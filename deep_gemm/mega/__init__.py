import math
import torch
import types
import warnings
from enum import Enum
from typing import Tuple, Optional, Union
from ..utils.math import align

# noinspection PyBroadException
try:
    # noinspection PyProtectedMember
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(
        f"Failed to load mega kernels, please check your PyTorch version: {exception}"
    )

from .. import _C


class RouteWeightMode(str, Enum):
    PRE_DOWN = 'pre_down'
    POST_DOWN = 'post_down'


class CombineOrderMode(str, Enum):
    FIXED_TOPK = 'fixed_topk'
    DEEPEP = 'deepep'
    DEEPEP_V1 = 'deepep_v1'


def _situ_kwargs(activation, beta, linear_beta, route_weight_mode, clamp, fast_math,
                 *, allow_fast_math=False):
    if activation != 'situ':
        return {}
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError('situ_beta must be positive and finite')
    linear_beta = float('inf') if linear_beta is None else linear_beta
    if math.isnan(linear_beta) or linear_beta <= 0:
        raise ValueError('situ_linear_beta must be positive or None')
    if RouteWeightMode(route_weight_mode) is not RouteWeightMode.POST_DOWN:
        raise ValueError('SiTU requires post_down route weights')
    if clamp is not None and clamp != float('inf'):
        raise ValueError('SiTU requires an unclamped activation')
    if fast_math and not allow_fast_math:
        raise ValueError('SiTU requires fast_math=False')
    return dict(situ_beta=beta, situ_linear_beta=linear_beta)


def _validate_num_sms(num_sms):
    if num_sms is None:
        return None
    if type(num_sms) is not int or num_sms <= 0 or num_sms % 2:
        raise ValueError("num_sms must be a positive even integer or None")
    if not getattr(_C, "BF16_MEGAMOE_SCOPED_SM_BUDGET", False):
        raise RuntimeError("Explicit BF16 MegaMoE num_sms requires a matching build")
    if num_sms > torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count:
        raise ValueError("num_sms exceeds the current device's SM count")
    return num_sms


def _scoped_num_sms_kwargs(sym_buffer):
    num_sms = getattr(sym_buffer, "num_sms", None)
    return {} if num_sms is None else {"num_sms": num_sms}


class SymmBuffer:
    def __init__(self, group: dist.ProcessGroup,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 num_ring_tokens: Optional[int] = None,
                 mma_type: str = 'fp8xfp4',
                 activation: str = 'swiglu',
                 base: Optional['SymmBuffer'] = None,
                 num_shared_experts: int = 0,
                 side_lora: bool = False,
                 bounded_lag: bool = False,
                 num_sms: Optional[int] = None):
        if num_sms is not None and (side_lora or mma_type != "bf16xbf16"):
            raise ValueError("Explicit num_sms requires merged-weight BF16")
        self._num_sms = _validate_num_sms(num_sms)
        if base is not None and getattr(base, "num_sms", None) != self._num_sms:
            raise ValueError("A reused symmetric buffer must have the same SM budget")
        # Align token count
        num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_mega_moe())

        if bounded_lag and (mma_type != 'bf16xbf16' or side_lora):
            raise ValueError("bounded lag requires the BF16 MegaMoE path")
        if bounded_lag and not hasattr(_C, "get_symm_buffer_size_for_bf16_mega_moe_bounded_lag"):
            raise RuntimeError("bounded lag requires a matching DeepGEMM build")

        if num_ring_tokens is not None and (
            num_ring_tokens <= 0 or num_ring_tokens % _C.get_token_alignment_for_mega_moe() != 0
        ):
            raise ValueError("num_ring_tokens must be positive and satisfy MegaMoE token alignment")

        # Init
        assert activation in ('swiglu', 'geglu', 'situ'), f'Unsupported activation: `{activation}`'
        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.num_shared_experts = num_shared_experts
        self.mma_type = mma_type
        self.activation = activation
        self.bounded_lag = bounded_lag

        # Allocate or reuse a symmetric buffer
        if side_lora:
            if num_shared_experts:
                raise ValueError("Fused side-LoRA does not support in-kernel shared experts")
            num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_mega_moe_side_lora(
                group.size(), num_experts, num_max_tokens_per_rank, num_topk,
                hidden, intermediate_hidden, mma_type, activation, num_ring_tokens or 0,
            )
        else:
            size_buffer = (_C.get_symm_buffer_size_for_bf16_mega_moe_bounded_lag if bounded_lag
                           else _C.get_symm_buffer_size_for_mega_moe_v3)
            num_bytes, slice_input_buffers = size_buffer(
                group.size(), num_experts, num_max_tokens_per_rank, num_topk,
                hidden, intermediate_hidden, mma_type, activation,
                num_ring_tokens or 0, num_shared_experts,
                **_scoped_num_sms_kwargs(self),
            )
        if base is None:
            allocator = torch if group.size() == 1 else symm_mem
            device = torch.device('cuda', torch.cuda.current_device())
            self.buffer = allocator.empty(num_bytes, dtype=torch.int8, device=device)
            self.handle = (
                types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
                if group.size() == 1
                else symm_mem.rendezvous(self.buffer, group=group)
            )
            self.buffer.zero_()
            self.group.barrier()
            torch.cuda.synchronize()
        else:
            assert base.buffer is not None and base.handle is not None and base.group is group, \
                'Cannot reuse an invalid symmetric buffer'
            assert num_bytes <= base.buffer.nbytes, \
                (f'The reused Mega MoE config requires {num_bytes} bytes, '
                 f'but the symmetric buffer only has {base.buffer.nbytes} bytes')
            self.buffer = base.buffer
            self.handle = base.handle

        views = slice_input_buffers(self.buffer)
        self.side_lora_source = views[-1] if side_lora else None
        if side_lora:
            views = views[:-1]

        # Create input buffer views
        # `token_src_metadata` exposes the Workspace combine source mapping
        # (rank_idx, token_idx, topk_idx) per pool row, needed by the training
        # backward to scatter per-expert grads back to source tokens/top-k slots.
        (
            self.x,
            self.x_sf,
            self.topk_idx,
            self.topk_weights,
            self.l1_acts,
            self.l1_acts_sf,
            self.l2_acts,
            self.l2_acts_sf,
            self.token_src_metadata,
            self.backward_grad_y,
            self.backward_grad_route,
            self.shared_l1_acts, self.shared_l1_acts_sf,
            self.shared_l2_acts, self.shared_l2_acts_sf,
        ) = views
        self.num_ring_tokens = self.l1_acts.size(0)
        self.num_sf_ring_tokens = self.l1_acts_sf.size(0) if self.l1_acts_sf is not None else 0

    @property
    def num_sms(self):
        """Rank-uniform launch budget fixed when this buffer is created."""
        return self._num_sms

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None
        self.token_src_metadata = None
        self.backward_grad_y = None
        self.backward_grad_route = None
        self.side_lora_source = None


# TODO: remove this function
def get_symm_buffer_for_mega_moe(group: dist.ProcessGroup,
                                 num_experts: int,
                                 num_max_tokens_per_rank: int, num_topk: int,
                                 hidden: int, intermediate_hidden: int,
                                 use_fp8_dispatch: Union[bool, None] = None,
                                 mma_type: str = 'fp8xfp4',
                                 activation: str = 'swiglu',
                                 num_ring_tokens: Optional[int] = None,
                                 num_shared_experts: int = 0,
                                 side_lora: bool = False,
                                 bounded_lag: bool = False,
                                 num_sms: Optional[int] = None,
                                 ) -> SymmBuffer:
    # Backward compat: derive `mma_type` from `use_fp8_dispatch` if provided
    if use_fp8_dispatch is not None:
        assert use_fp8_dispatch == (mma_type.split('x')[0] == 'fp8')
        warnings.warn(
            f'`use_fp8_dispatch` will be deprecated in the future, please use `mma_type`',
            DeprecationWarning, stacklevel=3
        )

    return SymmBuffer(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        num_ring_tokens,
        mma_type=mma_type, activation=activation, num_shared_experts=num_shared_experts,
        side_lora=side_lora, bounded_lag=bounded_lag, num_sms=num_sms,
    )


def _interleave_weights(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
    # [gate: 0..7, up: 0..7, gate: 8..15, up: 8..15, ...] instead of [gate | up]
    # Unsqueeze for 2D
    assert t.dim() in (2, 3)
    squeeze_group_dim = t.dim() == 2
    if squeeze_group_dim:
        t = t.unsqueeze(0)

    # Transpose
    g, n, *rest = t.shape
    half = n // 2
    gate = t[:, :half].reshape(g, half // gran, gran, *rest)
    up = t[:, half:].reshape(g, half // gran, gran, *rest)
    result = torch.empty_like(t).copy_(torch.stack([gate, up], dim=2).reshape(g, n, *rest))
    return result.squeeze(0) if squeeze_group_dim else result


def _transpose_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    # Unsqueeze for 2D
    assert sf.dtype == torch.int and sf.dim() in (2, 3)
    squeeze_group_dim = sf.dim() == 2
    if squeeze_group_dim:
        sf = sf.unsqueeze(0)

    # Transpose
    num_groups, mn, packed_sf_k = sf.shape
    assert mn % 128 == 0
    result = (sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
                .transpose(2, 3)
                .reshape(num_groups, mn, packed_sf_k))
    result = torch.empty_like(sf).copy_(result)
    return result.squeeze(0) if squeeze_group_dim else result


def transform_weights_for_mega_moe(
    l1_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    l2_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    activation: str = 'swiglu'
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
             Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
    # Gate/up interleaving is independent of the gated-activation variant
    assert activation in ('swiglu', 'geglu', 'situ'), f'Unsupported activation: `{activation}`'
    if isinstance(l1_weights, tuple):
        # FP8/FP4: interleave gate/up for weight and SF, then transpose L1 SF for UTCCP
        l1_w = _interleave_weights(l1_weights[0])
        l1_sf = _transpose_sf_for_utccp(_interleave_weights(l1_weights[1]))
        l1_transformed = (l1_w, l1_sf)
        # L2: only transpose SF for UTCCP
        l2_transformed = (l2_weights[0], _transpose_sf_for_utccp(l2_weights[1]))
    else:
        # BF16: L1 interleave gate/up, L2 unchanged
        l1_transformed = _interleave_weights(l1_weights)
        l2_transformed = l2_weights
    return l1_transformed, l2_transformed


def transform_side_lora_for_mega_moe(
    side_lora: Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                     torch.Tensor, torch.Tensor, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create the K-major rank-128 views consumed by the fused TC path.

    Input tensors use the conventional ``(A1, B1, A3, B3, A2, B2)``
    layouts from the shared side-LoRA contract: A1/A3 are independent
    ``[H, R]`` matrices shared across experts, B1/B3 and A2 are expert-local,
    and B2 is one shared ``[R, H]`` matrix. Keep and refresh this transformed
    tuple after optimizer updates; it must not be rebuilt for every forward.
    """
    if len(side_lora) != 6:
        raise ValueError(
            "side_lora must contain (A1, B1, A3, B3, A2, B2)")
    rank = side_lora[0].size(-1)
    if rank != 128:
        raise ValueError(
            "the tensor-core MegaMoE side-LoRA path currently requires "
            "rank 128")
    a1, b1, a3, b3, a2, b2 = side_lora
    if a1.dim() != 2 or a3.dim() != 2 or b2.dim() != 2:
        raise ValueError(
            "side-LoRA requires shared A1/A3 and shared B2")
    if b1.dim() != 3 or b3.dim() != 3 or a2.dim() != 3:
        raise ValueError(
            "side-LoRA requires expert-local B1/B3/A2")
    hidden, rank = a1.shape
    experts, b1_rank, intermediate = b1.shape
    expected_shapes = (
        ("A3", a3.shape, (hidden, rank)),
        ("B1", b1.shape, (experts, rank, intermediate)),
        ("B3", b3.shape, (experts, rank, intermediate)),
        ("A2", a2.shape, (experts, intermediate, rank)),
        ("B2", b2.shape, (rank, hidden)),
    )
    for name, actual, expected in expected_shapes:
        if tuple(actual) != expected:
            raise ValueError(
                f"invalid {name} shape {tuple(actual)}; expected {expected}")
    if b1_rank != rank:
        raise ValueError("B1 rank must match A1 rank")
    if any(tensor.dtype != torch.bfloat16 for tensor in side_lora):
        raise TypeError("side-LoRA tensors must all be BF16")
    return tuple(tensor.transpose(-1, -2).contiguous()
                 for tensor in side_lora)



def fp8_fp4_mega_moe(
    y: torch.Tensor,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    sym_buffer: SymmBuffer,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    recipe: Tuple[int, int, int] = (1, 1, 32),
    activation: str = "swiglu",
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
    saved_l1_preact: Optional[torch.Tensor] = None,
    route_weight_mode: RouteWeightMode = RouteWeightMode.PRE_DOWN,
    saved_down_unweighted: Optional[torch.Tensor] = None,
    saved_l1_acts: Optional[torch.Tensor] = None,
    saved_l1_acts_sf: Optional[torch.Tensor] = None,
    num_config_tokens: Optional[int] = None,
    shared_l1_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    shared_l2_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    *,
    situ_beta: Optional[float] = None,
    situ_linear_beta: Optional[float] = None,
    local_claim: bool = False,
    comm_claim: bool = False,
):
    """Run MXFP8/FP4 MegaMoE with an explicit route-weight boundary.

    POST_DOWN quantizes the unweighted gated activation for W2 and optionally
    saves each route's exact BF16 unweighted W2 output before applying its
    score at the remote combine write.

    local_claim (one-rank buffers only) launches the kernel without grid-wide
    barriers: work is claimed in dependency order, so it completes with any
    number of free SMs. The caller must run deep_gemm.mega.local.LocalPlan.plan
    on the launch stream before each launch.

    comm_claim (no in-kernel shared experts) launches a multi-rank kernel without
    grid-wide barriers: dispatch, pull, GEMM and combine work is claimed in
    dependency order and only the last CTA of each rank meets the other ranks.
    It completes while other kernels hold any number of SMs, as long as one SM
    is free on every rank. Every rank of the group must use the same mode.
    """
    if local_claim and sym_buffer.group.size() != 1:
        raise ValueError("local_claim requires a one-rank symmetric buffer")
    if comm_claim and (local_claim or shared_l1_weights is not None):
        raise ValueError("comm_claim excludes local_claim and in-kernel shared experts")
    route_weight_mode = RouteWeightMode(route_weight_mode)
    has_explicit_config_tokens = num_config_tokens is not None
    if num_config_tokens is None:
        num_config_tokens = y.size(0)
    if (
        (saved_l1_preact is not None or saved_l1_acts is not None or
         saved_down_unweighted is not None) and
        not has_explicit_config_tokens and
        sym_buffer.group.size() > 1
    ):
        # Every rank must instantiate the same persistent grid/barrier
        # specialization, including empty source ranks. Select from the
        # collective maximum source-token extent while retaining each rank's
        # local runtime token count.
        rank_uniform_num_tokens = torch.tensor(
            num_config_tokens, dtype=torch.int32, device=y.device)
        dist.all_reduce(
            rank_uniform_num_tokens,
            op=dist.ReduceOp.MAX,
            group=sym_buffer.group)
        num_config_tokens = int(rank_uniform_num_tokens.item())
    situ_kwargs = {}
    if situ_beta is not None or situ_linear_beta is not None:
        situ_kwargs = dict(situ_beta_opt=situ_beta, situ_linear_beta_opt=situ_linear_beta)
    _C.fp8_fp4_mega_moe(
        y,
        l1_weights,
        l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts,
        sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math,
        sym_buffer.num_ring_tokens,
        saved_l1_preact,
        route_weight_mode.value,
        saved_down_unweighted,
        saved_l1_acts,
        saved_l1_acts_sf,
        num_config_tokens,
        shared_l1_weights, shared_l2_weights,
        **situ_kwargs,
        local_claim=local_claim,
        comm_claim=comm_claim,
    )


def fp8_fp4_mega_moe_side_lora(
    y: torch.Tensor,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    sym_buffer: SymmBuffer,
    side_lora_input: torch.Tensor,
    side_lora: Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                     torch.Tensor, torch.Tensor, torch.Tensor],
    saved_x: torch.Tensor,
    saved_h_unweighted: torch.Tensor,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    recipe: Tuple[int, int, int] = (1, 1, 32),
    activation: str = "swiglu",
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
    saved_l1_preact: Optional[torch.Tensor] = None,
    route_weight_mode: RouteWeightMode = RouteWeightMode.PRE_DOWN,
    saved_down_unweighted: Optional[torch.Tensor] = None,
    num_config_tokens: Optional[int] = None,
    side_lora_scale: float = 1.0,
    side_lora_scratch: Optional[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the dedicated MXFP4 base + BF16 rank-128 side-LoRA kernel.

    Only q1/q3/q2 are persisted. Rank expansion accumulators remain in TMEM
    beside the MXFP4 base accumulators and are BF16-rounded before addition;
    no hidden- or intermediate-width side delta is allocated.
    """
    route_weight_mode = RouteWeightMode(route_weight_mode)
    if len(side_lora) != 6:
        raise ValueError(
            "side_lora must contain (A1, B1, A3, B3, A2, B2)")
    if side_lora[0].size(0) != 128:
        raise ValueError("MXFP4 side-LoRA requires rank 128")
    if side_lora_input.dtype != torch.bfloat16:
        raise TypeError("side_lora_input must be BF16")
    if side_lora_input.shape != (y.size(0), sym_buffer.hidden):
        raise ValueError("side_lora_input must have shape [tokens, hidden]")
    if sym_buffer.side_lora_source is None or sym_buffer.side_lora_source.numel() == 0:
        raise ValueError("MXFP4 side-LoRA requires a buffer created with side_lora=True")
    sym_buffer.side_lora_source[:y.size(0)].copy_(
        side_lora_input.contiguous())
    pool_rows = saved_x.size(0)
    if side_lora_scratch is None:
        side_lora_scratch = (
            torch.empty(
                (pool_rows, 2, 128), dtype=torch.bfloat16,
                device=y.device),
            torch.empty(
                (pool_rows, 128), dtype=torch.bfloat16,
                device=y.device),
            torch.zeros(
                4 * sym_buffer.num_ring_tokens // 8,
                dtype=torch.int32, device=y.device),
        )
    if len(side_lora_scratch) != 3:
        raise ValueError("side_lora_scratch must contain (q13, q2, ready)")
    q13, q2, ready = side_lora_scratch
    has_explicit_config_tokens = num_config_tokens is not None
    if num_config_tokens is None:
        num_config_tokens = y.size(0)
    if not has_explicit_config_tokens and sym_buffer.group.size() > 1:
        rank_uniform_num_tokens = torch.tensor(
            num_config_tokens, dtype=torch.int32, device=y.device)
        dist.all_reduce(
            rank_uniform_num_tokens, op=dist.ReduceOp.MAX,
            group=sym_buffer.group)
        num_config_tokens = int(rank_uniform_num_tokens.item())
    _C.fp8_fp4_mega_moe_side_lora(
        y, l1_weights, l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer, sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(), sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk, recipe,
        activation, activation_clamp, fast_math,
        sym_buffer.num_ring_tokens, saved_l1_preact,
        route_weight_mode.value, saved_down_unweighted,
        num_config_tokens, saved_x, saved_h_unweighted,
        *side_lora, q13, q2, ready, float(side_lora_scale))
    return q13, q2, ready

def bf16_mega_moe(y: torch.Tensor,
                  l1_weights: torch.Tensor,
                  l2_weights: torch.Tensor,
                  sym_buffer: SymmBuffer,
                  cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                  activation: str = 'swiglu',
                  activation_clamp: Optional[float] = None,
                  fast_math: bool = True,
                  saved_l1_preact: Optional[torch.Tensor] = None,
                  route_weight_mode: RouteWeightMode = RouteWeightMode.PRE_DOWN,
                  saved_h_unweighted: Optional[torch.Tensor] = None,
                  saved_h_weighted: Optional[torch.Tensor] = None,
                  saved_down_unweighted: Optional[torch.Tensor] = None,
                  combine_order_mode: CombineOrderMode =
                  CombineOrderMode.FIXED_TOPK,
                  precomputed_route_counts: Optional[torch.Tensor] = None,
                  active_pool_rows: Optional[int] = None,
                  route_count_mismatch: Optional[torch.Tensor] = None,
                  num_config_tokens: Optional[int] = None,
                  saved_x: Optional[torch.Tensor] = None,
                  shared_l1_weights: Optional[torch.Tensor] = None,
                  shared_l2_weights: Optional[torch.Tensor] = None,
                  situ_beta: float = 4.0,
                  situ_linear_beta: Optional[float] = None,
                  l1_up_weights: Optional[torch.Tensor] = None,
                  local_claim: bool = False):
    """Run BF16 MegaMoE with an explicit route-weight boundary.

    ``y`` is BF16 by default. An explicitly supplied FP32 output is supported
    only with FIXED_TOPK combine, preserving the routed accumulator for a
    caller that adds a shared expert before final BF16 rounding. The BF16
    specialization and all defaults are unchanged.

    The optional stage saves expose unweighted/weighted activation and W2
    output boundaries for strict parity checks. The fused backward currently
    supports the production ``pre_down`` route-weight boundary only.

    Training callers may provide an exact local source-route histogram,
    rank-uniform ``active_pool_rows``, and a scalar mismatch flag to size saved
    pools from actual receive counts. The kernel publishes the precomputed
    counts, verifies them against its internal dispatch count, and sets the
    flag before any caller can accept a truncated result.

    ``l1_up_weights`` selects separate contiguous gate/up tensors instead of
    an eight-row-interleaved ``l1_weights`` tensor. Fused shared experts are
    not supported with this layout. MMA and activation rounding are unchanged.

    A buffer constructed with ``bounded_lag=True`` selects the bounded L1/L2
    task stream. Its ring and schedule are derived from fixed buffer capacity,
    independent of the current rank's token count.

    local_claim (one-rank buffers only) launches without grid-wide barriers,
    like fp8_fp4_mega_moe. Run deep_gemm.mega.local.LocalPlan.plan on the
    launch stream first. The in-kernel route-count check is skipped.
    """
    if local_claim and sym_buffer.group.size() != 1:
        raise ValueError("local_claim requires a one-rank symmetric buffer")
    situ_kwargs = _situ_kwargs(activation, situ_beta, situ_linear_beta,
                               route_weight_mode, activation_clamp, fast_math)
    weight_kwargs = {}
    if getattr(sym_buffer, "bounded_lag", False):
        weight_kwargs["bounded_lag"] = True
    if l1_up_weights is not None:
        if not getattr(_C, "BF16_MEGAMOE_SEPARATE_L1", False):
            raise RuntimeError("separate forward weights require a matching DeepGEMM build")
        if shared_l1_weights is not None:
            raise ValueError("separate forward weights require shared experts outside MegaMoE")
        weight_kwargs["l1_up_weights"] = l1_up_weights
    if activation == "situ" and shared_l1_weights is not None:
        raise ValueError("SiTU shared experts must run outside MegaMoE")
    route_weight_mode = RouteWeightMode(route_weight_mode)
    combine_order_mode = CombineOrderMode(combine_order_mode)
    if (
        (saved_h_unweighted is None) !=
        (saved_h_weighted is None)
    ):
        raise ValueError(
            "both activation stage outputs must be provided together")
    active_plan = (
        precomputed_route_counts is not None,
        active_pool_rows is not None,
        route_count_mismatch is not None,
    )
    if any(active_plan) and not all(active_plan):
        raise ValueError(
            "precomputed_route_counts, active_pool_rows, and "
            "route_count_mismatch must be provided together")
    has_precomputed_config_tokens = num_config_tokens is not None
    if num_config_tokens is None:
        num_config_tokens = y.size(0)
    if (
        (saved_l1_preact is not None or saved_h_unweighted is not None or
         saved_down_unweighted is not None or saved_x is not None)
        and not has_precomputed_config_tokens
        and sym_buffer.group.size() > 1
    ):
        # The config selects BLOCK_M, which defines the persistent launch and
        # pool packing. Empty source ranks can still receive expert rows, so
        # every rank must select the config from the same source-token extent.
        rank_uniform_num_tokens = torch.tensor(
            num_config_tokens, dtype=torch.int32, device=y.device)
        dist.all_reduce(
            rank_uniform_num_tokens,
            op=dist.ReduceOp.MAX,
            group=sym_buffer.group)
        num_config_tokens = int(rank_uniform_num_tokens.item())
    _C.bf16_mega_moe(
        y,
        l1_weights,
        l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts,
        sym_buffer.num_topk,
        activation, activation_clamp,
        fast_math,
        sym_buffer.num_ring_tokens,
        saved_l1_preact,
        route_weight_mode.value,
        saved_h_unweighted,
        saved_h_weighted,
        saved_down_unweighted,
        num_config_tokens,
        combine_order_mode.value,
        precomputed_route_counts,
        active_pool_rows,
        route_count_mismatch,
        saved_x,
        shared_l1_weights, shared_l2_weights,
        **situ_kwargs,
        **weight_kwargs,
        **_scoped_num_sms_kwargs(sym_buffer),
        local_claim=local_claim,
    )

def bf16_mega_moe_side_lora(y: torch.Tensor,
                  l1_weights: torch.Tensor,
                  l2_weights: torch.Tensor,
                  sym_buffer: SymmBuffer,
                  cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                  activation: str = 'swiglu',
                  activation_clamp: Optional[float] = None,
                  fast_math: bool = True,
                  saved_l1_preact: Optional[torch.Tensor] = None,
                  route_weight_mode: RouteWeightMode = RouteWeightMode.PRE_DOWN,
                  saved_h_unweighted: Optional[torch.Tensor] = None,
                  saved_h_weighted: Optional[torch.Tensor] = None,
                  saved_down_unweighted: Optional[torch.Tensor] = None,
                  combine_order_mode: CombineOrderMode =
                  CombineOrderMode.FIXED_TOPK,
                  precomputed_route_counts: Optional[torch.Tensor] = None,
                  active_pool_rows: Optional[int] = None,
                  route_count_mismatch: Optional[torch.Tensor] = None,
                  num_config_tokens: Optional[int] = None,
                  saved_x: Optional[torch.Tensor] = None,
                  side_lora: Optional[Tuple[torch.Tensor, torch.Tensor,
                                            torch.Tensor, torch.Tensor,
                                            torch.Tensor, torch.Tensor]] = None,
                  side_lora_scale: float = 1.0,
                  side_lora_scratch: Optional[
                      Tuple[torch.Tensor, torch.Tensor,
                            torch.Tensor]] = None):
    """Run the dedicated native rank-128 BF16 side-LoRA MegaMoE kernel.

    The optional stage saves expose unweighted/weighted activation and W2
    output boundaries for strict parity checks. ``saved_down_unweighted`` is
    also used by post-down backward for the exact router gradient.

    Training callers may provide an exact local source-route histogram,
    rank-uniform ``active_pool_rows``, and a scalar mismatch flag to size saved
    pools from actual receive counts. The kernel publishes the precomputed
    counts, verifies them against its internal dispatch count, and sets the
    flag before any caller can accept a truncated result.

    ``side_lora`` is the K-major tuple returned by
    :func:`transform_side_lora_for_mega_moe`. The optional scratch tuple is
    ``(l1_shrink, l2_shrink, ready)``. Callers on a hot
    path should retain and reuse it; otherwise correctly-sized buffers are
    allocated.
    """
    if sym_buffer.side_lora_source is None:
        raise ValueError("BF16 side-LoRA requires a buffer created with side_lora=True")
    route_weight_mode = RouteWeightMode(route_weight_mode)
    combine_order_mode = CombineOrderMode(combine_order_mode)
    if (
        (saved_h_unweighted is None) !=
        (saved_h_weighted is None)
    ):
        raise ValueError(
            "both activation stage outputs must be provided together")
    active_plan = (
        precomputed_route_counts is not None,
        active_pool_rows is not None,
        route_count_mismatch is not None,
    )
    if any(active_plan) and not all(active_plan):
        raise ValueError(
            "precomputed_route_counts, active_pool_rows, and "
            "route_count_mismatch must be provided together")
    has_precomputed_config_tokens = num_config_tokens is not None
    if num_config_tokens is None:
        num_config_tokens = y.size(0)
    if (
        not has_precomputed_config_tokens
        and sym_buffer.group.size() > 1
    ):
        # The config selects BLOCK_M, which defines the persistent launch and
        # pool packing. Empty source ranks can still receive expert rows, so
        # every rank must select the config from the same source-token extent.
        rank_uniform_num_tokens = torch.tensor(
            num_config_tokens, dtype=torch.int32, device=y.device)
        dist.all_reduce(
            rank_uniform_num_tokens,
            op=dist.ReduceOp.MAX,
            group=sym_buffer.group)
        num_config_tokens = int(rank_uniform_num_tokens.item())
    if side_lora is None:
        raise ValueError("the dedicated side-LoRA kernel requires adapters")
    else:
        if len(side_lora) != 6:
            raise ValueError(
                "side_lora must contain (A1, B1, A3, B3, A2, B2)")
        side_lora_rank = side_lora[0].size(0)
        if side_lora_rank != 128:
            raise ValueError(
                "side_lora must be the rank-128 K-major tuple returned by "
                "transform_side_lora_for_mega_moe")
        if side_lora_scratch is None:
            side_lora_scratch = (
                torch.empty(
                    (active_pool_rows or
                     sym_buffer.token_src_metadata.size(0),
                     2, side_lora_rank),
                    dtype=torch.bfloat16, device=y.device),
                torch.empty(
                    (active_pool_rows or
                     sym_buffer.token_src_metadata.size(0),
                     side_lora_rank),
                    dtype=torch.bfloat16, device=y.device),
                torch.empty(
                    (4, sym_buffer.num_ring_tokens // 8),
                    dtype=torch.int32, device=y.device),
            )
        if len(side_lora_scratch) != 3:
            raise ValueError(
                "side_lora_scratch must contain "
                "(l1_shrink, l2_shrink, ready)")
        side_lora_args = (*side_lora, *side_lora_scratch)
    _C.bf16_mega_moe_side_lora(
        y,
        l1_weights,
        l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts,
        sym_buffer.num_topk,
        activation, activation_clamp,
        fast_math,
        sym_buffer.num_ring_tokens,
        saved_l1_preact,
        route_weight_mode.value,
        saved_h_unweighted,
        saved_h_weighted,
        saved_down_unweighted,
        num_config_tokens,
        combine_order_mode.value,
        precomputed_route_counts,
        active_pool_rows,
        route_count_mismatch,
        saved_x,
        *side_lora_args,
        float(side_lora_scale),
    )
