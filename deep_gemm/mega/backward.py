import time
from typing import Any, Callable, NamedTuple, Optional, Tuple

import torch
import torch.distributed as dist

from .. import _C
from . import CombineOrderMode, RouteWeightMode, _situ_kwargs

_BF16_BACKWARD_TIMING_HOOK: Optional[Callable[[str, torch.cuda.Event, torch.cuda.Event, float], None]] = None
_BF16_BACKWARD_KERNEL_TRACE_HOOK: Optional[
    Callable[[torch.Tensor, float, torch.Tensor, int], None]
] = None
_BF16_BACKWARD_KERNEL_TRACE_SITES = (
    "kernel_total",
    "initial_cluster_arrival",
    "initial_cluster_ready",
    "dispatch_rank_arrival",
    "dispatch_grid_completion",
    "local_dispatch_grid_completion",
    "w2_input_grid_completion",
    "before_direct_grad_x",
    "after_grad_y_clear",
    "w2_phase_cluster_arrival",
    "w2_phase_cluster_handoff",
    "w2_pipeline_ready",
    "w2_to_w13_grid_handoff",
    "w13_padding_clear",
    "predown_route_alias_handoff",
    "w13_phase_cluster_arrival",
    "w13_pipeline_ready",
    "direct_grad_x_local_grid_completion",
    "direct_grad_x_rank_signal",
    "direct_grad_x_publish_grid_completion",
    "kernel_final_cluster_sync",
    "w13_compute_and_grad_x_publication",
)
_BF16_SIDE_LORA_BACKWARD_KERNEL_TRACE_SITES = _BF16_BACKWARD_KERNEL_TRACE_SITES + (
    "canonical_gate_up_grad_deinterleave_handoff",
)
_BF16_BACKWARD_KERNEL_TRACE_FIELDS = (
    "begin_cycle",
    "end_cycle",
    "begin_global_ns",
    "end_global_ns",
    "sm_id",
)
_BF16_BACKWARD_DECOMPOSE_PRELUDE = False


def _timed_cuda_phase(name: str, operation: Callable[[], Any]) -> Any:
    hook = _BF16_BACKWARD_TIMING_HOOK
    if hook is None:
        return operation()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    wall_start = time.perf_counter()
    result = operation()
    wall_ms = (time.perf_counter() - wall_start) * 1_000
    end.record()
    hook(name, start, end, wall_ms)
    return result


def _allocate_backward_kernel_trace(
    sites: tuple[str, ...], device: torch.device,
) -> tuple[torch.Tensor, float]:
    properties = torch.cuda.get_device_properties(device)
    return (
        torch.zeros(
            (
                len(sites),
                properties.multi_processor_count,
                len(_BF16_BACKWARD_KERNEL_TRACE_FIELDS),
            ),
            dtype=torch.int64,
            device=device,
        ),
        float(properties.clock_rate),
    )


def _storage_byte_range(tensor: torch.Tensor) -> tuple[int, int]:
    start = tensor.data_ptr()
    return start, start + tensor.numel() * tensor.element_size()


def _exact_alias(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    return lhs.data_ptr() == rhs.data_ptr() and lhs.numel() == rhs.numel()


def _validate_bf16_backward_alias_contract(
    *,
    memory_mode: str,
    route_weight_mode: RouteWeightMode,
    gate_up_output: torch.Tensor,
    grad_gate_up_output: torch.Tensor,
    grad_h_output: torch.Tensor,
    h_act_output: torch.Tensor,
    h_weighted_output: torch.Tensor,
    x_pool_output: torch.Tensor,
    grad_x_pool_output: torch.Tensor,
    grad_ye: torch.Tensor,
    grad_y_unweighted_output: torch.Tensor,
    down_unweighted_output: torch.Tensor,
) -> None:
    """Validate every overlapping BF16 pool against the phase contract."""
    if memory_mode not in ("legacy", "phase_ordered"):
        raise ValueError("memory_mode must be 'legacy' or 'phase_ordered', got " f"{memory_mode!r}")

    tensors = {
        "gate_up": gate_up_output,
        "grad_gate_up": grad_gate_up_output,
        "grad_h": grad_h_output,
        "h_act": h_act_output,
        "h_weighted": h_weighted_output,
        "x_pool": x_pool_output,
        "grad_x_pool": grad_x_pool_output,
        "grad_ye": grad_ye,
        "grad_y_unweighted": grad_y_unweighted_output,
        "down_unweighted": down_unweighted_output,
    }
    # Empty debug outputs have data_ptr()==0 and do not overlap anything.
    ranges = {name: _storage_byte_range(tensor) for name, tensor in tensors.items() if tensor.numel()}
    overlaps = set()
    names = list(ranges)
    for idx, lhs_name in enumerate(names):
        lhs_start, lhs_end = ranges[lhs_name]
        for rhs_name in names[idx + 1 :]:
            rhs_start, rhs_end = ranges[rhs_name]
            if max(lhs_start, rhs_start) < min(lhs_end, rhs_end):
                overlaps.add(frozenset((lhs_name, rhs_name)))

    # These aliases predate phase-ordered reuse and are safe in legacy mode.
    allowed = set()
    if route_weight_mode is RouteWeightMode.PRE_DOWN:
        allowed.add(frozenset(("grad_ye", "grad_y_unweighted")))
        allowed.add(frozenset(("grad_ye", "down_unweighted")))
        allowed.add(frozenset(("grad_y_unweighted", "down_unweighted")))
    else:
        allowed.add(frozenset(("h_act", "h_weighted")))

    if memory_mode == "phase_ordered":
        if route_weight_mode is RouteWeightMode.POST_DOWN:
            required_exact = (
                ("grad_ye", grad_ye, "down_unweighted", down_unweighted_output),
                ("h_act", h_act_output, "grad_h", grad_h_output),
                ("h_weighted", h_weighted_output, "grad_h", grad_h_output),
                ("grad_gate_up", grad_gate_up_output, "gate_up", gate_up_output),
            )
            if (
                grad_h_output.data_ptr() != grad_y_unweighted_output.data_ptr()
                or grad_h_output.numel() > grad_y_unweighted_output.numel()
            ):
                raise ValueError(
                    "phase_ordered post_down requires grad_h to be a " "contiguous prefix view of grad_y_unweighted"
                )
            allowed.add(frozenset(("grad_h", "grad_y_unweighted")))
            allowed.add(frozenset(("h_act", "grad_y_unweighted")))
            allowed.add(frozenset(("h_weighted", "grad_y_unweighted")))
        else:
            required_exact = (
                ("grad_ye", grad_ye, "grad_y_unweighted", grad_y_unweighted_output),
                ("h_act", h_act_output, "h_weighted", h_weighted_output),
                ("grad_gate_up", grad_gate_up_output, "gate_up", gate_up_output),
            )
        for lhs_name, lhs, rhs_name, rhs in required_exact:
            if not _exact_alias(lhs, rhs):
                raise ValueError(
                    f"phase_ordered {route_weight_mode.value} requires " f"{lhs_name} to exactly alias {rhs_name}"
                )
            allowed.add(frozenset((lhs_name, rhs_name)))

    unexpected = overlaps - allowed
    if unexpected:
        pairs = sorted("/".join(sorted(pair)) for pair in unexpected)
        raise ValueError(f"unsafe BF16 backward storage overlap for {memory_mode}: " + ", ".join(pairs))


def _sort_bf16_pool_in_deepep_order(
    tensors: tuple[torch.Tensor, ...],
    token_src_metadata: torch.Tensor,
    valid_rows: torch.Tensor,
    expert_counts: torch.Tensor,
    num_max_tokens_per_rank: int,
    num_topk: int,
    num_ranks: int,
) -> None:
    """Sort valid expert rows into DeepEP's stable source-route order."""
    pool_rows = valid_rows.nonzero().flatten()
    if pool_rows.numel() <= 1:
        return
    expert_ids = torch.repeat_interleave(
        torch.arange(
            expert_counts.numel(),
            dtype=torch.long,
            device=expert_counts.device,
        ),
        expert_counts.to(torch.long),
    )
    if expert_ids.numel() != pool_rows.numel():
        raise RuntimeError("valid BF16 pool rows do not match expert counts")
    metadata = token_src_metadata[pool_rows].long()
    routes_per_rank = num_max_tokens_per_rank * num_topk
    route_keys = metadata[:, 0] * routes_per_rank + metadata[:, 1] * num_topk + metadata[:, 2]
    order = torch.argsort(
        expert_ids * (num_ranks * routes_per_rank) + route_keys,
        stable=True,
    )
    ordered_rows = pool_rows[order]
    if torch.equal(ordered_rows, pool_rows):
        return

    seen_ptrs = set()
    for tensor in tensors:
        ptr = tensor.data_ptr()
        if ptr in seen_ptrs:
            continue
        seen_ptrs.add(ptr)
        tensor[pool_rows] = tensor[ordered_rows].clone()


def bf16_mega_moe_backward_dgrad(
    gate_up_output: torch.Tensor,
    grad_h_output: torch.Tensor,
    grad_gate_up_output: torch.Tensor,
    h_act_output: torch.Tensor,
    h_weighted_output: torch.Tensor,
    x_pool_output: torch.Tensor,
    grad_x_pool_output: Optional[torch.Tensor],
    grad_route_output: torch.Tensor,
    grad_ye: torch.Tensor,
    route_weights: torch.Tensor,
    w2_weights: torch.Tensor,
    w13_weights: torch.Tensor,
    expert_counts: torch.Tensor,
    grid_sync_counter: torch.Tensor,
    grad_y: torch.Tensor,
    sym_buffer: Any,
    activation_limit: float,
    block_m: int,
    activation: str = "swiglu",
    fast_math: bool = False,
    direct_remote_grad_x: bool = True,
    write_grad_x_pool: bool = True,
    clear_wgrad_padding: bool = True,
    route_weight_mode: RouteWeightMode = RouteWeightMode.PRE_DOWN,
    grad_y_unweighted_output: Optional[torch.Tensor] = None,
    down_unweighted_output: Optional[torch.Tensor] = None,
    python_numerical_correction: bool = False,
    combine_order_mode: CombineOrderMode = CombineOrderMode.FIXED_TOPK,
    memory_mode: str = "legacy",
    rank_uniform_block_m: bool = False,
    x_prepared: bool = False,
    situ_beta: float = 4.0,
    situ_linear_beta: Optional[float] = None,
    w3_weights: Optional[torch.Tensor] = None,
    *,
    split_barrier_fused_route: bool = True,
    route_prelude_threads: int = 128,
) -> None:
    """Run BF16 reverse dispatch, dgrad, activation, and direct grad-x.

    ``memory_mode="phase_ordered"`` enables destructive pool reuse. POST_DOWN
    requires ``grad_ye == down_unweighted``, a ``grad_h`` prefix view of
    ``grad_y_unweighted``, ``h_act == h_weighted == grad_h``, and
    ``grad_gate_up == gate_up``. PRE_DOWN requires
    ``grad_y_unweighted == grad_ye``, ``h_act == h_weighted``, and
    ``grad_gate_up == gate_up``. Logical tensors whose storage is reused are
    no longer observable after this call. ``"legacy"`` keeps independent
    storage, apart from the historical semantic aliases validated above.

    For SiTU, ``fast_math=True`` uses the hardware tanh approximation in
    backward only. Sigmoid evaluation, FP32 arithmetic order and BF16
    materialization are unchanged. The default retains precise tanh.

    Route-prelude scheduling is configured per call. The defaults preserve
    the split-barrier fused route prelude and the 128-thread H2048 route dot.
    Other hidden widths retain their 256-thread prelude.
    """
    if route_prelude_threads not in (128, 256):
        raise ValueError(f"route_prelude_threads must be 128 or 256, got {route_prelude_threads!r}")
    situ_kwargs = _situ_kwargs(activation, situ_beta, situ_linear_beta,
                               route_weight_mode, activation_limit, fast_math,
                               allow_fast_math=bool(getattr(
                                   _C, "BF16_MEGAMOE_SITU_FAST_BACKWARD", False)))
    # With a separate up weight, w13_weights is the gate tensor [E, I, H].
    # Preserve the packed ABI when callers do not request this layout.
    if w3_weights is not None:
        if python_numerical_correction:
            raise ValueError("separate weights do not support Python numerical correction")
        situ_kwargs["w3_weights"] = w3_weights
    if activation == "situ" and python_numerical_correction:
        raise ValueError("SiTU does not support Python numerical correction")
    route_weight_mode = RouteWeightMode(route_weight_mode)
    combine_order_mode = CombineOrderMode(combine_order_mode)
    if activation not in ("swiglu", "geglu", "situ"):
        raise ValueError(f"unsupported activation: {activation}")
    if not write_grad_x_pool and not direct_remote_grad_x:
        raise ValueError("grad-x requires a local or direct remote output")
    if grad_x_pool_output is None:
        if write_grad_x_pool:
            raise ValueError("write_grad_x_pool requires grad_x_pool_output")
        grad_x_pool_output = grad_ye.new_empty((0, grad_ye.size(1)))
    if python_numerical_correction and not write_grad_x_pool:
        raise ValueError("Python numerical correction requires grad_x_pool_output")
    if python_numerical_correction and memory_mode != "legacy":
        raise ValueError("Python numerical correction does not support destructive " "phase-ordered aliases")
    if x_prepared and memory_mode != "phase_ordered":
        raise ValueError("prepared x pool requires phase_ordered memory")
    if route_weight_mode is RouteWeightMode.POST_DOWN:
        if grad_y_unweighted_output is None:
            raise ValueError("post_down requires grad_y_unweighted_output")
        if down_unweighted_output is None:
            raise ValueError("post_down requires saved down_unweighted_output")
    else:
        if grad_y_unweighted_output is None:
            grad_y_unweighted_output = grad_ye
        if down_unweighted_output is None:
            down_unweighted_output = grad_ye
    _validate_bf16_backward_alias_contract(
        memory_mode=memory_mode,
        route_weight_mode=route_weight_mode,
        gate_up_output=gate_up_output,
        grad_gate_up_output=grad_gate_up_output,
        grad_h_output=grad_h_output,
        h_act_output=h_act_output,
        h_weighted_output=h_weighted_output,
        x_pool_output=x_pool_output,
        grad_x_pool_output=grad_x_pool_output,
        grad_ye=grad_ye,
        grad_y_unweighted_output=grad_y_unweighted_output,
        down_unweighted_output=down_unweighted_output,
    )
    if sym_buffer.group.size() > 1 and not rank_uniform_block_m:
        # BLOCK_M determines the persistent grid shape and therefore every
        # grid/NVLink barrier's participant count. A rank with no source
        # tokens may choose a smaller local bucket than its peers; launching
        # those different specializations deadlocks at the first grid sync.
        # Canonicalize at the API boundary so every caller is safe.
        rank_uniform_block_m = torch.tensor(block_m, dtype=torch.int32, device=grad_y.device)
        _timed_cuda_phase(
            "block_m_all_reduce",
            lambda: dist.all_reduce(rank_uniform_block_m, op=dist.ReduceOp.MAX, group=sym_buffer.group),
        )
        block_m = int(_timed_cuda_phase("block_m_host_item", rank_uniform_block_m.item))
    num_tokens = grad_y.shape[0]
    backward_grad_y = sym_buffer.backward_grad_y
    # This dedicated FP32 source plane is backward-only and retains one
    # scalar per physical top-k slot. Every valid expert row publishes
    # directly to its unique source slot; invalid routes remain zero.
    _timed_cuda_phase(
        "grad_route_plane_clear",
        sym_buffer.backward_grad_route.zero_,
    )
    if direct_remote_grad_x:
        # Kernel A reuses this region as [topk, token, hidden] direct-write
        # planes after every rank has pulled grad-y from plane zero. Clear all
        # planes so masked routes and repeated backward calls reduce as zero.
        backward_grad_y = torch.as_strided(
            backward_grad_y,
            size=(
                sym_buffer.num_topk,
                sym_buffer.num_max_tokens_per_rank,
                sym_buffer.hidden,
            ),
            stride=(
                sym_buffer.num_max_tokens_per_rank * sym_buffer.hidden,
                sym_buffer.hidden,
                1,
            ),
        )
        _timed_cuda_phase(
            "grad_y_expose_copy",
            lambda: backward_grad_y[0, :num_tokens].copy_(grad_y.to(torch.bfloat16).contiguous()),
        )
    else:
        _timed_cuda_phase(
            "grad_y_expose_copy",
            lambda: backward_grad_y[:num_tokens].copy_(grad_y.to(torch.bfloat16).contiguous()),
        )
    dispatch_inputs_prepared = memory_mode == "phase_ordered"
    if dispatch_inputs_prepared:
        # The prelude starts with a device-side NVLink barrier. It publishes
        # local grad-y before pulling grad-y, x, and route weights with the
        # full persistent grid. POST_DOWN also performs its exact route dot
        # and weighted-grad write; PRE_DOWN needs only the input dispatch.
        prepare_route_and_weighted = route_weight_mode is RouteWeightMode.POST_DOWN

        def launch_prelude(
            *,
            do_reverse_dispatch: bool,
            compute_route_dot: bool,
            write_weighted: bool,
            synchronize_ranks: bool,
            synchronize_after_dispatch: bool,
            barrier_only: bool = False,
        ) -> None:
            threads = 256
            if (
                compute_route_dot
                and grad_y.shape[1] == 2048
                and combine_order_mode is not CombineOrderMode.FIXED_TOPK
            ):
                threads = route_prelude_threads
            _C.bf16_mega_moe_backward_post_down_prelude_v2(
                grad_y_unweighted_output,
                grad_ye,
                x_pool_output,
                route_weights,
                grad_route_output,
                down_unweighted_output,
                expert_counts,
                sym_buffer.backward_grad_y,
                sym_buffer.x,
                sym_buffer.topk_weights,
                sym_buffer.backward_grad_route,
                sym_buffer.token_src_metadata,
                sym_buffer.handle.buffer_ptrs,
                sym_buffer.group.rank(),
                sym_buffer.num_topk,
                block_m,
                combine_order_mode.value,
                do_reverse_dispatch,
                compute_route_dot,
                write_weighted,
                synchronize_ranks,
                synchronize_after_dispatch,
                barrier_only,
                x_prepared,
                threads,
            )

        def launch_decomposed_route_prelude() -> None:
            _timed_cuda_phase(
                "rank_arrival_barrier",
                lambda: launch_prelude(
                    do_reverse_dispatch=False,
                    compute_route_dot=False,
                    write_weighted=False,
                    synchronize_ranks=True,
                    synchronize_after_dispatch=False,
                    barrier_only=True,
                ),
            )
            _timed_cuda_phase(
                "reverse_dispatch_nvlink_pull",
                lambda: launch_prelude(
                    do_reverse_dispatch=True,
                    compute_route_dot=False,
                    write_weighted=False,
                    synchronize_ranks=False,
                    synchronize_after_dispatch=False,
                ),
            )
            _timed_cuda_phase(
                "final_rank_synchronization",
                lambda: launch_prelude(
                    do_reverse_dispatch=False,
                    compute_route_dot=False,
                    write_weighted=False,
                    synchronize_ranks=False,
                    synchronize_after_dispatch=True,
                    barrier_only=True,
                ),
            )
            _timed_cuda_phase(
                "exact_route_dot_reduction",
                lambda: launch_prelude(
                    do_reverse_dispatch=False,
                    compute_route_dot=True,
                    write_weighted=False,
                    synchronize_ranks=False,
                    synchronize_after_dispatch=False,
                ),
            )
            _timed_cuda_phase(
                "route_weight_multiply_write",
                lambda: launch_prelude(
                    do_reverse_dispatch=False,
                    compute_route_dot=False,
                    write_weighted=True,
                    synchronize_ranks=False,
                    synchronize_after_dispatch=False,
                ),
            )

        def launch_decomposed_input_prelude() -> None:
            _timed_cuda_phase(
                "rank_arrival_barrier",
                lambda: launch_prelude(
                    do_reverse_dispatch=False,
                    compute_route_dot=False,
                    write_weighted=False,
                    synchronize_ranks=True,
                    synchronize_after_dispatch=False,
                    barrier_only=True,
                ),
            )
            _timed_cuda_phase(
                "reverse_dispatch_nvlink_pull",
                lambda: launch_prelude(
                    do_reverse_dispatch=True,
                    compute_route_dot=False,
                    write_weighted=False,
                    synchronize_ranks=False,
                    synchronize_after_dispatch=False,
                ),
            )
            _timed_cuda_phase(
                "final_rank_synchronization",
                lambda: launch_prelude(
                    do_reverse_dispatch=False,
                    compute_route_dot=False,
                    write_weighted=False,
                    synchronize_ranks=False,
                    synchronize_after_dispatch=True,
                    barrier_only=True,
                ),
            )

        def launch_chunked_route_prelude() -> None:
            # Pull the two remote BF16 planes with the dispatch-only vector
            # path, then preserve the exact Triton route reduction tree while
            # consuming the local grad-y plane. Splitting here avoids coupling
            # NVLink load coalescing to Triton's strided reduction lane map.
            _timed_cuda_phase(
                "reverse_dispatch_nvlink_pull",
                lambda: launch_prelude(
                    do_reverse_dispatch=True,
                    compute_route_dot=False,
                    write_weighted=False,
                    synchronize_ranks=True,
                    synchronize_after_dispatch=True,
                ),
            )
            _timed_cuda_phase(
                "exact_route_dot_weighted_write",
                lambda: launch_prelude(
                    do_reverse_dispatch=False,
                    compute_route_dot=True,
                    write_weighted=True,
                    synchronize_ranks=False,
                    # Route gradients are written directly into each source
                    # rank's symmetric top-k plane. W2 combine's entry
                    # barrier publishes these earlier same-stream writes
                    # before W13 combine and the source-plane clone.
                    synchronize_after_dispatch=False,
                ),
            )

        def launch_split_barrier_fused_route_prelude() -> None:
            _timed_cuda_phase(
                "rank_arrival_barrier",
                lambda: launch_prelude(
                    do_reverse_dispatch=False,
                    compute_route_dot=False,
                    write_weighted=False,
                    synchronize_ranks=True,
                    synchronize_after_dispatch=False,
                    barrier_only=True,
                ),
            )
            _timed_cuda_phase(
                "fused_reverse_dispatch_route",
                lambda: launch_prelude(
                    do_reverse_dispatch=True,
                    compute_route_dot=True,
                    write_weighted=True,
                    synchronize_ranks=False,
                    synchronize_after_dispatch=False,
                ),
            )
            _timed_cuda_phase(
                "final_rank_synchronization",
                lambda: launch_prelude(
                    do_reverse_dispatch=False,
                    compute_route_dot=False,
                    write_weighted=False,
                    synchronize_ranks=False,
                    synchronize_after_dispatch=True,
                    barrier_only=True,
                ),
            )

        _timed_cuda_phase(
            ("reverse_dispatch_route_prelude" if prepare_route_and_weighted else "reverse_dispatch_input_prelude"),
            (
                launch_decomposed_route_prelude
                if prepare_route_and_weighted and _BF16_BACKWARD_DECOMPOSE_PRELUDE
                else (
                    launch_split_barrier_fused_route_prelude
                    if (prepare_route_and_weighted and split_barrier_fused_route)
                    else (
                        launch_chunked_route_prelude
                        if prepare_route_and_weighted
                        else (
                            launch_decomposed_input_prelude
                            if _BF16_BACKWARD_DECOMPOSE_PRELUDE
                            else lambda: launch_prelude(
                                do_reverse_dispatch=True,
                                compute_route_dot=False,
                                write_weighted=False,
                                synchronize_ranks=True,
                                synchronize_after_dispatch=True,
                            )
                        )
                    )
                )
            ),
        )
    kernel_trace_hook = _BF16_BACKWARD_KERNEL_TRACE_HOOK
    kernel_trace = None
    trace_clock_rate_khz = 0.0
    if kernel_trace_hook is not None:
        kernel_trace, trace_clock_rate_khz = (
            _allocate_backward_kernel_trace(
                _BF16_BACKWARD_KERNEL_TRACE_SITES, grad_y.device,
            )
        )
    _timed_cuda_phase(
        "kernel_a_dgrad",
        lambda: _C.bf16_mega_moe_backward_dgrad_v2(
            gate_up_output,
            grad_h_output,
            grad_gate_up_output,
            h_act_output,
            h_weighted_output,
            x_pool_output,
            grad_x_pool_output,
            grad_route_output,
            grad_ye,
            grad_y_unweighted_output,
            route_weights,
            w2_weights,
            w13_weights,
            expert_counts,
            grid_sync_counter,
            activation_limit,
            activation,
            fast_math,
            route_weight_mode.value,
            down_unweighted_output,
            block_m,
            direct_remote_grad_x,
            write_grad_x_pool,
            clear_wgrad_padding,
            sym_buffer.backward_grad_y,
            sym_buffer.x,
            sym_buffer.topk_weights,
            sym_buffer.backward_grad_route,
            sym_buffer.token_src_metadata,
            sym_buffer.handle.buffer_ptrs,
            sym_buffer.group.rank(),
            sym_buffer.num_max_tokens_per_rank,
            sym_buffer.num_topk,
            combine_order_mode.value,
            memory_mode,
            kernel_trace,
            **situ_kwargs,
        ),
    )
    if kernel_trace_hook is not None:
        kernel_trace_hook(
            kernel_trace,
            trace_clock_rate_khz,
            expert_counts,
            block_m,
        )
    if not python_numerical_correction:
        return
    # FireTitan uses two native grouped GEMMs for W1/W3 dgrad and rounds each
    # result to BF16 before the in-place add. The fused W13 dgrad above uses
    # one FP32 accumulation across [gate | up], which is not bitwise
    # equivalent. Materialize the native boundary with the same grouped-MM
    # calls and offsets.
    valid_rows = torch.isfinite(gate_up_output[:, 0])
    row_indices = valid_rows.nonzero().flatten()
    intermediate_hidden = grad_gate_up_output.size(1) // 2
    if activation == "swiglu" and activation_limit == float("inf"):
        # Standard FireTitan grouped experts run BF16 SiLU followed by a
        # separate BF16 multiply. Reuse ATen's native BF16 forward/backward
        # operations so both materialization boundaries and SiLU's derivative
        # operation order match exactly.
        gate = gate_up_output[row_indices, :intermediate_hidden].contiguous()
        up = gate_up_output[row_indices, intermediate_hidden:].contiguous()
        native_silu = torch.nn.functional.silu(gate)
        native_h = native_silu * up
        native_grad_h = grad_h_output[row_indices]
        if route_weight_mode is RouteWeightMode.PRE_DOWN:
            native_grad_h = (native_grad_h.float() * route_weights[row_indices].float().unsqueeze(1)).to(
                native_grad_h.dtype
            )
        native_grad_silu = native_grad_h * up
        native_grad_gate = torch.ops.aten.silu_backward(native_grad_silu, gate)
        native_grad_up = native_grad_h * native_silu
        grad_gate_up_output[row_indices] = torch.cat((native_grad_gate, native_grad_up), dim=1)
        h_act_output[row_indices] = native_h
        h_weighted_output[row_indices] = (
            (native_h.float() * route_weights[row_indices].float().unsqueeze(1)).to(native_h.dtype)
            if route_weight_mode is RouteWeightMode.PRE_DOWN
            else native_h
        )
    elif activation == "geglu":
        # GeGLU's tanh approximation is expressed as separate ATen FP32
        # operations in FireTitan. Preserve each materialization point here;
        # reassociating the polynomial or derivative in CUDA changes rare
        # BF16 round-to-nearest ties.
        gate_unclamped = gate_up_output[row_indices, :intermediate_hidden].contiguous()
        up_unclamped = gate_up_output[row_indices, intermediate_hidden:].contiguous()
        gate = torch.clamp(gate_unclamped, max=activation_limit).float()
        up = torch.clamp(
            up_unclamped,
            min=-activation_limit,
            max=activation_limit,
        ).float()
        alpha = 1.5957691216057308
        beta = 0.044715
        gate_sq = gate * gate
        z = (alpha * gate) * (1.0 + beta * gate_sq)
        dz = alpha * (1.0 + (3.0 * beta) * gate_sq)
        sig = torch.sigmoid(z)
        activated_gate = gate * sig
        native_h = (activated_gate * up).to(torch.bfloat16)
        native_grad_h = grad_h_output[row_indices]
        if route_weight_mode is RouteWeightMode.PRE_DOWN:
            native_grad_h = (native_grad_h.float() * route_weights[row_indices].float().unsqueeze(1)).to(
                native_grad_h.dtype
            )
        activation_grad = sig + gate * sig * (1.0 - sig) * dz
        native_grad_gate = native_grad_h.float() * up * activation_grad
        native_grad_gate = torch.where(
            gate_unclamped.float() <= activation_limit,
            native_grad_gate,
            torch.zeros_like(native_grad_gate),
        ).to(torch.bfloat16)
        native_grad_up = native_grad_h.float() * activated_gate
        native_grad_up = torch.where(
            (up_unclamped.float() >= -activation_limit) & (up_unclamped.float() <= activation_limit),
            native_grad_up,
            torch.zeros_like(native_grad_up),
        ).to(torch.bfloat16)
        grad_gate_up_output[row_indices] = torch.cat((native_grad_gate, native_grad_up), dim=1)
        h_act_output[row_indices] = native_h
        h_weighted_output[row_indices] = (
            (native_h.float() * route_weights[row_indices].float().unsqueeze(1)).to(native_h.dtype)
            if route_weight_mode is RouteWeightMode.PRE_DOWN
            else native_h
        )
    offsets = expert_counts.cumsum(0).to(torch.int32)
    w1_weights = w13_weights[:, :intermediate_hidden].contiguous()
    w3_weights = w13_weights[:, intermediate_hidden:].contiguous()
    if row_indices.numel():
        native_grad_x = torch._grouped_mm(
            grad_gate_up_output[row_indices, :intermediate_hidden].contiguous(),
            w1_weights,
            offs=offsets,
        )
        native_grad_x.add_(
            torch._grouped_mm(
                grad_gate_up_output[row_indices, intermediate_hidden:].contiguous(),
                w3_weights,
                offs=offsets,
            )
        )
    else:
        native_grad_x = grad_x_pool_output.new_empty((0, grad_x_pool_output.size(1)))
    grad_x_pool_output.zero_()
    grad_x_pool_output[row_indices] = native_grad_x

    # Match PyTorch's deterministic FP32 reduction tree for router gradients;
    # the fused kernel's scalar left fold has a different rounding order.
    if route_weight_mode is RouteWeightMode.POST_DOWN:
        route_lhs = grad_y_unweighted_output[row_indices]
        route_rhs = down_unweighted_output[row_indices]
    else:
        route_lhs = grad_h_output[row_indices]
        route_rhs = h_act_output[row_indices]
    native_grad_route = (route_lhs.float() * route_rhs.float()).sum(dim=1)
    grad_route_output.zero_()
    grad_route_output[row_indices] = native_grad_route.to(grad_route_output.dtype)

    if direct_remote_grad_x:
        metadata = sym_buffer.token_src_metadata[row_indices].long()
        native_grad_x_planes = torch.zeros(
            (
                sym_buffer.group.size(),
                sym_buffer.num_max_tokens_per_rank,
                sym_buffer.num_topk,
                grad_x_pool_output.size(1),
            ),
            dtype=grad_x_pool_output.dtype,
            device=grad_x_pool_output.device,
        )
        native_grad_x_planes[metadata[:, 0], metadata[:, 1], metadata[:, 2]] = native_grad_x
        dist.all_reduce(native_grad_x_planes, group=sym_buffer.group)
        # Install the corrected values into the real slot-major symmetric
        # planes before the fused W13 combine. Reducing a shadow tensor after
        # combine masks destination/slot bugs and leaves direct-plane users
        # observing the fused W13 dgrad's different accumulation boundary.
        backward_grad_y.copy_(native_grad_x_planes[sym_buffer.group.rank()].permute(1, 0, 2))

    # MegaMoE's communication scheduler deliberately interleaves source
    # ranks, while DeepEP's native grouped wgrads consume stable
    # expert/rank/token/slot order. Canonicalize the retained pool before the
    # standalone wgrad kernels so their K traversal matches FireTitan.
    _sort_bf16_pool_in_deepep_order(
        (
            grad_gate_up_output,
            h_weighted_output,
            x_pool_output,
            grad_ye,
        ),
        sym_buffer.token_src_metadata,
        valid_rows,
        expert_counts,
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_topk,
        sym_buffer.group.size(),
    )


def fp8_fp4_mega_moe_backward_dgrad_swiglu(
    gate_up_output: torch.Tensor,
    grad_h_output: torch.Tensor,
    grad_gate_up_output: torch.Tensor,
    h_act_output: torch.Tensor,
    h_weighted_output: torch.Tensor,
    x_pool_output: torch.Tensor,
    grad_x_pool_output: torch.Tensor,
    l1_acts: torch.Tensor,
    l1_acts_sf: torch.Tensor,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    grad_ye: torch.Tensor,
    route_weights: torch.Tensor,
    w2_weights: Tuple[torch.Tensor, torch.Tensor],
    w2_dequant_scratch: torch.Tensor,
    w13_weights: Tuple[torch.Tensor, torch.Tensor],
    w13_dequant_scratch: torch.Tensor,
    expert_counts: torch.Tensor,
    grid_sync_counter: torch.Tensor,
    activation_limit: float,
    block_m: int,
    compute_w13_dgrad: bool = True,
    direct_remote_grad_x: bool = False,
    write_grad_x_pool: bool = True,
    clear_wgrad_padding: bool = True,
    sym_buffer: Optional[Any] = None,
    grad_y: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    token_src_metadata: Optional[torch.Tensor] = None,
    route_weight_mode: RouteWeightMode = RouteWeightMode.PRE_DOWN,
    grad_y_unweighted_output: Optional[torch.Tensor] = None,
    down_unweighted_output: Optional[torch.Tensor] = None,
    grad_route_output: Optional[torch.Tensor] = None,
    rank_uniform_block_m: bool = False,
) -> None:
    """Run the production L1 replay, dgrad/SwiGLU, and grad-x dispatch.

    POST_DOWN writes BF16 ``score * grad_y`` to ``grad_ye``, retains
    unweighted ``h_act`` in ``h_weighted_output`` for W2 wgrad, and computes
    ``grad_route_output = dot(grad_y_unweighted, down_unweighted)``.
    """
    route_weight_mode = RouteWeightMode(route_weight_mode)
    if route_weight_mode is RouteWeightMode.POST_DOWN:
        if route_weights.dtype != torch.float32:
            raise TypeError(
                "post_down requires FP32 route_weights")
        if grad_y_unweighted_output is None:
            raise ValueError(
                "post_down requires grad_y_unweighted_output")
        if down_unweighted_output is None:
            raise ValueError(
                "post_down requires saved down_unweighted_output")
        if grad_route_output is None and sym_buffer is None:
            raise ValueError(
                "post_down without a symmetric buffer requires "
                "grad_route_output")
    elif grad_y_unweighted_output is None:
        grad_y_unweighted_output = grad_ye
    backward_grad_y = None
    backward_topk_weights = None
    backward_grad_route = None
    backward_sym_buffer_ptrs = []
    backward_rank = 0
    num_max_tokens_per_rank = 0
    num_topk = 0
    if (
        compute_w13_dgrad and
        not write_grad_x_pool and
        not direct_remote_grad_x
    ):
        raise ValueError("grad-x requires a local or direct remote output")
    if direct_remote_grad_x and not compute_w13_dgrad:
        raise ValueError("direct remote grad-x requires W13 dgrad")
    if direct_remote_grad_x and sym_buffer is None:
        raise ValueError("direct remote grad-x requires a symmetric buffer")
    if sym_buffer is not None:
        if grad_y is None or topk_weights is None or token_src_metadata is None:
            raise ValueError("in-kernel EP dispatch requires grad_y, topk_weights, " "and token_src_metadata")
        num_tokens = grad_y.shape[0]
        sym_buffer.backward_grad_y[:num_tokens].copy_(grad_y.to(torch.bfloat16).contiguous())
        sym_buffer.topk_weights[:num_tokens].copy_(topk_weights.float().contiguous())
        sym_buffer.backward_grad_route.zero_()
        backward_grad_y = sym_buffer.backward_grad_y
        backward_topk_weights = sym_buffer.topk_weights
        backward_grad_route = sym_buffer.backward_grad_route
        backward_sym_buffer_ptrs = sym_buffer.handle.buffer_ptrs
        backward_rank = sym_buffer.group.rank()
        num_max_tokens_per_rank = sym_buffer.num_max_tokens_per_rank
        num_topk = sym_buffer.num_topk
        if grad_route_output is None:
            grad_route_output = torch.empty(
                route_weights.numel(),
                dtype=torch.float32,
                device=route_weights.device,
            )
        if sym_buffer.group.size() > 1 and not rank_uniform_block_m:
            rank_uniform_block_m_tensor = torch.tensor(
                block_m,
                dtype=torch.int32,
                device=grad_ye.device,
            )
            dist.all_reduce(
                rank_uniform_block_m_tensor,
                op=dist.ReduceOp.MAX,
                group=sym_buffer.group,
            )
            block_m = int(rank_uniform_block_m_tensor.item())

    _C.fp8_fp4_mega_moe_backward_dgrad_swiglu_v2(
        gate_up_output,
        grad_h_output,
        grad_gate_up_output,
        h_act_output,
        h_weighted_output,
        x_pool_output,
        grad_x_pool_output,
        l1_acts,
        l1_acts_sf,
        l1_weights,
        grad_ye,
        route_weights,
        w2_weights,
        w2_dequant_scratch,
        w13_weights,
        w13_dequant_scratch,
        expert_counts,
        grid_sync_counter,
        activation_limit,
        compute_w13_dgrad,
        block_m,
        direct_remote_grad_x,
        write_grad_x_pool,
        clear_wgrad_padding,
        backward_grad_y,
        backward_topk_weights,
        backward_grad_route,
        token_src_metadata,
        backward_sym_buffer_ptrs,
        backward_rank,
        num_max_tokens_per_rank,
        num_topk,
        route_weight_mode.value,
        grad_y_unweighted_output,
        down_unweighted_output,
        grad_route_output,
    )


class MegaMoESideLoraBackwardResult(NamedTuple):
    """Outputs of the native side-LoRA training backward.

    ``grad_side_lora`` uses the conventional trainable layouts supplied to
    :func:`transform_side_lora_for_mega_moe`, not the transposed inference
    layouts. Shared A1/A3/B2 gradients are EP-local partials, matching the
    caller contract that reduces replicated 2-D factors across EP ranks.
    The frozen base W1/W2/W3 gradients are deliberately absent.
    """

    grad_x_pool: torch.Tensor
    grad_x: Optional[torch.Tensor]
    grad_route: torch.Tensor
    grad_ye: torch.Tensor
    grad_h: torch.Tensor
    grad_gate_up: torch.Tensor
    h_act: torch.Tensor
    h_weighted: torch.Tensor
    x_pool: torch.Tensor
    route_weights: torch.Tensor
    t13: torch.Tensor
    t2: torch.Tensor
    grad_side_lora: Tuple[torch.Tensor, ...]


def _allocate_side_lora_backward_outputs(
    gate_up: torch.Tensor,
    side_lora: Tuple[torch.Tensor, ...],
    hidden: int,
    intermediate_hidden: int,
    write_grad_x_pool: bool = True,
    reuse_gate_for_grad_x: bool = False,
    saved_x_pool: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, ...]:
    if not write_grad_x_pool:
        raise ValueError("side-LoRA backward requires write_grad_x_pool=True")
    pool_rows = gate_up.size(0)
    options = dict(dtype=torch.bfloat16, device=gate_up.device)
    grad_ye = torch.empty((pool_rows, hidden), **options)
    grad_h = torch.empty((pool_rows, intermediate_hidden), **options)
    grad_gate_up = torch.empty_like(gate_up)
    h_act = torch.empty_like(grad_h)
    h_weighted = torch.empty_like(grad_h)
    if saved_x_pool is not None:
        if saved_x_pool.dtype != torch.bfloat16:
            raise TypeError("saved_x_pool must be BF16")
        if saved_x_pool.device != gate_up.device:
            raise ValueError("saved_x_pool must be on the same device as gate_up")
        if not saved_x_pool.is_contiguous():
            raise ValueError("saved_x_pool must be contiguous")
        if tuple(saved_x_pool.shape) != (pool_rows, hidden):
            raise ValueError(
                "saved_x_pool must have shape "
                f"{(pool_rows, hidden)}; got {tuple(saved_x_pool.shape)}"
            )
        x_pool = saved_x_pool
    else:
        x_pool = torch.empty_like(grad_ye)
    if reuse_gate_for_grad_x:
        if gate_up.numel() != pool_rows * hidden:
            raise ValueError(
                "gate_up storage cannot cover the grad-x pool"
            )
        grad_x_pool = gate_up.view(pool_rows, hidden)
    else:
        grad_x_pool = torch.empty_like(grad_ye)
    route_weights = torch.empty(
        pool_rows, dtype=torch.float32, device=gate_up.device)
    grad_route = torch.empty_like(route_weights)
    t13 = torch.empty((pool_rows, 2, 128), **options)
    t2 = torch.empty((pool_rows, 128), **options)
    # Every inference tensor is K-major. Transposing only its shape yields the
    # conventional optimizer layout without performing a runtime data copy.
    # Some expert-local output tiles can be structurally empty for a routing
    # realization.  The native wgrad kernels overwrite every visited tile but
    # intentionally skip those empty tiles, so these buffers must start at
    # zero rather than exposing allocator contents to the optimizer.
    grad_side_lora = tuple(
        torch.zeros(
            (*tensor.shape[:-2], tensor.size(-1), tensor.size(-2)),
            dtype=torch.bfloat16,
            device=gate_up.device,
        )
        for tensor in side_lora
    )
    return (
        grad_ye, grad_h, grad_gate_up, h_act, h_weighted, x_pool,
        grad_x_pool, route_weights, grad_route, t13, t2,
        grad_side_lora,
    )


def _side_backward_block_m(
    block_m: int,
    sym_buffer: Any,
    device: torch.device,
    rank_uniform_block_m: bool,
) -> int:
    """Match ordinary backward's rank-uniform persistent-grid contract.

    Callers that retained a rank-uniform forward tile may opt out, avoiding
    a duplicate collective and host scalar read (including during capture).
    Otherwise the MAX reduction must precede both prelude and native wave.
    """
    if rank_uniform_block_m or sym_buffer.group.size() == 1:
        return block_m
    uniform = torch.tensor(block_m, dtype=torch.int32, device=device)
    dist.all_reduce(uniform, op=dist.ReduceOp.MAX, group=sym_buffer.group)
    return int(uniform.item())


def bf16_mega_moe_side_lora_backward(
    gate_up_output: torch.Tensor,
    saved_h: torch.Tensor,
    saved_down_unweighted: torch.Tensor,
    q13: torch.Tensor,
    q2: torch.Tensor,
    side_lora: Tuple[torch.Tensor, ...],
    w2_weights: torch.Tensor,
    w13_weights: torch.Tensor,
    expert_counts: torch.Tensor,
    padded_expert_counts: torch.Tensor,
    grad_y: torch.Tensor,
    sym_buffer: Any,
    block_m: int,
    activation_limit: float = float("inf"),
    activation: str = "swiglu",
    fast_math: bool = False,
    route_weight_mode: RouteWeightMode = RouteWeightMode.PRE_DOWN,
    combine_order_mode: CombineOrderMode = CombineOrderMode.FIXED_TOPK,
    side_lora_scale: float = 1.0,
    direct_remote_grad_x: Optional[bool] = None,
    combine_grad_x: bool = True,
    write_grad_x_pool: bool = True,
    out: Optional[MegaMoESideLoraBackwardResult] = None,
    grid_sync_counter: Optional[torch.Tensor] = None,
    expert_psum_rows: Optional[torch.Tensor] = None,
    saved_x_pool: Optional[torch.Tensor] = None,
    *,
    rank_uniform_block_m: bool = False,
) -> MegaMoESideLoraBackwardResult:
    """Run the dedicated BF16 base-dgrad + rank-128 LoRA backward.

    The native path produces six adapter gradients and intentionally performs
    no frozen base-weight gradients. It also avoids hidden-width LoRA delta or
    gradient scratch buffers. ``w13_weights`` uses the same 8-row gate/up
    interleave returned by :func:`transform_weights_for_mega_moe`; the
    internal ``grad_gate_up`` result is emitted in that matching interleave.

    ``rank_uniform_block_m=True`` requires a caller-retained forward tile
    proven identical across all EP ranks, including recomputed forwards. It
    skips the default MAX collective/host read, as in ordinary backward, and
    is the capture-safe path; never set it for a local token-derived tile.
    """
    route_weight_mode = RouteWeightMode(route_weight_mode)
    combine_order_mode = CombineOrderMode(combine_order_mode)
    if route_weight_mode is RouteWeightMode.POST_DOWN:
        raise NotImplementedError(
            "BF16 side-LoRA backward currently supports pre_down routing only")
    if len(side_lora) != 6 or side_lora[0].size(0) != 128:
        raise ValueError("side_lora must be a rank-128 transformed tuple")
    if direct_remote_grad_x is None:
        direct_remote_grad_x = sym_buffer.group.size() > 1
    if not write_grad_x_pool:
        raise ValueError("side-LoRA backward requires write_grad_x_pool=True")
    block_m = _side_backward_block_m(
        block_m, sym_buffer, grad_y.device, rank_uniform_block_m)
    # The native specialization materializes the base dgrad pool, folds both
    # side branches into it, then publishes the final value once. This is the
    # ordinary base-dgrad output, not an additional side-LoRA wide scratch.
    outputs = (
        _allocate_side_lora_backward_outputs(
            gate_up_output, side_lora, w13_weights.size(2),
            w2_weights.size(2), write_grad_x_pool,
            saved_x_pool=saved_x_pool)
        if out is None else (
            out.grad_ye, out.grad_h, out.grad_gate_up, out.h_act,
            out.h_weighted, out.x_pool, out.grad_x_pool,
            out.route_weights, out.grad_route, out.t13, out.t2,
            out.grad_side_lora)
    )
    (grad_ye, grad_h, grad_gate_up, h_act, h_weighted, x_pool,
     grad_x_pool, route_weights, grad_route, t13, t2,
     grad_side_lora) = outputs
    if saved_x_pool is not None and x_pool.data_ptr() != saved_x_pool.data_ptr():
        raise ValueError("out.x_pool must alias saved_x_pool when it is provided")
    sym_buffer.backward_grad_y[:grad_y.size(0)].copy_(
        grad_y.to(torch.bfloat16).contiguous())
    sym_buffer.backward_grad_route.zero_()
    num_grid_states = (
        expert_counts.numel() *
        ((w13_weights.size(2) // 64) * (w2_weights.size(2) // 128) +
         (w13_weights.size(1) // 64) * (w13_weights.size(2) // 128)) + 2)
    if grid_sync_counter is None:
        grid_sync_counter = torch.zeros(
            num_grid_states, dtype=torch.int32, device=gate_up_output.device)
    if expert_psum_rows is None:
        expert_psum_rows = padded_expert_counts.cumsum(0).to(torch.int32)
    # Publish and pull grad-y/x/route metadata before the rank shrink. The
    # following persistent specialization is told that dispatch is complete,
    # so it does not repeat communication. This is a native CUDA prelude; no
    # framework GEMM participates in the production backward.
    _C.bf16_mega_moe_backward_post_down_prelude_v2(
        grad_ye, grad_ye, x_pool, route_weights, grad_route,
        saved_down_unweighted, expert_counts,
        sym_buffer.backward_grad_y, sym_buffer.x,
        sym_buffer.topk_weights, sym_buffer.backward_grad_route,
        sym_buffer.token_src_metadata, sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(), sym_buffer.num_topk, block_m,
        combine_order_mode.value, True, False, False, True, True,
        False, saved_x_pool is not None, 256)
    kernel_trace_hook = _BF16_BACKWARD_KERNEL_TRACE_HOOK
    kernel_trace = None
    trace_clock_rate_khz = 0.0
    if kernel_trace_hook is not None:
        kernel_trace, trace_clock_rate_khz = (
            _allocate_backward_kernel_trace(
                _BF16_SIDE_LORA_BACKWARD_KERNEL_TRACE_SITES,
                grad_y.device,
            )
        )
    _C.bf16_mega_moe_side_lora_backward(
        gate_up_output, grad_h, grad_gate_up, h_act, h_weighted,
        x_pool, grad_x_pool, grad_route, grad_ye, grad_ye,
        route_weights, w2_weights, w13_weights, expert_counts,
        grid_sync_counter, float(activation_limit), activation,
        bool(fast_math), route_weight_mode.value,
        combine_order_mode.value, saved_down_unweighted, block_m,
        bool(direct_remote_grad_x),
        bool(write_grad_x_pool), True,
        sym_buffer.backward_grad_y, sym_buffer.x,
        sym_buffer.topk_weights, sym_buffer.backward_grad_route,
        sym_buffer.token_src_metadata, sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(), sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_topk, "dispatch_prepared", *side_lora, q13, q2, saved_h,
        t13, t2, *grad_side_lora, expert_psum_rows,
        padded_expert_counts, float(side_lora_scale), kernel_trace)
    if kernel_trace_hook is not None:
        kernel_trace_hook(
            kernel_trace,
            trace_clock_rate_khz,
            expert_counts,
            block_m,
        )
    grad_x = None
    if direct_remote_grad_x and combine_grad_x:
        grad_x = (
            torch.empty_like(grad_y, dtype=torch.bfloat16)
            if out is None or out.grad_x is None else out.grad_x)
        mega_moe_backward_combine_grad_x(
            grad_x, sym_buffer, combine_order_mode)
    return MegaMoESideLoraBackwardResult(
        grad_x_pool, grad_x, grad_route, grad_ye, grad_h,
        grad_gate_up, h_act, h_weighted, x_pool, route_weights,
        t13, t2, grad_side_lora)


def fp8_fp4_mega_moe_side_lora_backward(
    gate_up_output: torch.Tensor,
    saved_h: torch.Tensor,
    saved_down_unweighted: torch.Tensor,
    q13: torch.Tensor,
    q2: torch.Tensor,
    side_lora: Tuple[torch.Tensor, ...],
    l1_acts: torch.Tensor,
    l1_acts_sf: torch.Tensor,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    w13_weights: Tuple[torch.Tensor, torch.Tensor],
    w2_weights: Tuple[torch.Tensor, torch.Tensor],
    w13_dequant_scratch: torch.Tensor,
    w2_dequant_scratch: torch.Tensor,
    expert_counts: torch.Tensor,
    padded_expert_counts: torch.Tensor,
    grad_y: torch.Tensor,
    sym_buffer: Any,
    block_m: int,
    activation_limit: float = float("inf"),
    activation: str = "swiglu",
    fast_math: bool = False,
    route_weight_mode: RouteWeightMode = RouteWeightMode.PRE_DOWN,
    side_lora_scale: float = 1.0,
    direct_remote_grad_x: Optional[bool] = None,
    combine_grad_x: bool = True,
    write_grad_x_pool: bool = True,
    out: Optional[MegaMoESideLoraBackwardResult] = None,
    grid_sync_counter: Optional[torch.Tensor] = None,
    expert_psum_rows: Optional[torch.Tensor] = None,
    reuse_gate_for_grad_x: bool = False,
    saved_x_pool: Optional[torch.Tensor] = None,
    *,
    rank_uniform_block_m: bool = False,
) -> MegaMoESideLoraBackwardResult:
    """Run the dedicated MXFP4 base-dgrad + BF16 side-LoRA backward.

    ``rank_uniform_block_m=True`` has the same caller-proven forward-tile
    contract and capture-safe collective bypass as the BF16 side wrapper.
    """
    if activation not in ("swiglu", "geglu"):
        raise ValueError(f"unsupported activation: {activation}")
    route_weight_mode = RouteWeightMode(route_weight_mode)
    if route_weight_mode is RouteWeightMode.POST_DOWN:
        raise NotImplementedError(
            "MXFP4 side-LoRA backward currently supports pre_down routing only")
    if len(side_lora) != 6 or side_lora[0].size(0) != 128:
        raise ValueError("side_lora must be a rank-128 transformed tuple")
    if direct_remote_grad_x is None:
        direct_remote_grad_x = sym_buffer.group.size() > 1
    if not write_grad_x_pool:
        raise ValueError("side-LoRA backward requires write_grad_x_pool=True")
    block_m = _side_backward_block_m(
        block_m, sym_buffer, grad_y.device, rank_uniform_block_m)
    hidden = w2_weights[0].size(1)
    intermediate_hidden = w2_dequant_scratch.size(2)
    expected_down_shape = (gate_up_output.size(0), hidden)
    if saved_down_unweighted.dtype != torch.bfloat16:
        raise TypeError("saved_down_unweighted must be BF16")
    if saved_down_unweighted.device != gate_up_output.device:
        raise ValueError(
            "saved_down_unweighted must be on the same device as "
            "gate_up_output"
        )
    if not saved_down_unweighted.is_contiguous():
        raise ValueError("saved_down_unweighted must be contiguous")
    if tuple(saved_down_unweighted.shape) != expected_down_shape:
        raise ValueError(
            "saved_down_unweighted must cover the full route pool with "
            f"shape {expected_down_shape}; got "
            f"{tuple(saved_down_unweighted.shape)}"
        )
    outputs = (
        _allocate_side_lora_backward_outputs(
            gate_up_output, side_lora, hidden, intermediate_hidden,
            write_grad_x_pool,
            reuse_gate_for_grad_x=reuse_gate_for_grad_x,
            saved_x_pool=saved_x_pool)
        if out is None else (
            out.grad_ye, out.grad_h, out.grad_gate_up, out.h_act,
            out.h_weighted, out.x_pool, out.grad_x_pool,
            out.route_weights, out.grad_route, out.t13, out.t2,
            out.grad_side_lora)
    )
    (grad_ye, grad_h, grad_gate_up, h_act, h_weighted, x_pool,
     grad_x_pool, route_weights, grad_route, t13, t2,
     grad_side_lora) = outputs
    if saved_x_pool is not None and x_pool.data_ptr() != saved_x_pool.data_ptr():
        raise ValueError("out.x_pool must alias saved_x_pool when it is provided")
    if reuse_gate_for_grad_x:
        if (
            grad_x_pool.data_ptr() != gate_up_output.data_ptr()
            or grad_x_pool.numel() != gate_up_output.numel()
        ):
            raise ValueError(
                "reuse_gate_grad_x requires grad_x_pool to reuse gate_up"
            )
    sym_buffer.backward_grad_y[:grad_y.size(0)].copy_(
        grad_y.to(torch.bfloat16).contiguous())
    sym_buffer.backward_grad_route.zero_()
    num_grid_states = (
        expert_counts.numel() *
        ((hidden // 64) * (intermediate_hidden // 128) +
         ((2 * intermediate_hidden) // 64) * (hidden // 128)) + 2)
    if grid_sync_counter is None:
        grid_sync_counter = torch.zeros(
            num_grid_states, dtype=torch.int32, device=gate_up_output.device)
    if expert_psum_rows is None:
        expert_psum_rows = padded_expert_counts.cumsum(0).to(torch.int32)
    _C.bf16_mega_moe_backward_post_down_prelude_v2(
        grad_ye, grad_ye, x_pool, route_weights, grad_route,
        saved_down_unweighted, expert_counts,
        sym_buffer.backward_grad_y, sym_buffer.side_lora_source,
        sym_buffer.topk_weights, sym_buffer.backward_grad_route,
        sym_buffer.token_src_metadata, sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(), sym_buffer.num_topk, block_m,
        CombineOrderMode.FIXED_TOPK.value, True, False,
        False, True, True, False, saved_x_pool is not None, 256)
    _C.fp8_fp4_mega_moe_side_lora_backward(
        gate_up_output, grad_h, grad_gate_up, h_act, h_weighted,
        x_pool, grad_x_pool, l1_acts, l1_acts_sf,
        l1_weights[0], l1_weights[1], grad_ye, route_weights,
        w2_weights[0], w2_weights[1], w2_dequant_scratch,
        w13_weights[0], w13_weights[1], w13_dequant_scratch,
        expert_counts, grid_sync_counter, float(activation_limit),
        activation, bool(fast_math), True,
        bool(direct_remote_grad_x),
        bool(write_grad_x_pool), True, block_m,
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(), sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_topk,
        sym_buffer.backward_grad_y, sym_buffer.topk_weights,
        sym_buffer.backward_grad_route, sym_buffer.token_src_metadata,
        route_weight_mode.value, grad_ye, saved_down_unweighted,
        grad_route, *side_lora, q13, q2, saved_h, t13, t2,
        *grad_side_lora, expert_psum_rows, padded_expert_counts,
        float(side_lora_scale))
    grad_x = None
    if direct_remote_grad_x and combine_grad_x:
        grad_x = (
            torch.empty_like(grad_y, dtype=torch.bfloat16)
            if out is None or out.grad_x is None else out.grad_x)
        mega_moe_backward_combine_grad_x(
            grad_x, sym_buffer, CombineOrderMode.FIXED_TOPK)
    return MegaMoESideLoraBackwardResult(
        grad_x_pool, grad_x, grad_route, grad_ye, grad_h,
        grad_gate_up, h_act, h_weighted, x_pool, route_weights,
        t13, t2, grad_side_lora)


def bf16_mega_moe_backward_w2(
    grad_w2_output: torch.Tensor,
    grad_ye: torch.Tensor,
    h_weighted: torch.Tensor,
    padded_expert_counts: torch.Tensor,
    pool_block_m: int,
    route_weight_mode: RouteWeightMode = RouteWeightMode.PRE_DOWN,
) -> None:
    """Run standalone single-CTA BF16 W2 wgrad for local experts."""
    route_weight_mode = RouteWeightMode(route_weight_mode)
    _C.bf16_mega_moe_backward_w2(
        grad_w2_output,
        grad_ye,
        h_weighted,
        padded_expert_counts,
        pool_block_m,
        route_weight_mode.value,
    )


def bf16_mega_moe_backward_w13(
    grad_w13_output: torch.Tensor,
    grad_gate_up: torch.Tensor,
    x_pool: torch.Tensor,
    padded_expert_counts: torch.Tensor,
    pool_block_m: int,
) -> None:
    """Run standalone single-CTA BF16 combined W1/W3 wgrad."""
    _C.bf16_mega_moe_backward_w13(
        grad_w13_output,
        grad_gate_up,
        x_pool,
        padded_expert_counts,
        pool_block_m,
    )


def _direct_grad_x_planes(sym_buffer: Any) -> torch.Tensor:
    return torch.as_strided(
        sym_buffer.backward_grad_y,
        size=(
            sym_buffer.num_topk * sym_buffer.num_max_tokens_per_rank,
            sym_buffer.hidden,
        ),
        stride=(sym_buffer.hidden, 1),
    )


def mega_moe_backward_combine_grad_x(
    grad_x_output: torch.Tensor,
    sym_buffer: Any,
    combine_order_mode: CombineOrderMode = CombineOrderMode.FIXED_TOPK,
) -> None:
    """Reduce direct-write fixed-slot grad-x planes without a wgrad kernel."""
    combine_order_mode = CombineOrderMode(combine_order_mode)
    num_ranks = sym_buffer.group.size()
    if sym_buffer.num_experts % num_ranks:
        raise ValueError("num_experts must be divisible by the EP group size")
    _C.mega_moe_backward_combine_grad_x(
        grad_x_output,
        _direct_grad_x_planes(sym_buffer),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_topk,
        num_ranks,
        sym_buffer.num_experts // num_ranks,
        (
            None
            if combine_order_mode is CombineOrderMode.FIXED_TOPK
            else sym_buffer.topk_idx
        ),
        combine_order_mode.value,
    )


def bf16_mega_moe_backward_w2_combine(
    grad_w2_output: torch.Tensor,
    grad_ye: torch.Tensor,
    h_weighted: torch.Tensor,
    padded_expert_counts: torch.Tensor,
    pool_block_m: int,
    grad_x_output: torch.Tensor,
    sym_buffer: Any,
    route_weight_mode: RouteWeightMode = RouteWeightMode.PRE_DOWN,
) -> None:
    """Run W2 wgrad while protecting the direct-write receive planes."""
    route_weight_mode = RouteWeightMode(route_weight_mode)
    _C.bf16_mega_moe_backward_w2_combine(
        grad_w2_output,
        grad_ye,
        h_weighted,
        padded_expert_counts,
        pool_block_m,
        grad_x_output,
        _direct_grad_x_planes(sym_buffer),
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_topk,
        route_weight_mode.value,
    )


def bf16_mega_moe_backward_w13_combine(
    grad_w13_output: torch.Tensor,
    grad_gate_up: torch.Tensor,
    x_pool: torch.Tensor,
    padded_expert_counts: torch.Tensor,
    pool_block_m: int,
    grad_x_output: torch.Tensor,
    sym_buffer: Any,
    combine_order_mode: CombineOrderMode = CombineOrderMode.FIXED_TOPK,
) -> None:
    """Run W13 wgrad while reducing direct-write grad-x planes."""
    combine_order_mode = CombineOrderMode(combine_order_mode)
    if pool_block_m < 64:
        # Small expert buckets commonly use the grouped-MM wgrad path. Keep
        # the expert-major wgrad independent and consume the fixed source
        # planes with the standalone reduction.
        bf16_mega_moe_backward_w13(
            grad_w13_output,
            grad_gate_up,
            x_pool,
            padded_expert_counts,
            pool_block_m,
        )
        mega_moe_backward_combine_grad_x(
            grad_x_output,
            sym_buffer,
            combine_order_mode,
        )
        return
    _C.bf16_mega_moe_backward_w13_combine(
        grad_w13_output,
        grad_gate_up,
        x_pool,
        padded_expert_counts,
        pool_block_m,
        grad_x_output,
        _direct_grad_x_planes(sym_buffer),
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_topk,
        sym_buffer.topk_idx,
        combine_order_mode.value,
    )
