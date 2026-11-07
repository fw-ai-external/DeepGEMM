"""BF16 inference compatibility and clamped training-state ownership."""

import math
from collections.abc import Iterator

import pytest
import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.mega.backward import _validate_bf16_backward_alias_contract


def _buffers(clamped: bool) -> dict:
    def plane() -> torch.Tensor:
        return torch.empty((8, 16), dtype=torch.bfloat16)

    gate, grad_y, activation = plane(), plane(), plane()
    return dict(
        memory_mode="phase_ordered",
        route_weight_mode=deep_gemm.RouteWeightMode.PRE_DOWN,
        gate_up_output=gate,
        grad_gate_up_output=gate,
        grad_h_output=plane(),
        h_act_output=activation,
        h_weighted_output=plane() if clamped else activation,
        x_pool_output=plane(),
        grad_x_pool_output=plane(),
        grad_ye=grad_y,
        grad_y_unweighted_output=grad_y,
        down_unweighted_output=grad_y,
        activation="swiglu",
        activation_limit=10.0 if clamped else math.inf,
    )


@pytest.mark.parametrize("clamped", [False, True])
def test_phase_ordered_activation_ownership(clamped: bool) -> None:
    _validate_bf16_backward_alias_contract(**_buffers(clamped))


@pytest.mark.parametrize("violation", ["exact", "partial", "unclamped", "pool", "gate"])
def test_unsafe_activation_ownership_is_rejected(violation: str) -> None:
    args = _buffers(clamped=violation != "unclamped")
    if violation == "exact":
        args["h_weighted_output"] = args["h_act_output"]
    elif violation == "partial":
        backing = torch.empty(256, dtype=torch.bfloat16)
        args["h_act_output"] = backing[:128].view(8, 16)
        args["h_weighted_output"] = backing[64:192].view(8, 16)
    elif violation == "unclamped":
        args["h_weighted_output"] = torch.empty_like(args["h_act_output"])
    elif violation == "pool":
        args["x_pool_output"] = args["grad_h_output"]
    else:
        args["grad_gate_up_output"] = torch.empty_like(args["gate_up_output"])
    with pytest.raises(ValueError, match="overlap|exactly alias"):
        _validate_bf16_backward_alias_contract(**args)


@pytest.fixture(scope="module")
def single_rank_group(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dist.ProcessGroup]:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("BF16 MegaMoE requires a free Blackwell GPU")
    if dist.is_initialized():
        pytest.skip("this single-rank test owns its process group")
    rendezvous = tmp_path_factory.mktemp("bf16-mega") / "rendezvous"
    dist.init_process_group("nccl", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    try:
        yield dist.group.WORLD
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("route_mode", ["pre_down", "post_down"])
@pytest.mark.parametrize("activation,limit", [("swiglu", math.inf), ("swiglu", 10.0), ("geglu", 10.0)])
@torch.no_grad()
def test_fp32_is_optional_and_default_bf16_stays_graph_safe(
    single_rank_group: dist.ProcessGroup, route_mode: str, activation: str, limit: float
) -> None:
    tokens, hidden, intermediate, experts, topk = 128, 1024, 256, 8, 2
    torch.manual_seed(41)
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    w13 = torch.randn(experts, intermediate * 2, hidden, device="cuda", dtype=torch.bfloat16) / math.sqrt(hidden)
    w2 = torch.randn(experts, hidden, intermediate, device="cuda", dtype=torch.bfloat16) / math.sqrt(intermediate)
    w13, w2 = deep_gemm.transform_weights_for_mega_moe(w13, w2, activation=activation)
    buffer = deep_gemm.SymmBuffer(
        single_rank_group, experts, tokens, topk, hidden, intermediate, mma_type="bf16xbf16", activation=activation
    )
    buffer.x[:tokens].copy_(x)
    indices = torch.rand(tokens, experts, device="cuda").topk(topk, dim=-1).indices
    buffer.topk_idx[:tokens].copy_(indices)
    buffer.topk_weights[:tokens].copy_(torch.softmax(torch.randn(tokens, topk, device="cuda"), dim=-1))
    bf16 = torch.empty_like(x)
    fp32 = torch.empty_like(x, dtype=torch.float32)

    def forward(output: torch.Tensor) -> None:
        deep_gemm.bf16_mega_moe(
            output,
            w13,
            w2,
            buffer,
            activation=activation,
            activation_clamp=limit,
            route_weight_mode=deep_gemm.RouteWeightMode(route_mode),
            fast_math=False,
        )

    forward(bf16)
    forward(fp32)
    assert bf16.dtype == torch.bfloat16 and torch.isfinite(fp32).all()
    assert torch.equal(bf16, fp32.bfloat16())
    assert not torch.equal(fp32, bf16.float()), "FP32 output must retain the routed sum before BF16 rounding"
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward(bf16)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(bf16, fp32.bfloat16())
