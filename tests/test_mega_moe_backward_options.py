"""CPU contracts for explicit backward scheduling and collective order."""

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
from deep_gemm.mega import CombineOrderMode, RouteWeightMode

BACKWARD = Path(__file__).resolve().parents[1] / "deep_gemm/mega/backward.py"


def load(path: Path, suffix: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location("deep_gemm.mega." + suffix, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def invoke(
    module: ModuleType, hidden: int, calls: list[tuple[str, tuple]] | None = None, **options: Any
) -> list[tuple[str, tuple]]:
    calls = [] if calls is None else calls
    module._C = SimpleNamespace(
        bf16_mega_moe_backward_post_down_prelude_v2=lambda *args: calls.append(("prelude", args[17:])),
        bf16_mega_moe_backward_dgrad_v2=lambda *args, **kwargs: calls.append(
            ("dgrad", (kwargs["vectorized_grad_x_store"], kwargs["wide_grad_x_store"]))
        ),
    )
    # Real CPU tensors retain the wrapper's alias/lifetime checks. Only the
    # native GPU launches are intercepted; no numerical result is claimed.
    gate = torch.empty(2, 8, dtype=torch.bfloat16)
    weighted = torch.empty(2, hidden, dtype=torch.bfloat16)
    unweighted = torch.empty_like(weighted)
    grad_h = unweighted.flatten()[:8].view(2, 4)
    buffer = SimpleNamespace(
        group=SimpleNamespace(size=lambda: 1, rank=lambda: 0),
        backward_grad_y=torch.empty_like(weighted),
        backward_grad_route=torch.empty(2, 1),
        x=object(),
        topk_weights=object(),
        token_src_metadata=object(),
        handle=SimpleNamespace(buffer_ptrs=[1]),
        num_topk=1,
        num_max_tokens_per_rank=2,
        hidden=hidden,
    )
    module.bf16_mega_moe_backward_dgrad(
        gate,
        grad_h,
        gate,
        grad_h,
        grad_h,
        torch.empty_like(weighted),
        torch.empty_like(weighted),
        torch.empty(2),
        weighted,
        torch.empty(2),
        torch.empty(1),
        torch.empty(1),
        torch.empty(1, dtype=torch.int32),
        torch.empty(1, dtype=torch.int32),
        torch.ones_like(weighted),
        buffer,
        float("inf"),
        128,
        direct_remote_grad_x=False,
        route_weight_mode=RouteWeightMode.POST_DOWN,
        grad_y_unweighted_output=unweighted,
        down_unweighted_output=weighted,
        combine_order_mode=CombineOrderMode.DEEPEP,
        memory_mode="phase_ordered",
        **options,
    )
    return calls


def expected_calls(
    hidden: int, fused: bool = True, threads: int = 128, vectorized: bool = True, wide: bool = False
) -> list[tuple[str, tuple]]:
    route_threads = threads if hidden == 2048 else 256
    if fused:
        schedule = [
            (False, False, False, True, False, True, False, 256),
            (True, True, True, False, False, False, False, route_threads),
            (False, False, False, False, True, True, False, 256),
        ]
    else:
        schedule = [
            (True, False, False, True, True, False, False, 256),
            (False, True, True, False, False, False, False, route_threads),
        ]
    return [("prelude", arguments) for arguments in schedule] + [("dgrad", (vectorized, wide))]


@pytest.mark.parametrize("hidden", [2048, 3584])
def test_defaults_ignore_legacy_environment(monkeypatch: pytest.MonkeyPatch, hidden: int) -> None:
    monkeypatch.setenv("DG_BF16_SPLIT_BARRIER_FUSED_ROUTE", "0")
    monkeypatch.setenv("DG_BF16_ROUTE_PRELUDE_THREADS", "invalid")
    monkeypatch.setenv("DG_BF16_MEGA_MOE_VECTORIZED_GRAD_X_STORE", "0")
    monkeypatch.setenv("DG_BF16_MEGA_MOE_WIDE_GRAD_X_STORE", "1")
    module = load(BACKWARD, "candidate_defaults")
    assert invoke(module, hidden) == expected_calls(hidden)


@pytest.mark.parametrize("hidden", [2048, 3584])
@pytest.mark.parametrize("threads", [128, 256])
@pytest.mark.parametrize("fused", [True, False])
def test_explicit_controls_preserve_collective_order(
    monkeypatch: pytest.MonkeyPatch, hidden: int, threads: int, fused: bool
) -> None:
    monkeypatch.setenv("DG_BF16_SPLIT_BARRIER_FUSED_ROUTE", "0" if fused else "1")
    monkeypatch.setenv("DG_BF16_ROUTE_PRELUDE_THREADS", "256" if threads == 128 else "128")
    module = load(BACKWARD, "candidate_explicit")
    assert invoke(module, hidden, route_prelude_threads=threads, split_barrier_fused_route=fused) == expected_calls(
        hidden, fused, threads
    )


@pytest.mark.parametrize("invalid", [0, 64, 192, "128"])
def test_invalid_threads_fail_before_native_launch(invalid: object) -> None:
    module = load(BACKWARD, "candidate_invalid")
    calls = []
    with pytest.raises(ValueError, match="route_prelude_threads must be 128 or 256"):
        invoke(module, 2048, calls=calls, route_prelude_threads=invalid)
    assert not calls


@pytest.mark.parametrize("hidden", [2048, 3584])
def test_store_options_can_change_between_calls_without_environment_state(
    monkeypatch: pytest.MonkeyPatch, hidden: int
) -> None:
    module = load(BACKWARD, "candidate_store_options")
    for vectorized, wide in ((False, False), (True, True), (False, True), (True, False)):
        monkeypatch.setenv("DG_BF16_MEGA_MOE_VECTORIZED_GRAD_X_STORE", "0" if vectorized else "1")
        monkeypatch.setenv("DG_BF16_MEGA_MOE_WIDE_GRAD_X_STORE", "0" if wide else "1")
        assert invoke(module, hidden, vectorized_grad_x_store=vectorized, wide_grad_x_store=wide) == expected_calls(
            hidden, vectorized=vectorized, wide=wide
        )
