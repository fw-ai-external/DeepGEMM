"""CPU checks for bounded-ring allocation/forward selection and legacy defaults."""

from types import SimpleNamespace

import pytest
import torch

import deep_gemm


@pytest.mark.parametrize("bounded", [False, True])
def test_buffer_and_forward_share_schedule(monkeypatch, bounded):
    group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    base = SimpleNamespace(
        buffer=torch.empty(16, dtype=torch.int8), handle=SimpleNamespace(buffer_ptrs=[1]), group=group
    )
    views = [torch.full((2, 4), i) for i in range(15)]
    sizing, forwards = [], []

    def size(label):
        def call(*args):
            sizing.append((label, args))
            return 16, lambda _: views

        return call

    monkeypatch.setattr(deep_gemm._C, "get_token_alignment_for_mega_moe", lambda: 1920)
    monkeypatch.setattr(deep_gemm._C, "get_symm_buffer_size_for_mega_moe_v3", size("legacy"))
    monkeypatch.setattr(deep_gemm._C, "get_symm_buffer_size_for_bf16_mega_moe_bounded_lag", size("bounded"))
    monkeypatch.setattr(deep_gemm._C, "bf16_mega_moe", lambda *args, **kwargs: forwards.append((args, kwargs)))
    buffer = deep_gemm.SymmBuffer(group, 4, 128, 2, 4, 4, mma_type="bf16xbf16", base=base, bounded_lag=bounded)
    assert buffer.bounded_lag == bounded
    assert sizing == [("bounded" if bounded else "legacy", (1, 4, 1920, 2, 4, 4, "bf16xbf16", "swiglu", 0, 0))]
    for length in (2, 0, 1):
        deep_gemm.bf16_mega_moe(torch.empty(length, 4), object(), object(), buffer)
    assert all(kwargs == ({"bounded_lag": True} if bounded else {}) for _, kwargs in forwards)
    assert all(args[7] == 1920 for args, _ in forwards)


@pytest.mark.parametrize("mma_type,side_lora", [("fp8xfp4", False), ("bf16xbf16", True)])
def test_rejects_unsupported_paths_before_allocation(mma_type, side_lora):
    with pytest.raises(ValueError, match="BF16 MegaMoE path"):
        deep_gemm.SymmBuffer(None, 896, 1024, 16, 3584, 3072, mma_type=mma_type, side_lora=side_lora, bounded_lag=True)


def test_requires_matching_native_build(monkeypatch):
    monkeypatch.delattr(deep_gemm._C, "get_symm_buffer_size_for_bf16_mega_moe_bounded_lag")
    with pytest.raises(RuntimeError, match="matching DeepGEMM build"):
        deep_gemm.SymmBuffer(None, 896, 1024, 16, 3584, 3072, mma_type="bf16xbf16", bounded_lag=True)


def test_factory_propagates_explicit_option(monkeypatch):
    import deep_gemm.mega as mega

    calls = []
    monkeypatch.setattr(mega, "SymmBuffer", lambda *args, **kwargs: calls.append(kwargs))
    for bounded in (False, True):
        mega.get_symm_buffer_for_mega_moe(None, 896, 1024, 16, 3584, 3072, mma_type="bf16xbf16", bounded_lag=bounded)
    assert [c["bounded_lag"] for c in calls] == [False, True]
