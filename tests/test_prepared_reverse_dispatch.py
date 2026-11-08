"""Run the real Python dispatch/alias control flow with CPU pool storage.

Stop at Kernel A: these checks cover publication and lifetime contracts, not
CUDA arithmetic or distributed synchronization. Those require the EP4 test.
"""
import ast
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
import math

import pytest
import torch


class RouteWeightMode(str, Enum):
    PRE_DOWN = "pre_down"
    POST_DOWN = "post_down"


class CombineOrderMode(str, Enum):
    FIXED_TOPK = "fixed_topk"


class KernelReached(Exception):
    pass


def runtime():
    source = Path(__file__).parents[1] / "deep_gemm/mega/backward.py"
    tree = ast.parse(source.read_text())
    names = {"_storage_byte_range", "_exact_alias", "_validate_bf16_backward_alias_contract",
             "bf16_mega_moe_backward_dgrad"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(functions) == len(names)
    namespace = dict(torch=torch, math=math, Optional=Optional, Any=Any,
                     RouteWeightMode=RouteWeightMode, CombineOrderMode=CombineOrderMode,
                     _BF16_BACKWARD_KERNEL_TRACE_HOOK=None,
                     _BF16_BACKWARD_DECOMPOSE_PRELUDE=False,
                     _timed_cuda_phase=lambda name, fn: fn())
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def arguments():
    gate = torch.full((4, 8), 3, dtype=torch.bfloat16)
    grad = torch.full((4, 4), 11, dtype=torch.bfloat16)
    h = torch.zeros_like(grad)
    group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    sym = SimpleNamespace(backward_grad_y=torch.full((8, 4), 7, dtype=torch.bfloat16),
                          backward_grad_route=torch.ones(4, 2), group=group,
                          num_topk=2, num_max_tokens_per_rank=4, hidden=4,
                          x=torch.ones_like(grad), topk_weights=torch.ones(4, 2),
                          token_src_metadata=torch.zeros(4, 3, dtype=torch.int32),
                          handle=SimpleNamespace(buffer_ptrs=[1, 2]))
    return dict(gate_up_output=gate, grad_gate_up_output=gate, grad_h_output=torch.zeros_like(h),
                h_act_output=h, h_weighted_output=h, x_pool_output=torch.ones_like(grad),
                grad_x_pool_output=None, grad_route_output=torch.empty(4), grad_ye=grad,
                route_weights=torch.full((4,), .5), w2_weights=torch.empty(1, 4, 4),
                w13_weights=torch.empty(1, 8, 4), expert_counts=torch.tensor([4], dtype=torch.int32),
                grid_sync_counter=torch.zeros(2, dtype=torch.int32),
                grad_y=torch.full_like(grad, 9), sym_buffer=sym, activation_limit=float("inf"),
                block_m=128, write_grad_x_pool=False, memory_mode="phase_ordered",
                rank_uniform_block_m=True, x_prepared=True)


@pytest.mark.parametrize("prepared", [False, True])
def test_prepared_reuse_keeps_rank_barriers_and_route_clear(prepared):
    ns, args = runtime(), arguments()
    sym = args["sym_buffer"]
    calls = []

    def prelude(*values):
        assert torch.count_nonzero(sym.backward_grad_route) == 0
        calls.append(values)
        # No dispatch/copy is allowed on the reuse path. Both rank barriers
        # must still execute after the route-plane clear and before Kernel A.
        assert values[17:] == (not prepared, False, False, True, True, prepared, True, 256)

    def kernel(*values):
        assert len(calls) == 1
        assert values[8] is args["grad_ye"]
        assert values[10] is args["route_weights"]
        raise KernelReached

    ns["_C"] = SimpleNamespace(bf16_mega_moe_backward_post_down_prelude_v2=prelude,
                               bf16_mega_moe_backward_dgrad_v2=kernel)
    before_grad = args["grad_ye"].clone()
    before_weights = args["route_weights"].clone()
    with pytest.raises(KernelReached):
        ns["bf16_mega_moe_backward_dgrad"](**args, reverse_dispatch_prepared=prepared)
    torch.testing.assert_close(args["grad_ye"], before_grad, rtol=0, atol=0)
    torch.testing.assert_close(args["route_weights"], before_weights, rtol=0, atol=0)
    assert torch.all(sym.backward_grad_y[:4] == (7 if prepared else 9))
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("override", [dict(x_prepared=False), dict(memory_mode="legacy"),
                                       dict(route_weight_mode=RouteWeightMode.POST_DOWN)])
def test_incompatible_prepared_contract_rejected_before_source_mutation(override):
    ns, args = runtime(), arguments()
    args.update(override)
    source = args["sym_buffer"].backward_grad_route
    with pytest.raises(ValueError, match="prepared"):
        ns["bf16_mega_moe_backward_dgrad"](**args, reverse_dispatch_prepared=True)
    assert torch.all(source == 1)
