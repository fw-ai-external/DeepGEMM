"""Communication-free (one-rank) MegaMoE: host-free planning for the local kernel mode.

On a one-rank buffer the kernels skip in-kernel token counting, every grid-wide
barrier and the final cleanup. Pull, task and combine work are claimed through
atomic counters in one global order where each task depends only on tasks claimed
before it, so the kernel finishes with any number of co-resident CTAs and cannot
deadlock against concurrent kernels that hold SMs.

The caller replaces the removed in-kernel work with LocalPlan.plan on the launch
stream before every launch: per-expert counts, the expert-major route table and
the reset of the claim counters and ring signals. It does not synchronize with
the host.
"""

import torch

# layout/mega_moe.cuh: MegaMoESignals
_HEAD_BYTES = 128
_MAX_RANKS = 72
_MAX_EXPERTS = 2048
_MAX_EXPERTS_PER_RANK = 512
_MIN_BLOCK_M = 8


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class LocalPlan:
    """Workspace views of a one-rank MegaMoE buffer and the per-launch plan."""

    def __init__(self, sym_buffer):
        if sym_buffer.group.size() != 1:
            raise ValueError("MegaMoE local mode requires a one-rank buffer")
        base = sym_buffer.buffer
        experts, max_tokens = sym_buffer.num_experts, sym_buffer.num_max_tokens_per_rank
        peer = _align(_HEAD_BYTES + _MAX_RANKS * 8, 128)
        send = _align(peer + _MAX_RANKS * 8, 128)
        recv = send + _MAX_EXPERTS * 8
        recv_sum = recv + _MAX_EXPERTS * 8
        # Ring and shared-expert signals follow the control prefix, sized from capacity
        # (Workspace::get_num_signal_bytes): l1_full, l1_empty, l2_full_mask (2 strides), l2_empty, shared.
        ring_blocks = -(-sym_buffer.num_ring_tokens // _MIN_BLOCK_M)
        shared_blocks = -(-max_tokens // _MIN_BLOCK_M)
        stride = _align(ring_blocks * 4, 128)
        l1_full = _align(recv_sum + _MAX_EXPERTS_PER_RANK * 8, 128)
        l1_empty = l1_full + stride
        l2_full = l1_full + 2 * stride
        l2_empty = l1_full + 4 * stride
        shared = l1_full + 5 * stride
        route_table = shared + _align(shared_blocks * 4, 128)
        metadata = sym_buffer.token_src_metadata.data_ptr() - base.data_ptr()
        if metadata != route_table + experts * max_tokens * 4:
            raise RuntimeError("MegaMoE workspace layout does not match deep_gemm.mega.local")

        def view(offset, count, dtype):
            size = torch.empty((), dtype=dtype).element_size()
            return base[offset : offset + count * size].view(dtype)

        self.experts = experts
        self.recv_count = view(recv, experts, torch.int64)
        self.recv_count_sum = view(recv_sum, experts, torch.int64)
        self.route_table = view(route_table, experts * max_tokens, torch.int32).view(experts, max_tokens)
        self.signals = [
            view(0, _HEAD_BYTES // 4, torch.int32),
            view(l1_full, ring_blocks, torch.int32),
            view(l1_empty, ring_blocks, torch.int32),
            view(l2_full, ring_blocks, torch.int64),
            view(l2_empty, ring_blocks, torch.int32),
            view(shared, shared_blocks, torch.int32),
        ]

    def plan(self, topk_idx: torch.Tensor, counts: torch.Tensor, num_routes: int) -> None:
        """Write counts and the route table and reset signals for the next launch.

        topk_idx: [tokens, topk] local expert ids, -1 for none.
        counts: [experts] device route count per expert.
        num_routes: host total of counts.
        """
        if counts.numel() != self.experts:
            raise ValueError("counts must cover every local expert")
        flat = topk_idx.reshape(-1)
        keys = torch.where(flat >= 0, flat, self.experts).to(torch.int32)
        keys, order = torch.sort(keys, stable=True)
        keys, order = keys[:num_routes].to(torch.int64), order[:num_routes]
        counts = counts.to(torch.int64)
        starts = counts.cumsum(0) - counts
        slot = torch.arange(num_routes, device=flat.device) - starts.index_select(0, keys)
        self.route_table.index_put_((keys, slot), order.to(torch.int32))
        self.recv_count.copy_(counts)
        self.recv_count_sum.copy_(counts)
        torch._foreach_zero_(self.signals)
