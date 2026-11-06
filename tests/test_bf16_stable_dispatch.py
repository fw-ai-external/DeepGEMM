"""GPU regression for stable BF16 dispatch with more experts than warps.

Run with torchrun --standalone --nproc-per-node=2 (or 8) on SM100/SM103.
The route oracle comes directly from source IDs, independently of dispatch
counts, offsets, or saved metadata. It detects duplicates, omissions, wrong
expert ownership, and order changes even when numerical outputs look plausible.
"""

import os
import hashlib
from pathlib import Path

import torch
import torch.distributed as dist

import deep_gemm


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", device_id=torch.device("cuda", torch.cuda.current_device()))
    rank, world = dist.get_rank(), dist.get_world_size()
    print(f"rank={rank} harness_sha256={hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}", flush=True)
    experts, hidden, intermediate, topk, capacity = 896, 2048, 256, 16, 256
    assert experts % world == 0
    assert experts // world <= 512, "896 experts require at least two ranks"
    sms = deep_gemm.get_num_sms()
    assert experts > sms * 4, "Regression requires experts beyond dispatch-warp coverage"
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        dist.group.WORLD,
        experts,
        4096,
        topk,
        hidden,
        intermediate,
        mma_type="bf16xbf16",
        activation="swiglu",
    )
    assert (
        3 * buffer.token_src_metadata.size(0) >= sms * 4 * experts
    ), "Regression must exercise parallel stable compaction"
    local_experts = experts // world
    gen = torch.Generator(device="cuda").manual_seed(6000 + rank)
    w13 = (
        torch.randn(local_experts, 2 * intermediate, hidden, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.02
    )
    w2 = torch.randn(local_experts, hidden, intermediate, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.02
    w13, w2 = deep_gemm.transform_weights_for_mega_moe(w13, w2, activation="swiglu")
    x = torch.randn(capacity, hidden, device="cuda", dtype=torch.bfloat16, generator=gen)
    # Cover every expert repeatedly, including those above the warp threshold.
    base_ids = (torch.arange(capacity * topk, device="cuda").view(capacity, topk) + rank * 37) % experts
    saved = torch.empty(buffer.token_src_metadata.size(0), 2 * intermediate, device="cuda", dtype=torch.bfloat16)
    block_m = deep_gemm.get_block_m_for_mega_moe(
        world, experts, buffer.num_max_tokens_per_rank, capacity, topk, "bf16xbf16"
    )
    for case, length in [
        ("full", capacity),
        ("odd", 129 + rank),
        ("short", 7 + rank),
        ("empty_rank", 0 if rank == 0 else rank),
        ("all_masked", 0),
    ]:
        ids = base_ids.clone()
        ids[length:] = -1
        ids[::3, -1] = -1
        scores = torch.full((capacity, topk), 1 / topk, device="cuda")
        scores.masked_fill_(ids < 0, 0)
        all_ids = [torch.empty_like(ids) for _ in range(world)]
        dist.all_gather(all_ids, ids)
        all_ids_cpu = torch.stack(all_ids).cpu()
        expected_by_expert = [
            (all_ids_cpu == expert).nonzero() for expert in range(rank * local_experts, (rank + 1) * local_experts)
        ]
        first_output = first_metadata = None
        for repeat in range(3):
            buffer.x[:capacity].copy_(x)
            buffer.topk_idx[:capacity].copy_(ids)
            buffer.topk_weights[:capacity].copy_(scores)
            saved.fill_(float("nan"))
            output = torch.empty_like(x)
            deep_gemm.bf16_mega_moe(
                output,
                w13,
                w2,
                buffer,
                activation="swiglu",
                fast_math=False,
                saved_l1_preact=saved,
                num_config_tokens=capacity,
            )
            metadata = buffer.token_src_metadata.cpu()
            offset = 0
            actual_routes = []
            for expert, expected in enumerate(expected_by_expert):
                count = expected.size(0)
                actual = metadata[offset : offset + count].long()
                assert torch.equal(actual, expected), (
                    f"{case=} {rank=} global_expert={rank * local_experts + expert} "
                    f"has missing, duplicate, or unordered routes"
                )
                assert torch.isfinite(
                    saved[offset : offset + count]
                ).all(), f"{case=} {rank=} expert={expert} has an unwritten preactivation row"
                actual_routes.append(actual)
                offset += (count + block_m - 1) // block_m * block_m
            actual_routes = torch.cat(actual_routes)
            host_output = output.cpu()
            assert torch.isfinite(host_output).all()
            assert torch.count_nonzero(host_output[length:]) == 0
            if repeat:
                assert torch.equal(first_output, host_output), f"{case=} output is nondeterministic"
                assert torch.equal(first_metadata, actual_routes), f"{case=} routes are nondeterministic"
            else:
                first_output, first_metadata = host_output, actual_routes
        print(f"rank={rank} case={case} routes={len(actual_routes)} PASS", flush=True)
    dist.barrier()
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
