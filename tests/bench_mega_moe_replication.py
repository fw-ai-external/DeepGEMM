"""冻结权重的热点专家复制前向原型；GPU 运行只由 RSI 驱动发起。

复用已封存 R2 的路由。物理槽 E'=1.25E，原专家仍在原卡；副本只接收
完整 block_m。固定放置和配额在测试准备期规划，在线路径重新计算每源卡
计数及稳定行号并改写 ID。无 GPU->主机计数读取，不修改生产代码或梯度。
--cpu-check 不导入 torch、不初始化 CUDA。
"""

import argparse
from collections import Counter
from datetime import timedelta
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import statistics
import time

from bench_mega_moe_curve import (
    HOT_RANK, OUTPUT_CAP, SHAPES, TOKENS, TOPK, WORLD,
    check_output_bound, make_routes, paired_interval,
)


ROUTINGS = ("ratio126", "ratio170")
CASES = ("balanced",) + tuple(
    f"{routing}_{mode}" for routing in ROUTINGS
    for mode in ("native", "slots", "prepared", "online")
)


def predicted_block_m(experts):
    """只用于 CPU 检查；GPU 准入必须与安装包返回值相等。"""
    expected = TOKENS * WORLD * TOPK / experts
    covered = expected + math.sqrt(expected)
    blocks = math.ceil(covered / 240)
    return next(b for b in (64, 128, 192, 240) if blocks * b >= covered)


def placement(experts):
    local = experts // WORLD
    extra = local // 4
    physical_local = local + extra
    cold = [r for r in range(WORLD) if r != HOT_RANK]
    owner_map = [e // local * physical_local + e % local for e in range(experts)]
    replica_map = owner_map.copy()
    inverse = [-1] * (physical_local * WORLD)
    for logical, physical in enumerate(owner_map):
        inverse[physical] = logical
    # 固定候选放置；前缀热专家按三张冷卡轮转，每卡最多 extra 个副本槽。
    for index in range(extra * len(cold)):
        logical = HOT_RANK * local + index
        physical = cold[index % len(cold)] * physical_local + local + index // len(cold)
        replica_map[logical] = physical
        inverse[physical] = logical
    return owner_map, replica_map, inverse


def padded_by_rank(counts, block, local):
    return [sum(math.ceil(c / block) * block for c in counts[r * local:(r + 1) * local])
            for r in range(WORLD)]


def plan_split(experts, routing, block):
    started = time.perf_counter()
    rows, source = make_routes(experts, routing)
    counts = source["expert_counts"]
    local = experts // WORLD
    physical_local = local * 5 // 4
    physical_experts = physical_local * WORLD
    owner_map, replica_map, inverse = placement(experts)
    physical_counts = [0] * physical_experts
    for e, count in enumerate(counts):
        physical_counts[owner_map[e]] = count
    unsplit = physical_counts.copy()
    loads = padded_by_rank(physical_counts, block, physical_local)
    cutoffs = [0] * experts
    # 每次移走一个整块。优先冷卡，再复用已启用的副本，减少复制权重数。
    # 同一 logical expert 只有一个副本，余数始终留在 owner；不增加补齐块。
    while True:
        candidates = []
        for e in range(HOT_RANK * local, (HOT_RANK + 1) * local):
            if replica_map[e] == owner_map[e] or counts[e] - cutoffs[e] <= block:
                continue
            dst = replica_map[e] // physical_local
            if loads[HOT_RANK] - loads[dst] <= block:
                continue
            candidates.append((loads[dst], cutoffs[e] == 0, e))
        if not candidates:
            break
        _, _, e = min(candidates)
        dst = replica_map[e] // physical_local
        cutoffs[e] += block
        physical_counts[owner_map[e]] -= block
        physical_counts[replica_map[e]] += block
        loads[HOT_RANK] -= block
        loads[dst] += block

    seen = [0] * experts
    rewritten, expanded = [], []
    for row in rows:
        new_row = []
        for e in row:
            new_row.append(replica_map[e] if seen[e] < cutoffs[e] else owner_map[e])
            seen[e] += 1
        rewritten.append(new_row)
        expanded.append([owner_map[e] for e in row])
        assert [inverse[e] for e in new_row] == row
        assert len(set(new_row)) == TOPK
    observed = Counter(e for row in rewritten for e in row)
    assert [observed[e] for e in range(physical_experts)] == physical_counts
    assert sum(physical_counts) == WORLD * TOKENS * TOPK
    assert loads == padded_by_rank(physical_counts, block, physical_local)
    assert sum(loads) == sum(padded_by_rank(unsplit, block, physical_local))
    copies = [{"logical": e, "source_rank": HOT_RANK, "source_slot": e % local,
               "destination_rank": replica_map[e] // physical_local,
               "destination_slot": replica_map[e] % physical_local,
               "rows": c} for e, c in enumerate(cutoffs) if c]
    assert all(copy["rows"] % block == 0 for copy in copies)
    metadata = {
        "routing": routing, "logical_experts": experts, "physical_experts": physical_experts,
        "reserved_extra_slots": experts // 4, "active_replicas": len(copies),
        "block_m": block, "original_received_rows": source["received_rows"],
        "native_block_m": predicted_block_m(experts),
        "native_padded_received_rows": padded_by_rank(counts, predicted_block_m(experts), local),
        "slots_padded_received_rows": padded_by_rank(unsplit, block, physical_local),
        "replica_received_rows": [sum(physical_counts[r * physical_local:(r + 1) * physical_local])
                                  for r in range(WORLD)],
        "replica_padded_received_rows": loads,
        "original_fanout": source["token_rank_fanout_histogram"],
        "replica_fanout": dict(Counter(len({e // physical_local for e in row}) for row in rewritten)),
        "copies": copies, "cutoffs": cutoffs, "expert_counts": physical_counts,
        "logical_route_sha256": source["route_sha256"],
        "replica_route_sha256": hashlib.sha256(json.dumps(rewritten).encode()).hexdigest(),
        "cpu_preparation_ms": (time.perf_counter() - started) * 1000,
    }
    return rows, expanded, rewritten, metadata


def round_orders():
    # 9 组各占每个位置 2 次；反序轮转分散运行位置影响。
    forward = [CASES[i:] + CASES[:i] for i in range(len(CASES))]
    return forward + [tuple(reversed(order)) for order in forward]


def aggregate(samples):
    result = {}
    for name in dict.fromkeys(sample["case"] for sample in samples):
        selected = [s for s in samples if s["case"] == name]
        ranks = [statistics.median(s["rank_us"][r] for s in selected) for r in range(WORLD)]
        result[name] = {"rank_us": ranks, "max_us": max(ranks),
                        "mean_us": statistics.mean(ranks),
                        "round_max_us": [s["max_us"] for s in selected]}
    return result


def comparison(samples, candidate, reference):
    selected = {(s["round"], s["case"]): s["max_us"] for s in samples}
    ratios = [selected[r, candidate] / selected[r, reference]
              for r in sorted({s["round"] for s in samples})]
    return {"candidate": candidate, "reference": reference,
            "paired_round_ratios": ratios, "median_paired_ratio": statistics.median(ratios),
            "bootstrap_95pct_median_ratio": paired_interval(ratios)}


def rewrite_ids(torch, dist, logical, counts, gathered, ones, order_index,
                owner_map, replica_map, cutoffs, rank, group, output):
    """设备上重新编号；固定大小计数避免 bincount 的动态输出/主机读。"""
    flat = logical.reshape(-1)
    counts.zero_()
    counts.scatter_add_(0, flat, ones)
    dist.all_gather_into_tensor(gathered, counts, group=group)
    sorted_ids, permutation = torch.sort(flat, stable=True)
    local_start = torch.cumsum(counts, 0) - counts
    # rank==0 是静态 Python 分支，不读取 GPU 值。
    source_start = (gathered.view(WORLD, -1)[:rank].sum(0)
                    if rank else torch.zeros_like(counts))
    ordinal = order_index - local_start[sorted_ids] + source_start[sorted_ids]
    physical = torch.where(ordinal < cutoffs[sorted_ids],
                           replica_map[sorted_ids], owner_map[sorted_ids])
    output.reshape(-1).scatter_(0, permutation, physical)


def cpu_check():
    orders = round_orders()
    for name in CASES:
        assert Counter(order.index(name) for order in orders) == {i: 2 for i in range(len(CASES))}
    for shape, (_, _, experts) in SHAPES.items():
        for routing in ROUTINGS:
            block = predicted_block_m(experts * 5 // 4)
            rows, _, rewritten, metadata = plan_split(experts, routing, block)
            owner_map, replica_map, _ = placement(experts)
            counts_before_source = [0] * experts
            for rank in range(WORLD):
                flat = [e for row in rows[rank * TOKENS:(rank + 1) * TOKENS] for e in row]
                permutation = sorted(range(len(flat)), key=lambda i: flat[i])
                local_counts = Counter(flat)
                local_start, offset = {}, 0
                for e in range(experts):
                    local_start[e] = offset
                    offset += local_counts[e]
                actual = [-1] * len(flat)
                for i, position in enumerate(permutation):
                    e = flat[position]
                    ordinal = i - local_start[e] + counts_before_source[e]
                    actual[position] = (replica_map[e] if ordinal < metadata["cutoffs"][e]
                                        else owner_map[e])
                assert actual == [e for row in rewritten[rank * TOKENS:(rank + 1) * TOKENS] for e in row]
                counts_before_source = [n + local_counts[e] for e, n in enumerate(counts_before_source)]
            print(json.dumps({"shape": shape, **{k: metadata[k] for k in (
                "routing", "physical_experts", "active_replicas", "block_m",
                "slots_padded_received_rows", "replica_received_rows", "replica_padded_received_rows",
                "replica_fanout")}}, ensure_ascii=False))
    print("CPU 检查通过：逐 slot 逻辑身份、行数守恒、块对齐、无新增补齐、各源卡行号和计时顺序。GPU 未运行。")


def cpu_torch_worker(rank, rendezvous):
    """真实 Gloo/torch 分布式接口检查，不用模拟通信，不接触 CUDA。"""
    import torch
    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous,
                            rank=rank, world_size=WORLD, timeout=timedelta(seconds=60))
    for shape, (_, _, experts) in SHAPES.items():
        owner, replica, _ = placement(experts)
        for routing in ROUTINGS:
            rows, _, prepared, plan = plan_split(experts, routing, predicted_block_m(experts * 5 // 4))
            logical = torch.tensor(rows[rank * TOKENS:(rank + 1) * TOKENS], dtype=torch.int64)
            output = torch.empty_like(logical)
            rewrite_ids(torch, dist, logical, torch.zeros(experts, dtype=torch.int64),
                        torch.empty(WORLD * experts, dtype=torch.int64),
                        torch.ones(TOKENS * TOPK, dtype=torch.int64), torch.arange(TOKENS * TOPK),
                        torch.tensor(owner), torch.tensor(replica), torch.tensor(plan["cutoffs"]),
                        rank, dist.group.WORLD, output)
            assert torch.equal(output, torch.tensor(prepared[rank * TOKENS:(rank + 1) * TOKENS]))
            if rank == 0:
                print(f"真实 CPU torch/Gloo 改写通过：{shape}/{routing}", flush=True)
    assert not torch.cuda.is_initialized()
    dist.destroy_process_group()


def worker(rank, args):
    import torch
    import torch.distributed as dist
    import deep_gemm
    from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.set_default_device("cuda")
    torch.manual_seed(20261108 + rank)
    installed_version = version("deep_gemm")
    assert "6ac0f15" in installed_version
    dist.init_process_group(backend="nccl", init_method="file://" + str(args.output / "rendezvous"),
                            world_size=WORLD, rank=rank, device_id=torch.device(f"cuda:{rank}"),
                            timeout=timedelta(minutes=15))
    group = dist.group.WORLD
    hidden, inter, experts = SHAPES[args.shape]
    local, physical_local = experts // WORLD, experts // WORLD * 5 // 4
    physical_experts = physical_local * WORLD
    buffers, blocks = {}, {}
    for key, n in (("native", experts), ("physical", physical_experts)):
        buffers[key] = deep_gemm.get_symm_buffer_for_mega_moe(
            group, n, TOKENS, TOPK, hidden, inter, mma_type="fp8xfp4")
        blocks[key] = deep_gemm._C.get_block_m_for_mega_moe(
            WORLD, n, buffers[key].num_max_tokens_per_rank, TOKENS, TOPK, "fp8xfp4")
        assert blocks[key] == predicted_block_m(n), (key, blocks[key])

    def cast_weights(n, k):
        packed = torch.empty((local, n, k // 2), dtype=torch.int8)
        sf = torch.empty((local, n, k // 32), dtype=torch.float32)
        for e in range(local):
            original = torch.randn((n, k), dtype=torch.bfloat16)
            packed[e], sf[e] = per_token_cast_to_fp4(original, use_ue8m0=True, gran_k=32)
        return packed, deep_gemm.transform_sf_into_required_layout(sf, n, k, (1, 32), local)

    l1, l2 = deep_gemm.transform_weights_for_mega_moe(
        cast_weights(2 * inter, hidden), cast_weights(hidden, inter))
    native = [*l1, *l2]
    expanded = []
    for tensor in native:
        assert tensor.stride(0) == tensor.numel() // local
        new = torch.empty_strided((physical_local, *tensor.shape[1:]), tensor.stride(), dtype=tensor.dtype)
        new.zero_()
        new[:local].copy_(tensor)
        assert torch.equal(new[:local], tensor)
        expanded.append(new)

    x = per_token_cast_to_fp8(torch.randn((TOKENS, hidden), dtype=torch.bfloat16),
                              use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    scores = torch.rand((TOKENS, TOPK), dtype=torch.float32)
    scores /= scores.sum(dim=1, keepdim=True)
    y = torch.empty((TOKENS, hidden), dtype=torch.bfloat16)
    stats = {"native": torch.zeros(local, dtype=torch.int32),
             "physical": torch.zeros(physical_local, dtype=torch.int32)}
    inputs, plans, expected_counts = {}, {}, {}
    balanced, balanced_info = make_routes(experts, "balanced")
    inputs["balanced"] = torch.tensor(balanced[rank * TOKENS:(rank + 1) * TOKENS], dtype=torch.int64)
    expected_counts["balanced"] = balanced_info["expert_counts"]
    owner, replica, _ = placement(experts)
    device_owner = torch.tensor(owner, dtype=torch.int64)
    device_replica = torch.tensor(replica, dtype=torch.int64)
    device_cutoffs = {}
    for routing in ROUTINGS:
        logical, slots, prepared, metadata = plan_split(experts, routing, blocks["physical"])
        plans[routing] = metadata
        for mode, rows in (("native", logical), ("slots", slots), ("prepared", prepared)):
            name = f"{routing}_{mode}"
            inputs[name] = torch.tensor(rows[rank * TOKENS:(rank + 1) * TOKENS], dtype=torch.int64)
            expected_counts[name] = [0] * (experts if mode == "native" else physical_experts)
            for row in rows:
                for e in row:
                    expected_counts[name][e] += 1
        expected_counts[f"{routing}_online"] = expected_counts[f"{routing}_prepared"]
        device_cutoffs[routing] = torch.tensor(metadata["cutoffs"], dtype=torch.int64)

    copy_buffers = {}
    per_expert_bytes = sum(t.numel() * t.element_size() // local for t in native)
    for routing in ROUTINGS:
        copies = plans[routing]["copies"]
        send_order = sorted((c for c in copies if rank == c["source_rank"]),
                            key=lambda c: (c["destination_rank"], c["destination_slot"]))
        recv_order = sorted((c for c in copies if rank == c["destination_rank"]),
                            key=lambda c: (c["source_rank"], c["destination_slot"]))
        send_ids = torch.tensor([c["source_slot"] for c in send_order], dtype=torch.int64)
        recv_ids = torch.tensor([c["destination_slot"] for c in recv_order], dtype=torch.int64)
        parts = []
        for src, dst in zip(native, expanded):
            elements = src.stride(0)
            src_storage = src.as_strided((local, elements), (elements, 1))
            dst_storage = dst.as_strided((physical_local, elements), (elements, 1))
            send = torch.empty((len(send_order), elements), dtype=src.dtype)
            recv = torch.empty((len(recv_order), elements), dtype=src.dtype)
            send_sizes = [sum(c["destination_rank"] == r for c in send_order) * elements for r in range(WORLD)]
            recv_sizes = [sum(c["source_rank"] == r for c in recv_order) * elements for r in range(WORLD)]
            parts.append((src_storage, dst_storage, send, recv, send_sizes, recv_sizes))
        copy_buffers[routing] = (parts, send_ids, recv_ids)
        plans[routing]["replica_weight_payload_bytes"] = len(copies) * per_expert_bytes

    def copy_weights(routing):
        parts, send_ids, recv_ids = copy_buffers[routing]
        for src, dst, send, recv, send_sizes, recv_sizes in parts:
            torch.index_select(src, 0, send_ids, out=send)
            dist.all_to_all_single(recv.reshape(-1), send.reshape(-1),
                                   output_split_sizes=recv_sizes, input_split_sizes=send_sizes, group=group)
            dst.index_copy_(0, recv_ids, recv)

    def time_once(name, round_idx, fn, iterations=1):
        dist.barrier(group=group)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        duration = start.elapsed_time(end) * 1000.0 / iterations
        scalar = torch.tensor([duration], dtype=torch.float64)
        gathered_times = [torch.empty_like(scalar) for _ in range(WORLD)]
        dist.all_gather(gathered_times, scalar, group=group)
        times = [float(t.item()) for t in gathered_times]
        assert all(math.isfinite(t) and t > 0 for t in times)
        return {"case": name, "round": round_idx, "rank_us": times, "max_us": max(times)}

    copy_samples = []
    for routing in ROUTINGS:
        for _ in range(2):
            copy_weights(routing)
        torch.cuda.synchronize()
        for round_idx in range(12):
            copy_samples.append(time_once(routing, round_idx, lambda r=routing: copy_weights(r)))
        for _, dst, _, recv, _, _ in copy_buffers[routing][0]:
            assert torch.equal(dst.index_select(0, copy_buffers[routing][2]), recv)
    # 两套规划的副本权重相同；按并集初始化，不能依赖一次计时的残留值。
    for routing in ROUTINGS:
        copy_weights(routing)
    torch.cuda.synchronize()

    route_counts = torch.zeros(experts, dtype=torch.int64)
    route_counts_all = torch.empty(WORLD * experts, dtype=torch.int64)
    ones = torch.ones(TOKENS * TOPK, dtype=torch.int64)
    order_index = torch.arange(TOKENS * TOPK, dtype=torch.int64)
    rewritten_ids = torch.empty((TOKENS, TOPK), dtype=torch.int64)

    def rewrite(routing):
        rewrite_ids(torch, dist, inputs[f"{routing}_native"], route_counts, route_counts_all,
                    ones, order_index, device_owner, device_replica, device_cutoffs[routing],
                    rank, group, rewritten_ids)

    # 首次 collectives 和 sort 在 capture 外预热；实际 GPU 改写须逐 slot 等于 CPU 参考。
    for routing in ROUTINGS:
        rewrite(routing)
        assert torch.equal(rewritten_ids, inputs[f"{routing}_prepared"])

    def forward(name):
        is_native = name == "balanced" or name.endswith("_native")
        key = "native" if is_native else "physical"
        if name.endswith("_online"):
            rewrite(name.split("_")[0])
            ids = rewritten_ids
        else:
            ids = inputs[name]
        buffer = buffers[key]
        weights = native if is_native else expanded
        buffer.x[:TOKENS].copy_(x[0])
        buffer.x_sf[:TOKENS].copy_(x[1])
        buffer.topk_idx[:TOKENS].copy_(ids)
        buffer.topk_weights[:TOKENS].copy_(scores)
        deep_gemm.fp8_fp4_mega_moe(
            y=y, l1_weights=tuple(weights[:2]), l2_weights=tuple(weights[2:]), sym_buffer=buffer,
            cumulative_local_expert_recv_stats=stats[key], activation="swiglu",
            activation_clamp=10.0, fast_math=True,
            route_weight_mode=deep_gemm.RouteWeightMode.PRE_DOWN, num_config_tokens=TOKENS)

    checks, references, graphs = [], {}, {}
    stream = torch.cuda.Stream()

    def verify(name, reference):
        different = int((y.view(torch.int16) != reference.view(torch.int16)).sum().item())
        max_abs = float((y.float() - reference.float()).abs().max().item())
        finite = bool(torch.isfinite(y).all())
        local_check = {"rank": rank, "case": name, "different_elements": different,
                       "max_abs": max_abs, "finite": finite}
        all_checks = [None] * WORLD
        dist.all_gather_object(all_checks, local_check, group=group)
        checks.extend(all_checks)
        passed = all(c["different_elements"] == 0 and c["finite"] for c in all_checks)
        if not passed:
            if rank == 0:
                failure = {"schema": "megamoe-replication-r3-v1", "shape": args.shape,
                           "correctness": {"status": "fail", "checks": checks},
                           "block_m": blocks, "plans": plans,
                           "copy_cases": aggregate(copy_samples),
                           "reason": "前向逐位合同失败；不进行性能测量，不放宽容差。"}
                (args.output / "metrics.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n")
            dist.barrier(group=group)
            raise AssertionError(f"{name}: 前向逐位合同失败")

    for name in CASES:
        is_native = name == "balanced" or name.endswith("_native")
        key = "native" if is_native else "physical"
        n_local = local if is_native else physical_local
        expected = torch.tensor(expected_counts[name][rank * n_local:(rank + 1) * n_local], dtype=torch.int32)
        for _ in range(3):
            forward(name)
        torch.cuda.synchronize()
        stats[key].zero_()
        forward(name)
        torch.cuda.synchronize()
        assert torch.equal(stats[key], expected), f"{name}: kernel 接收计数不符"
        routing = "balanced" if name == "balanced" else name.split("_")[0]
        if is_native:
            references[routing] = y.clone()
        verify(name, references[routing])
        graph = torch.cuda.CUDAGraph()
        stream.wait_stream(torch.cuda.current_stream())
        dist.barrier(group=group)
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(args.graph_iterations):
                forward(name)
        torch.cuda.current_stream().wait_stream(stream)
        stats[key].zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(stats[key], expected * args.graph_iterations)
        verify(name + "_graph", references[routing])
        graphs[name] = graph
        dist.barrier(group=group)
        if rank == 0:
            print(f"逐位检查通过：{name}", flush=True)
            check_output_bound(args.output)

    rewrite_graphs = {}
    for routing in ROUTINGS:
        graph = torch.cuda.CUDAGraph()
        stream.wait_stream(torch.cuda.current_stream())
        dist.barrier(group=group)
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(args.graph_iterations):
                rewrite(routing)
        torch.cuda.current_stream().wait_stream(stream)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(rewritten_ids, inputs[f"{routing}_prepared"])
        rewrite_graphs[routing] = graph

    samples, rewrite_samples = [], []
    for _ in range(3):
        for name in CASES:
            graphs[name].replay()
        torch.cuda.synchronize()
    for round_idx, order in enumerate(round_orders()):
        for name in order:
            graphs[name].replay()
            torch.cuda.synchronize()
            sample = time_once(name, round_idx, graphs[name].replay, args.graph_iterations)
            sample["order"] = list(order)
            samples.append(sample)
        if rank == 0:
            print(f"第 {round_idx + 1}/{len(round_orders())} 轮：" + ", ".join(
                f"{s['case']}={s['max_us']:.3f} 微秒" for s in samples[-len(CASES):]), flush=True)
            check_output_bound(args.output)
    for round_idx in range(12):
        for routing in ROUTINGS if round_idx % 2 == 0 else tuple(reversed(ROUTINGS)):
            rewrite_graphs[routing].replay()
            torch.cuda.synchronize()
            rewrite_samples.append(time_once(routing, round_idx, rewrite_graphs[routing].replay, args.graph_iterations))

    dist.barrier(group=group)
    if rank == 0:
        cases, copy_cases, rewrite_cases = aggregate(samples), aggregate(copy_samples), aggregate(rewrite_samples)
        comparisons, recovery = {}, {}
        for routing in ROUTINGS:
            for mode in ("slots", "prepared", "online"):
                comparisons[f"{routing}_{mode}_vs_native"] = comparison(samples, f"{routing}_{mode}", f"{routing}_native")
            comparisons[f"{routing}_prepared_vs_slots"] = comparison(samples, f"{routing}_prepared", f"{routing}_slots")
            original = cases[f"{routing}_native"]["max_us"]
            candidate = cases[f"{routing}_online"]["max_us"]
            balanced_us = cases["balanced"]["max_us"]
            saved = original - candidate
            copy_us = copy_cases[routing]["max_us"]
            recovery[routing] = {
                "saved_us_including_online_rewrite": saved,
                "fraction_of_imbalance_penalty_recovered": saved / (original - balanced_us) if original > balanced_us else None,
                "weight_refresh_max_us": copy_us,
                "sum_of_separate_refresh_and_online_max_us": copy_us + candidate,
                "refresh_every_call_sum_is_estimate": True,
                "amortization_calls_at_fixed_weights_and_routes": math.ceil(copy_us / saved) if saved > 0 else None,
                "paired_online_gain_above_3pct": comparisons[f"{routing}_online_vs_native"]["bootstrap_95pct_median_ratio"][1] < 0.97,
            }
        metrics = {
            "schema": "megamoe-replication-r3-v1", "shape": args.shape,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "deep_gemm_version": installed_version, "deep_gemm_module": deep_gemm.__file__,
            "hidden": hidden, "intermediate_hidden": inter, "num_experts": experts,
            "physical_experts": physical_experts, "num_ranks": WORLD,
            "tokens_per_rank": TOKENS, "topk": TOPK, "mma_type": "fp8xfp4",
            "block_m": blocks, "rounds": len(round_orders()), "graph_iterations": args.graph_iterations,
            "top_level_case": "ratio126_online", "plans": plans,
            "timing_boundary": "所有组含输入写入、dispatch、专家计算、combine 和内核尾屏障。online 另含 GPU 固定大小计数、all_gather、稳定排序编号和 ID 改写；prepared/slots 的 ID 在准备期算好。固定放置/配额规划、量化、分配和计时外 barrier 均不计入；无 profiler。",
            "copy_timing_boundary": "单独测已量化/转换权重的 GPU 打包、all_to_all 单副本传输和写入槽；含设备等待与调用间隙。不含首次扩槽/分配、量化、优化器或梯度归约。使用 CUDA event，12 次全部样本。",
            "aggregation": "max_us=max(各卡全部轮次中位数)；同时保留每轮 rank-max 与配对 bootstrap 区间。",
            "cases": cases, "comparisons": comparisons, "recovery": recovery,
            "copy_cases": copy_cases, "rewrite_cases": rewrite_cases,
            "samples": samples, "copy_samples": copy_samples, "rewrite_samples": rewrite_samples,
            "memory": {"weight_bytes_per_expert": per_expert_bytes,
                       "native_weight_bytes_per_rank": per_expert_bytes * local,
                       "expanded_weight_bytes_per_rank": per_expert_bytes * physical_local,
                       "reserved_extra_weight_bytes_per_rank": per_expert_bytes * (physical_local - local),
                       "native_symm_buffer_bytes_per_rank": buffers["native"].buffer.numel(),
                       "expanded_symm_buffer_bytes_per_rank": buffers["physical"].buffer.numel(),
                       "note": "对照测试同时持有两套权重/缓冲区及复制临时区；不把它说成生产峰值显存。"},
            "correctness": {"status": "pass", "checks": checks,
                            "scope": "副本槽逻辑身份、GPU ID 逐 slot、逐专家 kernel 接收行数、直接和图内前向对相同原始路由逐位相等且有限。dX、dW、反向和训练更新未测。"},
            "limitations": "固定合成放置和配额，不是动态 GPU 规划器；CPU 测试准备毫秒数包括路由生成和检查，不代表产品规划耗时。扇出可因分流改变，记录直方图。每次重新读取所有源卡计数，未扣 native 已有 all_reduce 成本，因此 online 结果计入整套改写成本。仅冻结权重前向原型；未接入产品、不实现梯度。不能外推 AF、全参数训练或端到端收益。",
            "output_bytes_before_metrics": check_output_bound(args.output), "output_cap_bytes": OUTPUT_CAP,
        }
        payload = json.dumps(metrics, ensure_ascii=False, indent=2)
        assert len(payload.encode()) <= 256 * 1024
        (args.output / "metrics.json").write_text(payload + "\n")
        for r, duration in enumerate(cases["ratio126_online"]["rank_us"]):
            print(f"EP: {r}/{WORLD} | 1.26倍在线改写复制前向 | {duration:.6f} us, 含通信和改写", flush=True)
    dist.barrier(group=group)
    graphs.clear()
    rewrite_graphs.clear()
    for buffer in buffers.values():
        buffer.destroy()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=tuple(SHAPES), default="dsv41")
    parser.add_argument("--graph-iterations", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("/out"))
    parser.add_argument("--cpu-check", action="store_true")
    parser.add_argument("--cpu-torch-check", action="store_true")
    args = parser.parse_args()
    if args.cpu_check:
        cpu_check()
        return
    if args.cpu_torch_check:
        import tempfile
        import torch
        import deep_gemm
        import torch.multiprocessing as mp
        installed = version("deep_gemm")
        assert "6ac0f15" in installed
        print(f"依赖导入：{installed}，{deep_gemm.__file__}，CUDA 未初始化={not torch.cuda.is_initialized()}", flush=True)
        rendezvous_dir = tempfile.mkdtemp(prefix="mmib-r3-cpu-")
        mp.spawn(cpu_torch_worker, args=(str(Path(rendezvous_dir) / "gloo"),), nprocs=WORLD)
        return
    assert 1 <= args.graph_iterations <= 64
    args.output.mkdir(parents=True, exist_ok=True)
    for name, directory in (("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "cache"),
                            ("CUDA_CACHE_PATH", "cuda-cache"), ("DG_JIT_CACHE_DIR", "deep-gemm-cache"),
                            ("TRITON_CACHE_DIR", "triton-cache")):
        path = args.output / directory
        path.mkdir(exist_ok=True)
        os.environ[name] = str(path)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    import torch.multiprocessing as mp
    mp.spawn(worker, args=(args,), nprocs=WORLD)


if __name__ == "__main__":
    main()
