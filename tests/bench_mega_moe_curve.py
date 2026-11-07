"""测逐调用不均比例的前向延迟曲线；由本地 RSI 驱动租卡后运行。

权重转换和前向调用取自 test_mega_moe_training.py（6ac0f15）。
不修改原测试。--cpu-check 不导入 torch，也不访问 GPU。
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
import random
import statistics


SHAPES = {"dsv4": (4096, 2048, 256), "dsv41": (5120, 2304, 384)}
RATIOS = (1.0, 1.18, 1.26, 1.50, 1.70, 2.0)
CASES = ("balanced", "ratio118", "ratio126", "ratio150", "ratio170", "ratio200")
HOT_RANK = 2
WORLD = 4
TOKENS = 8192
TOPK = 6
# 每项作业预算 2 GiB，另给驱动的 local.log / local_result 留出 16 MiB。
OUTPUT_CAP = 2 * 1024**3 - 16 * 1024**2


def quotas(total, weights):
    denominator = sum(weights)
    counts = [total * w // denominator for w in weights]
    order = sorted(range(len(weights)),
                   key=lambda i: (-(total * weights[i] % denominator), i))
    for i in order[:total - sum(counts)]:
        counts[i] += 1
    return counts


def make_routes(num_experts, case):
    """先保留每 token 的四卡访问，再精确分配其余两个专家槽。"""
    assert case in CASES and num_experts % WORLD == 0
    per_rank = num_experts // WORLD
    n = WORLD * TOKENS
    mean = n * TOPK // WORLD
    ratio = RATIOS[CASES.index(case)]
    hot_rows = round(mean * ratio)
    cold = quotas(n * TOPK - hot_rows, [1] * (WORLD - 1))
    targets = cold[:HOT_RANK] + [hot_rows] + cold[HOT_RANK:]
    # 每个 token 至少一条路由到每张卡，最重卡最多三条，故上限为 2.0 倍。
    assert all(n <= rows <= 3 * n for rows in targets)
    extras = [r for r, count in enumerate(targets) for _ in range(count - n)]
    assert len(extras) == n * (TOPK - WORLD)
    random.Random(20261108).shuffle(extras)
    used = [0] * WORLD
    routes, source_rows = [], [[0] * WORLD for _ in range(WORLD)]
    for t in range(n):
        owners = list(range(WORLD)) + extras[2 * t:2 * t + 2]
        # 各卡的基础槽位置随 token 轮转；不改变六个 slot 的合并规则。
        offset = t % TOPK
        owners = owners[offset:] + owners[:offset]
        row = []
        for owner in owners:
            row.append(owner * per_rank + used[owner] % per_rank)
            used[owner] += 1
            source_rows[t // TOKENS][owner] += 1
        routes.append(row)
    counts, fanout = [0] * num_experts, Counter()
    for row in routes:
        assert len(row) == TOPK and len(set(row)) == TOPK
        assert all(0 <= expert < num_experts for expert in row)
        for expert in row:
            counts[expert] += 1
        fanout[len({expert // per_rank for expert in row})] += 1
    rank_rows = [sum(counts[r * per_rank:(r + 1) * per_rank]) for r in range(WORLD)]
    assert rank_rows == targets and sum(rank_rows) == n * TOPK
    assert fanout == {WORLD: n}
    for r in range(WORLD):
        local = counts[r * per_rank:(r + 1) * per_rank]
        assert max(local) - min(local) <= 1
    assert all(sum(row) == TOKENS * TOPK for row in source_rows)
    return routes, {
        "requested_rank_max_mean": ratio,
        "actual_rank_max_mean": max(rank_rows) / mean,
        "hot_rank": HOT_RANK,
        "received_rows": rank_rows,
        "source_destination_rows": source_rows,
        "expert_counts": counts,
        "token_rank_fanout_histogram": dict(fanout),
        "route_sha256": hashlib.sha256(json.dumps(routes).encode()).hexdigest(),
    }


def round_orders():
    # 六组 Williams 顺序：每组在每个位置出现一次，覆盖所有相邻有向对。
    pattern = (0, 1, 5, 2, 4, 3)
    first = [tuple(CASES[(i + shift) % len(CASES)] for i in pattern)
             for shift in range(len(CASES))]
    return first + [tuple(reversed(order)) for order in first]


def paired_interval(values):
    rng = random.Random(20261108)
    samples = sorted(statistics.median(rng.choices(values, k=len(values)))
                     for _ in range(5000))
    return [samples[124], samples[4874]]


def summarize(samples):
    cases = {}
    for name in CASES:
        selected = [s for s in samples if s["routing"] == name]
        ranks = [statistics.median(s["rank_us"][rank] for s in selected)
                 for rank in range(WORLD)]
        maxima = [s["max_us"] for s in selected]
        cases[name] = {
            "rank_us": ranks,
            "max_us": max(ranks),
            "mean_us": statistics.mean(ranks),
            "round_max_us": maxima,
            "median_round_max_us": statistics.median(maxima),
            "round_max_relative_range": (max(maxima) - min(maxima))
            / statistics.median(maxima),
        }
    comparisons = {}
    for name in CASES[1:]:
        ratios = []
        for round_idx in sorted({s["round"] for s in samples}):
            by_case = {s["routing"]: s["max_us"] for s in samples
                       if s["round"] == round_idx}
            ratios.append(by_case[name] / by_case["balanced"])
        interval = paired_interval(ratios)
        comparisons[name] = {
            "paired_round_ratios": ratios,
            "median_paired_ratio": statistics.median(ratios),
            "bootstrap_95pct_median_ratio": interval,
            "max_us_ratio": cases[name]["max_us"] / cases["balanced"]["max_us"],
            "below_3pct_with_sampling_margin": interval[1] < 1.03,
        }
    return cases, comparisons


def project_curve(path, cases, samples):
    """对 <=2.0 倍的旧调用分布分段线性插值；未覆盖尾部不补造收益。"""
    evidence = json.loads(path.read_text())
    assert evidence["ratio_grid"] == list(RATIOS)
    indices = sorted({sample["round"] for sample in samples})
    by_round = {(s["round"], s["routing"]): s for s in samples}
    point = [cases[name]["max_us"] / cases["balanced"]["max_us"] for name in CASES]
    # 配对抽样保留同轮各比例的相关性；统计口径仍是 max(各卡中位数)。
    rng = random.Random(20261108)
    boot = []
    for _ in range(2000):
        selected = rng.choices(indices, k=len(indices))
        maxima = [max(statistics.median(by_round[r, name]["rank_us"][rank]
                                       for r in selected)
                      for rank in range(WORLD)) for name in CASES]
        boot.append([value / maxima[0] for value in maxima])
    result = {
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "method": "以旧调用四卡平均耗时为权重，在已测行数比网格分段线性插值；仅作代理估算。",
        "uncertainty": "95% 区间仅含本作业配对抽样，不含 trace 顺序匹配、时间比到行数比或线性插值的模型误差。",
        "stop_rule": "只有含尾部的加权惩罚上界低于 3% 才能支持停止；任一未覆盖尾部不能由本结果置零。",
        "traces": {},
    }
    for name, trace in evidence["traces"].items():
        weights = trace["interpolation_weights_of_total_mean_time"]
        coverage = sum(weights)
        assert 0 < coverage <= 1.00000001
        projected = sum(w * (ratio - 1) for w, ratio in zip(weights, point))
        draws = sorted(sum(w * (ratio - 1) for w, ratio in zip(weights, row))
                       for row in boot)
        interval = [draws[49], draws[1949]]
        result["traces"][name] = {
            "calls_per_rank": trace["calls_per_rank"],
            "supported_mean_time_fraction": coverage,
            "unsupported_tail": trace["unsupported_tail"],
            "supported_excess_over_total_proxy_time": projected,
            "supported_excess_bootstrap_95pct": interval,
            "supported_only_relative_penalty": projected / coverage,
            "full_distribution_relative_penalty": None,
            "full_distribution_stop_supported": False,
            "supported_component_above_3pct_with_sampling_margin": interval[0] >= 0.03,
        }
    return result


def cpu_check(trace_weights):
    orders = round_orders()
    assert all(set(order) == set(CASES) for order in orders)
    for name in CASES:
        assert Counter(order.index(name) for order in orders) == {i: 2 for i in range(6)}
    samples = []
    for r, order in enumerate(orders):
        for name in order:
            times = [100 * RATIOS[CASES.index(name)] * (1 + rank / 100)
                     for rank in range(WORLD)]
            samples.append({"round": r, "routing": name, "rank_us": times,
                            "max_us": max(times), "mean_us": statistics.mean(times)})
    cases, comparisons = summarize(samples)
    for name in CASES[1:]:
        assert math.isclose(comparisons[name]["max_us_ratio"], RATIOS[CASES.index(name)])
    if trace_weights is not None:
        projection = project_curve(trace_weights, cases, samples)
        evidence = json.loads(trace_weights.read_text())
        for name, trace in evidence["traces"].items():
            expected = trace["supported_linear_excess_over_total_mean_time"]
            actual = projection["traces"][name]["supported_excess_over_total_proxy_time"]
            assert math.isclose(actual, expected, abs_tol=1e-12), (name, actual, expected)
    print("CPU 顺序、rank-max 聚合与时间加权检查通过；合成测试数字不是 GPU 实测。")


def check_output_bound(output):
    total = 0
    for directory, _, files in os.walk(output):
        for filename in files:
            try:
                total += (Path(directory) / filename).stat().st_size
            except FileNotFoundError:
                continue  # 编译临时文件可在读取时正常消失。
    assert total <= OUTPUT_CAP, f"输出 {total} B 超过 {OUTPUT_CAP} B"
    return total


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
    assert "6ac0f15" in installed_version, installed_version
    dist.init_process_group(
        backend="nccl", init_method="file://" + str(args.output / "rendezvous"),
        world_size=WORLD, rank=rank, device_id=torch.device(f"cuda:{rank}"),
        timeout=timedelta(minutes=15))
    group = dist.group.WORLD
    hidden, inter, experts = SHAPES[args.shape]
    per_rank = experts // WORLD
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, experts, TOKENS, TOPK, hidden, inter, mma_type="fp8xfp4")
    block_m = deep_gemm._C.get_block_m_for_mega_moe(
        WORLD, experts, buffer.num_max_tokens_per_rank, TOKENS, TOPK, "fp8xfp4")

    def weights(n, k):
        # 与原测试相同的量化边界；逐专家生成，避免保存额外 BF16 权重副本。
        packed = torch.empty((per_rank, n, k // 2), dtype=torch.int8, device="cuda")
        sf = torch.empty((per_rank, n, k // 32), dtype=torch.float32, device="cuda")
        for expert in range(per_rank):
            original = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
            packed[expert], sf[expert] = per_token_cast_to_fp4(
                original, use_ue8m0=True, gran_k=32)
        return packed, deep_gemm.transform_sf_into_required_layout(
            sf, n, k, (1, 32), per_rank)

    l1, l2 = deep_gemm.transform_weights_for_mega_moe(
        weights(2 * inter, hidden), weights(hidden, inter))
    x = per_token_cast_to_fp8(
        torch.randn((TOKENS, hidden), dtype=torch.bfloat16, device="cuda"),
        use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    scores = torch.rand((TOKENS, TOPK), dtype=torch.float32, device="cuda")
    scores /= scores.sum(dim=1, keepdim=True)
    y = torch.empty((TOKENS, hidden), dtype=torch.bfloat16, device="cuda")
    stats = torch.zeros(per_rank, dtype=torch.int32, device="cuda")
    inputs, routing = {}, {}
    for name in CASES:
        rows, metadata = make_routes(experts, name)
        inputs[name] = torch.tensor(rows[rank * TOKENS:(rank + 1) * TOKENS],
                                    dtype=torch.int64, device="cuda")
        count = metadata["expert_counts"]
        metadata["padded_received_rows"] = [
            sum((c + block_m - 1) // block_m * block_m
                for c in count[r * per_rank:(r + 1) * per_rank])
            for r in range(WORLD)]
        routing[name] = metadata
        # 在 GPU 上独立计数，核对所有源卡实际路由，不把规划配额冒充接收量。
        actual_counts = torch.bincount(inputs[name].flatten(), minlength=experts)
        dist.all_reduce(actual_counts, group=group)
        assert actual_counts.cpu().tolist() == count

    def forward(ids):
        buffer.x[:TOKENS].copy_(x[0])
        buffer.x_sf[:TOKENS].copy_(x[1])
        buffer.topk_idx[:TOKENS].copy_(ids)
        buffer.topk_weights[:TOKENS].copy_(scores)
        deep_gemm.fp8_fp4_mega_moe(
            y=y, l1_weights=l1, l2_weights=l2, sym_buffer=buffer,
            cumulative_local_expert_recv_stats=stats,
            activation="swiglu", activation_clamp=10.0, fast_math=True,
            route_weight_mode=deep_gemm.RouteWeightMode.PRE_DOWN,
            num_config_tokens=TOKENS)

    graphs = {}
    capture_stream = torch.cuda.Stream()
    for name in CASES:
        expected = torch.tensor(
            routing[name]["expert_counts"][rank * per_rank:(rank + 1) * per_rank],
            dtype=torch.int32, device="cuda")
        for _ in range(3):
            forward(inputs[name])
        torch.cuda.synchronize()
        stats.zero_()
        forward(inputs[name])
        torch.cuda.synchronize()
        assert torch.equal(stats, expected), f"{name}: kernel 接收计数不一致"
        assert bool(torch.isfinite(y).all()), f"{name}: 输出非有限值"
        reference = y.clone()
        graph = torch.cuda.CUDAGraph()
        capture_stream.wait_stream(torch.cuda.current_stream())
        dist.barrier(group=group)
        with torch.cuda.graph(graph, stream=capture_stream):
            for _ in range(args.graph_iterations):
                forward(inputs[name])
        torch.cuda.current_stream().wait_stream(capture_stream)
        stats.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(stats, expected * args.graph_iterations)
        assert torch.equal(y, reference), f"{name}: graph 与直接调用输出不逐位相等"
        graphs[name] = graph
        dist.barrier(group=group)
        if rank == 0:
            print(f"正确性通过：{name}，接收行数 {routing[name]['received_rows']} 行，"
                  f"补齐行数 {routing[name]['padded_received_rows']} 行，block_m={block_m} 行",
                  flush=True)
            check_output_bound(args.output)

    samples = []
    # 所有组先共同预热，保留随后所有正式样本，不丢弃首轮。
    for _ in range(4):
        for name in CASES:
            graphs[name].replay()
        torch.cuda.synchronize()
    dist.barrier(group=group)
    orders = round_orders()
    for round_idx in range(args.rounds):
        order = orders[round_idx % len(orders)]
        for name in order:
            # 每次切换先预热同一路由；以下 barrier、统计及主机读取均不在计时内。
            dist.barrier(group=group)
            graphs[name].replay()
            torch.cuda.synchronize()
            dist.barrier(group=group)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            duration = start.elapsed_time(end) * 1000.0 / args.graph_iterations
            local_time = torch.tensor([duration], dtype=torch.float64, device="cuda")
            all_times = [torch.empty_like(local_time) for _ in range(WORLD)]
            dist.all_gather(all_times, local_time, group=group)
            rank_us = [float(t.item()) for t in all_times]
            assert all(math.isfinite(t) and t > 0 for t in rank_us)
            samples.append({"round": round_idx, "order": list(order), "routing": name,
                            "rank_us": rank_us, "max_us": max(rank_us),
                            "mean_us": statistics.mean(rank_us)})
        if rank == 0:
            print(f"第 {round_idx + 1}/{args.rounds} 轮完成："
                  + ", ".join(f"{s['routing']}={s['max_us']:.3f} 微秒"
                              for s in samples[-len(CASES):]), flush=True)
            check_output_bound(args.output)

    dist.barrier(group=group)
    if rank == 0:
        cases, comparisons = summarize(samples)
        metrics = {
            "schema": "megamoe-imbalance-r2-curve-v1", "shape": args.shape,
            "deep_gemm_version": installed_version,
            "deep_gemm_module": deep_gemm.__file__,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "hidden": hidden, "intermediate_hidden": inter, "num_experts": experts,
            "num_ranks": WORLD, "tokens_per_rank": TOKENS, "topk": TOPK,
            "mma_type": "fp8xfp4", "block_m": block_m,
            "rounds": args.rounds, "graph_iterations": args.graph_iterations,
            "device_name": torch.cuda.get_device_name(rank),
            "timing_boundary": "每次前向含输入写入对称缓冲区、dispatch、专家计算、combine、kernel 尾屏障；不含量化、路由生成、统计、进程间起跑 barrier；无 profiler",
            "aggregation": f"各路由 max_us=max(各 rank 的 {args.rounds} 轮中位数)；另保留每轮 max_us 和配对比值",
            "top_level_routing": "ratio126",
            "knee_first_tested_ratio_above_3pct": next((RATIOS[CASES.index(name)] for name in CASES[1:] if comparisons[name]["bootstrap_95pct_median_ratio"][0] > 1.03), None),
            "time_weighted_projection": project_curve(args.trace_weights, cases, samples),
            "routing": routing, "cases": cases, "comparisons": comparisons,
            "samples": samples,
            "correctness": {"status": "pass", "scope": "所有路由唯一且计数精确；kernel 接收计数与全局路由一致；每组 graph 对直接调用输出逐位相等且有限。不同路由数学输出不同，不相互比较；未验证反向或复制。"},
            "output_bytes_before_metrics": check_output_bound(args.output),
            "output_cap_bytes": OUTPUT_CAP,
            "limitations": "合成路由把逐调用 grouped GEMM 时间比当作行数比，卡内专家均匀且每 token 扇出为四卡。按序号配对的 device_kernel 可能混有非 MoE 和 dX/dW；曲线投影不是实测整层分布、AF 或整步收益。超过 2.0 倍部分不外推；未测反向、复制和复制成本。",
        }
        payload = json.dumps(metrics, ensure_ascii=False, indent=2)
        assert len(payload.encode()) <= 256 * 1024
        (args.output / "metrics.json").write_text(payload + "\n")
        # run_dg.sh 只解析这四行。顶层 max_us 仅代表 ratio126 组。
        for r, duration in enumerate(cases["ratio126"]["rank_us"]):
            print(f"EP: {r}/{WORLD} | 1.26倍路由中位数 | {duration:.6f} us, 前向含通信",
                  flush=True)
    dist.barrier(group=group)
    graphs.clear()
    buffer.destroy()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=tuple(SHAPES), default="dsv4")
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--graph-iterations", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("/out"))
    parser.add_argument("--cpu-check", action="store_true")
    parser.add_argument("--trace-weights", type=Path)
    args = parser.parse_args()
    assert args.rounds >= 12 and args.rounds % 12 == 0
    assert 1 <= args.graph_iterations <= 64
    if args.cpu_check:
        cpu_check(args.trace_weights)
        for shape, (_, _, experts) in SHAPES.items():
            for case in CASES:
                _, metadata = make_routes(experts, case)
                print(json.dumps({"shape": shape, "routing": case,
                                  "received_rows": metadata["received_rows"],
                                  "fanout": metadata["token_rank_fanout_histogram"]},
                                 ensure_ascii=False))
        print("CPU 路由检查通过；GPU 导入、正确性与计时尚未执行。")
        return
    assert args.trace_weights is not None and args.trace_weights.is_file()
    args.output.mkdir(parents=True, exist_ok=True)
    # 临时文件和库缓存计入同一个作业输出目录，保留失败证据。
    for name, directory in (("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "cache"),
                            ("CUDA_CACHE_PATH", "cuda-cache"),
                            ("DG_JIT_CACHE_DIR", "deep-gemm-cache"),
                            ("TRITON_CACHE_DIR", "triton-cache")):
        path = args.output / directory
        path.mkdir(exist_ok=True)
        os.environ[name] = str(path)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    import torch.multiprocessing as mp
    mp.spawn(worker, args=(args,), nprocs=WORLD)


if __name__ == "__main__":
    main()
