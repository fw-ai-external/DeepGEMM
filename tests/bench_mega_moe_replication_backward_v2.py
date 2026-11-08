"""冻结专家的 FP8×FP4 复制反向验证：修复跨配置同步状态别名。

沿用封存 R2 路由和 R3 块分流。复用 6ac0f15 的真实 dgrad/通信/combine，
不计算或保存 dW。两臂共享等容量梯度临时区；副本额外权重单独记账。
--cpu-check 不导入 torch。独立同步状态消除 R4 测试错误；梯度池不增加。
诊断进度及超时只在计时外工作，成功后直接退出，避开 R3 清理挂起。
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
import sys
import threading
import time

from bench_mega_moe_curve import (
    OUTPUT_CAP, SHAPES, TOKENS, TOPK, WORLD, check_output_bound, make_routes,
)
from bench_mega_moe_replication import (
    ROUTINGS, aggregate, comparison, padded_by_rank, placement, plan_split,
    predicted_block_m, rewrite_ids,
)


CASES = ("balanced", "ratio126_native", "ratio126_online",
         "ratio170_native", "ratio170_online")
STAGES = ("backward", "forward_backward")


def write_small_json(path, value):
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    assert len(payload.encode()) <= 256 * 1024
    with path.open("w") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


class StageWatchdog:
    """只终止当前实验子进程；计时区间外写进度，超时记录保留。"""

    def __init__(self, output, rank, shape):
        self.output, self.rank, self.shape = output, rank, shape
        self.lock = threading.Lock()
        self.finished = threading.Event()
        self.phase, self.deadline = "启动", time.monotonic() + 480
        self.thread = threading.Thread(target=self.watch, daemon=True)
        self.thread.start()

    def mark(self, phase, seconds):
        with self.lock:
            self.phase, self.deadline = phase, time.monotonic() + seconds
        write_small_json(self.output / f"progress_rank{self.rank}.json", {
            "rank": self.rank, "shape": self.shape, "phase": phase,
            "timeout_seconds": seconds, "unix_time_seconds": time.time()})
        print(f"进度 rank{self.rank}：{phase}，阶段上限 {seconds} 秒", flush=True)

    def watch(self):
        while not self.finished.wait(1):
            with self.lock:
                phase, deadline = self.phase, self.deadline
            if time.monotonic() > deadline:
                record = {"rank": self.rank, "shape": self.shape, "phase": phase,
                          "reason": "当前实验子进程阶段超时，退出码124；不接触其他任务。"}
                try:
                    write_small_json(self.output / f"timeout_rank{self.rank}.json", record)
                    print("阶段超时：" + json.dumps(record, ensure_ascii=False), flush=True)
                finally:
                    os._exit(124)

    def stop(self):
        self.finished.set()


def counter_layout(hidden, inter, local, physical_local):
    result = {}
    for key, count in (("native", local), ("physical", physical_local)):
        w2 = count * (hidden // 64) * (inter // 128)
        w13 = count * ((2 * inter) // 64) * (hidden // 128)
        result[key] = {"local_experts": count, "w2_words": w2,
                       "w13_words": w13, "state_words": w2 + w13,
                       "phase_count_offset_words": w2 + w13,
                       "phase_sense_offset_words": w2 + w13 + 1}
    return result


def orders():
    forward = [CASES[i:] + CASES[:i] for i in range(len(CASES))]
    return forward + [tuple(reversed(order)) for order in forward]


def prepare_routes(experts):
    result, plans = {}, {}
    rows, info = make_routes(experts, "balanced")
    result["balanced"] = (rows, info["expert_counts"])
    for routing in ROUTINGS:
        logical, _, physical, plan = plan_split(
            experts, routing, predicted_block_m(experts * 5 // 4))
        _, info = make_routes(experts, routing)
        result[routing + "_native"] = (logical, info["expert_counts"])
        result[routing + "_online"] = (physical, plan["expert_counts"])
        plans[routing] = plan
    return result, plans


def pool_capacity(experts, routes):
    # 单层实验共用原映射的最大有效池；不得通过加大复制臂的梯度区过关。
    native_block = predicted_block_m(experts)
    physical_block = predicted_block_m(experts * 5 // 4)
    native_max = max(max(padded_by_rank(counts, native_block, experts // WORLD))
                     for name, (_, counts) in routes.items() if not name.endswith("online"))
    physical_max = max(max(padded_by_rank(counts, physical_block, experts * 5 // 4 // WORLD))
                       for name, (_, counts) in routes.items() if name.endswith("online"))
    assert physical_max <= native_max
    for routing in ROUTINGS:
        before = padded_by_rank(routes[routing + "_native"][1], native_block, experts // WORLD)
        after = padded_by_rank(routes[routing + "_online"][1], physical_block, experts * 5 // 4 // WORLD)
        assert max(after) <= max(before) and sum(after) <= sum(before)
    common = math.lcm(native_block, physical_block)
    assert native_max % common == 0, (native_max, common)
    return native_max


def cpu_check():
    for name in CASES:
        assert Counter(order.index(name) for order in orders()) == {i: 2 for i in range(len(CASES))}
    for shape, (hidden, inter, experts) in SHAPES.items():
        routes, plans = prepare_routes(experts)
        capacity = pool_capacity(experts, routes)
        layout = counter_layout(hidden, inter, experts // WORLD, experts * 5 // 4 // WORLD)
        native_phase = layout["native"]["state_words"]
        assert layout["physical"]["w2_words"] <= native_phase < layout["physical"]["state_words"]
        # W13 写入 epoch ^ 0x80000000。若当作 phase_count，首批 CTA 不可能拿到最后票。
        poisoned_count = 1 ^ 0x80000000
        for grid_ctas in (2, 132, 148, 256, 1024):
            assert not poisoned_count <= grid_ctas - 1 < poisoned_count + grid_ctas
        assert layout["native"]["state_words"] < layout["physical"]["state_words"]
        print(f"CPU 屏障偏移证据：{shape}，原映射 {native_phase}，扩槽 {layout['physical']['state_words']}，原位置落在扩槽 W13 就绪区", flush=True)
        assert capacity == 92160
        owner, replica, inverse = placement(experts)
        for routing, plan in plans.items():
            original = routes[routing + "_native"][0]
            rewritten = routes[routing + "_online"][0]
            assert all([inverse[e] for e in b] == a for a, b in zip(original, rewritten))
            assert sum(plan["expert_counts"]) == WORLD * TOKENS * TOPK
            assert all(c["rows"] % plan["block_m"] == 0 for c in plan["copies"])
        print(f"CPU 路由与共享梯度容量检查通过：{shape}，每卡容量 {capacity} 行", flush=True)
    sample = [{"case": c, "round": r, "rank_us": [10.0, 11.0, 12.0, 13.0], "max_us": 13.0}
              for r in range(10) for c in CASES]
    assert aggregate(sample)["balanced"]["max_us"] == 13.0
    assert comparison(sample, "ratio126_online", "ratio126_native")["median_paired_ratio"] == 1.0


def worker(rank, args):
    import torch
    import torch.distributed as dist
    import deep_gemm
    from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8

    guard = StageWatchdog(args.output, rank, args.shape)
    guard.mark("初始化与权重准备", seconds=480)
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.set_default_device("cuda")
    torch.manual_seed(20261108 + rank)
    installed = version("deep_gemm")
    assert "6ac0f15" in installed
    dist.init_process_group(
        backend="nccl", init_method="file://" + str(args.output / "rendezvous"),
        world_size=WORLD, rank=rank, device_id=torch.device(f"cuda:{rank}"),
        timeout=timedelta(minutes=12))
    group = dist.group.WORLD
    hidden, inter, experts = SHAPES[args.shape]
    physical_experts = experts * 5 // 4
    local, physical_local = experts // WORLD, physical_experts // WORLD
    routes, plans = prepare_routes(experts)
    capacity = pool_capacity(experts, routes)
    blocks, buffers = {}, {}
    for key, n in (("native", experts), ("physical", physical_experts)):
        buffers[key] = deep_gemm.get_symm_buffer_for_mega_moe(
            group, n, TOKENS, TOPK, hidden, inter, mma_type="fp8xfp4")
        blocks[key] = deep_gemm._C.get_block_m_for_mega_moe(
            WORLD, n, buffers[key].num_max_tokens_per_rank, TOKENS, TOPK, "fp8xfp4")
        assert blocks[key] == predicted_block_m(n)
        assert buffers[key].token_src_metadata.size(0) >= capacity
    agreed = [None] * WORLD
    dist.all_gather_object(agreed, blocks, group=group)
    assert all(x == blocks for x in agreed)
    assert buffers["native"].backward_grad_y.shape == buffers["physical"].backward_grad_y.shape
    assert buffers["native"].backward_grad_route.shape == buffers["physical"].backward_grad_route.shape

    def cast_weight(n, k):
        packed = torch.empty((local, n, k // 2), dtype=torch.int8)
        scales = torch.empty((local, n, k // 32), dtype=torch.float32)
        for e in range(local):
            raw = torch.randn((n, k), dtype=torch.bfloat16)
            packed[e], scales[e] = per_token_cast_to_fp4(raw, use_ue8m0=True, gran_k=32)
        return packed, scales

    raw_l1, raw_l2 = cast_weight(2 * inter, hidden), cast_weight(hidden, inter)
    transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe(
        (raw_l1[0], deep_gemm.transform_sf_into_required_layout(
            raw_l1[1], 2 * inter, hidden, (1, 32), local)),
        (raw_l2[0], deep_gemm.transform_sf_into_required_layout(
            raw_l2[1], hidden, inter, (1, 32), local)))
    # 前四张量为前向布局，后四张量为真实反向要求的 FP4/FP32 scales 布局。
    native_weights = [*transformed_l1, *transformed_l2, *raw_l1, *raw_l2]
    physical_weights = []
    for src in native_weights:
        assert src.stride(0) == src.numel() // local
        dst = torch.empty_strided((physical_local, *src.shape[1:]), src.stride(), dtype=src.dtype)
        dst.zero_()
        dst[:local].copy_(src)
        physical_weights.append(dst)
    weight_sets = {"native": native_weights, "physical": physical_weights}

    def time_once(case, round_index, fn, iterations=1):
        dist.barrier(group=group)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        fn()
        end.record()
        end.synchronize()
        elapsed = begin.elapsed_time(end) * 1000 / iterations
        value = torch.tensor([elapsed], dtype=torch.float64)
        all_values = [torch.empty_like(value) for _ in range(WORLD)]
        dist.all_gather(all_values, value, group=group)
        times = [float(v.item()) for v in all_values]
        assert all(math.isfinite(v) and v > 0 for v in times)
        return {"case": case, "round": round_index, "rank_us": times, "max_us": max(times)}

    copy_buffers = {}
    bytes_per_expert = sum(t.numel() * t.element_size() // local for t in native_weights)
    for routing, plan in plans.items():
        copies = plan["copies"]
        send_order = sorted((c for c in copies if rank == c["source_rank"]),
                            key=lambda c: (c["destination_rank"], c["destination_slot"]))
        recv_order = sorted((c for c in copies if rank == c["destination_rank"]),
                            key=lambda c: (c["source_rank"], c["destination_slot"]))
        send_ids = torch.tensor([c["source_slot"] for c in send_order], dtype=torch.int64)
        recv_ids = torch.tensor([c["destination_slot"] for c in recv_order], dtype=torch.int64)
        parts = []
        for src, dst in zip(native_weights, physical_weights):
            elements = src.stride(0)
            send = torch.empty((len(send_order), elements), dtype=src.dtype)
            recv = torch.empty((len(recv_order), elements), dtype=src.dtype)
            parts.append((src.as_strided((local, elements), (elements, 1)),
                          dst.as_strided((physical_local, elements), (elements, 1)), send, recv,
                          [sum(c["destination_rank"] == r for c in send_order) * elements for r in range(WORLD)],
                          [sum(c["source_rank"] == r for c in recv_order) * elements for r in range(WORLD)]))
        copy_buffers[routing] = (parts, send_ids, recv_ids)
        plan["forward_backward_replica_payload_bytes"] = len(copies) * bytes_per_expert

    def copy_weights(routing):
        parts, send_ids, recv_ids = copy_buffers[routing]
        for src, dst, send, recv, send_sizes, recv_sizes in parts:
            torch.index_select(src, 0, send_ids, out=send)
            dist.all_to_all_single(recv.reshape(-1), send.reshape(-1),
                                   output_split_sizes=recv_sizes, input_split_sizes=send_sizes, group=group)
            dst.index_copy_(0, recv_ids, recv)

    copy_samples = []
    for routing in ROUTINGS:
        guard.mark("测独立权重复制/" + routing, seconds=180)
        for _ in range(2):
            copy_weights(routing)
        for i in range(12):
            copy_samples.append(time_once(routing, i, lambda r=routing: copy_weights(r)))
        for _, dst, _, received, _, _ in copy_buffers[routing][0]:
            assert torch.equal(dst.index_select(0, copy_buffers[routing][2]), received)
    for routing in ROUTINGS:
        copy_weights(routing)
    torch.cuda.synchronize()
    if rank == 0:
        write_small_json(args.output / "copy_metrics.json", {
            "schema": "r5-independent-weight-copy", "shape": args.shape,
            "copy_cases": aggregate(copy_samples), "copy_samples": copy_samples,
            "note": "独立权重复制计时；不代表前后向已完成。"})
    guard.mark("分配共享梯度池与路由", seconds=180)

    x = per_token_cast_to_fp8(torch.randn((TOKENS, hidden), dtype=torch.bfloat16),
                              use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    dy = torch.randn((TOKENS, hidden), dtype=torch.bfloat16)
    scores = torch.rand((TOKENS, TOPK), dtype=torch.float32)
    scores /= scores.sum(1, keepdim=True)
    y, dx = torch.empty_like(dy), torch.empty_like(dy)
    inputs = {name: torch.tensor(rows[rank * TOKENS:(rank + 1) * TOKENS], dtype=torch.int64)
              for name, (rows, _) in routes.items()}
    owner, replica, _ = placement(experts)
    owner_gpu, replica_gpu = torch.tensor(owner), torch.tensor(replica)
    cutoffs = {r: torch.tensor(plans[r]["cutoffs"]) for r in ROUTINGS}
    counts = torch.zeros(experts, dtype=torch.int64)
    counts_all = torch.empty(WORLD * experts, dtype=torch.int64)
    ones = torch.ones(TOKENS * TOPK, dtype=torch.int64)
    order_index = torch.arange(TOKENS * TOPK, dtype=torch.int64)
    rewritten = torch.empty((TOKENS, TOPK), dtype=torch.int64)

    # 只保留一套梯度临时区，两臂完全复用。grad_x_pool 为 x_pool 的别名。
    scratch = {
        "gate": torch.zeros((capacity, 2 * inter), dtype=torch.bfloat16),
        "grad_gate": torch.zeros((capacity, 2 * inter), dtype=torch.bfloat16),
        "grad_h": torch.zeros((capacity, inter), dtype=torch.bfloat16),
        "h_act": torch.zeros((capacity, inter), dtype=torch.bfloat16),
        "h_weighted": torch.zeros((capacity, inter), dtype=torch.bfloat16),
        "x_pool": torch.zeros((capacity, hidden), dtype=torch.bfloat16),
        "grad_ye": torch.zeros((capacity, hidden), dtype=torch.bfloat16),
        "route_weights": torch.zeros(capacity, dtype=torch.float32),
        "grad_route": torch.zeros(capacity, dtype=torch.float32),
    }
    acts = torch.empty((capacity, hidden), dtype=torch.float8_e4m3fn)
    sf_rows = {key: capacity // block * deep_gemm.align(block, 128) for key, block in blocks.items()}
    sf_storage = torch.empty(max(sf_rows.values()) * (hidden // 128), dtype=torch.int32)
    saved_sf = {key: sf_storage.as_strided((n, hidden // 128), (1, n)) for key, n in sf_rows.items()}
    w2_scratch = torch.empty((physical_local, hidden, inter), dtype=torch.bfloat16)
    w13_scratch = torch.empty((physical_local, 2 * inter, hidden), dtype=torch.bfloat16)
    # phase_count/sense 的偏移随本地专家数变化，不能共用不同模板的状态。
    state_layout = counter_layout(hidden, inter, local, physical_local)
    grids = {key: torch.zeros(row["state_words"] + 2, dtype=torch.int32)
             for key, row in state_layout.items()}
    assert grids["native"].data_ptr() != grids["physical"].data_ptr()
    recv_counts = {key: torch.zeros(n, dtype=torch.int32)
                   for key, n in (("native", local), ("physical", physical_local))}
    # 视图预先构造，反向计时不创建权重梯度、不作计数主机读取。
    bwd_weights = {}
    for key, tensors in weight_sets.items():
        n = local if key == "native" else physical_local
        bwd_weights[key] = ((tensors[6].view(torch.float8_e4m3fn), tensors[7]),
                            (tensors[4].view(torch.float8_e4m3fn).view(2 * n, inter, hidden // 2),
                             tensors[5].view(2 * n, inter, hidden // 32)))

    def rewrite(routing):
        rewrite_ids(torch, dist, inputs[routing + "_native"], counts, counts_all, ones,
                    order_index, owner_gpu, replica_gpu, cutoffs[routing], rank, group, rewritten)

    for routing in ROUTINGS:
        rewrite(routing)
        assert torch.equal(rewritten, inputs[routing + "_online"])

    def forward(name):
        key = "physical" if name.endswith("online") else "native"
        if key == "physical":
            rewrite(name.split("_")[0])
            ids = rewritten
        else:
            ids = inputs[name]
        buffer, weights = buffers[key], weight_sets[key]
        recv_counts[key].zero_()
        buffer.x[:TOKENS].copy_(x[0])
        buffer.x_sf[:TOKENS].copy_(x[1])
        buffer.topk_idx[:TOKENS].copy_(ids)
        buffer.topk_weights[:TOKENS].copy_(scores)
        deep_gemm.fp8_fp4_mega_moe(
            y=y, l1_weights=tuple(weights[:2]), l2_weights=tuple(weights[2:4]), sym_buffer=buffer,
            cumulative_local_expert_recv_stats=recv_counts[key], activation="swiglu",
            activation_clamp=10.0, fast_math=True, saved_l1_preact=scratch["gate"],
            saved_l1_acts=acts, saved_l1_acts_sf=saved_sf[key],
            route_weight_mode=deep_gemm.RouteWeightMode.PRE_DOWN, num_config_tokens=TOKENS)

    def backward_dgrad(name):
        key = "physical" if name.endswith("online") else "native"
        buffer, weights = buffers[key], weight_sets[key]
        n = physical_local if key == "physical" else local
        w2, w13 = bwd_weights[key]
        deep_gemm.fp8_fp4_mega_moe_backward_dgrad_swiglu(
            gate_up_output=scratch["gate"], grad_h_output=scratch["grad_h"],
            grad_gate_up_output=scratch["grad_gate"], h_act_output=scratch["h_act"],
            h_weighted_output=scratch["h_weighted"], x_pool_output=scratch["x_pool"],
            grad_x_pool_output=scratch["x_pool"], l1_acts=acts, l1_acts_sf=saved_sf[key],
            l1_weights=tuple(weights[:2]), grad_ye=scratch["grad_ye"],
            route_weights=scratch["route_weights"], w2_weights=w2, w2_dequant_scratch=w2_scratch[:n],
            w13_weights=w13, w13_dequant_scratch=w13_scratch[:n],
            expert_counts=recv_counts[key], grid_sync_counter=grids[key], activation_limit=10.0,
            block_m=blocks[key], compute_w13_dgrad=True, direct_remote_grad_x=True,
            write_grad_x_pool=False, clear_wgrad_padding=False,
            sym_buffer=buffer, grad_y=dy, topk_weights=scores,
            token_src_metadata=buffer.token_src_metadata,
            route_weight_mode=deep_gemm.RouteWeightMode.PRE_DOWN,
            grad_route_output=scratch["grad_route"], rank_uniform_block_m=True)

    def backward_combine(name):
        key = "physical" if name.endswith("online") else "native"
        deep_gemm.mega_moe_backward_combine_grad_x(
            dx, buffers[key], combine_order_mode=deep_gemm.CombineOrderMode.FIXED_TOPK)

    def backward(name):
        backward_dgrad(name)
        backward_combine(name)

    diagnostics = []

    def diagnostic_call(label, fn):
        # 首次调用可能进行数分钟 JIT；函数返回后，GPU 完成等待最多 45 秒。
        guard.mark(label + "/提交（允许首次JIT）", seconds=480)
        fn()
        guard.mark(label + "/等待GPU", seconds=45)
        torch.cuda.synchronize()
        guard.mark(label + "/完成", seconds=480)

    # 先单独验证高偏斜 direct，再检验触发 R4 的配置切换，不先进入 graph。
    for probe_index, name in enumerate(("ratio170_native", "ratio126_online", "ratio170_native")):
        label = f"直接探针{probe_index}/{name}"
        diagnostic_call(label + "/前向", lambda n=name: forward(n))
        diagnostic_call(label + "/dgrad", lambda n=name: backward_dgrad(n))
        diagnostic_call(label + "/combine", lambda n=name: backward_combine(n))
        key = "physical" if name.endswith("online") else "native"
        n = physical_local if key == "physical" else local
        expected = torch.tensor(routes[name][1][rank * n:(rank + 1) * n], dtype=torch.int32)
        assert torch.equal(recv_counts[key], expected)
        states = {k: {"phase_count": int(g[-2].item()), "phase_sense": int(g[-1].item())}
                  for k, g in grids.items()}
        assert all(v["phase_count"] == 0 for v in states.values()), states
        old_alias = int(grids["physical"][state_layout["native"]["state_words"]].item()) & 0xffffffff
        row = {"rank": rank, "probe": probe_index, "case": name, "states": states,
               "legacy_native_phase_location_in_physical_states_uint32": old_alias,
               "received_rows": int(expected.sum().item()), "common_pool_rows": capacity}
        diagnostics.append(row)
        write_small_json(args.output / f"diagnostics_rank{rank}.json", diagnostics)
        print("同步状态诊断：" + json.dumps(row, ensure_ascii=False), flush=True)

    checks, refs = [], {}

    def verify(name, suffix, reference):
        entries = []
        for kind, value, expected in zip(("forward", "dX"), (y, dx), reference):
            entries.append({"rank": rank, "case": name, "mode": suffix, "tensor": kind,
                            "different_elements": int((value.view(torch.int16) != expected.view(torch.int16)).sum().item()),
                            "finite": bool(torch.isfinite(value).all()),
                            "max_abs": float((value.float() - expected.float()).abs().max().item())})
        gathered = [None] * WORLD
        dist.all_gather_object(gathered, entries, group=group)
        all_entries = [e for part in gathered for e in part]
        checks.extend(all_entries)
        if not all(e["different_elements"] == 0 and e["finite"] for e in all_entries):
            if rank == 0:
                failure = {"schema": "megamoe-replication-backward-r5-v2", "shape": args.shape,
                           "correctness": {"status": "fail", "checks": checks},
                           "reason": "前向或 dX 逐位合同失败；不放宽阈值，不进入正式计时。"}
                (args.output / "metrics.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n")
                print("正确性失败，已保存差异，退出码 1", flush=True)
            raise AssertionError(f"{name}/{suffix}: 位模式不相等或非有限")

    graphs = {}
    stream = torch.cuda.Stream()
    for name in CASES:
        guard.mark("正确性预热/" + name, seconds=480)
        key = "physical" if name.endswith("online") else "native"
        n = physical_local if key == "physical" else local
        for _ in range(2):
            forward(name)
            backward(name)
        torch.cuda.synchronize()
        expected = torch.tensor(routes[name][1][rank * n:(rank + 1) * n], dtype=torch.int32)
        assert torch.equal(recv_counts[key], expected)
        reference_name = name.replace("_online", "_native")
        if key == "native":
            refs[name] = (y.clone(), dx.clone())
        verify(name, "direct", refs[reference_name])
        for stage in STAGES:
            guard.mark("图预热/" + name + "/" + stage, seconds=480)
            forward(name)
            torch.cuda.synchronize()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    if stage == "forward_backward":
                        forward(name)
                    backward(name)
            stream.synchronize()
            dist.barrier(group=group)
            guard.mark("图捕获/" + name + "/" + stage, seconds=480)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(args.graph_iterations):
                    if stage == "forward_backward":
                        forward(name)
                    backward(name)
            guard.mark("图执行/" + name + "/" + stage, seconds=45)
            graph.replay()
            torch.cuda.synchronize()
            verify(name, stage + "_graph", refs[reference_name])
            graphs[name, stage] = graph
        check_output_bound(args.output)
        if rank == 0:
            write_small_json(args.output / "validation_checks.json", checks)
            print(f"前向/dX 逐位及图执行检查通过：{name}", flush=True)

    guard.mark("正式测量前共同预热", seconds=60)
    # 共同预热后才收样本；每轮两种时间边界交替先后。
    for name in CASES:
        forward(name)
        graphs[name, "backward"].replay()
        graphs[name, "forward_backward"].replay()
    torch.cuda.synchronize()
    samples = {stage: [] for stage in STAGES}
    for round_index, order in enumerate(orders()):
        guard.mark(f"正式计时第{round_index + 1}轮（两阶段各五组）", seconds=120)
        stages = STAGES if round_index % 2 == 0 else tuple(reversed(STAGES))
        for stage in stages:
            for name in order:
                if stage == "backward":
                    # backward-only 使用刚完成的对应前向状态；前向不在这个时间窗内。
                    forward(name)
                    torch.cuda.synchronize()
                row = time_once(name, round_index, graphs[name, stage].replay, args.graph_iterations)
                row["order"] = list(order)
                samples[stage].append(row)
        check_output_bound(args.output)
        if rank == 0:
            latest = samples["forward_backward"][-len(CASES):]
            print(f"第 {round_index + 1}/{len(orders())} 轮前后向：" +
                  ", ".join(f"{r['case']}={r['max_us']:.3f} 微秒" for r in latest), flush=True)

    torch.cuda.synchronize()
    # 必须在所有 GPU 工作结束后才交换终态；此后绝不再做 NCCL barrier 或 destroy。
    guard.mark("终态交换与写盘", seconds=60)
    done = {"rank": rank, "diagnostics": diagnostics, "samples": {s: len(v) for s, v in samples.items()},
            "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
    receipts = [None] * WORLD
    dist.all_gather_object(receipts, done, group=group)
    torch.cuda.synchronize()
    if rank == 0:
        stage_cases = {stage: aggregate(rows) for stage, rows in samples.items()}
        comparisons = {stage: {
            routing: comparison(rows, routing + "_online", routing + "_native")
            for routing in ROUTINGS} for stage, rows in samples.items()}
        copy_cases = aggregate(copy_samples)
        recovery = {}
        for routing in ROUTINGS:
            values = stage_cases["forward_backward"]
            saved = values[routing + "_native"]["max_us"] - values[routing + "_online"]["max_us"]
            copy_us = copy_cases[routing]["max_us"]
            recovery[routing] = {"saved_forward_backward_us": saved,
                                  "copy_all_weight_formats_max_us": copy_us,
                                  "fixed_weight_amortization_calls_estimate": math.floor(copy_us / saved) + 1 if saved > 0 else None,
                                  "sum_of_separately_measured_copy_and_online_us": copy_us + values[routing + "_online"]["max_us"]}
        scratch_bytes = sum(t.numel() * t.element_size() for t in scratch.values())
        metrics = {
            "schema": "megamoe-replication-backward-r5-v2", "shape": args.shape,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "deep_gemm_version": installed, "deep_gemm_module": deep_gemm.__file__,
            "hidden": hidden, "intermediate_hidden": inter, "num_experts": experts,
            "physical_experts": physical_experts, "num_ranks": WORLD, "tokens_per_rank": TOKENS,
            "topk": TOPK, "mma_type": "fp8xfp4", "block_m": blocks,
            "rounds": len(orders()), "graph_iterations": args.graph_iterations,
            "top_level_case": "forward_backward/ratio126_online",
            "cases": stage_cases["forward_backward"], "stage_cases": stage_cases,
            "comparisons": comparisons, "samples": samples,
            "copy_cases": copy_cases, "copy_samples": copy_samples, "recovery": recovery, "plans": plans,
            "aggregation": "各组 max_us 为所有轮次各卡中位数的最大值；同时保留每轮 rank-max 和配对 bootstrap 区间。",
            "timing_boundary": "forward_backward 含在线改写（复制臂）、前向输入写入/dispatch/保存激活/计算/combine、dY 分发、L1 replay、W2/W13 dgrad、远端 dX 写回及 fixed-topk combine。backward 只含上述反向部分，使用计时外刚完成的前向状态。量化、分配、固定计划、外部 barrier 和权重复制不在两个主时间窗内。无 profiler。",
            "copy_timing_boundary": "独立测前向和反向共八个权重张量格式的打包、all_to_all 与写入；12 次完整样本。不含重新量化和计划；分别测得的复制和前后向之和只作估算。",
            "memory": {"common_pool_rows": capacity, "shared_pool_scratch_bytes_per_rank": scratch_bytes,
                       "extra_replica_pool_gradient_bytes": 0, "weight_gradient_allocations": 0,
                       "weight_gradient_calls": 0, "dgrad_calls_per_backward": 1,
                       "local_grad_x_pool_bytes": 0,
                       "counter_state_layout": state_layout,
                       "counter_state_bytes_per_rank": sum(t.numel() * t.element_size() for t in grids.values()),
                       "extra_benchmark_counter_bytes_vs_r4": grids["native"].numel() * 4,
                       "all_weight_formats_bytes_per_expert": bytes_per_expert,
                       "reserved_extra_weight_bytes_per_rank": bytes_per_expert * (physical_local - local),
                       "extra_weight_dequant_scratch_bytes_per_rank": (physical_local - local) * 3 * hidden * inter * 2,
                       "rank_receipts": receipts,
                       "note": "测试同时持有两套对照权重和对称缓冲，不代表生产峰值。梯度临时区为原映射五组的最大容量且两臂完全复用；权重反量化空间是权重暂存，不存 dW。"},
            "correctness": {"status": "pass", "checks": checks, "dW": "NOT_TESTED",
                            "scope": "相同逻辑路由/FP4权重/FP32分数/dY下，复制前向与dX按BF16原始位模式比较，包括符号零；直接与图执行均有限、逐位相等；GPU改写和实际逐专家接收计数符合CPU参考。冻结专家不生成dW；未验证全参数更新。"},
            "harness_fix": "源码诊断：R4 原/扩槽配置共用 grid，同步屏障位置与较大配置 W13 就绪标记重叠；现在独立配置状态。直接高偏斜和跨配置探针独立记录；这些诊断同步在正式计时外，不扩大梯度池、不改kernel。",
            "limitations": "仅冻结专家的合成单层前后向，不代表完整训练或AF。计划静态，复制后扇出可能变化；原映射和扩槽 block_m 可能不同。既有生产计数/MAX主机读取不在此固定形状实验中；两臂均无新增计数主机读取。无 dW 容差或副本梯度归约证据，不准据此交付 full-param。",
            "output_bytes_before_metrics": check_output_bound(args.output), "output_cap_bytes": OUTPUT_CAP,
        }
        payload = json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
        assert len(payload.encode()) <= 256 * 1024
        with (args.output / "metrics.json").open("w") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        for r, value in enumerate(stage_cases["forward_backward"]["ratio126_online"]["rank_us"]):
            print(f"EP: {r}/{WORLD} | 冻结专家1.26倍在线复制前后向 | {value:.6f} us, 含通信及改写", flush=True)
        print("R5_DONE：四卡计算与检查已结束，metrics 已落盘；跳过可能挂起的分布式析构。", flush=True)
    guard.stop()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=tuple(SHAPES), default="dsv41")
    parser.add_argument("--graph-iterations", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("/out"))
    parser.add_argument("--cpu-check", action="store_true")
    parser.add_argument("--import-check", action="store_true")
    args = parser.parse_args()
    if args.cpu_check:
        cpu_check()
        return
    if args.import_check:
        import inspect
        import torch
        import deep_gemm
        params = inspect.signature(deep_gemm.fp8_fp4_mega_moe_backward_dgrad_swiglu).parameters
        assert all(p in params for p in ("rank_uniform_block_m", "grad_route_output", "write_grad_x_pool"))
        assert hasattr(deep_gemm, "mega_moe_backward_combine_grad_x")
        assert not torch.cuda.is_initialized()
        print(f"真实镜像导入通过：{version('deep_gemm')}；CUDA 未初始化", flush=True)
        return
    assert 1 <= args.graph_iterations <= 32
    args.output.mkdir(parents=True, exist_ok=True)
    for key, subdir in (("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "cache"), ("CUDA_CACHE_PATH", "cuda-cache"),
                        ("DG_JIT_CACHE_DIR", "deep-gemm-cache"), ("TRITON_CACHE_DIR", "triton-cache")):
        path = args.output / subdir
        path.mkdir(exist_ok=True)
        os.environ[key] = str(path)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    import torch.multiprocessing as mp
    mp.spawn(worker, args=(args,), nprocs=WORLD)
    print("R5 父进程完成：四个子进程均退出码 0", flush=True)


if __name__ == "__main__":
    main()
