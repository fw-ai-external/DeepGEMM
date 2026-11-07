"""只读已有 SQLite，导出逐调用曲线的积分权重；不执行 GPU 工作。"""

import argparse
import bisect
import hashlib
import json
from pathlib import Path
import sqlite3

from bench_mega_moe_curve import RATIOS


def extract(root):
    result = {
        "schema": "megamoe-percall-curve-weights-v1",
        "ratio_grid": list(RATIOS),
        "matching": "四卡 device_kernel 按 start 排序后以序号配对；不证明同层同算子。",
        "limitation": "device_kernel 可能包含非 MoE、dX/dW；耗时比仅作为行数比代理，不是实际专家计数。",
        "traces": {},
    }
    for name in ("native-light-trace", "rot-trace-r1"):
        path = root / name / "execution/targeted-one-FB.sqlite"
        before = path.stat()
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        try:
            connection.execute("PRAGMA query_only=ON")
            ids = [row[0] for row in connection.execute(
                "SELECT id FROM StringIds WHERE value='device_kernel'")]
            assert ids
            placeholders = ",".join("?" for _ in ids)
            query = ("SELECT end-start FROM CUPTI_ACTIVITY_KIND_KERNEL "
                     f"WHERE deviceId=? AND shortName IN ({placeholders}) ORDER BY start")
            per_rank = [[row[0] for row in connection.execute(query, [rank, *ids])]
                        for rank in range(4)]
        finally:
            connection.close()
        assert len({len(rows) for rows in per_rank}) == 1, "不能静默截短调用序列"
        assert path.stat().st_size == before.st_size and path.stat().st_mtime_ns == before.st_mtime_ns
        weighted, samples = [0.0] * len(RATIOS), []
        total, excess, supported_excess, tail_time, tail_excess, tail_calls = 0, 0, 0, 0, 0, 0
        for durations in zip(*per_rank):
            assert min(durations) > 0
            mean = sum(durations) / 4
            ratio = max(durations) / mean
            total += mean
            excess += max(durations) - mean
            samples.append((ratio, mean))
            if ratio > RATIOS[-1]:
                tail_time += mean
                tail_excess += max(durations) - mean
                tail_calls += 1
                continue
            supported_excess += max(durations) - mean
            i = min(bisect.bisect_right(RATIOS, ratio) - 1, len(RATIOS) - 2)
            alpha = (ratio - RATIOS[i]) / (RATIOS[i + 1] - RATIOS[i])
            assert 0 <= alpha <= 1
            weighted[i] += mean * (1 - alpha)
            weighted[i + 1] += mean * alpha
        ratios = sorted(ratio for ratio, _ in samples)
        weights = [value / total for value in weighted]
        assert abs(sum(weights) + tail_time / total - 1) < 1e-12
        result["traces"][name] = {
            "source": str(path),
            "source_bytes": before.st_size,
            "source_mtime_ns": before.st_mtime_ns,
            "duration_rows_sha256": hashlib.sha256(json.dumps(per_rank).encode()).hexdigest(),
            "calls_per_rank": [len(rows) for rows in per_rank],
            "ratio_quantiles": {str(q): ratios[int(q * (len(ratios) - 1))]
                                for q in (0.1, 0.5, 0.9, 0.99)},
            "total_mean_time_us": total / 1000,
            "observed_time_weighted_slowest_excess": excess / total,
            "supported_linear_excess_over_total_mean_time": supported_excess / total,
            "interpolation_weights_of_total_mean_time": weights,
            "unsupported_tail": {
                "ratio_above": RATIOS[-1], "calls": tail_calls,
                "call_fraction": tail_calls / len(ratios),
                "mean_time_fraction": tail_time / total,
                "observed_excess_over_total_mean_time": tail_excess / total,
                "max_ratio": max(ratios),
            },
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = extract(args.root)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    assert len(payload.encode()) < 32 * 1024
    with args.output.open("x") as output:
        output.write(payload)
    for name, trace in result["traces"].items():
        print(name, json.dumps({key: trace[key] for key in (
            "calls_per_rank", "ratio_quantiles", "observed_time_weighted_slowest_excess",
            "unsupported_tail")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
