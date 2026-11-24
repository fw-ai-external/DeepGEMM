"""Exercise the native training prefix loop with more experts than warps."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


KERNEL = (
    Path(__file__).resolve().parents[1]
    / "deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_moe.cuh"
)


def prefix_loop():
    source = KERNEL.read_text()
    prefix = source.index("uint32_t prefix = 0;")
    start = source.rfind("if (", 0, prefix)
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


class TestStableExpertPrefix(unittest.TestCase):
    def test_native_loop_covers_experts_beyond_dispatch_warps(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("C++ compiler unavailable")
        # Compile the actual kernel loop. Only CUDA thread identifiers and the
        # backing allocation are supplied by this CPU harness.
        harness = r"""
#include <cstdint>
#include <iostream>
#include <vector>
template <uint32_t kNumGlobalWarps, uint32_t kNumExperts>
bool check() {
    std::vector<uint32_t> storage(kNumGlobalWarps * kNumExperts);
    auto count = [](uint32_t warp, uint32_t expert) {
        return ((warp * 7 + expert * 3) % 11 == 0) ? 0u :
            1u + ((warp + expert) % 5);
    };
    for (uint32_t warp = 0; warp < kNumGlobalWarps; ++warp)
        for (uint32_t expert = 0; expert < kNumExperts; ++expert)
            storage[warp * kNumExperts + expert] = count(warp, expert);
    auto* warp_expert_prefix = storage.data();
    const uint32_t lane_idx = 0;
    for (uint32_t global_warp_idx = 0;
         global_warp_idx < kNumGlobalWarps; ++global_warp_idx) {
""" + prefix_loop() + r"""
    }
    for (uint32_t expert = 0; expert < kNumExperts; ++expert) {
        uint32_t expected = 0;
        for (uint32_t warp = 0; warp < kNumGlobalWarps; ++warp) {
            const auto actual = storage[warp * kNumExperts + expert];
            if (actual != expected) {
                std::cerr << "warps=" << kNumGlobalWarps
                          << " expert=" << expert << " warp=" << warp
                          << " expected=" << expected << " actual=" << actual;
                return false;
            }
            expected += count(warp, expert);
        }
    }
    return true;
}
int main() {
    return !(check<560, 896>() && check<592, 896>() &&
             check<1184, 896>() && check<8, 17>() && check<8, 8>());
}
"""
        with tempfile.TemporaryDirectory(prefix="megamoe-stable-prefix-") as tmp:
            binary = str(Path(tmp) / "prefix")
            subprocess.run(
                [compiler, "-std=c++17", "-O2", "-x", "c++", "-", "-o", binary],
                input=harness,
                text=True,
                check=True,
                capture_output=True,
            )
            result = subprocess.run([binary], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
