#pragma once

#include <cutlass/cutlass.h>

namespace deep_gemm::epilogue {

// A lane-dependent select kept as one SEL: as a C++ ternary, nvcc turns a quad's selects into a divergent branch
CUTLASS_DEVICE float lane_select(const bool& p, const float& a, const float& b) {
    float r;
    asm("{ .reg .pred p; setp.ne.b32 p, %3, 0; selp.f32 %0, %1, %2, p; }" : "=f"(r) : "f"(a), "f"(b), "r"(static_cast<int>(p)));
    return r;
}

// The SM90 MQA logits reduction across a quad: each of its 4 lanes holds partial sums `v[row][half]` for all `kNumRows`
// rows. Each xor step (1, then 2) halves the rows a lane keeps while it holds more than one and all-reduces once it
// holds one, so a logit is stored once instead of by all 4 lanes (twice for 2 rows, 4 times for 1). Every sum still
// associates as `(a0 + a1) + (a2 + a3)`, as the butterfly all-reduce did.
template <uint32_t kNumRows>
struct QuadReduce {
    static constexpr uint32_t kNumRowsAfterXor1 = kNumRows > 1 ? kNumRows / 2 : 1;
    static constexpr uint32_t kNumOwnedRows = kNumRowsAfterXor1 > 1 ? kNumRowsAfterXor1 / 2 : 1;

    // The first of the `kNumOwnedRows` rows lane `lane_idx` holds after `reduce`
    CUTLASS_DEVICE static uint32_t owned_row(const uint32_t& lane_idx) {
        return (kNumRows > 1 ? (lane_idx & 1) * kNumRowsAfterXor1 : 0) +
               (kNumRowsAfterXor1 > 1 ? ((lane_idx >> 1) & 1) * kNumOwnedRows : 0);
    }

    // Leaves the sums of rows `[owned_row, owned_row + kNumOwnedRows)` in `v[0, kNumOwnedRows)`
    CUTLASS_DEVICE static void reduce(float (&v)[kNumRows][2], const uint32_t& lane_idx) {
        step<kNumRows>(v, lane_idx & 1, 1);
        step<kNumRowsAfterXor1>(v, (lane_idx >> 1) & 1, 2);
    }

private:
    // Lanes with `upper` keep rows `[n / 2, n)` of the first `n`, the others `[0, n / 2)`, each adding its partner's
    // copy, into `v[0, n / 2)`
    template <uint32_t n>
    CUTLASS_DEVICE static void step(float (&v)[kNumRows][2], const bool& upper, const int& lane_mask) {
        if constexpr (n == 1) {
            #pragma unroll
            for (uint32_t h = 0; h < 2; ++ h)
                v[0][h] += __shfl_xor_sync(0xffffffffu, v[0][h], lane_mask);
        } else {
            #pragma unroll
            for (uint32_t r = 0; r < n / 2; ++ r) {
                #pragma unroll
                for (uint32_t h = 0; h < 2; ++ h) {
                    const float keep = lane_select(upper, v[n / 2 + r][h], v[r][h]);
                    const float send = lane_select(upper, v[r][h], v[n / 2 + r][h]);
                    v[r][h] = keep + __shfl_xor_sync(0xffffffffu, send, lane_mask);
                }
            }
        }
    }
};

} // namespace deep_gemm::epilogue
