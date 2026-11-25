#pragma once
#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe_backward.cuh>
namespace deep_gemm {
template <uint32_t kHidden, uint32_t kNumExperts, uint32_t BLOCK_M,
          uint32_t kNumSMs, uint32_t kNumReadyStates>
CUTLASS_GLOBAL void sm100_bf16_dy_pipeline_prepare(
    const int* expert_counts, const uint32_t num_pool_rows,
    cutlass::bfloat16_t* grad_y_weighted_output,
    float* route_weights_output, float* grad_route_output,
    uint32_t* dy_block_ready) {
    constexpr uint32_t kNumThreads = 256;
    constexpr bool kDoReverseDispatch = true;
    constexpr bool kWriteWeighted = true;
    constexpr bool kComputeRouteDot = true;
    DG_STATIC_ASSERT(kHidden == 3584 &&
                     ((kNumExperts == 112 && kNumReadyStates == 150528 && (BLOCK_M == 240 || BLOCK_M == 192)) ||
                      (kNumExperts == 150 && kNumReadyStates == 201600 && BLOCK_M == 240) ||
                      (kNumExperts == 22 && kNumReadyStates == 29568 && BLOCK_M == 240) ||
                      (kNumExperts == 64 && kNumReadyStates == 86016 && BLOCK_M == 240)),
                     "dY prepare requires an explicitly supported development shape");
    DG_DEVICE_ASSERT(math::ceil_div(num_pool_rows, BLOCK_M) <= kNumReadyStates);
    for (uint32_t i = blockIdx.x * kNumThreads + threadIdx.x;
         i < kNumReadyStates; i += kNumSMs * kNumThreads) {
        dy_block_ready[i] = 0;
    }
    uint32_t padded_pool_blocks = 0;
    #pragma unroll
    for (uint32_t expert_idx = 0;
         expert_idx < kNumExperts; ++expert_idx) {
        const int count = __ldg(expert_counts + expert_idx);
        DG_DEVICE_ASSERT(count >= 0 && static_cast<uint32_t>(count) <= num_pool_rows);
        const uint32_t num_tokens = static_cast<uint32_t>(count);
        const uint32_t num_blocks =
            math::ceil_div(num_tokens, BLOCK_M);
        DG_DEVICE_ASSERT((static_cast<uint64_t>(padded_pool_blocks) + num_blocks) * BLOCK_M <= num_pool_rows);
        const uint32_t num_padded_tokens =
            num_blocks * BLOCK_M;
        const uint32_t num_padding_rows =
            num_padded_tokens - num_tokens;
        for (uint64_t linear =
                 static_cast<uint64_t>(blockIdx.x) *
                     kNumThreads +
                 threadIdx.x;
             linear <
                 static_cast<uint64_t>(
                     num_padding_rows) *
                     kHidden;
             linear +=
                 static_cast<uint64_t>(kNumSMs) *
                     kNumThreads) {
            const uint32_t padding_row =
                linear / kHidden;
            const uint32_t col =
                linear -
                static_cast<uint64_t>(padding_row) *
                    kHidden;
            const uint32_t pool_row =
                padded_pool_blocks * BLOCK_M +
                num_tokens + padding_row;
            const uint64_t offset =
                static_cast<uint64_t>(pool_row) *
                    kHidden +
                col;
            if constexpr (kWriteWeighted) {
                grad_y_weighted_output[offset] =
                    cutlass::bfloat16_t(0.0f);
            }
        }
        for (uint32_t padding_row =
                 blockIdx.x * kNumThreads +
                 threadIdx.x;
             padding_row < num_padding_rows;
             padding_row +=
                 kNumSMs * kNumThreads) {
            const uint32_t pool_row =
                padded_pool_blocks * BLOCK_M +
                num_tokens + padding_row;
            if constexpr (kDoReverseDispatch)
                route_weights_output[pool_row] = 0.0f;
            if constexpr (kComputeRouteDot)
                grad_route_output[pool_row] = 0.0f;
        }
        padded_pool_blocks += num_blocks;
    }
    DG_DEVICE_ASSERT(padded_pool_blocks * BLOCK_M <= num_pool_rows);
}
}
