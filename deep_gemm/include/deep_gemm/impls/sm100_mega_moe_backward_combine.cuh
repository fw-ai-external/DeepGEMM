#pragma once

#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/utils.cuh>

namespace deep_gemm {

// Reduce fixed source-slot planes written by the expert-major backward.
// The arithmetic deliberately matches the fused BF16-wgrad combine path:
// FixedTopK accumulates slots in slot order, while DeepEP modes round one
// partial per destination rank to BF16 before the cross-rank accumulation.
template <
    uint32_t kNumRanks,
    uint32_t kNumLocalExperts,
    CombineOrderMode kCombineOrderMode>
CUTLASS_GLOBAL void sm100_mega_moe_backward_combine_grad_x(
    cutlass::bfloat16_t* grad_x_output,
    const cutlass::bfloat16_t* combine_buffer,
    const int64_t* topk_ids,
    uint32_t num_tokens,
    uint32_t num_max_tokens,
    uint32_t num_topk,
    uint32_t hidden) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)) || defined(__CLION_IDE__)
    constexpr uint32_t kValuesPerVector =
        2 * sizeof(uint4) / sizeof(cutlass::bfloat16_t);
    const uint32_t num_vectors_per_token =
        hidden / kValuesPerVector;
    const uint64_t num_vectors =
        static_cast<uint64_t>(num_tokens) *
        num_vectors_per_token;

    for (uint64_t linear =
             static_cast<uint64_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         linear < num_vectors;
         linear += static_cast<uint64_t>(gridDim.x) *
                   blockDim.x) {
        const uint32_t token_idx =
            linear / num_vectors_per_token;
        const uint32_t vector_idx =
            linear -
            static_cast<uint64_t>(token_idx) *
                num_vectors_per_token;
        float values[kValuesPerVector] = {0.0f};

        const auto accumulate_slot =
            [&](float* destination, uint32_t slot) {
                const uint64_t packed_idx =
                    ((static_cast<uint64_t>(slot) *
                          num_max_tokens +
                      token_idx) *
                         num_vectors_per_token +
                     vector_idx) *
                    2;
                uint4 packed[2];
                packed[0] =
                    reinterpret_cast<const uint4*>(
                        combine_buffer)[packed_idx];
                packed[1] =
                    reinterpret_cast<const uint4*>(
                        combine_buffer)[packed_idx + 1];
                const auto* packed_values =
                    reinterpret_cast<
                        const cutlass::bfloat16_t*>(packed);
                #pragma unroll
                for (uint32_t i = 0;
                     i < kValuesPerVector; ++i) {
                    destination[i] +=
                        static_cast<float>(packed_values[i]);
                }
            };

        if constexpr (
            kCombineOrderMode == CombineOrderMode::FixedTopK) {
            for (uint32_t slot = 0; slot < num_topk; ++slot)
                accumulate_slot(values, slot);
        } else {
            // H3584 contains 224 vectors/token: every participating warp
            // owns one token, including odd token counts and the final warp.
            // Keep scalar fallback for other layouts, launch sizes and modes.
            if (kNumRanks == 6 && kNumLocalExperts == 150 &&
                kCombineOrderMode == CombineOrderMode::DeepEP &&
                hidden == 3584 && num_topk == 16 && blockDim.x % 32 == 0) {
                constexpr uint32_t full_warp = 0xffffffffu;
                const uint32_t lane = threadIdx.x & 31u;
                const int64_t expert = lane < num_topk
                    ? topk_ids[static_cast<uint64_t>(token_idx) * num_topk + lane]
                    : -1;
                const int owner = expert >= 0 ? static_cast<int>(expert / kNumLocalExperts) : -1;
                const uint32_t same_owner = __match_any_sync(full_warp, owner);
                const uint32_t own_mask = owner >= 0 ? same_owner : 0u;
                for (uint32_t last_slot = 0; last_slot < num_topk; ++last_slot) {
                    uint32_t rank_mask = __shfl_sync(full_warp, own_mask, last_slot);
                    // DeepEP visits owners by their last occurrence in top-k.
                    // Every lane sees the same mask and takes the same branch.
                    if (!rank_mask || 31u - __clz(rank_mask) != last_slot)
                        continue;
                    float rank_values[kValuesPerVector] = {0.0f};
                    while (rank_mask) {
                        const uint32_t selected_slot = __ffs(rank_mask) - 1;
                        rank_mask &= rank_mask - 1;
                        accumulate_slot(rank_values, selected_slot);
                    }
                    #pragma unroll
                    for (uint32_t i = 0; i < kValuesPerVector; ++i) {
                        const cutlass::bfloat16_t rank_partial(rank_values[i]);
                        values[i] += static_cast<float>(rank_partial);
                    }
                }
            } else {
            DG_DEVICE_ASSERT(topk_ids != nullptr);
            DG_DEVICE_ASSERT(num_topk <= 32);
            const uint32_t num_rank_iterations =
                kCombineOrderMode == CombineOrderMode::DeepEPV1
                ? kNumRanks
                : num_topk;
            for (uint32_t rank_iteration = 0;
                 rank_iteration < num_rank_iterations;
                 ++rank_iteration) {
                int destination_rank;
                if constexpr (
                    kCombineOrderMode ==
                    CombineOrderMode::DeepEPV1) {
                    destination_rank =
                        static_cast<int>(rank_iteration);
                    bool rank_is_present = false;
                    for (uint32_t slot = 0;
                         slot < num_topk; ++slot) {
                        const int64_t expert_idx =
                            topk_ids[
                                static_cast<uint64_t>(
                                    token_idx) *
                                    num_topk +
                                slot];
                        rank_is_present |=
                            expert_idx >= 0 &&
                            expert_idx / kNumLocalExperts ==
                                destination_rank;
                    }
                    if (!rank_is_present)
                        continue;
                } else {
                    const int64_t expert_idx =
                        topk_ids[
                            static_cast<uint64_t>(token_idx) *
                                num_topk +
                            rank_iteration];
                    if (expert_idx < 0)
                        continue;
                    destination_rank =
                        static_cast<int>(
                            expert_idx / kNumLocalExperts);
                    bool appears_later = false;
                    for (uint32_t slot = rank_iteration + 1;
                         slot < num_topk; ++slot) {
                        const int64_t later_expert_idx =
                            topk_ids[
                                static_cast<uint64_t>(
                                    token_idx) *
                                    num_topk +
                                slot];
                        appears_later |=
                            later_expert_idx >= 0 &&
                            later_expert_idx /
                                    kNumLocalExperts ==
                                destination_rank;
                    }
                    if (appears_later)
                        continue;
                }

                float rank_values[kValuesPerVector] = {0.0f};
                uint32_t rank_mask = 0;
                for (uint32_t slot = 0;
                     slot < num_topk; ++slot) {
                    const int64_t expert_idx =
                        topk_ids[
                            static_cast<uint64_t>(token_idx) *
                                num_topk +
                            slot];
                    if (expert_idx >= 0 &&
                        expert_idx / kNumLocalExperts ==
                            destination_rank) {
                        rank_mask |= 1u << slot;
                    }
                }
                while (rank_mask) {
                    const uint32_t selected_slot =
                        __ffs(rank_mask) - 1;
                    rank_mask &= ~(1u << selected_slot);
                    accumulate_slot(
                        rank_values, selected_slot);
                }
                #pragma unroll
                for (uint32_t i = 0;
                     i < kValuesPerVector; ++i) {
                    const cutlass::bfloat16_t rank_partial(
                        rank_values[i]);
                    values[i] +=
                        static_cast<float>(rank_partial);
                }
            }
            }
        }

        uint4 packed_output[2];
        auto* output_values =
            reinterpret_cast<cutlass::bfloat16_t*>(
                packed_output);
        #pragma unroll
        for (uint32_t i = 0; i < kValuesPerVector; ++i)
            output_values[i] =
                cutlass::bfloat16_t(values[i]);
        auto* output =
            reinterpret_cast<uint4*>(grad_x_output) +
            linear * 2;
        output[0] = packed_output[0];
        output[1] = packed_output[1];
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0)
        DG_DEVICE_ASSERT(
            false && "This kernel only supports sm_100f");
#endif
}

}  // namespace deep_gemm
