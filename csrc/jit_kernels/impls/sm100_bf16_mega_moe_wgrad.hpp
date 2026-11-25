#pragma once

#include "sm100_bf16_gemm.hpp"

namespace deep_gemm {

struct MegaMoEBackwardCombineArgs {
    bool enabled = false;
    int num_ranks = 1;
    layout::SymBuffer<> sym_buffer{};
    layout::Workspace workspace{nullptr, 1, 1, 1, 1, 1};
    cutlass::bfloat16_t* grad_x_output = nullptr;
    cutlass::bfloat16_t* combine_buffer = nullptr;
    const int64_t* topk_ids = nullptr;
    uint32_t num_tokens = 0;
    uint32_t num_max_tokens = 0;
    uint32_t num_topk = 0;
    uint32_t hidden = 0;
    bool reduce = false;
    std::string order_mode = "fixed_topk";
};

// Dedicated single-CTA specialization for MegaMoE local expert wgrad.
// Kernel A stores each expert in a BLOCK_M-padded contiguous pool; callers pass
// those padded counts as grouped_layout so zero tail rows participate harmlessly.
static void sm100_bf16_mega_moe_wgrad_1sm(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& d,
    const torch::Tensor& padded_expert_counts,
    const int pool_block_m,
    const MegaMoEBackwardCombineArgs& combine = {},
    const bool allow_row_strided_inputs = false,
    const std::string& kernel_name = "sm100_bf16_mega_moe_wgrad_1sm") {
    const auto [num_groups, m, n] = get_shape<3>(d);
    const auto [pool_rows_a, m_] = get_shape<2>(a);
    const auto [pool_rows_b, n_] = get_shape<2>(b);
    DG_HOST_ASSERT(m == m_ and n == n_);
    DG_HOST_ASSERT(pool_rows_a == pool_rows_b);
    DG_HOST_ASSERT(
        padded_expert_counts.numel() == num_groups and
        padded_expert_counts.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(
        a.scalar_type() == torch::kBFloat16 and
        b.scalar_type() == torch::kBFloat16 and
        (d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat));
    DG_HOST_ASSERT(d.is_contiguous());
    DG_HOST_ASSERT(
        allow_row_strided_inputs
            ? (a.stride(1) == 1 and b.stride(1) == 1)
            : (a.is_contiguous() and b.is_contiguous()));

    DG_HOST_ASSERT(
        pool_block_m == 16 || pool_block_m == 32 ||
        pool_block_m == 64 || pool_block_m == 96 ||
        pool_block_m == 128 || pool_block_m == 192 ||
        pool_block_m == 240);
    const int kBlockM = deep_jit::get_env<int>(
        "DG_BF16_MEGA_MOE_WGRAD_BLOCK_M", 128);
    // Amortize each A tile and scheduler assignment across twice as much
    // tensor-core work whenever the output width permits a full 256-column
    // tile. Keep the 128-column fallback for non-divisible model dimensions.
    const int kBlockN = deep_jit::get_env<int>(
        "DG_BF16_MEGA_MOE_WGRAD_BLOCK_N",
        n % 256 == 0 ? 256 : 128);
    // Preserve the physical expert offsets while amortizing TMA/barrier cost
    // over K=64. For pools such as 96/240, mask whole 16-wide MMA atoms in the
    // final tile rather than accumulating rows from the following expert.
    const int kBlockK = deep_jit::get_env<int>(
        "DG_BF16_MEGA_MOE_WGRAD_BLOCK_K", 64);
    DG_HOST_ASSERT(kBlockK == 16 or kBlockK == 32 or kBlockK == 64);
    DG_HOST_ASSERT(pool_block_m % 16 == 0);
    const bool mask_grouped_k_tail = pool_block_m % kBlockK != 0;
    const int kNumStages = deep_jit::get_env<int>(
        "DG_BF16_MEGA_MOE_WGRAD_NUM_STAGES", 4);
    const int kSwizzle =
        kBlockK * static_cast<int>(sizeof(cutlass::bfloat16_t));
    const int output_element_size = static_cast<int>(d.element_size());
    const int kStoreBlockN = deep_jit::get_env<int>(
        "DG_BF16_MEGA_MOE_WGRAD_STORE_BLOCK_N", 128 / output_element_size);
    // The output store width is independent of the A/B K tile.
    const int kSwizzleCD = kStoreBlockN * output_element_size;
    DG_HOST_ASSERT(kSwizzleCD == 32 or kSwizzleCD == 64 or kSwizzleCD == 128);
    constexpr int kNumNonEpilogueThreads = 128;
    constexpr int kNumEpilogueThreads = 128;
    // Production combine always uses four warps: the two original non-MMA
    // warps plus two appended warps that leave existing warp IDs unchanged.
    const int num_extra_combine_threads = combine.enabled ? 64 : 0;
    const int num_threads =
        kNumNonEpilogueThreads + kNumEpilogueThreads +
        num_extra_combine_threads;
    const int kSmemSize =
        2 * kBlockM * kStoreBlockN * output_element_size +
        kNumStages * (kBlockM + kBlockN) * kBlockK *
            sizeof(cutlass::bfloat16_t) +
        1024;

    // Join the shared MegaMoE headroom policy: the combine warps enter a
    // whole-grid NVLink barrier (sm100_bf16_gemm.cuh), so this grid must
    // reserve SMs for concurrent communication kernels like every other
    // cooperative mega launch. `jit->device.get_num_sms()` would ignore both
    // the DG_MEGA_MOE_SM_HEADROOM env and the process-wide set_num_sms
    // override and always launch a full-device grid (INC-1291).
    const int num_sms = get_mega_moe_num_sms();
    const auto desc = GemmDesc{
        .gemm_type = GemmType::KGroupedContiguous,
        .kernel_type = KernelType::KernelNoSF,
        .m = m,
        .n = n,
        .k = pool_rows_a,
        .num_groups = num_groups,
        .a_dtype = a.scalar_type(),
        .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = cute::UMMA::Major::MN,
        .major_b = cute::UMMA::Major::MN,
        .with_accumulation = false,
        .num_sms = num_sms,
        .tc_util = 100,
        .compiled_dims = "mn",
        .expected_m = m,
        .expected_n = n,
        .expected_k = pool_rows_a,
        .expected_num_groups = num_groups,
    };
    const auto config = GemmConfig{
        .layout =
            Layout{
                .swap_ab = false,
                .block_m = kBlockM,
                .block_n = kBlockN,
                .block_k = kBlockK,
                .cluster_m = 1,
                .cluster_n = 1,
            },
        .storage_config =
            StorageConfig{
                .load_block_m = kBlockM,
                .load_block_n = kBlockN,
                .store_block_m = kBlockM,
                .store_block_n = kStoreBlockN,
                .swizzle_a_mode = kSwizzle,
                .swizzle_b_mode = kSwizzle,
                .swizzle_cd_mode = kSwizzleCD,
            },
        .pipeline_config =
            PipelineConfig{
                .smem_size = kSmemSize,
                .num_stages = kNumStages,
            },
        .launch_config =
            LaunchConfig{
                .num_sms = num_sms,
                .num_sms_per_cluster = 1,
                .num_threads = num_threads,
                .num_tma_threads = 32,
                .num_math_threads = 32,
                .num_non_epilogue_threads = kNumNonEpilogueThreads,
                .num_epilogue_threads = kNumEpilogueThreads,
            },
    };

    const auto tensor_map_a = make_tma_a_desc(
        cute::UMMA::Major::MN, a, m, pool_rows_a,
        kBlockM, kBlockK, static_cast<int>(a.stride(0)), 1, kSwizzle);
    const auto tensor_map_b = make_tma_b_desc(
        cute::UMMA::Major::MN, b, n, pool_rows_b,
        kBlockN, kBlockK, static_cast<int>(b.stride(0)), 1, kSwizzle);
    // Upstream K-grouped epilogues address [N, M, group] with a 3D TMA
    // store. A flattened 2D descriptor compiles but traps on the first store.
    const auto tensor_map_d = make_tma_3d_desc(
        d, n, m, num_groups, kStoreBlockN, kBlockM, 1,
        static_cast<int>(d.stride(1)), static_cast<int>(d.stride(0)),
        kSwizzleCD);

    const SM100BF16GemmRuntime::Args args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .options =
            make_mega_moe_launch_options(num_sms, num_threads, kSmemSize, 1),
        .grouped_layout = padded_expert_counts.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_d,
        .epilogue_class = std::make_shared<IdentityEpilogue>(),
        .combine_num_ranks = combine.num_ranks,
        .fuse_combine = combine.enabled,
        .combine_sym_buffer = combine.sym_buffer,
        .combine_workspace = combine.workspace,
        .grad_x_output = combine.grad_x_output,
        .combine_buffer = combine.combine_buffer,
        .combine_topk_ids = combine.topk_ids,
        .combine_num_tokens = combine.num_tokens,
        .combine_num_max_tokens = combine.num_max_tokens,
        .combine_num_topk = combine.num_topk,
        .combine_hidden = combine.hidden,
        .combine_reduce = combine.reduce,
        .combine_order_mode = combine.order_mode,
        .combine_num_extra_threads =
            static_cast<uint32_t>(num_extra_combine_threads),
        .mask_grouped_k_tail = mask_grouped_k_tail,
        // Count-based BF16 groups use explicit physical offsets and no SF.
        // Do not inherit unrelated global grouped-GEMM alignment (e.g. 224),
        // which need not be divisible by this kernel's K tile.
        .k_alignment_override = kBlockK,
    };
    SM100BF16GemmRuntime::compile_and_launch(kernel_name, args);
}

} // namespace deep_gemm
