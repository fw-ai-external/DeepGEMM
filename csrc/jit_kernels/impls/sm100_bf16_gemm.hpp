#pragma once

#include <array>
#include <format>
#include <torch/python.h>

#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm100.hpp"
#include "epilogue_class.hpp"
#include "runtime_utils.hpp"
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

namespace deep_gemm {

static std::string get_bf16_gemm_combine_order_mode_name(
    const std::string& combine_order_mode) {
    if (combine_order_mode == "fixed_topk")
        return "CombineOrderMode::FixedTopK";
    if (combine_order_mode == "deepep")
        return "CombineOrderMode::DeepEP";
    if (combine_order_mode == "deepep_v1")
        return "CombineOrderMode::DeepEPV1";
    DG_HOST_UNREACHABLE("Unsupported combine order mode");
}

class SM100BF16GemmRuntime final {
public:
    struct Args {
        GemmDesc gemm_desc;
        GemmConfig gemm_config;
        deep_jit::cuda::LaunchOptions options;

        void* grouped_layout;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_cd;
        std::shared_ptr<EpilogueClass> epilogue_class;
        int combine_num_ranks = 1;
        bool fuse_combine = false;
        layout::SymBuffer<> combine_sym_buffer{};
        layout::Workspace combine_workspace{
            nullptr, 1, 1, 1, 1, 1};
        cutlass::bfloat16_t* grad_x_output = nullptr;
        cutlass::bfloat16_t* combine_buffer = nullptr;
        const int64_t* combine_topk_ids = nullptr;
        uint32_t combine_num_tokens = 0;
        uint32_t combine_num_max_tokens = 0;
        uint32_t combine_num_topk = 0;
        uint32_t combine_hidden = 0;
        bool combine_reduce = false;
        std::string combine_order_mode = "fixed_topk";
        uint32_t combine_num_extra_threads = 0;
        bool mask_grouped_k_tail = false;
        int k_alignment_override = 0;
    };

    static void compile_and_launch(const std::string& tag, const Args& args) {
        const auto kernel = jit->compile(tag, std::format(R"(
#include <deep_gemm/impls/sm100_bf16_gemm.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_gemm_impl<
        {}, {},
        {}, {}, {},
        {}, {}, {},
        {},
        {}, {}, {},
        {},
        {}, {},
        {}, {},
        {},
        {},
        {}, {},
        {}, {}, {},
        {},
        {},
        {}, {}, {}, {}, {}
    >);
}};
)",
        to_string(args.gemm_desc.major_a), to_string(args.gemm_desc.major_b),
        get_compiled_dim(args.gemm_desc.m, 'm', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.n, 'n', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.k, 'k', args.gemm_desc.compiled_dims),
        args.gemm_config.layout.block_m, args.gemm_config.layout.block_n, args.gemm_config.layout.block_k,
        args.gemm_desc.num_groups,
        args.gemm_config.storage_config.swizzle_a_mode, args.gemm_config.storage_config.swizzle_b_mode, args.gemm_config.storage_config.swizzle_cd_mode,
        args.gemm_config.pipeline_config.num_stages,
        args.gemm_config.launch_config.num_non_epilogue_threads, args.gemm_config.launch_config.num_epilogue_threads,
        args.gemm_config.layout.get_cluster_size(), args.gemm_config.layout.cluster_n > 1,
        args.gemm_config.launch_config.num_sms,
        args.k_alignment_override != 0 ? args.k_alignment_override :
            heuristics_runtime->get_mk_alignment_for_contiguous_layout(),
        args.gemm_config.layout.swap_ab, args.gemm_desc.ensure_zero_padding,
        to_string(args.gemm_desc.gemm_type), args.gemm_desc.with_accumulation,
        to_string(args.gemm_desc.cd_dtype),
        args.epilogue_class->get_epilogue_operator_type(),
        args.gemm_desc.tc_util,
        args.combine_num_ranks, args.fuse_combine,
        get_bf16_gemm_combine_order_mode_name(args.combine_order_mode),
        args.combine_num_extra_threads, args.mask_grouped_k_tail), args.epilogue_class->compiler_options());

        // Launch
        jit->launch(
            kernel, args.options,
            args.grouped_layout, args.gemm_desc.m, args.gemm_desc.n, args.gemm_desc.k, args.epilogue_class->make_epilogue_operator_args(args.gemm_desc.m, args.gemm_desc.n),
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_cd,
            args.combine_sym_buffer, args.combine_workspace,
            args.grad_x_output,
            args.combine_buffer, args.combine_topk_ids,
            args.combine_num_tokens,
            args.combine_num_max_tokens, args.combine_num_topk,
            args.combine_hidden, args.combine_reduce);
    }
};

// A capacity-selected layout must remain valid when runtime M/N change.
// Multicast along a dynamic dimension requires divisibility of its tile count,
// which is not guaranteed by capacity planning. Keep multicast along static
// dimensions available, without padding or allocating to the tuning shape.
struct SM100DynamicBF16GemmArchSpec : SM100ArchSpec {
    static std::vector<Layout> get_layout_candidates(const GemmDesc& desc) {
        auto candidates = SM100ArchSpec::get_layout_candidates(desc);
        std::erase_if(candidates, [&](const Layout& layout) {
            return (desc.compiled_dims.find('m') == std::string::npos && layout.cluster_m > 1) ||
                   (desc.compiled_dims.find('n') == std::string::npos && layout.cluster_n > 1);
        });
        return candidates;
    }
};

static void sm100_bf16_gemm(const torch::Tensor& a,
                            const torch::Tensor& b,
                            const std::optional<torch::Tensor>& c,
                            const torch::Tensor& d,
                            const int& m, const int& n, const int& k,
                            const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                            const std::string& compiled_dims,
                            const std::shared_ptr<EpilogueClass>& epilogue_class,
                            const std::optional<std::array<int, 3>>& tuning_shape = std::nullopt) {
    const auto desc = GemmDesc {
        .gemm_type = GemmType::Normal,
        .kernel_type = KernelType::KernelNoSF,
        .m = m, .n = n, .k = k, .num_groups = 1,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = c.has_value(),
        .num_sms = runtime->get_num_sms(),
        .tc_util = runtime->get_tc_util(),
        .compiled_dims = compiled_dims
    };
    const auto config = [&]() {
        if (!tuning_shape.has_value())
            return get_best_config<SM100ArchSpec>(desc);
        auto tuning_desc = desc;
        const std::array<int, 3> actual_shape{m, n, k};
        for (int i = 0; i < 3; ++i) {
            DG_HOST_ASSERT((*tuning_shape)[i] > 0);
            DG_HOST_ASSERT(compiled_dims.find("mnk"[i]) == std::string::npos ||
                           (*tuning_shape)[i] == actual_shape[i]);
        }
        tuning_desc.m = (*tuning_shape)[0];
        tuning_desc.n = (*tuning_shape)[1];
        tuning_desc.k = (*tuning_shape)[2];
        return get_best_config<SM100DynamicBF16GemmArchSpec>(tuning_desc);
    }();

    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), 1,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                config.storage_config.store_block_m,
                                                config.storage_config.store_block_n,
                                                static_cast<int>(d.stride(-2)), 1,
                                                config.storage_config.swizzle_cd_mode);

    // Compile and launch
    SM100BF16GemmRuntime::compile_and_launch("sm100_bf16_gemm", {
        .gemm_desc = desc,
        .gemm_config = config,
        .options = {
            .num_smem_bytes = config.pipeline_config.smem_size,
            .grid_dim = dim3(config.launch_config.num_sms, 1, 1),
            .block_dim = dim3(config.launch_config.num_threads, 1, 1),
            .cluster_dim = dim3(config.layout.get_cluster_size(), 1, 1),
        },
        .grouped_layout = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd,
        .epilogue_class = epilogue_class
    });
}

static void sm100_m_grouped_bf16_gemm_contiguous(const torch::Tensor& a,
                                                 const torch::Tensor& b,
                                                 const torch::Tensor& d,
                                                 const torch::Tensor& grouped_layout,
                                                 const int& num_groups, const int& m, const int& n, const int& k,
                                                 const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                 const std::string& compiled_dims,
                                                 const bool& use_psum_layout,
                                                 const bool& ensure_zero_padding,
                                                 const std::optional<int>& expected_m_for_psum_layout,
                                                 const std::shared_ptr<EpilogueClass>& epilogue_class) {
    const auto gemm_type = use_psum_layout ?
        GemmType::MGroupedContiguousWithPsumLayout : GemmType::MGroupedContiguous;

    // Only psum layout can use expected m
    if (expected_m_for_psum_layout)
        DG_HOST_ASSERT(use_psum_layout);

    // NOTES: If actual M is dynamic, estimate config via `num_groups` and `expected_m`.
    //        Otherwise, treat the contiguous layout as a whole.
    const auto desc = GemmDesc {
        .gemm_type = gemm_type,
        .kernel_type = KernelType::KernelNoSF,
        .m = m, .n = n, .k = k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = false,
        .num_sms = runtime->get_num_sms(),
        .tc_util = runtime->get_tc_util(),
        .compiled_dims = compiled_dims,
        .ensure_zero_padding = ensure_zero_padding,
        .expected_m = expected_m_for_psum_layout.value_or(m),
        .expected_n = n, .expected_k = k,
        .expected_num_groups = expected_m_for_psum_layout.has_value() ? num_groups : 1
    };
    const auto config = get_best_config<SM100ArchSpec>(desc);

    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                config.storage_config.store_block_m,
                                                config.storage_config.store_block_n,
                                                static_cast<int>(d.stride(-2)), 1,
                                                config.storage_config.swizzle_cd_mode);

    // Compile and launch
    SM100BF16GemmRuntime::compile_and_launch("sm100_bf16_m_grouped_gemm_contiguous", {
        .gemm_desc = desc,
        .gemm_config = config,
        .options = {
            .num_smem_bytes = config.pipeline_config.smem_size,
            .grid_dim = dim3(config.launch_config.num_sms, 1, 1),
            .block_dim = dim3(config.launch_config.num_threads, 1, 1),
            .cluster_dim = dim3(config.layout.get_cluster_size(), 1, 1),
        },
        .grouped_layout = grouped_layout.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd,
        .epilogue_class = epilogue_class
    });
}

static void sm100_m_grouped_bf16_gemm_masked(const torch::Tensor& a,
                                             const torch::Tensor& b,
                                             const torch::Tensor& d,
                                             const torch::Tensor& masked_m,
                                             const int& num_groups, const int& m, const int& n, const int& k,
                                             const int& expected_m,
                                             const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                             const std::string& compiled_dims,
                                             const std::shared_ptr<EpilogueClass>& epilogue_class) {
    const auto desc = GemmDesc {
        .gemm_type = GemmType::MGroupedMasked,
        .kernel_type = KernelType::KernelNoSF,
        .m = m, .n = n, .k = k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = false,
        .num_sms = runtime->get_num_sms(),
        .tc_util = runtime->get_tc_util(),
        .compiled_dims = compiled_dims,
        .expected_m = expected_m, .expected_n = n, .expected_k = k, .expected_num_groups = num_groups
    };
    const auto config = get_best_config<SM100ArchSpec>(desc);

    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), num_groups,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                config.storage_config.store_block_m,
                                                config.storage_config.store_block_n,
                                                static_cast<int>(d.stride(-2)), num_groups,
                                                config.storage_config.swizzle_cd_mode);

    // Compile and launch
    SM100BF16GemmRuntime::compile_and_launch("sm100_bf16_m_grouped_gemm_masked", {
        .gemm_desc = desc,
        .gemm_config = config,
        .options = {
            .num_smem_bytes = config.pipeline_config.smem_size,
            .grid_dim = dim3(config.launch_config.num_sms, 1, 1),
            .block_dim = dim3(config.launch_config.num_threads, 1, 1),
            .cluster_dim = dim3(config.layout.get_cluster_size(), 1, 1),
        },
        .grouped_layout = masked_m.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd,
        .epilogue_class = epilogue_class
    });
}

static void sm100_bf16_k_grouped_gemm(const torch::Tensor& a,
                                      const torch::Tensor& b,
                                      const std::optional<torch::Tensor>& c,
                                      const torch::Tensor& d,
                                      const int& m, const int& n,
                                      const torch::Tensor& grouped_layout,
                                      const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                      const std::string& compiled_dims,
                                      const bool& use_psum_layout,
                                      const std::shared_ptr<EpilogueClass>& epilogue_class) {
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::MN and major_b == cute::UMMA::Major::MN);

    const auto sum_k = static_cast<int>(a.size(0));
    const auto num_groups = static_cast<int>(grouped_layout.numel());
    const auto expected_k = ceil_div(sum_k, num_groups);
    const auto desc = GemmDesc {
        .gemm_type = use_psum_layout ? GemmType::KGroupedContiguousWithPsumLayout : GemmType::KGroupedContiguous,
        .kernel_type = KernelType::KernelNoSF,
        .m = m, .n = n, .k = sum_k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = c.has_value(),
        .num_sms = runtime->get_num_sms(),
        .tc_util = runtime->get_tc_util(),
        .compiled_dims = compiled_dims,
        // NOTES: expected_k is not used in SM100 get_best_config yet.
        .expected_m = m, .expected_n = n, .expected_k = expected_k, .expected_num_groups = num_groups
    };
    const auto config = get_best_config<SM100ArchSpec>(desc);

    // Create tensor descriptors
    const auto tensor_map_a = make_tma_a_desc(cute::UMMA::Major::MN, a, m, sum_k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(0)), 1,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(cute::UMMA::Major::MN, b, n, sum_k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(0)), 1,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_cd = make_tma_3d_desc(d, n, m, num_groups,
                                                config.storage_config.store_block_n,
                                                config.storage_config.store_block_m, 1,
                                                static_cast<int>(d.stride(1)),
                                                static_cast<int>(d.stride(0)),
                                                config.storage_config.swizzle_cd_mode);

    // Compile and launch
    SM100BF16GemmRuntime::compile_and_launch("sm100_bf16_k_grouped_gemm", {
        .gemm_desc = desc,
        .gemm_config = config,
        .options = {
            .num_smem_bytes = config.pipeline_config.smem_size,
            .grid_dim = dim3(config.launch_config.num_sms, 1, 1),
            .block_dim = dim3(config.launch_config.num_threads, 1, 1),
            .cluster_dim = dim3(config.layout.get_cluster_size(), 1, 1),
        },
        .grouped_layout = grouped_layout.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd,
        .epilogue_class = epilogue_class
    });
}

static void sm100_bf16_bhr_hdr_bhd(const torch::Tensor& tensor_a,
                                   const torch::Tensor& tensor_b,
                                   const std::optional<torch::Tensor>& tensor_c,
                                   const torch::Tensor& tensor_d,
                                   const int& b, const int& h, const int& r, const int& d,
                                   const std::string& compiled_dims,
                                   const std::shared_ptr<EpilogueClass>& epilogue_class) {
    const auto desc = GemmDesc {
        .gemm_type = GemmType::Batched,
        .kernel_type = KernelType::KernelNoSF,
        .m = b, .n = d, .k = r, .num_groups = h,
        .a_dtype = tensor_a.scalar_type(), .b_dtype = tensor_b.scalar_type(),
        .cd_dtype = tensor_d.scalar_type(),
        .major_a = cute::UMMA::Major::K, .major_b = cute::UMMA::Major::K,
        .with_accumulation = tensor_c.has_value(),
        .num_sms = runtime->get_num_sms(),
        .tc_util = runtime->get_tc_util(),
        .compiled_dims = compiled_dims
    };
    const auto config = get_best_config<SM100ArchSpec>(desc);

    const auto tensor_map_a = make_tma_3d_desc(tensor_a, r, b, h,
                                               config.layout.block_k, config.storage_config.load_block_m, 1,
                                               tensor_a.stride(0), tensor_a.stride(1),
                                               config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_3d_desc(tensor_b, r, d, h,
                                               config.layout.block_k, config.storage_config.load_block_n, 1,
                                               tensor_b.stride(1), tensor_b.stride(0),
                                               config.storage_config.swizzle_b_mode);
    const auto tensor_map_cd = make_tma_3d_desc(tensor_d, d, b, h,
                                                config.storage_config.store_block_n, config.storage_config.store_block_m, 1,
                                                tensor_d.stride(0), tensor_d.stride(1),
                                                config.storage_config.swizzle_cd_mode);

    // Compile and launch
    SM100BF16GemmRuntime::compile_and_launch("sm100_bf16_bhr_hdr_bhd", {
        .gemm_desc = desc,
        .gemm_config = config,
        .options = {
            .num_smem_bytes = config.pipeline_config.smem_size,
            .grid_dim = dim3(config.launch_config.num_sms, 1, 1),
            .block_dim = dim3(config.launch_config.num_threads, 1, 1),
            .cluster_dim = dim3(config.layout.get_cluster_size(), 1, 1),
        },
        .grouped_layout = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd,
        .epilogue_class = epilogue_class
    });
}

static void sm100_bf16_bhd_hdr_bhr(const torch::Tensor& tensor_a,
                                   const torch::Tensor& tensor_b,
                                   const std::optional<torch::Tensor>& tensor_c,
                                   const torch::Tensor& tensor_d,
                                   const int& b, const int& h, const int& r, const int& d,
                                   const std::string& compiled_dims,
                                   const std::shared_ptr<EpilogueClass>& epilogue_class) {
    const auto desc = GemmDesc {
        .gemm_type = GemmType::Batched,
        .kernel_type = KernelType::KernelNoSF,
        .m = b, .n = r, .k = d, .num_groups = h,
        .a_dtype = tensor_a.scalar_type(), .b_dtype = tensor_b.scalar_type(),
        .cd_dtype = tensor_d.scalar_type(),
        .major_a = cute::UMMA::Major::K, .major_b = cute::UMMA::Major::MN,
        .with_accumulation = tensor_c.has_value(),
        .num_sms = runtime->get_num_sms(),
        .tc_util = runtime->get_tc_util(),
        .compiled_dims = compiled_dims
    };
    const auto config = get_best_config<SM100ArchSpec>(desc);

    const auto tensor_map_a = make_tma_3d_desc(tensor_a, d, b, h,
                                               config.layout.block_k, config.storage_config.load_block_m, 1,
                                               tensor_a.stride(0), tensor_a.stride(1),
                                               config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_3d_desc(tensor_b, r, d, h,
                                               config.storage_config.load_block_n, config.layout.block_k, 1,
                                               tensor_b.stride(1), tensor_b.stride(0),
                                               config.storage_config.swizzle_b_mode);
    const auto tensor_map_cd = make_tma_3d_desc(tensor_d, r, b, h,
                                                config.storage_config.store_block_n, config.storage_config.store_block_m, 1,
                                                tensor_d.stride(0), tensor_d.stride(1),
                                                config.storage_config.swizzle_cd_mode);

    // Compile and launch
    SM100BF16GemmRuntime::compile_and_launch("sm100_bf16_bhd_hdr_bhr", {
        .gemm_desc = desc,
        .gemm_config = config,
        .options = {
            .num_smem_bytes = config.pipeline_config.smem_size,
            .grid_dim = dim3(config.launch_config.num_sms, 1, 1),
            .block_dim = dim3(config.launch_config.num_threads, 1, 1),
            .cluster_dim = dim3(config.layout.get_cluster_size(), 1, 1),
        },
        .grouped_layout = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd,
        .epilogue_class = epilogue_class
    });
}

} // namespace deep_gemm
