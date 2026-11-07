#pragma once

#include <format>
#include <torch/python.h>

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "../heuristics/mega_moe.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

static std::string get_bf16_activation_type_name(
    const std::string& activation) {
    if (activation == "swiglu")
        return "ActivationType::SwiGLU";
    if (activation == "geglu")
        return "ActivationType::GeGLU";
    DG_HOST_UNREACHABLE("Unsupported activation");
}

static std::string get_route_weight_mode_name(
    const std::string& route_weight_mode) {
    if (route_weight_mode == "pre_down")
        return "RouteWeightMode::PreDown";
    if (route_weight_mode == "post_down")
        return "RouteWeightMode::PostDown";
    DG_HOST_UNREACHABLE("Unsupported route weight mode");
}

static std::string get_combine_order_mode_name(
    const std::string& combine_order_mode) {
    if (combine_order_mode == "fixed_topk")
        return "CombineOrderMode::FixedTopK";
    if (combine_order_mode == "deepep")
        return "CombineOrderMode::DeepEP";
    if (combine_order_mode == "deepep_v1")
        return "CombineOrderMode::DeepEPV1";
    DG_HOST_UNREACHABLE("Unsupported combine order mode");
}

static void sm100_bf16_mega_moe(
    const torch::Tensor& y,
    const std::optional<torch::Tensor>& saved_l1_preact,
    const torch::Tensor& l1_acts, const torch::Tensor& l2_acts,
    const torch::Tensor& shared_l1_acts, const torch::Tensor& shared_l2_acts,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const torch::Tensor& shared_l1_weights, const torch::Tensor& shared_l2_weights,
    const torch::Tensor& sm_locality_domains,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_shared_experts,
    const int& num_tokens, const int& num_config_tokens,
    const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const std::string& activation,
    const float& activation_clamp,
    const bool& fast_math,
    const std::string& route_weight_mode,
    const std::optional<torch::Tensor>& saved_h_unweighted,
    const std::optional<torch::Tensor>& saved_h_weighted,
    const std::optional<torch::Tensor>& saved_down_unweighted,
    const std::string& combine_order_mode,
    const std::optional<torch::Tensor>& precomputed_route_counts,
    const std::optional<int>& active_pool_rows,
    const std::optional<torch::Tensor>& route_count_mismatch,
    const std::optional<torch::Tensor>& saved_x
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_ring_tokens = static_cast<int>(l1_acts.size(0));
    const auto shared_intermediate_hidden = intermediate_hidden * num_shared_experts;

    // Heuristics
    const auto config = get_mega_moe_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_config_tokens, num_topk,
        hidden, intermediate_hidden,
        num_ring_tokens, 0, MmaKind::BF16);
    const auto num_max_pool_tokens =
        layout::get_num_max_pool_tokens(
            num_ranks, num_max_tokens_per_rank, num_topk,
            num_experts_per_rank);
    const auto num_saved_pool_tokens =
        active_pool_rows.value_or(num_max_pool_tokens);
    DG_HOST_ASSERT(
        num_saved_pool_tokens > 0 &&
        num_saved_pool_tokens <= num_max_pool_tokens);
    if (saved_l1_preact.has_value()) {
        DG_HOST_ASSERT(saved_l1_preact->scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(saved_l1_preact->is_contiguous());
        DG_HOST_ASSERT(
            saved_l1_preact->sizes() ==
            torch::IntArrayRef(
                {num_saved_pool_tokens, 2 * intermediate_hidden}));
    }
    DG_HOST_ASSERT(
        route_weight_mode == "pre_down" ||
        route_weight_mode == "post_down");
    DG_HOST_ASSERT(
        combine_order_mode == "fixed_topk" ||
        combine_order_mode == "deepep" ||
        combine_order_mode == "deepep_v1");
    DG_HOST_ASSERT(
        saved_h_unweighted.has_value() ==
        saved_h_weighted.has_value());
    if (saved_h_unweighted.has_value()) {
        for (const auto* saved :
             {&*saved_h_unweighted, &*saved_h_weighted}) {
            DG_HOST_ASSERT(
                saved->scalar_type() == torch::kBFloat16);
            DG_HOST_ASSERT(saved->is_contiguous());
            DG_HOST_ASSERT(
                saved->sizes() == torch::IntArrayRef(
                    {num_saved_pool_tokens, intermediate_hidden}));
        }
    }
    if (saved_down_unweighted.has_value()) {
        DG_HOST_ASSERT(
            saved_down_unweighted->scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(saved_down_unweighted->is_contiguous());
        DG_HOST_ASSERT(
            saved_down_unweighted->sizes() ==
            torch::IntArrayRef({num_saved_pool_tokens, hidden}));
    }
    if (saved_x.has_value()) {
        DG_HOST_ASSERT(saved_x->scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(saved_x->is_contiguous());
        DG_HOST_ASSERT(
            saved_x->sizes() ==
            torch::IntArrayRef({num_saved_pool_tokens, hidden}));
    }
    // The BF16 kernel has a single store block size for both layers
    DG_HOST_ASSERT(config.store_block_m_l1 == config.store_block_m_l2);

    // Make tensormap
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden, config.num_ring_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l1_weights = make_tma_weights_3d_desc(l1_weights, config.block_k, config.load_block_n, config.swizzle_weights_mode);
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       intermediate_hidden, config.num_ring_tokens,
                                                       config.block_n / 2, config.store_block_m_l1,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       config.swizzle_acts_mode);
    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     intermediate_hidden, config.num_ring_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l2_weights = make_tma_weights_3d_desc(l2_weights, config.block_k, config.load_block_n, config.swizzle_weights_mode);
    const auto tensor_map_down_unweighted =
        saved_down_unweighted.has_value()
        ? make_tma_2d_desc(
              *saved_down_unweighted,
              hidden, saved_down_unweighted->size(0),
              config.block_n, config.store_block_m_l2,
              static_cast<int>(saved_down_unweighted->stride(-2)),
              config.swizzle_acts_mode)
        : tensor_map_l2_acts;

    const auto tensor_map_shared_l1_acts = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l1_acts,
        hidden, num_max_tokens_per_rank,
        config.block_k, config.load_block_m,
        static_cast<int>(shared_l1_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l1_acts;
    const auto tensor_map_shared_l1_weights = num_shared_experts > 0 ?
        make_tma_weights_2d_desc(shared_l1_weights, config.block_k, config.load_block_n, config.swizzle_weights_mode) : tensor_map_l1_weights;
    const auto tensor_map_shared_l1_output = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l2_acts,
        shared_intermediate_hidden, num_max_tokens_per_rank,
        config.block_n / 2, config.store_block_m_l1,
        static_cast<int>(shared_l2_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l1_output;
    const auto tensor_map_shared_l2_acts = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l2_acts,
        shared_intermediate_hidden, num_max_tokens_per_rank,
        config.block_k, config.load_block_m,
        static_cast<int>(shared_l2_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l2_acts;
    const auto tensor_map_shared_l2_weights = num_shared_experts > 0 ?
        make_tma_weights_2d_desc(shared_l2_weights, config.block_k, config.load_block_n, config.swizzle_weights_mode) : tensor_map_l2_weights;

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();

    const auto num_sms = get_mega_moe_num_sms();

    // Compile
    const auto kernel = jit->compile("sm100_bf16_mega_moe", std::format(R"(
#include <deep_gemm/impls/sm100_bf16_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_mega_moe_impl<
        {},
        {}, {},
        {}, {},
        {}, {}, {},
        {},
        {},
        {},
        {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {}, {}, {}, {}, {}, {}, {}, {}, {}
    >);
}};
)", num_max_tokens_per_rank,
        hidden, intermediate_hidden,
        num_experts, num_shared_experts,
        num_topk,
        config.block_m, config.block_n, config.block_k,
        config.store_block_m_l1,
        config.num_ring_tokens,
        config.num_stages,
        config.num_bytes_per_pull,
        config.num_dispatch_threads, config.num_non_epilogue_threads, config.num_epilogue_threads,
        num_sms, num_ranks,
        to_string(activation_clamp),
        fast_math ? "true" : "false",
        get_bf16_activation_type_name(activation), saved_l1_preact.has_value(),
        saved_h_unweighted.has_value(), get_route_weight_mode_name(route_weight_mode),
        get_combine_order_mode_name(combine_order_mode), saved_down_unweighted.has_value(), saved_x.has_value(),
        y.scalar_type() == torch::kFloat32));

    // Launch
    jit->launch(
        kernel, {
            .num_smem_bytes = config.smem_size,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads, 1, 1),
            .cluster_dim = dim3(2, 1, 1),
        },
        y.data_ptr(),
        saved_l1_preact.has_value() ? saved_l1_preact->data_ptr() : nullptr,
        saved_h_unweighted.has_value() ? saved_h_unweighted->data_ptr() : nullptr,
        saved_h_weighted.has_value() ? saved_h_weighted->data_ptr() : nullptr,
        saved_x.has_value() ? saved_x->data_ptr() : nullptr,
        cumulative_local_expert_recv_stats_ptr,
        precomputed_route_counts.has_value() ? precomputed_route_counts->data_ptr<int>() : nullptr,
        route_count_mismatch.has_value() ? route_count_mismatch->data_ptr<int>() : nullptr,
        num_tokens,
        num_saved_pool_tokens,
        layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        tensor_map_l1_acts,
        tensor_map_l1_weights,
        tensor_map_l1_output,
        tensor_map_l2_acts,
        tensor_map_l2_weights,
        tensor_map_down_unweighted,
        tensor_map_shared_l1_acts,
        tensor_map_shared_l1_weights,
        tensor_map_shared_l1_output,
        tensor_map_shared_l2_acts,
        tensor_map_shared_l2_weights,
        sm_locality_domains.data_ptr<uint8_t>());
}

} // namespace deep_gemm
