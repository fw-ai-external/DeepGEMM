#pragma once

#include "../jit_kernels/impls/sm100_bf16_mega_moe_wgrad.hpp"
#include "../jit_kernels/impls/sm100_fp8_fp4_mega_moe_backward.hpp"
#include "../jit_kernels/impls/sm100_bf16_mega_moe_side_lora_backward.hpp"

namespace deep_gemm::mega_backward {

static void fp8_fp4_mega_moe_backward_dgrad_swiglu_v2(
    const torch::Tensor& gate_up_output,
    const torch::Tensor& grad_h_output,
    const torch::Tensor& grad_gate_up_output,
    const torch::Tensor& h_act_output,
    const torch::Tensor& h_weighted_output,
    const torch::Tensor& x_pool_output,
    const torch::Tensor& grad_x_pool_output,
    const torch::Tensor& l1_acts,
    const torch::Tensor& l1_acts_sf,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const torch::Tensor& grad_ye,
    const torch::Tensor& route_weights,
    const std::tuple<torch::Tensor, torch::Tensor>& w2_weights_tuple,
    const torch::Tensor& w2_dequant_scratch,
    const std::tuple<torch::Tensor, torch::Tensor>& w13_weights_tuple,
    const torch::Tensor& w13_dequant_scratch,
    const torch::Tensor& expert_counts,
    const torch::Tensor& grid_sync_counter,
    const float& activation_limit,
    const bool& compute_w13_dgrad,
    const int& block_m,
    const bool& direct_remote_grad_x,
    const bool& write_grad_x_pool,
    const bool& clear_wgrad_padding,
    const std::optional<torch::Tensor>& backward_grad_y,
    const std::optional<torch::Tensor>& backward_topk_weights,
    const std::optional<torch::Tensor>& backward_grad_route,
    const std::optional<torch::Tensor>& token_src_metadata,
    const std::vector<int64_t>& backward_sym_buffer_ptrs,
    const int& backward_rank,
    const int& num_max_tokens_per_rank,
    const int& num_topk,
    const std::string& route_weight_mode,
    const std::optional<torch::Tensor>& grad_y_unweighted_output,
    const std::optional<torch::Tensor>& down_unweighted_output,
    const std::optional<torch::Tensor>& grad_route_output) {
    const auto [l1_weights, l1_weights_sf] = l1_weights_tuple;
    const auto [w2_weights, w2_scales] = w2_weights_tuple;
    const auto [w13_weights, w13_scales] = w13_weights_tuple;
    deep_gemm::sm100_fp8_fp4_mega_moe_backward_dgrad_swiglu(
        gate_up_output, grad_h_output, grad_gate_up_output, h_act_output,
        h_weighted_output, x_pool_output, grad_x_pool_output,
        l1_acts, l1_acts_sf, l1_weights, l1_weights_sf,
        grad_ye, route_weights, w2_weights, w2_scales,
        w2_dequant_scratch, w13_weights, w13_scales,
        w13_dequant_scratch, expert_counts, grid_sync_counter,
        activation_limit, compute_w13_dgrad,
        direct_remote_grad_x, write_grad_x_pool,
        clear_wgrad_padding,
        block_m,
        backward_sym_buffer_ptrs, backward_rank,
        num_max_tokens_per_rank, num_topk,
        backward_grad_y, backward_topk_weights,
        backward_grad_route, token_src_metadata, route_weight_mode,
        grad_y_unweighted_output.value_or(grad_ye),
        down_unweighted_output, grad_route_output);
}

// Legacy raw binding: preserve the original positional ABI and PRE_DOWN
// behavior. Expanded route-gradient publication is exposed through _v2.
static void fp8_fp4_mega_moe_backward_dgrad_swiglu(
    const torch::Tensor& gate_up_output,
    const torch::Tensor& grad_h_output,
    const torch::Tensor& grad_gate_up_output,
    const torch::Tensor& h_act_output,
    const torch::Tensor& h_weighted_output,
    const torch::Tensor& x_pool_output,
    const torch::Tensor& grad_x_pool_output,
    const torch::Tensor& l1_acts,
    const torch::Tensor& l1_acts_sf,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const torch::Tensor& grad_ye,
    const torch::Tensor& route_weights,
    const std::tuple<torch::Tensor, torch::Tensor>& w2_weights_tuple,
    const torch::Tensor& w2_dequant_scratch,
    const std::tuple<torch::Tensor, torch::Tensor>& w13_weights_tuple,
    const torch::Tensor& w13_dequant_scratch,
    const torch::Tensor& expert_counts,
    const torch::Tensor& grid_sync_counter,
    const float& activation_limit,
    const bool& compute_w13_dgrad,
    const int& block_m,
    const bool& direct_remote_grad_x,
    const bool& write_grad_x_pool,
    const bool& clear_wgrad_padding,
    const std::optional<torch::Tensor>& backward_grad_y,
    const std::optional<torch::Tensor>& backward_topk_weights,
    const std::optional<torch::Tensor>& token_src_metadata,
    const std::vector<int64_t>& backward_sym_buffer_ptrs,
    const int& backward_rank,
    const int& num_max_tokens_per_rank,
    const int& num_topk) {
    fp8_fp4_mega_moe_backward_dgrad_swiglu_v2(
        gate_up_output, grad_h_output, grad_gate_up_output,
        h_act_output, h_weighted_output, x_pool_output,
        grad_x_pool_output, l1_acts, l1_acts_sf,
        l1_weights_tuple, grad_ye, route_weights,
        w2_weights_tuple, w2_dequant_scratch,
        w13_weights_tuple, w13_dequant_scratch,
        expert_counts, grid_sync_counter, activation_limit,
        compute_w13_dgrad, block_m, direct_remote_grad_x,
        write_grad_x_pool, clear_wgrad_padding,
        backward_grad_y, backward_topk_weights, std::nullopt,
        token_src_metadata, backward_sym_buffer_ptrs,
        backward_rank, num_max_tokens_per_rank, num_topk,
        "pre_down", grad_ye, std::nullopt, std::nullopt);
}

static void bf16_mega_moe_backward_dgrad_v2(
    const torch::Tensor& gate_up_output,
    const torch::Tensor& grad_h_output,
    const torch::Tensor& grad_gate_up_output,
    const torch::Tensor& h_act_output,
    const torch::Tensor& h_weighted_output,
    const torch::Tensor& x_pool_output,
    const torch::Tensor& grad_x_pool_output,
    const torch::Tensor& grad_route_output,
    const torch::Tensor& grad_ye,
    const torch::Tensor& grad_y_unweighted_output,
    const torch::Tensor& route_weights,
    const torch::Tensor& w2_weights,
    const torch::Tensor& w13_weights,
    const torch::Tensor& expert_counts,
    const torch::Tensor& grid_sync_counter,
    const float& activation_limit,
    const std::string& activation,
    const bool& fast_math,
    const std::string& route_weight_mode,
    const torch::Tensor& down_unweighted_output,
    const int& block_m,
    const bool& direct_remote_grad_x,
    const bool& write_grad_x_pool,
    const bool& clear_wgrad_padding,
    const torch::Tensor& backward_grad_y,
    const torch::Tensor& backward_x,
    const torch::Tensor& backward_topk_weights,
    const std::optional<torch::Tensor>& backward_grad_route,
    const torch::Tensor& token_src_metadata,
    const std::vector<int64_t>& backward_sym_buffer_ptrs,
    const int& backward_rank,
    const int& num_max_tokens_per_rank,
    const int& num_topk,
    const std::string& combine_order_mode,
    const std::string& memory_mode,
    const std::optional<torch::Tensor>& kernel_trace =
        std::nullopt,
    const float situ_beta = 4.0f,
    const float situ_linear_beta = std::numeric_limits<float>::infinity(),
    const std::optional<torch::Tensor>& w3_weights = std::nullopt,
    const bool vectorized_grad_x_store = true,
    const bool wide_grad_x_store = false) {
    deep_gemm::sm100_bf16_mega_moe_backward_dgrad(
        gate_up_output, grad_h_output, grad_gate_up_output,
        h_act_output, h_weighted_output, x_pool_output,
        grad_x_pool_output, grad_route_output, grad_ye,
        grad_y_unweighted_output,
        route_weights, w2_weights, w13_weights,
        expert_counts, grid_sync_counter, activation_limit,
        activation, fast_math, route_weight_mode,
        combine_order_mode,
        down_unweighted_output, block_m,
        direct_remote_grad_x, write_grad_x_pool,
        clear_wgrad_padding, backward_grad_y, backward_x,
        backward_topk_weights, backward_grad_route,
        token_src_metadata,
        backward_sym_buffer_ptrs, backward_rank,
        num_max_tokens_per_rank, num_topk, memory_mode,
        kernel_trace, situ_beta, situ_linear_beta, w3_weights,
        vectorized_grad_x_store, wide_grad_x_store);
}

static void bf16_mega_moe_backward_dgrad(
    const torch::Tensor& gate_up_output,
    const torch::Tensor& grad_h_output,
    const torch::Tensor& grad_gate_up_output,
    const torch::Tensor& h_act_output,
    const torch::Tensor& h_weighted_output,
    const torch::Tensor& x_pool_output,
    const torch::Tensor& grad_x_pool_output,
    const torch::Tensor& grad_route_output,
    const torch::Tensor& grad_ye,
    const torch::Tensor& grad_y_unweighted_output,
    const torch::Tensor& route_weights,
    const torch::Tensor& w2_weights,
    const torch::Tensor& w13_weights,
    const torch::Tensor& expert_counts,
    const torch::Tensor& grid_sync_counter,
    const float& activation_limit,
    const std::string& activation,
    const bool& fast_math,
    const std::string& route_weight_mode,
    const torch::Tensor& down_unweighted_output,
    const int& block_m,
    const bool& direct_remote_grad_x,
    const bool& write_grad_x_pool,
    const bool& clear_wgrad_padding,
    const torch::Tensor& backward_grad_y,
    const torch::Tensor& backward_x,
    const torch::Tensor& backward_topk_weights,
    const torch::Tensor& token_src_metadata,
    const std::vector<int64_t>& backward_sym_buffer_ptrs,
    const int& backward_rank,
    const int& num_max_tokens_per_rank,
    const int& num_topk,
    const std::string& combine_order_mode,
    const std::string& memory_mode,
    const std::optional<torch::Tensor>& kernel_trace =
        std::nullopt) {
    bf16_mega_moe_backward_dgrad_v2(
        gate_up_output, grad_h_output, grad_gate_up_output,
        h_act_output, h_weighted_output, x_pool_output,
        grad_x_pool_output, grad_route_output, grad_ye,
        grad_y_unweighted_output, route_weights, w2_weights,
        w13_weights, expert_counts, grid_sync_counter,
        activation_limit, activation, fast_math,
        route_weight_mode, down_unweighted_output, block_m,
        direct_remote_grad_x, write_grad_x_pool,
        clear_wgrad_padding, backward_grad_y, backward_x,
        backward_topk_weights, std::nullopt,
        token_src_metadata, backward_sym_buffer_ptrs,
        backward_rank, num_max_tokens_per_rank, num_topk,
        combine_order_mode, memory_mode, kernel_trace);
}

static void bf16_mega_moe_backward_post_down_prelude_v2(
    const torch::Tensor& grad_y_unweighted_output,
    const torch::Tensor& grad_y_weighted_output,
    const torch::Tensor& x_pool_output,
    const torch::Tensor& route_weights_output,
    const torch::Tensor& grad_route_output,
    const torch::Tensor& down_unweighted_output,
    const torch::Tensor& expert_counts,
    const torch::Tensor& backward_grad_y,
    const torch::Tensor& backward_x,
    const torch::Tensor& backward_topk_weights,
    const std::optional<torch::Tensor>& backward_grad_route,
    const torch::Tensor& token_src_metadata,
    const std::vector<int64_t>& backward_sym_buffer_ptrs,
    const int& backward_rank,
    const int& num_topk,
    const int& block_m,
    const std::string& combine_order_mode,
    const bool& do_reverse_dispatch,
    const bool& compute_route_dot,
    const bool& write_weighted,
    const bool& synchronize_ranks,
    const bool& synchronize_after_dispatch,
    const bool& barrier_only,
    const bool& x_prepared,
    const int& route_prelude_threads) {
    deep_gemm::
        sm100_bf16_mega_moe_backward_post_down_prelude(
            grad_y_unweighted_output,
            grad_y_weighted_output, x_pool_output,
            route_weights_output, grad_route_output,
            down_unweighted_output, expert_counts,
            backward_grad_y, backward_x,
            backward_topk_weights, backward_grad_route,
            token_src_metadata,
            backward_sym_buffer_ptrs, backward_rank,
            num_topk, block_m, combine_order_mode,
            do_reverse_dispatch, compute_route_dot,
            write_weighted, synchronize_ranks,
            synchronize_after_dispatch,
            barrier_only, x_prepared,
            route_prelude_threads);
}

static void bf16_mega_moe_backward_post_down_prelude(
    const torch::Tensor& grad_y_unweighted_output,
    const torch::Tensor& grad_y_weighted_output,
    const torch::Tensor& x_pool_output,
    const torch::Tensor& route_weights_output,
    const torch::Tensor& grad_route_output,
    const torch::Tensor& down_unweighted_output,
    const torch::Tensor& expert_counts,
    const torch::Tensor& backward_grad_y,
    const torch::Tensor& backward_x,
    const torch::Tensor& backward_topk_weights,
    const torch::Tensor& token_src_metadata,
    const std::vector<int64_t>& backward_sym_buffer_ptrs,
    const int& backward_rank,
    const int& num_topk,
    const int& block_m,
    const std::string& combine_order_mode,
    const bool& do_reverse_dispatch,
    const bool& compute_route_dot,
    const bool& write_weighted,
    const bool& synchronize_ranks,
    const bool& synchronize_after_dispatch,
    const bool& barrier_only,
    const bool& x_prepared,
    const int& route_prelude_threads) {
    bf16_mega_moe_backward_post_down_prelude_v2(
        grad_y_unweighted_output, grad_y_weighted_output,
        x_pool_output, route_weights_output, grad_route_output,
        down_unweighted_output, expert_counts, backward_grad_y,
        backward_x, backward_topk_weights,
        std::nullopt, token_src_metadata,
        backward_sym_buffer_ptrs, backward_rank, num_topk,
        block_m, combine_order_mode, do_reverse_dispatch,
        compute_route_dot, write_weighted, synchronize_ranks,
        synchronize_after_dispatch, barrier_only, x_prepared,
        route_prelude_threads);
}

static void mega_moe_backward_combine_grad_x(
    const torch::Tensor& grad_x_output,
    const torch::Tensor& combine_buffer,
    const int& num_max_tokens_per_rank,
    const int& num_topk,
    const int& num_ranks,
    const int& num_local_experts,
    const std::optional<torch::Tensor>& topk_ids,
    const std::string& combine_order_mode) {
    deep_gemm::sm100_mega_moe_backward_combine_grad_x(
        grad_x_output, combine_buffer, topk_ids,
        num_max_tokens_per_rank, num_topk, num_ranks,
        num_local_experts, combine_order_mode);
}

static void bf16_mega_moe_backward_w2(
    const torch::Tensor& grad_w2_output,
    const torch::Tensor& grad_ye,
    const torch::Tensor& h_weighted,
    const torch::Tensor& padded_expert_counts,
    const int& pool_block_m,
    const std::string& route_weight_mode) {
    DG_HOST_ASSERT(
        route_weight_mode == "pre_down" ||
        route_weight_mode == "post_down");
    deep_gemm::sm100_bf16_mega_moe_wgrad_1sm(
        grad_ye, h_weighted, grad_w2_output,
        padded_expert_counts, pool_block_m);
}

static void bf16_mega_moe_backward_w13(
    const torch::Tensor& grad_w13_output,
    const torch::Tensor& grad_gate_up,
    const torch::Tensor& x_pool,
    const torch::Tensor& padded_expert_counts,
    const int& pool_block_m) {
    deep_gemm::sm100_bf16_mega_moe_wgrad_1sm(
        grad_gate_up, x_pool, grad_w13_output,
        padded_expert_counts, pool_block_m);
}

static MegaMoEBackwardCombineArgs make_backward_combine_args(
    const torch::Tensor& grad_x_output,
    const torch::Tensor& combine_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank,
    const int& num_max_tokens_per_rank,
    const int& num_topk,
    const int num_local_experts,
    const bool reduce,
    const std::optional<torch::Tensor>& topk_ids =
        std::nullopt,
    const std::string& combine_order_mode =
        "fixed_topk") {
    const int num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    DG_HOST_ASSERT(num_ranks >= 1);
    DG_HOST_ASSERT(rank >= 0 and rank < num_ranks);
    DG_HOST_ASSERT(grad_x_output.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(grad_x_output.is_contiguous());
    DG_HOST_ASSERT(combine_buffer.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(combine_buffer.is_contiguous());
    DG_HOST_ASSERT(combine_buffer.dim() == 2);
    DG_HOST_ASSERT(
        combine_buffer.size(0) >=
        static_cast<int64_t>(num_topk) *
            num_max_tokens_per_rank);
    DG_HOST_ASSERT(
        combine_buffer.size(1) == grad_x_output.size(1));
    DG_HOST_ASSERT(grad_x_output.dim() == 2);
    DG_HOST_ASSERT(num_max_tokens_per_rank >= grad_x_output.size(0));
    DG_HOST_ASSERT(
        combine_order_mode == "fixed_topk" ||
        combine_order_mode == "deepep" ||
        combine_order_mode == "deepep_v1");
    if (combine_order_mode != "fixed_topk") {
        DG_HOST_ASSERT(topk_ids.has_value());
        DG_HOST_ASSERT(topk_ids->is_cuda());
        DG_HOST_ASSERT(
            topk_ids->scalar_type() == torch::kInt64);
        DG_HOST_ASSERT(topk_ids->is_contiguous());
        DG_HOST_ASSERT(topk_ids->dim() == 2);
        DG_HOST_ASSERT(
            topk_ids->size(0) >=
            grad_x_output.size(0));
        DG_HOST_ASSERT(
            topk_ids->size(1) == num_topk);
        DG_HOST_ASSERT(num_topk <= 32);
    }

    MegaMoEBackwardCombineArgs combine;
    combine.enabled = true;
    combine.num_ranks = num_ranks;
    combine.sym_buffer =
        layout::SymBuffer<>(sym_buffer_ptrs, rank);
    combine.workspace = layout::Workspace(
        reinterpret_cast<void*>(sym_buffer_ptrs[rank]),
        num_ranks, num_local_experts * num_ranks,
        num_max_tokens_per_rank, num_topk,
        layout::kMinCandidateBlockM);
    combine.grad_x_output =
        reinterpret_cast<cutlass::bfloat16_t*>(
            grad_x_output.data_ptr<at::BFloat16>());
    combine.combine_buffer =
        reinterpret_cast<cutlass::bfloat16_t*>(
            combine_buffer.data_ptr<at::BFloat16>());
    combine.topk_ids =
        topk_ids.has_value()
        ? topk_ids->data_ptr<int64_t>()
        : nullptr;
    combine.num_tokens =
        static_cast<uint32_t>(grad_x_output.size(0));
    combine.num_max_tokens =
        static_cast<uint32_t>(num_max_tokens_per_rank);
    combine.num_topk = static_cast<uint32_t>(num_topk);
    combine.hidden = static_cast<uint32_t>(grad_x_output.size(1));
    combine.reduce = reduce;
    combine.order_mode = combine_order_mode;
    return combine;
}

static void bf16_mega_moe_backward_w2_combine(
    const torch::Tensor& grad_w2_output,
    const torch::Tensor& grad_ye,
    const torch::Tensor& h_weighted,
    const torch::Tensor& padded_expert_counts,
    const int& pool_block_m,
    const torch::Tensor& grad_x_output,
    const torch::Tensor& combine_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank,
    const int& num_max_tokens_per_rank,
    const int& num_topk,
    const std::string& route_weight_mode) {
    DG_HOST_ASSERT(
        route_weight_mode == "pre_down" ||
        route_weight_mode == "post_down");
    const auto combine = make_backward_combine_args(
        grad_x_output, combine_buffer, sym_buffer_ptrs, rank,
        num_max_tokens_per_rank, num_topk,
        static_cast<int>(padded_expert_counts.numel()), false);
    deep_gemm::sm100_bf16_mega_moe_wgrad_1sm(
        grad_ye, h_weighted, grad_w2_output,
        padded_expert_counts, pool_block_m, combine);
}

static void bf16_mega_moe_backward_w13_combine(
    const torch::Tensor& grad_w13_output,
    const torch::Tensor& grad_gate_up,
    const torch::Tensor& x_pool,
    const torch::Tensor& padded_expert_counts,
    const int& pool_block_m,
    const torch::Tensor& grad_x_output,
    const torch::Tensor& combine_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank,
    const int& num_max_tokens_per_rank,
    const int& num_topk,
    const std::optional<torch::Tensor>& topk_ids,
    const std::string& combine_order_mode) {
    const auto combine = make_backward_combine_args(
        grad_x_output, combine_buffer, sym_buffer_ptrs, rank,
        num_max_tokens_per_rank, num_topk,
        static_cast<int>(padded_expert_counts.numel()), true,
        topk_ids, combine_order_mode);
    deep_gemm::sm100_bf16_mega_moe_wgrad_1sm(
        grad_gate_up, x_pool, grad_w13_output,
        padded_expert_counts, pool_block_m, combine);
}

static void register_apis(pybind11::module_& m) {
    m.def(
        "fp8_fp4_mega_moe_side_lora_backward",
        &deep_gemm::sm100_fp8_fp4_mega_moe_side_lora_backward);
    m.def(
        "bf16_mega_moe_side_lora_backward",
        &deep_gemm::sm100_bf16_mega_moe_side_lora_backward,
        py::arg("gate_up_output"), py::arg("grad_h_output"),
        py::arg("grad_gate_up_output"), py::arg("h_act_output"),
        py::arg("h_weighted_output"), py::arg("x_pool_output"),
        py::arg("grad_x_pool_output"), py::arg("grad_route_output"),
        py::arg("grad_ye"), py::arg("grad_y_unweighted_output"),
        py::arg("route_weights"), py::arg("w2_weights"),
        py::arg("w13_weights"), py::arg("expert_counts"),
        py::arg("grid_sync_counter"), py::arg("activation_limit"),
        py::arg("activation"), py::arg("fast_math"),
        py::arg("route_weight_mode"), py::arg("combine_order_mode"),
        py::arg("down_unweighted_output"), py::arg("block_m"),
        py::arg("direct_remote_grad_x"), py::arg("write_grad_x_pool"),
        py::arg("clear_wgrad_padding"), py::arg("backward_grad_y"),
        py::arg("backward_x"), py::arg("backward_topk_weights"),
        py::arg("backward_grad_route"), py::arg("token_src_metadata"),
        py::arg("backward_sym_buffer_ptrs"), py::arg("backward_rank"),
        py::arg("num_max_tokens_per_rank"), py::arg("num_topk"),
        py::arg("memory_mode"),
        py::arg("side_lora_a1"), py::arg("side_lora_b1"),
        py::arg("side_lora_a3"), py::arg("side_lora_b3"),
        py::arg("side_lora_a2"), py::arg("side_lora_b2"),
        py::arg("side_lora_q13"), py::arg("side_lora_q2"),
        py::arg("side_lora_saved_h"), py::arg("side_lora_t13"),
        py::arg("side_lora_t2"), py::arg("grad_side_lora_a1"),
        py::arg("grad_side_lora_b1"), py::arg("grad_side_lora_a3"),
        py::arg("grad_side_lora_b3"), py::arg("grad_side_lora_a2"),
        py::arg("grad_side_lora_b2"), py::arg("expert_psum_rows"),
        py::arg("padded_expert_counts"), py::arg("side_lora_scale"),
        py::arg("kernel_trace") = py::none());
    m.def("fp8_fp4_mega_moe_backward_dgrad_swiglu_v2",
          &fp8_fp4_mega_moe_backward_dgrad_swiglu_v2,
          py::arg("gate_up_output"), py::arg("grad_h_output"),
          py::arg("grad_gate_up_output"), py::arg("h_act_output"),
          py::arg("h_weighted_output"), py::arg("x_pool_output"),
          py::arg("grad_x_pool_output"),
          py::arg("l1_acts"), py::arg("l1_acts_sf"),
          py::arg("l1_weights"), py::arg("grad_ye"),
          py::arg("route_weights"), py::arg("w2_weights"),
          py::arg("w2_dequant_scratch"),
          py::arg("w13_weights"), py::arg("w13_dequant_scratch"),
          py::arg("expert_counts"), py::arg("grid_sync_counter"),
          py::arg("activation_limit"),
          py::arg("compute_w13_dgrad"),
          py::arg("block_m"),
          py::arg("direct_remote_grad_x") = false,
          py::arg("write_grad_x_pool") = true,
          py::arg("clear_wgrad_padding") = true,
          py::arg("backward_grad_y") = py::none(),
          py::arg("backward_topk_weights") = py::none(),
          py::arg("backward_grad_route") = py::none(),
          py::arg("token_src_metadata") = py::none(),
          py::arg("backward_sym_buffer_ptrs") =
              std::vector<int64_t>{},
          py::arg("backward_rank") = 0,
          py::arg("num_max_tokens_per_rank") = 0,
          py::arg("num_topk") = 0,
          py::arg("route_weight_mode") = "pre_down",
          py::arg("grad_y_unweighted_output") = py::none(),
          py::arg("down_unweighted_output") = py::none(),
          py::arg("grad_route_output") = py::none());
    m.def("fp8_fp4_mega_moe_backward_dgrad_swiglu",
          &fp8_fp4_mega_moe_backward_dgrad_swiglu,
          py::arg("gate_up_output"), py::arg("grad_h_output"),
          py::arg("grad_gate_up_output"), py::arg("h_act_output"),
          py::arg("h_weighted_output"), py::arg("x_pool_output"),
          py::arg("grad_x_pool_output"),
          py::arg("l1_acts"), py::arg("l1_acts_sf"),
          py::arg("l1_weights"), py::arg("grad_ye"),
          py::arg("route_weights"), py::arg("w2_weights"),
          py::arg("w2_dequant_scratch"),
          py::arg("w13_weights"), py::arg("w13_dequant_scratch"),
          py::arg("expert_counts"), py::arg("grid_sync_counter"),
          py::arg("activation_limit"),
          py::arg("compute_w13_dgrad"),
          py::arg("block_m"),
          py::arg("direct_remote_grad_x") = false,
          py::arg("write_grad_x_pool") = true,
          py::arg("clear_wgrad_padding") = true,
          py::arg("backward_grad_y") = py::none(),
          py::arg("backward_topk_weights") = py::none(),
          py::arg("token_src_metadata") = py::none(),
          py::arg("backward_sym_buffer_ptrs") =
              std::vector<int64_t>{},
          py::arg("backward_rank") = 0,
          py::arg("num_max_tokens_per_rank") = 0,
          py::arg("num_topk") = 0);
    m.def(
        "bf16_mega_moe_backward_dgrad_v2",
        &bf16_mega_moe_backward_dgrad_v2,
        py::arg("gate_up_output"),
        py::arg("grad_h_output"),
        py::arg("grad_gate_up_output"),
        py::arg("h_act_output"),
        py::arg("h_weighted_output"),
        py::arg("x_pool_output"),
        py::arg("grad_x_pool_output"),
        py::arg("grad_route_output"),
        py::arg("grad_ye"),
        py::arg("grad_y_unweighted_output"),
        py::arg("route_weights"),
        py::arg("w2_weights"),
        py::arg("w13_weights"),
        py::arg("expert_counts"),
        py::arg("grid_sync_counter"),
        py::arg("activation_limit"),
        py::arg("activation"),
        py::arg("fast_math"),
        py::arg("route_weight_mode"),
        py::arg("down_unweighted_output"),
        py::arg("block_m"),
        py::arg("direct_remote_grad_x"),
        py::arg("write_grad_x_pool"),
        py::arg("clear_wgrad_padding"),
        py::arg("backward_grad_y"),
        py::arg("backward_x"),
        py::arg("backward_topk_weights"),
        py::arg("backward_grad_route"),
        py::arg("token_src_metadata"),
        py::arg("backward_sym_buffer_ptrs"),
        py::arg("backward_rank"),
        py::arg("num_max_tokens_per_rank"),
        py::arg("num_topk"),
        py::arg("combine_order_mode") = "fixed_topk",
        py::arg("memory_mode") = "legacy",
        py::arg("kernel_trace") = py::none(),
        py::arg("situ_beta") = 4.0f,
        py::arg("situ_linear_beta") = std::numeric_limits<float>::infinity(),
        py::arg("w3_weights") = py::none(),
        py::arg("vectorized_grad_x_store") = true,
        py::arg("wide_grad_x_store") = false);
    m.def(
        "bf16_mega_moe_backward_dgrad",
        &bf16_mega_moe_backward_dgrad,
        py::arg("gate_up_output"),
        py::arg("grad_h_output"),
        py::arg("grad_gate_up_output"),
        py::arg("h_act_output"),
        py::arg("h_weighted_output"),
        py::arg("x_pool_output"),
        py::arg("grad_x_pool_output"),
        py::arg("grad_route_output"),
        py::arg("grad_ye"),
        py::arg("grad_y_unweighted_output"),
        py::arg("route_weights"),
        py::arg("w2_weights"),
        py::arg("w13_weights"),
        py::arg("expert_counts"),
        py::arg("grid_sync_counter"),
        py::arg("activation_limit"),
        py::arg("activation"),
        py::arg("fast_math"),
        py::arg("route_weight_mode"),
        py::arg("down_unweighted_output"),
        py::arg("block_m"),
        py::arg("direct_remote_grad_x"),
        py::arg("write_grad_x_pool"),
        py::arg("clear_wgrad_padding"),
        py::arg("backward_grad_y"),
        py::arg("backward_x"),
        py::arg("backward_topk_weights"),
        py::arg("token_src_metadata"),
        py::arg("backward_sym_buffer_ptrs"),
        py::arg("backward_rank"),
        py::arg("num_max_tokens_per_rank"),
        py::arg("num_topk"),
        py::arg("combine_order_mode") = "fixed_topk",
        py::arg("memory_mode") = "legacy",
        py::arg("kernel_trace") = py::none());
    m.def(
        "bf16_mega_moe_backward_post_down_prelude_v2",
        &bf16_mega_moe_backward_post_down_prelude_v2,
        py::arg("grad_y_unweighted_output"),
        py::arg("grad_y_weighted_output"),
        py::arg("x_pool_output"),
        py::arg("route_weights_output"),
        py::arg("grad_route_output"),
        py::arg("down_unweighted_output"),
        py::arg("expert_counts"),
        py::arg("backward_grad_y"),
        py::arg("backward_x"),
        py::arg("backward_topk_weights"),
        py::arg("backward_grad_route"),
        py::arg("token_src_metadata"),
        py::arg("backward_sym_buffer_ptrs"),
        py::arg("backward_rank"),
        py::arg("num_topk"),
        py::arg("block_m"),
        py::arg("combine_order_mode") = "fixed_topk",
        py::arg("do_reverse_dispatch") = true,
        py::arg("compute_route_dot") = true,
        py::arg("write_weighted") = true,
        py::arg("synchronize_ranks") = true,
        py::arg("synchronize_after_dispatch") = true,
        py::arg("barrier_only") = false,
        py::arg("x_prepared") = false,
        py::arg("route_prelude_threads") = 256);
    m.def(
        "bf16_mega_moe_backward_post_down_prelude",
        &bf16_mega_moe_backward_post_down_prelude,
        py::arg("grad_y_unweighted_output"),
        py::arg("grad_y_weighted_output"),
        py::arg("x_pool_output"),
        py::arg("route_weights_output"),
        py::arg("grad_route_output"),
        py::arg("down_unweighted_output"),
        py::arg("expert_counts"),
        py::arg("backward_grad_y"),
        py::arg("backward_x"),
        py::arg("backward_topk_weights"),
        py::arg("token_src_metadata"),
        py::arg("backward_sym_buffer_ptrs"),
        py::arg("backward_rank"),
        py::arg("num_topk"),
        py::arg("block_m"),
        py::arg("combine_order_mode") = "fixed_topk",
        py::arg("do_reverse_dispatch") = true,
        py::arg("compute_route_dot") = true,
        py::arg("write_weighted") = true,
        py::arg("synchronize_ranks") = true,
        py::arg("synchronize_after_dispatch") = true,
        py::arg("barrier_only") = false,
        py::arg("x_prepared") = false,
        py::arg("route_prelude_threads") = 256);
    m.def(
        "mega_moe_backward_combine_grad_x",
        &mega_moe_backward_combine_grad_x,
        py::arg("grad_x_output"),
        py::arg("combine_buffer"),
        py::arg("num_max_tokens_per_rank"),
        py::arg("num_topk"),
        py::arg("num_ranks"),
        py::arg("num_local_experts"),
        py::arg("topk_ids") = py::none(),
        py::arg("combine_order_mode") = "fixed_topk");
    m.def("bf16_mega_moe_backward_w2",
          &bf16_mega_moe_backward_w2,
          py::arg("grad_w2_output"), py::arg("grad_ye"),
          py::arg("h_weighted"),
          py::arg("padded_expert_counts"),
          py::arg("pool_block_m"),
          py::arg("route_weight_mode") = "pre_down");
    m.def("bf16_mega_moe_backward_w13",
          &bf16_mega_moe_backward_w13,
          py::arg("grad_w13_output"),
          py::arg("grad_gate_up"), py::arg("x_pool"),
          py::arg("padded_expert_counts"),
          py::arg("pool_block_m"));
    m.def("bf16_mega_moe_backward_w2_combine",
          &bf16_mega_moe_backward_w2_combine,
          py::arg("grad_w2_output"),
          py::arg("grad_ye"), py::arg("h_weighted"),
          py::arg("padded_expert_counts"),
          py::arg("pool_block_m"),
          py::arg("grad_x_output"),
          py::arg("combine_buffer"),
          py::arg("sym_buffer_ptrs"), py::arg("rank"),
          py::arg("num_max_tokens_per_rank"),
          py::arg("num_topk"),
          py::arg("route_weight_mode") = "pre_down");
    m.def("bf16_mega_moe_backward_w13_combine",
          &bf16_mega_moe_backward_w13_combine,
          py::arg("grad_w13_output"),
          py::arg("grad_gate_up"), py::arg("x_pool"),
          py::arg("padded_expert_counts"),
          py::arg("pool_block_m"),
          py::arg("grad_x_output"),
          py::arg("combine_buffer"),
          py::arg("sym_buffer_ptrs"), py::arg("rank"),
          py::arg("num_max_tokens_per_rank"),
          py::arg("num_topk"),
          py::arg("topk_ids") = py::none(),
          py::arg("combine_order_mode") = "fixed_topk");
}

}  // namespace deep_gemm::mega_backward
