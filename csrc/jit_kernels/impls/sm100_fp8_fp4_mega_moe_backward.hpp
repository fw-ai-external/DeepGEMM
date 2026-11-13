#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>

#include <torch/python.h>

#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "runtime_utils.hpp"
#include "../../runtime/runtime.hpp"
#include <format>

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

namespace deep_gemm {

static std::string get_backward_route_weight_mode_name(
    const std::string& route_weight_mode) {
    if (route_weight_mode == "pre_down")
        return "RouteWeightMode::PreDown";
    if (route_weight_mode == "post_down")
        return "RouteWeightMode::PostDown";
    DG_HOST_UNREACHABLE("Unsupported route weight mode");
}

static std::string get_backward_combine_order_mode_name(
    const std::string& combine_order_mode) {
    if (combine_order_mode == "fixed_topk")
        return "CombineOrderMode::FixedTopK";
    if (combine_order_mode == "deepep")
        return "CombineOrderMode::DeepEP";
    if (combine_order_mode == "deepep_v1")
        return "CombineOrderMode::DeepEPV1";
    DG_HOST_UNREACHABLE("Unsupported combine order mode");
}

class SM100MegaMoEBackwardCombineRuntime final {
public:
    struct Args {
        int num_ranks;
        int num_local_experts;
        std::string combine_order_mode = "fixed_topk";
        cutlass::bfloat16_t* grad_x_output;
        const cutlass::bfloat16_t* combine_buffer;
        const int64_t* topk_ids;
        uint32_t num_tokens;
        uint32_t num_max_tokens;
        uint32_t num_topk;
        uint32_t hidden;
        deep_jit::cuda::LaunchOptions launch_args;
    };

    static std::string generate(const Args& args) {
        return std::format(R"(
#include <deep_gemm/impls/sm100_mega_moe_backward_combine.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &sm100_mega_moe_backward_combine_grad_x<
            {}, {}, {}>);
}};
)",
            args.num_ranks, args.num_local_experts,
            get_backward_combine_order_mode_name(
                args.combine_order_mode));
    }

    static void launch(const auto& kernel, const Args& args) {
        jit->launch(
            kernel, args.launch_args, args.grad_x_output,
            args.combine_buffer, args.topk_ids,
            args.num_tokens, args.num_max_tokens,
            args.num_topk, args.hidden);
    }
};

class SM100BF16MegaMoEBackwardPostDownPreludeRuntime final {
public:
    struct Args {
        int hidden;
        int num_experts;
        int block_m;
        int num_sms;
        int num_ranks;
        std::string combine_order_mode = "fixed_topk";
        bool do_reverse_dispatch = true;
        bool compute_route_dot = true;
        bool write_weighted = true;
        bool synchronize_ranks = true;
        bool synchronize_after_dispatch = true;
        bool barrier_only = false;
        bool x_prepared = false;
        int route_prelude_threads = 256;
        const int* expert_counts;
        layout::Workspace backward_workspace;
        layout::SymBuffer<> backward_sym_buffer;
        const cutlass::bfloat16_t* backward_grad_y;
        const cutlass::bfloat16_t* backward_x;
        const float* backward_topk_weights;
        float* backward_grad_route;
        const layout::TokenSrcMetadata* token_src_metadata;
        uint32_t num_topk;
        uint32_t num_pool_rows;
        cutlass::bfloat16_t* grad_y_unweighted_output;
        cutlass::bfloat16_t* grad_y_weighted_output;
        cutlass::bfloat16_t* x_pool_output;
        float* route_weights_output;
        const cutlass::bfloat16_t* down_unweighted;
        float* grad_route_output;
        deep_jit::cuda::LaunchOptions launch_args;
    };

    static std::string generate(const Args& args) {
        return std::format(R"(
#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe_backward.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &sm100_bf16_mega_moe_backward_post_down_prelude<
            {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
        >);
}};
)",
            args.hidden, args.num_experts, args.block_m,
            args.num_sms, args.num_ranks,
            get_backward_combine_order_mode_name(
                args.combine_order_mode),
            args.do_reverse_dispatch ? "true" : "false",
            args.compute_route_dot ? "true" : "false",
            args.write_weighted ? "true" : "false",
            "false",
            args.synchronize_ranks ? "true" : "false",
            args.synchronize_after_dispatch ? "true" : "false",
            args.barrier_only ? "true" : "false",
            args.x_prepared ? "true" : "false",
            args.route_prelude_threads);
    }

    static void launch(const auto& kernel, const Args& args) {
        jit->launch(
            kernel, args.launch_args,
            args.expert_counts,
            args.backward_workspace,
            args.backward_sym_buffer,
            args.backward_grad_y,
            args.backward_x,
            args.backward_topk_weights,
            args.backward_grad_route,
            args.token_src_metadata,
            args.num_topk,
            args.num_pool_rows,
            args.grad_y_unweighted_output,
            args.grad_y_weighted_output,
            args.x_pool_output,
            args.route_weights_output,
            args.down_unweighted,
            args.grad_route_output);
    }
};

class SM100FP8FP4MegaMoEBackwardWaveRuntime final {
public:
    struct Args {
        int hidden;
        int intermediate_hidden;
        int num_experts;
        int num_pool_rows;
        int num_acts_rows;
        int num_sf_pool_rows;
        int block_m;
        int block_n;
        int block_k;
        int sf_block_m;
        int sf_block_n;
        int num_stages;
        int num_sms;
        int num_ranks;
        bool bf16_mode = false;
        std::string activation = "swiglu";
        bool fast_math = false;
        std::string route_weight_mode = "pre_down";
        std::string combine_order_mode = "fixed_topk";

        const int* expert_counts;
        layout::SymBuffer<> backward_sym_buffer;
        layout::Workspace backward_workspace;
        const cutlass::bfloat16_t* backward_grad_y;
        const cutlass::bfloat16_t* backward_x;
        const float* backward_topk_weights;
        float* backward_grad_route;
        const layout::TokenSrcMetadata* token_src_metadata;
        uint32_t num_topk;
        uint32_t acts_sf_stride;
        CUtensorMap tensor_map_acts;
        CUtensorMap tensor_map_acts_sf;
        CUtensorMap tensor_map_weights;
        CUtensorMap tensor_map_weights_sf;
        CUtensorMap tensor_map_output;
        CUtensorMap tensor_map_grad_ye;
        CUtensorMap tensor_map_w2_dequant;
        CUtensorMap tensor_map_w2_weights;
        CUtensorMap tensor_map_w2_scales;
        CUtensorMap tensor_map_w13_dequant;
        CUtensorMap tensor_map_w13_weights;
        CUtensorMap tensor_map_w13_scales;
        CUtensorMap tensor_map_grad_gate_up;
        const cutlass::float_e4m3_t* acts_ptr;
        const uint32_t* acts_sf_ptr;
        const int8_t* w2_weights;
        const float* w2_scales;
        cutlass::bfloat16_t* w2_dequant_scratch;
        const int8_t* w13_weights;
        const float* w13_scales;
        cutlass::bfloat16_t* w13_dequant_scratch;
        const cutlass::bfloat16_t* gate_up_output;
        cutlass::bfloat16_t* grad_ye_output;
        cutlass::bfloat16_t* grad_y_unweighted_output;
        cutlass::bfloat16_t* route_weights;
        float* route_weights_fp32;
        cutlass::bfloat16_t* grad_h_output;
        cutlass::bfloat16_t* grad_gate_up_output;
        cutlass::bfloat16_t* h_act_output;
        cutlass::bfloat16_t* h_weighted_output;
        cutlass::bfloat16_t* x_pool_output;
        cutlass::bfloat16_t* grad_x_pool_output;
        const cutlass::bfloat16_t* down_unweighted_output;
        float* grad_route_output;
        uint32_t* grid_sync_counter;
        uint32_t launch_epoch;
        float activation_limit;
        bool compute_w13_dgrad;
        bool direct_remote_grad_x;
        bool write_grad_x_pool;
        bool clear_wgrad_padding;
        bool compute_route_grad = false;
        bool trace_kernel = false;
        bool vectorized_grad_x_store = false;
        bool wide_grad_x_store = false;
        uint64_t* kernel_trace = nullptr;
        bool inputs_prepared = false;
        bool dispatch_inputs_prepared = false;
        deep_jit::cuda::LaunchOptions launch_args;
    };

    static std::string generate(const Args& args) {
        return std::format(R"(
#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe_backward.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &sm100_fp8_fp4_mega_moe_backward_wave_impl<
            {}, {},
            {},
            {}, {}, {},
            {}, {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {}
        >);
}};
)",
            args.hidden, args.intermediate_hidden,
            args.num_experts,
            args.block_m, args.block_n, args.block_k,
            args.sf_block_m, args.sf_block_n,
            args.num_stages,
            args.num_sms,
            args.num_ranks,
            args.compute_w13_dgrad ? "true" : "false",
            args.bf16_mode ? "true" : "false",
            args.activation == "geglu"
                ? "ActivationType::GeGLU"
                : "ActivationType::SwiGLU",
            args.fast_math ? "true" : "false",
            get_backward_route_weight_mode_name(
                args.route_weight_mode),
            get_backward_combine_order_mode_name(
                args.combine_order_mode),
            args.inputs_prepared ? "true" : "false",
            args.dispatch_inputs_prepared ? "true" : "false",
            args.direct_remote_grad_x ? "true" : "false",
            args.write_grad_x_pool ? "true" : "false",
            args.clear_wgrad_padding ? "true" : "false",
            args.compute_route_grad ? "true" : "false",
            args.trace_kernel ? "true" : "false",
            args.vectorized_grad_x_store ? "true" : "false",
            args.wide_grad_x_store ? "true" : "false");
    }

    static void launch(const auto& kernel, const Args& args) {
        jit->launch(
            kernel, args.launch_args,
            args.expert_counts,
            args.backward_sym_buffer,
            args.backward_workspace,
            args.backward_grad_y,
            args.backward_x,
            args.backward_topk_weights,
            args.backward_grad_route,
            args.token_src_metadata,
            args.num_topk,
            args.num_pool_rows,
            args.num_acts_rows,
            args.acts_sf_stride,
            args.tensor_map_acts,
            args.tensor_map_acts_sf,
            args.tensor_map_weights,
            args.tensor_map_weights_sf,
            args.tensor_map_output,
            args.tensor_map_grad_ye,
            args.tensor_map_w2_dequant,
            args.tensor_map_w2_weights,
            args.tensor_map_w2_scales,
            args.tensor_map_w13_dequant,
            args.tensor_map_w13_weights,
            args.tensor_map_w13_scales,
            args.tensor_map_grad_gate_up,
            args.acts_ptr,
            args.acts_sf_ptr,
            args.w2_weights,
            args.w2_scales,
            args.w2_dequant_scratch,
            args.w13_weights,
            args.w13_scales,
            args.w13_dequant_scratch,
            args.gate_up_output,
            args.grad_ye_output,
            args.grad_y_unweighted_output,
            args.route_weights,
            args.route_weights_fp32,
            args.grad_h_output,
            args.grad_gate_up_output,
            args.h_act_output,
            args.h_weighted_output,
            args.x_pool_output,
            args.grad_x_pool_output,
            args.down_unweighted_output,
            args.grad_route_output,
            args.grid_sync_counter,
            args.launch_epoch,
            args.activation_limit,
            args.kernel_trace);
    }
};

static void sm100_fp8_fp4_mega_moe_backward_dgrad_swiglu(
    const torch::Tensor& gate_up_output,
    const torch::Tensor& grad_h_output,
    const torch::Tensor& grad_gate_up_output,
    const torch::Tensor& h_act_output,
    const torch::Tensor& h_weighted_output,
    const torch::Tensor& x_pool_output,
    const torch::Tensor& grad_x_pool_output,
    const torch::Tensor& acts,
    const torch::Tensor& acts_sf,
    const torch::Tensor& l1_weights,
    const torch::Tensor& l1_weights_sf,
    const torch::Tensor& grad_ye,
    const torch::Tensor& route_weights,
    const torch::Tensor& w2_weights,
    const torch::Tensor& w2_scales,
    const torch::Tensor& w2_dequant_scratch,
    const torch::Tensor& w13_weights,
    const torch::Tensor& w13_scales,
    const torch::Tensor& w13_dequant_scratch,
    const torch::Tensor& expert_counts,
    const torch::Tensor& grid_sync_counter,
    const float& activation_limit,
    const bool& compute_w13_dgrad,
    const bool& direct_remote_grad_x,
    const bool& write_grad_x_pool,
    const bool& clear_wgrad_padding,
    const int& block_m,
    const std::vector<int64_t>& backward_sym_buffer_ptrs = {},
    const int& backward_rank = 0,
    const int& num_max_tokens_per_rank = 0,
    const int& num_topk = 0,
    const std::optional<torch::Tensor>& backward_grad_y = std::nullopt,
    const std::optional<torch::Tensor>& backward_topk_weights = std::nullopt,
    const std::optional<torch::Tensor>& backward_grad_route = std::nullopt,
    const std::optional<torch::Tensor>& token_src_metadata = std::nullopt,
    const std::string& route_weight_mode = "pre_down",
    const std::optional<torch::Tensor>& grad_y_unweighted_output =
        std::nullopt,
    const std::optional<torch::Tensor>& down_unweighted_output =
        std::nullopt,
    const std::optional<torch::Tensor>& grad_route_output =
        std::nullopt) {
    constexpr int block_n = 128;
    constexpr int block_k = 128;
    constexpr int dgrad_block_k = 64;
    constexpr int store_block_m = 16;
    constexpr int gran_k = 32;
    constexpr int smem_capacity = 232448;
    constexpr int num_epilogue_stages = 2;
    constexpr int num_tma_store_stages = 2;

    const auto [num_experts, intermediate_hidden_2, hidden] =
        check_grouped_ab_fp8_fp4(
            l1_weights, cute::UMMA::Major::K,
            jit->device.get_arch_major());
    const int intermediate_hidden = intermediate_hidden_2 / 2;
    const int num_pool_rows = static_cast<int>(grad_ye.size(0));
    const int num_acts_rows = static_cast<int>(acts.size(0));
    const int num_sf_pool_rows = static_cast<int>(acts_sf.size(0));
    const int sf_block_m = align(block_m, 128);
    const int sf_block_n = block_n;
    const int load_block_m = block_m / 2;
    const int load_block_n = block_n;
    const int num_ranks = backward_sym_buffer_ptrs.empty()
        ? 1
        : static_cast<int>(backward_sym_buffer_ptrs.size());
    const int num_dispatch_warps = num_ranks > 1 ? 4 : 0;

    DG_HOST_ASSERT(jit->device.get_arch_major() == 10);
    DG_HOST_ASSERT(num_ranks >= 1);
    DG_HOST_ASSERT(
        route_weight_mode == "pre_down" ||
        route_weight_mode == "post_down");
    DG_HOST_ASSERT(num_ranks == 1 ||
                   (backward_rank >= 0 && backward_rank < num_ranks));
    DG_HOST_ASSERT(block_m % 16 == 0);
    DG_HOST_ASSERT(hidden % block_k == 0);
    DG_HOST_ASSERT(hidden % dgrad_block_k == 0);
    DG_HOST_ASSERT(intermediate_hidden_2 % block_n == 0);
    DG_HOST_ASSERT(acts.dim() == 2 and acts.size(1) == hidden);
    DG_HOST_ASSERT(num_acts_rows > 0 && num_acts_rows <= num_pool_rows);
    DG_HOST_ASSERT(num_sf_pool_rows > 0);
    DG_HOST_ASSERT(grad_ye.dim() == 2);
    DG_HOST_ASSERT(grad_ye.size(0) == num_pool_rows);
    DG_HOST_ASSERT(grad_ye.size(1) == hidden);
    DG_HOST_ASSERT(grad_ye.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(grad_ye.is_contiguous());
    DG_HOST_ASSERT(
        gate_up_output.sizes() ==
        torch::IntArrayRef({num_pool_rows, intermediate_hidden_2}));
    DG_HOST_ASSERT(grad_gate_up_output.sizes() ==
                   gate_up_output.sizes());
    DG_HOST_ASSERT(grad_h_output.sizes() ==
                   torch::IntArrayRef({num_pool_rows, intermediate_hidden}));
    DG_HOST_ASSERT(h_act_output.sizes() == grad_h_output.sizes());
    DG_HOST_ASSERT(h_weighted_output.sizes() == grad_h_output.sizes());
    DG_HOST_ASSERT(
        x_pool_output.sizes() ==
        torch::IntArrayRef({num_pool_rows, hidden}));
    DG_HOST_ASSERT(grad_x_pool_output.sizes() ==
                   torch::IntArrayRef({num_pool_rows, hidden}));
    DG_HOST_ASSERT(gate_up_output.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(grad_h_output.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(grad_gate_up_output.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(h_act_output.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(
        h_weighted_output.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(x_pool_output.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(grad_x_pool_output.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(gate_up_output.is_contiguous());
    DG_HOST_ASSERT(grad_h_output.is_contiguous());
    DG_HOST_ASSERT(grad_gate_up_output.is_contiguous());
    DG_HOST_ASSERT(h_act_output.is_contiguous());
    DG_HOST_ASSERT(h_weighted_output.is_contiguous());
    DG_HOST_ASSERT(x_pool_output.is_contiguous());
    DG_HOST_ASSERT(grad_x_pool_output.is_contiguous());
    DG_HOST_ASSERT(
        route_weights.scalar_type() == torch::kFloat ||
        route_weights.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(route_weights.numel() == num_pool_rows);
    DG_HOST_ASSERT(route_weights.is_contiguous());
    const auto check_bf16_hidden_pool =
        [&](const torch::Tensor& tensor) {
            DG_HOST_ASSERT(
                tensor.scalar_type() == torch::kBFloat16);
            DG_HOST_ASSERT(tensor.is_contiguous());
            DG_HOST_ASSERT(
                tensor.sizes() ==
                torch::IntArrayRef(
                    {num_pool_rows, hidden}));
        };
    if (grad_y_unweighted_output.has_value())
        check_bf16_hidden_pool(
            *grad_y_unweighted_output);
    if (down_unweighted_output.has_value()) {
        DG_HOST_ASSERT(
            down_unweighted_output->scalar_type() ==
            torch::kBFloat16);
        DG_HOST_ASSERT(down_unweighted_output->is_contiguous());
        DG_HOST_ASSERT(down_unweighted_output->dim() == 2);
        DG_HOST_ASSERT(
            down_unweighted_output->size(1) == hidden);
        DG_HOST_ASSERT(
            down_unweighted_output->size(0) > 0 &&
            down_unweighted_output->size(0) <= num_pool_rows);
    }
    if (grad_route_output.has_value()) {
        DG_HOST_ASSERT(
            grad_route_output->scalar_type() ==
            torch::kFloat);
        DG_HOST_ASSERT(grad_route_output->is_contiguous());
        DG_HOST_ASSERT(
            grad_route_output->numel() ==
            num_pool_rows);
    }
    if (route_weight_mode == "post_down") {
        DG_HOST_ASSERT(
            route_weights.scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(
            grad_y_unweighted_output.has_value());
        DG_HOST_ASSERT(
            down_unweighted_output.has_value());
        DG_HOST_ASSERT(grad_route_output.has_value());
    }
    DG_HOST_ASSERT(
        w2_weights.scalar_type() ==
        torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(w2_weights.dim() == 3);
    DG_HOST_ASSERT(w2_weights.size(0) == num_experts);
    DG_HOST_ASSERT(w2_weights.size(1) == hidden);
    DG_HOST_ASSERT(w2_weights.size(2) == intermediate_hidden / 2);
    DG_HOST_ASSERT(w2_weights.is_contiguous());
    DG_HOST_ASSERT(w2_scales.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(w2_scales.dim() == 3);
    DG_HOST_ASSERT(w2_scales.size(0) == num_experts);
    DG_HOST_ASSERT(w2_scales.size(1) == hidden);
    DG_HOST_ASSERT(w2_scales.size(2) == intermediate_hidden / gran_k);
    DG_HOST_ASSERT(w2_scales.is_contiguous());
    DG_HOST_ASSERT(
        w2_dequant_scratch.sizes() ==
        torch::IntArrayRef(
            {num_experts, hidden, intermediate_hidden}));
    DG_HOST_ASSERT(
        w2_dequant_scratch.scalar_type() ==
        torch::kBFloat16);
    DG_HOST_ASSERT(w2_dequant_scratch.is_contiguous());
    DG_HOST_ASSERT(
        w13_weights.scalar_type() ==
        torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(w13_weights.dim() == 3);
    DG_HOST_ASSERT(w13_weights.size(0) == 2 * num_experts);
    DG_HOST_ASSERT(w13_weights.size(1) == intermediate_hidden);
    DG_HOST_ASSERT(w13_weights.size(2) == hidden / 2);
    DG_HOST_ASSERT(w13_weights.is_contiguous());
    DG_HOST_ASSERT(w13_scales.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(w13_scales.dim() == 3);
    DG_HOST_ASSERT(w13_scales.size(0) == 2 * num_experts);
    DG_HOST_ASSERT(w13_scales.size(1) == intermediate_hidden);
    DG_HOST_ASSERT(w13_scales.size(2) == hidden / gran_k);
    DG_HOST_ASSERT(w13_scales.is_contiguous());
    DG_HOST_ASSERT(
        w13_dequant_scratch.sizes() ==
        torch::IntArrayRef(
            {num_experts, intermediate_hidden_2, hidden}));
    DG_HOST_ASSERT(
        w13_dequant_scratch.scalar_type() ==
        torch::kBFloat16);
    DG_HOST_ASSERT(w13_dequant_scratch.is_contiguous());
    DG_HOST_ASSERT(expert_counts.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(expert_counts.numel() == num_experts);
    DG_HOST_ASSERT(expert_counts.is_contiguous());
    DG_HOST_ASSERT(grid_sync_counter.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(
        grid_sync_counter.numel() >=
        num_experts *
            ((hidden / dgrad_block_k) *
                 (intermediate_hidden / block_n) +
             (intermediate_hidden_2 / dgrad_block_k) *
                 (hidden / block_n)) +
            2);
    DG_HOST_ASSERT(grid_sync_counter.is_contiguous());
    if (compute_w13_dgrad)
        DG_HOST_ASSERT(write_grad_x_pool || direct_remote_grad_x);
    else
        DG_HOST_ASSERT(!direct_remote_grad_x);
    if (direct_remote_grad_x) {
        DG_HOST_ASSERT(compute_w13_dgrad);
        DG_HOST_ASSERT(num_ranks > 1);
    }
    if (!backward_sym_buffer_ptrs.empty()) {
        DG_HOST_ASSERT(backward_grad_y.has_value());
        DG_HOST_ASSERT(backward_topk_weights.has_value());
        DG_HOST_ASSERT(token_src_metadata.has_value());
        DG_HOST_ASSERT(num_max_tokens_per_rank > 0);
        DG_HOST_ASSERT(num_topk > 0);
        DG_HOST_ASSERT(
            backward_grad_y->scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(backward_grad_y->dim() == 2);
        DG_HOST_ASSERT(backward_grad_y->size(0) >= num_max_tokens_per_rank);
        DG_HOST_ASSERT(backward_grad_y->size(1) == hidden);
        DG_HOST_ASSERT(backward_grad_y->is_contiguous());
        DG_HOST_ASSERT(
            backward_topk_weights->scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(backward_topk_weights->dim() == 2);
        DG_HOST_ASSERT(backward_topk_weights->size(0) >=
                       num_max_tokens_per_rank);
        DG_HOST_ASSERT(backward_topk_weights->size(1) == num_topk);
        DG_HOST_ASSERT(backward_topk_weights->is_contiguous());
        if (grad_route_output.has_value()) {
            DG_HOST_ASSERT(backward_grad_route.has_value());
            DG_HOST_ASSERT(
                backward_grad_route->scalar_type() == torch::kFloat);
            DG_HOST_ASSERT(backward_grad_route->dim() == 2);
            DG_HOST_ASSERT(backward_grad_route->size(0) >=
                           num_max_tokens_per_rank);
            DG_HOST_ASSERT(backward_grad_route->size(1) == num_topk);
            DG_HOST_ASSERT(backward_grad_route->is_contiguous());
        }
        DG_HOST_ASSERT(
            token_src_metadata->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(token_src_metadata->dim() == 2);
        // Metadata is an immutable active-layout snapshot. Bucket-only output
        // rows are never visited because expert_counts guards every access.
        DG_HOST_ASSERT(token_src_metadata->size(0) >= num_acts_rows);
        DG_HOST_ASSERT(token_src_metadata->size(1) == 3);
        DG_HOST_ASSERT(token_src_metadata->is_contiguous());
    }

    check_sf_layout(
        l1_weights_sf, intermediate_hidden_2, hidden,
        1, gran_k, num_experts, true, false, torch::kInt);

    const int smem_cd =
        store_block_m * block_n *
        static_cast<int>(sizeof(cutlass::bfloat16_t)) *
        num_tma_store_stages;
    const int smem_per_stage =
        load_block_m * block_k +
        load_block_n * block_k +
        sf_block_m * static_cast<int>(sizeof(uint32_t)) +
        sf_block_n * static_cast<int>(sizeof(uint32_t)) +
        2 * static_cast<int>(sizeof(uint64_t));
    const int smem_fixed =
        num_dispatch_warps * hidden *
            static_cast<int>(sizeof(cutlass::bfloat16_t)) +
        smem_cd +
        2 * num_epilogue_stages *
            static_cast<int>(sizeof(uint64_t)) +
        num_dispatch_warps *
            static_cast<int>(sizeof(uint64_t)) +
        static_cast<int>(sizeof(uint32_t));
    const int num_stages =
        std::min(32, (smem_capacity - smem_fixed) / smem_per_stage);
    const int smem_size =
        align(smem_fixed + num_stages * smem_per_stage, 1024);
    DG_HOST_ASSERT(num_stages >= 2);

    const auto tensor_map_acts = make_tma_2d_desc(
        acts, hidden, num_acts_rows,
        block_k, load_block_m,
        static_cast<int>(acts.stride(-2)), 128);
    const auto tensor_map_acts_sf = make_tma_sf_desc(
        cute::UMMA::Major::MN, acts_sf,
        num_sf_pool_rows, hidden,
        sf_block_m, gran_k, 1, 0);
    const auto tensor_map_l1_weights = make_tma_2d_desc(
        l1_weights, hidden, num_experts * intermediate_hidden_2,
        block_k, load_block_n,
        static_cast<int>(l1_weights.stride(-2)), 128);
    const auto tensor_map_l1_weights_sf = make_tma_sf_desc(
        cute::UMMA::Major::MN, l1_weights_sf,
        intermediate_hidden_2, hidden,
        block_n, gran_k, num_experts, 0);
    const auto tensor_map_gate_up = make_tma_2d_desc(
        gate_up_output, intermediate_hidden_2, num_pool_rows,
        block_n, store_block_m,
        static_cast<int>(gate_up_output.stride(-2)), 128);
    const auto tensor_map_grad_ye = make_tma_2d_desc(
        grad_ye, hidden, num_pool_rows,
        dgrad_block_k, load_block_m,
        static_cast<int>(grad_ye.stride(-2)), 128);
    const auto tensor_map_w2_dequant = make_tma_2d_desc(
        w2_dequant_scratch, intermediate_hidden,
        num_experts * hidden,
        load_block_n, dgrad_block_k,
        intermediate_hidden, 128);
    const auto tensor_map_w2_weights = make_tma_2d_desc(
        w2_weights, intermediate_hidden / 2,
        num_experts * hidden,
        load_block_n / 2, 256,
        intermediate_hidden / 2, 0);
    const auto tensor_map_w2_scales = make_tma_2d_desc(
        w2_scales, intermediate_hidden / gran_k,
        num_experts * hidden,
        load_block_n / gran_k, 256,
        intermediate_hidden / gran_k, 0);
    const auto tensor_map_w13_dequant = make_tma_2d_desc(
        w13_dequant_scratch, hidden,
        num_experts * intermediate_hidden_2,
        load_block_n, dgrad_block_k,
        hidden, 128);
    const auto tensor_map_w13_weights = make_tma_2d_desc(
        w13_weights, hidden / 2,
        num_experts * intermediate_hidden_2,
        load_block_n / 2, 256,
        hidden / 2, 0);
    const auto tensor_map_w13_scales = make_tma_2d_desc(
        w13_scales, hidden / gran_k,
        num_experts * intermediate_hidden_2,
        load_block_n / gran_k, 256,
        hidden / gran_k, 0);
    const auto tensor_map_grad_gate_up = make_tma_2d_desc(
        grad_gate_up_output, intermediate_hidden_2,
        num_pool_rows,
        dgrad_block_k, load_block_m,
        static_cast<int>(grad_gate_up_output.stride(-2)), 128);
    // Each launch gets a unique readiness epoch; no host memset is required.
    const int num_sms = get_mega_moe_num_sms();
    DG_HOST_ASSERT(num_sms % 2 == 0);
    static std::atomic<uint32_t> next_launch_epoch{1};
    uint32_t launch_epoch =
        next_launch_epoch.fetch_add(1, std::memory_order_relaxed);
    if (launch_epoch == 0)
        launch_epoch =
            next_launch_epoch.fetch_add(1, std::memory_order_relaxed);
    layout::SymBuffer<> backward_sym_buffer{};
    void* backward_workspace_base = nullptr;
    if (!backward_sym_buffer_ptrs.empty()) {
        backward_sym_buffer =
            layout::SymBuffer<>(
                backward_sym_buffer_ptrs, backward_rank);
        backward_workspace_base =
            reinterpret_cast<void*>(
                backward_sym_buffer_ptrs[backward_rank]);
    }
    const auto backward_workspace = layout::Workspace(
        backward_workspace_base, num_ranks,
        num_experts * num_ranks,
        std::max(num_max_tokens_per_rank, 1),
        std::max(num_topk, 1),
        layout::kMinCandidateBlockM);
    const SM100FP8FP4MegaMoEBackwardWaveRuntime::Args args = {
        .hidden = hidden,
        .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts,
        .num_pool_rows = num_pool_rows,
        .num_acts_rows = num_acts_rows,
        .num_sf_pool_rows = num_sf_pool_rows,
        .block_m = block_m,
        .block_n = block_n,
        .block_k = block_k,
        .sf_block_m = sf_block_m,
        .sf_block_n = sf_block_n,
        .num_stages = num_stages,
        .num_sms = num_sms,
        .num_ranks = num_ranks,
        .route_weight_mode = route_weight_mode,
        .expert_counts = expert_counts.data_ptr<int>(),
        .backward_sym_buffer = backward_sym_buffer,
        .backward_workspace = backward_workspace,
        .backward_grad_y = num_ranks > 1
            ? reinterpret_cast<const cutlass::bfloat16_t*>(
                  backward_grad_y->data_ptr<at::BFloat16>())
            : nullptr,
        .backward_x = nullptr,
        .backward_topk_weights = num_ranks > 1
            ? backward_topk_weights->data_ptr<float>()
            : nullptr,
        .backward_grad_route =
            backward_grad_route.has_value()
            ? backward_grad_route->data_ptr<float>()
            : nullptr,
        .token_src_metadata = !backward_sym_buffer_ptrs.empty()
            ? reinterpret_cast<const layout::TokenSrcMetadata*>(
                  token_src_metadata->data_ptr<int>())
            : nullptr,
        .num_topk = static_cast<uint32_t>(num_topk),
        .acts_sf_stride =
            static_cast<uint32_t>(acts_sf.stride(1)),
        .tensor_map_acts = tensor_map_acts,
        .tensor_map_acts_sf = tensor_map_acts_sf,
        .tensor_map_weights = tensor_map_l1_weights,
        .tensor_map_weights_sf = tensor_map_l1_weights_sf,
        .tensor_map_output = tensor_map_gate_up,
        .tensor_map_grad_ye = tensor_map_grad_ye,
        .tensor_map_w2_dequant = tensor_map_w2_dequant,
        .tensor_map_w2_weights = tensor_map_w2_weights,
        .tensor_map_w2_scales = tensor_map_w2_scales,
        .tensor_map_w13_dequant = tensor_map_w13_dequant,
        .tensor_map_w13_weights = tensor_map_w13_weights,
        .tensor_map_w13_scales = tensor_map_w13_scales,
        .tensor_map_grad_gate_up = tensor_map_grad_gate_up,
        .acts_ptr =
            reinterpret_cast<const cutlass::float_e4m3_t*>(
                acts.data_ptr()),
        .acts_sf_ptr =
            reinterpret_cast<const uint32_t*>(
                acts_sf.data_ptr<int>()),
        .w2_weights =
            reinterpret_cast<const int8_t*>(
                w2_weights.data_ptr()),
        .w2_scales = w2_scales.data_ptr<float>(),
        .w2_dequant_scratch =
            reinterpret_cast<cutlass::bfloat16_t*>(
                w2_dequant_scratch.data_ptr<at::BFloat16>()),
        .w13_weights =
            reinterpret_cast<const int8_t*>(
                w13_weights.data_ptr()),
        .w13_scales = w13_scales.data_ptr<float>(),
        .w13_dequant_scratch =
            reinterpret_cast<cutlass::bfloat16_t*>(
                w13_dequant_scratch.data_ptr<at::BFloat16>()),
        .gate_up_output =
            reinterpret_cast<const cutlass::bfloat16_t*>(
                gate_up_output.data_ptr<at::BFloat16>()),
        .grad_ye_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                grad_ye.data_ptr<at::BFloat16>()),
        .grad_y_unweighted_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                grad_y_unweighted_output
                    .value_or(grad_ye)
                    .data_ptr<at::BFloat16>()),
        .route_weights =
            route_weights.scalar_type() == torch::kBFloat16
            ? reinterpret_cast<cutlass::bfloat16_t*>(
                  route_weights.data_ptr<at::BFloat16>())
            : nullptr,
        .route_weights_fp32 =
            route_weights.scalar_type() == torch::kFloat
            ? route_weights.data_ptr<float>()
            : nullptr,
        .grad_h_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                grad_h_output.data_ptr<at::BFloat16>()),
        .grad_gate_up_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                grad_gate_up_output.data_ptr<at::BFloat16>()),
        .h_act_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                h_act_output.data_ptr<at::BFloat16>()),
        .h_weighted_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                h_weighted_output.data_ptr<at::BFloat16>()),
        .x_pool_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                x_pool_output.data_ptr<at::BFloat16>()),
        .grad_x_pool_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                grad_x_pool_output.data_ptr<at::BFloat16>()),
        .down_unweighted_output =
            down_unweighted_output.has_value()
            ? reinterpret_cast<
                  const cutlass::bfloat16_t*>(
                  down_unweighted_output
                      ->data_ptr<at::BFloat16>())
            : nullptr,
        .grad_route_output =
            grad_route_output.has_value()
            ? grad_route_output->data_ptr<float>()
            : nullptr,
        .grid_sync_counter =
            reinterpret_cast<uint32_t*>(
                grid_sync_counter.data_ptr<int>()),
        .launch_epoch = launch_epoch,
        .activation_limit = activation_limit,
        .compute_w13_dgrad = compute_w13_dgrad,
        .direct_remote_grad_x = direct_remote_grad_x,
        .write_grad_x_pool = write_grad_x_pool,
        .clear_wgrad_padding = clear_wgrad_padding,
        .compute_route_grad =
            grad_route_output.has_value(),
        .launch_args = make_mega_moe_launch_options(
            num_sms, 1024, smem_size, 2),
    };
    const auto code =
        SM100FP8FP4MegaMoEBackwardWaveRuntime::generate(args);
    const auto runtime = jit->compile(std::format(
        "sm100_fp8_fp4_mega_moe_backward_dgrad_swiglu_{}_r{}",
        route_weight_mode,
        grad_route_output.has_value()), code);
    SM100FP8FP4MegaMoEBackwardWaveRuntime::launch(runtime, args);
}

static void sm100_mega_moe_backward_combine_grad_x(
    const torch::Tensor& grad_x_output,
    const torch::Tensor& combine_buffer,
    const std::optional<torch::Tensor>& topk_ids,
    const int& num_max_tokens,
    const int& num_topk,
    const int& num_ranks,
    const int& num_local_experts,
    const std::string& combine_order_mode) {
    const auto [num_tokens, hidden] =
        get_shape<2>(grad_x_output);
    DG_HOST_ASSERT(jit->device.get_arch_major() == 10);
    DG_HOST_ASSERT(num_ranks >= 1);
    DG_HOST_ASSERT(num_local_experts >= 1);
    DG_HOST_ASSERT(num_max_tokens >= num_tokens);
    DG_HOST_ASSERT(num_topk >= 1 && num_topk <= 32);
    DG_HOST_ASSERT(hidden % 16 == 0);
    DG_HOST_ASSERT(
        combine_order_mode == "fixed_topk" ||
        combine_order_mode == "deepep" ||
        combine_order_mode == "deepep_v1");
    DG_HOST_ASSERT(
        grad_x_output.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(grad_x_output.is_contiguous());
    DG_HOST_ASSERT(
        combine_buffer.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(combine_buffer.is_contiguous());
    DG_HOST_ASSERT(combine_buffer.dim() == 2);
    DG_HOST_ASSERT(
        combine_buffer.size(0) >=
        static_cast<int64_t>(num_topk) * num_max_tokens);
    DG_HOST_ASSERT(combine_buffer.size(1) == hidden);
    if (combine_order_mode != "fixed_topk") {
        DG_HOST_ASSERT(topk_ids.has_value());
        DG_HOST_ASSERT(topk_ids->is_cuda());
        DG_HOST_ASSERT(
            topk_ids->scalar_type() == torch::kInt64);
        DG_HOST_ASSERT(topk_ids->is_contiguous());
        DG_HOST_ASSERT(topk_ids->dim() == 2);
        DG_HOST_ASSERT(topk_ids->size(0) >= num_tokens);
        DG_HOST_ASSERT(topk_ids->size(1) == num_topk);
    }

    const int num_sms = get_mega_moe_num_sms();
    const SM100MegaMoEBackwardCombineRuntime::Args args = {
        .num_ranks = num_ranks,
        .num_local_experts = num_local_experts,
        .combine_order_mode = combine_order_mode,
        .grad_x_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                grad_x_output.data_ptr<at::BFloat16>()),
        .combine_buffer =
            reinterpret_cast<const cutlass::bfloat16_t*>(
                combine_buffer.data_ptr<at::BFloat16>()),
        .topk_ids =
            topk_ids.has_value()
            ? topk_ids->data_ptr<int64_t>()
            : nullptr,
        .num_tokens = static_cast<uint32_t>(num_tokens),
        .num_max_tokens =
            static_cast<uint32_t>(num_max_tokens),
        .num_topk = static_cast<uint32_t>(num_topk),
        .hidden = static_cast<uint32_t>(hidden),
        .launch_args = make_mega_moe_launch_options(num_sms, 256),
    };
    const auto code =
        SM100MegaMoEBackwardCombineRuntime::generate(args);
    const auto runtime = jit->compile(std::format(
        "sm100_mega_moe_backward_combine_grad_x_r{}_e{}_{}",
        num_ranks, num_local_experts, combine_order_mode),
        code);
    SM100MegaMoEBackwardCombineRuntime::launch(runtime, args);
}

static void sm100_bf16_mega_moe_backward_post_down_prelude(
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
    const auto [num_pool_rows, hidden] =
        get_shape<2>(grad_y_unweighted_output);
    const int num_experts =
        static_cast<int>(expert_counts.numel());
    const int num_ranks =
        static_cast<int>(backward_sym_buffer_ptrs.size());
    DG_HOST_ASSERT(jit->device.get_arch_major() == 10);
    DG_HOST_ASSERT(
        combine_order_mode == "fixed_topk" ||
        combine_order_mode == "deepep" ||
        combine_order_mode == "deepep_v1");
    DG_HOST_ASSERT(num_ranks >= 1);
    DG_HOST_ASSERT(
        backward_rank >= 0 && backward_rank < num_ranks);
    DG_HOST_ASSERT(hidden % 256 == 0);
    DG_HOST_ASSERT(block_m % 16 == 0);
    DG_HOST_ASSERT(num_topk > 0);
    DG_HOST_ASSERT(
        route_prelude_threads == 128 ||
        route_prelude_threads == 256);
    if (route_prelude_threads == 128) {
        DG_HOST_ASSERT(hidden == 2048);
        DG_HOST_ASSERT(compute_route_dot);
        DG_HOST_ASSERT(combine_order_mode != "fixed_topk");
    }
    const auto check_bf16_pool = [=](
        const torch::Tensor& tensor) {
        DG_HOST_ASSERT(
            tensor.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(tensor.is_contiguous());
        DG_HOST_ASSERT(
            tensor.sizes() ==
            torch::IntArrayRef({num_pool_rows, hidden}));
    };
    check_bf16_pool(grad_y_unweighted_output);
    check_bf16_pool(x_pool_output);
    if (write_weighted) {
        check_bf16_pool(grad_y_weighted_output);
    }
    if (compute_route_dot) {
        check_bf16_pool(down_unweighted_output);
    }
    DG_HOST_ASSERT(
        backward_grad_y.scalar_type() ==
            torch::kBFloat16 &&
        backward_grad_y.is_contiguous() &&
        backward_grad_y.size(1) == hidden);
    DG_HOST_ASSERT(
        backward_x.scalar_type() == torch::kBFloat16 &&
        backward_x.is_contiguous() &&
        backward_x.size(1) == hidden);
    DG_HOST_ASSERT(
        backward_topk_weights.scalar_type() ==
            torch::kFloat &&
        backward_topk_weights.is_contiguous() &&
        backward_topk_weights.size(1) == num_topk);
    if (backward_grad_route.has_value()) {
        DG_HOST_ASSERT(
            backward_grad_route->scalar_type() == torch::kFloat &&
            backward_grad_route->is_contiguous() &&
            backward_grad_route->size(0) >=
                backward_grad_y.size(0) &&
            backward_grad_route->size(1) == num_topk);
    }
    DG_HOST_ASSERT(
        expert_counts.scalar_type() == torch::kInt &&
        expert_counts.is_contiguous());
    DG_HOST_ASSERT(
        token_src_metadata.scalar_type() == torch::kInt &&
        token_src_metadata.is_contiguous() &&
        token_src_metadata.size(0) >= num_pool_rows &&
        token_src_metadata.size(1) == 3);
    DG_HOST_ASSERT(
        route_weights_output.scalar_type() ==
            torch::kFloat &&
        route_weights_output.is_contiguous() &&
        route_weights_output.numel() == num_pool_rows);
    if (compute_route_dot) {
        DG_HOST_ASSERT(
            grad_route_output.scalar_type() ==
                torch::kFloat &&
            grad_route_output.is_contiguous() &&
            grad_route_output.numel() == num_pool_rows);
    }

    const auto exact_alias = [](
        const torch::Tensor& lhs,
        const torch::Tensor& rhs) {
        return lhs.data_ptr() == rhs.data_ptr() &&
               lhs.numel() == rhs.numel();
    };
    const auto overlaps = [](
        const torch::Tensor& lhs,
        const torch::Tensor& rhs) {
        const auto lhs_begin =
            reinterpret_cast<uintptr_t>(lhs.data_ptr());
        const auto rhs_begin =
            reinterpret_cast<uintptr_t>(rhs.data_ptr());
        const auto lhs_end =
            lhs_begin + lhs.numel() * lhs.element_size();
        const auto rhs_end =
            rhs_begin + rhs.numel() * rhs.element_size();
        return std::max(lhs_begin, rhs_begin) <
               std::min(lhs_end, rhs_end);
    };
    DG_HOST_ASSERT(!overlaps(
        x_pool_output, grad_y_unweighted_output));
    if (compute_route_dot || write_weighted) {
        // The weighted destination may replace the retained down pool only
        // because every row is read and overwritten by the same route group.
        if (compute_route_dot && write_weighted) {
            DG_HOST_ASSERT(
                exact_alias(
                    grad_y_weighted_output,
                    down_unweighted_output) ||
                !overlaps(
                    grad_y_weighted_output,
                    down_unweighted_output));
        }
        if (compute_route_dot) {
            DG_HOST_ASSERT(!overlaps(
                grad_y_unweighted_output,
                down_unweighted_output));
            DG_HOST_ASSERT(!overlaps(
                x_pool_output, down_unweighted_output));
        }
        if (write_weighted) {
            DG_HOST_ASSERT(!overlaps(
                grad_y_unweighted_output,
                grad_y_weighted_output));
            DG_HOST_ASSERT(!overlaps(
                x_pool_output, grad_y_weighted_output));
        }
    }

    const int num_sms = get_mega_moe_num_sms();
    const auto backward_sym_buffer = layout::SymBuffer<>(
        backward_sym_buffer_ptrs, backward_rank);
    const auto backward_workspace = layout::Workspace(
        reinterpret_cast<void*>(
            backward_sym_buffer_ptrs[backward_rank]),
        num_ranks,
        num_experts * num_ranks,
        static_cast<uint32_t>(backward_grad_y.size(0)),
        static_cast<uint32_t>(num_topk),
        0);
    SM100BF16MegaMoEBackwardPostDownPreludeRuntime::Args
        args = {
            .hidden = hidden,
            .num_experts = num_experts,
            .block_m = block_m,
            .num_sms = num_sms,
            .num_ranks = num_ranks,
            .combine_order_mode = combine_order_mode,
            .do_reverse_dispatch = do_reverse_dispatch,
            .compute_route_dot = compute_route_dot,
            .write_weighted = write_weighted,
            .synchronize_ranks = synchronize_ranks,
            .synchronize_after_dispatch =
                synchronize_after_dispatch,
            .barrier_only = barrier_only,
            .x_prepared = x_prepared,
            .route_prelude_threads =
                route_prelude_threads,
            .expert_counts = expert_counts.data_ptr<int>(),
            .backward_workspace = backward_workspace,
            .backward_sym_buffer = backward_sym_buffer,
            .backward_grad_y =
                reinterpret_cast<
                    const cutlass::bfloat16_t*>(
                    backward_grad_y
                        .data_ptr<at::BFloat16>()),
            .backward_x =
                reinterpret_cast<
                    const cutlass::bfloat16_t*>(
                    backward_x.data_ptr<at::BFloat16>()),
            .backward_topk_weights =
                backward_topk_weights.data_ptr<float>(),
            .backward_grad_route =
                backward_grad_route.has_value()
                ? backward_grad_route->data_ptr<float>()
                : nullptr,
            .token_src_metadata =
                reinterpret_cast<
                    const layout::TokenSrcMetadata*>(
                    token_src_metadata.data_ptr<int>()),
            .num_topk =
                static_cast<uint32_t>(num_topk),
            .num_pool_rows =
                static_cast<uint32_t>(num_pool_rows),
            .grad_y_unweighted_output =
                reinterpret_cast<cutlass::bfloat16_t*>(
                    grad_y_unweighted_output
                        .data_ptr<at::BFloat16>()),
            .grad_y_weighted_output =
                reinterpret_cast<cutlass::bfloat16_t*>(
                    grad_y_weighted_output
                        .data_ptr<at::BFloat16>()),
            .x_pool_output =
                reinterpret_cast<cutlass::bfloat16_t*>(
                    x_pool_output
                        .data_ptr<at::BFloat16>()),
            .route_weights_output =
                route_weights_output.data_ptr<float>(),
            .down_unweighted =
                reinterpret_cast<
                    const cutlass::bfloat16_t*>(
                    down_unweighted_output
                        .data_ptr<at::BFloat16>()),
            .grad_route_output =
                grad_route_output.data_ptr<float>(),
            .launch_args =
                make_mega_moe_launch_options(num_sms, 1024, 4096),
        };
    const auto code =
        SM100BF16MegaMoEBackwardPostDownPreludeRuntime::
            generate(args);
    const auto runtime = jit->compile(std::format(
        "sm100_bf16_mega_moe_backward_prelude_r{}_d{}_w{}_s{}_c{}_b{}_x{}_t{}",
        do_reverse_dispatch, compute_route_dot,
        write_weighted, synchronize_ranks,
        synchronize_after_dispatch, barrier_only,
        x_prepared, route_prelude_threads), code);
    SM100BF16MegaMoEBackwardPostDownPreludeRuntime::launch(
        runtime, args);
}

static void sm100_bf16_mega_moe_backward_dgrad(
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
    const std::string& combine_order_mode,
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
    const std::string& memory_mode,
    const std::optional<torch::Tensor>& kernel_trace =
        std::nullopt) {
    constexpr int block_n = 128;
    constexpr int block_k = 128;
    constexpr int dgrad_block_k = 64;
    constexpr int store_block_m = 16;
    constexpr int smem_capacity = 232448;
    constexpr int num_epilogue_stages = 2;
    constexpr int num_tma_store_stages = 2;

    const auto [num_experts, intermediate_hidden_2, hidden] =
        get_shape<3>(w13_weights);
    const auto [num_experts_w2, hidden_w2, intermediate_hidden] =
        get_shape<3>(w2_weights);
    const int num_pool_rows = static_cast<int>(grad_ye.size(0));
    const int num_ranks =
        static_cast<int>(backward_sym_buffer_ptrs.size());
    const int sf_block_m = align(block_m, 128);
    const int sf_block_n = block_n;
    const int load_block_m = block_m / 2;
    const int load_block_n = block_n;
    const int num_dispatch_warps = num_ranks > 1 ? 4 : 0;

    DG_HOST_ASSERT(jit->device.get_arch_major() == 10);
    DG_HOST_ASSERT(activation == "swiglu" || activation == "geglu");
    DG_HOST_ASSERT(
        route_weight_mode == "pre_down" ||
        route_weight_mode == "post_down");
    DG_HOST_ASSERT(
        memory_mode == "legacy" ||
        memory_mode == "phase_ordered");
    DG_HOST_ASSERT(num_ranks >= 1);
    DG_HOST_ASSERT(
        backward_rank >= 0 && backward_rank < num_ranks);
    DG_HOST_ASSERT(num_experts == num_experts_w2);
    DG_HOST_ASSERT(hidden == hidden_w2);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(block_m % 16 == 0);
    DG_HOST_ASSERT(hidden % 256 == 0);
    DG_HOST_ASSERT(intermediate_hidden % 256 == 0);
    DG_HOST_ASSERT(num_pool_rows > 0);
    DG_HOST_ASSERT(num_max_tokens_per_rank > 0);
    DG_HOST_ASSERT(num_topk > 0);

    const auto check_bf16_contiguous = [](const torch::Tensor& tensor) {
        DG_HOST_ASSERT(tensor.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(tensor.is_contiguous());
    };
    check_bf16_contiguous(gate_up_output);
    check_bf16_contiguous(grad_h_output);
    check_bf16_contiguous(grad_gate_up_output);
    check_bf16_contiguous(h_act_output);
    check_bf16_contiguous(h_weighted_output);
    check_bf16_contiguous(x_pool_output);
    check_bf16_contiguous(grad_x_pool_output);
    check_bf16_contiguous(grad_ye);
    check_bf16_contiguous(grad_y_unweighted_output);
    check_bf16_contiguous(down_unweighted_output);
    check_bf16_contiguous(w2_weights);
    check_bf16_contiguous(w13_weights);
    check_bf16_contiguous(backward_grad_y);
    check_bf16_contiguous(backward_x);
    DG_HOST_ASSERT(backward_topk_weights.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(backward_topk_weights.is_contiguous());
    if (backward_grad_route.has_value()) {
        DG_HOST_ASSERT(
            backward_grad_route->scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(backward_grad_route->is_contiguous());
    }
    DG_HOST_ASSERT(route_weights.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(route_weights.is_contiguous());
    DG_HOST_ASSERT(grad_route_output.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(grad_route_output.is_contiguous());
    DG_HOST_ASSERT(expert_counts.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(expert_counts.is_contiguous());
    DG_HOST_ASSERT(expert_counts.numel() == num_experts);
    DG_HOST_ASSERT(grid_sync_counter.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(grid_sync_counter.is_contiguous());
    DG_HOST_ASSERT(token_src_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(token_src_metadata.is_contiguous());

    DG_HOST_ASSERT(
        gate_up_output.sizes() ==
        torch::IntArrayRef(
            {num_pool_rows, intermediate_hidden_2}));
    DG_HOST_ASSERT(grad_gate_up_output.sizes() ==
                   gate_up_output.sizes());
    DG_HOST_ASSERT(
        grad_h_output.sizes() ==
        torch::IntArrayRef(
            {num_pool_rows, intermediate_hidden}));
    DG_HOST_ASSERT(h_act_output.sizes() == grad_h_output.sizes());
    DG_HOST_ASSERT(
        h_weighted_output.sizes() == grad_h_output.sizes());
    DG_HOST_ASSERT(
        x_pool_output.sizes() ==
        torch::IntArrayRef({num_pool_rows, hidden}));
    DG_HOST_ASSERT(
        write_grad_x_pool
            ? grad_x_pool_output.sizes() == x_pool_output.sizes()
            : grad_x_pool_output.sizes() ==
                  torch::IntArrayRef({0, hidden}));
    DG_HOST_ASSERT(grad_ye.sizes() == x_pool_output.sizes());
    DG_HOST_ASSERT(
        grad_y_unweighted_output.sizes() ==
        x_pool_output.sizes());
    DG_HOST_ASSERT(
        down_unweighted_output.sizes() ==
        x_pool_output.sizes());
    DG_HOST_ASSERT(route_weights.numel() == num_pool_rows);
    DG_HOST_ASSERT(grad_route_output.numel() == num_pool_rows);
    DG_HOST_ASSERT(
        backward_grad_y.size(0) >= num_max_tokens_per_rank &&
        backward_grad_y.size(1) == hidden);
    DG_HOST_ASSERT(
        backward_x.size(0) >= num_max_tokens_per_rank &&
        backward_x.size(1) == hidden);
    DG_HOST_ASSERT(
        backward_topk_weights.size(0) >=
            num_max_tokens_per_rank &&
        backward_topk_weights.size(1) == num_topk);
    if (backward_grad_route.has_value()) {
        DG_HOST_ASSERT(
            backward_grad_route->size(0) >=
                num_max_tokens_per_rank &&
            backward_grad_route->size(1) == num_topk);
    }
    DG_HOST_ASSERT(
        token_src_metadata.size(0) >= num_pool_rows &&
        token_src_metadata.size(1) == 3);
    DG_HOST_ASSERT(write_grad_x_pool || direct_remote_grad_x);
    if (kernel_trace.has_value()) {
        DG_HOST_ASSERT(kernel_trace->is_cuda());
        DG_HOST_ASSERT(
            kernel_trace->scalar_type() == torch::kInt64);
        DG_HOST_ASSERT(kernel_trace->is_contiguous());
        DG_HOST_ASSERT(kernel_trace->device() == grad_ye.device());
    }

    const auto exact_alias = [](
        const torch::Tensor& lhs,
        const torch::Tensor& rhs) {
        return lhs.data_ptr() == rhs.data_ptr() &&
               lhs.numel() == rhs.numel();
    };
    const auto overlaps = [](
        const torch::Tensor& lhs,
        const torch::Tensor& rhs) {
        if (lhs.numel() == 0 || rhs.numel() == 0)
            return false;
        const auto lhs_begin = reinterpret_cast<uintptr_t>(
            lhs.data_ptr());
        const auto rhs_begin = reinterpret_cast<uintptr_t>(
            rhs.data_ptr());
        const auto lhs_end =
            lhs_begin + lhs.numel() * lhs.element_size();
        const auto rhs_end =
            rhs_begin + rhs.numel() * rhs.element_size();
        return std::max(lhs_begin, rhs_begin) <
               std::min(lhs_end, rhs_end);
    };
    const std::array<std::reference_wrapper<const torch::Tensor>, 10>
        alias_tensors = {
            gate_up_output, grad_gate_up_output,
            grad_h_output, h_act_output, h_weighted_output,
            x_pool_output, grad_x_pool_output, grad_ye,
            grad_y_unweighted_output, down_unweighted_output};
    const auto allowed_overlap = [&](const int lhs, const int rhs) {
        const auto pair_is = [=](
            const int first, const int second) {
            return (lhs == first && rhs == second) ||
                   (lhs == second && rhs == first);
        };
        if (route_weight_mode == "pre_down" &&
            pair_is(7, 8))
            return exact_alias(
                grad_ye, grad_y_unweighted_output);
        if (route_weight_mode == "pre_down" &&
            pair_is(7, 9))
            return exact_alias(
                grad_ye, down_unweighted_output);
        if (route_weight_mode == "pre_down" &&
            pair_is(8, 9))
            return exact_alias(
                grad_y_unweighted_output,
                down_unweighted_output);
        if (route_weight_mode == "post_down" &&
            pair_is(3, 4))
            return exact_alias(
                h_act_output, h_weighted_output);
        if (memory_mode != "phase_ordered")
            return false;
        if (pair_is(0, 1))
            return exact_alias(
                gate_up_output, grad_gate_up_output);
        if (route_weight_mode == "post_down") {
            if (pair_is(7, 9))
                return exact_alias(
                    grad_ye, down_unweighted_output);
            if (pair_is(2, 8))
                return
                    grad_h_output.data_ptr() ==
                        grad_y_unweighted_output.data_ptr() &&
                    grad_h_output.numel() <=
                        grad_y_unweighted_output.numel();
            if (pair_is(2, 3) || pair_is(2, 4))
                return exact_alias(
                    grad_h_output,
                    pair_is(2, 3)
                        ? h_act_output
                        : h_weighted_output);
            if (pair_is(3, 8) || pair_is(4, 8))
                return
                    grad_y_unweighted_output.data_ptr() ==
                    (pair_is(3, 8)
                         ? h_act_output.data_ptr()
                         : h_weighted_output.data_ptr());
        } else if (pair_is(3, 4)) {
            return exact_alias(
                h_act_output, h_weighted_output);
        }
        return false;
    };
    for (int lhs = 0; lhs < alias_tensors.size(); ++lhs) {
        for (int rhs = lhs + 1;
             rhs < alias_tensors.size(); ++rhs) {
            DG_HOST_ASSERT(
                !overlaps(
                    alias_tensors[lhs].get(),
                    alias_tensors[rhs].get()) ||
                allowed_overlap(lhs, rhs));
        }
    }
    if (memory_mode == "phase_ordered") {
        DG_HOST_ASSERT(exact_alias(
            gate_up_output, grad_gate_up_output));
        if (route_weight_mode == "post_down") {
            DG_HOST_ASSERT(exact_alias(
                grad_ye, down_unweighted_output));
            DG_HOST_ASSERT(
                grad_h_output.data_ptr() ==
                    grad_y_unweighted_output.data_ptr() &&
                grad_h_output.numel() <=
                    grad_y_unweighted_output.numel());
            DG_HOST_ASSERT(exact_alias(
                grad_h_output, h_act_output));
            DG_HOST_ASSERT(exact_alias(
                grad_h_output, h_weighted_output));
        } else {
            DG_HOST_ASSERT(exact_alias(
                grad_ye, grad_y_unweighted_output));
            if (activation == "swiglu" &&
                activation_limit !=
                    cute::numeric_limits<float>::infinity()) {
                // Finite-clamped PRE_DOWN keeps FP32 activation*route
                // until its single BF16 store; the route plane is separate.
                DG_HOST_ASSERT(!overlaps(
                    h_act_output, h_weighted_output));
            } else {
                DG_HOST_ASSERT(exact_alias(
                    h_act_output, h_weighted_output));
            }
        }
    }

    const int num_w2_states =
        num_experts * (hidden / dgrad_block_k) *
        (intermediate_hidden / block_n);
    const int num_w13_states =
        num_experts *
        (intermediate_hidden_2 / dgrad_block_k) *
        (hidden / block_n);
    DG_HOST_ASSERT(
        grid_sync_counter.numel() >=
        num_w2_states + num_w13_states + 2);

    const int smem_cd =
        store_block_m * block_n *
        static_cast<int>(sizeof(cutlass::bfloat16_t)) *
        num_tma_store_stages;
    const int smem_per_stage =
        load_block_m * block_k +
        load_block_n * block_k +
        sf_block_m * static_cast<int>(sizeof(uint32_t)) +
        sf_block_n * static_cast<int>(sizeof(uint32_t)) +
        2 * static_cast<int>(sizeof(uint64_t));
    const int smem_fixed =
        num_dispatch_warps * hidden *
            static_cast<int>(sizeof(cutlass::bfloat16_t)) +
        smem_cd +
        2 * num_epilogue_stages *
            static_cast<int>(sizeof(uint64_t)) +
        num_dispatch_warps *
            static_cast<int>(sizeof(uint64_t)) +
        static_cast<int>(sizeof(uint32_t));
    const int num_stages =
        std::min(
            32,
            (smem_capacity - smem_fixed) / smem_per_stage);
    const int smem_size =
        align(
            smem_fixed + num_stages * smem_per_stage,
            1024);
    DG_HOST_ASSERT(num_stages >= 2);

    // Recompute descriptors are compile-time dead in BF16 mode. Keep valid
    // descriptors in every slot so descriptor prefetch remains well-defined.
    const auto tensor_map_acts = make_tma_2d_desc(
        x_pool_output, hidden, num_pool_rows,
        block_k, load_block_m,
        static_cast<int>(x_pool_output.stride(-2)), 128);
    const auto tensor_map_w13 = make_tma_2d_desc(
        w13_weights, hidden,
        num_experts * intermediate_hidden_2,
        block_k, load_block_n,
        static_cast<int>(w13_weights.stride(-2)), 128);
    const auto tensor_map_w13_dgrad = make_tma_2d_desc(
        w13_weights, hidden,
        num_experts * intermediate_hidden_2,
        load_block_n, dgrad_block_k,
        static_cast<int>(w13_weights.stride(-2)), 128);
    const auto tensor_map_gate_up = make_tma_2d_desc(
        gate_up_output, intermediate_hidden_2, num_pool_rows,
        block_n, store_block_m,
        static_cast<int>(gate_up_output.stride(-2)), 128);
    const auto tensor_map_grad_ye = make_tma_2d_desc(
        grad_ye, hidden, num_pool_rows,
        dgrad_block_k, load_block_m,
        static_cast<int>(grad_ye.stride(-2)), 128);
    const auto tensor_map_w2 = make_tma_2d_desc(
        w2_weights, intermediate_hidden,
        num_experts * hidden,
        load_block_n, dgrad_block_k,
        intermediate_hidden, 128);
    const auto tensor_map_grad_gate_up = make_tma_2d_desc(
        grad_gate_up_output, intermediate_hidden_2,
        num_pool_rows,
        dgrad_block_k, load_block_m,
        static_cast<int>(
            grad_gate_up_output.stride(-2)), 128);

    const int num_sms = get_mega_moe_num_sms();
    DG_HOST_ASSERT(num_sms % 2 == 0);
    constexpr int num_trace_sites = 22;
    constexpr int num_trace_values = 5;
    if (kernel_trace.has_value()) {
        DG_HOST_ASSERT(
            kernel_trace->dim() == 3 &&
            kernel_trace->size(0) == num_trace_sites &&
            kernel_trace->size(1) == num_sms &&
            kernel_trace->size(2) == num_trace_values);
    }
    static std::atomic<uint32_t> next_launch_epoch{1};
    uint32_t launch_epoch =
        next_launch_epoch.fetch_add(
            1, std::memory_order_relaxed);
    if (launch_epoch == 0) {
        launch_epoch =
            next_launch_epoch.fetch_add(
                1, std::memory_order_relaxed);
    }

    const auto backward_sym_buffer =
        layout::SymBuffer<>(
            backward_sym_buffer_ptrs, backward_rank);
    const auto backward_workspace = layout::Workspace(
        reinterpret_cast<void*>(
            backward_sym_buffer_ptrs[backward_rank]),
        num_ranks, num_experts * num_ranks,
        num_max_tokens_per_rank, num_topk,
        layout::kMinCandidateBlockM);

    const SM100FP8FP4MegaMoEBackwardWaveRuntime::Args args = {
        .hidden = hidden,
        .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts,
        .num_pool_rows = num_pool_rows,
        .num_acts_rows = 0,
        .num_sf_pool_rows = 1,
        .block_m = block_m,
        .block_n = block_n,
        .block_k = block_k,
        .sf_block_m = sf_block_m,
        .sf_block_n = sf_block_n,
        .num_stages = num_stages,
        .num_sms = num_sms,
        .num_ranks = num_ranks,
        .bf16_mode = true,
        .activation = activation,
        .fast_math = fast_math,
        .route_weight_mode = route_weight_mode,
        .combine_order_mode = combine_order_mode,
        .expert_counts = expert_counts.data_ptr<int>(),
        .backward_sym_buffer = backward_sym_buffer,
        .backward_workspace = backward_workspace,
        .backward_grad_y =
            reinterpret_cast<const cutlass::bfloat16_t*>(
                backward_grad_y.data_ptr<at::BFloat16>()),
        .backward_x =
            reinterpret_cast<const cutlass::bfloat16_t*>(
                backward_x.data_ptr<at::BFloat16>()),
        .backward_topk_weights =
            backward_topk_weights.data_ptr<float>(),
        .backward_grad_route =
            backward_grad_route.has_value()
            ? backward_grad_route->data_ptr<float>()
            : nullptr,
        .token_src_metadata =
            reinterpret_cast<
                const layout::TokenSrcMetadata*>(
                token_src_metadata.data_ptr<int>()),
        .num_topk = static_cast<uint32_t>(num_topk),
        .acts_sf_stride = 0,
        .tensor_map_acts = tensor_map_acts,
        .tensor_map_acts_sf = tensor_map_acts,
        .tensor_map_weights = tensor_map_w13,
        .tensor_map_weights_sf = tensor_map_w13,
        .tensor_map_output = tensor_map_gate_up,
        .tensor_map_grad_ye = tensor_map_grad_ye,
        .tensor_map_w2_dequant = tensor_map_w2,
        .tensor_map_w2_weights = tensor_map_w2,
        .tensor_map_w2_scales = tensor_map_w2,
        .tensor_map_w13_dequant = tensor_map_w13_dgrad,
        .tensor_map_w13_weights = tensor_map_w13,
        .tensor_map_w13_scales = tensor_map_w13,
        .tensor_map_grad_gate_up =
            tensor_map_grad_gate_up,
        .acts_ptr = nullptr,
        .acts_sf_ptr = nullptr,
        .w2_weights = nullptr,
        .w2_scales = nullptr,
        .w2_dequant_scratch =
            reinterpret_cast<cutlass::bfloat16_t*>(
                w2_weights.data_ptr<at::BFloat16>()),
        .w13_weights = nullptr,
        .w13_scales = nullptr,
        .w13_dequant_scratch =
            reinterpret_cast<cutlass::bfloat16_t*>(
                w13_weights.data_ptr<at::BFloat16>()),
        .gate_up_output =
            reinterpret_cast<const cutlass::bfloat16_t*>(
                gate_up_output.data_ptr<at::BFloat16>()),
        .grad_ye_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                grad_ye.data_ptr<at::BFloat16>()),
        .grad_y_unweighted_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                grad_y_unweighted_output
                    .data_ptr<at::BFloat16>()),
        .route_weights =
            nullptr,
        .route_weights_fp32 = route_weights.data_ptr<float>(),
        .grad_h_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                grad_h_output.data_ptr<at::BFloat16>()),
        .grad_gate_up_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                grad_gate_up_output.data_ptr<at::BFloat16>()),
        .h_act_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                h_act_output.data_ptr<at::BFloat16>()),
        .h_weighted_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                h_weighted_output.data_ptr<at::BFloat16>()),
        .x_pool_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                x_pool_output.data_ptr<at::BFloat16>()),
        .grad_x_pool_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                grad_x_pool_output.data_ptr<at::BFloat16>()),
        .down_unweighted_output =
            reinterpret_cast<const cutlass::bfloat16_t*>(
                down_unweighted_output
                    .data_ptr<at::BFloat16>()),
        .grad_route_output =
            grad_route_output.data_ptr<float>(),
        .grid_sync_counter =
            reinterpret_cast<uint32_t*>(
                grid_sync_counter.data_ptr<int>()),
        .launch_epoch = launch_epoch,
        .activation_limit = activation_limit,
        .compute_w13_dgrad = true,
        .direct_remote_grad_x = direct_remote_grad_x,
        .write_grad_x_pool = write_grad_x_pool,
        .clear_wgrad_padding = clear_wgrad_padding,
        .compute_route_grad = true,
        .trace_kernel = kernel_trace.has_value(),
        .vectorized_grad_x_store = deep_jit::get_env<int>(
            "DG_BF16_MEGA_MOE_VECTORIZED_GRAD_X_STORE",
            1) == 1,
        .wide_grad_x_store = deep_jit::get_env<int>(
            "DG_BF16_MEGA_MOE_WIDE_GRAD_X_STORE",
            0) == 1,
        .kernel_trace =
            kernel_trace.has_value()
            ? reinterpret_cast<uint64_t*>(
                  kernel_trace->data_ptr<int64_t>())
            : nullptr,
        .inputs_prepared =
            memory_mode == "phase_ordered" &&
            route_weight_mode == "post_down",
        .dispatch_inputs_prepared =
            memory_mode == "phase_ordered",
        .launch_args =
            make_mega_moe_launch_options(num_sms, 1024, smem_size, 2),
    };
    const auto code =
        SM100FP8FP4MegaMoEBackwardWaveRuntime::generate(args);
    const auto runtime = jit->compile(
        std::format(
            "sm100_bf16_mega_moe_backward_dgrad_trace{}_vec{}_wide{}",
            kernel_trace.has_value(),
            args.vectorized_grad_x_store,
            args.wide_grad_x_store),
        code);
    SM100FP8FP4MegaMoEBackwardWaveRuntime::launch(
        runtime, args);
}

}  // namespace deep_gemm
