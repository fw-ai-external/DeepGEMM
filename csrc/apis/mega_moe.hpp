#pragma once

#include <functional>
#include <string>
#include <pybind11/functional.h>

#include <deep_gemm/common/types.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>

#include "../runtime/runtime.hpp"
#include "../jit_kernels/impls/sm100_bf16_mega_moe.hpp"
#include "../jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp"
#include "../jit_kernels/impls/sm100_bf16_mega_moe_side_lora_forward.hpp"
#include "../jit_kernels/impls/sm100_fp8_fp4_mega_moe_side_lora_forward.hpp"
#include "locality_domain.hpp"

namespace deep_gemm::mega {

using MegaMoELegacySlices = std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor>;
using MegaMoEExpandedSlices = std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor>;

using MegaMoESharedSlices = std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;

static int get_token_alignment_for_mega_moe() {
    return layout::kLCMCandidateBlockM;
}

static int get_block_m_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const std::string& mma_type) {
    DG_HOST_ASSERT(num_tokens >= 0);
    const auto mma_kind = parse_mma_kind(mma_type);
    const auto [cluster_size, block_m, store_block_m_l1, store_block_m_l2, block_k, num_epilogue_threads] =
        get_block_config_for_mega_moe(num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_tokens, mma_kind);
    return block_m;
}

static int get_num_ring_tokens_for_mega_moe(
    const int num_ranks, const int num_experts,
    const int num_max_tokens_per_rank, const int num_topk,
    const int hidden, const int intermediate_hidden) {
    DG_HOST_ASSERT(num_ranks > 0 && num_experts > 0 && num_experts % num_ranks == 0);
    // Ring capacity: worst-case live pool blocks over all candidate BLOCK_M; mirrors the kernel assert.
    // TODO: we temporarily assume the SM count is consistent with the runtime value
    const auto num_sms = get_mega_moe_num_sms();
    const auto num_experts_per_rank = num_experts / num_ranks;
    const auto num_active_topk = std::min(num_topk, num_experts_per_rank);
    const auto num_max_routed_tokens = num_max_tokens_per_rank * num_ranks * num_active_topk;

    // Iterate all block candidates to get the maximum ring size
    int num_ring_tokens = 0;
    for (const auto& block_m: layout::kCandidateBlockM) {
        const auto num_pool_blocks = ceil_div(num_max_routed_tokens, block_m) + num_experts_per_rank;
        const auto num_live_pool_blocks = sched::get_num_max_live_pool_blocks(
            num_pool_blocks, num_sms, hidden, intermediate_hidden);
        num_ring_tokens = std::max(num_ring_tokens, num_live_pool_blocks * block_m);
    }
    num_ring_tokens = math::align(num_ring_tokens, layout::kLCMCandidateBlockM);
    return num_ring_tokens;
}

static std::tuple<
    int64_t,
    std::function<MegaMoESharedSlices(const torch::Tensor&)>>
get_symm_buffer_size_for_mega_moe_v3(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const std::string& mma_type, const std::string& activation,
    const int& requested_num_ring_tokens,
    const int& num_shared_experts = 0) {
    DG_HOST_ASSERT(num_ranks > 0 and num_experts > 0 and num_topk > 0);
    DG_HOST_ASSERT(num_experts % num_ranks == 0);
    DG_HOST_ASSERT(activation == "swiglu" or activation == "geglu" or activation == "situ");
    DG_HOST_ASSERT(num_shared_experts >= 0);

    const auto num_experts_per_rank = num_experts / num_ranks;
    const int shared_intermediate_hidden = intermediate_hidden * num_shared_experts;
    int num_ring_tokens = get_num_ring_tokens_for_mega_moe(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden);
    DG_HOST_ASSERT(requested_num_ring_tokens >= 0);
    if (requested_num_ring_tokens > 0) {
        DG_HOST_ASSERT(requested_num_ring_tokens % layout::kLCMCandidateBlockM == 0);
        DG_HOST_ASSERT(requested_num_ring_tokens >= num_ring_tokens and
                       "Explicit ring capacity is too small for the upstream scheduler");
        num_ring_tokens = requested_num_ring_tokens;
    }

    // Parse MMA type
    const auto mma_kind = parse_mma_kind(mma_type);
    const auto with_sf = is_mma_with_sf(mma_kind);

    // Compute num_sf_ring_tokens (max across all candidate block sizes)
    int num_sf_ring_tokens = 0;
    if (with_sf) {
        for (auto block_m: layout::kCandidateBlockM) {
            num_sf_ring_tokens = std::max(
                num_sf_ring_tokens,
                layout::get_num_sf_ring_tokens(num_ring_tokens, block_m));
        }
    }

    // All buffers
    const auto mega_buffer = layout::MegaMoEBuffer(
        nullptr, hidden, intermediate_hidden,
        num_ranks, num_experts, num_max_tokens_per_rank,
        num_topk, num_ring_tokens, num_sf_ring_tokens, with_sf,
        num_shared_experts
    );
    const auto& workspace = mega_buffer.workspace;
    const auto& combine_token_buffer = mega_buffer.combine_token_buffer;
    const auto backward_route_grad_buffer = layout::Buffer(
        layout::Data(num_topk * sizeof(float), false), 1, num_max_tokens_per_rank,
        combine_token_buffer.get_end_ptr());

    // Check SF buffer requirements
    if (with_sf) {
        DG_HOST_ASSERT(hidden % 128 == 0 and intermediate_hidden % 128 == 0);
        DG_HOST_ASSERT(shared_intermediate_hidden % 128 == 0);
        DG_HOST_ASSERT(num_sf_ring_tokens % 4 == 0);
    }

    // Slice function: creates tensor views from the raw buffer.
    // NOTES: `x_sf` is K-major, while `l1_acts_sf` and `l2_acts_sf` are M-major
    auto slice_input_buffers = [=](const torch::Tensor& buffer) {
        auto x = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.input_token_buffer.base)),
            {num_max_tokens_per_rank, hidden},
            torch::TensorOptions().dtype(with_sf ? torch::kFloat8_e4m3fn : torch::kBFloat16).device(buffer.device()));
        auto x_sf = with_sf ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.input_sf_buffer.base)),
            {num_max_tokens_per_rank, hidden / 128},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
        auto topk_idx = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.input_topk_idx_buffer.base)),
            {num_max_tokens_per_rank, num_topk},
            torch::TensorOptions().dtype(torch::kInt64).device(buffer.device()));
        auto topk_weights = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.input_topk_weights_buffer.base)),
            {num_max_tokens_per_rank, num_topk},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));

        auto shared_l1_acts = x;
        auto shared_l1_acts_sf = (with_sf and num_shared_experts > 0) ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.shared_l1_sf_buffer.base)),
            {layout::get_num_max_shared_sf_tokens(num_max_tokens_per_rank), hidden / 128},
            {1, layout::get_num_max_shared_sf_tokens(num_max_tokens_per_rank)},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
        auto shared_l2_acts = num_shared_experts > 0 ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.shared_l2_token_buffer.base)),
            {num_max_tokens_per_rank, shared_intermediate_hidden},
            torch::TensorOptions().dtype(with_sf ? torch::kFloat8_e4m3fn : torch::kBFloat16).device(buffer.device())) : torch::Tensor();
        auto shared_l2_acts_sf = (with_sf and num_shared_experts > 0) ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.shared_l2_sf_buffer.base)),
            {layout::get_num_max_shared_sf_tokens(num_max_tokens_per_rank), shared_intermediate_hidden / 128},
            {1, layout::get_num_max_shared_sf_tokens(num_max_tokens_per_rank)},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();

        auto l1_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l1_token_buffer.base)),
            {num_ring_tokens, hidden},
            torch::TensorOptions().dtype(with_sf ? torch::kFloat8_e4m3fn : torch::kBFloat16).device(buffer.device()));
        auto l1_acts_sf = with_sf ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l1_sf_buffer.base)),
            {num_sf_ring_tokens, hidden / 128},
            {1, num_sf_ring_tokens},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
        auto l2_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l2_token_buffer.base)),
            {num_ring_tokens, intermediate_hidden},
            torch::TensorOptions().dtype(with_sf ? torch::kFloat8_e4m3fn : torch::kBFloat16).device(buffer.device()));
        auto l2_acts_sf = with_sf ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l2_sf_buffer.base)),
            {num_sf_ring_tokens, intermediate_hidden / 128},
            {1, num_sf_ring_tokens},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
        // Training-backward metadata (Workspace region, base of the buffer):
        //   `token_src_metadata`: per pool-token (rank_idx, token_idx, topk_idx)
        //       source mapping for combine write-back -> lets the backward scatter
        //       per-expert grads back to source tokens/top-k slots without
        //       reverse-engineering the (swizzled) pool packing. Every populated
        //       pool row self-describes its source token + top-k slot.
        // `workspace` above is nullptr-based, so its (now HOST_DEVICE) accessor
        // returns a byte offset into the symmetric buffer.
        auto token_src_metadata = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(workspace.get_token_src_metadata_ptr(0))),
            {static_cast<int64_t>(workspace.num_max_pool_tokens), 3},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device()));
        // The combine region is dead after the forward returns. Keep the public
        // grad-y view two-dimensional, but retain storage for every top-k plane
        // so BF16 backward can reuse the region for direct grad-x write-back.
        // MXFP4 callers continue to observe the original [tokens, hidden] view.
        auto backward_combine_planes = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(combine_token_buffer.base)),
            {num_topk * num_max_tokens_per_rank, hidden},
            torch::TensorOptions().dtype(torch::kBFloat16).device(buffer.device()));
        auto backward_grad_y = backward_combine_planes.narrow(
            0, 0, num_max_tokens_per_rank);
        auto backward_grad_route =
            buffer.nbytes() >= static_cast<size_t>(
                reinterpret_cast<int64_t>(
                    backward_route_grad_buffer.get_end_ptr()))
            ? torch::from_blob(
                  math::advance_ptr(
                      buffer.data_ptr(),
                      reinterpret_cast<int64_t>(
                          backward_route_grad_buffer.base)),
                  {num_max_tokens_per_rank, num_topk},
                  torch::TensorOptions()
                      .dtype(torch::kFloat32)
                      .device(buffer.device()))
            : torch::Tensor();
        return std::make_tuple(x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf,
                               token_src_metadata, backward_grad_y, backward_grad_route,
                               shared_l1_acts, shared_l1_acts_sf, shared_l2_acts, shared_l2_acts_sf);
    };
    return {reinterpret_cast<int64_t>(backward_route_grad_buffer.get_end_ptr()), slice_input_buffers};
}

static std::tuple<int64_t, std::function<MegaMoEExpandedSlices(const torch::Tensor&)>>
get_symm_buffer_size_for_mega_moe_v2(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const std::string& mma_type, const std::string& activation,
    const int& num_ring_tokens) {
    auto [num_bytes, slicer] = get_symm_buffer_size_for_mega_moe_v3(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden, mma_type, activation, num_ring_tokens, 0);
    return {num_bytes, [slicer](const torch::Tensor& buffer) {
        auto [x, x_sf, ids, weights, l1, l1_sf, l2, l2_sf, metadata, grad_y, grad_route,
              shared_l1, shared_l1_sf, shared_l2, shared_l2_sf] = slicer(buffer);
        return std::make_tuple(x, x_sf, ids, weights, l1, l1_sf, l2, l2_sf, metadata, grad_y, grad_route);
    }};
}

// Keep the original raw _C slicer ABI and allocation size. Training callers
// opt in to the appended backward plane through the versioned v2 API.
static std::tuple<
    int64_t,
    std::function<MegaMoELegacySlices(const torch::Tensor&)>>
get_symm_buffer_size_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const std::string& mma_type, const std::string& activation,
    const int& num_ring_tokens) {
    auto [expanded_num_bytes, expanded_slicer] =
        get_symm_buffer_size_for_mega_moe_v2(
            num_ranks, num_experts, num_max_tokens_per_rank,
            num_topk, hidden, intermediate_hidden, mma_type,
            activation, num_ring_tokens);
    const int64_t route_plane_bytes =
        static_cast<int64_t>(num_max_tokens_per_rank) *
        num_topk * sizeof(float);
    auto legacy_slicer =
        [expanded_slicer](const torch::Tensor& buffer) {
            auto [x, x_sf, topk_idx, topk_weights, l1_acts,
                  l1_acts_sf, l2_acts, l2_acts_sf,
                  token_src_metadata, backward_grad_y,
                  _backward_grad_route] =
                expanded_slicer(buffer);
            return std::make_tuple(
                x, x_sf, topk_idx, topk_weights, l1_acts,
                l1_acts_sf, l2_acts, l2_acts_sf,
                token_src_metadata, backward_grad_y);
        };
    return {
        expanded_num_bytes - route_plane_bytes,
        legacy_slicer};
}


using MegaMoESideLoraSlices = decltype(std::tuple_cat(
    std::declval<MegaMoESharedSlices>(), std::declval<std::tuple<torch::Tensor>>()));

// Side-LoRA retains its own fused phase scheduler. Its wave must fit at
// least one expert's worst-case input, in addition to the upstream ring bound.
static std::tuple<int64_t, std::function<MegaMoESideLoraSlices(const torch::Tensor&)>>
get_symm_buffer_size_for_mega_moe_side_lora(
    const int num_ranks, const int num_experts,
    const int num_max_tokens_per_rank, const int num_topk,
    const int hidden, const int intermediate_hidden,
    const std::string& mma_type, const std::string& activation,
    const int requested_num_ring_tokens) {
    const auto minimum = std::max(
        num_ranks * num_max_tokens_per_rank,
        get_num_ring_tokens_for_mega_moe(
            num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden));
    const auto ring_tokens = requested_num_ring_tokens > 0 ? requested_num_ring_tokens : minimum;
    DG_HOST_ASSERT(ring_tokens >= minimum);
    const auto [base_bytes, base_slice] = get_symm_buffer_size_for_mega_moe_v3(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden, mma_type, activation, ring_tokens, 0);
    const bool with_sf = is_mma_with_sf(parse_mma_kind(mma_type));
    const int64_t extra_bytes = with_sf ? int64_t(num_max_tokens_per_rank) * hidden * 2 : 0;
    auto slice = [=](const torch::Tensor& buffer) {
        DG_HOST_ASSERT(int64_t(buffer.nbytes()) >= base_bytes + extra_bytes);
        auto views = base_slice(buffer);
        auto source = with_sf ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), base_bytes),
            {num_max_tokens_per_rank, hidden},
            torch::TensorOptions().dtype(torch::kBFloat16).device(buffer.device()))
            : std::get<0>(views);
        return std::tuple_cat(views, std::make_tuple(source));
    };
    return {base_bytes + extra_bytes, slice};
}

static void fp8_fp4_mega_moe(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math,
    const int& num_ring_tokens,
    const std::optional<torch::Tensor>& saved_l1_preact,
    const std::string& route_weight_mode,
    const std::optional<torch::Tensor>& saved_down_unweighted,
    const std::optional<torch::Tensor>& saved_l1_acts,
    const std::optional<torch::Tensor>& saved_l1_acts_sf,
    const std::optional<int>& num_config_tokens_opt,
    const std::optional<std::tuple<torch::Tensor, torch::Tensor>>& shared_l1_weights_tuple_opt,
    const std::optional<std::tuple<torch::Tensor, torch::Tensor>>& shared_l2_weights_tuple_opt,
    const std::optional<float>& situ_beta_opt,
    const std::optional<float>& situ_linear_beta_opt
) {
    const auto [l1_weights, l1_weights_sf] = l1_weights_tuple;
    const auto [l2_weights, l2_weights_sf] = l2_weights_tuple;

    // Config checks
    const auto num_tokens = static_cast<int>(y.size(0));
    const auto num_config_tokens =
        num_config_tokens_opt.value_or(num_tokens);
    const auto [rm, rn, rk] = recipe;
    DG_HOST_ASSERT(rm == 1 and rn == 1 and rk == 32);
    DG_HOST_ASSERT(activation == "swiglu" or activation == "geglu" or activation == "situ");
    DG_HOST_ASSERT(
        route_weight_mode == "pre_down" ||
        route_weight_mode == "post_down");
    DG_HOST_ASSERT(shared_l1_weights_tuple_opt.has_value() == shared_l2_weights_tuple_opt.has_value());

    // Activation checks
    const auto situ_beta = situ_beta_opt.value_or(1.0f);
    const auto situ_linear_beta = situ_linear_beta_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(situ_beta > 0);
    DG_HOST_ASSERT(situ_linear_beta > 0);
    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

    // Tensor checks
    const auto arch_major = jit->device.get_arch_major();
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] = check_weights_layout_3d(l1_weights);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] = check_weights_layout_3d(l2_weights);
    DG_HOST_ASSERT(is_localized(l1_weights) == is_localized(l2_weights));
    const auto weight_dtype = l1_weights.scalar_type();
    DG_HOST_ASSERT(weight_dtype == torch::kFloat8_e4m3fn or weight_dtype == kPackedFP4);
    DG_HOST_ASSERT(l2_weights.scalar_type() == weight_dtype);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(num_config_tokens >= num_tokens);
    DG_HOST_ASSERT(num_config_tokens <= num_max_tokens_per_rank);

    // Check weight SF layout for UE8M0 packing, MN-major, and TMA alignment
    constexpr int kGranMN = 1, kGranK = 32;
    check_sf_layout(l1_weights_sf, intermediate_hidden * 2, hidden, kGranMN, kGranK,
                    num_experts_per_rank, true, false, torch::kInt);
    check_sf_layout(l2_weights_sf, hidden, intermediate_hidden, kGranMN, kGranK,
                    num_experts_per_rank, true, false, torch::kInt);

    int num_shared_experts = 0, shared_intermediate_hidden = 0;
    torch::Tensor shared_l1_weights, shared_l1_weights_sf, shared_l2_weights, shared_l2_weights_sf;
    if (shared_l1_weights_tuple_opt.has_value()) {
        std::tie(shared_l1_weights, shared_l1_weights_sf) = shared_l1_weights_tuple_opt.value();
        std::tie(shared_l2_weights, shared_l2_weights_sf) = shared_l2_weights_tuple_opt.value();
        const auto [shared_l1_n, shared_l1_k] = check_weights_layout_2d(shared_l1_weights);
        const auto [shared_l2_n, shared_l2_k] = check_weights_layout_2d(shared_l2_weights);
        DG_HOST_ASSERT(is_localized(shared_l1_weights) == is_localized(l1_weights));
        DG_HOST_ASSERT(is_localized(shared_l2_weights) == is_localized(l2_weights));
        shared_intermediate_hidden = shared_l2_k;
        num_shared_experts = shared_intermediate_hidden / intermediate_hidden;

        DG_HOST_ASSERT(shared_intermediate_hidden % intermediate_hidden == 0);
        DG_HOST_ASSERT(shared_l1_n == shared_intermediate_hidden * 2);
        DG_HOST_ASSERT(shared_l1_k == hidden);
        DG_HOST_ASSERT(shared_l2_n == hidden);
        DG_HOST_ASSERT(shared_l1_weights.scalar_type() == torch::kFloat8_e4m3fn);
        DG_HOST_ASSERT(shared_l2_weights.scalar_type() == torch::kFloat8_e4m3fn);
        check_sf_layout(shared_l1_weights_sf, shared_intermediate_hidden * 2, hidden, kGranMN, kGranK,
                        std::nullopt, true, false, torch::kInt);
        check_sf_layout(shared_l2_weights_sf, hidden, shared_intermediate_hidden, kGranMN, kGranK,
                        std::nullopt, true, false, torch::kInt);
    }

    // Training saves use routed-expert pool offsets; shared experts have separate storage.
    DG_HOST_ASSERT(num_shared_experts == 0 or
        (not saved_l1_preact.has_value() and not saved_l1_acts.has_value() and
         not saved_down_unweighted.has_value() and route_weight_mode == "pre_down"));

    // Check stats counter
    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }

    // Check buffer bytes
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    const auto num_max_pool_tokens =
        layout::get_num_max_pool_tokens(
            num_ranks, num_max_tokens_per_rank, num_topk,
            num_experts_per_rank);
    if (saved_down_unweighted.has_value()) {
        DG_HOST_ASSERT(
            saved_down_unweighted->scalar_type() ==
            torch::kBFloat16);
        DG_HOST_ASSERT(saved_down_unweighted->is_contiguous());
        DG_HOST_ASSERT(saved_down_unweighted->dim() == 2);
        DG_HOST_ASSERT(saved_down_unweighted->size(1) == hidden);
        DG_HOST_ASSERT(saved_down_unweighted->size(0) > 0);
        DG_HOST_ASSERT(
            saved_down_unweighted->size(0) <=
            num_max_pool_tokens);
    }
    DG_HOST_ASSERT(
        saved_l1_acts.has_value() ==
        saved_l1_acts_sf.has_value());
    if (saved_l1_acts.has_value()) {
        DG_HOST_ASSERT(
            saved_l1_acts->scalar_type() ==
            torch::kFloat8_e4m3fn);
        DG_HOST_ASSERT(saved_l1_acts->is_contiguous());
        DG_HOST_ASSERT(saved_l1_acts->dim() == 2);
        DG_HOST_ASSERT(saved_l1_acts->size(1) == hidden);
        DG_HOST_ASSERT(saved_l1_acts->size(0) > 0);
        DG_HOST_ASSERT(
            saved_l1_acts->size(0) <= num_max_pool_tokens);

        const auto& saved_sf = *saved_l1_acts_sf;
        DG_HOST_ASSERT(saved_sf.scalar_type() == torch::kInt);
        DG_HOST_ASSERT(saved_sf.dim() == 2);
        DG_HOST_ASSERT(saved_sf.size(1) == hidden / 128);
        DG_HOST_ASSERT(saved_sf.stride(0) == 1);
        DG_HOST_ASSERT(saved_sf.stride(1) == saved_sf.size(0));
    }
    const auto [expanded_num_required_bytes, slice] =
        get_symm_buffer_size_for_mega_moe_v3(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        weight_dtype == torch::kFloat8_e4m3fn ? "fp8xfp8" : "fp8xfp4", activation, num_ring_tokens, num_shared_experts);
    const auto num_required_bytes =
        expanded_num_required_bytes -
        static_cast<int64_t>(num_max_tokens_per_rank) *
            num_topk * sizeof(float);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    // Already registered tensors
    const auto [x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf,
                token_src_metadata, backward_grad_y, _backward_grad_route,
                shared_l1_acts, shared_l1_acts_sf, shared_l2_acts, shared_l2_acts_sf] = slice(sym_buffer);

    // Dispatch into different architectures
    if (arch_major == 10) {
        sm100_fp8_fp4_mega_moe(y,
                               saved_l1_preact,
                               l1_acts, l1_acts_sf,
                               l2_acts, l2_acts_sf,
                               shared_l1_acts, shared_l1_acts_sf,
                               shared_l2_acts, shared_l2_acts_sf,
                               l1_weights, l2_weights,
                               l1_weights_sf, l2_weights_sf,
                               shared_l1_weights, shared_l2_weights,
                               shared_l1_weights_sf, shared_l2_weights_sf,
                               locality_domain::get_balanced_sm_locality_domains(),
                               cumulative_local_expert_recv_stats,
                               sym_buffer_ptrs,
                               rank_idx, num_max_tokens_per_rank,
                               num_experts_per_rank,
                               num_shared_experts,
                               num_tokens, num_config_tokens, num_topk,
                               hidden, intermediate_hidden,
                               activation, activation_clamp, situ_beta, situ_linear_beta, fast_math,
                               route_weight_mode,
                               saved_down_unweighted,
                               saved_l1_acts,
                               saved_l1_acts_sf);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Zero the entire symmetric buffer for debug mode
    // NOTES: caller must re-copy inputs into the buffer before each kernel call
    if (deep_jit::get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}

static void bf16_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_weights,
    const torch::Tensor& l2_weights,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math,
    const int& num_ring_tokens,
    const std::optional<torch::Tensor>& saved_l1_preact,
    const std::string& route_weight_mode,
    const std::optional<torch::Tensor>& saved_h_unweighted,
    const std::optional<torch::Tensor>& saved_h_weighted,
    const std::optional<torch::Tensor>& saved_down_unweighted,
    const int& num_config_tokens,
    const std::string& combine_order_mode,
    const std::optional<torch::Tensor>& precomputed_route_counts,
    const std::optional<int>& active_pool_rows,
    const std::optional<torch::Tensor>& route_count_mismatch,
    const std::optional<torch::Tensor>& saved_x,
    const std::optional<torch::Tensor>& shared_l1_weights_opt,
    const std::optional<torch::Tensor>& shared_l2_weights_opt
) {
    // Config checks
    const auto num_tokens = static_cast<int>(y.size(0));
    DG_HOST_ASSERT(activation == "swiglu" or activation == "geglu");
    DG_HOST_ASSERT(
        route_weight_mode == "pre_down" ||
        route_weight_mode == "post_down");
    DG_HOST_ASSERT(
        combine_order_mode == "fixed_topk" ||
        combine_order_mode == "deepep" ||
        combine_order_mode == "deepep_v1");

    DG_HOST_ASSERT(shared_l1_weights_opt.has_value() == shared_l2_weights_opt.has_value());

    // Activation checks
    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

    // Tensor checks
    const auto arch_major = jit->device.get_arch_major();
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] = check_weights_layout_3d(l1_weights);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] = check_weights_layout_3d(l2_weights);
    DG_HOST_ASSERT(is_localized(l1_weights) == is_localized(l2_weights));
    DG_HOST_ASSERT(l1_weights.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(l2_weights.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(y.scalar_type() == torch::kBFloat16 || y.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(y.scalar_type() != torch::kFloat32 || combine_order_mode == "fixed_topk");
    DG_HOST_ASSERT(y.is_contiguous());
    DG_HOST_ASSERT(y.sizes() == torch::IntArrayRef({num_tokens, hidden}));
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_config_tokens >= num_tokens);
    DG_HOST_ASSERT(num_config_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);

    int num_shared_experts = 0, shared_intermediate_hidden = 0;
    torch::Tensor shared_l1_weights, shared_l2_weights;
    if (shared_l1_weights_opt.has_value()) {
        shared_l1_weights = shared_l1_weights_opt.value();
        shared_l2_weights = shared_l2_weights_opt.value();
        const auto [shared_l1_n, shared_l1_k] = check_weights_layout_2d(shared_l1_weights);
        const auto [shared_l2_n, shared_l2_k] = check_weights_layout_2d(shared_l2_weights);
        DG_HOST_ASSERT(is_localized(shared_l1_weights) == is_localized(l1_weights));
        DG_HOST_ASSERT(is_localized(shared_l2_weights) == is_localized(l2_weights));
        shared_intermediate_hidden = shared_l2_k;
        num_shared_experts = shared_intermediate_hidden / intermediate_hidden;

        DG_HOST_ASSERT(shared_intermediate_hidden % intermediate_hidden == 0);
        DG_HOST_ASSERT(shared_l1_n == shared_intermediate_hidden * 2);
        DG_HOST_ASSERT(shared_l1_k == hidden);
        DG_HOST_ASSERT(shared_l2_n == hidden);
        DG_HOST_ASSERT(shared_l1_weights.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(shared_l2_weights.scalar_type() == torch::kBFloat16);
    }

    DG_HOST_ASSERT(num_shared_experts == 0 or
        (not saved_l1_preact.has_value() and not saved_h_unweighted.has_value() and
         not saved_h_weighted.has_value() and not saved_down_unweighted.has_value() and
         not saved_x.has_value() and not precomputed_route_counts.has_value() and
         route_weight_mode == "pre_down" and combine_order_mode == "fixed_topk"));

    // Check stats counter
    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }
    DG_HOST_ASSERT(
        precomputed_route_counts.has_value() ==
        active_pool_rows.has_value());
    DG_HOST_ASSERT(
        precomputed_route_counts.has_value() ==
        route_count_mismatch.has_value());
    if (precomputed_route_counts.has_value()) {
        DG_HOST_ASSERT(
            precomputed_route_counts->scalar_type() ==
            torch::kInt);
        DG_HOST_ASSERT(
            precomputed_route_counts->numel() == num_experts);
        DG_HOST_ASSERT(precomputed_route_counts->is_contiguous());
        DG_HOST_ASSERT(
            route_count_mismatch->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(route_count_mismatch->numel() == 1);
        DG_HOST_ASSERT(route_count_mismatch->is_contiguous());
        DG_HOST_ASSERT(*active_pool_rows > 0);
    }

    // Check buffer bytes
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    const auto [expanded_num_required_bytes, slice] =
        get_symm_buffer_size_for_mega_moe_v3(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        "bf16xbf16", activation, num_ring_tokens, num_shared_experts);
    const auto num_required_bytes =
        expanded_num_required_bytes -
        static_cast<int64_t>(num_max_tokens_per_rank) *
            num_topk * sizeof(float);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    // Already registered tensors
    const auto [x, _x_sf, topk_idx, topk_weights, l1_acts, _l1_acts_sf, l2_acts, _l2_acts_sf,
                _token_src_metadata, _backward_grad_y, _backward_grad_route,
                shared_l1_acts, shared_l1_acts_sf, shared_l2_acts, shared_l2_acts_sf] = slice(sym_buffer);

    // Dispatch into different architectures
    if (arch_major == 10) {
        sm100_bf16_mega_moe(y,
                            saved_l1_preact,
                            l1_acts, l2_acts,
                            shared_l1_acts, shared_l2_acts,
                            l1_weights, l2_weights,
                            shared_l1_weights, shared_l2_weights,
                            locality_domain::get_balanced_sm_locality_domains(),
                            cumulative_local_expert_recv_stats,
                            sym_buffer_ptrs,
                            rank_idx, num_max_tokens_per_rank,
                            num_experts_per_rank,
                            num_shared_experts,
                            num_tokens, num_config_tokens, num_topk,
                            hidden, intermediate_hidden,
                            activation,
                            activation_clamp, fast_math,
                            route_weight_mode,
                            saved_h_unweighted,
                            saved_h_weighted,
                            saved_down_unweighted,
                            combine_order_mode,
                            precomputed_route_counts,
                            active_pool_rows,
                            route_count_mismatch,
                            saved_x);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Zero the entire symmetric buffer for debug mode
    // NOTES: caller must re-copy inputs into the buffer before each kernel call
    if (deep_jit::get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}

static void fp8_fp4_mega_moe_side_lora(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math,
    const int& num_ring_tokens,
    const std::optional<torch::Tensor>& saved_l1_preact,
    const std::string& route_weight_mode,
    const std::optional<torch::Tensor>& saved_down_unweighted,
    const std::optional<int>& num_config_tokens_opt,
    const torch::Tensor& saved_x,
    const torch::Tensor& saved_h_unweighted,
    const torch::Tensor& side_lora_a1,
    const torch::Tensor& side_lora_b1,
    const torch::Tensor& side_lora_a3,
    const torch::Tensor& side_lora_b3,
    const torch::Tensor& side_lora_a2,
    const torch::Tensor& side_lora_b2,
    const torch::Tensor& side_lora_l1_scratch,
    const torch::Tensor& side_lora_l2_scratch,
    const torch::Tensor& side_lora_ready,
    const float& side_lora_scale
) {
    const auto [l1_weights, l1_weights_sf] = l1_weights_tuple;
    const auto [l2_weights, l2_weights_sf] = l2_weights_tuple;

    // Config checks
    const auto num_tokens = static_cast<int>(y.size(0));
    const auto num_config_tokens =
        num_config_tokens_opt.value_or(num_tokens);
    const auto [rm, rn, rk] = recipe;
    DG_HOST_ASSERT(rm == 1 and rn == 1 and rk == 32);
    DG_HOST_ASSERT(activation == "swiglu" or activation == "geglu");
    DG_HOST_ASSERT(
        route_weight_mode == "pre_down" ||
        route_weight_mode == "post_down");

    // Activation checks
    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

    // Tensor checks
    DG_HOST_ASSERT(get_major_type_ab(l1_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(l2_weights) == cute::UMMA::Major::K);
    const auto arch_major = jit->device.get_arch_major();
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] =
        check_grouped_ab_fp8_fp4(l1_weights, cute::UMMA::Major::K, arch_major);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] =
        check_grouped_ab_fp8_fp4(l2_weights, cute::UMMA::Major::K, arch_major);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(l1_weights.is_contiguous() and l2_weights.is_contiguous());
    DG_HOST_ASSERT(num_config_tokens >= num_tokens);
    DG_HOST_ASSERT(num_config_tokens <= num_max_tokens_per_rank);

    // Check weight SF layout for UE8M0 packing, MN-major, and TMA alignment
    constexpr int kGranMN = 1, kGranK = 32;
    check_sf_layout(l1_weights_sf, intermediate_hidden * 2, hidden, kGranMN, kGranK,
                    num_experts_per_rank, true, false, torch::kInt);
    check_sf_layout(l2_weights_sf, hidden, intermediate_hidden, kGranMN, kGranK,
                    num_experts_per_rank, true, false, torch::kInt);

    // Check stats counter
    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }

    // Check buffer bytes
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    const auto num_max_pool_tokens =
        layout::get_num_max_pool_tokens(
            num_ranks, num_max_tokens_per_rank, num_topk,
            num_experts_per_rank);
    if (saved_down_unweighted.has_value()) {
        DG_HOST_ASSERT(
            saved_down_unweighted->scalar_type() ==
            torch::kBFloat16);
        DG_HOST_ASSERT(saved_down_unweighted->is_contiguous());
        DG_HOST_ASSERT(saved_down_unweighted->dim() == 2);
        DG_HOST_ASSERT(saved_down_unweighted->size(1) == hidden);
        DG_HOST_ASSERT(saved_down_unweighted->size(0) > 0);
        DG_HOST_ASSERT(
            saved_down_unweighted->size(0) <=
            num_max_pool_tokens);
    }
    const auto [expanded_num_required_bytes, slice] =
        get_symm_buffer_size_for_mega_moe_side_lora(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        "fp8xfp4", activation, num_ring_tokens);
    const auto num_required_bytes = expanded_num_required_bytes;
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    // Already registered tensors
    const auto [x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf,
                token_src_metadata, backward_grad_y, _backward_grad_route,
                _shared_l1, _shared_l1_sf, _shared_l2, _shared_l2_sf,
                side_lora_source] = slice(sym_buffer);

    // Dispatch into different architectures
    if (arch_major == 10) {
        sm100_fp8_fp4_mega_moe_side_lora_forward(y,
                               saved_l1_preact,
                               l1_acts, l1_acts_sf,
                               l2_acts, l2_acts_sf,
                               l1_weights, l2_weights,
                               l1_weights_sf, l2_weights_sf,
                               cumulative_local_expert_recv_stats,
                               sym_buffer_ptrs,
                               rank_idx, num_max_tokens_per_rank,
                               num_experts_per_rank,
                               num_tokens, num_config_tokens, num_topk,
                               hidden, intermediate_hidden,
                               activation, activation_clamp, fast_math,
                               route_weight_mode,
                               saved_down_unweighted,
                               side_lora_source,
                               saved_x,
                               saved_h_unweighted,
                               side_lora_a1, side_lora_b1,
                               side_lora_a3, side_lora_b3,
                               side_lora_a2, side_lora_b2,
                               side_lora_l1_scratch,
                               side_lora_l2_scratch,
                               side_lora_ready,
                               side_lora_scale);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Zero the entire symmetric buffer for debug mode
    // NOTES: caller must re-copy inputs into the buffer before each kernel call
    if (deep_jit::get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}


static void bf16_mega_moe_side_lora(
    const torch::Tensor& y,
    const torch::Tensor& l1_weights,
    const torch::Tensor& l2_weights,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math,
    const int& num_ring_tokens,
    const std::optional<torch::Tensor>& saved_l1_preact,
    const std::string& route_weight_mode,
    const std::optional<torch::Tensor>& saved_h_unweighted,
    const std::optional<torch::Tensor>& saved_h_weighted,
    const std::optional<torch::Tensor>& saved_down_unweighted,
    const int& num_config_tokens,
    const std::string& combine_order_mode,
    const std::optional<torch::Tensor>& precomputed_route_counts,
    const std::optional<int>& active_pool_rows,
    const std::optional<torch::Tensor>& route_count_mismatch,
    const std::optional<torch::Tensor>& saved_x,
    const std::optional<torch::Tensor>& side_lora_a1,
    const std::optional<torch::Tensor>& side_lora_b1,
    const std::optional<torch::Tensor>& side_lora_a3,
    const std::optional<torch::Tensor>& side_lora_b3,
    const std::optional<torch::Tensor>& side_lora_a2,
    const std::optional<torch::Tensor>& side_lora_b2,
    const std::optional<torch::Tensor>& side_lora_l1_scratch,
    const std::optional<torch::Tensor>& side_lora_l2_scratch,
    const std::optional<torch::Tensor>& side_lora_ready,
    const float& side_lora_scale
) {
    // Config checks
    const auto num_tokens = static_cast<int>(y.size(0));
    DG_HOST_ASSERT(activation == "swiglu" or activation == "geglu");
    DG_HOST_ASSERT(
        route_weight_mode == "pre_down" ||
        route_weight_mode == "post_down");
    DG_HOST_ASSERT(
        combine_order_mode == "fixed_topk" ||
        combine_order_mode == "deepep" ||
        combine_order_mode == "deepep_v1");

    // Activation checks
    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

    // Tensor checks
    DG_HOST_ASSERT(get_major_type_ab(l1_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(l2_weights) == cute::UMMA::Major::K);
    const auto arch_major = jit->device.get_arch_major();
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] = get_shape<3>(l1_weights);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] = get_shape<3>(l2_weights);
    DG_HOST_ASSERT(l1_weights.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(l2_weights.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(y.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(y.is_contiguous());
    DG_HOST_ASSERT(y.sizes() == torch::IntArrayRef({num_tokens, hidden}));
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_config_tokens >= num_tokens);
    DG_HOST_ASSERT(num_config_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(l1_weights.is_contiguous() and l2_weights.is_contiguous());

    // Check stats counter
    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }
    DG_HOST_ASSERT(
        precomputed_route_counts.has_value() ==
        active_pool_rows.has_value());
    DG_HOST_ASSERT(
        precomputed_route_counts.has_value() ==
        route_count_mismatch.has_value());
    if (precomputed_route_counts.has_value()) {
        DG_HOST_ASSERT(
            precomputed_route_counts->scalar_type() ==
            torch::kInt);
        DG_HOST_ASSERT(
            precomputed_route_counts->numel() == num_experts);
        DG_HOST_ASSERT(precomputed_route_counts->is_contiguous());
        DG_HOST_ASSERT(
            route_count_mismatch->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(route_count_mismatch->numel() == 1);
        DG_HOST_ASSERT(route_count_mismatch->is_contiguous());
        DG_HOST_ASSERT(*active_pool_rows > 0);
    }

    // Check buffer bytes
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    const auto [expanded_num_required_bytes, slice] =
        get_symm_buffer_size_for_mega_moe_side_lora(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        "bf16xbf16", activation, num_ring_tokens);
    DG_HOST_ASSERT(
        sym_buffer.nbytes() >=
        static_cast<size_t>(expanded_num_required_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    // Already registered tensors
    const auto [x, _x_sf, topk_idx, topk_weights, l1_acts, _l1_acts_sf,
                l2_acts, _l2_acts_sf, _token_src_metadata,
                _backward_grad_y, _backward_grad_route,
                _shared_l1, _shared_l1_sf, _shared_l2, _shared_l2_sf,
                _side_lora_source] =
        slice(sym_buffer);

    // Dispatch into different architectures
    if (arch_major == 10) {
        sm100_bf16_mega_moe_side_lora_forward(y,
                            saved_l1_preact,
                            l1_acts, l2_acts,
                            l1_weights, l2_weights,
                            cumulative_local_expert_recv_stats,
                            sym_buffer_ptrs,
                            rank_idx, num_max_tokens_per_rank,
                            num_experts_per_rank,
                            num_tokens, num_config_tokens, num_topk,
                            hidden, intermediate_hidden,
                            activation,
                            activation_clamp, fast_math,
                            route_weight_mode,
                            saved_h_unweighted,
                            saved_h_weighted,
                            saved_down_unweighted,
                            combine_order_mode,
                            precomputed_route_counts,
                            active_pool_rows,
                            route_count_mismatch,
                            saved_x,
                            side_lora_a1,
                            side_lora_b1,
                            side_lora_a3,
                            side_lora_b3,
                            side_lora_a2,
                            side_lora_b2,
                            side_lora_l1_scratch,
                            side_lora_l2_scratch,
                            side_lora_ready,
                            side_lora_scale);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Zero the entire symmetric buffer for debug mode
    // NOTES: caller must re-copy inputs into the buffer before each kernel call
    if (deep_jit::get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}


static void register_apis(pybind11::module_& m) {
    m.def("bf16_mega_moe_side_lora", &bf16_mega_moe_side_lora);
    m.def("get_symm_buffer_size_for_mega_moe_side_lora", &get_symm_buffer_size_for_mega_moe_side_lora);
    m.def("get_num_ring_tokens_for_mega_moe", &get_num_ring_tokens_for_mega_moe);
    m.def(
        "fp8_fp4_mega_moe_side_lora",
        &fp8_fp4_mega_moe_side_lora,
        py::arg("y"), py::arg("l1_weights_tuple"),
        py::arg("l2_weights_tuple"),
        py::arg("cumulative_local_expert_recv_stats"),
        py::arg("sym_buffer"), py::arg("sym_buffer_ptrs"),
        py::arg("rank_idx"), py::arg("num_max_tokens_per_rank"),
        py::arg("num_experts"), py::arg("num_topk"),
        py::arg("recipe"), py::arg("activation"),
        py::arg("activation_clamp_opt"), py::arg("fast_math"),
        py::arg("num_ring_tokens"), py::arg("saved_l1_preact"),
        py::arg("route_weight_mode"),
        py::arg("saved_down_unweighted"),
        py::arg("num_config_tokens"), py::arg("saved_x"),
        py::arg("saved_h_unweighted"), py::arg("side_lora_a1"),
        py::arg("side_lora_b1"), py::arg("side_lora_a3"),
        py::arg("side_lora_b3"), py::arg("side_lora_a2"),
        py::arg("side_lora_b2"), py::arg("side_lora_l1_scratch"),
        py::arg("side_lora_l2_scratch"), py::arg("side_lora_ready"),
        py::arg("side_lora_scale"));

    m.def("get_symm_buffer_size_for_mega_moe_v3", &get_symm_buffer_size_for_mega_moe_v3);
    m.def("get_token_alignment_for_mega_moe", &get_token_alignment_for_mega_moe);
    m.def("get_block_m_for_mega_moe", &get_block_m_for_mega_moe);
    m.def("get_symm_buffer_size_for_mega_moe", &get_symm_buffer_size_for_mega_moe);
    m.def("get_symm_buffer_size_for_mega_moe_v2",
          &get_symm_buffer_size_for_mega_moe_v2);
    m.def(
        "fp8_fp4_mega_moe", &fp8_fp4_mega_moe,
        py::arg("y"),
        py::arg("l1_weights_tuple"),
        py::arg("l2_weights_tuple"),
        py::arg("cumulative_local_expert_recv_stats"),
        py::arg("sym_buffer"),
        py::arg("sym_buffer_ptrs"),
        py::arg("rank_idx"),
        py::arg("num_max_tokens_per_rank"),
        py::arg("num_experts"),
        py::arg("num_topk"),
        py::arg("recipe"),
        py::arg("activation"),
        py::arg("activation_clamp_opt"),
        py::arg("fast_math"),
        py::arg("num_ring_tokens"),
        py::arg("saved_l1_preact") = py::none(),
        py::arg("route_weight_mode") = "pre_down",
        py::arg("saved_down_unweighted") = py::none(),
        py::arg("saved_l1_acts") = py::none(),
        py::arg("saved_l1_acts_sf") = py::none(),
        py::arg("num_config_tokens") = py::none(),
        py::arg("shared_l1_weights_tuple_opt") = py::none(),
        py::arg("shared_l2_weights_tuple_opt") = py::none(),
        py::arg("situ_beta_opt") = py::none(),
        py::arg("situ_linear_beta_opt") = py::none());
    m.def("bf16_mega_moe", &bf16_mega_moe,
        py::arg("y"),
        py::arg("l1_weights"),
        py::arg("l2_weights"),
        py::arg("cumulative_local_expert_recv_stats"),
        py::arg("sym_buffer"),
        py::arg("sym_buffer_ptrs"),
        py::arg("rank_idx"),
        py::arg("num_max_tokens_per_rank"),
        py::arg("num_experts"),
        py::arg("num_topk"),
        py::arg("activation"),
        py::arg("activation_clamp_opt"),
        py::arg("fast_math"),
        py::arg("num_ring_tokens"),
        py::arg("saved_l1_preact"),
        py::arg("route_weight_mode"),
        py::arg("saved_h_unweighted"),
        py::arg("saved_h_weighted"),
        py::arg("saved_down_unweighted"),
        py::arg("num_config_tokens"),
        py::arg("combine_order_mode"),
        py::arg("precomputed_route_counts"),
        py::arg("active_pool_rows"),
        py::arg("route_count_mismatch"),
        py::arg("saved_x"),
        py::arg("shared_l1_weights_opt") = py::none(),
        py::arg("shared_l2_weights_opt") = py::none());
}

} // namespace deep_gemm::mega
