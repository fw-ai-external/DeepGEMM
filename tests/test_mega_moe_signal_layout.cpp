#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>

#define DG_HOST_ASSERT(condition) \
    do { if (!(condition)) throw std::runtime_error(#condition); } while (0)
#include <deep_gemm/layout/mega_moe_side_lora.cuh>

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

static uint64_t aligned(uint64_t value) {
    return (value + 127) / 128 * 128;
}

static void check(uint32_t capacity, uint32_t ring_tokens) {
    using namespace deep_gemm::layout;
    const uint64_t ring_count = (uint64_t(ring_tokens) + 7) / 8;
    const uint64_t shared_count = (uint64_t(capacity) + 7) / 8;
    const uint64_t prefix = aligned(offsetof(MegaMoESignals, expert_recv_count_sum)
                                   + sizeof(MegaMoESignals::expert_recv_count_sum));
    const uint64_t stride = aligned(ring_count * sizeof(uint32_t));
    const uint64_t signal_end = prefix + 5 * stride + aligned(shared_count * sizeof(uint32_t));
    const uint64_t allocation = signal_end + 128;
    std::unique_ptr<void, decltype(&std::free)> storage(std::aligned_alloc(128, allocation), &std::free);
    require(storage != nullptr, "host allocation failed");
    std::memset(storage.get(), 0xa5, allocation);
    Workspace workspace(storage.get(), 8, 896, capacity, 16, ring_tokens);
    SideLoraWorkspace side(storage.get(), 8, 896, capacity, 16, ring_tokens);
    const auto base = reinterpret_cast<uintptr_t>(storage.get());
    auto offset = [&](const void* pointer) { return reinterpret_cast<uintptr_t>(pointer) - base; };
    require(offset(workspace.get_combine_ready_grid_idx_ptr()) == 128, "control prefix moved");
    require(offset(workspace.get_l1_full_count_ptr()) == prefix, "L1 full starts inside control prefix");
    require(offset(workspace.get_l1_empty_count_ptr()) == prefix + stride, "L1 arrays overlap or overallocate");
    require(offset(workspace.get_l2_full_mask_ptr()) == prefix + 2 * stride, "L2 mask offset mismatch");
    require(offset(workspace.get_l2_empty_count_ptr()) == prefix + 4 * stride, "L2 mask/empty overlap");
    require(offset(workspace.get_shared_l2_full_count_ptr()) == prefix + 5 * stride, "shared signal overlap");
    require(offset(workspace.get_src_token_topk_idx_ptr()) == signal_end, "metadata overlaps signals");
    const uint64_t metadata_bytes = uint64_t(896) * capacity * sizeof(uint32_t);
    require(offset(workspace.get_token_src_metadata_ptr()) == signal_end + metadata_bytes,
            "token metadata offset mismatch");
    const uint64_t end = signal_end + metadata_bytes + uint64_t(workspace.num_max_pool_tokens) * sizeof(TokenSrcMetadata);
    require(workspace.get_num_bytes() == (end + 15) / 16 * 16, "workspace allocation/offset mismatch");
    require(offset(workspace.get_end_ptr()) == workspace.get_num_bytes(), "workspace end mismatch");

    for (uint32_t i = 0; i < ring_count; ++i) {
        *workspace.get_l1_full_count_ptr(i) = 0x10000000u + i;
        *workspace.get_l1_empty_count_ptr(i) = 0x20000000u + i;
        *workspace.get_l2_full_mask_ptr(i) = 0x3456789000000000ull + i;
        *workspace.get_l2_empty_count_ptr(i) = 0x40000000u + i;
    }
    for (uint32_t i = 0; i < shared_count; ++i)
        *workspace.get_shared_l2_full_count_ptr(i) = 0x50000000u + i;
    for (uint32_t i = 0; i < ring_count; ++i) {
        require(*workspace.get_l1_full_count_ptr(i) == 0x10000000u + i, "L1 full corrupted");
        require(*workspace.get_l1_empty_count_ptr(i) == 0x20000000u + i, "L1 empty corrupted");
        require(*workspace.get_l2_full_mask_ptr(i) == 0x3456789000000000ull + i, "L2 mask corrupted");
        require(*workspace.get_l2_empty_count_ptr(i) == 0x40000000u + i, "L2 empty corrupted");
        require(offset(side.get_l2_full_count_ptr(i)) == offset(workspace.get_l2_full_mask_ptr(i)),
                "side-LoRA count does not alias low mask word");
        *side.get_l2_full_count_ptr(i) = 0x60000000u + i;
        uint64_t mask = 0;
        std::memcpy(&mask, workspace.get_l2_full_mask_ptr(i), sizeof(mask));
        require(mask == 0x3456789060000000ull + i, "side-LoRA count corrupted upper mask word");
    }
    for (uint32_t i = 0; i < shared_count; ++i)
        require(*workspace.get_shared_l2_full_count_ptr(i) == 0x50000000u + i, "shared signal corrupted");
    const auto bytes = static_cast<const uint8_t*>(storage.get());
    require(std::all_of(bytes, bytes + prefix, [](uint8_t x) { return x == 0xa5; }), "control prefix overwritten");
    require(std::all_of(bytes + signal_end, bytes + allocation, [](uint8_t x) { return x == 0xa5; }),
            "metadata guard overwritten");
}

int main() {
    try {
        for (uint32_t capacity : {0u, 1u, 7u, 8u, 9u, 4096u, 131072u, 262144u, 263040u})
            for (uint32_t ring : {0u, 1u, 8u, 9u, 105600u, 132480u})
                check(capacity, ring);
        std::cout << "54 workspace layouts passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
