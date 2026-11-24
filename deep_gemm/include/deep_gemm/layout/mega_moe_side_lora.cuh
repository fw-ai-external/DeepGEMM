#pragma once

#include <deep_gemm/layout/mega_moe.cuh>

namespace deep_gemm::layout {

// Side-LoRA's phase scheduler uses an arrival count, not the ordinary
// scheduler's per-N-tile mask. Reuse the low word of each reserved L2 signal
// slot, preserving the upstream workspace and token-buffer offsets.
// A symmetric buffer has one in-flight forward generation: the two scheduler
// protocols must never execute concurrently on the same buffer.
struct SideLoraWorkspace : Workspace {
    using Workspace::Workspace;

    CUTLASS_DEVICE uint32_t* get_l2_full_count_ptr(const uint32_t ring_block_idx) const {
        return reinterpret_cast<uint32_t*>(get_l2_full_mask_ptr(ring_block_idx));
    }
};

}  // namespace deep_gemm::layout
