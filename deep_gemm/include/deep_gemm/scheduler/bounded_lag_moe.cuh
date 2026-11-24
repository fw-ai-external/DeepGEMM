#pragma once

#include <cstdint>
#include <cutlass/cutlass.h>

namespace deep_gemm::sched {

// Adapted from the lag enumerator in deepseek-ai/DeepGEMM PR440
// (zkyue/DeepGEMM, 8cf8180fe6f97eb65ee6638a14a21497cc1cb5e0), MIT license.
// A unit contains at most eight consecutive M blocks of one expert. After
// the initial lag, issue four L1 units then four L2 units. Unlike SM90's full
// replay table, the caller publishes each result through SM100's two slots.
CUTLASS_HOST_DEVICE constexpr uint32_t get_bounded_lag_units(
    uint32_t capacity, uint32_t topk, uint32_t experts_per_rank) {
    const uint64_t denominator = static_cast<uint64_t>(experts_per_rank) * 1024;
    const uint64_t per_expert = (static_cast<uint64_t>(capacity) * topk + denominator - 1) / denominator;
    const uint64_t units = experts_per_rank * (per_expert > 0 ? per_expert : 1);
    const uint64_t lag = (45 * units + 50) / 100;
    return lag < 8 ? 8 : (lag > 64 ? 64 : static_cast<uint32_t>(lag));
}

CUTLASS_HOST_DEVICE constexpr uint32_t get_bounded_lag_ring_blocks(uint32_t lag) {
    return 8 * (lag + 4) + 4;
}

struct BoundedLagTask {
    uint32_t phase = 0, expert = 0, m = 0, n = 0;
    uint32_t pool_block = 0, valid_m = 0;
};

template <uint32_t kNumExperts, uint32_t BLOCK_M,
          uint32_t kL1Clusters, uint32_t kL2Clusters, uint32_t kLagUnits>
struct BoundedLagCursor {
    struct Cursor {
        uint32_t expert = 0, rows = 0, pool = 0, m = 0;
    } l1, l2;
    uint32_t position = 0, previous_ticket = 0;
    uint32_t num_l1_units = 0, num_l2_pending = 0;
    bool initialized = false;

    CUTLASS_HOST_DEVICE static uint32_t blocks(uint32_t rows) {
        return (rows + BLOCK_M - 1) / BLOCK_M;
    }

    template <typename GetCount>
    CUTLASS_HOST_DEVICE void skip_empty(Cursor& c, GetCount&& count) {
        while (c.expert < kNumExperts && c.m >= blocks(c.rows)) {
            c.pool += blocks(c.rows);
            ++c.expert;
            c.m = 0;
            c.rows = c.expert < kNumExperts ? count(c.expert) : 0;
        }
    }

    // Tickets must increase within each cursor. All schedulers claim from one
    // rank-local atomic stream only AFTER a metadata slot is available. This
    // bounds outstanding tasks and keeps every data/reuse dependency earlier
    // in issue order, including when a slow CTA falls behind its peers.
    template <typename GetCount>
    CUTLASS_HOST_DEVICE BoundedLagTask next(uint32_t ticket, GetCount&& count) {
        if (!initialized) {
            l1.rows = l2.rows = count(0);
            skip_empty(l1, count);
            skip_empty(l2, count);
            initialized = true;
        }
        position += ticket - previous_ticket;
        previous_ticket = ticket;
        while (l2.expert < kNumExperts) {
            const bool is_l2 = num_l2_pending > 0 || l1.expert >= kNumExperts;
            Cursor& c = is_l2 ? l2 : l1;
            const uint32_t remaining = blocks(c.rows) - c.m;
            const uint32_t unit_m = remaining < 8 ? remaining : 8;
            const uint32_t ns = is_l2 ? kL2Clusters : kL1Clusters;
            const uint32_t chunk = unit_m * ns;
            if (position < chunk) {
                const uint32_t m = c.m + position / ns;
                const uint32_t rows = c.rows - m * BLOCK_M;
                return {is_l2 ? 2u : 1u, c.expert, m, position % ns,
                        c.pool + m, rows < BLOCK_M ? rows : BLOCK_M};
            }
            position -= chunk;
            c.m += unit_m;
            skip_empty(c, count);
            if (is_l2) {
                if (num_l2_pending > 0)
                    --num_l2_pending;
            } else {
                ++num_l1_units;
                if (num_l1_units > kLagUnits && (num_l1_units - kLagUnits) % 4 == 0)
                    num_l2_pending = 4;
            }
        }
        return {};
    }
};

} // namespace deep_gemm::sched
