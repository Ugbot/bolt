// bolt/kernels/fintech/donchian_channel.h — Phase 5.R.3 Tier 2 kernel.
//
// DonchianChannel: rolling max of `high` and rolling min of `low` over a
// fixed window. Emits two output columns (upper = rolling max, lower =
// rolling min). Mid is left for a caller-side add — keeps the kernel lean.
//
//   upper[i] = NaN                   if seen < window     (warm-up)
//   upper[i] = max(high[i-w+1..i])   otherwise
//   lower[i] = NaN                   if seen < window     (warm-up)
//   lower[i] = min(low[i-w+1..i])    otherwise
//
// Source: chukonu/fintech/kernels.h :: CreateDonchianChannelProcessor (2832).
// Port shape: docs/research/gestalt-kernel-adapter.md (Tier 2 stateful).
//
// Deviation from chukonu: chukonu emits upper/lower/mid (3 columns) AND
// uses partial windows during warm-up. Bolt follows Phase 5.R spec: 2
// columns (upper, lower), NaN warm-up. O(1) amortised via monotonic deque,
// not O(w) per row like chukonu's std::deque min/max scan.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/fintech/monotonic_deque.h"

#include <cassert>
#include <cstdint>
#include <limits>

namespace bolt {
namespace fintech {

// DonchianChannelState<kCap>
// Two monotonic-index rings. kCap must be >= window.
template <uint32_t kCap>
struct DonchianChannelState {
    int32_t             high_col;
    int32_t             low_col;
    int32_t             upper_col;
    int32_t             lower_col;
    uint32_t            window;
    int64_t             row_counter;   // advances monotonically across calls
    IndexRing<kCap>     dq_max;
    IndexRing<kCap>     dq_min;

    inline void init(int32_t high, int32_t low,
                     int32_t upper, int32_t lower,
                     uint32_t w) noexcept {
        assert(w > 0u);
        assert(w <= kCap);
        high_col    = high;
        low_col     = low;
        upper_col   = upper;
        lower_col   = lower;
        window      = w;
        row_counter = 0;
        dq_max.init();
        dq_min.init();
    }
};

// Execute DonchianChannel across `n` rows.
// Inputs:  in[0] = high, in[1] = low.
// Outputs: out_upper[i] = rolling max of high, out_lower[i] = rolling min
//          of low. Warm-up rows (row_counter < window - 1) emit NaN.
template <uint32_t kCap>
inline void execute_donchian_channel(DonchianChannelState<kCap>* state,
                                     const double* BOLT_RESTRICT* in,
                                     int64_t n,
                                     double* BOLT_RESTRICT out_upper,
                                     double* BOLT_RESTRICT out_lower) noexcept {
    assert(state != nullptr);
    assert(in    != nullptr);
    assert(n >= 0);
    assert(out_upper != nullptr || n == 0);
    assert(out_lower != nullptr || n == 0);
    const double* BOLT_RESTRICT high = in[0];
    const double* BOLT_RESTRICT low  = in[1];
    assert(high != nullptr || n == 0);
    assert(low  != nullptr || n == 0);
    const int64_t w = static_cast<int64_t>(state->window);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    assert(w > 0);
    for (int64_t k = 0; k < n; ++k) {
        const int64_t i = state->row_counter;
        const int64_t min_keep = i - w + 1;
        state->dq_max.pop_front_while_stale(min_keep);
        state->dq_min.pop_front_while_stale(min_keep);
        state->dq_max.push_max(i, high[k]);
        state->dq_min.push_min(i, low[k]);
        if (i + 1 < w) {
            out_upper[k] = nan;
            out_lower[k] = nan;
        } else {
            out_upper[k] = state->dq_max.front_val();
            out_lower[k] = state->dq_min.front_val();
        }
        ++state->row_counter;
    }
}

}  // namespace fintech
}  // namespace bolt
