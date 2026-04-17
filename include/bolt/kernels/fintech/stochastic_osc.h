// bolt/kernels/fintech/stochastic_osc.h — Phase 5.R.7 Tier 2 kernel.
//
// Stochastic Oscillator %K: close's position within the rolling high-low
// range, expressed 0..100.
//
//   K[i] = NaN                                           if warm-up
//   K[i] = 50.0                                          if range == 0
//   K[i] = 100 * (close[i] - rolling_low[i]) / range     otherwise
//   where rolling_low[i]  = min(low[i-w+1..i])
//         rolling_high[i] = max(high[i-w+1..i])
//
// Source: chukonu/fintech/kernels.h :: CreateStochasticOscProcessor (1453).
// Port shape: docs/research/gestalt-kernel-adapter.md (Tier 2 stateful).
//
// Deviations from chukonu:
//   - chukonu emits %K and %D (SMA(3) of %K). Phase 5.R spec says "emits
//     1 output column" (just %K). %D is trivially composed as SMA(K, 3)
//     by the caller; we don't bake it in.
//   - chukonu uses partial windows during warm-up; Bolt emits NaN.
//   - The zero-range degenerate case returns 50.0 (matches chukonu).
//
// Implementation reuses the DonchianChannelState substrate so the two
// monotonic deques stay in sync across advances. This keeps the kernel
// small and avoids forking another deque-eviction implementation.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/fintech/donchian_channel.h"

#include <cassert>
#include <cstdint>
#include <limits>

namespace bolt {
namespace fintech {

// StochasticOscState<kCap>
// Composes DonchianChannelState (rolling high/low) + the close column.
template <uint32_t kCap>
struct StochasticOscState {
    int32_t                         close_col;
    int32_t                         high_col;
    int32_t                         low_col;
    int32_t                         k_col;
    uint32_t                        window;
    DonchianChannelState<kCap>      donchian;

    inline void init(int32_t close, int32_t high, int32_t low,
                     int32_t k_out, uint32_t w) noexcept {
        assert(w > 0u);
        assert(w <= kCap);
        close_col = close;
        high_col  = high;
        low_col   = low;
        k_col     = k_out;
        window    = w;
        // Donchian upper/lower column indices are unused here — the
        // values come out through donchian.dq_max/dq_min.front_val().
        donchian.init(high, low, /*upper*/ -1, /*lower*/ -1, w);
    }
};

// Execute Stochastic %K across `n` rows.
// Inputs:  in[0] = close, in[1] = high, in[2] = low.
// Output:  out[i] = K value (NaN during warm-up).
template <uint32_t kCap>
inline void execute_stochastic_osc(StochasticOscState<kCap>* state,
                                   const double* BOLT_RESTRICT* in,
                                   int64_t n,
                                   double* BOLT_RESTRICT out) noexcept {
    assert(state != nullptr);
    assert(in    != nullptr);
    assert(n >= 0);
    assert(out != nullptr || n == 0);
    const double* BOLT_RESTRICT close = in[0];
    const double* BOLT_RESTRICT high  = in[1];
    const double* BOLT_RESTRICT low   = in[2];
    assert(close != nullptr || n == 0);
    assert(high  != nullptr || n == 0);
    assert(low   != nullptr || n == 0);
    const int64_t w = static_cast<int64_t>(state->window);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    assert(w > 0);
    auto& dq = state->donchian;
    for (int64_t k = 0; k < n; ++k) {
        const int64_t i = dq.row_counter;
        const int64_t min_keep = i - w + 1;
        dq.dq_max.pop_front_while_stale(min_keep);
        dq.dq_min.pop_front_while_stale(min_keep);
        dq.dq_max.push_max(i, high[k]);
        dq.dq_min.push_min(i, low[k]);
        if (i + 1 < w) {
            out[k] = nan;
        } else {
            const double hi   = dq.dq_max.front_val();
            const double lo   = dq.dq_min.front_val();
            const double rng  = hi - lo;
            out[k] = (rng > 1e-12) ? (100.0 * (close[k] - lo) / rng) : 50.0;
        }
        ++dq.row_counter;
    }
}

}  // namespace fintech
}  // namespace bolt
