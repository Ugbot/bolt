// bolt/kernels/fintech/roll_spread.h — Phase 5.S.12 Tier 2 kernel.
//
// RollSpread (Roll 1984): bid-ask spread estimator derived from the
// first-order autocovariance of price changes.
//   delta_p[i]  = price[i] - price[i-1],   delta_p[0] = 0
//   cov        = population covariance of (delta_p[j], delta_p[j-1])
//                computed over the most-recent `w` pairs
//   spread     = 2 * sqrt(-cov)   if cov < 0, else 0
//
// Intuition: a two-sided dealer market produces a bid-ask bounce, which
// shows up as negatively autocorrelated price changes. A positive
// autocovariance (trending) gives no spread signal, so chukonu clamps to 0.
//
// Source: chukonu/fintech/kernels.h :: CreateRollSpreadProcessor (line 2579).
// Contract copied verbatim:
//   * No ring pushes for stream-row index < 2 (need both diff[i] and
//     diff[i-1] before the first pair is well-defined).
//   * w < 2 → output 0.0 (chukonu's `if (win_curr.size() < 2)` guard).
//   * cov >= 0 → output 0.0 (cmov guard).
//   * cov is divided by w (POPULATION, not sample — matches chukonu).
//
// State layout: two RollingRing<double, kCap> — current Δp and lagged Δp
// — plus the last price and last diff across calls (to compute diffs at
// a batch boundary) and a seen_count to gate the i<2 warmup at stream start.
//
// Branchless warmup-split: startup states (seen<2, and seen>=2 but ring
// fill < 2) are each emitted 0.0 in a tiny prefix loop that runs at most
// three times per filter lifetime. The steady-state loop has no data-
// dependent control-flow branches, only the `cov<0` cmov.

#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/bolt_port.h"
#include "bolt/kernels/fintech/state.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace bolt {
namespace fintech {

template <uint32_t kCap>
struct RollSpreadState {
    int32_t                      price_col;
    int32_t                      out_col;
    uint32_t                     window;
    double                       prev_price;  // price[i-1] across calls
    double                       prev_diff;   // Δp[i-1] across calls
    uint64_t                     seen;        // total rows fed so far
    RollingRing<double, kCap>    ring_curr;   // {Δp[i]}
    RollingRing<double, kCap>    ring_prev;   // {Δp[i-1]}

    inline void init(int32_t p, int32_t out, uint32_t w) noexcept {
        assert(w > 0u);
        assert(w <= kCap);
        price_col  = p;
        out_col    = out;
        window     = w;
        prev_price = 0.0;
        prev_diff  = 0.0;
        seen       = 0ull;
        ring_curr.init(w);
        ring_prev.init(w);
    }
};

// Warmup prefix: process a single row in "startup" mode. Runs at most
// three times per filter lifetime (seen=0, seen=1, seen=2). Always
// emits 0.0. Not inlined into the steady-state loop so the hot path
// stays branchless.
template <uint32_t kCap>
inline void roll_spread_warmup_row(RollSpreadState<kCap>* state,
                                   double px) noexcept {
    assert(state != nullptr);
    assert(state->window > 0u);
    const uint64_t idx = state->seen;
    const double diff_i = (idx >= 1ull) ? (px - state->prev_price) : 0.0;
    if (idx >= 2ull) {
        state->ring_curr.push(diff_i);
        state->ring_prev.push(state->prev_diff);
    }
    state->prev_diff  = diff_i;
    state->prev_price = px;
    ++state->seen;
}

template <uint32_t kCap>
inline void execute_roll_spread(RollSpreadState<kCap>* state,
                                const double* BOLT_RESTRICT price,
                                int64_t n,
                                double* BOLT_RESTRICT out) noexcept {
    assert(state != nullptr);
    assert(price != nullptr || n == 0);
    assert(out   != nullptr || n == 0);
    assert(n >= 0);
    // Warmup count: rows that should emit 0.0 at call entry. We are in
    // warmup until ring fill >= 2. Rows with stream-index 0, 1 do no push;
    // stream-index 2 pushes once (fill=1); stream-index 3 pushes again
    // (fill=2) → steady. So the steady state's first row is stream-index
    // 4, i.e. need_seen = 4. Deficit = max(0, 4 - seen).
    const uint64_t need_seen = 4ull;
    const uint64_t deficit   = (state->seen >= need_seen)
                             ? 0ull : (need_seen - state->seen);
    const int64_t warm_n     = (n < static_cast<int64_t>(deficit))
                             ? n : static_cast<int64_t>(deficit);

    for (int64_t i = 0; i < warm_n; ++i) {
        roll_spread_warmup_row(state, price[i]);
        out[i] = 0.0;
    }

    // Steady-state: seen >= 3 and ring fill >= 2 guaranteed. No per-row
    // startup branches; only the cov<0 cmov ternary.
    for (int64_t i = warm_n; i < n; ++i) {
        const double px     = price[i];
        const double diff_i = px - state->prev_price;
        state->ring_curr.push(diff_i);
        state->ring_prev.push(state->prev_diff);
        state->prev_diff  = diff_i;
        state->prev_price = px;
        ++state->seen;

        const uint32_t w = state->ring_curr.size();
        assert(w >= 2u);
        assert(w == state->ring_prev.size());
        double sum_c = 0.0, sum_p = 0.0;
        for (uint32_t j = 0; j < w; ++j) {
            sum_c += state->ring_curr.back(j);
            sum_p += state->ring_prev.back(j);
        }
        const double mean_c = sum_c / static_cast<double>(w);
        const double mean_p = sum_p / static_cast<double>(w);
        double cov = 0.0;
        for (uint32_t j = 0; j < w; ++j) {
            cov += (state->ring_curr.back(j) - mean_c) *
                   (state->ring_prev.back(j) - mean_p);
        }
        cov /= static_cast<double>(w);
        // cmov: cov >= 0 → 0.0. No mispredictable branch.
        out[i] = (cov < 0.0) ? 2.0 * std::sqrt(-cov) : 0.0;
    }
}

template <uint32_t kCap>
inline RollSpreadState<kCap>* make_roll_spread_state(::bolt::Arena* arena,
                                                     int32_t price_col,
                                                     int32_t out_col,
                                                     uint32_t window) noexcept {
    assert(arena != nullptr);
    assert(window > 0u);
    assert(window <= kCap);
    RollSpreadState<kCap>* s = arena->allocate_array<RollSpreadState<kCap>>(1);
    if (s == nullptr) return nullptr;
    s->init(price_col, out_col, window);
    return s;
}

}  // namespace fintech
}  // namespace bolt
