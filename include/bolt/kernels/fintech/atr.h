// bolt/kernels/fintech/atr.h — Phase 5.R.4 Tier 2 kernel.
//
// ATR (Average True Range): rolling mean of the True Range over a window.
//
//   TR[0]   = high[0] - low[0]   (no previous close)
//   TR[i]   = max(high[i] - low[i],
//                 |high[i] - close[i-1]|,
//                 |low[i]  - close[i-1]|)    for i >= 1
//   ATR[i]  = NaN                            if ring.size() < window
//   ATR[i]  = mean(TR over last `window`)    otherwise
//
// Source: chukonu/fintech/kernels.h :: CreateATRProcessor (line 987).
// Port shape: docs/research/gestalt-kernel-adapter.md (Tier 2 stateful).
//
// Deviation from chukonu: chukonu uses Wilder's exponential smoothing
// (alpha = 1/period, EMA-style). Phase 5.R spec explicitly calls for
// "rolling mean of TR" — the plain-SMA contract on top of the RollingRing
// substrate. Wilder's-smoothed ATR will land as a separate EMA-based kernel
// if needed (Phase 5.E territory).

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/fintech/state.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

namespace bolt {
namespace fintech {

// ATRState<kCap>
// A ring of True-Range values + carry of the previous close for TR's
// cross-row term. kCap must be >= window.
template <uint32_t kCap>
struct ATRState {
    int32_t                       high_col;
    int32_t                       low_col;
    int32_t                       close_col;
    int32_t                       out_col;
    uint32_t                      window;
    bool                          have_prev_close;
    double                        prev_close;
    RollingRing<double, kCap>     ring_tr;

    inline void init(int32_t high, int32_t low, int32_t close, int32_t out,
                     uint32_t w) noexcept {
        assert(w > 0u);
        assert(w <= kCap);
        high_col        = high;
        low_col         = low;
        close_col       = close;
        out_col         = out;
        window          = w;
        have_prev_close = false;
        prev_close      = 0.0;
        ring_tr.init(w);
    }
};

// Execute ATR across `n` rows.
// Inputs:  in[0] = high, in[1] = low, in[2] = close.
// Output:  out[i] = rolling mean of TR[i-w+1..i], NaN during warm-up.
template <uint32_t kCap>
inline void execute_atr(ATRState<kCap>* state,
                        const double* BOLT_RESTRICT* in,
                        int64_t n,
                        double* BOLT_RESTRICT out) noexcept {
    assert(state != nullptr);
    assert(in    != nullptr);
    assert(n >= 0);
    assert(out != nullptr || n == 0);
    const double* BOLT_RESTRICT high  = in[0];
    const double* BOLT_RESTRICT low   = in[1];
    const double* BOLT_RESTRICT close = in[2];
    assert(high  != nullptr || n == 0);
    assert(low   != nullptr || n == 0);
    assert(close != nullptr || n == 0);
    const uint32_t w = state->window;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    assert(w > 0u);
    for (int64_t i = 0; i < n; ++i) {
        const double h = high[i];
        const double l = low[i];
        double tr = h - l;
        if (state->have_prev_close) {
            const double pc = state->prev_close;
            const double t1 = std::fabs(h - pc);
            const double t2 = std::fabs(l - pc);
            if (t1 > tr) tr = t1;
            if (t2 > tr) tr = t2;
        }
        state->ring_tr.push(tr);
        state->prev_close      = close[i];
        state->have_prev_close = true;
        out[i] = (state->ring_tr.size() < w) ? nan : state->ring_tr.mean();
    }
}

}  // namespace fintech
}  // namespace bolt
