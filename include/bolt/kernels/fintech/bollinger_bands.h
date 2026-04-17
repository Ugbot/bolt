// bolt/kernels/fintech/bollinger_bands.h — Phase 5.R.2 Tier 2 kernel.
//
// Bollinger Bands: rolling mean ± k·stddev over a fixed window. Emits three
// output columns:
//   middle = rolling mean
//   upper  = middle + k · stddev
//   lower  = middle - k · stddev
//
// During warm-up (ring.size() < window) every output is NaN.
//
// Source: chukonu/fintech/kernels.h :: CreateBollingerBandsProcessor (line 566).
// Port shape: docs/research/gestalt-kernel-adapter.md (Tier 2 stateful).
//
// Numerical note: chukonu recomputes mean and stddev by scanning the window
// each row (O(w)). Bolt uses an O(1) amortised scheme — two RollingRings,
// one over x and one over x². Population variance = E[x²] - E[x]², then
// clamped at 0 to suppress cancellation-induced negatives. This mirrors
// chukonu's "population stddev, divide by N" convention (not Bessel).
// Drift bounded by window length × typical signal magnitude; acceptable
// for TA-indicator use. Welford is unsuitable here because it cannot
// undo-sample on eviction.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/fintech/state.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

namespace bolt {
namespace fintech {

// BollingerBandsState<kCap>
// Two rings (x and x²) + static parameters. kCap must be >= window.
template <uint32_t kCap>
struct BollingerBandsState {
    int32_t                       in_col;
    int32_t                       upper_col;
    int32_t                       middle_col;
    int32_t                       lower_col;
    uint32_t                      window;
    double                        k;       // typical: 2.0
    RollingRing<double, kCap>     ring_x;
    RollingRing<double, kCap>     ring_xx; // x²

    inline void init(int32_t in,
                     int32_t upper, int32_t middle, int32_t lower,
                     uint32_t w, double k_sigma) noexcept {
        assert(w > 0u);
        assert(w <= kCap);
        in_col     = in;
        upper_col  = upper;
        middle_col = middle;
        lower_col  = lower;
        window     = w;
        k          = k_sigma;
        ring_x.init(w);
        ring_xx.init(w);
    }
};

// Execute BollingerBands across `n` rows.
// Inputs:  in[0] = value column.
// Outputs: out_upper[i], out_middle[i], out_lower[i] — three buffers, caller
//          owns all three. Warm-up rows emit NaN across the board.
template <uint32_t kCap>
inline void execute_bollinger_bands(BollingerBandsState<kCap>* state,
                                    const double* BOLT_RESTRICT* in,
                                    int64_t n,
                                    double* BOLT_RESTRICT out_upper,
                                    double* BOLT_RESTRICT out_middle,
                                    double* BOLT_RESTRICT out_lower) noexcept {
    assert(state != nullptr);
    assert(in    != nullptr);
    assert(n >= 0);
    assert(out_upper  != nullptr || n == 0);
    assert(out_middle != nullptr || n == 0);
    assert(out_lower  != nullptr || n == 0);
    const double* BOLT_RESTRICT x = in[0];
    assert(x != nullptr || n == 0);
    const uint32_t w = state->window;
    const double k = state->k;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    assert(w > 0u);
    for (int64_t i = 0; i < n; ++i) {
        const double v = x[i];
        state->ring_x.push(v);
        state->ring_xx.push(v * v);
        if (state->ring_x.size() < w) {
            out_upper[i]  = nan;
            out_middle[i] = nan;
            out_lower[i]  = nan;
            continue;
        }
        const double mean = state->ring_x.mean();
        const double mean_sq = state->ring_xx.mean();
        double var = mean_sq - mean * mean;
        if (var < 0.0) var = 0.0;             // cancellation guard
        const double sd = std::sqrt(var);
        out_middle[i] = mean;
        out_upper[i]  = mean + k * sd;
        out_lower[i]  = mean - k * sd;
    }
}

}  // namespace fintech
}  // namespace bolt
