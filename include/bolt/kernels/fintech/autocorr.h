// bolt/kernels/fintech/autocorr.h — Phase 5.W.9 Tier 2 kernel.
//
// Autocorr: rolling Pearson autocorrelation between x_t and x_{t-lag} over
// a window of size `w`.
//
//   current series: x[i-w+1 .. i]
//   lagged series:  x[i-lag-w+1 .. i-lag]
//   autocorr_i    = corr(current, lagged)     (Pearson, population moments)
//
// Source: chukonu/fintech/kernels.h :: CreateAutocorrProcessor (1381).
// Contract copied verbatim:
//   * Both series are size `w`; denominator in the moments is `w`.
//   * Warm-up: rows where `i - w + 1 < 0 || i - lag - w + 1 < 0` emit 0.0
//     (not NaN). This means the first `w + lag - 1` rows emit 0.
//   * sqrt(var_curr * var_lag) <= 1e-12 → 0.0.
//
// State layout: a single RollingRing holds the last `w + lag` samples.
// back(0..w-1)       → current window, newest-first
// back(lag..lag+w-1) → lagged  window, newest-first
// The ring size is therefore `w + lag`; caller picks kCap >= w + lag at
// compile time. This is simpler (and cache-hotter) than maintaining two
// separate rings and a `lag_column` staging buffer.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/fintech/state.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace bolt {
namespace fintech {

template <uint32_t kCap>
struct AutocorrState {
    int32_t                      in_col;
    int32_t                      out_col;
    uint32_t                     window;
    uint32_t                     lag;
    uint32_t                     ring_size;     // window + lag
    RollingRing<double, kCap>    ring;

    inline void init(int32_t in, int32_t out,
                     uint32_t w, uint32_t k) noexcept {
        assert(w > 0u);
        assert(k > 0u);
        assert(w + k <= kCap);
        in_col    = in;
        out_col   = out;
        window    = w;
        lag       = k;
        ring_size = w + k;
        ring.init(w + k);
    }
};

template <uint32_t kCap>
inline void execute_autocorr(AutocorrState<kCap>* state,
                             const double* BOLT_RESTRICT x,
                             int64_t n,
                             double* BOLT_RESTRICT out) noexcept {
    assert(state != nullptr);
    assert(x     != nullptr || n == 0);
    assert(out   != nullptr || n == 0);
    assert(n >= 0);
    const uint32_t w   = state->window;
    const uint32_t lag = state->lag;
    assert(w > 0u);
    assert(lag > 0u);
    // Warmup split: chukonu emits 0.0 until ring fill reaches `w + lag`.
    // Compute the deficit up-front so the steady-state loop never branches
    // on ring size per row.
    // Warmup count: see rolling_correlation.h for the derivation. Original
    // check is post-push `size < need`, so warmup row count is
    // max(0, need - have - 1).
    const uint32_t have    = state->ring.size();
    const uint32_t need    = w + lag;
    const uint32_t warmup  = (have + 1u >= need) ? 0u : (need - have - 1u);
    const int64_t warm_n   = (n < static_cast<int64_t>(warmup))
                           ? n : static_cast<int64_t>(warmup);

    for (int64_t i = 0; i < warm_n; ++i) {
        state->ring.push(x[i]);
        out[i] = 0.0;
    }

    for (int64_t i = warm_n; i < n; ++i) {
        state->ring.push(x[i]);
        assert(state->ring.size() >= w + lag);

        // newest-first indexing into the ring:
        //   back(j)         j in [0, w)       → current window
        //   back(lag + j)   j in [0, w)       → lagged window
        double sum_curr = 0.0, sum_lag = 0.0;
        for (uint32_t j = 0; j < w; ++j) {
            sum_curr += state->ring.back(j);
            sum_lag  += state->ring.back(lag + j);
        }
        const double mean_curr = sum_curr / static_cast<double>(w);
        const double mean_lag  = sum_lag  / static_cast<double>(w);

        double cov = 0.0, var_c = 0.0, var_l = 0.0;
        for (uint32_t j = 0; j < w; ++j) {
            const double dc = state->ring.back(j)       - mean_curr;
            const double dl = state->ring.back(lag + j) - mean_lag;
            cov   += dc * dl;
            var_c += dc * dc;
            var_l += dl * dl;
        }
        // Guard: zero-variance window → 0.0 (cmov on x86).
        const double denom = std::sqrt(var_c * var_l);
        out[i] = (denom > 1e-12) ? (cov / denom) : 0.0;
    }
}

}  // namespace fintech
}  // namespace bolt
