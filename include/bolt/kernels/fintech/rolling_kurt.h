// bolt/kernels/fintech/rolling_kurt.h — Phase 5.W.8 Tier 2 kernel.
//
// RollingKurt: standardised 4th central moment over a rolling window,
// expressed as EXCESS kurtosis (subtract 3 so a Gaussian input → 0).
//
//   mean = mean(x[i-w+1 .. i])
//   m2   = (1/w) Σ (x_j - mean)²        (population variance)
//   m4   = (1/w) Σ (x_j - mean)⁴
//   kurt = m4 / m2² - 3                 (Pearson excess kurtosis)
//
// Source: chukonu/fintech/kernels.h :: CreateRollingKurtProcessor (1319).
// Contract copied verbatim:
//   * POPULATION (biased) moments — denominator `w`, not `w-1`.
//   * EXCESS kurtosis (the `- 3` subtraction is applied). This is the
//     industry-standard convention for financial time series where
//     the Gaussian baseline is the reference.
//   * w < 4 → output 0.0 (not NaN).
//   * variance guard: var <= 1e-10 → 0.0.
//
// Rolling-window Welford stability: as with skew, the 4th moment is
// highly sensitive to decrement cancellation — rescan-from-ring is the
// numerically safe default. Design-log "Rolling-window Welford —
// rescan vs decrement".

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/fintech/state.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace bolt {
namespace fintech {

template <uint32_t kCap>
struct RollingKurtState {
    int32_t                      in_col;
    int32_t                      out_col;
    uint32_t                     window;
    RollingRing<double, kCap>    ring;

    inline void init(int32_t in, int32_t out, uint32_t w) noexcept {
        assert(w > 0u);
        assert(w <= kCap);
        in_col  = in;
        out_col = out;
        window  = w;
        ring.init(w);
    }
};

template <uint32_t kCap>
inline void execute_rolling_kurt(RollingKurtState<kCap>* state,
                                 const double* BOLT_RESTRICT x,
                                 int64_t n,
                                 double* BOLT_RESTRICT out) noexcept {
    assert(state != nullptr);
    assert(x     != nullptr || n == 0);
    assert(out   != nullptr || n == 0);
    assert(n >= 0);
    // Warmup split: chukonu emits 0.0 while ring fill < 4. Hoist the
    // warmup into a tiny prefix loop so the steady-state body is branchless.
    // Warmup count: see rolling_correlation.h for the derivation. Original
    // check is post-push `size < need`, so warmup row count is
    // max(0, need - have - 1).
    const uint32_t have   = state->ring.size();
    const uint32_t need   = 4u;
    const uint32_t warmup = (have + 1u >= need) ? 0u : (need - have - 1u);
    const int64_t warm_n = (n < static_cast<int64_t>(warmup))
                         ? n : static_cast<int64_t>(warmup);

    for (int64_t i = 0; i < warm_n; ++i) {
        state->ring.push(x[i]);
        out[i] = 0.0;
    }

    for (int64_t i = warm_n; i < n; ++i) {
        state->ring.push(x[i]);
        const uint32_t w = state->ring.size();
        assert(w >= 4u);

        double sum = 0.0;
        for (uint32_t j = 0; j < w; ++j) sum += state->ring.back(j);
        const double mean = sum / static_cast<double>(w);

        double m2 = 0.0, m4 = 0.0;
        for (uint32_t j = 0; j < w; ++j) {
            const double d  = state->ring.back(j) - mean;
            const double d2 = d * d;
            m2 += d2;
            m4 += d2 * d2;
        }
        m2 /= static_cast<double>(w);
        m4 /= static_cast<double>(w);
        // Degenerate-variance guard: ternary → cmov.
        const double var = m2;
        out[i] = (var > 1e-10) ? ((m4 / (var * var)) - 3.0) : 0.0;
    }
}

}  // namespace fintech
}  // namespace bolt
