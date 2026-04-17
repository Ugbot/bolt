// bolt/kernels/fintech/rolling_skew.h — Phase 5.W.7 Tier 2 kernel.
//
// RollingSkew: standardised 3rd central moment over a rolling window.
//
//   mean = mean(x[i-w+1 .. i])
//   m2   = (1/w) Σ (x_j - mean)²
//   m3   = (1/w) Σ (x_j - mean)³
//   skew = m3 / (sqrt(m2))³       (raw / biased / population form)
//
// Source: chukonu/fintech/kernels.h :: CreateRollingSkewProcessor (1258).
// Contract copied verbatim:
//   * POPULATION (biased) moments — denominator `w`, not `w-1`.
//   * Raw skewness (Pearson's moment coefficient), not the adjusted
//     "sample skewness" from Fisher-Pearson g_1.
//   * w < 3 → output 0.0 (not NaN). Matches chukonu.
//   * stddev guard: stddev <= 1e-10 → 0.0.
//
// Rolling-window Welford stability: higher moments (m3) are even more
// sensitive to decrement cancellation than m2, so rescan-from-ring is the
// right default here. Design-log "Rolling-window Welford — rescan vs
// decrement" covers the decision.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/fintech/state.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace bolt {
namespace fintech {

template <uint32_t kCap>
struct RollingSkewState {
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
inline void execute_rolling_skew(RollingSkewState<kCap>* state,
                                 const double* BOLT_RESTRICT x,
                                 int64_t n,
                                 double* BOLT_RESTRICT out) noexcept {
    assert(state != nullptr);
    assert(x     != nullptr || n == 0);
    assert(out   != nullptr || n == 0);
    assert(n >= 0);
    // Warmup split: chukonu emits 0.0 while ring fill < 3. We compute how
    // many warmup rows this call owes up-front and run a tiny prefix loop,
    // so the steady-state body contains no data-dependent branch.
    // Warmup count: see rolling_correlation.h for the derivation. Original
    // check is post-push `size < need`, so warmup row count is
    // max(0, need - have - 1).
    const uint32_t have   = state->ring.size();
    const uint32_t need   = 3u;
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
        assert(w >= 3u);

        double sum = 0.0;
        for (uint32_t j = 0; j < w; ++j) sum += state->ring.back(j);
        const double mean = sum / static_cast<double>(w);

        double m2 = 0.0, m3 = 0.0;
        for (uint32_t j = 0; j < w; ++j) {
            const double d  = state->ring.back(j) - mean;
            const double d2 = d * d;
            m2 += d2;
            m3 += d2 * d;
        }
        m2 /= static_cast<double>(w);
        m3 /= static_cast<double>(w);
        // Guard for the degenerate all-equal window: `stddev <= 1e-10` →
        // emit 0.0. Ternary compiles to cmov — no mispredictable branch.
        const double stddev = std::sqrt(m2);
        out[i] = (stddev > 1e-10) ? (m3 / (stddev * stddev * stddev)) : 0.0;
    }
}

}  // namespace fintech
}  // namespace bolt
