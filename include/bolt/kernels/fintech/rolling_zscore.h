// bolt/kernels/fintech/rolling_zscore.h — Phase 5.W.3 Tier 2 kernel.
//
// RollingZScore: `(x - rolling_mean) / rolling_stddev` over a window.
//
// Source: chukonu/fintech/kernels.h :: CreateRollingZScoreProcessor (847).
// Contract copied verbatim:
//   * POPULATION stddev (sqrt(m2/count)).
//   * NO warm-up NaN — partial-window z-score emitted from row 0.
//   * When `stddev == 0.0` (constant window) → output 0.0  (NOT NaN).
//     This is the chukonu convention; it differs from the naive 0/0 → NaN
//     interpretation. Matched to preserve behaviour parity.
//
// Rolling-window Welford stability: rescan-from-ring, same as RollingStd —
// see rolling_std.h header + design-log "Rolling-window Welford — rescan".

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/fintech/rolling_std.h"   // welford_rescan_pop
#include "bolt/kernels/fintech/state.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace bolt {
namespace fintech {

// Pinned per-graph POD state.
template <uint32_t kCap>
struct RollingZScoreState {
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
inline void execute_rolling_zscore(RollingZScoreState<kCap>* state,
                                   const double* BOLT_RESTRICT x,
                                   int64_t n,
                                   double* BOLT_RESTRICT out) noexcept {
    assert(state != nullptr);
    assert(x     != nullptr || n == 0);
    assert(out   != nullptr || n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) {
        state->ring.push(x[i]);
        double mean, var;
        welford_rescan_pop(state->ring, &mean, &var);
        const double stddev = std::sqrt(var);
        out[i] = (stddev > 0.0) ? ((x[i] - mean) / stddev) : 0.0;
    }
}

}  // namespace fintech
}  // namespace bolt
