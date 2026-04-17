// bolt/kernels/fintech/sma.h — Phase 5.R.1 Tier 2 kernel.
//
// SMA: simple rolling mean over a fixed window.
//   out[i] = NaN                     if ring.size() < window     (warm-up)
//   out[i] = mean of last `window`   otherwise
//
// Source: chukonu/fintech/kernels.h :: CreateSMAProcessor (line 440).
// Port shape: docs/research/gestalt-kernel-adapter.md (Tier 2 stateful).
//
// Deviation from chukonu: chukonu emits partial-window means during warm-up.
// Bolt emits NaN until the window fills — matches Phase 5.R spec and is the
// conventional technical-analysis contract (PANDAS rolling min_periods=w).

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/fintech/state.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

namespace bolt {
namespace fintech {

// SMAState<kCap>
// Pinned per-graph POD state. Caller picks kCap >= window at compile time;
// `window` is set at graph init (via init()).
template <uint32_t kCap>
struct SMAState {
    int32_t                       in_col;
    int32_t                       out_col;
    uint32_t                      window;
    RollingRing<double, kCap>     ring;

    inline void init(int32_t in, int32_t out, uint32_t w) noexcept {
        assert(w > 0u);
        assert(w <= kCap);
        in_col  = in;
        out_col = out;
        window  = w;
        ring.init(w);
    }
};

// Execute SMA across `n` rows. Input is `in[0]` (the bound `in_col`
// tracking metadata only — this free function takes raw pointers).
// Warm-up rows emit NaN until the ring fills to `window`.
template <uint32_t kCap>
inline void execute_sma(SMAState<kCap>* state,
                        const double* BOLT_RESTRICT* in,
                        int64_t n,
                        double* BOLT_RESTRICT out) noexcept {
    assert(state != nullptr);
    assert(in    != nullptr);
    assert(out   != nullptr || n == 0);
    assert(n >= 0);
    const double* BOLT_RESTRICT x = in[0];
    assert(x != nullptr || n == 0);
    const uint32_t w = state->window;
    assert(w > 0u);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (int64_t i = 0; i < n; ++i) {
        state->ring.push(x[i]);
        out[i] = (state->ring.size() < w) ? nan : state->ring.mean();
    }
}

}  // namespace fintech
}  // namespace bolt
