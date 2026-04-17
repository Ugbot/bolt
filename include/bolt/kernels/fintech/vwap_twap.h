// bolt/kernels/fintech/vwap_twap.h — Phase 5.R.5 Tier 2 kernels.
//
// Two rolling-window kernels in one header:
//
//   VWAPState<kCap> / execute_vwap_rolling   — volume-weighted avg price.
//       Σ(price·vol) / Σ(vol) over last `window` samples. NaN warm-up.
//       NaN if Σ(vol) == 0 inside the window (guards against all-zero
//       volume windows without propagating inf/NaN via div-by-zero).
//
//   TWAPState<kCap> / execute_twap_rolling   — time-weighted avg price.
//       Simple mean of price over last `window`. Wrapper around the same
//       RollingRing shape as SMA but kept as its own kernel so the caller
//       contract (TWAP vs generic SMA) is explicit at the call site.
//
// Source: chukonu/fintech/kernels.h ::
//   CreateVWAPBatchProcessor (254) — BATCH aggregate (whole batch → one scalar)
//   CreateTWAPBatchProcessor (295) — BATCH aggregate (whole batch → one scalar)
//
// Deviation from chukonu: chukonu's *Batch processors emit one scalar per
// batch, fill-replicated across all rows. Phase 5.R spec calls for per-
// ROW rolling VWAP/TWAP over `window`. This is the semantically-useful
// rolling shape; the batch-aggregate shape falls out as `window == n`.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/fintech/state.h"

#include <cassert>
#include <cstdint>
#include <limits>

namespace bolt {
namespace fintech {

// ------------------------------------------------------------------
// VWAP (rolling)
// ------------------------------------------------------------------
template <uint32_t kCap>
struct VWAPState {
    int32_t                       price_col;
    int32_t                       volume_col;
    int32_t                       out_col;
    uint32_t                      window;
    RollingRing<double, kCap>     ring_pv;   // price * volume
    RollingRing<double, kCap>     ring_v;    // volume

    inline void init(int32_t price, int32_t volume, int32_t out,
                     uint32_t w) noexcept {
        assert(w > 0u);
        assert(w <= kCap);
        price_col  = price;
        volume_col = volume;
        out_col    = out;
        window     = w;
        ring_pv.init(w);
        ring_v.init(w);
    }
};

// Execute rolling VWAP across `n` rows.
// Inputs:  in[0] = price, in[1] = volume.
// Output:  out[i] = NaN                     if ring.size() < window
//          out[i] = NaN                     if Σ(vol) == 0 in the window
//          out[i] = Σ(p·v)/Σ(v)             otherwise
template <uint32_t kCap>
inline void execute_vwap_rolling(VWAPState<kCap>* state,
                                 const double* BOLT_RESTRICT* in,
                                 int64_t n,
                                 double* BOLT_RESTRICT out) noexcept {
    assert(state != nullptr);
    assert(in    != nullptr);
    assert(n >= 0);
    assert(out != nullptr || n == 0);
    const double* BOLT_RESTRICT price  = in[0];
    const double* BOLT_RESTRICT volume = in[1];
    assert(price  != nullptr || n == 0);
    assert(volume != nullptr || n == 0);
    const uint32_t w = state->window;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    assert(w > 0u);
    for (int64_t i = 0; i < n; ++i) {
        const double p = price[i];
        const double v = volume[i];
        state->ring_pv.push(p * v);
        state->ring_v.push(v);
        if (state->ring_v.size() < w) {
            out[i] = nan;
            continue;
        }
        const double sum_pv = static_cast<double>(state->ring_pv.sum);
        const double sum_v  = static_cast<double>(state->ring_v.sum);
        // Guard against zero-volume windows. We materialise a safe divisor
        // so MSVC's static analyser sees no possible zero-divide (avoids
        // spurious C4723) without branching in the hot path.
        const double safe_v = (sum_v != 0.0) ? sum_v : 1.0;
        const double div    = sum_pv / safe_v;
        out[i] = (sum_v != 0.0) ? div : nan;
    }
}

// ------------------------------------------------------------------
// TWAP (rolling) — same shape as SMA, distinct type for clarity.
// ------------------------------------------------------------------
template <uint32_t kCap>
struct TWAPState {
    int32_t                       price_col;
    int32_t                       out_col;
    uint32_t                      window;
    RollingRing<double, kCap>     ring;

    inline void init(int32_t price, int32_t out, uint32_t w) noexcept {
        assert(w > 0u);
        assert(w <= kCap);
        price_col = price;
        out_col   = out;
        window    = w;
        ring.init(w);
    }
};

// Execute rolling TWAP across `n` rows.
// Inputs:  in[0] = price.
// Output:  out[i] = NaN                  if ring.size() < window
//          out[i] = mean of last window  otherwise
template <uint32_t kCap>
inline void execute_twap_rolling(TWAPState<kCap>* state,
                                 const double* BOLT_RESTRICT* in,
                                 int64_t n,
                                 double* BOLT_RESTRICT out) noexcept {
    assert(state != nullptr);
    assert(in    != nullptr);
    assert(n >= 0);
    assert(out != nullptr || n == 0);
    const double* BOLT_RESTRICT price = in[0];
    assert(price != nullptr || n == 0);
    const uint32_t w = state->window;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    assert(w > 0u);
    for (int64_t i = 0; i < n; ++i) {
        state->ring.push(price[i]);
        out[i] = (state->ring.size() < w) ? nan : state->ring.mean();
    }
}

}  // namespace fintech
}  // namespace bolt
