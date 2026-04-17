// bolt/kernels/fintech/outlier_flag_mad.h — Phase 5.S.14 Tier 2 kernel.
//
// OutlierFlagMAD: median-absolute-deviation outlier flag.
//   median       = median(window)
//   mad          = median(|x_j - median| for j in window)
//   scaled_mad   = 1.4826 * mad          (normal-distribution consistency)
//   z            = |x_i - median| / scaled_mad
//   flag         = (scaled_mad > 1e-15 && z > k) ? 1 : 0
//
// Source: chukonu/fintech/kernels.h :: CreateOutlierFlagMADProcessor
// (line 2320). Contract copied verbatim:
//   * Emits int64_t flag (not int32 as the plan one-liner claimed — we
//     match chukonu's output type so downstream arithmetic lines up).
//   * MAD scaled by 1.4826 — standard normal-distribution factor;
//     for Gaussian x, E[1.4826 · MAD] = σ.
//   * scaled_mad <= 1e-15 → flag = 0 (degenerate cluster guard).
//   * NO warmup NaN — flag emitted from row 0 with partial window.
//   * Even-length windows use mean of the two middle order statistics
//     for both median and MAD. SortedRing::median() already does this.
//
// Implementation: two SortedRings. ring_x over x, ring_d over
// absolute deviations from the current median. Because |x_j - median|
// itself depends on the current median (which changes each row), we
// do NOT incrementally maintain ring_d as a rolling sorted window.
// Instead we recompute absolute deviations by scanning ring_x.sorted[]
// each row and push them into a fresh SortedRing<double, kCap> (via
// a reinit-then-push-all pattern). O(w log w) effective per row —
// same asymptotic as chukonu's std::nth_element-per-row but without
// heap allocation.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/fintech/sorted_ring.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace bolt {
namespace fintech {

template <uint32_t kCap>
struct OutlierFlagMADState {
    int32_t                        in_col;
    int32_t                        out_col;
    uint32_t                       window;
    double                         k;        // outlier threshold (e.g. 3.0)
    SortedRing<double, kCap>       ring_x;   // rolling sorted window over x
    SortedRing<double, kCap>       ring_d;   // scratch ring for |x-median|

    inline void init(int32_t in, int32_t out,
                     uint32_t w, double kk) noexcept {
        assert(w  > 0u);
        assert(w  <= kCap);
        assert(kk >= 0.0);
        in_col  = in;
        out_col = out;
        window  = w;
        k       = kk;
        ring_x.init(w);
        ring_d.init(w);
    }
};

// Execute OutlierFlagMAD across `n` rows. Per-row body is branchless:
// the flag emission is `(scaled_mad > eps && z > k) ? 1 : 0`, which
// lowers to two vcmpsd + an AND + a bitmask-to-int — no mispredictable
// control flow. The absolute-deviation recompute loop walks ring_x's
// sorted[] contiguously (cache-hot) and performs no branching inside
// its body.
template <uint32_t kCap>
inline void execute_outlier_flag_mad(OutlierFlagMADState<kCap>* state,
                                     const double* BOLT_RESTRICT x,
                                     int64_t n,
                                     int64_t* BOLT_RESTRICT out) noexcept {
    assert(state != nullptr);
    assert(x     != nullptr || n == 0);
    assert(out   != nullptr || n == 0);
    assert(n >= 0);
    constexpr double kEpsMAD     = 1e-15;
    constexpr double kMadScaling = 1.4826;

    for (int64_t i = 0; i < n; ++i) {
        const double val = x[i];
        state->ring_x.push(val);
        const uint32_t w = state->ring_x.size();
        assert(w > 0u);

        const double median = state->ring_x.median();

        // Rebuild ring_d from absolute deviations against the current
        // median. Re-init + push each live x order-stat. No allocation.
        state->ring_d.init(state->window);
        for (uint32_t j = 0; j < w; ++j) {
            const double xj = state->ring_x.order(j);
            const double d  = xj - median;
            state->ring_d.push(d < 0.0 ? -d : d);
        }
        const double mad        = state->ring_d.median();
        const double scaled_mad = kMadScaling * mad;

        // Branchless flag: two comparisons, bitwise-AND, cast to int64.
        const double absdev = (val - median);
        const double absd   = absdev < 0.0 ? -absdev : absdev;
        // z is only defined when scaled_mad > eps; we compute it
        // unconditionally and gate via the scaled_mad mask. When
        // scaled_mad ≈ 0 the division may produce inf, but the mask
        // zeros the flag regardless so no spurious 1 escapes.
        const double safe_denom = (scaled_mad > kEpsMAD) ? scaled_mad : 1.0;
        const double z          = absd / safe_denom;
        const bool   mad_ok     = (scaled_mad > kEpsMAD);
        const bool   big_z      = (z > state->k);
        out[i] = (mad_ok && big_z) ? int64_t{1} : int64_t{0};
    }
}

}  // namespace fintech
}  // namespace bolt
