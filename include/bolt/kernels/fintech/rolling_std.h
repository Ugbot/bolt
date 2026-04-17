// bolt/kernels/fintech/rolling_std.h — Phase 5.W.2 Tier 2 kernel.
//
// RollingStd: rolling standard deviation over a fixed window.
//   out[i] = sqrt(pop_variance(x[max(0, i-w+1) .. i]))
//
// Source: chukonu/fintech/kernels.h :: CreateRollingStdProcessor (3481).
// Contract copied verbatim:
//   * POPULATION variance (m2 / count), not sample (m2 / count-1).
//   * NO warm-up NaN — partial-window values emit sqrt(pop_var) from row 0.
//   * Variance floor: negative variance clamped to 0.0 to guard FP underflow
//     when all window samples are equal.
//
// Rolling-window Welford stability note:
//   Welford's online algorithm is append-only-stable. For a true rolling
//   window (add + drop) the decrement introduces cancellation. We instead
//   **rescan the RollingRing on each update** — O(w) per row, numerically
//   safe, and cache-hot for the small windows typical of rolling-std
//   (w <= 256). See design-log entry "Rolling-window Welford — rescan vs
//   decrement" (Phase 5.W).

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/fintech/state.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace bolt {
namespace fintech {

// Pinned per-graph POD state. kCap is the compile-time max window size.
template <uint32_t kCap>
struct RollingStdState {
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

// Rescan the ring for (mean, pop_variance). Separated so execute_rolling_std,
// execute_rolling_zscore, etc. can share the rescan path.
template <uint32_t kCap>
inline void welford_rescan_pop(const RollingRing<double, kCap>& r,
                               double* BOLT_RESTRICT out_mean,
                               double* BOLT_RESTRICT out_var) noexcept {
    assert(out_mean != nullptr);
    assert(out_var  != nullptr);
    const uint32_t n = r.size();
    if (n == 0u) { *out_mean = 0.0; *out_var = 0.0; return; }
    double sum = 0.0;
    for (uint32_t i = 0; i < n; ++i) sum += r.back(i);
    const double m = sum / static_cast<double>(n);
    double acc = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        const double d = r.back(i) - m;
        acc += d * d;
    }
    // FP-underflow guard, mirrors chukonu. Ternary → cmov on x86, csel on
    // ARM. No data-dependent branch on the hot path.
    const double raw_v = acc / static_cast<double>(n);
    const double v = (raw_v > 0.0) ? raw_v : 0.0;
    *out_mean = m;
    *out_var  = v;
}

// Execute RollingStd across `n` rows. Partial windows emit partial-window
// stddev (no NaN warmup — matches chukonu).
template <uint32_t kCap>
inline void execute_rolling_std(RollingStdState<kCap>* state,
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
        out[i] = std::sqrt(var);
    }
}

}  // namespace fintech
}  // namespace bolt
