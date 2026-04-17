// bolt/kernels/fintech/rolling_correlation.h — Phase 5.W.6 Tier 2 kernel.
//
// RollingCorrelation: Pearson correlation between two series over a
// rolling window of size `w`.
//
//   mean_a = mean(A[i-w+1..i]),  mean_b = mean(B[i-w+1..i])
//   cov    = Σ (A_j - mean_a)(B_j - mean_b)
//   var_a  = Σ (A_j - mean_a)²
//   var_b  = Σ (B_j - mean_b)²
//   corr   = cov / sqrt(var_a * var_b)
//
// Source: chukonu/fintech/kernels.h :: CreateRollingCorrelationProcessor
// (1183). Contract copied verbatim:
//   * No warm-up NaN — partial-window corr emitted from row 0.
//   * w < 2 → output 0.0 (not NaN). Matches chukonu.
//   * sqrt(var_a*var_b) <= 1e-12 → output 0.0.
//
// Rolling-window Welford stability: rescan both rings on each row —
// O(w) per row, numerically safe. Design-log "Rolling-window Welford
// — rescan vs decrement" applies to the two-variable case identically.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/fintech/state.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace bolt {
namespace fintech {

template <uint32_t kCap>
struct RollingCorrelationState {
    int32_t                      col_a;
    int32_t                      col_b;
    int32_t                      out_col;
    uint32_t                     window;
    RollingRing<double, kCap>    ring_a;
    RollingRing<double, kCap>    ring_b;

    inline void init(int32_t ca, int32_t cb, int32_t out, uint32_t w) noexcept {
        assert(w > 0u);
        assert(w <= kCap);
        col_a   = ca;
        col_b   = cb;
        out_col = out;
        window  = w;
        ring_a.init(w);
        ring_b.init(w);
    }
};

template <uint32_t kCap>
inline void execute_rolling_correlation(RollingCorrelationState<kCap>* state,
                                        const double* BOLT_RESTRICT a,
                                        const double* BOLT_RESTRICT b,
                                        int64_t n,
                                        double* BOLT_RESTRICT out) noexcept {
    assert(state != nullptr);
    assert(a     != nullptr || n == 0);
    assert(b     != nullptr || n == 0);
    assert(out   != nullptr || n == 0);
    assert(n >= 0);
    // Warmup split: chukonu emits 0.0 for rows where ring fill < 2. Rather
    // than branch per row on `w < 2u`, compute how many warmup rows this
    // call owes at entry (may be 0 if the ring is already primed from a
    // previous call), run a tiny prefix loop, then drop into a steady-state
    // loop whose per-row body is branchless.
    // Warmup count: original per-row check is `if (size_after_push < need)`.
    // After k pushes starting from `have`, size = have + k. Steady once
    // have + k >= need → k >= need - have. The FIRST steady push is
    // 1-based k = need - have; its array index is (need - have) - 1. So
    // warmup row count = max(0, need - have - 1). Saturates to 0 once the
    // ring is primed (have >= need-1) across a chained call.
    const uint32_t have   = state->ring_a.size();
    const uint32_t need   = 2u;
    const uint32_t warmup = (have + 1u >= need) ? 0u : (need - have - 1u);
    const int64_t warm_n = (n < static_cast<int64_t>(warmup))
                         ? n : static_cast<int64_t>(warmup);

    // Warmup prefix: push both rings and emit 0.0. No branch on `w<2` here.
    for (int64_t i = 0; i < warm_n; ++i) {
        state->ring_a.push(a[i]);
        state->ring_b.push(b[i]);
        out[i] = 0.0;
    }

    // Steady-state: from now on the ring is guaranteed w >= 2, so the
    // per-row body contains no data-dependent branch. The final `(denom>eps)
    // ? ... : 0.0` compiles to a cmov, which is a cycle-level select — not
    // a mispredictable control-flow branch.
    for (int64_t i = warm_n; i < n; ++i) {
        state->ring_a.push(a[i]);
        state->ring_b.push(b[i]);
        const uint32_t w = state->ring_a.size();
        assert(w >= 2u);
        assert(w == state->ring_b.size());

        double sum_a = 0.0, sum_b = 0.0;
        for (uint32_t j = 0; j < w; ++j) {
            sum_a += state->ring_a.back(j);
            sum_b += state->ring_b.back(j);
        }
        const double mean_a = sum_a / static_cast<double>(w);
        const double mean_b = sum_b / static_cast<double>(w);

        double cov = 0.0, var_a = 0.0, var_b = 0.0;
        for (uint32_t j = 0; j < w; ++j) {
            const double da = state->ring_a.back(j) - mean_a;
            const double db = state->ring_b.back(j) - mean_b;
            cov   += da * db;
            var_a += da * da;
            var_b += db * db;
        }
        const double denom = std::sqrt(var_a * var_b);
        out[i] = (denom > 1e-12) ? (cov / denom) : 0.0;
    }
}

}  // namespace fintech
}  // namespace bolt
