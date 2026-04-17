// bolt/kernels/fintech/welford_mean_var.h — Phase 5.W.1 Tier 2 kernel.
//
// WelfordMeanVar: streaming (non-rolling) running mean and variance. Emits
// two output columns per row — the running `(mean, var)` after observing
// samples [0..i].
//
//   out_mean[i] = mean of x[0..i]
//   out_var[i]  = var_pop  of x[0..i]           (chukonu convention: m2/count)
//
// Source: chukonu/fintech/kernels.h :: CreateWelfordMeanVarProcessor (3040).
// Contract: chukonu uses POPULATION variance (m2 / count) for every row,
// including row 0 (var=0). Matched verbatim — see BOLT_FINTECH_PORT.md P5.W.1.
//
// Numerical stability: Welford's online algorithm is unconditionally stable
// for the *append-only* stream (which this kernel is). No cancellation risk.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/fintech/state.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace fintech {

// Pinned per-graph POD state. `in_col`, `out_mean_col`, `out_var_col` are
// schema indices kept only as metadata (the execute function takes raw
// pointers). `acc` is the single streaming Welford accumulator.
struct WelfordMeanVarState {
    int32_t             in_col;
    int32_t             out_mean_col;
    int32_t             out_var_col;
    WelfordAccumulator  acc;

    inline void init(int32_t in, int32_t out_mean, int32_t out_var) noexcept {
        assert(in >= -1);
        assert(out_mean >= -1);
        in_col       = in;
        out_mean_col = out_mean;
        out_var_col  = out_var;
        acc.init();
    }
};

// Execute WelfordMeanVar across `n` rows. Two output buffers — `out_mean`
// and `out_var`. No warm-up sentinel (chukonu emits row-0 as `(x0, 0.0)`).
inline void execute_welford_mean_var(WelfordMeanVarState* state,
                                     const double* BOLT_RESTRICT x,
                                     int64_t n,
                                     double* BOLT_RESTRICT out_mean,
                                     double* BOLT_RESTRICT out_var) noexcept {
    assert(state    != nullptr);
    assert(x        != nullptr || n == 0);
    assert(out_mean != nullptr || n == 0);
    assert(out_var  != nullptr || n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) {
        state->acc.update(x[i]);
        out_mean[i] = state->acc.mean;
        // population variance: m2 / count. count >= 1 here (just updated).
        out_var[i] = state->acc.m2 / static_cast<double>(state->acc.count);
    }
}

}  // namespace fintech
}  // namespace bolt
