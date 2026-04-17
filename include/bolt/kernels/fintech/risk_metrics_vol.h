// bolt/kernels/fintech/risk_metrics_vol.h — Phase 5.W.10 Tier 2 kernel.
//
// RiskMetricsVol: JP Morgan RiskMetrics'94 exponentially-weighted volatility
// over a stream of returns.
//
//   var_0   = r_0²
//   var_t   = λ · var_{t-1} + (1 - λ) · r_t²       for t >= 1
//   out[t]  = sqrt(var_t)                          (volatility, not variance)
//
// Source: chukonu/fintech/kernels.h :: CreateRiskMetricsVolProcessor (2730).
// Contract copied verbatim:
//   * Seed at row 0 with var = r₀², output |r₀|.
//   * λ is typically 0.94 for daily returns (RiskMetrics'94).
//   * Output is stddev (sqrt of variance) per row, NOT variance.
//
// Uses EmaState as the substrate via a direct recurrence on r². We don't
// route through EmaState.update() because the seed is `r₀²` (not `r_0`)
// and the recurrence uses `λ` (not `1-α = λ`). Inlining the recurrence is
// cleaner than abusing EmaState semantics, but the state IS an EmaState
// shape — matching the Phase 2 P5.W.10 row in BOLT_FINTECH_PORT.md.
//
// Depends on: P2.3 EmaState (substrate), P1.4 sum_of_squares (conceptual).

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/fintech/state.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace bolt {
namespace fintech {

struct RiskMetricsVolState {
    int32_t   in_col;
    int32_t   out_col;
    double    lambda;
    EmaState  ema;        // holds (alpha=1-λ, value=var_t, initialized)

    inline void init(int32_t in, int32_t out, double lam) noexcept {
        assert(lam >= 0.0);
        assert(lam <= 1.0);
        in_col  = in;
        out_col = out;
        lambda  = lam;
        // alpha = 1 - λ, value seeded on first sample; we drive the
        // recurrence manually since the seed is r₀², not r₀.
        ema.alpha       = 1.0 - lam;
        ema.value       = 0.0;
        ema.initialized = false;
    }
};

inline void execute_risk_metrics_vol(RiskMetricsVolState* state,
                                     const double* BOLT_RESTRICT r,
                                     int64_t n,
                                     double* BOLT_RESTRICT out) noexcept {
    assert(state != nullptr);
    assert(r     != nullptr || n == 0);
    assert(out   != nullptr || n == 0);
    assert(n >= 0);
    if (n == 0) return;
    const double lam = state->lambda;
    assert(lam >= 0.0 && lam <= 1.0);

    double var_t;
    int64_t i0 = 0;
    if (!state->ema.initialized) {
        var_t = r[0] * r[0];
        out[0] = std::sqrt(var_t);
        state->ema.value       = var_t;
        state->ema.initialized = true;
        i0 = 1;
    } else {
        var_t = state->ema.value;
    }

    for (int64_t i = i0; i < n; ++i) {
        const double ri = r[i];
        var_t = lam * var_t + (1.0 - lam) * ri * ri;
        out[i] = std::sqrt(var_t);
    }
    state->ema.value = var_t;
}

}  // namespace fintech
}  // namespace bolt
