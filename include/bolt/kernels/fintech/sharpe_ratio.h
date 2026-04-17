// bolt/kernels/fintech/sharpe_ratio.h — Phase 5.W.4 Tier 2 kernel.
//
// SharpeRatio: scalar-broadcast Sharpe ratio over a batch of per-period
// returns.
//
//   rf_per_period = risk_free_rate / annualization_periods
//   mean, stddev  = Welford(returns[0..n))    (SAMPLE variance, m2/(n-1))
//   sharpe = (mean - rf_per_period) / stddev * sqrt(annualization_periods)
//
// Output shape: **scalar broadcast** — every row in the output column holds
// the same Sharpe value computed over the whole input. Matches chukonu
// CreateSharpeRatioProcessor (line 1040) exactly.
//
// Contract copied verbatim:
//   * SAMPLE variance (n-1) for the denominator.
//   * Annualization factor: `sqrt(annualization_periods)` (chukonu default 252).
//   * Numerical guard: `stddev > 1e-12` — below this, emit 0.0 (not NaN).
//   * Warm-up: none — the scalar applies to every row, including row 0.
//   * Risk-free rate is pre-divided by annualization_periods inside the fn.
//
// Note: Sharpe is fundamentally a whole-batch statistic. A rolling variant
// (per-window Sharpe) is a separate kernel not covered here.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/fintech/state.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace bolt {
namespace fintech {

// Pinned per-graph POD state. Parameters are set at init() and stay constant
// across the graph's lifetime. `acc` is reset per execute call (each batch
// is an independent Sharpe computation).
struct SharpeRatioState {
    int32_t             in_col;
    int32_t             out_col;
    double              risk_free_rate;
    int32_t             annualization_periods;
    WelfordAccumulator  acc;

    inline void init(int32_t in, int32_t out,
                     double rfr, int32_t periods) noexcept {
        assert(periods > 0);
        assert(periods <= (int32_t)1 << 20);
        in_col                = in;
        out_col               = out;
        risk_free_rate        = rfr;
        annualization_periods = periods;
        acc.init();
    }
};

inline void execute_sharpe_ratio(SharpeRatioState* state,
                                 const double* BOLT_RESTRICT x,
                                 int64_t n,
                                 double* BOLT_RESTRICT out) noexcept {
    assert(state != nullptr);
    assert(x     != nullptr || n == 0);
    assert(out   != nullptr || n == 0);
    assert(n >= 0);
    if (n == 0) return;

    const double periods = static_cast<double>(state->annualization_periods);
    const double rf_per  = state->risk_free_rate / periods;

    state->acc.init();
    for (int64_t i = 0; i < n; ++i) state->acc.update(x[i]);

    const double var = (n > 1)
        ? state->acc.m2 / static_cast<double>(n - 1)    // sample variance
        : 0.0;
    const double stddev = std::sqrt(var);

    const double sharpe = (stddev > 1e-12)
        ? ((state->acc.mean - rf_per) / stddev) * std::sqrt(periods)
        : 0.0;

    // Scalar broadcast: fill every row.
    for (int64_t i = 0; i < n; ++i) out[i] = sharpe;
}

}  // namespace fintech
}  // namespace bolt
