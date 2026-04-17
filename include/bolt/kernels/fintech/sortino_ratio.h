// bolt/kernels/fintech/sortino_ratio.h — Phase 5.W.5 Tier 2 kernel.
//
// SortinoRatio: scalar-broadcast Sortino ratio over a batch of per-period
// returns. Like Sharpe, but the denominator is "downside deviation" —
// stddev computed over only the rows where `excess < 0`.
//
//   rf_per_period       = risk_free_rate / annualization_periods
//   excess_i            = returns[i] - rf_per_period
//   downside_sum_sq     = Σ  excess_i²  over i s.t. excess_i < 0
//   downside_dev        = sqrt(downside_sum_sq / n)      ← / n, NOT n_down
//   sortino = (mean - rf_per_period) / downside_dev * sqrt(periods)
//
// Source: chukonu/fintech/kernels.h :: CreateSortinoRatioProcessor (1090).
// Contract copied verbatim — in particular chukonu divides the downside
// sum-of-squares by `n` (the total sample count), not by the number of
// downside observations. This is the Sortino-Sharpe-Ratio practitioner
// convention in chukonu; matched for parity.
//
// Output shape: **scalar broadcast** — every row gets the same value.
// Numerical guard: `downside_dev > 1e-12` else 0.0 (not NaN).

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace bolt {
namespace fintech {

struct SortinoRatioState {
    int32_t  in_col;
    int32_t  out_col;
    double   risk_free_rate;
    int32_t  annualization_periods;

    inline void init(int32_t in, int32_t out,
                     double rfr, int32_t periods) noexcept {
        assert(periods > 0);
        assert(periods <= (int32_t)1 << 20);
        in_col                = in;
        out_col               = out;
        risk_free_rate        = rfr;
        annualization_periods = periods;
    }
};

inline void execute_sortino_ratio(SortinoRatioState* state,
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

    double sum = 0.0;
    double downside_sum_sq = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double v = x[i];
        sum += v;
        const double excess = v - rf_per;
        // Branchless downside filter: ternary → cmov on x86, csel on ARM.
        // `neg = excess` if excess<0, else 0.0; squaring 0 is a no-op, so
        // this is equivalent to the conditional `+= excess*excess` but
        // with zero branches in the per-row body.
        const double neg = (excess < 0.0) ? excess : 0.0;
        downside_sum_sq += neg * neg;
    }
    const double mean = sum / static_cast<double>(n);
    // Chukonu convention: divide by n (not n_downside) for the downside dev.
    const double downside_dev = std::sqrt(downside_sum_sq
                                          / static_cast<double>(n));

    const double sortino = (downside_dev > 1e-12)
        ? ((mean - rf_per) / downside_dev) * std::sqrt(periods)
        : 0.0;

    for (int64_t i = 0; i < n; ++i) out[i] = sortino;
}

}  // namespace fintech
}  // namespace bolt
