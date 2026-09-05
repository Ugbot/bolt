// bolt/kernels/promql_deriv.h — Prometheus gauge derivative range functions.
//
//   promql_deriv          deriv(v[range])            — OLS slope (per second).
//   promql_predict_linear predict_linear(v[range],t) — OLS slope*t + intercept.
//
// Both are simple linear regression (ordinary least squares) over the
// windowed samples, with x measured in seconds relative to a chosen
// intercept time. No counter-reset handling — these are gauge functions and
// use raw values (the reset dip is a real slope input, per Prometheus).
//
// Semantics replicate Prometheus exactly:
//   promql/functions.go :: linearRegression / funcDeriv / funcPredictLinear.
//   Oracle: chukonu/tests/promql/testdata/upstream/functions.test
//   (READ-ONLY). Worked cases pinned in tests/test_promql_kernels.cpp.
//
//   deriv          uses intercept time = ts[0]        (slope is shift-invariant).
//   predict_linear uses intercept time = eval_ts_ms   (the query timestamp),
//                  then returns slope*predict_seconds + intercept, i.e. the
//                  value predicted `predict_seconds` after eval_ts_ms.
//
// Deviation from Prometheus: Prometheus accumulates the four regression sums
// with Kahan compensation. These kernels use plain double accumulation —
// bit-identical for the integer-valued oracle data, and within a few ULP for
// arbitrary inputs. If a workload needs full Kahan parity, that is a bounded,
// documented follow-up (compile-time switch), not a hot-path default.
//
// Tiger Style: noexcept, >= 2 asserts/fn, fixed-bound loops, no allocation,
// explicit integer sizes, functions <= 70 lines.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/promql_rate.h"   // promql_no_sample()

#include <cassert>
#include <cstdint>
#include <cmath>
#include <limits>

namespace bolt {
namespace promql {

// Ordinary-least-squares regression of val[] against time, with x in seconds
// relative to `intercept_time_ms`. Writes slope and the value at x==0
// (i.e. at intercept_time_ms) into the out-params. Returns false when the
// regression is undefined (fewer than two samples), leaving outputs at NaN.
//
// Matches Prometheus's constant-Y short-circuit: a series with a single
// distinct value has slope 0 and intercept == that value (and slope/intercept
// NaN if that value is +/-Inf).
inline bool promql_linear_regression(const int64_t* BOLT_RESTRICT ts,
                                     const double* BOLT_RESTRICT val,
                                     int64_t n,
                                     int64_t intercept_time_ms,
                                     double* BOLT_RESTRICT slope_out,
                                     double* BOLT_RESTRICT intercept_out) noexcept {
    assert(slope_out != nullptr);
    assert(intercept_out != nullptr);
    assert(ts != nullptr || n == 0);
    assert(val != nullptr || n == 0);
    assert(n >= 0);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    *slope_out = nan;
    *intercept_out = nan;
    if (n < 2) return false;

    const double init_y = val[0];
    bool const_y = true;
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0;
    for (int64_t i = 0; i < n; ++i) {          // bounded by n
        const double y = val[i];
        if (const_y && i > 0 && y != init_y) const_y = false;
        const double x = static_cast<double>(ts[i] - intercept_time_ms) / 1000.0;
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
    }
    if (const_y) {
        if (std::isinf(init_y)) return true;   // slope/intercept stay NaN
        *slope_out = 0.0;
        *intercept_out = init_y;
        return true;
    }
    const double nf = static_cast<double>(n);
    const double cov_xy = sum_xy - sum_x * sum_y / nf;
    const double var_x = sum_x2 - sum_x * sum_x / nf;
    const double slope = cov_xy / var_x;
    *slope_out = slope;
    *intercept_out = sum_y / nf - slope * sum_x / nf;
    assert(std::isnan(slope) || std::isfinite(slope) || std::isinf(slope));
    return true;
}

// deriv(v[range]) — the regression slope, per second. NaN when undefined
// (fewer than two samples, or a +/-Inf-only constant series).
inline double promql_deriv(const int64_t* BOLT_RESTRICT ts,
                           const double* BOLT_RESTRICT val,
                           int64_t n) noexcept {
    assert(ts != nullptr || n == 0);
    assert(n >= 0);
    double slope = 0.0, intercept = 0.0;
    const int64_t itime = (n > 0) ? ts[0] : 0;
    // The regression is only reported undefined for fewer than two samples,
    // which is ABSENCE. An Inf-only constant series returns true with a NaN
    // slope, which is a NaN VALUE and flows through unchanged.
    if (!promql_linear_regression(ts, val, n, itime, &slope, &intercept)) {
        return promql_no_sample();
    }
    return slope;
}

// predict_linear(v[range], predict_seconds) — the value the regression line
// predicts `predict_seconds` after `eval_ts_ms`. NaN when the regression is
// undefined.
inline double promql_predict_linear(const int64_t* BOLT_RESTRICT ts,
                                    const double* BOLT_RESTRICT val,
                                    int64_t n,
                                    int64_t eval_ts_ms,
                                    double predict_seconds) noexcept {
    assert(ts != nullptr || n == 0);
    assert(n >= 0);
    double slope = 0.0, intercept = 0.0;
    if (!promql_linear_regression(ts, val, n, eval_ts_ms, &slope, &intercept)) {
        return promql_no_sample();
    }
    return slope * predict_seconds + intercept;
}

}  // namespace promql
}  // namespace bolt
