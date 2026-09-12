// bolt/kernels/promql_smoothing.h — Prometheus double exponential smoothing.
//
//   promql_double_exp_smoothing  double_exponential_smoothing(v[range], sf, tf)
//
// Holt's linear (double exponential) smoothing: a weighted moving average in
// which historical data has exponentially less influence, plus a trend term.
// `sf` is the smoothing factor and `tf` the trend factor; both must lie
// strictly inside (0, 1).
//
// Semantics replicate Prometheus exactly:
//   promql/functions.go :: calcTrendValue / funcDoubleExponentialSmoothing
//   (transcribed from the v3.7.3 tag, not from memory).
//   Oracle: chukonu/tests/promql/testdata/upstream/functions.test (READ-ONLY)
//   AND a differential run of prom/prometheus:v3.7.3 over non-linear data —
//   benchmarks/promql/des_oracle.py. The second oracle is load-bearing, not
//   belt-and-braces: EVERY window in the vendored corpus is a perfect
//   arithmetic ramp, and on a perfect ramp this function returns the last
//   sample for ANY sf and tf, so the corpus alone cannot tell a correct
//   implementation from `return val[n-1]` nor sf from tf. That is measured,
//   not asserted — see chukonu/tests/promql/test_promql_des.cpp, whose pins
//   use non-linear data precisely because the corpus cannot.
//
// TIMESTAMPS ARE NOT READ. Unlike rate/deriv/predict_linear this function is
// a pure sequence fold: it depends only on the ORDER of the samples, never on
// their spacing. The caller must therefore hand it the window already sorted
// ascending by timestamp — the same contract the other range kernels rely on.
//
// HISTOGRAM SAMPLES MUST NOT APPEAR IN `val`. Upstream folds over
// `samples.Floats`, a slice that never contains a histogram, and `l` is that
// slice's length. A caller that passes a merged float+histogram window with
// placeholder values would fold the placeholders into the answer AND get the
// `n < 2` short-circuit wrong. Compacting is the caller's job; it is stated
// here because the failure is a silent wrong number, not a crash.
//
// Tiger Style: noexcept, >= 2 asserts/fn, fixed-bound loops, no allocation,
// explicit integer sizes, functions <= 70 lines.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/promql_rate.h"   // promql_no_sample()

#include <cassert>
#include <cstdint>
#include <cmath>

namespace bolt {
namespace promql {

// True when `f` is a factor upstream ACCEPTS. Spelled as the negation of
// upstream's own reject test rather than as the positive interval, because on
// NaN the two are not the same predicate and this file's first version got
// that backwards.
//
//   upstream (promql/functions.go, v3.7.3):  if sf <= 0 || sf >= 1 { error }
//   NaN <= 0 is false and NaN >= 1 is false, so upstream ACCEPTS a NaN factor
//   and folds it through, emitting a NaN SAMPLE.
//
// MEASURED, not reasoned — promtool 3.7.3 over `3 1 4 1 5`:
//   double_exponential_smoothing(m[5m], NaN, 0.5)  ->  {k="a"} NaN
//   double_exponential_smoothing(m[5m], 0.0, 0.5)  ->  err: invalid smoothing
//                                     factor. Expected: 0 < sf < 1, got: 0.0
// So 0 and 1 are query errors while NaN is a legal query with a NaN answer.
// A `(f > 0 && f < 1)` interval test rejects all three alike and would refuse
// a query the reference answers. Pinned in test_promql_des.cpp.
//
// The emitted NaN cannot be mistaken for absence: `promql_no_sample()` is the
// payload-3 NaN 0x7ff0000000000003 and `std::nan("")` — what the PromQL
// parser builds for the literal `NaN` — is 0x7ff8000000000000. Checked, since
// a collision here would turn a sample the reference emits into a missing
// series, which is the one failure this campaign ranks above every other.
inline bool promql_des_factor_valid(double f) noexcept {
    const bool ok = !(f <= 0.0 || f >= 1.0);
    assert(!ok || !(f <= 0.0));
    assert(!ok || !(f >= 1.0));
    return ok;
}

// The trend value at index `i`. `i == 0` keeps the seed trend; every later
// index blends the change in the smoothed level with the previous trend.
// Mirrors upstream `calcTrendValue` argument for argument, including its
// parameter order (s0 = previous smoothed value, s1 = current, b = trend).
inline double promql_des_trend(std::int64_t i, double tf,
                               double s0, double s1, double b) noexcept {
    assert(i >= 0);
    // Not `tf > 0 && tf < 1`: a NaN trend factor is legal upstream (see
    // promql_des_factor_valid), and an interval assert would abort an
    // asserts-live build on a query the reference answers with NaN.
    assert(promql_des_factor_valid(tf));
    if (i == 0) return b;
    return tf * (s1 - s0) + (1.0 - tf) * b;
}

// double_exponential_smoothing(v[range], sf, tf) over one series' window.
//
// `val` holds the window's FLOAT samples in ascending timestamp order, with
// every histogram sample already removed (see the header note). Returns the
// final smoothed level, or `promql_no_sample()` when fewer than two float
// samples are present — upstream emits no sample there, and the caller
// compacts a no-sample away rather than reporting a value.
//
// `sf` and `tf` are preconditions, not runtime conditions: an out-of-range
// factor is an invalid QUERY whatever the data is — upstream v3.7.3 answers
// it with `err: invalid smoothing factor. Expected: 0 < sf < 1, got:
// 0.000000` (a query error; an earlier version of this comment said "PANICS",
// which was never checked and is wrong for the pinned release) — and this
// engine rejects it in the runner before any data is touched. Asserted here,
// and defended with a no-sample return in a release build rather than folding
// a garbage factor into a confident number.
//
// "Before any data is touched" is where this engine and upstream DIVERGE, by
// one measured case: upstream validates per SELECTED series, so an invalid
// factor over a selector matching nothing is not an error there at all. The
// divergence is named and pinned in test_promql_des.cpp; it costs an error
// where upstream returns empty, never a value where upstream returns another.
inline double promql_double_exp_smoothing(const double* BOLT_RESTRICT val,
                                          std::int64_t n,
                                          double sf, double tf) noexcept {
    assert(val != nullptr || n == 0);
    assert(n >= 0);
    assert(promql_des_factor_valid(sf));
    assert(promql_des_factor_valid(tf));
    if (!promql_des_factor_valid(sf) || !promql_des_factor_valid(tf))
        return promql_no_sample();
    if (n < 2) return promql_no_sample();

    double s0 = 0.0;
    double s1 = val[0];
    double b = val[1] - val[0];
    for (std::int64_t i = 1; i < n; ++i) {          // bounded by n
        const double x = sf * val[i];
        b = promql_des_trend(i - 1, tf, s0, s1, b);
        const double y = (1.0 - sf) * (s1 + b);
        s0 = s1;
        s1 = x + y;
    }
    assert(n >= 2);
    return s1;
}

}  // namespace promql
}  // namespace bolt
