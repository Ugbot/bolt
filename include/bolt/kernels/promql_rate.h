// bolt/kernels/promql_rate.h — Prometheus counter/gauge RANGE-function math.
//
// Pure, allocation-free kernels for the PromQL range-vector functions whose
// input is one series' windowed samples plus the window's [start,end] bounds:
//
//   promql_rate      rate(v[range])      — per-second avg rate (counter-reset
//                                          aware + boundary extrapolation).
//   promql_increase  increase(v[range])  — rate * range_seconds.
//   promql_delta     delta(v[range])     — gauge last-first, extrapolated.
//   promql_irate     irate(v[range])     — instantaneous rate, last 2 samples.
//   promql_idelta    idelta(v[range])    — gauge last 2 samples' difference.
//   promql_resets    resets(v[range])    — count of counter resets.
//   promql_changes   changes(v[range])   — count of value changes (NaN-aware).
//
// Semantics replicate Prometheus exactly:
//   promql/functions.go :: extrapolatedRate / instantValue / funcResets /
//   funcChanges. The oracle is chukonu/tests/promql/testdata/upstream/
//   functions.test (READ-ONLY — do not edit). Worked cases pinned in
//   tests/test_promql_kernels.cpp.
//
// DIVISION OF LABOUR: the CALLER (e.g. chukonu's window operator) is
// responsible for windowing — selecting the samples whose timestamp lies in
// Prometheus's half-open range (mint, maxt] — and passing them here already
// gathered per-series. These kernels do only the math, given:
//   ts[]            — ascending sample timestamps, milliseconds.
//   val[]           — sample values (float), same length n.
//   n               — number of windowed samples (>= 0).
//   range_start_ms  — window start boundary (Prometheus rangeStart == mint).
//   range_end_ms    — window end boundary   (Prometheus rangeEnd   == maxt).
//
// A NaN return means "no result" (Prometheus emits no sample; the caller
// drops it). This mirrors the < 2-sample and zero-interval degenerate cases.
//
// Tiger Style: noexcept, >= 2 asserts/fn (pre + post, positive + negative
// space, no side effects in assert), fixed-bound loops, no allocation, no
// std:: containers, explicit integer sizes, functions <= 70 lines.

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>
#include <cmath>
#include <limits>

namespace bolt {
namespace promql {

// NaN sentinel used for "no result".
BOLT_FORCE_INLINE double promql_nan() noexcept {
    return std::numeric_limits<double>::quiet_NaN();
}

// ---------------------------------------------------------------------------
// extrapolatedRate — the shared core of rate/increase/delta.
//
// is_counter : apply counter-reset correction (rate/increase; false for delta).
// is_rate    : divide by the range seconds (rate; false for increase/delta).
//
// Returns NaN when there are fewer than two samples or a zero sampling
// interval (both degenerate; Prometheus produces no output).
// ---------------------------------------------------------------------------
inline double promql_extrapolated_rate(const int64_t* BOLT_RESTRICT ts,
                                       const double* BOLT_RESTRICT val,
                                       int64_t n,
                                       int64_t range_start_ms,
                                       int64_t range_end_ms,
                                       bool is_counter,
                                       bool is_rate) noexcept {
    assert(ts != nullptr || n == 0);
    assert(val != nullptr || n == 0);
    assert(range_end_ms >= range_start_ms);
    assert(n >= 0);
    if (n < 2) return promql_nan();

    // Counter-reset correction: every time the series drops, the pre-drop
    // value is re-added (Prometheus assumes the counter restarted at 0).
    double counter_correction = 0.0;
    double prev = val[0];
    for (int64_t i = 1; i < n; ++i) {          // bounded by n
        const double curr = val[i];
        if (is_counter && curr < prev) counter_correction += prev;
        prev = curr;
    }
    const double first_v = val[0];
    const double last_v = val[n - 1];
    double result = last_v - first_v + counter_correction;

    const int64_t first_t = ts[0];
    const int64_t last_t = ts[n - 1];
    const double sampled_interval = static_cast<double>(last_t - first_t) / 1000.0;
    if (sampled_interval <= 0.0) return promql_nan();
    const double avg_interval = sampled_interval / static_cast<double>(n - 1);
    const double threshold = avg_interval * 1.1;

    double duration_to_start = static_cast<double>(first_t - range_start_ms) / 1000.0;
    double duration_to_end = static_cast<double>(range_end_ms - last_t) / 1000.0;
    double extrapolate = sampled_interval;

    if (duration_to_start >= threshold) duration_to_start = avg_interval / 2.0;
    // Counters cannot go negative: cap left-extrapolation at the zero crossing.
    if (is_counter && result > 0.0 && first_v >= 0.0) {
        const double duration_to_zero = sampled_interval * (first_v / result);
        if (duration_to_zero < duration_to_start) duration_to_start = duration_to_zero;
    }
    extrapolate += duration_to_start;
    if (duration_to_end >= threshold) duration_to_end = avg_interval / 2.0;
    extrapolate += duration_to_end;

    double factor = extrapolate / sampled_interval;
    if (is_rate) {
        const double range_seconds =
            static_cast<double>(range_end_ms - range_start_ms) / 1000.0;
        assert(range_seconds > 0.0);
        factor /= range_seconds;
    }
    assert(std::isfinite(factor) || std::isnan(factor));
    return result * factor;
}

// rate(v[range]) — counter-reset aware, extrapolated, per-second.
inline double promql_rate(const int64_t* BOLT_RESTRICT ts,
                          const double* BOLT_RESTRICT val,
                          int64_t n,
                          int64_t range_start_ms,
                          int64_t range_end_ms) noexcept {
    assert(n >= 0);
    assert(range_end_ms >= range_start_ms);
    return promql_extrapolated_rate(ts, val, n, range_start_ms, range_end_ms,
                                    /*is_counter=*/true, /*is_rate=*/true);
}

// increase(v[range]) — rate * range_seconds (no per-second division).
inline double promql_increase(const int64_t* BOLT_RESTRICT ts,
                              const double* BOLT_RESTRICT val,
                              int64_t n,
                              int64_t range_start_ms,
                              int64_t range_end_ms) noexcept {
    assert(n >= 0);
    assert(range_end_ms >= range_start_ms);
    return promql_extrapolated_rate(ts, val, n, range_start_ms, range_end_ms,
                                    /*is_counter=*/true, /*is_rate=*/false);
}

// delta(v[range]) — gauge last-first, boundary-extrapolated, no reset logic.
inline double promql_delta(const int64_t* BOLT_RESTRICT ts,
                           const double* BOLT_RESTRICT val,
                           int64_t n,
                           int64_t range_start_ms,
                           int64_t range_end_ms) noexcept {
    assert(n >= 0);
    assert(range_end_ms >= range_start_ms);
    return promql_extrapolated_rate(ts, val, n, range_start_ms, range_end_ms,
                                    /*is_counter=*/false, /*is_rate=*/false);
}

// ---------------------------------------------------------------------------
// instantValue — the shared core of irate/idelta (last two samples only).
// Range bounds are intentionally not consumed (Prometheus ignores them here).
// ---------------------------------------------------------------------------
inline double promql_instant_value(const int64_t* BOLT_RESTRICT ts,
                                   const double* BOLT_RESTRICT val,
                                   int64_t n,
                                   bool is_rate) noexcept {
    assert(ts != nullptr || n == 0);
    assert(val != nullptr || n == 0);
    assert(n >= 0);
    if (n < 2) return promql_nan();
    const double last_v = val[n - 1];
    const double prev_v = val[n - 2];
    // Counter-reset aware: on a drop, the delta is the last value itself.
    double result = (is_rate && last_v < prev_v) ? last_v : (last_v - prev_v);
    const int64_t dt_ms = ts[n - 1] - ts[n - 2];
    if (dt_ms == 0) return promql_nan();
    if (is_rate) result /= static_cast<double>(dt_ms) / 1000.0;
    assert(dt_ms > 0);
    return result;
}

// irate(v[range]) — instantaneous per-second rate, counter-reset aware.
inline double promql_irate(const int64_t* BOLT_RESTRICT ts,
                           const double* BOLT_RESTRICT val,
                           int64_t n) noexcept {
    assert(n >= 0);
    assert(ts != nullptr || n == 0);
    return promql_instant_value(ts, val, n, /*is_rate=*/true);
}

// idelta(v[range]) — last two samples' difference (gauge).
inline double promql_idelta(const int64_t* BOLT_RESTRICT ts,
                            const double* BOLT_RESTRICT val,
                            int64_t n) noexcept {
    assert(n >= 0);
    assert(val != nullptr || n == 0);
    return promql_instant_value(ts, val, n, /*is_rate=*/false);
}

// resets(v[range]) — number of counter resets (value strictly decreases).
inline int64_t promql_resets(const double* BOLT_RESTRICT val, int64_t n) noexcept {
    assert(val != nullptr || n == 0);
    assert(n >= 0);
    int64_t count = 0;
    for (int64_t i = 1; i < n; ++i) {          // bounded by n
        count += (val[i] < val[i - 1]) ? 1 : 0;
    }
    assert(count >= 0);
    assert(count <= (n > 0 ? n - 1 : 0));
    return count;
}

// changes(v[range]) — number of value changes; NaN==NaN is NOT a change.
inline int64_t promql_changes(const double* BOLT_RESTRICT val, int64_t n) noexcept {
    assert(val != nullptr || n == 0);
    assert(n >= 0);
    int64_t count = 0;
    for (int64_t i = 1; i < n; ++i) {          // bounded by n
        const double a = val[i];
        const double b = val[i - 1];
        const bool both_nan = std::isnan(a) && std::isnan(b);
        count += (a != b && !both_nan) ? 1 : 0;
    }
    assert(count >= 0);
    assert(count <= (n > 0 ? n - 1 : 0));
    return count;
}

}  // namespace promql
}  // namespace bolt
