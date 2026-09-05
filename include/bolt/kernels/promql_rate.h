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
#include <cstring>
#include <limits>

namespace bolt {
namespace promql {

// An ordinary quiet NaN: a real PromQL sample whose VALUE is NaN.
BOLT_FORCE_INLINE double promql_nan() noexcept {
    return std::numeric_limits<double>::quiet_NaN();
}

// ---------------------------------------------------------------------------
// "No sample" versus "a sample whose value is NaN"
//
// PromQL distinguishes these and a range function needs both:
//
//   * NO SAMPLE — the structural precondition failed (fewer than two samples
//     in the window, a zero sampling interval, an empty window). Prometheus
//     emits NOTHING for the series at that step.
//   * A NaN VALUE — the arithmetic legitimately produced NaN, e.g.
//     `irate(http_requests_nan[15m1s])` over `1 NaN NaN 5 11`, which
//     Prometheus asserts as `_ NaN NaN NaN 0.02`
//     (functions.test:241 / :324). The sample EXISTS and its value is NaN.
//
// Returning a plain quiet NaN for the first case makes the two indis-
// tinguishable, and every caller that compacts on `isnan` then silently DROPS
// legitimately-NaN results — an `Ok` reply with an empty series where a value
// is due.
//
// The distinction is carried in the NaN PAYLOAD, which is exactly how
// Prometheus itself carries its own stale marker (payload 2). `no_sample` is
// still a quiet NaN, so `std::isnan` remains true for it and every existing
// caller keeps its current behaviour until it opts in to the finer test —
// this addition changes no result on its own.
constexpr uint64_t k_promql_no_sample_bits = 0x7ff0000000000003ULL;
constexpr uint64_t k_promql_stale_bits     = 0x7ff0000000000002ULL;
static_assert(k_promql_no_sample_bits != k_promql_stale_bits,
              "the no-sample payload must not collide with Prometheus's stale marker");

BOLT_FORCE_INLINE double promql_no_sample() noexcept {
    // Not a reinterpret_cast: type punning through a union or a pointer cast is
    // UB. memcpy is the sanctioned spelling and compiles to a register move.
    double v;
    const uint64_t bits = k_promql_no_sample_bits;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

// True only for the sentinel above — never for an ordinary NaN result.
BOLT_FORCE_INLINE bool promql_is_no_sample(double v) noexcept {
    uint64_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    return bits == k_promql_no_sample_bits;
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
    if (n < 2) return promql_no_sample();

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
    if (sampled_interval <= 0.0) return promql_no_sample();
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
    if (n < 2) return promql_no_sample();
    const double last_v = val[n - 1];
    const double prev_v = val[n - 2];
    // Counter-reset aware: on a drop, the delta is the last value itself.
    double result = (is_rate && last_v < prev_v) ? last_v : (last_v - prev_v);
    const int64_t dt_ms = ts[n - 1] - ts[n - 2];
    if (dt_ms == 0) return promql_no_sample();
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


// ===========================================================================
// PromQL 3 range MODIFIERS: `anchored` and `smoothed`.
//
// Both change WHICH values the range function differences, not the function.
// Prometheus reference: the PROM-52 extended-range-vector proposal; the oracle
// is chukonu/tests/promql/testdata/upstream/extended_vectors.test (READ-ONLY),
// whose per-case comments state the intended arithmetic in words, e.g.
// "Anchored rate at 50s: left=count:1 (t=0), right=count:4 (t=45), rate=3/60".
//
// CALLER CONTRACT DIFFERS from the plain kernels above: these need samples
// OUTSIDE the range, so the caller must pass a WIDENED, ascending sample array
// covering at least (range_start - lookback, range_end + lookback] and give the
// TRUE range bounds separately. The kernels locate the range inside the array.
//
//   anchored — prepend the last sample at or before range_start (within
//              `lookback_ms`) to the in-range samples, then difference with NO
//              extrapolation. "Where did this counter start the window at?"
//   smoothed — linearly interpolate the series at BOTH range boundaries and
//              difference those two interpolated values. Boundaries beyond the
//              first/last sample clamp to it.
// ===========================================================================

// Index of the last sample with ts <= t, or -1 when none. Bounded linear scan
// (a windowed series is short; no allocation, no branchy binary search).
BOLT_FORCE_INLINE int64_t promql_last_at_or_before(const int64_t* BOLT_RESTRICT ts,
                                                   int64_t n, int64_t t) noexcept {
    assert(ts != nullptr || n == 0);
    assert(n >= 0);
    int64_t j = -1;
    for (int64_t i = 0; i < n; ++i) {          // bounded by n, ts ascending
        if (ts[i] > t) break;
        j = i;
    }
    assert(j >= -1 && j < n);
    return j;
}

// The ANCHORED sample slice for (range_start, range_end]: [*begin, *begin+*cnt).
// Returns false when the range holds no sample (Prometheus emits nothing).
// The slice starts at the anchor sample when one exists in
// (range_start - lookback_ms, range_start]; otherwise at the first in-range
// sample. `lookback_ms <= 0` disables anchoring (slice == the in-range samples).
inline bool promql_anchored_span(const int64_t* BOLT_RESTRICT ts, int64_t n,
                                 int64_t range_start_ms, int64_t range_end_ms,
                                 int64_t lookback_ms,
                                 int64_t* BOLT_RESTRICT begin,
                                 int64_t* BOLT_RESTRICT cnt) noexcept {
    assert(ts != nullptr || n == 0);
    assert(begin != nullptr && cnt != nullptr);
    assert(range_end_ms >= range_start_ms);
    *begin = 0; *cnt = 0;
    const int64_t hi = promql_last_at_or_before(ts, n, range_end_ms);
    if (hi < 0) return false;                        // nothing at or before end
    const int64_t before = promql_last_at_or_before(ts, n, range_start_ms);
    const int64_t lo = before + 1;                   // first sample IN range
    if (lo > hi) return false;                       // no sample inside range
    int64_t start = lo;
    if (lookback_ms > 0 && before >= 0 &&
        ts[before] > range_start_ms - lookback_ms) {
        start = before;                              // the anchor
    }
    *begin = start;
    *cnt = hi - start + 1;
    assert(*cnt >= 1);
    assert(*begin >= 0 && *begin + *cnt <= n);
    return true;
}

// anchored rate/increase/delta over a widened sample array.
inline double promql_anchored_rate(const int64_t* BOLT_RESTRICT ts,
                                   const double* BOLT_RESTRICT val,
                                   int64_t n,
                                   int64_t range_start_ms,
                                   int64_t range_end_ms,
                                   int64_t lookback_ms,
                                   bool is_counter, bool is_rate) noexcept {
    assert(ts != nullptr || n == 0);
    assert(val != nullptr || n == 0);
    assert(range_end_ms >= range_start_ms);
    int64_t b = 0, c = 0;
    if (!promql_anchored_span(ts, n, range_start_ms, range_end_ms,
                              lookback_ms, &b, &c)) return promql_no_sample();
    // Unlike the extrapolating kernels, ONE sample is a result, not a
    // degenerate case: an anchored window holding a single sample and no
    // anchor has provably not increased, and upstream asserts exactly that
    // (`increase(metric[1m] anchored)` at 5s is 0, not absent). Absence is
    // reserved for a window holding NO sample, which the span check above
    // already returned false for.
    double correction = 0.0;
    double prev = val[b];
    for (int64_t i = b + 1; i < b + c; ++i) {        // bounded by c
        const double curr = val[i];
        if (is_counter && curr < prev) correction += prev;
        prev = curr;
    }
    double result = val[b + c - 1] - val[b] + correction;
    if (is_rate) {
        const double range_seconds =
            static_cast<double>(range_end_ms - range_start_ms) / 1000.0;
        if (range_seconds <= 0.0) return promql_no_sample();
        result /= range_seconds;
    }
    assert(!(range_end_ms < range_start_ms));
    return result;
}

// Value of the series at instant `t`, linearly interpolated between the two
// bracketing samples; clamped to the first/last sample outside their span.
// `counter_adjust` adds Prometheus's counter-reset correction so a series that
// restarts is interpolated on its monotonic reconstruction, not across the drop.
// Returns NaN when the array is empty.
inline double promql_interpolated_value_at(const int64_t* BOLT_RESTRICT ts,
                                           const double* BOLT_RESTRICT val,
                                           int64_t n, int64_t t,
                                           bool counter_adjust) noexcept {
    assert(ts != nullptr || n == 0);
    assert(val != nullptr || n == 0);
    assert(n >= 0);
    if (n <= 0) return promql_no_sample();
    const int64_t j = promql_last_at_or_before(ts, n, t);
    if (j < 0) return val[0];                        // before the first sample
    // Correction accumulated up to j (a constant offset across both boundary
    // reads of one query, so differencing them is unaffected by where it starts).
    double corr = 0.0;
    if (counter_adjust) {
        for (int64_t i = 1; i <= j; ++i)             // bounded by n
            if (val[i] < val[i - 1]) corr += val[i - 1];
    }
    const double vj = val[j] + corr;
    if (j == n - 1) return vj;                       // after the last sample
    if (ts[j] == t) return vj;                       // exact hit
    double next = val[j + 1];
    if (counter_adjust && next < val[j]) next += val[j];
    next += corr;
    const double span = static_cast<double>(ts[j + 1] - ts[j]);
    assert(span > 0.0);
    const double frac = static_cast<double>(t - ts[j]) / span;
    assert(frac >= 0.0 && frac <= 1.0);
    return vj + (next - vj) * frac;
}

// smoothed rate/increase/delta: difference of the interpolated values at the
// two range boundaries. Returns NaN ("no result") when the data lies wholly
// before or wholly after the range — the case upstream marks
// "Smoothed rate returns empty when data is only before or only after the
// range" — because then both boundaries would clamp to the same sample.
inline double promql_smoothed_rate(const int64_t* BOLT_RESTRICT ts,
                                   const double* BOLT_RESTRICT val,
                                   int64_t n,
                                   int64_t range_start_ms,
                                   int64_t range_end_ms,
                                   bool is_counter, bool is_rate) noexcept {
    assert(ts != nullptr || n == 0);
    assert(val != nullptr || n == 0);
    assert(range_end_ms >= range_start_ms);
    if (n <= 0) return promql_no_sample();
    if (ts[0] > range_end_ms) return promql_no_sample();       // data only after
    if (ts[n - 1] <= range_start_ms) return promql_no_sample(); // data only before
    const double hi =
        promql_interpolated_value_at(ts, val, n, range_end_ms, is_counter);
    const double lo =
        promql_interpolated_value_at(ts, val, n, range_start_ms, is_counter);
    // A boundary that is itself absent (empty array) is absence; a boundary
    // whose VALUE is NaN yields a NaN result, which Prometheus emits.
    if (promql_is_no_sample(hi) || promql_is_no_sample(lo)) return promql_no_sample();
    double result = hi - lo;
    if (is_rate) {
        const double range_seconds =
            static_cast<double>(range_end_ms - range_start_ms) / 1000.0;
        if (range_seconds <= 0.0) return promql_no_sample();
        result /= range_seconds;
    }
    assert(n > 0);
    return result;
}

}  // namespace promql
}  // namespace bolt
