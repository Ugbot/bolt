// bolt/kernels/promql_nh_extended.h — the PromQL 3 EXTENDED range selectors
// (`anchored` / `smoothed`) over NATIVE HISTOGRAMS.
//
// promql_rate.h carries the FLOAT half of this feature
// (`promql_anchored_span` / `promql_anchored_rate` /
// `promql_interpolated_value_at` / `promql_smoothed_rate`). This is the
// histogram half, and it is not that arithmetic wearing a different type:
//
//   * INTERPOLATING a counter histogram across a reset is not a linear blend.
//     Upstream models the counter as restarting from zero and returns
//     `h2 * fraction`; blending would answer with a distribution that is
//     neither endpoint and no row count could see it.
//   * The reset CORRECTION is taken over an index range that deliberately
//     EXCLUDES both boundary samples, because `right` is either a copy of
//     `h[lastSampleIndex]` or an interpolation that INHERITS its
//     CounterResetHint — including that sample would make `right.DetectReset`
//     self-detect and double-count the correction.
//   * When the smoothed LEFT interpolation already spanned a reset, that reset
//     is spent: the loop must skip the sample after it and use that sample as
//     the comparison anchor instead.
//
// SEMANTICS are a port of prometheus **main** (the branch whose
// `promql/promqltest/testdata/extended_vectors.test` this tree vendors).
// Named sources, all in `promql/functions.go` unless stated:
//   interpolateHistograms, pickOrInterpolateLeftHistogram,
//   pickOrInterpolateRightHistogram, validateHistogramRange,
//   correctForCounterResetsHistogram, extendedHistogramRate
//   promql/engine.go :: the `smoothed` vector-selector histogram branch
//   model/histogram/float_histogram.go :: adjustCounterReset, Mul, Div
//
// THREE OUTCOMES, the same contract promql_nh_rate.h states and for the same
// reason — an empty result and a declined query are different claims:
//   Ok       — a histogram was produced.
//   NoSample — Prometheus produces NOTHING here and says so with a warning:
//              a window with no sample past the range start, a ladder that
//              mixes exponential and custom buckets, or (for the selector) a
//              lookback holding both floats and histograms.
//   Refused  — this kernel cannot answer EXACTLY (an undecidable reset, or a
//              combine upstream reconciles and `nh_combine` does not). The
//              caller must decline the query, never emit a partial answer.
//
// Tiger Style: noexcept, >= 2 asserts/fn, fixed-bound loops, NO allocation
// (the caller supplies every histogram, including scratch), explicit integer
// sizes, functions <= 70 lines.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/promql_native_histogram.h"
#include "bolt/kernels/promql_nh_rate.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace promql {

// Scratch slots `nh_extended_rate` needs from its caller: left boundary, right
// boundary, accumulated correction. None may alias `out`.
inline constexpr int32_t k_nh_ext_scratch = 3;

// prometheus/model/histogram/float_histogram.go :: adjustCounterReset — run by
// Add and Sub on the RECEIVER. It matters because the result of an
// interpolation is later fed to `nh_detect_reset` AS THE RECEIVER, and the
// receiver's hint short-circuits that test in both directions. `nh_combine`
// deliberately leaves the hint alone (its other callers depend on that), so the
// adjustment is applied here, next to the combine that earns it.
//
// UNORACLED (W13-L2). Making this a no-op leaves extended_vectors.test
// BIT-IDENTICAL: no vendored case produces a CounterReset/NotReset collision
// whose resulting hint a later DetectReset actually reads. It is kept because
// it is what upstream does on every Add/Sub and a real dataset can reach the
// shape — but this corpus does not prove it, and it must not be cited as if
// it did.
BOLT_FORCE_INLINE void nh_adjust_reset_hint(NativeHistogram* BOLT_RESTRICT h,
                                            const NativeHistogram* BOLT_RESTRICT other) noexcept {
    assert(h != nullptr && other != nullptr);
    assert(h != other);
    constexpr uint8_t gauge = static_cast<uint8_t>(NhResetHint::Gauge);
    constexpr uint8_t unk   = static_cast<uint8_t>(NhResetHint::Unknown);
    const uint8_t hh = h->reset_hint, oh = other->reset_hint;
    if (oh == hh) return;                    // apples to apples
    if (hh == gauge) return;                 // already a gauge
    if (oh == gauge) { h->reset_hint = gauge; return; }
    if (hh == unk) return;
    // Either `other` is unknown, or CounterReset collides with NotReset.
    // Upstream conservatively answers "unknown" for both.
    h->reset_hint = unk;
}

// `dst = dst + src`, with the hint adjustment upstream's Add performs.
BOLT_FORCE_INLINE bool nh_add_adjusted(NativeHistogram* BOLT_RESTRICT dst,
                                       const NativeHistogram* BOLT_RESTRICT src) noexcept {
    assert(dst != nullptr && src != nullptr);
    assert(dst != src);
    if (!nh_combine(dst, src, /*subtract=*/false)) return false;
    nh_adjust_reset_hint(dst, src);
    return true;
}

// ---------------------------------------------------------------------------
// interpolateHistograms — the value of the series at instant `t`, between the
// bracketing samples (h1 @ t1) and (h2 @ t2).
//
// The counter-reset branch is the whole point: when `is_counter` and h2 is a
// reset of h1, upstream returns `h2 * fraction` — the counter is modelled as
// having restarted from zero at t1 — NOT a blend of the two. The corpus pins
// it: `histogram_count(reset_middle smoothed)` at 20s is 0.6666…, which is
// count:2 * (5/15), and any blend of count:10 and count:2 is > 2.
// ---------------------------------------------------------------------------
inline NhRangeStatus nh_interpolate(const NativeHistogram* h1, int64_t t1,
                                    const NativeHistogram* h2, int64_t t2,
                                    int64_t t, bool is_counter,
                                    NativeHistogram* BOLT_RESTRICT out) noexcept {
    assert(h1 != nullptr && h2 != nullptr && out != nullptr);
    assert(t2 > t1);
    if (t == t1) { *out = *h1; return NhRangeStatus::Ok; }
    if (t == t2) { *out = *h2; return NhRangeStatus::Ok; }
    // Upstream's Sub reports ErrHistogramsIncompatibleSchema for a ladder mix,
    // which the caller turns into a warning and an EMPTY result — that is
    // Prometheus's answer, not a gap in this kernel.
    if (nh_uses_custom(h1) != nh_uses_custom(h2)) return NhRangeStatus::NoSample;
    const double fraction =
        static_cast<double>(t - t1) / static_cast<double>(t2 - t1);
    assert(fraction > 0.0 && fraction < 1.0);
    if (is_counter) {
        const NhReset r = nh_detect_reset(h2, h1);
        if (r == NhReset::Undecidable) return NhRangeStatus::Refused;
        if (r == NhReset::Yes) {
            // Restart-from-zero. The hint rides along on the copy, which is
            // what makes a later `right.DetectReset(prev)` fire.
            *out = *h2;
            nh_mul(out, fraction);
            return NhRangeStatus::Ok;
        }
    }
    *out = *h2;                                   // result = h1 + (h2 - h1)*f
    if (!nh_combine(out, h1, /*subtract=*/true)) return NhRangeStatus::Refused;
    nh_adjust_reset_hint(out, h1);
    nh_mul(out, fraction);
    if (!nh_add_adjusted(out, h1)) return NhRangeStatus::Refused;
    return NhRangeStatus::Ok;
}

namespace detail {

// Index of the first sample in [0, hi) with ts[i] > bound (Go's sort.Search
// predicate), or `hi` when none satisfies it. Bounded linear scan: a windowed
// series is short and this mirrors promql_last_at_or_before's shape.
BOLT_FORCE_INLINE int32_t nh_search_gt(const int64_t* BOLT_RESTRICT ts,
                                       int32_t hi, int64_t bound) noexcept {
    assert(ts != nullptr || hi == 0);
    assert(hi >= 0);
    for (int32_t i = 0; i < hi; ++i) if (ts[i] > bound) return i;   // bounded
    return hi;
}

// The same, for `ts[i] >= bound`.
BOLT_FORCE_INLINE int32_t nh_search_ge(const int64_t* BOLT_RESTRICT ts,
                                       int32_t hi, int64_t bound) noexcept {
    assert(ts != nullptr || hi == 0);
    assert(hi >= 0);
    for (int32_t i = 0; i < hi; ++i) if (ts[i] >= bound) return i;  // bounded
    return hi;
}

// correctForCounterResetsHistogram. `corr` is the caller's accumulator and
// `*have` tracks whether anything has landed in it yet (upstream's nil check).
inline NhRangeStatus nh_reset_correction(const int64_t* BOLT_RESTRICT ts,
                                         const NativeHistogram* const* points,
                                         int32_t first_idx, int32_t last_idx,
                                         const NativeHistogram* left,
                                         const NativeHistogram* right,
                                         int64_t range_start_ms, bool smoothed,
                                         NativeHistogram* BOLT_RESTRICT corr,
                                         bool* BOLT_RESTRICT have) noexcept {
    assert(ts != nullptr && points != nullptr && corr != nullptr);
    assert(left != nullptr && right != nullptr && have != nullptr);
    *have = false;
    int32_t first = first_idx + 1;      // first_idx is represented by `left`
    const NativeHistogram* prev = left;
    if (smoothed && ts[first_idx] < range_start_ms) {
        const NhReset r = nh_detect_reset(points[first_idx + 1], points[first_idx]);
        if (r == NhReset::Undecidable) return NhRangeStatus::Refused;
        if (r == NhReset::Yes) {
            // The left interpolation already spanned this reset. Skip the
            // sample and make it the anchor for whatever follows.
            prev = points[first_idx + 1];
            ++first;
        }
    }
    // last_idx is ALWAYS excluded: `right` is a copy of it or an interpolation
    // carrying its hint, so keeping it as `prev` would let right.DetectReset
    // self-detect and count the same reset twice.
    const int32_t last = last_idx - 1;
    if (first > last + 1) return NhRangeStatus::Ok;   // nothing between them
    for (int32_t i = first; i <= last; ++i) {                    // bounded
        const NhReset r = nh_detect_reset(points[i], prev);
        if (r == NhReset::Undecidable) return NhRangeStatus::Refused;
        if (r == NhReset::Yes) {
            if (!*have) { *corr = *prev; *have = true; }
            else if (!nh_add_adjusted(corr, prev)) return NhRangeStatus::Refused;
        }
        prev = points[i];
    }
    const NhReset r = nh_detect_reset(right, prev);
    if (r == NhReset::Undecidable) return NhRangeStatus::Refused;
    if (r == NhReset::Yes) {
        if (!*have) { *corr = *prev; *have = true; }
        else if (!nh_add_adjusted(corr, prev)) return NhRangeStatus::Refused;
    }
    return NhRangeStatus::Ok;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// extendedHistogramRate — rate / increase / delta under `anchored` or
// `smoothed` over an all-histogram window.
//
// CALLER CONTRACT (identical to promql_rate.h's modifier kernels): `ts`/`points`
// are the WIDENED, ascending window — at least (range_start - lookback,
// range_end] for anchored and (range_start - lookback, range_end + lookback]
// for smoothed — and the TRUE range bounds arrive separately. A window that
// mixes float and histogram samples must be rejected by the caller before this
// is reached (upstream warns and emits nothing for that shape).
// `scratch` is k_nh_ext_scratch slots; none may alias `out`.
// ---------------------------------------------------------------------------
inline NhRangeStatus nh_extended_rate(const int64_t* BOLT_RESTRICT ts,
                                      const NativeHistogram* const* points,
                                      int32_t n,
                                      int64_t range_start_ms,
                                      int64_t range_end_ms,
                                      bool is_counter, bool is_rate, bool smoothed,
                                      NativeHistogram* scratch,
                                      NativeHistogram* BOLT_RESTRICT out) noexcept {
    assert(ts != nullptr || n == 0);
    assert(points != nullptr || n == 0);
    assert(scratch != nullptr && out != nullptr);
    assert(range_end_ms >= range_start_ms);
    if (n <= 0) return NhRangeStatus::NoSample;
    int32_t last = n - 1;
    int32_t first = detail::nh_search_gt(ts, last, range_start_ms) - 1;
    if (first < 0) first = 0;
    if (smoothed) last = detail::nh_search_ge(ts, last, range_end_ms);
    assert(first >= 0 && last < n);
    if (ts[last] <= range_start_ms) return NhRangeStatus::NoSample;
    if (smoothed && ts[first] > range_end_ms) return NhRangeStatus::NoSample;
    // validateHistogramRange: a window whose ladder changes kind has no common
    // basis. (Its other job is the not-a-counter/not-a-gauge WARNING, which
    // this engine has no annotation channel for and which changes no value.)
    const bool custom = nh_uses_custom(points[first]);
    for (int32_t i = first; i <= last; ++i)                      // bounded
        if (nh_uses_custom(points[i]) != custom) return NhRangeStatus::NoSample;

    NativeHistogram* left  = &scratch[0];
    NativeHistogram* right = &scratch[1];
    NativeHistogram* corr  = &scratch[2];
    if (smoothed && ts[first] < range_start_ms) {
        assert(first + 1 <= last);      // ts[last] > range_start > ts[first]
        const NhRangeStatus s = nh_interpolate(points[first], ts[first],
                                               points[first + 1], ts[first + 1],
                                               range_start_ms, is_counter, left);
        if (s != NhRangeStatus::Ok) return s;
    } else { *left = *points[first]; }
    if (smoothed && last > 0 && ts[last] > range_end_ms) {
        const NhRangeStatus s = nh_interpolate(points[last - 1], ts[last - 1],
                                               points[last], ts[last],
                                               range_end_ms, is_counter, right);
        if (s != NhRangeStatus::Ok) return s;
    } else { *right = *points[last]; }

    *out = *right;
    if (!nh_combine(out, left, /*subtract=*/true)) return NhRangeStatus::Refused;
    nh_adjust_reset_hint(out, left);
    if (is_counter) {
        bool have = false;
        const NhRangeStatus s = detail::nh_reset_correction(
            ts, points, first, last, left, right, range_start_ms, smoothed,
            corr, &have);
        if (s != NhRangeStatus::Ok) return s;
        if (have && !nh_add_adjusted(out, corr)) return NhRangeStatus::Refused;
    }
    if (is_rate) {
        const double range_seconds =
            static_cast<double>(range_end_ms - range_start_ms) / 1000.0;
        if (!(range_seconds > 0.0)) return NhRangeStatus::NoSample;
        nh_div(out, range_seconds);   // Div, not Mul(1/x): see promql_nh_rate.h
    }
    out->reset_hint = static_cast<uint8_t>(NhResetHint::Gauge);
    assert(out->n_pos <= k_nh_max_side && out->n_neg <= k_nh_max_side);
    return NhRangeStatus::Ok;
}

// ---------------------------------------------------------------------------
// The `smoothed` BARE VECTOR SELECTOR over histograms (promql/engine.go). Not
// a range function: the answer is the series interpolated AT the eval instant.
//
// `isCounter` here is upstream's own rule and it is the inverse of the obvious
// one: a pair is treated as a COUNTER unless BOTH samples explicitly carry the
// gauge hint. Getting it backwards silently blends across a reset.
//
// UNORACLED (W13-L2). Flipping that `||` to `&&` leaves the whole board
// BIT-IDENTICAL, because the two rules differ only when EXACTLY ONE of the
// bracketing samples carries the gauge hint and no vendored case has that
// shape. The `||` is what promql/engine.go does; this corpus does not
// discriminate it, and no oracle here can — `anchored`/`smoothed` postdate the
// Prometheus v3.7.3 container and GreptimeDB does not implement them.
//
// Past the last sample the previous value is carried forward with the hint
// RESET TO UNKNOWN — the hint describes the relationship between consecutive
// samples, not the value, and carrying `CounterReset` forward would make the
// next comparison fire on a sample that never reset.
// ---------------------------------------------------------------------------
inline NhRangeStatus nh_smoothed_instant(const int64_t* BOLT_RESTRICT ts,
                                         const NativeHistogram* const* points,
                                         int32_t n, int64_t at_ms,
                                         NativeHistogram* BOLT_RESTRICT out) noexcept {
    assert(ts != nullptr || n == 0);
    assert(points != nullptr || n == 0);
    assert(out != nullptr);
    if (n <= 0) return NhRangeStatus::NoSample;
    const int32_t i = detail::nh_search_ge(ts, n, at_ms);
    if (i < n && ts[i] == at_ms) { *out = *points[i]; return NhRangeStatus::Ok; }
    if (i > 0 && i < n) {
        const NativeHistogram* prev = points[i - 1];
        const NativeHistogram* next = points[i];
        assert(prev != nullptr && next != nullptr);
        if (nh_uses_custom(prev) != nh_uses_custom(next))
            return NhRangeStatus::NoSample;
        constexpr uint8_t gauge = static_cast<uint8_t>(NhResetHint::Gauge);
        const bool is_counter =
            prev->reset_hint != gauge || next->reset_hint != gauge;
        return nh_interpolate(prev, ts[i - 1], next, ts[i], at_ms, is_counter, out);
    }
    if (i > 0) {                                  // no next sample yet
        *out = *points[i - 1];
        out->reset_hint = static_cast<uint8_t>(NhResetHint::Unknown);
        return NhRangeStatus::Ok;
    }
    assert(i == 0);
    return NhRangeStatus::NoSample;               // nothing at or before at_ms
}

}  // namespace promql
}  // namespace bolt
