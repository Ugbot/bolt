// bolt/kernels/promql_nh_rate.h — PromQL RANGE functions over NATIVE
// histograms: rate / increase / delta / irate / idelta / resets / changes.
//
// promql_rate.h does the same seven functions over FLOAT samples. This is the
// histogram half, and it is not the same arithmetic wearing a different type:
//
//   * A reset is not "the count went down". It is upstream's four-part
//     DetectReset (explicit CounterResetHint, observation count, zero bucket
//     under a possibly-different threshold, and every bucket compared on a
//     COMMON schema) — see promql_native_histogram.h :: nh_detect_reset.
//   * Two histograms at DIFFERENT schemas cannot be compared or subtracted at
//     all until both are coarsened to the lower one. Skipping that alignment
//     makes every bucket-level difference quietly wrong.
//   * rate/increase/delta/irate/idelta return a HISTOGRAM, not a float, and
//     resets/changes return a float over histogram INPUTS.
//
// SEMANTICS are a port of prometheus **v3.12.0** — the version whose
// `promql/promqltest/testdata/native_histograms.test` this tree vendors (the
// two differ by one comment typo). Named sources:
//   promql/functions.go  extrapolatedRate, histogramRate, instantValue,
//                        funcResets, funcChanges
//   model/histogram/float_histogram.go  DetectReset, CopyToSchema, Add, Sub
//
// THREE OUTCOMES, deliberately, because two would be a wrong answer:
//   Ok       — a value (or histogram) was produced.
//   NoSample — Prometheus produces NOTHING for this input and says so with a
//              warning: fewer than two samples, a zero sampling interval, or a
//              window mixing exponential and custom-bucket histograms. An
//              empty result here is the CORRECT answer, not a failure.
//   Refused  — this kernel cannot answer EXACTLY. Upstream can (it reconciles
//              mismatched NHCB bounds and differing zero thresholds); this
//              kernel does not, and a guessed reset is a silent wrong rate on
//              every histogram latency panel. The caller must decline the
//              query, never emit a partial answer and never report empty —
//              "no series" and "a series Prometheus would have returned" are
//              different claims.
//
// Tiger Style: noexcept, >= 2 asserts/fn, fixed-bound loops, no allocation
// (caller supplies every histogram), explicit integer sizes, fns <= 70 lines.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/promql_native_histogram.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace bolt {
namespace promql {

enum class NhRangeStatus : uint8_t { Ok = 0, NoSample = 1, Refused = 2 };

// ---------------------------------------------------------------------------
// histogramRate — the counter-reset-corrected difference across the window,
// BEFORE boundary extrapolation. `points[i]` must be non-null and ascending in
// time; `n >= 2`.
// ---------------------------------------------------------------------------
inline NhRangeStatus nh_histogram_rate(const NativeHistogram* const* points,
                                       int32_t n, bool is_counter,
                                       NativeHistogram* BOLT_RESTRICT out) noexcept {
    assert(points != nullptr && out != nullptr);
    assert(n >= 2);
    const NativeHistogram* prev = points[0];
    const NativeHistogram* last = points[n - 1];
    assert(prev != nullptr && last != nullptr);
    bool using_custom = nh_uses_custom(prev);

    // Upstream nulls out the 1st sample when there is a reset between the 1st
    // and the 2nd: its bucket layout is then irrelevant, so an incompatibility
    // there must NOT decide the query. `zeroed` stands in for it, and being
    // empty it contributes nothing to any Sub/Add — which is why those are
    // SKIPPED rather than run against a zero-threshold that would not match.
    NativeHistogram zeroed;
    bool prev_nulled = false;
    if (is_counter) {
        const NativeHistogram* second = points[1];
        assert(second != nullptr);
        const NhReset r = nh_detect_reset(second, prev);
        if (r == NhReset::Undecidable) return NhRangeStatus::Refused;
        if (r == NhReset::Yes) {
            nh_init(&zeroed);
            zeroed.schema = second->schema;
            zeroed.n_custom = second->n_custom;
            for (int32_t i = 0; i < second->n_custom; ++i)   // bounded
                zeroed.custom[i] = second->custom[i];
            prev = &zeroed;
            prev_nulled = true;
            using_custom = nh_uses_custom(second);
        }
    }
    // A window that mixes exponential and custom-bucket histograms has no
    // common ladder; upstream warns and emits nothing.
    if (nh_uses_custom(last) != using_custom) return NhRangeStatus::NoSample;

    int32_t min_schema = last->schema < prev->schema ? last->schema : prev->schema;
    for (int32_t i = 1; i + 1 < n; ++i) {                    // bounded by n
        const NativeHistogram* curr = points[i];
        assert(curr != nullptr);
        if (!is_counter) continue;      // delta looks only at the endpoints
        if (curr->schema < min_schema) min_schema = curr->schema;
        if (nh_uses_custom(curr) != using_custom) return NhRangeStatus::NoSample;
    }
    if (!nh_copy_to_schema(out, last, min_schema)) return NhRangeStatus::Refused;
    if (!prev_nulled && !nh_combine(out, prev, /*subtract=*/true))
        return NhRangeStatus::Refused;
    if (is_counter) {
        for (int32_t i = 1; i < n; ++i) {                    // bounded by n
            const NativeHistogram* curr = points[i];
            const NhReset r = nh_detect_reset(curr, prev);
            if (r == NhReset::Undecidable) return NhRangeStatus::Refused;
            if (r == NhReset::Yes && prev != &zeroed &&
                !nh_combine(out, prev, /*subtract=*/false))
                return NhRangeStatus::Refused;
            prev = curr;
        }
    }
    out->reset_hint = static_cast<uint8_t>(NhResetHint::Gauge);
    // Upstream finishes with `h.Compact(0)`, which DELETES empty bucket runs
    // and can open interior gaps. This representation is dense and cannot hold
    // a gap, so compaction is not modelled — and it changes no answer, because
    // an absent bucket and an explicit zero are the same distribution to every
    // consumer here (quantile, fraction and variance all skip a zero-count
    // bucket, and the corpus comparison reads both sides by absolute index).
    assert(out->n_pos <= k_nh_max_side && out->n_neg <= k_nh_max_side);
    return NhRangeStatus::Ok;
}

// ---------------------------------------------------------------------------
// extrapolatedRate over histograms: rate (is_counter, is_rate),
// increase (is_counter), delta (neither). The boundary extrapolation is the
// SAME arithmetic promql_rate.h applies to floats — the one histogram-specific
// term is the zero-crossing cap, which reads the observation COUNT rather than
// the value.
// ---------------------------------------------------------------------------
inline NhRangeStatus nh_extrapolated_rate(const int64_t* BOLT_RESTRICT ts,
                                          const NativeHistogram* const* points,
                                          int32_t n,
                                          int64_t range_start_ms,
                                          int64_t range_end_ms,
                                          bool is_counter, bool is_rate,
                                          NativeHistogram* BOLT_RESTRICT out) noexcept {
    assert(ts != nullptr && out != nullptr);
    assert(range_end_ms >= range_start_ms);
    if (n < 2) return NhRangeStatus::NoSample;
    const NhRangeStatus st = nh_histogram_rate(points, n, is_counter, out);
    if (st != NhRangeStatus::Ok) return st;

    const int64_t first_t = ts[0], last_t = ts[n - 1];
    const double sampled_interval = static_cast<double>(last_t - first_t) / 1000.0;
    if (sampled_interval <= 0.0) return NhRangeStatus::NoSample;
    const double avg_interval = sampled_interval / static_cast<double>(n - 1);
    const double threshold = avg_interval * 1.1;
    double to_start = static_cast<double>(first_t - range_start_ms) / 1000.0;
    double to_end   = static_cast<double>(range_end_ms - last_t) / 1000.0;
    if (to_start >= threshold) to_start = avg_interval / 2.0;
    // `points[0]->count >= 0.0` and not `!(… < 0.0)`: a NaN count must fail
    // this test the way Go's `>=` does, or a NaN histogram would take the
    // zero-crossing branch upstream never takes.
    if (is_counter && out->count > 0.0 && points[0]->count >= 0.0) {
        const double to_zero = sampled_interval * (points[0]->count / out->count);
        if (to_zero < to_start) to_start = to_zero;
    }
    if (to_end >= threshold) to_end = avg_interval / 2.0;
    double factor = (sampled_interval + to_start + to_end) / sampled_interval;
    if (is_rate) {
        const double range_seconds =
            static_cast<double>(range_end_ms - range_start_ms) / 1000.0;
        if (!(range_seconds > 0.0)) return NhRangeStatus::NoSample;
        factor /= range_seconds;
    }
    nh_mul(out, factor);
    assert(out->n_pos <= k_nh_max_side);
    return NhRangeStatus::Ok;
}

// ---------------------------------------------------------------------------
// instantValue over histograms: irate (is_rate) / idelta. Only the last two
// samples of the window matter, and a reset makes the LAST sample itself the
// answer — exactly as it does for floats.
// ---------------------------------------------------------------------------
inline NhRangeStatus nh_instant_value(const int64_t* BOLT_RESTRICT ts,
                                      const NativeHistogram* a,
                                      const NativeHistogram* b,
                                      bool is_rate,
                                      NativeHistogram* BOLT_RESTRICT out) noexcept {
    assert(ts != nullptr && out != nullptr);
    assert(a != nullptr && b != nullptr);
    const int64_t dt_ms = ts[1] - ts[0];
    if (dt_ms == 0) return NhRangeStatus::NoSample;
    *out = *b;
    bool subtract = true;
    if (is_rate) {
        const NhReset r = nh_detect_reset(b, a);
        if (r == NhReset::Undecidable) return NhRangeStatus::Refused;
        subtract = (r == NhReset::No);
    }
    // An exponential and a custom-bucket histogram have no common ladder, so
    // the subtraction is impossible rather than merely unimplemented: upstream
    // gets `ErrHistogramsIncompatibleSchema` back from Sub, warns, and emits
    // NOTHING (functions.test:350 — `idelta(…{path="/f"}[20m])` asserts an
    // empty result). Two NHCBs with DIFFERENT ladders are the other case and
    // are not this one: upstream reconciles those, so they refuse below.
    if (subtract && nh_uses_custom(a) != nh_uses_custom(out))
        return NhRangeStatus::NoSample;
    if (subtract && !nh_combine(out, a, /*subtract=*/true))
        return NhRangeStatus::Refused;
    out->reset_hint = static_cast<uint8_t>(NhResetHint::Gauge);
    if (is_rate) nh_div(out, static_cast<double>(dt_ms) / 1000.0);
    assert(out->n_pos <= k_nh_max_side && out->n_neg <= k_nh_max_side);
    return NhRangeStatus::Ok;
}

// ---------------------------------------------------------------------------
// resets() / changes() over a window that may hold BOTH floats and histograms.
// `hist[i] == nullptr` means sample i is an ordinary float carried in `val[i]`;
// the arrays are one merged, ascending-by-timestamp sequence, which is the same
// order upstream produces by interleaving its two per-type slices.
//
// A transition between a float and a histogram counts for BOTH functions:
// upstream treats a type change as a reset and as a change.
// ---------------------------------------------------------------------------
inline NhRangeStatus nh_resets(const double* BOLT_RESTRICT val,
                               const NativeHistogram* const* hist,
                               int32_t n, int64_t* BOLT_RESTRICT out) noexcept {
    assert(val != nullptr || n == 0);
    assert(hist != nullptr || n == 0);
    assert(out != nullptr);
    int64_t resets = 0;
    for (int32_t i = 1; i < n; ++i) {                        // bounded by n
        const NativeHistogram* p = hist[i - 1];
        const NativeHistogram* c = hist[i];
        if (p == nullptr && c == nullptr) {
            if (val[i] < val[i - 1]) ++resets;
        } else if (p == nullptr || c == nullptr) {
            ++resets;
        } else {
            const NhReset r = nh_detect_reset(c, p);
            if (r == NhReset::Undecidable) return NhRangeStatus::Refused;
            if (r == NhReset::Yes) ++resets;
        }
    }
    *out = resets;
    assert(resets >= 0 && resets < n + 1);
    return NhRangeStatus::Ok;
}

inline NhRangeStatus nh_changes(const double* BOLT_RESTRICT val,
                                const NativeHistogram* const* hist,
                                int32_t n, int64_t* BOLT_RESTRICT out) noexcept {
    assert(val != nullptr || n == 0);
    assert(hist != nullptr || n == 0);
    assert(out != nullptr);
    int64_t changes = 0;
    for (int32_t i = 1; i < n; ++i) {                        // bounded by n
        const NativeHistogram* p = hist[i - 1];
        const NativeHistogram* c = hist[i];
        if (p == nullptr && c == nullptr) {
            const double a = val[i - 1], b = val[i];
            if (a != b && !(std::isnan(a) && std::isnan(b))) ++changes;
        } else if (p == nullptr || c == nullptr) {
            ++changes;
        } else if (!nh_equals(c, p)) {
            ++changes;
        }
    }
    *out = changes;
    assert(changes >= 0 && changes < n + 1);
    return NhRangeStatus::Ok;
}

}  // namespace promql
}  // namespace bolt
