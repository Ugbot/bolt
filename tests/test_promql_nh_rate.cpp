// test_promql_nh_rate.cpp — oracle test for the NATIVE-histogram range
// functions (rate / increase / delta / irate / idelta / resets / changes) and
// for the counter-reset detection underneath them.
//
// THE ORACLE IS PROMETHEUS'S OWN VENDORED CORPUS, cited by file and line:
//   chukonu/tests/promql/testdata/upstream/native_histograms.test
//   chukonu/tests/promql/testdata/upstream/functions.test
// Both are READ-ONLY here. Every `{{...}}` literal and every expected number
// below is transcribed from the cited line, so this file reads as a copy of the
// answer key rather than a record of what this implementation returns.
//
// WHICH PROMETHEUS. The vendored `native_histograms.test` matches upstream
// **v3.12.0** to within one comment typo (v3.11.x differ by 12 lines, v3.7.3 —
// the version the docker differential oracle runs — by 1,095, i.e. 842 lines
// shorter). v3.12.0 is therefore the version this kernel ports, and where the
// live v3.7.3 oracle disagrees with the vendored key, the key wins.
//
// A WRONG COUNTER-RESET DETECTION IS A SILENT WRONG RATE, never an error, on
// every histogram latency panel — so the cases below deliberately include the
// ones where a plausible shortcut gives a plausible wrong answer:
//   * `reset_in_bucket` — the TOTAL count RISES at every step while one bucket
//     falls. "The count went down" detects nothing here and returns 6 instead
//     of the corpus's 9.
//   * an explicit `counter_reset_hint` that CONTRADICTS the counts, in both
//     directions.
//   * two histograms at DIFFERENT schemas, which cannot be compared bucket for
//     bucket until the finer one is coarsened.

#include <gtest/gtest.h>

#include "bolt/kernels/promql.h"

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>

using bolt::promql::NativeHistogram;
using bolt::promql::NhRangeStatus;
using bolt::promql::NhReset;
using bolt::promql::NhResetHint;
using bolt::promql::k_nh_schema_custom;
using bolt::promql::nh_changes;
using bolt::promql::nh_detect_reset;
using bolt::promql::nh_equals;
using bolt::promql::nh_extrapolated_rate;
using bolt::promql::nh_init;
using bolt::promql::nh_instant_value;
using bolt::promql::nh_resets;

namespace {

// Build from the `{{...}}` fields exactly as the corpus writes them; a
// `buckets:` list's first entry is bucket index 0 unless an offset is given,
// which is the indexing native_histograms.test:111 states in prose.
NativeHistogram mk(int32_t schema, double sum, double count,
                   std::initializer_list<double> pos,
                   std::initializer_list<double> neg = {},
                   double z_count = 0.0, double z_width = 0.0,
                   std::initializer_list<double> custom = {},
                   NhResetHint hint = NhResetHint::Unknown,
                   int32_t pos_offset = 0) {
    NativeHistogram h;
    nh_init(&h);
    h.schema = schema;
    h.sum = sum;
    h.count = count;
    h.zero_count = z_count;
    h.zero_threshold = z_width;
    h.reset_hint = static_cast<uint8_t>(hint);
    h.pos_offset = pos_offset;
    h.n_pos = static_cast<int32_t>(pos.size());
    int32_t i = 0;
    for (double v : pos) h.pos[i++] = v;
    i = 0;
    h.n_neg = static_cast<int32_t>(neg.size());
    for (double v : neg) h.neg[i++] = v;
    i = 0;
    h.n_custom = static_cast<int32_t>(custom.size());
    for (double v : custom) h.custom[i++] = v;
    return h;
}

void ExpectClose(double got, double want) {
    if (std::isnan(want)) { EXPECT_TRUE(std::isnan(got)); return; }
    EXPECT_NEAR(got, want, std::fabs(want) * 1e-12 + 1e-12);
}

// Compare against a corpus `{{...}}` literal the way the corpus's own
// comparison does: by the bucket SEQUENCE at absolute indices, so an implicit
// (absent) bucket and an explicit zero are the same distribution.
double BucketAt(const NativeHistogram& h, int32_t idx) {
    if (idx < h.pos_offset || idx >= h.pos_offset + h.n_pos) return 0.0;
    return h.pos[idx - h.pos_offset];
}

void ExpectHistogram(const NativeHistogram& got, int32_t schema, double sum,
                     double count, std::initializer_list<double> pos,
                     std::initializer_list<double> custom = {}) {
    EXPECT_EQ(got.schema, schema);
    ExpectClose(got.sum, sum);
    ExpectClose(got.count, count);
    EXPECT_EQ(got.n_custom, static_cast<int32_t>(custom.size()));
    int32_t i = 0;
    for (double v : custom) ExpectClose(got.custom[i++], v);
    const int32_t hi = static_cast<int32_t>(pos.size());
    i = 0;
    for (double v : pos) ExpectClose(BucketAt(got, i++), v);
    // Nothing outside the expected run: a fabricated bucket is as wrong as a
    // dropped one, and comparing only the expected span cannot see it.
    for (int32_t j = got.pos_offset; j < got.pos_offset + got.n_pos; ++j)
        if (j < 0 || j >= hi) ExpectClose(got.pos[j - got.pos_offset], 0.0);
}

}  // namespace

// ---------------------------------------------------------------------------
// nh_detect_reset — the four independent things upstream checks.
// ---------------------------------------------------------------------------
TEST(PromqlNhReset, HintOverridesTheCounts) {
    // float_histogram.go :: DetectReset shortcuts on the hint BEFORE looking at
    // anything else, in both directions. A detector that only compared counts
    // would answer the opposite on both of these.
    const NativeHistogram grew = mk(0, 2, 2, {2});
    const NativeHistogram base = mk(0, 1, 1, {1});
    NativeHistogram grew_but_reset = grew;
    grew_but_reset.reset_hint = static_cast<uint8_t>(NhResetHint::CounterReset);
    EXPECT_EQ(nh_detect_reset(&grew_but_reset, &base), NhReset::Yes);

    NativeHistogram shrank_but_not = mk(0, 0.5, 0.5, {0.5});
    shrank_but_not.reset_hint = static_cast<uint8_t>(NhResetHint::NotReset);
    EXPECT_EQ(nh_detect_reset(&shrank_but_not, &base), NhReset::No);

    // Gauge is NOT a shortcut: PromQL still runs counter functions over gauge
    // histograms (warning, not refusal), so the detection must proceed.
    // native_histograms.test:1093-1099 asserts a real rate for a gauge series.
    NativeHistogram gauge = grew;
    gauge.reset_hint = static_cast<uint8_t>(NhResetHint::Gauge);
    EXPECT_EQ(nh_detect_reset(&gauge, &base), NhReset::No);
}

TEST(PromqlNhReset, SingleBucketFallWithARisingTotal) {
    // native_histograms.test:1059-1061 — "Counter reset only noticeable in a
    // single bucket". count goes 4 -> 5 (UP) while bucket 1 goes 2 -> 1.
    const NativeHistogram a = mk(0, 5, 4, {1, 2, 1});
    const NativeHistogram b = mk(0, 6, 5, {1, 1, 3});
    EXPECT_EQ(nh_detect_reset(&b, &a), NhReset::Yes);
    // ... and the step after it is NOT a reset (every bucket grows or holds).
    const NativeHistogram c = mk(0, 7, 6, {1, 2, 3});
    EXPECT_EQ(nh_detect_reset(&c, &b), NhReset::No);
}

TEST(PromqlNhReset, SchemaAndZeroThresholdRules) {
    const NativeHistogram coarse = mk(0, 5, 4, {1, 2, 1});
    // An increase of resolution can only happen together with a reset.
    NativeHistogram finer = mk(1, 5, 4, {1, 2, 1});
    EXPECT_EQ(nh_detect_reset(&finer, &coarse), NhReset::Yes);
    // A DECREASE of resolution is ordinary: the previous histogram is merged
    // onto the coarser ladder before comparing. schema 1 [0 2 1] reduces to
    // schema 0 {0:0, 1:3} — the reduction native_histograms.test pins through
    // `histogram_sub_2`. Against schema 0 [0 3] that is not a reset; against
    // schema 0 [0 2] one bucket fell, so it is.
    const NativeHistogram prev_fine = mk(1, 5, 3, {0, 2, 1});
    const NativeHistogram same_mass = mk(0, 5, 3, {0, 3});
    EXPECT_EQ(nh_detect_reset(&same_mass, &prev_fine), NhReset::No);
    const NativeHistogram lost_mass = mk(0, 5, 3, {0, 2});
    EXPECT_EQ(nh_detect_reset(&lost_mass, &prev_fine), NhReset::Yes);

    // A decrease of the zero threshold can only happen together with a reset.
    const NativeHistogram wide = mk(0, 5, 4, {1, 2, 1}, {}, 1.0, 0.01);
    const NativeHistogram narrow = mk(0, 9, 8, {1, 2, 1}, {}, 4.0, 0.001);
    EXPECT_EQ(nh_detect_reset(&narrow, &wide), NhReset::Yes);
    // Widening it is not, when the previous histogram's mass that falls inside
    // the new threshold is accounted for.
    EXPECT_EQ(nh_detect_reset(&wide, &wide), NhReset::No);
}

TEST(PromqlNhReset, CustomBucketLadders) {
    // functions.test:227 — a float/exponential sample followed by an NHCB is a
    // reset (the schema jumped from -53 to 0's ladder, or vice versa).
    const NativeHistogram expo = mk(0, 1, 1, {});
    const NativeHistogram nhcb = mk(k_nh_schema_custom, 3, 3, {3}, {}, 0, 0, {5, 10});
    EXPECT_EQ(nh_detect_reset(&nhcb, &expo), NhReset::Yes);
    // Same ladder, growing: no reset. native_histograms.test:1291 (the first
    // two nhcb_metric samples are identical).
    const NativeHistogram same = mk(k_nh_schema_custom, 1, 1, {1}, {}, 0, 0, {5});
    EXPECT_EQ(nh_detect_reset(&same, &same), NhReset::No);
    // MISMATCHED ladders: upstream reconciles them bucket by bucket
    // (detectResetWithMismatchedCustomBounds) and RETURNS an answer. This pin
    // demanded `Undecidable` until W25 — a refusal is honest only while the
    // kernel really cannot decide, and asserting one for a shape upstream
    // decides is what kept 22 corpus cells RED.
    //
    // Re-derived from Prometheus's OWN expectation rather than from this
    // kernel: native_histograms.test:1334 loads exactly these three samples
    // ([5], [5], [5 10], all sum 1 count 1 buckets [1]) and asserts
    // `resets(nhcb_metric[13m])` -> **0**. On the intersected ladder {5} both
    // histograms hold their whole mass in the one bounded bucket, so nothing
    // decreased. promtool 3.12.0 agrees; 3.7.3 answers 1 here and is a
    // characterised divergence (see nhcb_reconcile_oracle.py).
    const NativeHistogram other = mk(k_nh_schema_custom, 1, 1, {1}, {}, 0, 0, {5, 10});
    EXPECT_EQ(nh_detect_reset(&other, &same), NhReset::No);
    // The negative control, and the one this cell CANNOT supply: the corpus's
    // only `resets` over mismatched ladders expects 0, so "never a reset"
    // scores full marks there. Drop the mass that survives reconciliation and
    // the same pair must answer Yes — otherwise a missed reset is a silently
    // wrong `rate` on every panel.
    const NativeHistogram drained = mk(k_nh_schema_custom, 1, 1, {0.5}, {}, 0, 0, {5, 10});
    EXPECT_EQ(nh_detect_reset(&drained, &same), NhReset::Yes);
}

// ---------------------------------------------------------------------------
// rate / increase / delta.
// ---------------------------------------------------------------------------
TEST(PromqlNhRate, IncreaseOverAResetInOneBucket) {
    // native_histograms.test:1059-1065.
    //   load 5m
    //     reset_in_bucket {{schema:0 count:4 sum:5 buckets:[1 2 1]}}
    //                     {{schema:0 count:5 sum:6 buckets:[1 1 3]}}
    //                     {{schema:0 count:6 sum:7 buckets:[1 2 3]}}
    //   eval instant at 10m increase(reset_in_bucket[15m])
    //     {} {{count:9 sum:10.5 buckets:[1.5 3 4.5]}}
    const NativeHistogram s0 = mk(0, 5, 4, {1, 2, 1});
    const NativeHistogram s1 = mk(0, 6, 5, {1, 1, 3});
    const NativeHistogram s2 = mk(0, 7, 6, {1, 2, 3});
    const NativeHistogram* pts[3] = { &s0, &s1, &s2 };
    const int64_t ts[3] = { 0, 300000, 600000 };
    NativeHistogram out;
    // window (10m - 15m, 10m] = (-5m, 10m]
    ASSERT_EQ(nh_extrapolated_rate(ts, pts, 3, -300000, 600000,
                                   /*is_counter=*/true, /*is_rate=*/false, &out),
              NhRangeStatus::Ok);
    ExpectHistogram(out, 0, 10.5, 9.0, {1.5, 3.0, 4.5});
    // The result is a GAUGE histogram, as upstream's histogramRate stamps it.
    EXPECT_EQ(out.reset_hint, static_cast<uint8_t>(NhResetHint::Gauge));

    // Discriminating power: without the reset, the answer is 6/7.5 and not
    // 9/10.5, so this case can tell a working detector from a missing one.
    NativeHistogram s1_nr = s1;
    s1_nr.reset_hint = static_cast<uint8_t>(NhResetHint::NotReset);
    const NativeHistogram* pts_nr[3] = { &s0, &s1_nr, &s2 };
    ASSERT_EQ(nh_extrapolated_rate(ts, pts_nr, 3, -300000, 600000, true, false, &out),
              NhRangeStatus::Ok);
    ExpectClose(out.count, 3.0);   // (6 - 4) extrapolated by 1.5
}

TEST(PromqlNhRate, ConstantBucketsRateToZero) {
    // native_histograms.test:1147-1152. Five identical samples 1m apart:
    //   eval instant at 5m rate(const_histogram[5m])  ->  {{schema:0 sum:0 count:0}}
    const NativeHistogram s = mk(0, 1, 1, {1, 1, 1});
    const NativeHistogram* pts[4] = { &s, &s, &s, &s };
    const int64_t ts[4] = { 60000, 120000, 180000, 240000 };
    NativeHistogram out;
    ASSERT_EQ(nh_extrapolated_rate(ts, pts, 4, 0, 300000, true, true, &out),
              NhRangeStatus::Ok);
    ExpectHistogram(out, 0, 0.0, 0.0, {0.0, 0.0, 0.0});
}

TEST(PromqlNhRate, GaugeHintedSeriesStillRates) {
    // native_histograms.test:1090-1099. `counter_reset_hint:gauge` earns a
    // warning, not a refusal, and the numbers are the ordinary ones.
    const NativeHistogram a = mk(0, 1, 1, {1}, {}, 0, 0, {}, NhResetHint::Gauge);
    const NativeHistogram b = mk(0, 2, 2, {2}, {}, 0, 0, {}, NhResetHint::Gauge);
    const NativeHistogram* pts[2] = { &a, &b };
    const int64_t ts[2] = { 0, 30000 };
    NativeHistogram out;
    // eval instant at 30s rate(some_metric[1m])  ->  window (-30s, 30s]
    ASSERT_EQ(nh_extrapolated_rate(ts, pts, 2, -30000, 30000, true, true, &out),
              NhRangeStatus::Ok);
    ExpectHistogram(out, 0, 0.03333333333333333, 0.03333333333333333,
                    {0.03333333333333333});
}

TEST(PromqlNhRate, MixedExponentialAndCustomBuckets) {
    // native_histograms.test:1107-1145. One load block, four evals, three
    // different outcomes — which is what makes it a good discriminator.
    const NativeHistogram e0 = mk(0, 1, 1, {1});
    const NativeHistogram c1 = mk(k_nh_schema_custom, 1, 1, {1}, {}, 0, 0, {5, 10});
    const NativeHistogram e2 = mk(0, 5, 4, {1, 2, 1});
    const int64_t ts3[3] = { 0, 30000, 60000 };
    NativeHistogram out;

    // (a) :1112 — exponential at both ends, custom in the middle: upstream
    // warns and produces NO RESULT. That is an empty answer, not a refusal.
    {
        const NativeHistogram* pts[3] = { &e0, &c1, &e2 };
        EXPECT_EQ(nh_extrapolated_rate(ts3, pts, 3, -30000, 60000, true, true, &out),
                  NhRangeStatus::NoSample);
    }
    // (b) :1131 — start custom, end exponential. The 1st sample is nulled out
    // by the reset between it and the 2nd, so only its COUNT survives, and it
    // survives through the zero-crossing cap: 30s * (1/4) = 7.5s.
    //   {} {{count:0.0833... sum:0.104166... buckets:[0.0208... 0.0416... 0.0208...]}}
    {
        const NativeHistogram* pts[2] = { &c1, &e2 };
        const int64_t ts[2] = { 30000, 60000 };
        ASSERT_EQ(nh_extrapolated_rate(ts, pts, 2, 0, 60000, true, true, &out),
                  NhRangeStatus::Ok);
        ExpectHistogram(out, 0, 0.10416666666666666, 0.08333333333333333,
                        {0.020833333333333332, 0.041666666666666664,
                         0.020833333333333332});
    }
    // (c) :1138 — start exponential, end custom: the custom histogram divided
    // by 30, custom_values carried through.
    {
        const NativeHistogram* pts[2] = { &e0, &c1 };
        const int64_t ts[2] = { 0, 30000 };
        ASSERT_EQ(nh_extrapolated_rate(ts, pts, 2, -30000, 30000, true, true, &out),
                  NhRangeStatus::Ok);
        ExpectHistogram(out, k_nh_schema_custom, 0.03333333333333333,
                        0.03333333333333333, {0.03333333333333333}, {5, 10});
    }
}

TEST(PromqlNhRate, MismatchedCustomBoundsAreReconciledNotRefused) {
    // native_histograms.test:1291 + :1326 — nhcb_metric's third sample carries
    // custom_values [5 10] where the first two carry [5]. Upstream RECONCILES
    // the ladders during Sub, onto their intersection {5}; this pin demanded a
    // refusal until W25 and that refusal was the defect.
    //
    // The expectation is Prometheus's own, not this kernel's:
    // native_histograms.test:1326 asserts `rate(nhcb_metric[13m])` ->
    // `{{schema:-53 custom_values:[5] }}` — schema custom, ladder {5}, and
    // count/sum/buckets all ZERO, because on the intersected ladder every
    // sample holds mass 1 in the one bounded bucket and the delta is exactly
    // nothing. :1321 and :1324 assert the same value for `delta` and
    // `increase`. Asserting the LADDER as well as the zeros matters: a kernel
    // that returned an exponential zero histogram, or an empty ladder, would
    // also be "all zeros".
    const NativeHistogram a = mk(k_nh_schema_custom, 1, 1, {1}, {}, 0, 0, {5});
    const NativeHistogram c = mk(k_nh_schema_custom, 1, 1, {1}, {}, 0, 0, {5, 10});
    const NativeHistogram* pts[3] = { &a, &a, &c };
    const int64_t ts[3] = { 0, 360000, 720000 };
    NativeHistogram out;
    ASSERT_EQ(nh_extrapolated_rate(ts, pts, 3, -60000, 720000, true, true, &out),
              NhRangeStatus::Ok);
    ExpectHistogram(out, k_nh_schema_custom, 0.0, 0.0, {}, {5});
}

// ---------------------------------------------------------------------------
// irate / idelta.
// ---------------------------------------------------------------------------
TEST(PromqlNhInstant, IrateOverTheLastTwoSamples) {
    // functions.test:222-273, load 5m so the sampled interval is 300s.
    NativeHistogram out;
    const int64_t ts[2] = { 600000, 900000 };
    // :245 `/a` — {{sum:2 count:2}}+{{sum:3 count:3}}x5 -> 14 then 17.
    {
        const NativeHistogram a = mk(0, 14, 14, {});
        const NativeHistogram b = mk(0, 17, 17, {});
        ASSERT_EQ(nh_instant_value(ts, &a, &b, /*is_rate=*/true, &out),
                  NhRangeStatus::Ok);
        ExpectHistogram(out, 0, 0.01, 0.01, {});
        EXPECT_EQ(out.reset_hint, static_cast<uint8_t>(NhResetHint::Gauge));
    }
    // :256 `/c` — the newer sample is hinted `gauge`; the detection proceeds
    // anyway and finds no reset, so the ordinary difference is the answer.
    {
        const NativeHistogram a = mk(0, 1, 1, {});
        const NativeHistogram b = mk(0, 4, 4, {}, {}, 0, 0, {}, NhResetHint::Gauge);
        ASSERT_EQ(nh_instant_value(ts, &a, &b, true, &out), NhRangeStatus::Ok);
        ExpectHistogram(out, 0, 0.01, 0.01, {});
    }
    // :267 `/f` — exponential then NHCB. That IS a reset, so there is no
    // subtraction at all: the answer is the newer sample over the interval.
    {
        const NativeHistogram a = mk(0, 1, 1, {});
        const NativeHistogram b = mk(k_nh_schema_custom, 3, 3, {3}, {}, 0, 0, {5, 10});
        ASSERT_EQ(nh_instant_value(ts, &a, &b, true, &out), NhRangeStatus::Ok);
        ExpectHistogram(out, k_nh_schema_custom, 0.01, 0.01, {0.01}, {5, 10});
    }
    // :271 `/g` — two NHCBs with DIFFERENT ladders, and the intersection of
    // {1} and {5 10} is EMPTY. functions.test:273 expects
    // `{{schema:-53 counter_reset_hint:gauge}}`: no custom_values at all, no
    // buckets, count and sum zero. That is the empty-intersection result —
    // one implicit (-Inf, +Inf] bucket into which BOTH samples' mass folds, so
    // the subtraction cancels. Reconciled, not refused; the ladder being EMPTY
    // rather than {1} or {5 10} is the part a count-only assertion would miss.
    {
        const NativeHistogram a = mk(k_nh_schema_custom, 3, 3, {3}, {}, 0, 0, {1});
        const NativeHistogram b = mk(k_nh_schema_custom, 3, 3, {3}, {}, 0, 0, {5, 10});
        ASSERT_EQ(nh_instant_value(ts, &a, &b, true, &out), NhRangeStatus::Ok);
        ExpectHistogram(out, k_nh_schema_custom, 0.0, 0.0, {}, {});
    }
    // idelta always subtracts — it is the gauge form and never reset-corrects.
    {
        const NativeHistogram a = mk(0, 17, 17, {});
        const NativeHistogram b = mk(0, 14, 14, {});
        ASSERT_EQ(nh_instant_value(ts, &a, &b, /*is_rate=*/false, &out),
                  NhRangeStatus::Ok);
        ExpectHistogram(out, 0, -3.0, -3.0, {});
    }
    // functions.test:350 — the SAME `/f` pair under idelta is an empty result,
    // not a refusal: idelta has no reset shortcut, so it reaches a subtraction
    // between an exponential and a custom ladder, which upstream reports as
    // incompatible schemas and answers with nothing. The pair with :267 above
    // is what separates "cannot be computed" from "we cannot compute it".
    {
        const NativeHistogram a = mk(0, 1, 1, {});
        const NativeHistogram b = mk(k_nh_schema_custom, 3, 3, {3}, {}, 0, 0, {5, 10});
        EXPECT_EQ(nh_instant_value(ts, &a, &b, /*is_rate=*/false, &out),
                  NhRangeStatus::NoSample);
    }
}

// ---------------------------------------------------------------------------
// resets / changes — float results over histogram inputs.
// ---------------------------------------------------------------------------
TEST(PromqlNhResetsChanges, TypeTransitionsCountForBoth) {
    // functions.test:6-8 and :47/:90 — a float-to-histogram transition (and the
    // reverse) is BOTH a reset and a change, whatever the numbers say.
    const NativeHistogram h = mk(0, 1, 1, {});
    const double val[4] = { 0.0, 0.0, 0.0, 0.0 };
    const NativeHistogram* hist[4] = { nullptr, &h, nullptr, &h };
    int64_t n = 0;
    ASSERT_EQ(nh_resets(val, hist, 4, &n), NhRangeStatus::Ok);
    EXPECT_EQ(n, 3);
    ASSERT_EQ(nh_changes(val, hist, 4, &n), NhRangeStatus::Ok);
    EXPECT_EQ(n, 3);
}

TEST(PromqlNhResetsChanges, NhcbLadderChangeIsAChangeButNotACountedReset) {
    // native_histograms.test:1318 (`changes(nhcb_metric[13m])` -> 1) and :1334
    // (`resets(nhcb_metric[13m])` -> 0). The three samples are equal-valued;
    // only the third's custom_values differ, so `changes` sees one change...
    const NativeHistogram a = mk(k_nh_schema_custom, 1, 1, {1}, {}, 0, 0, {5});
    const NativeHistogram c = mk(k_nh_schema_custom, 1, 1, {1}, {}, 0, 0, {5, 10});
    const double val[3] = { 0.0, 0.0, 0.0 };
    const NativeHistogram* hist[3] = { &a, &a, &c };
    int64_t n = 0;
    ASSERT_EQ(nh_changes(val, hist, 3, &n), NhRangeStatus::Ok);
    EXPECT_EQ(n, 1);
    // ... while `resets` now performs that reconciliation and answers 0, which
    // is native_histograms.test:1334's own expectation. It used to decline.
    //
    // 0 is also what "never a reset" answers, so this assertion alone proves
    // nothing — the discriminating case is in
    // PromqlNhReset.CustomBucketLadders, where the same mismatched pair with
    // mass actually removed must answer Yes.
    ASSERT_EQ(nh_resets(val, hist, 3, &n), NhRangeStatus::Ok);
    EXPECT_EQ(n, 0);
}

TEST(PromqlNhResetsChanges, EqualityIsDataEqualityNotDistributionEquality) {
    const NativeHistogram a = mk(0, 5, 4, {1, 2, 1});
    NativeHistogram b = a;
    EXPECT_TRUE(nh_equals(&a, &b));
    // The hint is metadata about the sample, not part of its value: upstream's
    // Equals does not read it, so `changes` must not either.
    b.reset_hint = static_cast<uint8_t>(NhResetHint::CounterReset);
    EXPECT_TRUE(nh_equals(&a, &b));
    // A different zero threshold IS a different histogram even at equal mass.
    b = a; b.zero_threshold = 0.001;
    EXPECT_FALSE(nh_equals(&a, &b));
    // NaN == NaN, because the comparison is on bit patterns.
    NativeHistogram n1 = mk(0, std::nan(""), 1, {1});
    NativeHistogram n2 = n1;
    EXPECT_TRUE(nh_equals(&n1, &n2));
}

TEST(PromqlNhResetsChanges, FloatOnlyPathMatchesTheFloatKernel) {
    // The merged array must reproduce the float rules exactly when no histogram
    // is present: `changes` treats NaN -> NaN as no change (functions.test's
    // http_requests_nan cases), and `resets` counts only decreases.
    const double nan_v = std::nan("");
    const double val[5] = { 1.0, nan_v, nan_v, 5.0, 2.0 };
    const NativeHistogram* hist[5] = { nullptr, nullptr, nullptr, nullptr, nullptr };
    int64_t n = 0;
    ASSERT_EQ(nh_changes(val, hist, 5, &n), NhRangeStatus::Ok);
    EXPECT_EQ(n, 3);   // 1->NaN, NaN->5, 5->2 ; NaN->NaN is not a change
    ASSERT_EQ(nh_resets(val, hist, 5, &n), NhRangeStatus::Ok);
    EXPECT_EQ(n, 1);   // only 5 -> 2 ; NaN comparisons are false
}
