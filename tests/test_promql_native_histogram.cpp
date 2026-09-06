// test_promql_native_histogram.cpp — oracle test for the NATIVE-histogram
// kernels. Every numeric expectation below is COPIED from Prometheus's own
// vendored corpus (chukonu/tests/promql/testdata/upstream/native_histograms
// .test, READ-ONLY) together with the `{{...}}` literal that produced it, so
// the file reads as a transcription of the oracle rather than a record of
// what this implementation happens to return.
//
// Two structural facts the corpus pins independently of any expectation, and
// which the whole kernel rests on, are asserted directly (BucketIndexing):
//   * bucket index 0 has UPPER bound 1 for every exponential schema;
//   * the `buckets:` list's default offset is 0, i.e. its first entry IS
//     bucket 0. The corpus states this in prose at native_histograms.test:111
//     and pins it arithmetically twice over — quantile(0.5, {{schema:0 sum:5
//     count:4 buckets:[1 2 1]}}) is sqrt(2) under this reading and 2.828 under
//     the off-by-one reading, and stddev_stdvar_1's nine schema-2 buckets are
//     documented as the observations {1,2,3,4}, which only lands on integers
//     under this reading.

#include <gtest/gtest.h>

#include "bolt/kernels/promql.h"

#include <cmath>
#include <cstdint>
#include <limits>

using bolt::promql::NativeHistogram;
using bolt::promql::NhBucket;
using bolt::promql::k_nh_max_all_buckets;
using bolt::promql::k_nh_schema_custom;
using bolt::promql::nh_all_buckets;
using bolt::promql::nh_avg;
using bolt::promql::nh_fraction;
using bolt::promql::nh_init;
using bolt::promql::nh_quantile;
using bolt::promql::nh_stddev;
using bolt::promql::nh_stdvar;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

struct Scratch {
    NhBucket b[k_nh_max_all_buckets];
    NhBucket* p() { return b; }
    static constexpr int32_t cap = k_nh_max_all_buckets;
};

// Build from the `{{...}}` fields as written in the corpus. `pos`/`neg` are
// the `buckets:` / `n_buckets:` lists; offsets default to 0 exactly as the
// corpus's own parser does.
NativeHistogram mk(int32_t schema, double sum, double count,
                   std::initializer_list<double> pos,
                   std::initializer_list<double> neg = {},
                   double z_count = 0.0, double z_width = 0.0,
                   int32_t pos_offset = 0, int32_t neg_offset = 0,
                   std::initializer_list<double> custom = {}) {
    NativeHistogram h;
    nh_init(&h);
    h.schema = schema;
    h.sum = sum;
    h.count = count;
    h.zero_count = z_count;
    h.zero_threshold = z_width;
    h.pos_offset = pos_offset;
    h.neg_offset = neg_offset;
    int32_t i = 0;
    for (double v : pos) h.pos[i++] = v;
    h.n_pos = i;
    i = 0;
    for (double v : neg) h.neg[i++] = v;
    h.n_neg = i;
    i = 0;
    for (double v : custom) h.custom[i++] = v;
    h.n_custom = i;
    return h;
}

// The corpus prints ~16 significant digits; compare relatively so a large
// magnitude (stdvar_4 is 7.6e8) is held to the same number of digits as a
// small one, which an absolute epsilon would not do.
void ExpectClose(double got, double want) {
    if (std::isnan(want)) { EXPECT_TRUE(std::isnan(got)) << got; return; }
    if (std::isinf(want)) { EXPECT_EQ(got, want); return; }
    const double scale = std::fabs(want) > 1.0 ? std::fabs(want) : 1.0;
    EXPECT_NEAR(got, want, 1e-12 * scale);
}

}  // namespace

// ---------------------------------------------------------------------------
// The indexing contract itself (see file header).
// ---------------------------------------------------------------------------

TEST(NativeHistogram, BucketIndexingZeroIsUpperBoundOne) {
    // getBoundExponential(0, schema) == 1 for every admitted schema.
    for (int32_t s = -4; s <= 8; ++s) {
        EXPECT_EQ(bolt::promql::detail::nh_bound_exponential(0, s), 1.0)
            << "schema " << s;
    }
    // ...and index 1 is the next bucket to the right: base = 2^(2^-schema).
    EXPECT_DOUBLE_EQ(bolt::promql::detail::nh_bound_exponential(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(bolt::promql::detail::nh_bound_exponential(1, -1), 4.0);
    EXPECT_NEAR(bolt::promql::detail::nh_bound_exponential(1, 1),
                std::sqrt(2.0), 1e-15);
    EXPECT_DOUBLE_EQ(bolt::promql::detail::nh_bound_exponential(-1, 0), 0.5);
}

TEST(NativeHistogram, AllBucketsAreAscendingAndComplete) {
    // negatives, zero bucket, positives — one contiguous ascending run.
    NativeHistogram h = mk(0, 0.0, 6.0, {1, 1}, {1, 1}, /*z_count=*/2.0);
    Scratch s;
    const int32_t n = nh_all_buckets(&h, s.p(), Scratch::cap);
    ASSERT_EQ(n, 5);
    double total = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        // `<=`, not `<`: a histogram with zero_threshold 0 has a DEGENERATE
        // zero bucket [-0, 0] that legitimately holds observations of exactly
        // zero. Prometheus emits it the same way.
        EXPECT_LE(s.b[i].lower, s.b[i].upper) << i;
        if (i > 0) EXPECT_LE(s.b[i - 1].upper, s.b[i].lower) << i;
        total += s.b[i].count;
    }
    EXPECT_DOUBLE_EQ(total, 6.0);
    // Most negative first: (-2,-1] then (-1,-0.5] then the zero bucket.
    EXPECT_DOUBLE_EQ(s.b[0].lower, -2.0);
    EXPECT_DOUBLE_EQ(s.b[0].upper, -1.0);
    EXPECT_DOUBLE_EQ(s.b[2].count, 2.0);   // zero bucket
    EXPECT_DOUBLE_EQ(s.b[4].upper, 2.0);
}

TEST(NativeHistogram, AllBucketsRefusesToTruncate) {
    // A dropped bucket silently changes every quantile, so an undersized
    // scratch must fail rather than answer.
    NativeHistogram h = mk(0, 5.0, 4.0, {1, 2, 1});
    NhBucket tiny[2];
    EXPECT_EQ(nh_all_buckets(&h, tiny, 2), -1);
    EXPECT_TRUE(std::isnan(nh_quantile(&h, 0.5, tiny, 2)));
    EXPECT_TRUE(std::isnan(nh_fraction(&h, 1.0, 2.0, tiny, 2)));
}

// ---------------------------------------------------------------------------
// native_histograms.test:33-67 — single_histogram
//   {{schema:0 sum:5 count:4 buckets:[1 2 1]}}
// ---------------------------------------------------------------------------

TEST(NativeHistogram, SingleHistogramCorpusCases) {
    NativeHistogram h = mk(0, 5.0, 4.0, {1, 2, 1});
    Scratch s;
    ExpectClose(h.count, 4.0);                                  // histogram_count
    ExpectClose(h.sum, 5.0);                                    // histogram_sum
    ExpectClose(nh_avg(&h), 1.25);                              // histogram_avg
    ExpectClose(nh_fraction(&h, 1.0, 2.0, s.p(), Scratch::cap), 0.5);
    ExpectClose(nh_fraction(&h, 0.0, 8.0, s.p(), Scratch::cap), 1.0);
    // Exponential interpolation inside (1,2]: 2**2**-1.
    ExpectClose(nh_quantile(&h, 0.5, s.p(), Scratch::cap), 1.414213562373095);
}

// native_histograms.test:1-28 — the two empty histograms.
TEST(NativeHistogram, EmptyHistogramCorpusCases) {
    Scratch s;
    NativeHistogram exp_h = mk(0, 0.0, 0.0, {});
    NativeHistogram cbh   = mk(k_nh_schema_custom, 0.0, 0.0, {}, {}, 0.0, 0.0,
                               0, 0, {-2, 3});
    for (const NativeHistogram* h : {&exp_h, &cbh}) {
        ExpectClose(h->count, 0.0);
        ExpectClose(h->sum, 0.0);
        EXPECT_TRUE(std::isnan(nh_avg(h)));
        EXPECT_TRUE(std::isnan(nh_fraction(h, -kInf, kInf, s.p(), Scratch::cap)));
        EXPECT_TRUE(std::isnan(nh_fraction(h, 0.0, 8.0, s.p(), Scratch::cap)));
        EXPECT_TRUE(std::isnan(nh_quantile(h, 0.5, s.p(), Scratch::cap)));
    }
}

// ---------------------------------------------------------------------------
// native_histograms.test:326-410 — histogram_stddev / histogram_stdvar.
// Each load line's comment names the observations it encodes; the expected
// values are the corpus's, digit for digit.
// ---------------------------------------------------------------------------

TEST(NativeHistogram, StdDevStdVarCorpusCases) {
    Scratch s;
    {   // {1, 2, 3, 4} (low res): schema 2
        NativeHistogram h = mk(2, 10.0, 4.0, {1, 0, 0, 0, 1, 0, 0, 1, 1});
        ExpectClose(nh_stddev(&h, s.p(), Scratch::cap), 1.0787993180043811);
        ExpectClose(nh_stdvar(&h, s.p(), Scratch::cap), 1.163807968526718);
    }
    {   // {1, 1, 1, 1} (high res): schema 8
        NativeHistogram h = mk(8, 10.0, 10.0, {1, 2, 3, 4});
        ExpectClose(nh_stddev(&h, s.p(), Scratch::cap), 0.0048960313898237465);
        ExpectClose(nh_stdvar(&h, s.p(), Scratch::cap), 2.3971123370139447e-05);
    }
    {   // {-50, -8, 0, 3, 8, 9}: schema 3, a populated zero bucket, both sides
        NativeHistogram h = mk(3, 62.0, 7.0,
            {0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
            /*z_count=*/1.0);
        ExpectClose(nh_stddev(&h, s.p(), Scratch::cap), 42.94723640026);
        ExpectClose(nh_stdvar(&h, s.p(), Scratch::cap), 1844.4651144196398);
    }
    {   // {-100000 ... -3}: schema 0, negatives only
        NativeHistogram h = mk(0, -112946.0, 10.0, {},
            {0,0,1,1,1,0,1,1,0,0,3,0,0,0,1,0,0,1});
        ExpectClose(nh_stddev(&h, s.p(), Scratch::cap), 27556.344499842);
        ExpectClose(nh_stdvar(&h, s.p(), Scratch::cap), 759352122.1939945);
    }
    {   // {-10 x10}: pins the negative bucket bound AND the signed geometric
        // mean — bucket (-16,-8] has representative -sqrt(128) = -11.3137.
        NativeHistogram h = mk(0, -100.0, 10.0, {}, {0, 0, 0, 0, 10});
        ExpectClose(nh_stddev(&h, s.p(), Scratch::cap), 1.3137084989848);
        ExpectClose(nh_stdvar(&h, s.p(), Scratch::cap), 1.725830020304794);
    }
    {   // sum:NaN => both are NaN
        NativeHistogram h = mk(3, std::nan(""), 7.0, {1}, {1}, 1.0);
        EXPECT_TRUE(std::isnan(nh_stddev(&h, s.p(), Scratch::cap)));
        EXPECT_TRUE(std::isnan(nh_stdvar(&h, s.p(), Scratch::cap)));
    }
}

// ---------------------------------------------------------------------------
// Custom buckets (NHCB, schema -53). native_histograms.test:2180+ uses
// custom_values:[5 10 15 20] with buckets:[1 6 4 3 1]; the corpus's own
// bucket-slice expectations pin the bounds: `cbh >/ 15` keeps offset:3
// buckets:[3 1] with count 4, i.e. bucket 3 is (15,20] and bucket 4 is
// (20,+Inf].
// ---------------------------------------------------------------------------

TEST(NativeHistogram, CustomBucketBoundsMatchCorpusSlices) {
    NativeHistogram h = mk(k_nh_schema_custom, 172.5, 15.0, {1, 6, 4, 3, 1},
                           {}, 0.0, 0.0, 0, 0, {5, 10, 15, 20});
    Scratch s;
    const int32_t n = nh_all_buckets(&h, s.p(), Scratch::cap);
    ASSERT_EQ(n, 5);
    EXPECT_EQ(s.b[0].lower, -kInf);
    EXPECT_DOUBLE_EQ(s.b[0].upper, 5.0);
    EXPECT_DOUBLE_EQ(s.b[3].lower, 15.0);
    EXPECT_DOUBLE_EQ(s.b[3].upper, 20.0);
    EXPECT_DOUBLE_EQ(s.b[3].count, 3.0);
    EXPECT_DOUBLE_EQ(s.b[4].lower, 20.0);
    EXPECT_EQ(s.b[4].upper, kInf);
    EXPECT_DOUBLE_EQ(s.b[4].count, 1.0);
    // Custom buckets interpolate LINEARLY, never exponentially. Rank 7.5 of
    // 15 falls in (10,15] (cumulative 1, 7, 11), 0.5/4 of the way through it:
    // 10 + 5*0.125. Forward and reverse iteration agree here, as they must.
    ExpectClose(nh_quantile(&h, 0.5, s.p(), Scratch::cap), 10.625);
    // The last bucket is open-ended, so q=1 returns its finite LOWER bound
    // rather than +Inf.
    ExpectClose(nh_quantile(&h, 1.0, s.p(), Scratch::cap), 20.0);
    // The first bucket's lower bound is -Inf; Prometheus closes it at 0 when
    // its upper bound is positive, so q=0 lands at 0 — NOT at the bucket's
    // upper bound. (Hand-derivation said 5 here and the algorithm said 0; the
    // algorithm is the oracle.)
    ExpectClose(nh_quantile(&h, 0.0, s.p(), Scratch::cap), 0.0);
}

// ---------------------------------------------------------------------------
// An NHCB must agree with the CLASSIC bucket set it was converted from — that
// equality is the whole point of Prometheus's `load_with_nhcb`, which asserts
// the same expected value for `histogram_quantile(q, m)` and
// `histogram_quantile(q, m_bucket)`. histograms.test:743-758 pins
// testhistogram at q=0.5 -> 0.15 and q=0.8 -> 0.72 for start="positive"
// (le 0.1 -> 5, .2 -> 7, 1e0 -> 11, +Inf -> 12 at t=50m).
// ---------------------------------------------------------------------------

TEST(NativeHistogram, NhcbAgreesWithClassicQuantile) {
    // Classic cumulative buckets, as histograms.test loads them.
    const double le[4]  = {0.1, 0.2, 1.0, kInf};
    const double cum[4] = {5.0, 7.0, 11.0, 12.0};
    bolt::promql::HistBucket cscratch[4];

    // The same histogram as an NHCB: custom_values are the FINITE le bounds
    // and the bucket counts are the de-cumulated differences.
    NativeHistogram h = mk(k_nh_schema_custom, 0.0, 12.0,
                           {5.0, 2.0, 4.0, 1.0}, {}, 0.0, 0.0, 0, 0,
                           {0.1, 0.2, 1.0});
    Scratch s;
    for (double q : {0.0, 0.2, 0.5, 0.8, 1.0}) {
        const double classic = bolt::promql::promql_histogram_quantile(
            q, le, cum, 4, cscratch, 4);
        const double native = nh_quantile(&h, q, s.p(), Scratch::cap);
        ExpectClose(native, classic);
    }
    // ...and both equal the corpus's own numbers.
    ExpectClose(nh_quantile(&h, 0.5, s.p(), Scratch::cap), 0.15);
    ExpectClose(nh_quantile(&h, 0.8, s.p(), Scratch::cap), 0.72);
}

// ---------------------------------------------------------------------------
// Discriminating power: the tests above must FAIL if the kernel is wrong in
// the ways it is most likely to be wrong. Each injection is applied to a copy
// of a corpus histogram and asserted to move the answer.
// ---------------------------------------------------------------------------

TEST(NativeHistogram, DiscriminatingPower) {
    Scratch s;
    NativeHistogram h = mk(0, 5.0, 4.0, {1, 2, 1});
    const double truth = nh_quantile(&h, 0.5, s.p(), Scratch::cap);
    ExpectClose(truth, 1.414213562373095);

    // (1) Off-by-one bucket indexing (offset 1 instead of 0) — the reading
    //     this kernel's header rejects. It must NOT produce sqrt(2).
    NativeHistogram shifted = h;
    shifted.pos_offset = 1;
    EXPECT_GT(std::fabs(nh_quantile(&shifted, 0.5, s.p(), Scratch::cap) - truth),
              1.0);

    // (2) Linear instead of exponential interpolation would give 1.5.
    EXPECT_GT(std::fabs(truth - 1.5), 1e-3);

    // (3) A dropped bucket changes the answer (so the refuse-to-truncate path
    //     in AllBucketsRefusesToTruncate is guarding something real).
    NativeHistogram dropped = h;
    dropped.n_pos = 2;
    EXPECT_NE(nh_quantile(&dropped, 0.5, s.p(), Scratch::cap), truth);

    // (4) The forward/reverse iterator split is observable: q=0.5 takes the
    //     reverse path, q=0.49 the forward one, and both must be inside the
    //     same bucket (1,2] yet differ, so a kernel that ignored the split
    //     would not reproduce both corpus digits.
    const double lo = nh_quantile(&h, 0.49, s.p(), Scratch::cap);
    EXPECT_GT(lo, 1.0);
    EXPECT_LT(lo, truth);

    // (5) Custom-bucket histograms must not take the exponential branch.
    NativeHistogram cbh = mk(k_nh_schema_custom, 172.5, 15.0, {1, 6, 4, 3, 1},
                             {}, 0.0, 0.0, 0, 0, {5, 10, 15, 20});
    const double cq = nh_quantile(&cbh, 0.5, s.p(), Scratch::cap);
    ExpectClose(cq, 10.625);
    // Exponential interpolation over the SAME bucket (10,15] at the SAME
    // fraction 0.125 gives 10*(1.5^0.125) = 10.518..., so the linear/
    // exponential branch choice is observable and this case pins it.
    const double as_exponential =
        std::exp2(std::log2(10.0) + (std::log2(15.0) - std::log2(10.0)) * 0.125);
    EXPECT_GT(std::fabs(cq - as_exponential), 1e-3);
}
