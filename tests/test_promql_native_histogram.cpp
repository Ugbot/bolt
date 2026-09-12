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
// Prometheus's promqltest compares with a RELATIVE tolerance of 1e-6. Used
// only where the corpus rounded a value that this cutoff cannot make exact.
void ExpectCloseAsOracle(double got, double want) {
    const double scale = std::fabs(want) > 1.0 ? std::fabs(want) : 1.0;
    EXPECT_NEAR(got, want, 1e-6 * scale);
}

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

// ---------------------------------------------------------------------------
// ARITHMETIC — nh_mul / nh_div / nh_reduce_resolution / nh_combine.
//
// Every expectation below is copied from native_histograms.test with the line
// cited. These four are the kernels behind PromQL's `hist*float`,
// `hist/float`, `-hist`, `hist+hist`, `hist-hist`, `sum(hist)` and
// `avg(hist)`; before them the frontend answered those with a float, with no
// series, or with an unrelated pass-through histogram.
// ---------------------------------------------------------------------------

namespace {

// True iff two histograms describe the same distribution, comparing by
// ABSOLUTE bucket index with an absent index read as a zero count — the same
// rule Prometheus's FloatHistogram.Equals applies after Compact(0), and the
// same one the PromQL corpus harness uses.
bool SameHistogram(const NativeHistogram& a, const NativeHistogram& b) {
    if (a.schema != b.schema) return false;
    if (!(a.count == b.count || (std::isnan(a.count) && std::isnan(b.count)))) return false;
    if (!(a.sum == b.sum || (std::isnan(a.sum) && std::isnan(b.sum)))) return false;
    if (!(a.zero_count == b.zero_count ||
          (std::isnan(a.zero_count) && std::isnan(b.zero_count)))) return false;
    if (a.zero_threshold != b.zero_threshold) return false;
    if (a.schema == k_nh_schema_custom) {
        if (a.n_custom != b.n_custom) return false;
        for (int32_t i = 0; i < a.n_custom; ++i)
            if (a.custom[i] != b.custom[i]) return false;
    }
    for (int side = 0; side < 2; ++side) {
        const double* pa = (side == 0) ? a.pos : a.neg;
        const double* pb = (side == 0) ? b.pos : b.neg;
        const int32_t oa = (side == 0) ? a.pos_offset : a.neg_offset;
        const int32_t ob = (side == 0) ? b.pos_offset : b.neg_offset;
        const int32_t na = (side == 0) ? a.n_pos : a.n_neg;
        const int32_t nb = (side == 0) ? b.n_pos : b.n_neg;
        const int32_t lo = (na == 0) ? ob : (nb == 0 ? oa : (oa < ob ? oa : ob));
        const int32_t hi = (na == 0) ? ob + nb
                         : (nb == 0 ? oa + na
                            : ((oa + na) > (ob + nb) ? (oa + na) : (ob + nb)));
        for (int32_t i = lo; i < hi; ++i) {
            const double va = (i >= oa && i < oa + na) ? pa[i - oa] : 0.0;
            const double vb = (i >= ob && i < ob + nb) ? pb[i - ob] : 0.0;
            if (va != vb) return false;
        }
    }
    return true;
}

}  // namespace

// native_histograms.test:952-1022. `histogram_mul_div` is
// {{schema:0 count:30 sum:33 z_bucket:3 z_bucket_w:0.001
//   buckets:[3 3 3] n_buckets:[6 6 6]}}.
TEST(NativeHistogram, MulAndDivMatchCorpus) {
    const NativeHistogram base =
        mk(0, 33.0, 30.0, {3, 3, 3}, {6, 6, 6}, 3.0, 0.001);

    NativeHistogram x3 = base;
    bolt::promql::nh_mul(&x3, 3.0);
    EXPECT_TRUE(SameHistogram(
        x3, mk(0, 99.0, 90.0, {9, 9, 9}, {18, 18, 18}, 9.0, 0.001)));

    // `histogram_mul_div*-1` and `-histogram_mul_div` are the same answer.
    NativeHistogram neg = base;
    bolt::promql::nh_mul(&neg, -1.0);
    EXPECT_TRUE(SameHistogram(
        neg, mk(0, -33.0, -30.0, {-3, -3, -3}, {-6, -6, -6}, -3.0, 0.001)));

    NativeHistogram d3 = base;
    bolt::promql::nh_div(&d3, 3.0);
    EXPECT_TRUE(SameHistogram(
        d3, mk(0, 11.0, 10.0, {1, 1, 1}, {2, 2, 2}, 1.0, 0.001)));

    NativeHistogram dm3 = base;
    bolt::promql::nh_div(&dm3, -3.0);
    EXPECT_TRUE(SameHistogram(
        dm3, mk(0, -11.0, -10.0, {-1, -1, -1}, {-2, -2, -2}, -1.0, 0.001)));

    // `histogram_mul_div*0` KEEPS its buckets, at count 0.
    NativeHistogram x0 = base;
    bolt::promql::nh_mul(&x0, 0.0);
    EXPECT_EQ(x0.n_pos, 3);
    EXPECT_TRUE(SameHistogram(x0, mk(0, 0.0, 0.0, {0, 0, 0}, {0, 0, 0}, 0.0, 0.001)));

    // `histogram_mul_div/0` REMOVES every bucket — the corpus's expected
    // literal has no `buckets:` or `n_buckets:` key at all, only
    // count:Inf sum:Inf z_bucket:Inf. That asymmetry with Mul is upstream's,
    // not a simplification: a bucket count of +Inf is meaningless.
    NativeHistogram d0 = base;
    bolt::promql::nh_div(&d0, 0.0);
    EXPECT_EQ(d0.n_pos, 0);
    EXPECT_EQ(d0.n_neg, 0);
    EXPECT_EQ(d0.count, kInf);
    EXPECT_EQ(d0.sum, kInf);
    EXPECT_EQ(d0.zero_count, kInf);

    // `histogram_mul_div*0/0` -> NaN aggregates, buckets gone.
    NativeHistogram nan00 = base;
    bolt::promql::nh_mul(&nan00, 0.0);
    bolt::promql::nh_div(&nan00, 0.0);
    EXPECT_TRUE(std::isnan(nan00.count));
    EXPECT_TRUE(std::isnan(nan00.sum));
    EXPECT_TRUE(std::isnan(nan00.zero_count));
    EXPECT_EQ(nan00.n_pos, 0);
}

// The resolution reduction is the part of Add/Sub that MOVES counts between
// buckets, so it is pinned directly and then again through nh_combine.
// Derived from native_histograms.test:1586-1594: reducing schema-1
// `buckets:[0 2 1]` to schema 0 must give {0:0, 1:3}, because
// `histogram_sub_2{idx="0"} - ignoring(idx) histogram_sub_2{idx="1"}` is
// `buckets:[1 0 1 2 1 1 1]` and the minuend is `[1 3 1 2 1 1 1]`.
TEST(NativeHistogram, ReduceResolutionMatchesCorpusDerivation) {
    NativeHistogram fine = mk(1, 1234.5, 11.0, {0, 2, 1}, {0, 0, 3, 2}, 3.0, 0.001);
    ASSERT_TRUE(bolt::promql::nh_reduce_resolution(&fine, 0));
    EXPECT_EQ(fine.schema, 0);
    EXPECT_TRUE(SameHistogram(fine, mk(0, 1234.5, 11.0, {0, 3}, {0, 3, 2}, 3.0, 0.001)));

    // Only coarser is legal, and a custom-bucket ladder has no coarser form.
    NativeHistogram f2 = mk(1, 0.0, 0.0, {1, 1});
    EXPECT_FALSE(bolt::promql::nh_reduce_resolution(&f2, 2));
    NativeHistogram cbh = mk(k_nh_schema_custom, 1.0, 1.0, {1}, {}, 0.0, 0.0, 0, 0, {5});
    EXPECT_FALSE(bolt::promql::nh_reduce_resolution(&cbh, -54));
}

// native_histograms.test:1590-1598, the three `histogram_sub_*` cases: same
// schema, finer-minus-coarser, and coarser-minus-finer.
TEST(NativeHistogram, CombineSubtractMatchesCorpus) {
    const NativeHistogram big = mk(0, 2345.6, 41.0,
        {1, 3, 1, 2, 1, 1, 1}, {0, 1, 4, 2, 7, 0, 0, 0, 0, 5, 5, 2}, 5.0, 0.001);
    const NativeHistogram small_s0 = mk(0, 1234.5, 11.0,
        {0, 2, 1}, {0, 0, 3, 2}, 3.0, 0.001);
    const NativeHistogram small_s1 = mk(1, 1234.5, 11.0,
        {0, 2, 1}, {0, 0, 3, 2}, 3.0, 0.001);

    NativeHistogram a = big;
    ASSERT_TRUE(bolt::promql::nh_combine(&a, &small_s0, true));
    EXPECT_TRUE(SameHistogram(a, mk(0, 1111.1, 30.0,
        {1, 1, 0, 2, 1, 1, 1}, {0, 1, 1, 0, 7, 0, 0, 0, 0, 5, 5, 2}, 2.0, 0.001)));

    NativeHistogram b = big;
    ASSERT_TRUE(bolt::promql::nh_combine(&b, &small_s1, true));
    EXPECT_TRUE(SameHistogram(b, mk(0, 1111.1, 30.0,
        {1, 0, 1, 2, 1, 1, 1}, {0, -2, 2, 2, 7, 0, 0, 0, 0, 5, 5, 2}, 2.0, 0.001)));

    NativeHistogram c = small_s1;
    ASSERT_TRUE(bolt::promql::nh_combine(&c, &big, true));
    EXPECT_TRUE(SameHistogram(c, mk(0, -1111.1, -30.0,
        {-1, 0, -1, -2, -1, -1, -1},
        {0, 2, -2, -2, -7, 0, 0, 0, 0, -5, -5, -2}, -2.0, 0.001)));
}

// native_histograms.test:1409-1420 — `sum(histogram_sum)` over four series,
// three at schema 0 and one at schema 1, sums to schema 0
// buckets:[5 14 7 7 3 2 2]. This is nh_combine's ADD direction across a
// schema boundary, which is what makes the reduction observable.
TEST(NativeHistogram, CombineAddAcrossSchemasMatchesCorpusSum) {
    const NativeHistogram s0 = mk(0, 3.1, 25.0, {1, 2, 0, 1, 1},
                                  {2, 4, 0, 0, 1, 9}, 4.0, 0.001);
    const NativeHistogram s1 = mk(0, 1e100, 41.0, {1, 3, 1, 2, 1, 1, 1},
                                  {0, 1, 4, 2, 7, 0, 0, 0, 0, 5, 5, 2}, 5.0, 0.001);
    const NativeHistogram s2 = mk(0, -1e100, 41.0, {1, 3, 1, 2, 1, 1, 1},
                                  {0, 1, 4, 2, 7, 0, 0, 0, 0, 5, 5, 2}, 5.0, 0.001);
    const NativeHistogram s3 = mk(1, 1.3, 0.0, {2, 4, 2, 3, 2, 2},
                                  {1, 2, 5, 3, 8, 1, 1, 1, 1, 6, 3}, 3.0, 0.001);
    NativeHistogram acc = s0;
    ASSERT_TRUE(bolt::promql::nh_combine(&acc, &s1, false));
    ASSERT_TRUE(bolt::promql::nh_combine(&acc, &s2, false));
    ASSERT_TRUE(bolt::promql::nh_combine(&acc, &s3, false));
    EXPECT_EQ(acc.schema, 0);
    EXPECT_EQ(acc.count, 107.0);
    EXPECT_EQ(acc.zero_count, 17.0);
    const NativeHistogram want = mk(0, 0.0, 107.0, {5, 14, 7, 7, 3, 2, 2},
        {3, 13, 19, 6, 17, 18, 0, 0, 0, 10, 10, 4}, 17.0, 0.001);
    NativeHistogram got = acc; got.sum = 0.0;   // sum compensation is the caller's
    EXPECT_TRUE(SameHistogram(got, want));
    // …and the reason it is the caller's: plain accumulation of
    // 3.1 + 1e100 + -1e100 + 1.3 loses the small terms, while the corpus
    // asserts 4.4. The kernel does not compensate; the aggregation does.
    EXPECT_EQ(acc.sum, 1.3);
}

// A pair nh_combine cannot represent must be REFUSED, leaving `dst`
// untouched, never partially combined.
TEST(NativeHistogram, CombineRefusesIncompatiblePairs) {
    const NativeHistogram exp0 = mk(0, 1.0, 1.0, {1});
    const NativeHistogram cbh  = mk(k_nh_schema_custom, 1.0, 1.0, {1}, {}, 0.0, 0.0,
                                    0, 0, {5, 10});
    const NativeHistogram cbh2 = mk(k_nh_schema_custom, 1.0, 1.0, {1}, {}, 0.0, 0.0,
                                    0, 0, {5, 11});
    NativeHistogram a = exp0;
    EXPECT_FALSE(bolt::promql::nh_combine(&a, &cbh, false));
    EXPECT_TRUE(SameHistogram(a, exp0));           // untouched

    // W26-L3: two NHCBs whose bound ladders DIFFER are NOT in this class. This
    // file demanded a refusal for them until W25, which is what made 22 corpus
    // cells RED — a pin asserting a refusal for a shape upstream answers is a
    // pin that blocks the fix, so it is re-derived here rather than relaxed.
    // Upstream reconciles onto the INTERSECTION of the two ladders
    // ({5,10} n {5,11} = {5}), and both source buckets then land in the single
    // bounded bucket. Oracle: promtool 3.12.0 (revision 9f27dffc), read by
    // planting a wrong 987654 expectation so it prints what it computed:
    //   sum(m)                        -> {{schema:-53 count:2 sum:2
    //   m{k="a"} + ignoring(k) m{k="b"}   custom_values:[5] buckets:[2]}}
    // for m{k="a"} = {{count:1 sum:1 custom_values:[5 10] buckets:[1]}} and
    // m{k="b"} = {{... custom_values:[5 11] ...}}. The aggregation and the
    // binary-operator spellings agree, and the value is asserted BUCKET-wise:
    // the right total in the wrong bucket preserves count, sum and cardinality
    // and is still a wrong distribution.
    NativeHistogram b = cbh;
    EXPECT_TRUE(bolt::promql::nh_combine(&b, &cbh2, false));
    const NativeHistogram reconciled =
        mk(k_nh_schema_custom, 2.0, 2.0, {2}, {}, 0.0, 0.0, 0, 0, {5});
    EXPECT_TRUE(SameHistogram(b, reconciled));

    // W31-L5x: differing zero thresholds are NOT all in this class either, and
    // this pin demanded a refusal for the half that is exact — the same shape
    // as the NHCB paragraph above, one field over. Re-derived rather than
    // relaxed. Neither histogram here has a bucket inside the wider threshold
    // (schema-0 bucket 0 is (0.5, 1], and 0.5 > 0.01), so upstream's
    // widen-until-they-meet loop absorbs nothing, adjusts no boundary and
    // simply agrees on the wider threshold. Asserted by VALUE — a refusal
    // reinstated here would block the fix, and a count that MOVED would be a
    // wrong distribution wearing the right total.
    const NativeHistogram zt1 = mk(0, 1.0, 1.0, {1}, {}, 1.0, 0.001);
    const NativeHistogram zt2 = mk(0, 1.0, 1.0, {1}, {}, 1.0, 0.01);
    NativeHistogram c = zt1;
    EXPECT_TRUE(bolt::promql::nh_combine(&c, &zt2, false));
    EXPECT_EQ(c.zero_threshold, 0.01);          // the WIDER of the two
    ExpectClose(c.count, 2.0);
    ExpectClose(c.zero_count, 2.0);             // nothing absorbed into it
    ExpectClose(c.pos[0], 2.0);                 // and nothing moved out of it

    // What REMAINS in this class, and what the paragraph above used to stand
    // for: a pair where the widening genuinely has work to do. Bucket -20 of
    // schema 0 covers (2^-11, 2^-10], entirely below 0.01, so upstream folds
    // it into the zero bucket and re-raises the threshold. Still refused.
    const NativeHistogram zt3 = mk(0, 1.0, 1.0, {1}, {}, 1.0, 0.01);
    const NativeHistogram low = mk(0, 1.0, 1.0, {1}, {}, 1.0, 0.001, -20);
    NativeHistogram d = zt3;
    EXPECT_FALSE(bolt::promql::nh_combine(&d, &low, false));
    EXPECT_TRUE(SameHistogram(d, zt3));
}

// Discriminating power: each assertion above fails under a plausible WRONG
// implementation, so a green run means something.
TEST(NativeHistogram, ArithmeticDiscriminatingPower) {
    const NativeHistogram base =
        mk(0, 33.0, 30.0, {3, 3, 3}, {6, 6, 6}, 3.0, 0.001);

    // (1) Div-by-zero implemented as Mul(1/0) would keep buckets at +Inf.
    NativeHistogram as_mul = base;
    bolt::promql::nh_mul(&as_mul, 1.0 / 0.0);
    EXPECT_EQ(as_mul.n_pos, 3);
    NativeHistogram real_div = base;
    bolt::promql::nh_div(&real_div, 0.0);
    EXPECT_NE(real_div.n_pos, as_mul.n_pos);

    // (2) The reduction's `((idx-1)>>shift)+1` is not `idx>>shift`: the naive
    //     form maps schema-1 [0 2 1] to {0:2, 1:1}, not the {0:0, 1:3} the
    //     corpus requires.
    NativeHistogram fine = mk(1, 0.0, 3.0, {0, 2, 1});
    ASSERT_TRUE(bolt::promql::nh_reduce_resolution(&fine, 0));
    double at0 = 0.0, at1 = 0.0;
    for (int32_t i = 0; i < fine.n_pos; ++i) {
        if (fine.pos_offset + i == 0) at0 = fine.pos[i];
        if (fine.pos_offset + i == 1) at1 = fine.pos[i];
    }
    EXPECT_EQ(at0, 0.0);
    EXPECT_EQ(at1, 3.0);

    // (3) Subtraction that reduced to the FINER schema instead of the coarser
    //     would leave schema 1 on the result; the corpus says schema 0.
    const NativeHistogram s0 = mk(0, 0.0, 41.0, {1, 3, 1, 2, 1, 1, 1}, {}, 0.0, 0.001);
    const NativeHistogram s1 = mk(1, 0.0, 11.0, {0, 2, 1}, {}, 0.0, 0.001);
    NativeHistogram r = s0;
    ASSERT_TRUE(bolt::promql::nh_combine(&r, &s1, true));
    EXPECT_EQ(r.schema, 0);

    // (4) Subtraction is not addition: flipping the flag changes the answer.
    NativeHistogram added = s0;
    ASSERT_TRUE(bolt::promql::nh_combine(&added, &s1, false));
    EXPECT_FALSE(SameHistogram(added, r));
}

// ---------------------------------------------------------------------------
// TRIM operators (`h </ x`, `h >/ x`).
//
// Every expectation below is transcribed from native_histograms.test's trim
// section (the block introduced at line 2039, `# Test native histogram with
// trim operators ("</": TRIM_UPPER, ">/": TRIM_LOWER)`), together with the
// `{{...}}` literal that produced it. The corpus states the contract for the
// COUNT itself:
//     histogram_count(h </ x) == histogram_fraction(-Inf, x, h) * histogram_count(h)
//     histogram_count(h >/ x) == histogram_fraction(x, +Inf, h) * histogram_count(h)
// so `TrimCountMatchesFractionIdentity` checks the kernel against a second,
// already-shipped kernel rather than against itself.
//
// The SUM is the part with no second source, and it is where a wrong
// implementation is plausible and silent: a trimmed histogram's recorded sum
// describes observations it no longer holds, so upstream re-estimates it from
// the retained buckets' representatives. Each case below therefore asserts the
// sum to the corpus's full printed precision.
// ---------------------------------------------------------------------------
namespace {

// native_histograms.test:2041
NativeHistogram h_test() {
    // h_test {{schema:0 sum:123.75 count:34 z_bucket:1 z_bucket_w:0.001
    //          buckets:[2 4 8 16] n_buckets:[1 2]}}
    return mk(0, 123.75, 34, {2, 4, 8, 16}, {1, 2}, /*z_count=*/1,
              /*z_width=*/0.001);
}

// native_histograms.test:2069
NativeHistogram h_test_2() {
    // {{schema:2 sum:12.8286080906 count:28 z_bucket:1 z_bucket_w:0.001
    //   buckets:[1 2 4 7 3] n_buckets:[1 5 3 1]}}
    return mk(2, 12.8286080906, 28, {1, 2, 4, 7, 3}, {1, 5, 3, 1}, 1, 0.001);
}

// native_histograms.test:2179
NativeHistogram cbh() {
    // {{schema:-53 sum:172.5 count:15 custom_values:[5 10 15 20]
    //   buckets:[1 6 4 3 1]}}
    return mk(k_nh_schema_custom, 172.5, 15, {1, 6, 4, 3, 1}, {}, 0, 0, 0, 0,
              {5, 10, 15, 20});
}

// native_histograms.test:2312
NativeHistogram cbh_split_at_positive() {
    // {{schema:-53 sum:33 count:101 custom_values:[5] buckets:[1 100]}}
    return mk(k_nh_schema_custom, 33, 101, {1, 100}, {}, 0, 0, 0, 0, {5});
}

// Trim `h` and return it, asserting the kernel accepted the cutoff.
NativeHistogram Trim(NativeHistogram h, double cutoff, bool keep_above) {
    EXPECT_TRUE(bolt::promql::nh_trim(&h, cutoff, keep_above));
    return h;
}

}  // namespace

TEST(NativeHistogram, TrimExponentialCorpusCases) {
    // :2044  h_test >/ -Inf  -> the input, untouched. A trim that removes
    // nothing is the identity: the recorded sum 123.75 is exact, while the
    // bucket-representative re-estimate of the same histogram is 116.67.
    EXPECT_TRUE(SameHistogram(Trim(h_test(), -kInf, true), h_test()));
    // :2047  h_test </ +Inf -> the input, untouched.
    EXPECT_TRUE(SameHistogram(Trim(h_test(), kInf, false), h_test()));

    // :2050  h_test >/ +Inf -> {{schema:0 z_bucket_w:0.001}}  (everything gone)
    {
        const NativeHistogram r = Trim(h_test(), kInf, true);
        ExpectClose(r.count, 0.0);
        ExpectClose(r.sum, 0.0);
        ExpectClose(r.zero_count, 0.0);
        EXPECT_EQ(r.zero_threshold, 0.001);   // the shape survives the trim
        EXPECT_EQ(r.schema, 0);
    }
    // :2053  h_test </ -Inf -> {{schema:0 z_bucket_w:0.001}}
    ExpectClose(Trim(h_test(), -kInf, false).count, 0.0);

    // :2056  h_test >/ 0
    //   {{schema:0 sum:120.20840280171308 count:30.5 z_bucket:0.5
    //     z_bucket_w:0.001 buckets:[2 4 8 16]}}
    {
        const NativeHistogram r = Trim(h_test(), 0.0, true);
        ExpectClose(r.count, 30.5);
        ExpectClose(r.sum, 120.20840280171308);
        ExpectClose(r.zero_count, 0.5);       // zero bucket split at its middle
        ExpectClose(r.neg[0], 0.0);
        ExpectClose(r.neg[1], 0.0);
        ExpectClose(r.pos[3], 16.0);
    }
    // :2059  h_test </ 0
    //   {{schema:0 sum:-3.53578390593273768 count:3.5 z_bucket:0.5 ...
    //     n_buckets:[1 2]}}
    {
        const NativeHistogram r = Trim(h_test(), 0.0, false);
        ExpectClose(r.count, 3.5);
        ExpectClose(r.sum, -3.53578390593273768);
        ExpectClose(r.zero_count, 0.5);
    }

    // :2064  h_test </ 1.4142135624 — the cutoff falls INSIDE bucket (1,2],
    // and the corpus says so: "Trim at sqrt(2) yields half the area between 1
    // and 2 boundaries." The retained half is represented at the geometric
    // mean of (1, sqrt(2)], not of the whole bucket.
    //   {{count:8 sum:0.2570938865989847 z_bucket:1 z_bucket_w:0.001
    //     buckets:[2 2] n_buckets:[1 2]}}
    {
        const NativeHistogram r = Trim(h_test(), 1.4142135624, false);
        // The corpus prints this count as a round 8, but 1.4142135624 is a
        // TRUNCATED sqrt(2) — very slightly above it — so the exact answer is
        // 8.0000000001, and Prometheus's own promqltest comparison (relative
        // 1e-6) is what makes 8 the right thing to write. The SUM beside it is
        // held to 1e-12 because the corpus printed all sixteen of its digits,
        // and it carries the same perturbation: reproducing 0.257093886598...
        // exactly is the stronger of the two statements.
        ExpectCloseAsOracle(r.count, 8.0);
        ExpectClose(r.sum, 0.2570938865989847);
        EXPECT_NEAR(r.pos[1], 2.0, 1e-9);     // half of 4 — the bucket WAS split
        ExpectClose(r.pos[2], 0.0);
    }
    // :2067  h_test >/ 1.4142135624
    //   {{count:26 sum:116.50067065070982 z_bucket_w:0.001 buckets:[0 2 8 16]}}
    {
        const NativeHistogram r = Trim(h_test(), 1.4142135624, true);
        ExpectCloseAsOracle(r.count, 26.0);   // truncated sqrt(2), as above
        ExpectClose(r.sum, 116.50067065070982);
        ExpectClose(r.zero_count, 0.0);
    }

    // :2074  h_test_2 </ 1.13 (schema 2 — a finer ladder, so the interpolation
    // is exercised at a different resolution)
    //   {{schema:2 count:13.410582181123704 sum:-9.385798726068233 ...}}
    {
        const NativeHistogram r = Trim(h_test_2(), 1.13, false);
        ExpectClose(r.count, 13.410582181123704);
        ExpectClose(r.sum, -9.385798726068233);
        ExpectClose(r.pos[1], 1.410582181123704);
    }
    // :2077  h_test_2 >/ 1.13
    //   {{schema:2 count:14.589417818876296 sum:22.168126492693734 ...}}
    {
        const NativeHistogram r = Trim(h_test_2(), 1.13, true);
        ExpectClose(r.count, 14.589417818876296);
        ExpectClose(r.sum, 22.168126492693734);
    }
    // :2083  h_test_2 </ -1.3 — a NEGATIVE bucket split, whose representative
    // is the geometric mean of the retained magnitudes, negated.
    //   {{schema:2 count:2.45786052095524 sum:-3.5189307983595066 ...
    //     n_offset:2 n_buckets:[1.45786052095524 1]}}
    {
        const NativeHistogram r = Trim(h_test_2(), -1.3, false);
        ExpectClose(r.count, 2.45786052095524);
        ExpectClose(r.sum, -3.5189307983595066);
        ExpectClose(r.neg[2], 1.45786052095524);
        ExpectClose(r.neg[0], 0.0);
    }

    // :2087  h_test </ 2 — the cutoff sits exactly ON a bucket boundary, so no
    // interpolation happens ("trim on bucket boundary uses no interpolation").
    //   {{count:10 sum:3.5355339059327373 z_bucket:1 ... buckets:[2 4]
    //     n_buckets:[1 2]}}
    {
        const NativeHistogram r = Trim(h_test(), 2.0, false);
        ExpectClose(r.count, 10.0);
        ExpectClose(r.sum, 3.5355339059327373);
    }
    // :2090  h_test >/ 2 -> {{count:24 sum:113.13708498984761 offset:2 ...}}
    {
        const NativeHistogram r = Trim(h_test(), 2.0, true);
        ExpectClose(r.count, 24.0);
        ExpectClose(r.sum, 113.13708498984761);
    }
    // :2093 / :2096  the same on the negative side.
    ExpectClose(Trim(h_test(), -1.0, true).sum, 119.50104602052653);
    ExpectClose(Trim(h_test(), -1.0, true).count, 32.0);
    ExpectClose(Trim(h_test(), -1.0, false).sum, -2.8284271247461903);
    ExpectClose(Trim(h_test(), -1.0, false).count, 2.0);
}

TEST(NativeHistogram, TrimZeroBucketBiasCorpusCases) {
    // The zero bucket is uniform over [-w, w], closed at zero on whichever
    // side the histogram carries no buckets at all. The corpus tests all three
    // biases with the same numbers, which is what makes the rule visible.

    // :2101 h_positive_buckets {{schema:0 sum:8.0210678118654755 count:12
    //        z_bucket:2 z_bucket_w:0.5 buckets:[10]}} — "positive-biased
    //        (because of the presence of positive buckets)", so [0, 0.5].
    const NativeHistogram pos = mk(0, 8.0210678118654755, 12, {10}, {}, 2, 0.5);
    // :2104  >/ 0.5 -> count 10 sum 7.0710678118654755 z_bucket 0
    ExpectClose(Trim(pos, 0.5, true).count, 10.0);
    ExpectClose(Trim(pos, 0.5, true).sum, 7.0710678118654755);
    ExpectClose(Trim(pos, 0.5, true).zero_count, 0.0);
    // :2107  >/ 0.1 -> count 11.6 sum 7.551067811865476 z_bucket 1.6
    ExpectClose(Trim(pos, 0.1, true).count, 11.6);
    ExpectClose(Trim(pos, 0.1, true).sum, 7.551067811865476);
    ExpectClose(Trim(pos, 0.1, true).zero_count, 1.6);
    // :2110  >/ 0 -> unchanged (nothing is below zero when the bias is [0,w])
    EXPECT_TRUE(SameHistogram(Trim(pos, 0.0, true), pos));
    // :2113  </ 0.5 -> count 2 sum 0.5 ; :2116  </ 0.1 -> count 0.4 sum 0.02
    ExpectClose(Trim(pos, 0.5, false).count, 2.0);
    ExpectClose(Trim(pos, 0.5, false).sum, 0.5);
    ExpectClose(Trim(pos, 0.1, false).count, 0.4);
    ExpectClose(Trim(pos, 0.1, false).sum, 0.02);

    // :2124 h_negative_buckets — the mirror image, bias [-w, 0].
    const NativeHistogram neg = mk(0, -8.0210678118654755, 12, {}, {10}, 2, 0.5);
    ExpectClose(Trim(neg, -0.5, false).count, 10.0);
    ExpectClose(Trim(neg, -0.5, false).sum, -7.0710678118654755);
    ExpectClose(Trim(neg, -0.1, false).count, 11.6);
    ExpectClose(Trim(neg, -0.1, false).sum, -7.551067811865476);

    // :2147 zero_bucket_only {{schema:0 count:5 sum:0 z_bucket:5
    //       z_bucket_w:0.1}} — no other buckets, so the span is symmetric.
    const NativeHistogram only = mk(0, 0, 5, {}, {}, 5, 0.1);
    ExpectClose(Trim(only, 0.05, true).count, 1.25);     // :2156
    ExpectClose(Trim(only, 0.05, true).sum, 0.09375);
    ExpectClose(Trim(only, 0.05, false).count, 3.75);    // :2159
    ExpectClose(Trim(only, 0.05, false).sum, -0.09375);
    ExpectClose(Trim(only, 0.0, true).count, 2.5);       // :2162 — HALF, which
    ExpectClose(Trim(only, 0.0, true).sum, 0.125);       // the [0,w] bias
    ExpectClose(Trim(only, 0.0, false).count, 2.5);      // would not give
    ExpectClose(Trim(only, 0.0, false).sum, -0.125);
    ExpectClose(Trim(only, -0.05, true).count, 3.75);    // :2168
    ExpectClose(Trim(only, -0.05, true).sum, 0.09375);
    ExpectClose(Trim(only, 0.1, true).count, 0.0);       // :2150
    EXPECT_TRUE(SameHistogram(Trim(only, 0.1, false), only));   // :2153 no-op

    // :2263 zero_bucket {{schema:0 sum:-6.75 z_bucket:5 z_bucket_w:0.01
    //       buckets:[2 3] n_buckets:[1 2 3]}} — both sides present, symmetric.
    const NativeHistogram both = mk(0, -6.75, 0, {2, 3}, {1, 2, 3}, 5, 0.01);
    ExpectClose(Trim(both, -0.005, false).count, 7.25);          // :2266
    ExpectClose(Trim(both, -0.005, false).sum, -12.03019028017131);
    ExpectClose(Trim(both, 0.0, true).count, 7.5);               // :2269
    ExpectClose(Trim(both, 0.0, true).sum, 5.669354249492381);
}

TEST(NativeHistogram, TrimCustomBucketCorpusCases) {
    const NativeHistogram h = cbh();
    // :2182  cbh </ 15 — on a boundary, no interpolation.
    //   {{count:11 sum:97.5 buckets:[1 6 4]}}
    // 97.5 = 1*2.5 + 6*7.5 + 4*12.5: the lowest bucket is (-Inf, 5], and its
    // 2.5 is the midpoint of [0,5] — a histogram with no negative bound
    // records no negative observations, so its first bucket starts at 0.
    ExpectClose(Trim(h, 15.0, false).count, 11.0);
    ExpectClose(Trim(h, 15.0, false).sum, 97.5);
    // :2185  cbh >/ 15 -> {{count:4 sum:72.5 offset:3 buckets:[3 1]}}
    // 72.5 = 3*17.5 + 1*20: the (20, +Inf] bucket cannot be split, so it is
    // kept whole and represented at its only finite bound.
    ExpectClose(Trim(h, 15.0, true).count, 4.0);
    ExpectClose(Trim(h, 15.0, true).sum, 72.5);
    // :2189  cbh </ 13 — inside a bucket: custom buckets interpolate LINEARLY
    //   {{count:9.4 sum:75.1 buckets:[1 6 2.4]}}
    ExpectClose(Trim(h, 13.0, false).count, 9.4);
    ExpectClose(Trim(h, 13.0, false).sum, 75.1);
    ExpectClose(Trim(h, 13.0, false).pos[2], 2.4);
    // :2192  cbh >/ 13 -> {{count:5.6 sum:94.9 offset:2 buckets:[1.6 3 1]}}
    ExpectClose(Trim(h, 13.0, true).count, 5.6);
    ExpectClose(Trim(h, 13.0, true).sum, 94.9);
    // :2195  cbh </ 7.5 -> {{count:4 sum:21.25 buckets:[1 3]}}
    ExpectClose(Trim(h, 7.5, false).count, 4.0);
    ExpectClose(Trim(h, 7.5, false).sum, 21.25);
    // :2199  cbh </ 50 — "trim drops +Inf bucket entirely even if cutoff is
    // above its lower bound".  {{count:14 sum:150.0 buckets:[1 6 4 3]}}
    ExpectClose(Trim(h, 50.0, false).count, 14.0);
    ExpectClose(Trim(h, 50.0, false).sum, 150.0);
    // :2202 / :2205  an infinity in the discarding direction empties it, and
    // the custom_values ladder survives.
    {
        const NativeHistogram r = Trim(h, -kInf, false);
        ExpectClose(r.count, 0.0);
        ASSERT_EQ(r.n_custom, 4);
        EXPECT_EQ(r.custom[3], 20.0);
    }
    ExpectClose(Trim(h, kInf, true).count, 0.0);
    // :2208 / :2211  and in the keeping direction is the identity.
    EXPECT_TRUE(SameHistogram(Trim(h, kInf, false), h));
    EXPECT_TRUE(SameHistogram(Trim(h, -kInf, true), h));
    // :2215  cbh >/ 0 is a no-op — every bucket starts at or above 0 once the
    // unbounded first bucket is closed there.  :2218  cbh </ 0 is empty.
    EXPECT_TRUE(SameHistogram(Trim(h, 0.0, true), h));
    ExpectClose(Trim(h, 0.0, false).count, 0.0);

    // :2223 cbh_has_neg {{custom_values:[-10 5 10 15 20] buckets:[2 1 6 4 3 1]}}
    // Its first bucket is (-Inf, -10]: it does NOT reach above zero, so it is
    // not closed at zero and stays unsplittable. Under `</` it is kept whole
    // and represented at the retained slice's only finite end, min(upper, x) —
    // which is -10 at :2226 and the CUTOFF -15 at :2232.
    const NativeHistogram hn = mk(k_nh_schema_custom, 172.5, 15,
                                  {2, 1, 6, 4, 3, 1}, {}, 0, 0, 0, 0,
                                  {-10, 5, 10, 15, 20});
    ExpectClose(Trim(hn, 2.0, false).count, 2.8);       // :2226
    ExpectClose(Trim(hn, 2.0, false).sum, -23.2);
    ExpectClose(Trim(hn, -4.0, false).count, 2.4);      // :2229
    ExpectClose(Trim(hn, -4.0, false).sum, -22.8);
    ExpectClose(Trim(hn, -15.0, false).count, 2.0);     // :2232
    ExpectClose(Trim(hn, -15.0, false).sum, -30.0);

    // :2273 cbh_one_bucket {{schema:-53 sum:100.0 count:100 buckets:[100]}} —
    // a single [-Inf, +Inf] bucket. Unsplittable from either side, so both
    // directions drop it for any finite cutoff, and only an infinity in the
    // keeping direction returns it.
    const NativeHistogram one = mk(k_nh_schema_custom, 100.0, 100, {100});
    ExpectClose(Trim(one, 10.0, false).count, 0.0);     // :2276
    ExpectClose(Trim(one, 10.0, true).count, 0.0);      // :2280
    EXPECT_TRUE(SameHistogram(Trim(one, kInf, false), one));    // :2284
    ExpectClose(Trim(one, kInf, true).count, 0.0);              // :2288
    EXPECT_TRUE(SameHistogram(Trim(one, -kInf, true), one));    // :2292
    ExpectClose(Trim(one, -kInf, false).count, 0.0);            // :2296

    // :2301 cbh_two_buckets_split_at_zero {{custom_values:[0] buckets:[1 100]}}
    const NativeHistogram z = mk(k_nh_schema_custom, 33.0, 100, {1, 100}, {},
                                 0, 0, 0, 0, {0});
    ExpectClose(Trim(z, 10.0, false).sum, 0.0);         // :2304 rep = upper 0
    ExpectClose(Trim(z, 10.0, false).count, 1.0);
    ExpectClose(Trim(z, -10.0, false).sum, -10.0);      // :2312 rep = cutoff
    ExpectClose(Trim(z, -10.0, false).count, 1.0);
    ExpectClose(Trim(z, 10.0, true).sum, 1000.0);       // :2324 rep = cutoff
    ExpectClose(Trim(z, 10.0, true).count, 100.0);
    ExpectClose(Trim(z, -10.0, true).sum, 0.0);         // :2316 rep = lower 0
}

TEST(NativeHistogram, TrimLowerBelowZeroDropsAnUnboundedFirstBucket) {
    // The one asymmetry, and the pair of corpus lines that establishes it.
    // cbh_two_buckets_split_at_positive is {{custom_values:[5] buckets:[1 100]}}
    // — first bucket (-Inf, 5], closed at zero because it reaches above zero.
    //
    //   :2352  >/ 0.0  -> "# Noop."  count 101, sum 33 (the ORIGINAL sum)
    //   :2344  >/ -10.0 -> "# Skip [0, 5] bucket (1)." count 100, sum 500
    //
    // A HIGHER cutoff keeps MORE. Closing the bucket at zero asserts where the
    // mass sits at or above zero and nothing below it, so `>/ -10` asks how
    // much lies above a point inside a span nothing can split, and the bucket
    // contributes nothing. Getting this wrong is silent: 101 instead of 100.
    const NativeHistogram h = cbh_split_at_positive();
    EXPECT_TRUE(SameHistogram(Trim(h, 0.0, true), h));
    ExpectClose(Trim(h, -10.0, true).count, 100.0);
    ExpectClose(Trim(h, -10.0, true).sum, 500.0);

    // :2332  </ 10.0 -> count 1 sum 2.5 (midpoint of the closed [0,5])
    ExpectClose(Trim(h, 10.0, false).count, 1.0);
    ExpectClose(Trim(h, 10.0, false).sum, 2.5);
    // :2336  </ 2.0 -> "Skip (5, +Inf] bucket (100) and 3/5 of [0, 5] bucket"
    //   count 0.4 sum 0.4
    ExpectClose(Trim(h, 2.0, false).count, 0.4);
    ExpectClose(Trim(h, 2.0, false).sum, 0.4);
    // :2356  >/ 2.0 -> count 100.6 sum 502.1
    ExpectClose(Trim(h, 2.0, true).count, 100.6);
    ExpectClose(Trim(h, 2.0, true).sum, 502.1);
    // :2340  </ 0.0 -> empty ; :2348  </ -10.0 -> empty
    ExpectClose(Trim(h, 0.0, false).count, 0.0);
    ExpectClose(Trim(h, -10.0, false).count, 0.0);
    // :2360  >/ 10.0 -> count 100 sum 1000
    ExpectClose(Trim(h, 10.0, true).count, 100.0);
    ExpectClose(Trim(h, 10.0, true).sum, 1000.0);
}

TEST(NativeHistogram, TrimCountMatchesFractionIdentity) {
    // The corpus states this identity for itself, and lists the exact pairs it
    // holds for ("Verify trim operators satisfy the identity for non-empty
    // native histograms h: h </ x == histogram_fraction(-Inf, x, h) *
    // histogram_count(h)"). nh_fraction is a separate, already-shipped kernel,
    // so this is a second source for the retained COUNT rather than the kernel
    // checking itself. The pairs below are the corpus's own, no others.
    Scratch sc;
    struct Case { NativeHistogram h; double x; };
    const Case cases[] = {
        { h_test(),   2.0 },            // native_histograms.test:2450 / :2456
        { h_test(),  -1.0 },            // :2462 / :2468
        { h_test(),   0.0 },            // :2474 / :2480
        { h_test(),   1.4142135624 },   // :2486 / :2492
        { h_test_2(), 1.13 },           // :2498 / :2504
        { cbh(),     15.0 },            // :2510 / :2516
        { cbh(),     13.0 },            // :2522 / :2528
    };
    int checked = 0;
    for (const Case& c : cases) {
        const double below =
            nh_fraction(&c.h, -kInf, c.x, sc.p(), Scratch::cap) * c.h.count;
        const double above =
            nh_fraction(&c.h, c.x, kInf, sc.p(), Scratch::cap) * c.h.count;
        EXPECT_NEAR(Trim(c.h, c.x, false).count, below, 1e-9)
            << "TRIM_UPPER at " << c.x;
        EXPECT_NEAR(Trim(c.h, c.x, true).count, above, 1e-9)
            << "TRIM_LOWER at " << c.x;
        EXPECT_NEAR(below + above, c.h.count, 1e-9);   // and they partition it
        ++checked;
    }
    EXPECT_EQ(checked, 7);   // the loop really ran
}

TEST(NativeHistogram, TrimAndFractionDivergeBelowAClosedFirstBucket) {
    // A FINDING, not a bug, recorded because the identity above is easy to
    // over-generalise: it does NOT hold for a custom-bucket histogram whose
    // first bucket is unbounded below, at a cutoff below zero.
    //
    // nh_fraction closes such a bucket at zero and then reports everything in
    // it as lying above -10. nh_trim does not: closing at zero says where the
    // mass sits at or above zero and nothing about the span below, which no
    // interpolation can split, so the bucket contributes nothing. The corpus
    // pins TRIM's answer directly and is silent on the fraction's:
    //   native_histograms.test:2344  cbh_two_buckets_split_at_positive >/ -10.0
    //     -> "# Skip [0, 5] bucket (1)."  count 100 of 101.
    // Anyone extending either kernel should know the two disagree HERE and
    // agree everywhere the corpus checks them, rather than discover it as a
    // silent one-off in a count.
    Scratch sc;
    const NativeHistogram h = cbh_split_at_positive();
    const double frac_above =
        nh_fraction(&h, -10.0, kInf, sc.p(), Scratch::cap) * h.count;
    EXPECT_NEAR(frac_above, 101.0, 1e-9);            // the fraction kernel
    ExpectClose(Trim(h, -10.0, true).count, 100.0);  // the corpus, via trim
}

TEST(NativeHistogram, TrimDiscriminatingPower) {
    // Each block states a plausible WRONG implementation and shows a corpus
    // expectation that separates it from the shipped one. A test that cannot
    // do this is green on broken code.
    Scratch sc;

    // (1) Carrying the recorded sum through instead of re-estimating it. The
    // no-op cases would still pass; this one would not.
    const NativeHistogram r = Trim(h_test(), 2.0, false);
    EXPECT_NE(r.sum, h_test().sum);
    ExpectClose(r.sum, 3.5355339059327373);

    // (2) Re-estimating the sum even when NOTHING was trimmed. The corpus
    // returns the input for `h_test >/ -Inf`, and the re-estimate of the whole
    // h_test is 116.672..., not 123.75 — so the two are distinguishable.
    EXPECT_EQ(Trim(h_test(), -kInf, true).sum, 123.75);

    // (3) Representing a SPLIT bucket at the whole bucket's midpoint rather
    // than the retained slice's. `h_test </ 1.4142135624` keeps half of (1,2];
    // at the whole bucket's geometric mean sqrt(2) the sum would be
    // 0.7071067811865475, and the corpus says 0.2570938865989847.
    ExpectClose(Trim(h_test(), 1.4142135624, false).sum, 0.2570938865989847);

    // (4) Interpolating a CUSTOM bucket geometrically instead of linearly.
    // cbh </ 13 splits (10,15] at 13; linearly that is 0.6 of 4 = 2.4, and
    // geometrically it would be 4*log(1.3)/log(1.5) = 2.588...
    ExpectClose(Trim(cbh(), 13.0, false).pos[2], 2.4);

    // (5) Keeping the +Inf bucket under `</`. The corpus's own comment says it
    // is dropped "even if cutoff is above its lower bound"; keeping it would
    // make `cbh </ 50` a 15-count no-op instead of 14.
    ExpectClose(Trim(cbh(), 50.0, false).count, 14.0);

    // (6) Treating an exponential bucket's representative as the arithmetic
    // midpoint. For h_test >/ 2 that would give 8*3 + 16*6 = 120, not
    // 113.13708498984761.
    ExpectClose(Trim(h_test(), 2.0, true).sum, 113.13708498984761);

    // (7) Ignoring the zero bucket's BIAS and always splitting it in half.
    // h_positive_buckets >/ 0 is a no-op precisely because its zero bucket is
    // [0, w]; a symmetric span would return 1 of its 2, and count 11 not 12.
    const NativeHistogram pos = mk(0, 8.0210678118654755, 12, {10}, {}, 2, 0.5);
    EXPECT_TRUE(SameHistogram(Trim(pos, 0.0, true), pos));

    // (8) A NaN cutoff has no answer and must be REFUSED, not guessed: no
    // ordering places a bucket above or below it.
    NativeHistogram nan_case = h_test();
    EXPECT_FALSE(bolt::promql::nh_trim(&nan_case, std::nan(""), true));
    EXPECT_TRUE(SameHistogram(nan_case, h_test()));   // and left untouched

    // (9) The identity check has teeth only if fraction and trim can disagree:
    // confirm nh_fraction sees a non-trivial split here.
    const double frac = nh_fraction(&nan_case, -kInf, 2.0, sc.p(), Scratch::cap);
    EXPECT_GT(frac, 0.0);
    EXPECT_LT(frac, 1.0);
}

// ---------------------------------------------------------------------------
// W31-L5x — COMBINING TWO HISTOGRAMS WHOSE ZERO THRESHOLDS DIFFER.
//
// `nh_combine` used to refuse any such pair outright. Prometheus's
// reconcileZeroBuckets widens the narrower zero bucket until the thresholds
// meet, absorbing every bucket that falls inside the wider one and re-raising
// the threshold when it lands mid-bucket. That loop MOVES COUNTS, so only the
// case where it does ZERO iterations is performed here; everything else stays
// refused.
//
// THE CORPUS CANNOT GRADE THE REFUSAL HALF, which is the whole reason this
// test exists. `native_histograms.test`'s only differing-threshold pair is
// `sum_over_time(histogram_sum_over_time[4m:1m])`, whose narrow side is the
// EMPTY `{{schema:1 count:0}}` sample — it has no bucket to absorb, so an
// implementation that simply overwrote the threshold and ignored absorbed
// buckets scores the identical GREEN. Measured, not assumed: that injection
// leaves the corpus at GREEN 2137 / RED 31 / FAIL 0 / VALUE-DIFF 0.
TEST(PromqlNativeHistogram, ZeroThresholdReconciliation) {
    // The corpus's own pair, transcribed: three samples at z_bucket_w 0.001
    // already folded, plus `{{schema:1 count:0}}` which carries no width.
    NativeHistogram acc = mk(0, 4691.2, 107, {3, 8, 2, 5, 3, 2, 2}, {}, 14.0, 0.001);
    const NativeHistogram empty = mk(1, 0.0, 0, {}, {}, 0.0, 0.0);
    ASSERT_TRUE(bolt::promql::nh_combine(&acc, &empty, false));
    EXPECT_EQ(acc.zero_threshold, 0.001);   // the WIDER of the two
    ExpectClose(acc.count, 107.0);          // absorbed nothing
    ExpectClose(acc.zero_count, 14.0);

    // REFUSAL: the narrow side has a populated bucket INSIDE the wider
    // threshold. Schema-0 bucket -20 covers (2^-11, 2^-10] ~ (4.9e-4, 9.8e-4],
    // entirely below 0.001, so upstream would absorb it into the zero bucket
    // and this kernel must decline rather than claim the counts stayed put.
    NativeHistogram wide = mk(0, 1.0, 107, {1}, {}, 14.0, 0.001);
    const NativeHistogram low = mk(0, 1.0, 3, {3}, {}, 0.0, 0.0, -20);
    const NativeHistogram wide_before = wide;
    EXPECT_FALSE(bolt::promql::nh_combine(&wide, &low, false));
    EXPECT_TRUE(SameHistogram(wide, wide_before));   // and left untouched

    // The same shape on the NEGATIVE side — a bucket's distance from zero is
    // what matters, not its sign. Without the negative test a histogram whose
    // mass is all below zero would be reconciled as if it were empty.
    NativeHistogram wide2 = mk(0, 1.0, 107, {1}, {}, 14.0, 0.001);
    const NativeHistogram low_neg = mk(0, -1.0, 3, {}, {3}, 0.0, 0.0, 0, -20);
    EXPECT_FALSE(bolt::promql::nh_combine(&wide2, &low_neg, false));

    // CONTROL: the same bucket count moved ABOVE the wider threshold is the
    // identity again, so the refusal above is about the BOUND and not merely
    // about having buckets at all.
    NativeHistogram wide3 = mk(0, 1.0, 107, {1}, {}, 14.0, 0.001);
    const NativeHistogram high = mk(0, 1.0, 3, {3}, {}, 0.0, 0.0, 1);
    ASSERT_TRUE(bolt::promql::nh_combine(&wide3, &high, false));
    ExpectClose(wide3.count, 110.0);
    EXPECT_EQ(wide3.zero_threshold, 0.001);

    // CONTROL: widening works when the RECEIVER is the narrow side too. `dst`
    // is the mutable one, so a version that only ever widened `src` would pass
    // every case above and fail here.
    NativeHistogram narrow = mk(0, 1.0, 3, {3}, {}, 0.0, 0.0, 1);
    const NativeHistogram wider = mk(0, 4691.2, 107, {1}, {}, 14.0, 0.001);
    ASSERT_TRUE(bolt::promql::nh_combine(&narrow, &wider, false));
    EXPECT_EQ(narrow.zero_threshold, 0.001);
    ExpectClose(narrow.count, 110.0);

    // CONTROL: equal thresholds still take the untouched path.
    NativeHistogram a = mk(0, 1234.5, 25, {1, 2}, {}, 4.0, 0.001);
    const NativeHistogram b = mk(0, 2345.6, 41, {1, 3}, {}, 5.0, 0.001);
    ASSERT_TRUE(bolt::promql::nh_combine(&a, &b, false));
    ExpectClose(a.count, 66.0);
    ExpectClose(a.zero_count, 9.0);
}

// W31-L5x — a NaN zero threshold must not become COMBINABLE.
//
// nh_combine's reconciliation is entered on `dst->zero_threshold !=
// src->zero_threshold`, and NaN compares unequal to ITSELF — so without an
// explicit non-finite guard a malformed pair that the old flat refusal
// rejected would fall through every comparison and combine. Widening what
// nh_combine accepts is only ever allowed where the answer is exact; this is
// the boundary where it would not have been.
TEST(PromqlNativeHistogram, NonFiniteZeroThresholdStillRefused) {
    const double nan_v = std::nan("");
    NativeHistogram a = mk(0, 1.0, 1.0, {1}, {}, 1.0, nan_v);
    const NativeHistogram b = mk(0, 1.0, 1.0, {1}, {}, 1.0, nan_v);
    EXPECT_FALSE(bolt::promql::nh_combine(&a, &b, false));
    // `a` is untouched, asserted field-wise rather than with SameHistogram:
    // that helper compares every field with ==, and a NaN threshold is not
    // equal to itself, so it would report a difference the engine never made.
    ExpectClose(a.count, 1.0);
    ExpectClose(a.sum, 1.0);
    ExpectClose(a.pos[0], 1.0);
    ExpectClose(a.zero_count, 1.0);
    EXPECT_TRUE(std::isnan(a.zero_threshold));

    // One side finite, one NaN — likewise refused, in both directions.
    NativeHistogram c = mk(0, 1.0, 1.0, {1}, {}, 1.0, 0.001);
    const NativeHistogram d = mk(0, 1.0, 1.0, {1}, {}, 1.0, nan_v);
    EXPECT_FALSE(bolt::promql::nh_combine(&c, &d, false));
    NativeHistogram e = mk(0, 1.0, 1.0, {1}, {}, 1.0, nan_v);
    const NativeHistogram f = mk(0, 1.0, 1.0, {1}, {}, 1.0, 0.001);
    EXPECT_FALSE(bolt::promql::nh_combine(&e, &f, false));
}

// ---------------------------------------------------------------------------
// W31-L6d — `nh_format`, and the fence around what the oracle could not say.
//
// The corpus contains exactly ONE rendering of a native histogram, repeated
// across the five `count_values` cells, and no second oracle is reachable on
// this box (no promtool; the Docker daemon does not answer). So the ORACLE CASE
// below is transcribed from `aggregators.test` and everything the single
// example cannot ground is REFUSED rather than guessed.
//
// The corpus can only ever exercise the one shape, so every refusal here is
// invisible to it — which is exactly why they are pinned in this file.
TEST(PromqlNativeHistogram, FormatMatchesTheCorpusRendering) {
    // {{schema:0 sum:10 count:20 z_bucket_w:0.001 z_bucket:2
    //   buckets:[1 2] n_buckets:[1 2]}}
    NativeHistogram h = mk(0, 10.0, 20.0, {1, 2}, {1, 2}, 2.0, 0.001);
    char buf[512];
    const int32_t n = bolt::promql::nh_format(&h, buf, sizeof(buf));
    ASSERT_GT(n, 0);
    EXPECT_STREQ(buf,
        "{count:20, sum:10, [-2,-1):2, [-1,-0.5):1, [-0.001,0.001]:2, "
        "(0.5,1]:1, (1,2]:2}");
    EXPECT_EQ(n, static_cast<int32_t>(std::strlen(buf)));

    // The BRACKETS differ per side and are the easiest thing to get subtly
    // wrong: negative half-open low, zero CLOSED both ends, positive half-open
    // high. Asserted as substrings so a change to one side is named.
    EXPECT_NE(std::strstr(buf, "[-2,-1):2"), nullptr);
    EXPECT_NE(std::strstr(buf, "[-0.001,0.001]:2"), nullptr);
    EXPECT_NE(std::strstr(buf, "(1,2]:2"), nullptr);
}

TEST(PromqlNativeHistogram, FormatRefusesWhatTheOracleCannotGround) {
    char buf[512];
    // (1) An NHCB. Its bounds are a user-supplied ladder and the corpus has no
    // example of how upstream prints one.
    NativeHistogram cbh = mk(k_nh_schema_custom, 1.0, 1.0, {1}, {}, 0.0, 0.0,
                             0, 0, {5, 10});
    EXPECT_LT(bolt::promql::nh_format(&cbh, buf, sizeof(buf)), 0);

    // (2) A ZERO-COUNT bucket inside a populated run. Upstream may well skip
    // empty buckets, but the corpus's own histogram has none, so whether to
    // print it is precisely what one example cannot say.
    NativeHistogram gap = mk(0, 3.0, 3.0, {1, 0, 2}, {}, 0.0, 0.0);
    EXPECT_LT(bolt::promql::nh_format(&gap, buf, sizeof(buf)), 0);

    // (3) A value needing EXPONENT notation. The oracle's exponents run -3 to
    // 1; outside that the spelling (`1e+06` vs `1000000`) is a guess.
    NativeHistogram big = mk(0, 1e30, 1.0, {1}, {}, 0.0, 0.0);
    EXPECT_LT(bolt::promql::nh_format(&big, buf, sizeof(buf)), 0);

    // (4) A NaN sum — no example, and NaN has more than one spelling.
    NativeHistogram nan_sum = mk(0, std::nan(""), 1.0, {1}, {}, 0.0, 0.0);
    EXPECT_LT(bolt::promql::nh_format(&nan_sum, buf, sizeof(buf)), 0);

    // CONTROL: the refusals above are about the SHAPE, not about nh_format
    // declining everything. A histogram with no zero bucket and no negative
    // side still renders — the zero bucket is omitted, which is
    // `nh_all_buckets`' existing tested behaviour rather than a new rule.
    NativeHistogram pos_only = mk(0, 3.0, 3.0, {1, 2}, {}, 0.0, 0.0);
    const int32_t m = bolt::promql::nh_format(&pos_only, buf, sizeof(buf));
    ASSERT_GT(m, 0);
    EXPECT_STREQ(buf, "{count:3, sum:3, (0.5,1]:1, (1,2]:2}");

    // CONTROL: a buffer too small must REFUSE, never truncate — a truncated
    // histogram rendering would be a plausible-looking wrong label value.
    NativeHistogram h = mk(0, 10.0, 20.0, {1, 2}, {1, 2}, 2.0, 0.001);
    char tiny[16];
    EXPECT_LT(bolt::promql::nh_format(&h, tiny, sizeof(tiny)), 0);
}
