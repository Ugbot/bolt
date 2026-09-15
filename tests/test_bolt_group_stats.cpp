// Tests for bolt::kernels::group_stats — exact MEDIAN / QUANTILE over a
// value slice (sort-based) and single-pass Pearson CORRELATION / COVARIANCE.
// Mirrors test_bolt_window_agg.cpp in shape.

#include "bolt/kernels/group_stats.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>

namespace {

using namespace bolt::kernels::group_stats;

// Scratch big enough for every case below.
constexpr std::int64_t kCap = 64;
std::int64_t g_keys[kCap];
std::int32_t g_perm[kCap];
std::int32_t g_sort[kCap];

double median(const double* v, std::int64_t n) {
    return median_f64(v, n, g_keys, g_perm, g_sort);
}
double quantile(const double* v, std::int64_t n, double q) {
    return quantile_f64(v, n, q, g_keys, g_perm, g_sort);
}

TEST(GroupStats, MedianOddCount) {
    const double v[] = {5.0, 1.0, 3.0, 2.0, 4.0};   // sorted: 1,2,3,4,5
    EXPECT_DOUBLE_EQ(median(v, 5), 3.0);
}

TEST(GroupStats, MedianEvenCount) {
    const double v[] = {10.0, 2.0, 8.0, 4.0};        // sorted: 2,4,8,10 -> (4+8)/2
    EXPECT_DOUBLE_EQ(median(v, 4), 6.0);
}

TEST(GroupStats, MedianNegativesAndSingle) {
    // CONTROL, and the reason this file needs the MIXED-SIGN case below.
    //
    // This case is named for negatives and CANNOT FAIL on the negatives bug:
    // `f64_sortable_key` used to emit the UNSIGNED-comparable key while the
    // argsort compares SIGNED, which swapped the POSITIVE and NEGATIVE BLOCKS
    // wholesale and left the order WITHIN each block correct. An all-negative
    // input is one block, so it sorted right throughout. Kept as the control
    // that states that.
    const double v[] = {-3.0, -1.0, -2.0};           // sorted: -3,-2,-1
    EXPECT_DOUBLE_EQ(median(v, 3), -2.0);
    const double one[] = {42.5};
    EXPECT_DOUBLE_EQ(median(one, 1), 42.5);
    EXPECT_DOUBLE_EQ(median(one, 0), 0.0);
}

// THE DISCRIMINATING CASE: both signs present. Under the old key every
// non-negative sorted BEFORE every negative, so this returned 3.0 -- a value
// from the set, plausible, and wrong. Every other case in this file (all
// positive, or all negative) passes either way.
TEST(GroupStats, MedianMixedSignsIsTheDiscriminator) {
    const double v[] = {-10.0, -5.0, 1.0, 2.0, 3.0};  // sorted: -10,-5,1,2,3
    EXPECT_DOUBLE_EQ(median(v, 5), 1.0);              // old key gave 3.0

    const double w[] = {1.5, 2.5, -1.0, 0.25};        // sorted: -1,.25,1.5,2.5
    EXPECT_DOUBLE_EQ(median(w, 4), 0.875);            // old key gave 2.0

    // Quantile shares the sort, so it moves with it.
    EXPECT_DOUBLE_EQ(quantile(v, 5, 0.0), -10.0);
    EXPECT_DOUBLE_EQ(quantile(v, 5, 1.0), 3.0);
    EXPECT_DOUBLE_EQ(quantile(v, 5, 0.25), -5.0);
}

// The key itself, asserted directly over the shapes a value test reaches
// only indirectly: mixed signs, both zeroes, subnormals and the extremes.
// The contract is that ASCENDING SIGNED int64 order of the keys equals
// ascending order of the doubles.
TEST(GroupStats, SortableKeyIsSignedComparable) {
    // Ascending by value. -0.0 and 0.0 compare equal as doubles; the key is
    // allowed to order them either way, so they are adjacent here and the
    // assertion below is non-strict across that one pair.
    const double asc[] = {-1e308, -1.0, -5e-324, -0.0, 0.0, 5e-324, 1.0, 1e308};
    const int n = static_cast<int>(sizeof(asc) / sizeof(asc[0]));
    for (int i = 1; i < n; ++i) {
        const std::int64_t a = f64_sortable_key(asc[i - 1]);
        const std::int64_t b = f64_sortable_key(asc[i]);
        EXPECT_LE(a, b) << "key order broke between " << asc[i - 1]
                        << " and " << asc[i];
    }
    // And strictly, everywhere the doubles are strictly ordered.
    for (int i = 1; i < n; ++i) {
        if (asc[i - 1] == asc[i]) continue;           // the +/-0 pair
        EXPECT_LT(f64_sortable_key(asc[i - 1]), f64_sortable_key(asc[i]));
    }
    // A negative key must sort below a non-negative one -- the exact
    // relation the old unsigned-comparable key inverted.
    EXPECT_LT(f64_sortable_key(-1.0), f64_sortable_key(0.0));
    EXPECT_LT(f64_sortable_key(-1e-300), f64_sortable_key(1e-300));
}

TEST(GroupStats, QuantileType7Interpolation) {
    // type-7: pos = q*(n-1). n=5 -> q=0.25 -> pos 1.0 -> exact element.
    const double v[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT_DOUBLE_EQ(quantile(v, 5, 0.0), 1.0);
    EXPECT_DOUBLE_EQ(quantile(v, 5, 0.25), 2.0);
    EXPECT_DOUBLE_EQ(quantile(v, 5, 0.5), 3.0);
    EXPECT_DOUBLE_EQ(quantile(v, 5, 1.0), 5.0);
    // n=4 -> q=0.5 -> pos 1.5 -> 2 + 0.5*(3-2) = 2.5
    const double w[] = {1.0, 2.0, 3.0, 4.0};
    EXPECT_DOUBLE_EQ(quantile(w, 4, 0.5), 2.5);
}

TEST(GroupStats, CovariancePopAndSamp) {
    // x = 1..4, y = 2,4,6,8 (= 2x). cov_pop = E[xy]-E[x]E[y].
    const double xs[] = {1.0, 2.0, 3.0, 4.0};
    const double ys[] = {2.0, 4.0, 6.0, 8.0};
    CorrMoments m;
    corr_moments_init(&m);
    for (int i = 0; i < 4; ++i) corr_moments_update(&m, xs[i], ys[i]);
    // mean x = 2.5, mean y = 5; cov_pop = sum((x-2.5)(y-5))/4
    // deviations: (-1.5,-3),(-.5,-1),(.5,1),(1.5,3) -> 4.5+0.5+0.5+4.5 = 10 ; /4 = 2.5
    EXPECT_DOUBLE_EQ(covar_pop_finalize(&m), 2.5);
    EXPECT_DOUBLE_EQ(covar_samp_finalize(&m), 10.0 / 3.0);
}

TEST(GroupStats, CorrelationPerfectAndNegative) {
    CorrMoments m;
    corr_moments_init(&m);
    const double xs[] = {1.0, 2.0, 3.0, 4.0};
    const double yp[] = {2.0, 4.0, 6.0, 8.0};        // perfectly positive
    for (int i = 0; i < 4; ++i) corr_moments_update(&m, xs[i], yp[i]);
    EXPECT_NEAR(corr_finalize(&m), 1.0, 1e-12);

    corr_moments_init(&m);
    const double yn[] = {8.0, 6.0, 4.0, 2.0};        // perfectly negative
    for (int i = 0; i < 4; ++i) corr_moments_update(&m, xs[i], yn[i]);
    EXPECT_NEAR(corr_finalize(&m), -1.0, 1e-12);
}

TEST(GroupStats, CorrelationDegenerateGuards) {
    CorrMoments m;
    corr_moments_init(&m);
    EXPECT_DOUBLE_EQ(corr_finalize(&m), 0.0);        // n=0
    corr_moments_update(&m, 5.0, 7.0);
    EXPECT_DOUBLE_EQ(corr_finalize(&m), 0.0);        // n=1
    corr_moments_update(&m, 5.0, 9.0);               // x constant -> vx==0
    EXPECT_DOUBLE_EQ(corr_finalize(&m), 0.0);
}

}  // namespace
