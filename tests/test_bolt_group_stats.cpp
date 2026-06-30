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
    const double v[] = {-3.0, -1.0, -2.0};           // sorted: -3,-2,-1
    EXPECT_DOUBLE_EQ(median(v, 3), -2.0);
    const double one[] = {42.5};
    EXPECT_DOUBLE_EQ(median(one, 1), 42.5);
    EXPECT_DOUBLE_EQ(median(one, 0), 0.0);
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
