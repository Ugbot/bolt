// test_promql_kernels.cpp — Prometheus counter/gauge/histogram kernel oracle.
//
// Every numeric expectation is cross-checked against Prometheus's own promql
// test suite (chukonu/tests/promql/testdata/upstream/{functions,histograms}
// .test, READ-ONLY). The kernels receive samples ALREADY windowed to
// Prometheus's half-open (mint, maxt] range — exactly what the chukonu window
// operator will hand them — so each test constructs the windowed arrays by
// hand and pins the expected value from the oracle.

#include <gtest/gtest.h>

#include "bolt/kernels/promql.h"

#include <cstdint>
#include <limits>

using bolt::promql::HistBucket;
using bolt::promql::promql_changes;
using bolt::promql::promql_delta;
using bolt::promql::promql_deriv;
using bolt::promql::promql_histogram_quantile;
using bolt::promql::promql_idelta;
using bolt::promql::promql_increase;
using bolt::promql::promql_irate;
using bolt::promql::promql_predict_linear;
using bolt::promql::promql_rate;
using bolt::promql::promql_resets;

namespace {
constexpr int64_t k5m = 5 * 60 * 1000;    // 5 minutes in ms
constexpr double kInf = std::numeric_limits<double>::infinity();

// Build ascending timestamps: start_ms, start_ms+step_ms, ...
void mk_ts(int64_t* ts, int64_t start_ms, int64_t step_ms, int64_t n) {
    for (int64_t i = 0; i < n; ++i) ts[i] = start_ms + i * step_ms;
}
}  // namespace

// ---------------------------------------------------------------------------
// rate() — functions.test:
//   calculate_rate_window_total 0+80x10; rate(...[50m]) @50m == 0.26666...
// Windowed (0,50m]: drops t=0, keeps t=5m..50m -> vals 80..800.
// ---------------------------------------------------------------------------
TEST(PromqlRate, WindowPerSecond) {
    int64_t ts[10];
    mk_ts(ts, k5m, k5m, 10);
    double val[10];
    for (int i = 0; i < 10; ++i) val[i] = 80.0 * (i + 1);
    const double r = promql_rate(ts, val, 10, /*start*/ 0, /*end*/ 10 * k5m);
    EXPECT_NEAR(r, 0.26666666666666666, 1e-12);
}

// rate() with a counter reset in the middle — functions.test:
//   testcounter_reset_middle_total 0+27x4 0+27x5; rate(...[50m]) @50m == 0.08
// Windowed (0,50m]: vals [27,54,81,108, 0,27,54,81,108,135].
TEST(PromqlRate, CounterResetMiddle) {
    int64_t ts[10];
    mk_ts(ts, k5m, k5m, 10);
    const double val[10] = {27, 54, 81, 108, 0, 27, 54, 81, 108, 135};
    const double r = promql_rate(ts, val, 10, 0, 10 * k5m);
    EXPECT_NEAR(r, 0.08, 1e-12);
}

// ---------------------------------------------------------------------------
// increase() — functions.test.
//   /foo   0+10x10 [50m]  -> 100  (half-interval left cutoff)
//   /dings 10+10x10 [100m]-> 105  (zero-cutoff not reached)
//   /bumms 1+10x10 [100m] -> 101  (zero-cutoff limits extrapolation)
// ---------------------------------------------------------------------------
TEST(PromqlIncrease, FooHalfIntervalCutoff) {
    int64_t ts[10];
    mk_ts(ts, k5m, k5m, 10);              // t=5m..50m
    double val[10];
    for (int i = 0; i < 10; ++i) val[i] = 10.0 * (i + 1);  // 10..100
    EXPECT_NEAR(promql_increase(ts, val, 10, 0, 10 * k5m), 100.0, 1e-9);
}

TEST(PromqlIncrease, DingsZeroCutoffNotReached) {
    int64_t ts[11];
    mk_ts(ts, 0, k5m, 11);                // t=0..50m
    double val[11];
    for (int i = 0; i < 11; ++i) val[i] = 10.0 + 10.0 * i;  // 10..110
    // [100m] @50m -> range (-50m, 50m].
    EXPECT_NEAR(promql_increase(ts, val, 11, -10 * k5m, 10 * k5m), 105.0, 1e-9);
}

TEST(PromqlIncrease, BummsZeroCutoffLimits) {
    int64_t ts[11];
    mk_ts(ts, 0, k5m, 11);
    double val[11];
    for (int i = 0; i < 11; ++i) val[i] = 1.0 + 10.0 * i;   // 1..101
    EXPECT_NEAR(promql_increase(ts, val, 11, -10 * k5m, 10 * k5m), 101.0, 1e-9);
}

// ---------------------------------------------------------------------------
// delta() — functions.test: http_requests /foo 0 50 100 150 200,
//   /bar 200 150 100 50 0; delta(...[20m]) @20m -> 200 / -200.
// Windowed (0,20m]: t=5m..20m.
// ---------------------------------------------------------------------------
TEST(PromqlDelta, GaugeUpAndDown) {
    int64_t ts[4];
    mk_ts(ts, k5m, k5m, 4);               // t=5m..20m
    const double up[4] = {50, 100, 150, 200};
    const double down[4] = {150, 100, 50, 0};
    EXPECT_NEAR(promql_delta(ts, up, 4, 0, 4 * k5m), 200.0, 1e-9);
    EXPECT_NEAR(promql_delta(ts, down, 4, 0, 4 * k5m), -200.0, 1e-9);
}

// ---------------------------------------------------------------------------
// irate() — functions.test.
//   /foo 0+10x10 irate[50m] @50m -> 0.03333...
//   /bar 0+10x5 0+10x5 irate[50m] @30m -> 0 (last two are a reset).
// ---------------------------------------------------------------------------
TEST(PromqlIrate, LastTwoPerSecond) {
    int64_t ts[10];
    mk_ts(ts, k5m, k5m, 10);
    double val[10];
    for (int i = 0; i < 10; ++i) val[i] = 10.0 * (i + 1);
    EXPECT_NEAR(promql_irate(ts, val, 10), 0.03333333333333333, 1e-15);
}

TEST(PromqlIrate, ResetOnLastPairIsZero) {
    // Windowed (-20m,30m] of 0+10x5 0+10x5: [0,10,20,30,40,50,0] t=0..30m.
    int64_t ts[7];
    mk_ts(ts, 0, k5m, 7);
    const double val[7] = {0, 10, 20, 30, 40, 50, 0};
    EXPECT_NEAR(promql_irate(ts, val, 7), 0.0, 1e-15);
}

// ---------------------------------------------------------------------------
// idelta() — functions.test: /foo 0 50 100 150 -> 50, /bar 0 50 100 50 -> -50.
// ---------------------------------------------------------------------------
TEST(PromqlIdelta, LastTwoDifference) {
    int64_t ts[3];
    mk_ts(ts, k5m, k5m, 3);               // t=5m..15m (windowed (0,20m])
    const double a[3] = {50, 100, 150};
    const double b[3] = {50, 100, 50};
    EXPECT_NEAR(promql_idelta(ts, a, 3), 50.0, 1e-12);
    EXPECT_NEAR(promql_idelta(ts, b, 3), -50.0, 1e-12);
}

// ---------------------------------------------------------------------------
// resets() / changes() — functions.test http_requests /foo windowed [50m]:
//   [2,3,0,1,0,0,1,2,0] -> resets 3, changes 7.
// ---------------------------------------------------------------------------
TEST(PromqlResetsChanges, FooSeries) {
    const double val[9] = {2, 3, 0, 1, 0, 0, 1, 2, 0};
    EXPECT_EQ(promql_resets(val, 9), 3);
    EXPECT_EQ(promql_changes(val, 9), 7);
}

// changes() is NaN-aware — functions.test: x{a="c"} 0 NaN 0 -> 2,
//   x{a="b"} NaN NaN NaN -> 0.
TEST(PromqlChanges, NanAware) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double c[3] = {0, nan, 0};
    const double b[3] = {nan, nan, nan};
    EXPECT_EQ(promql_changes(c, 3), 2);
    EXPECT_EQ(promql_changes(b, 3), 0);
    EXPECT_EQ(promql_resets(b, 3), 0);   // NaN comparisons never count as reset
}

// ---------------------------------------------------------------------------
// deriv() — functions.test: testcounter_reset_middle_total 0+10x4 0+10x5,
//   deriv(...[100m]) @50m == 0.010606060606060607.
// Windowed (-50m,50m]: all 11 samples [0,10,20,30,40,0,10,20,30,40,50].
// ---------------------------------------------------------------------------
TEST(PromqlDeriv, KnownSlope) {
    int64_t ts[11];
    mk_ts(ts, 0, k5m, 11);
    const double val[11] = {0, 10, 20, 30, 40, 0, 10, 20, 30, 40, 50};
    EXPECT_NEAR(promql_deriv(ts, val, 11), 0.010606060606060607, 1e-15);
}

// deriv of a constant series is exactly 0.
TEST(PromqlDeriv, ConstantIsZero) {
    int64_t ts[4];
    mk_ts(ts, 0, k5m, 4);
    const double val[4] = {7, 7, 7, 7};
    EXPECT_DOUBLE_EQ(promql_deriv(ts, val, 4), 0.0);
}

// ---------------------------------------------------------------------------
// predict_linear() — functions.test:
//   predict_linear(testcounter_reset_middle_total[50m], 3600) @50m == 70.
// Windowed (0,50m]: [10,20,30,40,0,10,20,30,40,50] t=5m..50m; eval_ts=50m.
// ---------------------------------------------------------------------------
TEST(PromqlPredictLinear, PredictOneHour) {
    int64_t ts[10];
    mk_ts(ts, k5m, k5m, 10);
    const double val[10] = {10, 20, 30, 40, 0, 10, 20, 30, 40, 50};
    const double p = promql_predict_linear(ts, val, 10, /*eval_ts*/ 10 * k5m, 3600.0);
    EXPECT_NEAR(p, 70.0, 1e-9);
}

// ---------------------------------------------------------------------------
// histogram_quantile() — histograms.test testhistogram3 (positive):
//   le = [0, 0.1, 0.2, 1.0, +Inf], cum = [0, 50, 70, 110, 110].
// ---------------------------------------------------------------------------
TEST(PromqlHistogramQuantile, PositiveBuckets) {
    const double le[5] = {0.0, 0.1, 0.2, 1.0, kInf};
    const double cum[5] = {0, 50, 70, 110, 110};
    HistBucket scratch[8];
    auto q = [&](double phi) {
        return promql_histogram_quantile(phi, le, cum, 5, scratch, 8);
    };
    EXPECT_NEAR(q(0.0), 0.0, 1e-12);
    EXPECT_NEAR(q(0.25), 0.055, 1e-12);
    EXPECT_NEAR(q(0.5), 0.125, 1e-12);
    EXPECT_NEAR(q(0.75), 0.45, 1e-12);
    EXPECT_NEAR(q(1.0), 1.0, 1e-12);
}

// testhistogram3 (negative): le = [-0.25,-0.2,-0.1,0.3,+Inf], cum=[0,10,20,20,20].
TEST(PromqlHistogramQuantile, NegativeBuckets) {
    const double le[5] = {-0.25, -0.2, -0.1, 0.3, kInf};
    const double cum[5] = {0, 10, 20, 20, 20};
    HistBucket scratch[8];
    EXPECT_NEAR(promql_histogram_quantile(0.0, le, cum, 5, scratch, 8), -0.25, 1e-12);
    EXPECT_NEAR(promql_histogram_quantile(0.5, le, cum, 5, scratch, 8), -0.2, 1e-12);
    EXPECT_NEAR(promql_histogram_quantile(1.0, le, cum, 5, scratch, 8), -0.1, 1e-12);
}

// Unsorted input is sorted internally — same answer as PositiveBuckets 0.5.
TEST(PromqlHistogramQuantile, UnsortedInput) {
    const double le[5] = {1.0, kInf, 0.1, 0.0, 0.2};
    const double cum[5] = {110, 110, 50, 0, 70};
    HistBucket scratch[8];
    EXPECT_NEAR(promql_histogram_quantile(0.5, le, cum, 5, scratch, 8), 0.125, 1e-12);
}

// Non-monotonic counts are clamped to monotonic before interpolation.
//   le=[1,2,+Inf] cum=[10,5,12] -> clamp 5->10 -> obs=12, phi=0.5 -> 0.6.
TEST(PromqlHistogramQuantile, NonMonotonicClamped) {
    const double le[3] = {1.0, 2.0, kInf};
    const double cum[3] = {10, 5, 12};
    HistBucket scratch[4];
    EXPECT_NEAR(promql_histogram_quantile(0.5, le, cum, 3, scratch, 4), 0.6, 1e-12);
}

// Negative space: out-of-range phi, NaN phi, missing +Inf, too few buckets.
TEST(PromqlHistogramQuantile, DegenerateCases) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double le[5] = {0.0, 0.1, 0.2, 1.0, kInf};
    const double cum[5] = {0, 50, 70, 110, 110};
    HistBucket scratch[8];

    EXPECT_EQ(promql_histogram_quantile(-0.1, le, cum, 5, scratch, 8), -kInf);
    EXPECT_EQ(promql_histogram_quantile(1.01, le, cum, 5, scratch, 8), kInf);
    EXPECT_TRUE(std::isnan(promql_histogram_quantile(nan, le, cum, 5, scratch, 8)));

    // Highest bucket is not +Inf -> NaN.
    const double no_inf_le[3] = {0.1, 0.2, 1.0};
    const double no_inf_cum[3] = {5, 7, 11};
    EXPECT_TRUE(std::isnan(
        promql_histogram_quantile(0.5, no_inf_le, no_inf_cum, 3, scratch, 8)));

    // Fewer than two buckets -> NaN.
    const double one_le[1] = {kInf};
    const double one_cum[1] = {10};
    EXPECT_TRUE(std::isnan(
        promql_histogram_quantile(0.5, one_le, one_cum, 1, scratch, 8)));

    // Zero total observations -> NaN.
    const double zero_cum[5] = {0, 0, 0, 0, 0};
    EXPECT_TRUE(std::isnan(
        promql_histogram_quantile(0.5, le, zero_cum, 5, scratch, 8)));
}

// Degenerate range-function inputs: < 2 samples -> NaN.
TEST(PromqlRate, TooFewSamplesIsNan) {
    int64_t ts[1] = {0};
    double val[1] = {5};
    EXPECT_TRUE(std::isnan(promql_rate(ts, val, 1, 0, k5m)));
    EXPECT_TRUE(std::isnan(promql_rate(ts, val, 0, 0, k5m)));
    EXPECT_TRUE(std::isnan(promql_irate(ts, val, 1)));
    EXPECT_TRUE(std::isnan(promql_deriv(ts, val, 1)));
    EXPECT_EQ(promql_resets(val, 1), 0);
    EXPECT_EQ(promql_changes(val, 1), 0);
}
