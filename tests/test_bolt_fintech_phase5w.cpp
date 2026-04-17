// test_bolt_fintech_phase5w.cpp — Phase 5.W Tier 2 Welford-based kernels.
//
// Ten kernels built on WelfordAccumulator (and RollingRing for the rolling
// variants). Each test is a round-trip against a hand-computed scalar
// reference. Warmup / guard conventions match chukonu verbatim —
// see each kernel header for the contract.

#include <gtest/gtest.h>

#include "bolt/kernels/fintech/autocorr.h"
#include "bolt/kernels/fintech/risk_metrics_vol.h"
#include "bolt/kernels/fintech/rolling_correlation.h"
#include "bolt/kernels/fintech/rolling_kurt.h"
#include "bolt/kernels/fintech/rolling_skew.h"
#include "bolt/kernels/fintech/rolling_std.h"
#include "bolt/kernels/fintech/rolling_zscore.h"
#include "bolt/kernels/fintech/sharpe_ratio.h"
#include "bolt/kernels/fintech/sortino_ratio.h"
#include "bolt/kernels/fintech/welford_mean_var.h"

#include <cmath>
#include <cstdint>

using namespace bolt::fintech;

namespace {
constexpr double kEps = 1e-9;
}  // namespace

// ---------------------------------------------------------------------------
// P5.W.1 WelfordMeanVar — streaming running (mean, pop_var) per row.
// ---------------------------------------------------------------------------
TEST(Phase5W_WelfordMeanVar, ClassicEightSample) {
    // {2,4,4,4,5,5,7,9}: final mean=5, pop_var=4, sample_var=32/7.
    // chukonu uses pop_var; the last row should match 4.0.
    const double xs[] = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    double out_mean[8] = {};
    double out_var[8]  = {};
    WelfordMeanVarState st{};
    st.init(0, 1, 2);
    execute_welford_mean_var(&st, xs, 8, out_mean, out_var);

    // Row 0: mean=2, var=0.
    EXPECT_NEAR(out_mean[0], 2.0, kEps);
    EXPECT_NEAR(out_var[0],  0.0, kEps);

    // Row 1: mean=3, pop_var = ((2-3)^2 + (4-3)^2)/2 = 1.
    EXPECT_NEAR(out_mean[1], 3.0, kEps);
    EXPECT_NEAR(out_var[1],  1.0, kEps);

    // Final row 7.
    EXPECT_NEAR(out_mean[7], 5.0, kEps);
    EXPECT_NEAR(out_var[7],  4.0, kEps);
}

TEST(Phase5W_WelfordMeanVar, ConstantSequenceZeroVar) {
    const double xs[] = {7.5, 7.5, 7.5, 7.5};
    double out_mean[4] = {};
    double out_var[4]  = {};
    WelfordMeanVarState st{};
    st.init(0, 1, 2);
    execute_welford_mean_var(&st, xs, 4, out_mean, out_var);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(out_mean[i], 7.5, kEps);
        EXPECT_NEAR(out_var[i],  0.0, kEps);
    }
}

// ---------------------------------------------------------------------------
// P5.W.2 RollingStd — sqrt(pop_variance) per row, no NaN warmup.
// ---------------------------------------------------------------------------
TEST(Phase5W_RollingStd, Window3On1Through5) {
    // Chukonu emits pop_stddev with no warmup sentinel.
    //   row 0: [1]        → stddev = 0
    //   row 1: [1,2]      → pop_var = 0.25   → stddev = 0.5
    //   row 2: [1,2,3]    → pop_var = 2/3    → stddev = sqrt(2/3)
    //   row 3: [2,3,4]    → same             → sqrt(2/3)
    //   row 4: [3,4,5]    → same             → sqrt(2/3)
    const double xs[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double out[5] = {};
    RollingStdState<8> st{};
    st.init(0, 1, /*w=*/3);
    execute_rolling_std(&st, xs, 5, out);

    const double s = std::sqrt(2.0 / 3.0);
    EXPECT_NEAR(out[0], 0.0, kEps);
    EXPECT_NEAR(out[1], 0.5, kEps);
    EXPECT_NEAR(out[2], s,   kEps);
    EXPECT_NEAR(out[3], s,   kEps);
    EXPECT_NEAR(out[4], s,   kEps);
}

// ---------------------------------------------------------------------------
// P5.W.3 RollingZScore — constant input → 0 (chukonu convention, not NaN).
// ---------------------------------------------------------------------------
TEST(Phase5W_RollingZScore, ConstantYieldsZero) {
    const double xs[] = {5.0, 5.0, 5.0, 5.0};
    double out[4] = {};
    RollingZScoreState<8> st{};
    st.init(0, 1, 3);
    execute_rolling_zscore(&st, xs, 4, out);
    for (int i = 0; i < 4; ++i) EXPECT_NEAR(out[i], 0.0, kEps);
}

TEST(Phase5W_RollingZScore, NonConstantWindowFiniteZ) {
    // Window 3 over {1, 2, 3}:
    //   at row 2: mean=2, pop_var=2/3, stddev=sqrt(2/3).
    //   z = (3 - 2) / sqrt(2/3) = sqrt(3/2) = sqrt(1.5)
    const double xs[] = {1.0, 2.0, 3.0};
    double out[3] = {};
    RollingZScoreState<8> st{};
    st.init(0, 1, 3);
    execute_rolling_zscore(&st, xs, 3, out);
    EXPECT_NEAR(out[2], std::sqrt(1.5), kEps);
    // Output must be finite at every position (no NaN, no inf).
    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(std::isfinite(out[i]));
    }
}

// ---------------------------------------------------------------------------
// P5.W.4 SharpeRatio — scalar broadcast, sample variance, annualized.
// ---------------------------------------------------------------------------
TEST(Phase5W_SharpeRatio, KnownReturnsRfrZeroPeriodsOne) {
    // Returns: {0.01, -0.005, 0.02, 0.01, -0.01}.
    //   sum   = 0.025, mean = 0.005
    //   deviations: {0.005, -0.01, 0.015, 0.005, -0.015}
    //   sum_sq = 2.5e-5 + 1e-4 + 2.25e-4 + 2.5e-5 + 2.25e-4 = 6e-4
    //   sample var = 6e-4 / 4 = 1.5e-4
    //   stddev = sqrt(1.5e-4)
    //   sharpe = (0.005 - 0) / stddev * sqrt(1) = 0.005 / sqrt(1.5e-4)
    const double xs[] = {0.01, -0.005, 0.02, 0.01, -0.01};
    double out[5] = {};
    SharpeRatioState st{};
    st.init(0, 1, /*rfr=*/0.0, /*periods=*/1);
    execute_sharpe_ratio(&st, xs, 5, out);

    const double stddev   = std::sqrt(1.5e-4);
    const double expected = (0.005 / stddev) * 1.0;
    for (int i = 0; i < 5; ++i) EXPECT_NEAR(out[i], expected, 1e-9);
}

TEST(Phase5W_SharpeRatio, ConstantReturnsGuardReturnsZero) {
    // All returns equal → stddev = 0 → guard → 0.0.
    const double xs[] = {0.01, 0.01, 0.01};
    double out[3] = {};
    SharpeRatioState st{};
    st.init(0, 1, 0.0, 252);
    execute_sharpe_ratio(&st, xs, 3, out);
    for (int i = 0; i < 3; ++i) EXPECT_NEAR(out[i], 0.0, kEps);
}

// ---------------------------------------------------------------------------
// P5.W.5 SortinoRatio — scalar broadcast; downside_dev = sqrt(sum_sq/n).
// ---------------------------------------------------------------------------
TEST(Phase5W_SortinoRatio, KnownReturnsMixedSign) {
    // Returns: {0.01, -0.02, 0.03, -0.01, 0.02}, rfr=0, periods=1.
    //   sum  = 0.03,  mean = 0.006
    //   downside mask (excess < 0): {-0.02, -0.01}
    //   downside_sum_sq = 4e-4 + 1e-4 = 5e-4
    //   downside_dev    = sqrt(5e-4 / 5) = sqrt(1e-4) = 0.01
    //   sortino = (0.006 - 0) / 0.01 * sqrt(1) = 0.6
    const double xs[] = {0.01, -0.02, 0.03, -0.01, 0.02};
    double out[5] = {};
    SortinoRatioState st{};
    st.init(0, 1, 0.0, 1);
    execute_sortino_ratio(&st, xs, 5, out);
    for (int i = 0; i < 5; ++i) EXPECT_NEAR(out[i], 0.6, 1e-9);
}

TEST(Phase5W_SortinoRatio, AllPositiveReturnsZeroDownside) {
    // No negative excess → downside_dev = 0 → guard → 0.0.
    const double xs[] = {0.01, 0.02, 0.03};
    double out[3] = {};
    SortinoRatioState st{};
    st.init(0, 1, 0.0, 1);
    execute_sortino_ratio(&st, xs, 3, out);
    for (int i = 0; i < 3; ++i) EXPECT_NEAR(out[i], 0.0, kEps);
}

// ---------------------------------------------------------------------------
// P5.W.6 RollingCorrelation — identical → r=1, anti-correlated → r=-1.
// ---------------------------------------------------------------------------
TEST(Phase5W_RollingCorrelation, IdenticalSeriesCorrOne) {
    const double a[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    const double b[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double out[5] = {};
    RollingCorrelationState<8> st{};
    st.init(0, 1, 2, 3);
    execute_rolling_correlation(&st, a, b, 5, out);
    // w < 2 at row 0 → 0. From row 1 onwards → 1.
    EXPECT_NEAR(out[0], 0.0, kEps);
    for (int i = 1; i < 5; ++i) EXPECT_NEAR(out[i], 1.0, kEps);
}

TEST(Phase5W_RollingCorrelation, AntiCorrelated) {
    const double a[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    const double b[] = {5.0, 4.0, 3.0, 2.0, 1.0};
    double out[5] = {};
    RollingCorrelationState<8> st{};
    st.init(0, 1, 2, 3);
    execute_rolling_correlation(&st, a, b, 5, out);
    EXPECT_NEAR(out[0], 0.0, kEps);
    for (int i = 1; i < 5; ++i) EXPECT_NEAR(out[i], -1.0, kEps);
}

// ---------------------------------------------------------------------------
// P5.W.7 RollingSkew — symmetric data → skew = 0.
// ---------------------------------------------------------------------------
TEST(Phase5W_RollingSkew, SymmetricSeriesZero) {
    // {1,2,3,4,5} is symmetric around mean=3 → skew = 0.
    const double xs[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double out[5] = {};
    RollingSkewState<8> st{};
    st.init(0, 1, /*w=*/5);
    execute_rolling_skew(&st, xs, 5, out);
    // Warmup (w<3): rows 0,1 → 0.
    // Rows 2,3,4 once w>=3 — all symmetric partials also sum to zero.
    EXPECT_NEAR(out[0], 0.0, kEps);
    EXPECT_NEAR(out[1], 0.0, kEps);
    EXPECT_NEAR(out[4], 0.0, kEps);
}

TEST(Phase5W_RollingSkew, PositiveSkewFromOutlier) {
    // Right-tail outlier at the end: {1,1,1,1,10}.
    // window=5: mean=2.8, deltas = {-1.8,-1.8,-1.8,-1.8, 7.2}
    //   m2 = (4*3.24 + 51.84)/5 = (12.96 + 51.84)/5 = 64.8/5 = 12.96
    //   m3 = (4*(-5.832) + 373.248)/5 = (-23.328 + 373.248)/5 = 349.92/5 = 69.984
    //   stddev = sqrt(12.96) = 3.6
    //   skew = 69.984 / (3.6^3) = 69.984 / 46.656 = 1.5
    const double xs[] = {1.0, 1.0, 1.0, 1.0, 10.0};
    double out[5] = {};
    RollingSkewState<8> st{};
    st.init(0, 1, 5);
    execute_rolling_skew(&st, xs, 5, out);
    EXPECT_GT(out[4], 0.0);
    EXPECT_NEAR(out[4], 1.5, 1e-9);
}

// ---------------------------------------------------------------------------
// P5.W.8 RollingKurt — excess kurtosis; {1..5} → -1.3.
// ---------------------------------------------------------------------------
TEST(Phase5W_RollingKurt, FiveRampExcessKurt) {
    // {1,2,3,4,5}: mean=3, m2=2, m4=34/5=6.8.
    //   kurt = m4/m2^2 - 3 = 6.8/4 - 3 = 1.7 - 3 = -1.3
    const double xs[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double out[5] = {};
    RollingKurtState<8> st{};
    st.init(0, 1, 5);
    execute_rolling_kurt(&st, xs, 5, out);
    // Warmup w<4: rows 0,1,2 → 0.
    EXPECT_NEAR(out[0], 0.0, kEps);
    EXPECT_NEAR(out[4], -1.3, 1e-9);
}

TEST(Phase5W_RollingKurt, ConstantWindowZero) {
    // Constant → var=0 → guard → 0.0.
    const double xs[] = {4.0, 4.0, 4.0, 4.0, 4.0};
    double out[5] = {};
    RollingKurtState<8> st{};
    st.init(0, 1, 4);
    execute_rolling_kurt(&st, xs, 5, out);
    for (int i = 0; i < 5; ++i) EXPECT_NEAR(out[i], 0.0, kEps);
}

// ---------------------------------------------------------------------------
// P5.W.9 Autocorr — period-2 alternating series with lag=2 → corr=+1.
// ---------------------------------------------------------------------------
TEST(Phase5W_Autocorr, Period2SeriesLag2PerfectCorr) {
    // x alternates; x[t] = x[t-2] exactly, so corr(current, lagged) = 1.
    // We need enough samples: ring_size = w + lag. With w=4, lag=2,
    // first non-warmup row is index (4 + 2 - 1) = 5.
    const double xs[] = {1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0};
    double out[8] = {};
    AutocorrState<16> st{};
    st.init(0, 1, /*w=*/4, /*lag=*/2);
    execute_autocorr(&st, xs, 8, out);
    // Warmup rows 0..4 → 0.
    for (int i = 0; i < 5; ++i) EXPECT_NEAR(out[i], 0.0, kEps);
    // Rows 5..7: perfect autocorrelation at lag 2.
    EXPECT_NEAR(out[5], 1.0, 1e-12);
    EXPECT_NEAR(out[6], 1.0, 1e-12);
    EXPECT_NEAR(out[7], 1.0, 1e-12);
}

TEST(Phase5W_Autocorr, PeriodMismatchAntiCorr) {
    // Period 2, lag 1: x[t] = -x[t-1] → corr = -1.
    const double xs[] = {1.0, -1.0, 1.0, -1.0, 1.0, -1.0};
    double out[6] = {};
    AutocorrState<16> st{};
    st.init(0, 1, /*w=*/3, /*lag=*/1);
    execute_autocorr(&st, xs, 6, out);
    // ring_size = 4; first non-warmup row = 3.
    for (int i = 0; i < 3; ++i) EXPECT_NEAR(out[i], 0.0, kEps);
    EXPECT_NEAR(out[3], -1.0, 1e-12);
    EXPECT_NEAR(out[4], -1.0, 1e-12);
    EXPECT_NEAR(out[5], -1.0, 1e-12);
}

// ---------------------------------------------------------------------------
// P5.W.10 RiskMetricsVol — λ=0.94 on {0.01, -0.02, 0.015}.
// ---------------------------------------------------------------------------
TEST(Phase5W_RiskMetricsVol, Lambda094ThreeSamples) {
    //   var_0 = 0.01²            = 0.0001              → out = 0.01
    //   var_1 = 0.94·0.0001 + 0.06·0.0004
    //         = 0.000094 + 0.000024 = 0.000118         → out = sqrt(...)
    //   var_2 = 0.94·0.000118 + 0.06·0.000225
    //         = 0.00011092 + 0.0000135 = 0.00012442    → out = sqrt(...)
    const double r[] = {0.01, -0.02, 0.015};
    double out[3] = {};
    RiskMetricsVolState st{};
    st.init(0, 1, 0.94);
    execute_risk_metrics_vol(&st, r, 3, out);

    EXPECT_NEAR(out[0], 0.01,                 1e-12);
    EXPECT_NEAR(out[1], std::sqrt(0.000118),  1e-12);
    EXPECT_NEAR(out[2], std::sqrt(0.00012442),1e-12);
}

TEST(Phase5W_RiskMetricsVol, StatePersistsAcrossCalls) {
    // Second call picks up where first left off.
    const double r1[] = {0.01};
    const double r2[] = {-0.02, 0.015};
    double out1[1] = {};
    double out2[2] = {};
    RiskMetricsVolState st{};
    st.init(0, 1, 0.94);
    execute_risk_metrics_vol(&st, r1, 1, out1);
    execute_risk_metrics_vol(&st, r2, 2, out2);

    EXPECT_NEAR(out1[0], 0.01,                   1e-12);
    EXPECT_NEAR(out2[0], std::sqrt(0.000118),    1e-12);
    EXPECT_NEAR(out2[1], std::sqrt(0.00012442),  1e-12);
}
