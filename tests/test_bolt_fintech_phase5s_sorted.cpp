// test_bolt_fintech_phase5s_sorted.cpp — Phase 5.S sorted-window kernels.
//
// Four Tier-2 specialised fintech kernels that share SortedRing<T, kCap>
// (rolling quantile / median / MAD):
//   - HistoricalVaR       (P5.S.4)
//   - HistoricalCVaR      (P5.S.5)
//   - MedRV               (P5.S.8)
//   - OutlierFlagMAD      (P5.S.14)
//
// Each test is a round-trip against a hand-computed scalar reference.
// Warmup / guard conventions match chukonu verbatim — see each kernel
// header for the contract.

#include <gtest/gtest.h>

#include "bolt/kernels/fintech/historical_cvar.h"
#include "bolt/kernels/fintech/historical_var.h"
#include "bolt/kernels/fintech/med_rv.h"
#include "bolt/kernels/fintech/outlier_flag_mad.h"
#include "bolt/kernels/fintech/sorted_ring.h"

#include <cmath>
#include <cstdint>

using namespace bolt::fintech;

namespace {
constexpr double kEps = 1e-9;
}  // namespace

// ---------------------------------------------------------------------------
// SortedRing — rolling sorted window primitive.
// ---------------------------------------------------------------------------
TEST(Phase5S_SortedRing, FiveInsertionsThenEviction) {
    SortedRing<double, 16> r{};
    r.init(5);
    // Insertion order {5, 1, 3, 4, 2} into window=5.
    const double xs[] = {5.0, 1.0, 3.0, 4.0, 2.0};
    for (double v : xs) r.push(v);
    EXPECT_EQ(r.size(), 5u);
    EXPECT_TRUE(r.full());

    // sorted[] must be {1,2,3,4,5}.
    double sorted_out[5] = {};
    r.copy_sorted(sorted_out, 5);
    EXPECT_NEAR(sorted_out[0], 1.0, kEps);
    EXPECT_NEAR(sorted_out[1], 2.0, kEps);
    EXPECT_NEAR(sorted_out[2], 3.0, kEps);
    EXPECT_NEAR(sorted_out[3], 4.0, kEps);
    EXPECT_NEAR(sorted_out[4], 5.0, kEps);

    EXPECT_NEAR(r.median(), 3.0, kEps);
    EXPECT_NEAR(r.quantile(0.5), 3.0, kEps);
    EXPECT_NEAR(r.quantile(0.0), 1.0, kEps);
    EXPECT_NEAR(r.quantile(1.0), 5.0, kEps);

    // Push a 6th value: raw[0]=5 (first in) gets evicted.
    r.push(6.0);
    EXPECT_EQ(r.size(), 5u);
    r.copy_sorted(sorted_out, 5);
    // Insertion order so far: {5,1,3,4,2,6} with window=5 — after evict,
    // live set = {1,3,4,2,6}, sorted = {1,2,3,4,6}.
    EXPECT_NEAR(sorted_out[0], 1.0, kEps);
    EXPECT_NEAR(sorted_out[1], 2.0, kEps);
    EXPECT_NEAR(sorted_out[2], 3.0, kEps);
    EXPECT_NEAR(sorted_out[3], 4.0, kEps);
    EXPECT_NEAR(sorted_out[4], 6.0, kEps);
    EXPECT_NEAR(r.median(), 3.0, kEps);
}

TEST(Phase5S_SortedRing, PartialWindowBuildUp) {
    SortedRing<double, 8> r{};
    r.init(4);
    r.push(10.0);
    EXPECT_EQ(r.size(), 1u);
    EXPECT_NEAR(r.median(), 10.0, kEps);

    r.push(2.0);
    EXPECT_EQ(r.size(), 2u);
    // sorted = {2, 10}, median = 6.
    EXPECT_NEAR(r.median(), 6.0, kEps);

    r.push(7.0);
    // sorted = {2,7,10}, median = 7.
    EXPECT_NEAR(r.median(), 7.0, kEps);
}

TEST(Phase5S_SortedRing, EvenLengthMedianAveragesMiddles) {
    SortedRing<double, 8> r{};
    r.init(4);
    // insert {4,1,3,2} → sorted = {1,2,3,4}, median = (2+3)/2 = 2.5.
    r.push(4.0); r.push(1.0); r.push(3.0); r.push(2.0);
    EXPECT_NEAR(r.median(), 2.5, kEps);
    EXPECT_NEAR(r.quantile(0.5), 2.5, kEps);
}

TEST(Phase5S_SortedRing, EvictionWithDuplicates) {
    SortedRing<double, 8> r{};
    r.init(3);
    // {2, 2, 2} then push 5 → evict a 2 → sorted = {2,2,5}, median=2.
    r.push(2.0); r.push(2.0); r.push(2.0);
    EXPECT_NEAR(r.median(), 2.0, kEps);
    r.push(5.0);
    EXPECT_NEAR(r.median(), 2.0, kEps);
    // Push another 5 → evict another 2 → sorted = {2,5,5}, median=5.
    r.push(5.0);
    EXPECT_NEAR(r.median(), 5.0, kEps);
}

// ---------------------------------------------------------------------------
// P5.S.4 HistoricalVaR — rolling (1-confidence) quantile of losses.
// ---------------------------------------------------------------------------
TEST(Phase5S_HistoricalVaR, HandComputedWindow10) {
    // Ten returns; confidence = 0.75 (alpha = 0.25 — exact in binary
    // floating-point, unlike 0.9 which produces a subtle
    // (1-0.9)*10 = 0.999... < 1 → floor = 0 rounding surprise we want
    // to avoid in the reference). nw=10 → var_index = floor(0.25*10) = 2.
    // Series: (-0.05, 0.01, -0.02, 0.03, -0.01, 0.02, -0.04, 0.00, -0.03, 0.01)
    // Sorted: -0.05, -0.04, -0.03, -0.02, -0.01, 0.00, 0.01, 0.01, 0.02, 0.03
    // var_index = 2 → sorted[2] = -0.03 → VaR = 0.03.
    const double xs[] = {-0.05, 0.01, -0.02, 0.03, -0.01,
                          0.02, -0.04, 0.00, -0.03, 0.01};
    double out[10] = {};
    HistoricalVaRState<16> st{};
    st.init(0, 1, 10, 0.75);
    execute_historical_var(&st, xs, 10, out);

    // Row 0: nw=1, var_index=floor(0.25*1)=0 → -sorted[0]=-(-0.05)=0.05.
    EXPECT_NEAR(out[0], 0.05, kEps);
    // Row 3: nw=4, sorted={-0.05,-0.02,0.01,0.03}, var_index=floor(1.0)=1
    //   → -sorted[1] = -(-0.02) = 0.02.
    EXPECT_NEAR(out[3], 0.02, kEps);
    // Final row 9: VaR = 0.03 as derived above.
    EXPECT_NEAR(out[9], 0.03, kEps);
}

TEST(Phase5S_HistoricalVaR, ConfidenceBoundaryZeroTail) {
    // When alpha * nw < 1 everywhere, var_index stays at 0 and VaR is
    // just -min(window). Verify that trivial case row-by-row on a short
    // increasing series.
    const double xs[] = {-0.01, -0.02, -0.03};
    double out[3] = {};
    HistoricalVaRState<8> st{};
    st.init(0, 1, 8, 0.99);  // alpha = 0.01; tiny tail
    execute_historical_var(&st, xs, 3, out);
    // Row 0: sorted=[-0.01], idx=0, VaR = 0.01.
    EXPECT_NEAR(out[0], 0.01, kEps);
    // Row 1: sorted=[-0.02,-0.01], idx=0, VaR = 0.02.
    EXPECT_NEAR(out[1], 0.02, kEps);
    // Row 2: sorted=[-0.03,-0.02,-0.01], idx=0, VaR = 0.03.
    EXPECT_NEAR(out[2], 0.03, kEps);
}

// ---------------------------------------------------------------------------
// P5.S.5 HistoricalCVaR — mean of tail beyond VaR threshold.
// ---------------------------------------------------------------------------
TEST(Phase5S_HistoricalCVaR, HandComputedWindow10) {
    // Same series as HistoricalVaR test. With confidence=0.75, alpha=0.25
    // (exact in binary), nw=10, var_index=2 → tail=sorted[0..2)={-0.05,-0.04}.
    // CVaR = -mean({-0.05,-0.04}) = -(-0.045) = 0.045.
    const double xs[] = {-0.05, 0.01, -0.02, 0.03, -0.01,
                          0.02, -0.04, 0.00, -0.03, 0.01};
    double out[10] = {};
    HistoricalCVaRState<16> st{};
    st.init(0, 1, 10, 0.75);
    execute_historical_cvar(&st, xs, 10, out);

    // Row 0: nw=1, var_index=0 → CVaR = 0.
    EXPECT_NEAR(out[0], 0.0, kEps);
    // Row 9: CVaR = 0.045 as computed above.
    EXPECT_NEAR(out[9], 0.045, kEps);
}

TEST(Phase5S_HistoricalCVaR, LowerConfidenceWidensTail) {
    // confidence=0.5 → alpha=0.5 (exact). nw=10, var_index=5,
    // tail=sorted[0..5) = {-0.05,-0.04,-0.03,-0.02,-0.01}, mean=-0.03,
    // CVaR = 0.03.
    const double xs[] = {-0.05, 0.01, -0.02, 0.03, -0.01,
                          0.02, -0.04, 0.00, -0.03, 0.01};
    double out[10] = {};
    HistoricalCVaRState<16> st{};
    st.init(0, 1, 10, 0.5);
    execute_historical_cvar(&st, xs, 10, out);
    EXPECT_NEAR(out[9], 0.03, kEps);
}

// ---------------------------------------------------------------------------
// P5.S.8 MedRV — batch-scalar median realised volatility.
// ---------------------------------------------------------------------------
TEST(Phase5S_MedRV, HandComputedFive) {
    // Returns: r = {0.01, -0.02, 0.03, -0.04, 0.05}, n = 5.
    //   i=1: median(|0.01|,|-0.02|,|0.03|) = median(0.01,0.02,0.03) = 0.02
    //   i=2: median(|-0.02|,|0.03|,|-0.04|) = median(0.02,0.03,0.04) = 0.03
    //   i=3: median(|0.03|,|-0.04|,|0.05|) = median(0.03,0.04,0.05) = 0.04
    // sum_med_sq = 0.02² + 0.03² + 0.04² = 0.0004 + 0.0009 + 0.0016 = 0.0029
    // scale = π / (6 - 4√3 + π)
    // bias  = 5 / 3
    const double xs[] = {0.01, -0.02, 0.03, -0.04, 0.05};
    double out[5] = {};
    MedRVState st{};
    st.init(0, 1);
    execute_med_rv(&st, xs, 5, out);

    const double scale = M_PI / (6.0 - 4.0 * std::sqrt(3.0) + M_PI);
    const double bias  = 5.0 / 3.0;
    const double expected = scale * bias * 0.0029;
    // Every row should hold the same scalar.
    for (int i = 0; i < 5; ++i) {
        EXPECT_NEAR(out[i], expected, 1e-12);
    }
}

TEST(Phase5S_MedRV, BelowMinRowsEmitsZero) {
    // n < 3 → medrv = 0 broadcast.
    const double xs[] = {0.01, -0.02};
    double out[2] = {};
    MedRVState st{};
    st.init(0, 1);
    execute_med_rv(&st, xs, 2, out);
    EXPECT_NEAR(out[0], 0.0, kEps);
    EXPECT_NEAR(out[1], 0.0, kEps);
}

// ---------------------------------------------------------------------------
// P5.S.14 OutlierFlagMAD — median + 1.4826·MAD outlier detection.
// ---------------------------------------------------------------------------
TEST(Phase5S_OutlierFlagMAD, TightClusterThenOutlier) {
    // Seven samples: tight cluster around 10.0, then a huge jump.
    // window = 7, k = 3.0.
    //
    // After 7 pushes, ring_x sorted = {9.9, 9.95, 10.0, 10.0, 10.05, 10.1, 50.0}
    // median = 10.0
    // abs devs = {0.1, 0.05, 0.0, 0.0, 0.05, 0.1, 40.0}
    // sorted_d = {0.0, 0.0, 0.05, 0.05, 0.1, 0.1, 40.0}
    // MAD = sorted_d[3] = 0.05
    // scaled_mad = 1.4826 * 0.05 = 0.07413
    // For row 6 (value=50.0): |50.0-10.0|/0.07413 ≈ 539 >> 3 → flag = 1.
    const double xs[] = {10.0, 10.05, 9.95, 10.1, 9.9, 10.0, 50.0};
    int64_t out[7] = {};
    OutlierFlagMADState<16> st{};
    st.init(0, 1, 7, 3.0);
    execute_outlier_flag_mad(&st, xs, 7, out);
    EXPECT_EQ(out[6], 1);
    // The first row has a single-sample window, median == x → absdev == 0
    // so the flag must be 0.
    EXPECT_EQ(out[0], 0);
}

TEST(Phase5S_OutlierFlagMAD, ConstantSeriesNoFlag) {
    // Degenerate cluster: scaled_mad ≈ 0 → flag always 0.
    const double xs[] = {7.0, 7.0, 7.0, 7.0, 7.0};
    int64_t out[5] = {};
    OutlierFlagMADState<8> st{};
    st.init(0, 1, 5, 3.0);
    execute_outlier_flag_mad(&st, xs, 5, out);
    for (int i = 0; i < 5; ++i) EXPECT_EQ(out[i], 0);
}

TEST(Phase5S_OutlierFlagMAD, NormalClusterInlier) {
    // window=5, cluster mean ~ 100 with stddev ~1. A value within k·MAD
    // of the median should NOT trip the flag.
    const double xs[] = {100.0, 101.0, 99.0, 100.5, 99.5};
    int64_t out[5] = {};
    OutlierFlagMADState<8> st{};
    st.init(0, 1, 5, 3.0);
    execute_outlier_flag_mad(&st, xs, 5, out);
    // All values are within ~1 of median; none should flag.
    for (int i = 0; i < 5; ++i) EXPECT_EQ(out[i], 0);
}
