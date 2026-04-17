// test_bolt_fintech_phase5s_kalman.cpp — Phase 5.S Kalman / rolling
// regression cluster (Kalman1D, KalmanHedgeRatio, KyleLambda, RollSpread,
// CornishFisherVaR, Amihud).
//
// Each test round-trips against a hand-computed scalar reference; no Arrow
// dependency in-tree. Warmup / guard conventions match chukonu verbatim —
// see each kernel header for the contract.

#include <gtest/gtest.h>

#include "bolt/bolt_arena.h"
#include "bolt/kernels/fintech/amihud.h"
#include "bolt/kernels/fintech/cornish_fisher_var.h"
#include "bolt/kernels/fintech/kalman1d.h"
#include "bolt/kernels/fintech/kalman_hedge_ratio.h"
#include "bolt/kernels/fintech/kyle_lambda.h"
#include "bolt/kernels/fintech/roll_spread.h"

#include <cmath>
#include <cstdint>

using bolt::Arena;
using namespace bolt::fintech;

namespace {
constexpr double kEps = 1e-9;
}  // namespace

// ===========================================================================
// P5.S.6 Kalman1D — scalar single-state Kalman filter.
// ===========================================================================
TEST(Phase5S_Kalman1D, RampSignalHandComputed) {
    // z = {0, 2, 4, 6}, q=0, r=1, p_init=1.0. Chukonu seeds from obs[0].
    //   Row 0: seed x=0, P=1.
    //   Row 1: P_pred=1, K=0.5, x = 0 + 0.5*(2-0) = 1.0, P = 0.5.
    //   Row 2: P_pred=0.5, K=1/3, x = 1 + (1/3)*(4-1) = 2.0, P = 1/3.
    //   Row 3: P_pred=1/3, K=0.25, x = 2 + 0.25*(6-2) = 3.0, P = 0.25.
    Arena arena;
    Kalman1DState* s = make_kalman1d_state(&arena, /*obs_col=*/0,
                                           /*out_col=*/1,
                                           /*q=*/0.0, /*r=*/1.0);
    ASSERT_NE(s, nullptr);

    const double z[] = {0.0, 2.0, 4.0, 6.0};
    double out[4] = {};
    execute_kalman1d(s, z, 4, out);

    EXPECT_NEAR(out[0], 0.0, kEps);
    EXPECT_NEAR(out[1], 1.0, kEps);
    EXPECT_NEAR(out[2], 2.0, kEps);
    EXPECT_NEAR(out[3], 3.0, kEps);
}

TEST(Phase5S_Kalman1D, ConstantSignalConverges) {
    // z = {1,1,1,1}, q=0, r=1. Seed x=1 from obs[0]. All subsequent rows
    // emit 1 (innovation z - x = 0 ⇒ no update).
    Arena arena;
    Kalman1DState* s = make_kalman1d_state(&arena, 0, 1, 0.0, 1.0);
    ASSERT_NE(s, nullptr);
    const double z[] = {1.0, 1.0, 1.0, 1.0};
    double out[4] = {};
    execute_kalman1d(s, z, 4, out);
    for (int i = 0; i < 4; ++i) EXPECT_NEAR(out[i], 1.0, kEps);
}

TEST(Phase5S_Kalman1D, StatePersistsAcrossCalls) {
    Arena arena;
    Kalman1DState* s = make_kalman1d_state(&arena, 0, 1, 0.0, 1.0);
    ASSERT_NE(s, nullptr);
    const double z1[] = {0.0, 2.0};
    double out1[2] = {};
    execute_kalman1d(s, z1, 2, out1);
    EXPECT_NEAR(out1[0], 0.0, kEps);
    EXPECT_NEAR(out1[1], 1.0, kEps);

    const double z2[] = {4.0, 6.0};
    double out2[2] = {};
    execute_kalman1d(s, z2, 2, out2);
    EXPECT_NEAR(out2[0], 2.0, kEps);
    EXPECT_NEAR(out2[1], 3.0, kEps);
}

// ===========================================================================
// P5.S.7 KalmanHedgeRatio — hedge ratio via 1-D Kalman on pa/pb.
// ===========================================================================
TEST(Phase5S_KalmanHedgeRatio, LinearRelationConvergesToTwo) {
    // pa = 2 * pb exactly → observation = 2 at every row. With beta_init=1,
    // the filter drives beta toward 2 at rate governed by q, r. Use
    // q=0.01, r=0.0001 (tight measurement noise) to converge quickly.
    Arena arena;
    KalmanHedgeRatioState* s = make_kalman_hedge_ratio_state(
        &arena, /*pa=*/0, /*pb=*/1, /*ratio=*/2, /*spread=*/3,
        /*q=*/0.01, /*r=*/0.0001);
    ASSERT_NE(s, nullptr);
    constexpr int N = 32;
    double pa[N], pb[N];
    for (int i = 0; i < N; ++i) {
        pb[i] = 1.0 + static_cast<double>(i) * 0.5;  // 1, 1.5, 2, 2.5, ...
        pa[i] = 2.0 * pb[i];
    }
    double ratio[N] = {}, spread[N] = {};
    execute_kalman_hedge_ratio(s, pa, pb, N, ratio, spread);
    // After enough steps beta should be within 0.01 of 2.
    EXPECT_NEAR(ratio[N - 1], 2.0, 0.01);
    // Spread = pa - beta*pb → ~0 once beta≈2.
    EXPECT_NEAR(spread[N - 1], 0.0, 0.05);
}

TEST(Phase5S_KalmanHedgeRatio, FirstRowRecurrenceHandComputed) {
    // q=0.01, r=0.01. Row 0: pa=2, pb=1. beta_pred=1, P_pred=1+0.01=1.01.
    //   obs = 2/1 = 2. K = 1.01 / (1.01+0.01) = 1.01/1.02.
    //   beta = 1 + (1.01/1.02)*(2-1) = 1 + 1.01/1.02.
    //   P    = (1 - 1.01/1.02) * 1.01 = (0.01/1.02) * 1.01.
    //   spread = 2 - beta * 1 = 2 - beta.
    Arena arena;
    KalmanHedgeRatioState* s = make_kalman_hedge_ratio_state(
        &arena, 0, 1, 2, 3, /*q=*/0.01, /*r=*/0.01);
    ASSERT_NE(s, nullptr);

    const double pa[] = {2.0};
    const double pb[] = {1.0};
    double ratio[1] = {}, spread[1] = {};
    execute_kalman_hedge_ratio(s, pa, pb, 1, ratio, spread);
    const double expected_beta = 1.0 + 1.01 / 1.02;
    EXPECT_NEAR(ratio[0], expected_beta, kEps);
    EXPECT_NEAR(spread[0], 2.0 - expected_beta, kEps);
}

TEST(Phase5S_KalmanHedgeRatio, NearZeroPbFallsBackToPrediction) {
    // pb≈0 → observation fallback = beta_pred → innovation = 0 → beta
    // unchanged. P still inflates by q and contracts by (1-K).
    Arena arena;
    KalmanHedgeRatioState* s = make_kalman_hedge_ratio_state(
        &arena, 0, 1, 2, 3, /*q=*/0.01, /*r=*/0.01,
        /*beta_init=*/1.5, /*p_init=*/1.0);
    ASSERT_NE(s, nullptr);

    const double pa[] = {3.0};
    const double pb[] = {1e-20};  // triggers fallback
    double ratio[1] = {}, spread[1] = {};
    execute_kalman_hedge_ratio(s, pa, pb, 1, ratio, spread);
    EXPECT_NEAR(ratio[0], 1.5, kEps);  // beta unchanged
}

// ===========================================================================
// P5.S.11 KyleLambda — rolling OLS slope of dp on sv.
// ===========================================================================
TEST(Phase5S_KyleLambda, LinearImpactHandComputed) {
    // dp = 0.1*sv. Rolling slope should be 0.1 once window fills.
    // dp = {0.1, 0.2, 0.3, 0.4}, sv = {1, 2, 3, 4}, window=3.
    //   Row 0: ring size 1, warmup → 0.
    //   Row 1: {(0.1,1),(0.2,2)}, mean_dp=0.15, mean_sv=1.5.
    //     cov = (-0.05)(-0.5)+(0.05)(0.5) = 0.05
    //     var_sv = 0.25+0.25 = 0.5 → lambda = 0.1
    //   Row 2: {(0.1,1),(0.2,2),(0.3,3)}, mean_dp=0.2, mean_sv=2.
    //     cov = 0.2, var_sv = 2 → lambda = 0.1
    //   Row 3: {(0.2,2),(0.3,3),(0.4,4)}, lambda = 0.1
    Arena arena;
    KyleLambdaState<8>* s = make_kyle_lambda_state<8>(
        &arena, /*dp=*/0, /*sv=*/1, /*out=*/2, /*window=*/3);
    ASSERT_NE(s, nullptr);

    const double dp[] = {0.1, 0.2, 0.3, 0.4};
    const double sv[] = {1.0, 2.0, 3.0, 4.0};
    double out[4] = {};
    execute_kyle_lambda(s, dp, sv, 4, out);

    EXPECT_NEAR(out[0], 0.0, kEps);
    EXPECT_NEAR(out[1], 0.1, kEps);
    EXPECT_NEAR(out[2], 0.1, kEps);
    EXPECT_NEAR(out[3], 0.1, kEps);
}

TEST(Phase5S_KyleLambda, ZeroVarianceGuardYieldsZero) {
    // sv constant → var_sv = 0 → guard → 0.
    Arena arena;
    KyleLambdaState<8>* s = make_kyle_lambda_state<8>(&arena, 0, 1, 2, 3);
    ASSERT_NE(s, nullptr);
    const double dp[] = {0.1, 0.2, 0.3, 0.4};
    const double sv[] = {5.0, 5.0, 5.0, 5.0};
    double out[4] = {};
    execute_kyle_lambda(s, dp, sv, 4, out);
    // Row 0 warmup, rows 1..3 steady-with-zero-var → 0.
    for (int i = 0; i < 4; ++i) EXPECT_NEAR(out[i], 0.0, kEps);
}

// ===========================================================================
// P5.S.12 RollSpread — 2 * sqrt(-cov(Δp, Δp_{-1})) gated on cov<0.
// ===========================================================================
TEST(Phase5S_RollSpread, OscillatingPricesNegativeCov) {
    // prices = {10, 11, 10, 11, 10, 11}, window=3.
    // Bid-ask-bounce pattern → cov(Δp, Δp_{-1}) = -8/9 at rows 4,5.
    //   spread = 2 * sqrt(8/9) = 4*sqrt(2)/3 ≈ 1.8856.
    Arena arena;
    RollSpreadState<8>* s = make_roll_spread_state<8>(
        &arena, /*price=*/0, /*out=*/1, /*window=*/3);
    ASSERT_NE(s, nullptr);

    const double px[] = {10.0, 11.0, 10.0, 11.0, 10.0, 11.0};
    double out[6] = {};
    execute_roll_spread(s, px, 6, out);

    // Rows 0..3 are warmup (chukonu emits 0.0).
    EXPECT_NEAR(out[0], 0.0, kEps);
    EXPECT_NEAR(out[1], 0.0, kEps);
    EXPECT_NEAR(out[2], 0.0, kEps);
    EXPECT_NEAR(out[3], 0.0, kEps);
    const double expected = 4.0 * std::sqrt(2.0) / 3.0;
    EXPECT_NEAR(out[4], expected, kEps);
    EXPECT_NEAR(out[5], expected, kEps);
}

TEST(Phase5S_RollSpread, TrendingPricesPositiveCovYieldsZero) {
    // Monotone ramp → diffs all = 1 → cov of constant = 0 → clamp to 0.
    Arena arena;
    RollSpreadState<8>* s = make_roll_spread_state<8>(&arena, 0, 1, 3);
    ASSERT_NE(s, nullptr);
    const double px[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double out[6] = {};
    execute_roll_spread(s, px, 6, out);
    for (int i = 0; i < 6; ++i) EXPECT_NEAR(out[i], 0.0, kEps);
}

TEST(Phase5S_RollSpread, StatePersistsAcrossCalls) {
    // Same oscillating series, split across two execute calls — must match
    // the single-call output.
    Arena arena;
    RollSpreadState<8>* s = make_roll_spread_state<8>(&arena, 0, 1, 3);
    ASSERT_NE(s, nullptr);
    const double px1[] = {10.0, 11.0, 10.0};
    const double px2[] = {11.0, 10.0, 11.0};
    double out[6] = {};
    execute_roll_spread(s, px1, 3, out + 0);
    execute_roll_spread(s, px2, 3, out + 3);
    const double expected = 4.0 * std::sqrt(2.0) / 3.0;
    EXPECT_NEAR(out[4], expected, kEps);
    EXPECT_NEAR(out[5], expected, kEps);
}

// ===========================================================================
// P5.S.9 CornishFisherVaR — parametric VaR with skew/kurt correction.
// ===========================================================================
TEST(Phase5S_CornishFisherVaR, SymmetricSeriesMatchesGaussianVaR) {
    // Symmetric series {−0.02, −0.01, 0.0, 0.01, 0.02} has skew=0 and
    // excess kurt ≈ -1.3 (platykurtic). Compute by hand.
    //   mean = 0.
    //   m2 = (4e-4 + 1e-4 + 0 + 1e-4 + 4e-4)/5 = 1e-3/5 = 2e-4.
    //   stddev = sqrt(2e-4) = 0.01 * sqrt(2).
    //   m3 = 0  (symmetric).
    //   m4 = (16e-8 + 1e-8 + 0 + 1e-8 + 16e-8)/5 = 34e-8/5 = 6.8e-8.
    //   kurt_raw = 6.8e-8 / (2e-4)² = 6.8e-8 / 4e-8 = 1.7.
    //   excess_kurt = 1.7 - 3 = -1.3.
    // confidence = 0.95 → p = 0.05. z_normal ≈ -1.64485 (from A&S 26.2.3).
    // CF adjustment: skew=0 kills the (z²-1)·skew/6 and the (2z³-5z)·skew²/36
    // terms; only the (z³-3z)·kurt/24 term survives.
    //   z = -1.64485..., z³ = -4.4492..., (z³ - 3z) = -4.4492 + 4.9345 = 0.4854.
    //   cf_z = z + 0.4854 * (-1.3) / 24 = z - 0.02629.
    //          ≈ -1.64485 - 0.02629 = -1.67114.
    //   var = -(0 + cf_z * stddev) = -(-1.67114 * 0.014142) = 0.02364.
    Arena arena;
    CornishFisherVaRState* s = make_cornish_fisher_var_state(
        &arena, /*return_col=*/0, /*out_col=*/1, /*confidence=*/0.95);
    ASSERT_NE(s, nullptr);

    const double r[] = {-0.02, -0.01, 0.0, 0.01, 0.02};
    double out[5] = {};
    execute_cornish_fisher_var(s, r, 5, out);

    // All rows broadcast the same scalar.
    for (int i = 1; i < 5; ++i) EXPECT_NEAR(out[i], out[0], kEps);
    // Allow 1e-3 tolerance for the A&S rational-approximation error on z.
    EXPECT_NEAR(out[0], 0.02364, 1e-3);
}

TEST(Phase5S_CornishFisherVaR, ZeroVarianceGuardYieldsNegativeMean) {
    // Constant returns → stddev=0, skew=0, kurt=0 by guard. cf_z = z.
    // var = -(mean + z*0) = -mean. With mean=0.01, confidence=0.95 → var = -0.01.
    Arena arena;
    CornishFisherVaRState* s = make_cornish_fisher_var_state(&arena, 0, 1, 0.95);
    ASSERT_NE(s, nullptr);
    const double r[] = {0.01, 0.01, 0.01, 0.01};
    double out[4] = {};
    execute_cornish_fisher_var(s, r, 4, out);
    EXPECT_NEAR(out[0], -0.01, kEps);
    for (int i = 1; i < 4; ++i) EXPECT_NEAR(out[i], out[0], kEps);
}

// ===========================================================================
// P5.S.10 Amihud — |r|/v with v>0 guard, per-row (no rolling window).
// ===========================================================================
TEST(Phase5S_Amihud, HandComputedSequence) {
    // r = {0.02, -0.04, 0.0, 0.01},  v = {100, 200, 50, 0}.
    //   row 0: 0.02 / 100 = 2e-4
    //   row 1: 0.04 / 200 = 2e-4  (fabs on negative)
    //   row 2: 0.0  / 50  = 0
    //   row 3: v = 0 → guard → 0
    Arena arena;
    AmihudState* s = make_amihud_state(&arena, /*r=*/0, /*v=*/1, /*out=*/2);
    ASSERT_NE(s, nullptr);
    const double r[] = {0.02, -0.04, 0.0, 0.01};
    const double v[] = {100.0, 200.0, 50.0, 0.0};
    double out[4] = {};
    execute_amihud(s, r, v, 4, out);
    EXPECT_NEAR(out[0], 2e-4, kEps);
    EXPECT_NEAR(out[1], 2e-4, kEps);
    EXPECT_NEAR(out[2], 0.0,  kEps);
    EXPECT_NEAR(out[3], 0.0,  kEps);
}

TEST(Phase5S_Amihud, NegativeVolumeGuarded) {
    // Guard is v > 0 — zero AND negative v both → 0.
    Arena arena;
    AmihudState* s = make_amihud_state(&arena, 0, 1, 2);
    ASSERT_NE(s, nullptr);
    const double r[] = {0.01};
    const double v[] = {-10.0};  // shouldn't happen in real data but guard it.
    double out[1] = {};
    execute_amihud(s, r, v, 1, out);
    EXPECT_NEAR(out[0], 0.0, kEps);
}
