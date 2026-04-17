// test_bolt_fintech_phase5e.cpp — Phase 5.E EMA-based Tier 2 kernels.
//
// Covers EWMA, EWCOV, MACD, RSI — all built on Phase 2's EmaState
// substrate. Each test round-trips against a hand-computed scalar
// reference; there is no Arrow dependency in-tree.

#include <gtest/gtest.h>

#include "bolt/bolt_arena.h"
#include "bolt/kernels/fintech/ewcov.h"
#include "bolt/kernels/fintech/ewma.h"
#include "bolt/kernels/fintech/macd.h"
#include "bolt/kernels/fintech/rsi.h"

#include <cmath>
#include <cstdint>

using bolt::Arena;
using namespace bolt::fintech;

namespace {
constexpr double kEps = 1e-12;
}

// ===========================================================================
// P5.E.2 EWMA — caller-supplied alpha
// ===========================================================================
TEST(Phase5E_EWMA, Alpha05Sequence) {
    // alpha = 0.5 on {10, 20, 30}:
    //   seed 10 -> 10
    //   0.5*20 + 0.5*10 = 15
    //   0.5*30 + 0.5*15 = 22.5
    Arena arena;
    EWMAState* s = make_ewma_state(&arena, /*in_col=*/0, /*out_col=*/1,
                                   /*alpha=*/0.5);
    ASSERT_NE(s, nullptr);

    const double x[] = {10.0, 20.0, 30.0};
    const double* in[] = {x};
    double out[3] = {};
    execute_ewma(s, in, 3, out);

    EXPECT_NEAR(out[0], 10.0,  kEps);
    EXPECT_NEAR(out[1], 15.0,  kEps);
    EXPECT_NEAR(out[2], 22.5,  kEps);
}

TEST(Phase5E_EWMA, StatePersistsAcrossCalls) {
    Arena arena;
    EWMAState* s = make_ewma_state(&arena, 0, 1, 0.5);
    ASSERT_NE(s, nullptr);

    const double x1[] = {10.0};
    const double* in1[] = {x1};
    double out1[1] = {};
    execute_ewma(s, in1, 1, out1);
    EXPECT_NEAR(out1[0], 10.0, kEps);

    const double x2[] = {20.0, 30.0};
    const double* in2[] = {x2};
    double out2[2] = {};
    execute_ewma(s, in2, 2, out2);
    EXPECT_NEAR(out2[0], 15.0,  kEps);
    EXPECT_NEAR(out2[1], 22.5,  kEps);
}

// ===========================================================================
// P5.E.3 EWCOV — two-variable exponentially-weighted covariance
// ===========================================================================
TEST(Phase5E_EWCOV, IdenticalSeriesCovEqualsVar) {
    // lambda = 0.5, x == y == {1,2,3,4,5}. Because x == y, the running
    // mean of x and y are identical and dx == dy, so cov == "variance"
    // under the same EW scheme. Hand-computed:
    //   i=0: seed mu_x=mu_y=1, cov=0       -> 0
    //   i=1: dx=dy=1;   mu_x=1.5;   cov=0.5*0   + 0.5*1*1    = 0.5
    //   i=2: dx=dy=1.5; mu_x=2.25;  cov=0.5*0.5 + 0.5*1.5*1.5 = 1.375
    //   i=3: dx=dy=1.75;mu_x=3.125; cov=0.5*1.375+0.5*1.75^2  = 2.21875
    //   i=4: dx=dy=1.875;           cov=0.5*2.21875+0.5*1.875^2 = 2.8671875
    Arena arena;
    EWCOVState* s = make_ewcov_state(&arena,
                                     /*x_col=*/0, /*y_col=*/1,
                                     /*out_col=*/2, /*lambda=*/0.5);
    ASSERT_NE(s, nullptr);

    const double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    const double y[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    const double* in[] = {x, y};
    double out[5] = {};
    execute_ewcov(s, in, 5, out);

    EXPECT_NEAR(out[0], 0.0,       kEps);
    EXPECT_NEAR(out[1], 0.5,       kEps);
    EXPECT_NEAR(out[2], 1.375,     kEps);
    EXPECT_NEAR(out[3], 2.21875,   kEps);
    EXPECT_NEAR(out[4], 2.8671875, kEps);
}

TEST(Phase5E_EWCOV, ConstantSeriesCovStaysZero) {
    // Two constant series -> dx = dy = 0 for every sample -> cov remains 0.
    Arena arena;
    EWCOVState* s = make_ewcov_state(&arena, 0, 1, 2, 0.94);
    ASSERT_NE(s, nullptr);

    const double x[] = {5.0, 5.0, 5.0, 5.0};
    const double y[] = {7.0, 7.0, 7.0, 7.0};
    const double* in[] = {x, y};
    double out[4] = {};
    execute_ewcov(s, in, 4, out);
    for (int i = 0; i < 4; ++i) EXPECT_NEAR(out[i], 0.0, kEps);
}

TEST(Phase5E_EWCOV, ResetReArms) {
    Arena arena;
    EWCOVState* s = make_ewcov_state(&arena, 0, 1, 2, 0.5);
    ASSERT_NE(s, nullptr);

    const double x[] = {1.0, 2.0};
    const double y[] = {1.0, 2.0};
    const double* in[] = {x, y};
    double out[2] = {};
    execute_ewcov(s, in, 2, out);
    EXPECT_NEAR(out[1], 0.5, kEps);

    ewcov_reset(s);
    EXPECT_FALSE(s->initialized);
    EXPECT_DOUBLE_EQ(s->cov, 0.0);

    // Re-feed fresh → same as a fresh state.
    execute_ewcov(s, in, 2, out);
    EXPECT_NEAR(out[0], 0.0, kEps);
    EXPECT_NEAR(out[1], 0.5, kEps);
}

// ===========================================================================
// P5.E.4 MACD — three EMAs chained
// ===========================================================================
TEST(Phase5E_MACD, ConstantInputAllZero) {
    // Any constant input: ema_fast = ema_slow = x; macd_line = 0 from
    // i=0 (seed path sets both to x). signal = EMA(0, signal_period) = 0.
    // histogram = 0. Holds for every i.
    Arena arena;
    MACDState* s = make_macd_state(&arena,
                                   /*in_col=*/0, /*line_col=*/1,
                                   /*signal_col=*/2, /*hist_col=*/3,
                                   /*fast=*/12, /*slow=*/26, /*signal=*/9);
    ASSERT_NE(s, nullptr);

    const int N = 16;
    double x[N];
    for (int i = 0; i < N; ++i) x[i] = 100.0;
    const double* in[] = {x};
    double line[N] = {}, sig[N] = {}, hist[N] = {};
    execute_macd(s, in, N, line, sig, hist);

    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(line[i], 0.0, kEps) << "line[" << i << "]";
        EXPECT_NEAR(sig[i],  0.0, kEps) << "sig["  << i << "]";
        EXPECT_NEAR(hist[i], 0.0, kEps) << "hist[" << i << "]";
    }
}

TEST(Phase5E_MACD, TwoStepHandComputed) {
    // fast=2, slow=4, signal=2.
    //   alpha_fast = 2/3 ≈ 0.666667
    //   alpha_slow = 2/5 = 0.4
    //   alpha_sig  = 2/3
    // Input {10, 20, 30}.
    //   i=0: seed all EMAs at 10. line=0, signal=0, hist=0.
    //   i=1: x=20.
    //     ema_fast = 2/3 * 20 + 1/3 * 10 = 16.666666...
    //     ema_slow = 0.4 * 20 + 0.6 * 10 = 14.0
    //     line     = 2.666666...
    //     signal   = 2/3 * 2.666666... + 1/3 * 0 = 1.777777...
    //     hist     = line - signal = 0.888888...
    //   i=2: x=30.
    //     ema_fast = 2/3*30 + 1/3*16.666666... = 20 + 5.555555... = 25.555555...
    //     ema_slow = 0.4*30 + 0.6*14 = 12 + 8.4 = 20.4
    //     line     = 5.155555...
    //     signal   = 2/3*5.155555... + 1/3*1.777777... = 3.437037... + 0.592593... = 4.029629...
    //     hist     = 5.155555... - 4.029629... = 1.125925...
    Arena arena;
    MACDState* s = make_macd_state(&arena, 0, 1, 2, 3,
                                   /*fast=*/2, /*slow=*/4, /*signal=*/2);
    ASSERT_NE(s, nullptr);

    const double x[] = {10.0, 20.0, 30.0};
    const double* in[] = {x};
    double line[3] = {}, sig[3] = {}, hist[3] = {};
    execute_macd(s, in, 3, line, sig, hist);

    // i=0
    EXPECT_NEAR(line[0], 0.0, kEps);
    EXPECT_NEAR(sig[0],  0.0, kEps);
    EXPECT_NEAR(hist[0], 0.0, kEps);

    // i=1
    const double af = 2.0 / 3.0;
    const double as = 0.4;
    const double ag = 2.0 / 3.0;
    const double ema_f1 = af * 20.0 + (1.0 - af) * 10.0;
    const double ema_s1 = as * 20.0 + (1.0 - as) * 10.0;
    const double line1  = ema_f1 - ema_s1;
    const double sig1   = ag * line1 + (1.0 - ag) * 0.0;
    const double hist1  = line1 - sig1;
    EXPECT_NEAR(line[1], line1, 1e-12);
    EXPECT_NEAR(sig[1],  sig1,  1e-12);
    EXPECT_NEAR(hist[1], hist1, 1e-12);

    // i=2
    const double ema_f2 = af * 30.0 + (1.0 - af) * ema_f1;
    const double ema_s2 = as * 30.0 + (1.0 - as) * ema_s1;
    const double line2  = ema_f2 - ema_s2;
    const double sig2   = ag * line2 + (1.0 - ag) * sig1;
    const double hist2  = line2 - sig2;
    EXPECT_NEAR(line[2], line2, 1e-12);
    EXPECT_NEAR(sig[2],  sig2,  1e-12);
    EXPECT_NEAR(hist[2], hist2, 1e-12);
}

TEST(Phase5E_MACD, ResetReArms) {
    Arena arena;
    MACDState* s = make_macd_state(&arena, 0, 1, 2, 3, 12, 26, 9);
    ASSERT_NE(s, nullptr);

    const double x[] = {1.0, 2.0, 3.0, 4.0};
    const double* in[] = {x};
    double line[4] = {}, sig[4] = {}, hist[4] = {};
    execute_macd(s, in, 4, line, sig, hist);
    EXPECT_TRUE(s->ema_fast.initialized);

    macd_reset(s);
    EXPECT_FALSE(s->ema_fast.initialized);
    EXPECT_FALSE(s->ema_slow.initialized);
    EXPECT_FALSE(s->ema_signal.initialized);

    // After reset, new run should mirror a fresh state.
    double line2[4] = {}, sig2[4] = {}, hist2[4] = {};
    execute_macd(s, in, 4, line2, sig2, hist2);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(line[i], line2[i], 1e-12);
        EXPECT_NEAR(sig[i],  sig2[i],  1e-12);
        EXPECT_NEAR(hist[i], hist2[i], 1e-12);
    }
}

// ===========================================================================
// P5.E.5 RSI — Wilder smoothing, alpha = 1/period
// ===========================================================================
TEST(Phase5E_RSI, FirstSampleIsNeutralFifty) {
    // Chukonu convention: first sample emits 50.0 (no prior to diff).
    Arena arena;
    RSIState* s = make_rsi_state(&arena, 0, 1, /*period=*/14);
    ASSERT_NE(s, nullptr);
    // Wilder alpha: 1/14 ≈ 0.0714285...
    EXPECT_NEAR(s->avg_gain.alpha, 1.0 / 14.0, 1e-15);
    EXPECT_NEAR(s->avg_loss.alpha, 1.0 / 14.0, 1e-15);

    const double x[] = {50.0};
    const double* in[] = {x};
    double out[1] = {};
    execute_rsi(s, in, 1, out);
    EXPECT_NEAR(out[0], 50.0, kEps);
}

TEST(Phase5E_RSI, AllGainsApproachesUpperBound) {
    // Strictly increasing sequence: every change is +1, avg_loss stays 0.
    // Chukonu treats avg_loss == 0 as RS = 100, RSI = 100 - 100/101
    // = 99.00990099... (NOT 100 exactly). We follow the source we port;
    // the value must be >= 99 for every non-seed sample.
    Arena arena;
    RSIState* s = make_rsi_state(&arena, 0, 1, /*period=*/14);
    ASSERT_NE(s, nullptr);

    const int N = 64;
    double x[N];
    for (int i = 0; i < N; ++i) x[i] = 10.0 + static_cast<double>(i);
    const double* in[] = {x};
    double out[N] = {};
    execute_rsi(s, in, N, out);

    EXPECT_NEAR(out[0], 50.0, kEps);     // seed row = neutral.
    for (int i = 1; i < N; ++i) {
        EXPECT_GT(out[i], 99.0)      << "i=" << i;
        EXPECT_LT(out[i], 100.0 + kEps) << "i=" << i;
    }
    // By the time period has elapsed (i >= 14), RSI is essentially at the
    // chukonu ceiling 99.00990099...
    const double ceiling = 100.0 - 100.0 / 101.0;
    for (int i = 14; i < N; ++i) {
        EXPECT_NEAR(out[i], ceiling, 1e-9) << "i=" << i;
    }
}

TEST(Phase5E_RSI, AllLossesApproachesZero) {
    // Strictly decreasing sequence: change = -1 every step. avg_gain = 0,
    // RS = 0, RSI = 0.
    Arena arena;
    RSIState* s = make_rsi_state(&arena, 0, 1, 14);
    ASSERT_NE(s, nullptr);

    const int N = 32;
    double x[N];
    for (int i = 0; i < N; ++i) x[i] = 100.0 - static_cast<double>(i);
    const double* in[] = {x};
    double out[N] = {};
    execute_rsi(s, in, N, out);

    EXPECT_NEAR(out[0], 50.0, kEps);
    for (int i = 1; i < N; ++i) {
        EXPECT_NEAR(out[i], 0.0, 1e-12) << "i=" << i;
    }
}

TEST(Phase5E_RSI, MixedSequenceKnownRatio) {
    // period=2, alpha = 1/2 = 0.5.
    // x = {10, 11, 10, 11, 10} — alternating +1, -1, +1, -1.
    // Seeded: avg_gain=0, avg_loss=0, has_prev=false.
    // i=0: seed -> RSI=50, prev=10.
    // i=1: x=11, change=+1, gain=1, loss=0.
    //      avg_gain = 0.5*1 + 0.5*0 = 0.5
    //      avg_loss = 0.5*0 + 0.5*0 = 0.0
    //      avg_loss == 0 -> RS=100, RSI = 100 - 100/101 ≈ 99.0099.
    // i=2: x=10, change=-1, gain=0, loss=1.
    //      avg_gain = 0.5*0 + 0.5*0.5 = 0.25
    //      avg_loss = 0.5*1 + 0.5*0.0 = 0.5
    //      RS = 0.25/0.5 = 0.5; RSI = 100 - 100/1.5 = 33.333...
    // i=3: x=11, change=+1, gain=1, loss=0.
    //      avg_gain = 0.5*1 + 0.5*0.25 = 0.625
    //      avg_loss = 0.5*0 + 0.5*0.5  = 0.25
    //      RS = 2.5; RSI = 100 - 100/3.5 = 71.42857...
    // i=4: x=10, change=-1, gain=0, loss=1.
    //      avg_gain = 0.5*0 + 0.5*0.625 = 0.3125
    //      avg_loss = 0.5*1 + 0.5*0.25  = 0.625
    //      RS = 0.5; RSI = 100 - 100/1.5 = 33.333...
    Arena arena;
    RSIState* s = make_rsi_state(&arena, 0, 1, 2);
    ASSERT_NE(s, nullptr);

    const double x[] = {10.0, 11.0, 10.0, 11.0, 10.0};
    const double* in[] = {x};
    double out[5] = {};
    execute_rsi(s, in, 5, out);

    EXPECT_NEAR(out[0], 50.0,                              1e-12);
    EXPECT_NEAR(out[1], 100.0 - 100.0 / 101.0,             1e-9);
    EXPECT_NEAR(out[2], 100.0 - 100.0 / 1.5,               1e-12);
    EXPECT_NEAR(out[3], 100.0 - 100.0 / 3.5,               1e-12);
    EXPECT_NEAR(out[4], 100.0 - 100.0 / 1.5,               1e-12);
}

TEST(Phase5E_RSI, StatePersistsAcrossCalls) {
    // Splitting a series across two calls must produce the same output
    // as running it in one call.
    Arena arena;
    RSIState* s1 = make_rsi_state(&arena, 0, 1, 2);
    RSIState* s2 = make_rsi_state(&arena, 0, 1, 2);
    ASSERT_NE(s1, nullptr);
    ASSERT_NE(s2, nullptr);

    const double x[] = {10.0, 11.0, 10.0, 11.0, 10.0};
    const double* in_full[] = {x};
    double ref[5] = {};
    execute_rsi(s1, in_full, 5, ref);

    // Split {10,11} then {10,11,10}.
    const double x_a[] = {10.0, 11.0};
    const double x_b[] = {10.0, 11.0, 10.0};
    const double* in_a[] = {x_a};
    const double* in_b[] = {x_b};
    double out_a[2] = {}, out_b[3] = {};
    execute_rsi(s2, in_a, 2, out_a);
    execute_rsi(s2, in_b, 3, out_b);

    EXPECT_NEAR(out_a[0], ref[0], 1e-12);
    EXPECT_NEAR(out_a[1], ref[1], 1e-12);
    EXPECT_NEAR(out_b[0], ref[2], 1e-12);
    EXPECT_NEAR(out_b[1], ref[3], 1e-12);
    EXPECT_NEAR(out_b[2], ref[4], 1e-12);
}
