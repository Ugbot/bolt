// test_bolt_fintech_phase5r.cpp — Phase 5.R RollingRing-based Tier 2 kernels.
//
// Seven kernels, one file. Each has a round-trip test against a hand-
// computed scalar reference. Warm-up rows emit NaN (matches the Phase 5.R
// contract — see per-kernel headers for deviations from chukonu).

#include <gtest/gtest.h>

#include "bolt/kernels/fintech/atr.h"
#include "bolt/kernels/fintech/bollinger_bands.h"
#include "bolt/kernels/fintech/donchian_channel.h"
#include "bolt/kernels/fintech/rolling_min_max.h"
#include "bolt/kernels/fintech/sma.h"
#include "bolt/kernels/fintech/stochastic_osc.h"
#include "bolt/kernels/fintech/vwap_twap.h"

#include <cmath>
#include <cstdint>

using namespace bolt::fintech;

namespace {

constexpr double kEps = 1e-9;

inline bool is_nan(double x) noexcept { return x != x; }

}  // namespace

// ===========================================================================
// P5.R.1  SMA
// ===========================================================================
TEST(Phase5R_SMA, Window3OnFiveRows) {
    const double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double out[5] = {};
    const double* in[] = {x};

    SMAState<8> state{};
    state.init(/*in*/0, /*out*/1, /*window*/3);
    execute_sma(&state, in, 5, out);

    EXPECT_TRUE(is_nan(out[0]));
    EXPECT_TRUE(is_nan(out[1]));
    EXPECT_NEAR(out[2], 2.0, kEps);    // (1+2+3)/3
    EXPECT_NEAR(out[3], 3.0, kEps);    // (2+3+4)/3
    EXPECT_NEAR(out[4], 4.0, kEps);    // (3+4+5)/3
}

TEST(Phase5R_SMA, Window1IsIdentity) {
    const double x[] = {10.0, 20.0, -5.0};
    double out[3] = {};
    const double* in[] = {x};

    SMAState<4> state{};
    state.init(0, 1, 1);
    execute_sma(&state, in, 3, out);
    EXPECT_NEAR(out[0], 10.0, kEps);
    EXPECT_NEAR(out[1], 20.0, kEps);
    EXPECT_NEAR(out[2], -5.0, kEps);
}

// ===========================================================================
// P5.R.2  BollingerBands
// ===========================================================================
TEST(Phase5R_BollingerBands, ConstantInputYieldsZeroStddev) {
    const double x[] = {7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5};
    double up[7] = {}, mid[7] = {}, lo[7] = {};
    const double* in[] = {x};

    BollingerBandsState<8> state{};
    state.init(/*in*/0, /*up*/1, /*mid*/2, /*lo*/3, /*w*/5, /*k*/2.0);
    execute_bollinger_bands(&state, in, 7, up, mid, lo);

    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(is_nan(up[i]));
        EXPECT_TRUE(is_nan(mid[i]));
        EXPECT_TRUE(is_nan(lo[i]));
    }
    for (int i = 4; i < 7; ++i) {
        EXPECT_NEAR(mid[i], 7.5, kEps);
        EXPECT_NEAR(up[i],  7.5, kEps);
        EXPECT_NEAR(lo[i],  7.5, kEps);
    }
}

TEST(Phase5R_BollingerBands, KnownVarianceWindow) {
    // Window = 4, k = 2.0. Input {2, 4, 4, 4, 5, 5, 7, 9}
    // When the full window first fills (i=3): values = {2,4,4,4}
    //   mean = 3.5
    //   pop var = ((2-3.5)² + 3·(4-3.5)²) / 4 = (2.25 + 0.75) / 4 = 0.75
    //   stddev = sqrt(0.75) ≈ 0.8660254037844386
    const double x[] = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    double up[8] = {}, mid[8] = {}, lo[8] = {};
    const double* in[] = {x};

    BollingerBandsState<8> state{};
    state.init(0, 1, 2, 3, 4, 2.0);
    execute_bollinger_bands(&state, in, 8, up, mid, lo);

    const double sd = std::sqrt(0.75);
    EXPECT_NEAR(mid[3], 3.5, kEps);
    EXPECT_NEAR(up[3], 3.5 + 2.0 * sd, 1e-8);
    EXPECT_NEAR(lo[3], 3.5 - 2.0 * sd, 1e-8);
}

// ===========================================================================
// P5.R.3  DonchianChannel
// ===========================================================================
TEST(Phase5R_Donchian, Window3OnFiveRows) {
    // Using the same sequence for both high and low (simplifies the check)
    const double h[] = {1.0, 3.0, 2.0, 5.0, 4.0};
    const double l[] = {1.0, 3.0, 2.0, 5.0, 4.0};
    double up[5] = {}, lo[5] = {};
    const double* in[] = {h, l};

    DonchianChannelState<8> state{};
    state.init(/*high*/0, /*low*/1, /*up*/2, /*lo*/3, /*w*/3);
    execute_donchian_channel(&state, in, 5, up, lo);

    EXPECT_TRUE(is_nan(up[0]));
    EXPECT_TRUE(is_nan(up[1]));
    EXPECT_NEAR(up[2], 3.0, kEps);  // max(1,3,2)
    EXPECT_NEAR(up[3], 5.0, kEps);  // max(3,2,5)
    EXPECT_NEAR(up[4], 5.0, kEps);  // max(2,5,4)

    EXPECT_TRUE(is_nan(lo[0]));
    EXPECT_TRUE(is_nan(lo[1]));
    EXPECT_NEAR(lo[2], 1.0, kEps);  // min(1,3,2)
    EXPECT_NEAR(lo[3], 2.0, kEps);  // min(3,2,5)
    EXPECT_NEAR(lo[4], 2.0, kEps);  // min(2,5,4)
}

// ===========================================================================
// P5.R.4  ATR
// ===========================================================================
TEST(Phase5R_ATR, Window2OnThreeRows) {
    // Row 0: H=10, L=8  → TR = 10-8 = 2    (no prev close)
    // Row 1: H=12, L=9, prev_close=9 → TR = max(3, |12-9|=3, |9-9|=0) = 3
    // Row 2: H=11, L=7, prev_close=11 → TR = max(4, |11-11|=0, |7-11|=4) = 4
    // Window 2 → ATR warmup = NaN, NaN, (3+4)/2 = 3.5  (row 0 NaN since
    // only 1 sample in ring; row 1: ring has {2, 3}, ATR = 2.5; row 2:
    // ring has {3, 4}, ATR = 3.5)
    //
    // Wait: with w=2, ring becomes full after 2 pushes → row 1 is first
    // valid output. Let's re-check:
    //   row 0: push TR=2  → ring.size()=1 < 2 → NaN
    //   row 1: push TR=3  → ring.size()=2 == 2 → mean({2,3}) = 2.5
    //   row 2: push TR=4  → ring evicts 2, ring={3,4} → mean = 3.5
    const double h[] = {10.0, 12.0, 11.0};
    const double l[] = { 8.0,  9.0,  7.0};
    const double c[] = { 9.0, 11.0, 10.0};
    double out[3] = {};
    const double* in[] = {h, l, c};

    ATRState<4> state{};
    state.init(/*h*/0, /*l*/1, /*c*/2, /*out*/3, /*w*/2);
    execute_atr(&state, in, 3, out);

    EXPECT_TRUE(is_nan(out[0]));
    EXPECT_NEAR(out[1], 2.5, kEps);
    EXPECT_NEAR(out[2], 3.5, kEps);
}

// ===========================================================================
// P5.R.5  VWAP / TWAP (rolling)
// ===========================================================================
TEST(Phase5R_VWAP, Window2BasicTwoRows) {
    // Row 0: price=10 vol=1 → ring.size()=1 < 2 → NaN
    // Row 1: price=20 vol=3 → ring.size()=2; Σpv = 10 + 60 = 70, Σv = 4
    //        → 70/4 = 17.5
    const double p[] = {10.0, 20.0};
    const double v[] = { 1.0,  3.0};
    double out[2] = {};
    const double* in[] = {p, v};

    VWAPState<4> state{};
    state.init(/*price*/0, /*volume*/1, /*out*/2, /*w*/2);
    execute_vwap_rolling(&state, in, 2, out);

    EXPECT_TRUE(is_nan(out[0]));
    EXPECT_NEAR(out[1], 17.5, kEps);
}

TEST(Phase5R_VWAP, ZeroVolumeWindowEmitsNaN) {
    const double p[] = {10.0, 20.0};
    const double v[] = { 0.0,  0.0};
    double out[2] = {};
    const double* in[] = {p, v};

    VWAPState<4> state{};
    state.init(0, 1, 2, 2);
    execute_vwap_rolling(&state, in, 2, out);
    EXPECT_TRUE(is_nan(out[0]));
    EXPECT_TRUE(is_nan(out[1]));
}

TEST(Phase5R_TWAP, Window3IsRollingMean) {
    const double p[] = {10.0, 20.0, 30.0, 40.0};
    double out[4] = {};
    const double* in[] = {p};

    TWAPState<4> state{};
    state.init(0, 1, 3);
    execute_twap_rolling(&state, in, 4, out);
    EXPECT_TRUE(is_nan(out[0]));
    EXPECT_TRUE(is_nan(out[1]));
    EXPECT_NEAR(out[2], 20.0, kEps);        // (10+20+30)/3
    EXPECT_NEAR(out[3], 30.0, kEps);        // (20+30+40)/3
}

// ===========================================================================
// P5.R.6  RollingMin / RollingMax
// ===========================================================================
TEST(Phase5R_RollingMinMax, Window3OnFiveRows) {
    const double x[] = {5.0, 1.0, 3.0, 2.0, 4.0};

    double mn[5] = {}, mx[5] = {};
    const double* in[] = {x};

    RollingMinState<8> s_min{};
    s_min.init(0, 1, 3);
    execute_rolling_min(&s_min, in, 5, mn);

    RollingMaxState<8> s_max{};
    s_max.init(0, 1, 3);
    execute_rolling_max(&s_max, in, 5, mx);

    // mins: NaN, NaN, min(5,1,3)=1, min(1,3,2)=1, min(3,2,4)=2
    EXPECT_TRUE(is_nan(mn[0]));
    EXPECT_TRUE(is_nan(mn[1]));
    EXPECT_NEAR(mn[2], 1.0, kEps);
    EXPECT_NEAR(mn[3], 1.0, kEps);
    EXPECT_NEAR(mn[4], 2.0, kEps);

    // maxes: NaN, NaN, max(5,1,3)=5, max(1,3,2)=3, max(3,2,4)=4
    EXPECT_TRUE(is_nan(mx[0]));
    EXPECT_TRUE(is_nan(mx[1]));
    EXPECT_NEAR(mx[2], 5.0, kEps);
    EXPECT_NEAR(mx[3], 3.0, kEps);
    EXPECT_NEAR(mx[4], 4.0, kEps);
}

TEST(Phase5R_RollingMinMax, MonotonicIncreasing) {
    // Each new value replaces the previous extrema: max follows the input.
    const double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double mx[5] = {};
    const double* in[] = {x};
    RollingMaxState<8> s{};
    s.init(0, 1, 2);
    execute_rolling_max(&s, in, 5, mx);
    EXPECT_TRUE(is_nan(mx[0]));
    EXPECT_NEAR(mx[1], 2.0, kEps);
    EXPECT_NEAR(mx[2], 3.0, kEps);
    EXPECT_NEAR(mx[3], 4.0, kEps);
    EXPECT_NEAR(mx[4], 5.0, kEps);
}

// ===========================================================================
// P5.R.7  StochasticOsc (%K)
// ===========================================================================
TEST(Phase5R_StochasticOsc, Window3HandComputed) {
    // high / low / close over 4 rows, window=3.
    // i=0: warmup NaN
    // i=1: warmup NaN
    // i=2: range = max(h[0..2]) - min(l[0..2]) = 5 - 1 = 4,
    //      K = 100 * (close[2] - 1) / 4
    // i=3: range = max(h[1..3]) - min(l[1..3]) = 5 - 2 = 3,
    //      K = 100 * (close[3] - 2) / 3
    const double h[] = {3.0, 4.0, 5.0, 4.0};
    const double l[] = {1.0, 2.0, 3.0, 2.0};
    const double c[] = {2.0, 3.0, 4.0, 3.0};
    double out[4] = {};
    const double* in[] = {c, h, l};

    StochasticOscState<8> state{};
    state.init(/*close*/0, /*high*/1, /*low*/2, /*k_out*/3, /*w*/3);
    execute_stochastic_osc(&state, in, 4, out);

    EXPECT_TRUE(is_nan(out[0]));
    EXPECT_TRUE(is_nan(out[1]));
    EXPECT_NEAR(out[2], 100.0 * (4.0 - 1.0) / (5.0 - 1.0), 1e-8); // 75
    EXPECT_NEAR(out[3], 100.0 * (3.0 - 2.0) / (5.0 - 2.0), 1e-8); // 33.33..
}

TEST(Phase5R_StochasticOsc, ZeroRangeYieldsFifty) {
    // Constant high and low → range is 0 → kernel emits 50.
    const double h[] = {5.0, 5.0, 5.0};
    const double l[] = {5.0, 5.0, 5.0};
    const double c[] = {5.0, 5.0, 5.0};
    double out[3] = {};
    const double* in[] = {c, h, l};

    StochasticOscState<4> state{};
    state.init(0, 1, 2, 3, 2);
    execute_stochastic_osc(&state, in, 3, out);
    EXPECT_TRUE(is_nan(out[0]));
    EXPECT_NEAR(out[1], 50.0, kEps);
    EXPECT_NEAR(out[2], 50.0, kEps);
}
