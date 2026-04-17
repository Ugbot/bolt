// test_bolt_fintech_phase5s_core.cpp — Phase 5.S "specialised" subset.
//
// Covers the simple POD-state specialised kernels from Phase 5.S:
//   P5.S.1  MaxDrawdown    — streaming peak, emit absolute drawdown.
//   P5.S.2  Drawdown       — streaming peak, emit fractional drawdown.
//   P5.S.3  CUSUM          — two-sided Page CUSUM with threshold reset.
//   P5.S.13 AlmgrenChriss  — precomputed optimal-execution schedule.
//   P5.S.15 ThrottleCheck  — sliding-window event count tripwire.
//
// Each test round-trips against a hand-computed scalar reference. No
// Arrow dependency; no other external libs.

#include <gtest/gtest.h>

#include "bolt/bolt_arena.h"
#include "bolt/kernels/fintech/almgren_chriss.h"
#include "bolt/kernels/fintech/cusum.h"
#include "bolt/kernels/fintech/drawdown.h"
#include "bolt/kernels/fintech/max_drawdown.h"
#include "bolt/kernels/fintech/throttle_check.h"

#include <cmath>
#include <cstdint>

using bolt::Arena;
using namespace bolt::fintech;

namespace {
constexpr double kEps = 1e-12;
}

// ===========================================================================
// P5.S.1 MaxDrawdown — running peak, absolute drop (peak - x)
// ===========================================================================
TEST(Phase5S_MaxDrawdown, SingleBatchReferenceVector) {
    // Reference vector from the plan: {100, 120, 110, 130, 90}
    //   peak:       100, 120, 120, 130, 130
    //   drawdown:   0,   0,   10,  0,   40
    Arena arena;
    MaxDrawdownState* s = make_max_drawdown_state(&arena, /*in=*/0, /*out=*/1);
    ASSERT_NE(s, nullptr);

    const double x[]    = {100.0, 120.0, 110.0, 130.0, 90.0};
    const double* in[]  = {x};
    double out[5]       = {};
    execute_max_drawdown(s, in, 5, out);

    EXPECT_NEAR(out[0],  0.0, kEps);
    EXPECT_NEAR(out[1],  0.0, kEps);
    EXPECT_NEAR(out[2], 10.0, kEps);
    EXPECT_NEAR(out[3],  0.0, kEps);
    EXPECT_NEAR(out[4], 40.0, kEps);
}

TEST(Phase5S_MaxDrawdown, StatePersistsAcrossBatches) {
    // Feeding {100, 120} then {110, 130, 90} should produce the same
    // output as feeding the full {100, 120, 110, 130, 90} in one call.
    Arena arena;
    MaxDrawdownState* s = make_max_drawdown_state(&arena, 0, 1);
    ASSERT_NE(s, nullptr);

    const double x1[] = {100.0, 120.0};
    const double* in1[] = {x1};
    double out1[2] = {};
    execute_max_drawdown(s, in1, 2, out1);
    EXPECT_NEAR(out1[0], 0.0, kEps);
    EXPECT_NEAR(out1[1], 0.0, kEps);

    const double x2[] = {110.0, 130.0, 90.0};
    const double* in2[] = {x2};
    double out2[3] = {};
    execute_max_drawdown(s, in2, 3, out2);
    EXPECT_NEAR(out2[0], 10.0, kEps);
    EXPECT_NEAR(out2[1],  0.0, kEps);
    EXPECT_NEAR(out2[2], 40.0, kEps);
}

TEST(Phase5S_MaxDrawdown, MonotonicRiseYieldsZero) {
    Arena arena;
    MaxDrawdownState* s = make_max_drawdown_state(&arena, 0, 1);
    ASSERT_NE(s, nullptr);
    const double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    const double* in[] = {x};
    double out[5] = {};
    execute_max_drawdown(s, in, 5, out);
    for (int i = 0; i < 5; ++i) EXPECT_NEAR(out[i], 0.0, kEps);
}

// ===========================================================================
// P5.S.2 Drawdown — running peak, fractional drop (peak - x) / peak
// ===========================================================================
TEST(Phase5S_Drawdown, ReferenceVectorFractional) {
    // Same sequence as MaxDrawdown but fractional:
    //   {100, 120, 110, 130, 90}
    //   peak:     100,   120,            120,   130,              130
    //   fraction: 0,     0,              10/120,0,                40/130
    Arena arena;
    DrawdownState* s = make_drawdown_state(&arena, 0, 1);
    ASSERT_NE(s, nullptr);

    const double x[]   = {100.0, 120.0, 110.0, 130.0, 90.0};
    const double* in[] = {x};
    double out[5]      = {};
    execute_drawdown(s, in, 5, out);

    EXPECT_NEAR(out[0], 0.0,            kEps);
    EXPECT_NEAR(out[1], 0.0,            kEps);
    EXPECT_NEAR(out[2], 10.0 / 120.0,   kEps);
    EXPECT_NEAR(out[3], 0.0,            kEps);
    EXPECT_NEAR(out[4], 40.0 / 130.0,   kEps);
}

TEST(Phase5S_Drawdown, FirstRowSeedsZeroNotNaN) {
    // Guard: first-row x[0] seeds peak=x[0]; (peak - x[0]) / peak = 0,
    // never NaN. Even for x[0] = 0 or negative, we must emit 0.0.
    Arena arena;
    DrawdownState* s = make_drawdown_state(&arena, 0, 1);
    ASSERT_NE(s, nullptr);
    const double x[] = {0.0};
    const double* in[] = {x};
    double out[1] = {};
    execute_drawdown(s, in, 1, out);
    EXPECT_FALSE(std::isnan(out[0]));
    EXPECT_NEAR(out[0], 0.0, kEps);
}

TEST(Phase5S_Drawdown, CrossBatchPeakSurvives) {
    Arena arena;
    DrawdownState* s = make_drawdown_state(&arena, 0, 1);
    ASSERT_NE(s, nullptr);

    const double x1[] = {100.0, 120.0};
    const double* in1[] = {x1};
    double out1[2] = {};
    execute_drawdown(s, in1, 2, out1);

    const double x2[] = {110.0, 130.0, 90.0};
    const double* in2[] = {x2};
    double out2[3] = {};
    execute_drawdown(s, in2, 3, out2);

    EXPECT_NEAR(out2[0], 10.0 / 120.0, kEps);
    EXPECT_NEAR(out2[1], 0.0,          kEps);
    EXPECT_NEAR(out2[2], 40.0 / 130.0, kEps);
}

// ===========================================================================
// P5.S.3 CUSUM — two-sided, reset on threshold cross
// ===========================================================================
TEST(Phase5S_CUSUM, ReferenceVectorTwoSided) {
    // Plan-specified reference: {1, 2, 3, -5, 1, 1, 5}, target=0, threshold=3.
    //   i=0 d=1:  cp=max(0,0+1)=1,  cn=min(0,0+1)=0.    trip? no.
    //   i=1 d=2:  cp=max(0,1+2)=3,  cn=min(0,0+2)=0.    cp==3 not > 3. no.
    //   i=2 d=3:  cp=max(0,3+3)=6,  cn=min(0,0+3)=0.    cp=6>3 -> trip -> reset.
    //   i=3 d=-5: cp=max(0,0-5)=0,  cn=min(0,0-5)=-5.   cn=-5<-3 -> trip -> reset.
    //   i=4 d=1:  cp=max(0,0+1)=1,  cn=0.               no trip.
    //   i=5 d=1:  cp=max(0,1+1)=2,  cn=0.               no trip.
    //   i=6 d=5:  cp=max(0,2+5)=7,  cn=0.               cp=7>3 -> trip -> reset.
    // Emitted pre-reset values: cp={1,3,6,0,1,2,7}, cn={0,0,0,-5,0,0,0}.
    Arena arena;
    CUSUMState* s = make_cusum_state(&arena,
                                     /*in=*/0, /*cp=*/1, /*cn=*/2,
                                     /*target=*/0.0, /*threshold=*/3.0);
    ASSERT_NE(s, nullptr);

    const double x[]   = {1.0, 2.0, 3.0, -5.0, 1.0, 1.0, 5.0};
    const double* in[] = {x};
    double cp_out[7]   = {};
    double cn_out[7]   = {};
    execute_cusum(s, in, 7, cp_out, cn_out);

    const double cp_expect[7] = {1.0, 3.0, 6.0, 0.0, 1.0, 2.0, 7.0};
    const double cn_expect[7] = {0.0, 0.0, 0.0, -5.0, 0.0, 0.0, 0.0};
    for (int i = 0; i < 7; ++i) {
        EXPECT_NEAR(cp_out[i], cp_expect[i], kEps) << "i=" << i;
        EXPECT_NEAR(cn_out[i], cn_expect[i], kEps) << "i=" << i;
    }
}

TEST(Phase5S_CUSUM, ResetBetweenBatches) {
    // Reset mid-stream: state.cum_pos / cum_neg should zero when trip fires.
    Arena arena;
    CUSUMState* s = make_cusum_state(&arena, 0, 1, 2, 0.0, 3.0);
    ASSERT_NE(s, nullptr);

    const double x1[] = {1.0, 2.0};
    const double* in1[] = {x1};
    double cp1[2] = {};
    double cn1[2] = {};
    execute_cusum(s, in1, 2, cp1, cn1);
    EXPECT_NEAR(cp1[0], 1.0, kEps);
    EXPECT_NEAR(cp1[1], 3.0, kEps);
    EXPECT_EQ(s->cum_pos, 3.0);   // still exactly 3, not tripped

    const double x2[] = {3.0};
    const double* in2[] = {x2};
    double cp2[1] = {};
    double cn2[1] = {};
    execute_cusum(s, in2, 1, cp2, cn2);
    EXPECT_NEAR(cp2[0], 6.0, kEps);
    // 6 > 3 -> state was reset after emit.
    EXPECT_EQ(s->cum_pos, 0.0);
    EXPECT_EQ(s->cum_neg, 0.0);
}

TEST(Phase5S_CUSUM, CustomTargetOffset) {
    // target = 10, threshold = 5, x = {10,10,10}. d=0 every row -> no trip.
    Arena arena;
    CUSUMState* s = make_cusum_state(&arena, 0, 1, 2, 10.0, 5.0);
    ASSERT_NE(s, nullptr);
    const double x[] = {10.0, 10.0, 10.0};
    const double* in[] = {x};
    double cp[3] = {}, cn[3] = {};
    execute_cusum(s, in, 3, cp, cn);
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(cp[i], 0.0, kEps);
        EXPECT_NEAR(cn[i], 0.0, kEps);
    }
}

// ===========================================================================
// P5.S.13 AlmgrenChriss — precomputed optimal-execution schedule
// ===========================================================================
TEST(Phase5S_AlmgrenChriss, UniformFallbackWhenSigmaIsZero) {
    // denom = 2*eta*sigma^2 = 0  -> uniform schedule = 1 / n_slices.
    Arena arena;
    AlmgrenChrissState* s = make_almgren_chriss_state(
        &arena, /*out=*/0, /*n_slices=*/4,
        /*sigma=*/0.0, /*gamma=*/1.0, /*eta=*/1.0);
    ASSERT_NE(s, nullptr);

    double out[8] = {};
    execute_almgren_chriss(s, nullptr, 8, out);
    const double expect = 0.25;
    for (int i = 0; i < 8; ++i) EXPECT_NEAR(out[i], expect, kEps);
}

TEST(Phase5S_AlmgrenChriss, AnalyticSchedule_Kappa1_Nslices4) {
    // sigma=1, gamma=2, eta=1 -> kappa = sqrt(2 / (2*1*1)) = 1.
    // sinh(kT)=sinh(1). Slice[k] = sinh(1-k/4)/sinh(1) - sinh(1-(k+1)/4)/sinh(1).
    Arena arena;
    AlmgrenChrissState* s = make_almgren_chriss_state(
        &arena, 0, 4, /*sigma=*/1.0, /*gamma=*/2.0, /*eta=*/1.0);
    ASSERT_NE(s, nullptr);

    const double sh1 = std::sinh(1.0);
    const double exp0 = (std::sinh(1.00) - std::sinh(0.75)) / sh1;
    const double exp1 = (std::sinh(0.75) - std::sinh(0.50)) / sh1;
    const double exp2 = (std::sinh(0.50) - std::sinh(0.25)) / sh1;
    const double exp3 = (std::sinh(0.25) - std::sinh(0.00)) / sh1;

    // Emit 8 rows -> schedule wraps twice.
    double out[8] = {};
    execute_almgren_chriss(s, nullptr, 8, out);
    EXPECT_NEAR(out[0], exp0, kEps);
    EXPECT_NEAR(out[1], exp1, kEps);
    EXPECT_NEAR(out[2], exp2, kEps);
    EXPECT_NEAR(out[3], exp3, kEps);
    // Second cycle — cyclic wrap.
    EXPECT_NEAR(out[4], exp0, kEps);
    EXPECT_NEAR(out[5], exp1, kEps);
    EXPECT_NEAR(out[6], exp2, kEps);
    EXPECT_NEAR(out[7], exp3, kEps);

    // Schedule sums to Q = 1.
    const double total = exp0 + exp1 + exp2 + exp3;
    EXPECT_NEAR(total, 1.0, 1e-10);
}

TEST(Phase5S_AlmgrenChriss, RowCountPersistsAcrossBatches) {
    // First batch emits rows 0,1 -> schedule[0], schedule[1].
    // Second batch emits rows 2,3 -> schedule[2], schedule[3].
    Arena arena;
    AlmgrenChrissState* s = make_almgren_chriss_state(
        &arena, 0, 4, 1.0, 2.0, 1.0);
    ASSERT_NE(s, nullptr);

    double out1[2] = {};
    execute_almgren_chriss(s, nullptr, 2, out1);
    double out2[2] = {};
    execute_almgren_chriss(s, nullptr, 2, out2);

    EXPECT_NEAR(out1[0], s->schedule[0], kEps);
    EXPECT_NEAR(out1[1], s->schedule[1], kEps);
    EXPECT_NEAR(out2[0], s->schedule[2], kEps);
    EXPECT_NEAR(out2[1], s->schedule[3], kEps);
}

// ===========================================================================
// P5.S.15 ThrottleCheck — sliding window event count tripwire
// ===========================================================================
TEST(Phase5S_ThrottleCheck, BasicWindow_AllWithinLimit) {
    // window = 1000, max_count = 3. Timestamps: 0, 100, 200, 300 — all
    // within the same window at t=300 -> count=4 > 3 -> flag 0 at t=300.
    Arena arena;
    auto* s = make_throttle_check_state<16>(
        &arena, /*ts=*/0, /*out=*/1, /*window=*/1000, /*max_count=*/3);
    ASSERT_NE(s, nullptr);

    const int64_t ts[] = {0, 100, 200, 300};
    int64_t out[4] = {};
    execute_throttle_check(s, ts, 4, out);

    // count after each insert:
    //   i=0: ring={0},           count=1 <= 3 -> 1
    //   i=1: ring={0,100},       count=2 <= 3 -> 1
    //   i=2: ring={0,100,200},   count=3 <= 3 -> 1
    //   i=3: ring={0,100,200,300}, count=4 > 3 -> 0
    EXPECT_EQ(out[0], 1);
    EXPECT_EQ(out[1], 1);
    EXPECT_EQ(out[2], 1);
    EXPECT_EQ(out[3], 0);
}

TEST(Phase5S_ThrottleCheck, EvictionRestoresCapacity) {
    // window = 100. Timestamps 0, 50, 150, 250 with max_count = 2.
    // At t=0:   ring={0},           count=1 -> 1
    // At t=50:  t_start=-50, ring={0,50}, count=2 <= 2 -> 1
    // At t=150: t_start=50, evict 0;   ring={50,150}, count=2 <= 2 -> 1
    // At t=250: t_start=150,evict 50;  ring={150,250}, count=2 <= 2 -> 1
    Arena arena;
    auto* s = make_throttle_check_state<16>(&arena, 0, 1, 100, 2);
    ASSERT_NE(s, nullptr);

    const int64_t ts[] = {0, 50, 150, 250};
    int64_t out[4] = {};
    execute_throttle_check(s, ts, 4, out);
    EXPECT_EQ(out[0], 1);
    EXPECT_EQ(out[1], 1);
    EXPECT_EQ(out[2], 1);
    EXPECT_EQ(out[3], 1);
}

TEST(Phase5S_ThrottleCheck, BurstTripsAndClearsAfterGap) {
    // window = 100, max_count = 2.
    // Burst at t=0,10,20 -> count=1,2,3. out = 1,1,0 (third trips).
    // Then gap to t=500 -> all old evicted -> count=1 -> 1.
    Arena arena;
    auto* s = make_throttle_check_state<16>(&arena, 0, 1, 100, 2);
    ASSERT_NE(s, nullptr);

    const int64_t ts[] = {0, 10, 20, 500};
    int64_t out[4] = {};
    execute_throttle_check(s, ts, 4, out);
    EXPECT_EQ(out[0], 1);
    EXPECT_EQ(out[1], 1);
    EXPECT_EQ(out[2], 0);
    EXPECT_EQ(out[3], 1);
}

TEST(Phase5S_ThrottleCheck, StatePersistsAcrossBatches) {
    Arena arena;
    auto* s = make_throttle_check_state<16>(&arena, 0, 1, 100, 2);
    ASSERT_NE(s, nullptr);

    const int64_t ts1[] = {0, 10};
    int64_t out1[2] = {};
    execute_throttle_check(s, ts1, 2, out1);
    EXPECT_EQ(out1[0], 1);
    EXPECT_EQ(out1[1], 1);

    // Third event inside the same window trips.
    const int64_t ts2[] = {20};
    int64_t out2[1] = {};
    execute_throttle_check(s, ts2, 1, out2);
    EXPECT_EQ(out2[0], 0);
}
