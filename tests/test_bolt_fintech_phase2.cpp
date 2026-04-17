// test_bolt_fintech_phase2.cpp — Phase 2 fintech state substrate tests.
//
// Covers the three POD state types in bolt/kernels/fintech/state.h:
//   RollingRing<T, kCap>
//   WelfordAccumulator
//   EmaState
//
// Each test is a round-trip against a hand-computed scalar reference. No
// Arrow / external library comparisons — Bolt is zero-dependency.

#include <gtest/gtest.h>

#include "bolt/kernels/fintech/state.h"

#include <cmath>
#include <cstdint>

using bolt::fintech::EmaState;
using bolt::fintech::RollingRing;
using bolt::fintech::WelfordAccumulator;

// ===========================================================================
// RollingRing
// ===========================================================================

TEST(BoltFintechRollingRing, FillToCapacityMeanIsHandComputed) {
    RollingRing<double, 8> r{};
    r.init(4);
    EXPECT_FALSE(r.full());
    EXPECT_EQ(r.size(), 0u);

    r.push(1.0);
    r.push(2.0);
    r.push(3.0);
    r.push(4.0);

    EXPECT_TRUE(r.full());
    EXPECT_EQ(r.size(), 4u);
    EXPECT_DOUBLE_EQ(r.mean(), 2.5);   // (1+2+3+4)/4

    // back(0) is the most-recent sample; back(3) is the oldest.
    EXPECT_DOUBLE_EQ(r.back(0), 4.0);
    EXPECT_DOUBLE_EQ(r.back(1), 3.0);
    EXPECT_DOUBLE_EQ(r.back(2), 2.0);
    EXPECT_DOUBLE_EQ(r.back(3), 1.0);
}

TEST(BoltFintechRollingRing, EvictionReplacesOldestAndUpdatesSum) {
    // Cap 4, window 4, push 5 values — first value must be evicted.
    RollingRing<double, 4> r{};
    r.init(4);
    r.push(1.0);
    r.push(2.0);
    r.push(3.0);
    r.push(4.0);
    // Evict 1.0, append 5.0. Ring now holds {5, 2, 3, 4} in ring[]
    // logical order (oldest → newest): 2, 3, 4, 5.
    r.push(5.0);

    EXPECT_TRUE(r.full());
    EXPECT_EQ(r.size(), 4u);
    EXPECT_DOUBLE_EQ(r.mean(), 3.5);   // (2+3+4+5)/4

    EXPECT_DOUBLE_EQ(r.back(0), 5.0);
    EXPECT_DOUBLE_EQ(r.back(1), 4.0);
    EXPECT_DOUBLE_EQ(r.back(2), 3.0);
    EXPECT_DOUBLE_EQ(r.back(3), 2.0);
}

TEST(BoltFintechRollingRing, EvictionIntegerTypeWidenedSum) {
    RollingRing<int32_t, 3> r{};
    r.init(3);
    r.push(10);
    r.push(20);
    r.push(30);          // ring full: sum=60
    r.push(40);          // evict 10: sum=90
    r.push(50);          // evict 20: sum=120
    EXPECT_TRUE(r.full());
    EXPECT_DOUBLE_EQ(r.mean(), 40.0);
    EXPECT_EQ(r.back(0), 50);
    EXPECT_EQ(r.back(1), 40);
    EXPECT_EQ(r.back(2), 30);
}

TEST(BoltFintechRollingRing, CapacityOneEdge) {
    RollingRing<double, 1> r{};
    r.init(1);
    EXPECT_FALSE(r.full());
    r.push(7.5);
    EXPECT_TRUE(r.full());
    EXPECT_EQ(r.size(), 1u);
    EXPECT_DOUBLE_EQ(r.mean(), 7.5);
    EXPECT_DOUBLE_EQ(r.back(0), 7.5);

    // Eviction path with cap=1: every push replaces the single slot.
    r.push(-3.25);
    EXPECT_TRUE(r.full());
    EXPECT_EQ(r.size(), 1u);
    EXPECT_DOUBLE_EQ(r.mean(), -3.25);
    EXPECT_DOUBLE_EQ(r.back(0), -3.25);
}

TEST(BoltFintechRollingRing, PartialFillMeanAndSize) {
    RollingRing<double, 8> r{};
    r.init(4);
    r.push(10.0);
    r.push(20.0);
    EXPECT_FALSE(r.full());
    EXPECT_EQ(r.size(), 2u);
    EXPECT_DOUBLE_EQ(r.mean(), 15.0);
    EXPECT_DOUBLE_EQ(r.back(0), 20.0);
    EXPECT_DOUBLE_EQ(r.back(1), 10.0);
}

// ===========================================================================
// WelfordAccumulator
// ===========================================================================

TEST(BoltFintechWelford, ClassicEightSampleSequence) {
    // Sequence {2,4,4,4,5,5,7,9}: mean=5, sum_sq_dev=32,
    // variance_pop = 32/8 = 4, variance_sample = 32/7.
    WelfordAccumulator w{};
    w.init();
    const double xs[] = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    for (double x : xs) w.update(x);

    EXPECT_EQ(w.count, 8ull);
    EXPECT_NEAR(w.mean, 5.0, 1e-9);
    EXPECT_NEAR(w.variance_pop(),     4.0,          1e-9);
    EXPECT_NEAR(w.variance_sample(),  32.0 / 7.0,   1e-9);
    EXPECT_NEAR(w.stddev_pop(),       2.0,          1e-9);
    EXPECT_NEAR(w.stddev_sample(),    std::sqrt(32.0 / 7.0), 1e-9);
}

TEST(BoltFintechWelford, EmptyAfterInit) {
    // Design note: variance on an empty accumulator asserts (count > 0 / > 1).
    // We do NOT run a death test here — Tiger Style bans exceptions and the
    // assertion macro is disabled in Release builds. We only verify the
    // zero-init path: count=0, mean=0, m2=0.
    WelfordAccumulator w{};
    w.init();
    EXPECT_EQ(w.count, 0ull);
    EXPECT_DOUBLE_EQ(w.mean, 0.0);
    EXPECT_DOUBLE_EQ(w.m2,   0.0);
}

TEST(BoltFintechWelford, ConstantSequenceHasZeroVariance) {
    WelfordAccumulator w{};
    w.init();
    for (int i = 0; i < 16; ++i) w.update(3.14159);
    EXPECT_EQ(w.count, 16ull);
    EXPECT_NEAR(w.mean, 3.14159, 1e-12);
    EXPECT_NEAR(w.variance_pop(),    0.0, 1e-12);
    EXPECT_NEAR(w.variance_sample(), 0.0, 1e-12);
}

TEST(BoltFintechWelford, SingleSampleMeanIsValue) {
    WelfordAccumulator w{};
    w.init();
    w.update(42.0);
    EXPECT_EQ(w.count, 1ull);
    EXPECT_DOUBLE_EQ(w.mean, 42.0);
    EXPECT_DOUBLE_EQ(w.m2,   0.0);
    EXPECT_NEAR(w.variance_pop(), 0.0, 1e-12);
}

// ===========================================================================
// EmaState
// ===========================================================================

TEST(BoltFintechEma, Period4KnownSequence) {
    // period = 4 → alpha = 2 / (4+1) = 0.4.
    // Feed {10, 20, 30, 40}:
    //   update(10): seed → 10
    //   update(20): 0.4*20 + 0.6*10  = 14.0
    //   update(30): 0.4*30 + 0.6*14  = 20.4
    //   update(40): 0.4*40 + 0.6*20.4 = 28.24
    EmaState e{};
    e.init(4);
    EXPECT_DOUBLE_EQ(e.alpha, 0.4);
    EXPECT_FALSE(e.initialized);

    EXPECT_NEAR(e.update(10.0), 10.0,   1e-12);
    EXPECT_TRUE(e.initialized);
    EXPECT_NEAR(e.update(20.0), 14.0,   1e-12);
    EXPECT_NEAR(e.update(30.0), 20.4,   1e-12);
    EXPECT_NEAR(e.update(40.0), 28.24,  1e-12);
}

TEST(BoltFintechEma, ResetReArmsInitialized) {
    EmaState e{};
    e.init(4);
    e.update(100.0);
    e.update(200.0);
    EXPECT_TRUE(e.initialized);

    e.reset();
    EXPECT_FALSE(e.initialized);
    EXPECT_DOUBLE_EQ(e.value, 0.0);
    EXPECT_DOUBLE_EQ(e.alpha, 0.4);    // alpha preserved across reset

    // After reset, the first update seeds again.
    EXPECT_NEAR(e.update(7.0), 7.0, 1e-12);
    EXPECT_NEAR(e.update(17.0), 0.4 * 17.0 + 0.6 * 7.0, 1e-12);
}

TEST(BoltFintechEma, Period1DegeneratesToIdentity) {
    // alpha = 2/(1+1) = 1 → every update fully replaces state.
    EmaState e{};
    e.init(1);
    EXPECT_DOUBLE_EQ(e.alpha, 1.0);

    EXPECT_NEAR(e.update(5.0),   5.0,   1e-12);
    EXPECT_NEAR(e.update(99.0),  99.0,  1e-12);
    EXPECT_NEAR(e.update(-3.5), -3.5,   1e-12);
}
