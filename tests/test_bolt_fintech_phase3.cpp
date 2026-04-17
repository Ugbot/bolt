// test_bolt_fintech_phase3.cpp — Phase 3 worked EMA example.
//
// Validates the end-to-end Tier-2 shape from the gestalt-kernel-adapter
// template: POD operator state pinned in an arena at graph-compile
// time, persistent across execute_ema() calls, zero hot-path allocation.
//
// Three angles:
//   1. EMA worked: period=4 on {10,20,30,40,50} — hand-computed against
//      Phase 2's substrate (alpha = 2/(p+1) = 0.4, first sample seeds).
//   2. State persistence: feeding {10} then {20} across two separate
//      execute_ema() calls must match feeding {10,20} in one call.
//   3. reset() re-arms the state for graph re-entry.

#include <gtest/gtest.h>

#include "bolt/bolt_arena.h"
#include "bolt/kernels/fintech/ema.h"

#include <cstdint>

using bolt::Arena;
using bolt::fintech::EMAOpDesc;
using bolt::fintech::EMAState;
using bolt::fintech::execute_ema;
using bolt::fintech::make_ema_state;

namespace {
constexpr double kEps = 1e-12;
}

// ---------------------------------------------------------------------------
// P3.1 EMA worked example — period=4, alpha=0.4.
//   update(10): seed → 10
//   update(20): 0.4*20 + 0.6*10   = 14.0
//   update(30): 0.4*30 + 0.6*14   = 20.4
//   update(40): 0.4*40 + 0.6*20.4 = 28.24
//   update(50): 0.4*50 + 0.6*28.24 = 36.944
// ---------------------------------------------------------------------------
TEST(Phase3_EMA, Period4KnownSequenceSingleCall) {
    Arena arena;
    EMAState* s = make_ema_state(&arena, /*in_col=*/0, /*out_col=*/1,
                                 /*period=*/4);
    ASSERT_NE(s, nullptr);
    EXPECT_DOUBLE_EQ(s->ema.alpha, 0.4);

    const double x[] = {10.0, 20.0, 30.0, 40.0, 50.0};
    const double* in[] = {x};
    double out[5] = {};
    execute_ema(s, in, 5, out);

    EXPECT_NEAR(out[0], 10.0,    kEps);
    EXPECT_NEAR(out[1], 14.0,    kEps);
    EXPECT_NEAR(out[2], 20.4,    kEps);
    EXPECT_NEAR(out[3], 28.24,   kEps);
    EXPECT_NEAR(out[4], 36.944,  kEps);
}

// ---------------------------------------------------------------------------
// State persistence across execute_ema() calls.
// Feed {10} then {20}; second call's first output must equal
//   0.4 * 20 + 0.6 * 10 = 14.0
// (identical to feeding {10, 20} in one call). This is the whole point of
// the port — chukonu's lambda captures break this invariant; Bolt must
// not.
// ---------------------------------------------------------------------------
TEST(Phase3_EMA, StatePersistsAcrossCalls) {
    Arena arena;
    EMAState* s = make_ema_state(&arena, 0, 1, 4);
    ASSERT_NE(s, nullptr);

    const double x1[] = {10.0};
    const double* in1[] = {x1};
    double out1[1] = {};
    execute_ema(s, in1, 1, out1);
    EXPECT_NEAR(out1[0], 10.0, kEps);
    EXPECT_TRUE(s->ema.initialized);
    EXPECT_NEAR(s->ema.value, 10.0, kEps);

    const double x2[] = {20.0};
    const double* in2[] = {x2};
    double out2[1] = {};
    execute_ema(s, in2, 1, out2);
    EXPECT_NEAR(out2[0], 14.0, kEps);        // 0.4*20 + 0.6*10
    EXPECT_NEAR(s->ema.value, 14.0, kEps);
}

// ---------------------------------------------------------------------------
// reset() zeros the state for graph re-entry.
// ---------------------------------------------------------------------------
TEST(Phase3_EMA, ResetReArmsForGraphReEntry) {
    Arena arena;
    EMAState* s = make_ema_state(&arena, 0, 1, 4);
    ASSERT_NE(s, nullptr);

    const double x1[] = {100.0, 200.0};
    const double* in1[] = {x1};
    double out1[2] = {};
    execute_ema(s, in1, 2, out1);
    EXPECT_TRUE(s->ema.initialized);

    s->ema.reset();
    EXPECT_FALSE(s->ema.initialized);
    EXPECT_DOUBLE_EQ(s->ema.value, 0.0);
    EXPECT_DOUBLE_EQ(s->ema.alpha, 0.4);     // alpha preserved.

    // After reset, first update seeds again.
    const double x2[] = {7.0, 17.0};
    const double* in2[] = {x2};
    double out2[2] = {};
    execute_ema(s, in2, 2, out2);
    EXPECT_NEAR(out2[0],  7.0, kEps);                        // seed
    EXPECT_NEAR(out2[1],  0.4 * 17.0 + 0.6 * 7.0, kEps);     // 11.0
}

// ---------------------------------------------------------------------------
// Arena pinning — the state pointer is arena-owned, bounded by arena
// lifetime, and can be reused across many calls without realloc.
// ---------------------------------------------------------------------------
TEST(Phase3_EMA, ArenaPinnedNoAllocOnHotPath) {
    Arena arena;
    EMAState* s = make_ema_state(&arena, 0, 1, 4);
    ASSERT_NE(s, nullptr);
    const size_t alloc_after_make = arena.total_allocated();

    // Run many execute_ema() calls; arena allocation must not grow.
    const double x[] = {1.0, 2.0, 3.0, 4.0};
    const double* in[] = {x};
    double out[4] = {};
    for (int i = 0; i < 32; ++i) {
        execute_ema(s, in, 4, out);
    }
    EXPECT_EQ(arena.total_allocated(), alloc_after_make);
}
