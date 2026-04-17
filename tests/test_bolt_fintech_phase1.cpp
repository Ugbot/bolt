// test_bolt_fintech_phase1.cpp — Phase 1 fintech column primitives.
//
// Covers: diff_column, lag_column, log_column, sum_of_squares,
// running_max_drawdown. Each kernel gets a happy-path test + a
// degenerate-case test (empty, single-element, or boundary-shift).

#include <gtest/gtest.h>

#include "bolt/kernels/fintech/column_primitives.h"

#include <cmath>
#include <cstdint>
#include <limits>

using bolt::fintech::diff_column;
using bolt::fintech::lag_column;
using bolt::fintech::log_column;
using bolt::fintech::running_max_drawdown;
using bolt::fintech::sum_of_squares;

// ---- diff_column ---------------------------------------------------------

TEST(BoltFintechPhase1, DiffColumn_Int64_Happy) {
    const int64_t in[8]  = {10, 12, 15, 11, 11, 20, 19, 30};
    int64_t       out[8] = {};
    diff_column<int64_t>(in, 8, out);
    // out[0] = 0 sentinel; rest = in[i]-in[i-1].
    EXPECT_EQ(out[0], 0);
    EXPECT_EQ(out[1], 2);
    EXPECT_EQ(out[2], 3);
    EXPECT_EQ(out[3], -4);
    EXPECT_EQ(out[4], 0);
    EXPECT_EQ(out[5], 9);
    EXPECT_EQ(out[6], -1);
    EXPECT_EQ(out[7], 11);
}

TEST(BoltFintechPhase1, DiffColumn_Double_Happy) {
    const double in[4]  = {1.0, 1.5, 1.25, 2.0};
    double       out[4] = {};
    diff_column<double>(in, 4, out);
    EXPECT_DOUBLE_EQ(out[0], 0.0);
    EXPECT_DOUBLE_EQ(out[1], 0.5);
    EXPECT_DOUBLE_EQ(out[2], -0.25);
    EXPECT_DOUBLE_EQ(out[3], 0.75);
}

TEST(BoltFintechPhase1, DiffColumn_Degenerate_EmptyAndSingle) {
    int64_t out1[1] = {123};
    diff_column<int64_t>(nullptr, 0, nullptr);           // no-op, no assert fire
    diff_column<int64_t>(out1, 0, out1);                 // n=0 path
    EXPECT_EQ(out1[0], 123);                             // untouched

    const int64_t one_in[1]  = {42};
    int64_t       one_out[1] = {};
    diff_column<int64_t>(one_in, 1, one_out);
    EXPECT_EQ(one_out[0], 0);
}

// ---- lag_column ----------------------------------------------------------

TEST(BoltFintechPhase1, LagColumn_ShiftZeroIdentity) {
    const int64_t in[5]  = {7, 8, 9, 10, 11};
    int64_t       out[5] = {};
    lag_column<int64_t>(in, 5, 0, out);
    for (int i = 0; i < 5; ++i) EXPECT_EQ(out[i], in[i]);
}

TEST(BoltFintechPhase1, LagColumn_ShiftOne) {
    const double in[5]  = {1.0, 2.0, 3.0, 4.0, 5.0};
    double       out[5] = {};
    lag_column<double>(in, 5, 1, out);
    EXPECT_DOUBLE_EQ(out[0], 0.0);
    EXPECT_DOUBLE_EQ(out[1], 1.0);
    EXPECT_DOUBLE_EQ(out[2], 2.0);
    EXPECT_DOUBLE_EQ(out[3], 3.0);
    EXPECT_DOUBLE_EQ(out[4], 4.0);
}

TEST(BoltFintechPhase1, LagColumn_ShiftNMinusOne) {
    // Maximum meaningful shift: only the last slot is a real sample.
    const int64_t in[4]  = {100, 200, 300, 400};
    int64_t       out[4] = {};
    lag_column<int64_t>(in, 4, 3, out);
    EXPECT_EQ(out[0], 0);
    EXPECT_EQ(out[1], 0);
    EXPECT_EQ(out[2], 0);
    EXPECT_EQ(out[3], 100);
}

// ---- log_column ----------------------------------------------------------

TEST(BoltFintechPhase1, LogColumn_Happy) {
    const double e = std::exp(1.0);
    const double in[3]  = {1.0, e, e * e};
    double       out[3] = {};
    log_column(in, 3, out);
    EXPECT_DOUBLE_EQ(out[0], 0.0);
    EXPECT_NEAR(out[1], 1.0, 1e-15);
    EXPECT_NEAR(out[2], 2.0, 1e-15);
}

TEST(BoltFintechPhase1, LogColumn_IEEESemanticsForBadInputs) {
    // Zero and negatives: contract says no branch/filter — IEEE semantics.
    const double in[3]  = {0.0, -1.0, 1.0};
    double       out[3] = {};
    log_column(in, 3, out);
    EXPECT_TRUE(std::isinf(out[0]) && out[0] < 0.0);   // log(0) = -inf
    EXPECT_TRUE(std::isnan(out[1]));                   // log(-1) = NaN
    EXPECT_DOUBLE_EQ(out[2], 0.0);
}

// ---- sum_of_squares ------------------------------------------------------

TEST(BoltFintechPhase1, SumOfSquares_Int64_SmallSet) {
    const int64_t in[4] = {1, 2, 3, 4};
    const int64_t s = sum_of_squares<int64_t>(in, 4);
    EXPECT_EQ(s, 30);  // 1+4+9+16
    static_assert(std::is_same_v<decltype(sum_of_squares<int64_t>(in, 4)),
                                 int64_t>,
                  "integral T must widen to int64_t");
}

TEST(BoltFintechPhase1, SumOfSquares_Int64_WideningCheck) {
    // 5 * (2^30)^2 = 5 * 2^60 — fits in int64_t (max 2^63-1) but would
    // overflow an int32 accumulator. Validates the widened acc contract.
    const int64_t x = int64_t{1} << 30;
    const int64_t in[5] = {x, x, x, x, x};
    const int64_t s = sum_of_squares<int64_t>(in, 5);
    const int64_t expected = int64_t{5} * (x * x);
    EXPECT_EQ(s, expected);
    EXPECT_GT(s, int64_t{std::numeric_limits<int32_t>::max()});
}

TEST(BoltFintechPhase1, SumOfSquares_Double_Happy) {
    const double in[3] = {1.5, -2.0, 0.5};
    const double s = sum_of_squares<double>(in, 3);
    EXPECT_DOUBLE_EQ(s, 2.25 + 4.0 + 0.25);
    static_assert(std::is_same_v<decltype(sum_of_squares<double>(in, 3)),
                                 double>,
                  "floating T must accumulate in double");
}

TEST(BoltFintechPhase1, SumOfSquares_Degenerate_Empty) {
    const double in_empty[1] = {99.0};
    EXPECT_EQ(sum_of_squares<int64_t>(nullptr, 0), 0);
    EXPECT_DOUBLE_EQ(sum_of_squares<double>(in_empty, 0), 0.0);
}

// ---- running_max_drawdown -----------------------------------------------

TEST(BoltFintechPhase1, RunningMaxDrawdown_Canonical) {
    // Spec example: {10, 20, 15, 30, 25} → {0, 0, 5, 0, 5}
    const int64_t in[5]  = {10, 20, 15, 30, 25};
    int64_t       out[5] = {};
    running_max_drawdown<int64_t>(in, 5, out);
    EXPECT_EQ(out[0], 0);
    EXPECT_EQ(out[1], 0);
    EXPECT_EQ(out[2], 5);
    EXPECT_EQ(out[3], 0);
    EXPECT_EQ(out[4], 5);
}

TEST(BoltFintechPhase1, RunningMaxDrawdown_Double_MonotonicUp) {
    // Monotonically non-decreasing input → zero drawdown everywhere.
    const double in[4]  = {1.0, 1.0, 2.0, 2.0};
    double       out[4] = {};
    running_max_drawdown<double>(in, 4, out);
    for (int i = 0; i < 4; ++i) EXPECT_DOUBLE_EQ(out[i], 0.0);
}

TEST(BoltFintechPhase1, RunningMaxDrawdown_Degenerate_EmptyAndSingle) {
    int64_t out0[1] = {7};
    running_max_drawdown<int64_t>(nullptr, 0, nullptr);  // no-op
    running_max_drawdown<int64_t>(out0, 0, out0);
    EXPECT_EQ(out0[0], 7);                               // untouched

    const int64_t one_in[1]  = {42};
    int64_t       one_out[1] = {};
    running_max_drawdown<int64_t>(one_in, 1, one_out);
    EXPECT_EQ(one_out[0], 0);
}
