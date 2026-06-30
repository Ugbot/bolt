// Tests for bolt::kernels::window_agg::window_agg_f64 — per-row windowed
// aggregate over a single sorted partition slice. Covers cumulative
// (UNBOUNDED PRECEDING → CURRENT), trailing-n, MIN/MAX, COUNT, and
// sample/population variance + stddev.

#include "bolt/kernels/window_agg.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>

namespace {

using namespace bolt::kernels::window_agg;

constexpr Frame kCumulative{0, 0, /*start_unbounded=*/true, /*end_unbounded=*/false, {}};
Frame trailing(std::int64_t n_preceding) {
    return Frame{n_preceding, 0, false, false, {}};
}

TEST(WindowAgg, CumulativeSum) {
    const double v[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double out[5] = {};
    window_agg_f64(WinAgg::Sum, v, 5, kCumulative, out);
    EXPECT_DOUBLE_EQ(out[0], 1.0);
    EXPECT_DOUBLE_EQ(out[1], 3.0);
    EXPECT_DOUBLE_EQ(out[2], 6.0);
    EXPECT_DOUBLE_EQ(out[3], 10.0);
    EXPECT_DOUBLE_EQ(out[4], 15.0);
}

TEST(WindowAgg, TrailingAvg) {
    // 3-row trailing window (2 PRECEDING + CURRENT): mean of last 3.
    const double v[] = {10.0, 20.0, 30.0, 40.0};
    double out[4] = {};
    window_agg_f64(WinAgg::Mean, v, 4, trailing(2), out);
    EXPECT_DOUBLE_EQ(out[0], 10.0);              // [10]
    EXPECT_DOUBLE_EQ(out[1], 15.0);              // [10,20]
    EXPECT_DOUBLE_EQ(out[2], 20.0);              // [10,20,30]
    EXPECT_DOUBLE_EQ(out[3], 30.0);              // [20,30,40]
}

TEST(WindowAgg, TrailingMinMaxCount) {
    const double v[] = {5.0, 2.0, 8.0, 1.0, 9.0};
    double mn[5] = {}, mx[5] = {}, cn[5] = {};
    window_agg_f64(WinAgg::Min,   v, 5, trailing(2), mn);
    window_agg_f64(WinAgg::Max,   v, 5, trailing(2), mx);
    window_agg_f64(WinAgg::Count, v, 5, trailing(2), cn);
    EXPECT_DOUBLE_EQ(mn[2], 2.0);   // min(5,2,8)
    EXPECT_DOUBLE_EQ(mx[2], 8.0);   // max(5,2,8)
    EXPECT_DOUBLE_EQ(mn[4], 1.0);   // min(8,1,9)
    EXPECT_DOUBLE_EQ(mx[4], 9.0);   // max(8,1,9)
    EXPECT_DOUBLE_EQ(cn[0], 1.0);
    EXPECT_DOUBLE_EQ(cn[4], 3.0);
}

TEST(WindowAgg, SampleVarianceAndStddev) {
    // Whole-partition variance (UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING).
    const double v[] = {2.0, 4.0, 6.0, 8.0};   // mean 5, ss = 9+1+1+9 = 20
    Frame full{0, 0, true, true, {}};
    double vs[4] = {}, vp[4] = {}, ss[4] = {};
    window_agg_f64(WinAgg::VarSamp, v, 4, full, vs);
    window_agg_f64(WinAgg::VarPop,  v, 4, full, vp);
    window_agg_f64(WinAgg::StdSamp, v, 4, full, ss);
    EXPECT_DOUBLE_EQ(vs[0], 20.0 / 3.0);   // sample: /(n-1)
    EXPECT_DOUBLE_EQ(vp[0], 20.0 / 4.0);   // population: /n
    EXPECT_DOUBLE_EQ(ss[0], std::sqrt(20.0 / 3.0));
}

TEST(WindowAgg, SingleRowSampleVarianceIsZero) {
    const double v[] = {42.0};
    double out[1] = {};
    window_agg_f64(WinAgg::VarSamp, v, 1, kCumulative, out);
    EXPECT_DOUBLE_EQ(out[0], 0.0);   // n-1 == 0 → defined as 0 here
}

// n PRECEDING wider than the partition clamps to row 0 (no out-of-bounds).
TEST(WindowAgg, PrecedingWiderThanPartition) {
    const double v[] = {1.0, 2.0, 3.0};
    double out[3] = {};
    window_agg_f64(WinAgg::Sum, v, 3, trailing(100), out);   // 100 PRECEDING
    EXPECT_DOUBLE_EQ(out[0], 1.0);
    EXPECT_DOUBLE_EQ(out[1], 3.0);
    EXPECT_DOUBLE_EQ(out[2], 6.0);   // whole partition (clamped)
}

// CURRENT ROW AND UNBOUNDED FOLLOWING — reverse cumulative (suffix sum).
TEST(WindowAgg, CurrentToUnboundedFollowing) {
    const double v[] = {1.0, 2.0, 3.0, 4.0};
    double out[4] = {};
    Frame fr{0, 0, /*start_unbounded=*/false, /*end_unbounded=*/true, {}};  // start=CURRENT
    window_agg_f64(WinAgg::Sum, v, 4, fr, out);
    EXPECT_DOUBLE_EQ(out[0], 10.0);  // 1+2+3+4
    EXPECT_DOUBLE_EQ(out[1], 9.0);   // 2+3+4
    EXPECT_DOUBLE_EQ(out[2], 7.0);   // 3+4
    EXPECT_DOUBLE_EQ(out[3], 4.0);   // 4
}

// Centered frame: 1 PRECEDING AND 1 FOLLOWING (3-wide, clamped at edges).
TEST(WindowAgg, CenteredFrame) {
    const double v[] = {10.0, 20.0, 30.0, 40.0};
    double out[4] = {};
    Frame fr{1, 1, false, false, {}};
    window_agg_f64(WinAgg::Mean, v, 4, fr, out);
    EXPECT_DOUBLE_EQ(out[0], 15.0);  // [10,20]
    EXPECT_DOUBLE_EQ(out[1], 20.0);  // [10,20,30]
    EXPECT_DOUBLE_EQ(out[2], 30.0);  // [20,30,40]
    EXPECT_DOUBLE_EQ(out[3], 35.0);  // [30,40]
}

}  // namespace
