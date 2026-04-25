#include <gtest/gtest.h>
#include "bolt/kernels/bolt_column_stats.h"

#include <cmath>
#include <limits>

using bolt::ColumnStatsF32;
using bolt::column_stats_f32;
using bolt::column_stats_f32_strided;

TEST(ColumnStatsF32, KnownSmallArray) {
    const float data[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    ColumnStatsF32 s{};
    column_stats_f32(data, 5, &s);

    EXPECT_FLOAT_EQ(s.mean, 3.0f);
    EXPECT_FLOAT_EQ(s.var, 2.0f);  // population variance = ((-2)^2 + (-1)^2 + 0 + 1 + 4)/5 = 10/5 = 2.0
    EXPECT_FLOAT_EQ(s.min, 1.0f);
    EXPECT_FLOAT_EQ(s.max, 5.0f);
    EXPECT_EQ(s.n, 5u);
}

TEST(ColumnStatsF32, EmptyReturnsSentinels) {
    ColumnStatsF32 s{};
    column_stats_f32(nullptr, 0, &s);

    EXPECT_FLOAT_EQ(s.mean, 0.0f);
    EXPECT_FLOAT_EQ(s.var, 0.0f);
    EXPECT_TRUE(std::isinf(s.min) && s.min > 0.0f);
    EXPECT_TRUE(std::isinf(s.max) && s.max < 0.0f);
    EXPECT_EQ(s.n, 0u);
}

TEST(ColumnStatsF32, SingleElement) {
    const float data[1] = {7.5f};
    ColumnStatsF32 s{};
    column_stats_f32(data, 1, &s);

    EXPECT_FLOAT_EQ(s.mean, 7.5f);
    EXPECT_FLOAT_EQ(s.var, 0.0f);
    EXPECT_FLOAT_EQ(s.min, 7.5f);
    EXPECT_FLOAT_EQ(s.max, 7.5f);
    EXPECT_EQ(s.n, 1u);
}

TEST(ColumnStatsF32, StridedMatchesContiguous) {
    const float data[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    ColumnStatsF32 contig{};
    ColumnStatsF32 strided{};
    column_stats_f32(data, 5, &contig);
    column_stats_f32_strided(data, 5, sizeof(float), &strided);

    EXPECT_FLOAT_EQ(strided.mean, contig.mean);
    EXPECT_FLOAT_EQ(strided.var,  contig.var);
    EXPECT_FLOAT_EQ(strided.min,  contig.min);
    EXPECT_FLOAT_EQ(strided.max,  contig.max);
    EXPECT_EQ(strided.n, contig.n);
}

TEST(ColumnStatsF32, StridedSkipsEverySecondElement) {
    // Interleaved buffer: [1.0, junk, 2.0, junk, 3.0, junk, 4.0, junk, 5.0, junk]
    const float interleaved[10] = {
        1.0f, 99.0f,
        2.0f, 99.0f,
        3.0f, 99.0f,
        4.0f, 99.0f,
        5.0f, 99.0f,
    };
    ColumnStatsF32 picked{};
    column_stats_f32_strided(interleaved, 5, sizeof(float) * 2, &picked);

    // Should match the contiguous {1,2,3,4,5} stats.
    EXPECT_FLOAT_EQ(picked.mean, 3.0f);
    EXPECT_FLOAT_EQ(picked.var,  2.0f);
    EXPECT_FLOAT_EQ(picked.min,  1.0f);
    EXPECT_FLOAT_EQ(picked.max,  5.0f);
    EXPECT_EQ(picked.n, 5u);
}

TEST(ColumnStatsF32, NaNAtIndexTwoPropagatesIntoMean) {
    const float data[5] = {
        1.0f,
        2.0f,
        std::numeric_limits<float>::quiet_NaN(),
        4.0f,
        5.0f,
    };
    ColumnStatsF32 s{};
    column_stats_f32(data, 5, &s);

    // Welford running mean absorbs NaN — propagation is the documented behaviour.
    EXPECT_TRUE(std::isnan(s.mean));
    // n counts every input slot regardless of NaN.
    EXPECT_EQ(s.n, 5u);
    // For this scalar implementation, NaN compares false to everything, so
    // min/max keep tracking only the non-NaN values they have seen.
    EXPECT_FLOAT_EQ(s.min, 1.0f);
    EXPECT_FLOAT_EQ(s.max, 5.0f);
}
