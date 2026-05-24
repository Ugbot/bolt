// test_bolt_interval.cpp — SQL INTERVAL arithmetic kernels.

#include <gtest/gtest.h>

#include <cstdint>

#include "bolt/kernels/bolt_interval.h"
#include "bolt/kernels/bolt_temporal.h"

namespace {

using bolt::kernels::interval::Interval;
using bolt::kernels::interval::IntervalDate;
using bolt::kernels::interval::add_interval_one;
using bolt::kernels::interval::days_from_civil;
using bolt::kernels::interval::days_in_month;
using bolt::kernels::interval::date32_add_interval;
using bolt::kernels::interval::date32_add_interval_scalar;
using bolt::kernels::interval::date32_sub_date32;
using bolt::kernels::interval::interval_mul_scalar;
using bolt::kernels::interval::interval_filter_gt;

// Helper: day-serial for civil date.
static int32_t serial(int y, unsigned m, unsigned d) noexcept {
    return static_cast<int32_t>(
        days_from_civil(y, m, d));
}

TEST(BoltInterval, ReprSizes) {
    EXPECT_EQ(sizeof(Interval), 16u);
    EXPECT_EQ(sizeof(IntervalDate), 8u);
}

TEST(BoltInterval, DaysFromCivilRoundTrip) {
    // Known epoch.
    EXPECT_EQ(days_from_civil(1970, 1, 1), 0);
    // 1998-12-01 is the TPC-H reference date. Round trip via civil_from_days.
    int64_t d = days_from_civil(1998, 12, 1);
    bolt::kernels::YMD r = bolt::kernels::civil_from_days(d);
    EXPECT_EQ(r.y, 1998);
    EXPECT_EQ(r.m, 12u);
    EXPECT_EQ(r.d, 1u);
}

TEST(BoltInterval, DaysInMonth) {
    EXPECT_EQ(days_in_month(2020, 2), 29u);  // leap
    EXPECT_EQ(days_in_month(1900, 2), 28u);  // century non-leap
    EXPECT_EQ(days_in_month(2000, 2), 29u);  // 400-year leap
    EXPECT_EQ(days_in_month(2021, 2), 28u);
    EXPECT_EQ(days_in_month(1998, 12), 31u);
    EXPECT_EQ(days_in_month(1998, 4), 30u);
}

TEST(BoltInterval, AddDaysOnly) {
    int32_t in[4]  = { serial(1998, 12, 1), serial(2020, 1, 1),
                        serial(1970, 1, 1), serial(2026, 5, 16) };
    IntervalDate iv[4] = { {0, -90}, {0, 30}, {0, 0}, {0, 1} };
    int32_t out[4];
    date32_add_interval(in, iv, out, 4);
    EXPECT_EQ(out[0], serial(1998, 9, 2));   // 1998-12-01 - 90 days
    EXPECT_EQ(out[1], serial(2020, 1, 31));
    EXPECT_EQ(out[2], serial(1970, 1, 1));
    EXPECT_EQ(out[3], serial(2026, 5, 17));
}

TEST(BoltInterval, AddMonthsClamps) {
    // 2020-01-31 + 1 month should clamp to 2020-02-29 (leap).
    int32_t in = serial(2020, 1, 31);
    int32_t got = add_interval_one(in, 1, 0);
    EXPECT_EQ(got, serial(2020, 2, 29));
    // 2021-01-31 + 1 month -> 2021-02-28.
    EXPECT_EQ(add_interval_one(serial(2021, 1, 31), 1, 0), serial(2021, 2, 28));
    // 2020-03-31 + 1 month -> 2020-04-30.
    EXPECT_EQ(add_interval_one(serial(2020, 3, 31), 1, 0), serial(2020, 4, 30));
}

TEST(BoltInterval, AddYearsAndMonthsNegative) {
    // 1998-12-01 - 1 year -> 1997-12-01.
    EXPECT_EQ(add_interval_one(serial(1998, 12, 1), -12, 0),
              serial(1997, 12, 1));
    // 1998-01-15 - 2 months -> 1997-11-15.
    EXPECT_EQ(add_interval_one(serial(1998, 1, 15), -2, 0),
              serial(1997, 11, 15));
    // 2020-02-29 + 12 months -> 2021-02-28 (leap clamp).
    EXPECT_EQ(add_interval_one(serial(2020, 2, 29), 12, 0),
              serial(2021, 2, 28));
}

TEST(BoltInterval, AddMixedMonthsAndDays) {
    // 1998-12-01 - 3 months + 5 days.
    int32_t got = add_interval_one(serial(1998, 12, 1), -3, 5);
    EXPECT_EQ(got, serial(1998, 9, 6));
}

TEST(BoltInterval, AddIntervalScalarBroadcast) {
    int32_t in[3]  = { serial(1998, 12, 1), serial(1998, 12, 1),
                        serial(1998, 12, 1) };
    int32_t out[3];
    date32_add_interval_scalar(in, IntervalDate{0, -90}, out, 3);
    EXPECT_EQ(out[0], serial(1998, 9, 2));
    EXPECT_EQ(out[1], serial(1998, 9, 2));
    EXPECT_EQ(out[2], serial(1998, 9, 2));

    // Year shift via months path.
    date32_add_interval_scalar(in, IntervalDate{-12, 0}, out, 3);
    EXPECT_EQ(out[0], serial(1997, 12, 1));
}

TEST(BoltInterval, AddIntervalFastPathAllMonthsZero) {
    // 16 rows, all months==0 -> exercises pre-scan.
    int32_t in[16];
    IntervalDate iv[16];
    int32_t out[16];
    for (int i = 0; i < 16; ++i) {
        in[i] = serial(2020, 1, 1) + i;
        iv[i] = IntervalDate{0, i};
    }
    date32_add_interval(in, iv, out, 16);
    for (int i = 0; i < 16; ++i) {
        EXPECT_EQ(out[i], in[i] + i) << "row " << i;
    }
}

TEST(BoltInterval, SubDate) {
    int32_t a[3] = { serial(1998, 12, 1), serial(2020, 1, 1),
                     serial(2026, 5, 16) };
    int32_t b[3] = { serial(1998, 9, 2),  serial(2019, 12, 31),
                     serial(2026, 5, 1) };
    IntervalDate out[3];
    date32_sub_date32(a, b, out, 3);
    EXPECT_EQ(out[0].months, 0);
    EXPECT_EQ(out[0].days,   90);
    EXPECT_EQ(out[1].months, 0);
    EXPECT_EQ(out[1].days,   1);
    EXPECT_EQ(out[2].days,   15);
}

TEST(BoltInterval, MulScalar) {
    IntervalDate in[3]  = { {1, 2}, {0, 7}, {-1, -1} };
    IntervalDate out[3];
    interval_mul_scalar(in, 3, out, 3);
    EXPECT_EQ(out[0].months, 3);
    EXPECT_EQ(out[0].days,   6);
    EXPECT_EQ(out[1].months, 0);
    EXPECT_EQ(out[1].days,   21);
    EXPECT_EQ(out[2].months, -3);
    EXPECT_EQ(out[2].days,   -3);
}

TEST(BoltInterval, FilterGt) {
    IntervalDate data[5] = {
        {0, 1},     // 1 day
        {0, 31},    // 31 days
        {1, 0},     // 1 month -> 30 days for cmp
        {0, 100},   // 100 days
        {-1, 31},   // -30 + 31 = 1 day for cmp
    };
    int32_t sel[5];
    // gt 29 days: row 1 (31), row 2 (30), row 3 (100).
    int64_t n = interval_filter_gt(data, 5, IntervalDate{0, 29}, sel);
    EXPECT_EQ(n, 3);
    EXPECT_EQ(sel[0], 1);
    EXPECT_EQ(sel[1], 2);
    EXPECT_EQ(sel[2], 3);
}

TEST(BoltInterval, EmptyInputsAreSafe) {
    int32_t* p = nullptr;
    IntervalDate* iv = nullptr;
    date32_add_interval(p, iv, p, 0);
    date32_add_interval_scalar(p, IntervalDate{0, 0}, p, 0);
    date32_sub_date32(p, p, iv, 0);
    interval_mul_scalar(iv, 7, iv, 0);
    int32_t* sel = nullptr;
    EXPECT_EQ(interval_filter_gt(iv, 0, IntervalDate{0, 0}, sel), 0);
}

}  // namespace
