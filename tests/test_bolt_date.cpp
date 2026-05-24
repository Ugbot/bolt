// test_bolt_date.cpp — Date32 extract kernel correctness + microbench.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "bolt/kernels/bolt_date.h"

namespace bk = bolt::kernels::date;

namespace {

// Reference: compute days-since-1970 for a civil (y,m,d) via the same Hinnant
// inverse. We mirror days_from_civil here so the test isn't tautological if
// someone breaks the header — both must agree.
static int32_t ref_days(int y, unsigned m, unsigned d) {
    int yy = y - (m <= 2 ? 1 : 0);
    int era = (yy >= 0 ? yy : yy - 399) / 400;
    unsigned yoe = static_cast<unsigned>(yy - era * 400);
    unsigned doy = (153u * (m > 2 ? m - 3 : m + 9) + 2) / 5 + d - 1;
    unsigned doe = yoe * 365 + yoe/4 - yoe/100 + doy;
    return static_cast<int32_t>(era * 146097 + static_cast<int>(doe) - 719468);
}

static bool is_leap(int y) {
    return (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
}

}  // namespace

// ---------------------------------------------------------------------------
// Anchors
// ---------------------------------------------------------------------------
TEST(BoltDate, EpochAnchor) {
    int32_t d = 0;  // 1970-01-01 Thursday
    int32_t y = 0, m = 0, dd = 0, dow = 0;
    bk::date32_extract_year (&d, &y,   1);
    bk::date32_extract_month(&d, &m,   1);
    bk::date32_extract_day  (&d, &dd,  1);
    bk::date32_extract_day_of_week(&d, &dow, 1);
    EXPECT_EQ(y,  1970);
    EXPECT_EQ(m,  1);
    EXPECT_EQ(dd, 1);
    EXPECT_EQ(dow, 4);  // Thursday
}

TEST(BoltDate, KnownDates) {
    // (days, expected y, m, d, dow Mon=1..Sun=7)
    struct C { int32_t d; int32_t y; int32_t m; int32_t dd; int32_t dow; };
    C tab[] = {
        { ref_days(1900,  1,  1), 1900,  1,  1, 1 },  // Mon
        { ref_days(2000,  2, 29), 2000,  2, 29, 2 },  // Tue (leap)
        { ref_days(2024,  2, 29), 2024,  2, 29, 4 },  // Thu (leap)
        { ref_days(2024, 12, 31), 2024, 12, 31, 2 },  // Tue
        { ref_days(2025,  1,  1), 2025,  1,  1, 3 },  // Wed
        { ref_days(2026,  5, 17), 2026,  5, 17, 7 },  // Sun
        { ref_days(2100,  3,  1), 2100,  3,  1, 1 },  // Mon (2100 NOT leap)
        { ref_days(2200,  1,  1), 2200,  1,  1, 3 },  // Wed
        { ref_days(1969, 12, 31), 1969, 12, 31, 3 },  // Wed
        { ref_days(1900,  2, 28), 1900,  2, 28, 3 },  // Wed (1900 NOT leap)
    };
    for (const auto& c : tab) {
        int32_t y=0, m=0, dd=0, dow=0;
        bk::date32_extract_year (&c.d, &y,   1);
        bk::date32_extract_month(&c.d, &m,   1);
        bk::date32_extract_day  (&c.d, &dd,  1);
        bk::date32_extract_day_of_week(&c.d, &dow, 1);
        EXPECT_EQ(y,  c.y)  << "y for d=" << c.d;
        EXPECT_EQ(m,  c.m)  << "m for d=" << c.d;
        EXPECT_EQ(dd, c.dd) << "d for d=" << c.d;
        EXPECT_EQ(dow,c.dow)<< "dow for d=" << c.d;
    }
}

// ---------------------------------------------------------------------------
// Roundtrip over 1900..2200 — every day must round-trip through extract.
// This is ~110k iterations; quick.
// ---------------------------------------------------------------------------
TEST(BoltDate, FullRangeRoundtrip1900To2200) {
    int errs = 0;
    for (int y = 1900; y <= 2200 && errs < 5; ++y) {
        int dim_table[13] = {0,31,28,31,30,31,30,31,31,30,31,30,31};
        if (is_leap(y)) dim_table[2] = 29;
        for (int m = 1; m <= 12 && errs < 5; ++m) {
            for (int d = 1; d <= dim_table[m] && errs < 5; ++d) {
                int32_t days = ref_days(y, m, d);
                int32_t y2=0, m2=0, d2=0;
                bk::date32_extract_year (&days, &y2, 1);
                bk::date32_extract_month(&days, &m2, 1);
                bk::date32_extract_day  (&days, &d2, 1);
                if (y2 != y || m2 != m || d2 != d) {
                    ADD_FAILURE() << "roundtrip mismatch y=" << y << " m=" << m
                                  << " d=" << d << " got (" << y2 << "," << m2
                                  << "," << d2 << ") days=" << days;
                    ++errs;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Joint YMD agrees with separate extracts.
// ---------------------------------------------------------------------------
TEST(BoltDate, JointYmdAgreesWithSeparate) {
    constexpr int N = 4096;
    std::vector<int32_t> days(N);
    std::mt19937 rng(0xC0FFEEu);
    // Span ~1900..2200 in day-space: 1900-01-01 ≈ -25567, 2200-01-01 ≈ 83977.
    std::uniform_int_distribution<int32_t> dist(-25567, 83977);
    for (int i = 0; i < N; ++i) days[i] = dist(rng);

    std::vector<int32_t> ys(N), ms(N), ds(N);
    bk::date32_extract_year (days.data(), ys.data(), N);
    bk::date32_extract_month(days.data(), ms.data(), N);
    bk::date32_extract_day  (days.data(), ds.data(), N);

    std::vector<bk::YMD32> joint(N);
    bk::date32_to_ymd(days.data(), joint.data(), N);

    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(joint[i].y, ys[i]);
        EXPECT_EQ(static_cast<int32_t>(joint[i].m), ms[i]);
        EXPECT_EQ(static_cast<int32_t>(joint[i].d), ds[i]);
    }
}

// ---------------------------------------------------------------------------
// Quarter / day-of-year sanity.
// ---------------------------------------------------------------------------
TEST(BoltDate, QuarterAndDayOfYear) {
    struct C { int32_t d; int32_t q; int32_t doy; };
    C tab[] = {
        { ref_days(2024,  1,  1), 1,   1 },
        { ref_days(2024,  3, 31), 1,  91 },  // leap
        { ref_days(2024,  4,  1), 2,  92 },
        { ref_days(2024,  7,  1), 3, 183 },
        { ref_days(2024, 10,  1), 4, 275 },
        { ref_days(2024, 12, 31), 4, 366 },  // leap year last doy
        { ref_days(2023, 12, 31), 4, 365 },
    };
    for (const auto& c : tab) {
        int32_t q=0, doy=0;
        bk::date32_extract_quarter   (&c.d, &q,   1);
        bk::date32_extract_day_of_year(&c.d, &doy, 1);
        EXPECT_EQ(q,   c.q)   << "q  for d=" << c.d;
        EXPECT_EQ(doy, c.doy) << "doy for d=" << c.d;
    }
}

// ---------------------------------------------------------------------------
// ISO week — anchors from Wikipedia ISO 8601 table.
// ---------------------------------------------------------------------------
TEST(BoltDate, IsoWeek) {
    struct C { int32_t d; int32_t w; };
    C tab[] = {
        { ref_days(2024,  1,  1),  1 },  // Mon — start of W1
        { ref_days(2024, 12, 30),  1 },  // Mon — belongs to 2025-W1
        { ref_days(2024, 12, 29), 52 },  // Sun — last day of 2024-W52
        { ref_days(2020, 12, 31), 53 },  // 2020 is an ISO 53-week year
        { ref_days(2021,  1,  1), 53 },  // Fri — still 2020-W53
        { ref_days(2021,  1,  4),  1 },  // Mon — start of 2021-W1
        { ref_days(2005,  1,  1), 53 },  // 2004-W53
        { ref_days(2026,  5, 17), 20 },  // Sun of week 20 of 2026
    };
    for (const auto& c : tab) {
        int32_t w = 0;
        bk::date32_extract_week(&c.d, &w, 1);
        EXPECT_EQ(w, c.w) << "week for d=" << c.d;
    }
}

// ---------------------------------------------------------------------------
// Filter wrappers — Date32 ≥ '1995-01-01' style predicate.
// ---------------------------------------------------------------------------
TEST(BoltDate, FilterGeWrapper) {
    const int32_t cutoff = ref_days(1995, 1, 1);
    std::vector<int32_t> days = {
        ref_days(1994, 12, 31),
        ref_days(1995,  1,  1),
        ref_days(1995,  6, 15),
        ref_days(1994,  1,  1),
        ref_days(2024,  2, 29),
    };
    std::vector<int32_t> sel(days.size(), 0);
    int64_t k = bk::date32_filter_ge(days.data(), static_cast<int64_t>(days.size()),
                                     cutoff, sel.data());
    EXPECT_EQ(k, 3);
    EXPECT_EQ(sel[0], 1);
    EXPECT_EQ(sel[1], 2);
    EXPECT_EQ(sel[2], 4);
}

// ---------------------------------------------------------------------------
// Microbench — print ns/row. Not a gate; informational. Skipped under DEBUG.
// ---------------------------------------------------------------------------
TEST(BoltDate, BenchExtract) {
#ifndef NDEBUG
    GTEST_SKIP() << "skip bench under debug";
#endif
    constexpr int N = 1 << 20;  // 1Mi rows
    std::vector<int32_t> days(N);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> dist(-25567, 83977);
    for (int i = 0; i < N; ++i) days[i] = dist(rng);

    std::vector<int32_t> out(N);
    std::vector<bk::YMD32> joint(N);

    auto bench = [&](const char* name, auto fn) {
        // warmup
        fn();
        constexpr int iters = 8;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        double nsr = ns / (double)(iters) / (double)N;
        std::printf("[bench] %-28s %.3f ns/row\n", name, nsr);
    };

    bench("date32_extract_year",  [&]{ bk::date32_extract_year (days.data(), out.data(), N); });
    bench("date32_extract_month", [&]{ bk::date32_extract_month(days.data(), out.data(), N); });
    bench("date32_extract_day",   [&]{ bk::date32_extract_day  (days.data(), out.data(), N); });
    bench("date32_extract_quarter",[&]{bk::date32_extract_quarter(days.data(), out.data(), N); });
    bench("date32_extract_dow",   [&]{ bk::date32_extract_day_of_week(days.data(), out.data(), N); });
    bench("date32_extract_doy",   [&]{ bk::date32_extract_day_of_year(days.data(), out.data(), N); });
    bench("date32_extract_week",  [&]{ bk::date32_extract_week (days.data(), out.data(), N); });
    bench("date32_to_ymd",        [&]{ bk::date32_to_ymd       (days.data(), joint.data(), N); });

    std::vector<int32_t> sel(N);
    const int32_t cutoff = ref_days(2000, 1, 1);
    bench("date32_filter_gt",     [&]{ (void)bk::date32_filter_gt(days.data(), N, cutoff, sel.data()); });
}
