// bolt::tz / bolt::cal / bolt::dtfmt: embedded IANA rules, calendar math and
// Spark datetime patterns. Expected offsets come from CPython's zoneinfo over
// the same tzdata; pattern outputs from java.time (Spark 4) semantics.
#include <gtest/gtest.h>

#include <cstring>
#include <string>

#include "bolt/datetime/bolt_calendar.h"
#include "bolt/datetime/bolt_datetime_fmt.h"
#include "bolt/datetime/bolt_datetime_kernels.h"
#include "bolt/datetime/bolt_tz.h"

namespace {

using bolt::tz::ZoneId;

ZoneId zone(const char* name) {
    ZoneId z = bolt::tz::kInvalidZone;
    EXPECT_TRUE(bolt::tz::zone_parse(name, static_cast<std::uint32_t>(std::strlen(name)), &z)) << name;
    return z;
}

std::string fmt(const char* pattern, std::int64_t utc_us, ZoneId z) {
    bolt::dtfmt::Pattern p{};
    EXPECT_EQ(bolt::dtfmt::compile(pattern, static_cast<std::uint32_t>(std::strlen(pattern)), &p),
              bolt::dtfmt::CompileStatus::Ok) << pattern;
    char buf[256];
    const std::uint32_t n = bolt::dtfmt::format_instant(p, utc_us, z, buf, sizeof(buf));
    return std::string(buf, n);
}

bool parse(const char* pattern, const char* text, ZoneId z, std::int64_t* out) {
    bolt::dtfmt::Pattern p{};
    EXPECT_EQ(bolt::dtfmt::compile(pattern, static_cast<std::uint32_t>(std::strlen(pattern)), &p),
              bolt::dtfmt::CompileStatus::Ok);
    return bolt::dtfmt::parse_instant(p, text, static_cast<std::uint32_t>(std::strlen(text)), z, out);
}

constexpr std::int64_t kUs = 1000000;

TEST(BoltTz, ParsesRegionsOffsetsAndAliases) {
    ZoneId z = 0;
    EXPECT_TRUE(bolt::tz::zone_parse("UTC", 3, &z));
    EXPECT_EQ(z, bolt::tz::kUtc);
    EXPECT_TRUE(bolt::tz::zone_parse("+08:00", 6, &z));
    EXPECT_EQ(bolt::tz::offset_at(z, 0), 8 * 3600);
    EXPECT_TRUE(bolt::tz::zone_parse("GMT-0530", 8, &z));
    EXPECT_EQ(bolt::tz::offset_at(z, 0), -(5 * 3600 + 30 * 60));
    EXPECT_TRUE(bolt::tz::zone_parse("america/new_york", 16, &z));
    char buf[64];
    EXPECT_EQ(std::string(buf, bolt::tz::zone_name(z, buf, sizeof(buf))), "America/New_York");
    EXPECT_EQ(bolt::tz::offset_at(zone("PST"), 1705276800), -8 * 3600);
    EXPECT_FALSE(bolt::tz::zone_parse("Mars/Olympus", 12, &z));
    EXPECT_FALSE(bolt::tz::zone_parse("+19:00", 6, &z));
    EXPECT_EQ(z, bolt::tz::kInvalidZone);
}

TEST(BoltTz, OffsetsMatchZoneinfo) {
    struct Case { const char* zone; std::int64_t utc; std::int32_t off; };
    const Case cases[] = {
        {"America/New_York", 1705276800, -18000}, {"America/New_York", 1721001600, -14400},
        {"America/New_York", 4118083200, -14400}, {"America/New_York", -615513600, -14400},
        {"Australia/Sydney", 1705276800, 39600},  {"Australia/Sydney", 1721001600, 36000},
        {"Pacific/Chatham", 1705276800, 49500},   {"Asia/Kolkata", 1705276800, 19800},
        {"Europe/London", 1719792000, 3600},      {"Europe/London", -2208988800, 0},
    };
    for (const Case& c : cases) EXPECT_EQ(bolt::tz::offset_at(zone(c.zone), c.utc), c.off) << c.zone;
}

TEST(BoltTz, LocalToUtcFollowsJavaGapAndOverlapRules) {
    const ZoneId ny = zone("America/New_York");
    const std::int64_t gap_local = bolt::cal::days_from_ymd(2024, 3, 10) * 86400 + 2 * 3600 + 1800;
    EXPECT_EQ(bolt::tz::local_to_utc(ny, gap_local), gap_local + 5 * 3600);   // 03:30 EDT
    const std::int64_t overlap = bolt::cal::days_from_ymd(2024, 11, 3) * 86400 + 3600 + 1800;
    EXPECT_EQ(bolt::tz::local_to_utc(ny, overlap), overlap + 4 * 3600);       // earlier offset
    const std::int64_t plain = bolt::cal::days_from_ymd(2024, 1, 15) * 86400;
    EXPECT_EQ(bolt::tz::local_to_utc(ny, plain), plain + 5 * 3600);
    char buf[16];
    EXPECT_EQ(std::string(buf, bolt::tz::zone_abbr(ny, 1705276800, buf, sizeof(buf))), "EST");
    EXPECT_EQ(std::string(buf, bolt::tz::zone_abbr(ny, 1721001600, buf, sizeof(buf))), "EDT");
}

TEST(BoltCal, MonthArithmeticMatchesSpark) {
    using namespace bolt::cal;
    EXPECT_EQ(add_months(days_from_ymd(2024, 1, 31), 1), days_from_ymd(2024, 2, 29));
    EXPECT_EQ(add_months(days_from_ymd(2024, 3, 31), -13), days_from_ymd(2023, 2, 28));
    EXPECT_EQ(last_day(days_from_ymd(2023, 2, 10)), days_from_ymd(2023, 2, 28));
    EXPECT_EQ(next_day(days_from_ymd(2024, 3, 4), 1), days_from_ymd(2024, 3, 11));   // Mon -> Mon
    const std::int64_t a = days_from_ymd(2024, 1, 1) * kMicrosPerDay;
    EXPECT_DOUBLE_EQ(months_between(a, days_from_ymd(2021, 3, 15) * kMicrosPerDay, true), 33.5483871);
    EXPECT_DOUBLE_EQ(months_between(days_from_ymd(2024, 2, 29) * kMicrosPerDay,
                                    days_from_ymd(2024, 1, 31) * kMicrosPerDay, true), 1.0);
    EXPECT_EQ(trunc_days(days_from_ymd(2024, 5, 17), TruncUnit::Quarter), days_from_ymd(2024, 4, 1));
}

TEST(BoltDtfmt, FormatsSparkPatterns) {
    const std::int64_t ts = 1709300710LL * kUs + 123456;   // 2024-03-01 13:45:10.123456Z
    EXPECT_EQ(fmt("yyyy/MM", ts, bolt::tz::kUtc), "2024/03");
    EXPECT_EQ(fmt("yyyy-MM-dd'T'HH:mm:ss.SSSXXX", ts, zone("Asia/Kolkata")),
              "2024-03-01T19:15:10.123+05:30");
    EXPECT_EQ(fmt("EEE, d MMM yyyy hh:mm a", ts, bolt::tz::kUtc), "Fri, 1 Mar 2024 01:45 PM");
    EXPECT_EQ(fmt("EEEE MMMM D QQQ G", ts, bolt::tz::kUtc), "Friday March 61 Q1 AD");
    EXPECT_EQ(fmt("HH:mm z VV X Z O", ts, zone("America/New_York")),
              "08:45 EST America/New_York -05 -0500 GMT-5");
    EXPECT_EQ(fmt("X x", ts, bolt::tz::kUtc), "Z +00");
    EXPECT_EQ(fmt("''yy''", ts, bolt::tz::kUtc), "'24'");
    bolt::dtfmt::Pattern p{};
    EXPECT_EQ(bolt::dtfmt::compile("YYYY-ww", 7, &p), bolt::dtfmt::CompileStatus::IllegalLetter);
    EXPECT_EQ(bolt::dtfmt::compile("yyyy-[MM", 8, &p), bolt::dtfmt::CompileStatus::UnbalancedSection);
}

TEST(BoltDtfmt, ParsesWithPatterns) {
    std::int64_t us = 0;
    ASSERT_TRUE(parse("yyyy-MM-dd HH:mm:ss", "2024-03-01 13:45:10", bolt::tz::kUtc, &us));
    EXPECT_EQ(us, 1709300710LL * kUs);
    ASSERT_TRUE(parse("yyyyMMdd", "20240301", bolt::tz::kUtc, &us));
    EXPECT_EQ(us, 19783LL * 86400 * kUs);
    ASSERT_TRUE(parse("dd/MMM/yyyy:HH:mm:ss Z", "01/mar/2024:13:45:10 +0100", bolt::tz::kUtc, &us));
    EXPECT_EQ(us, (1709300710LL - 3600) * kUs);
    ASSERT_TRUE(parse("yyyy-MM-dd[ HH:mm]", "2024-03-01", zone("+02:00"), &us));
    EXPECT_EQ(us, (19783LL * 86400 - 7200) * kUs);
    ASSERT_TRUE(parse("yyyy-MM-dd HH:mm:ss.SSS", "2024-03-01 13:45:10.5", bolt::tz::kUtc, &us));
    EXPECT_EQ(us % kUs, 500000);
    EXPECT_FALSE(parse("yyyy-MM-dd", "2024-02-30", bolt::tz::kUtc, &us));
    EXPECT_FALSE(parse("yyyy-MM-dd", "2024-03-01x", bolt::tz::kUtc, &us));
    EXPECT_FALSE(parse("yyyy-MM-dd", "2024-3-1", bolt::tz::kUtc, &us));
}

TEST(BoltDtfmt, CastsTextLikeSpark) {
    std::int64_t d = 0;
    ASSERT_TRUE(bolt::dtfmt::parse_date_text(" 2021-3-5 ", 10, &d));
    EXPECT_EQ(d, bolt::cal::days_from_ymd(2021, 3, 5));
    ASSERT_TRUE(bolt::dtfmt::parse_date_text("2021-03-15T10:00", 16, &d));
    ASSERT_TRUE(bolt::dtfmt::parse_date_text("2021", 4, &d));
    EXPECT_EQ(d, bolt::cal::days_from_ymd(2021, 1, 1));
    EXPECT_FALSE(bolt::dtfmt::parse_date_text("21-03-15", 8, &d));
    EXPECT_FALSE(bolt::dtfmt::parse_date_text("2021-13-01", 10, &d));
    std::int64_t us = 0;
    ASSERT_TRUE(bolt::dtfmt::parse_timestamp_text("2024-03-01 13:45:10Z", 20, zone("+05:00"), &us));
    EXPECT_EQ(us, 1709300710LL * kUs);
    ASSERT_TRUE(bolt::dtfmt::parse_timestamp_text("2024-03-01T13:45:10.25+08:00", 28, bolt::tz::kUtc, &us));
    EXPECT_EQ(us, (1709300710LL - 8 * 3600) * kUs + 250000);
    const char* la = "2024-03-01 05:45:10 America/Los_Angeles";
    ASSERT_TRUE(bolt::dtfmt::parse_timestamp_text(la, static_cast<std::uint32_t>(std::strlen(la)),
                                                  bolt::tz::kUtc, &us));
    EXPECT_EQ(us, 1709300710LL * kUs);
    ASSERT_TRUE(bolt::dtfmt::parse_timestamp_text("2024-03-01 13", 13, bolt::tz::kUtc, &us));
    EXPECT_EQ(us, (19783LL * 86400 + 13 * 3600) * kUs);
    EXPECT_FALSE(bolt::dtfmt::parse_timestamp_text("2024-03-01 25:00", 16, bolt::tz::kUtc, &us));
    char buf[64];
    EXPECT_EQ(std::string(buf, bolt::dtfmt::format_timestamp_default(1709300710LL * kUs + 120000, buf, 64)),
              "2024-03-01 13:45:10.12");
    EXPECT_EQ(std::string(buf, bolt::dtfmt::format_date_default(19783, buf, 64)), "2024-03-01");
}

// Column kernels against the scalar oracle, over sorted runs (cursor reuse,
// DST edges) and scrambled order (cursor misses), in several zones.
TEST(BoltDtk, ColumnKernelsMatchScalarOracle) {
    constexpr std::int64_t kN = 20000;
    static std::int64_t in[kN], out[kN], local[kN], f[kN];
    static std::int32_t days[kN];
    std::uint64_t x = 0x9E3779B97F4A7C15ull;
    for (std::int64_t i = 0; i < kN; ++i) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        const std::int64_t sorted = -2208988800LL * kUs + i * 1800LL * 7 * kUs + i * 13;   // 1900.. step ~3.5h
        const std::int64_t rnd = static_cast<std::int64_t>(x % (8000000000ULL * 1000000ULL)) - 3000000000LL * kUs;
        in[i] = (i % 3 == 0) ? rnd : sorted;
    }
    const char* zones[] = {"America/New_York", "Australia/Lord_Howe", "Europe/London", "Asia/Kolkata",
                           "+05:30", "Pacific/Chatham", "America/Sao_Paulo"};
    for (const char* zn : zones) {
        const ZoneId z = zone(zn);
        bolt::dtk::utc_to_local_us(z, in, kN, local);
        for (std::int64_t i = 0; i < kN; ++i) ASSERT_EQ(local[i], bolt::tz::utc_micros_to_local(z, in[i])) << zn << i;
        bolt::dtk::local_to_utc_us(z, in, kN, out);
        for (std::int64_t i = 0; i < kN; ++i) ASSERT_EQ(out[i], bolt::tz::local_micros_to_utc(z, in[i])) << zn << i;
        bolt::dtk::trunc_utc_us(z, bolt::cal::TruncUnit::Hour, in, kN, out);
        for (std::int64_t i = 0; i < kN; ++i) {
            const std::int64_t t = bolt::cal::trunc_local_micros(bolt::tz::utc_micros_to_local(z, in[i]),
                                                                 bolt::cal::TruncUnit::Hour);
            ASSERT_EQ(out[i], bolt::tz::local_micros_to_utc(z, t)) << zn << i;
        }
        bolt::dtk::utc_us_to_days(z, in, kN, days);
        for (std::int64_t i = 0; i < kN; ++i)
            ASSERT_EQ(days[i], bolt::tz::floor_div(local[i], bolt::cal::kMicrosPerDay)) << zn << i;
        bolt::dtk::days_to_utc_us(z, days, kN, out);
        for (std::int64_t i = 0; i < kN; ++i)
            ASSERT_EQ(out[i], bolt::tz::local_micros_to_utc(z, days[i] * bolt::cal::kMicrosPerDay)) << zn << i;
    }
    using F = bolt::dtk::Field;
    for (F fld : {F::Year, F::Month, F::Day, F::Hour, F::Minute, F::Second, F::DayOfWeek, F::DayOfYear,
                  F::WeekOfYear, F::Quarter}) {
        bolt::dtk::field_local_us(fld, in, kN, f);
        for (std::int64_t i = 0; i < kN; ++i) {
            bolt::cal::LocalFields lf{};
            bolt::cal::fields_from_local_micros(in[i], &lf);
            const std::int64_t d = bolt::tz::floor_div(in[i], bolt::cal::kMicrosPerDay);
            const std::int64_t doy = d - bolt::cal::days_from_ymd(lf.year, 1, 1) + 1;
            std::int64_t want = 0;
            switch (fld) {
                case F::Year: want = lf.year; break;
                case F::Month: want = lf.month; break;
                case F::Day: want = lf.day; break;
                case F::Hour: want = lf.hour; break;
                case F::Minute: want = lf.minute; break;
                case F::Second: want = lf.second; break;
                case F::DayOfWeek: want = bolt::cal::iso_dow(d) % 7 + 1; break;
                case F::DayOfYear: want = doy; break;
                case F::WeekOfYear:
                    want = bolt::kernels::date::iso_week_from_components(lf.year, static_cast<std::int32_t>(doy),
                                                                         bolt::cal::iso_dow(d));
                    break;
                case F::Quarter: want = (lf.month - 1) / 3 + 1; break;
                default: break;
            }
            ASSERT_EQ(f[i], want) << static_cast<int>(fld) << " " << i;
        }
    }
}

TEST(BoltDtk, FixedFormatAndTextDatesMatchOracle) {
    const char* pats[] = {"yyyy-MM-dd HH:mm:ss", "yyyyMMdd'T'HHmmss.SSS", "yy/MM/dd", "yyyy-MM-dd HH:mm:ss.SSSSSS"};
    for (const char* pat : pats) {
        bolt::dtfmt::Pattern p{};
        ASSERT_EQ(bolt::dtfmt::compile(pat, static_cast<std::uint32_t>(std::strlen(pat)), &p),
                  bolt::dtfmt::CompileStatus::Ok);
        const std::uint32_t w = bolt::dtk::format_fixed_width(p);
        ASSERT_NE(w, 0u) << pat;
        for (std::int64_t us = -1000000LL * kUs; us < 4000000000LL * kUs; us += 7777777777123LL) {
            char a[64], b[64];
            bolt::dtk::format_fixed(p, us, a);
            const std::uint32_t n = bolt::dtfmt::format_instant(p, us, bolt::tz::kUtc, b, sizeof(b));
            ASSERT_EQ(std::string(a, w), std::string(b, n)) << pat;
        }
    }
    bolt::dtfmt::Pattern q{};
    ASSERT_EQ(bolt::dtfmt::compile("EEE yyyy", 8, &q), bolt::dtfmt::CompileStatus::Ok);
    EXPECT_EQ(bolt::dtk::format_fixed_width(q), 0u);

    const char* texts[] = {"2024-02-29", "1969-12-31", "2023-02-29", "2021-3-5", " 2021-03-05", "2021-03-05T10:00:00",
                           "abcd-ef-gh"};
    const bool ok[] = {true, true, false, true, true, true, false};
    for (int k = 0; k < 7; ++k) {
        static char pool[64];
        const std::uint32_t len = static_cast<std::uint32_t>(std::strlen(texts[k]));
        bolt::StringView v;
        std::memset(&v, 0, sizeof(v));
        v.length = len;
        if (len <= 12) {
            char tmp[12] = {0};
            std::memcpy(tmp, texts[k], len);
            std::memcpy(v.prefix, tmp, 4);
            std::memcpy(v.inline_data, tmp + 4, 8);
        } else {
            std::memcpy(pool, texts[k], len);
            std::memcpy(v.prefix, texts[k], 4);
            v.ref.offset = 0;
        }
        std::int32_t d = 0;
        const bool got = bolt::dtk::text_to_days(&v, pool, 1, &d);
        ASSERT_EQ(got, ok[k]) << texts[k];
        std::int64_t want = 0;
        if (got) {
            ASSERT_TRUE(bolt::dtfmt::parse_date_text(texts[k], len, &want));
            EXPECT_EQ(d, want) << texts[k];
        }
    }
}

}  // namespace
