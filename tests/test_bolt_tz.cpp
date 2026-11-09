// bolt::tz / bolt::cal / bolt::dtfmt: embedded IANA rules, calendar math and
// Spark datetime patterns. Expected offsets come from CPython's zoneinfo over
// the same tzdata; pattern outputs from java.time (Spark 4) semantics.
#include <gtest/gtest.h>

#include <cstring>
#include <string>

#include "bolt/datetime/bolt_calendar.h"
#include "bolt/datetime/bolt_datetime_fmt.h"
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

}  // namespace
