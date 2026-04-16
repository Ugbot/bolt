// test_bolt_strtemp.cpp — String and temporal kernels.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_types.h"
#include "bolt/kernels/bolt_string.h"
#include "bolt/kernels/bolt_temporal.h"

namespace {

static bolt::StringView make_inline(const char* s) noexcept {
    return bolt::StringView::from_cstr(s);
}

// Build a "spilled" StringView (length > 12). Fills length + 4-byte prefix;
// ref fields point nowhere valid — we only exercise prefix-level comparisons.
static bolt::StringView make_spilled(const char* s, uint32_t len) noexcept {
    bolt::StringView v;
    std::memset(&v, 0, sizeof(v));
    v.length = len;
    std::memcpy(v.prefix, s, 4);
    v.ref.buf_idx = 0;
    v.ref.offset  = 0;
    return v;
}

TEST(BoltStrTemp, Utf8EqualsMixedInlineSpilled) {
    // 6 rows: 4 inline, 2 spilled; needle is "hello" (inline). Expect 2 matches.
    bolt::StringView rows[6];
    rows[0] = make_inline("hello");              // match
    rows[1] = make_inline("world");              // length 5 != 5 but prefix differs
    rows[2] = make_inline("help");               // length differs
    rows[3] = make_inline("hello");              // match
    rows[4] = make_spilled("hellothere_long",   15);  // spilled, length != 5
    rows[5] = make_spilled("helloworld_stuff",  16);  // spilled, length != 5

    bolt::StringView needle = make_inline("hello");
    int32_t out[6] = {};
    int64_t hits = bolt::kernels::utf8_equals(rows, 6, needle, out);
    EXPECT_EQ(hits, 2);
    EXPECT_EQ(out[0], 0);
    EXPECT_EQ(out[1], 3);
}

TEST(BoltStrTemp, Utf8StartsWithFoo) {
    bolt::StringView rows[4];
    rows[0] = make_inline("foobar");
    rows[1] = make_inline("food");
    rows[2] = make_inline("bar");
    rows[3] = make_inline("fo");

    bolt::StringView prefix = make_inline("foo");
    int32_t out[4] = {};
    int64_t hits = bolt::kernels::utf8_starts_with(rows, 4, prefix, out);
    EXPECT_EQ(hits, 2);
    EXPECT_EQ(out[0], 0);
    EXPECT_EQ(out[1], 1);
}

TEST(BoltStrTemp, Utf8LengthBytes) {
    bolt::StringView rows[3];
    rows[0] = make_inline("a");
    rows[1] = make_inline("abcdef");
    rows[2] = make_inline("");

    int32_t lens[3] = {};
    bolt::kernels::utf8_length_bytes(rows, 3, lens);
    EXPECT_EQ(lens[0], 1);
    EXPECT_EQ(lens[1], 6);
    EXPECT_EQ(lens[2], 0);
}

TEST(BoltStrTemp, Utf8ContainsInline) {
    bolt::StringView rows[3];
    rows[0] = make_inline("foobar");
    rows[1] = make_inline("abc");
    rows[2] = make_inline("oobaz");

    bolt::StringView needle = make_inline("oob");
    int32_t out[3] = {};
    int64_t hits = bolt::kernels::utf8_contains(rows, 3, needle, out);
    EXPECT_EQ(hits, 2);
    EXPECT_EQ(out[0], 0);
    EXPECT_EQ(out[1], 2);
}

TEST(BoltStrTemp, Utf8UpperAsciiInline) {
    bolt::Arena arena{};
    bolt::StringView rows[2];
    rows[0] = make_inline("Hello");
    rows[1] = make_inline("abc");

    bolt::StringView out[2];
    bolt::kernels::utf8_upper_ascii(rows, 2, out, &arena);
    EXPECT_EQ(out[0].length, 5u);
    EXPECT_EQ(std::memcmp(out[0].prefix, "HELL", 4), 0);
    EXPECT_EQ(out[0].inline_data[0], 'O');
    EXPECT_EQ(std::memcmp(out[1].prefix, "ABC", 3), 0);
}

TEST(BoltStrTemp, TimestampAddSeconds) {
    int64_t ts[3]  = {0, 100, -50};
    int64_t out[3] = {};
    bolt::kernels::timestamp_add_seconds(ts, 3, 3600, out);
    EXPECT_EQ(out[0], 3600);
    EXPECT_EQ(out[1], 3700);
    EXPECT_EQ(out[2], 3550);
}

TEST(BoltStrTemp, TimestampDiffSeconds) {
    int64_t a[2] = {1000, 500};
    int64_t b[2] = {400, 500};
    int64_t out[2] = {};
    bolt::kernels::timestamp_diff_seconds(a, b, 2, out);
    EXPECT_EQ(out[0], 600);
    EXPECT_EQ(out[1], 0);
}

TEST(BoltStrTemp, TimestampExtractYearEpoch) {
    int64_t ts[3] = {0, 946684800 /* 2000-01-01 */, 1704067200 /* 2024-01-01 */};
    int32_t out[3] = {};
    bolt::kernels::timestamp_extract_year(ts, 3, out);
    EXPECT_EQ(out[0], 1970);
    EXPECT_EQ(out[1], 2000);
    EXPECT_EQ(out[2], 2024);
}

TEST(BoltStrTemp, TimestampExtractMonthDay) {
    // 2000-03-15 00:00:00 UTC = 953078400 (2000-03-03 is 952041600; +12 days).
    int64_t ts[1] = {953078400};
    int32_t m[1] = {};
    int32_t d[1] = {};
    bolt::kernels::timestamp_extract_month(ts, 1, m);
    bolt::kernels::timestamp_extract_day(ts, 1, d);
    EXPECT_EQ(m[0], 3);
    EXPECT_EQ(d[0], 15);
}

TEST(BoltStrTemp, TimestampDateTruncDayIdempotent) {
    int64_t ts[3] = {0, 86400, 86400 + 5};
    int64_t out1[3] = {};
    int64_t out2[3] = {};
    bolt::kernels::timestamp_date_trunc_day(ts, 3, out1);
    bolt::kernels::timestamp_date_trunc_day(out1, 3, out2);
    EXPECT_EQ(out1[0], 0);
    EXPECT_EQ(out1[1], 86400);
    EXPECT_EQ(out1[2], 86400);
    // Idempotent at midnight.
    EXPECT_EQ(out2[0], out1[0]);
    EXPECT_EQ(out2[1], out1[1]);
    EXPECT_EQ(out2[2], out1[2]);
}

TEST(BoltStrTemp, TimestampDateTruncDayNegative) {
    // -1 second → -86400 (start of prev day)
    int64_t ts[1] = {-1};
    int64_t out[1] = {};
    bolt::kernels::timestamp_date_trunc_day(ts, 1, out);
    EXPECT_EQ(out[0], -86400);
}

}  // namespace
