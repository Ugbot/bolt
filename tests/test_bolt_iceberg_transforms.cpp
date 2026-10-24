// G2ICE-292 — Iceberg partition transform value kernels against the spec's own vectors (Appendix B: 32-bit hash
// requirements; "Truncate Transform Details" examples) and hand-derived calendar values for the time transforms.

#include "bolt/lakehouse/iceberg/transform.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

namespace {

using namespace bolt::lakehouse::iceberg;

constexpr int64_t kTs = INT64_C(1510871468000000);   // 2017-11-16T22:31:08 (micros)

TEST(IcebergHash, AppendixBVectors) {
    EXPECT_EQ(hash_long(34), 2017239379);                                  // int (promoted) and long
    EXPECT_EQ(hash_long(static_cast<int32_t>(34)), 2017239379);
    EXPECT_EQ(hash_decimal(0, 1420u), -500754589);                         // 14.20, unscaled 1420
    EXPECT_EQ(hash_long(17486), -653330422);                               // date 2017-11-16
    EXPECT_EQ(hash_long(INT64_C(81068000000)), -662762989);                // time 22:31:08
    EXPECT_EQ(hash_long(kTs), -2047944441);                                // timestamp / timestamptz
    EXPECT_EQ(hash_long(kTs + 1), -1207196810);
    EXPECT_EQ(hash_long(micros_from_nanos(kTs * 1000)), -2047944441);      // timestamp_ns / timestamptz_ns
    EXPECT_EQ(hash_long(micros_from_nanos(kTs * 1000 + 1001)), -1207196810);
    EXPECT_EQ(hash_bytes("iceberg", 7), 1210000089);                      // string
    const uint8_t uuid[16] = {0xF7, 0x9C, 0x3E, 0x09, 0x67, 0x7C, 0x4B, 0xBD,
                              0xA4, 0x79, 0x3F, 0x34, 0x9C, 0xB7, 0x85, 0xE7};
    EXPECT_EQ(hash_bytes(uuid, 16), 1488055340);                           // uuid f79c3e09-677c-4bbd-a479-3f349cb785e7
    const uint8_t fixed[4] = {0, 1, 2, 3};
    EXPECT_EQ(hash_bytes(fixed, 4), -188683207);                           // fixed(4) and binary
    // Not bucketable today, but the spec fixes their hash.
    EXPECT_EQ(hash_bool(true), 1392991556);
    EXPECT_EQ(hash_double(static_cast<double>(1.0f)), -142385009);
    EXPECT_EQ(hash_double(0.0), 1669671676);
    EXPECT_EQ(hash_double(-0.0), 1669671676);
    EXPECT_EQ(hash_double(1.0), -142385009);
    double nan_a = 0, nan_b = 0;
    const uint64_t qa = UINT64_C(0x7ff8000000000001), qb = UINT64_C(0xfff0000000000123);
    std::memcpy(&nan_a, &qa, 8); std::memcpy(&nan_b, &qb, 8);
    EXPECT_EQ(hash_double(nan_a), hash_double(nan_b));
    EXPECT_EQ(hash_double(nan_a), hash_long(static_cast<int64_t>(UINT64_C(0x7ff8000000000000))));
}

TEST(IcebergHash, BucketIsSignStrippedModN) {
    EXPECT_EQ(bucket_i32(34, 16), 2017239379 % 16);
    EXPECT_EQ(bucket_i64(34, 16), bucket_i32(34, 16));
    EXPECT_EQ(bucket_bytes("iceberg", 7, 16), 1210000089 % 16);
    EXPECT_EQ(bucket_str("iceberg", 7, 16), bucket_bytes("iceberg", 7, 16));
    // A negative hash drops its sign bit before the modulus (not abs, not a negative remainder).
    EXPECT_EQ(bucket_of_hash(-500754589, 100), static_cast<int32_t>((static_cast<uint32_t>(-500754589) & 0x7fffffffu) % 100u));
    EXPECT_EQ(bucket_decimal(0, 1420u, 7), bucket_of_hash(-500754589, 7));
    for (int32_t v = -1000; v < 1000; ++v) {
        EXPECT_GE(bucket_i32(v, 3), 0);
        EXPECT_LT(bucket_i32(v, 3), 3);
    }
}

TEST(IcebergHash, DecimalMinimalBigEndianBytes) {
    uint8_t b[16];
    ASSERT_EQ(decimal_min_be_bytes(0, 1420u, b), 2u);
    EXPECT_EQ(b[0], 0x05); EXPECT_EQ(b[1], 0x8C);
    ASSERT_EQ(decimal_min_be_bytes(-1, static_cast<uint64_t>(-1420), b), 2u);
    EXPECT_EQ(b[0], 0xFA); EXPECT_EQ(b[1], 0x74);
    ASSERT_EQ(decimal_min_be_bytes(0, 0u, b), 1u);   EXPECT_EQ(b[0], 0x00);
    ASSERT_EQ(decimal_min_be_bytes(-1, ~0ull, b), 1u); EXPECT_EQ(b[0], 0xFF);
    ASSERT_EQ(decimal_min_be_bytes(0, 128u, b), 2u); EXPECT_EQ(b[0], 0x00); EXPECT_EQ(b[1], 0x80);
    ASSERT_EQ(decimal_min_be_bytes(-1, static_cast<uint64_t>(-128), b), 1u); EXPECT_EQ(b[0], 0x80);
    ASSERT_EQ(decimal_min_be_bytes(1, 0u, b), 9u);   EXPECT_EQ(b[0], 0x01);   // 2^64
    ASSERT_EQ(decimal_min_be_bytes(INT64_MIN, 0u, b), 16u); EXPECT_EQ(b[0], 0x80);
}

TEST(IcebergTruncate, SpecExamples) {
    EXPECT_EQ(truncate_i32(1, 10), 0);
    EXPECT_EQ(truncate_i32(-1, 10), -10);
    EXPECT_EQ(truncate_i64(1, 10), 0);
    EXPECT_EQ(truncate_i64(-1, 10), -10);
    EXPECT_EQ(truncate_i64(INT64_C(-10), 10), -10);
    int32_t t32 = 0; int64_t t64 = 0;
    EXPECT_FALSE(truncate_i32_checked(INT32_MIN, 10, &t32));   // INT32_MIN - 2 is not an int
    EXPECT_TRUE(truncate_i32_checked(INT32_MIN + 8, 10, &t32)); EXPECT_EQ(t32, INT32_MIN + 8);
    EXPECT_TRUE(truncate_i32_checked(-1, 10, &t32)); EXPECT_EQ(t32, -10);
    EXPECT_FALSE(truncate_i64_checked(INT64_MIN, 10, &t64));
    EXPECT_TRUE(truncate_i64_checked(INT64_MAX, 10, &t64)); EXPECT_EQ(t64, INT64_MAX - 7);
    int64_t hi = 0; uint64_t lo = 0;
    truncate_decimal(0, 1065u, 50, &hi, &lo);                 // 10.65 -> 10.50 (W = 50 at scale 2)
    EXPECT_EQ(hi, 0); EXPECT_EQ(lo, 1050u);
    truncate_decimal(-1, static_cast<uint64_t>(-1065), 50, &hi, &lo);   // -10.65 -> -11.00
    EXPECT_EQ(hi, -1); EXPECT_EQ(static_cast<int64_t>(lo), -1100);
    truncate_decimal(-1, static_cast<uint64_t>(-1050), 50, &hi, &lo);   // on a boundary: unchanged
    EXPECT_EQ(static_cast<int64_t>(lo), -1050);
    truncate_decimal(1, 5u, 10, &hi, &lo);                    // 2^64 + 5 -> 2^64 + 4: 2^64 mod 10 = 6, so the value mod 10 is 1
    EXPECT_EQ(hi, 1); EXPECT_EQ(lo, 4u);
    EXPECT_EQ(truncate_utf8_len("iceberg", 7, 3), 3u);        // "ice"
    EXPECT_EQ(truncate_utf8_len("ab", 2, 3), 2u);
    const char euro[] = "a\xE2\x82\xAC" "b";                  // a, U+20AC, b
    EXPECT_EQ(truncate_utf8_len(euro, 5, 2), 4u);
    EXPECT_EQ(truncate_utf8_len(euro, 5, 1), 1u);
    const char emoji[] = "\xF0\x9F\x98\x80" "x";              // U+1F600, x
    EXPECT_EQ(truncate_utf8_len(emoji, 5, 1), 4u);
    EXPECT_EQ(truncate_binary_len(5, 3), 3u);                 // \x01..\x05 -> \x01\x02\x03
    EXPECT_EQ(truncate_binary_len(2, 3), 2u);
}

TEST(IcebergTimeTransforms, DateTimestampAndNanos) {
    EXPECT_EQ(transform_year(kTs, TimeUnit::kMicros), 47);
    EXPECT_EQ(transform_month(kTs, TimeUnit::kMicros), 47 * 12 + 10);
    EXPECT_EQ(transform_day(kTs, TimeUnit::kMicros), 17486);
    EXPECT_EQ(transform_hour(kTs, TimeUnit::kMicros), 17486 * 24 + 22);
    EXPECT_EQ(transform_year(17486, TimeUnit::kDays), 47);
    EXPECT_EQ(transform_month(17486, TimeUnit::kDays), 574);
    EXPECT_EQ(transform_day(17486, TimeUnit::kDays), 17486);
    EXPECT_EQ(transform_hour(kTs * 1000 + 999, TimeUnit::kNanos), 17486 * 24 + 22);
    EXPECT_EQ(transform_day(kTs * 1000, TimeUnit::kNanos), 17486);
    // Before the epoch every transform floors: 1969-12-31T23:59:59.999999 is year -1, month -1, day -1, hour -1.
    EXPECT_EQ(transform_year(-1, TimeUnit::kMicros), -1);
    EXPECT_EQ(transform_month(-1, TimeUnit::kMicros), -1);
    EXPECT_EQ(transform_day(-1, TimeUnit::kMicros), -1);
    EXPECT_EQ(transform_hour(-1, TimeUnit::kMicros), -1);
    EXPECT_EQ(transform_hour(-1, TimeUnit::kNanos), -1);
    EXPECT_EQ(transform_year(-1, TimeUnit::kDays), -1);
    EXPECT_EQ(transform_month(-31, TimeUnit::kDays), -1);    // 1969-12-01
    EXPECT_EQ(transform_month(-32, TimeUnit::kDays), -2);    // 1969-11-30
    EXPECT_EQ(transform_year(-365, TimeUnit::kDays), -1);    // 1969-01-01
    EXPECT_EQ(transform_year(-366, TimeUnit::kDays), -2);    // 1968-12-31
    EXPECT_EQ(transform_month(59, TimeUnit::kDays), 2);      // 1970-03-01
    EXPECT_EQ(transform_month(58, TimeUnit::kDays), 1);      // 1970-02-28
    EXPECT_EQ(transform_year(11016, TimeUnit::kDays), 30);   // 2000-02-29 (leap day of a century leap year)
    EXPECT_EQ(transform_month(11016, TimeUnit::kDays), 30 * 12 + 1);
    EXPECT_EQ(transform_year(0, TimeUnit::kDays), 0);
}

TEST(IcebergTransformParse, RoundTripsEveryKind) {
    const char* names[] = {"identity", "year", "month", "day", "hour", "void", "bucket[16]", "truncate[4]"};
    for (const char* n : names) {
        const Transform t = transform_parse(n, static_cast<uint32_t>(std::strlen(n)));
        ASSERT_NE(t.kind, TransformKind::kUnknown) << n;
        char buf[kIcebergMaxTransformName];
        ASSERT_GT(transform_format(t, buf, sizeof(buf)), 0u);
        EXPECT_STREQ(buf, n);
    }
    EXPECT_EQ(transform_parse("zorder", 6).kind, TransformKind::kUnknown);
}

}  // namespace
