// ---------------------------------------------------------------------------
// bolt::parse unit tests.
// ---------------------------------------------------------------------------
#include "bolt/parse/bolt_ascii.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>

using bolt::parse::parse_int;
using bolt::parse::parse_int10th;
using bolt::parse::parse_int10th_word;

namespace {

int32_t ref_int10th(double v) {
    // Round-to-nearest in tenths.
    return static_cast<int32_t>(v < 0 ? v * 10 - 0.5 : v * 10 + 0.5);
}

uint64_t load_word(const char* s, int len) {
    uint64_t w = 0;
    std::memcpy(&w, s, static_cast<size_t>(len));
    return w;
}

}  // namespace

TEST(BoltParseInt10th, KnownValues) {
    struct Case { const char* s; int32_t expected; };
    const Case cases[] = {
        {"-99.9", -999},
        {"-12.3", -123},
        {"-1.0",  -10 },
        {"-0.1",  -1  },
        {"0.0",   0   },
        {"0.5",   5   },
        {"9.9",   99  },
        {"12.3",  123 },
        {"99.9",  999 },
    };
    for (const auto& c : cases) {
        const int len = static_cast<int>(std::strlen(c.s));
        EXPECT_EQ(parse_int10th(c.s, len), c.expected) << "input=" << c.s;
    }
}

TEST(BoltParseInt10th, RoundTripRandom) {
    std::mt19937 rng(0xC0FFEEu);
    std::uniform_int_distribution<int32_t> dist(-999, 999);
    char buf[8];
    constexpr int N = 1'000'000;
    for (int i = 0; i < N; ++i) {
        const int32_t v = dist(rng);
        const int32_t whole = v / 10;
        const int32_t frac = v < 0 ? -(v % 10) : (v % 10);
        int len;
        if (v < 0 && whole == 0) {
            // e.g. -0.1 — emit the sign explicitly since whole rounds to 0.
            len = std::snprintf(buf, sizeof(buf), "-0.%d", frac);
        } else {
            len = std::snprintf(buf, sizeof(buf), "%d.%d", whole, frac);
        }
        ASSERT_GE(len, 3);
        ASSERT_LE(len, 5);
        const int32_t got = parse_int10th(buf, len);
        ASSERT_EQ(got, v) << "buf=" << std::string(buf, static_cast<size_t>(len));
    }
}

TEST(BoltParseInt10th, WordVariant) {
    const char* s = "12.3";
    const int len = 4;
    const uint64_t w = load_word(s, len);
    EXPECT_EQ(parse_int10th_word(w, len), 123);

    const char* neg = "-12.3";
    const int nlen = 5;
    const uint64_t wn = load_word(neg, nlen);
    EXPECT_EQ(parse_int10th_word(wn, nlen), -123);

    const char* zero = "0.0";
    const int zlen = 3;
    const uint64_t wz = load_word(zero, zlen);
    EXPECT_EQ(parse_int10th_word(wz, zlen), 0);
}

TEST(BoltParseInt, Basic) {
    EXPECT_EQ(parse_int("0", 1), 0);
    EXPECT_EQ(parse_int("12345", 5), 12345);
    EXPECT_EQ(parse_int("-67890", 6), -67890);
    EXPECT_EQ(parse_int("+42", 3), 42);
}

TEST(BoltParseInt, LargeValues) {
    // 18 digits max.
    EXPECT_EQ(parse_int("123456789012345678", 18),
              INT64_C(123456789012345678));
    EXPECT_EQ(parse_int("-123456789012345678", 19),
              INT64_C(-123456789012345678));
    EXPECT_EQ(parse_int("999999999999999999", 18),
              INT64_C(999999999999999999));
}

TEST(BoltParseInt, CStyleSurface) {
    EXPECT_EQ(bolt_parse_int("42", 2), 42);
    EXPECT_EQ(bolt_parse_int10th("99.9", 4), 999);
    const uint64_t w = load_word("99.9", 4);
    EXPECT_EQ(bolt_parse_int10th_word(w, 4), 999);
}
