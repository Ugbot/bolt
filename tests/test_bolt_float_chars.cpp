// test_bolt_float_chars.cpp — bolt_float_chars.h / bolt_float_parse.h.
// Fixed expectations everywhere; where the host's <charconv> has the
// floating-point overloads, a differential sweep against them as the oracle.

#include "bolt/kernels/bolt_float_chars.h"
#include "bolt/kernels/bolt_float_parse.h"

#include <gtest/gtest.h>

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>

#if defined(__APPLE__)
#if defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__) && \
    __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ >= 260000
#define BOLT_FC_ORACLE 1
#endif
#elif defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
#define BOLT_FC_ORACLE 1
#endif

namespace fc = bolt::kernels::float_chars;

namespace {

std::string fmt(double v) {
    char buf[64];
    const uint32_t n = fc::f64_to_chars(buf, sizeof(buf), v);
    EXPECT_GT(n, 0u);
    return std::string(buf, n);
}

double from_bits(uint64_t b) {
    double d = 0.0;
    std::memcpy(&d, &b, sizeof(d));
    return d;
}

uint64_t to_bits(double d) {
    uint64_t b = 0;
    std::memcpy(&b, &d, sizeof(b));
    return b;
}

fc::ParseStatus parse(const std::string& s, double* v) {
    const fc::ParseResult r = fc::f64_from_chars(s.data(), s.data() + s.size(), v);
    if (r.status == fc::ParseStatus::ok) EXPECT_EQ(r.ptr, s.data() + s.size()) << s;
    return r.status;
}

}  // namespace

TEST(BoltFloatChars, FormatLayout) {
    EXPECT_EQ(fmt(0.0), "0");
    EXPECT_EQ(fmt(-0.0), "-0");
    EXPECT_EQ(fmt(1.25), "1.25");
    EXPECT_EQ(fmt(-3.5), "-3.5");
    EXPECT_EQ(fmt(0.1), "0.1");
    EXPECT_EQ(fmt(0.001), "0.001");
    EXPECT_EQ(fmt(0.0001), "1e-04");
    EXPECT_EQ(fmt(0.00012), "0.00012");
    EXPECT_EQ(fmt(10000.0), "10000");
    EXPECT_EQ(fmt(100000.0), "1e+05");
    EXPECT_EQ(fmt(123456789012.0), "123456789012");
    EXPECT_EQ(fmt(1e100), "1e+100");
    EXPECT_EQ(fmt(1.7976931348623157e308), "1.7976931348623157e+308");
    EXPECT_EQ(fmt(5e-324), "5e-324");
    EXPECT_EQ(fmt(2.2250738585072014e-308), "2.2250738585072014e-308");
    EXPECT_EQ(fmt(0.30000000000000004), "0.30000000000000004");
    EXPECT_EQ(fmt(9007199254740993.0), "9007199254740992");
    EXPECT_EQ(fmt(std::numeric_limits<double>::infinity()), "inf");
    EXPECT_EQ(fmt(-std::numeric_limits<double>::infinity()), "-inf");
    char small[3];
    EXPECT_EQ(fc::f64_to_chars(small, sizeof(small), 1.25), 0u);
}

TEST(BoltFloatChars, ParseGrammarAndRounding) {
    double v = -1.0;
    EXPECT_EQ(parse("2.5", &v), fc::ParseStatus::ok);  EXPECT_EQ(v, 2.5);
    EXPECT_EQ(parse(".5", &v), fc::ParseStatus::ok);   EXPECT_EQ(v, 0.5);
    EXPECT_EQ(parse("5.", &v), fc::ParseStatus::ok);   EXPECT_EQ(v, 5.0);
    EXPECT_EQ(parse("-1E3", &v), fc::ParseStatus::ok); EXPECT_EQ(v, -1000.0);
    EXPECT_EQ(parse("0.1", &v), fc::ParseStatus::ok);  EXPECT_EQ(v, 0.1);
    EXPECT_EQ(parse("9007199254740993", &v), fc::ParseStatus::ok);
    EXPECT_EQ(v, 9007199254740992.0);
    EXPECT_EQ(parse("9007199254740995", &v), fc::ParseStatus::ok);
    EXPECT_EQ(v, 9007199254740996.0);
    EXPECT_EQ(parse("4.9e-324", &v), fc::ParseStatus::ok);
    EXPECT_EQ(to_bits(v), 1u);
    EXPECT_EQ(parse("2.4703282292062328e-324", &v), fc::ParseStatus::ok);
    EXPECT_EQ(to_bits(v), 1u);
    EXPECT_EQ(parse("1.7976931348623157e308", &v), fc::ParseStatus::ok);
    EXPECT_EQ(v, std::numeric_limits<double>::max());
    EXPECT_EQ(parse("1e309", &v), fc::ParseStatus::out_of_range);
    EXPECT_EQ(parse("1e-400", &v), fc::ParseStatus::out_of_range);
    EXPECT_EQ(parse("0e999999", &v), fc::ParseStatus::ok);  EXPECT_EQ(v, 0.0);
    EXPECT_EQ(parse("Infinity", &v), fc::ParseStatus::ok);
    EXPECT_TRUE(std::isinf(v));
    EXPECT_EQ(parse("-nan(x_1)", &v), fc::ParseStatus::ok);
    EXPECT_TRUE(std::isnan(v) && std::signbit(v));
    for (const char* bad : {"", ".", "-", "+1", " 1", "e5", "x"}) {
        EXPECT_EQ(parse(bad, &v), fc::ParseStatus::invalid) << bad;
    }
    const std::string partial = "1e+";
    const fc::ParseResult r =
        fc::f64_from_chars(partial.data(), partial.data() + 3, &v);
    EXPECT_EQ(r.status, fc::ParseStatus::ok);
    EXPECT_EQ(r.ptr, partial.data() + 1);
    std::string longd = "1." + std::string(900, '0') + "1";
    EXPECT_EQ(parse(longd, &v), fc::ParseStatus::ok);
    EXPECT_EQ(v, 1.0);
}

TEST(BoltFloatChars, RoundTripRandomBits) {
    std::mt19937_64 rng(0x6a4c5u);
    for (int i = 0; i < 300000; ++i) {
        const double d = from_bits(rng());
        if (!std::isfinite(d)) continue;
        const std::string s = fmt(d);
        double back = 0.0;
        ASSERT_EQ(parse(s, &back), fc::ParseStatus::ok) << s;
        ASSERT_EQ(to_bits(back), to_bits(d)) << s;
    }
}

#if defined(BOLT_FC_ORACLE)
namespace {

void oracle_format(double d) {
    char want[64];
    const auto w = std::to_chars(want, want + sizeof(want), d);
    ASSERT_EQ(fmt(d), std::string(want, w.ptr)) << to_bits(d);
}

void oracle_parse(const std::string& s) {
    double want = 12345.0, got = 12345.0;
    const auto w = std::from_chars(s.data(), s.data() + s.size(), want);
    const fc::ParseResult g = fc::f64_from_chars(s.data(), s.data() + s.size(), &got);
    const bool w_ok = w.ec == std::errc();
    ASSERT_EQ(w_ok, g.status == fc::ParseStatus::ok) << s;
    ASSERT_EQ(w.ec == std::errc::result_out_of_range,
              g.status == fc::ParseStatus::out_of_range) << s;
    if (w.ec != std::errc::invalid_argument) ASSERT_EQ(w.ptr, g.ptr) << s;
    if (w_ok && !std::isnan(want)) ASSERT_EQ(to_bits(want), to_bits(got)) << s;
}

// Exact decimal (a + b) / 2 of two finite positive doubles, fixed-point.
std::string midpoint(double a, double b) {
    std::string sa(1300, '\0'), sb(1300, '\0');
    sa.resize(static_cast<size_t>(std::snprintf(sa.data(), sa.size(), "%.1100f", a)));
    sb.resize(static_cast<size_t>(std::snprintf(sb.data(), sb.size(), "%.1100f", b)));
    while (sa.size() < sb.size()) sa.insert(sa.begin(), '0');
    while (sb.size() < sa.size()) sb.insert(sb.begin(), '0');
    const size_t dot = sa.find('.');
    std::string digits;
    for (size_t i = 0; i < sa.size(); ++i) if (i != dot) digits.push_back('0');
    int carry = 0;
    for (size_t i = sa.size(), j = digits.size(); i > 0; --i) {
        if (i - 1 == dot) continue;
        const int t = (sa[i - 1] - '0') + (sb[i - 1] - '0') + carry;
        digits[--j] = static_cast<char>('0' + t % 10);
        carry = t / 10;
    }
    if (carry) digits.insert(digits.begin(), '1');
    digits.push_back('0');
    std::string half;
    int rem = 0;
    for (char c : digits) {
        const int t = rem * 10 + (c - '0');
        half.push_back(static_cast<char>('0' + t / 2));
        rem = t % 2;
    }
    const size_t int_len = dot + (carry ? 1u : 0u);
    half.insert(int_len, ".");
    return half;
}

}  // namespace

TEST(BoltFloatChars, OracleFormat) {
    std::mt19937_64 rng(0x5eedu);
    for (int i = 0; i < 1000000; ++i) {
        const double d = from_bits(rng());
        if (std::isfinite(d)) oracle_format(d);
    }
    for (uint64_t b = 0; b < 20000; ++b) oracle_format(from_bits(b));
    for (int e = 1; e < 2046; ++e) {
        const uint64_t base = static_cast<uint64_t>(e) << 52;
        for (uint64_t m : {uint64_t{0}, uint64_t{1}, (uint64_t{1} << 52) - 1u})
            oracle_format(from_bits(base | m));
    }
    for (int i = -400000; i < 400000; ++i) {
        oracle_format(i / 100.0);
        oracle_format(i * 0.001);
        oracle_format(static_cast<double>(i) * 1e10);
    }
    for (int e = -330; e <= 310; ++e) oracle_format(std::pow(10.0, e));
}

TEST(BoltFloatChars, OracleParse) {
    std::mt19937_64 rng(0xfeedu);
    char buf[64];
    for (int i = 0; i < 300000; ++i) {
        const double d = from_bits(rng());
        if (!std::isfinite(d)) continue;
        const int prec = static_cast<int>(rng() % 25u);
        std::snprintf(buf, sizeof(buf), "%.*e", prec, d);
        oracle_parse(buf);
    }
    std::uniform_real_distribution<double> mag(-6.0, 15.0);
    for (int i = 0; i < 20000; ++i) {
        const double d = std::pow(10.0, mag(rng));
        const std::string m = midpoint(d, from_bits(to_bits(d) + 1u));
        oracle_parse(m);
        std::string lo = m, hi = m;
        lo.push_back('0');
        hi.push_back('1');
        oracle_parse(hi);
        while (lo.back() == '0' || lo.back() == '5') lo.pop_back();
        oracle_parse(lo + "4999");
    }
    for (int i = 0; i < 200000; ++i) {
        std::string s;
        const int n = 1 + static_cast<int>(rng() % 30u);
        for (int j = 0; j < n; ++j) s.push_back("0123456789.eE-+x"[rng() % 16u]);
        oracle_parse(s);
    }
    for (const char* s : {"1e-400", "1e400", "2.4703282292062327e-324",
                          "2.4703282292062328e-324", "4.9406564584124654e-324",
                          "1.7976931348623158e308", "1.7976931348623159e308",
                          "inf", "INFINITY", "infin", "nan()", "nan(", "NaN(ab)"}) {
        oracle_parse(s);
    }
}
#endif
