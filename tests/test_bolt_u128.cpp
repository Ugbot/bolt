// G2CHK-990: bolt_u128.h (portable 128-bit, no __int128 in kernels) against the compiler's 128-bit integer as an
// oracle where one exists, over edge and pseudo-random values; and the kernels ported onto it (Spark bround of an
// integer, CPython round, format_number of a decimal, the try_avg / try_sum fold cells).
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

#include "bolt/join/bolt_groupby.h"
#include "bolt/kernels/bolt_math_spark.h"
#include "bolt/kernels/bolt_pystr.h"
#include "bolt/kernels/bolt_text_numfmt.h"
#include "bolt/kernels/bolt_u128.h"

namespace u = bolt::kernels::u128;

#if defined(__SIZEOF_INT128__)
using O = unsigned __int128;
static O o(u::U128 x) { return (static_cast<O>(x.hi) << 64) | x.lo; }
static u::U128 w(O x) { return u::U128{static_cast<uint64_t>(x >> 64), static_cast<uint64_t>(x)}; }

TEST(U128, MatchesCompilerOracle) {
    uint64_t seed = 0x9E3779B97F4A7C15ull;
    auto next = [&] { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return seed; };
    const uint64_t edge[] = {0, 1, 2, 9, 10, 0xFFFFFFFFull, 0x100000000ull, (1ull << 63) - 1, 1ull << 63, ~0ull};
    for (int i = 0; i < 20000; ++i) {
        const uint64_t a = i < 100 ? edge[i % 10] : next(), b = i < 100 ? edge[i / 10] : next();
        EXPECT_EQ(o(u::mul_u64(a, b)), static_cast<O>(a) * b);
        const u::U128 x{next() >> (i % 64), next()}, y{next() >> (i % 61), next()};
        EXPECT_EQ(o(u::add(x, y)), o(x) + o(y));
        EXPECT_EQ(o(u::sub(x, y)), o(x) - o(y));
        EXPECT_EQ(u::less(x, y), o(x) < o(y));
        const uint32_t k = static_cast<uint32_t>(next() % 128);
        EXPECT_EQ(o(u::shr(x, k)), o(x) >> k);
        EXPECT_EQ(o(u::shl(x, k)), o(x) << k);
        EXPECT_EQ(o(u::bit(k)), static_cast<O>(1) << k);
        u::U128 p{};
        const uint64_t m = next() >> (i % 64);
        const bool fits = u::mul_small(x, m, &p);
        const O wide_hi = (static_cast<O>(x.hi) * m) + ((static_cast<O>(x.lo) * m) >> 64);
        EXPECT_EQ(fits, (wide_hi >> 64) == 0);
        if (fits) EXPECT_EQ(o(p), o(x) * m);
        uint64_t r = 0;
        const uint64_t d = (next() >> (i % 63)) | 1u;
        EXPECT_EQ(o(u::div_small(x, d, &r)), o(x) / d);
        EXPECT_EQ(r, static_cast<uint64_t>(o(x) % d));
    }
    u::U128 q{};
    EXPECT_TRUE(u::mul(u::pow10(19), u::pow10(19), &q));
    EXPECT_EQ(o(q), o(u::pow10(38)));
    EXPECT_FALSE(u::mul(u::pow10(20), u::pow10(19), &q));
    EXPECT_EQ(o(w(o(q))), o(q));
}
#endif

TEST(U128, PortedKernels) {
    namespace ms = bolt::kernels::math_spark;
    bool ovf = false;
    EXPECT_EQ(ms::bround_int(125, -1, 64, &ovf), 120);   // HALF_EVEN
    EXPECT_FALSE(ovf);
    EXPECT_EQ(ms::bround_int(-135, -1, 64, &ovf), -140);
    EXPECT_EQ(ms::bround_int(INT64_MAX, -2, 64, &ovf), static_cast<int64_t>(9223372036854775800ll));
    EXPECT_FALSE(ovf);
    ms::bround_int(INT64_MAX - 1, -1, 64, &ovf);   // ...806 -> ...810 past BIGINT
    EXPECT_TRUE(ovf);
    ms::bround_int(127, -1, 8, &ovf);
    EXPECT_TRUE(ovf);   // 130 is outside TINYINT
    EXPECT_DOUBLE_EQ(bolt::kernels::utf8::f64_round_python(2.675, 2), 2.67);
    EXPECT_DOUBLE_EQ(bolt::kernels::utf8::f64_round_python(0.125, 2), 0.12);
    EXPECT_DOUBLE_EQ(bolt::kernels::utf8::f64_round_python(1e-20, 22), 1e-20);
    char buf[64];
    const int64_t n = bolt::kernels::text::format_number_dec(-1, ~0ull - 123456u, 2, 1, buf, sizeof buf);   // -1234.57
    ASSERT_GT(n, 0);
    EXPECT_EQ(std::string(buf, static_cast<size_t>(n)), "-1,234.6");
    bolt::GbCell16 out{};
    ASSERT_TRUE(bolt::gb_fold_d128_avg(bolt::GbCell16{-7, -1}, 2, 0, 1, &out));   // -7 / 2 = -3.5
    EXPECT_EQ(out.a, -35);
    EXPECT_EQ(out.b, -1);
    ASSERT_TRUE(bolt::gb_fold_d128_avg(bolt::GbCell16{5, 0}, 3, 0, 4, &out));    // 1.66666... -> 1.6667
    EXPECT_EQ(out.a, 16667);
    int64_t cnt = 1;
    bolt::GbCell16 acc{static_cast<int64_t>(u::sub(u::pow10(38), u::U128{0, 2}).lo),
                       static_cast<int64_t>(u::sub(u::pow10(38), u::U128{0, 2}).hi)};   // 10^38 - 2
    bolt::gb_fold_combine(bolt::AggKind::TrySum, bolt::BoltType::Decimal128, &acc, &cnt, bolt::GbCell16{1, 0}, 1);
    EXPECT_EQ(cnt, 2);
    bolt::gb_fold_combine(bolt::AggKind::TrySum, bolt::BoltType::Decimal128, &acc, &cnt, bolt::GbCell16{1, 0}, 1);
    EXPECT_EQ(cnt, bolt::kGbFoldOverflow);
}
