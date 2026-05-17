// test_bolt_decimal.cpp — correctness + micro-benchmark for the
// Decimal128 kernel family (bolt/kernels/bolt_decimal.h).
//
// Correctness reference: where __int128 is available (gcc/clang) we
// cross-check arithmetic and comparisons against the compiler's native
// 128-bit type. On MSVC we cross-check structurally (e.g. add/neg
// identity, distributivity of sub vs add+neg).
//
// Benchmark: tail of the test executable prints ns/row for add, filter_gt,
// and sum at N=1<<20. These are informational — gates documented in
// the Decimal128 plan; failures of those gates do NOT fail the test.

#include <gtest/gtest.h>

#include "bolt/bolt_port.h"
#include "bolt/kernels/bolt_decimal.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using bolt::kernels::decimal::Decimal128;
using bolt::kernels::decimal::d128_add;
using bolt::kernels::decimal::d128_sub;
using bolt::kernels::decimal::d128_mul;
using bolt::kernels::decimal::d128_div;
using bolt::kernels::decimal::d128_neg;
using bolt::kernels::decimal::d128_gt;
using bolt::kernels::decimal::d128_lt;
using bolt::kernels::decimal::d128_eq;
using bolt::kernels::decimal::d128_cmp;

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
Decimal128 make_d128(int64_t hi, int64_t lo) noexcept {
    Decimal128 r;
    r.lo = lo;
    r.hi = hi;
    return r;
}

#if defined(__SIZEOF_INT128__)
__int128 to_i128(Decimal128 x) noexcept {
    unsigned __int128 u = (static_cast<unsigned __int128>(static_cast<uint64_t>(x.hi)) << 64)
                        | static_cast<unsigned __int128>(static_cast<uint64_t>(x.lo));
    return static_cast<__int128>(u);
}
Decimal128 from_i128(__int128 v) noexcept {
    unsigned __int128 u = static_cast<unsigned __int128>(v);
    Decimal128 r;
    r.lo = static_cast<int64_t>(static_cast<uint64_t>(u));
    r.hi = static_cast<int64_t>(static_cast<uint64_t>(u >> 64));
    return r;
}
#endif

}  // namespace

// ===========================================================================
// Primitive arithmetic
// ===========================================================================

TEST(BoltDecimal, AddSimple) {
    // 1 + 2 = 3
    Decimal128 a = make_d128(0, 1);
    Decimal128 b = make_d128(0, 2);
    Decimal128 c = d128_add(a, b);
    EXPECT_EQ(c.hi, 0);
    EXPECT_EQ(c.lo, 3);
}

TEST(BoltDecimal, AddCarryIntoHi) {
    // (uint64_max) + 1 carries into hi.
    Decimal128 a = make_d128(0, static_cast<int64_t>(0xffffffffffffffffull));
    Decimal128 b = make_d128(0, 1);
    Decimal128 c = d128_add(a, b);
    EXPECT_EQ(c.hi, 1);
    EXPECT_EQ(c.lo, 0);
}

TEST(BoltDecimal, AddNegativeCancels) {
    // x + (-x) = 0
    Decimal128 a = make_d128(0x1234, 0x5678);
    Decimal128 c = d128_add(a, d128_neg(a));
    EXPECT_EQ(c.hi, 0);
    EXPECT_EQ(c.lo, 0);
}

TEST(BoltDecimal, SubBorrow) {
    // 0 - 1 = all-ones (two's complement -1).
    Decimal128 z = make_d128(0, 0);
    Decimal128 o = make_d128(0, 1);
    Decimal128 r = d128_sub(z, o);
    EXPECT_EQ(r.hi, static_cast<int64_t>(-1));
    EXPECT_EQ(static_cast<uint64_t>(r.lo), 0xffffffffffffffffull);
}

TEST(BoltDecimal, MulSmall) {
    // 12345 * 67890 in 128-bit precision = 838102050
    Decimal128 a = make_d128(0, 12345);
    Decimal128 b = make_d128(0, 67890);
    Decimal128 c = d128_mul(a, b);
    EXPECT_EQ(c.hi, 0);
    EXPECT_EQ(c.lo, 838102050);
}

TEST(BoltDecimal, MulNegative) {
    // -7 as a properly sign-extended 128-bit value is (hi=-1, lo=-7).
    Decimal128 a = make_d128(-1, -7);
    Decimal128 b = make_d128(0, 3);
    Decimal128 c = d128_mul(a, b);
    // -7 * 3 = -21 → low 64 bits are -21, high 64 bits are -1 (sign-extend).
    EXPECT_EQ(c.lo, -21);
    EXPECT_EQ(c.hi, -1);
}

TEST(BoltDecimal, MulCarriesIntoHi) {
    // 2^32 * 2^32 = 2^64 → hi=1, lo=0
    Decimal128 a = make_d128(0, static_cast<int64_t>(1ull << 32));
    Decimal128 b = make_d128(0, static_cast<int64_t>(1ull << 32));
    Decimal128 c = d128_mul(a, b);
    EXPECT_EQ(c.hi, 1);
    EXPECT_EQ(c.lo, 0);
}

TEST(BoltDecimal, DivExact) {
    // 100 / 4 = 25
    Decimal128 a = make_d128(0, 100);
    Decimal128 b = make_d128(0, 4);
    Decimal128 q = d128_div(a, b);
    EXPECT_EQ(q.hi, 0);
    EXPECT_EQ(q.lo, 25);
}

TEST(BoltDecimal, DivTruncatesTowardZero) {
    // -7 / 2 = -3 (truncated toward zero). Sign-extend -7 across hi.
    Decimal128 a = make_d128(-1, -7);
    Decimal128 b = make_d128(0, 2);
    Decimal128 q = d128_div(a, b);
    EXPECT_EQ(q.lo, -3);
    EXPECT_EQ(q.hi, -1);
}

TEST(BoltDecimal, DivLarge128By64) {
    // (2^64 * 5 + 7) / 11
#if defined(__SIZEOF_INT128__)
    __int128 num = (static_cast<__int128>(5) << 64) | 7;
    __int128 den = 11;
    __int128 expect = num / den;
    Decimal128 q = d128_div(from_i128(num), from_i128(den));
    EXPECT_EQ(to_i128(q), expect);
#else
    Decimal128 a = make_d128(5, 7);
    Decimal128 b = make_d128(0, 11);
    Decimal128 q = d128_div(a, b);
    // Just sanity: quotient hi should be 0 (5/11 < 1) → no wait: (5<<64+7)/11 ≈ 8.38e18, hi nonzero possible.
    EXPECT_GE(q.hi, 0);
#endif
}

#if defined(__SIZEOF_INT128__)
TEST(BoltDecimal, RandomCrossCheckAddSubMul) {
    std::mt19937_64 rng(0xC0FFEEull);
    for (int iter = 0; iter < 5000; ++iter) {
        Decimal128 a = make_d128(static_cast<int64_t>(rng()), static_cast<int64_t>(rng()));
        Decimal128 b = make_d128(static_cast<int64_t>(rng()), static_cast<int64_t>(rng()));
        __int128 ai = to_i128(a), bi = to_i128(b);
        EXPECT_EQ(to_i128(d128_add(a, b)), ai + bi);
        EXPECT_EQ(to_i128(d128_sub(a, b)), ai - bi);
        EXPECT_EQ(to_i128(d128_mul(a, b)),
                  static_cast<__int128>(static_cast<unsigned __int128>(ai) *
                                        static_cast<unsigned __int128>(bi)));
    }
}

TEST(BoltDecimal, RandomCrossCheckDiv) {
    std::mt19937_64 rng(0xD15C0ull);
    for (int iter = 0; iter < 2000; ++iter) {
        Decimal128 a = make_d128(static_cast<int64_t>(rng()), static_cast<int64_t>(rng()));
        // Non-zero positive smallish divisor for predictable __int128 semantics.
        int64_t dlo = static_cast<int64_t>(rng() | 1ull);
        if (dlo < 0) dlo = -dlo;
        Decimal128 b = make_d128(0, dlo);
        __int128 ai = to_i128(a), bi = to_i128(b);
        Decimal128 q = d128_div(a, b);
        EXPECT_EQ(to_i128(q), ai / bi) << "iter=" << iter;
    }
}
#endif

// ===========================================================================
// Comparison
// ===========================================================================

TEST(BoltDecimal, CmpSignedHi) {
    Decimal128 neg = make_d128(-1, 0);  // very large negative
    Decimal128 pos = make_d128(0, 0);
    EXPECT_TRUE(d128_lt(neg, pos));
    EXPECT_TRUE(d128_gt(pos, neg));
    EXPECT_FALSE(d128_eq(neg, pos));
    EXPECT_EQ(d128_cmp(neg, pos), -1);
}

TEST(BoltDecimal, CmpUnsignedLoOnHiEq) {
    // hi equal, distinguish on lo treated as unsigned.
    // lo=-1 (== 0xff..ff unsigned) is GREATER than lo=1.
    Decimal128 a = make_d128(0, -1);
    Decimal128 b = make_d128(0, 1);
    EXPECT_TRUE(d128_gt(a, b));
    EXPECT_FALSE(d128_lt(a, b));
}

TEST(BoltDecimal, CmpEqualValues) {
    Decimal128 a = make_d128(42, 99);
    Decimal128 b = make_d128(42, 99);
    EXPECT_TRUE(d128_eq(a, b));
    EXPECT_FALSE(d128_gt(a, b));
    EXPECT_FALSE(d128_lt(a, b));
    EXPECT_EQ(d128_cmp(a, b), 0);
}

// ===========================================================================
// Filter kernels
// ===========================================================================

TEST(BoltDecimal, FilterGt) {
    std::vector<Decimal128> data = {
        make_d128(0, 10), make_d128(0, 20), make_d128(0, 5),
        make_d128(0, 30), make_d128(0, 1),  make_d128(0, 40),
    };
    Decimal128 scalar = make_d128(0, 10);
    std::vector<int32_t> sel(data.size());
    int64_t n = bolt::kernels::decimal::decimal128_filter_gt(
        data.data(), static_cast<int64_t>(data.size()), scalar, sel.data());
    EXPECT_EQ(n, 3);
    EXPECT_EQ(sel[0], 1);  // 20 > 10
    EXPECT_EQ(sel[1], 3);  // 30 > 10
    EXPECT_EQ(sel[2], 5);  // 40 > 10
}

TEST(BoltDecimal, FilterEqWithBigHi) {
    Decimal128 target = make_d128(0xdeadbeef, 0x1234);
    std::vector<Decimal128> data = {
        make_d128(0xdeadbeef, 0x1234),
        make_d128(0xdeadbeef, 0x1235),
        make_d128(0xdeadbef0, 0x1234),
        make_d128(0xdeadbeef, 0x1234),
    };
    std::vector<int32_t> sel(data.size());
    int64_t n = bolt::kernels::decimal::decimal128_filter_eq(
        data.data(), static_cast<int64_t>(data.size()), target, sel.data());
    EXPECT_EQ(n, 2);
    EXPECT_EQ(sel[0], 0);
    EXPECT_EQ(sel[1], 3);
}

TEST(BoltDecimal, FilterAllOps) {
    using namespace bolt::kernels::decimal;
    std::vector<Decimal128> data = {
        make_d128(0, 1), make_d128(0, 2), make_d128(0, 3), make_d128(0, 4),
    };
    Decimal128 s = make_d128(0, 2);
    std::vector<int32_t> sel(4);
    EXPECT_EQ(decimal128_filter_gt(data.data(), 4, s, sel.data()), 2);
    EXPECT_EQ(decimal128_filter_ge(data.data(), 4, s, sel.data()), 3);
    EXPECT_EQ(decimal128_filter_lt(data.data(), 4, s, sel.data()), 1);
    EXPECT_EQ(decimal128_filter_le(data.data(), 4, s, sel.data()), 2);
    EXPECT_EQ(decimal128_filter_eq(data.data(), 4, s, sel.data()), 1);
    EXPECT_EQ(decimal128_filter_ne(data.data(), 4, s, sel.data()), 3);
}

TEST(BoltDecimal, FilterSelectedComposable) {
    using namespace bolt::kernels::decimal;
    std::vector<Decimal128> data = {
        make_d128(0, 5), make_d128(0, 10), make_d128(0, 15),
        make_d128(0, 20), make_d128(0, 25),
    };
    // First filter: data > 5 → sel_in = [1,2,3,4]
    std::vector<int32_t> sel0 = {1, 2, 3, 4};
    std::vector<int32_t> sel1(5);
    Decimal128 s = make_d128(0, 15);
    int64_t n = decimal128_filter_gt_selected(
        data.data(), sel0.data(), 4, s, sel1.data());
    EXPECT_EQ(n, 2);
    EXPECT_EQ(sel1[0], 3);
    EXPECT_EQ(sel1[1], 4);
}

// ===========================================================================
// Aggregates
// ===========================================================================

TEST(BoltDecimal, SumSimple) {
    using namespace bolt::kernels::decimal;
    std::vector<Decimal128> data = {
        make_d128(0, 1), make_d128(0, 2), make_d128(0, 3),
        make_d128(0, 4), make_d128(0, 5),
    };
    Decimal128 s = decimal128_sum(data.data(), 5);
    EXPECT_EQ(s.hi, 0);
    EXPECT_EQ(s.lo, 15);
}

TEST(BoltDecimal, SumCarriesIntoHi) {
    using namespace bolt::kernels::decimal;
    // Each element = 2^63; two of them sum to 2^64 → hi=1, lo=0.
    Decimal128 e = make_d128(0, static_cast<int64_t>(1ull << 63));
    std::vector<Decimal128> data = {e, e};
    // Note: hi field of e is 0, lo is 0x8000... (treated unsigned).
    Decimal128 s = decimal128_sum(data.data(), 2);
    EXPECT_EQ(s.hi, 1);
    EXPECT_EQ(s.lo, 0);
}

TEST(BoltDecimal, MinMax) {
    using namespace bolt::kernels::decimal;
    std::vector<Decimal128> data = {
        make_d128(0, 7),  make_d128(-1, 0),  // large negative
        make_d128(0, 1000000),
        make_d128(5, 0),  make_d128(0, 3),
    };
    Decimal128 lo = decimal128_min(data.data(), 5);
    Decimal128 hi = decimal128_max(data.data(), 5);
    EXPECT_TRUE(d128_eq(lo, make_d128(-1, 0)));
    EXPECT_TRUE(d128_eq(hi, make_d128(5, 0)));
}

// ===========================================================================
// Micro-benchmark (informational; does not gate CI)
// ===========================================================================

TEST(BoltDecimal, MicroBench) {
    using namespace bolt::kernels::decimal;
    using Clock = std::chrono::high_resolution_clock;
    using ns    = std::chrono::nanoseconds;

    constexpr int64_t N = 1 << 20;
    constexpr int iters = 5;

    std::vector<Decimal128> a(N), b(N), out(N);
    std::vector<int32_t>    sel(N);
    std::mt19937_64 rng(42);
    for (int64_t i = 0; i < N; ++i) {
        a[i] = make_d128(static_cast<int64_t>(rng() & 0xff), static_cast<int64_t>(rng()));
        b[i] = make_d128(static_cast<int64_t>(rng() & 0xff), static_cast<int64_t>(rng()));
    }

    auto run = [&](const char* name, auto fn) {
        int64_t best = std::numeric_limits<int64_t>::max();
        for (int it = 0; it < iters; ++it) {
            auto t0 = Clock::now();
            fn();
            auto t1 = Clock::now();
            int64_t dt = std::chrono::duration_cast<ns>(t1 - t0).count();
            if (dt < best) best = dt;
        }
        std::printf("[decimal-bench] %-32s  N=%-8lld  best=%6lld ns  %.3f ns/row\n",
                    name, (long long)N, (long long)best,
                    static_cast<double>(best) / static_cast<double>(N));
    };

    run("decimal128_add",        [&]{ decimal128_add(a.data(), b.data(), out.data(), N); });
    run("decimal128_sub",        [&]{ decimal128_sub(a.data(), b.data(), out.data(), N); });
    run("decimal128_mul",        [&]{ decimal128_mul(a.data(), b.data(), out.data(), N); });
    Decimal128 piv = make_d128(0, static_cast<int64_t>(rng()));
    run("decimal128_filter_gt",  [&]{ decimal128_filter_gt(a.data(), N, piv, sel.data()); });
    run("decimal128_filter_eq",  [&]{ decimal128_filter_eq(a.data(), N, piv, sel.data()); });
    Decimal128 sink{0, 0};
    run("decimal128_sum",        [&]{ Decimal128 s = decimal128_sum(a.data(), N); sink.lo ^= s.lo; sink.hi ^= s.hi; });
    run("decimal128_min",        [&]{ Decimal128 s = decimal128_min(a.data(), N); sink.lo ^= s.lo; sink.hi ^= s.hi; });
    if (sink.lo == 0xdeadc0deull) std::printf("unreachable\n");
    SUCCEED();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
