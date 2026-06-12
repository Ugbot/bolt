// Tests for the Perf Wave 2 kernel additions:
//
//   bolt_numeric.h — col-literal arithmetic (add_lit / sub_lit / lit_sub /
//     mul_lit), f64 mixed-compare filter (f64_filter_cmp_col[_selected]),
//     widen_i64_to_f64 / widen_i32_to_f64.
//   bolt_decimal.h — Decimal128 col-literal arithmetic (d128_add_lit /
//     d128_sub_lit / d128_lit_sub_col / d128_mul_lit), scale-aware col-col
//     compare-filter (d128_filter_cmp_col[_selected]), d128_to_f64 /
//     d64_to_f64 coercion.
//
// Every kernel is checked against a scalar oracle. The decimal filter
// oracle replicates chukonu filter_op.cpp dense_filter_decimal_col verbatim
// (per-element d128_rescale both sides to max(sa,sb), then d128_cmp), which
// is the semantic contract these kernels replace. Coverage includes negative
// values, INT64_MAX-adjacent lane patterns, sa!=sb in both directions, all
// six compare ops, empty/odd lengths, and sparse selection vectors.

#include "bolt/kernels/bolt_numeric.h"
#include "bolt/kernels/bolt_decimal.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

using namespace bolt::kernels;
namespace dec = bolt::kernels::decimal;

// ----------------------------------------------------------------------------
// Deterministic RNG (splitmix64) — reproducible inputs, no <random> overhead.
// ----------------------------------------------------------------------------
struct Rng {
    uint64_t state;
    explicit Rng(uint64_t seed) noexcept : state(seed) {}
    uint64_t next() noexcept {
        uint64_t z = (state += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    int64_t next_i64_small() noexcept {  // ±~1e6, scale-friendly
        return static_cast<int64_t>(next() % 2000001ull) - 1000000;
    }
};

dec::Decimal128 mk(int64_t hi, uint64_t lo) noexcept {
    dec::Decimal128 d;
    d.lo = static_cast<int64_t>(lo);
    d.hi = hi;
    return d;
}

bool d128_bits_eq(dec::Decimal128 a, dec::Decimal128 b) noexcept {
    return a.lo == b.lo && a.hi == b.hi;
}

// Edge-heavy value pool: zero, ±1, INT64_MAX-adjacent lo lanes, carry
// boundaries (lo = UINT64_MAX so +1 carries into hi), negative hi.
std::vector<dec::Decimal128> edge_values() {
    const uint64_t umax = std::numeric_limits<uint64_t>::max();
    const int64_t  imax = std::numeric_limits<int64_t>::max();
    const int64_t  imin = std::numeric_limits<int64_t>::min();
    return {
        mk(0, 0),
        mk(0, 1),
        mk(-1, umax),                       // -1
        mk(0, static_cast<uint64_t>(imax)), // INT64_MAX
        mk(0, static_cast<uint64_t>(imax) - 1),
        mk(0, static_cast<uint64_t>(imax) + 1),
        mk(-1, static_cast<uint64_t>(imin)),// INT64_MIN
        mk(0, umax),                        // 2^64-1 (positive, hi=0)
        mk(1, 0),                           // 2^64
        mk(1, umax),                        // carry-dense
        mk(-2, 5),                          // negative with small lo
        mk(42, 0xDEADBEEFCAFEBABEull),
        mk(-43, 0x0123456789ABCDEFull),
        mk(imax, umax),                     // near +2^127
        mk(imin, 0),                        // -2^127
    };
}

// Oracle keep: chukonu filter_op.cpp dec_keep, op 0..5 = eq,ne,lt,le,gt,ge.
bool oracle_keep(int op, int sign) noexcept {
    switch (op) {
        case 0: return sign == 0;
        case 1: return sign != 0;
        case 2: return sign <  0;
        case 3: return sign <= 0;
        case 4: return sign >  0;
        case 5: return sign >= 0;
        default: return false;
    }
}

// Oracle filter: verbatim dense_filter_decimal_col semantics.
int64_t oracle_filter_d128(const dec::Decimal128* a, uint8_t sa,
                           const dec::Decimal128* b, uint8_t sb,
                           int64_t n, int op, int32_t* out) {
    const uint8_t t = (sa > sb) ? sa : sb;
    int64_t kept = 0;
    for (int64_t i = 0; i < n; ++i) {
        const dec::Decimal128 x = dec::d128_rescale(a[i], sa, t);
        const dec::Decimal128 y = dec::d128_rescale(b[i], sb, t);
        if (oracle_keep(op, dec::d128_cmp(x, y))) out[kept++] = static_cast<int32_t>(i);
    }
    return kept;
}

// IEEE-direct keep (a NaN-aware sign collapse would wrongly keep eq/le/ge,
// so each predicate is evaluated literally). if-chain instead of a switch
// over FP compares: the switch shape ICEs MSVC 19.x under /O2 here.
bool f64_keep(int op, double x, double y) {
    if (op == 0) return x == y;
    if (op == 1) return x != y;
    if (op == 2) return x <  y;
    if (op == 3) return x <= y;
    if (op == 4) return x >  y;
    if (op == 5) return x >= y;
    return false;
}

int64_t oracle_filter_f64(const double* a, const double* b, int64_t n,
                          int op, int32_t* out) {
    int64_t kept = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (f64_keep(op, a[i], b[i])) out[kept++] = static_cast<int32_t>(i);
    }
    return kept;
}

// ----------------------------------------------------------------------------
// Numeric col-literal arithmetic
// ----------------------------------------------------------------------------
TEST(NumericLit, AddSubMulLitInt64) {
    const std::vector<int64_t> lens = {0, 1, 2, 7, 63, 1000};
    Rng rng(1);
    for (int64_t n : lens) {
        std::vector<int64_t> a(static_cast<size_t>(n));
        std::vector<int64_t> out(static_cast<size_t>(n), -777);
        for (auto& v : a) v = rng.next_i64_small();
        const int64_t lits[3] = {0, -12345, std::numeric_limits<int64_t>::max()};
        for (int64_t lit : lits) {
            add_lit<int64_t>(a.data(), n, lit, out.data());
            for (int64_t i = 0; i < n; ++i)
                EXPECT_EQ(out[static_cast<size_t>(i)],
                          static_cast<int64_t>(a[static_cast<size_t>(i)] + lit));
            sub_lit<int64_t>(a.data(), n, lit, out.data());
            for (int64_t i = 0; i < n; ++i)
                EXPECT_EQ(out[static_cast<size_t>(i)],
                          static_cast<int64_t>(a[static_cast<size_t>(i)] - lit));
            mul_lit<int64_t>(a.data(), n, lit, out.data());
            for (int64_t i = 0; i < n; ++i)
                EXPECT_EQ(out[static_cast<size_t>(i)],
                          static_cast<int64_t>(a[static_cast<size_t>(i)] * lit));
        }
    }
}

TEST(NumericLit, LitSubInt64) {
    Rng rng(2);
    const int64_t n = 257;
    std::vector<int64_t> a(static_cast<size_t>(n));
    std::vector<int64_t> out(static_cast<size_t>(n));
    for (auto& v : a) v = rng.next_i64_small();
    lit_sub<int64_t>(1000, a.data(), n, out.data());
    for (int64_t i = 0; i < n; ++i)
        EXPECT_EQ(out[static_cast<size_t>(i)], 1000 - a[static_cast<size_t>(i)]);
    lit_sub<int64_t>(-7, a.data(), n, out.data());
    for (int64_t i = 0; i < n; ++i)
        EXPECT_EQ(out[static_cast<size_t>(i)], -7 - a[static_cast<size_t>(i)]);
}

TEST(NumericLit, AddSubMulLitSubDouble) {
    Rng rng(3);
    const int64_t n = 1001;
    std::vector<double> a(static_cast<size_t>(n));
    std::vector<double> out(static_cast<size_t>(n));
    for (auto& v : a) v = static_cast<double>(rng.next_i64_small()) / 100.0;
    const double lit = 0.94;  // 1 - l_discount shape
    add_lit<double>(a.data(), n, lit, out.data());
    for (int64_t i = 0; i < n; ++i)
        EXPECT_EQ(out[static_cast<size_t>(i)], a[static_cast<size_t>(i)] + lit);
    sub_lit<double>(a.data(), n, lit, out.data());
    for (int64_t i = 0; i < n; ++i)
        EXPECT_EQ(out[static_cast<size_t>(i)], a[static_cast<size_t>(i)] - lit);
    mul_lit<double>(a.data(), n, lit, out.data());
    for (int64_t i = 0; i < n; ++i)
        EXPECT_EQ(out[static_cast<size_t>(i)], a[static_cast<size_t>(i)] * lit);
    lit_sub<double>(1.0, a.data(), n, out.data());
    for (int64_t i = 0; i < n; ++i)
        EXPECT_EQ(out[static_cast<size_t>(i)], 1.0 - a[static_cast<size_t>(i)]);
}

// ----------------------------------------------------------------------------
// Decimal128 col-literal arithmetic vs d128_add/sub/mul oracles
// ----------------------------------------------------------------------------
TEST(DecimalLit, AddLitEdgeAndRandom) {
    auto edges = edge_values();
    Rng rng(11);
    std::vector<dec::Decimal128> a = edges;
    for (int i = 0; i < 500; ++i)
        a.push_back(mk(static_cast<int64_t>(rng.next() % 7) - 3, rng.next()));
    const int64_t n = static_cast<int64_t>(a.size());
    std::vector<dec::Decimal128> out(a.size());
    for (const auto& lit : edges) {
        dec::d128_add_lit(a.data(), n, lit, out.data());
        for (int64_t i = 0; i < n; ++i) {
            const dec::Decimal128 want = dec::d128_add(a[static_cast<size_t>(i)], lit);
            EXPECT_TRUE(d128_bits_eq(out[static_cast<size_t>(i)], want))
                << "i=" << i << " lit.hi=" << lit.hi;
        }
    }
    dec::d128_add_lit(a.data(), 0, edges[0], out.data());  // empty: no-op
}

TEST(DecimalLit, SubLitEdgeAndRandom) {
    auto edges = edge_values();
    Rng rng(12);
    std::vector<dec::Decimal128> a = edges;
    for (int i = 0; i < 500; ++i)
        a.push_back(mk(static_cast<int64_t>(rng.next() % 7) - 3, rng.next()));
    const int64_t n = static_cast<int64_t>(a.size());
    std::vector<dec::Decimal128> out(a.size());
    for (const auto& lit : edges) {
        dec::d128_sub_lit(a.data(), n, lit, out.data());
        for (int64_t i = 0; i < n; ++i) {
            const dec::Decimal128 want = dec::d128_sub(a[static_cast<size_t>(i)], lit);
            EXPECT_TRUE(d128_bits_eq(out[static_cast<size_t>(i)], want))
                << "i=" << i << " lit.hi=" << lit.hi;
        }
    }
}

TEST(DecimalLit, LitSubColEdgeAndRandom) {
    auto edges = edge_values();
    Rng rng(13);
    std::vector<dec::Decimal128> a = edges;
    for (int i = 0; i < 500; ++i)
        a.push_back(mk(static_cast<int64_t>(rng.next() % 7) - 3, rng.next()));
    const int64_t n = static_cast<int64_t>(a.size());
    std::vector<dec::Decimal128> out(a.size());
    for (const auto& lit : edges) {
        dec::d128_lit_sub_col(lit, a.data(), n, out.data());
        for (int64_t i = 0; i < n; ++i) {
            const dec::Decimal128 want = dec::d128_sub(lit, a[static_cast<size_t>(i)]);
            EXPECT_TRUE(d128_bits_eq(out[static_cast<size_t>(i)], want))
                << "i=" << i << " lit.hi=" << lit.hi;
        }
    }
}

TEST(DecimalLit, MulLitBitIdenticalToD128Mul) {
    auto edges = edge_values();
    Rng rng(14);
    std::vector<dec::Decimal128> a = edges;
    for (int i = 0; i < 500; ++i)
        a.push_back(mk(static_cast<int64_t>(rng.next()), rng.next()));
    const int64_t n = static_cast<int64_t>(a.size());
    std::vector<dec::Decimal128> out(a.size());
    for (const auto& lit : edges) {
        dec::d128_mul_lit(a.data(), n, lit, out.data());
        for (int64_t i = 0; i < n; ++i) {
            // d128_mul does sign-abs-negate; the lane kernel multiplies raw
            // two's-complement lanes. Low 128 bits of the product must be
            // bit-identical (mod-2^128 multiply is sign-agnostic).
            const dec::Decimal128 want = dec::d128_mul(a[static_cast<size_t>(i)], lit);
            EXPECT_TRUE(d128_bits_eq(out[static_cast<size_t>(i)], want))
                << "i=" << i << " lit=(" << lit.hi << "," << lit.lo << ")";
        }
    }
}

// ----------------------------------------------------------------------------
// Decimal128 scale-aware compare-filter
// ----------------------------------------------------------------------------
void run_filter_case(uint8_t sa, uint8_t sb, uint64_t seed) {
    Rng rng(seed);
    const int64_t n = 777;  // odd length on purpose
    std::vector<dec::Decimal128> a, b;
    a.reserve(static_cast<size_t>(n));
    b.reserve(static_cast<size_t>(n));
    auto edges = edge_values();
    for (int64_t i = 0; i < n; ++i) {
        if (i < static_cast<int64_t>(edges.size())) {
            a.push_back(edges[static_cast<size_t>(i)]);
            b.push_back(edges[edges.size() - 1 - static_cast<size_t>(i)]);
        } else {
            // small-magnitude values so rescale stays exact; force frequent
            // equality after rescale (v and v*10 at scales differing by 1).
            const int64_t v = rng.next_i64_small() % 1000;
            a.push_back(dec::d128_from_i64(v));
            const bool make_eq = (rng.next() & 3u) == 0;
            const int64_t bv = make_eq ? v : rng.next_i64_small() % 1000;
            // express b at scale sb of the same logical value when make_eq
            const int delta = static_cast<int>(sb) - static_cast<int>(sa);
            int64_t scaled = bv;
            for (int d = 0; d < delta; ++d) scaled *= 10;
            for (int d = 0; d > delta; --d) scaled /= 10;
            b.push_back(dec::d128_from_i64(make_eq ? scaled : bv));
        }
    }
    std::vector<int32_t> got(static_cast<size_t>(n)), want(static_cast<size_t>(n));
    for (int op = 0; op <= 5; ++op) {
        const int64_t kw = oracle_filter_d128(a.data(), sa, b.data(), sb, n, op,
                                              want.data());
        const int64_t kg = dec::d128_filter_cmp_col(a.data(), sa, b.data(), sb,
                                                    n, op, got.data());
        ASSERT_EQ(kg, kw) << "op=" << op << " sa=" << int(sa) << " sb=" << int(sb);
        for (int64_t i = 0; i < kw; ++i)
            EXPECT_EQ(got[static_cast<size_t>(i)], want[static_cast<size_t>(i)])
                << "op=" << op << " i=" << i;
    }
}

TEST(DecimalFilter, SameScaleAllOps)   { run_filter_case(2, 2, 21); }
TEST(DecimalFilter, ScaleUpLhsAllOps)  { run_filter_case(0, 2, 22); }  // sa<sb
TEST(DecimalFilter, ScaleUpRhsAllOps)  { run_filter_case(3, 1, 23); }  // sa>sb
TEST(DecimalFilter, ZeroScaleVsLarge)  { run_filter_case(0, 6, 24); }

TEST(DecimalFilter, EmptyAndTiny) {
    dec::Decimal128 a1 = dec::d128_from_i64(5);
    dec::Decimal128 b1 = dec::d128_from_i64(50);
    int32_t out[4] = {-1, -1, -1, -1};
    EXPECT_EQ(dec::d128_filter_cmp_col(nullptr, 0, nullptr, 0, 0, 4, out), 0);
    // 5 (scale 0) == 50 (scale 1) after rescale → eq keeps row 0.
    EXPECT_EQ(dec::d128_filter_cmp_col(&a1, 0, &b1, 1, 1, 0, out), 1);
    EXPECT_EQ(out[0], 0);
    // gt: 5.0 > 5.0 false.
    EXPECT_EQ(dec::d128_filter_cmp_col(&a1, 0, &b1, 1, 1, 4, out), 0);
}

TEST(DecimalFilter, SelectedSparseAllOps) {
    Rng rng(31);
    const int64_t n = 500;
    std::vector<dec::Decimal128> a, b;
    for (int64_t i = 0; i < n; ++i) {
        a.push_back(dec::d128_from_i64(rng.next_i64_small() % 200));
        b.push_back(dec::d128_from_i64(rng.next_i64_small() % 200));
    }
    // Sparse sel: every 3rd row, then a clustered run, then stragglers.
    std::vector<int32_t> sel;
    for (int32_t i = 0; i < n; i += 3) sel.push_back(i);
    sel.push_back(498); sel.push_back(499);
    const int64_t sel_n = static_cast<int64_t>(sel.size());
    std::vector<int32_t> got(sel.size()), want(sel.size());
    const uint8_t scales[3][2] = {{2, 2}, {1, 3}, {4, 0}};
    for (const auto& sc : scales) {
        for (int op = 0; op <= 5; ++op) {
            // Oracle: walk the sel, keep rows passing.
            const uint8_t t = (sc[0] > sc[1]) ? sc[0] : sc[1];
            int64_t kw = 0;
            for (int64_t i = 0; i < sel_n; ++i) {
                const int32_t r = sel[static_cast<size_t>(i)];
                const dec::Decimal128 x =
                    dec::d128_rescale(a[static_cast<size_t>(r)], sc[0], t);
                const dec::Decimal128 y =
                    dec::d128_rescale(b[static_cast<size_t>(r)], sc[1], t);
                if (oracle_keep(op, dec::d128_cmp(x, y))) want[static_cast<size_t>(kw++)] = r;
            }
            const int64_t kg = dec::d128_filter_cmp_col_selected(
                a.data(), sc[0], b.data(), sc[1], sel.data(), sel_n, op,
                got.data());
            ASSERT_EQ(kg, kw) << "op=" << op << " sa=" << int(sc[0])
                              << " sb=" << int(sc[1]);
            for (int64_t i = 0; i < kw; ++i)
                EXPECT_EQ(got[static_cast<size_t>(i)], want[static_cast<size_t>(i)]);
        }
    }
    // Empty sel.
    EXPECT_EQ(dec::d128_filter_cmp_col_selected(a.data(), 2, b.data(), 2,
                                                nullptr, 0, 0, got.data()), 0);
}

// ----------------------------------------------------------------------------
// f64 mixed-compare filter + widen/coerce helpers
// ----------------------------------------------------------------------------
TEST(F64Filter, DenseAllOpsIncludingNaN) {
    Rng rng(41);
    const int64_t n = 333;
    std::vector<double> a(static_cast<size_t>(n)), b(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        a[static_cast<size_t>(i)] = static_cast<double>(rng.next_i64_small() % 50);
        b[static_cast<size_t>(i)] = static_cast<double>(rng.next_i64_small() % 50);
    }
    // Sprinkle NaN and exact ties.
    a[7] = std::numeric_limits<double>::quiet_NaN();
    b[8] = std::numeric_limits<double>::quiet_NaN();
    a[9] = b[9];
    a[10] = -0.0; b[10] = 0.0;  // -0 == +0
    std::vector<int32_t> got(static_cast<size_t>(n)), want(static_cast<size_t>(n));
    for (int op = 0; op <= 5; ++op) {
        const int64_t kw = oracle_filter_f64(a.data(), b.data(), n, op, want.data());
        const int64_t kg = f64_filter_cmp_col(a.data(), b.data(), n, op, got.data());
        ASSERT_EQ(kg, kw) << "op=" << op;
        for (int64_t i = 0; i < kw; ++i)
            EXPECT_EQ(got[static_cast<size_t>(i)], want[static_cast<size_t>(i)]);
    }
    EXPECT_EQ(f64_filter_cmp_col(nullptr, nullptr, 0, 0, got.data()), 0);
}

TEST(F64Filter, SelectedSparse) {
    Rng rng(42);
    const int64_t n = 400;
    std::vector<double> a(static_cast<size_t>(n)), b(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        a[static_cast<size_t>(i)] = static_cast<double>(rng.next_i64_small()) / 4.0;
        b[static_cast<size_t>(i)] = static_cast<double>(rng.next_i64_small()) / 4.0;
    }
    std::vector<int32_t> sel;
    for (int32_t i = 1; i < n; i += 7) sel.push_back(i);
    const int64_t sel_n = static_cast<int64_t>(sel.size());
    std::vector<int32_t> got(sel.size()), want(sel.size());
    for (int op = 0; op <= 5; ++op) {
        int64_t kw = 0;
        for (int64_t i = 0; i < sel_n; ++i) {
            const int32_t r = sel[static_cast<size_t>(i)];
            const double x = a[static_cast<size_t>(r)];
            const double y = b[static_cast<size_t>(r)];
            if (f64_keep(op, x, y)) want[static_cast<size_t>(kw++)] = r;
        }
        const int64_t kg = f64_filter_cmp_col_selected(
            a.data(), b.data(), sel.data(), sel_n, op, got.data());
        ASSERT_EQ(kg, kw) << "op=" << op;
        for (int64_t i = 0; i < kw; ++i)
            EXPECT_EQ(got[static_cast<size_t>(i)], want[static_cast<size_t>(i)]);
    }
}

TEST(Coerce, WidenI64AndI32) {
    const int64_t a64[5] = {0, -1, 42, std::numeric_limits<int64_t>::max(),
                            std::numeric_limits<int64_t>::min()};
    const int32_t a32[5] = {0, -1, 42, std::numeric_limits<int32_t>::max(),
                            std::numeric_limits<int32_t>::min()};
    double out[5];
    widen_i64_to_f64(a64, 5, out);
    for (int i = 0; i < 5; ++i) EXPECT_EQ(out[i], static_cast<double>(a64[i]));
    widen_i32_to_f64(a32, 5, out);
    for (int i = 0; i < 5; ++i) EXPECT_EQ(out[i], static_cast<double>(a32[i]));
    widen_i64_to_f64(nullptr, 0, nullptr);  // empty no-op
}

// numeric_as_double oracle for Decimal128 (chukonu filter_op.cpp).
double oracle_d128_to_f64(dec::Decimal128 d, uint8_t scale) {
    double p = 1.0;
    for (uint8_t s = 0; s < scale; ++s) p *= 10.0;
    const double v = static_cast<double>(d.hi) * 18446744073709551616.0
                   + static_cast<double>(static_cast<uint64_t>(d.lo));
    return (scale == 0) ? v : v / p;
}

TEST(Coerce, D128ToF64MatchesNumericAsDouble) {
    auto vals = edge_values();
    Rng rng(51);
    for (int i = 0; i < 200; ++i)
        vals.push_back(mk(static_cast<int64_t>(rng.next() % 9) - 4, rng.next()));
    const int64_t n = static_cast<int64_t>(vals.size());
    std::vector<double> out(vals.size());
    const uint8_t scales[4] = {0, 1, 2, 6};
    for (uint8_t sc : scales) {
        dec::d128_to_f64(vals.data(), sc, n, out.data());
        for (int64_t i = 0; i < n; ++i) {
            const double want = oracle_d128_to_f64(vals[static_cast<size_t>(i)], sc);
            EXPECT_EQ(out[static_cast<size_t>(i)], want)
                << "i=" << i << " scale=" << int(sc);
        }
    }
}

TEST(Coerce, D64ToF64) {
    const int64_t m[6] = {0, -1, 12345, -98765,
                          std::numeric_limits<int64_t>::max(),
                          std::numeric_limits<int64_t>::min()};
    double out[6];
    const uint8_t scales[3] = {0, 2, 4};
    for (uint8_t sc : scales) {
        double p = 1.0;
        for (uint8_t s = 0; s < sc; ++s) p *= 10.0;
        dec::d64_to_f64(m, sc, 6, out);
        for (int i = 0; i < 6; ++i)
            EXPECT_EQ(out[i], static_cast<double>(m[i]) / p) << "scale=" << int(sc);
    }
}

// End-to-end mixed-compare shape: Decimal128(scale 0) vs Float64 column, the
// exact TPC-H Q17 `l_quantity < 0.2*avg` combo — coerce then f64-filter must
// match chukonu's per-row numeric_as_double loop.
TEST(F64Filter, MixedD128VsF64EndToEnd) {
    Rng rng(61);
    const int64_t n = 300;
    std::vector<dec::Decimal128> q;
    std::vector<double> thr(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        q.push_back(dec::d128_from_i64(static_cast<int64_t>(rng.next() % 50)));
        thr[static_cast<size_t>(i)] = static_cast<double>(rng.next() % 500) / 10.0;
    }
    std::vector<double> qd(static_cast<size_t>(n));
    dec::d128_to_f64(q.data(), 0, n, qd.data());
    std::vector<int32_t> got(static_cast<size_t>(n)), want(static_cast<size_t>(n));
    int64_t kw = 0;
    for (int64_t i = 0; i < n; ++i) {
        const double a = oracle_d128_to_f64(q[static_cast<size_t>(i)], 0);
        if (a < thr[static_cast<size_t>(i)]) want[static_cast<size_t>(kw++)] =
            static_cast<int32_t>(i);
    }
    const int64_t kg = f64_filter_cmp_col(qd.data(), thr.data(), n, /*lt*/2,
                                          got.data());
    ASSERT_EQ(kg, kw);
    for (int64_t i = 0; i < kw; ++i)
        EXPECT_EQ(got[static_cast<size_t>(i)], want[static_cast<size_t>(i)]);
}

}  // namespace
