// test_bolt_stats_fused.cpp — MSEG K7: fused copy + page stats vs a naive
// reference written from the spec in bolt_stats_fused.h.

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "bolt/kernels/bolt_stats_fused.h"

using namespace bolt;
using namespace bolt::stats;

namespace {

// ---- naive reference ---------------------------------------------------------

struct Val {  // one row decoded independently of the kernel's key tricks
    bool valid = false, nan = false;
    int64_t i = 0;      // signed kinds
    uint64_t u = 0;     // unsigned kinds
    double f = 0;       // floats
    uint64_t lo = 0; int64_t hi = 0;  // Decimal128
    uint8_t raw[16] = {};
};

bool ref_less(StatsKind k, const Val& a, const Val& b) {
    switch (k) {
        case StatsKind::kSigned: return a.i < b.i;
        case StatsKind::kUnsigned: return a.u < b.u;
        case StatsKind::kD128: return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
        default:
            if (a.f < b.f) return true;
            if (a.f > b.f) return false;
            return std::signbit(a.f) && !std::signbit(b.f);
    }
}

Val decode(BoltType t, const uint8_t* p, bool valid) {
    Val v;
    v.valid = valid;
    const size_t w = type_size(t);
    std::memcpy(v.raw, p, w);
    const StatsKind k = stats_kind(t);
    if (k == StatsKind::kSigned) {
        if (w == 1) v.i = static_cast<int8_t>(p[0]);
        else if (w == 2) { int16_t x; std::memcpy(&x, p, 2); v.i = x; }
        else if (w == 4) { int32_t x; std::memcpy(&x, p, 4); v.i = x; }
        else std::memcpy(&v.i, p, 8);
    } else if (k == StatsKind::kUnsigned) {
        std::memcpy(&v.u, p, w);
    } else if (k == StatsKind::kF32) {
        float x; std::memcpy(&x, p, 4); v.f = x; v.nan = std::isnan(x);
    } else if (k == StatsKind::kF64) {
        std::memcpy(&v.f, p, 8); v.nan = std::isnan(v.f);
    } else {
        std::memcpy(&v.lo, p, 8); std::memcpy(&v.hi, p + 8, 8);
    }
    return v;
}

void ref_slot(BoltType t, const Val& v, uint8_t* out) {
    std::memset(out, 0, 16);
    const StatsKind k = stats_kind(t);
    if (k == StatsKind::kSigned) {
        std::memcpy(out, &v.i, 8);
        const int64_t ext = v.i < 0 ? -1 : 0;
        std::memcpy(out + 8, &ext, 8);
    } else if (k == StatsKind::kUnsigned) {
        std::memcpy(out, &v.u, 8);
    } else {
        std::memcpy(out, v.raw, type_size(t));
    }
}

// 128-bit total from separately summed 32-bit halves.
void ref_int_sum(StatsKind k, const std::vector<Val>& vals, uint64_t* lo, int64_t* hi) {
    int64_t sh = 0; uint64_t uh = 0, sl = 0;
    for (const Val& v : vals) {
        if (!v.valid) continue;
        const uint64_t bits = k == StatsKind::kSigned ? static_cast<uint64_t>(v.i) : v.u;
        sl += bits & 0xFFFFFFFFull;
        if (k == StatsKind::kSigned) sh += v.i >> 32; else uh += bits >> 32;
    }
    if (k == StatsKind::kSigned) {
        const uint64_t a = static_cast<uint64_t>(sh) << 32;
        *lo = a + sl;
        *hi = (sh >> 32) + (*lo < a ? 1 : 0);
    } else {
        const uint64_t a = uh << 32;
        *lo = a + sl;
        *hi = static_cast<int64_t>((uh >> 32) + (*lo < a ? 1 : 0));
    }
}

FusedStats reference(BoltType t, const std::vector<Val>& vals) {
    const StatsKind k = stats_kind(t);
    FusedStats o;
    std::memset(&o, 0, sizeof(o));
    o.type = static_cast<uint8_t>(t);
    o.row_count = static_cast<uint32_t>(vals.size());
    const Val* mn = nullptr; const Val* mx = nullptr; const Val* prev = nullptr;
    bool asc = true, desc = true, cons = true;
    for (const Val& v : vals) {
        if (!v.valid) { ++o.null_count; continue; }
        if (v.nan) { ++o.nan_count; continue; }
        if (prev) {
            const bool fl = k == StatsKind::kF32 || k == StatsKind::kF64;
            if (fl ? v.f < prev->f : ref_less(k, v, *prev)) asc = false;
            if (fl ? v.f > prev->f : ref_less(k, *prev, v)) desc = false;
        }
        if (!mn || ref_less(k, v, *mn)) mn = &v;
        if (!mx || ref_less(k, *mx, v)) mx = &v;
        prev = &v;
    }
    const Val* first = nullptr;
    for (const Val& v : vals) {
        if (!v.valid || v.nan) continue;
        if (!first) first = &v;
        else if (std::memcmp(v.raw, first->raw, type_size(t)) != 0) cons = false;
    }
    if (mn) {
        o.flags |= kStatsExtremesValid;
        ref_slot(t, *mn, o.min);
        ref_slot(t, *mx, o.max);
    }
    if (o.nan_count == 0) {
        if (asc) o.flags |= kStatsSortedAsc;
        if (desc) o.flags |= kStatsSortedDesc;
        if (cons && first) o.flags |= kStatsConstant;
    } else {
        o.flags |= kStatsHasNan;
    }
    if (o.row_count > 0 && o.null_count == o.row_count) o.flags |= kStatsAllNull;
    if (k == StatsKind::kF32 || k == StatsKind::kF64) {
        double lane[8] = {};
        for (size_t r = 0; r < vals.size(); ++r)
            if (vals[r].valid && !vals[r].nan) lane[r % 8] += vals[r].f;
        const double s = ((lane[0] + lane[1]) + (lane[2] + lane[3])) +
                         ((lane[4] + lane[5]) + (lane[6] + lane[7]));
        std::memcpy(&o.sum, &s, 8);
        if (std::isfinite(s)) o.flags |= kStatsSumValid;
    } else if (k != StatsKind::kD128) {
        uint64_t lo; int64_t hi;
        ref_int_sum(k, vals, &lo, &hi);
        o.sum = static_cast<int64_t>(lo);
        o.sum_hi = hi;
        const bool fits = k == StatsKind::kSigned ? hi == (o.sum < 0 ? -1 : 0) : hi == 0;
        if (fits) o.flags |= kStatsSumValid;
    }
    return o;
}

// ---- generator ---------------------------------------------------------------

const BoltType kTypes[] = {
    BoltType::Int8, BoltType::Int16, BoltType::Int32, BoltType::Int64,
    BoltType::UInt8, BoltType::UInt16, BoltType::UInt32, BoltType::UInt64,
    BoltType::Float32, BoltType::Float64, BoltType::Date32, BoltType::Date64,
    BoltType::Timestamp, BoltType::Duration, BoltType::Decimal64, BoltType::Decimal128,
};

enum Pattern { kRandom, kSmall, kConstant, kAsc, kDesc, kAscBreak, kSpecials, kExtremes, kZeros, kNumPatterns };

uint64_t gen_bits(std::mt19937_64& g, Pattern p, size_t r, size_t n, uint64_t base) {
    switch (p) {
        case kRandom: return g();
        case kSmall: return g() % 5;
        case kConstant: return base;
        case kAsc: return base + r / 3;
        case kDesc: return base + (n - r) / 3;
        case kAscBreak: return base + r - ((r == n / 2 || r % 256 == 0) ? 2 : 0);
        case kExtremes: return (g() & 1) ? ~0ull >> (g() % 2) : (1ull << 63) + (g() % 3);
        default: return g();
    }
}

void write_value(BoltType t, Pattern p, std::mt19937_64& g, size_t r, size_t n,
                 uint64_t base, uint8_t* out) {
    const StatsKind k = stats_kind(t);
    const size_t w = type_size(t);
    if ((k == StatsKind::kF32 || k == StatsKind::kF64) &&
        (p != kRandom || g() % 2 == 0)) {
        static const double specials[] = {0.0, -0.0, 1.5, -1.5, INFINITY, -INFINITY,
                                          std::nan(""), -std::nan(""), 1e308, -1e308};
        double d;
        if (p == kSpecials) d = specials[g() % 10];
        else if (p == kZeros) d = specials[g() % 4];
        else if (p == kExtremes) d = (g() & 1) ? 1e308 : -1e308;
        else d = static_cast<double>(static_cast<int64_t>(gen_bits(g, p, r, n, base) % 2001) - 1000) * 0.25;
        if (k == StatsKind::kF32) {
            float f = static_cast<float>(d);
            if (p == kSpecials && g() % 7 == 0) { uint32_t b = 0x7FC00001u | (static_cast<uint32_t>(g()) & 0xFFFF); std::memcpy(&f, &b, 4); }
            std::memcpy(out, &f, 4);
        } else {
            std::memcpy(out, &d, 8);
        }
        return;
    }
    uint64_t v = gen_bits(g, p, r, n, base);
    if (p == kSpecials) v = g() % 3 == 0 ? (1ull << (8 * (w < 8 ? w : 8) - 1)) : g();
    std::memcpy(out, &v, w < 8 ? w : 8);
    if (w == 16) {
        uint64_t hi = (p == kRandom || p == kSpecials) ? g()
                    : (p == kExtremes ? (g() & 1 ? 0x7FFFFFFFFFFFFFFFull : 0x8000000000000000ull)
                                      : (static_cast<int64_t>(v) < 0 ? ~0ull : 0));
        std::memcpy(out + 8, &hi, 8);
    }
}

struct Case {
    BoltType t;
    std::vector<uint8_t> data, validity;
    uint64_t voff = 0;
    size_t n = 0;
};

Case make_case(BoltType t, std::mt19937_64& g) {
    static const size_t sizes[] = {0, 1, 2, 7, 8, 9, 63, 64, 255, 256, 257, 511, 1000, 4096, 4099};
    Case c;
    c.t = t;
    c.n = g() % 3 == 0 ? g() % 6000 : sizes[g() % (sizeof(sizes) / sizeof(sizes[0]))];
    const Pattern p = static_cast<Pattern>(g() % kNumPatterns);
    const size_t w = type_size(t);
    c.data.assign(c.n * w + 1, 0);
    const uint64_t base = g() % 1000;
    for (size_t r = 0; r < c.n; ++r) write_value(t, p, g, r, c.n, base, c.data.data() + r * w);
    const int vmode = static_cast<int>(g() % 5);  // 0 none, 1 all valid, 2 sparse, 3 half, 4 all null
    if (vmode != 0) {
        c.voff = g() % 13;
        c.validity.assign((c.n + c.voff + 7) / 8 + 1, 0);
        for (size_t r = 0; r < c.n; ++r) {
            bool on = vmode == 1 || (vmode == 2 && g() % 97 != 0) || (vmode == 3 && g() % 2);
            if (on) c.validity[(r + c.voff) >> 3] |= static_cast<uint8_t>(1u << ((r + c.voff) & 7));
        }
    }
    return c;
}

std::vector<Val> decode_all(const Case& c) {
    std::vector<Val> v(c.n);
    const size_t w = type_size(c.t);
    for (size_t r = 0; r < c.n; ++r) {
        const bool valid = c.validity.empty() ||
            ((c.validity[(r + c.voff) >> 3] >> ((r + c.voff) & 7)) & 1);
        v[r] = decode(c.t, c.data.data() + r * w, valid);
    }
    return v;
}

struct CountExt {
    uint64_t rows = 0, dense_rows = 0;
    template <class T>
    void on_rows(const T*, uint32_t n, uint64_t, bool dense) { rows += n; dense_rows += dense ? n : 0; }
};

// Runs the kernel over `c` in random chunks; checks the copy.
FusedStats run_kernel(const Case& c, std::mt19937_64& g, bool copy, CountExt* ext) {
    StatsFusedState st;
    EXPECT_EQ(stats_fused_init(&st, c.t), StatsStatus::kOk);
    const size_t w = type_size(c.t);
    std::vector<uint8_t> dst(c.n * w + 1, 0xAB);
    size_t i = 0;
    while (i < c.n) {
        const size_t left = c.n - i;
        size_t m = g() % 4 == 0 ? left : 1 + g() % (left < 700 ? left : 700);
        const uint8_t* val = c.validity.empty() ? nullptr : c.validity.data();
        EXPECT_EQ(stats_fused_update(&st, c.data.data() + i * w, val, c.voff + i,
                                     static_cast<uint32_t>(m), copy ? dst.data() + i * w : nullptr, ext),
                  StatsStatus::kOk);
        i += m;
    }
    if (copy) {
        for (size_t r = 0; r < c.n; ++r) {
            const bool valid = c.validity.empty() ||
                ((c.validity[(r + c.voff) >> 3] >> ((r + c.voff) & 7)) & 1);
            static const uint8_t zero[16] = {};
            EXPECT_EQ(std::memcmp(dst.data() + r * w, valid ? c.data.data() + r * w : zero, w), 0)
                << "copy row " << r;
        }
        EXPECT_EQ(dst[c.n * w], 0xAB) << "wrote past the end";
    }
    return stats_fused_finish(st);
}

::testing::AssertionResult same(const FusedStats& a, const FusedStats& b) {
    if (std::memcmp(&a, &b, sizeof(a)) == 0) return ::testing::AssertionSuccess();
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "flags %x vs %x, nulls %u/%u, nans %u/%u, rows %u/%u, sum %lld/%lld hi %lld/%lld, min/max eq %d/%d",
                  a.flags, b.flags, a.null_count, b.null_count, a.nan_count, b.nan_count,
                  a.row_count, b.row_count, (long long)a.sum, (long long)b.sum,
                  (long long)a.sum_hi, (long long)b.sum_hi,
                  std::memcmp(a.min, b.min, 16) == 0, std::memcmp(a.max, b.max, 16) == 0);
    return ::testing::AssertionFailure() << buf;
}

}  // namespace

TEST(StatsFused, DifferentialFuzzEveryType) {
    uint64_t dense[sizeof(kTypes) / sizeof(kTypes[0])] = {};
    for (uint64_t seed = 1; seed <= 20; ++seed) {
        std::mt19937_64 g(seed * 0x9E3779B97F4A7C15ull);
        for (size_t ti = 0; ti < sizeof(kTypes) / sizeof(kTypes[0]); ++ti) {
            const BoltType t = kTypes[ti];
            for (int rep = 0; rep < 40; ++rep) {
                const Case c = make_case(t, g);
                const FusedStats want = reference(t, decode_all(c));
                CountExt ext;
                const FusedStats got = run_kernel(c, g, rep % 3 != 0, &ext);
                ASSERT_TRUE(same(got, want)) << "seed " << seed << " type " << int(t)
                                             << " rep " << rep << " n " << c.n;
                ASSERT_EQ(ext.rows, t == BoltType::Decimal128 ? 0u : c.n);
                dense[ti] += ext.dense_rows;
            }
        }
    }
    for (size_t ti = 0; ti + 1 < sizeof(kTypes) / sizeof(kTypes[0]); ++ti)
        EXPECT_GT(dense[ti], 100000u) << "type " << int(kTypes[ti]) << " never took the dense path";
}

TEST(StatsFused, SignedZerosOrderAndSort) {
    std::vector<double> v(2048, 0.0);
    for (size_t r = 0; r < v.size(); ++r) v[r] = (r % 3 == 0) ? -0.0 : 0.0;
    FusedStats s;
    ASSERT_EQ(stats_fused_copy(BoltType::Float64, v.data(), nullptr, 0, 2048, nullptr, &s),
              StatsStatus::kOk);
    EXPECT_TRUE(std::signbit(*reinterpret_cast<const double*>(s.min)));
    EXPECT_FALSE(std::signbit(*reinterpret_cast<const double*>(s.max)));
    EXPECT_EQ(s.flags & (kStatsSortedAsc | kStatsSortedDesc | kStatsConstant),
              kStatsSortedAsc | kStatsSortedDesc);  // numerically flat, not bitwise constant
    std::vector<float> f(1024, 0.0f);
    ASSERT_EQ(stats_fused_copy(BoltType::Float32, f.data(), nullptr, 0, 1024, nullptr, &s),
              StatsStatus::kOk);
    EXPECT_FALSE(std::signbit(*reinterpret_cast<const float*>(s.min)));
    EXPECT_TRUE(s.flags & kStatsConstant);
}

TEST(StatsFused, MergeEqualsSinglePass) {
    for (uint64_t seed = 1; seed <= 20; ++seed) {
        std::mt19937_64 g(seed ^ 0xC0FFEEull);
        for (BoltType t : kTypes) {
            for (int rep = 0; rep < 20; ++rep) {
                const Case c = make_case(t, g);
                const std::vector<Val> vals = decode_all(c);
                const size_t cut = c.n ? g() % (c.n + 1) : 0;
                const FusedStats a = reference(t, std::vector<Val>(vals.begin(), vals.begin() + static_cast<std::ptrdiff_t>(cut)));
                const FusedStats b = reference(t, std::vector<Val>(vals.begin() + static_cast<std::ptrdiff_t>(cut), vals.end()));
                FusedStats m;
                ASSERT_EQ(stats_fused_merge(a, b, &m), StatsStatus::kOk);
                FusedStats want = reference(t, vals);
                const StatsKind k = stats_kind(t);
                if (k == StatsKind::kF32 || k == StatsKind::kF64) {
                    double x, y; std::memcpy(&x, &a.sum, 8); std::memcpy(&y, &b.sum, 8);
                    const double s = x + y;
                    std::memcpy(&want.sum, &s, 8);
                    want.flags = static_cast<uint16_t>((want.flags & ~kStatsSumValid) |
                                                       (std::isfinite(s) ? kStatsSumValid : 0));
                }
                ASSERT_TRUE(same(m, want)) << "seed " << seed << " type " << int(t) << " cut " << cut;
            }
        }
    }
}

TEST(StatsFused, IntSumOverflowFlagsAndWraps) {
    const int64_t v[4] = {INT64_MAX, 1, INT64_MAX, -2};
    FusedStats s;
    ASSERT_EQ(stats_fused_copy(BoltType::Int64, v, nullptr, 0, 2, nullptr, &s), StatsStatus::kOk);
    EXPECT_FALSE(s.flags & kStatsSumValid);
    EXPECT_EQ(s.sum, INT64_MIN);  // wrapped low word
    EXPECT_EQ(s.sum_hi, 0);
    // back inside range: exact again
    ASSERT_EQ(stats_fused_copy(BoltType::Int64, v, nullptr, 0, 4, nullptr, &s), StatsStatus::kOk);
    EXPECT_FALSE(s.flags & kStatsSumValid);  // INT64_MAX*2 - 1 still overflows
    const int64_t w[3] = {INT64_MAX, 5, -10};
    ASSERT_EQ(stats_fused_copy(BoltType::Int64, w, nullptr, 0, 3, nullptr, &s), StatsStatus::kOk);
    EXPECT_TRUE(s.flags & kStatsSumValid);
    EXPECT_EQ(s.sum, INT64_MAX - 5);
    const uint64_t u[2] = {~0ull, 1};
    ASSERT_EQ(stats_fused_copy(BoltType::UInt64, u, nullptr, 0, 2, nullptr, &s), StatsStatus::kOk);
    EXPECT_FALSE(s.flags & kStatsSumValid);
    EXPECT_EQ(s.sum_hi, 1);
}

TEST(StatsFused, NanAllNullAndEmpty) {
    const double d[5] = {2.0, std::nan(""), -1.0, 3.0, 3.0};
    FusedStats s;
    ASSERT_EQ(stats_fused_copy(BoltType::Float64, d, nullptr, 0, 5, nullptr, &s), StatsStatus::kOk);
    EXPECT_EQ(s.nan_count, 1u);
    EXPECT_TRUE(s.flags & kStatsHasNan);
    EXPECT_FALSE(s.flags & (kStatsSortedAsc | kStatsSortedDesc | kStatsConstant));
    double mn, mx; std::memcpy(&mn, s.min, 8); std::memcpy(&mx, s.max, 8);
    EXPECT_EQ(mn, -1.0);
    EXPECT_EQ(mx, 3.0);
    ZoneMap z;
    stats_to_zonemap(s, &z);
    EXPECT_EQ(zone_min_f64(&z), -1.0);
    EXPECT_EQ(zone_max_f64(&z), 3.0);

    const uint8_t none[1] = {0};
    const int32_t i[3] = {7, 8, 9};
    ASSERT_EQ(stats_fused_copy(BoltType::Int32, i, none, 0, 3, nullptr, &s), StatsStatus::kOk);
    EXPECT_TRUE(s.flags & kStatsAllNull);
    EXPECT_FALSE(s.flags & (kStatsExtremesValid | kStatsConstant));
    EXPECT_EQ(s.null_count, 3u);
    stats_to_zonemap(s, &z);
    EXPECT_FALSE(zone_can_have_eq_i64(&z, 7));

    ASSERT_EQ(stats_fused_copy(BoltType::Int32, i, nullptr, 0, 0, nullptr, &s), StatsStatus::kOk);
    EXPECT_EQ(s.flags, kStatsSortedAsc | kStatsSortedDesc | kStatsSumValid);

    const uint8_t b[1] = {1};
    ASSERT_EQ(stats_fused_copy(BoltType::Bool, b, nullptr, 0, 1, nullptr, &s),
              StatsStatus::kUnsupportedType);
}

TEST(StatsFused, ZoneMapPrunesLikeTheValues) {
    std::vector<int64_t> v(1000);
    for (size_t r = 0; r < v.size(); ++r) v[r] = 100 + static_cast<int64_t>(r);
    FusedStats s;
    ASSERT_EQ(stats_fused_copy(BoltType::Int64, v.data(), nullptr, 0, 1000, nullptr, &s),
              StatsStatus::kOk);
    ZoneMap z;
    stats_to_zonemap(s, &z);
    EXPECT_TRUE(z.flags & kZoneFlagSortedAsc);
    EXPECT_TRUE(zone_can_have_eq_i64(&z, 100));
    EXPECT_TRUE(zone_can_have_eq_i64(&z, 1099));
    EXPECT_FALSE(zone_can_have_eq_i64(&z, 99));
    EXPECT_FALSE(zone_can_have_eq_i64(&z, 1100));
}

// Throughput at 1 thread (informational; the gate is recorded on the ticket).
TEST(StatsFused, ThroughputReport) {
    constexpr uint32_t n = 1u << 16;
    std::vector<int64_t> src(n), dst(n);
    std::mt19937_64 g(7);
    for (auto& x : src) x = static_cast<int64_t>(g() % 1000000);
    std::vector<double> fs(n), fd(n);
    for (auto& x : fs) x = static_cast<double>(g() % 1000000) * 0.5;
    constexpr int iters = 200;
    FusedStats s{};
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it)
        stats_fused_copy(BoltType::Int64, src.data(), nullptr, 0, n, dst.data(), &s);
    auto t1 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it)
        stats_fused_copy(BoltType::Float64, fs.data(), nullptr, 0, n, fd.data(), &s);
    auto t2 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) std::memcpy(dst.data(), src.data(), n * 8);
    auto t3 = std::chrono::steady_clock::now();
    const double rows = double(n) * iters;
    std::printf("stats_fused i64 %.3f ns/row  f64 %.3f ns/row  memcpy %.3f ns/row  (sink %lld)\n",
                std::chrono::duration<double, std::nano>(t1 - t0).count() / rows,
                std::chrono::duration<double, std::nano>(t2 - t1).count() / rows,
                std::chrono::duration<double, std::nano>(t3 - t2).count() / rows,
                static_cast<long long>(s.sum + dst[n / 2]));
    SUCCEED();
}
