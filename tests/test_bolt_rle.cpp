// test_bolt_rle.cpp — K3: ColumnFormat::RLE encode/decode + run-granular
// predicates, fuzzed against a naive scalar reference.

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/kernels/bolt_rle.h"

using namespace bolt;
namespace R = bolt::rle;

namespace {

constexpr int kSeeds = 24;
constexpr int64_t kMaxRunLen = 65536;

template <class T> bool same_bits(T a, T b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }

bool bit(const std::vector<uint8_t>& v, int64_t i) {
    return v.empty() || ((v[static_cast<size_t>(i >> 3)] >> (i & 7)) & 1u);
}

template <class T> T pick(std::mt19937_64& g, int domain) {
    const int k = static_cast<int>(g() % static_cast<uint64_t>(domain));
    if constexpr (std::is_floating_point_v<T>) {
        if (k == 0) return std::numeric_limits<T>::quiet_NaN();
        if (k == 1) return T(-0.0);
        return T(k) * T(0.5);
    } else {
        return static_cast<T>(k * 37 - 50);
    }
}

// Runs of log-uniform length 1..65,536; nulls by mode: 0 none, 1 sparse,
// 2 dense, 3 bursts, 4 all-null.
template <class T> struct Case {
    std::vector<T> v;
    std::vector<uint8_t> valid;  // empty = no bitmap
};

template <class T> Case<T> make_case(uint64_t seed, int64_t target_rows, int null_mode) {
    std::mt19937_64 g(seed);
    Case<T> c;
    const int domain = 2 + static_cast<int>(g() % 12);
    while (static_cast<int64_t>(c.v.size()) < target_rows) {
        const double e = std::ldexp(1.0, static_cast<int>(g() % 17));
        int64_t len = 1 + static_cast<int64_t>(g() % static_cast<uint64_t>(e));
        if (len > kMaxRunLen) len = kMaxRunLen;
        const T x = pick<T>(g, domain);
        c.v.insert(c.v.end(), static_cast<size_t>(len), x);
    }
    if (null_mode == 0) return c;
    const int64_t n = static_cast<int64_t>(c.v.size());
    c.valid.assign(static_cast<size_t>((n + 7) / 8), 0);
    bool in_burst = false;
    for (int64_t i = 0; i < n; ++i) {
        bool ok = true;
        if (null_mode == 1) ok = g() % 100 != 0;
        if (null_mode == 2) ok = g() % 2 == 0;
        if (null_mode == 3) { if (g() % 500 == 0) in_burst = !in_burst; ok = !in_burst; }
        if (null_mode == 4) ok = false;
        // Null slots hold junk so a run split on a null slot would show.
        if (!ok) c.v[static_cast<size_t>(i)] = pick<T>(g, 64);
        if (ok) c.valid[static_cast<size_t>(i >> 3)] |= uint8_t(1u << (i & 7));
    }
    return c;
}

// Naive reference: the number of maximal runs when nulls join their run.
template <class T> int64_t ref_runs(const Case<T>& c) {
    const int64_t n = static_cast<int64_t>(c.v.size());
    int64_t runs = n > 0 ? 1 : 0;
    bool have = false;
    T cur{};
    for (int64_t i = 0; i < n; ++i) {
        if (!bit(c.valid, i)) continue;
        if (have && !same_bits(cur, c.v[static_cast<size_t>(i)])) ++runs;
        cur = c.v[static_cast<size_t>(i)];
        have = true;
    }
    return runs;
}

template <class T> struct Enc {
    std::vector<T> vals;
    std::vector<int32_t> ends;
};

template <class T> Enc<T> encode_case(const Case<T>& c) {
    const int64_t n = static_cast<int64_t>(c.v.size());
    const uint8_t* vb = c.valid.empty() ? nullptr : c.valid.data();
    const int64_t want = R::count_runs(c.v.data(), vb, n);
    Enc<T> e;
    e.vals.resize(static_cast<size_t>(want));
    e.ends.resize(static_cast<size_t>(want));
    const int64_t got = R::encode(c.v.data(), vb, n, e.vals.data(), e.ends.data(), want);
    EXPECT_EQ(got, want);
    return e;
}

template <class T> void round_trip(uint64_t seed, int null_mode) {
    std::mt19937_64 g(seed * 7919 + static_cast<uint64_t>(null_mode));
    const Case<T> c = make_case<T>(seed, 1 + static_cast<int64_t>(g() % 300000), null_mode);
    const int64_t n = static_cast<int64_t>(c.v.size());
    const Enc<T> e = encode_case(c);
    const int64_t runs = static_cast<int64_t>(e.ends.size());
    ASSERT_EQ(runs, ref_runs(c)) << "seed " << seed;
    ASSERT_TRUE(R::run_ends_valid(e.ends.data(), runs, n));
    std::vector<T> out(static_cast<size_t>(n));
    ASSERT_TRUE(R::decode(e.vals.data(), e.ends.data(), runs, n, out.data()));
    int64_t r = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (i >= e.ends[static_cast<size_t>(r)]) ++r;
        ASSERT_TRUE(same_bits(out[static_cast<size_t>(i)], e.vals[static_cast<size_t>(r)]));
        if (bit(c.valid, i)) {
            ASSERT_TRUE(same_bits(out[static_cast<size_t>(i)], c.v[static_cast<size_t>(i)]))
                << "seed " << seed << " row " << i;
        }
    }
    // A too-small run budget refuses rather than truncating.
    if (runs > 1) {
        std::vector<T> sv(static_cast<size_t>(runs));
        std::vector<int32_t> se(static_cast<size_t>(runs));
        const uint8_t* vb = c.valid.empty() ? nullptr : c.valid.data();
        EXPECT_EQ(R::encode(c.v.data(), vb, n, sv.data(), se.data(), runs - 1), -1);
    }
}

}  // namespace

TEST(Rle, RoundTripFuzz) {
    for (int s = 1; s <= kSeeds; ++s) {
        for (int mode = 0; mode <= 4; ++mode) {
            round_trip<uint8_t>(static_cast<uint64_t>(s), mode);
            round_trip<int16_t>(static_cast<uint64_t>(s) + 100, mode);
            round_trip<int32_t>(static_cast<uint64_t>(s) + 200, mode);
            round_trip<int64_t>(static_cast<uint64_t>(s) + 300, mode);
            round_trip<double>(static_cast<uint64_t>(s) + 400, mode);
            round_trip<float>(static_cast<uint64_t>(s) + 500, mode);
        }
    }
}

TEST(Rle, NoNullsIsBitExactIncludingNaN) {
    const double nan2 = std::bit_cast<double>(uint64_t{0x7ff8000000000123ull});
    std::vector<double> v = {1.0, 1.0, -0.0, 0.0, nan2, nan2,
                             std::numeric_limits<double>::quiet_NaN(), 2.0};
    Case<double> c{v, {}};
    const Enc<double> e = encode_case(c);
    ASSERT_EQ(e.ends.size(), 6u);
    std::vector<double> out(v.size());
    ASSERT_TRUE(R::decode(e.vals.data(), e.ends.data(), 6, 8, out.data()));
    ASSERT_EQ(std::memcmp(out.data(), v.data(), v.size() * sizeof(double)), 0);
}

TEST(Rle, ColumnLayoutIsMakeRle) {
    Arena arena;
    const Case<int32_t> c = make_case<int32_t>(42, 50000, 0);
    const Enc<int32_t> e = encode_case(c);
    const int64_t n = static_cast<int64_t>(c.v.size());
    BoltColumn col = BoltColumn::make_rle(e.vals.data(), static_cast<int64_t>(e.ends.size()),
                                          e.ends.data(), n, BoltType::Int32, &arena);
    ASSERT_EQ(col.format, ColumnFormat::RLE);
    BoltColumn flat = col.materialize(&arena);
    ASSERT_NE(flat.data, nullptr);
    ASSERT_EQ(std::memcmp(flat.data, c.v.data(), c.v.size() * sizeof(int32_t)), 0);
}

TEST(Rle, EdgeShapes) {
    EXPECT_EQ(R::count_runs<int32_t>(nullptr, nullptr, 0), 0);
    EXPECT_TRUE(R::run_ends_valid(nullptr, 0, 0));
    EXPECT_TRUE(R::decode<int32_t>(nullptr, nullptr, 0, 0, nullptr));
    const uint8_t none = 0;
    int32_t v[3] = {5, 6, 7};
    int32_t ov[1] = {};
    int32_t oe[1] = {};
    ASSERT_EQ(R::encode(v, &none, 3, ov, oe, 1), 1);  // all null: one run
    EXPECT_EQ(oe[0], 3);
    EXPECT_EQ(ov[0], 0);
    int32_t bad_ends[2] = {3, 3};
    int32_t out[3];
    EXPECT_FALSE(R::run_ends_valid(bad_ends, 2, 3));
    EXPECT_FALSE(R::decode(v, bad_ends, 2, 3, out));
    int32_t short_ends[1] = {2};
    EXPECT_FALSE(R::decode(v, short_ends, 1, 3, out));
    int32_t ends3[3] = {1, 4, 9};
    EXPECT_EQ(R::find_run(ends3, 3, 0), 0);
    EXPECT_EQ(R::find_run(ends3, 3, 1), 1);
    EXPECT_EQ(R::find_run(ends3, 3, 3), 1);
    EXPECT_EQ(R::find_run(ends3, 3, 8), 2);
}

// The round-trip oracle catches an off-by-one run end (the K3 injection).
TEST(Rle, OffByOneRunEndIsCaught) {
    const Case<int16_t> c = make_case<int16_t>(9, 100000, 0);
    Enc<int16_t> e = encode_case(c);
    ASSERT_GT(e.ends.size(), 2u);
    const int64_t n = static_cast<int64_t>(c.v.size());
    for (int delta : {-1, +1}) {
        Enc<int16_t> bad = e;
        bad.ends[1] += delta;
        std::vector<int16_t> out(static_cast<size_t>(n));
        const bool ok = R::decode(bad.vals.data(), bad.ends.data(),
                                  static_cast<int64_t>(bad.ends.size()), n, out.data());
        EXPECT_TRUE(!ok || out != c.v) << "delta " << delta;
    }
}

// ---- predicates ----------------------------------------------------------

namespace {

template <class T> bool ref_cmp(T v, R::CmpOp op, T s) {
    switch (op) {
        case R::CmpOp::Eq: return v == s;
        case R::CmpOp::Ne: return v != s;
        case R::CmpOp::Lt: return v < s;
        case R::CmpOp::Le: return v <= s;
        case R::CmpOp::Gt: return v > s;
        case R::CmpOp::Ge: return v >= s;
    }
    return false;
}

template <class T, class Pred>
void check_against_ref(const Case<T>& c, const Enc<T>& e, Pred ref, int64_t got_sel_n,
                       const std::vector<int32_t>& sel, int64_t got_count,
                       const std::vector<int32_t>& st, const std::vector<int32_t>& en,
                       int64_t got_ranges, const char* what) {
    const int64_t n = static_cast<int64_t>(c.v.size());
    std::vector<int32_t> want;
    int64_t r = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (i >= e.ends[static_cast<size_t>(r)]) ++r;
        if (bit(c.valid, i) && ref(e.vals[static_cast<size_t>(r)])) {
            want.push_back(static_cast<int32_t>(i));
        }
    }
    ASSERT_EQ(got_sel_n, static_cast<int64_t>(want.size())) << what;
    ASSERT_EQ(got_count, static_cast<int64_t>(want.size())) << what;
    ASSERT_TRUE(std::equal(want.begin(), want.end(), sel.begin())) << what;
    ASSERT_GE(got_ranges, 0) << what;
    std::vector<int32_t> expanded;
    for (int64_t k = 0; k < got_ranges; ++k) {
        ASSERT_LT(st[static_cast<size_t>(k)], en[static_cast<size_t>(k)]) << what;
        if (k > 0) ASSERT_LT(en[static_cast<size_t>(k - 1)], st[static_cast<size_t>(k)]) << what;
        for (int32_t x = st[static_cast<size_t>(k)]; x < en[static_cast<size_t>(k)]; ++x) {
            expanded.push_back(x);
        }
    }
    ASSERT_EQ(expanded, want) << what;
}

template <class T> void predicate_fuzz(uint64_t seed, int null_mode) {
    std::mt19937_64 g(seed * 31 + static_cast<uint64_t>(null_mode));
    const Case<T> c = make_case<T>(seed, 1 + static_cast<int64_t>(g() % 200000), null_mode);
    const Enc<T> e = encode_case(c);
    const int64_t n = static_cast<int64_t>(c.v.size());
    const int64_t runs = static_cast<int64_t>(e.ends.size());
    const uint8_t* vb = c.valid.empty() ? nullptr : c.valid.data();
    const int64_t cap = vb ? (n + 1) / 2 : runs;
    std::vector<int32_t> sel(static_cast<size_t>(n)), st(static_cast<size_t>(cap)),
        en(static_cast<size_t>(cap));
    for (int op = 0; op <= 5; ++op) {
        const R::CmpOp o = static_cast<R::CmpOp>(op);
        const T s = pick<T>(g, 16);
        const int64_t ns = R::filter_sel(e.vals.data(), e.ends.data(), runs, vb, o, s, sel.data());
        const int64_t nc = R::filter_count(e.vals.data(), e.ends.data(), runs, vb, o, s);
        const int64_t nr = R::filter_ranges(e.vals.data(), e.ends.data(), runs, vb, o, s,
                                            st.data(), en.data(), cap);
        check_against_ref(c, e, [&](T v) { return ref_cmp(v, o, s); }, ns, sel, nc, st, en,
                          nr, "cmp");
    }
    for (int set_n : {0, 1, 3, 8, 9, 40}) {
        std::vector<T> set;
        for (int k = 0; k < 64 && static_cast<int>(set.size()) < set_n; ++k) {
            const T x = pick<T>(g, 64);
            if (x != x) continue;  // NaN has no place in an ordered set
            set.push_back(x);
        }
        std::sort(set.begin(), set.end());
        set.erase(std::unique(set.begin(), set.end()), set.end());
        const int32_t sn = static_cast<int32_t>(set.size());
        const int64_t ns = R::filter_in_sel(e.vals.data(), e.ends.data(), runs, vb, set.data(),
                                            sn, sel.data());
        const int64_t nc = R::filter_in_count(e.vals.data(), e.ends.data(), runs, vb,
                                              set.data(), sn);
        const int64_t nr = R::filter_in_ranges(e.vals.data(), e.ends.data(), runs, vb,
                                               set.data(), sn, st.data(), en.data(), cap);
        auto ref = [&](T v) {
            for (const T& x : set) if (x == v) return true;
            return false;
        };
        check_against_ref(c, e, ref, ns, sel, nc, st, en, nr, "in");
    }
}

}  // namespace

TEST(Rle, PredicatesMatchScalarReference) {
    for (int s = 1; s <= kSeeds; ++s) {
        for (int mode = 0; mode <= 4; ++mode) {
            predicate_fuzz<uint8_t>(static_cast<uint64_t>(s), mode);
            predicate_fuzz<int16_t>(static_cast<uint64_t>(s) + 100, mode);
            predicate_fuzz<int32_t>(static_cast<uint64_t>(s) + 200, mode);
            predicate_fuzz<int64_t>(static_cast<uint64_t>(s) + 300, mode);
            predicate_fuzz<double>(static_cast<uint64_t>(s) + 400, mode);
        }
    }
}

TEST(Rle, RangeCapacityRefusesAndUnsortedSetRefused) {
    int32_t v[4] = {1, 2, 1, 2};
    int32_t ends[4] = {2, 4, 6, 8};
    int32_t st[1], en[1];
    EXPECT_EQ(R::filter_ranges(v, ends, 4, nullptr, R::CmpOp::Eq, 1, st, en, 1), -1);
    EXPECT_EQ(R::filter_ranges(v, ends, 4, nullptr, R::CmpOp::Ge, 1, st, en, 1), 1);
    EXPECT_EQ(st[0], 0);
    EXPECT_EQ(en[0], 8);
    const int32_t unsorted[2] = {2, 1};
    const int32_t dup[2] = {1, 1};
    int32_t sel[8];
    EXPECT_EQ(R::filter_in_sel(v, ends, 4, nullptr, unsorted, 2, sel), -1);
    EXPECT_EQ(R::filter_in_count(v, ends, 4, nullptr, dup, 2), -1);
    const int32_t ok[2] = {1, 2};
    EXPECT_EQ(R::filter_in_count(v, ends, 4, nullptr, ok, 2), 8);
}

TEST(Rle, ValidityAcrossWordBoundaries) {
    // One run of 200 rows; every third row null; Eq matches the valid ones.
    std::vector<int32_t> v(200, 7);
    std::vector<uint8_t> vb(25, 0);
    for (int i = 0; i < 200; ++i) if (i % 3 != 0) vb[static_cast<size_t>(i >> 3)] |= uint8_t(1u << (i & 7));
    const int32_t val[1] = {7};
    const int32_t ends[1] = {200};
    std::vector<int32_t> sel(200), st(100), en(100);
    const int64_t ns = R::filter_sel(val, ends, 1, vb.data(), R::CmpOp::Eq, 7, sel.data());
    EXPECT_EQ(ns, 133);
    const int64_t nr = R::filter_ranges(val, ends, 1, vb.data(), R::CmpOp::Eq, 7, st.data(),
                                        en.data(), 100);
    EXPECT_EQ(nr, 67);
    EXPECT_EQ(st[66], 199);
    EXPECT_EQ(en[66], 200);
}
