// The value-canonical key contract (bolt/kernels/numeric_key.h) and the
// group-by paths that must honour it: G2CHK-276 (float keys by value),
// G2CHK-295 (Utf8 DISTINCT by content), G2CHK-311 (NULL key beside 0 / ''),
// G2CHK-301 (decimal literal -> double).

#include "bolt/join/bolt_groupby.h"
#include "bolt/kernels/numeric_key.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

using namespace bolt;
namespace nk = bolt::kernels::numeric_key;

constexpr double kInf = std::numeric_limits<double>::infinity();

double neg_nan() {
    const double n = std::numeric_limits<double>::quiet_NaN();
    return -n;   // sign bit set: a second NaN bit pattern
}

// V from the WI-1 plan, in ascending SQL order (-0.0 == 0.0, NaN last).
std::vector<double> ordered_v() {
    return {-kInf, -1e300, -2.0, -1.5, -0.0, 0.0, 1.5, kInf,
            std::numeric_limits<double>::quiet_NaN()};
}

AggSpec spec(AggKind k, uint8_t in_col, uint8_t distinct = 0) {
    AggSpec s{};
    s.kind = k; s.in_col = in_col; s.distinct = distinct;
    return s;
}

bool bit(const uint8_t* v, uint32_t i) { return ((v[i >> 3] >> (i & 7)) & 1) != 0; }

TEST(NumericKey, CanonAndSortableOrderOverV) {
    const std::vector<double> v = ordered_v();
    EXPECT_EQ(nk::canon_f64_bits_of(-0.0), nk::canon_f64_bits_of(0.0));
    EXPECT_EQ(nk::canon_f64_bits_of(neg_nan()),
              nk::canon_f64_bits_of(std::numeric_limits<double>::quiet_NaN()));
    EXPECT_NE(nk::canon_f64_bits_of(-1.5), nk::canon_f64_bits_of(1.5));
    for (size_t i = 0; i + 1 < v.size(); ++i) {
        const bool zero_pair = (v[i] == 0.0 && v[i + 1] == 0.0);
        if (zero_pair) {
            EXPECT_EQ(nk::f64_sortable_key(v[i]), nk::f64_sortable_key(v[i + 1]));
        } else {
            EXPECT_LT(nk::f64_sortable_key(v[i]), nk::f64_sortable_key(v[i + 1]))
                << v[i] << " vs " << v[i + 1];
        }
    }
    EXPECT_EQ(nk::f64_sortable_key(neg_nan()),
              nk::f64_sortable_key(std::numeric_limits<double>::quiet_NaN()));
    EXPECT_TRUE(nk::f64_total_less(kInf, neg_nan()));
}

TEST(NumericKey, Slot8PackWidensAndFlagsNull) {
    int32_t i32[] = {-1, 7};
    uint8_t valid[1] = {0b10};
    BoltColumn c32 = BoltColumn::make_flat(i32, valid, 2, BoltType::Int32);
    int64_t slot = 99;
    bool null = false;
    ASSERT_TRUE(nk::slot8_pack(c32, 0, &slot, &null));
    EXPECT_TRUE(null);
    EXPECT_EQ(slot, 0);
    ASSERT_TRUE(nk::slot8_pack(c32, 1, &slot, &null));
    EXPECT_FALSE(null);
    EXPECT_EQ(slot, 7);
    BoltColumn c32v = BoltColumn::make_flat(i32, nullptr, 2, BoltType::Int32);
    ASSERT_TRUE(nk::slot8_pack(c32v, 0, &slot, &null));
    EXPECT_EQ(slot, -1) << "Int32 sign-extends";

    double f[] = {-0.0, 0.0, neg_nan(), std::numeric_limits<double>::quiet_NaN()};
    BoltColumn cf = BoltColumn::make_flat(f, nullptr, 4, BoltType::Float64);
    int64_t s0, s1, s2, s3;
    ASSERT_TRUE(nk::slot8_pack(cf, 0, &s0, &null));
    ASSERT_TRUE(nk::slot8_pack(cf, 1, &s1, &null));
    ASSERT_TRUE(nk::slot8_pack(cf, 2, &s2, &null));
    ASSERT_TRUE(nk::slot8_pack(cf, 3, &s3, &null));
    EXPECT_EQ(s0, s1);
    EXPECT_EQ(s2, s3);

    kernels::decimal::Decimal128 d[1] = {};
    BoltColumn cd = BoltColumn::make_flat(d, nullptr, 1, BoltType::Decimal128);
    EXPECT_FALSE(nk::slot8_pack(cd, 0, &slot, &null)) << "16 bytes never truncate";
    StringView sv[1] = {};
    BoltColumn cs = BoltColumn::make_flat(sv, nullptr, 1, BoltType::Utf8);
    EXPECT_FALSE(nk::slot8_pack(cs, 0, &slot, &null));
}

TEST(NumericKey, DecimalLiteralToDoubleIsCorrectlyRounded) {
    EXPECT_EQ(nk::dec_lit_to_f64(694876139, 6), 694.876139);
    EXPECT_EQ(nk::dec_lit_to_f64(-6408019996606812LL, 10), -640801.9996606812);
    EXPECT_EQ(nk::dec_lit_to_f64(69487613899999997LL, 14),
              std::strtod("694.87613899999997", nullptr));
    EXPECT_EQ(nk::dec_lit_to_f64(1, 1), 0.1);
    // 1e20 / 1e38 needs the text path (128-bit mantissa).
    const int64_t hi = 0x5;                      // 5 * 2^64 + 7
    EXPECT_EQ(nk::dec_lit_to_f64(hi, 7, 3),
              std::strtod("92233720368547758087e-3", nullptr));
    EXPECT_EQ(nk::dec_lit_to_f64(-1, -15, 1), -1.5);
}

// Every V value as a GROUP BY key: -0.0/0.0 and the two NaNs merge.
TEST(NumericKey, GroupByFloatKeyByValue) {
    Arena a;
    std::vector<double> ks = ordered_v();
    ks.push_back(neg_nan());
    ks.push_back(0.0);
    const uint32_t n = static_cast<uint32_t>(ks.size());
    std::vector<int64_t> one(n, 1);
    BoltColumn key = BoltColumn::make_flat(ks.data(), nullptr, n, BoltType::Float64);
    BoltColumn val = BoltColumn::make_flat(one.data(), nullptr, n, BoltType::Int64);
    AggSpec s = spec(AggKind::Sum, 0);
    BoltColumn ok[1], oa[1];
    uint32_t ng = 0;
    ASSERT_TRUE(groupby_agg_multi_key_typed(&key, 1, &val, 1, &s, 1, n, ok, oa,
                                            &ng, &a, 16));
    EXPECT_EQ(ng, 8u);
    const auto* kb = static_cast<const double*>(ok[0].data);
    const auto* sb = static_cast<const int64_t*>(oa[0].data);
    for (uint32_t g = 0; g < ng; ++g) {
        if (kb[g] == 0.0) EXPECT_EQ(sb[g], 3) << "-0.0, 0.0, 0.0";
        if (std::isnan(kb[g])) EXPECT_EQ(sb[g], 2) << "both NaN patterns";
    }
}

TEST(NumericKey, CountDistinctFloatAndMinMaxNaN) {
    Arena a;
    std::vector<double> vs = ordered_v();
    vs.push_back(neg_nan());
    const uint32_t n = static_cast<uint32_t>(vs.size());
    std::vector<int64_t> ks(n, 1);
    BoltColumn key = BoltColumn::make_flat(ks.data(), nullptr, n, BoltType::Int64);
    BoltColumn val = BoltColumn::make_flat(vs.data(), nullptr, n, BoltType::Float64);
    AggSpec s[3] = {spec(AggKind::Count, 0, 1), spec(AggKind::Min, 0),
                    spec(AggKind::Max, 0)};
    BoltColumn ok[1], oa[3];
    uint32_t ng = 0;
    ASSERT_TRUE(groupby_agg_multi_key_typed(&key, 1, &val, 1, s, 3, n, ok, oa,
                                            &ng, &a, 4));
    ASSERT_EQ(ng, 1u);
    EXPECT_EQ(static_cast<const int64_t*>(oa[0].data)[0], 8);
    EXPECT_EQ(static_cast<const double*>(oa[1].data)[0], -kInf);
    EXPECT_TRUE(std::isnan(static_cast<const double*>(oa[2].data)[0]));

    // MIN over only NaN is NaN, not +inf.
    double only_nan[2] = {neg_nan(), std::numeric_limits<double>::quiet_NaN()};
    int64_t k2[2] = {1, 1};
    BoltColumn key2 = BoltColumn::make_flat(k2, nullptr, 2, BoltType::Int64);
    BoltColumn val2 = BoltColumn::make_flat(only_nan, nullptr, 2, BoltType::Float64);
    AggSpec s2 = spec(AggKind::Min, 0);
    BoltColumn ok2[1], oa2[1];
    ASSERT_TRUE(groupby_agg_multi_key_typed(&key2, 1, &val2, 1, &s2, 1, 2, ok2,
                                            oa2, &ng, &a, 4));
    EXPECT_TRUE(std::isnan(static_cast<const double*>(oa2[0].data)[0]));
}

StringView spilled(const std::string& pool, uint32_t off, uint32_t len) {
    StringView v{};
    v.length = len;
    std::memcpy(v.prefix, pool.data() + off, 4);
    v.ref.buf_idx = 0;
    v.ref.offset = off;
    return v;
}

StringView inline_sv(const char* s) {
    StringView v{};
    v.length = static_cast<uint32_t>(std::strlen(s));
    std::memcpy(v.prefix, s, v.length);
    return v;
}

// Equal long strings at two offsets + more than 64 distinct per group, with
// inline and spilled values mixed.
TEST(NumericKey, Utf8DistinctByContentPastTheInlineCap) {
    Arena a;
    std::string pool;
    std::vector<StringView> vs;
    std::vector<int64_t> ks;
    const std::string longs(30, 'q');
    for (int copy = 0; copy < 2; ++copy) {
        const uint32_t off = static_cast<uint32_t>(pool.size());
        pool += longs;
        vs.push_back(spilled(pool, off, 30));
        ks.push_back(1);
    }
    std::vector<std::string> shorts;
    for (int i = 0; i < 100; ++i) shorts.push_back("s" + std::to_string(i));
    for (int rep = 0; rep < 2; ++rep) {
        for (int i = 0; i < 100; ++i) {
            vs.push_back(inline_sv(shorts[static_cast<size_t>(i)].c_str()));
            ks.push_back(1);
            const std::string lg = "long-value-number-" + std::to_string(i);
            const uint32_t off = static_cast<uint32_t>(pool.size());
            pool += lg;
            vs.push_back(spilled(pool, off, static_cast<uint32_t>(lg.size())));
            ks.push_back(1);
        }
    }
    // Views were built before the pool stopped growing: rebuild prefixes.
    const uint32_t n = static_cast<uint32_t>(vs.size());
    for (uint32_t i = 0; i < n; ++i) {
        if (vs[i].length > 12) std::memcpy(vs[i].prefix, pool.data() + vs[i].ref.offset, 4);
    }
    BoltColumn key = BoltColumn::make_flat(ks.data(), nullptr, n, BoltType::Int64);
    BoltColumn val = BoltColumn::make_flat(vs.data(), nullptr, n, BoltType::Utf8);
    val.str_overflow_base = const_cast<char*>(pool.data());
    AggSpec s = spec(AggKind::Count, 0, 1);
    BoltColumn ok[1], oa[1];
    uint32_t ng = 0;
    ASSERT_TRUE(groupby_agg_multi_key_typed(&key, 1, &val, 1, &s, 1, n, ok, oa,
                                            &ng, &a, 4));
    ASSERT_EQ(ng, 1u);
    EXPECT_EQ(static_cast<const int64_t*>(oa[0].data)[0], 1 + 100 + 100);
}

// Decimal128 values that share a low word must stay distinct.
TEST(NumericKey, Decimal128DistinctUsesAllSixteenBytes) {
    Arena a;
    kernels::decimal::Decimal128 d[3] = {};
    d[0].lo = -1; d[0].hi = -1;      // -1
    d[1].lo = -1; d[1].hi = 0;       // 2^64 - 1
    d[2].lo = -1; d[2].hi = -1;      // -1 again
    int64_t ks[3] = {1, 1, 1};
    BoltColumn key = BoltColumn::make_flat(ks, nullptr, 3, BoltType::Int64);
    BoltColumn val = BoltColumn::make_flat(d, nullptr, 3, BoltType::Decimal128);
    AggSpec s = spec(AggKind::Count, 0, 1);
    BoltColumn ok[1], oa[1];
    uint32_t ng = 0;
    ASSERT_TRUE(groupby_agg_multi_key_typed(&key, 1, &val, 1, &s, 1, 3, ok, oa,
                                            &ng, &a, 4));
    ASSERT_EQ(ng, 1u);
    EXPECT_EQ(static_cast<const int64_t*>(oa[0].data)[0], 2);
}

// NULL beside 0 and '' in every key column: separate groups, one NULL group.
TEST(NumericKey, NullKeyBesideZeroAndEmpty) {
    Arena a;
    int64_t ik[4] = {0, 12345, 0, 0};
    uint8_t iv[1] = {0b1101};                // row 1 NULL
    StringView sk[4] = {inline_sv(""), inline_sv("x"), inline_sv(""), inline_sv("")};
    uint8_t sv_valid[1] = {0b1011};          // row 2 NULL
    double fk[4] = {0.0, 0.0, -0.0, 0.0};
    uint8_t fv[1] = {0b0111};                // row 3 NULL
    BoltColumn keys[3] = {
        BoltColumn::make_flat(ik, iv, 4, BoltType::Int64),
        BoltColumn::make_flat(sk, sv_valid, 4, BoltType::Utf8),
        BoltColumn::make_flat(fk, fv, 4, BoltType::Float64),
    };
    for (int k = 0; k < 3; ++k) {
        int64_t one[4] = {1, 1, 1, 1};
        BoltColumn val = BoltColumn::make_flat(one, nullptr, 4, BoltType::Int64);
        AggSpec s = spec(AggKind::Sum, 0);
        BoltColumn ok[1], oa[1];
        uint32_t ng = 0;
        ASSERT_TRUE(groupby_agg_multi_key_typed(&keys[k], 1, &val, 1, &s, 1, 4,
                                                ok, oa, &ng, &a, 8));
        ASSERT_NE(ok[0].validity, nullptr) << "key " << k;
        int nulls = 0, null_sum = 0;
        for (uint32_t g = 0; g < ng; ++g) {
            if (!bit(ok[0].validity, g)) {
                ++nulls;
                null_sum = static_cast<int>(static_cast<const int64_t*>(oa[0].data)[g]);
            }
        }
        EXPECT_EQ(nulls, 1) << "key " << k;
        EXPECT_EQ(null_sum, 1) << "key " << k;
        EXPECT_EQ(ng, k == 1 ? 3u : 2u) << "key " << k;
    }
}

}  // namespace
