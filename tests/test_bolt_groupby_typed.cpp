// Tests for bolt::groupby_agg_multi_key_typed (K-AGG-A + K-AGG-A.2).
//
// Scope: kernel-output correctness on small typed inputs.
//   - Int64 single-key + SUM
//   - Int32 multi-key + COUNT(*)
//   - Decimal128 SUM + AVG
//   - Date32 keys
//   - Composite-key (2) hash discriminates correctly
//   - K-AGG-A.2 item 1: NULL masking via BoltColumn::validity
//   - K-AGG-A.2 item 2: per-(agg, group) DISTINCT folding (Int64 + Utf8)
//   - K-AGG-A.2 item 3: Utf8 byte-lex MIN / MAX

#include "bolt/join/bolt_groupby.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

namespace {

using namespace bolt;
namespace dec = bolt::kernels::decimal;

// Build an AggSpec with explicit zero-init for `distinct` and padding so
// the kernel sees deterministic input even on stack-uninitialised flags.
inline AggSpec make_spec(AggKind k, uint8_t in_col,
                         uint8_t distinct = 0) noexcept {
    AggSpec s{};
    s.kind = k; s.in_col = in_col; s.distinct = distinct;
    return s;
}

TEST(BoltGroupbyTyped, Int64SingleKeySum) {
    Arena a;
    int64_t ks[] = {1, 2, 1, 2, 1, 3};
    int64_t vs[] = {10, 20, 30, 40, 50, 60};
    BoltColumn key = BoltColumn::make_flat(ks, nullptr, 6, BoltType::Int64);
    BoltColumn val = BoltColumn::make_flat(vs, nullptr, 6, BoltType::Int64);
    AggSpec spec = make_spec(AggKind::Sum, 0);
    BoltColumn ok[1], oa[1];
    uint32_t ng = 0;
    ASSERT_TRUE(groupby_agg_multi_key_typed(&key, 1, &val, 1, &spec, 1, 6,
                                            ok, oa, &ng, &a, /*hint=*/8));
    EXPECT_EQ(ng, 3u);
    int64_t* outk = static_cast<int64_t*>(ok[0].data);
    int64_t* outv = static_cast<int64_t*>(oa[0].data);
    int64_t sum_by_key[4] = {0, 0, 0, 0};
    for (uint32_t i = 0; i < ng; ++i) sum_by_key[outk[i]] = outv[i];
    EXPECT_EQ(sum_by_key[1], 90);
    EXPECT_EQ(sum_by_key[2], 60);
    EXPECT_EQ(sum_by_key[3], 60);
}

TEST(BoltGroupbyTyped, CountStarMatchesRowCount) {
    Arena a;
    int64_t ks[] = {5, 5, 5, 7, 7, 9};
    BoltColumn key = BoltColumn::make_flat(ks, nullptr, 6, BoltType::Int64);
    AggSpec spec = make_spec(AggKind::CountStar, 0);
    BoltColumn ok[1], oa[1];
    uint32_t ng = 0;
    ASSERT_TRUE(groupby_agg_multi_key_typed(&key, 1, nullptr, 0, &spec, 1, 6,
                                            ok, oa, &ng, &a, /*hint=*/4));
    EXPECT_EQ(ng, 3u);
    int64_t* outk = static_cast<int64_t*>(ok[0].data);
    int64_t* outc = static_cast<int64_t*>(oa[0].data);
    int64_t cnt_by_key[10] = {0};
    for (uint32_t i = 0; i < ng; ++i) cnt_by_key[outk[i]] = outc[i];
    EXPECT_EQ(cnt_by_key[5], 3);
    EXPECT_EQ(cnt_by_key[7], 2);
    EXPECT_EQ(cnt_by_key[9], 1);
}

TEST(BoltGroupbyTyped, CompositeKeyDiscriminates) {
    Arena a;
    int32_t k0[] = {1, 1, 2, 2};
    int64_t k1[] = {10, 20, 10, 10};
    int64_t vs[] = {1, 2, 4, 8};
    BoltColumn keys[2];
    keys[0] = BoltColumn::make_flat(k0, nullptr, 4, BoltType::Int32);
    keys[1] = BoltColumn::make_flat(k1, nullptr, 4, BoltType::Int64);
    BoltColumn val = BoltColumn::make_flat(vs, nullptr, 4, BoltType::Int64);
    AggSpec spec = make_spec(AggKind::Sum, 0);
    BoltColumn ok[2], oa[1];
    uint32_t ng = 0;
    ASSERT_TRUE(groupby_agg_multi_key_typed(keys, 2, &val, 1, &spec, 1, 4,
                                            ok, oa, &ng, &a, /*hint=*/4));
    EXPECT_EQ(ng, 3u);
    int64_t* outv = static_cast<int64_t*>(oa[0].data);
    int64_t total = 0;
    for (uint32_t i = 0; i < ng; ++i) total += outv[i];
    EXPECT_EQ(total, 1 + 2 + 12);
}

TEST(BoltGroupbyTyped, Decimal128SumPreservesScale) {
    Arena a;
    int64_t ks[] = {1, 1, 2};
    dec::Decimal128 vs[3] = {
        dec::d128_from_i64(100),
        dec::d128_from_i64(250),
        dec::d128_from_i64(700),
    };
    BoltColumn key = BoltColumn::make_flat(ks, nullptr, 3, BoltType::Int64);
    BoltColumn val = BoltColumn::make_flat(vs, nullptr, 3, BoltType::Decimal128);
    val.decimal_scale = 2;
    AggSpec spec = make_spec(AggKind::Sum, 0);
    BoltColumn ok[1], oa[1];
    uint32_t ng = 0;
    ASSERT_TRUE(groupby_agg_multi_key_typed(&key, 1, &val, 1, &spec, 1, 3,
                                            ok, oa, &ng, &a, /*hint=*/4));
    EXPECT_EQ(ng, 2u);
    EXPECT_EQ(oa[0].type, BoltType::Decimal128);
    EXPECT_EQ(oa[0].decimal_scale, 2);
    dec::Decimal128* outv = static_cast<dec::Decimal128*>(oa[0].data);
    int64_t* outk = static_cast<int64_t*>(ok[0].data);
    for (uint32_t i = 0; i < ng; ++i) {
        if (outk[i] == 1) EXPECT_EQ(dec::d128_cmp(outv[i], dec::d128_from_i64(350)), 0);
        if (outk[i] == 2) EXPECT_EQ(dec::d128_cmp(outv[i], dec::d128_from_i64(700)), 0);
    }
}

TEST(BoltGroupbyTyped, Date32KeysGroup) {
    Arena a;
    int32_t ks[] = {18000, 18001, 18000, 18001, 18000};
    int64_t vs[] = {1, 2, 4, 8, 16};
    BoltColumn key = BoltColumn::make_flat(ks, nullptr, 5, BoltType::Date32);
    BoltColumn val = BoltColumn::make_flat(vs, nullptr, 5, BoltType::Int64);
    AggSpec spec = make_spec(AggKind::Sum, 0);
    BoltColumn ok[1], oa[1];
    uint32_t ng = 0;
    ASSERT_TRUE(groupby_agg_multi_key_typed(&key, 1, &val, 1, &spec, 1, 5,
                                            ok, oa, &ng, &a, /*hint=*/4));
    EXPECT_EQ(ng, 2u);
    EXPECT_EQ(ok[0].type, BoltType::Date32);
    int32_t* outk = static_cast<int32_t*>(ok[0].data);
    int64_t* outv = static_cast<int64_t*>(oa[0].data);
    for (uint32_t i = 0; i < ng; ++i) {
        if (outk[i] == 18000) EXPECT_EQ(outv[i], 1 + 4 + 16);
        if (outk[i] == 18001) EXPECT_EQ(outv[i], 2 + 8);
    }
}

TEST(BoltGroupbyTyped, MinMaxInt64) {
    Arena a;
    int64_t ks[] = {1, 1, 1, 2, 2};
    int64_t vs[] = {30, 10, 20, 5, 50};
    BoltColumn key = BoltColumn::make_flat(ks, nullptr, 5, BoltType::Int64);
    BoltColumn val = BoltColumn::make_flat(vs, nullptr, 5, BoltType::Int64);
    AggSpec specs[2];
    specs[0] = make_spec(AggKind::Min, 0);
    specs[1] = make_spec(AggKind::Max, 0);
    BoltColumn ok[1], oa[2];
    uint32_t ng = 0;
    ASSERT_TRUE(groupby_agg_multi_key_typed(&key, 1, &val, 1, specs, 2, 5,
                                            ok, oa, &ng, &a, /*hint=*/4));
    EXPECT_EQ(ng, 2u);
    int64_t* outk = static_cast<int64_t*>(ok[0].data);
    int64_t* mins = static_cast<int64_t*>(oa[0].data);
    int64_t* maxs = static_cast<int64_t*>(oa[1].data);
    for (uint32_t i = 0; i < ng; ++i) {
        if (outk[i] == 1) { EXPECT_EQ(mins[i], 10); EXPECT_EQ(maxs[i], 30); }
        if (outk[i] == 2) { EXPECT_EQ(mins[i], 5);  EXPECT_EQ(maxs[i], 50); }
    }
}

// --- K-AGG-A.2 item 1: NULL masking via BoltColumn::validity ---
//
// Build a 6-row column with rows 1 and 4 NULL. Bits set: 0,2,3,5 → byte 0b00101101
// SUM should skip the NULL rows, COUNT should skip them, COUNT(*) counts every row.
TEST(BoltGroupbyTyped, NullMaskingSkipsNullsForSumCount) {
    Arena a;
    int64_t ks[] = {1, 1, 1, 2, 2, 2};
    int64_t vs[] = {10, 999, 20, 30, 999, 40};   // 999 are "null" payloads
    // Validity: bit i = 1 iff row i is non-NULL. Rows 1,4 are NULL.
    uint8_t validity[1] = {0b00101101};   // bits 0,2,3,5 set
    BoltColumn key = BoltColumn::make_flat(ks, nullptr, 6, BoltType::Int64);
    BoltColumn val = BoltColumn::make_flat(vs, validity, 6, BoltType::Int64);
    AggSpec specs[3];
    specs[0] = make_spec(AggKind::Sum, 0);
    specs[1] = make_spec(AggKind::Count, 0);
    specs[2] = make_spec(AggKind::CountStar, 0);
    BoltColumn ok[1], oa[3];
    uint32_t ng = 0;
    ASSERT_TRUE(groupby_agg_multi_key_typed(&key, 1, &val, 1, specs, 3, 6,
                                            ok, oa, &ng, &a, /*hint=*/4));
    EXPECT_EQ(ng, 2u);
    int64_t* outk = static_cast<int64_t*>(ok[0].data);
    int64_t* sums = static_cast<int64_t*>(oa[0].data);
    int64_t* cnts = static_cast<int64_t*>(oa[1].data);
    int64_t* css  = static_cast<int64_t*>(oa[2].data);
    for (uint32_t i = 0; i < ng; ++i) {
        if (outk[i] == 1) {
            EXPECT_EQ(sums[i], 30);   // 10 + 20 (skip null at row 1)
            EXPECT_EQ(cnts[i], 2);
            EXPECT_EQ(css[i],  3);
        }
        if (outk[i] == 2) {
            EXPECT_EQ(sums[i], 70);   // 30 + 40 (skip null at row 4)
            EXPECT_EQ(cnts[i], 2);
            EXPECT_EQ(css[i],  3);
        }
    }
}

// MIN / MAX with NULLs: must NOT mutate the slot identity when valid==false.
// If item 1 is broken, MIN would see the poison 999 and pick it.
TEST(BoltGroupbyTyped, NullMaskingSkipsNullsForMinMax) {
    Arena a;
    int64_t ks[] = {1, 1, 1};
    int64_t vs[] = {100, -999, 50};   // -999 is "null" payload
    uint8_t validity[1] = {0b00000101};   // bits 0,2 set (row 1 is NULL)
    BoltColumn key = BoltColumn::make_flat(ks, nullptr, 3, BoltType::Int64);
    BoltColumn val = BoltColumn::make_flat(vs, validity, 3, BoltType::Int64);
    AggSpec specs[2];
    specs[0] = make_spec(AggKind::Min, 0);
    specs[1] = make_spec(AggKind::Max, 0);
    BoltColumn ok[1], oa[2];
    uint32_t ng = 0;
    ASSERT_TRUE(groupby_agg_multi_key_typed(&key, 1, &val, 1, specs, 2, 3,
                                            ok, oa, &ng, &a, /*hint=*/4));
    EXPECT_EQ(ng, 1u);
    int64_t* mn = static_cast<int64_t*>(oa[0].data);
    int64_t* mx = static_cast<int64_t*>(oa[1].data);
    EXPECT_EQ(mn[0], 50);    // NOT -999
    EXPECT_EQ(mx[0], 100);
}

// --- K-AGG-A.2 item 2: DISTINCT folding ---
//
// Two groups, with duplicate values within each group. SUM(DISTINCT)
// must fold duplicates per group; COUNT(DISTINCT) must count uniques.
TEST(BoltGroupbyTyped, DistinctInt64SumAndCount) {
    Arena a;
    int64_t ks[] = {1, 1, 1, 1, 2, 2, 2};
    int64_t vs[] = {10, 20, 10, 30, 5, 5, 7};
    BoltColumn key = BoltColumn::make_flat(ks, nullptr, 7, BoltType::Int64);
    BoltColumn val = BoltColumn::make_flat(vs, nullptr, 7, BoltType::Int64);
    AggSpec specs[2];
    specs[0] = make_spec(AggKind::Sum,   0, /*distinct=*/1);
    specs[1] = make_spec(AggKind::Count, 0, /*distinct=*/1);
    BoltColumn ok[1], oa[2];
    uint32_t ng = 0;
    ASSERT_TRUE(groupby_agg_multi_key_typed(&key, 1, &val, 1, specs, 2, 7,
                                            ok, oa, &ng, &a, /*hint=*/4));
    EXPECT_EQ(ng, 2u);
    int64_t* outk = static_cast<int64_t*>(ok[0].data);
    int64_t* sums = static_cast<int64_t*>(oa[0].data);
    int64_t* cnts = static_cast<int64_t*>(oa[1].data);
    for (uint32_t i = 0; i < ng; ++i) {
        if (outk[i] == 1) { EXPECT_EQ(sums[i], 60); EXPECT_EQ(cnts[i], 3); }
        if (outk[i] == 2) { EXPECT_EQ(sums[i], 12); EXPECT_EQ(cnts[i], 2); }
    }
}

// --- K-AGG-A.2 item 3: Utf8 byte-lex MIN/MAX ---
//
// Three group-1 strings with the same length+prefix would have collided
// under the legacy Int64-cmp path. Verify the byte-lex compare picks the
// correct min/max.
static StringView sv_inline(const char* s, uint32_t len) noexcept {
    StringView v{};
    v.length = len;
    // 4-byte prefix copied first; tail in inline_data (max 8 bytes).
    // length ≤ 12 so the whole content fits inline.
    assert(len <= 12u);
    std::memcpy(v.prefix, s, len < 4u ? len : 4u);
    if (len > 4u) {
        std::memcpy(v.inline_data, s + 4, len - 4u);
    }
    return v;
}

TEST(BoltGroupbyTyped, Utf8ByteLexMinMax) {
    Arena a;
    int64_t ks[] = {1, 1, 1, 2, 2};
    StringView vs[5] = {
        sv_inline("banana", 6),
        sv_inline("apple",  5),
        sv_inline("cherry", 6),
        sv_inline("kiwi",   4),
        sv_inline("grape",  5),
    };
    BoltColumn key = BoltColumn::make_flat(ks, nullptr, 5, BoltType::Int64);
    BoltColumn val = BoltColumn::make_flat(vs, nullptr, 5, BoltType::Utf8);
    AggSpec specs[2];
    specs[0] = make_spec(AggKind::Min, 0);
    specs[1] = make_spec(AggKind::Max, 0);
    BoltColumn ok[1], oa[2];
    uint32_t ng = 0;
    ASSERT_TRUE(groupby_agg_multi_key_typed(&key, 1, &val, 1, specs, 2, 5,
                                            ok, oa, &ng, &a, /*hint=*/4));
    EXPECT_EQ(ng, 2u);
    EXPECT_EQ(oa[0].type, BoltType::Utf8);
    EXPECT_EQ(oa[1].type, BoltType::Utf8);
    int64_t* outk = static_cast<int64_t*>(ok[0].data);
    StringView* mn = static_cast<StringView*>(oa[0].data);
    StringView* mx = static_cast<StringView*>(oa[1].data);
    auto sv_eq = [](const StringView& a, const char* s, uint32_t n) {
        if (a.length != n) return false;
        const uint32_t hp = (n < 4u) ? n : 4u;
        if (std::memcmp(a.prefix, s, hp) != 0) return false;
        if (n > 4u && std::memcmp(a.inline_data, s + 4, n - 4u) != 0) return false;
        return true;
    };
    for (uint32_t i = 0; i < ng; ++i) {
        if (outk[i] == 1) {
            EXPECT_TRUE(sv_eq(mn[i], "apple",  5));
            EXPECT_TRUE(sv_eq(mx[i], "cherry", 6));
        }
        if (outk[i] == 2) {
            EXPECT_TRUE(sv_eq(mn[i], "grape", 5));
            EXPECT_TRUE(sv_eq(mx[i], "kiwi",  4));
        }
    }
}

// DISTINCT folding for Utf8 inline strings — exercises distinct_cells16.
TEST(BoltGroupbyTyped, Utf8DistinctCount) {
    Arena a;
    int64_t ks[] = {1, 1, 1, 1, 2};
    StringView vs[5] = {
        sv_inline("alpha", 5),
        sv_inline("beta",  4),
        sv_inline("alpha", 5),   // dup
        sv_inline("gamma", 5),
        sv_inline("delta", 5),
    };
    BoltColumn key = BoltColumn::make_flat(ks, nullptr, 5, BoltType::Int64);
    BoltColumn val = BoltColumn::make_flat(vs, nullptr, 5, BoltType::Utf8);
    AggSpec spec = make_spec(AggKind::Count, 0, /*distinct=*/1);
    BoltColumn ok[1], oa[1];
    uint32_t ng = 0;
    ASSERT_TRUE(groupby_agg_multi_key_typed(&key, 1, &val, 1, &spec, 1, 5,
                                            ok, oa, &ng, &a, /*hint=*/4));
    EXPECT_EQ(ng, 2u);
    int64_t* outk = static_cast<int64_t*>(ok[0].data);
    int64_t* cnts = static_cast<int64_t*>(oa[0].data);
    for (uint32_t i = 0; i < ng; ++i) {
        if (outk[i] == 1) EXPECT_EQ(cnts[i], 3);   // alpha, beta, gamma
        if (outk[i] == 2) EXPECT_EQ(cnts[i], 1);
    }
}

// --- K-AGG-A.4-PREREQ: Utf8 agg_identity must encode a valid inline StringView.
//
// Before the fix `agg_identity(Min, Utf8)` returned {INT64_MAX, 0}, aliased as
// a StringView with length = (INT64_MAX & 0xFFFFFFFF) = 0xFFFFFFFF — far above
// the 12-byte inline cap. The very first `apply()` against that identity
// asserted in `cur.length <= 12u`. In release builds NDEBUG hid the bug, but
// the StringView was still malformed. This test asserts the encoded identity
// is a well-formed inline StringView regardless of NDEBUG.
TEST(BoltGroupbyTyped, Utf8AggIdentityIsValidInlineStringView) {
    const GbCell16 id_min =
        gb_detail::agg_identity(AggKind::Min, BoltType::Utf8);
    const GbCell16 id_max =
        gb_detail::agg_identity(AggKind::Max, BoltType::Utf8);
    StringView sv_min{}, sv_max{};
    std::memcpy(&sv_min, &id_min, sizeof(sv_min));
    std::memcpy(&sv_max, &id_max, sizeof(sv_max));
    // Inline invariant: length must be ≤ 12.
    EXPECT_LE(sv_min.length, 12u);
    EXPECT_LE(sv_max.length, 12u);
    // MIN identity must be byte-lex-greater than any real ASCII string (a 12-
    // byte all-0xFF inline view is the byte-lex max among inline StringViews
    // built from ASCII input).
    EXPECT_EQ(sv_min.length, 12u);
    for (int i = 0; i < 4; ++i) EXPECT_EQ(static_cast<unsigned char>(sv_min.prefix[i]), 0xFFu);
    for (int i = 0; i < 8; ++i) EXPECT_EQ(static_cast<unsigned char>(sv_min.inline_data[i]), 0xFFu);
    // MAX identity must be byte-lex-less than any real string — empty works.
    EXPECT_EQ(sv_max.length, 0u);
    // Round-trip through apply(): the identity must not assert and must be
    // replaced by any real value on the very first application.
    GbCell16 slot = id_min;
    StringView v = sv_inline("apple", 5);
    GbCell16 vc{};
    std::memcpy(&vc, &v, sizeof(v));
    gb_detail::apply(AggKind::Min, BoltType::Utf8, &slot, vc, /*valid=*/true);
    StringView out{};
    std::memcpy(&out, &slot, sizeof(out));
    EXPECT_EQ(out.length, 5u);
    EXPECT_EQ(std::memcmp(out.prefix, "appl", 4), 0);
    EXPECT_EQ(out.inline_data[0], 'e');
}

}  // namespace
