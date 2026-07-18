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
#include <string>

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

// ---------------------------------------------------------------------------
// K-AGG-A.3 - cross-morsel begin/ingest/finalize + selection-vector ingest.
// ---------------------------------------------------------------------------
namespace xmorsel_anon {
using namespace bolt;
}

TEST(BoltGroupbyTypedXMorsel, ThreeMorselsSameAsOneShot) {
    int64_t ks_all[] = {1, 2, 1, 2, 1, 3, 3, 2, 1, 4};
    int64_t vs_all[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    constexpr int64_t N = 10;
    bolt::Arena a_one;
    uint32_t ng_one = 0;
    bolt::BoltColumn ok_one[1], oa_one[1];
    {
        bolt::BoltColumn key = bolt::BoltColumn::make_flat(ks_all, nullptr, N, bolt::BoltType::Int64);
        bolt::BoltColumn val = bolt::BoltColumn::make_flat(vs_all, nullptr, N, bolt::BoltType::Int64);
        bolt::AggSpec spec{}; spec.kind = bolt::AggKind::Sum; spec.in_col = 0; spec.distinct = 0;
        ASSERT_TRUE(bolt::groupby_agg_multi_key_typed(&key, 1, &val, 1, &spec, 1, N,
                                                ok_one, oa_one, &ng_one, &a_one, 8));
    }
    bolt::Arena a;
    bolt::GroupbyTypedState st{};
    bolt::BoltColumn key_desc = bolt::BoltColumn::make_flat(ks_all, nullptr, 0, bolt::BoltType::Int64);
    bolt::BoltColumn val_desc = bolt::BoltColumn::make_flat(vs_all, nullptr, 0, bolt::BoltType::Int64);
    bolt::AggSpec spec{}; spec.kind = bolt::AggKind::Sum; spec.in_col = 0; spec.distinct = 0;
    ASSERT_TRUE(bolt::groupby_agg_multi_key_typed_begin(
        &st, &a, &key_desc, 1, &val_desc, 1, &spec, 1, 8));
    const int64_t splits[3] = {4, 3, 3};
    int64_t off = 0;
    for (int s = 0; s < 3; ++s) {
        bolt::BoltColumn k = bolt::BoltColumn::make_flat(ks_all + off, nullptr, splits[s], bolt::BoltType::Int64);
        bolt::BoltColumn v = bolt::BoltColumn::make_flat(vs_all + off, nullptr, splits[s], bolt::BoltType::Int64);
        bolt::groupby_agg_multi_key_typed_ingest(&st, &k, &v, nullptr, 0, splits[s]);
        ASSERT_FALSE(st.oom);
        off += splits[s];
    }
    bolt::BoltColumn ok[1], oa[1];
    uint32_t ng = 0;
    ASSERT_TRUE(bolt::groupby_agg_multi_key_typed_finalize(&st, ok, oa, &ng));
    EXPECT_EQ(ng, ng_one);
    int64_t* k_one = static_cast<int64_t*>(ok_one[0].data);
    int64_t* s_one = static_cast<int64_t*>(oa_one[0].data);
    int64_t* k_x   = static_cast<int64_t*>(ok[0].data);
    int64_t* s_x   = static_cast<int64_t*>(oa[0].data);
    int64_t sum_one[8] = {0}, sum_x[8] = {0};
    for (uint32_t i = 0; i < ng_one; ++i) sum_one[k_one[i]] = s_one[i];
    for (uint32_t i = 0; i < ng;     ++i) sum_x[k_x[i]]     = s_x[i];
    for (int i = 0; i < 8; ++i) EXPECT_EQ(sum_one[i], sum_x[i]) << "k=" << i;
}

TEST(BoltGroupbyTypedXMorsel, SelectionVectorIngestSumsOnlySelected) {
    int64_t ks[] = {1, 2, 1, 2, 1, 2, 1, 2};
    int64_t vs[] = {10, 20, 30, 40, 50, 60, 70, 80};
    uint32_t sel[] = {1, 3, 5, 7};
    bolt::Arena a;
    bolt::BoltColumn key_desc = bolt::BoltColumn::make_flat(ks, nullptr, 0, bolt::BoltType::Int64);
    bolt::BoltColumn val_desc = bolt::BoltColumn::make_flat(vs, nullptr, 0, bolt::BoltType::Int64);
    bolt::AggSpec spec{}; spec.kind = bolt::AggKind::Sum; spec.in_col = 0; spec.distinct = 0;
    bolt::GroupbyTypedState st{};
    ASSERT_TRUE(bolt::groupby_agg_multi_key_typed_begin(
        &st, &a, &key_desc, 1, &val_desc, 1, &spec, 1, 4));
    bolt::BoltColumn k = bolt::BoltColumn::make_flat(ks, nullptr, 8, bolt::BoltType::Int64);
    bolt::BoltColumn v = bolt::BoltColumn::make_flat(vs, nullptr, 8, bolt::BoltType::Int64);
    bolt::groupby_agg_multi_key_typed_ingest(&st, &k, &v, sel, 4, 8);
    ASSERT_FALSE(st.oom);
    bolt::BoltColumn ok[1], oa[1];
    uint32_t ng = 0;
    ASSERT_TRUE(bolt::groupby_agg_multi_key_typed_finalize(&st, ok, oa, &ng));
    EXPECT_EQ(ng, 1u);
    EXPECT_EQ(static_cast<int64_t*>(ok[0].data)[0], 2);
    EXPECT_EQ(static_cast<int64_t*>(oa[0].data)[0], 200);
}

TEST(BoltGroupbyTypedXMorsel, MixedDenseAndSparseMorselsMergeCorrectly) {
    bolt::Arena a;
    bolt::BoltColumn key_desc = bolt::BoltColumn::make_flat(static_cast<int64_t*>(nullptr), nullptr, 0, bolt::BoltType::Int64);
    bolt::BoltColumn val_desc = bolt::BoltColumn::make_flat(static_cast<int64_t*>(nullptr), nullptr, 0, bolt::BoltType::Int64);
    bolt::AggSpec spec{}; spec.kind = bolt::AggKind::Sum; spec.in_col = 0; spec.distinct = 0;
    bolt::GroupbyTypedState st{};
    ASSERT_TRUE(bolt::groupby_agg_multi_key_typed_begin(
        &st, &a, &key_desc, 1, &val_desc, 1, &spec, 1, 4));
    int64_t ks1[] = {1, 2, 1};
    int64_t vs1[] = {10, 20, 30};
    bolt::BoltColumn k1 = bolt::BoltColumn::make_flat(ks1, nullptr, 3, bolt::BoltType::Int64);
    bolt::BoltColumn v1 = bolt::BoltColumn::make_flat(vs1, nullptr, 3, bolt::BoltType::Int64);
    bolt::groupby_agg_multi_key_typed_ingest(&st, &k1, &v1, nullptr, 0, 3);
    int64_t ks2[] = {2, 99, 2};
    int64_t vs2[] = {7, 0, 7};
    uint32_t sel2[] = {0, 2};
    bolt::BoltColumn k2 = bolt::BoltColumn::make_flat(ks2, nullptr, 3, bolt::BoltType::Int64);
    bolt::BoltColumn v2 = bolt::BoltColumn::make_flat(vs2, nullptr, 3, bolt::BoltType::Int64);
    bolt::groupby_agg_multi_key_typed_ingest(&st, &k2, &v2, sel2, 2, 3);
    ASSERT_FALSE(st.oom);
    bolt::BoltColumn ok[1], oa[1];
    uint32_t ng = 0;
    ASSERT_TRUE(bolt::groupby_agg_multi_key_typed_finalize(&st, ok, oa, &ng));
    EXPECT_EQ(ng, 2u);
    int64_t sums[3] = {0, 0, 0};
    int64_t* okp = static_cast<int64_t*>(ok[0].data);
    int64_t* oap = static_cast<int64_t*>(oa[0].data);
    for (uint32_t i = 0; i < ng; ++i) sums[okp[i]] = oap[i];
    EXPECT_EQ(sums[1], 40);
    EXPECT_EQ(sums[2], 34);
}

TEST(BoltGroupbyTypedXMorsel, AvgAccumulatesDenominatorAcrossMorsels) {
    bolt::Arena a;
    bolt::BoltColumn key_desc = bolt::BoltColumn::make_flat(static_cast<int64_t*>(nullptr), nullptr, 0, bolt::BoltType::Int64);
    bolt::BoltColumn val_desc = bolt::BoltColumn::make_flat(static_cast<int64_t*>(nullptr), nullptr, 0, bolt::BoltType::Int64);
    bolt::AggSpec spec{}; spec.kind = bolt::AggKind::Avg; spec.in_col = 0; spec.distinct = 0;
    bolt::GroupbyTypedState st{};
    ASSERT_TRUE(bolt::groupby_agg_multi_key_typed_begin(
        &st, &a, &key_desc, 1, &val_desc, 1, &spec, 1, 4));
    int64_t ks[] = {1, 1};
    int64_t v1[] = {2, 4};
    int64_t v2[] = {6, 8};
    bolt::BoltColumn k = bolt::BoltColumn::make_flat(ks, nullptr, 2, bolt::BoltType::Int64);
    bolt::BoltColumn V1 = bolt::BoltColumn::make_flat(v1, nullptr, 2, bolt::BoltType::Int64);
    bolt::BoltColumn V2 = bolt::BoltColumn::make_flat(v2, nullptr, 2, bolt::BoltType::Int64);
    bolt::groupby_agg_multi_key_typed_ingest(&st, &k, &V1, nullptr, 0, 2);
    bolt::groupby_agg_multi_key_typed_ingest(&st, &k, &V2, nullptr, 0, 2);
    bolt::BoltColumn ok[1], oa[1];
    uint32_t ng = 0;
    ASSERT_TRUE(bolt::groupby_agg_multi_key_typed_finalize(&st, ok, oa, &ng));
    EXPECT_EQ(ng, 1u);
    EXPECT_DOUBLE_EQ(static_cast<double*>(oa[0].data)[0], 5.0);
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

// ---------------------------------------------------------------------------
// Float64 aggregation. Before the fix, SUM/AVG/MIN/MAX over a Float64 payload
// accumulated the IEEE-754 bit pattern as int64 (apply()/finalize had no float
// arm) → garbage (~-1.5e17). These pin the corrected double-arithmetic path.
// ---------------------------------------------------------------------------
TEST(BoltGroupbyTyped, Float64SumAvgMinMax) {
    Arena a;
    int64_t ks[] = {1, 1, 1, 2, 2};
    double  vs[] = {1.5, 2.5, 3.0, 10.25, 100.75};   // g1: {1.5,2.5,3.0} g2:{10.25,100.75}
    BoltColumn key = BoltColumn::make_flat(ks, nullptr, 5, BoltType::Int64);
    BoltColumn val = BoltColumn::make_flat(vs, nullptr, 5, BoltType::Float64);
    AggSpec specs[4];
    specs[0] = make_spec(AggKind::Sum, 0);
    specs[1] = make_spec(AggKind::Avg, 0);
    specs[2] = make_spec(AggKind::Min, 0);
    specs[3] = make_spec(AggKind::Max, 0);
    BoltColumn ok[1], oa[4];
    uint32_t ng = 0;
    ASSERT_TRUE(groupby_agg_multi_key_typed(&key, 1, &val, 1, specs, 4, 5,
                                            ok, oa, &ng, &a, /*hint=*/4));
    EXPECT_EQ(ng, 2u);
    // SUM/MIN/MAX preserve Float64; AVG is Float64.
    for (int j = 0; j < 4; ++j) EXPECT_EQ(oa[j].type, BoltType::Float64);
    int64_t* outk = static_cast<int64_t*>(ok[0].data);
    double*  sums = static_cast<double*>(oa[0].data);
    double*  avgs = static_cast<double*>(oa[1].data);
    double*  mins = static_cast<double*>(oa[2].data);
    double*  maxs = static_cast<double*>(oa[3].data);
    for (uint32_t i = 0; i < ng; ++i) {
        if (outk[i] == 1) {
            EXPECT_DOUBLE_EQ(sums[i], 7.0);
            EXPECT_DOUBLE_EQ(avgs[i], 7.0 / 3.0);
            EXPECT_DOUBLE_EQ(mins[i], 1.5);
            EXPECT_DOUBLE_EQ(maxs[i], 3.0);
        }
        if (outk[i] == 2) {
            EXPECT_DOUBLE_EQ(sums[i], 111.0);
            EXPECT_DOUBLE_EQ(avgs[i], 55.5);
            EXPECT_DOUBLE_EQ(mins[i], 10.25);
            EXPECT_DOUBLE_EQ(maxs[i], 100.75);
        }
    }
}

// Negative floats: MIN/MAX must use IEEE compare, not int64 bit compare (a
// negative double's sign bit makes it the int64-largest pattern).
TEST(BoltGroupbyTyped, Float64NegativesMinMax) {
    Arena a;
    int64_t ks[] = {1, 1, 1};
    double  vs[] = {-2.0, -10.0, 5.0};
    BoltColumn key = BoltColumn::make_flat(ks, nullptr, 3, BoltType::Int64);
    BoltColumn val = BoltColumn::make_flat(vs, nullptr, 3, BoltType::Float64);
    AggSpec specs[2] = { make_spec(AggKind::Min, 0), make_spec(AggKind::Max, 0) };
    BoltColumn ok[1], oa[2];
    uint32_t ng = 0;
    ASSERT_TRUE(groupby_agg_multi_key_typed(&key, 1, &val, 1, specs, 2, 3,
                                            ok, oa, &ng, &a, /*hint=*/4));
    EXPECT_EQ(ng, 1u);
    EXPECT_DOUBLE_EQ(static_cast<double*>(oa[0].data)[0], -10.0);
    EXPECT_DOUBLE_EQ(static_cast<double*>(oa[1].data)[0], 5.0);
}

// Large-N single-group SUM/AVG exercises the two-pass columnar path (gid
// materialisation + gb_p2_generic → apply) rather than the per-row fallback.
TEST(BoltGroupbyTyped, Float64LargeNTwoPassSum) {
    Arena a;
    constexpr int64_t N = 4096;
    int64_t* ks = a.allocate_array<int64_t>(N);
    double*  vs = a.allocate_array<double>(N);
    ASSERT_NE(ks, nullptr);
    ASSERT_NE(vs, nullptr);
    double expect = 0.0;
    for (int64_t i = 0; i < N; ++i) { ks[i] = 1; vs[i] = 0.25 * static_cast<double>(i); expect += vs[i]; }
    BoltColumn key = BoltColumn::make_flat(ks, nullptr, N, BoltType::Int64);
    BoltColumn val = BoltColumn::make_flat(vs, nullptr, N, BoltType::Float64);
    AggSpec specs[2] = { make_spec(AggKind::Sum, 0), make_spec(AggKind::Avg, 0) };
    BoltColumn ok[1], oa[2];
    uint32_t ng = 0;
    ASSERT_TRUE(groupby_agg_multi_key_typed(&key, 1, &val, 1, specs, 2, N,
                                            ok, oa, &ng, &a, /*hint=*/4));
    EXPECT_EQ(ng, 1u);
    EXPECT_DOUBLE_EQ(static_cast<double*>(oa[0].data)[0], expect);
    EXPECT_DOUBLE_EQ(static_cast<double*>(oa[1].data)[0], expect / static_cast<double>(N));
}

// Cross-morsel partial merge over Float64 — exercises gb_merge_combine's
// apply() add arm on float partial cells.
TEST(BoltGroupbyTypedXMorsel, Float64TwoMorselsSumMerge) {
    bolt::Arena a;
    bolt::BoltColumn key_desc = bolt::BoltColumn::make_flat(static_cast<int64_t*>(nullptr), nullptr, 0, bolt::BoltType::Int64);
    bolt::BoltColumn val_desc = bolt::BoltColumn::make_flat(static_cast<double*>(nullptr), nullptr, 0, bolt::BoltType::Float64);
    bolt::AggSpec spec{}; spec.kind = bolt::AggKind::Sum; spec.in_col = 0; spec.distinct = 0;
    bolt::GroupbyTypedState st{};
    ASSERT_TRUE(bolt::groupby_agg_multi_key_typed_begin(
        &st, &a, &key_desc, 1, &val_desc, 1, &spec, 1, 4));
    int64_t ks1[] = {1, 2};
    double  v1[]  = {1.5, 10.0};
    int64_t ks2[] = {1, 2};
    double  v2[]  = {2.5, 20.0};
    bolt::BoltColumn k1 = bolt::BoltColumn::make_flat(ks1, nullptr, 2, bolt::BoltType::Int64);
    bolt::BoltColumn V1 = bolt::BoltColumn::make_flat(v1, nullptr, 2, bolt::BoltType::Float64);
    bolt::BoltColumn k2 = bolt::BoltColumn::make_flat(ks2, nullptr, 2, bolt::BoltType::Int64);
    bolt::BoltColumn V2 = bolt::BoltColumn::make_flat(v2, nullptr, 2, bolt::BoltType::Float64);
    bolt::groupby_agg_multi_key_typed_ingest(&st, &k1, &V1, nullptr, 0, 2);
    bolt::groupby_agg_multi_key_typed_ingest(&st, &k2, &V2, nullptr, 0, 2);
    ASSERT_FALSE(st.oom);
    bolt::BoltColumn ok[1], oa[1];
    uint32_t ng = 0;
    ASSERT_TRUE(bolt::groupby_agg_multi_key_typed_finalize(&st, ok, oa, &ng));
    EXPECT_EQ(ng, 2u);
    int64_t* okp = static_cast<int64_t*>(ok[0].data);
    double*  oap = static_cast<double*>(oa[0].data);
    for (uint32_t i = 0; i < ng; ++i) {
        if (okp[i] == 1) EXPECT_DOUBLE_EQ(oap[i], 4.0);
        if (okp[i] == 2) EXPECT_DOUBLE_EQ(oap[i], 30.0);
    }
}

// ---------------------------------------------------------------------------
// K-AGG-B: spilled (>12-byte) Utf8 GROUP BY keys through the ONE-SHOT
// groupby_agg_multi_key_typed. Grouping/hash/equality already resolve
// spilled keys by content, but the one-shot emit path must ALSO hand the
// output key column a str_overflow_base that owns the spilled bytes — the
// stored StringView's ref.offset is meaningless without a base, and a
// nullptr base dereferenced as `base + offset` is the exact base-anchor
// crash class (G2FEAT-283/295/307). These tests fail by construction
// against the pre-fix kernel (out_keys[k].str_overflow_base == nullptr).
// ---------------------------------------------------------------------------

// Build a spilled StringView (length > 12) referencing `base + off`.
static StringView sv_spilled(const char* base, uint32_t off, uint32_t len) {
    StringView v{};
    v.length = len;
    std::memcpy(v.prefix, base + off, 4);
    v.ref.buf_idx = 0;
    v.ref.offset  = off;
    return v;
}

// Build an inline StringView (length <= 12): prefix[0..4) + inline_data[0..8).
static StringView sv_inl(const char* s, uint32_t len) {
    StringView v{};
    v.length = len;
    const uint32_t p = (len < 4u) ? len : 4u;
    if (p > 0) std::memcpy(v.prefix, s, p);
    if (len > 4u) std::memcpy(v.inline_data, s + 4, len - 4u);
    return v;
}

// Resolve one output group key's content bytes as a std::string, via the
// output column's own str_overflow_base (inline views ignore the base).
static std::string resolve_key(const BoltColumn& col, uint32_t row) {
    StringView sv;
    std::memcpy(&sv, &static_cast<const StringView*>(col.data)[row], 16);
    const char* b = static_cast<const char*>(col.str_overflow_base);
    const char* p = bolt::kernels::utf8::sv_bytes(sv, b);
    return std::string(p, sv.length);
}

// Shared body: run the one-shot grouper over a Utf8 key column mixing two
// spilled keys and one inline key (with duplicates) and verify grouping,
// per-group SUM/COUNT, and — the K-AGG-B fix — that the output key column
// owns its spilled bytes and every group key resolves to the right string.
static void check_spilled_utf8_key_grouping(const BoltColumn& key_col,
                                            const int64_t* vals, int64_t n) {
    Arena a;
    BoltColumn val = BoltColumn::make_flat(const_cast<int64_t*>(vals), nullptr,
                                           n, BoltType::Int64);
    AggSpec specs[2] = {make_spec(AggKind::Sum, 0),
                        make_spec(AggKind::CountStar, 0)};
    BoltColumn ok[1], oa[2];
    uint32_t ng = 0;
    ASSERT_TRUE(groupby_agg_multi_key_typed(&key_col, 1, &val, 1, specs, 2, n,
                                            ok, oa, &ng, &a, /*hint=*/8));
    EXPECT_EQ(ng, 3u);
    EXPECT_EQ(ok[0].type, BoltType::Utf8);

    // Per-group aggregates verified WITHOUT touching the spilled base, so
    // this part passes both pre- and post-fix (grouping was already correct).
    const int64_t* sums = static_cast<const int64_t*>(oa[0].data);
    const int64_t* cnts = static_cast<const int64_t*>(oa[1].data);
    // Expected (sum,count) multiset: A=(40,2) B=(20,1) C=(15,3).
    int seen_A = 0, seen_B = 0, seen_C = 0;
    for (uint32_t i = 0; i < ng; ++i) {
        if (sums[i] == 40 && cnts[i] == 2) ++seen_A;
        else if (sums[i] == 20 && cnts[i] == 1) ++seen_B;
        else if (sums[i] == 15 && cnts[i] == 3) ++seen_C;
        else ADD_FAILURE() << "unexpected group sum=" << sums[i]
                           << " count=" << cnts[i];
    }
    EXPECT_EQ(seen_A, 1);
    EXPECT_EQ(seen_B, 1);
    EXPECT_EQ(seen_C, 1);

    // K-AGG-B fix: spilled group keys are present, so the output key column
    // MUST own a non-null str_overflow_base. ASSERT (not EXPECT) so a
    // pre-fix run returns here instead of dereferencing a nullptr base.
    ASSERT_NE(ok[0].str_overflow_base, nullptr)
        << "output Utf8 key column lacks a spilled-byte base — spilled group "
           "keys dangle";

    // Every group key resolves to the correct content bytes via the output
    // column's own base — proves the deep copy is correct (not just non-null).
    bool got_A = false, got_B = false, got_C = false;
    for (uint32_t i = 0; i < ng; ++i) {
        const std::string k = resolve_key(ok[0], i);
        if (k == "alpha_long_key_0001") { got_A = true; EXPECT_EQ(sums[i], 40); EXPECT_EQ(cnts[i], 2); }
        else if (k == "beta_longer_key_0002") { got_B = true; EXPECT_EQ(sums[i], 20); EXPECT_EQ(cnts[i], 1); }
        else if (k == "short") { got_C = true; EXPECT_EQ(sums[i], 15); EXPECT_EQ(cnts[i], 3); }
        else ADD_FAILURE() << "unresolved/garbled group key: '" << k << "'";
    }
    EXPECT_TRUE(got_A);
    EXPECT_TRUE(got_B);
    EXPECT_TRUE(got_C);
}

TEST(BoltGroupbyTyped, SpilledUtf8KeyGroupsAndResolvesFromOwnBase) {
    // Overflow buffer holding the two spilled (>12B) key strings.
    static char ovf[64];
    std::memset(ovf, 0, sizeof(ovf));
    std::memcpy(ovf + 0,  "alpha_long_key_0001", 19);   // offset 0,  len 19
    std::memcpy(ovf + 24, "beta_longer_key_0002", 20);  // offset 24, len 20
    StringView keys[6] = {
        sv_spilled(ovf, 0, 19),    // A
        sv_inl("short", 5),        // C
        sv_spilled(ovf, 24, 20),   // B
        sv_spilled(ovf, 0, 19),    // A (dup)
        sv_inl("short", 5),        // C (dup)
        sv_inl("short", 5),        // C (dup)
    };
    int64_t vals[6] = {10, 5, 20, 30, 5, 5};   // A:10+30=40, C:5+5+5=15, B:20
    BoltColumn key = BoltColumn::make_flat(keys, nullptr, 6, BoltType::Utf8);
    key.str_overflow_base = ovf;
    check_spilled_utf8_key_grouping(key, vals, 6);
}

// SLICE case: the key column's data pointer starts PAST row 0 of a larger
// StringView array while str_overflow_base points at the full overflow
// buffer. Spilled ref.offsets are relative to that full base — a fix that
// assumed the base begins at the slice's first row (offset 0) would resolve
// garbage. The two leading padding views deliberately differ in content.
TEST(BoltGroupbyTyped, SpilledUtf8KeySlicedColumnResolvesCorrectly) {
    static char ovf[96];
    std::memset(ovf, 0, sizeof(ovf));
    std::memcpy(ovf + 0,  "PADPADPADPADPADPAD00", 20);   // padding row's key
    std::memcpy(ovf + 32, "alpha_long_key_0001", 19);
    std::memcpy(ovf + 56, "beta_longer_key_0002", 20);
    // Full array of 8; real keys live at [2..8).
    StringView full[8] = {
        sv_spilled(ovf, 0, 20),    // padding row 0 (not scanned)
        sv_inl("pad", 3),          // padding row 1 (not scanned)
        sv_spilled(ovf, 32, 19),   // A
        sv_inl("short", 5),        // C
        sv_spilled(ovf, 56, 20),   // B
        sv_spilled(ovf, 32, 19),   // A (dup)
        sv_inl("short", 5),        // C (dup)
        sv_inl("short", 5),        // C (dup)
    };
    int64_t vals[6] = {10, 5, 20, 30, 5, 5};
    BoltColumn key = BoltColumn::make_flat(full + 2, nullptr, 6, BoltType::Utf8);
    key.str_overflow_base = ovf;   // full base; offsets are absolute into it
    check_spilled_utf8_key_grouping(key, vals, 6);
}

}  // namespace
