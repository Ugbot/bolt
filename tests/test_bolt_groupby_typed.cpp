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
}  // namespace
