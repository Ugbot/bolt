// Tests for bolt::groupby_agg_multi_key_typed (K-AGG-A).
//
// Scope: kernel-output correctness on small typed inputs.
//   - Int64 single-key + SUM
//   - Int32 multi-key + COUNT(*)
//   - Decimal128 SUM + AVG
//   - Date32 keys
//   - Composite-key (2) hash discriminates correctly

#include "bolt/join/bolt_groupby.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

namespace {

using namespace bolt;
namespace dec = bolt::kernels::decimal;

TEST(BoltGroupbyTyped, Int64SingleKeySum) {
    Arena a;
    int64_t ks[] = {1, 2, 1, 2, 1, 3};
    int64_t vs[] = {10, 20, 30, 40, 50, 60};
    BoltColumn key = BoltColumn::make_flat(ks, nullptr, 6, BoltType::Int64);
    BoltColumn val = BoltColumn::make_flat(vs, nullptr, 6, BoltType::Int64);
    AggSpec spec; spec.kind = AggKind::Sum; spec.in_col = 0;
    BoltColumn ok[1], oa[1];
    uint32_t ng = 0;
    ASSERT_TRUE(groupby_agg_multi_key_typed(&key, 1, &val, 1, &spec, 1, 6,
                                            ok, oa, &ng, &a, /*hint=*/8));
    EXPECT_EQ(ng, 3u);
    // Find each key's sum by scanning. Order is insertion-order.
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
    AggSpec spec; spec.kind = AggKind::CountStar; spec.in_col = 0;
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
    AggSpec spec; spec.kind = AggKind::Sum; spec.in_col = 0;
    BoltColumn ok[2], oa[1];
    uint32_t ng = 0;
    ASSERT_TRUE(groupby_agg_multi_key_typed(keys, 2, &val, 1, &spec, 1, 4,
                                            ok, oa, &ng, &a, /*hint=*/4));
    // Three distinct composite keys: (1,10), (1,20), (2,10).
    // Last two rows share (2,10) so sum = 12.
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
    AggSpec spec; spec.kind = AggKind::Sum; spec.in_col = 0;
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
    AggSpec spec; spec.kind = AggKind::Sum; spec.in_col = 0;
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
    specs[0].kind = AggKind::Min; specs[0].in_col = 0;
    specs[1].kind = AggKind::Max; specs[1].in_col = 0;
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

}  // namespace
