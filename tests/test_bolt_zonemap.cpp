// test_bolt_zonemap.cpp — Exhaustive tests for the ZoneMap primitive
// (bolt/bolt_zonemap.h). Bolted in as part of the MarbleDB Wave 10.1
// consolidation.

#include "bolt/bolt_zonemap.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>

using bolt::ZoneMap;

TEST(BoltZoneMap, SizeAndAlignment) {
    EXPECT_EQ(sizeof(ZoneMap), 32u);
    EXPECT_GE(alignof(ZoneMap), 32u);
}

TEST(BoltZoneMap, EmptyI64Sentinels) {
    ZoneMap z = bolt::zone_make_empty_i64();
    EXPECT_EQ(z.min_value, INT64_MAX);
    EXPECT_EQ(z.max_value, INT64_MIN);
    EXPECT_TRUE(std::isnan(z.mean));
    EXPECT_TRUE(std::isnan(z.variance));
    EXPECT_EQ(z.null_count, 0u);
    EXPECT_EQ(z.distinct_count, 0u);
    EXPECT_EQ(z.cardinality_class, bolt::kZoneCardUnknown);
    EXPECT_EQ(z.flags, 0u);
}

TEST(BoltZoneMap, EmptyF32Sentinels) {
    ZoneMap z = bolt::zone_make_empty_f32();
    EXPECT_TRUE(std::isinf(bolt::zone_min_f32(&z)));
    EXPECT_GT(bolt::zone_min_f32(&z), 0.0f);   // +inf
    EXPECT_TRUE(std::isinf(bolt::zone_max_f32(&z)));
    EXPECT_LT(bolt::zone_max_f32(&z), 0.0f);   // -inf
    EXPECT_TRUE(std::isnan(z.mean));
    EXPECT_TRUE(std::isnan(z.variance));
}

TEST(BoltZoneMap, F32RoundTrip) {
    ZoneMap z = bolt::zone_make_empty_f32();
    bolt::zone_set_min_f32(&z, 1.5f);
    bolt::zone_set_max_f32(&z, 9.25f);
    EXPECT_FLOAT_EQ(bolt::zone_min_f32(&z), 1.5f);
    EXPECT_FLOAT_EQ(bolt::zone_max_f32(&z), 9.25f);
}

TEST(BoltZoneMap, CanHaveEqI64) {
    ZoneMap z = bolt::zone_make_empty_i64();
    z.min_value = 10;
    z.max_value = 100;
    EXPECT_TRUE (bolt::zone_can_have_eq_i64(&z, 10));
    EXPECT_TRUE (bolt::zone_can_have_eq_i64(&z, 50));
    EXPECT_TRUE (bolt::zone_can_have_eq_i64(&z, 100));
    EXPECT_FALSE(bolt::zone_can_have_eq_i64(&z, 9));
    EXPECT_FALSE(bolt::zone_can_have_eq_i64(&z, 101));
}

TEST(BoltZoneMap, CanHaveRangeI64) {
    ZoneMap z = bolt::zone_make_empty_i64();
    z.min_value = 10;
    z.max_value = 100;
    EXPECT_TRUE (bolt::zone_can_have_range_i64(&z, 0, 15));       // overlap left
    EXPECT_TRUE (bolt::zone_can_have_range_i64(&z, 20, 50));      // inside
    EXPECT_TRUE (bolt::zone_can_have_range_i64(&z, 95, 200));     // overlap right
    EXPECT_TRUE (bolt::zone_can_have_range_i64(&z, 5, 200));      // spans
    EXPECT_FALSE(bolt::zone_can_have_range_i64(&z, 0, 9));        // left of min
    EXPECT_FALSE(bolt::zone_can_have_range_i64(&z, 101, 200));    // right of max
}

TEST(BoltZoneMap, CanHaveEqU64) {
    ZoneMap z = bolt::zone_make_empty_i64();
    uint64_t mn = 0xFFFFFFFF00000000ull;
    uint64_t mx = 0xFFFFFFFFFFFFFFFFull;
    std::memcpy(&z.min_value, &mn, sizeof(mn));
    std::memcpy(&z.max_value, &mx, sizeof(mx));
    EXPECT_TRUE (bolt::zone_can_have_eq_u64(&z, 0xFFFFFFFF80000000ull));
    EXPECT_FALSE(bolt::zone_can_have_eq_u64(&z, 0x0000000000000001ull));
}

TEST(BoltZoneMap, CanHaveEqF32) {
    ZoneMap z = bolt::zone_make_empty_f32();
    bolt::zone_set_min_f32(&z, -1.0f);
    bolt::zone_set_max_f32(&z, 10.0f);
    EXPECT_TRUE (bolt::zone_can_have_eq_f32(&z, 0.0f));
    EXPECT_TRUE (bolt::zone_can_have_eq_f32(&z, -1.0f));
    EXPECT_TRUE (bolt::zone_can_have_eq_f32(&z, 10.0f));
    EXPECT_FALSE(bolt::zone_can_have_eq_f32(&z, -1.5f));
    EXPECT_FALSE(bolt::zone_can_have_eq_f32(&z, 10.1f));
}

TEST(BoltZoneMap, CanHaveRangeF32) {
    ZoneMap z = bolt::zone_make_empty_f32();
    bolt::zone_set_min_f32(&z, 0.0f);
    bolt::zone_set_max_f32(&z, 5.0f);
    EXPECT_TRUE (bolt::zone_can_have_range_f32(&z, -1.0f, 0.5f));
    EXPECT_TRUE (bolt::zone_can_have_range_f32(&z, 4.5f, 10.0f));
    EXPECT_FALSE(bolt::zone_can_have_range_f32(&z, 5.5f, 10.0f));
    EXPECT_FALSE(bolt::zone_can_have_range_f32(&z, -2.0f, -0.5f));
}

TEST(BoltZoneMap, DimL2UpperBound) {
    ZoneMap z = bolt::zone_make_empty_f32();
    bolt::zone_set_min_f32(&z, 1.0f);
    bolt::zone_set_max_f32(&z, 5.0f);

    // q = 3 → max((3-1)^2, (3-5)^2) = max(4, 4) = 4.
    EXPECT_FLOAT_EQ(bolt::zone_dim_l2_upper_bound_f32(&z, 3.0f), 4.0f);

    // q = 0 → max((0-1)^2, (0-5)^2) = max(1, 25) = 25.
    EXPECT_FLOAT_EQ(bolt::zone_dim_l2_upper_bound_f32(&z, 0.0f), 25.0f);

    // q = 10 → max((10-1)^2, (10-5)^2) = max(81, 25) = 81.
    EXPECT_FLOAT_EQ(bolt::zone_dim_l2_upper_bound_f32(&z, 10.0f), 81.0f);

    // q exactly at min → still safe.
    EXPECT_FLOAT_EQ(bolt::zone_dim_l2_upper_bound_f32(&z, 1.0f), 16.0f);
}

TEST(BoltZoneMap, MergeI64) {
    ZoneMap a = bolt::zone_make_empty_i64();
    a.min_value = 10; a.max_value = 100;
    a.null_count = 2; a.distinct_count = 5;
    a.flags = bolt::kZoneFlagSortedAsc | bolt::kZoneFlagAllValid;

    ZoneMap b = bolt::zone_make_empty_i64();
    b.min_value = 50; b.max_value = 200;
    b.null_count = 3; b.distinct_count = 7;
    b.flags = bolt::kZoneFlagSortedAsc;  // not all_valid

    bolt::zone_merge_i64(&a, &b);
    EXPECT_EQ(a.min_value, 10);
    EXPECT_EQ(a.max_value, 200);
    EXPECT_EQ(a.null_count, 5u);
    EXPECT_EQ(a.distinct_count, 0u);   // conservative reset
    // sorted_asc keeps (both had it), all_valid drops.
    EXPECT_TRUE (a.flags & bolt::kZoneFlagSortedAsc);
    EXPECT_FALSE(a.flags & bolt::kZoneFlagAllValid);
}

TEST(BoltZoneMap, MergeF32) {
    ZoneMap a = bolt::zone_make_empty_f32();
    bolt::zone_set_min_f32(&a, -5.0f);
    bolt::zone_set_max_f32(&a, 10.0f);

    ZoneMap b = bolt::zone_make_empty_f32();
    bolt::zone_set_min_f32(&b, -20.0f);
    bolt::zone_set_max_f32(&b, 3.0f);

    bolt::zone_merge_f32(&a, &b);
    EXPECT_FLOAT_EQ(bolt::zone_min_f32(&a), -20.0f);
    EXPECT_FLOAT_EQ(bolt::zone_max_f32(&a), 10.0f);
}
