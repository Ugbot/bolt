// test_bolt_var_kernels.cpp — coverage for Layer 1.2:
//   * VarBinary string kernels (`bolt_string.h::varbinary_*`)
//   * String/VarBinary 8-byte prefix zone-map (`bolt_zonemap.h::zone_*_str8`)

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "bolt/bolt_zonemap.h"
#include "bolt/kernels/bolt_string.h"

using bolt::ZoneMap;
using bolt::zone_make_empty_str8;
using bolt::zone_str8_observe;
using bolt::zone_can_have_eq_str8;
using bolt::zone_can_have_prefix_str8;
using bolt::kernels::varbinary_equals;
using bolt::kernels::varbinary_starts_with;
using bolt::kernels::varbinary_contains;
using bolt::kernels::varbinary_length_in_range;
using bolt::kernels::varbinary_lengths;

namespace {

struct VarPayload {
    std::vector<uint8_t> data;
    std::vector<int32_t> offsets;  // length n+1
};

VarPayload pack(const std::vector<std::string>& rows) {
    VarPayload p;
    p.offsets.reserve(rows.size() + 1);
    p.offsets.push_back(0);
    for (const auto& r : rows) {
        p.data.insert(p.data.end(), r.begin(), r.end());
        p.offsets.push_back(static_cast<int32_t>(p.data.size()));
    }
    return p;
}

}  // namespace

TEST(VarBinaryKernels, EqualsExactMatch) {
    auto p = pack({"alpha", "bravo", "charlie", "alpha", ""});
    int32_t sel[5];
    const uint8_t needle[] = {'a','l','p','h','a'};
    int64_t n = varbinary_equals(p.data.data(), p.offsets.data(), 5,
                                  needle, 5, sel);
    ASSERT_EQ(n, 2);
    EXPECT_EQ(sel[0], 0);
    EXPECT_EQ(sel[1], 3);
}

TEST(VarBinaryKernels, EqualsEmptyNeedle) {
    auto p = pack({"", "x", "", "yz"});
    int32_t sel[4];
    int64_t n = varbinary_equals(p.data.data(), p.offsets.data(), 4,
                                  reinterpret_cast<const uint8_t*>(""), 0, sel);
    ASSERT_EQ(n, 2);
    EXPECT_EQ(sel[0], 0);
    EXPECT_EQ(sel[1], 2);
}

TEST(VarBinaryKernels, StartsWith) {
    auto p = pack({"foo", "foobar", "fo", "barfoo", "foo"});
    int32_t sel[5];
    const uint8_t pfx[] = {'f','o','o'};
    int64_t n = varbinary_starts_with(p.data.data(), p.offsets.data(), 5,
                                       pfx, 3, sel);
    ASSERT_EQ(n, 3);
    EXPECT_EQ(sel[0], 0);
    EXPECT_EQ(sel[1], 1);
    EXPECT_EQ(sel[2], 4);
}

TEST(VarBinaryKernels, ContainsSubstring) {
    auto p = pack({"red apple", "blueberry", "banana", "pineapple", ""});
    int32_t sel[5];
    const uint8_t needle[] = {'a','p','p','l','e'};
    int64_t n = varbinary_contains(p.data.data(), p.offsets.data(), 5,
                                    needle, 5, sel);
    ASSERT_EQ(n, 2);
    EXPECT_EQ(sel[0], 0);  // "red apple"
    EXPECT_EQ(sel[1], 3);  // "pineapple"
}

TEST(VarBinaryKernels, ContainsEmptyNeedleMatchesAll) {
    auto p = pack({"a", "", "bc"});
    int32_t sel[3];
    int64_t n = varbinary_contains(p.data.data(), p.offsets.data(), 3,
                                    reinterpret_cast<const uint8_t*>(""), 0, sel);
    EXPECT_EQ(n, 3);
}

TEST(VarBinaryKernels, LengthRange) {
    auto p = pack({"a", "bb", "ccc", "dddd", "eeeee"});
    int32_t sel[5];
    int64_t n = varbinary_length_in_range(p.offsets.data(), 5, 2, 4, sel);
    ASSERT_EQ(n, 3);
    EXPECT_EQ(sel[0], 1);
    EXPECT_EQ(sel[1], 2);
    EXPECT_EQ(sel[2], 3);
}

TEST(VarBinaryKernels, Lengths) {
    auto p = pack({"a", "bb", "ccc", ""});
    int32_t lens[4] = {0};
    varbinary_lengths(p.offsets.data(), 4, lens);
    EXPECT_EQ(lens[0], 1);
    EXPECT_EQ(lens[1], 2);
    EXPECT_EQ(lens[2], 3);
    EXPECT_EQ(lens[3], 0);
}

// ---------------------------------------------------------------------------
// Str8 zone-map.
// ---------------------------------------------------------------------------

TEST(ZoneMapStr8, EmptyZoneRejectsAll) {
    ZoneMap z = zone_make_empty_str8();
    const uint8_t q[] = {'a'};
    EXPECT_FALSE(zone_can_have_eq_str8(&z, q, 1));
}

TEST(ZoneMapStr8, ObserveSetsMinAndMax) {
    ZoneMap z = zone_make_empty_str8();
    const uint8_t a[] = "carrot";
    const uint8_t b[] = "alpha";
    const uint8_t c[] = "delta";
    zone_str8_observe(&z, a, 6);
    zone_str8_observe(&z, b, 5);
    zone_str8_observe(&z, c, 5);

    // "alpha" inside [alpha, delta] → possible.
    EXPECT_TRUE (zone_can_have_eq_str8(&z, a, 6));
    EXPECT_TRUE (zone_can_have_eq_str8(&z, b, 5));
    EXPECT_TRUE (zone_can_have_eq_str8(&z, c, 5));

    // "zoo" past max → skip.
    const uint8_t zoo[] = "zoo";
    EXPECT_FALSE(zone_can_have_eq_str8(&z, zoo, 3));

    // "aaa" before min → skip.
    const uint8_t aaa[] = "aaa";
    EXPECT_FALSE(zone_can_have_eq_str8(&z, aaa, 3));
}

TEST(ZoneMapStr8, PrefixPredicatePadsCorrectly) {
    ZoneMap z = zone_make_empty_str8();
    const uint8_t a[] = "betagamma";  // 9 bytes — prefix "betagamm"
    const uint8_t b[] = "deltaone";
    zone_str8_observe(&z, a, 9);
    zone_str8_observe(&z, b, 8);

    // Prefix "beta" matches a row → possible.
    const uint8_t beta[] = "beta";
    EXPECT_TRUE (zone_can_have_prefix_str8(&z, beta, 4));

    // Prefix "alpha" — entirely below "betagamm" → skip.
    const uint8_t alpha[] = "alpha";
    EXPECT_FALSE(zone_can_have_prefix_str8(&z, alpha, 5));

    // Prefix "zoo" — entirely above "deltaone" → skip.
    const uint8_t zoo[] = "zoo";
    EXPECT_FALSE(zone_can_have_prefix_str8(&z, zoo, 3));

    // Prefix "c" sits between "beta..." and "delta..." → possible (no
    // row may *actually* exist, but predicate is conservative).
    const uint8_t c[] = "c";
    EXPECT_TRUE (zone_can_have_prefix_str8(&z, c, 1));
}
