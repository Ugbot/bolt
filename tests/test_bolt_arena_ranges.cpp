#include <gtest/gtest.h>
#include "bolt/bolt_arena.h"
#include <cstdint>
#include <limits>

namespace {
const void* address(std::uintptr_t p) { return reinterpret_cast<const void*>(p); }
TEST(BoltArenaRanges, NullZeroInteriorEdgesAndPartialBothDirections) {
    bolt::Arena arena({4096,4096});
    EXPECT_FALSE(arena.overlaps_backing(nullptr,0)); EXPECT_FALSE(arena.overlaps_backing(nullptr,1));
    auto* bytes = arena.allocate_array<std::uint8_t>(4096); ASSERT_NE(bytes,nullptr);
    const auto start = reinterpret_cast<std::uintptr_t>(bytes);
    EXPECT_FALSE(arena.overlaps_backing(bytes,0));
    EXPECT_TRUE(arena.overlaps_backing(bytes,1)); EXPECT_TRUE(arena.overlaps_backing(bytes+4095,1));
    EXPECT_TRUE(arena.overlaps_backing(bytes+31,10));
    EXPECT_FALSE(arena.overlaps_backing(address(start-1),1));
    EXPECT_TRUE(arena.overlaps_backing(address(start-1),2));
    EXPECT_FALSE(arena.overlaps_backing(bytes+4096,1));
    EXPECT_TRUE(arena.overlaps_backing(bytes+4095,2));
    EXPECT_TRUE(arena.overlaps_backing(address(start-10),4116));
}
TEST(BoltArenaRanges, RetainsUnusedResetAndOversizeBackingWithoutAllocation) {
    bolt::Arena arena({4096,8192});
    auto* first = arena.allocate_array<std::uint8_t>(16); ASSERT_NE(first,nullptr);
    // Unused bytes in the first retained block count too.
    EXPECT_TRUE(arena.overlaps_backing(first+2048,1));
    auto* second = arena.allocate_array<std::uint8_t>(8192); ASSERT_NE(second,nullptr);
    auto* oversize = arena.allocate_array<std::uint8_t>(32768); ASSERT_NE(oversize,nullptr);
    ASSERT_GE(arena.num_blocks(),3u);
    const auto reserved = arena.total_reserved(); const auto used = arena.total_allocated();
    EXPECT_TRUE(arena.overlaps_backing(first,16)); EXPECT_TRUE(arena.overlaps_backing(second,8192));
    EXPECT_TRUE(arena.overlaps_backing(oversize+32767,1));
    EXPECT_EQ(arena.total_reserved(),reserved); EXPECT_EQ(arena.total_allocated(),used);
    arena.reset();
    EXPECT_TRUE(arena.overlaps_backing(first,16)); EXPECT_TRUE(arena.overlaps_backing(second,8192));
    EXPECT_TRUE(arena.overlaps_backing(oversize,32768)); EXPECT_EQ(arena.total_allocated(),0u);
}
TEST(BoltArenaRanges, ExtremeAddressAndLengthUseDifferencesWithoutWrapping) {
    bolt::Arena arena({4096,4096});
    auto* p = arena.allocate_array<std::uint8_t>(1); ASSERT_NE(p,nullptr);
    const auto start = reinterpret_cast<std::uintptr_t>(p);
    EXPECT_FALSE(arena.overlaps_backing(address(UINTPTR_MAX),1));
    EXPECT_FALSE(arena.overlaps_backing(address(UINTPTR_MAX-1),SIZE_MAX));
    EXPECT_TRUE(arena.overlaps_backing(address(1),SIZE_MAX));
    EXPECT_TRUE(arena.overlaps_backing(p,SIZE_MAX));
    EXPECT_FALSE(arena.overlaps_backing(address(1),start-1));
    EXPECT_TRUE(arena.overlaps_backing(address(1),start));
}
} // namespace
