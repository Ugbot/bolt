#include <gtest/gtest.h>
#include "bolt/bolt_arena.h"
#include <cstring>

TEST(BoltArenaFrozen, ExhaustionRefusesNormalAndOversizeGrowth) {
    bolt::Arena arena{{256, 256}};
    ASSERT_TRUE(arena.growth_enabled());
    ASSERT_EQ(arena.num_blocks(), 1u);
    arena.set_growth_enabled(false);
    auto* first = arena.allocate(224, 64);
    ASSERT_NE(first, nullptr);
    std::memset(first, 0x6b, 224);
    const auto reserved = arena.total_reserved();
    const auto allocated = arena.total_allocated();
    EXPECT_EQ(arena.allocate(64, 64), nullptr);
    EXPECT_EQ(arena.allocate(4096, 64), nullptr);
    EXPECT_EQ(arena.num_blocks(), 1u);
    EXPECT_EQ(arena.total_reserved(), reserved);
    EXPECT_EQ(arena.total_allocated(), allocated);
    for (unsigned i = 0; i < 224; ++i)
        EXPECT_EQ(static_cast<const unsigned char*>(first)[i], 0x6b);
    arena.reset();
    EXPECT_FALSE(arena.growth_enabled());
    EXPECT_EQ(arena.allocate(224, 64), first);
    EXPECT_EQ(arena.total_reserved(), reserved);
}

TEST(BoltArenaFrozen, ReusesWarmNormalAndOversizeBlocksWithoutOverlap) {
    bolt::Arena arena{{256, 256}};
    ASSERT_NE(arena.allocate(256, 64), nullptr);
    ASSERT_NE(arena.allocate(128, 64), nullptr);
    ASSERT_NE(arena.allocate(1024, 64), nullptr);
    ASSERT_EQ(arena.num_blocks(), 3u);
    const auto reserved = arena.total_reserved();
    arena.reset(); arena.set_growth_enabled(false);
    auto* a = arena.allocate(256, 64);
    auto* b = arena.allocate(128, 64);
    auto* c = arena.allocate(1024, 64);
    auto* d = arena.allocate(128, 64);
    ASSERT_NE(a, nullptr); ASSERT_NE(b, nullptr);
    ASSERT_NE(c, nullptr); ASSERT_NE(d, nullptr);
    std::memset(a, 1, 256); std::memset(b, 2, 128);
    std::memset(c, 3, 1024); std::memset(d, 4, 128);
    for (unsigned i = 0; i < 256; ++i) EXPECT_EQ(static_cast<unsigned char*>(a)[i], 1);
    for (unsigned i = 0; i < 128; ++i) EXPECT_EQ(static_cast<unsigned char*>(b)[i], 2);
    for (unsigned i = 0; i < 1024; ++i) EXPECT_EQ(static_cast<unsigned char*>(c)[i], 3);
    for (unsigned i = 0; i < 128; ++i) EXPECT_EQ(static_cast<unsigned char*>(d)[i], 4);
    EXPECT_EQ(arena.allocate(4096, 64), nullptr);
    EXPECT_EQ(arena.num_blocks(), 3u);
    EXPECT_EQ(arena.total_reserved(), reserved);
}

TEST(BoltArenaFrozen, ColdReenableAndTrimPreservePolicy) {
    bolt::Arena arena{{256, 256}};
    arena.set_growth_enabled(false);
    EXPECT_EQ(arena.allocate(1024, 64), nullptr);
    arena.set_growth_enabled(true);
    ASSERT_NE(arena.allocate(1024, 64), nullptr);
    ASSERT_EQ(arena.num_blocks(), 2u);
    arena.reset(); arena.set_growth_enabled(false); arena.compact();
    EXPECT_FALSE(arena.growth_enabled());
    EXPECT_EQ(arena.num_blocks(), 1u);
    EXPECT_EQ(arena.allocate(1024, 64), nullptr);
    EXPECT_NE(arena.allocate(128, 64), nullptr);
}
