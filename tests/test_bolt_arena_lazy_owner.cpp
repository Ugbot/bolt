// test_bolt_arena_lazy_owner.cpp — R0: a default arena holds nothing until
// used, starts small, and every backing block is charged to its MemOwner.
//
// Every default bolt::Arena used to malloc a 4 MiB first block in its
// constructor; the talk node held 407 of them (1.6 GiB reserved, 0.5 GiB
// dirty). These tests pin the replacement: lazy, kArenaLazyFirstBlockBytes,
// with explicit configs and frozen arenas keeping their old capacity.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_mem_owner.h"

namespace {

TEST(ArenaLazyFirstBlock, DefaultArenaHoldsNothingUntilFirstAllocate) {
    bolt::MemOwner owner;
    const std::uint64_t total0 = bolt::g_arena_total_owner.reserved.load();
    {
        bolt::ArenaConfig cfg{};
        cfg.owner = &owner;
        bolt::Arena a{cfg};
        EXPECT_EQ(a.num_blocks(), 0u);
        EXPECT_EQ(a.total_reserved(), 0u);
        EXPECT_EQ(owner.reserved.load(), 0u);
        EXPECT_EQ(owner.arenas.load(), 1u);
        a.reset();                      // no blocks: still fine
        a.reset_keep(0);
        a.compact();
        EXPECT_EQ(a.num_blocks(), 0u);

        void* p = a.allocate(100, 64);
        ASSERT_NE(p, nullptr);
        EXPECT_EQ(a.num_blocks(), 1u);
        EXPECT_EQ(a.total_reserved(), bolt::kArenaLazyFirstBlockBytes);
        EXPECT_EQ(owner.reserved.load(), bolt::kArenaLazyFirstBlockBytes);
        EXPECT_EQ(owner.blocks.load(), 1u);
        EXPECT_GE(bolt::g_arena_total_owner.reserved.load(),
                  total0 + bolt::kArenaLazyFirstBlockBytes);

        // Doubling continues from the small block, and the owner follows.
        for (int i = 0; i < 64; ++i) ASSERT_NE(a.allocate(32 * 1024, 64), nullptr);
        EXPECT_GT(a.num_blocks(), 1u);
        EXPECT_EQ(owner.reserved.load(), a.total_reserved());
        EXPECT_EQ(owner.blocks.load(), a.num_blocks());

        a.reset_keep(bolt::kArenaLazyFirstBlockBytes);
        EXPECT_EQ(a.num_blocks(), 1u);
        EXPECT_EQ(owner.reserved.load(), a.total_reserved());
    }
    EXPECT_EQ(owner.reserved.load(), 0u);
    EXPECT_EQ(owner.blocks.load(), 0u);
    EXPECT_EQ(owner.arenas.load(), 0u);
}

TEST(ArenaLazyFirstBlock, ZeroByteAllocateOnEmptyArenaIsNotNull) {
    bolt::Arena a;
    EXPECT_NE(a.allocate(0), nullptr);
    EXPECT_EQ(a.num_blocks(), 1u);
}

TEST(ArenaLazyFirstBlock, OversizeFirstAllocationThenSmallOnes) {
    bolt::ArenaConfig cfg{};
    cfg.max_block_size = 1024 * 1024;
    bolt::Arena a{cfg};
    auto* big = static_cast<unsigned char*>(a.allocate(4 * 1024 * 1024, 64));
    ASSERT_NE(big, nullptr);
    std::memset(big, 1, 4 * 1024 * 1024);
    auto* small = static_cast<unsigned char*>(a.allocate(128, 64));
    ASSERT_NE(small, nullptr);
    std::memset(small, 2, 128);
    EXPECT_EQ(big[4 * 1024 * 1024 - 1], 1);
    EXPECT_FALSE(small >= big && small < big + 4 * 1024 * 1024);
    a.reset();
    EXPECT_NE(a.allocate(128, 64), nullptr);
}

TEST(ArenaLazyFirstBlock, ExplicitFirstBlockStaysEager) {
    bolt::ArenaConfig cfg{};
    cfg.initial_block_size = 256 * 1024;
    bolt::Arena a{cfg};
    EXPECT_EQ(a.num_blocks(), 1u);
    EXPECT_EQ(a.total_reserved(), 256u * 1024u);
}

TEST(ArenaLazyFirstBlock, FreezingADefaultArenaKeepsTheLegacyCapacity) {
    bolt::Arena fresh;
    fresh.set_growth_enabled(false);
    EXPECT_GE(fresh.largest_block(), bolt::kArenaLegacyFirstBlockBytes);
    EXPECT_NE(fresh.allocate(3 * 1024 * 1024, 64), nullptr);

    bolt::Arena used;
    auto* p = static_cast<unsigned char*>(used.allocate(1000, 64));
    ASSERT_NE(p, nullptr);
    std::memset(p, 7, 1000);
    used.set_growth_enabled(false);
    EXPECT_GE(used.largest_block(), bolt::kArenaLegacyFirstBlockBytes);
    auto* q = static_cast<unsigned char*>(used.allocate(3 * 1024 * 1024, 64));
    ASSERT_NE(q, nullptr);
    EXPECT_EQ(p[999], 7);
}

TEST(MemOwnerRegistry, RegisteredOwnersAreListed) {
    static bolt::MemOwner owner;
    ASSERT_TRUE(bolt::mem_owner_register("test.lazy_owner", &owner));
    bolt::MemOwnerEntry list[bolt::kMemOwnerRegistryCap];
    const std::uint32_t n = bolt::mem_owner_list(list, bolt::kMemOwnerRegistryCap);
    bool found = false;
    for (std::uint32_t i = 0; i < n; ++i) {
        if (list[i].owner == &owner && std::strcmp(list[i].name, "test.lazy_owner") == 0) found = true;
    }
    EXPECT_TRUE(found);
}

}  // namespace
