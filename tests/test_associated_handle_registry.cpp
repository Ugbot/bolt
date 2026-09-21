#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include "../src/api/core/associated_handle_registry.h"

using bolt::api::core::AssociatedHandleRegistry;
using bolt::api::core::AssociationResult;

TEST(AssociatedHandleRegistry, PreservesFullWidthKeysAndAssociatesOnce) {
    AssociatedHandleRegistry registry;
    constexpr uint64_t kLowKey = 0x0000000087654321ULL;
    constexpr uint64_t kHighKey = 0xFEDCBA9887654321ULL;
    uint32_t calls = 0;

    EXPECT_EQ(registry.ensure(kLowKey, [&]() noexcept { ++calls; return true; }),
              AssociationResult::associated);
    EXPECT_EQ(registry.ensure(kHighKey, [&]() noexcept { ++calls; return true; }),
              AssociationResult::associated);
    EXPECT_EQ(registry.ensure(kHighKey, [&]() noexcept { ++calls; return true; }),
              AssociationResult::already_associated);

    EXPECT_EQ(calls, 2u);
    EXPECT_EQ(registry.size(), 2u);
    EXPECT_TRUE(registry.contains(kLowKey));
    EXPECT_TRUE(registry.contains(kHighKey));
}

TEST(AssociatedHandleRegistry, RollsBackPlatformFailureAndAllowsRetry) {
    AssociatedHandleRegistry registry;
    constexpr uint64_t kKey = 0xABCDEF0123456789ULL;
    uint32_t calls = 0;

    EXPECT_EQ(registry.ensure(kKey, [&]() noexcept { ++calls; return false; }),
              AssociationResult::platform_failed);
    EXPECT_EQ(registry.size(), 0u);
    EXPECT_FALSE(registry.contains(kKey));

    EXPECT_EQ(registry.ensure(kKey, [&]() noexcept { ++calls; return true; }),
              AssociationResult::associated);
    EXPECT_EQ(calls, 2u);
    EXPECT_TRUE(registry.contains(kKey));

    EXPECT_TRUE(registry.forget(kKey));
    EXPECT_FALSE(registry.contains(kKey));
    EXPECT_EQ(registry.size(), 0u);
}

TEST(AssociatedHandleRegistry, FailedCloseRetainsAssociationUntilSuccess) {
    AssociatedHandleRegistry registry;
    constexpr uint64_t kKey = 0xABCDEF0123456789ULL;
    uint32_t close_calls = 0;

    ASSERT_EQ(registry.ensure(kKey, []() noexcept { return true; }),
              AssociationResult::associated);
    EXPECT_FALSE(registry.close_and_forget(kKey, [&]() noexcept {
        ++close_calls;
        return false;
    }));
    EXPECT_EQ(close_calls, 1u);
    EXPECT_TRUE(registry.contains(kKey));
    EXPECT_EQ(registry.size(), 1u);

    EXPECT_TRUE(registry.close_and_forget(kKey, [&]() noexcept {
        ++close_calls;
        return true;
    }));
    EXPECT_EQ(close_calls, 2u);
    EXPECT_FALSE(registry.contains(kKey));
    EXPECT_EQ(registry.size(), 0u);
}

TEST(FdRegistry, InitialAllocationFailureIsSafe) {
    bolt::ArenaConfig config;
    config.initial_block_size = std::numeric_limits<size_t>::max();
    config.max_block_size = std::numeric_limits<size_t>::max();
    config.alignment = 64u;

    bolt::api::net::FdRegistry<uint8_t> registry(config);
    EXPECT_EQ(registry.find(7u), nullptr);
    EXPECT_EQ(registry.insert(7u), nullptr);
    EXPECT_FALSE(registry.erase(7u));
    EXPECT_EQ(registry.size(), 0u);
}

TEST(AssociatedHandleRegistry, RefusesPastCapacityBeforePlatformCall) {
    AssociatedHandleRegistry registry;
    uint32_t platform_calls = 0;

    for (uint32_t i = 0; i < AssociatedHandleRegistry::kCapacity; ++i) {
        const AssociationResult result = registry.ensure(
            static_cast<uint64_t>(i),
            [&]() noexcept { ++platform_calls; return true; });
        ASSERT_EQ(result, AssociationResult::associated) << "slot " << i;
    }
    ASSERT_EQ(registry.size(), AssociatedHandleRegistry::kCapacity);
    ASSERT_EQ(platform_calls, AssociatedHandleRegistry::kCapacity);

    EXPECT_EQ(registry.ensure(
                  17u, [&]() noexcept { ++platform_calls; return true; }),
              AssociationResult::already_associated);
    EXPECT_EQ(platform_calls, AssociatedHandleRegistry::kCapacity);

    EXPECT_EQ(registry.ensure(
                  static_cast<uint64_t>(AssociatedHandleRegistry::kCapacity),
                  [&]() noexcept { ++platform_calls; return true; }),
              AssociationResult::registry_refused);
    EXPECT_EQ(platform_calls, AssociatedHandleRegistry::kCapacity);

    ASSERT_TRUE(registry.forget(17u));
    EXPECT_EQ(registry.ensure(
                  static_cast<uint64_t>(AssociatedHandleRegistry::kCapacity),
                  [&]() noexcept { ++platform_calls; return true; }),
              AssociationResult::associated);
    EXPECT_EQ(platform_calls, AssociatedHandleRegistry::kCapacity + 1u);
}
