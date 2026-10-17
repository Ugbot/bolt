#include "bolt/crypto/random.h"

#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>

TEST(Entropy, RejectsInvalidRequestsWithoutTouchingOutput) {
    std::array<uint8_t, bolt::crypto::kRandomBytesCap + 1> output;
    output.fill(0xA5);
    EXPECT_FALSE(bolt::crypto::random_bytes(nullptr, 32));
    EXPECT_FALSE(bolt::crypto::random_bytes(nullptr, 0));
    EXPECT_FALSE(bolt::crypto::random_bytes(output.data(), 0));
    EXPECT_FALSE(bolt::crypto::random_bytes(
        output.data(), static_cast<uint32_t>(output.size())));
    EXPECT_FALSE(bolt::crypto::random_bytes(
        output.data(), std::numeric_limits<uint32_t>::max()));
    for (const uint8_t byte : output) EXPECT_EQ(byte, 0xA5);
}

TEST(Entropy, FillsEverySupportedRequestSizeWithinBounds) {
    constexpr uint32_t cap = bolt::crypto::kRandomBytesCap;
    std::array<uint8_t, cap + 2> output;
    for (uint32_t size = 1; size <= cap; ++size) {
        output.fill(0xA5);
        ASSERT_TRUE(bolt::crypto::random_bytes(output.data() + 1, size));
        EXPECT_EQ(output[0], 0xA5);
        for (uint32_t i = size + 1; i < output.size(); ++i) {
            EXPECT_EQ(output[i], 0xA5);
        }
    }
}

TEST(Entropy, IndependentFullRequestsDiffer) {
    std::array<uint8_t, bolt::crypto::kRandomBytesCap> first{};
    std::array<uint8_t, bolt::crypto::kRandomBytesCap> second{};
    ASSERT_TRUE(bolt::crypto::random_bytes(first.data(), bolt::crypto::kRandomBytesCap));
    ASSERT_TRUE(bolt::crypto::random_bytes(second.data(), bolt::crypto::kRandomBytesCap));
    EXPECT_NE(std::memcmp(first.data(), second.data(), first.size()), 0);
    const std::array<uint8_t, bolt::crypto::kRandomBytesCap> zero{};
    EXPECT_NE(first, zero);
}
