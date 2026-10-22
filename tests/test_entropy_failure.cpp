#include "bolt/crypto/random.h"

// Use the same native declaration as random.cpp. The target renames
// getentropy in this header as well as at the production call site.
#if defined(__APPLE__)
#include <sys/random.h>
#elif defined(__linux__)
#include <unistd.h>
#else
#error "the renamed getentropy test requires macOS or Linux"
#endif

#include <gtest/gtest.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {
uint32_t backend_calls = 0;
constexpr bool kBackendNoexcept = noexcept(::getentropy(nullptr, size_t{0}));
}

// random.cpp is compiled into this test with getentropy renamed here.
// Deliberately dirty only part of the buffer before reporting OS failure.
extern "C" int bolt_entropy_test_backend(void* output, size_t size) noexcept(kBackendNoexcept) {
    ++backend_calls;
    std::memset(output, 0xEF, (size + 1) / 2);
    return -1;
}

TEST(EntropyFailure, ClearsEntireRequestedRangeAndPreservesGuards) {
    constexpr uint32_t cap = bolt::crypto::kRandomBytesCap;
    std::array<uint8_t, cap + 2> output;
    backend_calls = 0;
    for (uint32_t size = 1; size <= cap; ++size) {
        output.fill(0xA5);
        EXPECT_FALSE(bolt::crypto::random_bytes(output.data() + 1, size));
        EXPECT_EQ(backend_calls, size);
        EXPECT_EQ(output[0], 0xA5);
        for (uint32_t i = 1; i <= size; ++i) EXPECT_EQ(output[i], 0);
        for (uint32_t i = size + 1; i < output.size(); ++i) {
            EXPECT_EQ(output[i], 0xA5);
        }
    }
}

TEST(EntropyFailure, InvalidRequestsNeverCallBackendOrTouchOutput) {
    std::array<uint8_t, bolt::crypto::kRandomBytesCap + 1> output;
    output.fill(0xA5);
    backend_calls = 0;
    EXPECT_FALSE(bolt::crypto::random_bytes(nullptr, 32));
    EXPECT_FALSE(bolt::crypto::random_bytes(output.data(), 0));
    EXPECT_FALSE(bolt::crypto::random_bytes(
        output.data(), static_cast<uint32_t>(output.size())));
    EXPECT_EQ(backend_calls, 0u);
    for (const uint8_t byte : output) EXPECT_EQ(byte, 0xA5);
}
