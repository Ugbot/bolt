#include "bolt/crypto/sha256.h"
#include "bolt/crypto/sigv4.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstring>

namespace {
using namespace bolt::crypto;
std::array<uint8_t, 32> unhex(const char* text) {
    std::array<uint8_t, 32> result{};
    const auto digit = [](char ch) { return ch <= '9' ? ch - '0' : ch - 'a' + 10; };
    for (uint32_t i = 0; i < 32; ++i)
        result[i] = static_cast<uint8_t>(16 * digit(text[2 * i]) + digit(text[2 * i + 1]));
    return result;
}
TEST(Sha256Stream, IndependentEmptyAndAbcKnownAnswers) {
    struct Answer { const char* input; const char* digest; };
    constexpr Answer answers[]{
        {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
        {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
         "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"}};
    for (const auto& answer : answers) {
        SCOPED_TRACE(answer.input);
        Sha256State state{}; std::array<uint8_t, 32> digest{};
        ASSERT_TRUE(sha256_stream_init(&state));
        ASSERT_TRUE(sha256_stream_update(&state,
            reinterpret_cast<const uint8_t*>(answer.input), std::strlen(answer.input)));
        ASSERT_TRUE(sha256_stream_final(&state, digest.data()));
        EXPECT_EQ(digest, unhex(answer.digest));
    }
}
TEST(Sha256Stream, IndependentMillionAUsesBoundedRepeatedInput) {
    Sha256State state{}; std::array<uint8_t, 1000> input{}; input.fill('a');
    ASSERT_TRUE(sha256_stream_init(&state));
    for (uint32_t i = 0; i < 1000; ++i)
        ASSERT_TRUE(sha256_stream_update(&state, input.data(), input.size()));
    std::array<uint8_t, 32> digest{};
    ASSERT_TRUE(sha256_stream_final(&state, digest.data()));
    EXPECT_EQ(digest, unhex("cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"));
}
TEST(Sha256Stream, SplitBoundariesAndCopiedStatesMatchExistingOneShot) {
    std::array<uint8_t, 257> input{};
    for (uint32_t i = 0; i < input.size(); ++i) input[i] = static_cast<uint8_t>(i * 131u);
    std::array<uint8_t, 32> expected{}; sha256(input.data(), input.size(), expected.data());
    for (const uint32_t split : {0u, 1u, 55u, 56u, 63u, 64u, 65u, 127u, 128u, 129u, 257u}) {
        SCOPED_TRACE(split);
        Sha256State first{};
        ASSERT_TRUE(sha256_stream_init(&first));
        ASSERT_TRUE(sha256_stream_update(&first, input.data(), split));
        Sha256State fork = first;
        ASSERT_TRUE(sha256_stream_update(&first, input.data() + split, input.size() - split));
        for (uint32_t i = split; i < input.size(); ++i)
            ASSERT_TRUE(sha256_stream_update(&fork, input.data() + i, 1));
        std::array<uint8_t, 32> a{}, b{};
        ASSERT_TRUE(sha256_stream_final(&first, a.data()));
        ASSERT_TRUE(sha256_stream_final(&fork, b.data()));
        EXPECT_EQ(a, expected); EXPECT_EQ(b, expected);
    }
}
TEST(Sha256Stream, PhaseAndArgumentRefusalsDoNotMutateStateOrOutput) {
    Sha256State state{}; std::array<uint8_t, 32> output{}; output.fill(0xa5);
    EXPECT_FALSE(sha256_stream_init(nullptr));
    const Sha256State uninitialized = state;
    EXPECT_FALSE(sha256_stream_update(&state, nullptr, 0));
    EXPECT_FALSE(sha256_stream_final(&state, output.data()));
    EXPECT_EQ(std::memcmp(&state, &uninitialized, sizeof(state)), 0);
    EXPECT_TRUE(std::all_of(output.begin(), output.end(), [](uint8_t ch) { return ch == 0xa5; }));
    ASSERT_TRUE(sha256_stream_init(&state));
    ASSERT_TRUE(sha256_stream_update(&state, nullptr, 0));
    const Sha256State active = state;
    EXPECT_FALSE(sha256_stream_update(&state, nullptr, 1));
    EXPECT_FALSE(sha256_stream_final(&state, nullptr));
    EXPECT_EQ(std::memcmp(&state, &active, sizeof(state)), 0);
    ASSERT_TRUE(sha256_stream_final(&state, output.data()));
    const Sha256State finished = state; const auto digest = output;
    EXPECT_FALSE(sha256_stream_update(&state, nullptr, 0));
    EXPECT_FALSE(sha256_stream_final(&state, output.data()));
    EXPECT_EQ(std::memcmp(&state, &finished, sizeof(state)), 0); EXPECT_EQ(output, digest);
    EXPECT_TRUE(sha256_stream_init(&state));
}
TEST(Sha256Stream, OverlapAndLengthOverflowRefuseBeforeReadingOrMutating) {
    Sha256State state{}; ASSERT_TRUE(sha256_stream_init(&state));
    const Sha256State before = state;
    const auto* state_bytes = reinterpret_cast<const uint8_t*>(&state);
    EXPECT_FALSE(sha256_stream_update(&state, state_bytes, sizeof(state)));
    EXPECT_FALSE(sha256_stream_final(&state, state.buf));
    EXPECT_FALSE(sha256_stream_update(&state, reinterpret_cast<const uint8_t*>(UINTPTR_MAX - 3u), 8));
    EXPECT_EQ(std::memcmp(&state, &before, sizeof(state)), 0);
    // Synthetic active state at the largest whole-byte message length. No
    // huge source allocation/read is necessary to prove preflight overflow.
    state.bitlen = kSha256MaxInputBytes * 8u; state.buflen = 63;
    const Sha256State full = state; const uint8_t byte = 1;
    EXPECT_FALSE(sha256_stream_update(&state, &byte, 1));
    EXPECT_FALSE(sha256_stream_update(&state, &byte, UINT64_MAX));
    EXPECT_EQ(std::memcmp(&state, &full, sizeof(state)), 0);
}
}  // namespace
