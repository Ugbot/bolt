// test_bolt_crc64nvme.cpp — CRC-64/NVME against the catalogue check value,
// a bitwise oracle, and chaining.

#include <gtest/gtest.h>

#include "bolt/io/bolt_crc64nvme.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace {

uint64_t bitwise(const uint8_t* p, size_t n) {
    uint64_t c = ~0ull;
    for (size_t i = 0; i < n; ++i) {
        c ^= p[i];
        for (int k = 0; k < 8; ++k) {
            c = (c & 1u) ? (0x9A6C9329AC4BC9B5ull ^ (c >> 1)) : (c >> 1);
        }
    }
    return ~c;
}

TEST(BoltCrc64Nvme, CheckValue) {
    EXPECT_EQ(bolt::io::crc64nvme("123456789", 9), 0xAE8B14860A799888ull);
    EXPECT_EQ(bolt::io::crc64nvme(nullptr, 0), 0ull);
}

TEST(BoltCrc64Nvme, MatchesBitwiseOracleAtEveryLengthAndOffset) {
    std::vector<uint8_t> buf(4096 + 16);
    uint64_t x = 0x243F6A8885A308D3ull;
    for (auto& b : buf) {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        b = static_cast<uint8_t>(x >> 56);
    }
    for (size_t off = 0; off < 8; ++off) {
        for (size_t n = 0; n < 300; ++n) {
            ASSERT_EQ(bolt::io::crc64nvme(buf.data() + off, n),
                      bitwise(buf.data() + off, n)) << off << " " << n;
        }
    }
    EXPECT_EQ(bolt::io::crc64nvme(buf.data(), 4096),
              bitwise(buf.data(), 4096));
}

TEST(BoltCrc64Nvme, ChainsLikeOneCall) {
    const char* s = "the quick brown fox jumps over the lazy dog";
    const size_t n = std::strlen(s);
    const uint64_t whole = bolt::io::crc64nvme(s, n);
    for (size_t cut = 0; cut <= n; ++cut) {
        const uint64_t a = bolt::io::crc64nvme(s, cut);
        EXPECT_EQ(bolt::io::crc64nvme_update(a, s + cut, n - cut), whole);
    }
    EXPECT_NE(bolt::io::crc64nvme("123456788", 9), 0xAE8B14860A799888ull);
}

}  // namespace
