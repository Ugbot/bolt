// test_bolt_bitmap.cpp — Tests for the bitmap kernels in
// bolt/kernels/bolt_bitmap.h.

#include "bolt/kernels/bolt_bitmap.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>

namespace {

// Helper: set bit i in `bm`.
inline void set_bit(uint8_t* bm, size_t i) noexcept {
    bm[i >> 3] |= static_cast<uint8_t>(1u << (i & 7u));
}

// Helper: read bit i from `bm`.
inline bool get_bit(const uint8_t* bm, size_t i) noexcept {
    return ((bm[i >> 3] >> (i & 7u)) & 1u) != 0u;
}

}  // namespace

TEST(BoltBitmap, PopcountPartialByteRange) {
    // 24 bits = 3 bytes; set bits 0,3,5,7,8,12,17,20,23.
    std::array<uint8_t, 4> bm{};
    bm.fill(0);
    for (size_t i : {0u, 3u, 5u, 7u, 8u, 12u, 17u, 20u, 23u}) {
        set_bit(bm.data(), i);
    }
    // Range [3, 21): bits 3,5,7,8,12,17,20 — count = 7.
    EXPECT_EQ(bolt::bitmap_popcount(bm.data(), 3, 21), 7u);
    // Empty range.
    EXPECT_EQ(bolt::bitmap_popcount(bm.data(), 5, 5), 0u);
    // Full range matches naive sum.
    EXPECT_EQ(bolt::bitmap_popcount(bm.data(), 0, 24), 9u);
}

TEST(BoltBitmap, PopcountLargeAcrossBodyAndTail) {
    // 200 bits = 25 bytes. Set every 3rd bit.
    constexpr size_t kBits = 200;
    std::array<uint8_t, 32> bm{};
    bm.fill(0);
    size_t expected = 0;
    for (size_t i = 0; i < kBits; i += 3) {
        set_bit(bm.data(), i);
        ++expected;
    }
    EXPECT_EQ(bolt::bitmap_popcount(bm.data(), 0, kBits), expected);
    // Non-byte-aligned head and tail.
    size_t naive = 0;
    for (size_t i = 5; i < 197; ++i) {
        if (get_bit(bm.data(), i)) ++naive;
    }
    EXPECT_EQ(bolt::bitmap_popcount(bm.data(), 5, 197), naive);
}

TEST(BoltBitmap, ToIndicesSparseAscending) {
    constexpr size_t kBits = 64;
    std::array<uint8_t, 8> bm{};
    bm.fill(0);
    const std::array<uint32_t, 5> expected_idx{2u, 7u, 13u, 41u, 60u};
    for (uint32_t i : expected_idx) set_bit(bm.data(), i);

    std::array<uint32_t, 64> out{};
    out.fill(0xFFFFFFFFu);
    const size_t n =
        bolt::bitmap_to_indices(bm.data(), kBits, out.data());
    ASSERT_EQ(n, expected_idx.size());
    for (size_t k = 0; k < n; ++k) {
        EXPECT_EQ(out[k], expected_idx[k]);
    }
    // Strictly ascending.
    for (size_t k = 1; k < n; ++k) {
        EXPECT_LT(out[k - 1], out[k]);
    }
}

TEST(BoltBitmap, AndIdempotent) {
    constexpr size_t kBits = 100;
    std::array<uint8_t, 16> a{};
    a.fill(0);
    for (size_t i : {1u, 4u, 9u, 16u, 25u, 36u, 49u, 64u, 81u}) {
        set_bit(a.data(), i);
    }
    const std::array<uint8_t, 16> snapshot = a;

    bolt::bitmap_and(a.data(), snapshot.data(), kBits);
    EXPECT_EQ(std::memcmp(a.data(), snapshot.data(), a.size()), 0);
}

TEST(BoltBitmap, AndZeroAnnihilates) {
    constexpr size_t kBits = 80;
    std::array<uint8_t, 10> a{};
    std::array<uint8_t, 10> z{};
    a.fill(0xFF);
    z.fill(0x00);
    bolt::bitmap_and(a.data(), z.data(), kBits);
    for (size_t i = 0; i < kBits; ++i) {
        EXPECT_FALSE(get_bit(a.data(), i));
    }
}

TEST(BoltBitmap, OrCommutative) {
    constexpr size_t kBits = 96;
    std::array<uint8_t, 12> a0{};
    std::array<uint8_t, 12> b0{};
    a0.fill(0);
    b0.fill(0);
    for (size_t i : {0u, 3u, 17u, 33u, 64u, 95u}) set_bit(a0.data(), i);
    for (size_t i : {1u, 17u, 50u, 64u, 80u}) set_bit(b0.data(), i);

    // Path 1: a |= b
    auto a1 = a0;
    auto b1 = b0;
    bolt::bitmap_or(a1.data(), b1.data(), kBits);

    // Path 2: b |= a, then compare to result of path 1.
    auto a2 = a0;
    auto b2 = b0;
    bolt::bitmap_or(b2.data(), a2.data(), kBits);

    EXPECT_EQ(std::memcmp(a1.data(), b2.data(), a1.size()), 0);
}

TEST(BoltBitmap, FillNonByteAligned) {
    constexpr size_t kBits = 64;
    std::array<uint8_t, 8> bm{};
    bm.fill(0);
    // Fill [3, 21) — straddles bytes 0..2.
    bolt::bitmap_fill(bm.data(), 3, 21);
    for (size_t i = 0; i < kBits; ++i) {
        const bool inside = (i >= 3 && i < 21);
        EXPECT_EQ(get_bit(bm.data(), i), inside) << "bit " << i;
    }
    // Empty range is a no-op.
    auto before = bm;
    bolt::bitmap_fill(bm.data(), 7, 7);
    EXPECT_EQ(std::memcmp(bm.data(), before.data(), bm.size()), 0);
    // OR-into existing bits doesn't unset them.
    bolt::bitmap_fill(bm.data(), 19, 25);
    for (size_t i = 3; i < 25; ++i) {
        EXPECT_TRUE(get_bit(bm.data(), i)) << "bit " << i;
    }
}
