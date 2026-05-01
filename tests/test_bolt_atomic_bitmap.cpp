// test_bolt_atomic_bitmap.cpp — coverage for bolt::atomic_bitmap_*.
//
// Tests:
//   ZeroOnConstruct           — fresh atomics read 0
//   SetReturnsTrueOnce        — first set returns true; second returns false
//   ClearReturnsTrueOnce      — first clear of a set bit returns true; idempotent
//   PopcountAfterSet          — popcount matches number of set bits
//   ConcurrentSet             — many threads racing on the same bit yields
//                               exactly one true, all others false
//   BoundaryBits              — bit 0, bit 63, bit 64, last bit of last word
//
// Test code may use stdlib freely; the kernel under test does not.

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include "bolt/kernels/bolt_atomic_bitmap.h"

using bolt::atomic_bitmap_clear;
using bolt::atomic_bitmap_clear_all;
using bolt::atomic_bitmap_popcount;
using bolt::atomic_bitmap_set;
using bolt::atomic_bitmap_test;
using bolt::atomic_bitmap_words_for;

namespace {

constexpr size_t kBits = 256u;
constexpr size_t kWords = atomic_bitmap_words_for(kBits);
static_assert(kWords == 4u, "256 bits → 4 words");

}  // namespace

TEST(BoltAtomicBitmap, ZeroOnConstruct) {
    std::atomic<uint64_t> words[kWords]{};
    EXPECT_EQ(atomic_bitmap_popcount(words, kWords), 0u);
    for (size_t i = 0; i < kBits; ++i) {
        EXPECT_FALSE(atomic_bitmap_test(words, i)) << "bit " << i;
    }
}

TEST(BoltAtomicBitmap, SetReturnsTrueOnce) {
    std::atomic<uint64_t> words[kWords]{};
    EXPECT_TRUE (atomic_bitmap_set(words, 7u));
    EXPECT_FALSE(atomic_bitmap_set(words, 7u));
    EXPECT_TRUE (atomic_bitmap_test(words, 7u));
    EXPECT_FALSE(atomic_bitmap_test(words, 6u));
    EXPECT_FALSE(atomic_bitmap_test(words, 8u));
    EXPECT_EQ(atomic_bitmap_popcount(words, kWords), 1u);
}

TEST(BoltAtomicBitmap, ClearReturnsTrueOnce) {
    std::atomic<uint64_t> words[kWords]{};
    atomic_bitmap_set(words, 12u);
    atomic_bitmap_set(words, 199u);
    EXPECT_EQ(atomic_bitmap_popcount(words, kWords), 2u);
    EXPECT_TRUE (atomic_bitmap_clear(words, 12u));
    EXPECT_FALSE(atomic_bitmap_clear(words, 12u));
    EXPECT_EQ(atomic_bitmap_popcount(words, kWords), 1u);
}

TEST(BoltAtomicBitmap, PopcountAfterSet) {
    std::atomic<uint64_t> words[kWords]{};
    const std::vector<uint64_t> bits = {0u, 1u, 63u, 64u, 65u, 127u, 128u, 255u};
    for (auto b : bits) atomic_bitmap_set(words, b);
    EXPECT_EQ(atomic_bitmap_popcount(words, kWords), bits.size());
    for (auto b : bits) EXPECT_TRUE(atomic_bitmap_test(words, b));
}

TEST(BoltAtomicBitmap, BoundaryBits) {
    std::atomic<uint64_t> words[kWords]{};
    atomic_bitmap_set(words, 0u);
    atomic_bitmap_set(words, 63u);
    atomic_bitmap_set(words, 64u);
    atomic_bitmap_set(words, kBits - 1u);
    EXPECT_TRUE(atomic_bitmap_test(words, 0u));
    EXPECT_TRUE(atomic_bitmap_test(words, 63u));
    EXPECT_TRUE(atomic_bitmap_test(words, 64u));
    EXPECT_TRUE(atomic_bitmap_test(words, kBits - 1u));
    EXPECT_EQ(atomic_bitmap_popcount(words, kWords), 4u);
    atomic_bitmap_clear_all(words, kWords);
    EXPECT_EQ(atomic_bitmap_popcount(words, kWords), 0u);
}

TEST(BoltAtomicBitmap, ConcurrentSetExactlyOneTrue) {
    constexpr size_t kN = 8;
    std::atomic<uint64_t> words[kWords]{};
    constexpr uint64_t kTargetBit = 42u;
    std::atomic<int> true_count{0};
    std::vector<std::thread> ts;
    ts.reserve(kN);
    for (size_t i = 0; i < kN; ++i) {
        ts.emplace_back([&] {
            if (atomic_bitmap_set(words, kTargetBit)) {
                true_count.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto& t : ts) t.join();
    EXPECT_EQ(true_count.load(), 1);
    EXPECT_TRUE(atomic_bitmap_test(words, kTargetBit));
    EXPECT_EQ(atomic_bitmap_popcount(words, kWords), 1u);
}
