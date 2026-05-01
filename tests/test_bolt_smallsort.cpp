// test_bolt_smallsort.cpp — coverage for bolt::kernels::sort_small_*.
//
// Asserts insertion-sort correctness over a few representative inputs.
// Test code may use stdlib freely; the kernel under test does not.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include "bolt/kernels/bolt_smallsort.h"

using bolt::kernels::sort_small_i64_asc;
using bolt::kernels::sort_small_u32_asc;
using bolt::kernels::sort_small_u64_asc;

TEST(BoltSmallSort, EmptyAndSingleton) {
    uint64_t a[1]{99u};
    sort_small_u64_asc(a, 0u);  // no-op
    sort_small_u64_asc(a, 1u);
    EXPECT_EQ(a[0], 99u);
}

TEST(BoltSmallSort, AlreadySortedU64) {
    uint64_t a[8] = {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u};
    sort_small_u64_asc(a, 8u);
    for (size_t i = 0; i < 8; ++i) EXPECT_EQ(a[i], i + 1u);
}

TEST(BoltSmallSort, ReversedU64) {
    uint64_t a[8] = {8u, 7u, 6u, 5u, 4u, 3u, 2u, 1u};
    sort_small_u64_asc(a, 8u);
    for (size_t i = 0; i < 8; ++i) EXPECT_EQ(a[i], i + 1u);
}

TEST(BoltSmallSort, RandomU64StableAgainstStdSort) {
    std::mt19937_64 rng(0xC0FFEEu);
    constexpr size_t kN = 200u;
    std::vector<uint64_t> a(kN), b(kN);
    for (size_t i = 0; i < kN; ++i) a[i] = rng();
    b = a;
    sort_small_u64_asc(a.data(), a.size());
    std::sort(b.begin(), b.end());
    EXPECT_EQ(a, b);
}

TEST(BoltSmallSort, DuplicatesU64) {
    uint64_t a[6] = {3u, 3u, 1u, 2u, 1u, 2u};
    sort_small_u64_asc(a, 6u);
    const uint64_t expected[6] = {1u, 1u, 2u, 2u, 3u, 3u};
    for (size_t i = 0; i < 6; ++i) EXPECT_EQ(a[i], expected[i]);
}

TEST(BoltSmallSort, RandomI64StableAgainstStdSort) {
    std::mt19937_64 rng(0xBADBEEFu);
    constexpr size_t kN = 100u;
    std::vector<int64_t> a(kN), b(kN);
    for (size_t i = 0; i < kN; ++i) a[i] = static_cast<int64_t>(rng());
    b = a;
    sort_small_i64_asc(a.data(), a.size());
    std::sort(b.begin(), b.end());
    EXPECT_EQ(a, b);
}

TEST(BoltSmallSort, RandomU32StableAgainstStdSort) {
    std::mt19937 rng(0xFACEFEEDu);
    constexpr size_t kN = 64u;
    std::vector<uint32_t> a(kN), b(kN);
    for (size_t i = 0; i < kN; ++i) a[i] = rng();
    b = a;
    sort_small_u32_asc(a.data(), a.size());
    std::sort(b.begin(), b.end());
    EXPECT_EQ(a, b);
}
