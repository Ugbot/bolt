// test_bolt_msort_perm.cpp — tests for merge_sort_indirect_u32 in
// bolt/kernels/bolt_argsort.h: the explicit-comparator STABLE BOTTOM-UP
// MERGE SORT of a uint32 permutation (the std::sort replacement on the
// chukonu ORDER BY / agg-spill / window hot paths).
//
// Coverage: empty / n=1, sorted, reverse, duplicates (stability vs a
// reference std::stable_sort), descending-via-comparator, multi-key-via-
// comparator, and perm-is-a-valid-permutation. Deterministic data only.

#include "bolt/kernels/bolt_argsort.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

using bolt::kernels::merge_sort_indirect_u32;

// Initialise perm to identity [0,n) and run the kernel with a comparator.
template <typename Compare>
std::vector<std::uint32_t> run_msort(std::int64_t n, Compare less) {
    std::vector<std::uint32_t> perm(static_cast<std::size_t>(n));
    std::vector<std::uint32_t> scratch(static_cast<std::size_t>(n == 0 ? 1 : n));
    for (std::int64_t i = 0; i < n; ++i) {
        perm[static_cast<std::size_t>(i)] = static_cast<std::uint32_t>(i);
    }
    merge_sort_indirect_u32(perm.data(), scratch.data(), n, less);
    return perm;
}

// Reference: stable argsort via std::stable_sort over identity indices.
template <typename Compare>
std::vector<std::uint32_t> ref_stable(std::int64_t n, Compare less) {
    std::vector<std::uint32_t> perm(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) {
        perm[static_cast<std::size_t>(i)] = static_cast<std::uint32_t>(i);
    }
    std::stable_sort(perm.begin(), perm.end(), less);
    return perm;
}

// Assert `perm` is a genuine permutation of [0,n).
void expect_valid_permutation(const std::vector<std::uint32_t>& perm,
                              std::int64_t n) {
    std::vector<char> seen(static_cast<std::size_t>(n), 0);
    ASSERT_EQ(perm.size(), static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) {
        const std::uint32_t v = perm[static_cast<std::size_t>(i)];
        ASSERT_LT(v, static_cast<std::uint32_t>(n));
        EXPECT_EQ(seen[v], 0) << "duplicate index " << v;
        seen[v] = 1;
    }
}

// ---------------------------------------------------------------------------
// Empty and singleton.
// ---------------------------------------------------------------------------
TEST(MSortPermU32, Empty) {
    auto less = [](std::uint32_t, std::uint32_t) noexcept { return false; };
    std::vector<std::uint32_t> perm;       // n == 0 → nullptr-safe
    std::vector<std::uint32_t> scratch(1);
    merge_sort_indirect_u32(perm.data(), scratch.data(), 0, less);
    EXPECT_TRUE(perm.empty());
}

TEST(MSortPermU32, Single) {
    std::vector<std::int64_t> keys{42};
    auto less = [&](std::uint32_t a, std::uint32_t b) noexcept {
        return keys[a] < keys[b];
    };
    auto perm = run_msort(1, less);
    ASSERT_EQ(perm.size(), 1u);
    EXPECT_EQ(perm[0], 0u);
}

// ---------------------------------------------------------------------------
// Already sorted and fully reversed (exercises both the insertion base case
// at small n and the merge passes at large n).
// ---------------------------------------------------------------------------
TEST(MSortPermU32, SortedAndReverse) {
    for (std::int64_t n : {8, 33, 100, 1000}) {  // 8/33 straddle the cap (32)
        std::vector<std::int64_t> asc(static_cast<std::size_t>(n));
        std::vector<std::int64_t> desc(static_cast<std::size_t>(n));
        for (std::int64_t i = 0; i < n; ++i) {
            asc[static_cast<std::size_t>(i)] = i;
            desc[static_cast<std::size_t>(i)] = n - i;
        }
        auto less_asc = [&](std::uint32_t a, std::uint32_t b) noexcept {
            return asc[a] < asc[b];
        };
        auto less_desc = [&](std::uint32_t a, std::uint32_t b) noexcept {
            return desc[a] < desc[b];
        };
        auto p_sorted = run_msort(n, less_asc);
        for (std::int64_t i = 0; i < n; ++i) {
            EXPECT_EQ(p_sorted[static_cast<std::size_t>(i)],
                      static_cast<std::uint32_t>(i));
        }
        auto p_rev = run_msort(n, less_desc);
        expect_valid_permutation(p_rev, n);
        for (std::int64_t i = 0; i < n; ++i) {  // desc[idx] must be ascending
            const std::uint32_t idx = p_rev[static_cast<std::size_t>(i)];
            EXPECT_EQ(desc[idx], i + 1);
        }
    }
}

// ---------------------------------------------------------------------------
// Duplicates → stability matches std::stable_sort exactly (ties keep original
// index order). This is the key byte-identical guarantee for ORDER BY ties.
// ---------------------------------------------------------------------------
TEST(MSortPermU32, DuplicatesStableVsReference) {
    for (std::int64_t n : {16, 31, 32, 64, 257, 1024}) {
        std::vector<std::int64_t> keys(static_cast<std::size_t>(n));
        // Deterministic LCG → many collisions in a small key range.
        std::uint64_t s = 0x9E3779B97F4A7C15ull;
        for (std::int64_t i = 0; i < n; ++i) {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            keys[static_cast<std::size_t>(i)] =
                static_cast<std::int64_t>((s >> 33) % 8);  // 0..7 → dup-heavy
        }
        auto less = [&](std::uint32_t a, std::uint32_t b) noexcept {
            return keys[a] < keys[b];
        };
        auto got = run_msort(n, less);
        auto want = ref_stable(n, less);
        expect_valid_permutation(got, n);
        EXPECT_EQ(got, want) << "n=" << n << " stability mismatch";
    }
}

// ---------------------------------------------------------------------------
// Descending via comparator (negate the compare) — still stable on ties.
// ---------------------------------------------------------------------------
TEST(MSortPermU32, DescendingViaComparator) {
    const std::int64_t n = 500;
    std::vector<std::int64_t> keys(static_cast<std::size_t>(n));
    std::uint64_t s = 12345;
    for (std::int64_t i = 0; i < n; ++i) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        keys[static_cast<std::size_t>(i)] =
            static_cast<std::int64_t>((s >> 33) % 50);
    }
    auto less = [&](std::uint32_t a, std::uint32_t b) noexcept {
        return keys[a] > keys[b];  // descending
    };
    auto got = run_msort(n, less);
    auto want = ref_stable(n, less);
    expect_valid_permutation(got, n);
    EXPECT_EQ(got, want);
    for (std::int64_t i = 1; i < n; ++i) {  // non-increasing keys
        EXPECT_GE(keys[got[static_cast<std::size_t>(i - 1)]],
                  keys[got[static_cast<std::size_t>(i)]]);
    }
}

// ---------------------------------------------------------------------------
// Multi-key (lexicographic) via comparator: primary asc, secondary desc.
// ---------------------------------------------------------------------------
TEST(MSortPermU32, MultiKeyViaComparator) {
    const std::int64_t n = 777;
    std::vector<std::int64_t> k0(static_cast<std::size_t>(n));
    std::vector<std::int64_t> k1(static_cast<std::size_t>(n));
    std::uint64_t s = 99;
    for (std::int64_t i = 0; i < n; ++i) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        k0[static_cast<std::size_t>(i)] = static_cast<std::int64_t>((s >> 40) % 5);
        k1[static_cast<std::size_t>(i)] = static_cast<std::int64_t>((s >> 20) % 7);
    }
    auto less = [&](std::uint32_t a, std::uint32_t b) noexcept {
        if (k0[a] != k0[b]) return k0[a] < k0[b];      // primary ascending
        if (k1[a] != k1[b]) return k1[a] > k1[b];      // secondary descending
        return false;                                  // tie → stable false
    };
    auto got = run_msort(n, less);
    auto want = ref_stable(n, less);
    expect_valid_permutation(got, n);
    EXPECT_EQ(got, want);
}

}  // namespace
