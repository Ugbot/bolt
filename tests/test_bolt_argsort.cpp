// test_bolt_argsort.cpp — Coverage for the indirect / argsort kernels.
//
// Verifies sort_indirect_i64 (single key) and sort_indirect_multikey_i64
// (lexicographic) against a stable std::stable_sort reference over
// deterministic data. The key invariants under test:
//   * perm is a valid permutation of [0, n) (covers each index once),
//   * keys[perm[i]] is monotonic in the requested direction,
//   * STABILITY: equal keys keep their original (ascending) index order,
//     in BOTH ascending and descending output,
//   * edge cases n = 0, n = 1, already-sorted, reverse-sorted,
//   * the merge path (n > kArgsortInsertionCap) via large random inputs,
//   * multikey (p, o, s)-style ties resolve on later keys, then index.
//
// All inputs are built from fixed seeds so failures replay byte-for-byte.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include "bolt/kernels/bolt_argsort.h"

using namespace bolt::kernels;

namespace {

// Assert perm is a permutation of [0, n): every index appears once.
void expect_valid_permutation(const std::vector<int32_t>& perm, int64_t n) {
    ASSERT_EQ(static_cast<int64_t>(perm.size()), n);
    std::vector<char> seen(static_cast<size_t>(n), 0);
    for (int64_t i = 0; i < n; ++i) {
        const int32_t p = perm[static_cast<size_t>(i)];
        ASSERT_GE(p, 0);
        ASSERT_LT(p, n);
        ASSERT_EQ(seen[static_cast<size_t>(p)], 0) << "duplicate index " << p;
        seen[static_cast<size_t>(p)] = 1;
    }
}

// Stable reference permutation for a single int64 key.
std::vector<int32_t> ref_argsort_i64(const std::vector<int64_t>& keys,
                                     bool ascending) {
    const int64_t n = static_cast<int64_t>(keys.size());
    std::vector<int32_t> perm(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) perm[static_cast<size_t>(i)] =
            static_cast<int32_t>(i);
    std::stable_sort(perm.begin(), perm.end(), [&](int32_t a, int32_t b) {
        return ascending ? (keys[a] < keys[b]) : (keys[a] > keys[b]);
    });
    return perm;
}

void run_single_key_case(const std::vector<int64_t>& keys, bool ascending) {
    const int64_t n = static_cast<int64_t>(keys.size());
    std::vector<int32_t> perm(static_cast<size_t>(n));
    std::vector<int32_t> scratch(static_cast<size_t>(n > 0 ? n : 1));
    sort_indirect_i64_scratch(perm.data(), keys.data(), scratch.data(), n,
                              ascending);
    expect_valid_permutation(perm, n);
    // Must equal the stable reference exactly (proves stability + order).
    const std::vector<int32_t> expect = ref_argsort_i64(keys, ascending);
    EXPECT_EQ(perm, expect);
    // Monotonic check (redundant but localises a failure to ordering).
    for (int64_t i = 1; i < n; ++i) {
        const int64_t prev = keys[perm[static_cast<size_t>(i - 1)]];
        const int64_t cur = keys[perm[static_cast<size_t>(i)]];
        if (ascending) EXPECT_LE(prev, cur);
        else EXPECT_GE(prev, cur);
    }
}

}  // namespace

TEST(BoltArgsort, EmptyAndSingle) {
    run_single_key_case({}, true);
    run_single_key_case({}, false);
    run_single_key_case({42}, true);
    run_single_key_case({42}, false);
}

TEST(BoltArgsort, AlreadySorted) {
    std::vector<int64_t> keys;
    for (int64_t i = 0; i < 50; ++i) keys.push_back(i);
    run_single_key_case(keys, true);
    run_single_key_case(keys, false);
}

TEST(BoltArgsort, ReverseSorted) {
    std::vector<int64_t> keys;
    for (int64_t i = 50; i > 0; --i) keys.push_back(i);
    run_single_key_case(keys, true);
    run_single_key_case(keys, false);
}

// Small-n base case (insertion path) with heavy duplicates → stability.
TEST(BoltArgsort, SmallWithDuplicatesStable) {
    std::vector<int64_t> keys = {3, 1, 3, 2, 1, 3, 2, 1, 0, 3};
    run_single_key_case(keys, true);
    run_single_key_case(keys, false);
}

// All-equal keys: perm must be identity (stability) in both directions.
TEST(BoltArgsort, AllEqualIsIdentity) {
    const int64_t n = 40;  // straddles insertion/merge boundary low side
    std::vector<int64_t> keys(static_cast<size_t>(n), 7);
    std::vector<int32_t> perm(static_cast<size_t>(n));
    std::vector<int32_t> scratch(static_cast<size_t>(n));
    for (bool asc : {true, false}) {
        sort_indirect_i64_scratch(perm.data(), keys.data(), scratch.data(), n,
                                  asc);
        for (int64_t i = 0; i < n; ++i)
            EXPECT_EQ(perm[static_cast<size_t>(i)], static_cast<int32_t>(i));
    }
}

// Large random inputs WITH duplicates → exercises the bottom-up merge
// path and stability across many equal-key runs. Fixed seeds.
TEST(BoltArgsort, LargeRandomWithDuplicatesStable) {
    for (uint32_t seed : {1u, 7u, 12345u, 99999u}) {
        std::mt19937_64 rng(seed);
        const int64_t n = 5000;
        std::vector<int64_t> keys(static_cast<size_t>(n));
        // Small value range forces lots of duplicate keys.
        std::uniform_int_distribution<int64_t> dist(0, 50);
        for (int64_t i = 0; i < n; ++i)
            keys[static_cast<size_t>(i)] = dist(rng);
        run_single_key_case(keys, true);
        run_single_key_case(keys, false);
    }
}

// Negative keys + full int64 range spread.
TEST(BoltArgsort, NegativeAndWideRange) {
    std::mt19937_64 rng(2024u);
    const int64_t n = 1000;
    std::vector<int64_t> keys(static_cast<size_t>(n));
    std::uniform_int_distribution<int64_t> dist(-1000000, 1000000);
    for (int64_t i = 0; i < n; ++i) keys[static_cast<size_t>(i)] = dist(rng);
    run_single_key_case(keys, true);
    run_single_key_case(keys, false);
}

// Convenience on-stack-scratch entry agrees with the _scratch overload.
TEST(BoltArgsort, ConvenienceEntryMatches) {
    std::mt19937_64 rng(55u);
    const int64_t n = 300;
    std::vector<int64_t> keys(static_cast<size_t>(n));
    std::uniform_int_distribution<int64_t> dist(0, 20);
    for (int64_t i = 0; i < n; ++i) keys[static_cast<size_t>(i)] = dist(rng);
    std::vector<int32_t> perm(static_cast<size_t>(n));
    sort_indirect_i64(perm.data(), keys.data(), n, true);
    EXPECT_EQ(perm, ref_argsort_i64(keys, true));
}

// ---------------------------------------------------------------------------
// Multi-key (lexicographic) coverage.
// ---------------------------------------------------------------------------
namespace {

// Stable reference lexicographic permutation.
std::vector<int32_t> ref_argsort_multikey(
        const std::vector<std::vector<int64_t>>& cols, const std::vector<bool>& asc) {
    const int64_t n = static_cast<int64_t>(cols[0].size());
    const int64_t nk = static_cast<int64_t>(cols.size());
    std::vector<int32_t> perm(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) perm[static_cast<size_t>(i)] =
            static_cast<int32_t>(i);
    std::stable_sort(perm.begin(), perm.end(), [&](int32_t a, int32_t b) {
        for (int64_t k = 0; k < nk; ++k) {
            const int64_t va = cols[static_cast<size_t>(k)][a];
            const int64_t vb = cols[static_cast<size_t>(k)][b];
            if (va != vb)
                return asc[static_cast<size_t>(k)] ? (va < vb) : (va > vb);
        }
        return false;
    });
    return perm;
}

void run_multikey_case(const std::vector<std::vector<int64_t>>& cols,
                       const std::vector<bool>& asc) {
    const int64_t nk = static_cast<int64_t>(cols.size());
    const int64_t n = static_cast<int64_t>(cols[0].size());
    std::vector<const int64_t*> keyptrs(static_cast<size_t>(nk));
    for (int64_t k = 0; k < nk; ++k)
        keyptrs[static_cast<size_t>(k)] = cols[static_cast<size_t>(k)].data();
    // std::vector<bool> is not contiguous; copy into a plain bool buffer.
    std::vector<char> ascbuf(static_cast<size_t>(nk));
    for (int64_t k = 0; k < nk; ++k)
        ascbuf[static_cast<size_t>(k)] = asc[static_cast<size_t>(k)] ? 1 : 0;
    const bool* ascptr = reinterpret_cast<const bool*>(ascbuf.data());

    std::vector<int32_t> perm(static_cast<size_t>(n));
    std::vector<int32_t> scratch(static_cast<size_t>(n > 0 ? n : 1));
    sort_indirect_multikey_i64_scratch(perm.data(), keyptrs.data(),
                                       scratch.data(), nk, n, ascptr);
    expect_valid_permutation(perm, n);
    EXPECT_EQ(perm, ref_argsort_multikey(cols, asc));
}

}  // namespace

// (p, o, s)-style 3-key sort with deliberate ties on earlier keys so the
// later keys (and finally original index) decide order. Mirrors the
// triple-store index build.
TEST(BoltArgsort, MultikeyTriplePOSTies) {
    //                 p:  o:  s:
    std::vector<std::vector<int64_t>> cols = {
        {1, 1, 1, 1, 2, 2, 2, 0, 0, 0},   // p
        {5, 5, 3, 5, 1, 1, 9, 4, 4, 4},   // o
        {2, 1, 0, 2, 7, 7, 0, 8, 8, 8},   // s
    };
    run_multikey_case(cols, {true, true, true});
    // Mixed directions.
    run_multikey_case(cols, {true, false, true});
    run_multikey_case(cols, {false, true, false});
}

// Large random 3-key input with a tiny value range → many ties at every
// level, exercising the merge path's stability. Fixed seeds.
TEST(BoltArgsort, MultikeyLargeRandomStable) {
    for (uint32_t seed : {3u, 88u, 4242u}) {
        std::mt19937_64 rng(seed);
        const int64_t n = 4000;
        std::vector<std::vector<int64_t>> cols(3,
                std::vector<int64_t>(static_cast<size_t>(n)));
        std::uniform_int_distribution<int64_t> dist(0, 6);
        for (int64_t k = 0; k < 3; ++k)
            for (int64_t i = 0; i < n; ++i)
                cols[static_cast<size_t>(k)][static_cast<size_t>(i)] = dist(rng);
        run_multikey_case(cols, {true, true, true});
        run_multikey_case(cols, {true, true, false});
    }
}

// Single-key behaviour through the multikey path (n_keys == 1).
TEST(BoltArgsort, MultikeyDegenerateSingleKey) {
    std::vector<std::vector<int64_t>> cols = {{4, 4, 1, 9, 1, 4, 0}};
    run_multikey_case(cols, {true});
    run_multikey_case(cols, {false});
}
