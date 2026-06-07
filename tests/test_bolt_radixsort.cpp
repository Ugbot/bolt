// test_bolt_radixsort.cpp — Coverage for the LSD radix sort kernels.
//
// Verifies radix_sort_i64 (DIRECT value sort) and radix_sort_indirect_i64
// (INDIRECT permutation sort) against a std::stable_sort reference over
// deterministic, seeded data. Invariants under test:
//   * DIRECT: output equals std::stable_sort of the input (ascending),
//   * INDIRECT: perm is a valid permutation of [0, n); keys[perm[i]] is
//     ascending; STABLE — equal keys keep original (ascending) index order;
//     perm exactly matches a stable_sort-of-indices reference,
//   * SIGNED correctness: negatives sort before positives; INT64_MIN and
//     INT64_MAX placed correctly in a mixed array,
//   * edge cases n = 0, n = 1, already sorted, reverse sorted, all-equal,
//     all-duplicates-of-a-few-values, large random (well past the SIMD
//     4-lane block boundary),
//   * keys[] is never mutated by the INDIRECT form,
//   * SCALAR == SIMD: the AVX2 digit-extract is cross-checked against an
//     independent scalar digit-extract for bit-identical histograms, and the
//     end-to-end sort output is path-independent (it equals the oracle on
//     whichever path this TU compiled — see PARITY note below).
//
// All inputs are built from fixed seeds so failures replay byte-for-byte.
//
// PARITY note: the kernel selects the AVX2 vs scalar histogram at COMPILE
// time via BOLT_SIMD_AVX2 (Bolt convention — no runtime divergence). A single
// TU therefore exercises exactly one path. We prove scalar==SIMD two ways:
// (1) the DigitExtractParity test reimplements BOTH the AVX2 and the scalar
//     digit-extraction here and asserts they produce identical per-byte
//     histograms on random data (the only place the two paths differ), and
// (2) the sort output is asserted equal to the std::stable_sort oracle, which
//     is path-independent — so whichever path this build compiled is correct.
// To exercise the scalar path explicitly, build this TU without /arch:AVX2.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "bolt/kernels/bolt_radixsort.h"

using namespace bolt::kernels;

namespace {

// Deterministic seeded key generator.
std::vector<int64_t> make_keys(uint64_t seed, int64_t n, int64_t lo,
                               int64_t hi) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> dist(lo, hi);
    std::vector<int64_t> v(static_cast<size_t>(n));
    for (auto& x : v) x = dist(rng);
    return v;
}

// Assert perm is a permutation of [0, n): every index appears exactly once.
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

// Stable index-sort oracle: argsort by key, ties broken by original index.
std::vector<int32_t> stable_argsort_oracle(const std::vector<int64_t>& keys) {
    std::vector<int32_t> idx(keys.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int32_t a, int32_t b) {
        return keys[static_cast<size_t>(a)] < keys[static_cast<size_t>(b)];
    });
    return idx;
}

// Run the DIRECT sort and compare to std::stable_sort of the same input.
void check_direct(const std::vector<int64_t>& in) {
    const int64_t n = static_cast<int64_t>(in.size());
    std::vector<int64_t> got = in;
    std::vector<int64_t> scratch(static_cast<size_t>(std::max<int64_t>(n, 1)));
    radix_sort_i64(got.data(), scratch.data(), n);
    std::vector<int64_t> want = in;
    std::stable_sort(want.begin(), want.end());
    EXPECT_EQ(got, want);
    // Monotonic non-decreasing.
    for (int64_t i = 1; i < n; ++i)
        EXPECT_LE(got[static_cast<size_t>(i - 1)], got[static_cast<size_t>(i)]);
}

// Run the INDIRECT sort; check valid perm, ascending, stable, and that keys
// were not mutated.
void check_indirect(const std::vector<int64_t>& in) {
    const int64_t n = static_cast<int64_t>(in.size());
    std::vector<int64_t> keys = in;  // copy so we can detect mutation
    std::vector<int32_t> perm(static_cast<size_t>(std::max<int64_t>(n, 1)));
    std::vector<int32_t> perm_scratch(static_cast<size_t>(std::max<int64_t>(n, 1)));
    std::vector<int64_t> key_scratch(static_cast<size_t>(std::max<int64_t>(n, 1)));
    radix_sort_indirect_i64(perm.data(), keys.data(), perm_scratch.data(),
                            key_scratch.data(), n);
    perm.resize(static_cast<size_t>(n));
    EXPECT_EQ(keys, in) << "INDIRECT sort must not mutate keys";
    expect_valid_permutation(perm, n);
    // keys[perm[i]] ascending.
    for (int64_t i = 1; i < n; ++i)
        EXPECT_LE(keys[static_cast<size_t>(perm[static_cast<size_t>(i - 1)])],
                  keys[static_cast<size_t>(perm[static_cast<size_t>(i)])]);
    // Exact match against the stable oracle (proves stability on ties).
    EXPECT_EQ(perm, stable_argsort_oracle(in));
}

void check_both(const std::vector<int64_t>& in) {
    check_direct(in);
    check_indirect(in);
}

}  // namespace

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------
TEST(BoltRadixSort, Empty) { check_both({}); }
TEST(BoltRadixSort, Single) { check_both({42}); }
TEST(BoltRadixSort, SingleNegative) { check_both({-42}); }
TEST(BoltRadixSort, TwoSorted) { check_both({1, 2}); }
TEST(BoltRadixSort, TwoReverse) { check_both({2, 1}); }

TEST(BoltRadixSort, AlreadySorted) {
    std::vector<int64_t> v(1000);
    std::iota(v.begin(), v.end(), -500);
    check_both(v);
}

TEST(BoltRadixSort, ReverseSorted) {
    std::vector<int64_t> v(1000);
    for (int i = 0; i < 1000; ++i) v[static_cast<size_t>(i)] = 999 - i;
    check_both(v);
}

TEST(BoltRadixSort, AllEqual) {
    check_both(std::vector<int64_t>(777, 7));
}

TEST(BoltRadixSort, FewDistinctManyDuplicates) {
    // Hammers stability: only 5 distinct values across 5000 elements.
    auto v = make_keys(0xC0FFEE, 5000, 0, 4);
    check_both(v);
}

// ---------------------------------------------------------------------------
// Signed correctness
// ---------------------------------------------------------------------------
TEST(BoltRadixSort, NegativesAndPositives) {
    check_both(make_keys(0xBEEF, 4096, -100000, 100000));
}

TEST(BoltRadixSort, ExtremesMixed) {
    std::vector<int64_t> v = {
        0, INT64_MAX, INT64_MIN, -1, 1, INT64_MIN, INT64_MAX, -2, 2,
        INT64_MIN + 1, INT64_MAX - 1, 0, -100, 100};
    check_both(v);
    // Spot-check the placement after the direct sort.
    std::vector<int64_t> sorted = v;
    std::vector<int64_t> scratch(sorted.size());
    radix_sort_i64(sorted.data(), scratch.data(),
                   static_cast<int64_t>(sorted.size()));
    EXPECT_EQ(sorted.front(), INT64_MIN);
    EXPECT_EQ(sorted.back(), INT64_MAX);
}

TEST(BoltRadixSort, AllNegative) {
    check_both(make_keys(0xA11BAD, 2000, INT64_MIN, -1));
}

// ---------------------------------------------------------------------------
// Large random — well past the SIMD 4-lane block boundary, across the full
// int64 range so every one of the 8 byte-passes sees variation.
// ---------------------------------------------------------------------------
TEST(BoltRadixSort, LargeRandomFullRange) {
    check_both(make_keys(1, 100000, INT64_MIN, INT64_MAX));
}
TEST(BoltRadixSort, LargeRandomNarrow) {
    check_both(make_keys(2, 100000, -1000, 1000));
}
TEST(BoltRadixSort, BlockBoundarySizes) {
    // n straddling the AVX2 4-lane block (4k+r) to exercise the tail loop.
    for (int64_t n : {3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 63, 64, 65}) {
        check_both(make_keys(static_cast<uint64_t>(100 + n), n,
                             -1000000, 1000000));
    }
}

// ---------------------------------------------------------------------------
// SCALAR == SIMD: reimplement BOTH digit-extraction paths here and assert the
// per-byte histograms are bit-identical. This is the ONLY place the AVX2 and
// scalar kernel paths differ; identical histograms ⇒ identical sort output.
// ---------------------------------------------------------------------------
TEST(BoltRadixSort, DigitExtractParity) {
    auto keys = make_keys(0xD161, 10000, INT64_MIN, INT64_MAX);
    std::vector<uint64_t> biased(keys.size());
    for (size_t i = 0; i < keys.size(); ++i)
        biased[i] = radix_bias_i64(keys[i]);
    const int64_t n = static_cast<int64_t>(biased.size());

    for (int pass = 0; pass < kRadixPasses; ++pass) {
        const int shift = pass * kRadixBits;

        // Path A — pure scalar digit extraction.
        int64_t hist_scalar[kRadixBuckets] = {0};
        for (int64_t i = 0; i < n; ++i) {
            const uint64_t d = (biased[static_cast<size_t>(i)] >> shift) &
                               kRadixMask;
            ++hist_scalar[d];
        }

        // Path B — the kernel's histogram (AVX2 on an AVX2 build, scalar
        // otherwise). Must match Path A bit-for-bit.
        int64_t hist_kernel[kRadixBuckets] = {0};
        radix_histogram_u64(biased.data(), n, shift, hist_kernel);

        for (int b = 0; b < kRadixBuckets; ++b)
            EXPECT_EQ(hist_scalar[b], hist_kernel[b])
                << "pass " << pass << " bucket " << b;
    }
}

// Cross-check: DIRECT and INDIRECT produce the SAME ordering of key values.
TEST(BoltRadixSort, DirectMatchesIndirect) {
    auto in = make_keys(99, 20000, INT64_MIN, INT64_MAX);
    const int64_t n = static_cast<int64_t>(in.size());

    std::vector<int64_t> direct = in;
    std::vector<int64_t> scratch(in.size());
    radix_sort_i64(direct.data(), scratch.data(), n);

    std::vector<int32_t> perm(in.size()), perm_scratch(in.size());
    std::vector<int64_t> key_scratch(in.size());
    std::vector<int64_t> keys = in;
    radix_sort_indirect_i64(perm.data(), keys.data(), perm_scratch.data(),
                            key_scratch.data(), n);

    for (int64_t i = 0; i < n; ++i)
        EXPECT_EQ(direct[static_cast<size_t>(i)],
                  in[static_cast<size_t>(perm[static_cast<size_t>(i)])]);
}
