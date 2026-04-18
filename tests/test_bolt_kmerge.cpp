// test_bolt_kmerge.cpp — Coverage for the k-way sorted-run merge kernel.
//
// Tests the contract documented in bolt_kmerge.h:
//   - 0 / 1 / 2 / 64 input shapes
//   - empty, single-key, equal-length, highly unbalanced runs
//   - output is non-decreasing and contains exactly the multiset union
//   - duplicates within one input preserve arrival order (implied by
//     monotonic pos advance)

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <vector>

#include "bolt/kernels/bolt_kmerge.h"

using namespace bolt;

namespace {

// Build a KMergeInput over a std::vector<int64_t> backing store.
KMergeInput make_input(const std::vector<int64_t>& keys, uint32_t id) {
    KMergeInput in{};
    in.keys     = keys.data();
    in.pos      = 0;
    in.len      = static_cast<int64_t>(keys.size());
    in.input_id = id;
    return in;
}

// Verify that (out_id, out_key) for `count` rows:
//   - is non-decreasing in key
//   - matches the sorted multiset-union of all input keys
void verify_sorted_and_multiset(
    const std::vector<std::vector<int64_t>>& inputs,
    const std::vector<int32_t>& out_id,
    const std::vector<int64_t>& out_key,
    int64_t count) {
    // Non-decreasing.
    for (int64_t i = 1; i < count; ++i) {
        ASSERT_LE(out_key[i - 1], out_key[i]) << "output not sorted at " << i;
    }
    // Multiset union: concatenate every input, sort, compare to output.
    std::vector<int64_t> expected;
    for (const auto& run : inputs) {
        expected.insert(expected.end(), run.begin(), run.end());
    }
    std::sort(expected.begin(), expected.end());
    ASSERT_EQ(count, static_cast<int64_t>(expected.size()));
    for (int64_t i = 0; i < count; ++i) {
        ASSERT_EQ(out_key[i], expected[static_cast<size_t>(i)]) << "at " << i;
    }
    // input_id bounds: must match some input's id.
    for (int64_t i = 0; i < count; ++i) {
        ASSERT_GE(out_id[i], 0);
        ASSERT_LT(static_cast<size_t>(out_id[i]), inputs.size());
    }
}

int64_t total_len(const std::vector<std::vector<int64_t>>& inputs) {
    int64_t t = 0;
    for (const auto& r : inputs) t += static_cast<int64_t>(r.size());
    return t;
}

int64_t run_i64(const std::vector<std::vector<int64_t>>& runs,
                std::vector<int32_t>& out_id,
                std::vector<int64_t>& out_key) {
    std::vector<KMergeInput> inputs;
    inputs.reserve(runs.size());
    for (uint32_t i = 0; i < runs.size(); ++i) {
        inputs.push_back(make_input(runs[i], i));
    }
    const int64_t cap = total_len(runs);
    out_id.assign(static_cast<size_t>(cap), 0);
    out_key.assign(static_cast<size_t>(cap), 0);
    return kmerge_i64(inputs.data(), static_cast<uint32_t>(inputs.size()),
                      out_id.data(), out_key.data(), cap);
}

}  // namespace

// ---------------------------------------------------------------------------
// Shape tests
// ---------------------------------------------------------------------------

TEST(BoltKMerge, ZeroInputs) {
    int32_t dummy_id = 0;
    int64_t dummy_key = 0;
    const int64_t n = kmerge_i64(nullptr, 0, &dummy_id, &dummy_key, 0);
    EXPECT_EQ(n, 0);
}

TEST(BoltKMerge, SingleInputIdentity) {
    std::vector<std::vector<int64_t>> runs = {{1, 2, 5, 9, 12}};
    std::vector<int32_t> out_id;
    std::vector<int64_t> out_key;
    const int64_t n = run_i64(runs, out_id, out_key);
    ASSERT_EQ(n, 5);
    verify_sorted_and_multiset(runs, out_id, out_key, n);
    for (int64_t i = 0; i < n; ++i) EXPECT_EQ(out_id[i], 0);
}

TEST(BoltKMerge, TwoInputsFastPath) {
    std::vector<std::vector<int64_t>> runs = {
        {1, 3, 5, 7},
        {2, 4, 6, 8},
    };
    std::vector<int32_t> out_id;
    std::vector<int64_t> out_key;
    const int64_t n = run_i64(runs, out_id, out_key);
    ASSERT_EQ(n, 8);
    verify_sorted_and_multiset(runs, out_id, out_key, n);
    // Alternating source ids.
    const std::array<int32_t, 8> expect_id{0, 1, 0, 1, 0, 1, 0, 1};
    for (int64_t i = 0; i < 8; ++i) EXPECT_EQ(out_id[i], expect_id[i]);
}

TEST(BoltKMerge, TwoInputsEqualKeysFavourA) {
    // Equal keys: the 2-way fast path emits input 0 on ties (<=).
    std::vector<std::vector<int64_t>> runs = {
        {5, 5, 5},
        {5, 5},
    };
    std::vector<int32_t> out_id;
    std::vector<int64_t> out_key;
    const int64_t n = run_i64(runs, out_id, out_key);
    ASSERT_EQ(n, 5);
    verify_sorted_and_multiset(runs, out_id, out_key, n);
    const std::array<int32_t, 5> expect_id{0, 0, 0, 1, 1};
    for (int64_t i = 0; i < 5; ++i) EXPECT_EQ(out_id[i], expect_id[i]);
}

TEST(BoltKMerge, ThreeInputsTriggerHeapPath) {
    std::vector<std::vector<int64_t>> runs = {
        {1, 4, 7, 10},
        {2, 5, 8, 11},
        {3, 6, 9, 12},
    };
    std::vector<int32_t> out_id;
    std::vector<int64_t> out_key;
    const int64_t n = run_i64(runs, out_id, out_key);
    ASSERT_EQ(n, 12);
    verify_sorted_and_multiset(runs, out_id, out_key, n);
    for (int64_t i = 0; i < 12; ++i) EXPECT_EQ(out_key[i], i + 1);
}

TEST(BoltKMerge, EmptyRunsAreSkipped) {
    std::vector<std::vector<int64_t>> runs = {
        {},
        {10, 20, 30},
        {},
        {15, 25, 35},
        {},
    };
    std::vector<int32_t> out_id;
    std::vector<int64_t> out_key;
    const int64_t n = run_i64(runs, out_id, out_key);
    ASSERT_EQ(n, 6);
    verify_sorted_and_multiset(runs, out_id, out_key, n);
}

TEST(BoltKMerge, SingleKeyRuns) {
    std::vector<std::vector<int64_t>> runs;
    for (int i = 0; i < 10; ++i) runs.push_back({static_cast<int64_t>(i * 3)});
    std::vector<int32_t> out_id;
    std::vector<int64_t> out_key;
    const int64_t n = run_i64(runs, out_id, out_key);
    ASSERT_EQ(n, 10);
    verify_sorted_and_multiset(runs, out_id, out_key, n);
}

TEST(BoltKMerge, HighlyUnbalanced) {
    // One input is the bulk; nine others are tiny.
    std::vector<std::vector<int64_t>> runs;
    std::vector<int64_t> big;
    big.reserve(10000);
    for (int64_t v = 0; v < 10000; v += 2) big.push_back(v);  // 5000 evens
    runs.push_back(std::move(big));
    for (int i = 0; i < 9; ++i) {
        runs.push_back({static_cast<int64_t>(2 * i + 1)});
    }

    std::vector<int32_t> out_id;
    std::vector<int64_t> out_key;
    const int64_t n = run_i64(runs, out_id, out_key);
    ASSERT_EQ(n, 5000 + 9);
    verify_sorted_and_multiset(runs, out_id, out_key, n);
}

TEST(BoltKMerge, MaxFanIn64) {
    std::vector<std::vector<int64_t>> runs;
    for (uint32_t i = 0; i < kKMergeMaxInputs; ++i) {
        std::vector<int64_t> r;
        for (int j = 0; j < 16; ++j) {
            r.push_back(static_cast<int64_t>(j * kKMergeMaxInputs + i));
        }
        runs.push_back(std::move(r));
    }
    std::vector<int32_t> out_id;
    std::vector<int64_t> out_key;
    const int64_t n = run_i64(runs, out_id, out_key);
    ASSERT_EQ(n, 64 * 16);
    verify_sorted_and_multiset(runs, out_id, out_key, n);
    // This layout produces a perfectly ordered 0..1023 output.
    for (int64_t i = 0; i < n; ++i) EXPECT_EQ(out_key[i], i);
}

TEST(BoltKMerge, ExceedsMaxFanInRejected) {
    std::vector<KMergeInput> inputs(kKMergeMaxInputs + 1);
    int32_t out_id = 0;
    int64_t out_key = 0;
    const int64_t n = kmerge_i64(inputs.data(),
                                 kKMergeMaxInputs + 1,
                                 &out_id, &out_key, 1);
    EXPECT_EQ(n, -1);
}

TEST(BoltKMerge, BadInputReturnsNegative) {
    // Negative len is invalid.
    KMergeInput bad{};
    bad.keys = nullptr;
    bad.pos  = 0;
    bad.len  = -1;
    bad.input_id = 0;
    int32_t out_id = 0;
    int64_t out_key = 0;
    EXPECT_EQ(kmerge_i64(&bad, 1, &out_id, &out_key, 16), -1);
}

// ---------------------------------------------------------------------------
// Fuzz: 100 random trials, random fan-in 1..64, random run lengths 0..256
// ---------------------------------------------------------------------------

TEST(BoltKMerge, RandomFuzz) {
    std::mt19937_64 rng(0xC0FFEEULL);

    for (int trial = 0; trial < 100; ++trial) {
        std::uniform_int_distribution<uint32_t> fanin_dist(1u, kKMergeMaxInputs);
        std::uniform_int_distribution<int>      len_dist(0, 256);
        std::uniform_int_distribution<int64_t>  key_dist(-1000, 1000);

        const uint32_t fanin = fanin_dist(rng);
        std::vector<std::vector<int64_t>> runs(fanin);
        for (uint32_t i = 0; i < fanin; ++i) {
            const int len = len_dist(rng);
            runs[i].reserve(static_cast<size_t>(len));
            for (int j = 0; j < len; ++j) runs[i].push_back(key_dist(rng));
            std::sort(runs[i].begin(), runs[i].end());
        }

        std::vector<int32_t> out_id;
        std::vector<int64_t> out_key;
        const int64_t n = run_i64(runs, out_id, out_key);
        ASSERT_GE(n, 0) << "trial " << trial << " rejected";
        verify_sorted_and_multiset(runs, out_id, out_key, n);
    }
}

// ---------------------------------------------------------------------------
// u64 variant: verify unsigned ordering across the signed boundary.
// ---------------------------------------------------------------------------

TEST(BoltKMerge, U64SignedBoundaryOrdering) {
    // Reinterpret-cast signed int64 bits to uint64: INT64_MIN's bit pattern
    // is the u64 value 0x8000000000000000, larger than any positive u64.
    // So under u64 ordering, -1 (bits 0xFFFFFFFFFFFFFFFF = UINT64_MAX)
    // sorts LAST and positive ints sort first.
    std::vector<int64_t> a = {
        static_cast<int64_t>(0x0000000000000001ull),
        static_cast<int64_t>(0x7FFFFFFFFFFFFFFEull),
    };
    std::vector<int64_t> b = {
        static_cast<int64_t>(0x0000000000000002ull),
        static_cast<int64_t>(0x8000000000000000ull),  // == INT64_MIN, huge as u64
    };

    KMergeInput inputs[2];
    inputs[0] = make_input(a, 0);
    inputs[1] = make_input(b, 1);
    const int64_t cap = 4;
    std::vector<int32_t> out_id(cap, -1);
    std::vector<int64_t> out_key(cap, 0);
    const int64_t n = kmerge_u64(inputs, 2,
                                 out_id.data(), out_key.data(), cap);
    ASSERT_EQ(n, 4);

    // Expected ordering (unsigned): 1, 2, 0x7FFF..FE, 0x8000..00
    EXPECT_EQ(static_cast<uint64_t>(out_key[0]), 0x0000000000000001ull);
    EXPECT_EQ(static_cast<uint64_t>(out_key[1]), 0x0000000000000002ull);
    EXPECT_EQ(static_cast<uint64_t>(out_key[2]), 0x7FFFFFFFFFFFFFFEull);
    EXPECT_EQ(static_cast<uint64_t>(out_key[3]), 0x8000000000000000ull);
}
