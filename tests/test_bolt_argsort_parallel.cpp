// Tests for bolt::kernels::parallel_merge_sort_indirect_u32 — the PARALLEL
// multi-block indirect (argsort) sort. Verifies that for large random u32 and
// i64 key arrays the produced permutation is (a) a valid permutation of [0,n),
// (b) fully non-decreasing under the comparator, and (c) byte-identical in
// ordering to a std::sort reference (stable order on equal keys). Also prints
// ns/element single-threaded vs parallel at N = 1,000,000 and the speedup.

#include "bolt/kernels/bolt_argsort.h"
#include "bolt/kernels/bolt_argsort_parallel.h"
#include "bolt/bolt_scheduler.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

using bolt::kernels::merge_sort_indirect_u32;
using bolt::kernels::parallel_merge_sort_indirect_u32;

// A valid permutation of [0,n): every index appears exactly once.
template <typename T>
bool is_permutation_of_range(const std::vector<T>& perm) {
    std::vector<uint8_t> seen(perm.size(), 0);
    for (T v : perm) {
        if (v >= perm.size()) return false;
        if (seen[v]) return false;
        seen[v] = 1;
    }
    return true;
}

// std::stable_sort reference permutation under `keys` (ascending, stable).
template <typename Key>
std::vector<uint32_t> reference_perm(const std::vector<Key>& keys) {
    std::vector<uint32_t> ref(keys.size());
    for (uint32_t i = 0; i < keys.size(); ++i) ref[i] = i;
    std::stable_sort(ref.begin(), ref.end(),
                     [&](uint32_t a, uint32_t b) { return keys[a] < keys[b]; });
    return ref;
}

template <typename Key>
void check_fully_sorted(const std::vector<uint32_t>& perm,
                        const std::vector<Key>& keys) {
    for (size_t i = 1; i < perm.size(); ++i) {
        ASSERT_FALSE(keys[perm[i]] < keys[perm[i - 1]])
            << "not non-decreasing at i=" << i;
    }
}

}  // namespace

TEST(ArgsortParallel, U32KeysPermutationSortedAndStable) {
    constexpr int64_t kN = 1'200'000;  // the G2FEAT-115 ORDER BY size
    std::mt19937 rng(12345);
    std::uniform_int_distribution<uint32_t> dist(0, 5000);  // many ties → stable
    std::vector<uint32_t> keys(kN);
    for (auto& k : keys) k = dist(rng);

    bolt::Scheduler sched;
    ASSERT_TRUE(sched.init(0));  // 0 -> hardware concurrency

    std::vector<uint32_t> perm(kN), scratch(kN);
    auto less = [&](uint32_t a, uint32_t b) { return keys[a] < keys[b]; };
    parallel_merge_sort_indirect_u32(perm.data(), scratch.data(), kN, less, &sched);

    EXPECT_TRUE(is_permutation_of_range(perm));
    check_fully_sorted(perm, keys);
    EXPECT_EQ(perm, reference_perm(keys));  // identical stable ordering

    sched.shutdown();
}

TEST(ArgsortParallel, I64KeysPermutationSortedAndStable) {
    constexpr int64_t kN = 1'000'000;
    std::mt19937_64 rng(987654321ULL);
    std::uniform_int_distribution<int64_t> dist(-1'000'000, 1'000'000);
    std::vector<int64_t> keys(kN);
    for (auto& k : keys) k = dist(rng);

    bolt::Scheduler sched;
    ASSERT_TRUE(sched.init(0));

    std::vector<uint32_t> perm(kN), scratch(kN);
    auto less = [&](uint32_t a, uint32_t b) { return keys[a] < keys[b]; };
    parallel_merge_sort_indirect_u32(perm.data(), scratch.data(), kN, less, &sched);

    EXPECT_TRUE(is_permutation_of_range(perm));
    check_fully_sorted(perm, keys);
    EXPECT_EQ(perm, reference_perm(keys));

    sched.shutdown();
}

TEST(ArgsortParallel, SmallInputFallsBackCorrectly) {
    // Below kParallelSortMinRows → single-threaded fallback path.
    constexpr int64_t kN = 1000;
    std::mt19937 rng(7);
    std::uniform_int_distribution<int64_t> dist(0, 100);
    std::vector<int64_t> keys(kN);
    for (auto& k : keys) k = dist(rng);

    bolt::Scheduler sched;
    ASSERT_TRUE(sched.init(0));

    std::vector<uint32_t> perm(kN), scratch(kN);
    auto less = [&](uint32_t a, uint32_t b) { return keys[a] < keys[b]; };
    parallel_merge_sort_indirect_u32(perm.data(), scratch.data(), kN, less, &sched);

    EXPECT_TRUE(is_permutation_of_range(perm));
    EXPECT_EQ(perm, reference_perm(keys));
    sched.shutdown();
}

TEST(ArgsortParallel, Benchmark1MSingleVsParallel) {
    constexpr int64_t kN = 1'000'000;
    std::mt19937_64 rng(2024);
    std::uniform_int_distribution<int64_t> dist(-(1LL << 40), (1LL << 40));
    std::vector<int64_t> keys(kN);
    for (auto& k : keys) k = dist(rng);
    auto less = [&](uint32_t a, uint32_t b) { return keys[a] < keys[b]; };

    std::vector<uint32_t> perm_s(kN), scratch_s(kN);
    std::vector<uint32_t> perm_p(kN), scratch_p(kN);

    using Clock = std::chrono::steady_clock;

    // Single-threaded reference (the existing kernel).
    double best_single = 1e30;
    for (int rep = 0; rep < 3; ++rep) {
        for (int64_t i = 0; i < kN; ++i) perm_s[i] = static_cast<uint32_t>(i);
        auto t0 = Clock::now();
        merge_sort_indirect_u32(perm_s.data(), scratch_s.data(), kN, less);
        auto t1 = Clock::now();
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        best_single = std::min(best_single, ns);
    }

    // Parallel.
    bolt::Scheduler sched;
    ASSERT_TRUE(sched.init(0));
    const uint32_t workers = sched.thread_count();
    double best_par = 1e30;
    for (int rep = 0; rep < 3; ++rep) {
        auto t0 = Clock::now();
        parallel_merge_sort_indirect_u32(perm_p.data(), scratch_p.data(), kN,
                                         less, &sched);
        auto t1 = Clock::now();
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        best_par = std::min(best_par, ns);
    }
    sched.shutdown();

    // Correctness parity: both produce the same stable permutation.
    EXPECT_EQ(perm_s, perm_p);

    const double ns_per_elem_single = best_single / static_cast<double>(kN);
    const double ns_per_elem_par = best_par / static_cast<double>(kN);
    std::printf(
        "\n[argsort N=%lld] workers=%u  single=%.3f ns/elem  parallel=%.3f "
        "ns/elem  speedup=%.2fx\n",
        static_cast<long long>(kN), workers,
        ns_per_elem_single, ns_per_elem_par, best_single / best_par);
    std::fflush(stdout);

    EXPECT_LT(ns_per_elem_par, ns_per_elem_single);  // parallel must be faster
}
