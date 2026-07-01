// test_bolt_radixsort_parallel.cpp — parallel multi-key radix argsort.
//
// The kernel MSD-partitions by the primary key's top byte, so it only wins (and
// only engages) when that byte carries entropy — a well-spread split. On a
// skewed split (narrow-range primary → one giant bucket) it DECLINES (returns
// false) so the caller keeps the parallel merge. These tests cover both:
//   * PROCEEDS on high-entropy keys → byte-identical to std::stable_sort.
//   * DECLINES on a degenerate (constant / narrow) primary → returns false.
// Correctness is thread-count independent, so the gate holds on any host.

#include "bolt/kernels/bolt_radixsort_parallel.h"
#include "bolt/kernels/bolt_radixsort.h"
#include "bolt/bolt_scheduler.h"
#include "bolt/bolt_arena.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <thread>
#include <vector>

using namespace bolt::kernels;

namespace {

uint32_t hw_threads() { return std::max(2u, std::thread::hardware_concurrency()); }

// Multi-key: primary int64 DESC, secondary int64 ASC. Full-range primary → top
// byte is uniform → the kernel proceeds. Asserts byte-identical to stable sort.
void check_multikey_proceeds(int64_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> full(std::numeric_limits<int64_t>::min(),
                                                std::numeric_limits<int64_t>::max());
    std::uniform_int_distribution<int64_t> small(0, 32);  // ties on the primary
    std::vector<int64_t> pk(static_cast<size_t>(n)), sk(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        pk[static_cast<size_t>(i)] = full(rng) / 3;   // spread with occasional ties
        sk[static_cast<size_t>(i)] = small(rng);
    }
    std::vector<uint64_t> b0(static_cast<size_t>(n)), b1(static_cast<size_t>(n));
    radix_fill_biased_i64(b0.data(), pk.data(), n, /*ascending=*/false);  // DESC
    radix_fill_biased_i64(b1.data(), sk.data(), n, /*ascending=*/true);   // ASC
    const uint64_t* keys[2] = {b0.data(), b1.data()};

    std::vector<uint32_t> perm(static_cast<size_t>(n)), scratch(static_cast<size_t>(n));
    bolt::Scheduler sched; ASSERT_TRUE(sched.init(hw_threads()));
    bolt::Arena arena;
    const bool proceeded = parallel_radix_argsort_multikey_u64(
        perm.data(), scratch.data(), keys, 2, n, &sched, &arena);
    sched.shutdown();
    ASSERT_TRUE(proceeded) << "full-range primary should have enough entropy";

    std::vector<uint32_t> ref(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) ref[static_cast<size_t>(i)] = static_cast<uint32_t>(i);
    std::stable_sort(ref.begin(), ref.end(), [&](uint32_t a, uint32_t b) {
        if (pk[a] != pk[b]) return pk[a] > pk[b];   // DESC
        return sk[a] < sk[b];                        // ASC
    });
    for (int64_t i = 0; i < n; ++i)
        ASSERT_EQ(perm[static_cast<size_t>(i)], ref[static_cast<size_t>(i)])
            << "row " << i << " (n=" << n << ")";
}

}  // namespace

TEST(ParallelRadix, MultiKeyProceedsLarge)   { check_multikey_proceeds(200'000, 42); }
TEST(ParallelRadix, MultiKeyProceedsMidsize) { check_multikey_proceeds(50'000,  7); }

// Single full-range int64 key → proceeds → byte-identical.
TEST(ParallelRadix, SingleKeyProceeds) {
    const int64_t n = 150'000;
    std::mt19937_64 rng(123);
    std::uniform_int_distribution<int64_t> full(std::numeric_limits<int64_t>::min(),
                                                std::numeric_limits<int64_t>::max());
    std::vector<int64_t> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) v[static_cast<size_t>(i)] = full(rng);
    std::vector<uint64_t> b0(static_cast<size_t>(n));
    radix_fill_biased_i64(b0.data(), v.data(), n, /*ascending=*/true);
    const uint64_t* keys[1] = {b0.data()};

    std::vector<uint32_t> perm(static_cast<size_t>(n)), scratch(static_cast<size_t>(n));
    bolt::Scheduler sched; ASSERT_TRUE(sched.init(hw_threads()));
    bolt::Arena arena;
    ASSERT_TRUE(parallel_radix_argsort_multikey_u64(
        perm.data(), scratch.data(), keys, 1, n, &sched, &arena));
    sched.shutdown();

    std::vector<uint32_t> ref(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) ref[static_cast<size_t>(i)] = static_cast<uint32_t>(i);
    std::stable_sort(ref.begin(), ref.end(), [&](uint32_t a, uint32_t b) { return v[a] < v[b]; });
    for (int64_t i = 0; i < n; ++i)
        ASSERT_EQ(perm[static_cast<size_t>(i)], ref[static_cast<size_t>(i)]) << "row " << i;
}

// Small n → serial fallback (proceeds), still byte-identical.
TEST(ParallelRadix, SmallFallback) {
    const int64_t n = 1'000;
    std::mt19937_64 rng(9);
    std::uniform_int_distribution<int64_t> full(std::numeric_limits<int64_t>::min(),
                                                std::numeric_limits<int64_t>::max());
    std::vector<int64_t> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) v[static_cast<size_t>(i)] = full(rng);
    std::vector<uint64_t> b0(static_cast<size_t>(n));
    radix_fill_biased_i64(b0.data(), v.data(), n, true);
    const uint64_t* keys[1] = {b0.data()};
    std::vector<uint32_t> perm(static_cast<size_t>(n)), scratch(static_cast<size_t>(n));
    bolt::Scheduler sched; ASSERT_TRUE(sched.init(hw_threads()));
    bolt::Arena arena;
    ASSERT_TRUE(parallel_radix_argsort_multikey_u64(
        perm.data(), scratch.data(), keys, 1, n, &sched, &arena));
    sched.shutdown();
    std::vector<uint32_t> ref(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) ref[static_cast<size_t>(i)] = static_cast<uint32_t>(i);
    std::stable_sort(ref.begin(), ref.end(), [&](uint32_t a, uint32_t b) { return v[a] < v[b]; });
    for (int64_t i = 0; i < n; ++i) ASSERT_EQ(perm[static_cast<size_t>(i)], ref[static_cast<size_t>(i)]);
}

// Constant primary → one giant MSD bucket → the kernel DECLINES (caller merges).
TEST(ParallelRadix, SkewedPrimaryDeclines) {
    const int64_t n = 100'000;
    std::vector<int64_t> pk(static_cast<size_t>(n), 7);   // constant → 1 bucket
    std::vector<int64_t> sk(static_cast<size_t>(n));
    std::mt19937_64 rng(5);
    std::uniform_int_distribution<int64_t> td(0, 1'000'000);
    for (int64_t i = 0; i < n; ++i) sk[static_cast<size_t>(i)] = td(rng);
    std::vector<uint64_t> b0(static_cast<size_t>(n)), b1(static_cast<size_t>(n));
    radix_fill_biased_i64(b0.data(), pk.data(), n, true);
    radix_fill_biased_i64(b1.data(), sk.data(), n, true);
    const uint64_t* keys[2] = {b0.data(), b1.data()};
    std::vector<uint32_t> perm(static_cast<size_t>(n)), scratch(static_cast<size_t>(n));
    bolt::Scheduler sched; ASSERT_TRUE(sched.init(hw_threads()));
    bolt::Arena arena;
    // One bucket holds all n rows (>> n/8) → decline so the caller keeps the merge.
    ASSERT_FALSE(parallel_radix_argsort_multikey_u64(
        perm.data(), scratch.data(), keys, 2, n, &sched, &arena));
    sched.shutdown();
}
