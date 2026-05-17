// test_bolt_hll.cpp — correctness + microbench for the HyperLogLog++
// sketch in bolt/kernels/bolt_hll.h.
//
// Correctness: error rate within 2% on cardinalities 100, 10K, 1M,
// 100M (synthetic distinct keys, hashed with bolt::swiss_mix).
// Merge associativity: merge(merge(A,B),C) == merge(A,merge(B,C)).
// init zeros the sketch.
//
// Bench (recorded in test output, not gating): hll_add_batch ns/row,
// hll_estimate / hll_merge total latency at P=14.

#include "bolt/kernels/bolt_hll.h"
#include "bolt/bolt_hash.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

using bolt::kernels::hll::HllPp;
using bolt::kernels::hll::hll_init;
using bolt::kernels::hll::hll_add_batch;
using bolt::kernels::hll::hll_add_hash;
using bolt::kernels::hll::hll_estimate;
using bolt::kernels::hll::hll_merge;

// Deterministic distinct-key stream: i in [0, n) hashed once per call.
// Insert each distinct key 3x to stress register-update idempotence.
template <uint8_t P>
static double measure_error(uint64_t n_distinct, uint64_t seed = 0xC0FFEEu) {
    HllPp<P> s;
    hll_init(&s);
    // Sequential keys via hll_add_u64 — exercises the double-mix
    // path that gives full avalanche for ascending integer streams.
    for (uint64_t i = 0; i < n_distinct; ++i) {
        const uint64_t key = i + seed;
        hll_add_u64<P>(&s, key);
        // Re-insert to confirm idempotence under register-max.
        hll_add_u64<P>(&s, key);
    }
    const uint64_t est = hll_estimate(&s);
    const double err   = std::fabs(static_cast<double>(est) -
                                   static_cast<double>(n_distinct))
                       / static_cast<double>(n_distinct);
    std::printf("[HLL P=%u] n=%llu est=%llu err=%.4f%%\n",
                static_cast<unsigned>(P),
                static_cast<unsigned long long>(n_distinct),
                static_cast<unsigned long long>(est),
                err * 100.0);
    return err;
}

// ---------------------------------------------------------------------------
// Init / basic invariants.
// ---------------------------------------------------------------------------

TEST(BoltHll, InitZerosRegisters) {
    HllPp<14> s;
    std::memset(&s, 0xFFu, sizeof(s));
    hll_init(&s);
    for (uint32_t i = 0; i < HllPp<14>::k_num_registers; ++i) {
        ASSERT_EQ(s.registers[i], 0u) << "register " << i << " not zero";
    }
}

TEST(BoltHll, EmptySketchEstimatesZero) {
    HllPp<14> s;
    hll_init(&s);
    EXPECT_EQ(hll_estimate(&s), 0u);
}

TEST(BoltHll, SmallExactRange) {
    // With m=16384 and V=16384 - distinct hits, linear counting is
    // near-exact for cardinalities up to a few hundred.
    HllPp<14> s;
    hll_init(&s);
    for (uint64_t i = 0; i < 10; ++i) {
        hll_add_u64(&s, i);
    }
    const uint64_t est = hll_estimate(&s);
    // Up to one collision possible in linear counting but extremely unlikely
    // at 10 distinct hashes on 16384 buckets.
    EXPECT_GE(est, 9u);
    EXPECT_LE(est, 11u);
}

// ---------------------------------------------------------------------------
// Error-rate gates — within 2% on the four target cardinalities.
// ---------------------------------------------------------------------------

TEST(BoltHll, ErrorRate_100)        { EXPECT_LT(measure_error<14>(100),         0.02); }
TEST(BoltHll, ErrorRate_10K)        { EXPECT_LT(measure_error<14>(10'000),      0.02); }
TEST(BoltHll, ErrorRate_1M)         { EXPECT_LT(measure_error<14>(1'000'000),   0.02); }
TEST(BoltHll, ErrorRate_100M)       { EXPECT_LT(measure_error<14>(100'000'000), 0.02); }

// ---------------------------------------------------------------------------
// Merge correctness + associativity.
// ---------------------------------------------------------------------------

TEST(BoltHll, MergeMatchesUnion) {
    // Build A over [0, N), B over [N/2, 3N/2). Merge(A, B) should
    // estimate the union ≈ 3N/2 within HLL error.
    constexpr uint64_t N = 200'000;
    HllPp<14> a, b, both;
    hll_init(&a);
    hll_init(&b);
    hll_init(&both);
    for (uint64_t i = 0; i < N; ++i) {
        hll_add_u64(&a, i);
    }
    for (uint64_t i = N / 2; i < (3 * N) / 2; ++i) {
        hll_add_u64(&b, i);
    }
    // Reference union via a single combined sketch.
    for (uint64_t i = 0; i < (3 * N) / 2; ++i) {
        hll_add_u64(&both, i);
    }
    HllPp<14> merged = a;
    hll_merge(&merged, &b);

    // Registers must match bit-exactly: a ∪ b == both is structural.
    for (uint32_t i = 0; i < HllPp<14>::k_num_registers; ++i) {
        ASSERT_EQ(merged.registers[i], both.registers[i]) << "reg " << i;
    }
}

TEST(BoltHll, MergeAssociativity) {
    constexpr uint64_t N = 50'000;
    HllPp<14> a, b, c;
    hll_init(&a); hll_init(&b); hll_init(&c);
    for (uint64_t i = 0;       i < N;     ++i) hll_add_u64(&a, i);
    for (uint64_t i = N;       i < 2 * N; ++i) hll_add_u64(&b, i);
    for (uint64_t i = 2 * N;   i < 3 * N; ++i) hll_add_u64(&c, i);

    HllPp<14> left  = a;
    hll_merge(&left, &b);
    hll_merge(&left, &c);

    HllPp<14> right = b;
    hll_merge(&right, &c);
    HllPp<14> right_full = a;
    hll_merge(&right_full, &right);

    for (uint32_t i = 0; i < HllPp<14>::k_num_registers; ++i) {
        ASSERT_EQ(left.registers[i], right_full.registers[i]) << "reg " << i;
    }
}

// ---------------------------------------------------------------------------
// Microbenchmark — reported via printf, not gated. Captures ns/row for
// hll_add_batch and total latency for estimate + merge so we can chase
// the 3 ns/row / 1 µs / 0.5 µs targets in the wave doc.
// ---------------------------------------------------------------------------

TEST(BoltHll, Microbench) {
    constexpr uint64_t N = 1'000'000;
    std::vector<uint64_t> hashes(N);
    std::mt19937_64 rng(0xDEADBEEFu);
    for (uint64_t i = 0; i < N; ++i) {
        hashes[i] = bolt::swiss_mix(rng());
    }

    HllPp<14> s;
    hll_init(&s);

    // add_batch
    auto t0 = std::chrono::steady_clock::now();
    constexpr int reps = 4;
    for (int r = 0; r < reps; ++r) {
        hll_add_batch<14>(&s, hashes.data(), static_cast<int64_t>(N));
    }
    auto t1 = std::chrono::steady_clock::now();
    const double ns_total = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    const double ns_per_row = ns_total / static_cast<double>(N * reps);

    // estimate — loop to amortize chrono overhead.
    constexpr int est_reps = 256;
    volatile uint64_t est_sink = 0;
    auto e0 = std::chrono::steady_clock::now();
    for (int r = 0; r < est_reps; ++r) est_sink ^= hll_estimate(&s);
    auto e1 = std::chrono::steady_clock::now();
    const double ns_estimate = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(e1 - e0).count())
        / static_cast<double>(est_reps);
    volatile uint64_t est = est_sink;

    // merge — loop too.
    HllPp<14> other;
    hll_init(&other);
    hll_add_batch<14>(&other, hashes.data(), static_cast<int64_t>(N));
    constexpr int merge_reps = 256;
    auto m0 = std::chrono::steady_clock::now();
    for (int r = 0; r < merge_reps; ++r) hll_merge(&s, &other);
    auto m1 = std::chrono::steady_clock::now();
    const double ns_merge = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(m1 - m0).count())
        / static_cast<double>(merge_reps);

    std::printf("[HLL bench P=14] add_batch=%.2f ns/row  estimate=%.0f ns  merge=%.0f ns  est=%llu\n",
                ns_per_row, ns_estimate, ns_merge,
                static_cast<unsigned long long>(est));
    SUCCEED();
}

}  // namespace
