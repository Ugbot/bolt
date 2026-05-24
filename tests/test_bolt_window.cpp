// test_bolt_window.cpp — coverage + microbench for bolt::kernels::window.
//
// Window function ranking kernels over already-partitioned + sorted input.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "bolt/kernels/bolt_window.h"

using bolt::kernels::window::row_number;
using bolt::kernels::window::rank;
using bolt::kernels::window::dense_rank;
using bolt::kernels::window::lag;
using bolt::kernels::window::lead;
using bolt::kernels::window::first_value;
using bolt::kernels::window::last_value;
using bolt::kernels::window::ntile;

// ============================================================================
// ROW_NUMBER
// ============================================================================

TEST(BoltWindow, RowNumberSinglePartition) {
    const int64_t pk[8] = {7, 7, 7, 7, 7, 7, 7, 7};
    int32_t out[8] = {};
    row_number(pk, 8, out);
    for (int i = 0; i < 8; ++i) EXPECT_EQ(out[i], i + 1);
}

TEST(BoltWindow, RowNumberMultiPartition) {
    const int64_t pk[8] = {1, 1, 1, 2, 2, 3, 3, 3};
    int32_t out[8] = {};
    row_number(pk, 8, out);
    const int32_t want[8] = {1, 2, 3, 1, 2, 1, 2, 3};
    for (int i = 0; i < 8; ++i) EXPECT_EQ(out[i], want[i]) << "i=" << i;
}

TEST(BoltWindow, RowNumberEmpty) {
    int32_t out[1] = {99};
    row_number<int64_t>(nullptr, 0, nullptr);
    EXPECT_EQ(out[0], 99);
}

// ============================================================================
// RANK
// ============================================================================

TEST(BoltWindow, RankTiesWithGaps) {
    //                                    p   p   p   p   p   p   p
    const int32_t pk[7] = {1, 1, 1, 1, 2, 2, 2};
    const int32_t ok[7] = {10, 20, 20, 30, 5, 5, 9};
    int32_t out[7] = {};
    rank(pk, ok, 7, out);
    const int32_t want[7] = {1, 2, 2, 4, 1, 1, 3};
    for (int i = 0; i < 7; ++i) EXPECT_EQ(out[i], want[i]) << "i=" << i;
}

TEST(BoltWindow, RankAllTies) {
    const int32_t pk[5] = {1, 1, 1, 1, 1};
    const int32_t ok[5] = {7, 7, 7, 7, 7};
    int32_t out[5] = {};
    rank(pk, ok, 5, out);
    for (int i = 0; i < 5; ++i) EXPECT_EQ(out[i], 1);
}

// ============================================================================
// DENSE_RANK
// ============================================================================

TEST(BoltWindow, DenseRankTiesNoGaps) {
    const int32_t pk[7] = {1, 1, 1, 1, 2, 2, 2};
    const int32_t ok[7] = {10, 20, 20, 30, 5, 5, 9};
    int32_t out[7] = {};
    dense_rank(pk, ok, 7, out);
    const int32_t want[7] = {1, 2, 2, 3, 1, 1, 2};
    for (int i = 0; i < 7; ++i) EXPECT_EQ(out[i], want[i]) << "i=" << i;
}

// ============================================================================
// LAG / LEAD
// ============================================================================

TEST(BoltWindow, LagBasic) {
    const int32_t pk[6]  = {1, 1, 1, 2, 2, 2};
    const int64_t v[6]   = {10, 20, 30, 40, 50, 60};
    int64_t out[6] = {};
    lag<int64_t, int32_t>(v, pk, 6, 1, -1, out);
    const int64_t want[6] = {-1, 10, 20, -1, 40, 50};
    for (int i = 0; i < 6; ++i) EXPECT_EQ(out[i], want[i]) << "i=" << i;
}

TEST(BoltWindow, LagOffset2) {
    const int32_t pk[5] = {1, 1, 1, 1, 1};
    const int64_t v[5]  = {10, 20, 30, 40, 50};
    int64_t out[5] = {};
    lag<int64_t, int32_t>(v, pk, 5, 2, -999, out);
    const int64_t want[5] = {-999, -999, 10, 20, 30};
    for (int i = 0; i < 5; ++i) EXPECT_EQ(out[i], want[i]);
}

TEST(BoltWindow, LeadBasic) {
    const int32_t pk[6]  = {1, 1, 1, 2, 2, 2};
    const int64_t v[6]   = {10, 20, 30, 40, 50, 60};
    int64_t out[6] = {};
    lead<int64_t, int32_t>(v, pk, 6, 1, -1, out);
    const int64_t want[6] = {20, 30, -1, 50, 60, -1};
    for (int i = 0; i < 6; ++i) EXPECT_EQ(out[i], want[i]) << "i=" << i;
}

// ============================================================================
// FIRST_VALUE / LAST_VALUE
// ============================================================================

TEST(BoltWindow, FirstValue) {
    const int32_t pk[6] = {1, 1, 1, 2, 2, 2};
    const int64_t v[6]  = {10, 20, 30, 40, 50, 60};
    int64_t out[6] = {};
    first_value(v, pk, 6, out);
    const int64_t want[6] = {10, 10, 10, 40, 40, 40};
    for (int i = 0; i < 6; ++i) EXPECT_EQ(out[i], want[i]);
}

TEST(BoltWindow, LastValue) {
    const int32_t pk[6] = {1, 1, 1, 2, 2, 2};
    const int64_t v[6]  = {10, 20, 30, 40, 50, 60};
    int64_t out[6] = {};
    last_value(v, pk, 6, out);
    const int64_t want[6] = {30, 30, 30, 60, 60, 60};
    for (int i = 0; i < 6; ++i) EXPECT_EQ(out[i], want[i]);
}

// ============================================================================
// NTILE
// ============================================================================

TEST(BoltWindow, NtileEvenDivide) {
    const int32_t pk[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    int32_t out[8] = {};
    ntile(pk, 8, 4, out);
    const int32_t want[8] = {1, 1, 2, 2, 3, 3, 4, 4};
    for (int i = 0; i < 8; ++i) EXPECT_EQ(out[i], want[i]);
}

TEST(BoltWindow, NtileWithRemainder) {
    // size=7, n_tiles=3 -> ceil=3,3,1? actually 3,2,2
    const int32_t pk[7] = {1, 1, 1, 1, 1, 1, 1};
    int32_t out[7] = {};
    ntile(pk, 7, 3, out);
    const int32_t want[7] = {1, 1, 1, 2, 2, 3, 3};
    for (int i = 0; i < 7; ++i) EXPECT_EQ(out[i], want[i]) << "i=" << i;
}

TEST(BoltWindow, NtileSmallerThanTiles) {
    // size=2, n_tiles=5 -> each row in its own bucket: {1,2}
    const int32_t pk[2] = {1, 1};
    int32_t out[2] = {};
    ntile(pk, 2, 5, out);
    EXPECT_EQ(out[0], 1);
    EXPECT_EQ(out[1], 2);
}

TEST(BoltWindow, NtileMultiPartition) {
    const int32_t pk[6] = {1, 1, 1, 1, 2, 2};
    int32_t out[6] = {};
    ntile(pk, 6, 2, out);
    const int32_t want[6] = {1, 1, 2, 2, 1, 2};
    for (int i = 0; i < 6; ++i) EXPECT_EQ(out[i], want[i]);
}

// ============================================================================
// Microbench — runs as part of the test binary so we get a printout.
// Thresholds are *informative*; the gate is the perf doc.
// ============================================================================

namespace {

template <typename F>
double bench_ns_per_row(F&& f, int64_t n, int iters) {
    using clk = std::chrono::high_resolution_clock;
    // warmup
    f();
    const auto t0 = clk::now();
    for (int i = 0; i < iters; ++i) f();
    const auto t1 = clk::now();
    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    return ns / (static_cast<double>(n) * iters);
}

}  // namespace

TEST(BoltWindowBench, RowNumberThroughput) {
    constexpr int64_t N = 1 << 16;
    std::vector<int64_t> pk(N);
    std::vector<int32_t> out(N);
    // 64 partitions of 1024 rows each.
    for (int64_t i = 0; i < N; ++i) pk[i] = i >> 10;
    const double ns_row = bench_ns_per_row(
        [&]() { row_number(pk.data(), N, out.data()); }, N, 200);
    std::printf("[bench] row_number<i64>     %.3f ns/row\n", ns_row);
    EXPECT_LT(ns_row, 8.0);  // generous CI bound; target ≤ 2.0
}

TEST(BoltWindowBench, RankThroughput) {
    constexpr int64_t N = 1 << 16;
    std::vector<int64_t> pk(N), ok(N);
    std::vector<int32_t> out(N);
    for (int64_t i = 0; i < N; ++i) {
        pk[i] = i >> 10;
        ok[i] = i & 3;            // ties every 4
    }
    const double ns_row = bench_ns_per_row(
        [&]() { rank(pk.data(), ok.data(), N, out.data()); }, N, 200);
    std::printf("[bench] rank<i64,i64>       %.3f ns/row\n", ns_row);
    EXPECT_LT(ns_row, 8.0);  // target ≤ 3.0
}

TEST(BoltWindowBench, DenseRankThroughput) {
    constexpr int64_t N = 1 << 16;
    std::vector<int64_t> pk(N), ok(N);
    std::vector<int32_t> out(N);
    for (int64_t i = 0; i < N; ++i) {
        pk[i] = i >> 10;
        ok[i] = i & 3;
    }
    const double ns_row = bench_ns_per_row(
        [&]() { dense_rank(pk.data(), ok.data(), N, out.data()); }, N, 200);
    std::printf("[bench] dense_rank<i64,i64> %.3f ns/row\n", ns_row);
    EXPECT_LT(ns_row, 8.0);
}

TEST(BoltWindowBench, Lag1Throughput) {
    constexpr int64_t N = 1 << 16;
    std::vector<int64_t> pk(N), v(N), out(N);
    for (int64_t i = 0; i < N; ++i) { pk[i] = i >> 10; v[i] = i; }
    const double ns_row = bench_ns_per_row(
        [&]() { lag<int64_t, int64_t>(v.data(), pk.data(), N, 1, -1, out.data()); },
        N, 200);
    std::printf("[bench] lag<i64> k=1        %.3f ns/row\n", ns_row);
    EXPECT_LT(ns_row, 8.0);  // CI guard; target ≤ 1.0 with /arch:AVX2
}

TEST(BoltWindowBench, LeadThroughput) {
    constexpr int64_t N = 1 << 16;
    std::vector<int64_t> pk(N), v(N), out(N);
    for (int64_t i = 0; i < N; ++i) { pk[i] = i >> 10; v[i] = i; }
    const double ns_row = bench_ns_per_row(
        [&]() { lead<int64_t, int64_t>(v.data(), pk.data(), N, 1, -1, out.data()); },
        N, 200);
    std::printf("[bench] lead<i64> k=1       %.3f ns/row\n", ns_row);
    EXPECT_LT(ns_row, 8.0);
}

TEST(BoltWindowBench, FirstValueThroughput) {
    constexpr int64_t N = 1 << 16;
    std::vector<int64_t> pk(N), v(N), out(N);
    for (int64_t i = 0; i < N; ++i) { pk[i] = i >> 10; v[i] = i; }
    const double ns_row = bench_ns_per_row(
        [&]() { first_value(v.data(), pk.data(), N, out.data()); }, N, 200);
    std::printf("[bench] first_value<i64>    %.3f ns/row\n", ns_row);
    EXPECT_LT(ns_row, 8.0);
}
