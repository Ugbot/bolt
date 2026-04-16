// test_bolt_kernels.cpp — GTest suite for Wave A4 numeric kernels.
//
// Covers filter_{gt,lt,eq,ne,ge,le}, sum/min/max, cast saturation, and
// micro-adaptive filter_gt short-circuits based on ColumnStats zone map.

#include <gtest/gtest.h>

#include "bolt/bolt_types.h"
#include "bolt/bolt_column.h"
#include "bolt/kernels/bolt_numeric.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

using namespace bolt;
using namespace bolt::kernels;

// ---------------------------------------------------------------------------
// Filter: six ops over Int32, plus boundary conditions
// ---------------------------------------------------------------------------

TEST(BoltKernels, FilterGtInt32) {
    const int32_t data[] = {1, 5, 3, 9, 2, 7, 4, 8, 6, 0};
    int32_t out[10] = {};
    int64_t n = filter_gt<int32_t>(data, 10, 4, out);
    EXPECT_EQ(n, 5);
    // Indices where data[i] > 4: 1(5), 3(9), 5(7), 7(8), 8(6)
    EXPECT_EQ(out[0], 1); EXPECT_EQ(out[1], 3); EXPECT_EQ(out[2], 5);
    EXPECT_EQ(out[3], 7); EXPECT_EQ(out[4], 8);
}

TEST(BoltKernels, FilterLtInt64) {
    const int64_t data[] = {10, 20, 5, 30, 1, 40};
    int32_t out[6] = {};
    int64_t n = filter_lt<int64_t>(data, 6, 10, out);
    EXPECT_EQ(n, 2);
    EXPECT_EQ(out[0], 2); EXPECT_EQ(out[1], 4);
}

TEST(BoltKernels, FilterEqInt32) {
    const int32_t data[] = {7, 7, 0, 7, 3, 7};
    int32_t out[6] = {};
    int64_t n = filter_eq<int32_t>(data, 6, 7, out);
    EXPECT_EQ(n, 4);
}

TEST(BoltKernels, FilterNeInt32) {
    const int32_t data[] = {5, 5, 5, 6, 5};
    int32_t out[5] = {};
    int64_t n = filter_ne<int32_t>(data, 5, 5, out);
    EXPECT_EQ(n, 1);
    EXPECT_EQ(out[0], 3);
}

TEST(BoltKernels, FilterGeLeBoundary) {
    const int32_t data[] = {1, 2, 3, 4, 5};
    int32_t out[5] = {};
    EXPECT_EQ(filter_ge<int32_t>(data, 5, 3, out), 3);
    EXPECT_EQ(filter_le<int32_t>(data, 5, 3, out), 3);
    EXPECT_EQ(filter_ge<int32_t>(data, 5, 5, out), 1);
    EXPECT_EQ(filter_le<int32_t>(data, 5, 1, out), 1);
    EXPECT_EQ(filter_ge<int32_t>(data, 5, 6, out), 0);
    EXPECT_EQ(filter_le<int32_t>(data, 5, 0, out), 0);
}

TEST(BoltKernels, FilterFloat64) {
    const double data[] = {1.5, 2.5, 3.5, 4.5};
    int32_t out[4] = {};
    EXPECT_EQ(filter_gt<double>(data, 4, 2.5, out), 2);
    EXPECT_EQ(out[0], 2);
    EXPECT_EQ(out[1], 3);
}

TEST(BoltKernels, FilterEmpty) {
    int32_t out[1] = {};
    EXPECT_EQ(filter_gt<int32_t>(nullptr, 0, 0, out), 0);
}

TEST(BoltKernels, FilterSimdI32MatchesScalar) {
    // 17 elements — forces tail path after whole vectors.
    int32_t data[17];
    for (int i = 0; i < 17; ++i) data[i] = i;
    int32_t out_scalar[17] = {}, out_simd[17] = {};
    int64_t n1 = filter_gt<int32_t>(data, 17, 8, out_scalar);
    int64_t n2 = filter_gt_i32_simd(data, 17, 8, out_simd);
    EXPECT_EQ(n1, n2);
    EXPECT_EQ(n1, 8);
    for (int64_t i = 0; i < n1; ++i) EXPECT_EQ(out_scalar[i], out_simd[i]);
}

// ---------------------------------------------------------------------------
// Aggregates
// ---------------------------------------------------------------------------

TEST(BoltKernels, SumInt32Widens) {
    const int32_t data[] = {1, 2, 3, 4, 5};
    auto s = sum<int32_t>(data, 5);
    static_assert(std::is_same_v<decltype(s), int64_t>, "int32 sum widens to int64");
    EXPECT_EQ(s, 15);
}

TEST(BoltKernels, SumFloat32WidensToDouble) {
    const float data[] = {1.0f, 2.0f, 3.0f};
    auto s = sum<float>(data, 3);
    static_assert(std::is_same_v<decltype(s), double>, "float sum widens to double");
    EXPECT_DOUBLE_EQ(s, 6.0);
}

TEST(BoltKernels, MinMaxInt64) {
    const int64_t data[] = {5, 1, 9, 3, 7};
    EXPECT_EQ(min<int64_t>(data, 5), 1);
    EXPECT_EQ(max<int64_t>(data, 5), 9);
}

TEST(BoltKernels, CountIsN) {
    const int32_t data[] = {0, 0, 0};
    EXPECT_EQ(count<int32_t>(data, 3), 3);
}

TEST(BoltKernels, AvgFloat64) {
    const double data[] = {1.0, 2.0, 3.0, 4.0};
    EXPECT_DOUBLE_EQ(avg<double>(data, 4), 2.5);
}

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------

TEST(BoltKernels, AddSubMulDiv) {
    const int32_t a[] = {10, 20, 30, 40};
    const int32_t b[] = {1, 2, 3, 4};
    int32_t out[4] = {};
    add<int32_t>(a, b, 4, out);
    EXPECT_EQ(out[0], 11); EXPECT_EQ(out[3], 44);
    sub<int32_t>(a, b, 4, out);
    EXPECT_EQ(out[0], 9); EXPECT_EQ(out[3], 36);
    mul<int32_t>(a, b, 4, out);
    EXPECT_EQ(out[0], 10); EXPECT_EQ(out[3], 160);
    div<int32_t>(a, b, 4, out);
    EXPECT_EQ(out[0], 10); EXPECT_EQ(out[3], 10);
}

TEST(BoltKernels, DivByZeroInt) {
    const int32_t a[] = {10};
    const int32_t b[] = {0};
    int32_t out[1] = {42};
    div<int32_t>(a, b, 1, out);
    EXPECT_EQ(out[0], 0);  // saturated to 0 on integer div-by-zero
}

// ---------------------------------------------------------------------------
// Cast — narrowing saturation
// ---------------------------------------------------------------------------

TEST(BoltKernels, CastInt64ToInt32Saturates) {
    const int64_t src[] = {
        std::numeric_limits<int64_t>::max(),
        std::numeric_limits<int64_t>::min(),
        0,
        123,
        -456,
    };
    int32_t dst[5] = {};
    cast<int64_t, int32_t>(src, 5, dst);
    EXPECT_EQ(dst[0], std::numeric_limits<int32_t>::max());
    EXPECT_EQ(dst[1], std::numeric_limits<int32_t>::min());
    EXPECT_EQ(dst[2], 0);
    EXPECT_EQ(dst[3], 123);
    EXPECT_EQ(dst[4], -456);
}

TEST(BoltKernels, CastInt32WideningToInt64) {
    const int32_t src[] = {1, -1, std::numeric_limits<int32_t>::max()};
    int64_t dst[3] = {};
    cast<int32_t, int64_t>(src, 3, dst);
    EXPECT_EQ(dst[0], 1);
    EXPECT_EQ(dst[1], -1);
    EXPECT_EQ(dst[2], (int64_t)std::numeric_limits<int32_t>::max());
}

TEST(BoltKernels, CastFloatToIntSaturates) {
    const double src[] = {1.0e20, -1.0e20, 3.5, -3.5, 0.0};
    int32_t dst[5] = {};
    cast<double, int32_t>(src, 5, dst);
    EXPECT_EQ(dst[0], std::numeric_limits<int32_t>::max());
    EXPECT_EQ(dst[1], std::numeric_limits<int32_t>::min());
    EXPECT_EQ(dst[2], 3);        // truncation toward zero
    EXPECT_EQ(dst[3], -3);
    EXPECT_EQ(dst[4], 0);
}

TEST(BoltKernels, CastFloat32ToFloat64) {
    const float src[] = {1.5f, -2.25f};
    double dst[2] = {};
    cast<float, double>(src, 2, dst);
    EXPECT_DOUBLE_EQ(dst[0], 1.5);
    EXPECT_DOUBLE_EQ(dst[1], -2.25);
}

// ---------------------------------------------------------------------------
// Micro-adaptive filter_gt
// ---------------------------------------------------------------------------

TEST(BoltKernels, AdaptiveAllMatchShortCircuit) {
    const int32_t data[] = {10, 20, 30, 40};
    int32_t out[4] = {};
    ColumnStats s = ColumnStats::make_default();
    s.min_value = 10;  // all values are > 5
    s.max_value = 40;
    int64_t n = filter_gt_adaptive<int32_t>(data, 4, 5, out, &s);
    EXPECT_EQ(n, 4);
    EXPECT_EQ(out[0], 0); EXPECT_EQ(out[1], 1);
    EXPECT_EQ(out[2], 2); EXPECT_EQ(out[3], 3);
}

TEST(BoltKernels, AdaptiveNoneMatchShortCircuit) {
    const int32_t data[] = {1, 2, 3, 4};
    int32_t out[4] = {};
    ColumnStats s = ColumnStats::make_default();
    s.min_value = 1;
    s.max_value = 4;  // max <= 100, so nothing > 100
    int64_t n = filter_gt_adaptive<int32_t>(data, 4, 100, out, &s);
    EXPECT_EQ(n, 0);
}

TEST(BoltKernels, AdaptiveFallsThroughWithoutStats) {
    const int32_t data[] = {1, 5, 3, 9};
    int32_t out[4] = {};
    int64_t n = filter_gt_adaptive<int32_t>(data, 4, 4, out, nullptr);
    EXPECT_EQ(n, 2);
    EXPECT_EQ(out[0], 1); EXPECT_EQ(out[1], 3);
}

TEST(BoltKernels, AdaptiveMidSelectivity) {
    int32_t data[100];
    for (int i = 0; i < 100; ++i) data[i] = i;
    int32_t out[100] = {};
    ColumnStats s = ColumnStats::make_default();
    s.min_value = 0;
    s.max_value = 99;
    // scalar=49 → ~50% selectivity → branchless path
    int64_t n = filter_gt_adaptive<int32_t>(data, 100, 49, out, &s);
    EXPECT_EQ(n, 50);
    EXPECT_EQ(out[0], 50);
    EXPECT_EQ(out[49], 99);
}

// ---------------------------------------------------------------------------
// Matrix instantiation check
// ---------------------------------------------------------------------------

TEST(BoltKernels, MatrixInstantiates) {
    // Calling touch_kernels forces every BOLT_NUMERIC_TYPES specialization.
    // If any kernel fails to compile for any T, the TU would not build.
    touch_kernels();
    SUCCEED();
}

// ---------------------------------------------------------------------------
// gather_to_column<int32_t> — round trip
// ---------------------------------------------------------------------------
TEST(BoltKernels, GatherToColumnInt32RoundTrip) {
    Arena arena;
    int32_t src_data[8] = {10, 11, 12, 13, 14, 15, 16, 17};
    BoltColumn src = BoltColumn::make_flat(src_data, nullptr, 8, BoltType::Int32);

    const int32_t sel[5] = {7, 0, 3, 3, 5};
    BoltColumn out = gather_to_column<int32_t>(src, sel, 5, &arena);

    ASSERT_EQ(out.length, 5);
    ASSERT_EQ(out.format, ColumnFormat::Flat);
    ASSERT_EQ(out.type, BoltType::Int32);
    ASSERT_NE(out.data, nullptr);
    const int32_t* got = static_cast<const int32_t*>(out.data);
    EXPECT_EQ(got[0], 17);
    EXPECT_EQ(got[1], 10);
    EXPECT_EQ(got[2], 13);
    EXPECT_EQ(got[3], 13);
    EXPECT_EQ(got[4], 15);
}

// ---------------------------------------------------------------------------
// gather_to_column<int32_t> — empty selection
// ---------------------------------------------------------------------------
TEST(BoltKernels, GatherToColumnInt32Empty) {
    Arena arena;
    int32_t src_data[4] = {1, 2, 3, 4};
    BoltColumn src = BoltColumn::make_flat(src_data, nullptr, 4, BoltType::Int32);

    BoltColumn out = gather_to_column<int32_t>(src, nullptr, 0, &arena);
    EXPECT_EQ(out.length, 0);
    EXPECT_EQ(out.type, BoltType::Int32);
}

// ---------------------------------------------------------------------------
// gather_to_column<int32_t> — sparse selection from larger source
// ---------------------------------------------------------------------------
TEST(BoltKernels, GatherToColumnInt32Sparse) {
    Arena arena;
    int32_t src_data[1000];
    for (int i = 0; i < 1000; ++i) src_data[i] = i * 3;
    BoltColumn src = BoltColumn::make_flat(src_data, nullptr, 1000, BoltType::Int32);

    int32_t sel[4] = {0, 999, 500, 123};
    BoltColumn out = gather_to_column<int32_t>(src, sel, 4, &arena);

    ASSERT_EQ(out.length, 4);
    const int32_t* got = static_cast<const int32_t*>(out.data);
    EXPECT_EQ(got[0], 0);
    EXPECT_EQ(got[1], 999 * 3);
    EXPECT_EQ(got[2], 500 * 3);
    EXPECT_EQ(got[3], 123 * 3);
}

// ===========================================================================
// Wave F1 — sum_masked / filter_sum_gt (fused filter+sum)
// ===========================================================================

TEST(BoltKernels, SumMaskedInt32Matches) {
    constexpr int64_t N = 4096;
    std::vector<int32_t> data(N);
    std::vector<int32_t> sel;
    sel.reserve(N);
    std::mt19937 rng(0xA11CE);
    std::uniform_int_distribution<int32_t> dv(-1000, 1000);
    for (int64_t i = 0; i < N; ++i) data[i] = dv(rng);
    // Random selection vector (each row included with ~50% probability,
    // stored in ascending order so the pattern mirrors a real filter output).
    std::bernoulli_distribution dp(0.5);
    for (int64_t i = 0; i < N; ++i) if (dp(rng)) sel.push_back(static_cast<int32_t>(i));

    int64_t ref = 0;
    for (int32_t idx : sel) ref += data[idx];
    int64_t got = sum_masked<int32_t>(data.data(), sel.data(),
                                      static_cast<int64_t>(sel.size()));
    EXPECT_EQ(got, ref);
}

TEST(BoltKernels, SumMaskedInt64Matches) {
    constexpr int64_t N = 4096;
    std::vector<int64_t> data(N);
    std::vector<int32_t> sel;
    sel.reserve(N);
    std::mt19937 rng(0xBEEF);
    std::uniform_int_distribution<int64_t> dv(-1'000'000, 1'000'000);
    for (int64_t i = 0; i < N; ++i) data[i] = dv(rng);
    std::bernoulli_distribution dp(0.3);
    for (int64_t i = 0; i < N; ++i) if (dp(rng)) sel.push_back(static_cast<int32_t>(i));

    int64_t ref = 0;
    for (int32_t idx : sel) ref += data[idx];
    int64_t got = sum_masked<int64_t>(data.data(), sel.data(),
                                      static_cast<int64_t>(sel.size()));
    EXPECT_EQ(got, ref);
}

TEST(BoltKernels, SumMaskedEmpty) {
    int32_t data[4] = {1, 2, 3, 4};
    int32_t sel[1]  = {0};
    EXPECT_EQ(sum_masked<int32_t>(data, sel, 0), 0);
    EXPECT_EQ(sum_masked<int32_t>(nullptr, nullptr, 0), 0);
    int64_t data64[4] = {1, 2, 3, 4};
    EXPECT_EQ(sum_masked<int64_t>(data64, sel, 0), 0);
}

TEST(BoltKernels, FilterSumGtInt32Matches) {
    constexpr int64_t N = 4096;
    std::vector<int32_t> data(N);
    std::vector<int32_t> sel(N);
    std::mt19937 rng(0xC0DE);
    std::uniform_int_distribution<int32_t> dv(-500, 500);
    for (int64_t i = 0; i < N; ++i) data[i] = dv(rng);

    const int32_t scalar = 42;
    int64_t sel_n = filter_gt<int32_t>(data.data(), N, scalar, sel.data());
    int64_t two_pass = sum_masked<int32_t>(data.data(), sel.data(), sel_n);
    int64_t fused    = filter_sum_gt<int32_t>(data.data(), N, scalar);
    EXPECT_EQ(fused, two_pass);
}

TEST(BoltKernels, FilterSumGtInt64Matches) {
    constexpr int64_t N = 3000;
    std::vector<int64_t> data(N);
    std::vector<int32_t> sel(N);
    std::mt19937 rng(0xDADA);
    std::uniform_int_distribution<int64_t> dv(-100'000, 100'000);
    for (int64_t i = 0; i < N; ++i) data[i] = dv(rng);

    const int64_t scalar = -17;
    int64_t sel_n = filter_gt<int64_t>(data.data(), N, scalar, sel.data());
    int64_t two_pass = sum_masked<int64_t>(data.data(), sel.data(), sel_n);
    int64_t fused    = filter_sum_gt<int64_t>(data.data(), N, scalar);
    EXPECT_EQ(fused, two_pass);
}

TEST(BoltKernels, FilterSumGtBoundary) {
    // INT_MAX / INT_MIN scalars with threshold = MAX-1 / MIN+1.
    // The all-ones mask path must not trip signed overflow UB.
    constexpr int32_t MAX = std::numeric_limits<int32_t>::max();
    constexpr int32_t MIN = std::numeric_limits<int32_t>::min();
    const int32_t data[] = {MAX, MAX - 1, 0, MIN + 1, MIN};

    // threshold MAX-1 → only MAX qualifies.
    int64_t s1 = filter_sum_gt<int32_t>(data, 5, MAX - 1);
    EXPECT_EQ(s1, static_cast<int64_t>(MAX));

    // threshold MIN → MAX, MAX-1, 0, MIN+1 qualify.
    int64_t s2 = filter_sum_gt<int32_t>(data, 5, MIN);
    int64_t expect2 = static_cast<int64_t>(MAX) + (MAX - 1) + 0 + (MIN + 1);
    EXPECT_EQ(s2, expect2);

    // threshold MAX → nothing qualifies.
    int64_t s3 = filter_sum_gt<int32_t>(data, 5, MAX);
    EXPECT_EQ(s3, 0);
}

TEST(BoltKernels, FilterSumGtAllPass) {
    const int32_t data[] = {10, 20, 30, 40, 50};
    int64_t s = filter_sum_gt<int32_t>(data, 5, 0);
    EXPECT_EQ(s, 150);
}

TEST(BoltKernels, FilterSumGtNonePass) {
    const int32_t data[] = {1, 2, 3, 4, 5};
    int64_t s = filter_sum_gt<int32_t>(data, 5, 1000);
    EXPECT_EQ(s, 0);

    // Degenerate n == 0.
    EXPECT_EQ(filter_sum_gt<int32_t>(nullptr, 0, 42), 0);
}

TEST(BoltKernels, FilterSumGtDouble) {
    const double data[] = {1.5, -2.0, 3.25, 0.0, 4.75};
    double s = filter_sum_gt<double>(data, 5, 0.0);
    EXPECT_DOUBLE_EQ(s, 1.5 + 3.25 + 4.75);
}

// In-place mini-bench: filter_sum_gt vs filter_gt + sum_masked two-pass.
// Not a correctness assert — logs throughput so reviewers can eyeball the
// fused-kernel win. Size chosen to blow past L1 so cache pressure shows.
TEST(BoltKernels, FilterSumGtMiniBench) {
    constexpr int64_t N = 1 << 20;  // 4 MiB of int32 = 1M rows
    std::vector<int32_t> data(N);
    std::vector<int32_t> sel(N);
    std::mt19937 rng(0x1234);
    std::uniform_int_distribution<int32_t> dv(0, 1000);
    for (int64_t i = 0; i < N; ++i) data[i] = dv(rng);
    const int32_t scalar = 500;  // ~50% selectivity

    using clk = std::chrono::steady_clock;
    constexpr int kReps = 20;

    // Two-pass: filter_gt then sum_masked (materializes selection vector).
    int64_t two_pass_sum = 0;
    auto t0 = clk::now();
    for (int r = 0; r < kReps; ++r) {
        int64_t sel_n = filter_gt<int32_t>(data.data(), N, scalar, sel.data());
        two_pass_sum += sum_masked<int32_t>(data.data(), sel.data(), sel_n);
    }
    auto t1 = clk::now();

    // Fused: no selection vector.
    int64_t fused_sum = 0;
    auto t2 = clk::now();
    for (int r = 0; r < kReps; ++r) {
        fused_sum += filter_sum_gt<int32_t>(data.data(), N, scalar);
    }
    auto t3 = clk::now();

    EXPECT_EQ(fused_sum, two_pass_sum);

    auto ns_two = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    auto ns_fus = std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
    double rows = static_cast<double>(N) * kReps;
    std::printf("[mini-bench] two-pass: %.2f M rows/s  fused: %.2f M rows/s  speedup=%.2fx\n",
                rows / (ns_two / 1e3), rows / (ns_fus / 1e3),
                static_cast<double>(ns_two) / static_cast<double>(ns_fus));
}
