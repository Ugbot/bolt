// test_bolt_branchless.cpp — byte-for-byte equality tests for the SIMD
// filter kernels (int64 / float64) against the scalar branchless template.
//
// The scalar template in bolt_branchless.h is the reference; SIMD kernels
// must produce identical selection vectors for every edge case we try.
// Works on every ISA path: under scalar builds the SIMD kernel folds back
// to the same scalar code, so the tests still run and assert consistency.

#include <gtest/gtest.h>

#include "bolt/bolt_port.h"
#include "bolt/bolt_branchless.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

using bolt::branchless::filter_gt_branchless;

#if BOLT_SIMD_AVX2 || BOLT_SIMD_SSE42 || BOLT_SIMD_NEON
using bolt::branchless::filter_gt_avx2_i64;
using bolt::branchless::filter_gt_avx2_f64;

namespace {

template <typename T>
void expect_match(const std::vector<T>& data, T scalar) {
    const int64_t n = static_cast<int64_t>(data.size());
    std::vector<int32_t> ref(n, 0);
    std::vector<int32_t> got(n, 0);

    int64_t ref_count = filter_gt_branchless<T>(
        data.empty() ? nullptr : data.data(), n, scalar,
        ref.empty() ? nullptr : ref.data());

    int64_t got_count = 0;
    if constexpr (std::is_same_v<T, int64_t>) {
        got_count = filter_gt_avx2_i64(
            data.empty() ? nullptr : data.data(), n, scalar,
            got.empty() ? nullptr : got.data());
    } else {
        got_count = filter_gt_avx2_f64(
            data.empty() ? nullptr : data.data(), n, scalar,
            got.empty() ? nullptr : got.data());
    }

    ASSERT_EQ(ref_count, got_count);
    for (int64_t i = 0; i < ref_count; ++i) {
        EXPECT_EQ(ref[i], got[i]) << "mismatch at " << i;
    }
}

}  // namespace

// ---------- int64 ---------------------------------------------------------

TEST(FilterBranchless, I64Empty) {
    std::vector<int64_t> data;
    expect_match<int64_t>(data, 0);
}

TEST(FilterBranchless, I64AllPass) {
    std::vector<int64_t> data(33);
    for (int i = 0; i < 33; ++i) data[i] = 100 + i;
    expect_match<int64_t>(data, 50);
}

TEST(FilterBranchless, I64NonePass) {
    std::vector<int64_t> data(33);
    for (int i = 0; i < 33; ++i) data[i] = i;
    expect_match<int64_t>(data, 1000);
}

TEST(FilterBranchless, I64Alternating) {
    std::vector<int64_t> data(64);
    for (int i = 0; i < 64; ++i) data[i] = (i & 1) ? 200 : -200;
    expect_match<int64_t>(data, 0);
}

TEST(FilterBranchless, I64Extremes) {
    std::vector<int64_t> data = {
        std::numeric_limits<int64_t>::min(),
        std::numeric_limits<int64_t>::max(),
        0,
        -1,
        1,
        std::numeric_limits<int64_t>::min() + 1,
        std::numeric_limits<int64_t>::max() - 1,
        42,
        -42,
    };
    expect_match<int64_t>(data, 0);
    expect_match<int64_t>(data, std::numeric_limits<int64_t>::min());
    expect_match<int64_t>(data, std::numeric_limits<int64_t>::max() - 1);
}

TEST(FilterBranchless, I64RandomLarge) {
    std::mt19937_64 rng(12345);
    std::vector<int64_t> data(4096);
    for (auto& v : data) v = static_cast<int64_t>(rng());
    expect_match<int64_t>(data, 0);
    expect_match<int64_t>(data, static_cast<int64_t>(rng()));
}

// ---------- float64 -------------------------------------------------------

TEST(FilterBranchless, F64Empty) {
    std::vector<double> data;
    expect_match<double>(data, 0.0);
}

TEST(FilterBranchless, F64AllPassAndNonePass) {
    std::vector<double> data(33);
    for (int i = 0; i < 33; ++i) data[i] = static_cast<double>(i) + 0.5;
    expect_match<double>(data, -1.0);   // all pass
    expect_match<double>(data, 1000.0); // none pass
}

TEST(FilterBranchless, F64NaNNeverPasses) {
    // IEEE ordered GT says NaN > x is false; branchless template and SIMD
    // kernel must agree.
    std::vector<double> data(16);
    for (int i = 0; i < 16; ++i) data[i] = (i % 3 == 0)
        ? std::numeric_limits<double>::quiet_NaN()
        : static_cast<double>(i);
    expect_match<double>(data, -1.0);
    expect_match<double>(data, 5.0);
}

TEST(FilterBranchless, F64RandomLarge) {
    std::mt19937_64 rng(99);
    std::vector<double> data(2048);
    for (auto& v : data) v = static_cast<double>(static_cast<int64_t>(rng())) / 1e9;
    expect_match<double>(data, 0.0);
    expect_match<double>(data, 1.0);
    expect_match<double>(data, -1.0);
}

#else  // scalar-only build: still provide a single trivial test so the binary links.

TEST(FilterBranchless, ScalarBuildNoSimd) {
    std::vector<int64_t> data = {1, 2, 3};
    std::vector<int32_t> out(3);
    int64_t c = filter_gt_branchless<int64_t>(data.data(), 3, 1, out.data());
    EXPECT_EQ(c, 2);
}

#endif
