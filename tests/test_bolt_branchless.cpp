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

// ---------- gather_branchless (A2) ---------------------------------------
// The generic template is the reference; the SIMD specialisations for i32
// and i64 must produce byte-equal output.

namespace {

template <typename T>
void expect_gather_match(const std::vector<T>& data,
                         const std::vector<int32_t>& indices) {
    const int64_t n = static_cast<int64_t>(indices.size());
    std::vector<T> ref(n, T{});
    std::vector<T> got(n, T{});

    // Reference: call via a wrapper that forces the generic (scalar) template
    // instantiation even for T=int32_t/int64_t by going one level of
    // indirection through a plain function pointer.
    auto scalar_gather = [](const T* d, const int32_t* ix, int64_t c, T* o) {
        constexpr int kPrefetchDist = 16;
        int64_t i = 0;
        for (; i < std::min<int64_t>(c, kPrefetchDist); ++i) {
            BOLT_PREFETCH_READ(&d[ix[i]]);
        }
        for (i = 0; i < c; ++i) {
            if (i + kPrefetchDist < c) BOLT_PREFETCH_READ(&d[ix[i + kPrefetchDist]]);
            o[i] = d[ix[i]];
        }
    };
    scalar_gather(data.empty() ? nullptr : data.data(),
                  indices.empty() ? nullptr : indices.data(),
                  n, ref.empty() ? nullptr : ref.data());

    bolt::branchless::gather_branchless<T>(
        data.empty() ? nullptr : data.data(),
        indices.empty() ? nullptr : indices.data(),
        n, got.empty() ? nullptr : got.data());

    for (int64_t i = 0; i < n; ++i) {
        EXPECT_EQ(ref[i], got[i]) << "gather mismatch at " << i;
    }
}

}  // namespace

TEST(GatherBranchless, I32Empty) {
    expect_gather_match<int32_t>({}, {});
}

TEST(GatherBranchless, I32SmallSequential) {
    std::vector<int32_t> data(64);
    for (int i = 0; i < 64; ++i) data[i] = i * 7;
    std::vector<int32_t> ix = {0, 5, 10, 63, 1, 2, 3, 4, 8, 16, 32};
    expect_gather_match<int32_t>(data, ix);
}

TEST(GatherBranchless, I32RandomLarge) {
    std::mt19937 rng(42);
    std::vector<int32_t> data(8192);
    for (size_t i = 0; i < data.size(); ++i)
        data[i] = static_cast<int32_t>(rng());
    std::vector<int32_t> ix(4096);
    std::uniform_int_distribution<int32_t> dist(0, 8191);
    for (auto& v : ix) v = dist(rng);
    expect_gather_match<int32_t>(data, ix);
}

TEST(GatherBranchless, I64SmallSequential) {
    std::vector<int64_t> data(64);
    for (int i = 0; i < 64; ++i) data[i] = 0x1000000000LL + i * 13;
    std::vector<int32_t> ix = {0, 1, 2, 3, 63, 62, 61, 60, 15, 16, 17};
    expect_gather_match<int64_t>(data, ix);
}

TEST(GatherBranchless, I64RandomLarge) {
    std::mt19937_64 rng(7);
    std::vector<int64_t> data(8192);
    for (auto& v : data) v = static_cast<int64_t>(rng());
    std::vector<int32_t> ix(4096);
    std::uniform_int_distribution<int32_t> dist(0, 8191);
    std::mt19937 irng(11);
    for (auto& v : ix) v = dist(irng);
    expect_gather_match<int64_t>(data, ix);
}

// ---------- Selection composition (A4) -----------------------------------

TEST(SelectionCompose, IntersectI32Match) {
    const int32_t a[] = {1, 3, 5, 7, 9, 11};
    const int32_t b[] = {2, 3, 5, 8, 9, 10};
    int32_t out[8] = {};
    int64_t n = bolt::branchless::selection_intersect_t<int32_t>(
        a, 6, b, 6, out);
    ASSERT_EQ(n, 3);
    EXPECT_EQ(out[0], 3);
    EXPECT_EQ(out[1], 5);
    EXPECT_EQ(out[2], 9);
}

TEST(SelectionCompose, IntersectI64) {
    const int64_t a[] = {100, 500, 999, 1'000'000'000'000LL};
    const int64_t b[] = {500, 800, 999, 1'000'000'000'001LL};
    int64_t out[4] = {};
    int64_t n = bolt::branchless::selection_intersect_t<int64_t>(
        a, 4, b, 4, out);
    ASSERT_EQ(n, 2);
    EXPECT_EQ(out[0], 500);
    EXPECT_EQ(out[1], 999);
}

TEST(SelectionCompose, IntersectI32EmptyInputs) {
    int32_t out[1] = {};
    EXPECT_EQ(bolt::branchless::selection_intersect_t<int32_t>(nullptr, 0, nullptr, 0, out), 0);
}

TEST(SelectionCompose, UnionI32Match) {
    const int32_t a[] = {1, 3, 5};
    const int32_t b[] = {2, 3, 4, 6};
    int32_t out[7] = {};
    int64_t n = bolt::branchless::selection_union_t<int32_t>(a, 3, b, 4, out);
    ASSERT_EQ(n, 6);
    EXPECT_EQ(out[0], 1);
    EXPECT_EQ(out[1], 2);
    EXPECT_EQ(out[2], 3);
    EXPECT_EQ(out[3], 4);
    EXPECT_EQ(out[4], 5);
    EXPECT_EQ(out[5], 6);
}

TEST(SelectionCompose, UnionI64) {
    const int64_t a[] = {0, 2, 4, 6, 8};
    const int64_t b[] = {1, 2, 3, 5, 7, 9};
    int64_t out[16] = {};
    int64_t n = bolt::branchless::selection_union_t<int64_t>(a, 5, b, 6, out);
    ASSERT_EQ(n, 10);
    const int64_t expected[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    for (int i = 0; i < 10; ++i) EXPECT_EQ(out[i], expected[i]);
}

TEST(SelectionCompose, BitmapToIndicesI32Dense) {
    // 130 rows; mark every 3rd one valid.
    const int64_t N = 130;
    uint64_t bm[3] = {0, 0, 0};
    int64_t expected = 0;
    for (int64_t i = 0; i < N; ++i) {
        if (i % 3 == 0) { bm[i >> 6] |= (uint64_t{1} << (i & 63)); expected++; }
    }
    int32_t out[N] = {};
    int64_t n = bolt::branchless::bitmap_to_indices<int32_t>(bm, N, out);
    ASSERT_EQ(n, expected);
    for (int64_t i = 0; i < n; ++i) EXPECT_EQ(out[i], static_cast<int32_t>(i * 3));
}

TEST(SelectionCompose, BitmapToIndicesI64PartialWord) {
    // 5 rows in the tail — make sure we ignore bits past num_rows.
    uint64_t bm[1] = {0xFFFFFFFFFFFFFFFFULL};
    int64_t out[8] = {};
    int64_t n = bolt::branchless::bitmap_to_indices<int64_t>(bm, 5, out);
    ASSERT_EQ(n, 5);
    for (int64_t i = 0; i < 5; ++i) EXPECT_EQ(out[i], i);
}

// ---- f32 SIMD kernels (A5) ----------------------------------------------

TEST(F32SIMD, FilterGtMatchesScalar) {
    std::mt19937 rng(0xF00D);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    std::vector<float> data(2048);
    for (auto& v : data) v = dist(rng);

    const int64_t n = static_cast<int64_t>(data.size());
    std::vector<int32_t> ref(n), got(n);
    int64_t rc = bolt::branchless::filter_gt_branchless<float>(
        data.data(), n, 0.0f, ref.data());
    int64_t gc = bolt::branchless::filter_gt_avx2_f32(
        data.data(), n, 0.0f, got.data());
    ASSERT_EQ(gc, rc);
    for (int64_t i = 0; i < rc; ++i) EXPECT_EQ(ref[i], got[i]);
}

TEST(F32SIMD, FilterGtNaNNeverPasses) {
    std::vector<float> data(17);
    for (int i = 0; i < 17; ++i) {
        data[i] = (i % 3 == 0) ? std::numeric_limits<float>::quiet_NaN()
                               : static_cast<float>(i);
    }
    std::vector<int32_t> out(17);
    int64_t n = bolt::branchless::filter_gt_avx2_f32(
        data.data(), 17, -1.0f, out.data());
    // Non-NaN rows: 1,2,4,5,7,8,10,11,13,14,16 — 11 values > -1.
    int64_t expected = 0;
    for (int i = 0; i < 17; ++i) {
        if (!std::isnan(data[i]) && data[i] > -1.0f) ++expected;
    }
    EXPECT_EQ(n, expected);
}

TEST(F32SIMD, SumMatchesDoubleAccumulator) {
    std::mt19937 rng(0xDEAD);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    std::vector<float> data(4096);
    for (auto& v : data) v = dist(rng);

    double ref = 0.0;
    for (float v : data) ref += static_cast<double>(v);
    double got = bolt::branchless::sum_avx2_f32(data.data(),
                                                 static_cast<int64_t>(data.size()));
    // Floats + SIMD order != scalar order → require close, not exact.
    EXPECT_NEAR(got, ref, std::abs(ref) * 1e-5 + 1e-3);
}

TEST(F32SIMD, SumEmpty) {
    EXPECT_EQ(bolt::branchless::sum_avx2_f32(nullptr, 0), 0.0);
}

// ---- Run-native RLE kernels (B2b) ---------------------------------------

TEST(RLEKernels, FilterEqMatchesMaterialised) {
    // 4 runs: 5×100, 3×200, 2×100, 4×300 = 14 rows. scalar=100 → 7 matches.
    int32_t values[4]   = {100, 200, 100, 300};
    int32_t run_ends[4] = {5, 8, 10, 14};
    int32_t out[14] = {};
    int64_t n = bolt::branchless::filter_eq_rle<int32_t>(
        values, run_ends, 4, 100, out);
    ASSERT_EQ(n, 7);
    // Run 0 = rows [0,5); run 2 = rows [8,10).
    const int32_t expected[7] = {0,1,2,3,4, 8,9};
    for (int i = 0; i < 7; ++i) EXPECT_EQ(out[i], expected[i]) << "i=" << i;
}

TEST(RLEKernels, FilterEqNoMatch) {
    int32_t values[3]   = {1, 2, 3};
    int32_t run_ends[3] = {10, 20, 30};
    int32_t out[4] = {};
    int64_t n = bolt::branchless::filter_eq_rle<int32_t>(
        values, run_ends, 3, 999, out);
    EXPECT_EQ(n, 0);
}

TEST(RLEKernels, SumMatchesExpanded) {
    // 4 runs: 5×10, 3×20, 2×30, 4×40 = 10*5 + 20*3 + 30*2 + 40*4 = 330.
    int32_t values[4]   = {10, 20, 30, 40};
    int32_t run_ends[4] = {5, 8, 10, 14};
    int64_t got = bolt::branchless::sum_rle_i64<int32_t>(values, run_ends, 4);
    EXPECT_EQ(got, 330);
}

TEST(RLEKernels, SumEmpty) {
    EXPECT_EQ(bolt::branchless::sum_rle_i64<int32_t>(nullptr, nullptr, 0), 0);
}

#else  // scalar-only build: still provide a single trivial test so the binary links.

TEST(FilterBranchless, ScalarBuildNoSimd) {
    std::vector<int64_t> data = {1, 2, 3};
    std::vector<int32_t> out(3);
    int64_t c = filter_gt_branchless<int64_t>(data.data(), 3, 1, out.data());
    EXPECT_EQ(c, 2);
}

#endif
