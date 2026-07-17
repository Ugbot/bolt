// test_bolt_idot.cpp -- bolt::compute's int8-activation IDOT GEMV (BLLM-14).
// Consolidates boltllm's test_idot_kernels.cpp + test_idot_avx512vnni.cpp
// into one gtest TU (they now live in bolt). Verifies:
//   1. Every AVAILABLE SIMD tier (AVX2, AVX-512VNNI) agrees BIT-FOR-BIT with
//      the scalar reference -- the kernels compute an EXACT int64 sum then
//      the same final float expression, so SIMD-vs-scalar is == not ~=.
//   2. The dispatch entry (matmul_i8a_*) equals whichever tier the CPUID
//      probe selected.
//   3. Every tier is within a DERIVED tolerance of a naive dequant-then-dot
//      float reference (two independent quantization errors + different
//      summation order -- see dot_error_bound()).
//   4. Determinism (same inputs -> bit-identical output twice).
//
// CI-portable: each SIMD tier's checks are gated on #if BOLT_ARCH_X86 AND its
// runtime availability probe, so this is green on windows-msvc/linux-gcc/
// linux-clang (fast tiers exercised where present) AND macos-arm (scalar-only,
// SIMD comparisons GTEST_SKIP-equivalent via the guards).

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "bolt/bolt_port.h"
#include "bolt/bolt_tensor.h"
#include "bolt/compute_idot.h"

#include "../src/compute/idot_kernels_internal.h"

using bolt::DType;
using bolt::Tensor;
using bolt::tensor_make_cpu_2d;
namespace bc = bolt::compute;

namespace {

// int8 weight-row quantizer == the activation quantizer's math (per-row
// max-abs scale, round-half-even, clamp [-128,127]).
void quantize_int8_rows(const float* w, int32_t rows, int32_t cols, int8_t* q_out,
                        float* scale_out) {
    for (int32_t r = 0; r < rows; ++r) {
        bc::quantize_act_row_int8(w + static_cast<size_t>(r) * cols, cols,
                                  q_out + static_cast<size_t>(r) * cols, &scale_out[r]);
    }
}

// int4 weight-row quantizer (scale = max|w|/7, round-half-even, clamp [-8,7],
// pack low-nibble-first with offset +8). Matches bolt::compute::detail::
// kInt4DequantOffset's convention.
void quantize_int4_rows(const float* w, int32_t rows, int32_t cols, uint8_t* packed_out,
                        float* scale_out) {
    const size_t stride = static_cast<size_t>((cols + 1) / 2);
    for (int32_t r = 0; r < rows; ++r) {
        const float* row = w + static_cast<size_t>(r) * cols;
        float amax = 0.0f;
        for (int32_t i = 0; i < cols; ++i) amax = std::max(amax, std::fabs(row[i]));
        float s = amax / 7.0f;
        if (s < 1e-8f) s = 1e-8f;
        scale_out[r] = s;
        uint8_t* prow = packed_out + static_cast<size_t>(r) * stride;
        auto q = [&](float x) -> int32_t {
            const float f = std::floor(x / s);
            const float d = (x / s) - f;
            int32_t v = static_cast<int32_t>(f);
            if (d > 0.5f) v += 1;
            else if (d == 0.5f) v = (v % 2 == 0) ? v : v + 1;
            if (v < -8) v = -8;
            if (v > 7) v = 7;
            return v;
        };
        for (int32_t i = 0; i < cols; i += 2) {
            const int32_t v0 = q(row[i]) + 8;
            const int32_t v1 = (i + 1 < cols) ? (q(row[i + 1]) + 8) : 8;
            prow[i >> 1] = static_cast<uint8_t>(v0 | (v1 << 4));
        }
    }
}

float dot_error_bound(int32_t cols, float act_scale, float wscale, int32_t qmax_w) {
    const float amax_a = 127.0f * act_scale;
    const float amax_w = static_cast<float>(qmax_w) * wscale;
    const float per_elem =
        amax_a * 0.5f * wscale + amax_w * 0.5f * act_scale + 0.25f * act_scale * wscale;
    return static_cast<float>(cols) * per_elem + 1e-6f;
}

// Runs one int8 case through all AVAILABLE tiers + the dispatch entry, and
// asserts (1) exact cross-tier agreement, (2) dispatch == selected tier,
// (3) tolerance vs a dequant-dot reference.
void run_int8_case(std::mt19937& rng, int32_t rows, int32_t cols) {
    std::uniform_real_distribution<float> vdist(-4.0f, 4.0f);
    std::vector<float> w(static_cast<size_t>(rows) * cols);
    for (auto& x : w) x = vdist(rng);
    std::vector<int8_t> wq(w.size());
    std::vector<float> wscale(static_cast<size_t>(rows));
    quantize_int8_rows(w.data(), rows, cols, wq.data(), wscale.data());

    std::vector<float> act(static_cast<size_t>(cols));
    for (auto& x : act) x = vdist(rng);
    std::vector<int8_t> actq(static_cast<size_t>(cols));
    float act_scale = 0.0f;
    bc::quantize_act_row_int8(act.data(), cols, actq.data(), &act_scale);

    Tensor t = tensor_make_cpu_2d(DType::Int8, rows, cols, wq.data(), wscale.data());

    std::vector<float> out_scalar(static_cast<size_t>(rows));
    std::vector<float> out_dispatch(static_cast<size_t>(rows));
    bc::detail::idot_i8a_int8_scalar(actq.data(), act_scale, t, out_scalar.data(), rows);
    bc::matmul_i8a_int8(actq.data(), act_scale, t, out_dispatch.data(), rows);

#if BOLT_ARCH_X86
    std::vector<float> out_avx2(static_cast<size_t>(rows));
    std::vector<float> out_vnni(static_cast<size_t>(rows));
    const bool have_avx2 = bc::idot_avx2_available();
    const bool have_vnni = bc::idot_avx512vnni_available();
    if (have_avx2) bc::detail::idot_i8a_int8_avx2(actq.data(), act_scale, t, out_avx2.data(), rows);
    if (have_vnni) bc::detail::idot_i8a_int8_avx512vnni(actq.data(), act_scale, t, out_vnni.data(), rows);
#endif

    for (int32_t r = 0; r < rows; ++r) {
#if BOLT_ARCH_X86
        if (have_avx2) EXPECT_EQ(out_avx2[r], out_scalar[r]) << "avx2 vs scalar row " << r;
        if (have_vnni) EXPECT_EQ(out_vnni[r], out_scalar[r]) << "vnni vs scalar row " << r;
        const float selected = have_vnni ? out_vnni[r] : (have_avx2 ? out_avx2[r] : out_scalar[r]);
#else
        const float selected = out_scalar[r];
#endif
        EXPECT_EQ(out_dispatch[r], selected) << "dispatch vs selected tier row " << r;

        // Tolerance vs dequant-dot reference.
        double ref = 0.0;
        for (int32_t i = 0; i < cols; ++i) {
            const float wd = static_cast<float>(wq[static_cast<size_t>(r) * cols + i]) * wscale[r];
            ref += static_cast<double>(actq[i]) * act_scale * wd;
        }
        EXPECT_LE(std::fabs(out_dispatch[r] - static_cast<float>(ref)),
                  dot_error_bound(cols, act_scale, wscale[r], 127));
    }
}

void run_int4_case(std::mt19937& rng, int32_t rows, int32_t cols) {
    std::uniform_real_distribution<float> vdist(-4.0f, 4.0f);
    std::vector<float> w(static_cast<size_t>(rows) * cols);
    for (auto& x : w) x = vdist(rng);
    const size_t stride = static_cast<size_t>((cols + 1) / 2);
    std::vector<uint8_t> wq(stride * static_cast<size_t>(rows));
    std::vector<float> wscale(static_cast<size_t>(rows));
    quantize_int4_rows(w.data(), rows, cols, wq.data(), wscale.data());

    std::vector<float> act(static_cast<size_t>(cols));
    for (auto& x : act) x = vdist(rng);
    std::vector<int8_t> actq(static_cast<size_t>(cols));
    float act_scale = 0.0f;
    bc::quantize_act_row_int8(act.data(), cols, actq.data(), &act_scale);

    Tensor t = tensor_make_cpu_2d(DType::Int4, rows, cols, wq.data(), wscale.data());

    std::vector<float> out_scalar(static_cast<size_t>(rows));
    std::vector<float> out_dispatch(static_cast<size_t>(rows));
    bc::detail::idot_i8a_int4_scalar(actq.data(), act_scale, t, out_scalar.data(), rows);
    bc::matmul_i8a_int4(actq.data(), act_scale, t, out_dispatch.data(), rows);

#if BOLT_ARCH_X86
    std::vector<float> out_avx2(static_cast<size_t>(rows));
    std::vector<float> out_vnni(static_cast<size_t>(rows));
    const bool have_avx2 = bc::idot_avx2_available();
    const bool have_vnni = bc::idot_avx512vnni_available();
    if (have_avx2) bc::detail::idot_i8a_int4_avx2(actq.data(), act_scale, t, out_avx2.data(), rows);
    if (have_vnni) bc::detail::idot_i8a_int4_avx512vnni(actq.data(), act_scale, t, out_vnni.data(), rows);
#endif

    for (int32_t r = 0; r < rows; ++r) {
#if BOLT_ARCH_X86
        if (have_avx2) EXPECT_EQ(out_avx2[r], out_scalar[r]) << "avx2 vs scalar row " << r;
        if (have_vnni) EXPECT_EQ(out_vnni[r], out_scalar[r]) << "vnni vs scalar row " << r;
        const float selected = have_vnni ? out_vnni[r] : (have_avx2 ? out_avx2[r] : out_scalar[r]);
#else
        const float selected = out_scalar[r];
#endif
        EXPECT_EQ(out_dispatch[r], selected) << "dispatch vs selected tier row " << r;

        double ref = 0.0;
        const uint8_t* prow = wq.data() + static_cast<size_t>(r) * stride;
        for (int32_t i = 0; i < cols; ++i) {
            const uint8_t byte = prow[i >> 1];
            const int32_t nib = (i & 1) ? (byte >> 4) : (byte & 0x0F);
            const float wd = static_cast<float>(nib - 8) * wscale[r];
            ref += static_cast<double>(actq[i]) * act_scale * wd;
        }
        EXPECT_LE(std::fabs(out_dispatch[r] - static_cast<float>(ref)),
                  dot_error_bound(cols, act_scale, wscale[r], 7));
    }
}

}  // namespace

TEST(BoltIdot, Int8BoundarySizes) {
    std::mt19937 rng(42);
    for (int32_t cols : {1, 2, 3, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257}) {
        run_int8_case(rng, 3, cols);
    }
}

TEST(BoltIdot, Int4BoundarySizes) {
    std::mt19937 rng(43);
    for (int32_t cols : {1, 2, 3, 63, 64, 65, 127, 128, 129, 255, 256, 257, 513}) {
        run_int4_case(rng, 3, cols);
    }
}

TEST(BoltIdot, Int8RandomShapes) {
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int32_t> rdist(1, 8), cdist(1, 300);
    for (int trial = 0; trial < 120; ++trial) run_int8_case(rng, rdist(rng), cdist(rng));
}

TEST(BoltIdot, Int4RandomShapes) {
    std::mt19937 rng(54321);
    std::uniform_int_distribution<int32_t> rdist(1, 8), cdist(1, 300);
    for (int trial = 0; trial < 120; ++trial) run_int4_case(rng, rdist(rng), cdist(rng));
}

TEST(BoltIdot, Determinism) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> vdist(-3.0f, 3.0f);
    const int32_t rows = 5, cols = 191;  // deliberately not a multiple of 32/64/128
    std::vector<float> w(static_cast<size_t>(rows) * cols);
    for (auto& x : w) x = vdist(rng);
    std::vector<int8_t> wq(w.size());
    std::vector<float> wscale(static_cast<size_t>(rows));
    quantize_int8_rows(w.data(), rows, cols, wq.data(), wscale.data());
    std::vector<float> act(static_cast<size_t>(cols));
    for (auto& x : act) x = vdist(rng);
    std::vector<int8_t> actq(static_cast<size_t>(cols));
    float act_scale = 0.0f;
    bc::quantize_act_row_int8(act.data(), cols, actq.data(), &act_scale);
    Tensor t = tensor_make_cpu_2d(DType::Int8, rows, cols, wq.data(), wscale.data());

    std::vector<float> a(static_cast<size_t>(rows)), b(static_cast<size_t>(rows));
    bc::matmul_i8a_int8(actq.data(), act_scale, t, a.data(), rows);
    bc::matmul_i8a_int8(actq.data(), act_scale, t, b.data(), rows);
    for (int32_t r = 0; r < rows; ++r) EXPECT_EQ(a[r], b[r]);
}

TEST(BoltIdot, ProbesAreCallableAndConsistent) {
    // The probes must be callable on any arch and stable across calls.
    const bool avx2a = bc::idot_avx2_available();
    const bool vnnia = bc::idot_avx512vnni_available();
    EXPECT_EQ(avx2a, bc::idot_avx2_available());
    EXPECT_EQ(vnnia, bc::idot_avx512vnni_available());
#if !BOLT_ARCH_X86
    EXPECT_FALSE(avx2a);
    EXPECT_FALSE(vnnia);
#endif
    // VNNI implies AVX2 (a VNNI-capable CPU is always AVX2-capable).
    if (vnnia) EXPECT_TRUE(avx2a);
}
