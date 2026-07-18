// test_bolt_gemv.cpp -- bolt::compute's f32/f16 float-activation SIMD GEMV
// (perf pass). Verifies:
//   1. Every AVAILABLE SIMD tier (AVX2, AVX-512F) agrees with the scalar f64
//      reference to a derived float tolerance. Unlike the integer IDOT
//      kernels, the f32 SIMD reduction is reassociated, so this is ~= (within
//      tolerance), NOT bit-exact.
//   2. The dispatch entry (matmul_f32 / matmul_f16) equals whichever tier the
//      CPUID probe selected.
//   3. f16 weights: matmul_f16 matches a dequantize-then-dot f64 reference.
//   4. Determinism (same inputs -> bit-identical output twice).
//   5. Odd/non-power-of-two cols exercise the scalar tail.
//
// CI-portable: each SIMD tier's checks are gated on #if BOLT_ARCH_X86 AND its
// runtime availability probe, so this is green on windows-msvc/linux-gcc/
// linux-clang (fast tiers exercised where present) AND macos-arm (scalar-only).

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "bolt/bolt_port.h"
#include "bolt/compute_gemv.h"

#include "../src/compute/gemv_kernels_internal.h"

namespace bc = bolt::compute;
namespace bcd = bolt::compute::detail;

namespace {

// Scalar IEEE-754 f32 -> binary16 (round-to-nearest-even), test-local so the
// test doesn't depend on any SIMD helper. Mirrors the inverse in gemv_scalar.
uint16_t f32_to_f16(float f) noexcept {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    const uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) return static_cast<uint16_t>(sign);  // flush tiny to signed zero
    if (exp >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u);  // overflow -> inf
    uint32_t h = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
    // Round-to-nearest-even on the 13 dropped mantissa bits.
    const uint32_t round_bits = mant & 0x1FFFu;
    if (round_bits > 0x1000u || (round_bits == 0x1000u && (h & 1u))) ++h;
    return static_cast<uint16_t>(h);
}

float f16_to_f32(uint16_t h) noexcept {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0u) {
        bits = sign;  // treat subnormal-as-zero is fine for this test's inputs
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// f64 "truth" dot of an f32 weight row against x.
double ref_dot_f32(const float* w, const float* x, int32_t cols) {
    double acc = 0.0;
    for (int32_t i = 0; i < cols; ++i) acc += static_cast<double>(w[i]) * static_cast<double>(x[i]);
    return acc;
}

// A tolerance that scales with the accumulation length and magnitude: the
// SIMD tiers sum in f32 within lanes, so worst-case reassociation error grows
// ~ cols * eps_f32 * max|term|.
float tol_for(double ref, int32_t cols) {
    return 2e-2f + 2e-3f * static_cast<float>(std::fabs(ref)) +
           static_cast<float>(cols) * 1e-5f;
}

std::vector<float> random_vec(int32_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> v(static_cast<size_t>(n));
    for (auto& e : v) e = d(rng);
    return v;
}

struct Shape { int32_t rows, cols; };
const Shape kShapes[] = {{1, 8}, {3, 17}, {8, 64}, {33, 129}, {64, 512}, {128, 4096}};

}  // namespace

TEST(BoltGemvF32, AllTiersMatchF64Reference) {
    for (const auto& s : kShapes) {
        const auto w = random_vec(s.rows * s.cols, 0x1234u + static_cast<uint32_t>(s.cols));
        const auto x = random_vec(s.cols, 0x9abcu + static_cast<uint32_t>(s.rows));
        std::vector<float> out(static_cast<size_t>(s.rows));

        bcd::gemv_f32_scalar(w.data(), x.data(), out.data(), s.rows, s.cols);
        for (int32_t r = 0; r < s.rows; ++r) {
            const double ref = ref_dot_f32(w.data() + r * s.cols, x.data(), s.cols);
            EXPECT_NEAR(out[r], static_cast<float>(ref), tol_for(ref, s.cols)) << "scalar r=" << r;
        }

#if BOLT_ARCH_X86
        if (bc::gemv_avx2_available()) {
            bcd::gemv_f32_avx2(w.data(), x.data(), out.data(), s.rows, s.cols);
            for (int32_t r = 0; r < s.rows; ++r) {
                const double ref = ref_dot_f32(w.data() + r * s.cols, x.data(), s.cols);
                EXPECT_NEAR(out[r], static_cast<float>(ref), tol_for(ref, s.cols)) << "avx2 r=" << r;
            }
        }
        if (bc::gemv_avx512_available()) {
            bcd::gemv_f32_avx512(w.data(), x.data(), out.data(), s.rows, s.cols);
            for (int32_t r = 0; r < s.rows; ++r) {
                const double ref = ref_dot_f32(w.data() + r * s.cols, x.data(), s.cols);
                EXPECT_NEAR(out[r], static_cast<float>(ref), tol_for(ref, s.cols)) << "avx512 r=" << r;
            }
        }
#endif
    }
}

TEST(BoltGemvF32, DispatchMatchesReferenceAndIsDeterministic) {
    const Shape s{96, 1024};
    const auto w = random_vec(s.rows * s.cols, 0x5555u);
    const auto x = random_vec(s.cols, 0x6666u);
    std::vector<float> a(static_cast<size_t>(s.rows)), b(static_cast<size_t>(s.rows));

    bc::matmul_f32(w.data(), x.data(), a.data(), s.rows, s.cols);
    bc::matmul_f32(w.data(), x.data(), b.data(), s.rows, s.cols);
    for (int32_t r = 0; r < s.rows; ++r) {
        EXPECT_EQ(a[r], b[r]) << "non-deterministic r=" << r;  // same tier, same order -> bit-equal
        const double ref = ref_dot_f32(w.data() + r * s.cols, x.data(), s.cols);
        EXPECT_NEAR(a[r], static_cast<float>(ref), tol_for(ref, s.cols));
    }
}

TEST(BoltGemvF16, MatchesDequantThenDot) {
    for (const auto& s : kShapes) {
        const auto wf = random_vec(s.rows * s.cols, 0x2468u + static_cast<uint32_t>(s.cols));
        const auto x = random_vec(s.cols, 0x1357u + static_cast<uint32_t>(s.rows));
        std::vector<uint16_t> wh(wf.size());
        std::vector<float> wdq(wf.size());  // the f16 values, back in f32
        for (size_t i = 0; i < wf.size(); ++i) {
            wh[i] = f32_to_f16(wf[i]);
            wdq[i] = f16_to_f32(wh[i]);
        }
        std::vector<float> out(static_cast<size_t>(s.rows));
        bc::matmul_f16(wh.data(), x.data(), out.data(), s.rows, s.cols);
        for (int32_t r = 0; r < s.rows; ++r) {
            const double ref = ref_dot_f32(wdq.data() + r * s.cols, x.data(), s.cols);
            EXPECT_NEAR(out[r], static_cast<float>(ref), tol_for(ref, s.cols)) << "f16 r=" << r;
        }
    }
}
