// gemv_avx2.cpp -- AVX2 (+FMA +F16C) tier for bolt::compute's f32/f16 GEMV
// (perf pass). #if BOLT_ARCH_X86 (empty TU on non-x86, so ARM consumers link a
// scalar-only path with no dispatch reference to these symbols). Compiled at
// /arch:AVX2 (MSVC) or -mavx2 -mfma -mf16c (GCC/Clang).
//
// Algorithm (from ggml's ggml_vec_dot_f32, written bolt-native): per output
// row, FOUR independent __m256 accumulators over a 32-wide unrolled body to
// hide the ~4-cycle FMA latency, then an 8-wide remainder, then a scalar tail.
// The four accumulators are tree-reduced and horizontally summed in f64. f16
// weights are upcast on the fly with _mm256_cvtph_ps (the zero-cost F16C
// promotion bolt already uses in bolt_vector_distance_f16.h).

#include "bolt/bolt_port.h"

#if BOLT_ARCH_X86

#include "gemv_kernels_internal.h"

#include <cassert>
#include <cstdint>

#include <immintrin.h>

namespace bolt::compute::detail {

namespace {

// One binary16 -> f32 via F16C (portable across MSVC/GCC/Clang, unlike the
// GCC-only _cvtsh_ss spelling). Used only in the scalar tail.
inline float half1_to_float(uint16_t h) noexcept {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(h))));
}

// Horizontal sum of one __m256 into a double (final reduction in f64).
inline double hsum256_to_f64(__m256 v) noexcept {
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    const __m128 shuf = _mm_movehdup_ps(s);   // [1,1,3,3]
    __m128 sums = _mm_add_ps(s, shuf);
    s = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, s);
    return static_cast<double>(_mm_cvtss_f32(sums));
}

// dot(w[0..cols), x[0..cols)) with 4x __m256 FMA accumulators + f64 tail.
inline double dot_row_f32_avx2(const float* BOLT_RESTRICT w, const float* BOLT_RESTRICT x,
                               int32_t cols) noexcept {
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps();
    __m256 a3 = _mm256_setzero_ps();
    int32_t i = 0;
    for (; i + 32 <= cols; i += 32) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + i), _mm256_loadu_ps(x + i), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(w + i + 8), _mm256_loadu_ps(x + i + 8), a1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(w + i + 16), _mm256_loadu_ps(x + i + 16), a2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(w + i + 24), _mm256_loadu_ps(x + i + 24), a3);
    }
    const __m256 s = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    double acc = hsum256_to_f64(s);
    for (; i + 8 <= cols; i += 8) {
        acc += hsum256_to_f64(_mm256_mul_ps(_mm256_loadu_ps(w + i), _mm256_loadu_ps(x + i)));
    }
    for (; i < cols; ++i) acc += static_cast<double>(w[i] * x[i]);
    return acc;
}

// Same, but the weight row is f16 (uint16_t) upcast via F16C 8 at a time.
inline double dot_row_f16_avx2(const uint16_t* BOLT_RESTRICT w, const float* BOLT_RESTRICT x,
                               int32_t cols) noexcept {
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps();
    __m256 a3 = _mm256_setzero_ps();
    int32_t i = 0;
    for (; i + 32 <= cols; i += 32) {
        a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i))),
                             _mm256_loadu_ps(x + i), a0);
        a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i + 8))),
                             _mm256_loadu_ps(x + i + 8), a1);
        a2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i + 16))),
                             _mm256_loadu_ps(x + i + 16), a2);
        a3 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i + 24))),
                             _mm256_loadu_ps(x + i + 24), a3);
    }
    const __m256 s = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    double acc = hsum256_to_f64(s);
    for (; i + 8 <= cols; i += 8) {
        const __m256 wf = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i)));
        acc += hsum256_to_f64(_mm256_mul_ps(wf, _mm256_loadu_ps(x + i)));
    }
    for (; i < cols; ++i) acc += static_cast<double>(half1_to_float(w[i]) * x[i]);
    return acc;
}

}  // namespace

void gemv_f32_avx2(const float* weight, const float* x, float* out, int32_t rows,
                   int32_t cols) noexcept {
    assert(weight != nullptr && x != nullptr && out != nullptr);
    assert(rows > 0 && cols > 0);
    for (int32_t r = 0; r < rows; ++r) {
        out[r] = static_cast<float>(dot_row_f32_avx2(weight + static_cast<int64_t>(r) * cols, x, cols));
    }
}

void gemv_f16_avx2(const uint16_t* weight, const float* x, float* out, int32_t rows,
                   int32_t cols) noexcept {
    assert(weight != nullptr && x != nullptr && out != nullptr);
    assert(rows > 0 && cols > 0);
    for (int32_t r = 0; r < rows; ++r) {
        out[r] = static_cast<float>(dot_row_f16_avx2(weight + static_cast<int64_t>(r) * cols, x, cols));
    }
}

}  // namespace bolt::compute::detail

#endif  // BOLT_ARCH_X86
