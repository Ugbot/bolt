// gemv_avx512.cpp -- AVX-512F tier for bolt::compute's f32/f16 GEMV (perf
// pass). #if BOLT_ARCH_X86 (empty TU on non-x86). Compiled at /arch:AVX512
// (MSVC) or -mavx512f (GCC/Clang).
//
// Same algorithm as the AVX2 tier (4 independent accumulators + f64 final
// reduction), widened to __m512 (16 f32/reg, 64-wide unrolled body). f16
// weights upcast via _mm512_cvtph_ps -- native to AVX-512F, so no separate
// F16C feature is required for this tier.

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

// dot(w[0..cols), x[0..cols)) with 4x __m512 FMA accumulators + f64 tail.
inline double dot_row_f32_avx512(const float* BOLT_RESTRICT w, const float* BOLT_RESTRICT x,
                                 int32_t cols) noexcept {
    __m512 a0 = _mm512_setzero_ps();
    __m512 a1 = _mm512_setzero_ps();
    __m512 a2 = _mm512_setzero_ps();
    __m512 a3 = _mm512_setzero_ps();
    int32_t i = 0;
    for (; i + 64 <= cols; i += 64) {
        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(w + i), _mm512_loadu_ps(x + i), a0);
        a1 = _mm512_fmadd_ps(_mm512_loadu_ps(w + i + 16), _mm512_loadu_ps(x + i + 16), a1);
        a2 = _mm512_fmadd_ps(_mm512_loadu_ps(w + i + 32), _mm512_loadu_ps(x + i + 32), a2);
        a3 = _mm512_fmadd_ps(_mm512_loadu_ps(w + i + 48), _mm512_loadu_ps(x + i + 48), a3);
    }
    const __m512 s = _mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3));
    double acc = static_cast<double>(_mm512_reduce_add_ps(s));
    for (; i + 16 <= cols; i += 16) {
        acc += static_cast<double>(
            _mm512_reduce_add_ps(_mm512_mul_ps(_mm512_loadu_ps(w + i), _mm512_loadu_ps(x + i))));
    }
    for (; i < cols; ++i) acc += static_cast<double>(w[i] * x[i]);
    return acc;
}

// Same, but the weight row is f16 upcast via _mm512_cvtph_ps (16 at a time).
inline double dot_row_f16_avx512(const uint16_t* BOLT_RESTRICT w, const float* BOLT_RESTRICT x,
                                 int32_t cols) noexcept {
    __m512 a0 = _mm512_setzero_ps();
    __m512 a1 = _mm512_setzero_ps();
    __m512 a2 = _mm512_setzero_ps();
    __m512 a3 = _mm512_setzero_ps();
    int32_t i = 0;
    for (; i + 64 <= cols; i += 64) {
        a0 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i))),
                             _mm512_loadu_ps(x + i), a0);
        a1 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i + 16))),
                             _mm512_loadu_ps(x + i + 16), a1);
        a2 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i + 32))),
                             _mm512_loadu_ps(x + i + 32), a2);
        a3 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i + 48))),
                             _mm512_loadu_ps(x + i + 48), a3);
    }
    const __m512 s = _mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3));
    double acc = static_cast<double>(_mm512_reduce_add_ps(s));
    for (; i + 16 <= cols; i += 16) {
        const __m512 wf = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i)));
        acc += static_cast<double>(_mm512_reduce_add_ps(_mm512_mul_ps(wf, _mm512_loadu_ps(x + i))));
    }
    for (; i < cols; ++i) acc += static_cast<double>(half1_to_float(w[i]) * x[i]);
    return acc;
}

}  // namespace

void gemv_f32_avx512(const float* weight, const float* x, float* out, int32_t rows,
                     int32_t cols) noexcept {
    assert(weight != nullptr && x != nullptr && out != nullptr);
    assert(rows > 0 && cols > 0);
    for (int32_t r = 0; r < rows; ++r) {
        out[r] = static_cast<float>(dot_row_f32_avx512(weight + static_cast<int64_t>(r) * cols, x, cols));
    }
}

void gemv_f16_avx512(const uint16_t* weight, const float* x, float* out, int32_t rows,
                     int32_t cols) noexcept {
    assert(weight != nullptr && x != nullptr && out != nullptr);
    assert(rows > 0 && cols > 0);
    for (int32_t r = 0; r < rows; ++r) {
        out[r] = static_cast<float>(dot_row_f16_avx512(weight + static_cast<int64_t>(r) * cols, x, cols));
    }
}

}  // namespace bolt::compute::detail

#endif  // BOLT_ARCH_X86
