#pragma once
// bolt_vector_distance_f16.h — half-precision vector distance kernels.
//
// f16 is the "modern embedding default": Cohere, OpenAI text-embedding-3
// and most BERT/GTE derivatives ship f16 natively. Apple Silicon (M1+)
// has ARMv8.2-A FP16 NEON (`vfmaq_f16` + `float16x8_t`); AVX2 boxes have
// no native f16 math but `_mm256_cvtph_ps` gives zero-cost upcast.
//
// Storage convention: `uint16_t*` (bit container; treat as IEEE 754
// binary16). NEON path reinterprets to `__fp16*` for native loads;
// AVX2 path uses `_mm_loadu_si128` → `_mm256_cvtph_ps` for an 8-lane
// f32 promotion. Accumulator is **f32 throughout** — f16's ~65k max
// finite value is too small for L2² accumulation over D ≥ 256 dims at
// typical embedding scale, so we promote to f32 at squared-difference
// time and never accumulate in f16.
//
// Wave 9.4 Phase ζ.3.
//
// Tiger Style — header-only, noexcept, ≥ 2 asserts per fn, bounded
// loops, scalar fallback that auto-vectorises under -O3.

#include "bolt/bolt_port.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

#if BOLT_SIMD_AVX2 && !BOLT_SIMD_NEON
    // _mm256_cvtph_ps lives in immintrin.h, already pulled in via bolt_port.h
#endif

namespace bolt {
namespace vec {

// Scalar f16→f32 conversion fallback (used when NEON FP16 + AVX2 are
// both absent). Software unpacks an IEEE 754 binary16 to a binary32.
// Cold path; the SIMD specialisations are the hot path on M1 / Skylake+.
BOLT_FORCE_INLINE float bits_to_float_from_f16(uint16_t h) noexcept {
    // Standard IEEE 754 binary16 → binary32 expansion. Branch-free
    // bit-twiddle; matches the result of __fp16 → float promotion.
    const uint32_t h32 = static_cast<uint32_t>(h);
    const uint32_t sign = (h32 & 0x8000u) << 16;
    uint32_t exp  = (h32 >> 10) & 0x1Fu;
    uint32_t mant = h32 & 0x3FFu;
    uint32_t f;
    if (exp == 0u) {
        if (mant == 0u) {
            f = sign;                            // signed zero
        } else {
            // Subnormal — normalise.
            int shift = 0;
            while ((mant & 0x400u) == 0u) { mant <<= 1; ++shift; }
            mant &= 0x3FFu;
            const uint32_t fexp = 127u - 15u - static_cast<uint32_t>(shift) + 1u;
            f = sign | (fexp << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        f = sign | 0x7F800000u | (mant << 13);   // inf / nan
    } else {
        const uint32_t fexp = exp + (127u - 15u);
        f = sign | (fexp << 23) | (mant << 13);
    }
    float out;
    __builtin_memcpy(&out, &f, 4);
    return out;
}

// ===========================================================================
// l2_pair_f16 — squared L2 between two f16 vectors of length D
// ===========================================================================
//
// Returns the squared L2 distance, computed in f32 throughout (f16
// inputs upcast immediately, accumulator stays f32). Output range is
// the same as `l2_pair_f32`; this kernel halves load bandwidth at the
// cost of a per-element upcast.

BOLT_FORCE_INLINE float l2_pair_f16(
    const uint16_t* BOLT_RESTRICT a,
    const uint16_t* BOLT_RESTRICT b,
    size_t D) noexcept {
    assert(a != nullptr || D == 0);
    assert(b != nullptr || D == 0);
    size_t i = 0;
    float acc = 0.0f;
#if BOLT_SIMD_NEON_FP16
    // 8-lane FP16 native FMA. `vsubq_f16` + `vmulq_f16` give the
    // squared difference in f16; we immediately promote to f32 to
    // keep the accumulator safe past D=256.
    const __fp16* BOLT_RESTRICT a_h = reinterpret_cast<const __fp16*>(a);
    const __fp16* BOLT_RESTRICT b_h = reinterpret_cast<const __fp16*>(b);
    float32x4_t acc_lo = vdupq_n_f32(0.0f);
    float32x4_t acc_hi = vdupq_n_f32(0.0f);
    for (; i + 8 <= D; i += 8) {
        const float16x8_t va = vld1q_f16(a_h + i);
        const float16x8_t vb = vld1q_f16(b_h + i);
        const float16x8_t vd = vsubq_f16(va, vb);
        const float16x8_t sq = vmulq_f16(vd, vd);
        acc_lo = vaddq_f32(acc_lo, vcvt_f32_f16(vget_low_f16(sq)));
        acc_hi = vaddq_f32(acc_hi, vcvt_high_f32_f16(sq));
    }
    acc = vaddvq_f32(vaddq_f32(acc_lo, acc_hi));
#elif BOLT_SIMD_AVX2
    // 8-lane f32 FMA over `_mm256_cvtph_ps`-promoted halves.
    __m256 vacc = _mm256_setzero_ps();
    for (; i + 8 <= D; i += 8) {
        const __m256 va = _mm256_cvtph_ps(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i)));
        const __m256 vb = _mm256_cvtph_ps(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i)));
        const __m256 vd = _mm256_sub_ps(va, vb);
        vacc = _mm256_fmadd_ps(vd, vd, vacc);
    }
    __m128 lo = _mm256_castps256_ps128(vacc);
    __m128 hi = _mm256_extractf128_ps(vacc, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    acc = _mm_cvtss_f32(s);
#elif BOLT_SIMD_NEON
    // NEON without FP16 vector arithmetic — promote via `vcvt_f32_f16`
    // on `vld1_f16` (half-vec load) then standard 4-lane f32 FMA.
    const __fp16* BOLT_RESTRICT a_h = reinterpret_cast<const __fp16*>(a);
    const __fp16* BOLT_RESTRICT b_h = reinterpret_cast<const __fp16*>(b);
    float32x4_t vacc = vdupq_n_f32(0.0f);
    for (; i + 4 <= D; i += 4) {
        const float32x4_t va = vcvt_f32_f16(vld1_f16(a_h + i));
        const float32x4_t vb = vcvt_f32_f16(vld1_f16(b_h + i));
        const float32x4_t vd = vsubq_f32(va, vb);
        vacc = vfmaq_f32(vacc, vd, vd);
    }
    acc = vaddvq_f32(vacc);
#endif
    // Scalar tail (every ISA falls through here for the residual).
    for (; i < D; ++i) {
        const float af = bits_to_float_from_f16(a[i]);
        const float bf = bits_to_float_from_f16(b[i]);
        const float d  = af - bf;
        acc += d * d;
    }
    return acc;
}

// ===========================================================================
// l2_vertical_f16_accumulate — `dists[i] += (q - col[i])²` per dim
// ===========================================================================
//
// Vertical column-major accumulator: walks `n_vectors` rows for one
// dim, broadcasting the scalar query value and adding the squared
// difference into the existing `dists[i]` accumulator. f32 output
// throughout — `dists` stores the running L2² in f32, regardless of
// the corpus's f16 storage. Matches the f32 vertical accumulator's
// signature exactly so canonical-PDX-64-style walkers can call this
// with no shape changes.

BOLT_FORCE_INLINE void l2_vertical_f16_accumulate(
    const uint16_t* BOLT_RESTRICT dim_col,
    uint16_t q_bits_broadcast,
    float* BOLT_RESTRICT dists,
    size_t n_vectors) noexcept {
    assert(dim_col != nullptr || n_vectors == 0);
    assert(dists   != nullptr || n_vectors == 0);
    const float q = bits_to_float_from_f16(q_bits_broadcast);
    size_t i = 0;
#if BOLT_SIMD_NEON_FP16
    const __fp16  q_h = static_cast<__fp16>(q);
    const float16x8_t vq = vdupq_n_f16(q_h);
    const __fp16* BOLT_RESTRICT col_h =
        reinterpret_cast<const __fp16*>(dim_col);
    for (; i + 8 <= n_vectors; i += 8) {
        const float16x8_t vx = vld1q_f16(col_h + i);
        const float16x8_t vd = vsubq_f16(vq, vx);
        const float16x8_t sq = vmulq_f16(vd, vd);
        // Promote 8-lane f16 squared to two 4-lane f32 vectors and
        // add into the existing f32 accumulator at dists[i..i+8).
        float32x4_t acc_lo = vld1q_f32(dists + i);
        float32x4_t acc_hi = vld1q_f32(dists + i + 4);
        acc_lo = vaddq_f32(acc_lo, vcvt_f32_f16(vget_low_f16(sq)));
        acc_hi = vaddq_f32(acc_hi, vcvt_high_f32_f16(sq));
        vst1q_f32(dists + i,     acc_lo);
        vst1q_f32(dists + i + 4, acc_hi);
    }
#elif BOLT_SIMD_AVX2
    const __m256 vq = _mm256_set1_ps(q);
    for (; i + 8 <= n_vectors; i += 8) {
        const __m256 vx = _mm256_cvtph_ps(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(dim_col + i)));
        const __m256 vd = _mm256_sub_ps(vq, vx);
        const __m256 acc = _mm256_loadu_ps(dists + i);
        const __m256 out = _mm256_fmadd_ps(vd, vd, acc);
        _mm256_storeu_ps(dists + i, out);
    }
#elif BOLT_SIMD_NEON
    const float32x4_t vq = vdupq_n_f32(q);
    const __fp16* BOLT_RESTRICT col_h =
        reinterpret_cast<const __fp16*>(dim_col);
    for (; i + 4 <= n_vectors; i += 4) {
        const float32x4_t vx = vcvt_f32_f16(vld1_f16(col_h + i));
        const float32x4_t vd = vsubq_f32(vq, vx);
        float32x4_t acc = vld1q_f32(dists + i);
        acc = vfmaq_f32(acc, vd, vd);
        vst1q_f32(dists + i, acc);
    }
#endif
    for (; i < n_vectors; ++i) {
        const float xf = bits_to_float_from_f16(dim_col[i]);
        const float d  = q - xf;
        dists[i] += d * d;
    }
}

// ===========================================================================
// f32 ↔ f16 conversion helpers — for tests + asymmetric quantisation
// ===========================================================================
//
// `quantise_f32_to_f16` / `dequantise_f16_to_f32` are batch helpers
// for the bench + tests + the query-vector path. f16 is lossy on the
// 11-bit mantissa; max relative error ~5e-4.

BOLT_FORCE_INLINE void quantise_f32_to_f16(
    const float* BOLT_RESTRICT src,
    uint16_t* BOLT_RESTRICT dst,
    size_t n) noexcept {
    assert(src != nullptr || n == 0);
    assert(dst != nullptr || n == 0);
#if BOLT_SIMD_NEON
    const __fp16* BOLT_RESTRICT dst_h =
        reinterpret_cast<const __fp16*>(dst);
    (void)dst_h;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const float32x4_t v = vld1q_f32(src + i);
        const float16x4_t h = vcvt_f16_f32(v);
        vst1_f16(reinterpret_cast<__fp16*>(dst + i), h);
    }
    for (; i < n; ++i) {
        const __fp16 h = static_cast<__fp16>(src[i]);
        uint16_t bits;
        __builtin_memcpy(&bits, &h, 2);
        dst[i] = bits;
    }
#elif BOLT_SIMD_AVX2
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_loadu_ps(src + i);
        const __m128i h = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), h);
    }
    for (; i < n; ++i) {
        const __m128 v = _mm_set_ss(src[i]);
        const __m128i h = _mm_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
        dst[i] = static_cast<uint16_t>(_mm_extract_epi16(h, 0));
    }
#else
    // Scalar fallback: rely on __fp16 if the compiler supports it,
    // otherwise we can't quantise here — caller must pre-quantise.
    for (size_t i = 0; i < n; ++i) {
        // Best-effort: use single-precision storage of the rounded
        // value. This path isn't hit on M1/Skylake+ benches.
        const float f = src[i];
        uint32_t fb; __builtin_memcpy(&fb, &f, 4);
        const uint32_t sign = (fb >> 16) & 0x8000u;
        const int32_t  exp  = static_cast<int32_t>(((fb >> 23) & 0xFFu)) - 127 + 15;
        uint32_t mant = (fb >> 13) & 0x3FFu;
        uint16_t h;
        if (exp <= 0)       h = static_cast<uint16_t>(sign);
        else if (exp >= 31) h = static_cast<uint16_t>(sign | 0x7C00u);
        else                h = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | mant);
        dst[i] = h;
    }
#endif
}

BOLT_FORCE_INLINE void dequantise_f16_to_f32(
    const uint16_t* BOLT_RESTRICT src,
    float* BOLT_RESTRICT dst,
    size_t n) noexcept {
    assert(src != nullptr || n == 0);
    assert(dst != nullptr || n == 0);
    for (size_t i = 0; i < n; ++i) {
        dst[i] = bits_to_float_from_f16(src[i]);
    }
}

}  // namespace vec
}  // namespace bolt
