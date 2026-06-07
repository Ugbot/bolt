#pragma once
// bolt_vector_distance_u8.h — 8-bit scalar-quantised vector distance
// kernels. Wave 9.4 Phase ζ.4.
//
// **Asymmetric quantisation**: corpus and query are BOTH `uint8_t`,
// quantised using the same per-row-group `QuantParams`. L2² is
// computed entirely in integer arithmetic (u8 inputs, u32
// accumulator); the f32 L2² is recoverable via multiplication by
// `inv_scale_squared` if the caller needs cross-precision comparison.
// For top-K within a single u8 column, the u32 accumulator's relative
// ordering matches the f32 ordering exactly — no fp dequant on the
// hot path.
//
// SIMD specialisations:
//   * **NEON DOTPROD (ARMv8.4-A / M1+)** — `vdotq_u32` 4-way 8-bit
//     dot product into u32 accumulator. After absolute-difference
//     `vabdq_u8`, computes `sum(|a-b|²)` as four parallel dot products
//     into four u32 lanes. 16 elements per loop iteration.
//   * **AVX2** — `_mm256_cvtepu8_epi16` 16-lane upcast, then
//     `_mm256_sub_epi16` + `_mm256_madd_epi16` for fused
//     multiply-accumulate of squared differences. 16 elements per loop.
//   * **NEON baseline (no DOTPROD)** — `vmovl_u8` upcast +
//     `vmulq_u16` square + `vpaddlq_u16` widening pairwise sum to u32.
//     Slower than DOTPROD but still SIMD.
//   * **Scalar** — straight integer loop; compiler auto-vectorises on
//     -O3.
//
// Tiger Style — header-only, noexcept, ≥ 2 asserts per fn, bounded
// loops. No exceptions, no RTTI, no STL.

#include "bolt/bolt_port.h"
#include "bolt/kernels/bolt_vector_traits.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace bolt {
namespace vec {

// ===========================================================================
// l2_pair_u8 — squared L2 between two u8 vectors of length D
// ===========================================================================
//
// Returns the u32 sum of squared differences. To convert to f32 L2²
// (compatible with the f32 kernel's output), multiply by
// `params.inv_scale_squared`. For top-K within a single column,
// callers can compare u32 values directly without dequantising.

BOLT_FORCE_INLINE uint32_t l2_pair_u8(
    const uint8_t* BOLT_RESTRICT a,
    const uint8_t* BOLT_RESTRICT b,
    size_t D) noexcept {
    assert(a != nullptr || D == 0);
    assert(b != nullptr || D == 0);
    size_t i = 0;
    uint32_t acc = 0u;
#if BOLT_SIMD_NEON_DOTPROD
    // 16-lane 8-bit absolute difference, then 4-way dot product into
    // u32 accumulator. `vdotq_u32(acc, x, x)` computes
    //   acc[j] += sum(x[4j..4j+3] * x[4j..4j+3]) for j in 0..3
    // so the four u32 lanes accumulate the sum-of-squares of the
    // 16-byte absolute-difference vector. Final horizontal sum at end.
    uint32x4_t vacc = vdupq_n_u32(0u);
    for (; i + 16 <= D; i += 16) {
        const uint8x16_t va = vld1q_u8(a + i);
        const uint8x16_t vb = vld1q_u8(b + i);
        const uint8x16_t vd = vabdq_u8(va, vb);
        vacc = vdotq_u32(vacc, vd, vd);
    }
    acc = vaddvq_u32(vacc);
#elif BOLT_SIMD_AVX2
    // 16-lane u8 upcast to i16, sub, then madd (sum of two products
    // into i32 lanes). `_mm256_madd_epi16(x, x)` gives `sum(x[2j] *
    // x[2j+1])` per i32 lane; for sum-of-squares we feed (diff, diff)
    // so each lane is `sum(diff[2j]² + diff[2j+1]²)`. 16 elements
    // per iteration.
    // Four independent i32 accumulators break the loop-carried add chain
    // (madd→add latency) so the pipeline can overlap iterations. They are
    // exact integer sums, so reassociating them changes nothing.
    __m256i vacc0 = _mm256_setzero_si256();
    __m256i vacc1 = _mm256_setzero_si256();
    __m256i vacc2 = _mm256_setzero_si256();
    __m256i vacc3 = _mm256_setzero_si256();
    for (; i + 64 <= D; i += 64) {
        const __m256i d0 = _mm256_sub_epi16(
            _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i))),
            _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i))));
        const __m256i d1 = _mm256_sub_epi16(
            _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i + 16))),
            _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i + 16))));
        const __m256i d2 = _mm256_sub_epi16(
            _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i + 32))),
            _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i + 32))));
        const __m256i d3 = _mm256_sub_epi16(
            _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i + 48))),
            _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i + 48))));
        vacc0 = _mm256_add_epi32(vacc0, _mm256_madd_epi16(d0, d0));
        vacc1 = _mm256_add_epi32(vacc1, _mm256_madd_epi16(d1, d1));
        vacc2 = _mm256_add_epi32(vacc2, _mm256_madd_epi16(d2, d2));
        vacc3 = _mm256_add_epi32(vacc3, _mm256_madd_epi16(d3, d3));
    }
    for (; i + 16 <= D; i += 16) {
        const __m128i va8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
        const __m128i vb8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i));
        const __m256i va16 = _mm256_cvtepu8_epi16(va8);
        const __m256i vb16 = _mm256_cvtepu8_epi16(vb8);
        const __m256i vd16 = _mm256_sub_epi16(va16, vb16);
        // vd16 lanes are i16 in [-255, 255]; madd squares + pairwise
        // sums into i32. Maxes out at 2 * 255² = 130050 per i32 lane
        // per iteration — fits in i32 for D ≤ 32k iterations.
        vacc0 = _mm256_add_epi32(vacc0, _mm256_madd_epi16(vd16, vd16));
    }
    const __m256i vacc = _mm256_add_epi32(_mm256_add_epi32(vacc0, vacc1),
                                          _mm256_add_epi32(vacc2, vacc3));
    // Horizontal sum of 8 i32 lanes.
    const __m128i lo = _mm256_castsi256_si128(vacc);
    const __m128i hi = _mm256_extracti128_si256(vacc, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    acc = static_cast<uint32_t>(_mm_cvtsi128_si32(s));
#elif BOLT_SIMD_NEON
    // Baseline NEON without DOTPROD: upcast u8 → u16, multiply, then
    // widen pairwise sum to u32. 8 elements per iteration.
    uint32x4_t vacc = vdupq_n_u32(0u);
    for (; i + 8 <= D; i += 8) {
        const uint8x8_t  va = vld1_u8(a + i);
        const uint8x8_t  vb = vld1_u8(b + i);
        const uint8x8_t  vd = vabd_u8(va, vb);
        const uint16x8_t vd16 = vmovl_u8(vd);
        const uint16x8_t vsq  = vmulq_u16(vd16, vd16);
        vacc = vaddq_u32(vacc, vpaddlq_u16(vsq));
    }
    acc = vaddvq_u32(vacc);
#endif
    for (; i < D; ++i) {
        const int32_t d = static_cast<int32_t>(a[i]) - static_cast<int32_t>(b[i]);
        acc += static_cast<uint32_t>(d * d);
    }
    return acc;
}

// ===========================================================================
// l2_vertical_u8_accumulate — `dists[i] += (q - col[i])²` per dim
// ===========================================================================
//
// Vertical column-major accumulator for u8 corpora. `dists` stays
// u32 (the L2² accumulator); the f32 conversion is the caller's
// responsibility at output time.

BOLT_FORCE_INLINE void l2_vertical_u8_accumulate(
    const uint8_t* BOLT_RESTRICT dim_col,
    uint8_t q_broadcast,
    uint32_t* BOLT_RESTRICT dists,
    size_t n_vectors) noexcept {
    assert(dim_col != nullptr || n_vectors == 0);
    assert(dists   != nullptr || n_vectors == 0);
    size_t i = 0;
#if BOLT_SIMD_NEON
    // Same idiom on both DOTPROD and baseline: per-element |q - x|,
    // square, accumulate. DOTPROD doesn't help here because we're
    // splatting q to 16 lanes and dotting against itself — the work
    // shape doesn't match vdotq_u32's 4-way grouping. Use widen+mul.
    const uint8x16_t vq = vdupq_n_u8(q_broadcast);
    for (; i + 16 <= n_vectors; i += 16) {
        const uint8x16_t vx = vld1q_u8(dim_col + i);
        const uint8x16_t vd = vabdq_u8(vq, vx);
        // Square each lane via widen-multiply on the two halves.
        const uint8x8_t  vd_lo = vget_low_u8(vd);
        const uint8x8_t  vd_hi = vget_high_u8(vd);
        const uint16x8_t sq_lo = vmull_u8(vd_lo, vd_lo);
        const uint16x8_t sq_hi = vmull_u8(vd_hi, vd_hi);
        // Promote each 8-lane u16 to two 4-lane u32 and add into dists.
        uint32x4_t acc0 = vld1q_u32(dists + i);
        uint32x4_t acc1 = vld1q_u32(dists + i + 4);
        uint32x4_t acc2 = vld1q_u32(dists + i + 8);
        uint32x4_t acc3 = vld1q_u32(dists + i + 12);
        acc0 = vaddq_u32(acc0, vmovl_u16(vget_low_u16 (sq_lo)));
        acc1 = vaddq_u32(acc1, vmovl_u16(vget_high_u16(sq_lo)));
        acc2 = vaddq_u32(acc2, vmovl_u16(vget_low_u16 (sq_hi)));
        acc3 = vaddq_u32(acc3, vmovl_u16(vget_high_u16(sq_hi)));
        vst1q_u32(dists + i,      acc0);
        vst1q_u32(dists + i + 4,  acc1);
        vst1q_u32(dists + i + 8,  acc2);
        vst1q_u32(dists + i + 12, acc3);
    }
#elif BOLT_SIMD_AVX2
    // Vertical accumulate needs PER-ELEMENT squares (each row i is an
    // independent accumulator), so `madd` (which pairwise-sums adjacent
    // lanes) is NOT usable here — that would merge rows i and i+1.
    // Widen the u8 diff to i32 BEFORE squaring so the square is computed
    // in 32-bit lanes. d ∈ [-255,255] ⇒ d² ≤ 65025, and the i32 square
    // is always exact (no reliance on the d² < 2^16 invariant that the
    // old `mullo_epi16` low-16-bits trick silently depended on).
    const __m256i vq32 = _mm256_set1_epi32(static_cast<int32_t>(q_broadcast));
    for (; i + 16 <= n_vectors; i += 16) {
        const __m128i vx8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dim_col + i));
        // Split the 16 u8 lanes into two 8-lane i32 vectors.
        const __m256i vx32_lo = _mm256_cvtepu8_epi32(vx8);
        const __m256i vx32_hi = _mm256_cvtepu8_epi32(_mm_srli_si128(vx8, 8));
        const __m256i vd_lo = _mm256_sub_epi32(vq32, vx32_lo);
        const __m256i vd_hi = _mm256_sub_epi32(vq32, vx32_hi);
        const __m256i sq32_lo = _mm256_mullo_epi32(vd_lo, vd_lo);
        const __m256i sq32_hi = _mm256_mullo_epi32(vd_hi, vd_hi);
        __m256i acc_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dists + i));
        __m256i acc_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dists + i + 8));
        acc_lo = _mm256_add_epi32(acc_lo, sq32_lo);
        acc_hi = _mm256_add_epi32(acc_hi, sq32_hi);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dists + i),     acc_lo);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dists + i + 8), acc_hi);
    }
#endif
    for (; i < n_vectors; ++i) {
        const int32_t d = static_cast<int32_t>(q_broadcast) - static_cast<int32_t>(dim_col[i]);
        dists[i] += static_cast<uint32_t>(d * d);
    }
}

// ===========================================================================
// quantise_f32_to_u8 — apply ScalarQuantizer<U8> to an f32 batch
// ===========================================================================
//
// Per-element: `q = clamp(round((x - base) * scale), 0, 255)`. Used
// to convert the query (or fresh corpus rows) to the column's u8
// storage at build / ingest time.

BOLT_FORCE_INLINE void quantise_f32_to_u8(
    const float* BOLT_RESTRICT src,
    uint8_t* BOLT_RESTRICT dst,
    size_t n,
    const QuantParams& params) noexcept {
    assert(src != nullptr || n == 0);
    assert(dst != nullptr || n == 0);
    for (size_t i = 0; i < n; ++i) {
        float q = (src[i] - params.base) * params.scale;
        if (q < 0.0f)        q = 0.0f;
        else if (q > 255.0f) q = 255.0f;
        // Banker's rounding via lrintf — matches PDXearch's convention.
        const int32_t qi = static_cast<int32_t>(std::rintf(q));
        dst[i] = static_cast<uint8_t>(qi);
    }
}

BOLT_FORCE_INLINE void dequantise_u8_to_f32(
    const uint8_t* BOLT_RESTRICT src,
    float* BOLT_RESTRICT dst,
    size_t n,
    const QuantParams& params) noexcept {
    assert(src != nullptr || n == 0);
    assert(dst != nullptr || n == 0);
    const float inv_scale = 1.0f / params.scale;
    for (size_t i = 0; i < n; ++i) {
        dst[i] = static_cast<float>(src[i]) * inv_scale + params.base;
    }
}

// ===========================================================================
// compute_quant_params_global_f32 — derive base+scale from an f32 corpus
// ===========================================================================
//
// Global per-row-group params: `base = min`, `scale = 255 / (max-min)`.
// Pre-fills the squared/inverse-squared memos for the hot kernel path.

BOLT_FORCE_INLINE QuantParams compute_quant_params_global_f32(
    const float* BOLT_RESTRICT data,
    size_t n_elements) noexcept {
    assert(data != nullptr || n_elements == 0);
    float min_v = 0.0f, max_v = 0.0f;
    if (n_elements > 0) {
        min_v = data[0]; max_v = data[0];
        for (size_t i = 1; i < n_elements; ++i) {
            const float v = data[i];
            if (v < min_v) min_v = v;
            if (v > max_v) max_v = v;
        }
    }
    QuantParams p{};
    p.base  = min_v;
    const float range = (max_v > min_v) ? (max_v - min_v) : 1.0f;
    p.scale = 255.0f / range;
    p.scale_squared     = p.scale * p.scale;
    p.inv_scale_squared = (p.scale_squared > 0.0f)
        ? (1.0f / p.scale_squared) : 1.0f;
    return p;
}

}  // namespace vec
}  // namespace bolt
