#pragma once
// bolt_vector_distance.h — Pair + vertical (column-major) distance kernels
// for f32 vectors. Header-only, scalar-first, AVX2 specialisation gated by
// `BOLT_SIMD_AVX2`. Mirrors the existing bolt_topk.h / bolt_zonemap.h shape.
//
// Six functions:
//   - l2_vertical_f32_accumulate          dists[i] += (q - col[i])^2
//   - ip_vertical_f32_accumulate          dists[i] += q * col[i]
//   - cosine_vertical_f32_accumulate_sxqx sxqx[i] += q * col[i];
//                                         sxx[i]  += col[i] * col[i]
//   - l2_pair_f32                          ||a - b||^2  (dim D)
//   - ip_pair_f32                          a · b        (dim D)
//   - cosine_pair_f32                      a · b / (||a|| * ||b||)
//
// Caller is responsible for zeroing accumulators before the first dim.
//
// Tiger Style: noexcept, ≥2 asserts per function, BOLT_RESTRICT on every
// pointer parameter, no STL containers, no exceptions.
//
// Promoted from `src/ext/pdx_kernels.cpp` (marbledb) so the math leaves
// live in bolt and become callable from any consumer (HNSW, BM25, future
// ANN code-paths). PDX-specific kernels (σ-bound, BOND rank-score,
// centroid-dists, ADSampling prune) stay in `src/ext/`; only the L2
// vertical/horizontal accumulators lift here, with IP and cosine added.

#include "bolt/bolt_port.h"

#include <cassert>
#include <cmath>
#include <cstddef>

#if BOLT_SIMD_AVX2
#include <immintrin.h>
#endif

namespace bolt {

// ---------------------------------------------------------------------------
// Vertical accumulators. Column-major sweep across `n_vectors` at one dim.
// Caller resets `dists` (and `sxqx`/`sxx` for cosine) to 0 before the first
// dim, then calls these once per dim with `q_broadcast = query[d]` and
// `dim_col = slab + d * stride`.
// ---------------------------------------------------------------------------

BOLT_FORCE_INLINE void l2_vertical_f32_accumulate(
    const float* BOLT_RESTRICT dim_col, float q_broadcast,
    float* BOLT_RESTRICT dists, size_t n_vectors) noexcept {
    assert(dim_col != nullptr || n_vectors == 0);
    assert(dists   != nullptr || n_vectors == 0);
    size_t i = 0;
#if BOLT_SIMD_AVX2
    const __m256 q = _mm256_set1_ps(q_broadcast);
    for (; i + 8 <= n_vectors; i += 8) {
        const __m256 x   = _mm256_loadu_ps(dim_col + i);
        const __m256 acc = _mm256_loadu_ps(dists   + i);
        const __m256 d   = _mm256_sub_ps(q, x);
        const __m256 out = _mm256_fmadd_ps(d, d, acc);
        _mm256_storeu_ps(dists + i, out);
    }
#elif BOLT_SIMD_NEON
    const float32x4_t q = vdupq_n_f32(q_broadcast);
    for (; i + 4 <= n_vectors; i += 4) {
        const float32x4_t x   = vld1q_f32(dim_col + i);
        const float32x4_t acc = vld1q_f32(dists   + i);
        const float32x4_t d   = vsubq_f32(q, x);
        const float32x4_t out = vfmaq_f32(acc, d, d);
        vst1q_f32(dists + i, out);
    }
#endif
    for (; i < n_vectors; ++i) {
        const float d = q_broadcast - dim_col[i];
        dists[i] += d * d;
    }
}

BOLT_FORCE_INLINE void ip_vertical_f32_accumulate(
    const float* BOLT_RESTRICT dim_col, float q_broadcast,
    float* BOLT_RESTRICT dists, size_t n_vectors) noexcept {
    assert(dim_col != nullptr || n_vectors == 0);
    assert(dists   != nullptr || n_vectors == 0);
    size_t i = 0;
#if BOLT_SIMD_AVX2
    const __m256 q = _mm256_set1_ps(q_broadcast);
    for (; i + 8 <= n_vectors; i += 8) {
        const __m256 x   = _mm256_loadu_ps(dim_col + i);
        const __m256 acc = _mm256_loadu_ps(dists   + i);
        const __m256 out = _mm256_fmadd_ps(q, x, acc);
        _mm256_storeu_ps(dists + i, out);
    }
#elif BOLT_SIMD_NEON
    const float32x4_t q = vdupq_n_f32(q_broadcast);
    for (; i + 4 <= n_vectors; i += 4) {
        const float32x4_t x   = vld1q_f32(dim_col + i);
        const float32x4_t acc = vld1q_f32(dists   + i);
        const float32x4_t out = vfmaq_f32(acc, q, x);
        vst1q_f32(dists + i, out);
    }
#endif
    for (; i < n_vectors; ++i) {
        dists[i] += q_broadcast * dim_col[i];
    }
}

BOLT_FORCE_INLINE void cosine_vertical_f32_accumulate_sxqx(
    const float* BOLT_RESTRICT dim_col, float q_broadcast,
    float* BOLT_RESTRICT sxqx, float* BOLT_RESTRICT sxx,
    size_t n_vectors) noexcept {
    assert(dim_col != nullptr || n_vectors == 0);
    assert(sxqx    != nullptr || n_vectors == 0);
    assert(sxx     != nullptr || n_vectors == 0);
    size_t i = 0;
#if BOLT_SIMD_AVX2
    const __m256 q = _mm256_set1_ps(q_broadcast);
    for (; i + 8 <= n_vectors; i += 8) {
        const __m256 x   = _mm256_loadu_ps(dim_col + i);
        const __m256 a   = _mm256_loadu_ps(sxqx    + i);
        const __m256 b   = _mm256_loadu_ps(sxx     + i);
        const __m256 oa  = _mm256_fmadd_ps(q, x, a);
        const __m256 ob  = _mm256_fmadd_ps(x, x, b);
        _mm256_storeu_ps(sxqx + i, oa);
        _mm256_storeu_ps(sxx  + i, ob);
    }
#elif BOLT_SIMD_NEON
    const float32x4_t q = vdupq_n_f32(q_broadcast);
    for (; i + 4 <= n_vectors; i += 4) {
        const float32x4_t x   = vld1q_f32(dim_col + i);
        const float32x4_t a   = vld1q_f32(sxqx    + i);
        const float32x4_t b   = vld1q_f32(sxx     + i);
        const float32x4_t oa  = vfmaq_f32(a, q, x);
        const float32x4_t ob  = vfmaq_f32(b, x, x);
        vst1q_f32(sxqx + i, oa);
        vst1q_f32(sxx  + i, ob);
    }
#endif
    for (; i < n_vectors; ++i) {
        const float x = dim_col[i];
        sxqx[i] += q_broadcast * x;
        sxx[i]  += x * x;
    }
}

// ---------------------------------------------------------------------------
// Pair distances — single vector × single vector, length D.
// ---------------------------------------------------------------------------

BOLT_FORCE_INLINE float l2_pair_f32(
    const float* BOLT_RESTRICT a,
    const float* BOLT_RESTRICT b,
    size_t D) noexcept {
    assert(a != nullptr || D == 0);
    assert(b != nullptr || D == 0);
    size_t i = 0;
    float acc = 0.0f;
#if BOLT_SIMD_AVX2
    __m256 vacc = _mm256_setzero_ps();
    for (; i + 8 <= D; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        const __m256 vd = _mm256_sub_ps(va, vb);
        vacc = _mm256_fmadd_ps(vd, vd, vacc);
    }
    // Horizontal sum.
    __m128 lo = _mm256_castps256_ps128(vacc);
    __m128 hi = _mm256_extractf128_ps(vacc, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    acc = _mm_cvtss_f32(s);
#elif BOLT_SIMD_NEON
    float32x4_t vacc = vdupq_n_f32(0.0f);
    for (; i + 4 <= D; i += 4) {
        const float32x4_t va = vld1q_f32(a + i);
        const float32x4_t vb = vld1q_f32(b + i);
        const float32x4_t vd = vsubq_f32(va, vb);
        vacc = vfmaq_f32(vacc, vd, vd);
    }
    acc = vaddvq_f32(vacc);
#endif
    for (; i < D; ++i) {
        const float d = a[i] - b[i];
        acc += d * d;
    }
    return acc;
}

BOLT_FORCE_INLINE float ip_pair_f32(
    const float* BOLT_RESTRICT a,
    const float* BOLT_RESTRICT b,
    size_t D) noexcept {
    assert(a != nullptr || D == 0);
    assert(b != nullptr || D == 0);
    size_t i = 0;
    float acc = 0.0f;
#if BOLT_SIMD_AVX2
    __m256 vacc = _mm256_setzero_ps();
    for (; i + 8 <= D; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        vacc = _mm256_fmadd_ps(va, vb, vacc);
    }
    __m128 lo = _mm256_castps256_ps128(vacc);
    __m128 hi = _mm256_extractf128_ps(vacc, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    acc = _mm_cvtss_f32(s);
#elif BOLT_SIMD_NEON
    float32x4_t vacc = vdupq_n_f32(0.0f);
    for (; i + 4 <= D; i += 4) {
        const float32x4_t va = vld1q_f32(a + i);
        const float32x4_t vb = vld1q_f32(b + i);
        vacc = vfmaq_f32(vacc, va, vb);
    }
    acc = vaddvq_f32(vacc);
#endif
    for (; i < D; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
}

BOLT_FORCE_INLINE float cosine_pair_f32(
    const float* BOLT_RESTRICT a,
    const float* BOLT_RESTRICT b,
    size_t D) noexcept {
    assert(a != nullptr || D == 0);
    assert(b != nullptr || D == 0);
    size_t i = 0;
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
#if BOLT_SIMD_AVX2
    __m256 vdot = _mm256_setzero_ps();
    __m256 vna  = _mm256_setzero_ps();
    __m256 vnb  = _mm256_setzero_ps();
    for (; i + 8 <= D; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        vdot = _mm256_fmadd_ps(va, vb, vdot);
        vna  = _mm256_fmadd_ps(va, va, vna);
        vnb  = _mm256_fmadd_ps(vb, vb, vnb);
    }
    auto hsum256 = [](__m256 v) noexcept -> float {
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 s  = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        return _mm_cvtss_f32(s);
    };
    dot = hsum256(vdot);
    na  = hsum256(vna);
    nb  = hsum256(vnb);
#elif BOLT_SIMD_NEON
    float32x4_t vdot = vdupq_n_f32(0.0f);
    float32x4_t vna  = vdupq_n_f32(0.0f);
    float32x4_t vnb  = vdupq_n_f32(0.0f);
    for (; i + 4 <= D; i += 4) {
        const float32x4_t va = vld1q_f32(a + i);
        const float32x4_t vb = vld1q_f32(b + i);
        vdot = vfmaq_f32(vdot, va, vb);
        vna  = vfmaq_f32(vna,  va, va);
        vnb  = vfmaq_f32(vnb,  vb, vb);
    }
    dot = vaddvq_f32(vdot);
    na  = vaddvq_f32(vna);
    nb  = vaddvq_f32(vnb);
#endif
    for (; i < D; ++i) {
        const float ai = a[i];
        const float bi = b[i];
        dot += ai * bi;
        na  += ai * ai;
        nb  += bi * bi;
    }
    const float denom = std::sqrt(na) * std::sqrt(nb);
    // Caller convention: zero-norm vector → cosine of 0.
    return (denom > 0.0f) ? (dot / denom) : 0.0f;
}

}  // namespace bolt
