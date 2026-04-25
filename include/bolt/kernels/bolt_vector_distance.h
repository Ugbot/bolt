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
    for (size_t i = 0; i < n_vectors; ++i) {
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
    for (size_t i = 0; i < n_vectors; ++i) {
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
    float acc = 0.0f;
    for (size_t i = 0; i < D; ++i) {
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
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < D; ++i) {
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
