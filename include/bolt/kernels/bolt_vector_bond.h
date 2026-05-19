#pragma once
// bolt_vector_bond.h — PDX-BOND zone ranking + σ-bound kernels.
//
// PDX-BOND (Kuffo/Krippner/Boncz SIGMOD 2025) orders dimension visits
// by per-cluster distance-to-mean (`|q_d − μ_{c,d}|`). The dim-zone
// variant groups consecutive dims into zones of `zone_size` and ranks
// zones for sequential memory access (paper Fig. 8: 30–40 % faster
// than per-dim ordering).
//
// σ-bound column-level early termination: once partial squared L2 is
// computed over a visited dim-subset `S`, a provable lower bound on
// the contribution of unvisited dims lets the search skip the whole
// cluster when `partial + bound > τ`:
//
//   exact (Weber 1998 VA-File):
//       UB_d = max((q_d − min_d)², (q_d − max_d)²)
//   probabilistic (Chebyshev–Cantelli, k=1):
//       UB_d^{σk} = (q_d − μ_d)² + (1+k)² × σ²_d
//
// Both kernels take a `visited_mask` bitmap so the caller can carry
// per-dim "already accumulated" state and update incrementally.
//
// Wave 9.4 Phase 1B. Tiger Style — noexcept, ≥ 2 asserts per fn,
// bounded loops, scalar-first inner loops that auto-vectorise under
// `-O3 -mavx2 -mfma`.

#include "bolt/bolt_port.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace bolt {
namespace vec {

inline constexpr uint32_t kBondMaxDim = 4096;

// ===========================================================================
// bond_zone_rank_scores_f32 — score each contiguous dim-zone
// ===========================================================================
//
// For each contiguous group of `zone_size` dimensions, write the sum
// of `|q_d - cluster_means[d]|` into `out_scores`. The caller then
// argsort-descends over `out_scores` to walk the most promising
// zones first.
//
// `out_scores` length must be at least `ceil(dim / zone_size)`.
// Returns the number of zones written.

BOLT_FORCE_INLINE uint32_t bond_zone_rank_scores_f32(
    const float* BOLT_RESTRICT q,
    const float* BOLT_RESTRICT cluster_means,
    uint32_t dim, uint32_t zone_size,
    float* BOLT_RESTRICT out_scores) noexcept {
    assert(q != nullptr); assert(cluster_means != nullptr);
    assert(out_scores != nullptr); assert(zone_size > 0u);
    assert(dim <= kBondMaxDim);
    if (dim == 0u) return 0u;

    const uint32_t n_zones = (dim + zone_size - 1u) / zone_size;
    for (uint32_t z = 0; z < n_zones; ++z) {
        const uint32_t lo = z * zone_size;
        const uint32_t hi = (lo + zone_size <= dim) ? (lo + zone_size) : dim;
        float acc = 0.0f;
        uint32_t d = lo;
#if BOLT_SIMD_NEON
        float32x4_t vacc = vdupq_n_f32(0.0f);
        for (; d + 4u <= hi; d += 4u) {
            const float32x4_t vq = vld1q_f32(q + d);
            const float32x4_t vm = vld1q_f32(cluster_means + d);
            vacc = vaddq_f32(vacc, vabdq_f32(vq, vm));
        }
        acc = vaddvq_f32(vacc);
#endif
        for (; d < hi; ++d) {
            const float diff = q[d] - cluster_means[d];
            acc += (diff >= 0.0f) ? diff : -diff;
        }
        out_scores[z] = acc;
    }
    return n_zones;
}

// ===========================================================================
// box_lower_bound_l2_f32 — exact lower bound for block-skip
// ===========================================================================
//
// Returns the minimum possible squared L2 distance from `q` to any
// point inside the axis-aligned box [min_d, max_d] in every
// dimension. Per-dim contribution:
//   if q_d < min_d : (min_d − q_d)²
//   if q_d > max_d : (q_d − max_d)²
//   else           : 0
//
// If this lower bound exceeds the current kth-best τ, every vector
// in the block is provably farther than τ — the block can be
// skipped entirely. Recall-1.0 preserved.
//
// `visited_mask` lets the orchestrator carry incremental row-level
// state (don't double-count dims already accumulated into the
// partial distance). nullptr ⇒ all dims contribute.

BOLT_FORCE_INLINE float box_lower_bound_l2_f32(
    const float* BOLT_RESTRICT q,
    const float* BOLT_RESTRICT min_d,
    const float* BOLT_RESTRICT max_d,
    const uint8_t* visited_mask,
    uint32_t dim) noexcept {
    assert(q != nullptr); assert(min_d != nullptr); assert(max_d != nullptr);
    assert(dim <= kBondMaxDim);
    float acc = 0.0f;
    uint32_t d = 0;
#if BOLT_SIMD_NEON
    if (visited_mask == nullptr) {
        // Branchless 4-lane:
        //   gap = max(min_d - q, q - max_d, 0)  (one side is ≤ 0 inside box)
        const float32x4_t vzero = vdupq_n_f32(0.0f);
        float32x4_t vacc = vzero;
        for (; d + 4u <= dim; d += 4u) {
            const float32x4_t vq  = vld1q_f32(q + d);
            const float32x4_t vmn = vld1q_f32(min_d + d);
            const float32x4_t vmx = vld1q_f32(max_d + d);
            const float32x4_t lo  = vsubq_f32(vmn, vq);   // > 0 if q < min
            const float32x4_t hi  = vsubq_f32(vq,  vmx);  // > 0 if q > max
            const float32x4_t gap = vmaxq_f32(vmaxq_f32(lo, hi), vzero);
            vacc = vfmaq_f32(vacc, gap, gap);
        }
        acc = vaddvq_f32(vacc);
    }
#endif
    for (; d < dim; ++d) {
        if (visited_mask != nullptr &&
            ((visited_mask[d >> 3] >> (d & 7u)) & 1u) != 0u) {
            continue;
        }
        const float qd = q[d];
        float gap = 0.0f;
        if (qd < min_d[d]) gap = min_d[d] - qd;
        else if (qd > max_d[d]) gap = qd - max_d[d];
        acc += gap * gap;
    }
    return acc;
}

// ===========================================================================
// sigma_bound_remaining_minmax_f32 — exact upper bound (row-level prune)
// ===========================================================================
//
// Returns Σ_{d ∉ visited} max((q_d − min_d)², (q_d − max_d)²). Adding
// this to a row's `partial_distance` (computed over the visited dim
// subset) gives an exact UPPER bound on the row's full L2². If
// `partial + bound ≤ τ`, the row is a guaranteed top-k candidate;
// if `partial > τ` already, drop.
//
// Use `box_lower_bound_l2_f32` for whole-block skip — that's the
// lower bound. This kernel is the upper bound, used by Phase 4's
// row-level ADSampling / σ-walk inside a probed block.
//
// Recall=1.0 preserved (no probabilistic argument). `visited_mask`
// is one bit per dim (LSB first). nullptr ⇒ no dims visited yet.

BOLT_FORCE_INLINE float sigma_bound_remaining_minmax_f32(
    const float* BOLT_RESTRICT q,
    const float* BOLT_RESTRICT min_d,
    const float* BOLT_RESTRICT max_d,
    const uint8_t* visited_mask,
    uint32_t dim) noexcept {
    assert(q != nullptr); assert(min_d != nullptr); assert(max_d != nullptr);
    assert(dim <= kBondMaxDim);
    float acc = 0.0f;
    uint32_t d = 0;
#if BOLT_SIMD_NEON
    if (visited_mask == nullptr) {
        float32x4_t vacc = vdupq_n_f32(0.0f);
        for (; d + 4u <= dim; d += 4u) {
            const float32x4_t vq  = vld1q_f32(q + d);
            const float32x4_t vmn = vld1q_f32(min_d + d);
            const float32x4_t vmx = vld1q_f32(max_d + d);
            const float32x4_t lo  = vsubq_f32(vq, vmn);
            const float32x4_t hi  = vsubq_f32(vq, vmx);
            const float32x4_t lo2 = vmulq_f32(lo, lo);
            const float32x4_t hi2 = vmulq_f32(hi, hi);
            vacc = vaddq_f32(vacc, vmaxq_f32(lo2, hi2));
        }
        acc = vaddvq_f32(vacc);
    } else {
        // Masked 4-lane: build lane-keep mask from up to 8 bits of one byte.
        float32x4_t vacc = vdupq_n_f32(0.0f);
        for (; d + 4u <= dim; d += 4u) {
            const uint8_t  byte  = visited_mask[d >> 3];
            const uint32_t shift = d & 7u;
            uint32_t m[4];
            for (uint32_t i = 0; i < 4u; ++i) {
                const uint32_t bit = (byte >> (shift + i)) & 1u;
                m[i] = bit ? 0u : 0xFFFFFFFFu;
            }
            const float32x4_t vq  = vld1q_f32(q + d);
            const float32x4_t vmn = vld1q_f32(min_d + d);
            const float32x4_t vmx = vld1q_f32(max_d + d);
            const float32x4_t lo  = vsubq_f32(vq, vmn);
            const float32x4_t hi  = vsubq_f32(vq, vmx);
            const float32x4_t lo2 = vmulq_f32(lo, lo);
            const float32x4_t hi2 = vmulq_f32(hi, hi);
            const float32x4_t ub  = vmaxq_f32(lo2, hi2);
            const uint32x4_t  keep = vld1q_u32(m);
            const uint32x4_t  ub_u = vreinterpretq_u32_f32(ub);
            const uint32x4_t  masked = vandq_u32(ub_u, keep);
            vacc = vaddq_f32(vacc, vreinterpretq_f32_u32(masked));
        }
        acc = vaddvq_f32(vacc);
    }
#endif
    for (; d < dim; ++d) {
        if (visited_mask != nullptr &&
            ((visited_mask[d >> 3] >> (d & 7u)) & 1u) != 0u) {
            continue;
        }
        const float lo = q[d] - min_d[d];
        const float hi = q[d] - max_d[d];
        const float lo2 = lo * lo;
        const float hi2 = hi * hi;
        acc += (lo2 > hi2) ? lo2 : hi2;
    }
    return acc;
}

// ===========================================================================
// sigma_bound_remaining_chebyshev_f32 — probabilistic σk bound
// ===========================================================================
//
// Looser-than-exact bound that uses per-dim mean / variance instead
// of min / max. Returns Σ_{d ∉ visited} ((q_d − μ_d)² + (1+k)² × σ²_d).
// Default OFF; gated by the caller because the false-drop rate is
// non-zero (≤ Chebyshev-Cantelli tail at k σ).

BOLT_FORCE_INLINE float sigma_bound_remaining_chebyshev_f32(
    const float* BOLT_RESTRICT q,
    const float* BOLT_RESTRICT mean_d,
    const float* BOLT_RESTRICT var_d,
    const uint8_t* visited_mask,
    uint32_t dim, float k_sigma) noexcept {
    assert(q != nullptr); assert(mean_d != nullptr); assert(var_d != nullptr);
    assert(dim <= kBondMaxDim); assert(k_sigma >= 0.0f);
    const float factor = (1.0f + k_sigma) * (1.0f + k_sigma);
    float acc = 0.0f;
    uint32_t d = 0;
#if BOLT_SIMD_NEON
    const float32x4_t vfactor = vdupq_n_f32(factor);
    if (visited_mask == nullptr) {
        float32x4_t vacc = vdupq_n_f32(0.0f);
        for (; d + 4u <= dim; d += 4u) {
            const float32x4_t vq  = vld1q_f32(q + d);
            const float32x4_t vm  = vld1q_f32(mean_d + d);
            const float32x4_t vv  = vld1q_f32(var_d + d);
            const float32x4_t df  = vsubq_f32(vq, vm);
            float32x4_t term = vmulq_f32(df, df);
            term = vfmaq_f32(term, vfactor, vv);
            vacc = vaddq_f32(vacc, term);
        }
        acc = vaddvq_f32(vacc);
    } else {
        float32x4_t vacc = vdupq_n_f32(0.0f);
        for (; d + 4u <= dim; d += 4u) {
            const uint8_t  byte  = visited_mask[d >> 3];
            const uint32_t shift = d & 7u;
            uint32_t m[4];
            for (uint32_t i = 0; i < 4u; ++i) {
                const uint32_t bit = (byte >> (shift + i)) & 1u;
                m[i] = bit ? 0u : 0xFFFFFFFFu;
            }
            const float32x4_t vq  = vld1q_f32(q + d);
            const float32x4_t vm  = vld1q_f32(mean_d + d);
            const float32x4_t vv  = vld1q_f32(var_d + d);
            const float32x4_t df  = vsubq_f32(vq, vm);
            float32x4_t term = vmulq_f32(df, df);
            term = vfmaq_f32(term, vfactor, vv);
            const uint32x4_t keep = vld1q_u32(m);
            const uint32x4_t term_u = vreinterpretq_u32_f32(term);
            const uint32x4_t masked = vandq_u32(term_u, keep);
            vacc = vaddq_f32(vacc, vreinterpretq_f32_u32(masked));
        }
        acc = vaddvq_f32(vacc);
    }
#endif
    for (; d < dim; ++d) {
        if (visited_mask != nullptr &&
            ((visited_mask[d >> 3] >> (d & 7u)) & 1u) != 0u) {
            continue;
        }
        const float diff = q[d] - mean_d[d];
        acc += diff * diff + factor * var_d[d];
    }
    return acc;
}

// ===========================================================================
// sigma_bound_remaining_chebyshev_lb_f32 — probabilistic σk LOWER bound
// ===========================================================================
//
// Returns Σ_{d ∉ visited} max(0, |q_d − μ_d| − k_σ × σ_d)².
// Probabilistic lower bound on the remaining-dim contribution to L2².
//
// Cf. Chebyshev–Cantelli: P(|X − μ| ≥ k σ) ≤ 1/k². With k_σ = 1, the
// LB holds for ≥ 0 fraction of values; with k_σ = 3, ≥ 88.9 %. The
// caller chooses k_σ as a recall/speed knob: higher k_σ → looser LB
// → fewer false drops but less aggressive prune.
//
// Recall: combining with `box_lower_bound_l2_f32` via
//   LB_combined = max(LB_box, LB_chebyshev(k_σ))
// is recall-preserving only when LB_chebyshev ≤ LB_box. The caller is
// responsible for ensuring that invariant by choosing k_σ high enough
// that the Chebyshev tail probability fits the data; default
// k_σ = 3 is the safe paper-canonical choice.
//
// Per-dim contribution (let s = √var_d[d], df = q[d] − μ_d[d]):
//   gap = |df| − k_σ × s
//   contrib = (gap > 0) ? gap² : 0
//
// At k_σ = 0 this reduces to Σ (q_d − μ_d)² (centroid-LB, deterministic).

BOLT_FORCE_INLINE float sigma_bound_remaining_chebyshev_lb_f32(
    const float* BOLT_RESTRICT q,
    const float* BOLT_RESTRICT mean_d,
    const float* BOLT_RESTRICT var_d,
    const uint8_t* visited_mask,
    uint32_t dim, float k_sigma) noexcept {
    assert(q != nullptr); assert(mean_d != nullptr); assert(var_d != nullptr);
    assert(dim <= kBondMaxDim); assert(k_sigma >= 0.0f);
    float acc = 0.0f;
    uint32_t d = 0;
#if BOLT_SIMD_NEON
    const float32x4_t vk    = vdupq_n_f32(k_sigma);
    const float32x4_t vzero = vdupq_n_f32(0.0f);
    if (visited_mask == nullptr) {
        float32x4_t vacc = vdupq_n_f32(0.0f);
        for (; d + 4u <= dim; d += 4u) {
            const float32x4_t vq  = vld1q_f32(q + d);
            const float32x4_t vm  = vld1q_f32(mean_d + d);
            const float32x4_t vv  = vld1q_f32(var_d + d);
            const float32x4_t vs  = vsqrtq_f32(vv);
            const float32x4_t df  = vabdq_f32(vq, vm);
            const float32x4_t ks  = vmulq_f32(vk, vs);
            const float32x4_t gap = vmaxq_f32(vsubq_f32(df, ks), vzero);
            vacc = vfmaq_f32(vacc, gap, gap);
        }
        acc = vaddvq_f32(vacc);
    } else {
        float32x4_t vacc = vdupq_n_f32(0.0f);
        for (; d + 4u <= dim; d += 4u) {
            const uint8_t  byte  = visited_mask[d >> 3];
            const uint32_t shift = d & 7u;
            uint32_t m[4];
            for (uint32_t i = 0; i < 4u; ++i) {
                const uint32_t bit = (byte >> (shift + i)) & 1u;
                m[i] = bit ? 0u : 0xFFFFFFFFu;
            }
            const float32x4_t vq  = vld1q_f32(q + d);
            const float32x4_t vm  = vld1q_f32(mean_d + d);
            const float32x4_t vv  = vld1q_f32(var_d + d);
            const float32x4_t vs  = vsqrtq_f32(vv);
            const float32x4_t df  = vabdq_f32(vq, vm);
            const float32x4_t ks  = vmulq_f32(vk, vs);
            const float32x4_t gap = vmaxq_f32(vsubq_f32(df, ks), vzero);
            const float32x4_t term = vmulq_f32(gap, gap);
            const uint32x4_t keep = vld1q_u32(m);
            const uint32x4_t term_u = vreinterpretq_u32_f32(term);
            const uint32x4_t masked = vandq_u32(term_u, keep);
            vacc = vaddq_f32(vacc, vreinterpretq_f32_u32(masked));
        }
        acc = vaddvq_f32(vacc);
    }
#endif
    for (; d < dim; ++d) {
        if (visited_mask != nullptr &&
            ((visited_mask[d >> 3] >> (d & 7u)) & 1u) != 0u) {
            continue;
        }
        const float diff = q[d] - mean_d[d];
        const float abs_diff = (diff >= 0.0f) ? diff : -diff;
        const float sigma = __builtin_sqrtf(var_d[d]);
        const float gap = abs_diff - k_sigma * sigma;
        if (gap > 0.0f) acc += gap * gap;
    }
    assert(acc >= 0.0f);
    return acc;
}

// ===========================================================================
// adsampling_threshold_f32 — ADSampling per-vector prune gate
// ===========================================================================
//
// Gao & Long SIGMOD 2023. Given the current kth-best distance `kth`,
// the dim-fraction d/D walked so far, and a tolerance ε, returns the
// threshold above which a partial-distance walk can be aborted.
//
//   T(d) = kth × (d / D) × (1 + ε / sqrt(d))²
//
// Caller compares the partial distance to T(d) per vector; if it
// exceeds T(d), the vector can be pruned without scanning the
// remaining dims.

BOLT_FORCE_INLINE float adsampling_threshold_f32(
    float kth, uint32_t d_visited, uint32_t dim, float epsilon) noexcept {
    assert(d_visited <= dim);
    assert(dim > 0u);
    assert(kth >= 0.0f); assert(epsilon >= 0.0f);
    if (d_visited == 0u) return 0.0f;
    const float frac = static_cast<float>(d_visited) /
                       static_cast<float>(dim);
    const float root = __builtin_sqrtf(static_cast<float>(d_visited));
    const float bend = 1.0f + epsilon / root;
    return kth * frac * bend * bend;
}

}  // namespace vec
}  // namespace bolt
