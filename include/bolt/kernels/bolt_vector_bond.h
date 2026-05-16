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
#include <cmath>
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
        for (uint32_t d = lo; d < hi; ++d) {
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
    for (uint32_t d = 0; d < dim; ++d) {
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
    for (uint32_t d = 0; d < dim; ++d) {
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
    for (uint32_t d = 0; d < dim; ++d) {
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
    const float root = std::sqrt(static_cast<float>(d_visited));
    const float bend = 1.0f + epsilon / root;
    return kth * frac * bend * bend;
}

}  // namespace vec
}  // namespace bolt
