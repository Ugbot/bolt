#pragma once
// bolt_vector_kmeans.h — Row-major k-means for Embedding columns.
//
// Companion to `bolt_kmeans.h`. The pre-existing header operates on a
// COLUMN-MAJOR slab (PDX-style: `slab[d * cluster_stride + i]`). This
// header operates on the ROW-MAJOR layout that backs
// `BoltType::Embedding` columns — vectors live contiguously, vector i
// at `vecs + i * dim`. Choosing one over the other avoids a transpose
// at the call boundary: the orchestrator (`marbledb::columnar_vector_
// search`) reads vectors directly out of an `Embedding` column.
//
// Wave 9.4 Phase 1B. Lives in Bolt because the centroid math is reused
// across every consumer that builds an IVF index on top of an
// Embedding column. Tiger Style:
//   - noexcept everywhere
//   - ≥ 2 asserts per hot-path function
//   - fixed constexpr caps on D and K
//   - no exceptions, no RTTI, no STL containers in hot paths
//   - scalar-first; AVX2 / NEON specialisations layer on later
//
// Companion header `bolt_vector_bond.h` carries the PDX-BOND
// zone-rank and σ-bound kernels that drive query-time pruning.

#include "bolt/bolt_port.h"
#include "bolt/bolt_arena.h"
#include "bolt/kernels/bolt_topk.h"             // argmin_f32

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace bolt {
namespace vec {

// Hard upper bounds. Match `bolt_kmeans.h` (kKmeansMaxDim / kKmeansMaxK)
// so the two paths can be swapped without surprise.
inline constexpr uint32_t kVectorKmeansMaxDim = 4096;
inline constexpr uint32_t kVectorKmeansMaxK   = 4096;

// ===========================================================================
// Internal: SplitMix64 — deterministic 64-bit hash → float in [0, 1).
// ===========================================================================
//
// k-means++ seed selection needs a seedable PRNG that is portable
// across runs. SplitMix64 is the simplest portable choice; for our
// scale (k ≤ 4096, n_vecs ≤ 2^32) bias is negligible.

namespace detail_km {

BOLT_FORCE_INLINE uint64_t splitmix64(uint64_t& state) noexcept {
    state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

BOLT_FORCE_INLINE float splitmix_unit(uint64_t& state) noexcept {
    // 53 bits of mantissa → [0, 1); cast back to float (≈ 24 bits).
    const uint64_t v = splitmix64(state) >> 11;
    return static_cast<float>(static_cast<double>(v) *
                              (1.0 / static_cast<double>(1ULL << 53)));
}

// Squared L2 between two row-major dim-aligned vectors. Scalar inner
// loop — the compiler vectorises this aggressively under
// -O3 -mavx2 -mfma.
BOLT_FORCE_INLINE float pair_l2_sq(const float* BOLT_RESTRICT a,
                                    const float* BOLT_RESTRICT b,
                                    uint32_t dim) noexcept {
    assert(a != nullptr); assert(b != nullptr);
    float acc = 0.0f;
    for (uint32_t d = 0; d < dim; ++d) {
        const float t = a[d] - b[d];
        acc += t * t;
    }
    return acc;
}

}  // namespace detail_km

// ===========================================================================
// kmeans_plusplus_seed_f32 — pick K initial centroids via k-means++
// ===========================================================================
//
// Standard k-means++ probability proportional to squared-distance to
// nearest already-chosen centroid. `vecs` is row-major:
// `vecs[i * dim + d]` is dim d of vector i.
//
// `scratch` must hold at least `n_vecs * sizeof(float)` bytes for the
// running min-distance array; we allocate from `arena` if `scratch` is
// nullptr.
//
// Returns true on success, false on bad arguments / OOM. The caller is
// expected to provide `out_centroids` sized `k * dim * sizeof(float)`.

BOLT_FORCE_INLINE bool kmeans_plusplus_seed_f32(
    const float* BOLT_RESTRICT vecs,
    uint32_t n_vecs, uint32_t dim,
    uint32_t k, uint64_t seed,
    float* BOLT_RESTRICT out_centroids,
    Arena* scratch) noexcept {
    assert(scratch != nullptr);
    assert(out_centroids != nullptr || k == 0);
    if (vecs == nullptr && n_vecs > 0) return false;
    if (n_vecs == 0 || k == 0) return false;
    if (k > n_vecs) return false;
    if (dim == 0u || dim > kVectorKmeansMaxDim) return false;
    if (k > kVectorKmeansMaxK) return false;

    auto* min_d2 = static_cast<float*>(
        scratch->allocate(static_cast<size_t>(n_vecs) * sizeof(float)));
    if (min_d2 == nullptr) return false;
    for (uint32_t i = 0; i < n_vecs; ++i) min_d2[i] = 1e30f;

    uint64_t state = seed;
    // First centroid: uniform random pick.
    {
        const uint64_t r = detail_km::splitmix64(state);
        const uint32_t first = static_cast<uint32_t>(r % n_vecs);
        std::memcpy(out_centroids,
                    vecs + static_cast<size_t>(first) * dim,
                    static_cast<size_t>(dim) * sizeof(float));
    }

    for (uint32_t c = 1; c < k; ++c) {
        const float* BOLT_RESTRICT prev = out_centroids +
            static_cast<size_t>(c - 1) * dim;
        // Update running min-distance for every vector.
        double total = 0.0;
        for (uint32_t i = 0; i < n_vecs; ++i) {
            const float* BOLT_RESTRICT vi = vecs +
                static_cast<size_t>(i) * dim;
            const float d2 = detail_km::pair_l2_sq(vi, prev, dim);
            if (d2 < min_d2[i]) min_d2[i] = d2;
            total += static_cast<double>(min_d2[i]);
        }
        // Weighted sample proportional to min_d2.
        const double target =
            total * static_cast<double>(detail_km::splitmix_unit(state));
        double accum = 0.0;
        uint32_t pick = n_vecs - 1u;
        for (uint32_t i = 0; i < n_vecs; ++i) {
            accum += static_cast<double>(min_d2[i]);
            if (accum >= target) { pick = i; break; }
        }
        std::memcpy(out_centroids + static_cast<size_t>(c) * dim,
                    vecs + static_cast<size_t>(pick) * dim,
                    static_cast<size_t>(dim) * sizeof(float));
        min_d2[pick] = 0.0f;
    }
    return true;
}

// ===========================================================================
// kmeans_lloyd_step_f32 — single Lloyd iteration on row-major data.
// ===========================================================================
//
// Assigns each vector to its nearest centroid, then recomputes each
// centroid as the mean of assigned vectors. Returns the total inertia
// (Σ ||vi - assigned_centroid||²) so the caller can detect
// convergence by tracking deltas across iterations.
//
// Empty clusters keep their prior centroid (not zeroed) so a
// subsequent restart can still split the largest cluster.

BOLT_FORCE_INLINE double kmeans_lloyd_step_f32(
    const float* BOLT_RESTRICT vecs,
    uint32_t n_vecs, uint32_t dim,
    uint32_t k,
    float* BOLT_RESTRICT centroids,
    uint32_t* BOLT_RESTRICT assign,
    Arena* scratch) noexcept {
    assert(centroids != nullptr); assert(scratch != nullptr);
    if (vecs == nullptr && n_vecs > 0) return 0.0;
    if (n_vecs == 0 || k == 0) return 0.0;
    if (dim == 0u || dim > kVectorKmeansMaxDim) return 0.0;
    if (k > kVectorKmeansMaxK) return 0.0;

    // Assignment pass.
    double inertia = 0.0;
    for (uint32_t i = 0; i < n_vecs; ++i) {
        const float* BOLT_RESTRICT vi = vecs +
            static_cast<size_t>(i) * dim;
        float best = 1e30f;
        uint32_t arg = 0u;
        for (uint32_t c = 0; c < k; ++c) {
            const float d2 = detail_km::pair_l2_sq(vi,
                centroids + static_cast<size_t>(c) * dim, dim);
            if (d2 < best) { best = d2; arg = c; }
        }
        assign[i] = arg;
        inertia += static_cast<double>(best);
    }

    // Accumulate pass — fresh sums.
    const size_t sum_bytes = static_cast<size_t>(k) *
        static_cast<size_t>(dim) * sizeof(double);
    const size_t cnt_bytes = static_cast<size_t>(k) * sizeof(uint32_t);
    auto* sums = static_cast<double*>(scratch->allocate(sum_bytes));
    auto* counts = static_cast<uint32_t*>(scratch->allocate(cnt_bytes));
    if (sums == nullptr || counts == nullptr) return inertia;
    std::memset(sums, 0, sum_bytes);
    std::memset(counts, 0, cnt_bytes);

    for (uint32_t i = 0; i < n_vecs; ++i) {
        const uint32_t c = assign[i];
        const float* BOLT_RESTRICT vi = vecs +
            static_cast<size_t>(i) * dim;
        double* BOLT_RESTRICT row = sums +
            static_cast<size_t>(c) * dim;
        for (uint32_t d = 0; d < dim; ++d) row[d] += vi[d];
        counts[c] += 1u;
    }
    // Mean pass — leave empty clusters alone (keep prior centroid).
    for (uint32_t c = 0; c < k; ++c) {
        if (counts[c] == 0u) continue;
        const double inv = 1.0 / static_cast<double>(counts[c]);
        double* BOLT_RESTRICT src = sums +
            static_cast<size_t>(c) * dim;
        float* BOLT_RESTRICT dst = centroids +
            static_cast<size_t>(c) * dim;
        for (uint32_t d = 0; d < dim; ++d) {
            dst[d] = static_cast<float>(src[d] * inv);
        }
    }
    return inertia;
}

// ===========================================================================
// kmeans_train_f32 — full driver (++seed + Lloyd iterations + restarts)
// ===========================================================================
//
// Returns the best inertia across `restarts` independent seedings.
// Each restart re-seeds via k-means++ then iterates Lloyd up to
// `max_iters` times or until inertia improvement < 0.1 %.

BOLT_FORCE_INLINE bool kmeans_train_f32(
    const float* BOLT_RESTRICT vecs,
    uint32_t n_vecs, uint32_t dim,
    uint32_t k,
    uint32_t max_iters, uint32_t restarts, uint64_t seed,
    float* BOLT_RESTRICT out_centroids,
    uint32_t* BOLT_RESTRICT out_assign,
    Arena* scratch) noexcept {
    assert(out_centroids != nullptr); assert(out_assign != nullptr);
    if (vecs == nullptr && n_vecs > 0) return false;
    if (n_vecs == 0 || k == 0 || k > n_vecs) return false;
    if (dim == 0u || dim > kVectorKmeansMaxDim) return false;
    if (k > kVectorKmeansMaxK) return false;
    if (restarts == 0u) restarts = 1u;
    if (max_iters == 0u) max_iters = 20u;

    const size_t cents_bytes = static_cast<size_t>(k) *
        static_cast<size_t>(dim) * sizeof(float);
    const size_t asgn_bytes  = static_cast<size_t>(n_vecs) *
        sizeof(uint32_t);
    auto* try_cents = static_cast<float*>(scratch->allocate(cents_bytes));
    auto* try_asgn  = static_cast<uint32_t*>(scratch->allocate(asgn_bytes));
    if (try_cents == nullptr || try_asgn == nullptr) return false;

    double best_inertia = 1e308;
    uint64_t state = seed;
    for (uint32_t r = 0; r < restarts; ++r) {
        const uint64_t restart_seed = detail_km::splitmix64(state);
        if (!kmeans_plusplus_seed_f32(vecs, n_vecs, dim, k,
                                       restart_seed, try_cents,
                                       scratch)) {
            return false;
        }
        double prev = 1e308;
        for (uint32_t it = 0; it < max_iters; ++it) {
            const double ine = kmeans_lloyd_step_f32(
                vecs, n_vecs, dim, k,
                try_cents, try_asgn, scratch);
            if (it > 0 && prev - ine < prev * 1e-3) break;
            prev = ine;
        }
        if (prev < best_inertia) {
            best_inertia = prev;
            std::memcpy(out_centroids, try_cents, cents_bytes);
            std::memcpy(out_assign,    try_asgn,  asgn_bytes);
        }
    }
    return true;
}

// ===========================================================================
// centroid_lower_bound_l2_f32 — Weber 1998 VA-File-style cluster prune
// ===========================================================================
//
// Given a query, a cluster centroid, and the cluster's radius (the
// maximum distance from the centroid to any vector in the cluster),
// returns a provable lower bound on the L2 distance from the query to
// any vector inside the cluster:
//
//     LB = max(0, ||q - centroid|| - radius)
//
// If this exceeds the current kth-best distance, the cluster can be
// skipped entirely. The caller is expected to pre-compute the
// per-cluster radius at build time (one Welford pass over the cluster
// assignments) and store it alongside the centroid.

BOLT_FORCE_INLINE float centroid_lower_bound_l2_f32(
    const float* BOLT_RESTRICT q,
    const float* BOLT_RESTRICT centroid,
    uint32_t dim, float radius) noexcept {
    assert(q != nullptr); assert(centroid != nullptr);
    assert(radius >= 0.0f);
    float d2 = 0.0f;
    for (uint32_t d = 0; d < dim; ++d) {
        const float t = q[d] - centroid[d];
        d2 += t * t;
    }
    // Convert to an L2 distance (sqrt) so the radius subtraction is
    // dimensionally correct; the caller compares against an L2 tau,
    // not an L2² tau.
    const float dist = (d2 > 0.0f) ? std::sqrt(d2) : 0.0f;
    const float lb = dist - radius;
    return (lb > 0.0f) ? lb : 0.0f;
}

}  // namespace vec
}  // namespace bolt
