#pragma once
// bolt_kmeans.h — Header-only k-means primitives (assignment, centroid
// update, convergence delta) for f32 vectors stored column-major. Plan D
// of Wave 19 (`docs/research/wave-19-bolt-vector-kernels.md`).
//
// Slab layout: column-major. Vector i, dimension d lives at
//   slab[d * cluster_stride + i]
// where `cluster_stride >= n_vectors`. This matches the PDX-style slab
// layout used elsewhere in the project so the same memory can be fed to
// both the L2 vertical accumulator and these k-means kernels without a
// transpose.
//
// Centroids are row-major K × D: centroid k, dim d at centroids[k*D + d].
//
// Tiger Style: noexcept, ≥2 asserts per function, BOLT_RESTRICT on every
// pointer parameter, no STL containers, no exceptions, scalar-first. AVX2
// specialisations are deferred — see TODOs below.
//
// Reuses `bolt::argmin_f32` from bolt_topk.h for the per-vector argmin.
// Pair distance math is inlined directly (rather than calling
// `bolt::l2_pair_f32`) because the hot inner loop reconstructs the i-th
// vector once into a stack array, then sweeps K centroid rows — both rows
// and the reconstructed vector are dense, so a tight scalar accumulator
// here is simpler and equivalent.

#include "bolt/bolt_port.h"
#include "bolt/kernels/bolt_topk.h"             // argmin_f32
#include "bolt/kernels/bolt_vector_distance.h"  // l2_pair_f32 (header dep)

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace bolt {

// Hard upper bounds (Tiger Style — every loop has a constexpr cap).
// These match `marbledb::ext::pdx::kMaxDim` and the K cap on the heap-
// based top-k. K must fit on a stack array for the per-vector argmin.
inline constexpr size_t kKmeansMaxDim = 4096;
inline constexpr size_t kKmeansMaxK   = 4096;

// ---------------------------------------------------------------------------
// kmeans_assign_f32_l2 — for each vector i, write argmin centroid id.
// ---------------------------------------------------------------------------
//
// `slab`         : column-major float slab, slab[d * cluster_stride + i].
// `D`            : vector dimension, ≤ kKmeansMaxDim.
// `n_vectors`    : number of vectors to assign.
// `cluster_stride`: stride between dim columns (≥ n_vectors).
// `centroids`    : row-major K × D, centroids[k*D + d].
// `K`            : centroid count, ≤ kKmeansMaxK.
// `assignments`  : output; assignments[i] := argmin_k ||vec_i - cent_k||^2.
BOLT_FORCE_INLINE void kmeans_assign_f32_l2(
    const float* BOLT_RESTRICT slab,
    size_t D, size_t n_vectors, size_t cluster_stride,
    const float* BOLT_RESTRICT centroids,
    size_t K,
    uint32_t* BOLT_RESTRICT assignments) noexcept {
    assert(slab        != nullptr || (D == 0 || n_vectors == 0));
    assert(centroids   != nullptr || K == 0);
    assert(assignments != nullptr || n_vectors == 0);
    assert(cluster_stride >= n_vectors);
    assert(D <= kKmeansMaxDim);
    assert(K <= kKmeansMaxK);
    if (n_vectors == 0 || K == 0) return;

    // Stack scratch — bounded by the constexpr caps above.
    float vec_i[kKmeansMaxDim];
    float dists[kKmeansMaxK];

    for (size_t i = 0; i < n_vectors; ++i) {
        // Gather i-th vector out of the column-major slab into a dense
        // row so the per-k inner loop is two contiguous walks.
        for (size_t d = 0; d < D; ++d) {
            vec_i[d] = slab[d * cluster_stride + i];
        }
        // Pair L2 against each centroid. Scalar; AVX2 TODO below.
        for (size_t k = 0; k < K; ++k) {
            const float* BOLT_RESTRICT cent_k = centroids + k * D;
            float acc = 0.0f;
            for (size_t d = 0; d < D; ++d) {
                const float diff = vec_i[d] - cent_k[d];
                acc += diff * diff;
            }
            dists[k] = acc;
        }
        // TODO(avx2): replace the scalar k×D accumulator with an FMA
        // sweep using `l2_pair_f32` — currently inlined for clarity.
        float mn = 0.0f;
        const size_t arg = bolt::argmin_f32(dists, K, &mn);
        assert(arg < K);
        assignments[i] = static_cast<uint32_t>(arg);
    }
}

// ---------------------------------------------------------------------------
// kmeans_update_centroids_f32 — sum/divide pass.
// ---------------------------------------------------------------------------
//
// Sums each cluster's assigned vectors dimension-by-dimension, divides by
// count. Empty clusters are *not* touched (cluster_sizes_out[k] == 0 and
// centroids_out[k*D + ..] == 0 after the zero-init below — caller is
// responsible for re-initialising those rows from a prior centroid copy
// if the convention is "empty cluster keeps prior centroid").
//
// `centroids_out`     : row-major K × D output (overwritten).
// `cluster_sizes_out` : K-element count vector (overwritten).
BOLT_FORCE_INLINE void kmeans_update_centroids_f32(
    const float* BOLT_RESTRICT slab,
    size_t D, size_t n_vectors, size_t cluster_stride,
    const uint32_t* BOLT_RESTRICT assignments,
    size_t K,
    float* BOLT_RESTRICT centroids_out,
    uint32_t* BOLT_RESTRICT cluster_sizes_out) noexcept {
    assert(slab              != nullptr || (D == 0 || n_vectors == 0));
    assert(assignments       != nullptr || n_vectors == 0);
    assert(centroids_out     != nullptr || K == 0);
    assert(cluster_sizes_out != nullptr || K == 0);
    assert(cluster_stride >= n_vectors);
    assert(D <= kKmeansMaxDim);
    assert(K <= kKmeansMaxK);

    // Zero accumulators.
    for (size_t k = 0; k < K; ++k) {
        cluster_sizes_out[k] = 0u;
        float* BOLT_RESTRICT row = centroids_out + k * D;
        for (size_t d = 0; d < D; ++d) row[d] = 0.0f;
    }
    if (n_vectors == 0 || K == 0) return;

    // Sum pass — walk vectors, accumulate into the assigned centroid row.
    for (size_t i = 0; i < n_vectors; ++i) {
        const uint32_t k = assignments[i];
        assert(static_cast<size_t>(k) < K);
        float* BOLT_RESTRICT row = centroids_out + static_cast<size_t>(k) * D;
        for (size_t d = 0; d < D; ++d) {
            row[d] += slab[d * cluster_stride + i];
        }
        cluster_sizes_out[k] += 1u;
    }
    // TODO(avx2): the per-dim accumulate is a column-major scatter-add —
    // replace with a transpose-then-vertical-accumulate when D is large.

    // Divide pass — only non-empty clusters; empties stay at zero so the
    // caller can detect and re-init from the prior centroid array.
    for (size_t k = 0; k < K; ++k) {
        const uint32_t n = cluster_sizes_out[k];
        if (n == 0u) continue;
        const float inv = 1.0f / static_cast<float>(n);
        float* BOLT_RESTRICT row = centroids_out + k * D;
        for (size_t d = 0; d < D; ++d) row[d] *= inv;
    }
}

// ---------------------------------------------------------------------------
// kmeans_centroid_delta_f32 — sum of squared deltas across all K centroids.
// ---------------------------------------------------------------------------
//
// Returns Σ_k Σ_d (new[k*D+d] - old[k*D+d])^2.  Caller compares the result
// against a convergence threshold (e.g. 1e-6).
BOLT_FORCE_INLINE float kmeans_centroid_delta_f32(
    const float* BOLT_RESTRICT centroids_old,
    const float* BOLT_RESTRICT centroids_new,
    size_t K, size_t D) noexcept {
    assert(centroids_old != nullptr || (K == 0 || D == 0));
    assert(centroids_new != nullptr || (K == 0 || D == 0));
    assert(D <= kKmeansMaxDim);
    assert(K <= kKmeansMaxK);
    float acc = 0.0f;
    const size_t total = K * D;
    for (size_t i = 0; i < total; ++i) {
        const float diff = centroids_new[i] - centroids_old[i];
        acc += diff * diff;
    }
    // TODO(avx2): replace with l2_pair_f32 over the (K*D)-flattened arrays.
    return acc;
}

}  // namespace bolt
