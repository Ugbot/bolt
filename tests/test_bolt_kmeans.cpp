// test_bolt_kmeans.cpp — tests for kmeans_assign_f32_l2,
// kmeans_update_centroids_f32, and kmeans_centroid_delta_f32 in
// bolt/kernels/bolt_kmeans.h. Plan D of Wave 19.

#include "bolt/kernels/bolt_kmeans.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace {

// Pack n vectors of dim D from a row-major source into a column-major
// slab of stride n. Helper for hand-built tiny cases.
static void pack_column_major(const float* src_row_major,
                              size_t n, size_t D,
                              float* slab_out /* size = D * n */) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t d = 0; d < D; ++d) {
            slab_out[d * n + i] = src_row_major[i * D + d];
        }
    }
}

// ---------------------------------------------------------------------------
// 1. Tiny case: 4 vectors in 2D, 2 centroids.
//    Vectors: (0,0) (0,1) (10,10) (10,11)
//    Centroids: (0, 0.5)  (10, 10.5)
//    Expected assignments: [0,0,1,1]
//    Expected updated centroids: (0, 0.5) and (10, 10.5) — already
//    optimal, so update is idempotent and centroid delta is 0.
// ---------------------------------------------------------------------------
TEST(BoltKmeans, TinyAssignAndUpdateHandComputed) {
    constexpr size_t D = 2;
    constexpr size_t N = 4;
    constexpr size_t K = 2;

    const float vecs_row[N * D] = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        10.0f, 10.0f,
        10.0f, 11.0f,
    };
    float slab[D * N];
    pack_column_major(vecs_row, N, D, slab);

    float centroids[K * D] = {
        0.0f, 0.5f,
        10.0f, 10.5f,
    };
    float centroids_new[K * D] = {0};
    uint32_t sizes[K] = {0};
    uint32_t assigns[N] = {0};

    bolt::kmeans_assign_f32_l2(slab, D, N, /*cluster_stride=*/N,
                               centroids, K, assigns);

    EXPECT_EQ(assigns[0], 0u);
    EXPECT_EQ(assigns[1], 0u);
    EXPECT_EQ(assigns[2], 1u);
    EXPECT_EQ(assigns[3], 1u);

    bolt::kmeans_update_centroids_f32(slab, D, N, /*cluster_stride=*/N,
                                      assigns, K,
                                      centroids_new, sizes);

    EXPECT_EQ(sizes[0], 2u);
    EXPECT_EQ(sizes[1], 2u);
    EXPECT_FLOAT_EQ(centroids_new[0 * D + 0], 0.0f);
    EXPECT_FLOAT_EQ(centroids_new[0 * D + 1], 0.5f);
    EXPECT_FLOAT_EQ(centroids_new[1 * D + 0], 10.0f);
    EXPECT_FLOAT_EQ(centroids_new[1 * D + 1], 10.5f);

    const float delta = bolt::kmeans_centroid_delta_f32(
        centroids, centroids_new, K, D);
    EXPECT_FLOAT_EQ(delta, 0.0f);
}

// ---------------------------------------------------------------------------
// 2. Convergence: 16 random points in 2D drawn from 4 well-separated
//    Gaussians; 4 centroids initialised at the corners of [-1,1]^2.
//    Run 20 iters of {assign, update}; assert delta drops below 1e-6.
// ---------------------------------------------------------------------------
TEST(BoltKmeans, ConvergesOnFourGaussiansIn20Iters) {
    constexpr size_t D = 2;
    constexpr size_t N = 16;
    constexpr size_t K = 4;

    // Four cluster centres, well separated.
    const float centres[K][D] = {
        {-10.0f, -10.0f},
        { 10.0f, -10.0f},
        {-10.0f,  10.0f},
        { 10.0f,  10.0f},
    };

    std::mt19937 rng(0xC0FFEE);
    std::normal_distribution<float> jitter(0.0f, 0.2f);

    float vecs_row[N * D];
    for (size_t i = 0; i < N; ++i) {
        const size_t c = i % K;  // 4 points per cluster.
        vecs_row[i * D + 0] = centres[c][0] + jitter(rng);
        vecs_row[i * D + 1] = centres[c][1] + jitter(rng);
    }
    float slab[D * N];
    pack_column_major(vecs_row, N, D, slab);

    // Initial centroids at unit-square corners — far from the Gaussians,
    // so the first iters must move them substantially.
    float centroids[K * D] = {
        -1.0f, -1.0f,
         1.0f, -1.0f,
        -1.0f,  1.0f,
         1.0f,  1.0f,
    };
    float centroids_new[K * D] = {0};
    uint32_t sizes[K] = {0};
    uint32_t assigns[N] = {0};

    float delta = 1.0f;
    for (int iter = 0; iter < 20; ++iter) {
        bolt::kmeans_assign_f32_l2(slab, D, N, /*cluster_stride=*/N,
                                   centroids, K, assigns);
        bolt::kmeans_update_centroids_f32(slab, D, N, /*cluster_stride=*/N,
                                          assigns, K,
                                          centroids_new, sizes);
        // Re-init any empty cluster from prior centroid (caller policy).
        for (size_t k = 0; k < K; ++k) {
            if (sizes[k] == 0u) {
                std::memcpy(centroids_new + k * D,
                            centroids + k * D,
                            sizeof(float) * D);
            }
        }
        delta = bolt::kmeans_centroid_delta_f32(centroids,
                                                centroids_new, K, D);
        std::memcpy(centroids, centroids_new, sizeof(centroids));
    }
    EXPECT_LT(delta, 1e-6f) << "k-means did not converge in 20 iters";

    // Sanity: each cluster has 4 points (the round-robin assignment
    // above guarantees an exact 4-way split when initial centroids are
    // closer to their respective Gaussian).
    bolt::kmeans_assign_f32_l2(slab, D, N, N, centroids, K, assigns);
    uint32_t counts[K] = {0};
    for (size_t i = 0; i < N; ++i) counts[assigns[i]]++;
    for (size_t k = 0; k < K; ++k) EXPECT_EQ(counts[k], 4u);
}

// ---------------------------------------------------------------------------
// 3. Empty cluster handling: place a centroid where no vectors are
//    nearest. After update, cluster_sizes_out[k] == 0, the row is zeroed
//    by the kernel (no NaN), and the caller's "keep prior centroid" path
//    leaves it sane.
// ---------------------------------------------------------------------------
TEST(BoltKmeans, EmptyClusterStaysFiniteAndSizeZero) {
    constexpr size_t D = 2;
    constexpr size_t N = 4;
    constexpr size_t K = 3;

    // All four points cluster around (0,0) and (10,10) only.
    const float vecs_row[N * D] = {
        0.0f, 0.0f,
        0.1f, 0.0f,
        10.0f, 10.0f,
        9.9f, 10.1f,
    };
    float slab[D * N];
    pack_column_major(vecs_row, N, D, slab);

    // Centroid 2 is parked far from any point — no vector picks it.
    float centroids[K * D] = {
        0.0f, 0.0f,
        10.0f, 10.0f,
        100.0f, 100.0f,  // empty cluster
    };
    const float prior2_x = centroids[2 * D + 0];
    const float prior2_y = centroids[2 * D + 1];

    float centroids_new[K * D] = {0};
    uint32_t sizes[K] = {0};
    uint32_t assigns[N] = {0};

    bolt::kmeans_assign_f32_l2(slab, D, N, /*cluster_stride=*/N,
                               centroids, K, assigns);
    // No assignment hits cluster 2.
    for (size_t i = 0; i < N; ++i) EXPECT_NE(assigns[i], 2u);

    bolt::kmeans_update_centroids_f32(slab, D, N, /*cluster_stride=*/N,
                                      assigns, K,
                                      centroids_new, sizes);

    EXPECT_EQ(sizes[2], 0u);
    // Kernel must not produce NaN/Inf for empty clusters — they stay at
    // the zero-init value the kernel writes during its zero pass.
    EXPECT_FALSE(std::isnan(centroids_new[2 * D + 0]));
    EXPECT_FALSE(std::isnan(centroids_new[2 * D + 1]));
    EXPECT_FALSE(std::isinf(centroids_new[2 * D + 0]));
    EXPECT_FALSE(std::isinf(centroids_new[2 * D + 1]));
    EXPECT_FLOAT_EQ(centroids_new[2 * D + 0], 0.0f);
    EXPECT_FLOAT_EQ(centroids_new[2 * D + 1], 0.0f);

    // Caller-side "keep prior centroid" policy must restore the row.
    if (sizes[2] == 0u) {
        centroids_new[2 * D + 0] = prior2_x;
        centroids_new[2 * D + 1] = prior2_y;
    }
    EXPECT_FLOAT_EQ(centroids_new[2 * D + 0], 100.0f);
    EXPECT_FLOAT_EQ(centroids_new[2 * D + 1], 100.0f);

    // Non-empty clusters got the correct mean.
    EXPECT_EQ(sizes[0], 2u);
    EXPECT_EQ(sizes[1], 2u);
    EXPECT_FLOAT_EQ(centroids_new[0 * D + 0], 0.05f);
    EXPECT_FLOAT_EQ(centroids_new[0 * D + 1], 0.0f);
    EXPECT_FLOAT_EQ(centroids_new[1 * D + 0], 9.95f);
    EXPECT_FLOAT_EQ(centroids_new[1 * D + 1], 10.05f);
}

// ---------------------------------------------------------------------------
// 4. cluster_stride > n_vectors — the slab may be over-allocated. The
//    kernel must read using stride, not n_vectors.
// ---------------------------------------------------------------------------
TEST(BoltKmeans, OverAllocatedSlabRespectsStride) {
    constexpr size_t D = 2;
    constexpr size_t N = 4;
    constexpr size_t STRIDE = 8;  // slab is over-allocated.
    constexpr size_t K = 2;

    const float vecs_row[N * D] = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        10.0f, 10.0f,
        10.0f, 11.0f,
    };

    float slab[D * STRIDE] = {0};
    // Pack into stride-N slots.
    for (size_t i = 0; i < N; ++i) {
        for (size_t d = 0; d < D; ++d) {
            slab[d * STRIDE + i] = vecs_row[i * D + d];
        }
    }
    // Garbage in the unused tail to make sure the kernel ignores it.
    for (size_t i = N; i < STRIDE; ++i) {
        for (size_t d = 0; d < D; ++d) {
            slab[d * STRIDE + i] = 999999.0f;
        }
    }

    float centroids[K * D] = {0.0f, 0.5f, 10.0f, 10.5f};
    float centroids_new[K * D] = {0};
    uint32_t sizes[K] = {0};
    uint32_t assigns[N] = {0};

    bolt::kmeans_assign_f32_l2(slab, D, N, STRIDE, centroids, K, assigns);
    EXPECT_EQ(assigns[0], 0u);
    EXPECT_EQ(assigns[3], 1u);

    bolt::kmeans_update_centroids_f32(slab, D, N, STRIDE,
                                      assigns, K,
                                      centroids_new, sizes);
    EXPECT_EQ(sizes[0], 2u);
    EXPECT_EQ(sizes[1], 2u);
    EXPECT_FLOAT_EQ(centroids_new[0 * D + 0], 0.0f);
    EXPECT_FLOAT_EQ(centroids_new[1 * D + 0], 10.0f);
}

}  // namespace
