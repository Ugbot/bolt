// test_bolt_topk.cpp — tests for argmin_f32 / argmin_i32 / topk_update_f32
// in bolt/kernels/bolt_topk.h.

#include "bolt/kernels/bolt_topk.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// argmin_f32
// ---------------------------------------------------------------------------

TEST(BoltTopK, ArgminF32_Small) {
    const float in[] = {3.0f, 1.0f, 4.0f, 1.5f, 9.0f, 2.0f, 6.0f};
    float mn = 0.0f;
    const size_t idx = bolt::argmin_f32(in, sizeof(in)/sizeof(in[0]), &mn);
    EXPECT_EQ(idx, 1u);
    EXPECT_FLOAT_EQ(mn, 1.0f);
}

TEST(BoltTopK, ArgminF32_TiesFirstWins) {
    const float in[] = {2.0f, 1.0f, 1.0f, 1.0f, 5.0f};
    float mn = 0.0f;
    const size_t idx = bolt::argmin_f32(in, 5, &mn);
    EXPECT_EQ(idx, 1u);
    EXPECT_FLOAT_EQ(mn, 1.0f);
}

TEST(BoltTopK, ArgminF32_NaNSkipped) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    // NaN at index 0 -> mn starts NaN; `<` against NaN is always false,
    // so first non-NaN candidate also fails to beat NaN. To match the
    // contract that NaN never wins, NaN must NOT be a valid leading
    // value. Test the "NaN in middle" case: it must never become the min.
    const float in[] = {3.0f, nan, 1.0f, nan, 2.0f};
    float mn = 0.0f;
    const size_t idx = bolt::argmin_f32(in, 5, &mn);
    EXPECT_EQ(idx, 2u);
    EXPECT_FLOAT_EQ(mn, 1.0f);
}

TEST(BoltTopK, ArgminF32_AllEqualReturnsZero) {
    const float in[] = {7.0f, 7.0f, 7.0f, 7.0f};
    float mn = 0.0f;
    const size_t idx = bolt::argmin_f32(in, 4, &mn);
    EXPECT_EQ(idx, 0u);
    EXPECT_FLOAT_EQ(mn, 7.0f);
}

TEST(BoltTopK, ArgminF32_Empty) {
    float mn = 99.0f;
    const size_t idx = bolt::argmin_f32(nullptr, 0, &mn);
    EXPECT_EQ(idx, 0u);
    EXPECT_FLOAT_EQ(mn, 0.0f);
}

TEST(BoltTopK, ArgminF32_Singleton) {
    const float in[] = {-3.5f};
    float mn = 0.0f;
    const size_t idx = bolt::argmin_f32(in, 1, &mn);
    EXPECT_EQ(idx, 0u);
    EXPECT_FLOAT_EQ(mn, -3.5f);
}

// ---------------------------------------------------------------------------
// argmin_i32
// ---------------------------------------------------------------------------

TEST(BoltTopK, ArgminI32_Small) {
    const int32_t in[] = {5, 3, 8, -1, 4, 0};
    int32_t mn = 0;
    const size_t idx = bolt::argmin_i32(in, 6, &mn);
    EXPECT_EQ(idx, 3u);
    EXPECT_EQ(mn, -1);
}

TEST(BoltTopK, ArgminI32_TiesFirstWins) {
    const int32_t in[] = {2, -5, 3, -5, 0, -5};
    int32_t mn = 0;
    const size_t idx = bolt::argmin_i32(in, 6, &mn);
    EXPECT_EQ(idx, 1u);
    EXPECT_EQ(mn, -5);
}

TEST(BoltTopK, ArgminI32_AllEqual) {
    const int32_t in[] = {42, 42, 42, 42, 42};
    int32_t mn = 0;
    const size_t idx = bolt::argmin_i32(in, 5, &mn);
    EXPECT_EQ(idx, 0u);
    EXPECT_EQ(mn, 42);
}

TEST(BoltTopK, ArgminI32_Empty) {
    int32_t mn = 99;
    const size_t idx = bolt::argmin_i32(nullptr, 0, &mn);
    EXPECT_EQ(idx, 0u);
    EXPECT_EQ(mn, 0);
}

TEST(BoltTopK, ArgminI32_MinSentinel) {
    const int32_t in[] = {0, 100, INT32_MIN, -50};
    int32_t mn = 0;
    const size_t idx = bolt::argmin_i32(in, 4, &mn);
    EXPECT_EQ(idx, 2u);
    EXPECT_EQ(mn, INT32_MIN);
}

// Tie-break across the 4-lane split: [3,1,1,5] -> the FIRST minimum (index 1)
// must win even though the equal value at index 2 lands in a different lane.
TEST(BoltTopK, ArgminTieAcrossLanes) {
    const int32_t i32[] = {3, 1, 1, 5};
    int32_t mn_i = 0;
    EXPECT_EQ(bolt::argmin_i32(i32, 4, &mn_i), 1u);
    EXPECT_EQ(mn_i, 1);

    const float f32[] = {3.0f, 1.0f, 1.0f, 5.0f};
    float mn_f = 0.0f;
    EXPECT_EQ(bolt::argmin_f32(f32, 4, &mn_f), 1u);
    EXPECT_FLOAT_EQ(mn_f, 1.0f);

    // Larger case where the global min appears at an index whose lane differs
    // from the lane of an equal-valued later element (exercises the unrolled
    // body + cross-lane lowest-index combine).
    const int32_t big[] = {9, 9, 9, 9, 2, 2, 9, 9, 2, 2};
    int32_t mn_b = 0;
    EXPECT_EQ(bolt::argmin_i32(big, 10, &mn_b), 4u);  // first 2 is at index 4
    EXPECT_EQ(mn_b, 2);
}

// Randomized parity: the 4-lane argmin must return the SAME (value, index) as a
// plain serial strict-`<` scan for every length class (n%4 = 0,1,2,3) and many
// duplicate values (forcing the tie-break path frequently).
TEST(BoltTopK, ArgminFourLaneMatchesSerial) {
    std::mt19937_64 rng(0xA12C0FFEEULL);
    std::uniform_int_distribution<int32_t> d(0, 7);  // tiny range -> many ties
    for (size_t n = 1; n <= 257; ++n) {
        std::vector<int32_t> v(n);
        for (auto& x : v) x = d(rng);
        // Serial reference (first index wins on ties).
        int32_t rmn = v[0]; size_t rmi = 0;
        for (size_t i = 1; i < n; ++i)
            if (v[i] < rmn) { rmn = v[i]; rmi = i; }
        int32_t mn = 0;
        const size_t mi = bolt::argmin_i32(v.data(), n, &mn);
        EXPECT_EQ(mi, rmi) << "n=" << n;
        EXPECT_EQ(mn, rmn) << "n=" << n;

        // Float lane parity (NaN-free data -> serial == 4-lane bit-identical).
        std::vector<float> vf(v.begin(), v.end());
        float rmnf = vf[0]; size_t rmif = 0;
        for (size_t i = 1; i < n; ++i)
            if (vf[i] < rmnf) { rmnf = vf[i]; rmif = i; }
        float mnf = 0.0f;
        const size_t mif = bolt::argmin_f32(vf.data(), n, &mnf);
        EXPECT_EQ(mif, rmif) << "n=" << n;
        EXPECT_FLOAT_EQ(mnf, rmnf) << "n=" << n;
    }
}

// ---------------------------------------------------------------------------
// topk_update_f32
// ---------------------------------------------------------------------------

template <size_t K>
static void init_heap(float* heap_dist, uint64_t* heap_pay) {
    const float inf = std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < K; ++i) {
        heap_dist[i] = inf;
        heap_pay[i]  = 0;
    }
}

TEST(BoltTopK, TopKUpdate_K10_Ascending) {
    constexpr size_t K = 10;
    float heap_dist[K];
    uint64_t heap_pay[K];
    init_heap<K>(heap_dist, heap_pay);

    // Insert 0,1,...,99 — heap should retain {0..9}.
    for (uint64_t i = 0; i < 100; ++i) {
        bolt::topk_update_f32<K>(heap_dist, heap_pay,
                                 static_cast<float>(i), i);
    }

    std::vector<float> got(heap_dist, heap_dist + K);
    std::sort(got.begin(), got.end());
    for (size_t i = 0; i < K; ++i) {
        EXPECT_FLOAT_EQ(got[i], static_cast<float>(i));
    }
}

TEST(BoltTopK, TopKUpdate_K32_Descending) {
    constexpr size_t K = 32;
    float heap_dist[K];
    uint64_t heap_pay[K];
    init_heap<K>(heap_dist, heap_pay);

    // Insert 99,98,...,0 — heap should retain {0..31}.
    for (int i = 99; i >= 0; --i) {
        bolt::topk_update_f32<K>(heap_dist, heap_pay,
                                 static_cast<float>(i),
                                 static_cast<uint64_t>(i));
    }

    std::vector<float> got(heap_dist, heap_dist + K);
    std::sort(got.begin(), got.end());
    for (size_t i = 0; i < K; ++i) {
        EXPECT_FLOAT_EQ(got[i], static_cast<float>(i));
    }
    // Heap[0] must be the max of the retained set.
    EXPECT_FLOAT_EQ(*std::max_element(heap_dist, heap_dist + K), 31.0f);
    EXPECT_FLOAT_EQ(heap_dist[0], 31.0f);
}

TEST(BoltTopK, TopKUpdate_K128_RandomVsNthElement) {
    constexpr size_t K = 128;
    constexpr size_t N = 1000;
    float heap_dist[K];
    uint64_t heap_pay[K];
    init_heap<K>(heap_dist, heap_pay);

    std::mt19937_64 rng(0xC0FFEEULL);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    std::vector<float> data(N);
    for (size_t i = 0; i < N; ++i) {
        data[i] = dist(rng);
        bolt::topk_update_f32<K>(heap_dist, heap_pay,
                                 data[i], static_cast<uint64_t>(i));
    }

    // Reference: lowest K via nth_element.
    std::vector<float> ref = data;
    std::nth_element(ref.begin(), ref.begin() + K, ref.end());
    std::vector<float> ref_topk(ref.begin(), ref.begin() + K);
    std::sort(ref_topk.begin(), ref_topk.end());

    std::vector<float> got(heap_dist, heap_dist + K);
    std::sort(got.begin(), got.end());

    ASSERT_EQ(got.size(), ref_topk.size());
    for (size_t i = 0; i < K; ++i) {
        EXPECT_FLOAT_EQ(got[i], ref_topk[i]) << "i=" << i;
    }
    // Heap invariant: heap[0] is max.
    EXPECT_FLOAT_EQ(heap_dist[0], ref_topk.back());
}

TEST(BoltTopK, TopKUpdate_K1_TracksMinimum) {
    constexpr size_t K = 1;
    float heap_dist[K];
    uint64_t heap_pay[K];
    init_heap<K>(heap_dist, heap_pay);

    const float vals[] = {5.0f, 3.0f, 9.0f, 1.0f, 4.0f, 1.0f};
    for (uint64_t i = 0; i < 6; ++i) {
        bolt::topk_update_f32<K>(heap_dist, heap_pay, vals[i], i);
    }
    EXPECT_FLOAT_EQ(heap_dist[0], 1.0f);
    // Tie -> first wins (later equal candidates fail `<` test).
    EXPECT_EQ(heap_pay[0], 3u);
}

TEST(BoltTopK, TopKUpdate_NoUpdateWhenNotSmaller) {
    constexpr size_t K = 4;
    float heap_dist[K];
    uint64_t heap_pay[K];
    init_heap<K>(heap_dist, heap_pay);

    // Fill with small values.
    bolt::topk_update_f32<K>(heap_dist, heap_pay, 1.0f, 1);
    bolt::topk_update_f32<K>(heap_dist, heap_pay, 2.0f, 2);
    bolt::topk_update_f32<K>(heap_dist, heap_pay, 3.0f, 3);
    bolt::topk_update_f32<K>(heap_dist, heap_pay, 4.0f, 4);

    // Snapshot.
    float before[K];
    uint64_t before_pay[K];
    for (size_t i = 0; i < K; ++i) {
        before[i] = heap_dist[i];
        before_pay[i] = heap_pay[i];
    }

    // Larger candidates must be rejected.
    bolt::topk_update_f32<K>(heap_dist, heap_pay, 100.0f, 999);
    bolt::topk_update_f32<K>(heap_dist, heap_pay, 4.0f, 888);  // tie with current max
    for (size_t i = 0; i < K; ++i) {
        EXPECT_FLOAT_EQ(heap_dist[i], before[i]);
        EXPECT_EQ(heap_pay[i], before_pay[i]);
    }
}

}  // namespace
