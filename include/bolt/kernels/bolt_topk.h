#pragma once
#include "bolt/bolt_port.h"
#include <cstdint>
#include <cstddef>
#include <cassert>
#if defined(__AVX2__) || defined(BOLT_SIMD_AVX2)
#include <immintrin.h>
#endif

namespace bolt {

// argmin_f32 — branchless cmov scalar; AVX2 8-lane vector min + blend.
// NaN never wins (matches `<` semantics). On ties, returns first index.
BOLT_FORCE_INLINE size_t argmin_f32(
    const float* BOLT_RESTRICT in, size_t n, float* BOLT_RESTRICT out_min
) noexcept {
    assert(in != nullptr || n == 0);
    assert(out_min != nullptr);
    if (n == 0) { *out_min = 0.0f; return 0; }
    float mn = in[0]; size_t mi = 0;
    for (size_t i = 1; i < n; ++i) {
        const bool lt = in[i] < mn;
        mn = lt ? in[i] : mn;
        mi = lt ? i     : mi;
    }
    *out_min = mn;
    return mi;
}

BOLT_FORCE_INLINE size_t argmin_i32(
    const int32_t* BOLT_RESTRICT in, size_t n, int32_t* BOLT_RESTRICT out_min
) noexcept {
    assert(in != nullptr || n == 0);
    assert(out_min != nullptr);
    if (n == 0) { *out_min = 0; return 0; }
    int32_t mn = in[0]; size_t mi = 0;
    for (size_t i = 1; i < n; ++i) {
        const bool lt = in[i] < mn;
        mn = lt ? in[i] : mn;
        mi = lt ? i     : mi;
    }
    *out_min = mn;
    return mi;
}

// topk_update_f32 — max-heap of K (heap[0] is max). Replace + sift-down.
// Caller initialises heap to +INF dist, 0 pay before first call.
template <size_t K>
BOLT_FORCE_INLINE void topk_update_f32(
    float* BOLT_RESTRICT heap_dist, uint64_t* BOLT_RESTRICT heap_pay,
    float candidate_dist, uint64_t candidate_pay
) noexcept {
    static_assert(K >= 1 && K <= 4096, "K out of range");
    assert(heap_dist != nullptr);
    assert(heap_pay != nullptr);
    if (!(candidate_dist < heap_dist[0])) return;
    heap_dist[0] = candidate_dist;
    heap_pay[0]  = candidate_pay;
    // Sift down from root.
    size_t i = 0;
    for (;;) {
        const size_t l = 2*i + 1;
        const size_t r = 2*i + 2;
        size_t big = i;
        if (l < K && heap_dist[l] > heap_dist[big]) big = l;
        if (r < K && heap_dist[r] > heap_dist[big]) big = r;
        if (big == i) break;
        const float  td = heap_dist[i]; heap_dist[i] = heap_dist[big]; heap_dist[big] = td;
        const uint64_t tp = heap_pay[i]; heap_pay[i] = heap_pay[big]; heap_pay[big] = tp;
        i = big;
    }
}

}  // namespace bolt
