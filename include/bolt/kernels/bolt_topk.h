#pragma once
#include "bolt/bolt_port.h"
#include <cstdint>
#include <cstddef>
#include <cassert>
// bolt_port.h #defines BOLT_SIMD_AVX2 to 0 on non-AVX2 hosts, so the
// previous guard `defined(BOLT_SIMD_AVX2)` was always true and pulled
// <immintrin.h> in on arm64 / NEON-only builds. Use the value, not the
// definition, to match every other consumer.
#if defined(__AVX2__) || BOLT_SIMD_AVX2
#include <immintrin.h>
#endif

namespace bolt {

// argmin_{f32,i32} — branchless cmov scalar with 4 INDEPENDENT min lanes to
// break the loop-carried dependency on (mn, mi). NaN never wins (matches `<`
// semantics: a NaN candidate fails the strict `<` test). On ties, returns the
// FIRST (lowest) index.
//
// Tie-break proof for the 4-lane split:
//   * Within a lane we update only on a STRICT `in[i] < mn`, so the earliest
//     index holding the lane-minimum is kept (a later equal value does not
//     replace it) — exactly the serial rule.
//   * The cross-lane combine compares lanes pairwise and, on EQUAL values,
//     keeps the candidate with the LOWER stored index (`val<best ||
//     (val==best && idx<bi)`). So the global winner on ties is the lowest
//     index across all lanes. This reproduces the serial scan's "first index
//     wins" contract bit-for-bit.
//   * Per-lane min/index value selection uses the identical `<` comparison as
//     the serial loop, so the chosen value (including +/-0.0 bit patterns and
//     NaN-never-wins behaviour) is unchanged.

namespace detail {

// Reduce one (val,idx) candidate into the running best, first-index-wins.
template <typename V>
BOLT_FORCE_INLINE void argmin_reduce(
    V val, size_t idx, V& best, size_t& bi
) noexcept {
    const bool take = (val < best) || (val == best && idx < bi);
    best = take ? val : best;
    bi   = take ? idx : bi;
}

}  // namespace detail

BOLT_FORCE_INLINE size_t argmin_f32(
    const float* BOLT_RESTRICT in, size_t n, float* BOLT_RESTRICT out_min
) noexcept {
    assert(in != nullptr || n == 0);
    assert(out_min != nullptr);
    if (n == 0) { *out_min = 0.0f; return 0; }
    // Lane L scans indices L, L+4, L+8, ... Each lane keeps its own (min,idx)
    // with strict-`<` (first-index-wins inside the lane).
    float m0 = in[0]; size_t i0 = 0;
    float m1 = in[0]; size_t i1 = 0;
    float m2 = in[0]; size_t i2 = 0;
    float m3 = in[0]; size_t i3 = 0;
    size_t i = 1;
    for (; i + 4 <= n; i += 4) {
        bool l0 = in[i + 0] < m0; m0 = l0 ? in[i + 0] : m0; i0 = l0 ? i + 0 : i0;
        bool l1 = in[i + 1] < m1; m1 = l1 ? in[i + 1] : m1; i1 = l1 ? i + 1 : i1;
        bool l2 = in[i + 2] < m2; m2 = l2 ? in[i + 2] : m2; i2 = l2 ? i + 2 : i2;
        bool l3 = in[i + 3] < m3; m3 = l3 ? in[i + 3] : m3; i3 = l3 ? i + 3 : i3;
    }
    for (; i < n; ++i) {
        bool l0 = in[i] < m0; m0 = l0 ? in[i] : m0; i0 = l0 ? i : i0;
    }
    // Combine lanes left-to-right, lowest index wins on equal values.
    float best = m0; size_t bi = i0;
    detail::argmin_reduce<float>(m1, i1, best, bi);
    detail::argmin_reduce<float>(m2, i2, best, bi);
    detail::argmin_reduce<float>(m3, i3, best, bi);
    assert(bi < n);
    *out_min = best;
    return bi;
}

BOLT_FORCE_INLINE size_t argmin_i32(
    const int32_t* BOLT_RESTRICT in, size_t n, int32_t* BOLT_RESTRICT out_min
) noexcept {
    assert(in != nullptr || n == 0);
    assert(out_min != nullptr);
    if (n == 0) { *out_min = 0; return 0; }
    int32_t m0 = in[0]; size_t i0 = 0;
    int32_t m1 = in[0]; size_t i1 = 0;
    int32_t m2 = in[0]; size_t i2 = 0;
    int32_t m3 = in[0]; size_t i3 = 0;
    size_t i = 1;
    for (; i + 4 <= n; i += 4) {
        bool l0 = in[i + 0] < m0; m0 = l0 ? in[i + 0] : m0; i0 = l0 ? i + 0 : i0;
        bool l1 = in[i + 1] < m1; m1 = l1 ? in[i + 1] : m1; i1 = l1 ? i + 1 : i1;
        bool l2 = in[i + 2] < m2; m2 = l2 ? in[i + 2] : m2; i2 = l2 ? i + 2 : i2;
        bool l3 = in[i + 3] < m3; m3 = l3 ? in[i + 3] : m3; i3 = l3 ? i + 3 : i3;
    }
    for (; i < n; ++i) {
        bool l0 = in[i] < m0; m0 = l0 ? in[i] : m0; i0 = l0 ? i : i0;
    }
    int32_t best = m0; size_t bi = i0;
    detail::argmin_reduce<int32_t>(m1, i1, best, bi);
    detail::argmin_reduce<int32_t>(m2, i2, best, bi);
    detail::argmin_reduce<int32_t>(m3, i3, best, bi);
    assert(bi < n);
    *out_min = best;
    return bi;
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
