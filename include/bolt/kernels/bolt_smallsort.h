// bolt_smallsort.h — In-place sorts for fixed small caps (n ≤ 1024).
//
// The hot consumers (BM25 boolean drain top-k, scatter-gather staging,
// per-batch result reordering) operate on n in 1..50 with a hard cap at
// kMaxTopK / kMaxQueryTerms. Insertion sort wins decisively below ~32
// elements (one cmov per shift, no recursion overhead), so we ship that
// as the default for `sort_small_*`. Above ~32 elements an unrolled-merge
// or bitonic-network sort would beat insertion; we leave that as a future
// SIMD upgrade path (research note: docs/research/smallsort-simd.md once
// the consumer profile shows the n>32 path is hot).
//
// Tiger Style:
//   * noexcept; no exceptions, no allocation
//   * BOLT_RESTRICT on the data pointer
//   * ≥2 assertions, ≤70-line functions
//   * the loop is bounded by `n`, which the caller asserts ≤ a `constexpr`
//     cap (typically kMaxTopK = 1024)
//   * no comparator function pointer — fixed types and ascending order; if
//     consumers want descending, post-reverse in O(n) at the call site
//
// Why not std::sort:
//   std::sort is allowed to allocate (stack frame from quicksort recursion)
//   and uses random-access iterators that defeat MSVC's branchless
//   transforms in /O2. Insertion-sort here is dependency-only on bolt_port
//   and emits a single cmov per shift on every supported toolchain.

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace bolt {
namespace kernels {

// Insertion sort, ascending. O(n²) worst case but cache-friendly and
// branchless on the inner shift. Hard upper-bound n ≤ kSmallSortCap so
// future SIMD specialisations can dispatch on n at compile time without
// adding a runtime cap check that the caller already enforces.
inline constexpr size_t kSmallSortCap = 1024u;

BOLT_FORCE_INLINE void sort_small_u64_asc(uint64_t* BOLT_RESTRICT a,
                                           size_t n) noexcept {
    assert(a != nullptr || n == 0u);
    assert(n <= kSmallSortCap);
    for (size_t i = 1; i < n; ++i) {
        const uint64_t v = a[i];
        size_t j = i;
        // Inner shift compiled to a tight cmov-driven copy on /O2.
        while (j > 0u && a[j - 1u] > v) {
            a[j] = a[j - 1u];
            --j;
        }
        a[j] = v;
    }
}

BOLT_FORCE_INLINE void sort_small_i64_asc(int64_t* BOLT_RESTRICT a,
                                           size_t n) noexcept {
    assert(a != nullptr || n == 0u);
    assert(n <= kSmallSortCap);
    for (size_t i = 1; i < n; ++i) {
        const int64_t v = a[i];
        size_t j = i;
        while (j > 0u && a[j - 1u] > v) {
            a[j] = a[j - 1u];
            --j;
        }
        a[j] = v;
    }
}

BOLT_FORCE_INLINE void sort_small_u32_asc(uint32_t* BOLT_RESTRICT a,
                                           size_t n) noexcept {
    assert(a != nullptr || n == 0u);
    assert(n <= kSmallSortCap);
    for (size_t i = 1; i < n; ++i) {
        const uint32_t v = a[i];
        size_t j = i;
        while (j > 0u && a[j - 1u] > v) {
            a[j] = a[j - 1u];
            --j;
        }
        a[j] = v;
    }
}

}  // namespace kernels
}  // namespace bolt
