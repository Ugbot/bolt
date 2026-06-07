// bolt_argsort.h — Indirect / index-permutation sorts (argsort).
//
// These kernels return a PERMUTATION of [0, n) rather than moving the
// key data: `perm[i]` is the original row index of the element that
// belongs in sorted position `i`. The caller then materialises sorted
// columns by feeding `perm` to `bolt::branchless::gather_branchless`
// (one gather per output column). This is the canonical "sort once,
// gather many" pattern used by:
//   - the triple-store index build (sort row ids by (p, o, s) then
//     gather every column through the permutation), and
//   - SQL ORDER BY (sort a key column, gather the projected payload).
//
// Algorithm
// ---------
//   * Base case (n <= kInsertionCap): a stable insertion sort over the
//     index array. Branch-reduced inner shift; O(n²) but n is tiny so
//     it wins on cache + no recursion.
//   * General case: a STABLE bottom-up (iterative) merge sort over the
//     index array, using a caller-provided int32 scratch buffer of n
//     entries. Bottom-up merge is allocation-free given scratch, has no
//     recursion (Tiger Style: bounded, no unbounded stack), and is
//     naturally stable — ties keep their original index order.
//   * The convenience public entries (`sort_indirect_i64`,
//     `sort_indirect_multikey_i64`) use a fixed on-stack scratch bounded
//     by kArgsortStackScratch and assert n is within it. Callers sorting
//     more rows MUST use the `*_scratch` overload with their own
//     pre-allocated scratch — misuse fails loudly rather than silently
//     growing the stack frame or allocating.
//
// ns/row floor (target)
// ---------------------
//   Stable merge sort is O(n log n) comparisons. On int64 keys the floor
//   we target is ~6–10 ns/row at n = 1e6 on a single core (comparison
//   sort, memory-bound on the index + key reload per compare). This is
//   the correctness-first default; an LSD-radix single-key specialisation
//   (linear, ~2 ns/row) is the documented future speed upgrade — see
//   docs/research/design-log.md "argsort — merge vs radix". Radix is left
//   out of the default until its scratch contract + key-range handling is
//   benchmarked; correctness first.
//
// Stability / determinism
// ------------------------
//   STABLE: equal keys preserve their original (ascending) index order in
//   BOTH ascending and descending output. We achieve descending NOT by
//   reversing (which would break tie order) but by inverting the key
//   comparison while keeping the "<= takes the left run first" merge rule,
//   so equal keys still emit lowest-original-index first. This makes the
//   triple index and ORDER BY byte-reproducible across runs and machines.
//
// Scratch / no-heap
// -----------------
//   The kernel NEVER calls malloc/new. The merge path requires an int32
//   scratch buffer of exactly n entries; the caller owns it (pass it to
//   the *_scratch overloads). The convenience overloads use a fixed
//   on-stack scratch bounded by kStackScratch and assert n is within it.
//
// Tiger Style:
//   * noexcept; no exceptions, no RTTI, no allocation.
//   * BOLT_RESTRICT on every pointer parameter.
//   * ≥2 assertions per function; functions ≤70 lines.
//   * explicit integer widths (int32_t perm, int64_t n, int64_t keys).

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace kernels {

// Below this length the stable insertion sort wins (no scratch, cache
// friendly). At/above it the bottom-up merge sort takes over.
inline constexpr int64_t kArgsortInsertionCap = 32;

// Convenience single-key entry on-stack scratch bound. Callers sorting
// more rows than this MUST use sort_indirect_i64_scratch with their own
// pre-allocated scratch (Tiger Style: no hidden large stack frames).
inline constexpr int64_t kArgsortStackScratch = 4096;

// ---------------------------------------------------------------------------
// Stable insertion sort of an index array by a single int64 key.
//
// `ascending` flips the compare. Stability: we only shift while the prior
// key is STRICTLY greater (asc) / STRICTLY less (desc) than the inserted
// key — never on equality — so equal keys keep their original index order.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE void argsort_insertion_i64(int32_t* BOLT_RESTRICT perm,
                                             const int64_t* BOLT_RESTRICT keys,
                                             int64_t n, bool ascending) noexcept {
    assert(perm != nullptr || n == 0);
    assert(keys != nullptr || n == 0);
    assert(n >= 0);
    for (int64_t i = 1; i < n; ++i) {
        const int32_t pi = perm[i];
        const int64_t ki = keys[pi];
        int64_t j = i;
        // Strict compare → stable (equal keys do not shift past pi).
        while (j > 0) {
            const int64_t kp = keys[perm[j - 1]];
            const bool gt = ascending ? (kp > ki) : (kp < ki);
            if (!gt) break;
            perm[j] = perm[j - 1];
            --j;
        }
        perm[j] = pi;
    }
}

// ---------------------------------------------------------------------------
// Stable merge of two adjacent runs [lo,mid) and [mid,hi) of `src` into
// `dst`, keyed by `keys`. On a tie the LEFT run's element is taken first
// (its original index is smaller), preserving stability.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE void argsort_merge_run_i64(const int32_t* BOLT_RESTRICT src,
                                             int32_t* BOLT_RESTRICT dst,
                                             const int64_t* BOLT_RESTRICT keys,
                                             int64_t lo, int64_t mid, int64_t hi,
                                             bool ascending) noexcept {
    assert(lo <= mid && mid <= hi);
    assert(src != nullptr && dst != nullptr);
    int64_t i = lo, j = mid, k = lo;
    while (i < mid && j < hi) {
        const int64_t ka = keys[src[i]];
        const int64_t kb = keys[src[j]];
        // take_left when left <= right (asc) / left >= right (desc).
        // Ties (ka == kb) take left → stable.
        const bool take_left = ascending ? (ka <= kb) : (ka >= kb);
        dst[k++] = take_left ? src[i] : src[j];
        i += static_cast<int64_t>(take_left);
        j += static_cast<int64_t>(!take_left);
    }
    while (i < mid) dst[k++] = src[i++];
    while (j < hi) dst[k++] = src[j++];
}

// ---------------------------------------------------------------------------
// Stable bottom-up merge sort of an index array by a single int64 key.
// `scratch` is a caller-owned int32 buffer of n entries. No allocation.
//
// `perm` must already be initialised to identity [0, n) by the caller's
// public entry. After return `perm` holds the sorted permutation.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE void argsort_mergesort_i64(int32_t* BOLT_RESTRICT perm,
                                             int32_t* BOLT_RESTRICT scratch,
                                             const int64_t* BOLT_RESTRICT keys,
                                             int64_t n, bool ascending) noexcept {
    assert(perm != nullptr && keys != nullptr);
    assert(scratch != nullptr || n == 0);
    int32_t* a = perm;
    int32_t* b = scratch;
    // Bounded outer loop: width doubles each pass, <= ceil(log2 n) passes.
    for (int64_t width = 1; width < n; width <<= 1) {
        for (int64_t lo = 0; lo < n; lo += (width << 1)) {
            const int64_t mid = (lo + width < n) ? (lo + width) : n;
            const int64_t hi = (lo + (width << 1) < n) ? (lo + (width << 1)) : n;
            argsort_merge_run_i64(a, b, keys, lo, mid, hi, ascending);
        }
        int32_t* tmp = a; a = b; b = tmp;  // ping-pong
    }
    // If the result landed in scratch, copy it back into perm.
    if (a != perm) {
        for (int64_t i = 0; i < n; ++i) perm[i] = a[i];
    }
}

// ===========================================================================
// Public API — single key
// ===========================================================================

// Stable argsort with a caller-provided int32 scratch buffer of n entries.
// out: perm[0..n) is a permutation of [0,n) such that keys[perm[i]] is
// ascending (ascending=true) or descending (ascending=false). Ties break
// by original index (stable). No heap allocation.
BOLT_FORCE_INLINE void sort_indirect_i64_scratch(
        int32_t* BOLT_RESTRICT perm, const int64_t* BOLT_RESTRICT keys,
        int32_t* BOLT_RESTRICT scratch, int64_t n, bool ascending) noexcept {
    assert(n >= 0);
    assert(perm != nullptr || n == 0);
    assert(keys != nullptr || n == 0);
    for (int64_t i = 0; i < n; ++i) perm[i] = static_cast<int32_t>(i);
    if (n <= kArgsortInsertionCap) {
        argsort_insertion_i64(perm, keys, n, ascending);
        return;
    }
    assert(scratch != nullptr);
    argsort_mergesort_i64(perm, scratch, keys, n, ascending);
}

// Convenience entry: uses a fixed on-stack scratch. Asserts
// n <= kArgsortStackScratch so large inputs route through the _scratch
// overload instead of silently growing the stack frame.
BOLT_FORCE_INLINE void sort_indirect_i64(int32_t* BOLT_RESTRICT perm,
                                         const int64_t* BOLT_RESTRICT keys,
                                         int64_t n, bool ascending) noexcept {
    assert(n >= 0);
    assert(n <= kArgsortStackScratch);
    int32_t scratch[kArgsortStackScratch];
    sort_indirect_i64_scratch(perm, keys, scratch, n, ascending);
}

// ===========================================================================
// Public API — lexicographic multi-key
// ===========================================================================

// Max key columns for a single lexicographic sort. Bounded (Tiger Style):
// triple store needs 3 (p, o, s); 8 covers any practical composite key.
inline constexpr int64_t kArgsortMaxKeys = 8;

// Lexicographic compare of rows `ra` vs `rb` across n_keys columns.
// Returns true iff row ra should sort BEFORE row rb (strict, stable when
// false-on-equal: caller's merge uses <= on this via the tie rule below).
// Direction is per-key via asc[k]. Bounded loop over n_keys (<= 8).
BOLT_FORCE_INLINE bool argsort_row_lt_multikey(
        const int64_t* const* BOLT_RESTRICT keys, int64_t n_keys,
        const bool* BOLT_RESTRICT asc, int32_t ra, int32_t rb) noexcept {
    assert(n_keys >= 1 && n_keys <= kArgsortMaxKeys);
    assert(keys != nullptr && asc != nullptr);
    for (int64_t k = 0; k < n_keys; ++k) {
        const int64_t va = keys[k][ra];
        const int64_t vb = keys[k][rb];
        if (va != vb) {
            const bool lt = (va < vb);
            return asc[k] ? lt : !lt;
        }
    }
    return false;  // fully equal → not "before" (stability handled by merge)
}

// Stable insertion sort of an index array under the lexicographic order.
BOLT_FORCE_INLINE void argsort_insertion_multikey(
        int32_t* BOLT_RESTRICT perm, const int64_t* const* BOLT_RESTRICT keys,
        int64_t n_keys, int64_t n, const bool* BOLT_RESTRICT asc) noexcept {
    assert(perm != nullptr || n == 0);
    assert(n_keys >= 1 && n_keys <= kArgsortMaxKeys);
    for (int64_t i = 1; i < n; ++i) {
        const int32_t pi = perm[i];
        int64_t j = i;
        // Shift while prior row is strictly AFTER pi (i.e. pi < prior) →
        // stable: never moves past an equal row.
        while (j > 0 && argsort_row_lt_multikey(keys, n_keys, asc, pi,
                                                perm[j - 1])) {
            perm[j] = perm[j - 1];
            --j;
        }
        perm[j] = pi;
    }
}

// Stable merge of two runs under the lexicographic order. Tie (neither row
// strictly before the other) takes the LEFT run first → stable.
BOLT_FORCE_INLINE void argsort_merge_run_multikey(
        const int32_t* BOLT_RESTRICT src, int32_t* BOLT_RESTRICT dst,
        const int64_t* const* BOLT_RESTRICT keys, int64_t n_keys,
        const bool* BOLT_RESTRICT asc, int64_t lo, int64_t mid,
        int64_t hi) noexcept {
    assert(lo <= mid && mid <= hi);
    assert(src != nullptr && dst != nullptr);
    int64_t i = lo, j = mid, k = lo;
    while (i < mid && j < hi) {
        // take_left unless the right row is STRICTLY before the left row.
        const bool right_first =
                argsort_row_lt_multikey(keys, n_keys, asc, src[j], src[i]);
        const bool take_left = !right_first;
        dst[k++] = take_left ? src[i] : src[j];
        i += static_cast<int64_t>(take_left);
        j += static_cast<int64_t>(!take_left);
    }
    while (i < mid) dst[k++] = src[i++];
    while (j < hi) dst[k++] = src[j++];
}

// Stable bottom-up merge sort under the lexicographic order. `scratch` is
// a caller-owned int32 buffer of n entries. `perm` must be identity on
// entry. No allocation. Bounded passes (<= ceil(log2 n)).
BOLT_FORCE_INLINE void argsort_mergesort_multikey(
        int32_t* BOLT_RESTRICT perm, int32_t* BOLT_RESTRICT scratch,
        const int64_t* const* BOLT_RESTRICT keys, int64_t n_keys, int64_t n,
        const bool* BOLT_RESTRICT asc) noexcept {
    assert(perm != nullptr && keys != nullptr);
    assert(scratch != nullptr || n == 0);
    int32_t* a = perm;
    int32_t* b = scratch;
    for (int64_t width = 1; width < n; width <<= 1) {
        for (int64_t lo = 0; lo < n; lo += (width << 1)) {
            const int64_t mid = (lo + width < n) ? (lo + width) : n;
            const int64_t hi = (lo + (width << 1) < n) ? (lo + (width << 1)) : n;
            argsort_merge_run_multikey(a, b, keys, n_keys, asc, lo, mid, hi);
        }
        int32_t* tmp = a; a = b; b = tmp;
    }
    if (a != perm) {
        for (int64_t i = 0; i < n; ++i) perm[i] = a[i];
    }
}

// Lexicographic multi-key argsort with caller-provided scratch (n int32).
// perm sorts rows by (keys[0][r], keys[1][r], ...) total order; asc[k]
// gives per-key direction. Ties (all keys equal) break by original index
// (stable). No heap allocation.
BOLT_FORCE_INLINE void sort_indirect_multikey_i64_scratch(
        int32_t* BOLT_RESTRICT perm, const int64_t* const* BOLT_RESTRICT keys,
        int32_t* BOLT_RESTRICT scratch, int64_t n_keys, int64_t n,
        const bool* BOLT_RESTRICT asc) noexcept {
    assert(n >= 0);
    assert(n_keys >= 1 && n_keys <= kArgsortMaxKeys);
    assert((keys != nullptr && asc != nullptr) || n == 0);
    for (int64_t i = 0; i < n; ++i) perm[i] = static_cast<int32_t>(i);
    if (n <= kArgsortInsertionCap) {
        argsort_insertion_multikey(perm, keys, n_keys, n, asc);
        return;
    }
    assert(scratch != nullptr);
    argsort_mergesort_multikey(perm, scratch, keys, n_keys, n, asc);
}

// Convenience multi-key entry using fixed on-stack scratch. Asserts
// n <= kArgsortStackScratch; larger inputs use the _scratch overload.
BOLT_FORCE_INLINE void sort_indirect_multikey_i64(
        int32_t* BOLT_RESTRICT perm, const int64_t* const* BOLT_RESTRICT keys,
        int64_t n_keys, int64_t n, const bool* BOLT_RESTRICT asc) noexcept {
    assert(n >= 0 && n <= kArgsortStackScratch);
    assert(n_keys >= 1 && n_keys <= kArgsortMaxKeys);
    int32_t scratch[kArgsortStackScratch];
    sort_indirect_multikey_i64_scratch(perm, keys, scratch, n_keys, n, asc);
}

// ===========================================================================
// Public API — explicit-comparator permutation merge sort (uint32)
// ===========================================================================
//
// A STABLE BOTTOM-UP MERGE SORT of a uint32 permutation by a caller
// comparator `less(a,b) -> bool` (true iff row a sorts strictly before b).
// This is NOT std::sort: the algorithm is FIXED and KNOWN — merge sort,
// O(n log n), STABLE (ties keep original index order), NO recursion (bounded
// passes, <= ceil(log2 n)), allocation-free given caller scratch. Insertion
// base case below kArgsortInsertionCap. `perm` must be caller-initialised to
// identity [0,n); `scratch` is a caller-owned uint32 buffer of n entries. The
// Compare functor is a template param so it INLINES (no indirect call) — same
// codegen quality as std::sort's lambda, but with an algorithm we control.
// For pure-int64 keys prefer sort_indirect_i64 (and the future radix variant).

// Stable insertion sort of a uint32 index array under an arbitrary strict
// comparator. We shift while the PRIOR element sorts strictly AFTER pi (i.e.
// less(pi, prior)); we never shift on equality (neither less), so equal keys
// keep their original index order → stable.
template <typename Compare>
BOLT_FORCE_INLINE void merge_sort_insertion_u32(
        uint32_t* BOLT_RESTRICT perm, int64_t n, Compare less) noexcept {
    assert(perm != nullptr || n == 0);
    assert(n >= 0);
    for (int64_t i = 1; i < n; ++i) {
        const uint32_t pi = perm[i];
        int64_t j = i;
        while (j > 0 && less(pi, perm[j - 1])) {
            perm[j] = perm[j - 1];
            --j;
        }
        perm[j] = pi;
    }
}

// Stable merge of two adjacent runs [lo,mid) and [mid,hi) of `src` into `dst`
// under `less`. Take the LEFT run's element unless the RIGHT element sorts
// strictly before it (less(right,left)); ties take left → stable.
template <typename Compare>
BOLT_FORCE_INLINE void merge_sort_run_u32(
        const uint32_t* BOLT_RESTRICT src, uint32_t* BOLT_RESTRICT dst,
        int64_t lo, int64_t mid, int64_t hi, Compare less) noexcept {
    assert(lo <= mid && mid <= hi);
    assert(src != nullptr && dst != nullptr);
    int64_t i = lo, j = mid, k = lo;
    while (i < mid && j < hi) {
        const bool take_left = !less(src[j], src[i]);
        dst[k++] = take_left ? src[i] : src[j];
        i += static_cast<int64_t>(take_left);
        j += static_cast<int64_t>(!take_left);
    }
    while (i < mid) dst[k++] = src[i++];
    while (j < hi)  dst[k++] = src[j++];
}

// Explicit STABLE BOTTOM-UP MERGE SORT of a uint32 permutation by `less`.
// `perm` is caller-initialised to identity [0,n); `scratch` is a caller-owned
// uint32 buffer of n entries (only required when n > kArgsortInsertionCap).
// No allocation; bounded passes; stable. See header comment above.
template <typename Compare>
BOLT_FORCE_INLINE void merge_sort_indirect_u32(
        uint32_t* BOLT_RESTRICT perm, uint32_t* BOLT_RESTRICT scratch,
        int64_t n, Compare less) noexcept {
    assert(perm != nullptr || n == 0);
    assert(scratch != nullptr || n <= kArgsortInsertionCap);
    if (n <= kArgsortInsertionCap) {
        merge_sort_insertion_u32(perm, n, less);
        return;
    }
    uint32_t* a = perm;
    uint32_t* b = scratch;
    // Bounded outer loop: width doubles each pass, <= ceil(log2 n) passes.
    for (int64_t width = 1; width < n; width <<= 1) {
        for (int64_t lo = 0; lo < n; lo += (width << 1)) {
            const int64_t mid = (lo + width < n) ? (lo + width) : n;
            const int64_t hi = (lo + (width << 1) < n) ? (lo + (width << 1)) : n;
            merge_sort_run_u32(a, b, lo, mid, hi, less);
        }
        uint32_t* tmp = a; a = b; b = tmp;  // ping-pong
    }
    // If the result landed in scratch, copy it back into perm.
    if (a != perm) {
        for (int64_t i = 0; i < n; ++i) perm[i] = a[i];
    }
}

}  // namespace kernels
}  // namespace bolt
