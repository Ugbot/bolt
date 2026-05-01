// bolt_binsearch.h — Branchless binary search over sorted flat arrays.
//
// Three-flavor API for int64_t / uint64_t / double; exact-match plus
// lower_bound / upper_bound. The hot inner loop is the classic
// "pivot-via-cmov" shape from Paul-Virak Khuong / Pat Morin:
//
//   size_t half = n / 2;
//   base += (key > base[half]) * (n - half);
//   n    = half;
//
// On MSVC / clang / gcc at /O2 this emits `cmp` + `cmovg` (or `csel` on
// ARM64) in the loop body — no conditional jumps, one dependency chain.
// Tail-compare at the end resolves the exact-match case without adding a
// branch to the loop.
//
// Shape rules (Tiger Style):
//   - every function ≥2 asserts, ≤70 lines
//   - all noexcept, no exceptions, no RTTI
//   - POD + free functions; pointers + length, no std::span / std::vector
//   - BOLT_RESTRICT on the data pointer so the compiler knows it doesn't
//     alias anything the caller might have stashed
//
// Hot-path consumers inside MarbleDB: SSTable block index lookup, zone
// map range probes, timestamp column range scans. Point-lookup budget
// (≤200 ns on L1-hit, per MarbleDB perf gates) assumes these lower out
// to straight-line cmov — do not introduce branches here.
//
// A templated internal helper does the real work; the public API is
// explicit non-template wrappers so C callers / FFI can name them with
// stable unmangled signatures if we ever expose them that way.
//
// Sorted-assumption: `data[0..n)` must be monotonically non-decreasing.
// Behaviour on unsorted input is undefined but will not crash (asserts
// aren't cheap enough to verify sortedness on the hot path).

#pragma once

#include "bolt/bolt_branchless.h"
#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace kernels {

// ============================================================================
// Internal templated core
// ============================================================================
//
// All three search flavors are the same loop shape; only the final
// disposition of `base` differs. Splitting them keeps each function ≤70
// lines and lets the compiler inline the type-specialised version at each
// call site.

// Branchless lower_bound. Returns the first index `i` with data[i] >= key,
// or `n` if every element is strictly less. Pure cmov loop; no branches
// in the hot path (the outer `while (m > 0)` is a counted loop — the
// branch is a predictable backward jump, not a data-dependent one).
//
// Select routed through `bolt::branchless::bselect` (Bolt convention —
// keeps the cmov shape discoverable and keeps the compile path identical
// across call sites that advance/retract cursors the same way).
// `(lo + hi) / 2` overflow is avoided structurally: `m` is the remaining
// count, monotonically shrinks, so `m >> 1` can never overflow.
template <typename T>
BOLT_FORCE_INLINE int64_t bolt_lower_bound_tmpl(
        const T* BOLT_RESTRICT data, int64_t n, T key) noexcept {
    assert(n >= 0);
    assert(n == 0 || data != nullptr);

    const T* b = data;
    int64_t  m = n;
    while (m > 0) {
        const int64_t half = m >> 1;
        // bselect → cmov on x86 / csel on ARM64 at /O2. Two dependency
        // chains, no conditional jumps inside the loop body.
        const bool go = (b[half] < key);
        b = bolt::branchless::bselect<const T*>(go, b + half + 1, b);
        m = bolt::branchless::bselect<int64_t>(go, m - half - 1, half);
    }
    assert(b >= data);
    return static_cast<int64_t>(b - data);
}

// Branchless upper_bound. First index `i` with data[i] > key, or n.
template <typename T>
BOLT_FORCE_INLINE int64_t bolt_upper_bound_tmpl(
        const T* BOLT_RESTRICT data, int64_t n, T key) noexcept {
    assert(n >= 0);
    assert(n == 0 || data != nullptr);

    const T* b = data;
    int64_t  m = n;
    while (m > 0) {
        const int64_t half = m >> 1;
        const bool    go   = !(key < b[half]);  // data[half] <= key → advance
        b = bolt::branchless::bselect<const T*>(go, b + half + 1, b);
        m = bolt::branchless::bselect<int64_t>(go, m - half - 1, half);
    }
    assert(b >= data);
    return static_cast<int64_t>(b - data);
}

// Exact-match binary search. Returns an index of *any* matching element,
// or -1 if none. Built on top of lower_bound: one extra compare after the
// loop picks up the match case without branching inside the loop.
template <typename T>
BOLT_FORCE_INLINE int64_t bolt_binsearch_tmpl(
        const T* BOLT_RESTRICT data, int64_t n, T key) noexcept {
    assert(n >= 0);
    assert(n == 0 || data != nullptr);

    const int64_t idx = bolt_lower_bound_tmpl<T>(data, n, key);
    assert(idx >= 0 && idx <= n);
    // Branchless disposition: a mismatch returns -1, a hit returns idx.
    // The final select is a single cmov.
    const bool hit = (idx < n) && !(key < data[idx]) && !(data[idx] < key);
    return bolt::branchless::bselect<int64_t>(hit, idx, int64_t{-1});
}

// ============================================================================
// Public API — explicit wrappers so C / FFI callers get stable names.
// ============================================================================

BOLT_FORCE_INLINE int64_t binsearch_i64(
        const int64_t* BOLT_RESTRICT data, int64_t n, int64_t key) noexcept {
    assert(n >= 0);
    assert(n == 0 || data != nullptr);
    return bolt_binsearch_tmpl<int64_t>(data, n, key);
}

BOLT_FORCE_INLINE int64_t binsearch_u64(
        const uint64_t* BOLT_RESTRICT data, int64_t n, uint64_t key) noexcept {
    assert(n >= 0);
    assert(n == 0 || data != nullptr);
    return bolt_binsearch_tmpl<uint64_t>(data, n, key);
}

BOLT_FORCE_INLINE int64_t binsearch_f64(
        const double* BOLT_RESTRICT data, int64_t n, double key) noexcept {
    assert(n >= 0);
    assert(n == 0 || data != nullptr);
    return bolt_binsearch_tmpl<double>(data, n, key);
}

BOLT_FORCE_INLINE int64_t binsearch_f32(
        const float* BOLT_RESTRICT data, int64_t n, float key) noexcept {
    assert(n >= 0);
    assert(n == 0 || data != nullptr);
    return bolt_binsearch_tmpl<float>(data, n, key);
}

BOLT_FORCE_INLINE int64_t lower_bound_i64(
        const int64_t* BOLT_RESTRICT data, int64_t n, int64_t key) noexcept {
    assert(n >= 0);
    assert(n == 0 || data != nullptr);
    return bolt_lower_bound_tmpl<int64_t>(data, n, key);
}

BOLT_FORCE_INLINE int64_t lower_bound_u64(
        const uint64_t* BOLT_RESTRICT data, int64_t n, uint64_t key) noexcept {
    assert(n >= 0);
    assert(n == 0 || data != nullptr);
    return bolt_lower_bound_tmpl<uint64_t>(data, n, key);
}

BOLT_FORCE_INLINE int64_t lower_bound_f64(
        const double* BOLT_RESTRICT data, int64_t n, double key) noexcept {
    assert(n >= 0);
    assert(n == 0 || data != nullptr);
    return bolt_lower_bound_tmpl<double>(data, n, key);
}

BOLT_FORCE_INLINE int64_t lower_bound_f32(
        const float* BOLT_RESTRICT data, int64_t n, float key) noexcept {
    assert(n >= 0);
    assert(n == 0 || data != nullptr);
    return bolt_lower_bound_tmpl<float>(data, n, key);
}

BOLT_FORCE_INLINE int64_t upper_bound_i64(
        const int64_t* BOLT_RESTRICT data, int64_t n, int64_t key) noexcept {
    assert(n >= 0);
    assert(n == 0 || data != nullptr);
    return bolt_upper_bound_tmpl<int64_t>(data, n, key);
}

BOLT_FORCE_INLINE int64_t upper_bound_u64(
        const uint64_t* BOLT_RESTRICT data, int64_t n, uint64_t key) noexcept {
    assert(n >= 0);
    assert(n == 0 || data != nullptr);
    return bolt_upper_bound_tmpl<uint64_t>(data, n, key);
}

BOLT_FORCE_INLINE int64_t upper_bound_f64(
        const double* BOLT_RESTRICT data, int64_t n, double key) noexcept {
    assert(n >= 0);
    assert(n == 0 || data != nullptr);
    return bolt_upper_bound_tmpl<double>(data, n, key);
}

BOLT_FORCE_INLINE int64_t upper_bound_f32(
        const float* BOLT_RESTRICT data, int64_t n, float key) noexcept {
    assert(n >= 0);
    assert(n == 0 || data != nullptr);
    return bolt_upper_bound_tmpl<float>(data, n, key);
}

// ============================================================================
// Membership tests (sorted-array `contains`)
// ============================================================================
//
// `contains_*` returns true iff `key` is present in the monotonically
// non-decreasing range `data[0..n)`. Implemented as `lower_bound` + a
// branchless tail compare so the cmov shape is preserved end-to-end. Use
// these in any hot path that wants "is this id in this allow-set?" rather
// than re-rolling a binary search at the call site.
template <typename T>
BOLT_FORCE_INLINE bool bolt_contains_tmpl(
        const T* BOLT_RESTRICT data, int64_t n, T key) noexcept {
    assert(n >= 0);
    assert(n == 0 || data != nullptr);
    const int64_t i = bolt_lower_bound_tmpl<T>(data, n, key);
    // Branchless: the AND folds the bounds check and the equality check
    // into one cmov on x86 / csel on ARM64. `i == n` is the only way the
    // first half can be false; in that case we must not deref data[i].
    const bool in_bounds = (i < n);
    const T    probe     = bolt::branchless::bselect<T>(in_bounds, data[bolt::branchless::bselect<int64_t>(in_bounds, i, 0)], key);
    return in_bounds && (probe == key);
}

BOLT_FORCE_INLINE bool contains_i64(
        const int64_t* BOLT_RESTRICT data, int64_t n, int64_t key) noexcept {
    return bolt_contains_tmpl<int64_t>(data, n, key);
}

BOLT_FORCE_INLINE bool contains_u64(
        const uint64_t* BOLT_RESTRICT data, int64_t n, uint64_t key) noexcept {
    return bolt_contains_tmpl<uint64_t>(data, n, key);
}

BOLT_FORCE_INLINE bool contains_f64(
        const double* BOLT_RESTRICT data, int64_t n, double key) noexcept {
    return bolt_contains_tmpl<double>(data, n, key);
}

BOLT_FORCE_INLINE bool contains_f32(
        const float* BOLT_RESTRICT data, int64_t n, float key) noexcept {
    return bolt_contains_tmpl<float>(data, n, key);
}

}  // namespace kernels
}  // namespace bolt
