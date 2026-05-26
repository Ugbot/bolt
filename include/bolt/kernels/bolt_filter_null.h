// bolt/kernels/bolt_filter_null.h — Null-bitmap-aware filter kernels.
//
// Scope: extend `bolt_branchless.h`'s comparison templates with the two
// missing pieces the predecessor papered over via an INT64_MIN sentinel:
//
//   1. SQL three-valued logic — a row with a NULL on either side never
//      passes a comparison predicate. We AND the column's Arrow-shape
//      validity bitmap into the predicate result before compress-store.
//   2. IS NULL / IS NOT NULL — pure bitmap kernels, no value compare.
//
// All entry points are header-only templates so the inner loop fully
// specialises at compile time (Tiger Style: compile-time type dispatch
// via the existing `BOLT_NUMERIC_TYPES` X-macro at the call site, not
// a runtime registry lookup).
//
// Validity bitmap shape: Arrow convention — LSB-first, bit `i` covers
// row `i`, `1` means valid, `0` means NULL. `nulls == nullptr` means
// "no nulls" (fast path, identical to `bolt_branchless.h`).
//
// Tiger Style:
//   - noexcept, ≥2 asserts/hot fn, ≤70 lines/fn, BOLT_RESTRICT pointers.
//   - zero allocation, fixed upper bounds on loops.
//   - no virtuals, no RTTI, no exceptions.
//   - scalar branchless default; matches `bolt_branchless.h` perf shape.
//
// API mirrors the card-K-FILT spec:
//
//   template <CmpOp Op, typename T>
//   size_t cmp(uint32_t* out_sel,
//              const uint32_t* in_sel, size_t in_sel_len,
//              size_t in_len,
//              const T* lhs, const T* rhs,
//              const uint64_t* nulls);
//
//   size_t cmp_is_null    (...);
//   size_t cmp_is_not_null(...);
//
// `out_sel`, `in_sel` carry 32-bit row indices (matches the rest of the
// codebase — `BoltColumn::length` is i64 but selection vectors are i32
// to keep the per-row index footprint tight).

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/bolt_branchless.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace bolt {
namespace kernels {
namespace filter {

// Re-export `bolt::branchless::CmpOp` so callers don't have to dig into
// the branchless namespace.
using bolt::branchless::CmpOp;

// ----------------------------------------------------------------------------
// Bit test: branchless, single-row.
// ----------------------------------------------------------------------------

BOLT_FORCE_INLINE bool bitmap_get(const uint64_t* BOLT_RESTRICT nulls,
                                  std::size_t row) noexcept {
    // `nulls == nullptr` ⇒ "all valid"; caller is responsible for the
    // nullptr fast path (we never deref here).
    assert(nulls != nullptr);
    return (nulls[row >> 6] >> (row & 63u)) & 1ull;
}

// ----------------------------------------------------------------------------
// Per-op evaluator — compile-time dispatch on CmpOp.
// ----------------------------------------------------------------------------

template <CmpOp Op, typename T>
BOLT_FORCE_INLINE bool cmp_one(T a, T b) noexcept {
    if constexpr (Op == CmpOp::Eq) return a == b;
    else if constexpr (Op == CmpOp::Ne) return a != b;
    else if constexpr (Op == CmpOp::Lt) return a <  b;
    else if constexpr (Op == CmpOp::Le) return a <= b;
    else if constexpr (Op == CmpOp::Gt) return a >  b;
    else /* Ge */                       return a >= b;
}

// ----------------------------------------------------------------------------
// Main entry point: cmp<Op,T>(out_sel, in_sel, in_sel_len, in_len, lhs, rhs,
//                              nulls).
//
// Four input shapes collapse into one body:
//
//    in_sel == nullptr , rhs == nullptr   ⇒ dense  vs scalar(*lhs source)
//    in_sel == nullptr , rhs != nullptr   ⇒ dense  col-col
//    in_sel != nullptr , rhs == nullptr   ⇒ sparse vs scalar (uncommon)
//    in_sel != nullptr , rhs != nullptr   ⇒ sparse col-col
//
// For col-vs-literal callers should set `rhs` to a pointer to the literal
// repeated, OR call the convenience overload below. We prefer the symmetric
// shape so the bit-mask + compare loop is one body.
// ----------------------------------------------------------------------------

template <CmpOp Op, typename T>
inline std::size_t cmp(
        uint32_t* BOLT_RESTRICT       out_sel,
        const uint32_t* BOLT_RESTRICT in_sel,
        std::size_t                   in_sel_len,
        std::size_t                   in_len,
        const T* BOLT_RESTRICT        lhs,
        const T* BOLT_RESTRICT        rhs,
        const uint64_t* BOLT_RESTRICT nulls) noexcept {
    assert(out_sel != nullptr || (in_sel_len == 0 && in_len == 0));
    assert(lhs != nullptr     || (in_sel_len == 0 && in_len == 0));
    assert(rhs != nullptr);  // caller passes either rhs vector or scalar-broadcast
    const bool dense = (in_sel == nullptr);
    const std::size_t n = dense ? in_len : in_sel_len;
    std::size_t k = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t row = dense ? i : static_cast<std::size_t>(in_sel[i]);
        const bool match = cmp_one<Op, T>(lhs[row], rhs[row]);
        const bool valid = (nulls == nullptr) ? true : bitmap_get(nulls, row);
        out_sel[k] = static_cast<uint32_t>(row);
        k += static_cast<std::size_t>(match & valid);
    }
    return k;
}

// Convenience: column vs scalar literal. Equivalent to building a
// broadcast pointer, but the inner loop only loads from `lhs`.
template <CmpOp Op, typename T>
inline std::size_t cmp_scalar(
        uint32_t* BOLT_RESTRICT       out_sel,
        const uint32_t* BOLT_RESTRICT in_sel,
        std::size_t                   in_sel_len,
        std::size_t                   in_len,
        const T* BOLT_RESTRICT        lhs,
        T                             rhs,
        const uint64_t* BOLT_RESTRICT nulls) noexcept {
    assert(out_sel != nullptr || (in_sel_len == 0 && in_len == 0));
    assert(lhs != nullptr     || (in_sel_len == 0 && in_len == 0));
    const bool dense = (in_sel == nullptr);
    const std::size_t n = dense ? in_len : in_sel_len;
    std::size_t k = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t row = dense ? i : static_cast<std::size_t>(in_sel[i]);
        const bool match = cmp_one<Op, T>(lhs[row], rhs);
        const bool valid = (nulls == nullptr) ? true : bitmap_get(nulls, row);
        out_sel[k] = static_cast<uint32_t>(row);
        k += static_cast<std::size_t>(match & valid);
    }
    return k;
}

// ----------------------------------------------------------------------------
// IS NULL / IS NOT NULL — pure bitmap, no value compare.
//
// `nulls == nullptr` ⇒ column has no null bitmap, so:
//   IS NULL     ⇒ 0 matches.
//   IS NOT NULL ⇒ all rows match (full pass-through of the input range).
// ----------------------------------------------------------------------------

inline std::size_t cmp_is_null(
        uint32_t* BOLT_RESTRICT       out_sel,
        const uint32_t* BOLT_RESTRICT in_sel,
        std::size_t                   in_sel_len,
        std::size_t                   in_len,
        const uint64_t* BOLT_RESTRICT nulls) noexcept {
    assert(out_sel != nullptr || (in_sel_len == 0 && in_len == 0));
    if (nulls == nullptr) return 0;
    const bool dense = (in_sel == nullptr);
    const std::size_t n = dense ? in_len : in_sel_len;
    std::size_t k = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t row = dense ? i : static_cast<std::size_t>(in_sel[i]);
        const bool is_null = !bitmap_get(nulls, row);
        out_sel[k] = static_cast<uint32_t>(row);
        k += static_cast<std::size_t>(is_null);
    }
    return k;
}

inline std::size_t cmp_is_not_null(
        uint32_t* BOLT_RESTRICT       out_sel,
        const uint32_t* BOLT_RESTRICT in_sel,
        std::size_t                   in_sel_len,
        std::size_t                   in_len,
        const uint64_t* BOLT_RESTRICT nulls) noexcept {
    assert(out_sel != nullptr || (in_sel_len == 0 && in_len == 0));
    const bool dense = (in_sel == nullptr);
    const std::size_t n = dense ? in_len : in_sel_len;
    if (nulls == nullptr) {
        // All rows pass — emit the (possibly dense) identity selvec.
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t row = dense ? i
                                          : static_cast<std::size_t>(in_sel[i]);
            out_sel[i] = static_cast<uint32_t>(row);
        }
        return n;
    }
    std::size_t k = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t row = dense ? i : static_cast<std::size_t>(in_sel[i]);
        const bool valid = bitmap_get(nulls, row);
        out_sel[k] = static_cast<uint32_t>(row);
        k += static_cast<std::size_t>(valid);
    }
    return k;
}

// ----------------------------------------------------------------------------
// i32-selvec convenience overloads — the rest of the codebase historically
// carries selection vectors as `int32_t*` (BoltColumn / Morsel::sel). Until
// the migration to `uint32_t*` selvecs lands everywhere, accept i32 here so
// callers don't have to cast at every site.
// ----------------------------------------------------------------------------

template <CmpOp Op, typename T>
inline std::size_t cmp_scalar_i32(
        int32_t* BOLT_RESTRICT        out_sel,
        const int32_t* BOLT_RESTRICT  in_sel,
        std::size_t                   in_sel_len,
        std::size_t                   in_len,
        const T* BOLT_RESTRICT        lhs,
        T                             rhs,
        const uint64_t* BOLT_RESTRICT nulls) noexcept {
    return cmp_scalar<Op, T>(reinterpret_cast<uint32_t*>(out_sel),
                             reinterpret_cast<const uint32_t*>(in_sel),
                             in_sel_len, in_len, lhs, rhs, nulls);
}

inline std::size_t cmp_is_null_i32(
        int32_t* BOLT_RESTRICT        out_sel,
        const int32_t* BOLT_RESTRICT  in_sel,
        std::size_t                   in_sel_len,
        std::size_t                   in_len,
        const uint64_t* BOLT_RESTRICT nulls) noexcept {
    return cmp_is_null(reinterpret_cast<uint32_t*>(out_sel),
                       reinterpret_cast<const uint32_t*>(in_sel),
                       in_sel_len, in_len, nulls);
}

inline std::size_t cmp_is_not_null_i32(
        int32_t* BOLT_RESTRICT        out_sel,
        const int32_t* BOLT_RESTRICT  in_sel,
        std::size_t                   in_sel_len,
        std::size_t                   in_len,
        const uint64_t* BOLT_RESTRICT nulls) noexcept {
    return cmp_is_not_null(reinterpret_cast<uint32_t*>(out_sel),
                           reinterpret_cast<const uint32_t*>(in_sel),
                           in_sel_len, in_len, nulls);
}

}  // namespace filter
}  // namespace kernels
}  // namespace bolt
