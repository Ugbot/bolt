// bolt/kernels/bolt_dict_filter.h — literal-resolve-once filter kernels
// for Dictionary-encoded columns (H1).
//
// QuestDB's `EqSymStrFunctionFactory.ConstSymIntCheckFunc` compiles
// `WHERE symbol = 'AAPL'` by resolving the literal to an int ID once
// at plan time, then doing `column[i] == code` on the hot path. This
// header ports the same shape to Bolt's `ColumnFormat::Dictionary`:
//
//   1. `dict_resolve_code<T>(child_values, distinct, scalar)` — walks
//      the dict_child once to find the code for `scalar`. Returns -1
//      if the literal never appears; the caller can then short-circuit
//      (filter returns 0) without touching the key buffer at all.
//   2. `filter_eq_dict_keys<KT>(keys, n, code, out)` — tight branchless
//      loop matching the existing filter-by-scalar shape, but on the
//      dict *keys* (uint8 / uint16 / uint32) instead of the logical
//      values.
//
// The caller (typically a BoltColumn filter dispatcher) ties them
// together: one dict-resolve call followed by one narrow integer scan.
// Compared to materialise-then-filter, we skip N × sizeof(value) bytes
// of scratch traffic and narrow the compare to the key width.
//
// RULES: Tiger Style — POD + free functions, no exceptions, noexcept,
// branchless hot loop, ≥2 asserts per function.

#pragma once

#include "bolt/bolt_branchless.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>
#include <cstring>

namespace bolt {
namespace kernels {

/// Resolve a typed literal `scalar` to its dictionary code by linear
/// scan of the `child_values[]` buffer. Returns the matching code, or
/// -1 if `scalar` is not in the dictionary (filter can short-circuit).
///
/// Dictionary children are typically small (<=256 entries for typical
/// tick columns — tickers, sides, exchanges) so the linear scan is
/// fine; when dict.length grows past, call sites can swap this for a
/// hashmap lookup without changing the consumer contract.
template <typename T>
inline int32_t dict_resolve_code(const T* BOLT_RESTRICT child_values,
                                 int64_t distinct_count,
                                 T scalar) noexcept {
    assert(child_values != nullptr || distinct_count == 0);
    assert(distinct_count >= 0);
    for (int64_t i = 0; i < distinct_count; ++i) {
        if (child_values[i] == scalar) return static_cast<int32_t>(i);
    }
    return -1;
}

/// Filter dict keys (narrow unsigned int) for equality against a
/// resolved code. Branchless — always writes `i`, conditionally
/// advances `count`. Same shape as `filter_eq_branchless<T>` but
/// narrowed to the key width (the whole point of dict encoding).
///
/// KT is typically `uint8_t` (dict <=256), `uint16_t`, or `uint32_t`.
template <typename KT>
inline int64_t filter_eq_dict_keys(const KT* BOLT_RESTRICT keys, int64_t n,
                                   uint32_t code,
                                   int32_t* BOLT_RESTRICT out) noexcept {
    assert(keys != nullptr || n == 0);
    assert(out  != nullptr || n == 0);
    assert(n >= 0);
    const KT code_kt = static_cast<KT>(code);
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        out[count] = static_cast<int32_t>(i);
        count += (keys[i] == code_kt);
    }
    return count;
}

// ---------------------------------------------------------------------------
// I3 — auto-dispatch `filter_eq` on Dictionary columns.
//
// Ties H1 (literal resolve) + H3 (popcount miss-accelerator) + C5
// (BitmapIndex sidecar). Decision tree:
//
//   1. Resolve the typed scalar against the dict_child (H1). If the
//      code is -1, return 0 — no row could possibly match.
//   2. If the column has a BitmapIndex with popcounts built (H3), and
//      `probably_absent(code)` is true, return 0 without scanning.
//   3. If the column has a BitmapIndex (C5), use `filter(code)` —
//      O(selectivity) in output cost, not O(length).
//   4. Otherwise fall through to `filter_eq_dict_keys` (H1) on the
//      narrow key buffer — O(length / key_width).
//
// `arena` may be null; without it, step 3 cannot auto-build a missing
// index and step 4 runs. Caller retains ownership of `out`; no arena
// allocation happens in this helper itself beyond the optional
// `ensure_bitmap_index`.
//
// Returns the number of matching rows, or -1 on unsupported shape
// (non-Dictionary column, type mismatch, or dict_child not Flat).
// ---------------------------------------------------------------------------
template <typename T>
inline int64_t filter_eq_dict_column(BoltColumn& col, T scalar,
                                     int32_t* BOLT_RESTRICT out,
                                     Arena* arena /* may be null */) noexcept {
    assert(out != nullptr || col.length == 0);
    if (col.format != ColumnFormat::Dictionary) return -1;
    if (col.dict_child == nullptr) return -1;
    if (col.dict_child->format != ColumnFormat::Flat) return -1;
    if (static_cast<size_t>(col.dict_child->type_size_bytes) != sizeof(T)) return -1;

    // Step 1 — literal resolve once.
    const T* child_values = static_cast<const T*>(col.dict_child->data);
    const int32_t code = dict_resolve_code<T>(child_values,
                                              col.dict_child->length, scalar);
    if (code < 0) return 0;

    // Step 2 + 3 — sidecar-driven dispatch.
    BitmapIndex* idx = static_cast<BitmapIndex*>(col.sidecars.bitmap_index);
    if (idx == nullptr && arena != nullptr) {
        idx = col.ensure_bitmap_index(arena);
    }
    if (idx != nullptr) {
        if (idx->probably_absent(static_cast<uint32_t>(code))) return 0;
        return idx->filter(static_cast<uint32_t>(code), out);
    }

    // Step 4 — narrow-key branchless scan.
    const uint16_t kw = col.type_size_bytes;
    const int64_t  n  = col.length;
    if (kw == 1) {
        return filter_eq_dict_keys<uint8_t>(
            static_cast<const uint8_t*>(col.data), n,
            static_cast<uint32_t>(code), out);
    } else if (kw == 2) {
        return filter_eq_dict_keys<uint16_t>(
            static_cast<const uint16_t*>(col.data), n,
            static_cast<uint32_t>(code), out);
    } else if (kw == 4) {
        return filter_eq_dict_keys<uint32_t>(
            static_cast<const uint32_t*>(col.data), n,
            static_cast<uint32_t>(code), out);
    }
    return -1;
}

}  // namespace kernels
}  // namespace bolt
