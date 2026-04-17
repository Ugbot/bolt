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

}  // namespace kernels
}  // namespace bolt
