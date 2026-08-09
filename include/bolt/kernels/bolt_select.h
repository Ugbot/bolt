// bolt_select.h — branchless column select (SQL CASE WHEN / ternary).
//
// out[i] = mask[i] ? a[i] : b[i], with the mask carried as the engine's
// canonical boolean lane (int64 0/1 — the same representation cmp_* and
// logical_and/or produce, so a comparison tree feeds straight in).
//
// Why here and not in the caller: chukonu's discipline is that every numeric
// loop is a Bolt kernel, so the vectorized CASE arm has exactly one
// implementation to optimize, audit, and benchmark. Written branchless (an
// index-free arithmetic blend for the 8-byte lanes) so a CASE cannot
// reintroduce a data-dependent branch into an otherwise branchless pipeline.
//
// Zero-dependency, noexcept, no allocation. Aliasing: `out` may alias `a` or
// `b` (each element is read before it is written at the same index).
#ifndef BOLT_KERNELS_BOLT_SELECT_H
#define BOLT_KERNELS_BOLT_SELECT_H

#include <cassert>
#include <cstdint>

#include "bolt/bolt_port.h"
#include "bolt/kernels/bolt_decimal.h"

namespace bolt::kernels {

// Truthiness of the canonical boolean lane. Any non-zero is true, matching
// the scalar evaluator's v_truthy so the two lanes cannot disagree.
BOLT_FORCE_INLINE int64_t select_mask_bit(int64_t m) noexcept {
    return static_cast<int64_t>(m != 0);
}

BOLT_FORCE_INLINE void select_i64(const int64_t* BOLT_RESTRICT mask,
                                  const int64_t* BOLT_RESTRICT a,
                                  const int64_t* BOLT_RESTRICT b,
                                  int64_t n,
                                  int64_t* BOLT_RESTRICT out) noexcept {
    assert((mask != nullptr && a != nullptr && b != nullptr && out != nullptr) ||
           n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) {
        // -1 (all ones) when true, 0 when false -> branchless blend.
        const int64_t t = -select_mask_bit(mask[i]);
        out[i] = (a[i] & t) | (b[i] & ~t);
    }
}

BOLT_FORCE_INLINE void select_f64(const int64_t* BOLT_RESTRICT mask,
                                  const double* BOLT_RESTRICT a,
                                  const double* BOLT_RESTRICT b,
                                  int64_t n,
                                  double* BOLT_RESTRICT out) noexcept {
    assert((mask != nullptr && a != nullptr && b != nullptr && out != nullptr) ||
           n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) {
        // Value-level select: a bit blend would have to type-pun, and the
        // compiler lowers this to a vector select anyway.
        out[i] = (mask[i] != 0) ? a[i] : b[i];
    }
}

BOLT_FORCE_INLINE void select_i32(const int64_t* BOLT_RESTRICT mask,
                                  const int32_t* BOLT_RESTRICT a,
                                  const int32_t* BOLT_RESTRICT b,
                                  int64_t n,
                                  int32_t* BOLT_RESTRICT out) noexcept {
    assert((mask != nullptr && a != nullptr && b != nullptr && out != nullptr) ||
           n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) {
        const int32_t t = -static_cast<int32_t>(select_mask_bit(mask[i]));
        out[i] = (a[i] & t) | (b[i] & ~t);
    }
}

// Decimal128 select. Both inputs must already be at the SAME scale — the
// caller owns scale alignment (d128_rescale), exactly as d128_add_col's
// callers do; a select cannot invent a common scale without changing values.
BOLT_FORCE_INLINE void select_d128(const int64_t* BOLT_RESTRICT mask,
                                   const decimal::Decimal128* BOLT_RESTRICT a,
                                   const decimal::Decimal128* BOLT_RESTRICT b,
                                   int64_t n,
                                   decimal::Decimal128* BOLT_RESTRICT out)
    noexcept {
    assert((mask != nullptr && a != nullptr && b != nullptr && out != nullptr) ||
           n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) {
        out[i] = (mask[i] != 0) ? a[i] : b[i];
    }
}

// Rescale a whole Decimal128 column from `from_scale` to `to_scale`
// (to_scale >= from_scale for the widening direction the select needs).
BOLT_FORCE_INLINE void d128_rescale_col(const decimal::Decimal128* BOLT_RESTRICT a,
                                        int64_t n, int from_scale, int to_scale,
                                        decimal::Decimal128* BOLT_RESTRICT out)
    noexcept {
    assert((a != nullptr && out != nullptr) || n == 0);
    assert(n >= 0 && from_scale >= 0 && to_scale >= 0);
    for (int64_t i = 0; i < n; ++i) {
        out[i] = decimal::d128_rescale(a[i], from_scale, to_scale);
    }
}

}  // namespace bolt::kernels

#endif  // BOLT_KERNELS_BOLT_SELECT_H
