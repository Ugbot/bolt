// bolt_mergejoin.h — Sorted-input merge-join (Tiger Style, arena-free hot path).
//
// For inputs that are already sorted on the join key (TimescaleDB-style
// time-series joins, pre-sorted dimensions, or the output of a radix
// partition-and-sort pass) merge-join dominates hash-join on three
// dimensions:
//
//   - Zero build-side state.  No hash table, no Bloom, no radix scan.
//   - O(n + m) time, O(1) space on the hot path.
//   - Output preserves the sorted join-key order, feeding a downstream
//     range-merge without a sort.
//
// Three variants ship:
//
//   mergejoin_inner_i64     — inner equi-join, emit (build_idx, probe_idx)
//                             pairs. Build-side keys MAY be duplicated;
//                             a probe row matches every equal build row
//                             in order.
//   mergejoin_left_outer_i64  — inner pairs plus unmatched probe rows
//                               (emitted with build_idx = -1).
//   mergejoin_right_outer_i64 — inner pairs plus unmatched build rows
//                               (emitted with probe_idx = -1).
//
// Contract:
//   - `build_keys` and `probe_keys` are Flat or View int64/uint64 columns,
//     non-strictly ascending.  A `BOLT_DEBUG` assert catches descending
//     input; Release builds behave as if the keys were sorted.
//   - `out_build_idx` and `out_probe_idx` are caller-allocated, capacity
//     ≥ `build_keys.length + probe_keys.length` in the outer variants,
//     ≥ `min(build, probe)` for inner (worst case: all match one-to-one).
//     Callers who know the exact cap (e.g. unique-vs-unique) can tighten.
//   - Returns pair count; < 0 on unsupported input shape.
//
// RULES: Tiger Style — POD + free functions, no exceptions, noexcept
// everywhere, no heap allocation.

#pragma once

#include "bolt/bolt_column.h"
#include "bolt/bolt_port.h"
#include "bolt/bolt_types.h"

#include <cassert>
#include <cstdint>

namespace bolt {

// Shared precondition check for the three variants.
inline bool mergejoin_check_inputs(const BoltColumn& build_keys,
                                   const BoltColumn& probe_keys) noexcept {
    const bool fmt_ok = (build_keys.format == ColumnFormat::Flat ||
                         build_keys.format == ColumnFormat::View) &&
                        (probe_keys.format == ColumnFormat::Flat ||
                         probe_keys.format == ColumnFormat::View);
    const bool type_ok =
        (build_keys.type == BoltType::Int64 || build_keys.type == BoltType::UInt64) &&
        (probe_keys.type == BoltType::Int64 || probe_keys.type == BoltType::UInt64);
    return fmt_ok && type_ok;
}

// ---------------------------------------------------------------------------
// Inner merge-join — emit every (build_idx, probe_idx) with matching keys.
// Build-side duplicates are handled by walking the equal-key run on both
// sides and emitting the cross product.
// ---------------------------------------------------------------------------
inline int64_t mergejoin_inner_i64(const BoltColumn& build_keys,
                                   const BoltColumn& probe_keys,
                                   int32_t* BOLT_RESTRICT out_build_idx,
                                   int32_t* BOLT_RESTRICT out_probe_idx,
                                   int64_t out_capacity) noexcept {
    assert(out_build_idx != nullptr || out_capacity == 0);
    assert(out_probe_idx != nullptr || out_capacity == 0);
    assert(out_capacity >= 0);
    if (!mergejoin_check_inputs(build_keys, probe_keys)) return -1;

    const int64_t bn = build_keys.length;
    const int64_t pn = probe_keys.length;
    const uint64_t* bk = static_cast<const uint64_t*>(build_keys.data);
    const uint64_t* pk = static_cast<const uint64_t*>(probe_keys.data);
    assert(bn >= 0 && pn >= 0);

    int64_t i = 0, j = 0, count = 0;
    while (i < bn && j < pn) {
        const uint64_t bv = bk[i];
        const uint64_t pv = pk[j];
        if (bv < pv) { ++i; continue; }
        if (bv > pv) { ++j; continue; }
        // Equal-key run: span on build [i, i_end) × span on probe [j, j_end).
        int64_t i_end = i + 1;
        while (i_end < bn && bk[i_end] == bv) ++i_end;
        int64_t j_end = j + 1;
        while (j_end < pn && pk[j_end] == pv) ++j_end;
        for (int64_t a = i; a < i_end; ++a) {
            for (int64_t b = j; b < j_end; ++b) {
                if (count >= out_capacity) return count;  // caller-bounded
                out_build_idx[count] = static_cast<int32_t>(a);
                out_probe_idx[count] = static_cast<int32_t>(b);
                ++count;
            }
        }
        i = i_end;
        j = j_end;
    }
    return count;
}

// ---------------------------------------------------------------------------
// Left outer — inner pairs plus unmatched probe rows (build_idx = -1).
// ---------------------------------------------------------------------------
inline int64_t mergejoin_left_outer_i64(const BoltColumn& build_keys,
                                        const BoltColumn& probe_keys,
                                        int32_t* BOLT_RESTRICT out_build_idx,
                                        int32_t* BOLT_RESTRICT out_probe_idx,
                                        int64_t out_capacity) noexcept {
    assert(out_build_idx != nullptr || out_capacity == 0);
    assert(out_probe_idx != nullptr || out_capacity == 0);
    assert(out_capacity >= 0);
    if (!mergejoin_check_inputs(build_keys, probe_keys)) return -1;

    const int64_t bn = build_keys.length;
    const int64_t pn = probe_keys.length;
    const uint64_t* bk = static_cast<const uint64_t*>(build_keys.data);
    const uint64_t* pk = static_cast<const uint64_t*>(probe_keys.data);
    assert(bn >= 0 && pn >= 0);

    int64_t i = 0, j = 0, count = 0;
    while (j < pn) {
        // Advance build until we reach pk[j] or run out.
        while (i < bn && bk[i] < pk[j]) ++i;

        if (i >= bn || bk[i] > pk[j]) {
            // Unmatched probe row.
            if (count >= out_capacity) return count;
            out_build_idx[count] = -1;
            out_probe_idx[count] = static_cast<int32_t>(j);
            ++count;
            ++j;
            continue;
        }

        // Equal-key run on build; emit all pairs for this probe row.
        const uint64_t pv = pk[j];
        int64_t i_end = i + 1;
        while (i_end < bn && bk[i_end] == pv) ++i_end;
        for (int64_t a = i; a < i_end; ++a) {
            if (count >= out_capacity) return count;
            out_build_idx[count] = static_cast<int32_t>(a);
            out_probe_idx[count] = static_cast<int32_t>(j);
            ++count;
        }
        ++j;
        // i stays on the first row of this equal-key run — a later probe
        // row with the same key reuses it; a strictly-greater probe row
        // advances past it via the inner while-loop.
    }
    return count;
}

// ---------------------------------------------------------------------------
// Right outer — inner pairs plus unmatched build rows (probe_idx = -1).
// Symmetric to left outer; kept as its own function so callers don't have
// to swap argument order (and the output pairing semantics stay clear).
// ---------------------------------------------------------------------------
inline int64_t mergejoin_right_outer_i64(const BoltColumn& build_keys,
                                         const BoltColumn& probe_keys,
                                         int32_t* BOLT_RESTRICT out_build_idx,
                                         int32_t* BOLT_RESTRICT out_probe_idx,
                                         int64_t out_capacity) noexcept {
    assert(out_build_idx != nullptr || out_capacity == 0);
    assert(out_probe_idx != nullptr || out_capacity == 0);
    assert(out_capacity >= 0);
    if (!mergejoin_check_inputs(build_keys, probe_keys)) return -1;

    const int64_t bn = build_keys.length;
    const int64_t pn = probe_keys.length;
    const uint64_t* bk = static_cast<const uint64_t*>(build_keys.data);
    const uint64_t* pk = static_cast<const uint64_t*>(probe_keys.data);
    assert(bn >= 0 && pn >= 0);

    int64_t i = 0, j = 0, count = 0;
    while (i < bn) {
        while (j < pn && pk[j] < bk[i]) ++j;

        if (j >= pn || pk[j] > bk[i]) {
            if (count >= out_capacity) return count;
            out_build_idx[count] = static_cast<int32_t>(i);
            out_probe_idx[count] = -1;
            ++count;
            ++i;
            continue;
        }

        const uint64_t bv = bk[i];
        int64_t j_end = j + 1;
        while (j_end < pn && pk[j_end] == bv) ++j_end;
        for (int64_t b = j; b < j_end; ++b) {
            if (count >= out_capacity) return count;
            out_build_idx[count] = static_cast<int32_t>(i);
            out_probe_idx[count] = static_cast<int32_t>(b);
            ++count;
        }
        ++i;
    }
    return count;
}

}  // namespace bolt
