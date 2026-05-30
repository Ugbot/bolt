// bolt_joinkernel.h — typed, all-shapes SwissTable join (index-pair model).
//
// Generalizes the int64-only chained join in bolt_hashjoin.h to the TYPED
// composite-key substrate already proven for GroupBy (bolt_groupby.h
// gb_detail: GbCell16 / read_cell16 / hash_keys / keys_equal / cell_valid) —
// so join keys get the same type coverage as group-by keys: Int64 / Int32 /
// Date32 / Float64 / Decimal128 / Utf8(inline), composite up to kHJMaxKeys.
//
// Model: the kernel owns the join MATH — build a chained SwissTable
// (hash -> head chain node; multi-match), probe emits (build_idx, probe_idx)
// index pairs for every key-equal build row, and a build-side `matched[]`
// bitmap + drain helpers cover the non-inner shapes. The caller (a chukonu
// operator) owns the SQL bookkeeping that the kernel must not know about:
// residual non-equi ON predicates (it sets matched[] only for pairs that pass
// the residual), payload gather (bolt_gather), and output assembly. NULL
// index (-1) on either side = "emit NULL for this side."
//
// SQL NULL semantics: a build/probe key row with any NULL key column never
// matches (NULL != NULL). Such build rows are stored (so OUTER/ANTI still emit
// them as unmatched) but NOT inserted into the table; such probe rows skip the
// lookup entirely.
//
// Tiger Style: noexcept, POD, arena-allocated (no hot-path growth), ≥2 asserts,
// bounded chains, compile-time type dispatch (via gb_detail).

#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_port.h"
#include "bolt/bolt_types.h"
#include "bolt/join/bolt_groupby.h"   // gb_detail::{read_cell16,hash_keys,keys_equal,cell_valid}, GbCell16
#include "bolt/join/bolt_hashjoin.h"  // kHJNumPartitions, hj_partition_of, HashJoinChainNode, kHJMaxKeys, kHJMaxChainLen
#include "bolt/join/bolt_swiss.h"

#include <cassert>
#include <cstdint>
#include <cstring>

namespace bolt {

// "No match -> emit NULL for this side." Used as a build_idx / probe_idx value.
inline constexpr int32_t kJoinNullIndex = -1;

// True iff any of row `r`'s composite key columns is NULL (Arrow validity).
BOLT_FORCE_INLINE bool join_row_has_null(const BoltColumn* keys, uint8_t n_keys,
                                         int64_t r) noexcept {
    assert(keys != nullptr || n_keys == 0);
    for (uint8_t k = 0; k < n_keys; ++k) {
        if (!gb_detail::cell_valid(keys[k], r)) return true;
    }
    return false;
}

// Build side: chained SwissTable (hash -> head chain node) + flat typed key
// cells + a matched bitmap. Composite up to kHJMaxKeys. One chain node per
// build row; node index == build row index == matched[] index.
struct JoinBuildTyped {
    SwissTable         partitions[kHJNumPartitions];  // hash -> head chain idx
    HashJoinChainNode* chain_nodes;                   // [build_rows]; -1 = end
    GbCell16*          keys_flat;                     // [build_rows * n_keys]
    uint8_t*           matched;                       // [build_rows]; caller-set
    uint64_t           build_rows;
    uint8_t            n_keys;
    uint8_t            _pad[7];
};

// Build the typed chained table from `n_keys` key columns (`build_rows` rows).
// Returns false on OOM / table-full. matched[] is zeroed (caller sets it as it
// applies residuals during the probe). Mirrors hash_join_build_chained but
// typed: the SwissTable key is the composite hash; collisions share a chain and
// are disambiguated by keys_equal at probe time.
inline bool join_build_typed(const BoltColumn* key_cols, uint8_t n_keys,
                             uint64_t build_rows, Arena* arena,
                             JoinBuildTyped* out) noexcept {
    assert(arena != nullptr && out != nullptr);
    assert(n_keys >= 1 && n_keys <= kHJMaxKeys);
    const uint64_t n = build_rows;
    const size_t   n_alloc = (n == 0 ? 1 : static_cast<size_t>(n));
    out->build_rows = n;
    out->n_keys     = n_keys;
    out->chain_nodes = arena->allocate_array<HashJoinChainNode>(n_alloc);
    out->keys_flat   = arena->allocate_array<GbCell16>(n_alloc * n_keys);
    out->matched     = static_cast<uint8_t*>(arena->allocate(n_alloc, 1));
    if (out->chain_nodes == nullptr || out->keys_flat == nullptr ||
        out->matched == nullptr) {
        return false;
    }
    std::memset(out->matched, 0, n_alloc);

    uint32_t counts[kHJNumPartitions];
    std::memset(counts, 0, sizeof(counts));
    for (uint64_t i = 0; i < n; ++i) {
        if (join_row_has_null(key_cols, n_keys, static_cast<int64_t>(i))) continue;
        const uint64_t h = gb_detail::hash_keys(key_cols, n_keys,
                                                static_cast<int64_t>(i));
        counts[hj_partition_of(h)]++;
    }
    for (uint32_t p = 0; p < kHJNumPartitions; ++p) {
        const uint64_t hint = counts[p] == 0 ? 1 : counts[p];
        if (!SwissTable::create(&out->partitions[p], hint, arena)) return false;
    }
    for (uint64_t i = 0; i < n; ++i) {
        const uint32_t slot = static_cast<uint32_t>(i);  // one node/row, in order
        GbCell16* krow = out->keys_flat + static_cast<size_t>(slot) * n_keys;
        for (uint8_t k = 0; k < n_keys; ++k) {
            krow[k] = gb_detail::read_cell16(key_cols[k], static_cast<int64_t>(i));
        }
        out->chain_nodes[slot].build_idx = slot;
        out->chain_nodes[slot].next      = kJoinNullIndex;
        if (join_row_has_null(key_cols, n_keys, static_cast<int64_t>(i))) {
            continue;  // present (for OUTER/ANTI) but unreachable (NULL != NULL)
        }
        const uint64_t h = gb_detail::hash_keys(key_cols, n_keys,
                                                static_cast<int64_t>(i));
        const uint32_t p = hj_partition_of(h);
        out->chain_nodes[slot].next = out->partitions[p].find(h);  // prev head/-1
        if (!out->partitions[p].insert(h, slot)) return false;
    }
    return true;
}

// INNER multi-match probe: for each probe row walk its chain and emit every
// key-equal (build_idx, probe_idx) pair. Does NOT touch matched[] — the caller
// sets matched[build_idx] only for pairs that also pass its residual, so the
// OUTER/SEMI/ANTI drain is residual-correct. Over-cap is a hard assert; the
// caller pre-sizes pairs_cap and chunks the probe morsel stream.
inline size_t join_probe_typed(const JoinBuildTyped* build,
                               const BoltColumn* probe_keys, int64_t n_probe,
                               int32_t* BOLT_RESTRICT out_build,
                               int32_t* BOLT_RESTRICT out_probe,
                               size_t pairs_cap) noexcept {
    assert(build != nullptr);
    assert(out_build != nullptr && out_probe != nullptr);
    (void)pairs_cap;  // referenced only by the release-stripped bounds assert
    const uint8_t nk = build->n_keys;
    size_t out = 0;
    for (int64_t r = 0; r < n_probe; ++r) {
        if (join_row_has_null(probe_keys, nk, r)) continue;  // NULL != NULL
        const uint64_t h = gb_detail::hash_keys(probe_keys, nk, r);
        const uint32_t p = hj_partition_of(h);
        int32_t  node = build->partitions[p].find(h);
        uint32_t walk = 0;
        while (node >= 0) {
            assert(walk < kHJMaxChainLen);
            const uint32_t slot = static_cast<uint32_t>(node);
            if (gb_detail::keys_equal(probe_keys, nk, r, build->keys_flat, slot)) {
                assert(out < pairs_cap);
                out_build[out] = static_cast<int32_t>(build->chain_nodes[slot].build_idx);
                out_probe[out] = static_cast<int32_t>(r);
                ++out;
            }
            node = build->chain_nodes[slot].next;
            ++walk;
        }
    }
    return out;
}

// Drain build-side rows by matched flag, for the finalize phase of the
// non-inner shapes. `want_matched`: 1 = SEMI (emit matched build rows once);
// 0 = ANTI / OUTER-unmatched (emit unmatched build rows once, probe side NULL).
// `*cursor` carries position across chunked calls; out_probe is the NULL
// sentinel. Returns rows written this call.
inline size_t join_drain_build(const JoinBuildTyped* build, uint8_t want_matched,
                               uint32_t* BOLT_RESTRICT cursor,
                               int32_t* BOLT_RESTRICT out_build,
                               int32_t* BOLT_RESTRICT out_probe,
                               size_t out_cap) noexcept {
    assert(build != nullptr && cursor != nullptr);
    assert(out_build != nullptr && out_probe != nullptr);
    size_t out = 0;
    while (static_cast<uint64_t>(*cursor) < build->build_rows && out < out_cap) {
        const uint32_t i = (*cursor)++;
        if (build->matched[i] == want_matched) {
            out_build[out] = static_cast<int32_t>(i);
            out_probe[out] = kJoinNullIndex;
            ++out;
        }
    }
    return out;
}

}  // namespace bolt
