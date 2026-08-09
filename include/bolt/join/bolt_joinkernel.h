// bolt_joinkernel.h — typed, all-shapes SwissTable join (index-pair model).
//
// Generalizes the int64-only chained join in bolt_hashjoin.h to the TYPED
// composite-key substrate already proven for GroupBy (bolt_groupby.h
// gb_detail: GbCell16 / read_cell16 / hash_keys / keys_equal / cell_valid) —
// so join keys get the same type coverage as group-by keys: Int64 / Int32 /
// Date32 / Float64 / Decimal128 / Utf8(inline+spilled), composite up to
// kHJMaxKeys.
//
// Hardening (Card A2) — all additive, no caller-visible signature changes:
//   * Float64 key canonicalization: -0.0 hashes/compares equal to +0.0, and
//     every NaN bit pattern collapses to one canonical quiet NaN so a NaN key
//     is equal-to-itself. Done in a JOIN-LOCAL cell-read wrapper
//     (jk_detail::read_cell16_canon) — gb_detail is NOT mutated.
//   * Decimal128 scale precondition: the probe asserts each Decimal128 key
//     column's scale matches the build side's captured scale, so equal nominal
//     values stored at different scales never silently mismatch (or match).
//   * Spilled-Utf8 key arm: GbCell16's raw-16-byte compare false-negatives on
//     >12-byte Utf8 keys (equal strings at different arena offsets differ in
//     the `ref`). The typed hash/equal helpers resolve Utf8 key bytes via
//     sv_bytes / sv_compare against each column's str_overflow_base, which the
//     build captures into the JoinBuildTyped side table from the passed
//     BoltColumns (BoltColumn already carries str_overflow_base).
//   * int64 OneInt64 fast path: a single Int64/UInt64/Int32/Date32 key bypasses
//     GbCell16 and uses the raw-value int64 hash/equality helpers from
//     bolt_hashjoin.h. Compile-time KeyShape dispatch; correctness identical.
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
#include "bolt/join/bolt_sbbf.h"      // SplitBlockBloom (W-J4 probe prefilter)
#include "bolt/join/bolt_swiss.h"
#include "bolt/kernels/bolt_utf8.h"   // sv_bytes / sv_compare for spilled-Utf8 key resolution

#include <cassert>
#include <cstdint>
#include <cstdio>    // latched CHUKONU_HJ_MLP_TRACE diagnostic only
#include <cstdlib>   // getenv/atoi — kill-switch latch only, never per row
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

// ===========================================================================
// jk_detail — JOIN-LOCAL typed cell helpers (additive; gb_detail is untouched).
//
// gb_detail::read_cell16 copies Float64 raw bits and copies the 16-byte
// StringView verbatim for Utf8. Two consequences the join must fix:
//   1. Float64: -0.0 (0x8000000000000000) and +0.0 (0) are distinct bit
//      patterns, and NaN has 2^53-2 distinct payloads — raw-bit hash/equality
//      would (a) put -0.0 and +0.0 in different groups and (b) make a NaN key
//      never equal to itself. SQL equi-join wants -0.0 == +0.0; NaN behaviour
//      we *define* as "a single canonical NaN compares equal to itself".
//   2. Utf8 spilled (>12 bytes): the cell's tail is a {buf_idx, offset} `ref`,
//      so two equal strings spilled at different offsets compare UNEQUAL. The
//      join resolves the real bytes via sv_bytes + sv_compare.
//
// Tiger Style: noexcept, branch-light, ≥2 asserts on the entry points, no
// allocation, compile-time type dispatch via the read wrapper's switch.
// ===========================================================================
namespace jk_detail {

// Canonicalize a Float64 bit pattern: map -0.0 -> +0.0 and every NaN to one
// canonical quiet NaN (0x7FF8000000000000). Finite/non-zero values unchanged.
// Branch-light: the two corrections are predicated selects.
BOLT_FORCE_INLINE int64_t canon_f64_bits(int64_t raw) noexcept {
    constexpr uint64_t kSignMask = 0x8000000000000000ULL;
    constexpr uint64_t kExpMask  = 0x7FF0000000000000ULL;
    constexpr uint64_t kFracMask = 0x000FFFFFFFFFFFFFULL;
    constexpr uint64_t kQNaN     = 0x7FF8000000000000ULL;  // canonical quiet NaN
    const uint64_t b = static_cast<uint64_t>(raw);
    // NaN iff all exponent bits set AND fraction != 0.
    const bool is_nan = ((b & kExpMask) == kExpMask) && ((b & kFracMask) != 0);
    // -0.0 iff bit pattern is exactly the sign bit.
    const bool is_neg_zero = (b == kSignMask);
    uint64_t out = b;
    out = is_neg_zero ? 0ULL : out;
    out = is_nan      ? kQNaN : out;
    return static_cast<int64_t>(out);
}

// Read one key cell with join-local canonicalization. Identical to
// gb_detail::read_cell16 except Float64 cells are canonicalized so the hash
// and the equality compare agree on -0.0/+0.0 and on NaN. All other types
// pass straight through gb_detail (single source of truth for their layout).
BOLT_FORCE_INLINE GbCell16 read_cell16_canon(const BoltColumn& c,
                                             int64_t r) noexcept {
    GbCell16 out = gb_detail::read_cell16(c, r);
    if (c.type == BoltType::Float64) out.a = canon_f64_bits(out.a);
    return out;
}

// Composite-key hash with Float64 canonicalization + Utf8 byte resolution.
// Mirrors gb_detail::hash_keys' mix exactly for non-Utf8/Float64 cells so the
// OneInt64 / cell paths interoperate; Utf8 keys fold utf8_hash_one over the
// resolved bytes (inline or spilled) instead of the raw 16-byte view, so equal
// spilled strings at different offsets hash identically.
BOLT_FORCE_INLINE uint64_t hash_keys_typed(const BoltColumn* keys,
                                           uint8_t n_keys, int64_t r) noexcept {
    assert(keys != nullptr || n_keys == 0);
    assert(n_keys <= kHJMaxKeys);
    uint64_t h = 0x9E3779B97F4A7C15ULL;
    for (uint8_t k = 0; k < n_keys; ++k) {
        const BoltColumn& c = keys[k];
        if (c.type == BoltType::Utf8) {
            const StringView& sv = static_cast<const StringView*>(c.data)[r];
            const char* base =
                static_cast<const char*>(c.str_overflow_base);
            const char* bytes = kernels::utf8::sv_bytes(sv, base);
            const uint64_t sh = kernels::utf8::utf8_hash_one(bytes, sv.length);
            h ^= swiss_mix_wyhash3(sh);
            h  = swiss_mix_wyhash3(h);
            continue;
        }
        const GbCell16 cell = read_cell16_canon(c, r);
        h ^= swiss_mix_wyhash3(static_cast<uint64_t>(cell.a));
        h  = swiss_mix_wyhash3(h ^ static_cast<uint64_t>(cell.b));
    }
    return h;
}

// True iff probe row `r`'s composite key equals the build group stored at
// `slot` in keys_flat. Utf8 keys compare resolved bytes via sv_compare
// (probe base from the live probe column; build base from the captured
// `build_str_base[k]`). Float64 cells are canonicalized on the probe side; the
// build side was canonicalized at store time, so a NaN/-0.0 build key matches.
BOLT_FORCE_INLINE bool keys_equal_typed(const BoltColumn* keys, uint8_t n_keys,
                                        int64_t r, const GbCell16* keys_flat,
                                        uint32_t slot,
                                        const BoltType* key_types,
                                        const char* const* build_str_base)
                                        noexcept {
    assert(keys != nullptr || n_keys == 0);
    assert(keys_flat != nullptr);
    const GbCell16* row = keys_flat + static_cast<size_t>(slot) * n_keys;
    for (uint8_t k = 0; k < n_keys; ++k) {
        if (key_types[k] == BoltType::Utf8) {
            const StringView& pv = static_cast<const StringView*>(keys[k].data)[r];
            StringView bv;
            std::memcpy(&bv, &row[k], sizeof(bv));
            const char* pbase =
                static_cast<const char*>(keys[k].str_overflow_base);
            const char* bbase = build_str_base[k];
            if (kernels::utf8::sv_compare(pv, pbase, bv, bbase) != 0) return false;
            continue;
        }
        const GbCell16 c = read_cell16_canon(keys[k], r);
        if (c.a != row[k].a || c.b != row[k].b) return false;
    }
    return true;
}

}  // namespace jk_detail

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
    // --- Additive (Card A2). Existing field offsets above are UNCHANGED, so
    // callers that read {partitions, chain_nodes, keys_flat, matched,
    // build_rows, n_keys} are unaffected. These are populated by
    // join_build_typed from the passed key columns and consumed by
    // join_probe_typed for the typed (Float64/Decimal128/Utf8) arms. ---
    BoltType    key_types[kHJMaxKeys];   // per-key column type (for probe dispatch)
    uint8_t     key_scales[kHJMaxKeys];  // Decimal128 scale per key; 0 otherwise
    const char* key_str_base[kHJMaxKeys];// Utf8 spilled-bytes base per key; null otherwise
    uint8_t     key_shape;               // JoinKeyShape (cached fast-path tag)
    uint8_t     _pad2[7];
    // --- Additive (W-J4). Split-block Bloom over the reachable build keys'
    // MIXED hashes (the same value used for partition selection, so the
    // probe tests it for free before the SwissTable find). Built always —
    // ~10 bits/key, one sbbf_add per inserted row — because the BUILD can't
    // know the probe's match rate; the PROBE decides per call whether to
    // consult it (join_probe_typed's use_bloom). has_bloom == 0 when the
    // arena couldn't fit the filter (never fatal: probes skip the test). ---
    SplitBlockBloom bloom;
    uint8_t         has_bloom;
    uint8_t         _pad3[7];
    // --- Additive (fan-out bound). Per-partition count of DISTINCT keys, i.e.
    // inserts that found no existing chain head. Written only by the partition
    // that owns it, so the parallel build needs no atomics. Consumers use
    // jk_max_fanout() rather than build_rows to bound how many probe rows can
    // safely go into one join_probe_typed call: `build_rows` is the true worst
    // case ("one probe row matches every build row") but is wildly pessimistic
    // for the common unique/PK build side, where the real fan-out is 1. A caller
    // dividing a fixed pairs budget by build_rows collapses to ONE ROW PER CALL
    // for any build over that budget, which drives this vectorized kernel
    // scalar. distinct_per_part is computed for free during the build.
    uint32_t distinct_per_part[kHJNumPartitions];
};

// Compile-/run-time key-shape tag. `OneInt64` (a single 64-bit-or-narrower
// integer key) takes the raw-value SwissTable fast path; `General` uses the
// typed GbCell16 path. Stored on the build so the probe dispatches once.
enum class JoinKeyShape : uint8_t {
    General  = 0,
    OneInt64 = 1,   // single Int64 / UInt64 / Int32 / Date32 key column
};

// True iff `t` is a 64-bit-or-narrower integer key that the OneInt64 fast path
// can represent losslessly as a raw uint64 SwissTable key.
BOLT_FORCE_INLINE bool jk_is_int64_shape_type(BoltType t) noexcept {
    return t == BoltType::Int64 || t == BoltType::UInt64 ||
           t == BoltType::Int32 || t == BoltType::Date32;
}

// Read one int-key cell as the raw uint64 value used by the OneInt64 path.
// Int32/Date32 are sign-extended to 64 bits so a negative key keeps its value.
BOLT_FORCE_INLINE uint64_t jk_read_int64_key(const BoltColumn& c,
                                             int64_t r) noexcept {
    assert(c.data != nullptr);
    switch (c.type) {
        case BoltType::Int64:
        case BoltType::UInt64: return static_cast<const uint64_t*>(c.data)[r];
        case BoltType::Int32:
        case BoltType::Date32:
            return static_cast<uint64_t>(
                static_cast<int64_t>(static_cast<const int32_t*>(c.data)[r]));
        default:
            assert(false && "jk_read_int64_key on non-int key");
            return static_cast<const uint64_t*>(c.data)[r];
    }
}

// Capture per-key metadata (types/scales/overflow bases) + decide the key
// shape. Pure bookkeeping over the additive JoinBuildTyped fields; no hashing.
BOLT_FORCE_INLINE void jk_capture_key_meta(const BoltColumn* key_cols,
                                           uint8_t n_keys,
                                           JoinBuildTyped* out) noexcept {
    assert(key_cols != nullptr || n_keys == 0);
    assert(out != nullptr);
    for (uint8_t k = 0; k < n_keys; ++k) {
        out->key_types[k]    = key_cols[k].type;
        out->key_scales[k]   = key_cols[k].decimal_scale;
        out->key_str_base[k] =
            static_cast<const char*>(key_cols[k].str_overflow_base);
    }
    const bool one_int = (n_keys == 1) &&
                         jk_is_int64_shape_type(key_cols[0].type);
    out->key_shape = static_cast<uint8_t>(
        one_int ? JoinKeyShape::OneInt64 : JoinKeyShape::General);
}

// Core build shared by the public auto-detect entry and the force-General
// reference builder. `force_general` overrides shape detection (used by the
// fast-path agreement test and as the General reference path). The SwissTable
// "table key" is the RAW value in OneInt64 mode (so the fast-path probe finds
// it with a plain value compare) and the composite hash in General mode.
inline bool jk_build_core(const BoltColumn* key_cols, uint8_t n_keys,
                          uint64_t build_rows, Arena* arena, bool force_general,
                          JoinBuildTyped* out) noexcept {
    assert(arena != nullptr && out != nullptr);
    assert(n_keys >= 1 && n_keys <= kHJMaxKeys);
    const uint64_t n = build_rows;
    const size_t   n_alloc = (n == 0 ? 1 : static_cast<size_t>(n));
    out->build_rows = n;
    out->n_keys     = n_keys;
    // Fan-out accounting starts at zero on every build (see jk_max_fanout).
    std::memset(out->distinct_per_part, 0, sizeof(out->distinct_per_part));
    jk_capture_key_meta(key_cols, n_keys, out);
    if (force_general) out->key_shape = static_cast<uint8_t>(JoinKeyShape::General);
    const bool one_int = (out->key_shape ==
                          static_cast<uint8_t>(JoinKeyShape::OneInt64));
    out->chain_nodes = arena->allocate_array<HashJoinChainNode>(n_alloc);
    out->keys_flat   = arena->allocate_array<GbCell16>(n_alloc * n_keys);
    out->matched     = static_cast<uint8_t*>(arena->allocate(n_alloc, 1));
    if (out->chain_nodes == nullptr || out->keys_flat == nullptr ||
        out->matched == nullptr) {
        return false;
    }
    std::memset(out->matched, 0, n_alloc);
    // W-J4 probe prefilter — sized to the row count (10 bits/key ≈ 1% FPR);
    // allocation failure just disables it (probe correctness never depends
    // on the bloom: it only skips DEFINITE non-members).
    out->has_bloom = sbbf_create(&out->bloom,
                                 static_cast<int64_t>(n_alloc), arena) ? 1 : 0;

    uint32_t counts[kHJNumPartitions];
    std::memset(counts, 0, sizeof(counts));
    for (uint64_t i = 0; i < n; ++i) {
        const int64_t r = static_cast<int64_t>(i);
        if (join_row_has_null(key_cols, n_keys, r)) continue;
        const uint64_t h = one_int ? swiss_mix(jk_read_int64_key(key_cols[0], r))
                                   : jk_detail::hash_keys_typed(key_cols, n_keys, r);
        counts[hj_partition_of(h)]++;
    }
    for (uint32_t p = 0; p < kHJNumPartitions; ++p) {
        const uint64_t hint = counts[p] == 0 ? 1 : counts[p];
        if (!SwissTable::create(&out->partitions[p], hint, arena)) return false;
    }
    for (uint64_t i = 0; i < n; ++i) {
        const int64_t  r    = static_cast<int64_t>(i);
        const uint32_t slot = static_cast<uint32_t>(i);  // one node/row, in order
        GbCell16* krow = out->keys_flat + static_cast<size_t>(slot) * n_keys;
        for (uint8_t k = 0; k < n_keys; ++k) {
            krow[k] = jk_detail::read_cell16_canon(key_cols[k], r);
        }
        out->chain_nodes[slot].build_idx = slot;
        out->chain_nodes[slot].next      = kJoinNullIndex;
        if (join_row_has_null(key_cols, n_keys, r)) {
            continue;  // present (for OUTER/ANTI) but unreachable (NULL != NULL)
        }
        const uint64_t tkey = one_int ? jk_read_int64_key(key_cols[0], r)
                                      : jk_detail::hash_keys_typed(key_cols, n_keys, r);
        const uint64_t h = one_int ? swiss_mix(tkey) : tkey;
        const uint32_t p = hj_partition_of(h);
        if (out->has_bloom != 0) sbbf_add(out->bloom, h);
        // ONE probe: link to the previous chain head (or -1) and become the
        // new head. Was a full find() THEN a full insert() — two walks over
        // the same probe chain per build row, measured at 23% of Q18 self.
        int32_t prev_head = -1;
        if (!out->partitions[p].upsert(tkey, static_cast<uint32_t>(slot),
                                       &prev_head)) {
            return false;
        }
        out->chain_nodes[slot].next = prev_head;
        // No prior head => this key is new. Counting here is free: the branch
        // is perfectly predicted for the all-distinct (PK) case.
        if (prev_head < 0) ++out->distinct_per_part[p];
    }
    return true;
}

// Build the typed chained table from `n_keys` key columns (`build_rows` rows).
// Returns false on OOM / table-full. matched[] is zeroed (caller sets it as it
// applies residuals during the probe). Mirrors hash_join_build_chained but
// typed: the SwissTable key is the composite hash; collisions share a chain and
// are disambiguated by keys_equal at probe time.
//
// Hardening: hashes via jk_detail::hash_keys_typed (Float64-canonical, Utf8
// byte-resolved) and stores Float64-canonicalized cells, so a -0.0/+0.0 or
// NaN build key is reachable and a spilled-Utf8 build key is comparable. For a
// single integer key the table key is the raw value (OneInt64 fast path).
inline bool join_build_typed(const BoltColumn* key_cols, uint8_t n_keys,
                             uint64_t build_rows, Arena* arena,
                             JoinBuildTyped* out) noexcept {
    return jk_build_core(key_cols, n_keys, build_rows, arena,
                         /*force_general=*/false, out);
}

// Reference / test builder that forces the General (GbCell16 composite-hash)
// path even for a single integer key. Used to assert the OneInt64 fast path
// and the general path emit identical pairs. Probe such a table with
// jk_probe_general (NOT join_probe_typed, which would dispatch by key_shape).
inline bool join_build_typed_general(const BoltColumn* key_cols, uint8_t n_keys,
                                     uint64_t build_rows, Arena* arena,
                                     JoinBuildTyped* out) noexcept {
    return jk_build_core(key_cols, n_keys, build_rows, arena,
                         /*force_general=*/true, out);
}

// ===========================================================================
// W-PAR: PARALLEL hash-join build. JoinBuildTyped is ALREADY partitioned by
// hj_partition_of into kHJNumPartitions (64) independent SwissTables, and the
// probe routes a key to its one partition — so a table whose partitions were
// each built by a different worker is PROBE-IDENTICAL to a serial build. We
// split the build in two phases: Phase 1 (calling thread) hashes every key
// once to histogram per partition, add to the bloom, init every chain node,
// write every key cell, and SCATTER the non-null row indices into per-
// partition groups IN ROW ORDER; Phase 2 (one task per partition range,
// worker threads) inserts each partition's rows into its own SwissTable +
// sets each row's chain next. Partitions are DISJOINT (a key hashes to one
// partition; all rows of a key share it), so workers touch disjoint tables +
// disjoint chain/insert slots — NO locks, NO atomics. Insert order within a
// partition is row order (the scatter preserves it), so chains are byte-
// identical to serial -> the probe emits identical pairs. The expensive
// cache-miss-bound SwissTable inserts parallelise; the cheap sequential
// Phase-1 work stays serial.
// ===========================================================================

// Row indices grouped by partition (Phase-1 output, Phase-2 input). row_ids
// holds only NON-NULL-key rows (null keys are present-but-unreachable); the
// `offsets` prefix-sum bounds each partition's slice in row_ids.
struct JoinBuildScatterCtx {
    int32_t* row_ids;                          // [<= build_rows], by partition
    uint32_t offsets[kHJNumPartitions + 1];    // prefix-sum boundaries
};

// Phase 1 (calling thread): capture meta, allocate the global arrays, init
// every chain node + key cell, fold the bloom over every reachable key, and
// scatter non-null row indices into per-partition groups in row order; then
// create the per-partition SwissTables sized to their counts. After this the
// tables are EMPTY (Phase 2 fills them). Returns false on OOM.
inline bool jk_build_scatter_and_alloc(const BoltColumn* key_cols,
        uint8_t n_keys, uint64_t build_rows, Arena* arena,
        JoinBuildTyped* out, JoinBuildScatterCtx* scat) noexcept {
    assert(arena != nullptr && out != nullptr && scat != nullptr);
    assert(n_keys >= 1 && n_keys <= kHJMaxKeys);
    const uint64_t n = build_rows;
    const size_t   n_alloc = (n == 0 ? 1 : static_cast<size_t>(n));
    out->build_rows = n;
    out->n_keys     = n_keys;
    // Fan-out accounting starts at zero on every build (see jk_max_fanout).
    std::memset(out->distinct_per_part, 0, sizeof(out->distinct_per_part));
    jk_capture_key_meta(key_cols, n_keys, out);
    const bool one_int = (out->key_shape ==
                          static_cast<uint8_t>(JoinKeyShape::OneInt64));
    out->chain_nodes = arena->allocate_array<HashJoinChainNode>(n_alloc);
    out->keys_flat   = arena->allocate_array<GbCell16>(n_alloc * n_keys);
    out->matched     = static_cast<uint8_t*>(arena->allocate(n_alloc, 1));
    scat->row_ids    = arena->allocate_array<int32_t>(n_alloc);
    if (out->chain_nodes == nullptr || out->keys_flat == nullptr ||
        out->matched == nullptr || scat->row_ids == nullptr) {
        return false;
    }
    std::memset(out->matched, 0, n_alloc);
    out->has_bloom = sbbf_create(&out->bloom,
                                 static_cast<int64_t>(n_alloc), arena) ? 1 : 0;
    uint32_t counts[kHJNumPartitions];
    std::memset(counts, 0, sizeof(counts));
    // Pass 1: init chain node + write key cell for EVERY row (SEQUENTIAL —
    // cache-friendly streaming read of key_cols, streaming write of keys_flat;
    // null rows must still drain for OUTER/ANTI); histogram + bloom over the
    // reachable rows. Phase 2 (parallel) then only does the random-access
    // SwissTable inserts — the cache-miss-bound part worth splitting.
    for (uint64_t i = 0; i < n; ++i) {
        const int64_t  r    = static_cast<int64_t>(i);
        const uint32_t slot = static_cast<uint32_t>(i);
        GbCell16* krow = out->keys_flat + static_cast<size_t>(slot) * n_keys;
        for (uint8_t k = 0; k < n_keys; ++k) {
            krow[k] = jk_detail::read_cell16_canon(key_cols[k], r);
        }
        out->chain_nodes[slot].build_idx = slot;
        out->chain_nodes[slot].next      = kJoinNullIndex;
        if (join_row_has_null(key_cols, n_keys, r)) continue;
        const uint64_t tkey = one_int ? jk_read_int64_key(key_cols[0], r)
                                      : jk_detail::hash_keys_typed(key_cols, n_keys, r);
        const uint64_t h = one_int ? swiss_mix(tkey) : tkey;
        counts[hj_partition_of(h)]++;
        if (out->has_bloom != 0) sbbf_add(out->bloom, h);
    }
    // Prefix-sum -> offsets; create tables sized to counts.
    uint32_t acc = 0;
    for (uint32_t p = 0; p < kHJNumPartitions; ++p) {
        scat->offsets[p] = acc;
        acc += counts[p];
        const uint64_t hint = counts[p] == 0 ? 1 : counts[p];
        if (!SwissTable::create(&out->partitions[p], hint, arena)) return false;
    }
    scat->offsets[kHJNumPartitions] = acc;
    // Pass 2: scatter non-null rows into per-partition groups IN ROW ORDER
    // (a moving cursor per partition over the prefix-sum slices).
    uint32_t cursor[kHJNumPartitions];
    std::memcpy(cursor, scat->offsets, sizeof(uint32_t) * kHJNumPartitions);
    for (uint64_t i = 0; i < n; ++i) {
        const int64_t r = static_cast<int64_t>(i);
        if (join_row_has_null(key_cols, n_keys, r)) continue;
        const uint64_t tkey = one_int ? jk_read_int64_key(key_cols[0], r)
                                      : jk_detail::hash_keys_typed(key_cols, n_keys, r);
        const uint64_t h = one_int ? swiss_mix(tkey) : tkey;
        scat->row_ids[cursor[hj_partition_of(h)]++] = static_cast<int32_t>(i);
    }
    return true;
}

// ---------------------------------------------------------------------------
// W-PAR inc 3 — PARALLEL Phase 1 (task bodies + serial glue; the caller owns
// the scheduler orchestration, exactly like jk_build_partition_range):
//
//   prepare (calling thread)  : meta capture + EVERY allocation (arena).
//   pass A  (task, per chunk) : key cells + chain init for rows [b,e), stash
//                               each non-null row's partition hash, fill a
//                               PRIVATE per-chunk histogram.
//   glue    (calling thread)  : bloom fold over the stashed hashes, column-
//                               major prefix sum -> scat->offsets + per-
//                               (chunk,partition) scatter bases, SwissTable
//                               creates sized to the counts.
//   pass B  (task, per chunk) : scatter row ids into the chunk's reserved
//                               bases (local cursors — zero synchronisation).
//
// Result is byte-identical to jk_build_scatter_and_alloc: chunks ascend in
// row order and each partition's slices are reserved in chunk order, so
// row_ids stay in ascending row order within every partition (the chain-
// order contract Phase 2 depends on); the bloom receives the same SET of
// adds (SBBF add is an idempotent OR — order-free). Null-key rows are
// re-checked per pass (the serial code paid the same check per pass);
// hashes[] is only ever read for non-null rows.

constexpr uint32_t kJkP1MaxChunks    = 64;     // hists/bases bound
constexpr uint64_t kJkP1MinChunkRows = 65536;  // below this, chunking can't pay

struct JkPhase1Ctx {
    const BoltColumn*    key_cols;
    JoinBuildTyped*      out;
    JoinBuildScatterCtx* scat;
    uint64_t*            hashes;   // [rows] partition hash per non-null row
    uint32_t*            hists;    // [n_chunks][kHJNumPartitions] private
    uint32_t*            bases;    // [n_chunks][kHJNumPartitions] scatter bases
    uint64_t             rows;
    uint64_t             chunk_rows;
    uint32_t             n_chunks;
    uint8_t              n_keys;
    uint8_t              one_int;
    uint8_t              _pad[2];
};

inline bool jk_build_phase1_prepare(const BoltColumn* key_cols, uint8_t n_keys,
        uint64_t build_rows, uint32_t max_chunks, Arena* arena,
        JoinBuildTyped* out, JoinBuildScatterCtx* scat,
        JkPhase1Ctx* c) noexcept {
    assert(arena != nullptr && out != nullptr && scat != nullptr);
    assert(c != nullptr && n_keys >= 1 && n_keys <= kHJMaxKeys);
    const uint64_t n = build_rows;
    if (n < kJkP1MinChunkRows * 2u) return false;      // not worth chunking
    const size_t n_alloc = static_cast<size_t>(n);
    out->build_rows = n;
    out->n_keys     = n_keys;
    // Fan-out accounting starts at zero on every build (see jk_max_fanout).
    std::memset(out->distinct_per_part, 0, sizeof(out->distinct_per_part));
    jk_capture_key_meta(key_cols, n_keys, out);
    out->chain_nodes = arena->allocate_array<HashJoinChainNode>(n_alloc);
    out->keys_flat   = arena->allocate_array<GbCell16>(n_alloc * n_keys);
    out->matched     = static_cast<uint8_t*>(arena->allocate(n_alloc, 1));
    scat->row_ids    = arena->allocate_array<int32_t>(n_alloc);
    c->hashes        = arena->allocate_array<uint64_t>(n_alloc);
    if (out->chain_nodes == nullptr || out->keys_flat == nullptr ||
        out->matched == nullptr || scat->row_ids == nullptr ||
        c->hashes == nullptr) {
        return false;
    }
    std::memset(out->matched, 0, n_alloc);
    out->has_bloom = sbbf_create(&out->bloom,
                                 static_cast<int64_t>(n_alloc), arena) ? 1 : 0;
    uint32_t nc = (max_chunks > kJkP1MaxChunks) ? kJkP1MaxChunks : max_chunks;
    if (nc < 1u) nc = 1u;
    uint64_t chunk = (n + nc - 1u) / nc;
    if (chunk < kJkP1MinChunkRows) chunk = kJkP1MinChunkRows;
    c->n_chunks   = static_cast<uint32_t>((n + chunk - 1u) / chunk);
    c->chunk_rows = chunk;
    assert(c->n_chunks >= 1 && c->n_chunks <= kJkP1MaxChunks);
    const size_t hb = static_cast<size_t>(c->n_chunks) * kHJNumPartitions;
    c->hists = arena->allocate_array<uint32_t>(hb);
    c->bases = arena->allocate_array<uint32_t>(hb);
    if (c->hists == nullptr || c->bases == nullptr) return false;
    std::memset(c->hists, 0, hb * sizeof(uint32_t));
    c->key_cols = key_cols;
    c->out      = out;
    c->scat     = scat;
    c->rows     = n;
    c->n_keys   = n_keys;
    c->one_int  = (out->key_shape ==
                   static_cast<uint8_t>(JoinKeyShape::OneInt64)) ? 1u : 0u;
    return true;
}

// Pass A task body — rows [chunk*chunk_rows, ...): disjoint writes only.
inline void jk_build_pass_a(const JkPhase1Ctx* c, uint32_t chunk) noexcept {
    assert(c != nullptr && chunk < c->n_chunks);
    const uint64_t b = static_cast<uint64_t>(chunk) * c->chunk_rows;
    uint64_t e = b + c->chunk_rows;
    if (e > c->rows) e = c->rows;
    assert(b < e);
    JoinBuildTyped* out = c->out;
    uint32_t* BOLT_RESTRICT hist =
        c->hists + static_cast<size_t>(chunk) * kHJNumPartitions;
    const bool one_int = (c->one_int != 0);
    for (uint64_t i = b; i < e; ++i) {
        const int64_t  r    = static_cast<int64_t>(i);
        const uint32_t slot = static_cast<uint32_t>(i);
        GbCell16* krow = out->keys_flat + static_cast<size_t>(slot) * c->n_keys;
        for (uint8_t k = 0; k < c->n_keys; ++k) {
            krow[k] = jk_detail::read_cell16_canon(c->key_cols[k], r);
        }
        out->chain_nodes[slot].build_idx = slot;
        out->chain_nodes[slot].next      = kJoinNullIndex;
        if (join_row_has_null(c->key_cols, c->n_keys, r)) continue;
        const uint64_t tkey = one_int
            ? jk_read_int64_key(c->key_cols[0], r)
            : jk_detail::hash_keys_typed(c->key_cols, c->n_keys, r);
        const uint64_t h = one_int ? swiss_mix(tkey) : tkey;
        c->hashes[i] = h;
        hist[hj_partition_of(h)]++;
    }
}

// Serial glue — bloom fold + column-major prefix sums + table creates.
// Returns false on SwissTable OOM (caller falls back to the serial build).
inline bool jk_build_phase1_glue(JkPhase1Ctx* c, Arena* arena) noexcept {
    assert(c != nullptr && arena != nullptr);
    assert(c->n_chunks >= 1);
    JoinBuildTyped* out = c->out;
    if (out->has_bloom != 0) {
        for (uint64_t i = 0; i < c->rows; ++i) {
            if (join_row_has_null(c->key_cols, c->n_keys,
                                  static_cast<int64_t>(i))) continue;
            sbbf_add(out->bloom, c->hashes[i]);
        }
    }
    uint32_t acc = 0;
    for (uint32_t p = 0; p < kHJNumPartitions; ++p) {
        c->scat->offsets[p] = acc;
        for (uint32_t k = 0; k < c->n_chunks; ++k) {
            const size_t at = static_cast<size_t>(k) * kHJNumPartitions + p;
            c->bases[at] = acc;
            acc += c->hists[at];
        }
        const uint32_t count = acc - c->scat->offsets[p];
        const uint64_t hint  = (count == 0u) ? 1u : count;
        if (!SwissTable::create(&out->partitions[p], hint, arena)) return false;
    }
    c->scat->offsets[kHJNumPartitions] = acc;
    return true;
}

// Pass B task body — scatter this chunk's non-null rows into its reserved
// per-partition slices (local cursors; slices are disjoint across chunks).
inline void jk_build_pass_b(const JkPhase1Ctx* c, uint32_t chunk) noexcept {
    assert(c != nullptr && chunk < c->n_chunks);
    const uint64_t b = static_cast<uint64_t>(chunk) * c->chunk_rows;
    uint64_t e = b + c->chunk_rows;
    if (e > c->rows) e = c->rows;
    assert(b < e);
    uint32_t cur[kHJNumPartitions];
    std::memcpy(cur, c->bases + static_cast<size_t>(chunk) * kHJNumPartitions,
                sizeof(cur));
    for (uint64_t i = b; i < e; ++i) {
        if (join_row_has_null(c->key_cols, c->n_keys,
                              static_cast<int64_t>(i))) continue;
        c->scat->row_ids[cur[hj_partition_of(c->hashes[i])]++] =
            static_cast<int32_t>(i);
    }
}

// Phase 2 (TASK BODY, parallel): insert the rows of partitions [p_lo, p_hi)
// from the scatter into their SwissTables + set each row's chain next. Rows
// arrive in ascending order within a partition, so the chain order is
// byte-identical to serial jk_build_core. Disjoint tables + disjoint chain
// slots across partitions -> no synchronisation. Returns false on table-full.
inline bool jk_build_partition_range(const BoltColumn* key_cols, uint8_t n_keys,
        const JoinBuildScatterCtx* scat, uint32_t p_lo, uint32_t p_hi,
        JoinBuildTyped* out) noexcept {
    assert(out != nullptr && scat != nullptr);
    assert(p_lo <= p_hi && p_hi <= kHJNumPartitions);
    const bool one_int = (out->key_shape ==
                          static_cast<uint8_t>(JoinKeyShape::OneInt64));
    for (uint32_t p = p_lo; p < p_hi; ++p) {
        const uint32_t lo = scat->offsets[p];
        const uint32_t hi = scat->offsets[p + 1];
        for (uint32_t idx = lo; idx < hi; ++idx) {
            const int32_t slot = scat->row_ids[idx];
            const int64_t r    = slot;
            const uint64_t tkey = one_int ? jk_read_int64_key(key_cols[0], r)
                                          : jk_detail::hash_keys_typed(key_cols, n_keys, r);
            // One-probe upsert: previous head out, new head in (see the
            // sibling site in jk_build_core for the measured motivation).
            int32_t prev_head = -1;
            if (!out->partitions[p].upsert(tkey, static_cast<uint32_t>(slot),
                                           &prev_head)) {
                return false;
            }
            out->chain_nodes[slot].next = prev_head;
            if (prev_head < 0) ++out->distinct_per_part[p];
        }
    }
    return true;
}

// Upper bound on how many build rows a single probe row can match.
//
// Exact (== 1) for the case that dominates analytical joins: a unique / primary
// key build side, where every insert found an empty chain so distinct == rows.
// With duplicates present, all of them could in principle sit on one chain, so
// the safe bound is (rows - distinct + 1) — still far tighter than `rows`, and
// additionally capped by kHJMaxChainLen, which the probe loops already assert
// no chain can exceed.
//
// Never returns 0 (a caller dividing by this must not trip), and never returns
// more than build_rows.
BOLT_FORCE_INLINE uint64_t jk_max_fanout(const JoinBuildTyped* build) noexcept {
    assert(build != nullptr);
    if (build->build_rows == 0) return 1;
    uint64_t distinct = 0;
    for (uint32_t p = 0; p < kHJNumPartitions; ++p) {
        distinct += build->distinct_per_part[p];
    }
    assert(distinct <= build->build_rows);
    if (distinct >= build->build_rows) return 1;          // all keys unique
    uint64_t fan = build->build_rows - distinct + 1;
    if (fan > kHJMaxChainLen) fan = kHJMaxChainLen;
    if (fan > build->build_rows) fan = build->build_rows;
    assert(fan >= 1 && fan <= build->build_rows);
    return fan;
}

// Decimal128 scale precondition: assert every Decimal128 key column's probe
// scale matches the build scale captured at build time. Equal nominal values
// stored at different scales (e.g. 12.30@2 vs 12.300@3) have different integer
// mantissas, so a silent scale mismatch would make them spuriously UNEQUAL (or,
// after a rescale elsewhere, spuriously equal). The join requires the caller to
// align decimal scales BEFORE the join (Tiger Style: assert the invariant; the
// caller's planner owns the rescale). Returns true iff all scales match.
BOLT_FORCE_INLINE bool join_decimal_scales_match(const JoinBuildTyped* build,
                                                 const BoltColumn* probe_keys)
                                                 noexcept {
    assert(build != nullptr);
    assert(probe_keys != nullptr || build->n_keys == 0);
    for (uint8_t k = 0; k < build->n_keys; ++k) {
        if (build->key_types[k] != BoltType::Decimal128) continue;
        assert(probe_keys[k].type == BoltType::Decimal128);
        if (probe_keys[k].decimal_scale != build->key_scales[k]) return false;
    }
    return true;
}

// OneInt64 fast-path probe: single integer key, raw-value SwissTable lookup
// (no GbCell16, no composite hash fold). Mirrors bolt_hashjoin.h's int64 probe
// but multi-match over the chain. NULL keys skipped (handled by caller's
// validity). Emits one (build_idx, probe_idx) per chain hit.
//
// UseBloom (W-J4): tests the build's split-block bloom on the mixed hash
// BEFORE the SwissTable find — one 256-bit load vs a table walk — so at low
// match rates most probe rows never touch the table. Compile-time flag (one
// instantiation per arm, zero per-row dispatch); caller gates on
// build->has_bloom.
template <bool UseBloom, bool Unique = false>
BOLT_FORCE_INLINE size_t jk_probe_one_int64(const JoinBuildTyped* build,
                                            const BoltColumn* probe_keys,
                                            int64_t n_probe,
                                            int32_t* BOLT_RESTRICT out_build,
                                            int32_t* BOLT_RESTRICT out_probe,
                                            size_t pairs_cap) noexcept {
    assert(build != nullptr && probe_keys != nullptr);
    assert(jk_is_int64_shape_type(probe_keys[0].type));
    assert(!UseBloom || build->has_bloom != 0);
    (void)pairs_cap;
    const BoltColumn& pk = probe_keys[0];
    size_t out = 0;
    for (int64_t r = 0; r < n_probe; ++r) {
        if (!gb_detail::cell_valid(pk, r)) continue;  // NULL != NULL
        const uint64_t key   = jk_read_int64_key(pk, r);
        const uint64_t mixed = swiss_mix(key);
        if constexpr (UseBloom) {
            if (!sbbf_test(build->bloom, mixed)) continue;  // definitely absent
        }
        const uint32_t p = hj_partition_of(mixed);
        int32_t  node = build->partitions[p].find(key);
        if constexpr (Unique) {
            // UNIQUE build side (jk_max_fanout()==1 — every key distinct, the
            // PK build every TPC-H join has): a find() hit IS the single
            // match, and every build init site stores build_idx == slot, so
            // the hit emits the slot directly. The chain_nodes array — one
            // extra dependent cache line per hit in the general loop, two
            // counting the next-link read that only ever says -1 — is never
            // touched. (DaMoN'24 unchained-layout reading applied to this
            // probe: fewer dependent lines per probe, same output.)
            if (node >= 0) {
                assert(out < pairs_cap);
                assert(build->chain_nodes[static_cast<uint32_t>(node)].next < 0);
                out_build[out] = node;
                out_probe[out] = static_cast<int32_t>(r);
                ++out;
            }
            continue;
        }
        uint32_t walk = 0;
        while (node >= 0) {
            assert(walk < kHJMaxChainLen);
            const uint32_t slot = static_cast<uint32_t>(node);
            // Build stored the raw value as the table key, so a find() hit on
            // `key` is already an exact key match — no per-row cell compare.
            assert(out < pairs_cap);
            out_build[out] = static_cast<int32_t>(build->chain_nodes[slot].build_idx);
            out_probe[out] = static_cast<int32_t>(r);
            ++out;
            node = build->chain_nodes[slot].next;
            ++walk;
        }
    }
    return out;
}

// ===========================================================================
// Windowed probe — memory-level parallelism (2026-08-09 perf campaign).
//
// The scalar loop above is one long dependent chain per row: hash -> bloom
// block -> ctrl line -> slot line -> chain node -> emit. Every link is a load
// that cannot issue until the previous one retires, so the core sustains ~ONE
// outstanding miss where Apple M-series can hold 10-16. Measured on TPC-H
// SF10 (60M lineitem probe rows, 12 workers): 15.2 core-ns/row against an
// L2-resident 100K build vs ~88-99 against a 2M-15M one — the out-of-cache
// figure is a full DRAM round trip per row, i.e. no overlap at all.
//
// This variant keeps the hash table EXACTLY as it is (the layout rewrite is a
// separate, much larger change) and instead splits the loop into phases over a
// window of W rows:
//     A  hash the window   (pure ALU, no table touch) + bloom-block prefetch
//     B  bloom test, compacting survivors            (the prefetched lines)
//     C  prefetch every survivor's ctrl/slot line    (W independent misses)
//     D  resolve + emit                              (lines already in flight)
// The misses in C are mutually independent, so they overlap; D then reads
// mostly-resident lines.
//
// EMISSION IS IDENTICAL, not merely equivalent: rows are visited in ascending
// order within a window and windows in ascending order, so the (build_idx,
// probe_idx) SEQUENCE matches the scalar path pair for pair.
// ===========================================================================

// Scratch window. 32 rows x 20 B = 640 B of stack, bounded by construction;
// W is a template parameter so the phase loops have compile-time trip counts.
inline constexpr uint32_t kJkProbeWindowMax = 64;

// Phase A — hash `[base,end)` into the window arrays, dropping NULL keys
// (NULL != NULL, same as the scalar loop's `continue`). Returns the count.
template <bool UseBloom>
BOLT_FORCE_INLINE uint32_t jk_window_hash(const JoinBuildTyped* build,
                                          const BoltColumn& pk,
                                          int64_t base, int64_t end,
                                          uint64_t* BOLT_RESTRICT w_key,
                                          uint64_t* BOLT_RESTRICT w_mix,
                                          int32_t*  BOLT_RESTRICT w_row) noexcept {
    assert(base >= 0 && end > base);
    assert(w_key != nullptr && w_mix != nullptr && w_row != nullptr);
    assert(end - base <= static_cast<int64_t>(kJkProbeWindowMax));
    (void)build;
    uint32_t n = 0;
    for (int64_t r = base; r < end; ++r) {
        if (!gb_detail::cell_valid(pk, r)) continue;   // NULL != NULL
        const uint64_t key   = jk_read_int64_key(pk, r);
        const uint64_t mixed = swiss_mix(key);
        w_key[n] = key;
        w_mix[n] = mixed;
        w_row[n] = static_cast<int32_t>(r);
        ++n;
        if constexpr (UseBloom) sbbf_prefetch(build->bloom, mixed);
    }
    assert(n <= static_cast<uint32_t>(end - base));
    return n;
}

// Phase B — bloom test over the (already prefetched) blocks, compacting the
// survivors down in place. Order-preserving, so phase D still emits in row
// order. Returns the surviving count.
BOLT_FORCE_INLINE uint32_t jk_window_bloom(const JoinBuildTyped* build,
                                           uint32_t n,
                                           uint64_t* BOLT_RESTRICT w_key,
                                           uint64_t* BOLT_RESTRICT w_mix,
                                           int32_t*  BOLT_RESTRICT w_row) noexcept {
    assert(build != nullptr && build->has_bloom != 0);
    assert(n <= kJkProbeWindowMax);
    uint32_t k = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (!sbbf_test(build->bloom, w_mix[i])) continue;  // definitely absent
        w_key[k] = w_key[i];
        w_mix[k] = w_mix[i];
        w_row[k] = w_row[i];
        ++k;
    }
    assert(k <= n);
    return k;
}

// Phase C — issue every survivor's ctrl/slot prefetch. THE point of the whole
// structure: these W loads are mutually independent, so they queue together.
BOLT_FORCE_INLINE void jk_window_prefetch(const JoinBuildTyped* build,
                                          uint32_t n,
                                          const uint64_t* BOLT_RESTRICT w_mix)
                                          noexcept {
    assert(build != nullptr);
    assert(n <= kJkProbeWindowMax);
    for (uint32_t i = 0; i < n; ++i) {
        build->partitions[hj_partition_of(w_mix[i])].prefetch_mixed(w_mix[i]);
    }
}

// Phase D — resolve and emit. `Unique` mirrors the scalar fast lane exactly
// (a find() hit IS the single match; chain_nodes never touched). Returns the
// new output cursor.
template <bool Unique>
BOLT_FORCE_INLINE size_t jk_window_resolve(const JoinBuildTyped* build,
                                           uint32_t n,
                                           const uint64_t* BOLT_RESTRICT w_key,
                                           const uint64_t* BOLT_RESTRICT w_mix,
                                           const int32_t*  BOLT_RESTRICT w_row,
                                           int32_t* BOLT_RESTRICT out_build,
                                           int32_t* BOLT_RESTRICT out_probe,
                                           size_t out, size_t pairs_cap) noexcept {
    assert(build != nullptr);
    assert(n <= kJkProbeWindowMax);
    (void)pairs_cap;
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t p = hj_partition_of(w_mix[i]);
        int32_t node = build->partitions[p].find_mixed(w_key[i], w_mix[i]);
        if constexpr (Unique) {
            if (node >= 0) {
                assert(out < pairs_cap);
                assert(build->chain_nodes[static_cast<uint32_t>(node)].next < 0);
                out_build[out] = node;
                out_probe[out] = w_row[i];
                ++out;
            }
            continue;
        }
        uint32_t walk = 0;
        while (node >= 0) {
            assert(walk < kHJMaxChainLen);
            const uint32_t slot = static_cast<uint32_t>(node);
            assert(out < pairs_cap);
            out_build[out] = static_cast<int32_t>(build->chain_nodes[slot].build_idx);
            out_probe[out] = w_row[i];
            ++out;
            node = build->chain_nodes[slot].next;
            ++walk;
        }
    }
    return out;
}

// Windowed OneInt64 probe. Same signature, same output sequence, same
// pairs_cap contract as jk_probe_one_int64 — only the memory schedule differs.
// Plain `inline` (not force-inline): it is entered once per multi-thousand-row
// block, so a real call costs nothing and the 12 template instantiations stay
// out of every caller.
template <bool UseBloom, bool Unique, uint32_t W>
inline size_t jk_probe_one_int64_mlp(const JoinBuildTyped* build,
                                     const BoltColumn* probe_keys,
                                     int64_t n_probe,
                                     int32_t* BOLT_RESTRICT out_build,
                                     int32_t* BOLT_RESTRICT out_probe,
                                     size_t pairs_cap) noexcept {
    assert(build != nullptr && probe_keys != nullptr);
    assert(jk_is_int64_shape_type(probe_keys[0].type));
    assert(!UseBloom || build->has_bloom != 0);
    static_assert(W >= 4 && W <= kJkProbeWindowMax, "window must stay in L1");
    const BoltColumn& pk = probe_keys[0];
    uint64_t w_key[W];
    uint64_t w_mix[W];
    int32_t  w_row[W];
    size_t out = 0;
    for (int64_t base = 0; base < n_probe; base += static_cast<int64_t>(W)) {
        int64_t end = base + static_cast<int64_t>(W);
        if (end > n_probe) end = n_probe;
        uint32_t n = jk_window_hash<UseBloom>(build, pk, base, end,
                                              w_key, w_mix, w_row);
        if constexpr (UseBloom) {
            n = jk_window_bloom(build, n, w_key, w_mix, w_row);
        }
        jk_window_prefetch(build, n, w_mix);
        out = jk_window_resolve<Unique>(build, n, w_key, w_mix, w_row,
                                        out_build, out_probe, out, pairs_cap);
    }
    return out;
}

// Switch + tuning knobs for the windowed probe, latched ONCE. Deliberately NOT
// a function-local static: this header sits on the probe path and a static's
// thread-safe-init guard load has already been measured as real overhead on
// this campaign.
//   CHUKONU_HJ_MLP=1              -> windowed schedule (DEFAULT OFF, see below)
//   CHUKONU_HJ_MLP_W=8|16|32|64   -> window size
//   CHUKONU_HJ_MLP_MINROWS=<n>    -> cache-residency gate (0 = always window)
//   CHUKONU_HJ_MLP_TRACE=1        -> latched proof that the arm was taken
//
// DEFAULT OFF, deliberately. At KERNEL level this is a clear, reproducible win
// (see the residency-gate note below: 1.26-1.57x on every out-of-cache build
// size TPC-H actually uses). At BOARD level it measured FLAT: TPC-H SF10 12w,
// interleaved OFF/ON/ON/OFF, rows 21/21 exact, GEO 104.78 -> 106.54 ms (+1.7%
// all-21) / -0.3% over the 19 join queries — and on that same run Q1, which
// contains NO JOIN and therefore has a true delta of exactly zero, moved
// +55.4%. The box (a working desktop with ~2.7 cores of Chrome) simply cannot
// resolve the ~5% this is worth, so turning it on by default would be adopting
// an unproven change. The campaign has already been burned twice by "obvious"
// wins that measured flat. Flip the default only after a quiet-box interleaved
// A/B shows the board move; the kernel harness that produced the ratios is
// scratchpad/probe_mlp_micro.cpp.
inline constexpr uint32_t kJkProbeWindowDefault = 32;

// Cache-residency gate. Windowing BUYS nothing when the table already sits in
// cache — there is no miss to overlap — and it costs a second pass over the
// window plus the prefetch issues. Measured at 12 threads, 24 M probe rows,
// paired within-rep ratios (W=32 vs scalar): 100 K entries / 1.6 MB **0.80x**
// (a real LOSS), 1.5 M / 24 MB 1.40x, 2 M / 32 MB 1.26x, 15 M / 240 MB 1.57x.
// So the windowed schedule engages only above a build size that cannot be
// resident.
// 262144 entries is ~8.9 MB of ctrl+slots, past a per-core slice of this
// machine's L2 once 12 workers share it, and is the SAME threshold the
// optimizer already uses to price a probe as cache-resident
// (CHUKONU_PROBE_CACHE_ROWS in memo_build_winners.inc) — one residency story,
// not two. Precedent: SwissTable::find_simd gates its prefetch on table
// capacity for exactly this reason (bolt_swiss.h).
inline constexpr uint64_t kJkMlpMinBuildRowsDefault = 262144;

inline int      g_jk_mlp_enabled  = -1;
inline uint32_t g_jk_mlp_window   = 0;
inline uint64_t g_jk_mlp_min_rows = 0;
// DEFAULT-ON RATIONALE (2026-08-09). This shipped opt-in because the only
// board runs available were on a loaded desktop where the no-join Q1 control
// swung +55% — a flat board there is not evidence. On a quiet box (control
// drift +-2.3%, 3 identical passes at 4% median spread) the A/B reads:
// TPC-H SF10 GEO 54.6 -> 53.7 ms (-1.5%), big-join subset -2.3%, Q13 -9.2%,
// 16 of 21 queries improved (sign test p ~ 0.01), 21/21 rows exact. Smaller
// than the ~5-7% the 25%-probe-share estimate suggested, because the
// residency gate correctly declines the many small/cache-resident builds --
// but real, consistent in direction, and free.
//
// CHUKONU_HJ_MLP_TRACE=1: latched proof that the windowed arm is actually
// TAKEN. This campaign has already shipped a fast path gated on a condition it
// could never satisfy (the parallel spill finalize) and no test could see it —
// a gated schedule needs a way to say out loud that it engaged.
inline int      g_jk_mlp_trace    = 0;
inline int      g_jk_mlp_shown    = 0;

inline void jk_mlp_resolve_env() noexcept {
    const char* e = std::getenv("CHUKONU_HJ_MLP");
    const char* w = std::getenv("CHUKONU_HJ_MLP_W");
    const char* m = std::getenv("CHUKONU_HJ_MLP_MINROWS");
    const char* t = std::getenv("CHUKONU_HJ_MLP_TRACE");
    g_jk_mlp_trace = (t != nullptr && t[0] == '1') ? 1 : 0;
    uint32_t win = kJkProbeWindowDefault;
    if (w != nullptr) {
        const int v = std::atoi(w);
        if (v == 8 || v == 16 || v == 32 || v == 64) win = static_cast<uint32_t>(v);
    }
    uint64_t minr = kJkMlpMinBuildRowsDefault;
    if (m != nullptr) {
        const long long v = std::atoll(m);
        if (v >= 0) minr = static_cast<uint64_t>(v);
    }
    g_jk_mlp_window   = win;
    g_jk_mlp_min_rows = minr;
    // Default ON as of the 2026-08-09 quiet-box A/B (see below);
    // CHUKONU_HJ_MLP=0 restores the scalar probe.
    g_jk_mlp_enabled  = (e != nullptr && e[0] == '0') ? 0 : 1;
}

// One latched read per kernel call (thousands of rows), never per row.
BOLT_FORCE_INLINE bool jk_mlp_enabled(const JoinBuildTyped* build) noexcept {
    assert(build != nullptr);
    if (g_jk_mlp_enabled < 0) jk_mlp_resolve_env();
    assert(g_jk_mlp_window == 8 || g_jk_mlp_window == 16 ||
           g_jk_mlp_window == 32 || g_jk_mlp_window == 64);
    return g_jk_mlp_enabled == 1 && build->build_rows >= g_jk_mlp_min_rows;
}

// TEST hook: force the schedule within one process, so an equivalence test can
// probe the SAME build with both paths and compare pair sequences. Sets the
// residency gate to 0 so a small fixture still exercises the windowed path.
inline void jk_mlp_force(bool enabled, uint32_t window) noexcept {
    assert(window == 8 || window == 16 || window == 32 || window == 64);
    g_jk_mlp_window   = window;
    g_jk_mlp_min_rows = 0;
    g_jk_mlp_enabled  = enabled ? 1 : 0;
}

// Dispatch the windowed probe on the latched window size — once per kernel
// call (thousands of rows), never per row.
template <bool UseBloom, bool Unique>
inline size_t jk_probe_one_int64_windowed(const JoinBuildTyped* build,
                                          const BoltColumn* probe_keys,
                                          int64_t n_probe,
                                          int32_t* BOLT_RESTRICT out_build,
                                          int32_t* BOLT_RESTRICT out_probe,
                                          size_t pairs_cap) noexcept {
    assert(build != nullptr && probe_keys != nullptr);
    assert(g_jk_mlp_window != 0);
    if (g_jk_mlp_trace != 0 && g_jk_mlp_shown < 4) {
        ++g_jk_mlp_shown;
        std::fprintf(stderr, "[hjmlp] windowed probe ENGAGED: W=%u "
                     "build_rows=%llu unique=%d bloom=%d\n", g_jk_mlp_window,
                     static_cast<unsigned long long>(build->build_rows),
                     Unique ? 1 : 0, UseBloom ? 1 : 0);
    }
    if (g_jk_mlp_window == 32) {
        return jk_probe_one_int64_mlp<UseBloom, Unique, 32>(
            build, probe_keys, n_probe, out_build, out_probe, pairs_cap);
    }
    if (g_jk_mlp_window == 64) {
        return jk_probe_one_int64_mlp<UseBloom, Unique, 64>(
            build, probe_keys, n_probe, out_build, out_probe, pairs_cap);
    }
    if (g_jk_mlp_window == 8) {
        return jk_probe_one_int64_mlp<UseBloom, Unique, 8>(
            build, probe_keys, n_probe, out_build, out_probe, pairs_cap);
    }
    return jk_probe_one_int64_mlp<UseBloom, Unique, 16>(
        build, probe_keys, n_probe, out_build, out_probe, pairs_cap);
}

// General typed probe: composite hash + typed (Float64-canonical, Utf8
// byte-resolved, Decimal128 raw-128-bit) equality on hit. UseBloom as above
// (tests the composite hash, which the loop computes anyway).
template <bool UseBloom>
BOLT_FORCE_INLINE size_t jk_probe_general(const JoinBuildTyped* build,
                                          const BoltColumn* probe_keys,
                                          int64_t n_probe,
                                          int32_t* BOLT_RESTRICT out_build,
                                          int32_t* BOLT_RESTRICT out_probe,
                                          size_t pairs_cap) noexcept {
    assert(build != nullptr);
    assert(out_build != nullptr && out_probe != nullptr);
    assert(!UseBloom || build->has_bloom != 0);
    (void)pairs_cap;
    const uint8_t nk = build->n_keys;
    size_t out = 0;
    for (int64_t r = 0; r < n_probe; ++r) {
        if (join_row_has_null(probe_keys, nk, r)) continue;  // NULL != NULL
        const uint64_t h = jk_detail::hash_keys_typed(probe_keys, nk, r);
        if constexpr (UseBloom) {
            if (!sbbf_test(build->bloom, h)) continue;  // definitely absent
        }
        const uint32_t p = hj_partition_of(h);
        int32_t  node = build->partitions[p].find(h);
        uint32_t walk = 0;
        while (node >= 0) {
            assert(walk < kHJMaxChainLen);
            const uint32_t slot = static_cast<uint32_t>(node);
            if (jk_detail::keys_equal_typed(probe_keys, nk, r, build->keys_flat,
                                            slot, build->key_types,
                                            build->key_str_base)) {
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

// INNER multi-match probe: for each probe row walk its chain and emit every
// key-equal (build_idx, probe_idx) pair. Does NOT touch matched[] — the caller
// sets matched[build_idx] only for pairs that also pass its residual, so the
// OUTER/SEMI/ANTI drain is residual-correct. Over-cap is a hard assert; the
// caller pre-sizes pairs_cap and chunks the probe morsel stream.
//
// Dispatches once on the build's cached key shape: a single integer key takes
// the raw-value fast path (jk_probe_one_int64); everything else takes the typed
// general path. Behaviour is identical across both for integer keys.
//
// `use_bloom` (W-J4, default off = byte-identical legacy behaviour): consult
// the build-side split-block bloom before each table lookup. EMITS THE SAME
// PAIRS either way (the bloom has no false negatives); only the lookup cost
// changes. Profitable when the probe match rate is low (misses skip the table
// walk for one 256-bit load); pure overhead near 100% match — the CALLER
// decides, ideally adaptively from observed pair/row rates, NOT from
// optimizer estimates.
inline size_t join_probe_typed(const JoinBuildTyped* build,
                               const BoltColumn* probe_keys, int64_t n_probe,
                               int32_t* BOLT_RESTRICT out_build,
                               int32_t* BOLT_RESTRICT out_probe,
                               size_t pairs_cap,
                               bool use_bloom = false) noexcept {
    assert(build != nullptr);
    assert(out_build != nullptr && out_probe != nullptr);
    // Decimal128 scale invariant (no-op unless a key is Decimal128).
    assert(join_decimal_scales_match(build, probe_keys));
    const bool bloom = use_bloom && build->has_bloom != 0;
    if (build->key_shape == static_cast<uint8_t>(JoinKeyShape::OneInt64)) {
        // jk_max_fanout is a 64-entry sum, paid once per multi-thousand-row
        // probe call — noise next to one probe's cache misses.
        const bool uniq = (jk_max_fanout(build) == 1);
        // Windowed (memory-level-parallel) schedule; identical pair sequence.
        // One latched read per kernel call, never per row.
        if (jk_mlp_enabled(build)) {
            if (uniq) {
                return bloom
                    ? jk_probe_one_int64_windowed<true, true>(
                          build, probe_keys, n_probe, out_build, out_probe, pairs_cap)
                    : jk_probe_one_int64_windowed<false, true>(
                          build, probe_keys, n_probe, out_build, out_probe, pairs_cap);
            }
            return bloom
                ? jk_probe_one_int64_windowed<true, false>(
                      build, probe_keys, n_probe, out_build, out_probe, pairs_cap)
                : jk_probe_one_int64_windowed<false, false>(
                      build, probe_keys, n_probe, out_build, out_probe, pairs_cap);
        }
        if (uniq) {
            return bloom
                ? jk_probe_one_int64<true, true>(build, probe_keys, n_probe,
                                                 out_build, out_probe, pairs_cap)
                : jk_probe_one_int64<false, true>(build, probe_keys, n_probe,
                                                  out_build, out_probe, pairs_cap);
        }
        return bloom ? jk_probe_one_int64<true>(build, probe_keys, n_probe,
                                                out_build, out_probe, pairs_cap)
                     : jk_probe_one_int64<false>(build, probe_keys, n_probe,
                                                 out_build, out_probe, pairs_cap);
    }
    return bloom ? jk_probe_general<true>(build, probe_keys, n_probe,
                                          out_build, out_probe, pairs_cap)
                 : jk_probe_general<false>(build, probe_keys, n_probe,
                                           out_build, out_probe, pairs_cap);
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
