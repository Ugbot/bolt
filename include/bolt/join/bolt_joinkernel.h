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
        out->chain_nodes[slot].next = out->partitions[p].find(tkey);  // prev head/-1
        if (!out->partitions[p].insert(tkey, slot)) return false;
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
template <bool UseBloom>
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
