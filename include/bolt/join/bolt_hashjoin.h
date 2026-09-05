// bolt_hashjoin.h — Partitioned hash join (Tiger Style, arena-allocated).
//
// Build phase:
//   - Hash all build keys.
//   - Radix-partition by top 6 hash bits → 64 partitions.
//   - One SwissTable per partition, sized from the partition's row count.
//   - Optional Bloom filter populated in-line when `build(..., build_bloom=true)`.
// Probe phase:
//   - Default: `probe()` — hash each probe key, dispatch to its partition's
//     SwissTable, emit any (build_idx, probe_idx) pair that matches.
//   - Low-match-rate opt-in: `probe_with_bloom()` — identical but gates each
//     probe on `bloom_test` before the SwissTable lookup.  Caller-selected;
//     zero runtime branch in either path (templated `probe_impl<bool>`).
//     Measured on Q3 (tpch_lite, ~100% match) Bloom is a net regression
//     (~8%) — only enable when the expected match rate is low.  See
//     `docs/research/design-log.md` "Hash-join Bloom pre-screen (C2)".
//
// Phase 3 scope: keys are int64_t / uint64_t columns in Flat format.

#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_port.h"
#include "bolt/bolt_types.h"
#include "bolt/join/bolt_bloom.h"
#include "bolt/join/bolt_swiss.h"

#include <cassert>
#include <cstdint>
#include <cstring>

namespace bolt {

inline constexpr uint32_t kHJPartitionBits  = 6;
inline constexpr uint32_t kHJNumPartitions  = 1u << kHJPartitionBits;  // 64

// Extract top `kHJPartitionBits` of the mixed 64-bit hash.
BOLT_FORCE_INLINE uint32_t hj_partition_of(uint64_t mixed) noexcept {
    return static_cast<uint32_t>(mixed >> (64 - kHJPartitionBits));
}

// ---------------------------------------------------------------------------
// Build side
// ---------------------------------------------------------------------------
struct HashJoinBuild {
    SwissTable  partitions[kHJNumPartitions];
    BloomFilter bloom;        // populated only when build(..., build_bloom=true)
    int64_t     build_rows;   // total rows inserted (for assertions)
    bool        has_bloom;    // true iff `bloom` was populated

    // Ingest an int64/uint64 flat column and partition-build. Returns false on
    // OOM or unsupported column shape. Set `build_bloom=true` to populate the
    // Bloom filter pre-screen for `probe_with_bloom()` (adds one O(n) pass
    // cost on build — same loop — and ~`16 * n` bits of arena).
    bool build(const BoltColumn& build_keys, Arena* arena,
               bool build_bloom = false) noexcept {
        assert(arena != nullptr);
        assert(build_keys.format == ColumnFormat::Flat ||
               build_keys.format == ColumnFormat::View);
        if (build_keys.type != BoltType::Int64 &&
            build_keys.type != BoltType::UInt64) return false;
        if (build_keys.format != ColumnFormat::Flat &&
            build_keys.format != ColumnFormat::View) return false;

        const int64_t n = build_keys.length;
        const uint64_t* keys = static_cast<const uint64_t*>(build_keys.data);

        // Pass 1: count per partition so each SwissTable can be sized once.
        // When Bloom is requested we fold bloom_add into the same pass so we
        // don't walk the keys twice (the cost is ~1 OR + 1 store per key).
        uint32_t counts[kHJNumPartitions];
        memset(counts, 0, sizeof(counts));
        assert(n >= 0 && n < (1LL << 40));  // 1T rows cap

        has_bloom = false;
        if (build_bloom) {
            if (!bloom_create(&bloom, n, arena)) return false;
            has_bloom = true;
        }

        for (int64_t i = 0; i < n; ++i) {
            uint64_t h = swiss_mix(keys[i]);
            counts[hj_partition_of(h)]++;
            if (build_bloom) bloom_add(bloom, h);
        }

        // Size each partition to its count (SwissTable::create doubles for
        // load factor). Empty partitions still get a minimum table.
        for (uint32_t p = 0; p < kHJNumPartitions; ++p) {
            uint64_t hint = counts[p] == 0 ? 1 : counts[p];
            if (!SwissTable::create(&partitions[p], hint, arena)) return false;
        }

        // Pass 2: insert.
        for (int64_t i = 0; i < n; ++i) {
            uint64_t h = swiss_mix(keys[i]);
            uint32_t p = hj_partition_of(h);
            assert(p < kHJNumPartitions);
            if (!partitions[p].insert(keys[i], static_cast<uint32_t>(i))) {
                return false;
            }
        }
        build_rows = n;
        return true;
    }
};

// ---------------------------------------------------------------------------
// Probe side — two named variants share a templated implementation.
// `probe()` is the default (no Bloom); `probe_with_bloom()` requires the
// build side to have been constructed with `build_bloom=true`.
// ---------------------------------------------------------------------------
struct HashJoinProbe {
    // Default inner equi-join — unchanged contract.
    static int64_t probe(const HashJoinBuild& build_side,
                         const BoltColumn& probe_keys,
                         int32_t* BOLT_RESTRICT out_build_idx,
                         int32_t* BOLT_RESTRICT out_probe_idx) noexcept {
        return probe_impl<false>(build_side, probe_keys,
                                 out_build_idx, out_probe_idx);
    }

    // Bloom-gated variant — call when expected match rate is low.
    // Requires `build_side.has_bloom == true` (enforced by assert).
    static int64_t probe_with_bloom(const HashJoinBuild& build_side,
                                    const BoltColumn& probe_keys,
                                    int32_t* BOLT_RESTRICT out_build_idx,
                                    int32_t* BOLT_RESTRICT out_probe_idx) noexcept {
        assert(build_side.has_bloom);
        return probe_impl<true>(build_side, probe_keys,
                                out_build_idx, out_probe_idx);
    }

private:
    template <bool UseBloom>
    static int64_t probe_impl(const HashJoinBuild& build_side,
                              const BoltColumn& probe_keys,
                              int32_t* BOLT_RESTRICT out_build_idx,
                              int32_t* BOLT_RESTRICT out_probe_idx) noexcept {
        assert(out_build_idx != nullptr);
        assert(out_probe_idx != nullptr);
        assert(probe_keys.format == ColumnFormat::Flat ||
               probe_keys.format == ColumnFormat::View);
        if (probe_keys.type != BoltType::Int64 &&
            probe_keys.type != BoltType::UInt64) return 0;

        const int64_t n = probe_keys.length;
        const uint64_t* keys = static_cast<const uint64_t*>(probe_keys.data);
        assert(n >= 0);

        int64_t count = 0;
        int64_t i = 0;

#if BOLT_SIMD_AVX2 || BOLT_SIMD_SSE42 || BOLT_SIMD_NEON
        count = probe_simd_block<UseBloom>(build_side, keys, n, out_build_idx,
                                           out_probe_idx, &i);
#endif

        // Scalar tail (and full path for fallback ISA): per-row branch.
        assert(i >= 0 && i <= n);
        for (; i < n; ++i) {
            uint64_t k = keys[i];
            uint64_t h = swiss_mix(k);
            int32_t bi = -1;
            if constexpr (UseBloom) {
                if (bloom_test(build_side.bloom, h)) {
                    uint32_t p = hj_partition_of(h);
                    bi = build_side.partitions[p].find(k);
                }
            } else {
                uint32_t p = hj_partition_of(h);
                bi = build_side.partitions[p].find(k);
            }
            bool hit = (bi >= 0);
            out_build_idx[count] = bi;
            out_probe_idx[count] = static_cast<int32_t>(i);
            count += hit ? 1 : 0;
        }
        return count;
    }

#if BOLT_SIMD_AVX2 || BOLT_SIMD_SSE42 || BOLT_SIMD_NEON
    // Branchless SIMD emit: process L probes, build a hit-mask vector, and
    // compress-store build_idx + probe_idx in lockstep. The per-key lookup
    // via SwissTable::find remains scalar (probe chains must branch).
    // Returns pairs emitted; updates *pi to first un-processed row.
    template <bool UseBloom>
    static int64_t probe_simd_block(const HashJoinBuild& build_side,
                                    const uint64_t* BOLT_RESTRICT keys,
                                    int64_t n,
                                    int32_t* BOLT_RESTRICT out_build_idx,
                                    int32_t* BOLT_RESTRICT out_probe_idx,
                                    int64_t* pi) noexcept {
        assert(keys != nullptr || n == 0);
        assert(pi != nullptr);
        assert(*pi == 0);
        assert(n >= 0);

        using namespace bolt::simd;
        constexpr int L = bmm_lanes_i32;
        int64_t count = 0;
        int64_t i = 0;
        alignas(32) int32_t build_buf[16];  // 16 >= L on every ISA.
        alignas(32) int32_t probe_buf[16];
        const bmm_vec_i32 v_neg1 = bmm_set1_i32(-1);

        for (; i + L <= n; i += L) {
            for (int k = 0; k < L; ++k) {
                uint64_t key = keys[i + k];
                uint64_t h   = swiss_mix(key);
                int32_t  bi  = -1;
                if constexpr (UseBloom) {
                    if (bloom_test(build_side.bloom, h)) {
                        uint32_t p = hj_partition_of(h);
                        bi = build_side.partitions[p].find(key);
                    }
                } else {
                    uint32_t p = hj_partition_of(h);
                    bi = build_side.partitions[p].find(key);
                }
                build_buf[k] = bi;
                probe_buf[k] = static_cast<int32_t>(i) + k;
            }
            bmm_vec_i32 v_build = bmm_loadu_i32(build_buf);
            bmm_vec_i32 v_probe = bmm_loadu_i32(probe_buf);
            // hit <=> build_idx > -1 (valid index is >= 0).
            bmm_vec_i32 v_hits  = bmm_cmpgt_i32(v_build, v_neg1);
            uint32_t    mask    = static_cast<uint32_t>(bmm_movemask_i32(v_hits));
            uint32_t wb = bmm_compressstore_i32(out_build_idx + count, v_build, mask);
            uint32_t wp = bmm_compressstore_i32(out_probe_idx + count, v_probe, mask);
            assert(wb == wp);
            count += wb;
            (void)wp;
        }
        *pi = i;
        return count;
    }
#endif
};

// ---------------------------------------------------------------------------
// Card K-JOIN-MM — chained build + multi-match probe + composite key (N≤8).
//
// Rationale: the default `HashJoinBuild` stores hash→slot in a SwissTable,
// so duplicate-key inserts overwrite — only one build row per key survives.
// Q13 (customer⨝orders avg 10 orders/customer), and the foundation test
// `test_sql_join_on_int64`, both require multi-match probe (one probe row
// emits one output per matching build row). This API adds a chained build:
// each bucket points at the head of a singly-linked list of build-row ids;
// the probe walks the chain and emits one (build_idx, probe_idx) per match.
//
// Composite key: up to `kHJMaxKeys = 8` int64 columns. Hash is `swiss_mix`
// folded over each key with a wyhash3-style mix. Compare on hit is N inline
// integer compares reduced with branchless AND.
//
// Chain arena: caller-owned slab sized at init from
// `build_rows × expected_dup_factor` (default 4). Over-cap returns false
// from build_chained (no silent growth — Tiger Style).
// ---------------------------------------------------------------------------

inline constexpr uint8_t  kHJMaxKeys                = 8;
inline constexpr uint32_t kHJDefaultExpectedDupFact = 4;
inline constexpr uint32_t kHJMaxChainLen            = 4096;

// Sentinel returned by every multi-match probe that would otherwise write past
// its caller's `pairs_cap`. SIZE_MAX is not a reachable pair count (the buffer
// it indexes could not exist), so no in-band value is stolen.
inline constexpr size_t kHJProbeOverflow = static_cast<size_t>(-1);

struct HashJoinChainNode {
    uint32_t build_idx;   // build-side row id
    int32_t  next;        // index into chain_nodes[], or -1 = end
};

struct HashJoinBuildCfgChained {
    BoltType keys[kHJMaxKeys];          // composite key types (Int64/UInt64)
    uint8_t  n_keys;                    // 1..kHJMaxKeys
    uint8_t  _pad[7];
    uint64_t build_rows;                // total build rows (for size hint)
    uint32_t expected_dup_factor;       // chain-arena multiplier; default 4
    uint32_t max_chain_len;             // assertion cap; default kHJMaxChainLen
};

struct HashJoinBuildChained {
    SwissTable        partitions[kHJNumPartitions];  // hash → head chain idx
    HashJoinChainNode* chain_nodes;                  // per-shard slab
    uint32_t          chain_size;                    // live node count
    uint32_t          chain_cap;                     // hard cap (fixed at init)
    uint64_t          build_rows;
    uint8_t           n_keys;
    uint8_t           _pad[7];
};

// Hash N composite int64 keys row-by-row. Folded wyhash3-style mix.
BOLT_FORCE_INLINE uint64_t hj_composite_hash_int64(
        const uint64_t* const* BOLT_RESTRICT key_cols,
        uint8_t  n_keys,
        int64_t  row) noexcept {
    assert(key_cols != nullptr);
    assert(n_keys >= 1 && n_keys <= kHJMaxKeys);
    uint64_t h = 0x9E3779B97F4A7C15ULL;  // golden-ratio seed
    for (uint8_t k = 0; k < n_keys; ++k) {
        h ^= swiss_mix(key_cols[k][row]);
        h  = swiss_mix(h);
    }
    return h;
}

// Branchless N-key Int64 equality at (build_row, probe_row).
BOLT_FORCE_INLINE bool hj_keys_equal_int64(
        const uint64_t* const* BOLT_RESTRICT lkeys,
        const uint64_t* const* BOLT_RESTRICT rkeys,
        uint8_t  n_keys,
        int64_t  lrow,
        int64_t  rrow) noexcept {
    assert(lkeys != nullptr && rkeys != nullptr);
    assert(n_keys >= 1 && n_keys <= kHJMaxKeys);
    // Branchless AND-reduction of per-key compares.
    uint32_t acc = 0;
    for (uint8_t k = 0; k < n_keys; ++k) {
        acc += static_cast<uint32_t>(lkeys[k][lrow] == rkeys[k][rrow]);
    }
    return acc == n_keys;
}

// Build the chained hash table. `key_cols[k]` is the int64 column data
// pointer for the k-th composite-key column on the build side.
inline bool hash_join_build_chained(
        const uint64_t* const* BOLT_RESTRICT key_cols,
        HashJoinBuildCfgChained cfg,
        Arena*                   arena,
        HashJoinBuildChained*    out) noexcept {
    assert(key_cols != nullptr);
    assert(arena != nullptr && out != nullptr);
    assert(cfg.n_keys >= 1 && cfg.n_keys <= kHJMaxKeys);
    const uint64_t n = cfg.build_rows;
    // `next` is an int32 index into chain_nodes[] and SwissTable::find()
    // returns int32 with -1 == absent, so a row id >= 2^31 aliases the
    // end-of-chain sentinel and loses matches silently. Refuse past the
    // structural limit (bolt_joinkernel.h, kJkMaxBuildRows) rather than
    // build a table that answers confidently and wrongly.
    if (n > 0x7FFFFFFFull) return false;
    out->chain_cap = static_cast<uint32_t>(n);  // exactly one node per row
    out->chain_nodes = static_cast<HashJoinChainNode*>(
        arena->allocate(sizeof(HashJoinChainNode) *
                            (n == 0 ? 1 : static_cast<size_t>(n)),
                        alignof(HashJoinChainNode)));
    if (out->chain_nodes == nullptr) return false;
    out->chain_size = 0;
    out->build_rows = n;
    out->n_keys     = cfg.n_keys;
    uint32_t counts[kHJNumPartitions]; memset(counts, 0, sizeof(counts));
    for (uint64_t i = 0; i < n; ++i) {
        const uint64_t h = hj_composite_hash_int64(key_cols, cfg.n_keys,
                                                   static_cast<int64_t>(i));
        counts[hj_partition_of(h)]++;
    }
    for (uint32_t p = 0; p < kHJNumPartitions; ++p) {
        const uint64_t hint = counts[p] == 0 ? 1 : counts[p];
        if (!SwissTable::create(&out->partitions[p], hint, arena)) return false;
    }
    for (uint64_t i = 0; i < n; ++i) {
        const uint64_t h = hj_composite_hash_int64(key_cols, cfg.n_keys,
                                                   static_cast<int64_t>(i));
        const uint32_t p = hj_partition_of(h);
        const int32_t prev_head = out->partitions[p].find(h);
        const uint32_t node_idx = out->chain_size++;
        // G2GRAPH-27: was `assert(node_idx < out->chain_cap)` -- an assert
        // standing directly in front of a write into chain_nodes[], i.e. absent
        // from any -DNDEBUG build. The arena is sized build_rows x dup_factor
        // so this cannot trip for a well-formed cfg; over-cap returns false
        // ("no silent growth", this header's own contract) rather than writing.
        if (node_idx >= out->chain_cap) return false;
        out->chain_nodes[node_idx].build_idx = static_cast<uint32_t>(i);
        out->chain_nodes[node_idx].next      = prev_head;
        if (!out->partitions[p].insert(h, node_idx)) return false;
    }
    return true;
}

// Multi-match probe: walk each probe row's chain and emit ALL matching
// (build_idx, probe_idx) pairs. Returns total pairs written, or
// kHJProbeOverflow if the emit would pass `pairs_cap` / the chain walk exceeds
// the build's structural node count (one node per build row, so a longer walk
// means a corrupt `next` link). Nothing is written past the cap; the caller
// pre-sizes `pairs_cap` (default: probe_rows × 8) and MUST test the sentinel.
// G2GRAPH-27: both bounds were `assert`s, i.e. absent from any -DNDEBUG build,
// which left an out-of-bounds write on a skewed build side.
inline size_t hash_join_probe_multi_match(
        const HashJoinBuildChained* build_side,
        const uint64_t* const* BOLT_RESTRICT lkeys,
        const uint64_t* const* BOLT_RESTRICT rkeys,
        size_t   n_probe_rows,
        uint32_t* BOLT_RESTRICT pairs_out_build,
        uint32_t* BOLT_RESTRICT pairs_out_probe,
        size_t   pairs_cap) noexcept {
    assert(build_side != nullptr);
    assert(lkeys != nullptr && rkeys != nullptr);
    assert(pairs_out_build != nullptr && pairs_out_probe != nullptr);
    const uint8_t nk = build_side->n_keys;
    // One chain node per build row: a longer walk is a corrupt `next` link.
    const uint64_t walk_limit = build_side->build_rows;
    size_t out = 0;
    for (size_t r = 0; r < n_probe_rows; ++r) {
        const uint64_t h = hj_composite_hash_int64(rkeys, nk,
                                                   static_cast<int64_t>(r));
        const uint32_t p = hj_partition_of(h);
        int32_t node = build_side->partitions[p].find(h);
        uint64_t chain_len = 0;
        while (node >= 0) {
            if (chain_len >= walk_limit) return kHJProbeOverflow;
            const HashJoinChainNode& nd =
                build_side->chain_nodes[static_cast<uint32_t>(node)];
            const uint32_t bi = nd.build_idx;
            if (hj_keys_equal_int64(lkeys, rkeys, nk,
                                    static_cast<int64_t>(bi),
                                    static_cast<int64_t>(r))) {
                if (out >= pairs_cap) return kHJProbeOverflow;
                pairs_out_build[out] = bi;
                pairs_out_probe[out] = static_cast<uint32_t>(r);
                ++out;
            }
            node = nd.next;
            ++chain_len;
        }
    }
    return out;
}

}  // namespace bolt
