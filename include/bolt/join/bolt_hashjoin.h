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

}  // namespace bolt
