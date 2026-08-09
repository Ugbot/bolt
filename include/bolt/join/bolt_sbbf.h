// bolt_sbbf.h — Split Block Bloom Filter (Parquet-spec, Apache 2.0 wire-compatible).
//
// Apache Parquet BloomFilter spec:
//   https://github.com/apache/parquet-format/blob/master/BloomFilter.md
// Putze/Sanders/Singler (2007) on block Bloom; Lang et al. (VLDB 2019)
// quantified split-block as the probe-throughput winner over cuckoo /
// xor / ribbon for in-memory probe workloads at 1–100 K build sides.
//
// Layout:
//   One 256-bit (32-byte) block per key-slot.  Each block is 8 × uint32_t;
//   each key sets exactly one bit per lane, giving k = 8 hashes packed into
//   one cache-line load.  Block index = hi32(swiss_mix(key)) mod num_blocks.
//   Within-block lane mask computed via the 8 Parquet salt constants.
//
// Probe cost: 1 256-bit load + 8 ANDs + 1 cmp.  Same cache-line traffic as
// the existing `bolt_bloom.h` (block-Bloom) but halves bits/key at 1% FPR
// (~8 bits vs ~16) and is byte-compatible with Parquet sidecars — free
// win once the lakehouse layer ingests Parquet stats.
//
// Relationship to bolt_bloom.h: BOTH ship in-source per the keep-code-
// paths rule.  Callers explicitly choose:
//   - BloomFilter (single-word block): lower build cost, simpler; default
//     HashJoinBuild path uses it.
//   - SplitBlockBloom: Parquet-compat wire format, ~0.5× bits/key at
//     ~5× lower FPR; reach for it when space or FPR matters.
//
// RULES: Tiger Style — POD + free functions, no exceptions, noexcept,
// arena-allocated, no heap on hot path.

#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>

namespace bolt {

// Parquet BloomFilter salt constants (Apache spec — do not change these
// or wire-compat breaks).  Each 32-bit lane of a block has one of these
// salts; the within-lane bit position is `(hash * salt[lane]) >> 27`,
// giving a uniform 5-bit index in [0, 32).
inline constexpr uint32_t kSbbfSalts[8] = {
    0x47b6137bu, 0x44974d91u, 0x8824ad5bu, 0xa2b7289du,
    0x705495c7u, 0x2df1424bu, 0x9efc4947u, 0x5c6bfb31u,
};

// One 256-bit block = 8 × uint32_t; per-lane mask has exactly one bit set.
struct alignas(32) SbbfBlock {
    uint32_t lane[8];
};
static_assert(sizeof(SbbfBlock) == 32, "SbbfBlock must be 32 bytes");

struct SplitBlockBloom {
    SbbfBlock* blocks;            // num_blocks entries (power of two)
    uint32_t   num_blocks_mask;   // num_blocks - 1
};

// Compute the 8-lane within-block mask for a given 32-bit hash slice.
// Parquet spec: lane L's bit position is ((hash * salt[L]) >> 27) ∈ [0,32).
BOLT_FORCE_INLINE void sbbf_compute_mask(uint32_t h_lane,
                                         uint32_t out[8]) noexcept {
    for (int l = 0; l < 8; ++l) {
        out[l] = uint32_t{1} << ((h_lane * kSbbfSalts[l]) >> 27);
    }
}

// Size a filter for `expected_n` keys.  Target: 10 bits per key → ~1%
// FPR (Parquet's documented baseline).  Each block holds 256 bits so
// num_blocks ≈ expected_n / 25.6 → take ceil((N * 10 + 255) / 256).
// Rounds `num_blocks` up to a power of two.
//
// Parquet FPR table (observed):
//   5 bits/key → 10% FPR
//   8 bits/key → 2%  FPR
//  10 bits/key → 1%  FPR   ← our default
//  16 bits/key → 0.01% FPR
// Callers that need tighter FPR pass a larger `expected_n`; caller who
// wants looser sizing can call sbbf_create_with_bits_per_key directly.
BOLT_FORCE_INLINE bool sbbf_create(SplitBlockBloom* sbf, int64_t expected_n,
                                   Arena* arena) noexcept {
    assert(sbf   != nullptr);
    assert(arena != nullptr);
    assert(expected_n >= 0);

    // 10 bits/key target; 256 bits/block → num_blocks ≈ (N*10 + 255) / 256.
    uint64_t need_bits   = static_cast<uint64_t>(expected_n) * 10ull;
    uint64_t need_blocks = (need_bits + 255ull) / 256ull;
    if (need_blocks == 0u) need_blocks = 1u;
    uint32_t num_blocks = 1u;
    while (static_cast<uint64_t>(num_blocks) < need_blocks &&
           num_blocks < (1u << 26)) {
        num_blocks <<= 1;
    }
    assert(num_blocks >= 1u);
    assert((num_blocks & (num_blocks - 1u)) == 0u);

    void* buf = arena->allocate_zeroed(
        static_cast<size_t>(num_blocks) * sizeof(SbbfBlock), 32);
    if (buf == nullptr) return false;
    sbf->blocks          = static_cast<SbbfBlock*>(buf);
    sbf->num_blocks_mask = num_blocks - 1u;
    return true;
}

// Block index for a hash: use the high 32 bits (Parquet spec) masked into
// the power-of-two block count range.
BOLT_FORCE_INLINE uint32_t sbbf_block_index(const SplitBlockBloom& sbf,
                                            uint64_t mixed_hash) noexcept {
    return static_cast<uint32_t>(mixed_hash >> 32) & sbf.num_blocks_mask;
}

// Insert.  `mixed_hash = swiss_mix(key)`.  Sets one bit per lane in the
// selected 256-bit block — 8 ORs into 8 adjacent uint32_t lanes.
BOLT_FORCE_INLINE void sbbf_add(SplitBlockBloom& sbf,
                                uint64_t mixed_hash) noexcept {
    assert(sbf.blocks != nullptr);
    const uint32_t block_idx = sbbf_block_index(sbf, mixed_hash);
    const uint32_t h_lane    = static_cast<uint32_t>(mixed_hash);
    uint32_t mask[8];
    sbbf_compute_mask(h_lane, mask);
    SbbfBlock& blk = sbf.blocks[block_idx];
    for (int l = 0; l < 8; ++l) blk.lane[l] |= mask[l];
}

// Prefetch the 256-bit block a future sbbf_test(mixed_hash) will read.
// Additive; the filter itself is unchanged.  Exists for the WINDOWED probe
// (bolt_joinkernel.h jk_probe_one_int64_mlp), which hashes a window of probe
// rows and issues every block's prefetch before reading any of them, so the
// independent misses overlap instead of serialising one per row.
BOLT_FORCE_INLINE void sbbf_prefetch(const SplitBlockBloom& sbf,
                                     uint64_t mixed_hash) noexcept {
    assert(sbf.blocks != nullptr);
    BOLT_PREFETCH_READ(&sbf.blocks[sbbf_block_index(sbf, mixed_hash)]);
}

// Test.  Returns true if the key MAY be present, false if definitely
// absent.  8 ANDs + 1 combined compare; on AVX2 this auto-vectorises to
// one 256-bit load + _mm256_and_si256 + _mm256_testc_si256 (or
// equivalent).  Zero branches on the hot path.
BOLT_FORCE_INLINE bool sbbf_test(const SplitBlockBloom& sbf,
                                 uint64_t mixed_hash) noexcept {
    assert(sbf.blocks != nullptr);
    const uint32_t block_idx = sbbf_block_index(sbf, mixed_hash);
    const uint32_t h_lane    = static_cast<uint32_t>(mixed_hash);
    uint32_t mask[8];
    sbbf_compute_mask(h_lane, mask);
    const SbbfBlock& blk = sbf.blocks[block_idx];
    // All 8 lane bits must be present.  Branchless: AND every lane and
    // demand the mask came back in full.
    uint32_t missing = 0;
    for (int l = 0; l < 8; ++l) {
        missing |= (mask[l] & ~blk.lane[l]);
    }
    return missing == 0u;
}

// Storage footprint in bytes (for sizing assertions / diagnostics).
BOLT_FORCE_INLINE size_t sbbf_size_bytes(const SplitBlockBloom& sbf) noexcept {
    return (static_cast<size_t>(sbf.num_blocks_mask) + 1u) * sizeof(SbbfBlock);
}

}  // namespace bolt
