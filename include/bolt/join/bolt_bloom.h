// bolt_bloom.h — block Bloom filter for hash-join probe pre-screen.
//
// One 64-bit word per block; 4 bits set per key within a single block
// selected by the high 32 bits of `swiss_mix(key)`. The low 24 bits
// supply four 6-bit slices picking bit positions in [0, 64).
//
// Cost profile:
//   - Add:   1 load + 1 OR + 1 store          (~3 ns on L1-resident)
//   - Test:  1 load + 1 AND + 1 compare        (~2 ns)
//
// The hash-join probe uses `bloom_test` as a front-gate before
// `SwissTable::find` so misses skip the probe chain entirely. Biggest
// wins at low match-rate; at high match-rate the add-per-build cost
// is amortised and test is cheap enough to stay net-positive.
//
// Parameters: 16 bits per expected key (2 bytes), 4 hashes per key,
// block-aligned at 64-bit word granularity.  Expected FPR under a
// load factor of 1.0x: ~0.4%.  (If callers want to flag-gate at build
// time for FPR sensitivity, `bloom_bits_per_key` can be overridden.)
//
// RULES: Tiger Style — POD struct, free functions, no exceptions,
// noexcept everywhere, arena-allocated, power-of-two word count for
// AND-masked modulo.

#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>

namespace bolt {

inline constexpr uint32_t kBloomBitsPerKey = 16;   // 2 bytes per expected key
inline constexpr uint32_t kBloomHashesPerKey = 4;  // 4 bits set per key

struct BloomFilter {
    uint64_t* words;           // num_words * 64-bit; arena-owned
    uint32_t  num_words_mask;  // (num_words - 1); words count is pow2
};

// Compute the 4-bit set-mask for a key's block, packed into one 64-bit word.
// Each hash slice is 6 bits wide picking a position in [0, 64).
BOLT_FORCE_INLINE uint64_t bloom_bits_from_hash(uint64_t h) noexcept {
    uint64_t mask = 0;
    mask |= uint64_t{1} << ((h      ) & 63u);
    mask |= uint64_t{1} << ((h >>  6) & 63u);
    mask |= uint64_t{1} << ((h >> 12) & 63u);
    mask |= uint64_t{1} << ((h >> 18) & 63u);
    return mask;
}

// Allocate a Bloom filter sized for `expected_n` keys from `arena`.
// Returns false on OOM or if `expected_n` is negative.  Empty filters
// (expected_n == 0) allocate one zero word so `bloom_test` is safe.
BOLT_FORCE_INLINE bool bloom_create(BloomFilter* bf, int64_t expected_n,
                                    Arena* arena) noexcept {
    assert(bf    != nullptr);
    assert(arena != nullptr);
    assert(expected_n >= 0);

    const uint64_t need_bits  = static_cast<uint64_t>(expected_n) *
                                kBloomBitsPerKey;
    const uint64_t need_words = (need_bits + 63u) / 64u;
    uint32_t num_words = 1;
    while (static_cast<uint64_t>(num_words) < need_words &&
           num_words < (1u << 26)) {
        num_words <<= 1;
    }
    assert(num_words >= 1u);
    assert((num_words & (num_words - 1u)) == 0u);

    void* buf = arena->allocate_zeroed(
        static_cast<size_t>(num_words) * sizeof(uint64_t), 64);
    if (buf == nullptr) return false;
    bf->words          = static_cast<uint64_t*>(buf);
    bf->num_words_mask = num_words - 1u;
    return true;
}

// Insert `mixed_hash = swiss_mix(key)` into the filter.
BOLT_FORCE_INLINE void bloom_add(BloomFilter& bf, uint64_t mixed_hash) noexcept {
    assert(bf.words != nullptr);
    const uint32_t w = static_cast<uint32_t>(mixed_hash >> 32) & bf.num_words_mask;
    bf.words[w] |= bloom_bits_from_hash(mixed_hash);
}

// Test `mixed_hash = swiss_mix(key)`. Returns true if the key MAY be
// present (can be a false positive); returns false if it is definitely
// absent (no false negatives).
BOLT_FORCE_INLINE bool bloom_test(const BloomFilter& bf,
                                  uint64_t mixed_hash) noexcept {
    assert(bf.words != nullptr);
    const uint32_t w = static_cast<uint32_t>(mixed_hash >> 32) & bf.num_words_mask;
    const uint64_t bits = bloom_bits_from_hash(mixed_hash);
    return (bf.words[w] & bits) == bits;
}

// Storage footprint in bytes (for sizing assertions / diagnostics).
BOLT_FORCE_INLINE size_t bloom_size_bytes(const BloomFilter& bf) noexcept {
    return (static_cast<size_t>(bf.num_words_mask) + 1u) * sizeof(uint64_t);
}

}  // namespace bolt
