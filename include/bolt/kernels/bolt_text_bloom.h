// bolt_text_bloom.h — per-page word filter kernels (G2CHK-1198, logs).
//
// A page of Utf8 rows is summarised as a split-block Bloom over its distinct
// words, so a reader can skip the page when a required word cannot be in it.
//
// WORD RULE: the one of bolt_text_search.h. A word byte is [A-Za-z0-9_]; every
// other byte, bytes >= 0x80 included, separates words. Words are ASCII
// case-folded to lower case before hashing, so a case-insensitive search uses
// the same filter. A word longer than kBloomWordMaxBytes is hashed on its first
// kBloomWordMaxBytes bytes (a prefix test: a necessary condition, so it never
// rejects a page that holds the word).
//
// HASH (persisted, never changes without a new algorithm tag): the folded bytes
// through bolt::hash_bytes_wyhash3_v1; a result of 0 is stored as 1 (0 marks an
// empty slot of the distinct-word set). The same value is probed with
// sbbf_fr_test.
//
// Nothing allocates: the caller owns the slot array and the block array.
#pragma once

#include "bolt/bolt_kernel_limits.h"
#include <cassert>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_hash.h"
#include "bolt/bolt_port.h"
#include "bolt/bolt_types.h"
#include "bolt/join/bolt_sbbf.h"

namespace bolt {
namespace kernels {
namespace text {


BOLT_FORCE_INLINE bool bloom_is_word_byte(uint8_t b) noexcept {
    return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || (b >= '0' && b <= '9') || b == '_';
}

// Hash of one word (any case); len >= 1. The value stored in, and probed
// against, a page filter.
BOLT_FORCE_INLINE uint64_t bloom_word_hash(const uint8_t* w, uint32_t len) noexcept {
    assert(w != nullptr && len >= 1u);
    uint8_t buf[kBloomWordMaxBytes];
    const uint32_t n = len < kBloomWordMaxBytes ? len : kBloomWordMaxBytes;
    for (uint32_t i = 0; i < n; ++i) {                  // bounded: kBloomWordMaxBytes
        const uint8_t b = w[i];
        buf[i] = (b >= 'A' && b <= 'Z') ? static_cast<uint8_t>(b + 32) : b;
    }
    const uint64_t h = hash_bytes_wyhash3_v1(buf, n);
    return h == 0u ? 1u : h;
}

// Open-addressing set of word hashes. `slots` is zeroed by the caller, a power
// of two long (mask + 1); `limit` < mask + 1 bounds the distinct words held.
struct WordSet {
    uint64_t* slots;
    uint32_t  mask;
    uint32_t  count;
    uint32_t  limit;
    uint32_t  _pad;
};

BOLT_FORCE_INLINE void wordset_init(WordSet* s, uint64_t* slots, uint32_t cap_pow2, uint32_t limit) noexcept {
    assert(s != nullptr && slots != nullptr);
    assert(cap_pow2 >= 2u && (cap_pow2 & (cap_pow2 - 1u)) == 0u && limit < cap_pow2);
    std::memset(slots, 0, size_t{cap_pow2} * sizeof(uint64_t));
    s->slots = slots;
    s->mask = cap_pow2 - 1u;
    s->count = 0;
    s->limit = limit;
    s->_pad = 0;
}

// False: a new word would pass `limit` (the set is left unchanged).
BOLT_FORCE_INLINE bool wordset_add(WordSet* s, uint64_t h) noexcept {
    assert(s != nullptr && h != 0u);
    uint32_t i = static_cast<uint32_t>(h >> 32) & s->mask;
    for (;;) {                                          // bounded: limit < capacity, so a slot is free
        const uint64_t v = s->slots[i];
        if (v == h) return true;
        if (v == 0u) break;
        i = (i + 1u) & s->mask;
    }
    if (s->count >= s->limit) return false;
    s->slots[i] = h;
    ++s->count;
    return true;
}

// The words of bytes[0, n) into the set. False: the set filled.
BOLT_FORCE_INLINE bool words_add_bytes(const uint8_t* b, uint32_t n, WordSet* s) noexcept {
    assert(b != nullptr || n == 0u);
    uint32_t i = 0;
    while (i < n) {                                     // bounded: n
        while (i < n && !bloom_is_word_byte(b[i])) ++i;
        const uint32_t beg = i;
        while (i < n && bloom_is_word_byte(b[i])) ++i;
        if (i > beg && !wordset_add(s, bloom_word_hash(b + beg, i - beg))) return false;
    }
    return true;
}

BOLT_FORCE_INLINE bool row_valid_bit(const uint8_t* validity, int64_t off, int64_t r) noexcept {
    if (validity == nullptr) return true;
    const int64_t bit = off + r;
    return ((validity[bit >> 3] >> (bit & 7)) & 1u) != 0u;
}

// Rows of a flat StringView column (inline up to 12 bytes, else in `overflow`).
BOLT_FORCE_INLINE bool words_add_views(const StringView* v, const uint8_t* validity, int64_t voff, int64_t n,
                                       const char* overflow, WordSet* s) noexcept {
    assert(v != nullptr || n == 0);
    for (int64_t r = 0; r < n; ++r) {                   // bounded: n
        if (!row_valid_bit(validity, voff, r)) continue;
        const uint32_t len = v[r].length;
        if (len == 0u) continue;
        const char* p = len <= 12u ? v[r].prefix : overflow + v[r].ref.offset;
        assert(len <= 12u || overflow != nullptr);
        if (!words_add_bytes(reinterpret_cast<const uint8_t*>(p), len, s)) return false;
    }
    return true;
}

// Rows of a VarBinary column: row r spans payload[offs[r], offs[r + 1]).
BOLT_FORCE_INLINE bool words_add_offsets(const uint8_t* payload, const int32_t* offs, const uint8_t* validity,
                                         int64_t voff, int64_t n, WordSet* s) noexcept {
    assert(offs != nullptr || n == 0);
    for (int64_t r = 0; r < n; ++r) {                   // bounded: n
        if (!row_valid_bit(validity, voff, r)) continue;
        const int32_t a = offs[r], e = offs[r + 1];
        assert(a >= 0 && e >= a);
        if (e == a) continue;
        if (!words_add_bytes(payload + a, static_cast<uint32_t>(e - a), s)) return false;
    }
    return true;
}

// Every hash of the set into the (zeroed) blocks.
BOLT_FORCE_INLINE void wordset_fill_bloom(const WordSet& s, SbbfBlock* blocks, uint32_t num_blocks) noexcept {
    assert(blocks != nullptr && num_blocks >= 1u);
    SplitBlockBloomFr bf{blocks, num_blocks};
    for (uint32_t i = 0; i <= s.mask; ++i)              // bounded: capacity
        if (s.slots[i] != 0u) sbbf_fr_add(bf, s.slots[i]);
}

// True when every word hash may be in the filter (AND). No false negatives.
BOLT_FORCE_INLINE bool bloom_has_all(const SbbfBlock* blocks, uint32_t num_blocks, const uint64_t* hashes,
                                     uint32_t n) noexcept {
    assert(blocks != nullptr && num_blocks >= 1u && (hashes != nullptr || n == 0u));
    SplitBlockBloomFr bf{const_cast<SbbfBlock*>(blocks), num_blocks};
    for (uint32_t i = 0; i < n; ++i)                    // bounded: n
        if (!sbbf_fr_test(bf, hashes[i])) return false;
    return true;
}

}  // namespace text
}  // namespace kernels
}  // namespace bolt
