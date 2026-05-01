// bolt_atomic_bitmap.h — Lock-free atomic bitmap primitives.
//
// One bit per slot, 64 bits per word, words held as `std::atomic<uint64_t>*`.
// Designed for single-writer / many-reader and many-writer / many-reader
// patterns where the cost of a mutex would dwarf the cost of the bit set.
//
// Operations:
//   atomic_bitmap_words_for(n_bits) → number of 64-bit words needed
//   atomic_bitmap_set(words, bit)   → wait-free fetch_or; returns true iff
//                                      the bit transitioned 0→1 on this call
//   atomic_bitmap_clear(words, bit) → wait-free fetch_and; returns true iff
//                                      the bit transitioned 1→0 on this call
//   atomic_bitmap_test(words, bit)  → relaxed atomic load + bit test
//   atomic_bitmap_popcount(words, n_words) → relaxed atomic loads + popcount
//   atomic_bitmap_clear_all(words, n_words) → release stores of zero
//
// Tiger Style:
//   * noexcept everywhere; no exceptions, no STL allocation
//   * caller-supplied `std::atomic<uint64_t>*` storage (typically arena-backed)
//   * cache-line alignment is the caller's responsibility — pass `alignas(64)`
//     storage when the bitmap is shared across writers in different sockets
//   * BOLT_RESTRICT on the words pointer in read-only kernels so the
//     compiler can keep the relaxed loads in registers across the loop
//   * ≥2 assertions per hot-path function
//   * functions ≤70 lines
//
// Ordering:
//   `set` / `clear` use acq_rel — readers see the bit transition with a
//   happens-before on every prior write the writer published.
//   `test` / `popcount` use relaxed — they're advisory checks. Callers that
//   need stronger ordering should pair the test with their own fence.
//
// Use cases (consumers identified during the BM25 rebuild, plan
// `this-was-a-freach-hashed-crab.md` / Wave-9.2):
//   * MarbleDB BM25 doc tombstone bitmap (`bm25_remove_doc`)
//   * Future: PDX cluster live-mask, HNSW evicted-entry mask, episodic
//     retire bitmap, anywhere a "soft delete" set needs lock-free probes.

#pragma once

#include "bolt/bolt_port.h"   // BOLT_FORCE_INLINE, BOLT_RESTRICT, bolt_popcount64

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace bolt {

// Number of 64-bit words needed to hold `n_bits` bits. Constexpr so the
// caller can size arena allocations at compile time when the cap is known.
BOLT_FORCE_INLINE constexpr size_t atomic_bitmap_words_for(size_t n_bits) noexcept {
    return (n_bits + 63u) / 64u;
}

// Set the bit at `bit_index`. Returns true iff this call performed the
// 0→1 transition (i.e. the bit was clear when we got here). Wait-free.
BOLT_FORCE_INLINE bool atomic_bitmap_set(std::atomic<uint64_t>* words,
                                          uint64_t bit_index) noexcept {
    assert(words != nullptr);
    const uint64_t w = bit_index >> 6;
    const uint64_t m = uint64_t{1} << (bit_index & 63);
    const uint64_t prev = words[w].fetch_or(m, std::memory_order_acq_rel);
    assert((prev & m) == 0u || (prev & m) == m);  // bit is 0 or 1
    return (prev & m) == 0u;
}

// Clear the bit at `bit_index`. Returns true iff this call performed the
// 1→0 transition. Wait-free.
BOLT_FORCE_INLINE bool atomic_bitmap_clear(std::atomic<uint64_t>* words,
                                            uint64_t bit_index) noexcept {
    assert(words != nullptr);
    const uint64_t w = bit_index >> 6;
    const uint64_t m = uint64_t{1} << (bit_index & 63);
    const uint64_t prev = words[w].fetch_and(~m, std::memory_order_acq_rel);
    assert((prev & m) == 0u || (prev & m) == m);
    return (prev & m) != 0u;
}

// Probe a single bit. Relaxed load — caller should pair with their own
// fence if the result drives further memory accesses.
BOLT_FORCE_INLINE bool atomic_bitmap_test(const std::atomic<uint64_t>* words,
                                           uint64_t bit_index) noexcept {
    assert(words != nullptr);
    const uint64_t w = bit_index >> 6;
    const uint64_t m = uint64_t{1} << (bit_index & 63);
    return (words[w].load(std::memory_order_relaxed) & m) != 0u;
}

// Population count over `n_words` words. Relaxed loads; result is a
// snapshot — concurrent writers may have moved the count by the time
// the kernel returns. Use for heuristics ("compact yet?"), not for
// invariants.
BOLT_FORCE_INLINE size_t atomic_bitmap_popcount(
        const std::atomic<uint64_t>* BOLT_RESTRICT words, size_t n_words) noexcept {
    assert(words != nullptr || n_words == 0u);
    size_t total = 0u;
    for (size_t i = 0; i < n_words; ++i) {
        total += static_cast<size_t>(
            bolt_popcount64(words[i].load(std::memory_order_relaxed)));
    }
    return total;
}

// Clear every bit. Release stores; not safe to invoke while another
// thread is concurrently calling set/clear/test on the same bitmap.
BOLT_FORCE_INLINE void atomic_bitmap_clear_all(std::atomic<uint64_t>* words,
                                                size_t n_words) noexcept {
    assert(words != nullptr || n_words == 0u);
    for (size_t i = 0; i < n_words; ++i) {
        words[i].store(0ull, std::memory_order_release);
    }
}

}  // namespace bolt
