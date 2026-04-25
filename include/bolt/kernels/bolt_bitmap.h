// bolt_bitmap.h — Bitmap primitives (popcount, expand-to-indices,
// in-place AND/OR, fill). Header-only, MSVC-clean. Walks 64-bit words
// where possible via bolt_popcount64.

#pragma once

#include "bolt/bolt_port.h"  // BOLT_FORCE_INLINE, BOLT_RESTRICT, bolt_popcount64

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace bolt {

// Popcount over [bit_lo, bit_hi). Walks 8 bytes at a time via memcpy +
// bolt_popcount64; partial-byte head/tail masked out.
BOLT_FORCE_INLINE size_t bitmap_popcount(
    const uint8_t* BOLT_RESTRICT bm, size_t bit_lo, size_t bit_hi
) noexcept {
    assert(bm != nullptr || bit_lo == bit_hi);
    if (bit_hi <= bit_lo) return 0;
    size_t total = 0;
    size_t i = bit_lo;
    // Head: align up to byte boundary if needed.
    while ((i & 7u) != 0u && i < bit_hi) {
        if ((bm[i >> 3] >> (i & 7u)) & 1u) ++total;
        ++i;
    }
    // Body: 64-bit chunks.
    while (i + 64 <= bit_hi) {
        uint64_t w;
        std::memcpy(&w, bm + (i >> 3), 8);
        total += static_cast<size_t>(bolt_popcount64(w));
        i += 64;
    }
    // Tail.
    while (i < bit_hi) {
        if ((bm[i >> 3] >> (i & 7u)) & 1u) ++total;
        ++i;
    }
    return total;
}

// Expand alive bitmap to a list of set-bit indices. Output capacity
// must be >= popcount of input. Returns count written.
BOLT_FORCE_INLINE size_t bitmap_to_indices(
    const uint8_t* BOLT_RESTRICT bm, size_t n_bits,
    uint32_t* BOLT_RESTRICT out_indices
) noexcept {
    assert(bm != nullptr || n_bits == 0);
    assert(out_indices != nullptr || n_bits == 0);
    size_t out = 0;
    for (size_t i = 0; i < n_bits; ++i) {
        if ((bm[i >> 3] >> (i & 7u)) & 1u) {
            out_indices[out++] = static_cast<uint32_t>(i);
        }
    }
    return out;
}

// In-place a &= b over n_bits.
BOLT_FORCE_INLINE void bitmap_and(
    uint8_t* BOLT_RESTRICT a, const uint8_t* BOLT_RESTRICT b, size_t n_bits
) noexcept {
    assert(a != nullptr || n_bits == 0);
    assert(b != nullptr || n_bits == 0);
    const size_t nb = (n_bits + 7u) >> 3;
    for (size_t i = 0; i < nb; ++i) a[i] &= b[i];
}

// In-place a |= b over n_bits.
BOLT_FORCE_INLINE void bitmap_or(
    uint8_t* BOLT_RESTRICT a, const uint8_t* BOLT_RESTRICT b, size_t n_bits
) noexcept {
    assert(a != nullptr || n_bits == 0);
    assert(b != nullptr || n_bits == 0);
    const size_t nb = (n_bits + 7u) >> 3;
    for (size_t i = 0; i < nb; ++i) a[i] |= b[i];
}

// Set all bits in [bit_lo, bit_hi).
BOLT_FORCE_INLINE void bitmap_fill(
    uint8_t* BOLT_RESTRICT bm, size_t bit_lo, size_t bit_hi
) noexcept {
    assert(bm != nullptr || bit_lo == bit_hi);
    for (size_t i = bit_lo; i < bit_hi; ++i) {
        bm[i >> 3] |= static_cast<uint8_t>(1u << (i & 7u));
    }
}

}  // namespace bolt
