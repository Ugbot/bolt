// bolt/kernels/bolt_run_bounds.h: run offsets of a batch from its key columns.
//
// mask[i] = 1 when logical row i differs from row i-1 on ANY key column (i >= 1; mask[0] is the caller's
// to settle against the previous batch). One call per key column ORs into the mask; mask_starts turns the
// mask into run-start offsets. NULL equals NULL and differs from every value; float keys compare by value
// with NaN == NaN and -0.0 == 0.0 (a group key's equality, not bit equality); Utf8 compares bytes.
// A selection vector maps logical to physical rows (phys(i) = sel ? sel[i] : i).
//
// Tiger Style: noexcept, no allocation, every loop bounded by n.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/bolt_types.h"
#include "bolt/kernels/bolt_utf8.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace bolt {
namespace kernels {
namespace run_bounds {

enum class KeyKind : std::uint8_t { Bytes, F32, F64, Utf8 };

struct KeyCol {
    const void*          data;          // fixed-width cells, or StringView[] for Utf8
    const std::uint8_t*  validity;      // Arrow bitmap, nullptr = all valid
    std::int64_t         validity_off;  // bit offset of physical row 0
    std::uint32_t        cell_w;        // bytes of one fixed-width cell
    KeyKind              kind;
    const char*          overflow;      // Utf8: long-string base
};

namespace detail {

BOLT_FORCE_INLINE bool bit_at(const std::uint8_t* v, std::int64_t off, std::int64_t p) noexcept {
    const std::int64_t b = p + off;
    return ((v[b >> 3] >> (b & 7)) & 1u) != 0u;
}

template <class T>
BOLT_FORCE_INLINE bool val_ne(const std::uint8_t* a, const std::uint8_t* b) noexcept {
    T x, y;
    std::memcpy(&x, a, sizeof(T));
    std::memcpy(&y, b, sizeof(T));
    return !(x == y || (x != x && y != y));
}

BOLT_FORCE_INLINE bool bytes_ne(const std::uint8_t* a, const std::uint8_t* b, std::uint32_t w) noexcept {
    return std::memcmp(a, b, w) != 0;
}

BOLT_FORCE_INLINE bool utf8_ne(const KeyCol& k, std::int64_t p, std::int64_t q) noexcept {
    const auto* sv = static_cast<const StringView*>(k.data);
    if (sv[p].length != sv[q].length) return true;
    const std::uint32_t len = sv[p].length;
    if (len == 0) return false;
    return std::memcmp(utf8::sv_bytes(sv[p], k.overflow), utf8::sv_bytes(sv[q], k.overflow), len) != 0;
}

// Values of physical rows p and q differ (both valid).
BOLT_FORCE_INLINE bool value_ne(const KeyCol& k, std::int64_t p, std::int64_t q) noexcept {
    if (k.kind == KeyKind::Utf8) return utf8_ne(k, p, q);
    const auto* base = static_cast<const std::uint8_t*>(k.data);
    const std::uint8_t* a = base + static_cast<std::size_t>(p) * k.cell_w;
    const std::uint8_t* b = base + static_cast<std::size_t>(q) * k.cell_w;
    if (k.kind == KeyKind::F64) return val_ne<double>(a, b);
    if (k.kind == KeyKind::F32) return val_ne<float>(a, b);
    return bytes_ne(a, b, k.cell_w);
}

}  // namespace detail

// OR the adjacent-inequality of one key column over logical rows [1, n) into mask[1, n).
inline void adjacent_ne(const KeyCol& k, const std::int32_t* sel, std::int64_t n, std::uint8_t* BOLT_RESTRICT mask) noexcept {
    assert(n >= 0 && (mask != nullptr || n == 0));
    assert(k.data != nullptr || n == 0);
    for (std::int64_t i = 1; i < n; ++i) {
        const std::int64_t p = sel != nullptr ? sel[i] : i;
        const std::int64_t q = sel != nullptr ? sel[i - 1] : i - 1;
        std::uint8_t ne;
        if (k.validity == nullptr) {
            ne = detail::value_ne(k, p, q) ? 1u : 0u;
        } else {
            const bool vp = detail::bit_at(k.validity, k.validity_off, p);
            const bool vq = detail::bit_at(k.validity, k.validity_off, q);
            ne = (vp != vq) ? 1u : ((vp && detail::value_ne(k, p, q)) ? 1u : 0u);
        }
        mask[i] |= ne;
    }
}

// Run-start offsets: every i in [0, n) with i == 0 or mask[i] != 0. Returns the count, or -1 when more than
// `cap` runs start (nothing past cap is written).
inline std::int64_t mask_starts(const std::uint8_t* BOLT_RESTRICT mask, std::int64_t n,
                                std::uint32_t* BOLT_RESTRICT offs, std::int64_t cap) noexcept {
    assert(n >= 0 && n <= 0xFFFFFFFFll);
    assert(offs != nullptr || n == 0);
    if (n == 0) return 0;
    std::int64_t c = 0;
    if (cap < 1) return -1;
    offs[c++] = 0;
    std::int64_t i = 1;
    while (i < n) {
        if (i + 8 <= n) {   // eight clear bytes at once: the common long run
            std::uint64_t w;
            std::memcpy(&w, mask + i, 8);
            if (w == 0) { i += 8; continue; }
        }
        if (mask[i] != 0) {
            if (c >= cap) return -1;
            offs[c++] = static_cast<std::uint32_t>(i);
        }
        ++i;
    }
    return c;
}

}  // namespace run_bounds
}  // namespace kernels
}  // namespace bolt
