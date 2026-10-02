// bolt_fastlanes.h — the FastLanes 1,024-value block layout that defines
// ColumnFormat::BitPacked, FrameOfRef and DeltaFOR (MSEG B1, decision U37/G8).
//
// Scalar reference encoder and decoder only. The NEON / AVX2 kernels (M4)
// must produce and consume exactly these bytes; this file is the spec they
// are tested against.
//
// Layout (Afroozeh & Boncz, "The FastLanes Compression Layout", VLDB 2023).
// index(), transpose() and the packed word order are those of the spiraldb
// `fastlanes` crate (BitPacking::pack, transpose, Delta::delta), which Vortex
// uses; DeltaFOR = that crate's Delta with per-lane bases, minus a frame of
// reference on the deltas, then BitPacking:
//
//   A column is cut into blocks of 1,024 values; the last block is padded.
//   Values are unsigned T-bit words, T = 8 / 16 / 32 / 64 (the type's width),
//   viewed as a 1,024-bit virtual register of LANES = 1024 / T lanes and T
//   rows. Value index of (row, lane):
//
//       index(row, lane) = ORDER[row / 8] * 16 + (row % 8) * 128 + lane
//       ORDER = {0, 4, 2, 6, 1, 5, 3, 7}
//
//   Packing to W bits (0 <= W <= T): each lane's T values, in row order, form
//   a W*T-bit little-endian bit stream cut into W words of T bits; stream
//   word k of lane l is stored at packed[LANES * k + l]. A block is
//   LANES * W words (W * 128 bytes): consecutive words are consecutive lanes,
//   so one SIMD load reads the same row-word of many lanes.
//
//   BitPacked  value = packed (unsigned W bits)
//   FrameOfRef value = ref + packed (wrapping in T bits); ref = BoltColumn::seq_offset
//   DeltaFOR   per block: [bases: LANES words][packed: LANES * W words].
//              The block's values are first transposed,
//                  t[i] = v[transpose(i)],
//                  transpose(i) = (i % 16) * 64 + ORDER[(i / 16) % 8] * 8 + i / 128,
//              then each lane is a delta chain in row order:
//                  t[index(row, lane)] = prev + dref + packed[...]
//              starting from prev = bases[lane]; dref = BoltColumn::seq_offset.
//              (For T = 64, lane l is exactly values [64 l, 64 l + 64) in
//              natural order.) Padding repeats the last value.
//
//   BoltColumn: data = the packed buffer, seq_step = W, seq_offset = ref /
//   dref, type = the logical integer type (its width is T), length = rows.
//
// Tiger Style: POD + free functions, noexcept, no allocation; per-call
// scratch is one 1,024-value block on the stack (<= 8 KiB); every loop is
// bounded by the block size or the row count.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "bolt/bolt_port.h"

namespace bolt {
namespace fastlanes {

inline constexpr uint32_t kBlockValues = 1024;
inline constexpr uint8_t  kOrder[8] = {0, 4, 2, 6, 1, 5, 3, 7};

template <class U> constexpr uint32_t bits() noexcept { return sizeof(U) * 8u; }
template <class U> constexpr uint32_t lanes() noexcept { return kBlockValues / bits<U>(); }

BOLT_FORCE_INLINE uint32_t index(uint32_t row, uint32_t lane) noexcept {
    assert(row < 64 && lane < 128);
    return kOrder[row >> 3] * 16u + (row & 7u) * 128u + lane;
}

BOLT_FORCE_INLINE uint32_t transpose(uint32_t i) noexcept {
    assert(i < kBlockValues);
    return (i % 16u) * 64u + kOrder[(i / 16u) % 8u] * 8u + i / 128u;
}

/// (row, lane) of value index i: the inverse of index() for type U.
template <class U>
BOLT_FORCE_INLINE void row_lane(uint32_t i, uint32_t* row, uint32_t* lane) noexcept {
    assert(i < kBlockValues);
    assert(row != nullptr && lane != nullptr);
    *lane = i % lanes<U>();
    const uint32_t s = i / 128u;
    const uint32_t o = kOrder[(i - s * 128u - *lane) / 16u];   // kOrder is its own inverse
    *row = o * 8u + s;
}

template <class U> constexpr uint64_t mask_of(uint32_t w) noexcept {
    return w >= 64 ? ~0ull : ((1ull << w) - 1ull);
}

/// Pack 1,024 values (`in`, natural index order) to W bits into
/// out[0, lanes * W).
template <class U>
inline void pack_block(const U* BOLT_RESTRICT in, uint32_t w, U* BOLT_RESTRICT out) noexcept {
    static_assert(std::is_unsigned_v<U>, "packed words are unsigned");
    assert(in != nullptr && w <= bits<U>());
    constexpr uint32_t T = bits<U>(), L = lanes<U>();
    if (w == 0) return;
    memset(out, 0, sizeof(U) * L * w);
    const uint64_t m = mask_of<U>(w);
    for (uint32_t lane = 0; lane < L; ++lane) {
        for (uint32_t row = 0; row < T; ++row) {
            const uint64_t v = static_cast<uint64_t>(in[index(row, lane)]) & m;
            const uint32_t bit = row * w, word = bit / T, off = bit % T;
            out[L * word + lane] = static_cast<U>(out[L * word + lane] | (v << off));
            if (off + w > T)
                out[L * (word + 1) + lane] =
                    static_cast<U>(out[L * (word + 1) + lane] | (v >> (T - off)));
        }
    }
}

/// Value i (natural index) of a packed block, without unpacking the rest.
template <class U>
BOLT_FORCE_INLINE U unpack_one(const U* BOLT_RESTRICT packed, uint32_t w, uint32_t i) noexcept {
    assert(i < kBlockValues);
    assert(w <= bits<U>());
    constexpr uint32_t T = bits<U>(), L = lanes<U>();
    if (w == 0) return 0;
    uint32_t row = 0, lane = 0;
    row_lane<U>(i, &row, &lane);
    const uint32_t bit = row * w, word = bit / T, off = bit % T;
    uint64_t v = static_cast<uint64_t>(packed[L * word + lane]) >> off;
    if (off + w > T) v |= static_cast<uint64_t>(packed[L * (word + 1) + lane]) << (T - off);
    return static_cast<U>(v & mask_of<U>(w));
}

/// Unpack a block into out[1024] (natural index order).
template <class U>
inline void unpack_block(const U* BOLT_RESTRICT packed, uint32_t w, U* BOLT_RESTRICT out) noexcept {
    assert(out != nullptr);
    assert(w <= bits<U>());
    for (uint32_t i = 0; i < kBlockValues; ++i) out[i] = unpack_one<U>(packed, w, i);
}

// ---------------------------------------------------------------------------
// Sizes and widths
// ---------------------------------------------------------------------------

BOLT_FORCE_INLINE int64_t blocks(int64_t n) noexcept {
    assert(n >= 0);
    return (n + kBlockValues - 1) / kBlockValues;
}

/// Bytes of a BitPacked / FrameOfRef buffer (type width T bits).
BOLT_FORCE_INLINE size_t packed_bytes(uint32_t t_bits, int64_t n, uint32_t w) noexcept {
    assert(t_bits == 8 || t_bits == 16 || t_bits == 32 || t_bits == 64);
    assert(w <= t_bits && n >= 0);
    return static_cast<size_t>(blocks(n)) * (kBlockValues / 8u) * w;
}

/// Bytes of a DeltaFOR buffer: per block LANES base words + LANES * W words.
BOLT_FORCE_INLINE size_t delta_bytes(uint32_t t_bits, int64_t n, uint32_t w) noexcept {
    assert(t_bits == 8 || t_bits == 16 || t_bits == 32 || t_bits == 64);
    assert(w <= t_bits && n >= 0);
    return static_cast<size_t>(blocks(n)) * (kBlockValues / 8u) * (w + 1u);
}

/// Bits needed for an unsigned value (0 for 0).
BOLT_FORCE_INLINE uint32_t width_of(uint64_t v) noexcept {
    uint32_t w = 0;
    while (v != 0 && w < 64) { v >>= 1; ++w; }   // bounded: 64
    return w;
}

// ---------------------------------------------------------------------------
// Column encoders / decoders. V is the logical integer type; U its unsigned.
// ---------------------------------------------------------------------------

template <class V> using unsigned_of = std::make_unsigned_t<V>;

/// FrameOfRef parameters: ref = min, W = bits(max - min). ref = 0 and
/// W = bits(max) for a non-negative BitPacked column is the caller's choice.
template <class V>
inline void choose_for(const V* BOLT_RESTRICT v, int64_t n, int64_t* ref, uint32_t* w) noexcept {
    static_assert(std::is_integral_v<V>, "integer types only");
    assert(ref != nullptr && w != nullptr);
    assert(v != nullptr || n == 0);
    using U = unsigned_of<V>;
    V lo = n ? v[0] : V(0), hi = lo;
    for (int64_t i = 1; i < n; ++i) { lo = v[i] < lo ? v[i] : lo; hi = v[i] > hi ? v[i] : hi; }
    *ref = static_cast<int64_t>(lo);
    *w = width_of(static_cast<uint64_t>(static_cast<U>(static_cast<U>(hi) - static_cast<U>(lo))));
}

/// Encode BitPacked (ref = 0) or FrameOfRef: packed = (v - ref) in W bits.
/// Values must fit (choose_for). out: packed_bytes(T, n, W).
template <class V>
inline void encode_for(const V* BOLT_RESTRICT v, int64_t n, int64_t ref, uint32_t w,
                       void* BOLT_RESTRICT out) noexcept {
    using U = unsigned_of<V>;
    assert(v != nullptr || n == 0);
    assert(w <= bits<U>());
    alignas(64) U block[kBlockValues];
    auto* dst = static_cast<U*>(out);
    const U r = static_cast<U>(ref);
    for (int64_t b = 0; b < blocks(n); ++b) {
        const int64_t base = b * kBlockValues;
        for (uint32_t i = 0; i < kBlockValues; ++i) {
            const int64_t k = base + i;
            block[i] = k < n ? static_cast<U>(static_cast<U>(v[k]) - r) : U(0);
        }
        pack_block<U>(block, w, dst + static_cast<size_t>(b) * lanes<U>() * w);
    }
}

template <class V>
inline void decode_for(const void* BOLT_RESTRICT in, int64_t n, int64_t ref, uint32_t w,
                       V* BOLT_RESTRICT out) noexcept {
    using U = unsigned_of<V>;
    assert(out != nullptr || n == 0);
    assert(w <= bits<U>());
    const auto* src = static_cast<const U*>(in);
    const U r = static_cast<U>(ref);
    for (int64_t b = 0; b < blocks(n); ++b) {
        const U* blk = src + static_cast<size_t>(b) * lanes<U>() * w;
        const int64_t base = b * kBlockValues;
        const uint32_t m = n - base < kBlockValues ? static_cast<uint32_t>(n - base) : kBlockValues;
        for (uint32_t i = 0; i < m; ++i)
            out[base + i] = static_cast<V>(static_cast<U>(unpack_one<U>(blk, w, i) + r));
    }
}

namespace detail {

// Value at natural position base + i of a padded block; past the end the
// last value repeats. encode and choose read the same padded values, so the
// chosen width covers every chain delta, padding included.
template <class V>
BOLT_FORCE_INLINE unsigned_of<V> delta_src(const V* v, int64_t n, int64_t base,
                                           uint32_t i) noexcept {
    assert(n > base);
    assert(i < kBlockValues);
    const int64_t k = base + i;
    return static_cast<unsigned_of<V>>(v[k < n ? k : n - 1]);
}

}  // namespace detail

/// DeltaFOR parameters: dref = the smallest lane-chain delta, W = bits of
/// (largest - smallest). Sorted runs with a fixed step give W = 0.
template <class V>
inline void choose_delta(const V* BOLT_RESTRICT v, int64_t n, int64_t* dref,
                         uint32_t* w) noexcept {
    using U = unsigned_of<V>;
    using S = std::make_signed_t<V>;
    assert(dref != nullptr && w != nullptr);
    assert(v != nullptr || n == 0);
    bool any = false;
    S lo = 0, hi = 0;
    for (int64_t b = 0; b < blocks(n); ++b) {
        const int64_t base = b * kBlockValues;
        for (uint32_t lane = 0; lane < lanes<U>(); ++lane) {
            for (uint32_t row = 1; row < bits<U>(); ++row) {
                const uint32_t i = index(row, lane), p = index(row - 1, lane);
                const U cur = detail::delta_src(v, n, base, transpose(i));
                const U prv = detail::delta_src(v, n, base, transpose(p));
                const S d = static_cast<S>(static_cast<U>(cur - prv));
                if (!any || d < lo) lo = d;
                if (!any || d > hi) hi = d;
                any = true;
            }
        }
    }
    *dref = static_cast<int64_t>(lo);
    *w = any ? width_of(static_cast<uint64_t>(static_cast<U>(static_cast<U>(hi) -
                                                              static_cast<U>(lo))))
             : 0u;
}

/// Encode DeltaFOR. out: delta_bytes(T, n, W).
template <class V>
inline void encode_delta_for(const V* BOLT_RESTRICT v, int64_t n, int64_t dref, uint32_t w,
                             void* BOLT_RESTRICT out) noexcept {
    using U = unsigned_of<V>;
    assert(v != nullptr || n == 0);
    assert(w <= bits<U>());
    constexpr uint32_t L = lanes<U>();
    alignas(64) U block[kBlockValues];
    auto* dst = static_cast<U*>(out);
    const U dr = static_cast<U>(dref);
    for (int64_t b = 0; b < blocks(n); ++b) {
        const int64_t base = b * kBlockValues;
        U* bases = dst + static_cast<size_t>(b) * L * (w + 1u);
        for (uint32_t lane = 0; lane < L; ++lane) {
            U prev = static_cast<U>(detail::delta_src(v, n, base, transpose(index(0, lane))) - dr);
            bases[lane] = prev;
            for (uint32_t row = 0; row < bits<U>(); ++row) {
                const uint32_t i = index(row, lane);
                const U cur = detail::delta_src(v, n, base, transpose(i));
                block[i] = static_cast<U>(static_cast<U>(cur - prev) - dr);
                prev = cur;
            }
        }
        pack_block<U>(block, w, bases + L);
    }
}

template <class V>
inline void decode_delta_for(const void* BOLT_RESTRICT in, int64_t n, int64_t dref, uint32_t w,
                             V* BOLT_RESTRICT out) noexcept {
    using U = unsigned_of<V>;
    assert(out != nullptr || n == 0);
    assert(w <= bits<U>());
    constexpr uint32_t L = lanes<U>();
    const auto* src = static_cast<const U*>(in);
    const U dr = static_cast<U>(dref);
    for (int64_t b = 0; b < blocks(n); ++b) {
        const U* bases = src + static_cast<size_t>(b) * L * (w + 1u);
        const int64_t base = b * kBlockValues;
        for (uint32_t lane = 0; lane < L; ++lane) {
            U prev = bases[lane];
            for (uint32_t row = 0; row < bits<U>(); ++row) {
                const uint32_t i = index(row, lane);
                prev = static_cast<U>(prev + dr + unpack_one<U>(bases + L, w, i));
                const int64_t k = base + transpose(i);
                if (k < n) out[k] = static_cast<V>(prev);
            }
        }
    }
}

}  // namespace fastlanes
}  // namespace bolt
