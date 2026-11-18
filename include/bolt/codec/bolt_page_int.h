// bolt/codec/bolt_page_int.h — page codecs for fixed-width values: runs (ColumnFormat::RLE), frame of reference
// (FrameOfRef) and delta with exceptions (DeltaFOR), on the FastLanes 1,024-value block layout of
// bolt_fastlanes.h. Values are opaque W-byte words (W = 1, 2, 4, 8): every codec is lossless on any bit
// pattern, so the same kernels serve ints, dates, timestamps, decimals-as-int and bools.
//
// Encoders choose their parameters from one statistics pass over the values (min / max, run count, a 65-bucket
// histogram of delta widths) and report the encoded size in closed form before any byte is written: a caller
// picks an encoding by comparing those sizes, never by encoding and measuring.
//
// Decoders treat their input as untrusted: every length is checked against the buffer and a bad stream returns
// false. No allocation; per-call scratch is one block on the stack (<= 8 KiB). Whole-page kernels, no per-row
// calls.
//
// DeltaFOR here differs from bolt_fastlanes.h's encode_delta_for in two stated ways: (1) positions past the
// last value pack 0 (the chain continues by dref) instead of repeating the last value, so a short page does
// not force delta 0 into the frame (8-byte words only; narrower words need whole blocks); (2) a delta outside the frame is an exception: its slot packs 0 and its
// full-width (delta - dref) is stored apart as {position, value}, patched into the unpacked block before the
// prefix sum.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_port.h"
#include "bolt/kernels/bolt_fastlanes.h"

namespace bolt {
namespace pagecodec {

inline constexpr uint32_t kBlock = fastlanes::kBlockValues;

/// One exception of a DeltaFOR or ALP page (16 B, little-endian).
struct Exception {
    uint32_t pos;     // DeltaFOR: block * 1024 + natural index in the block; ALP: row
    uint32_t _pad;    // zero
    uint64_t value;   // DeltaFOR: (delta - dref) in T bits; ALP: the value's raw bits
};
static_assert(sizeof(Exception) == 16, "persisted");

constexpr bool width_ok(uint32_t w) noexcept { return w == 1 || w == 2 || w == 4 || w == 8; }

// ---------------------------------------------------------------------------------------------------------
// Runs
// ---------------------------------------------------------------------------------------------------------

namespace detail {

template <class U>
inline uint32_t count_runs_t(const U* BOLT_RESTRICT v, uint32_t n) noexcept {
    uint32_t runs = n != 0 ? 1u : 0u;
    for (uint32_t i = 1; i < n; ++i) runs += v[i] != v[i - 1];     // bounded: n; no branch on the data
    return runs;
}

template <class U>
inline void rle_encode_t(const U* BOLT_RESTRICT v, uint32_t n, U* BOLT_RESTRICT values,
                         int32_t* BOLT_RESTRICT ends) noexcept {
    uint32_t k = 0;
    for (uint32_t i = 0; i < n; ++i) {                              // bounded: n
        values[k] = v[i];
        ends[k] = static_cast<int32_t>(i + 1);
        k += (i + 1 == n) || (v[i + 1 == n ? i : i + 1] != v[i]);
    }
}

template <class U>
inline bool rle_decode_t(const U* BOLT_RESTRICT values, const int32_t* BOLT_RESTRICT ends, uint32_t runs,
                         uint32_t n, U* BOLT_RESTRICT out) noexcept {
    uint32_t at = 0;
    for (uint32_t r = 0; r < runs; ++r) {                           // bounded: runs
        const int64_t e = ends[r];
        if (e <= static_cast<int64_t>(at) || e > static_cast<int64_t>(n)) return false;
        const U x = values[r];
        for (uint32_t i = at; i < static_cast<uint32_t>(e); ++i) out[i] = x;   // bounded: run length
        at = static_cast<uint32_t>(e);
    }
    return at == n;
}

}  // namespace detail

/// Runs of equal adjacent values (>= 1 for n >= 1).
inline uint32_t count_runs(const void* v, uint32_t n, uint32_t w) noexcept {
    assert(width_ok(w) && (v != nullptr || n == 0));
    switch (w) {
        case 1: return detail::count_runs_t(static_cast<const uint8_t*>(v), n);
        case 2: return detail::count_runs_t(static_cast<const uint16_t*>(v), n);
        case 4: return detail::count_runs_t(static_cast<const uint32_t*>(v), n);
        default: return detail::count_runs_t(static_cast<const uint64_t*>(v), n);
    }
}

constexpr uint64_t rle_bytes(uint32_t runs, uint32_t w) noexcept { return uint64_t{runs} * (w + 4u); }

/// values[runs] (W bytes each) and ends[runs] (exclusive end row of each run).
inline void rle_encode(const void* v, uint32_t n, uint32_t w, void* values, int32_t* ends) noexcept {
    assert(width_ok(w) && n > 0 && values != nullptr && ends != nullptr);
    switch (w) {
        case 1: detail::rle_encode_t(static_cast<const uint8_t*>(v), n, static_cast<uint8_t*>(values), ends); break;
        case 2: detail::rle_encode_t(static_cast<const uint16_t*>(v), n, static_cast<uint16_t*>(values), ends); break;
        case 4: detail::rle_encode_t(static_cast<const uint32_t*>(v), n, static_cast<uint32_t*>(values), ends); break;
        default: detail::rle_encode_t(static_cast<const uint64_t*>(v), n, static_cast<uint64_t*>(values), ends);
    }
}

/// False unless the run ends are strictly increasing and the last is n.
[[nodiscard]] inline bool rle_decode(const void* values, uint64_t values_len, const int32_t* ends, uint64_t ends_len,
                                     uint32_t runs, uint32_t w, uint32_t n, void* out) noexcept {
    if (!width_ok(w) || runs == 0 || runs > n || values_len != uint64_t{runs} * w ||
        ends_len != uint64_t{runs} * sizeof(int32_t) || out == nullptr)
        return false;
    switch (w) {
        case 1: return detail::rle_decode_t(static_cast<const uint8_t*>(values), ends, runs, n, static_cast<uint8_t*>(out));
        case 2: return detail::rle_decode_t(static_cast<const uint16_t*>(values), ends, runs, n, static_cast<uint16_t*>(out));
        case 4: return detail::rle_decode_t(static_cast<const uint32_t*>(values), ends, runs, n, static_cast<uint32_t*>(out));
        default: return detail::rle_decode_t(static_cast<const uint64_t*>(values), ends, runs, n, static_cast<uint64_t*>(out));
    }
}

// ---------------------------------------------------------------------------------------------------------
// Block unpack: one bit stream per lane, row order (the layout of fastlanes::pack_block).
// ---------------------------------------------------------------------------------------------------------

namespace detail {

template <class U>
inline void unpack_block_fast(const U* BOLT_RESTRICT packed, uint32_t w, U* BOLT_RESTRICT out) noexcept {
    constexpr uint32_t T = fastlanes::bits<U>(), L = fastlanes::lanes<U>();
    assert(w <= T);
    if (w == 0) { std::memset(out, 0, sizeof(U) * kBlock); return; }
    const uint64_t m = fastlanes::mask_of<U>(w);
    for (uint32_t lane = 0; lane < L; ++lane) {                     // bounded: lanes
        for (uint32_t row = 0; row < T; ++row) {                    // bounded: T
            const uint32_t bit = row * w, word = bit / T, off = bit % T;
            uint64_t x = static_cast<uint64_t>(packed[L * word + lane]) >> off;
            // off + w > T only when a next word exists (w * T bits = w words per lane)
            if (off + w > T) x |= static_cast<uint64_t>(packed[L * (word + 1) + lane]) << (T - off);
            out[fastlanes::index(row, lane)] = static_cast<U>(x & m);
        }
    }
}

template <class U> constexpr U sign_bit(bool is_signed) noexcept {
    return is_signed ? static_cast<U>(U{1} << (fastlanes::bits<U>() - 1)) : U{0};
}

}  // namespace detail

// ---------------------------------------------------------------------------------------------------------
// Frame of reference
// ---------------------------------------------------------------------------------------------------------

struct ForPlan {
    uint64_t ref;     // the minimum (signed order when is_signed), as raw T-bit word
    uint32_t width;   // bits of (max - min)
    uint64_t bytes;   // packed size
};

namespace detail {

template <class U>
inline ForPlan for_plan_t(const U* BOLT_RESTRICT v, uint32_t n, bool is_signed) noexcept {
    const U s = sign_bit<U>(is_signed);
    U lo = n ? static_cast<U>(v[0] ^ s) : U{0}, hi = lo;
    for (uint32_t i = 1; i < n; ++i) {                              // bounded: n
        const U x = static_cast<U>(v[i] ^ s);
        lo = x < lo ? x : lo;
        hi = x > hi ? x : hi;
    }
    ForPlan p;
    p.ref = static_cast<U>(lo ^ s);
    p.width = fastlanes::width_of(static_cast<uint64_t>(static_cast<U>(hi - lo)));
    p.bytes = fastlanes::packed_bytes(fastlanes::bits<U>(), n, p.width);
    return p;
}

template <class U>
inline void for_encode_t(const U* BOLT_RESTRICT v, uint32_t n, U ref, uint32_t w, U* BOLT_RESTRICT out) noexcept {
    alignas(64) U block[kBlock];
    const int64_t nb = fastlanes::blocks(n);
    for (int64_t b = 0; b < nb; ++b) {                              // bounded: blocks
        const uint32_t base = static_cast<uint32_t>(b) * kBlock;
        const uint32_t m = n - base < kBlock ? n - base : kBlock;
        for (uint32_t i = 0; i < m; ++i) block[i] = static_cast<U>(v[base + i] - ref);
        for (uint32_t i = m; i < kBlock; ++i) block[i] = U{0};
        fastlanes::pack_block<U>(block, w, out + static_cast<size_t>(b) * fastlanes::lanes<U>() * w);
    }
}

template <class U>
inline void for_decode_t(const U* BOLT_RESTRICT in, uint32_t n, U ref, uint32_t w, U* BOLT_RESTRICT out) noexcept {
    alignas(64) U block[kBlock];
    const int64_t nb = fastlanes::blocks(n);
    for (int64_t b = 0; b < nb; ++b) {                              // bounded: blocks
        unpack_block_fast<U>(in + static_cast<size_t>(b) * fastlanes::lanes<U>() * w, w, block);
        const uint32_t base = static_cast<uint32_t>(b) * kBlock;
        const uint32_t m = n - base < kBlock ? n - base : kBlock;
        for (uint32_t i = 0; i < m; ++i) out[base + i] = static_cast<U>(block[i] + ref);
    }
}

}  // namespace detail

inline ForPlan for_plan(const void* v, uint32_t n, uint32_t w, bool is_signed) noexcept {
    assert(width_ok(w) && n > 0);
    switch (w) {
        case 1: return detail::for_plan_t(static_cast<const uint8_t*>(v), n, is_signed);
        case 2: return detail::for_plan_t(static_cast<const uint16_t*>(v), n, is_signed);
        case 4: return detail::for_plan_t(static_cast<const uint32_t*>(v), n, is_signed);
        default: return detail::for_plan_t(static_cast<const uint64_t*>(v), n, is_signed);
    }
}

/// out: plan.bytes bytes, word aligned.
inline void for_encode(const void* v, uint32_t n, uint32_t w, const ForPlan& p, void* out) noexcept {
    assert(width_ok(w) && out != nullptr);
    switch (w) {
        case 1: detail::for_encode_t(static_cast<const uint8_t*>(v), n, static_cast<uint8_t>(p.ref), p.width, static_cast<uint8_t*>(out)); break;
        case 2: detail::for_encode_t(static_cast<const uint16_t*>(v), n, static_cast<uint16_t>(p.ref), p.width, static_cast<uint16_t*>(out)); break;
        case 4: detail::for_encode_t(static_cast<const uint32_t*>(v), n, static_cast<uint32_t>(p.ref), p.width, static_cast<uint32_t*>(out)); break;
        default: detail::for_encode_t(static_cast<const uint64_t*>(v), n, p.ref, p.width, static_cast<uint64_t*>(out));
    }
}

[[nodiscard]] inline bool for_decode(const void* in, uint64_t in_len, uint32_t n, uint32_t w, uint64_t ref,
                                     uint32_t width, void* out) noexcept {
    if (!width_ok(w) || n == 0 || width > w * 8u || out == nullptr) return false;
    if (in_len != fastlanes::packed_bytes(w * 8u, n, width)) return false;
    switch (w) {
        case 1: detail::for_decode_t(static_cast<const uint8_t*>(in), n, static_cast<uint8_t>(ref), width, static_cast<uint8_t*>(out)); break;
        case 2: detail::for_decode_t(static_cast<const uint16_t*>(in), n, static_cast<uint16_t>(ref), width, static_cast<uint16_t*>(out)); break;
        case 4: detail::for_decode_t(static_cast<const uint32_t*>(in), n, static_cast<uint32_t>(ref), width, static_cast<uint32_t*>(out)); break;
        default: detail::for_decode_t(static_cast<const uint64_t*>(in), n, ref, width, static_cast<uint64_t*>(out));
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------
// Delta with exceptions
// ---------------------------------------------------------------------------------------------------------

struct DeltaPlan {
    uint64_t dref;         // frame start of the chain deltas, raw T-bit word
    uint32_t width;        // bits of an in-frame (delta - dref)
    uint32_t exceptions;   // deltas outside the frame
    uint64_t bytes;        // packed size + exceptions
};

namespace detail {

// Chain delta of natural position k of a block: lane chains in row order over the transposed block.
// For each (lane, row >= 1) the pair is (index(row - 1, lane), index(row, lane)) through transpose().
template <class U, class F>
inline void for_each_delta(const U* BOLT_RESTRICT v, uint32_t n, F&& f) noexcept {
    constexpr uint32_t T = fastlanes::bits<U>(), L = fastlanes::lanes<U>();
    const int64_t nb = fastlanes::blocks(n);
    for (int64_t b = 0; b < nb; ++b) {                              // bounded: blocks
        const uint32_t base = static_cast<uint32_t>(b) * kBlock;
        for (uint32_t lane = 0; lane < L; ++lane) {                 // bounded: lanes
            uint32_t pk = base + fastlanes::transpose(fastlanes::index(0, lane));
            for (uint32_t row = 1; row < T; ++row) {                // bounded: T
                const uint32_t k = base + fastlanes::transpose(fastlanes::index(row, lane));
                if (k < n && pk < n) f(static_cast<U>(v[k] - v[pk]));
                pk = k;
            }
        }
    }
}

template <class U>
inline DeltaPlan delta_plan_t(const U* BOLT_RESTRICT v, uint32_t n) noexcept {
    constexpr uint32_t T = fastlanes::bits<U>();
    using S = std::make_signed_t<U>;
    // pass 1: the smallest non-negative delta, and the full range
    bool any = false, any_pos = false;
    S lo = 0, hi = 0, lo_pos = 0;
    for_each_delta<U>(v, n, [&](U d) noexcept {
        const S s = static_cast<S>(d);
        lo = (!any || s < lo) ? s : lo;
        hi = (!any || s > hi) ? s : hi;
        any = true;
        if (s >= 0) { lo_pos = (!any_pos || s < lo_pos) ? s : lo_pos; any_pos = true; }
    });
    DeltaPlan plain;
    plain.dref = static_cast<U>(lo);
    plain.width = any ? fastlanes::width_of(static_cast<uint64_t>(static_cast<U>(static_cast<U>(hi) - static_cast<U>(lo)))) : 0u;
    plain.exceptions = 0;
    plain.bytes = fastlanes::delta_bytes(T, n, plain.width);
    if (!any_pos) return plain;                                     // every step backward: no frame to keep
    // pass 2: widths of (delta - lo_pos) for deltas >= lo_pos; a smaller delta is an exception at any width
    uint32_t hist[65] = {};
    uint32_t below = 0;
    for_each_delta<U>(v, n, [&](U d) noexcept {
        const S s = static_cast<S>(d);
        const bool in = s >= lo_pos;
        below += !in;
        hist[in ? fastlanes::width_of(static_cast<uint64_t>(static_cast<U>(d - static_cast<U>(lo_pos)))) : 0u] += in;
    });
    DeltaPlan best = plain;
    uint32_t over = 0;
    for (uint32_t w = 1; w <= T; ++w) over += hist[w];              // bounded: 64
    for (uint32_t w = 0; w <= T; ++w) {                             // bounded: 65
        if (w != 0) over -= hist[w];
        const uint64_t bytes = fastlanes::delta_bytes(T, n, w) + uint64_t{below + over} * sizeof(Exception);
        if (bytes < best.bytes) {
            best.dref = static_cast<U>(lo_pos);
            best.width = w;
            best.exceptions = below + over;
            best.bytes = bytes;
        }
    }
    return best;
}

// out: per block [bases: L words][packed: L * w words]; exc: plan.exceptions entries, ascending pos.
template <class U>
inline uint32_t delta_encode_t(const U* BOLT_RESTRICT v, uint32_t n, U dref, uint32_t w, U* BOLT_RESTRICT out,
                               Exception* BOLT_RESTRICT exc, uint32_t exc_cap) noexcept {
    constexpr uint32_t T = fastlanes::bits<U>(), L = fastlanes::lanes<U>();
    alignas(64) U block[kBlock];
    const uint64_t m = fastlanes::mask_of<U>(w);
    const int64_t nb = fastlanes::blocks(n);
    uint32_t ne = 0;
    for (int64_t b = 0; b < nb; ++b) {                              // bounded: blocks
        const uint32_t base = static_cast<uint32_t>(b) * kBlock;
        U* bases = out + static_cast<size_t>(b) * L * (w + 1u);
        const uint32_t e0 = ne;
        for (uint32_t lane = 0; lane < L; ++lane) {                 // bounded: lanes
            const uint32_t i0 = fastlanes::index(0, lane);
            const uint32_t k0 = base + fastlanes::transpose(i0);
            // a lane that starts past the end has no real value: its base is 0 and its slots pack 0
            U prev = k0 < n ? v[k0] : U{0};
            bases[lane] = static_cast<U>(prev - dref);              // row 0 then adds dref + 0
            block[i0] = U{0};
            bool live = k0 < n;
            for (uint32_t row = 1; row < T; ++row) {                // bounded: T
                const uint32_t i = fastlanes::index(row, lane);
                const uint32_t k = base + fastlanes::transpose(i);
                U slot = U{0};
                if (k < n && live) {
                    const U u = static_cast<U>(static_cast<U>(v[k] - prev) - dref);
                    if ((static_cast<uint64_t>(u) & ~m) != 0) {
                        if (ne < exc_cap) exc[ne] = Exception{base + i, 0u, static_cast<uint64_t>(u)};
                        ++ne;
                    } else {
                        slot = u;
                    }
                    prev = v[k];
                } else if (k < n) {                                 // first real value of a lane whose row 0 is padding
                    const U u = static_cast<U>(static_cast<U>(v[k] - prev) - dref);
                    if (ne < exc_cap) exc[ne] = Exception{base + i, 0u, static_cast<uint64_t>(u)};
                    ++ne;
                    prev = v[k];
                    live = true;
                } else {
                    prev = static_cast<U>(prev + dref);             // padding continues the chain
                }
                block[i] = slot;
            }
        }
        // exceptions of one block ascend by position
        for (uint32_t a = e0 + 1; a < ne && a < exc_cap; ++a) {     // bounded: block exceptions (insertion sort)
            const Exception x = exc[a];
            uint32_t j = a;
            for (; j > e0 && exc[j - 1].pos > x.pos; --j) exc[j] = exc[j - 1];
            exc[j] = x;
        }
        fastlanes::pack_block<U>(block, w, bases + L);
    }
    return ne;
}

template <class U>
inline bool delta_decode_t(const U* BOLT_RESTRICT in, uint32_t n, U dref, uint32_t w,
                           const Exception* BOLT_RESTRICT exc, uint32_t n_exc, U* BOLT_RESTRICT out) noexcept {
    constexpr uint32_t T = fastlanes::bits<U>(), L = fastlanes::lanes<U>();
    alignas(64) U block[kBlock];
    const int64_t nb = fastlanes::blocks(n);
    uint32_t e = 0;
    for (int64_t b = 0; b < nb; ++b) {                              // bounded: blocks
        const U* bases = in + static_cast<size_t>(b) * L * (w + 1u);
        unpack_block_fast<U>(bases + L, w, block);
        const uint32_t base = static_cast<uint32_t>(b) * kBlock;
        for (; e < n_exc && exc[e].pos < base + kBlock; ++e) {      // bounded: exceptions
            if (exc[e].pos < base || (e != 0 && exc[e].pos <= exc[e - 1].pos)) return false;
            block[exc[e].pos - base] = static_cast<U>(exc[e].value);
        }
        for (uint32_t lane = 0; lane < L; ++lane) {                 // bounded: lanes
            U prev = bases[lane];
            for (uint32_t row = 0; row < T; ++row) {                // bounded: T
                const uint32_t i = fastlanes::index(row, lane);
                prev = static_cast<U>(prev + dref + block[i]);
                const uint32_t k = base + fastlanes::transpose(i);
                if (k < n) out[k] = prev;
            }
        }
    }
    return e == n_exc;
}

}  // namespace detail

/// A plan whose bytes is UINT64_MAX is not encodable: below 8-byte words a lane's chain is not in row order, so
/// a short last block would put padding between real values; those pages take another encoding.
inline DeltaPlan delta_plan(const void* v, uint32_t n, uint32_t w) noexcept {
    assert(width_ok(w) && n > 0);
    if (w != 8 && n % kBlock != 0) return DeltaPlan{0, 0, 0, UINT64_MAX};
    switch (w) {
        case 1: return detail::delta_plan_t(static_cast<const uint8_t*>(v), n);
        case 2: return detail::delta_plan_t(static_cast<const uint16_t*>(v), n);
        case 4: return detail::delta_plan_t(static_cast<const uint32_t*>(v), n);
        default: return detail::delta_plan_t(static_cast<const uint64_t*>(v), n);
    }
}

constexpr uint64_t delta_packed_bytes(uint32_t w, uint32_t n, uint32_t width) noexcept {
    return uint64_t{(n + kBlock - 1) / kBlock} * (kBlock / 8u) * (width + 1u);
}

/// out: delta_packed_bytes(); exc: plan.exceptions entries. Returns the exceptions written (== plan.exceptions).
inline uint32_t delta_encode(const void* v, uint32_t n, uint32_t w, const DeltaPlan& p, void* out,
                             Exception* exc) noexcept {
    assert(width_ok(w) && out != nullptr && (exc != nullptr || p.exceptions == 0));
    switch (w) {
        case 1: return detail::delta_encode_t(static_cast<const uint8_t*>(v), n, static_cast<uint8_t>(p.dref), p.width, static_cast<uint8_t*>(out), exc, p.exceptions);
        case 2: return detail::delta_encode_t(static_cast<const uint16_t*>(v), n, static_cast<uint16_t>(p.dref), p.width, static_cast<uint16_t*>(out), exc, p.exceptions);
        case 4: return detail::delta_encode_t(static_cast<const uint32_t*>(v), n, static_cast<uint32_t>(p.dref), p.width, static_cast<uint32_t*>(out), exc, p.exceptions);
        default: return detail::delta_encode_t(static_cast<const uint64_t*>(v), n, p.dref, p.width, static_cast<uint64_t*>(out), exc, p.exceptions);
    }
}

[[nodiscard]] inline bool delta_decode(const void* in, uint64_t in_len, uint32_t n, uint32_t w, uint64_t dref,
                                       uint32_t width, const Exception* exc, uint64_t exc_len, void* out) noexcept {
    if (!width_ok(w) || n == 0 || width > w * 8u || out == nullptr) return false;
    if (in_len != delta_packed_bytes(w, n, width) || exc_len % sizeof(Exception) != 0) return false;
    const uint64_t ne = exc_len / sizeof(Exception);
    if (ne > uint64_t{(n + kBlock - 1) / kBlock} * kBlock) return false;
    const uint32_t k = static_cast<uint32_t>(ne);
    switch (w) {
        case 1: return detail::delta_decode_t(static_cast<const uint8_t*>(in), n, static_cast<uint8_t>(dref), width, exc, k, static_cast<uint8_t*>(out));
        case 2: return detail::delta_decode_t(static_cast<const uint16_t*>(in), n, static_cast<uint16_t>(dref), width, exc, k, static_cast<uint16_t*>(out));
        case 4: return detail::delta_decode_t(static_cast<const uint32_t*>(in), n, static_cast<uint32_t>(dref), width, exc, k, static_cast<uint32_t*>(out));
        default: return detail::delta_decode_t(static_cast<const uint64_t*>(in), n, dref, width, exc, k, static_cast<uint64_t*>(out));
    }
}

}  // namespace pagecodec
}  // namespace bolt
