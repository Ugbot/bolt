// bolt/ingest/bolt_snappy.h — W-PQ: minimal snappy BLOCK decompressor
// (the format parquet SNAPPY pages use — raw block format, NOT the
// framing format). Zero dependencies.
//
// Format (https://github.com/google/snappy/blob/main/format_description.txt):
//   [uvarint uncompressed_length] then a tag stream:
//     tag & 3 == 0 : literal,  len = (tag>>2)+1, or 60..63 => 1..4 extra
//                    length bytes (little-endian)
//     tag & 3 == 1 : copy, len = ((tag>>2)&7)+4, offset = ((tag>>5)<<8)|byte
//     tag & 3 == 2 : copy, len = (tag>>2)+1, offset = 2-byte LE
//     tag & 3 == 3 : copy, len = (tag>>2)+1, offset = 4-byte LE
//   Copies may overlap forward (offset < len) — byte-by-byte semantics.
//
// Safety contract: every read/write is bounds-checked; corrupt input
// returns false, never UB (fuzzed in the parquet test suite under ASAN).
// Tiger Style: noexcept, no allocation, bounded loops, >=2 asserts.

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

namespace bolt {
namespace ingest {

// Parse the uncompressed-length preamble. Returns bytes consumed (1..5)
// or 0 on corrupt/oversized input.
inline uint32_t snappy_uncompressed_len(const uint8_t* src, uint64_t src_len,
                                        uint64_t* out_len) noexcept {
    assert(out_len != nullptr);
    if (src == nullptr || src_len == 0) return 0;
    uint64_t v = 0;
    uint32_t i = 0;
    for (; i < 5 && i < src_len; ++i) {
        v |= static_cast<uint64_t>(src[i] & 0x7Fu) << (7u * i);
        if ((src[i] & 0x80u) == 0) {
            if (v > (1ull << 32)) return 0;   // snappy caps at 2^32
            *out_len = v;
            return i + 1;
        }
    }
    return 0;
}

// Copy an 8-byte word. Fixed size, so this compiles to one load + one
// store — never a call into libc's runtime size-dispatch ladder.
inline void snappy_copy8(uint8_t* dst, const uint8_t* src) noexcept {
    uint64_t w;
    std::memcpy(&w, src, 8);
    std::memcpy(dst, &w, 8);
}

// Copy `len` bytes to dst+op from dst+op-off, where `off` may be SMALLER
// than `len` (a repeating pattern — snappy's overlapping copy).
//
// Why not std::memcpy: a snappy back-reference is 1..64 bytes BY FORMAT, so
// a runtime-length memcpy pays a call plus libc's size dispatch to move a
// handful of bytes. Measured at ~22% of total query CPU on TPC-H SF100
// (dtrace run 7f5b0152, `_platform_memmove` under decompress_page). This
// moves the same bytes with fixed-size 8-byte stores and no call.
//
// It OVERCOPIES up to 7 bytes past the run, so the caller must have proven
// `op + len + 8 <= dst_len`. Only the first `len` bytes are meaningful;
// anything past them is either rewritten by a later tag or lies beyond the
// final `op == dst_len` fill point.
//
// Every 8-byte read lands entirely inside bytes that are already final. For
// off >= 8 that is immediate. For off < 8 the offset is first grown to the
// smallest multiple of `off` that is >= 8 (legal because the source region
// is periodic with period `off`) and the few bytes that multiple needs are
// materialised one at a time — so this never reads uninitialised memory,
// which the naive "just read 16 bytes back" formulation does.
inline void snappy_copy_match(uint8_t* dst, uint64_t op, uint64_t off,
                              uint64_t len) noexcept {
    assert(off > 0 && off <= op);
    assert(len > 0);
    uint64_t k = 0;
    uint64_t d = off;
    if (d < 8) {
        while (d < 8) d += off;              // bounded: <= 8 iterations
        const uint64_t prime = d - off;      // <= 7
        const uint64_t stop = (prime < len) ? prime : len;
        for (; k < stop; ++k) dst[op + k] = dst[op + k - off];
    }
    for (; k < len; k += 8) {                // bounded: len <= 64 => <= 8
        snappy_copy8(dst + op + k, dst + op + k - d);
    }
}

// Decompress one snappy block into dst (exactly dst_len bytes — the
// caller sizes dst from snappy_uncompressed_len). False on any corrupt
// shape: truncated tags, out-of-range offsets, output over/underflow.
//
// Writes NEVER exceed dst_len. The wide-store fast lanes are taken only
// when there is provably room; the tail falls back to exact copies. That
// keeps the exact-fill contract every caller relies on (kafka_wire sizes
// its output buffer to exactly the decoded length, as do the tests) rather
// than requiring callers to append slop bytes.
inline bool snappy_decompress(const uint8_t* src, uint64_t src_len,
                              uint8_t* dst, uint64_t dst_len) noexcept {
    assert(src != nullptr || src_len == 0);
    assert(dst != nullptr || dst_len == 0);
    uint64_t hdr_len = 0;
    const uint32_t consumed = snappy_uncompressed_len(src, src_len, &hdr_len);
    if (consumed == 0 || hdr_len != dst_len) return false;
    uint64_t ip = consumed;
    uint64_t op = 0;
    while (ip < src_len) {                       // bounded: ip advances >=1
        const uint8_t tag = src[ip++];
        if ((tag & 3u) == 0) {                   // literal
            uint64_t len = (tag >> 2) + 1u;
            if (len > 60u) {
                const uint32_t nb = static_cast<uint32_t>(len - 60u);
                if (ip + nb > src_len) return false;
                uint64_t v = 0;
                for (uint32_t k = 0; k < nb; ++k) {
                    v |= static_cast<uint64_t>(src[ip + k]) << (8u * k);
                }
                ip += nb;
                len = v + 1u;
            }
            if (ip + len > src_len || op + len > dst_len) return false;
            // Short literals dominate dictionary/RLE-encoded parquet pages.
            // Two fixed 8-byte moves beat a call whose whole job is to
            // dispatch on a size that is almost always < 16.
            if (len <= 16u && ip + 16u <= src_len && op + 16u <= dst_len) {
                snappy_copy8(dst + op, src + ip);
                snappy_copy8(dst + op + 8u, src + ip + 8u);
            } else {
                std::memcpy(dst + op, src + ip, len);
            }
            ip += len;
            op += len;
            continue;
        }
        uint64_t len = 0, off = 0;
        switch (tag & 3u) {
            case 1:
                if (ip + 1 > src_len) return false;
                len = ((tag >> 2) & 7u) + 4u;
                off = (static_cast<uint64_t>(tag >> 5) << 8) | src[ip];
                ip += 1;
                break;
            case 2:
                if (ip + 2 > src_len) return false;
                len = (tag >> 2) + 1u;
                off = static_cast<uint64_t>(src[ip]) |
                      (static_cast<uint64_t>(src[ip + 1]) << 8);
                ip += 2;
                break;
            default:   // 3
                if (ip + 4 > src_len) return false;
                len = (tag >> 2) + 1u;
                std::uint32_t o32;
                std::memcpy(&o32, src + ip, 4);
                off = o32;
                ip += 4;
                break;
        }
        if (off == 0 || off > op) return false;          // bad back-ref
        if (op + len > dst_len) return false;            // output overflow
        if (op + len + 8u <= dst_len) {                  // room to overcopy
            snappy_copy_match(dst, op, off, len);
        } else if (off >= len) {                         // tail, no overlap
            std::memcpy(dst + op, dst + op - off, len);
        } else {                                         // tail, overlapping
            for (uint64_t k = 0; k < len; ++k) {
                dst[op + k] = dst[op + k - off];
            }
        }
        op += len;
    }
    return op == dst_len;                        // exact fill required
}

// ---------------------------------------------------------------------------
// W-PQ-W: minimal Snappy block COMPRESSOR.
//
// Encodes input as a single literal chunk preceded by the uncompressed-length
// varint. The format permits this — a literal-only stream is a valid Snappy
// block (just no compression benefit). Parquet's SNAPPY codec accepts it,
// and the decoder above round-trips it exactly. This keeps the encoder
// dependency-free and Tiger-Style tiny while still letting us claim
// "compression = 1 / SNAPPY" in the page metadata.
//
// Worst-case output is `src_len + varint_len(src_len) + 5` (one literal
// header per <= 4 GiB chunk). `snappy_max_compressed_len` gives a safe
// upper bound the caller sizes `dst` from.
// ---------------------------------------------------------------------------

inline uint64_t snappy_max_compressed_len(uint64_t src_len) noexcept {
    // 5 bytes for the uncompressed-length varint (covers up to 2^32),
    // 5 bytes for the worst-case literal-length tag prefix (60..63 means
    // 1..4 extra length bytes), then the data itself.
    return src_len + 32u;
}

inline bool snappy_compress(const uint8_t* src, uint64_t src_len,
                            uint8_t* dst, uint64_t dst_cap,
                            uint64_t* out_len) noexcept {
    assert(src != nullptr || src_len == 0);
    assert(dst != nullptr || dst_cap == 0);
    assert(out_len != nullptr);
    if (src_len > (uint64_t{1} << 32)) return false;       // snappy caps at 2^32
    // 1) uncompressed-length varint.
    uint64_t op = 0;
    {
        uint64_t v = src_len;
        for (int i = 0; i < 5; ++i) {
            if (op >= dst_cap) return false;
            const uint8_t b = static_cast<uint8_t>(v & 0x7Fu);
            v >>= 7;
            if (v == 0) { dst[op++] = b; break; }
            dst[op++] = static_cast<uint8_t>(b | 0x80u);
        }
    }
    if (src_len == 0) { *out_len = op; return true; }
    // 2) one literal chunk that covers the entire input.
    const uint64_t lit_len = src_len - 1u;       // tag encodes (len-1)
    if (lit_len < 60u) {
        if (op + 1u + src_len > dst_cap) return false;
        dst[op++] = static_cast<uint8_t>(lit_len << 2);
    } else {
        // Determine how many extra length bytes we need (1..4).
        uint32_t extra = 1u;
        if (lit_len > 0xFFu)     extra = 2u;
        if (lit_len > 0xFFFFu)   extra = 3u;
        if (lit_len > 0xFFFFFFu) extra = 4u;
        if (op + 1u + extra + src_len > dst_cap) return false;
        dst[op++] = static_cast<uint8_t>(((59u + extra) << 2));
        for (uint32_t k = 0; k < extra; ++k) {
            dst[op++] = static_cast<uint8_t>((lit_len >> (8u * k)) & 0xFFu);
        }
    }
    std::memcpy(dst + op, src, src_len);
    op += src_len;
    *out_len = op;
    return true;
}

}  // namespace ingest
}  // namespace bolt
