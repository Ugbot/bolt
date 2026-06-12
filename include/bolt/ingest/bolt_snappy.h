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

// Decompress one snappy block into dst (exactly dst_len bytes — the
// caller sizes dst from snappy_uncompressed_len). False on any corrupt
// shape: truncated tags, out-of-range offsets, output over/underflow.
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
            std::memcpy(dst + op, src + ip, len);
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
        if (off >= len) {                                // non-overlapping
            std::memcpy(dst + op, dst + op - off, len);
        } else {                                         // overlapping run
            for (uint64_t k = 0; k < len; ++k) {
                dst[op + k] = dst[op + k - off];
            }
        }
        op += len;
    }
    return op == dst_len;                        // exact fill required
}

}  // namespace ingest
}  // namespace bolt
