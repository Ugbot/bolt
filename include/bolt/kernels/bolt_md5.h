// bolt_md5.h — MD5 message digest (RFC 1321), header-only.
//
// Not a security primitive: MD5 is broken for collision resistance. It exists
// for SQL `md5(s)` (dbt.hash / generate_surrogate_key), where the contract is
// byte-exact agreement with DuckDB / Postgres / Python hashlib.
//
// RULES: No exceptions. No RTTI. No heap. All fns noexcept. Bounded loops.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_port.h"

namespace bolt {
namespace kernels {
namespace md5 {

constexpr uint32_t k_digest_bytes = 16;
constexpr uint32_t k_hex_chars    = 32;
constexpr uint32_t k_block_bytes  = 64;

namespace detail {

BOLT_FORCE_INLINE uint32_t rotl(uint32_t x, uint32_t c) noexcept {
    return (x << c) | (x >> (32u - c));
}

BOLT_FORCE_INLINE uint32_t load_le32(const uint8_t* p) noexcept {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

inline constexpr uint32_t k_shift[64] = {
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
};

// floor(abs(sin(i + 1)) * 2^32), RFC 1321 §3.4.
inline constexpr uint32_t k_sine[64] = {
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a,
    0xa8304613, 0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
    0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340,
    0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8,
    0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
    0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
    0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92,
    0xffeff47d, 0x85845dd1, 0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
    0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
};

inline void compress(uint32_t state[4], const uint8_t* block) noexcept {
    assert(state != nullptr);
    assert(block != nullptr);
    uint32_t m[16];
    for (uint32_t i = 0; i < 16; ++i) m[i] = load_le32(block + 4u * i);
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    for (uint32_t i = 0; i < 64; ++i) {
        uint32_t f = 0, g = 0;
        if (i < 16)      { f = (b & c) | (~b & d); g = i; }
        else if (i < 32) { f = (d & b) | (~d & c); g = (5u * i + 1u) & 15u; }
        else if (i < 48) { f = b ^ c ^ d;          g = (3u * i + 5u) & 15u; }
        else             { f = c ^ (b | ~d);       g = (7u * i) & 15u; }
        const uint32_t t = d;
        d = c;
        c = b;
        b = b + rotl(a + f + k_sine[i] + m[g], k_shift[i]);
        a = t;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
}

}  // namespace detail

// 16-byte digest of data[0, len). data may be nullptr only when len == 0.
inline void md5(const uint8_t* BOLT_RESTRICT data, uint64_t len,
                uint8_t out[k_digest_bytes]) noexcept {
    assert(out != nullptr);
    assert(data != nullptr || len == 0);
    uint32_t state[4] = {0x67452301u, 0xefcdab89u, 0x98badcfeu, 0x10325476u};
    const uint64_t full = len / k_block_bytes;
    for (uint64_t blk = 0; blk < full; ++blk) {
        detail::compress(state, data + blk * k_block_bytes);
    }
    // Tail: 0x80, zero pad, then the 64-bit little-endian bit length. One or
    // two final blocks depending on whether the length fits after the tail.
    uint8_t tail[2 * k_block_bytes];
    std::memset(tail, 0, sizeof(tail));
    const uint32_t rem = static_cast<uint32_t>(len - full * k_block_bytes);
    assert(rem < k_block_bytes);
    if (rem != 0) std::memcpy(tail, data + full * k_block_bytes, rem);
    tail[rem] = 0x80u;
    const uint32_t tail_len = (rem < 56u) ? k_block_bytes : 2u * k_block_bytes;
    const uint64_t bits = len * 8u;
    for (uint32_t i = 0; i < 8; ++i) {
        tail[tail_len - 8u + i] = static_cast<uint8_t>(bits >> (8u * i));
    }
    for (uint32_t off = 0; off < tail_len; off += k_block_bytes) {
        detail::compress(state, tail + off);
    }
    for (uint32_t i = 0; i < 4; ++i) {
        out[4u * i + 0u] = static_cast<uint8_t>(state[i]);
        out[4u * i + 1u] = static_cast<uint8_t>(state[i] >> 8);
        out[4u * i + 2u] = static_cast<uint8_t>(state[i] >> 16);
        out[4u * i + 3u] = static_cast<uint8_t>(state[i] >> 24);
    }
}

// 32 lowercase hex chars, NOT NUL-terminated.
inline void md5_hex(const uint8_t* BOLT_RESTRICT data, uint64_t len,
                    char out[k_hex_chars]) noexcept {
    assert(out != nullptr);
    assert(data != nullptr || len == 0);
    static constexpr char k_hex[] = "0123456789abcdef";
    uint8_t d[k_digest_bytes];
    md5(data, len, d);
    for (uint32_t i = 0; i < k_digest_bytes; ++i) {
        out[2u * i]      = k_hex[d[i] >> 4];
        out[2u * i + 1u] = k_hex[d[i] & 0x0fu];
    }
}

}  // namespace md5
}  // namespace kernels
}  // namespace bolt
