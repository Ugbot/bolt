// sha1.cpp — self-contained SHA-1.

#include "bolt/crypto/sha1.h"

#include <cassert>
#include <cstdint>
#include <cstring>

namespace bolt {
namespace crypto {

namespace {

inline uint32_t rol(uint32_t x, int n) noexcept {
    return (x << n) | (x >> (32 - n));
}

void sha1_block(uint32_t h[5], const uint8_t* blk) noexcept {
    assert(h != nullptr && blk != nullptr);
    uint32_t w[80];
    for (int i = 0; i < 16; ++i) {
        w[i] = (uint32_t{blk[4 * i]} << 24) | (uint32_t{blk[4 * i + 1]} << 16) |
               (uint32_t{blk[4 * i + 2]} << 8) | uint32_t{blk[4 * i + 3]};
    }
    for (int i = 16; i < 80; ++i) {
        w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
    for (int i = 0; i < 80; ++i) {
        uint32_t f = 0, k = 0;
        if (i < 20) {
            f = (b & c) | (~b & d);
            k = 0x5A827999u;
        } else if (i < 40) {
            f = b ^ c ^ d;
            k = 0x6ED9EBA1u;
        } else if (i < 60) {
            f = (b & c) | (b & d) | (c & d);
            k = 0x8F1BBCDCu;
        } else {
            f = b ^ c ^ d;
            k = 0xCA62C1D6u;
        }
        const uint32_t t = rol(a, 5) + f + e + k + w[i];
        e = d;
        d = c;
        c = rol(b, 30);
        b = a;
        a = t;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
}

}  // namespace

void sha1(const uint8_t* data, uint64_t len, uint8_t out[20]) noexcept {
    assert(data != nullptr || len == 0);
    assert(out != nullptr);
    uint32_t h[5] = {0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u,
                     0xC3D2E1F0u};
    uint64_t off = 0;
    while (len - off >= 64u) {
        sha1_block(h, data + off);
        off += 64u;
    }
    uint8_t tail[128] = {};
    const uint64_t rem = len - off;
    if (rem > 0u) std::memcpy(tail, data + off, static_cast<size_t>(rem));
    tail[rem] = 0x80u;
    const uint64_t tail_len = (rem < 56u) ? 64u : 128u;
    const uint64_t bits = len * 8u;
    for (int i = 0; i < 8; ++i) {
        tail[tail_len - 1 - static_cast<uint64_t>(i)] =
            static_cast<uint8_t>(bits >> (8 * i));
    }
    sha1_block(h, tail);
    if (tail_len == 128u) sha1_block(h, tail + 64);
    for (int i = 0; i < 5; ++i) {
        out[4 * i] = static_cast<uint8_t>(h[i] >> 24);
        out[4 * i + 1] = static_cast<uint8_t>(h[i] >> 16);
        out[4 * i + 2] = static_cast<uint8_t>(h[i] >> 8);
        out[4 * i + 3] = static_cast<uint8_t>(h[i]);
    }
}

}  // namespace crypto
}  // namespace bolt
