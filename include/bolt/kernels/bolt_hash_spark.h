// bolt_hash_spark.h -- digests and hashes for the Spark functions md5 / sha / sha1 / sha2 / crc32 /
// hash / xxhash64, header-only, no heap. Each kernel is bit-exact with Spark 4 (and the JDK):
//   * crc32  : zlib CRC-32 (IEEE 802.3, reflected 0xEDB88320), slicing-by-8; NOT CRC-32C.
//   * sha2   : SHA-224 / SHA-256 / SHA-384 / SHA-512 (FIPS 180-4); sha1 and sha256 one-shots come
//              from bolt::crypto, md5 from bolt_md5.h.
//   * hash   : Spark's Murmur3_x86_32 (hashInt / hashLong / hashUnsafeBytes -- the tail bytes are
//              mixed one at a time, each sign-extended, which is NOT stock murmur3).
//   * xxhash64: Spark's XXH64 (hashInt / hashLong / hashUnsafeBytes).
// RULES: No exceptions. No RTTI. No heap. All fns noexcept. Bounded loops.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_port.h"
#include "bolt/crypto/sha1.h"
#include "bolt/crypto/sigv4.h"
#include "bolt/kernels/bolt_md5.h"

namespace bolt {
namespace kernels {
namespace hashx {

BOLT_FORCE_INLINE std::uint32_t rotl32(std::uint32_t x, unsigned r) noexcept { return (x << r) | (x >> (32u - r)); }
BOLT_FORCE_INLINE std::uint64_t rotl64(std::uint64_t x, unsigned r) noexcept { return (x << r) | (x >> (64u - r)); }
BOLT_FORCE_INLINE std::uint32_t load32(const std::uint8_t* p) noexcept {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}
BOLT_FORCE_INLINE std::uint64_t load64(const std::uint8_t* p) noexcept {
    return static_cast<std::uint64_t>(load32(p)) | (static_cast<std::uint64_t>(load32(p + 4)) << 32);
}
BOLT_FORCE_INLINE std::uint64_t load_be64(const std::uint8_t* p) noexcept {
    std::uint64_t v = 0;
    for (unsigned i = 0; i < 8; ++i) v = (v << 8) | p[i];
    return v;
}

// ---- CRC-32 (zlib) -----------------------------------------------------------------------
struct Crc32Tables {
    std::uint32_t t[8][256];
    constexpr Crc32Tables() : t{} {
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            t[0][i] = c;
        }
        for (std::uint32_t i = 0; i < 256; ++i) {
            for (int s = 1; s < 8; ++s) t[s][i] = (t[s - 1][i] >> 8) ^ t[0][t[s - 1][i] & 0xFFu];
        }
    }
};
inline constexpr Crc32Tables k_crc32_tables{};

// CRC-32 of p[0, n): the value of java.util.zip.CRC32 (what Spark's crc32() returns).
inline std::uint32_t crc32(const std::uint8_t* p, std::uint64_t n) noexcept {
    assert(p != nullptr || n == 0);
    std::uint32_t c = 0xFFFFFFFFu;
    while (n >= 8) {
        const std::uint32_t a = load32(p) ^ c;
        const std::uint32_t b = load32(p + 4);
        c = k_crc32_tables.t[7][a & 0xFFu] ^ k_crc32_tables.t[6][(a >> 8) & 0xFFu] ^
            k_crc32_tables.t[5][(a >> 16) & 0xFFu] ^ k_crc32_tables.t[4][a >> 24] ^
            k_crc32_tables.t[3][b & 0xFFu] ^ k_crc32_tables.t[2][(b >> 8) & 0xFFu] ^
            k_crc32_tables.t[1][(b >> 16) & 0xFFu] ^ k_crc32_tables.t[0][b >> 24];
        p += 8;
        n -= 8;
    }
    while (n-- > 0) c = k_crc32_tables.t[0][(c ^ *p++) & 0xFFu] ^ (c >> 8);
    return ~c;
}

// ---- SHA-2 (FIPS 180-4) -------------------------------------------------------------------
inline constexpr std::uint32_t k_sha256_k[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};
inline constexpr std::uint32_t k_sha224_iv[8] = {
    0xc1059ed8u, 0x367cd507u, 0x3070dd17u, 0xf70e5939u,
    0xffc00b31u, 0x68581511u, 0x64f98fa7u, 0xbefa4fa4u
};
inline constexpr std::uint64_t k_sha512_k[80] = {
    0x428a2f98d728ae22ull, 0x7137449123ef65cdull, 0xb5c0fbcfec4d3b2full, 0xe9b5dba58189dbbcull,
    0x3956c25bf348b538ull, 0x59f111f1b605d019ull, 0x923f82a4af194f9bull, 0xab1c5ed5da6d8118ull,
    0xd807aa98a3030242ull, 0x12835b0145706fbeull, 0x243185be4ee4b28cull, 0x550c7dc3d5ffb4e2ull,
    0x72be5d74f27b896full, 0x80deb1fe3b1696b1ull, 0x9bdc06a725c71235ull, 0xc19bf174cf692694ull,
    0xe49b69c19ef14ad2ull, 0xefbe4786384f25e3ull, 0x0fc19dc68b8cd5b5ull, 0x240ca1cc77ac9c65ull,
    0x2de92c6f592b0275ull, 0x4a7484aa6ea6e483ull, 0x5cb0a9dcbd41fbd4ull, 0x76f988da831153b5ull,
    0x983e5152ee66dfabull, 0xa831c66d2db43210ull, 0xb00327c898fb213full, 0xbf597fc7beef0ee4ull,
    0xc6e00bf33da88fc2ull, 0xd5a79147930aa725ull, 0x06ca6351e003826full, 0x142929670a0e6e70ull,
    0x27b70a8546d22ffcull, 0x2e1b21385c26c926ull, 0x4d2c6dfc5ac42aedull, 0x53380d139d95b3dfull,
    0x650a73548baf63deull, 0x766a0abb3c77b2a8ull, 0x81c2c92e47edaee6ull, 0x92722c851482353bull,
    0xa2bfe8a14cf10364ull, 0xa81a664bbc423001ull, 0xc24b8b70d0f89791ull, 0xc76c51a30654be30ull,
    0xd192e819d6ef5218ull, 0xd69906245565a910ull, 0xf40e35855771202aull, 0x106aa07032bbd1b8ull,
    0x19a4c116b8d2d0c8ull, 0x1e376c085141ab53ull, 0x2748774cdf8eeb99ull, 0x34b0bcb5e19b48a8ull,
    0x391c0cb3c5c95a63ull, 0x4ed8aa4ae3418acbull, 0x5b9cca4f7763e373ull, 0x682e6ff3d6b2b8a3ull,
    0x748f82ee5defb2fcull, 0x78a5636f43172f60ull, 0x84c87814a1f0ab72ull, 0x8cc702081a6439ecull,
    0x90befffa23631e28ull, 0xa4506cebde82bde9ull, 0xbef9a3f7b2c67915ull, 0xc67178f2e372532bull,
    0xca273eceea26619cull, 0xd186b8c721c0c207ull, 0xeada7dd6cde0eb1eull, 0xf57d4f7fee6ed178ull,
    0x06f067aa72176fbaull, 0x0a637dc5a2c898a6ull, 0x113f9804bef90daeull, 0x1b710b35131c471bull,
    0x28db77f523047d84ull, 0x32caab7b40c72493ull, 0x3c9ebe0a15c9bebcull, 0x431d67c49c100d4cull,
    0x4cc5d4becb3e42b6ull, 0x597f299cfc657e2aull, 0x5fcb6fab3ad6faecull, 0x6c44198c4a475817ull
};
inline constexpr std::uint64_t k_sha512_iv[8] = {
    0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull, 0x3c6ef372fe94f82bull, 0xa54ff53a5f1d36f1ull,
    0x510e527fade682d1ull, 0x9b05688c2b3e6c1full, 0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull
};
inline constexpr std::uint64_t k_sha384_iv[8] = {
    0xcbbb9d5dc1059ed8ull, 0x629a292a367cd507ull, 0x9159015a3070dd17ull, 0x152fecd8f70e5939ull,
    0x67332667ffc00b31ull, 0x8eb44a8768581511ull, 0xdb0c2e0d64f98fa7ull, 0x47b5481dbefa4fa4ull
};

inline void sha256_block(std::uint32_t h[8], const std::uint8_t* p) noexcept {
    std::uint32_t w[64];
    for (unsigned i = 0; i < 16; ++i) {
        w[i] = (static_cast<std::uint32_t>(p[4 * i]) << 24) | (static_cast<std::uint32_t>(p[4 * i + 1]) << 16) |
               (static_cast<std::uint32_t>(p[4 * i + 2]) << 8) | p[4 * i + 3];
    }
    for (unsigned i = 16; i < 64; ++i) {
        const std::uint32_t s0 = rotl32(w[i - 15], 25) ^ rotl32(w[i - 15], 14) ^ (w[i - 15] >> 3);
        const std::uint32_t s1 = rotl32(w[i - 2], 15) ^ rotl32(w[i - 2], 13) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (unsigned i = 0; i < 64; ++i) {
        const std::uint32_t S1 = rotl32(e, 26) ^ rotl32(e, 21) ^ rotl32(e, 7);
        const std::uint32_t t1 = hh + S1 + ((e & f) ^ (~e & g)) + k_sha256_k[i] + w[i];
        const std::uint32_t S0 = rotl32(a, 30) ^ rotl32(a, 19) ^ rotl32(a, 10);
        const std::uint32_t t2 = S0 + ((a & b) ^ (a & c) ^ (b & c));
        hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

inline void sha512_block(std::uint64_t h[8], const std::uint8_t* p) noexcept {
    std::uint64_t w[80];
    for (unsigned i = 0; i < 16; ++i) w[i] = load_be64(p + 8 * i);
    for (unsigned i = 16; i < 80; ++i) {
        const std::uint64_t s0 = rotl64(w[i - 15], 63) ^ rotl64(w[i - 15], 56) ^ (w[i - 15] >> 7);
        const std::uint64_t s1 = rotl64(w[i - 2], 45) ^ rotl64(w[i - 2], 3) ^ (w[i - 2] >> 6);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint64_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (unsigned i = 0; i < 80; ++i) {
        const std::uint64_t S1 = rotl64(e, 50) ^ rotl64(e, 46) ^ rotl64(e, 23);
        const std::uint64_t t1 = hh + S1 + ((e & f) ^ (~e & g)) + k_sha512_k[i] + w[i];
        const std::uint64_t S0 = rotl64(a, 36) ^ rotl64(a, 30) ^ rotl64(a, 25);
        const std::uint64_t t2 = S0 + ((a & b) ^ (a & c) ^ (b & c));
        hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

// Digest of data[0, len) with a Merkle-Damgard 32-bit-word hash (block 64, length field 8 bytes).
inline void sha224(const std::uint8_t* data, std::uint64_t len, std::uint8_t out[28]) noexcept {
    assert(out != nullptr && (data != nullptr || len == 0));
    std::uint32_t h[8];
    std::memcpy(h, k_sha224_iv, sizeof h);
    std::uint64_t full = len / 64u;
    for (std::uint64_t i = 0; i < full; ++i) sha256_block(h, data + 64u * i);
    std::uint8_t tail[128];
    std::memset(tail, 0, sizeof tail);
    const std::uint32_t rem = static_cast<std::uint32_t>(len - full * 64u);
    if (rem != 0) std::memcpy(tail, data + full * 64u, rem);
    tail[rem] = 0x80u;
    const std::uint32_t tl = rem < 56u ? 64u : 128u;
    const std::uint64_t bits = len * 8u;
    for (unsigned i = 0; i < 8; ++i) tail[tl - 1u - i] = static_cast<std::uint8_t>(bits >> (8u * i));
    for (std::uint32_t o = 0; o < tl; o += 64u) sha256_block(h, tail + o);
    for (unsigned i = 0; i < 7; ++i) {
        out[4 * i] = static_cast<std::uint8_t>(h[i] >> 24);
        out[4 * i + 1] = static_cast<std::uint8_t>(h[i] >> 16);
        out[4 * i + 2] = static_cast<std::uint8_t>(h[i] >> 8);
        out[4 * i + 3] = static_cast<std::uint8_t>(h[i]);
    }
}

// SHA-512 (out 64 bytes) or SHA-384 (out 48 bytes; `is384`).
inline void sha512(const std::uint8_t* data, std::uint64_t len, bool is384, std::uint8_t out[64]) noexcept {
    assert(out != nullptr && (data != nullptr || len == 0));
    std::uint64_t h[8];
    std::memcpy(h, is384 ? k_sha384_iv : k_sha512_iv, sizeof h);
    const std::uint64_t full = len / 128u;
    for (std::uint64_t i = 0; i < full; ++i) sha512_block(h, data + 128u * i);
    std::uint8_t tail[256];
    std::memset(tail, 0, sizeof tail);
    const std::uint32_t rem = static_cast<std::uint32_t>(len - full * 128u);
    if (rem != 0) std::memcpy(tail, data + full * 128u, rem);
    tail[rem] = 0x80u;
    const std::uint32_t tl = rem < 112u ? 128u : 256u;
    const std::uint64_t bits = len * 8u;       // the high 64 bits of the 128-bit length stay zero
    for (unsigned i = 0; i < 8; ++i) tail[tl - 1u - i] = static_cast<std::uint8_t>(bits >> (8u * i));
    tail[tl - 9u] = static_cast<std::uint8_t>(len >> 61);
    for (std::uint32_t o = 0; o < tl; o += 128u) sha512_block(h, tail + o);
    const unsigned words = is384 ? 6u : 8u;
    for (unsigned i = 0; i < words; ++i) {
        for (unsigned b = 0; b < 8; ++b) out[8 * i + b] = static_cast<std::uint8_t>(h[i] >> (56u - 8u * b));
    }
}

// Spark sha2(x, bits): the digest size in bytes for `bits` in {0, 224, 256, 384, 512}; 0 otherwise.
inline std::uint32_t sha2_digest_bytes(std::int64_t bits) noexcept {
    switch (bits) {
        case 0: case 256: return 32;
        case 224: return 28;
        case 384: return 48;
        case 512: return 64;
        default: return 0;
    }
}

// The digest of data[0, len) for sha2_digest_bytes(bits) != 0; `out` holds 64 bytes.
inline void sha2(const std::uint8_t* data, std::uint64_t len, std::int64_t bits, std::uint8_t out[64]) noexcept {
    assert(sha2_digest_bytes(bits) != 0);
    switch (bits) {
        case 224: sha224(data, len, out); break;
        case 384: sha512(data, len, true, out); break;
        case 512: sha512(data, len, false, out); break;
        default: bolt::crypto::sha256(data, len, out); break;
    }
}

// ---- Murmur3_x86_32 (Spark) ---------------------------------------------------------------
BOLT_FORCE_INLINE std::uint32_t m3_mix_k1(std::uint32_t k1) noexcept {
    k1 *= 0xcc9e2d51u;
    k1 = rotl32(k1, 15);
    return k1 * 0x1b873593u;
}
BOLT_FORCE_INLINE std::uint32_t m3_mix_h1(std::uint32_t h1, std::uint32_t k1) noexcept {
    h1 ^= k1;
    h1 = rotl32(h1, 13);
    return h1 * 5u + 0xe6546b64u;
}
BOLT_FORCE_INLINE std::uint32_t m3_fmix(std::uint32_t h1, std::uint32_t len) noexcept {
    h1 ^= len;
    h1 ^= h1 >> 16;
    h1 *= 0x85ebca6bu;
    h1 ^= h1 >> 13;
    h1 *= 0xc2b2ae35u;
    h1 ^= h1 >> 16;
    return h1;
}
inline std::uint32_t m3_hash_int(std::uint32_t v, std::uint32_t seed) noexcept {
    return m3_fmix(m3_mix_h1(seed, m3_mix_k1(v)), 4u);
}
inline std::uint32_t m3_hash_long(std::uint64_t v, std::uint32_t seed) noexcept {
    std::uint32_t h1 = m3_mix_h1(seed, m3_mix_k1(static_cast<std::uint32_t>(v)));
    h1 = m3_mix_h1(h1, m3_mix_k1(static_cast<std::uint32_t>(v >> 32)));
    return m3_fmix(h1, 8u);
}
// Spark's hashUnsafeBytes: whole little-endian words, then every remaining byte as its own round,
// sign-extended (a byte >= 0x80 is a negative int).
inline std::uint32_t m3_hash_bytes(const std::uint8_t* p, std::uint64_t n, std::uint32_t seed) noexcept {
    assert(p != nullptr || n == 0);
    std::uint32_t h1 = seed;
    const std::uint64_t aligned = n - (n % 4u);
    for (std::uint64_t i = 0; i < aligned; i += 4) h1 = m3_mix_h1(h1, m3_mix_k1(load32(p + i)));
    for (std::uint64_t i = aligned; i < n; ++i) {
        h1 = m3_mix_h1(h1, m3_mix_k1(static_cast<std::uint32_t>(static_cast<std::int32_t>(static_cast<std::int8_t>(p[i])))));
    }
    return m3_fmix(h1, static_cast<std::uint32_t>(n));
}

// ---- XXH64 (Spark) ------------------------------------------------------------------------
inline constexpr std::uint64_t k_xp1 = 0x9E3779B185EBCA87ull, k_xp2 = 0xC2B2AE3D27D4EB4Full,
                               k_xp3 = 0x165667B19E3779F9ull, k_xp4 = 0x85EBCA77C2B2AE63ull,
                               k_xp5 = 0x27D4EB2F165667C5ull;
BOLT_FORCE_INLINE std::uint64_t xx_fmix(std::uint64_t h) noexcept {
    h ^= h >> 33;
    h *= k_xp2;
    h ^= h >> 29;
    h *= k_xp3;
    return h ^ (h >> 32);
}
BOLT_FORCE_INLINE std::uint64_t xx_round(std::uint64_t acc, std::uint64_t in) noexcept {
    acc += in * k_xp2;
    return rotl64(acc, 31) * k_xp1;
}
inline std::uint64_t xx_hash_int(std::uint32_t v, std::uint64_t seed) noexcept {
    std::uint64_t h = seed + k_xp5 + 4u;
    h ^= static_cast<std::uint64_t>(v) * k_xp1;
    h = rotl64(h, 23) * k_xp2 + k_xp3;
    return xx_fmix(h);
}
inline std::uint64_t xx_hash_long(std::uint64_t v, std::uint64_t seed) noexcept {
    std::uint64_t h = seed + k_xp5 + 8u;
    h ^= xx_round(0, v);
    h = rotl64(h, 27) * k_xp1 + k_xp4;
    return xx_fmix(h);
}
inline std::uint64_t xx_hash_bytes(const std::uint8_t* p, std::uint64_t n, std::uint64_t seed) noexcept {
    assert(p != nullptr || n == 0);
    const std::uint8_t* const end = p + n;
    std::uint64_t h;
    if (n >= 32) {
        std::uint64_t v1 = seed + k_xp1 + k_xp2, v2 = seed + k_xp2, v3 = seed, v4 = seed - k_xp1;
        const std::uint8_t* const lim = end - 32;
        do {
            v1 = xx_round(v1, load64(p));
            v2 = xx_round(v2, load64(p + 8));
            v3 = xx_round(v3, load64(p + 16));
            v4 = xx_round(v4, load64(p + 24));
            p += 32;
        } while (p <= lim);
        h = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        const std::uint64_t vs[4] = {v1, v2, v3, v4};
        for (unsigned i = 0; i < 4; ++i) h = (h ^ xx_round(0, vs[i])) * k_xp1 + k_xp4;
    } else {
        h = seed + k_xp5;
    }
    h += n;
    while (p + 8 <= end) {
        h ^= xx_round(0, load64(p));
        h = rotl64(h, 27) * k_xp1 + k_xp4;
        p += 8;
    }
    if (p + 4 <= end) {
        h ^= static_cast<std::uint64_t>(load32(p)) * k_xp1;
        h = rotl64(h, 23) * k_xp2 + k_xp3;
        p += 4;
    }
    while (p < end) {
        h ^= static_cast<std::uint64_t>(*p++) * k_xp5;
        h = rotl64(h, 11) * k_xp1;
    }
    return xx_fmix(h);
}

}  // namespace hashx
}  // namespace kernels
}  // namespace bolt
