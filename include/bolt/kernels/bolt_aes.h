// bolt_aes.h -- AES-128/192/256 (FIPS 197) with ECB and CBC under PKCS#7 padding and GCM with a
// 96-bit IV and 128-bit tag (NIST SP 800-38D), for Spark's aes_encrypt / aes_decrypt. Header-only,
// caller-owned buffers, no heap. Software tables: the S-box is derived at compile time and looked
// up per byte, so this is NOT constant-time with respect to cache side channels; it is a SQL
// function kernel, not a TLS record layer. GCM tag comparison is constant-time.
// RULES: No exceptions. No RTTI. No heap. All fns noexcept. Bounded loops.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_port.h"

namespace bolt {
namespace kernels {
namespace aes {

struct AesTables {
    std::uint8_t sbox[256];
    std::uint8_t inv[256];
    constexpr AesTables() : sbox{}, inv{} {
        std::uint8_t p = 1, q = 1;
        do {
            p = static_cast<std::uint8_t>(p ^ static_cast<std::uint8_t>(p << 1) ^ ((p & 0x80u) != 0 ? 0x1Bu : 0u));
            q = static_cast<std::uint8_t>(q ^ (q << 1));
            q = static_cast<std::uint8_t>(q ^ (q << 2));
            q = static_cast<std::uint8_t>(q ^ (q << 4));
            if ((q & 0x80u) != 0) q = static_cast<std::uint8_t>(q ^ 0x09u);
            const std::uint8_t x = static_cast<std::uint8_t>(
                q ^ static_cast<std::uint8_t>((q << 1) | (q >> 7)) ^ static_cast<std::uint8_t>((q << 2) | (q >> 6)) ^
                static_cast<std::uint8_t>((q << 3) | (q >> 5)) ^ static_cast<std::uint8_t>((q << 4) | (q >> 4)));
            sbox[p] = static_cast<std::uint8_t>(x ^ 0x63u);
        } while (p != 1);
        sbox[0] = 0x63u;
        for (unsigned i = 0; i < 256; ++i) inv[sbox[i]] = static_cast<std::uint8_t>(i);
    }
};
inline constexpr AesTables k_aes_tables{};

inline constexpr std::uint32_t k_block = 16;
inline constexpr std::uint32_t k_gcm_iv = 12;
inline constexpr std::uint32_t k_gcm_tag = 16;

struct AesKey {
    std::uint8_t rk[15 * 16];
    std::uint32_t rounds;
};

BOLT_FORCE_INLINE std::uint8_t xtime(std::uint8_t x) noexcept {
    return static_cast<std::uint8_t>((static_cast<std::uint32_t>(x) << 1) ^ ((x & 0x80u) != 0 ? 0x1Bu : 0u));
}
BOLT_FORCE_INLINE std::uint8_t gmul(std::uint8_t a, std::uint8_t b) noexcept {
    std::uint8_t r = 0;
    for (unsigned i = 0; i < 8; ++i) {
        if ((b & 1u) != 0) r = static_cast<std::uint8_t>(r ^ a);
        a = xtime(a);
        b = static_cast<std::uint8_t>(b >> 1);
    }
    return r;
}

// Expand a 16/24/32-byte key; false for any other length.
inline bool key_init(const std::uint8_t* key, std::uint32_t klen, AesKey* k) noexcept {
    assert(k != nullptr && (key != nullptr || klen == 0));
    if (klen != 16 && klen != 24 && klen != 32) return false;
    const std::uint32_t nk = klen / 4u;
    k->rounds = nk + 6u;
    const std::uint32_t words = 4u * (k->rounds + 1u);
    std::memcpy(k->rk, key, klen);
    std::uint8_t rcon = 1;
    for (std::uint32_t i = nk; i < words; ++i) {
        std::uint8_t t[4];
        std::memcpy(t, k->rk + 4u * (i - 1u), 4);
        if (i % nk == 0) {
            const std::uint8_t t0 = t[0];
            t[0] = static_cast<std::uint8_t>(k_aes_tables.sbox[t[1]] ^ rcon);
            t[1] = k_aes_tables.sbox[t[2]];
            t[2] = k_aes_tables.sbox[t[3]];
            t[3] = k_aes_tables.sbox[t0];
            rcon = xtime(rcon);
        } else if (nk > 6 && i % nk == 4) {
            for (unsigned j = 0; j < 4; ++j) t[j] = k_aes_tables.sbox[t[j]];
        }
        for (unsigned j = 0; j < 4; ++j) k->rk[4u * i + j] = static_cast<std::uint8_t>(k->rk[4u * (i - nk) + j] ^ t[j]);
    }
    return true;
}

inline void add_round_key(std::uint8_t s[16], const std::uint8_t* rk) noexcept {
    for (unsigned i = 0; i < 16; ++i) s[i] = static_cast<std::uint8_t>(s[i] ^ rk[i]);
}

inline void encrypt_block(const AesKey& k, const std::uint8_t in[16], std::uint8_t out[16]) noexcept {
    std::uint8_t s[16];
    std::memcpy(s, in, 16);
    add_round_key(s, k.rk);
    for (std::uint32_t r = 1; r <= k.rounds; ++r) {
        std::uint8_t t[16];
        for (unsigned c = 0; c < 4; ++c) {            // SubBytes + ShiftRows (column-major state)
            for (unsigned row = 0; row < 4; ++row) t[4 * c + row] = k_aes_tables.sbox[s[4 * ((c + row) & 3u) + row]];
        }
        if (r != k.rounds) {
            for (unsigned c = 0; c < 4; ++c) {        // MixColumns
                const std::uint8_t a0 = t[4 * c], a1 = t[4 * c + 1], a2 = t[4 * c + 2], a3 = t[4 * c + 3];
                const std::uint8_t all = static_cast<std::uint8_t>(a0 ^ a1 ^ a2 ^ a3);
                s[4 * c] = static_cast<std::uint8_t>(a0 ^ all ^ xtime(static_cast<std::uint8_t>(a0 ^ a1)));
                s[4 * c + 1] = static_cast<std::uint8_t>(a1 ^ all ^ xtime(static_cast<std::uint8_t>(a1 ^ a2)));
                s[4 * c + 2] = static_cast<std::uint8_t>(a2 ^ all ^ xtime(static_cast<std::uint8_t>(a2 ^ a3)));
                s[4 * c + 3] = static_cast<std::uint8_t>(a3 ^ all ^ xtime(static_cast<std::uint8_t>(a3 ^ a0)));
            }
        } else {
            std::memcpy(s, t, 16);
        }
        add_round_key(s, k.rk + 16u * r);
    }
    std::memcpy(out, s, 16);
}

inline void decrypt_block(const AesKey& k, const std::uint8_t in[16], std::uint8_t out[16]) noexcept {
    std::uint8_t s[16];
    std::memcpy(s, in, 16);
    add_round_key(s, k.rk + 16u * k.rounds);
    for (std::uint32_t r = k.rounds; r >= 1; --r) {
        std::uint8_t t[16];
        for (unsigned c = 0; c < 4; ++c) {            // InvShiftRows + InvSubBytes
            for (unsigned row = 0; row < 4; ++row) t[4 * ((c + row) & 3u) + row] = k_aes_tables.inv[s[4 * c + row]];
        }
        add_round_key(t, k.rk + 16u * (r - 1u));
        if (r != 1) {
            for (unsigned c = 0; c < 4; ++c) {        // InvMixColumns
                const std::uint8_t a0 = t[4 * c], a1 = t[4 * c + 1], a2 = t[4 * c + 2], a3 = t[4 * c + 3];
                s[4 * c] = static_cast<std::uint8_t>(gmul(a0, 14) ^ gmul(a1, 11) ^ gmul(a2, 13) ^ gmul(a3, 9));
                s[4 * c + 1] = static_cast<std::uint8_t>(gmul(a0, 9) ^ gmul(a1, 14) ^ gmul(a2, 11) ^ gmul(a3, 13));
                s[4 * c + 2] = static_cast<std::uint8_t>(gmul(a0, 13) ^ gmul(a1, 9) ^ gmul(a2, 14) ^ gmul(a3, 11));
                s[4 * c + 3] = static_cast<std::uint8_t>(gmul(a0, 11) ^ gmul(a1, 13) ^ gmul(a2, 9) ^ gmul(a3, 14));
            }
        } else {
            std::memcpy(s, t, 16);
        }
    }
    std::memcpy(out, s, 16);
}

enum class AesStatus : std::uint8_t {
    Ok = 0,
    BadLength,     // ciphertext not a multiple of 16 (padded modes)
    BadPadding,    // final block not properly padded
    TooShort,      // GCM input shorter than the tag
    TagMismatch,   // GCM authentication failed
};

// PKCS#7 ciphertext length for n plaintext bytes.
inline std::uint64_t padded_len(std::uint64_t n) noexcept { return (n / 16u + 1u) * 16u; }

// CBC (iv != nullptr) or ECB (iv == nullptr) with PKCS#7: writes padded_len(n) bytes.
inline void encrypt_padded(const AesKey& k, const std::uint8_t* iv, const std::uint8_t* p, std::uint64_t n,
                           std::uint8_t* out) noexcept {
    assert(out != nullptr && (p != nullptr || n == 0));
    std::uint8_t prev[16];
    if (iv != nullptr) std::memcpy(prev, iv, 16);
    const std::uint64_t total = padded_len(n);
    for (std::uint64_t off = 0; off < total; off += 16) {
        std::uint8_t blk[16];
        const std::uint64_t have = off < n ? (n - off < 16 ? n - off : 16) : 0;
        if (have != 0) std::memcpy(blk, p + off, have);
        for (std::uint64_t i = have; i < 16; ++i) blk[i] = static_cast<std::uint8_t>(16u - have);
        if (iv != nullptr) add_round_key(blk, prev);
        encrypt_block(k, blk, out + off);
        if (iv != nullptr) std::memcpy(prev, out + off, 16);
    }
}

// Inverse of encrypt_padded; *out_len is the plaintext length. n may be 0 (empty plaintext).
inline AesStatus decrypt_padded(const AesKey& k, const std::uint8_t* iv, const std::uint8_t* c, std::uint64_t n,
                                std::uint8_t* out, std::uint64_t* out_len) noexcept {
    assert(out != nullptr && out_len != nullptr && (c != nullptr || n == 0));
    *out_len = 0;
    if (n == 0) return AesStatus::Ok;
    if (n % 16u != 0) return AesStatus::BadLength;
    std::uint8_t prev[16];
    if (iv != nullptr) std::memcpy(prev, iv, 16);
    for (std::uint64_t off = 0; off < n; off += 16) {
        decrypt_block(k, c + off, out + off);
        if (iv != nullptr) { add_round_key(out + off, prev); std::memcpy(prev, c + off, 16); }
    }
    const std::uint32_t pad = out[n - 1];
    if (pad < 1 || pad > 16 || pad > n) return AesStatus::BadPadding;
    for (std::uint32_t i = 0; i < pad; ++i) if (out[n - 1 - i] != pad) return AesStatus::BadPadding;
    *out_len = n - pad;
    return AesStatus::Ok;
}

// ---- GCM -------------------------------------------------------------------------------------
struct Gf128 { std::uint64_t hi, lo; };

inline Gf128 gf_load(const std::uint8_t* b) noexcept {
    Gf128 r{0, 0};
    for (unsigned i = 0; i < 8; ++i) { r.hi = (r.hi << 8) | b[i]; r.lo = (r.lo << 8) | b[8 + i]; }
    return r;
}

inline Gf128 gf_mul(Gf128 x, Gf128 h) noexcept {
    Gf128 z{0, 0}, v = h;
    for (unsigned i = 0; i < 128; ++i) {
        const std::uint64_t bit = i < 64 ? (x.hi >> (63 - i)) & 1u : (x.lo >> (127 - i)) & 1u;
        const std::uint64_t mask = ~(bit - 1u);
        z.hi ^= v.hi & mask;
        z.lo ^= v.lo & mask;
        const std::uint64_t carry = v.lo & 1u;
        v.lo = (v.lo >> 1) | (v.hi << 63);
        v.hi = (v.hi >> 1) ^ (0xE100000000000000ull & ~(carry - 1u));
    }
    return z;
}

// GHASH over aad then ct, then the 128-bit length block.
inline Gf128 ghash(Gf128 h, const std::uint8_t* aad, std::uint64_t an, const std::uint8_t* ct, std::uint64_t cn) noexcept {
    Gf128 y{0, 0};
    for (int part = 0; part < 2; ++part) {
        const std::uint8_t* d = part == 0 ? aad : ct;
        const std::uint64_t n = part == 0 ? an : cn;
        for (std::uint64_t off = 0; off < n; off += 16) {
            std::uint8_t blk[16] = {};
            std::memcpy(blk, d + off, n - off < 16 ? n - off : 16);
            const Gf128 b = gf_load(blk);
            y.hi ^= b.hi;
            y.lo ^= b.lo;
            y = gf_mul(y, h);
        }
    }
    y.hi ^= an * 8u;
    y.lo ^= cn * 8u;
    return gf_mul(y, h);
}

inline void gcm_ctr(const AesKey& k, const std::uint8_t j0[16], const std::uint8_t* in, std::uint64_t n, std::uint8_t* out) noexcept {
    std::uint8_t ctr[16];
    std::memcpy(ctr, j0, 16);
    for (std::uint64_t off = 0; off < n; off += 16) {
        for (int i = 15; i >= 12; --i) { if (++ctr[i] != 0) break; }     // inc32
        std::uint8_t ks[16];
        encrypt_block(k, ctr, ks);
        const std::uint64_t m = n - off < 16 ? n - off : 16;
        for (std::uint64_t i = 0; i < m; ++i) out[off + i] = static_cast<std::uint8_t>(in[off + i] ^ ks[i]);
    }
}

inline void gcm_tag(const AesKey& k, const std::uint8_t j0[16], Gf128 h, const std::uint8_t* aad, std::uint64_t an,
                    const std::uint8_t* ct, std::uint64_t cn, std::uint8_t tag[16]) noexcept {
    const Gf128 g = ghash(h, aad, an, ct, cn);
    std::uint8_t ej[16];
    encrypt_block(k, j0, ej);
    for (unsigned i = 0; i < 8; ++i) {
        tag[i] = static_cast<std::uint8_t>(ej[i] ^ (g.hi >> (56 - 8 * i)));
        tag[8 + i] = static_cast<std::uint8_t>(ej[8 + i] ^ (g.lo >> (56 - 8 * i)));
    }
}

inline Gf128 gcm_h(const AesKey& k) noexcept {
    const std::uint8_t zero[16] = {};
    std::uint8_t hb[16];
    encrypt_block(k, zero, hb);
    return gf_load(hb);
}

// out = ciphertext (n bytes) then the 16-byte tag. iv is 12 bytes.
inline void gcm_encrypt(const AesKey& k, const std::uint8_t iv[12], const std::uint8_t* aad, std::uint64_t an,
                        const std::uint8_t* p, std::uint64_t n, std::uint8_t* out) noexcept {
    assert(out != nullptr && (p != nullptr || n == 0) && (aad != nullptr || an == 0));
    std::uint8_t j0[16];
    std::memcpy(j0, iv, 12);
    j0[12] = 0; j0[13] = 0; j0[14] = 0; j0[15] = 1;
    gcm_ctr(k, j0, p, n, out);
    gcm_tag(k, j0, gcm_h(k), aad, an, out, n, out + n);
}

// in = ciphertext then tag (n >= 16 bytes total); out gets n - 16 plaintext bytes.
inline AesStatus gcm_decrypt(const AesKey& k, const std::uint8_t iv[12], const std::uint8_t* aad, std::uint64_t an,
                             const std::uint8_t* in, std::uint64_t n, std::uint8_t* out) noexcept {
    assert(out != nullptr && (in != nullptr || n == 0) && (aad != nullptr || an == 0));
    if (n < 16) return AesStatus::TooShort;
    const std::uint64_t cn = n - 16;
    std::uint8_t j0[16];
    std::memcpy(j0, iv, 12);
    j0[12] = 0; j0[13] = 0; j0[14] = 0; j0[15] = 1;
    std::uint8_t tag[16];
    gcm_tag(k, j0, gcm_h(k), aad, an, in, cn, tag);
    std::uint8_t diff = 0;
    for (unsigned i = 0; i < 16; ++i) diff = static_cast<std::uint8_t>(diff | (tag[i] ^ in[cn + i]));
    if (diff != 0) return AesStatus::TagMismatch;
    gcm_ctr(k, j0, in, cn, out);
    return AesStatus::Ok;
}

}  // namespace aes
}  // namespace kernels
}  // namespace bolt
