// bolt_random_spark.h - the pseudo-random generator behind Spark's rand(),
// randn(), randstr() and uuid(): org.apache.spark.util.random.XORShiftRandom
// (a java.util.Random whose next(bits) is a 64-bit xorshift), seeded as Spark
// seeds it (MurmurHash3 of the seed's eight big-endian bytes), and the
// java.util.Random derivations on top of next(bits) (nextInt(bound),
// nextLong, nextDouble, polar nextGaussian). Header-only; state is a POD the
// caller owns, one per expression per partition: no allocation, no locks.
//
// Spark seeds a partition's generator with (seed + partitionIndex); the caller
// does that addition. Values are reproducible only given the same row order.
//
// RULES: No exceptions. No RTTI. No heap. All fns noexcept. Bounded loops.

#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_port.h"

namespace bolt {
namespace kernels {
namespace random_spark {

namespace detail {

BOLT_FORCE_INLINE uint32_t rotl32(uint32_t x, uint32_t r) noexcept {
    return (x << r) | (x >> (32u - r));
}

// scala.util.hashing.MurmurHash3.mixLast / mix / finalizeHash.
BOLT_FORCE_INLINE uint32_t mix_last(uint32_t h, uint32_t k) noexcept {
    k *= 0xcc9e2d51u;
    k = rotl32(k, 15u);
    k *= 0x1b873593u;
    return h ^ k;
}

BOLT_FORCE_INLINE uint32_t mix(uint32_t h, uint32_t k) noexcept {
    h = mix_last(h, k);
    h = rotl32(h, 13u);
    return h * 5u + 0xe6546b64u;
}

BOLT_FORCE_INLINE uint32_t finalize(uint32_t h, uint32_t len) noexcept {
    h ^= len;
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

// MurmurHash3.bytesHash over the eight big-endian bytes of `v`.
BOLT_FORCE_INLINE uint32_t bytes_hash_long(uint64_t v, uint32_t seed) noexcept {
    // little-endian 4-byte words of the big-endian byte string
    const uint32_t w0 = __builtin_bswap32(static_cast<uint32_t>(v >> 32));
    const uint32_t w1 = __builtin_bswap32(static_cast<uint32_t>(v));
    uint32_t h = seed;
    h = mix(h, w0);
    h = mix(h, w1);
    return finalize(h, 8u);
}

}  // namespace detail

// fdlibm e_log.c (what java.lang.StrictMath.log is): libm's log can differ in the last
// ulp, which would show in randn. x > 0 and finite here (polar method, 0 < s < 1).
BOLT_FORCE_INLINE double fdlibm_log(double x) noexcept {
#if defined(__clang__)
    #pragma clang fp contract(off)
#endif
    constexpr double ln2_hi = 6.93147180369123816490e-01, ln2_lo = 1.90821492927058770002e-10,
                     two54 = 1.80143985094819840000e+16,
                     Lg1 = 6.666666666666735130e-01, Lg2 = 3.999999999940941908e-01,
                     Lg3 = 2.857142874366239149e-01, Lg4 = 2.222219843214978396e-01,
                     Lg5 = 1.818357216161805012e-01, Lg6 = 1.531383769920937332e-01,
                     Lg7 = 1.479819860511658591e-01;
    uint64_t bits;
    std::memcpy(&bits, &x, 8);
    int32_t hx = static_cast<int32_t>(bits >> 32);
    uint32_t lx = static_cast<uint32_t>(bits);
    int32_t k = 0;
    if (hx < 0x00100000) {                       // subnormal
        k -= 54;
        x *= two54;
        std::memcpy(&bits, &x, 8);
        hx = static_cast<int32_t>(bits >> 32);
        lx = static_cast<uint32_t>(bits);
    }
    k += (hx >> 20) - 1023;
    hx &= 0x000fffff;
    const int32_t i = (hx + 0x95f64) & 0x100000;
    bits = (static_cast<uint64_t>(static_cast<uint32_t>(hx | (i ^ 0x3ff00000))) << 32) | lx;
    std::memcpy(&x, &bits, 8);
    k += (i >> 20);
    const double f = x - 1.0;
    if ((0x000fffff & (2 + hx)) < 3) {           // |f| < 2^-20
        if (f == 0.0) {
            if (k == 0) return 0.0;
            const double dk = static_cast<double>(k);
            return dk * ln2_hi + dk * ln2_lo;
        }
        const double R = f * f * (0.5 - 0.33333333333333333 * f);
        if (k == 0) return f - R;
        const double dk = static_cast<double>(k);
        return dk * ln2_hi - ((R - dk * ln2_lo) - f);
    }
    const double s = f / (2.0 + f);
    const double dk = static_cast<double>(k);
    const double z = s * s;
    int32_t ii = hx - 0x6147a;
    const double w = z * z;
    const int32_t j = 0x6b851 - hx;
    const double t1 = w * (Lg2 + w * (Lg4 + w * Lg6));
    const double t2 = z * (Lg1 + w * (Lg3 + w * (Lg5 + w * Lg7)));
    ii |= j;
    const double R = t2 + t1;
    if (ii > 0) {
        const double hfsq = 0.5 * f * f;
        if (k == 0) return f - (hfsq - s * (hfsq + R));
        return dk * ln2_hi - ((hfsq - (s * (hfsq + R) + dk * ln2_lo)) - f);
    }
    if (k == 0) return f - s * (f - R);
    return dk * ln2_hi - ((s * (f - R) - dk * ln2_lo) - f);
}

constexpr uint32_t k_murmur_array_seed = 0x3c074a61u;
constexpr uint32_t k_randstr_alphabet = 62;

struct XorShiftRandom {
    uint64_t seed;
    double   next_next_gaussian;
    uint8_t  have_next_gaussian;
};

// XORShiftRandom.hashSeed: the 64-bit state Spark derives from `init`.
BOLT_FORCE_INLINE uint64_t hash_seed(int64_t init) noexcept {
    const uint64_t u = static_cast<uint64_t>(init);
    const uint32_t low = detail::bytes_hash_long(u, k_murmur_array_seed);
    const uint32_t high = detail::bytes_hash_long(u, low);
    return (static_cast<uint64_t>(high) << 32) | static_cast<uint64_t>(low);
}

BOLT_FORCE_INLINE void init(XorShiftRandom* r, int64_t init_seed) noexcept {
    assert(r != nullptr);
    r->seed = hash_seed(init_seed);
    r->next_next_gaussian = 0.0;
    r->have_next_gaussian = 0;
}

// next(bits), 1 <= bits <= 32: the low `bits` of the next xorshift state, as a
// signed 32-bit int (bits == 32 may be negative, as in Java).
BOLT_FORCE_INLINE int32_t next_bits(XorShiftRandom* r, uint32_t bits) noexcept {
    assert(r != nullptr && bits >= 1 && bits <= 32);
    uint64_t s = r->seed;
    s ^= s << 21;
    s ^= s >> 35;
    s ^= s << 4;
    r->seed = s;
    return static_cast<int32_t>(static_cast<uint32_t>(s & ((uint64_t{1} << bits) - 1u)));
}

BOLT_FORCE_INLINE int64_t next_long(XorShiftRandom* r) noexcept {
    const int64_t hi = static_cast<int64_t>(next_bits(r, 32));
    const int64_t lo = static_cast<int64_t>(next_bits(r, 32));
    return static_cast<int64_t>(static_cast<uint64_t>(hi) * (uint64_t{1} << 32) + static_cast<uint64_t>(lo));
}

BOLT_FORCE_INLINE double next_double(XorShiftRandom* r) noexcept {
#if defined(__clang__)
    #pragma clang fp contract(off)
#endif
    const int64_t a = static_cast<int64_t>(next_bits(r, 26));
    const int64_t b = static_cast<int64_t>(next_bits(r, 27));
    return static_cast<double>((a << 27) + b) * 0x1.0p-53;
}

// java.util.Random.nextInt(bound), bound > 0. The rejection loop is bounded:
// each retry succeeds with probability >= 1/2.
BOLT_FORCE_INLINE int32_t next_int_bound(XorShiftRandom* r, int32_t bound) noexcept {
    assert(r != nullptr && bound > 0);
    int32_t v = next_bits(r, 31);
    const int32_t m = bound - 1;
    if ((bound & m) == 0) {
        return static_cast<int32_t>((static_cast<int64_t>(bound) * static_cast<int64_t>(v)) >> 31);
    }
    for (uint32_t guard = 0; guard < 1024u; ++guard) {
        const int32_t u = v;
        v = u % bound;
        if (static_cast<int32_t>(static_cast<uint32_t>(u) - static_cast<uint32_t>(v) +
                                 static_cast<uint32_t>(m)) >= 0) {
            return v;
        }
        v = next_bits(r, 31);
    }
    return v % bound;
}

// java.util.Random.nextGaussian (the polar method; the second value of each
// pair is kept in the state).
BOLT_FORCE_INLINE double next_gaussian(XorShiftRandom* r) noexcept {
#if defined(__clang__)
    #pragma clang fp contract(off)
#endif
    assert(r != nullptr);
    if (r->have_next_gaussian != 0) {
        r->have_next_gaussian = 0;
        return r->next_next_gaussian;
    }
    double v1 = 0.0, v2 = 0.0, s = 0.0;
    for (uint32_t guard = 0; guard < 4096u; ++guard) {
        v1 = 2.0 * next_double(r) - 1.0;
        v2 = 2.0 * next_double(r) - 1.0;
        s = v1 * v1 + v2 * v2;
        if (s < 1.0 && s != 0.0) break;
    }
    const double mult = std::sqrt(-2.0 * fdlibm_log(s) / s);
    r->next_next_gaussian = v2 * mult;
    r->have_next_gaussian = 1;
    return v1 * mult;
}

// Spark randstr: `len` characters, each from [0-9a-zA-Z] by abs(nextInt()) % 62.
BOLT_FORCE_INLINE void fill_randstr(XorShiftRandom* r, char* out, uint32_t len) noexcept {
    assert(r != nullptr && (out != nullptr || len == 0));
    for (uint32_t i = 0; i < len; ++i) {
        int64_t v = static_cast<int64_t>(next_bits(r, 32));
        if (v < 0) v = -v;
        const int32_t n = static_cast<int32_t>(v % k_randstr_alphabet);
        out[i] = n < 10 ? static_cast<char>('0' + n)
                        : (n < 36 ? static_cast<char>('a' + (n - 10)) : static_cast<char>('A' + (n - 36)));
    }
}

// An RFC 4122 version-4 UUID text (36 bytes, lowercase, no terminator) from
// two next_long draws, version and variant bits forced as java.util.UUID's
// randomUUID does.
BOLT_FORCE_INLINE void format_uuid_v4(XorShiftRandom* r, char out[36]) noexcept {
    assert(r != nullptr && out != nullptr);
    uint64_t hi = static_cast<uint64_t>(next_long(r));
    uint64_t lo = static_cast<uint64_t>(next_long(r));
    hi = (hi & 0xFFFFFFFFFFFF0FFFull) | 0x0000000000004000ull;
    lo = (lo | 0x8000000000000000ull) & 0xBFFFFFFFFFFFFFFFull;
    static const char hex[] = "0123456789abcdef";
    uint32_t p = 0;
    for (int shift = 60; shift >= 0; shift -= 4) {
        if (p == 8 || p == 13 || p == 18 || p == 23) out[p++] = '-';
        out[p++] = hex[(hi >> shift) & 0xF];
    }
    for (int shift = 60; shift >= 0; shift -= 4) {
        if (p == 8 || p == 13 || p == 18 || p == 23) out[p++] = '-';
        out[p++] = hex[(lo >> shift) & 0xF];
    }
    assert(p == 36);
}

}  // namespace random_spark
}  // namespace kernels
}  // namespace bolt
