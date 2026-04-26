// bolt_ribbon.cpp — Banded GF(2) Ribbon filter (Dillinger 2021).
//
// Body for the API declared in bolt/bolt_ribbon.h.
//
// Parameters (frozen in the header):
//   r       = 32   (band width — pivot row is one u32)
//   fp_bits = 8    (byte-aligned fingerprint; FPR ≈ 1/256 ≈ 0.39%)
//   m       = ceil(1.05 * n) + r
//   storage = m bytes ≈ 8.4 bits/key
//
// Tiger Style:
//   - All functions noexcept; ≥2 asserts per hot fn; ≤70 lines/fn.
//   - No exceptions / RTTI / smart pointers / std::vector.
//   - No heap on the hot path: scratch comes from the caller.
//   - Branchless query loop where possible.
//
// Algorithm (build):
//   For each key seeded by `seed`:
//     h     = mix(key ^ seed)
//     start = uniform_in([0, m - r))
//     coeff = (mix2(h) | 1) & ((1<<r) - 1)  — r-bit row, bit 0 = 1
//     fp    = top byte of h                 — fingerprint
//   Insert into banded reduced row-echelon matrix (per-slot pivot row +
//   RHS fingerprint).  Step:
//     pos  = ctz(c)
//     slot = base + pos
//     if pivot[slot] == 0: store (c >> pos) at pivot[slot], rhs[slot]=f, done
//     else:                c ^= (pivot[slot] << pos), f ^= rhs[slot], loop
//   Failures: c=0 with f≠0 → dependency conflict (retry with new seed).
//
//   Bit-truncation: the in-flight coefficient `c` lives in a u64 so
//   `pivot << pos` (pivot u32, pos ≤ 31) doesn't lose its high bits.
//   By induction the lowest set bit of c strictly increases each step,
//   and c's set bits remain in a 32-bit window above pos, so c >> pos
//   always fits in 32 bits when we land.
//
// Algorithm (query):
//   recompute (start, coeff, fp); xor payload[start + bit_idx] for
//   every set bit in coeff; compare to fp.  Match ⇒ might_contain.

#include "bolt/bolt_ribbon.h"

#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>
#include <cstring>

namespace bolt {
namespace ribbon {

namespace {

// -----------------------------------------------------------------------
// Hashing helpers — splitmix64-style finalizer composed with the seed.
// -----------------------------------------------------------------------

BOLT_FORCE_INLINE uint64_t mix64(uint64_t x) noexcept {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    x ^= x >> 33;
    return x;
}

BOLT_FORCE_INLINE uint64_t rotl64(uint64_t x, uint32_t k) noexcept {
    assert(k < 64u);
    return (x << k) | (x >> (64u - k));
}

struct KeyHash {
    uint32_t start;
    uint32_t coeff;   // r=32-bit row, bit 0 always set
    uint8_t  fp;
};

BOLT_FORCE_INLINE KeyHash hash_key(uint64_t key, uint64_t seed,
                                   uint32_t m) noexcept {
    assert(m > kBandWidth);
    const uint64_t h1 = mix64(key ^ seed);
    const uint64_t h2 = mix64(rotl64(h1, 32) ^ 0x9E3779B97F4A7C15ull);
    KeyHash kh;
    const uint32_t span = m - kBandWidth;
    kh.start = static_cast<uint32_t>((h1 >> 32) % span);
    // Force bit 0 = 1 so the band's first row is always non-zero.
    kh.coeff = static_cast<uint32_t>(h2) | 1u;
    kh.fp    = static_cast<uint8_t>(h1 >> 56);  // top byte of h1
    return kh;
}

// -----------------------------------------------------------------------
// Header read/write helpers (16-byte little-endian frame).
// -----------------------------------------------------------------------

BOLT_FORCE_INLINE void write_header(uint8_t* buf, uint32_t m,
                                    uint16_t seed_used) noexcept {
    assert(buf != nullptr);
    assert(m > 0u);
    std::memcpy(buf, &kMagic, sizeof(kMagic));
    std::memcpy(buf + 8, &m, sizeof(m));
    buf[12] = static_cast<uint8_t>(kBandWidth);
    buf[13] = static_cast<uint8_t>(kFpBits);
    std::memcpy(buf + 14, &seed_used, sizeof(seed_used));
}

struct ParsedHeader {
    uint32_t m;
    uint16_t seed_used;
    bool     ok;
};

BOLT_FORCE_INLINE ParsedHeader parse_header(const uint8_t* buf,
                                            size_t len) noexcept {
    ParsedHeader p{};
    if (buf == nullptr || len < kHeaderBytes) return p;
    uint64_t magic = 0;
    std::memcpy(&magic, buf, sizeof(magic));
    if (magic != kMagic) return p;
    std::memcpy(&p.m, buf + 8, sizeof(p.m));
    if (buf[12] != static_cast<uint8_t>(kBandWidth)) return p;
    if (buf[13] != static_cast<uint8_t>(kFpBits))    return p;
    std::memcpy(&p.seed_used, buf + 14, sizeof(p.seed_used));
    if (p.m <= kBandWidth)                           return p;
    if (kHeaderBytes + p.m > len)                    return p;
    p.ok = true;
    return p;
}

// -----------------------------------------------------------------------
// Banded insertion (Dillinger 2021 §3.2 / RocksDB `ribbon_alg.h` style).
// Re-anchor `c` to the current scan position after every step so it
// always fits in a u32 (kBandWidth=32 bits).
// -----------------------------------------------------------------------
bool insert_key(uint32_t* BOLT_RESTRICT coef,
                uint8_t*  BOLT_RESTRICT rhs,
                uint32_t m, KeyHash kh) noexcept {
    assert(coef != nullptr);
    assert(rhs  != nullptr);
    uint32_t s = kh.start;        // current anchor (slot)
    uint32_t c = kh.coeff;        // anchored at s; bit 0 = x[s]
    uint8_t  f = kh.fp;
    while (c != 0u) {
        // Shift forward to the next set bit so bit 0 corresponds to s.
        const uint32_t shift = static_cast<uint32_t>(bolt_ctz32(c));
        s += shift;
        c >>= shift;
        // Now bit 0 of c is set (we'll exit through this loop body).
        if (s >= m) return false;     // band overflow
        const uint32_t pivot = coef[s];
        if (pivot == 0u) {
            // Highest set bit of c determines how far the constraint
            // links forward — reject if it would index past slot m-1.
            const uint32_t hi_bit_pos = 31u
                - static_cast<uint32_t>(bolt_clz32(c));
            if (s + hi_bit_pos >= m) return false;
            coef[s] = c;
            rhs[s]  = f;
            return true;
        }
        // Both c and pivot are anchored at s.  XOR clears bit 0 of c
        // (both sides had a 1 there) and merges higher bits.  Then we
        // need to advance past bit 0 — that lives at the top of the
        // loop on the next ctz.
        c ^= pivot;
        f ^= rhs[s];
        // Strip the cleared bit 0 explicitly so the next ctz advances
        // s by at least one (loop terminates).
        c >>= 1;
        s += 1;
    }
    // c == 0 → row redundant (success) or conflict (fail).
    return f == 0u;
}

// -----------------------------------------------------------------------
// Back-substitute m-1..0 to materialise x[s].  For each slot s with
// pivot row p (bit 0 = x[s], bit i = x[s+i]), x[s] = rhs[s] XOR
// XOR_{i>0, p has bit i set} x[s+i].
// -----------------------------------------------------------------------
void back_substitute(const uint32_t* BOLT_RESTRICT coef,
                     const uint8_t*  BOLT_RESTRICT rhs,
                     uint32_t m,
                     uint8_t* BOLT_RESTRICT out) noexcept {
    assert(coef != nullptr);
    assert(rhs  != nullptr);
    assert(out  != nullptr);
    assert(m > 0u);
    for (uint32_t s = m; s-- > 0u; ) {
        const uint32_t pivot = coef[s];
        if (pivot == 0u) {
            out[s] = 0u;
            continue;
        }
        // Strip bit 0 (always 1; corresponds to x[s] itself).
        uint32_t mask = pivot >> 1;
        uint8_t f = rhs[s];
        uint32_t off = 1u;
        while (mask != 0u) {
            const uint32_t pos = static_cast<uint32_t>(bolt_ctz32(mask));
            const uint32_t target = s + off + pos;
            // Pivot row spans r slots starting at s, so target ≤ s+r-1.
            // hash_key + insert_key invariants make this < m.
            assert(target < m);
            f ^= out[target];
            const uint32_t drop = pos + 1u;
            mask >>= drop;
            off  += drop;
        }
        out[s] = f;
    }
}

// -----------------------------------------------------------------------
// One build attempt.  Returns false on dependency conflict.
// -----------------------------------------------------------------------
bool build_attempt(const uint64_t* keys, int64_t n,
                   uint64_t seed, uint32_t m,
                   uint32_t* coef, uint8_t* rhs,
                   uint8_t* out_payload) noexcept {
    assert(keys != nullptr || n == 0);
    assert(coef != nullptr);
    assert(rhs  != nullptr);
    assert(out_payload != nullptr);
    assert(m > kBandWidth);

    std::memset(coef, 0, static_cast<size_t>(m) * sizeof(uint32_t));
    std::memset(rhs,  0, static_cast<size_t>(m));

    for (int64_t i = 0; i < n; ++i) {
        const KeyHash kh = hash_key(keys[i], seed, m);
        if (!insert_key(coef, rhs, m, kh)) return false;
    }
    back_substitute(coef, rhs, m, out_payload);
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API — build.
// ---------------------------------------------------------------------------
size_t build(const uint64_t* keys, int64_t n,
             double target_fpr,
             void* scratch, size_t scratch_bytes,
             void* out_buf, size_t out_buf_bytes,
             uint64_t seed) noexcept {
    assert(out_buf != nullptr || out_buf_bytes == 0);
    assert(scratch != nullptr || scratch_bytes == 0);
    (void)target_fpr;  // advisory — fixed FPR per fp_bits in this build

    if (n < 0) return 0u;
    if (out_buf == nullptr) return 0u;

    const size_t need_out = size_bytes_for(n, target_fpr);
    if (out_buf_bytes < need_out) return 0u;

    auto* out_bytes = static_cast<uint8_t*>(out_buf);
    if (n == 0) {
        write_header(out_bytes, /*m=*/kBandWidth + 1u,
                     static_cast<uint16_t>(seed & 0xFFFFu));
        return kHeaderBytes;
    }

    if (keys == nullptr) return 0u;

    const size_t need_scratch = scratch_bytes_for(n);
    if (scratch == nullptr || scratch_bytes < need_scratch) return 0u;

    const uint32_t m = static_cast<uint32_t>(
        (static_cast<uint64_t>(n) * kOverProvisionN
            + kOverProvisionD - 1u) / kOverProvisionD
        + kBandWidth);
    assert(m > kBandWidth);
    assert(kHeaderBytes + m == need_out);

    // Carve scratch: m × u32 coefs (4-byte aligned), then m × u8 rhs.
    auto* scratch_p = static_cast<uint8_t*>(scratch);
    const uintptr_t addr = reinterpret_cast<uintptr_t>(scratch_p);
    const uintptr_t adj  = (4u - (addr & 3u)) & 3u;
    scratch_p += adj;
    auto* coef = reinterpret_cast<uint32_t*>(scratch_p);
    auto* rhs  = scratch_p + static_cast<size_t>(m) * sizeof(uint32_t);

    uint8_t* out_payload = out_bytes + kHeaderBytes;
    if (!build_attempt(keys, n, seed, m, coef, rhs, out_payload)) {
        return 0u;  // caller retries with a fresh seed
    }

    write_header(out_bytes, m, static_cast<uint16_t>(seed & 0xFFFFu));
    return need_out;
}

// ---------------------------------------------------------------------------
// Public API — might_contain.
// Branchless XOR-reduce of the m bytes selected by `coeff`, byte-compare
// against `fp`.
// ---------------------------------------------------------------------------
bool might_contain(const void* filter_bytes, size_t len,
                   uint64_t key) noexcept {
    assert(filter_bytes != nullptr || len == 0);
    if (filter_bytes == nullptr) return true;
    const auto* buf = static_cast<const uint8_t*>(filter_bytes);
    const ParsedHeader hdr = parse_header(buf, len);
    if (!hdr.ok) {
        // Conservative: treat malformed filter as "may be present".
        return true;
    }
    const uint64_t seed = static_cast<uint64_t>(hdr.seed_used);
    const KeyHash kh = hash_key(key, seed, hdr.m);
    const uint8_t* payload = buf + kHeaderBytes;

    // Branchless: walk all r=32 bits of coeff regardless of value.
    uint8_t acc = 0;
    const uint32_t start = kh.start;
    const uint32_t c     = kh.coeff;
    for (uint32_t i = 0; i < kBandWidth; ++i) {
        const uint32_t bit = (c >> i) & 1u;
        const uint8_t mask = static_cast<uint8_t>(0u - bit);
        acc ^= static_cast<uint8_t>(payload[start + i] & mask);
    }
    return acc == kh.fp;
}

}  // namespace ribbon
}  // namespace bolt
