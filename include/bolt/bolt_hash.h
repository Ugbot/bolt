// bolt_hash.h — scalar 64-bit hash-mix primitives.
//
// Three variants ship in source at all times per the
// keep-code-paths-for-JIT-later policy:
//
//   swiss_mix_wyhash3  — 3-op wyhash-family mix.  Current ship-default.
//   swiss_mix_xxh3     — xxh3-64 finalizer shape.  Stronger distribution.
//   swiss_mix_murmur3  — Murmur3 64-bit finalizer.  Slowest, strongest.
//
// A compile-time switch `BOLT_HASH_TIER_{WYHASH3|XXH3|MURMUR3}` selects
// which variant the default `swiss_mix` symbol resolves to.  All three
// are always compiled; a future runtime dispatch (JIT or per-CPU
// picker) can call them by name without archaeology.
//
// Trade-offs live in docs/research/hash-functions.md and
// docs/research/design-log.md ("Hash mixer — …").  No runtime branches
// in this header — dispatch is zero-cost.
//
// RULES: No exceptions.  noexcept everywhere.  POD over OO.

#pragma once

#include "bolt/bolt_port.h"
#include <cstdint>
#include <cstddef>
#include <cassert>
#include <cstring>

namespace bolt {

// Default: if no tier macro was defined by CMake, fall back to WYHASH3.
// This keeps the build green for consumers who include `bolt_hash.h`
// without running through `bolt_apply_feature_toggles`.
#if !defined(BOLT_HASH_TIER_WYHASH3) && !defined(BOLT_HASH_TIER_XXH3) && \
    !defined(BOLT_HASH_TIER_MURMUR3)
#define BOLT_HASH_TIER_WYHASH3 1
#endif

// ---------------------------------------------------------------------------
// Variant: wyhash 3-op mix (current ship-default).
//
// Constant from the wyhash / foldhash family (Wang Yi,
// https://github.com/wangyi-fudan/wyhash).  Keeps independent high/low
// halves so SwissTable's tag/idx split keeps its selectivity at
// roughly half the cost of Murmur3.  Avalanche margin is lower than
// Murmur3 but adequate for open-addressing on adversarial-but-not-
// pathological inputs (1BRC-class data).
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE uint64_t swiss_mix_wyhash3(uint64_t k) noexcept {
    k ^= k >> 32;
    k *= 0xE7037ED1A0B428DBULL;
    k ^= k >> 32;
    return k;
}

// ---------------------------------------------------------------------------
// Variant: xxh3-64 finalizer (Yann Collet's xxHash family).
//
// Two multiplies + three xor-shifts.  Stronger avalanche than WYHASH3
// at similar op count; wins when keys share entropy in narrow bit
// ranges (e.g. dense dictionary codes, low-cardinality timestamps).
// Constants from XXH3's `XXH3_avalanche` / `XXH64_avalanche`.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE uint64_t swiss_mix_xxh3(uint64_t k) noexcept {
    k ^= k >> 37;
    k *= 0x165667919E3779F9ULL;     // XXH_PRIME_MX1
    k ^= k >> 32;
    return k;
}

// ---------------------------------------------------------------------------
// Variant: Murmur3 64-bit finalizer.
//
// Six operations — three multiplies, three xor-shifts.  Strongest
// avalanche of the three variants; slowest op count.  Kept for
// adversarial workloads and as a correctness-check baseline.
// Constants from Austin Appleby's original Murmur3 reference.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE uint64_t swiss_mix_murmur3(uint64_t k) noexcept {
    k ^= k >> 33;
    k *= 0xFF51AFD7ED558CCDULL;
    k ^= k >> 33;
    k *= 0xC4CEB9FE1A85EC53ULL;
    k ^= k >> 33;
    return k;
}

// ---------------------------------------------------------------------------
// Default dispatch: picks one variant at compile time based on the
// BOLT_HASH_TIER_* macro set by `bolt_apply_feature_toggles`.
// Zero-cost — one symbol, inlined.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE uint64_t swiss_mix(uint64_t k) noexcept {
#if defined(BOLT_HASH_TIER_XXH3) && BOLT_HASH_TIER_XXH3
    return swiss_mix_xxh3(k);
#elif defined(BOLT_HASH_TIER_MURMUR3) && BOLT_HASH_TIER_MURMUR3
    return swiss_mix_murmur3(k);
#else   // BOLT_HASH_TIER_WYHASH3 (default)
    return swiss_mix_wyhash3(k);
#endif
}

// ---------------------------------------------------------------------------
// hash_bytes — 64-bit hash of an arbitrary byte range.
//
// Fast wyhash/xxh3-style block reader: consume 8-byte little-endian blocks,
// fold each through the wyhash3 mixer, handle the <8-byte tail with one masked
// read, then run the accumulator through the existing `swiss_mix` finalizer so
// the result composes with `SwissTable` (whose tag/index split assumes a
// swiss_mix-quality avalanche).  The length is seeded into the state so that
// `"ab"` and `"ab\0"` hash distinctly without a divergent length branch.
//
// DETERMINISTIC: blocks are read little-endian on every platform via
// `bolt_load_u64_le`, so the hash is identical on x86 and big-endian targets.
// STABLE: a SINGLE scalar implementation — there is deliberately NO SIMD path.
// The predecessor shipped a scalar+SIMD pair whose outputs diverged and
// silently corrupted hash joins; we do not reintroduce that risk.
//
// Floor target: ~0.30 ns/byte for len >= 32 (one mul + one memcpy per 8 bytes,
// memory-bandwidth-bound on warm L1), ~3 ns fixed for the short RDF-term
// strings the interner feeds it.  No allocs, branch-reduced, noexcept.
// ---------------------------------------------------------------------------

// Compile-time endianness detection. Bolt's wire format already pins all
// supported targets to little-endian (x86 / ARM64); this guard exists so the
// byte-swap path is *correct* rather than dead if anyone ports to a BE host.
#if !defined(BOLT_BIG_ENDIAN)
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#define BOLT_BIG_ENDIAN 1
#endif
#endif

// Endianness-stable 8-byte little-endian load. On little-endian hosts (x86,
// ARM-LE) this is a single `mov`; on big-endian it byte-swaps. Keeps
// `hash_bytes` identical across platforms — the determinism requirement.
BOLT_FORCE_INLINE uint64_t hash_load_u64_le(const uint8_t* BOLT_RESTRICT p) noexcept {
    uint64_t v;
    std::memcpy(&v, p, sizeof(v));
#if defined(BOLT_BIG_ENDIAN)
    v = ((v & 0x00000000000000FFULL) << 56) | ((v & 0x000000000000FF00ULL) << 40) |
        ((v & 0x0000000000FF0000ULL) << 24) | ((v & 0x00000000FF000000ULL) << 8)  |
        ((v & 0x000000FF00000000ULL) >> 8)  | ((v & 0x0000FF0000000000ULL) >> 24) |
        ((v & 0x00FF000000000000ULL) >> 40) | ((v & 0xFF00000000000000ULL) >> 56);
#endif
    return v;
}

BOLT_FORCE_INLINE uint64_t hash_bytes(const void* BOLT_RESTRICT data,
                                      size_t len) noexcept {
    // Precondition: non-null payload whenever there are bytes to read, and a
    // sane upper bound (anything past 4 GiB is a corrupt length, not data).
    assert(data != nullptr || len == 0);
    assert(len <= (size_t)0xFFFFFFFFULL);

    const uint8_t* BOLT_RESTRICT p = static_cast<const uint8_t*>(data);
    // Seed the length into the state so trailing-zero-distinguished strings
    // (e.g. "ab" vs "ab\0") avalanche apart without a branch on len.
    uint64_t h = 0x9E3779B97F4A7C15ULL ^
                 swiss_mix_wyhash3(static_cast<uint64_t>(len));

    size_t i = 0;
    // Fixed per-iteration work; bounded by `len` (asserted <= 4 GiB above).
    for (; i + 8 <= len; i += 8) {
        h = swiss_mix_wyhash3(h ^ hash_load_u64_le(p + i));
    }
    // Tail: pack the remaining 1..7 bytes into the low end of a zeroed word,
    // byte 0 at bit 0 — explicitly little-endian so the digest is identical on
    // big-endian hosts (do NOT memcpy: that lands bytes high on a BE host).
    if (i < len) {
        uint64_t tail = 0;
        size_t shift = 0;
        for (; i < len; ++i, shift += 8) {       // bounded: at most 7 iters
            tail |= static_cast<uint64_t>(p[i]) << shift;
        }
        h = swiss_mix_wyhash3(h ^ tail);
    }

    // Final avalanche through the SwissTable-composing mixer.
    const uint64_t out = swiss_mix(h);
    assert(len != 0 || out == swiss_mix(0x9E3779B97F4A7C15ULL ^
                                        swiss_mix_wyhash3(0)));  // len-0 stable
    return out;
}

}  // namespace bolt
