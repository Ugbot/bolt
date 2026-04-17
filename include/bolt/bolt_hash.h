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

}  // namespace bolt
