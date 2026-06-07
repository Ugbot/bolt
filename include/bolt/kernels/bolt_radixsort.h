// bolt_radixsort.h — LSD radix sort of int64 keys (the documented future
// speed upgrade promised by bolt_argsort.h's header). Two entries:
//
//   * radix_sort_i64           — DIRECT value sort (ascending). Moves the
//                                int64 keys themselves into sorted order.
//                                Backs a pure-int64 ORDER BY on a column that
//                                does NOT need to carry a payload (or whose
//                                payload is re-derived), and any internal
//                                "sort these keys" step.
//   * radix_sort_indirect_i64  — INDIRECT permutation sort. Produces a perm
//                                of [0,n) such that keys[perm[i]] is ascending.
//                                Same "sort once, gather many" contract as
//                                bolt_argsort.h's sort_indirect_i64 — this is
//                                the radix fast path for the triple-store
//                                index build (sort row ids by an int64 key,
//                                then gather every column through the perm)
//                                and a single-int64-key ORDER BY.
//
// Why a separate kernel from bolt_argsort's merge sort
// ----------------------------------------------------
//   Merge sort is O(n log n) comparisons (~6–10 ns/row at n=1e6). For a PURE
//   int64 key it is strictly beaten by LSD radix, which is O(n) — a fixed
//   number of linear passes independent of n. This is the explicit fast path
//   for "the key is exactly one int64 column"; the comparison sort stays the
//   correctness-first default for everything else (multi-key, Utf8, custom
//   comparator, floats).
//
// Algorithm — LSD radix, 8-bit digit, 8 passes
// --------------------------------------------
//   int64 is 8 bytes ⇒ 8 passes of an 8-bit (radix-256) digit, least-
//   significant byte first. Each pass is a STABLE counting sort on that byte:
//     1. histogram the 256 buckets for this byte over all n keys,
//     2. exclusive-prefix-sum the histogram into per-bucket write offsets,
//     3. scatter every element to scratch[offset[bucket]++] in input order.
//   Because each pass is stable and we proceed LSB→MSB, the final array is
//   fully sorted (classic LSD invariant). Ping-pong between the live buffer
//   and the scratch buffer; a single final copy lands the result back in the
//   caller's `keys` / `perm` buffer when an odd number of scatter passes ran.
//
//   Constant-byte skip: one combined sweep histograms ALL 8 bytes up front,
//   then any byte-pass whose histogram is a single full bucket (all keys equal
//   on that byte) is SKIPPED — it would be an identity reorder. Full-range
//   data still runs 8 scatters; narrow-range data (small dense int ids — the
//   triple-store index / ORDER-BY-on-keys common case) runs only 2–4. This is
//   the standard LSD optimization and is the reason the indirect form does its
//   histogram over the SEQUENTIAL biased keys, not the random perm gather.
//
//   Digit width tradeoff: 8-bit (256 buckets) keeps the histogram in L1
//   (256 × 8 B = 2 KiB) and needs 8 passes. 11-bit (2048 buckets) would cut
//   to 6 passes but a 16 KiB histogram thrashes L1 and the zeroing cost per
//   pass grows; 16-bit (65536 buckets) is 4 passes but a 512 KiB histogram
//   spills L2 and zeroing dominates at the n we target. 8-bit is the
//   measured sweet spot for n in [1e4, 1e7] single-thread — chosen here.
//
// Signed-key correctness
// ----------------------
//   Two's-complement negatives have bit 63 set, so a naive unsigned radix
//   would sort all positives before all negatives. We flip the sign bit:
//   `u = (uint64)key ^ 0x8000000000000000`. This maps INT64_MIN→0 … −1→
//   0x7FFF…F … 0→0x8000…0 … INT64_MAX→0xFFFF…F, i.e. a monotonic key→unsigned
//   map. We sort the biased unsigned values, then XOR the top bit back when
//   reading the key out. Tested with INT64_MIN / INT64_MAX / mixed negatives.
//
// SIMD
// ----
//   The hot inner work per pass is digit extraction `(u >> shift) & 0xFF`.
//   The COUNTING increment (`++hist[digit]`) and the SCATTER write
//   (`dst[off[digit]++]`) are data-dependent — vectorising them needs
//   conflict detection and risks a scalar≠SIMD divergence, exactly the class
//   of bug that corrupted the predecessor's hashed join. We DO NOT vectorise
//   those. SIMD (AVX2, guarded by BOLT_SIMD_AVX2) is applied ONLY to the
//   bias+digit-extract of a block of 4 int64 lanes feeding the histogram;
//   the extracted digits are then consumed by the IDENTICAL scalar counting
//   loop the no-SIMD build runs. The output is therefore BIT-IDENTICAL across
//   the AVX2 and scalar paths — SIMD is a pure speed path, selected at
//   compile time per Bolt's macros, never a result-changing branch. (Force
//   the scalar path in a translation unit by building without /arch:AVX2 —
//   the test cross-checks both.)
//
// ns/row floor (target)
// ---------------------
//   LSD radix is linear: 8 passes × (1 histogram read + 1 scatter write) per
//   element. The target floor is ~1–2 ns/row at n = 1e6 single core
//   (scatter-write memory bound). This beats the merge-sort floor (~6–10
//   ns/row) by ~4–8×, which is the whole reason this specialisation exists.
//
// Scratch / no-heap (Tiger Style)
// -------------------------------
//   NEVER calls malloc/new. Caller owns ALL scratch:
//     radix_sort_i64(keys, scratch, n)
//        keys    : int64[n]  in/out — sorted in place on return.
//        scratch : int64[n]  ping-pong buffer (contents undefined on return).
//     radix_sort_indirect_i64(perm, keys, perm_scratch, key_scratch, n)
//        perm        : int32[n] out — kernel fills identity then sorts it.
//        keys        : int64[n] in  — READ ONLY, never modified.
//        perm_scratch: int32[n] ping-pong for the perm.
//        key_scratch : int64[n] holds the biased keys reordered alongside the
//                      perm, so each pass reads the byte from a contiguous
//                      buffer instead of a random keys[perm[i]] gather.
//   The 256-entry histogram lives on the stack (2 KiB) — bounded, not heap.
//   key_scratch holds the biased key PER ORIGINAL ROW (filled once up front,
//   indexed by row id, never reordered). Each pass extracts the digit via a
//   key_scratch[perm[i]] gather, ping-ponging only the perm between perm and
//   perm_scratch. This keeps the int64 scratch to a single n-buffer (the
//   documented contract) — perm carries the result, keys stays read-only.
//
// Tiger Style
// -----------
//   * noexcept; no exceptions, no RTTI, no allocation.
//   * BOLT_RESTRICT on every pointer parameter.
//   * ≥2 assertions per function; functions ≤70 lines (passes split into
//     helpers); explicit integer widths; fixed constexpr pass count (8),
//     fixed radix (256) — every loop bound is compile-time bounded.
//   * Deterministic + STABLE (equal keys keep original index order in the
//     indirect form), so the triple index and ORDER BY are byte-reproducible.

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace kernels {

// LSD radix parameters. 8-bit digit ⇒ 256 buckets, 8 passes over int64.
// All loop bounds below derive from these constants (Tiger Style: bounded).
inline constexpr int      kRadixBits    = 8;
inline constexpr int      kRadixBuckets = 1 << kRadixBits;   // 256
inline constexpr uint64_t kRadixMask    = kRadixBuckets - 1; // 0xFF
inline constexpr int      kRadixPasses  = 8;                 // 64 / 8
inline constexpr uint64_t kRadixSignBit = 0x8000000000000000ull;

// ---------------------------------------------------------------------------
// Bias an int64 key into a radix-orderable uint64 (flip the sign bit so
// two's-complement negatives sort before positives). Self-inverse: applying
// it again recovers the original int64 bit pattern.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE uint64_t radix_bias_i64(int64_t key) noexcept {
    return static_cast<uint64_t>(key) ^ kRadixSignBit;
}
BOLT_FORCE_INLINE int64_t radix_unbias_u64(uint64_t u) noexcept {
    return static_cast<int64_t>(u ^ kRadixSignBit);
}

// ---------------------------------------------------------------------------
// Histogram one byte (`shift` = pass*8) of n biased keys into hist[256].
// `hist` MUST be zeroed by the caller. SIMD only extracts the digits of a
// 4-lane block (bit-identical to scalar); the counting increment stays
// scalar (data-dependent bucket — no SIMD divergence possible).
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE void radix_histogram_u64(const uint64_t* BOLT_RESTRICT keys,
                                           int64_t n, int shift,
                                           int64_t* BOLT_RESTRICT hist) noexcept {
    assert(keys != nullptr || n == 0);
    assert(hist != nullptr);
    assert(shift >= 0 && shift < 64);
    int64_t i = 0;
#if BOLT_SIMD_AVX2
    // Extract 4 digits per iteration via AVX2 srl+and, store to a small
    // stack array, then run the SAME scalar count loop the no-AVX2 build
    // runs — identical bucket increments ⇒ identical histogram.
    const __m256i mask = _mm256_set1_epi64x(static_cast<long long>(kRadixMask));
    // Variable (runtime) shift count: _mm256_srl_epi64 takes the count in the
    // low 64 bits of an __m128i and shifts ALL lanes by it. (_mm256_srli_epi64
    // needs a compile-time immediate, which `shift` is not.)
    const __m128i cnt = _mm_cvtsi64_si128(static_cast<long long>(shift));
    for (; i + 4 <= n; i += 4) {
        __m256i v = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(keys + i));
        __m256i d = _mm256_and_si256(_mm256_srl_epi64(v, cnt), mask);
        alignas(32) uint64_t digits[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(digits), d);
        ++hist[digits[0]];
        ++hist[digits[1]];
        ++hist[digits[2]];
        ++hist[digits[3]];
    }
#endif
    for (; i < n; ++i) {
        const uint64_t digit = (keys[i] >> shift) & kRadixMask;
        ++hist[digit];
    }
}

// ---------------------------------------------------------------------------
// Combined histogram of ALL 8 bytes in ONE sequential read of the biased keys.
// hists is kRadixPasses × kRadixBuckets (caller-zeroed). Computing every byte's
// histogram up front lets the sort SKIP any pass whose byte is constant across
// all keys (its histogram has a single non-zero bucket == n) — that pass would
// be an identity reorder. This is the standard LSD "skip constant digit"
// optimization: full-range data still does 8 passes, but narrow-range data
// (small dense int ids — the triple-store / ORDER-BY-on-keys common case)
// drops to 2–4 passes. One sequential sweep, no extra random access.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE void radix_histogram_all(const uint64_t* BOLT_RESTRICT keys,
                                           int64_t n,
                                           int64_t* BOLT_RESTRICT hists) noexcept {
    assert(keys != nullptr || n == 0);
    assert(hists != nullptr);
    for (int64_t i = 0; i < n; ++i) {
        const uint64_t k = keys[i];
        // Unrolled over the fixed 8 bytes (bounded, compile-time).
        for (int p = 0; p < kRadixPasses; ++p) {
            const uint64_t digit = (k >> (p * kRadixBits)) & kRadixMask;
            ++hists[p * kRadixBuckets + digit];
        }
    }
}

// True iff byte-pass `p`'s histogram is constant (one bucket holds all n) —
// that scatter pass is a no-op and can be skipped. Bounded over 256 buckets.
BOLT_FORCE_INLINE bool radix_pass_is_constant(const int64_t* BOLT_RESTRICT hist,
                                              int64_t n) noexcept {
    assert(hist != nullptr && n >= 0);
    for (int b = 0; b < kRadixBuckets; ++b) {
        if (hist[b] == n) return true;   // all keys share this byte value
        if (hist[b] != 0) return false;  // first non-empty, non-full bucket
    }
    return true;  // n == 0
}

// Exclusive prefix sum of hist[256] → write offsets (in place). Bounded loop
// over the fixed 256 buckets. After this, hist[b] is the start index of
// bucket b in the output; scatter post-increments it.
BOLT_FORCE_INLINE void radix_prefix_sum(int64_t* BOLT_RESTRICT hist) noexcept {
    assert(hist != nullptr);
    int64_t sum = 0;
    for (int b = 0; b < kRadixBuckets; ++b) {
        const int64_t c = hist[b];
        hist[b] = sum;
        sum += c;
    }
    assert(sum >= 0);
}

// ---------------------------------------------------------------------------
// One stable counting-sort pass of the DIRECT key sort: read byte `shift`
// from `src`, scatter into `dst` at the offsets in the PRE-COMPUTED prefix-
// summed `hist`. `hist` is the prefix-summed histogram for this byte (mutated
// in place as the scatter post-increments). Stable: input order preserved
// within a bucket because we sweep src front-to-back.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE void radix_scatter_direct(const uint64_t* BOLT_RESTRICT src,
                                            uint64_t* BOLT_RESTRICT dst,
                                            int64_t n, int shift,
                                            int64_t* BOLT_RESTRICT hist) noexcept {
    assert(src != nullptr && dst != nullptr);
    assert(n >= 0 && shift >= 0 && shift < 64);
    for (int64_t i = 0; i < n; ++i) {
        const uint64_t key   = src[i];
        const uint64_t digit = (key >> shift) & kRadixMask;
        dst[hist[digit]++] = key;
    }
}

// ===========================================================================
// Public API — DIRECT value sort
// ===========================================================================

// Ascending LSD radix sort of int64 `keys` in place. `scratch` is a caller-
// owned int64[n] ping-pong buffer. Stable, signed-correct, no allocation.
// Constant byte-passes are skipped; a final copy lands the result back in
// `keys` when an odd number of scatter passes ran.
BOLT_FORCE_INLINE void radix_sort_i64(int64_t* BOLT_RESTRICT keys,
                                      int64_t* BOLT_RESTRICT scratch,
                                      int64_t n) noexcept {
    assert(n >= 0);
    assert((keys != nullptr && scratch != nullptr) || n == 0);
    if (n < 2) return;  // 0 or 1 element is already sorted.
    // Work in biased-unsigned space so two's-complement negatives order
    // correctly. Accessing the int64 storage through a uint64_t* is the
    // STANDARD-PERMITTED signed/unsigned aliasing exception ([basic.lval]) —
    // same 8-byte storage, no copy, no UB.
    uint64_t* a = reinterpret_cast<uint64_t*>(keys);
    uint64_t* b = reinterpret_cast<uint64_t*>(scratch);
    for (int64_t i = 0; i < n; ++i) a[i] = radix_bias_i64(keys[i]);
    // One combined sweep computes all 8 byte-histograms; skip constant bytes.
    int64_t hists[kRadixPasses * kRadixBuckets] = {0};
    radix_histogram_all(a, n, hists);
    for (int pass = 0; pass < kRadixPasses; ++pass) {
        int64_t* hist = hists + pass * kRadixBuckets;
        if (radix_pass_is_constant(hist, n)) continue;  // identity pass
        radix_prefix_sum(hist);
        radix_scatter_direct(a, b, n, pass * kRadixBits, hist);
        uint64_t* tmp = a; a = b; b = tmp;
    }
    // Result is in `a`; if that is the scratch buffer, copy back into keys.
    uint64_t* keys_u = reinterpret_cast<uint64_t*>(keys);
    if (a != keys_u) {
        for (int64_t i = 0; i < n; ++i) keys_u[i] = a[i];
    }
    for (int64_t i = 0; i < n; ++i) keys[i] = radix_unbias_u64(keys_u[i]);
}

// ===========================================================================
// Public API — INDIRECT permutation sort
// ===========================================================================

// One stable counting-sort scatter of the INDIRECT sort. Reads byte `shift` of
// the biased key reached through `perm_src[i]` (bkeys indexed by ORIGINAL row
// id, never reordered), scatters the perm entry to perm_dst at the offset in
// the PRE-COMPUTED prefix-summed `hist`. Stable: src swept front-to-back so
// equal digits keep order. The bkeys[row] read is a data-dependent gather, so
// this stays scalar on every build (no SIMD divergence risk on the gather).
BOLT_FORCE_INLINE void radix_scatter_indirect(const int32_t* BOLT_RESTRICT perm_src,
                                              const uint64_t* BOLT_RESTRICT bkeys,
                                              int32_t* BOLT_RESTRICT perm_dst,
                                              int64_t n, int shift,
                                              int64_t* BOLT_RESTRICT hist) noexcept {
    assert(perm_src != nullptr && perm_dst != nullptr && bkeys != nullptr);
    assert(n >= 0 && shift >= 0 && shift < 64);
    for (int64_t i = 0; i < n; ++i) {
        const int32_t  row   = perm_src[i];
        const uint64_t digit = (bkeys[row] >> shift) & kRadixMask;
        perm_dst[hist[digit]++] = row;
    }
}

// Produce `perm` of [0,n) such that keys[perm[i]] is ascending, via LSD radix
// on the int64 key bytes. STABLE (ties keep original index order), signed-
// correct, no allocation. The kernel fills the identity perm itself (caller
// init NOT required). All scratch is caller-owned:
//   perm         : int32[n] out
//   keys         : int64[n] in (read only, never modified)
//   perm_scratch : int32[n] ping-pong
//   key_scratch  : int64[n] biased-key store, indexed by ORIGINAL row id
//                  (filled once up front, never reordered)
// Constant byte-passes are skipped (the histogram is built in one sequential
// sweep over bkeys, NOT over the perm gather); a final copy lands the result
// back in `perm` when an odd number of scatter passes ran.
BOLT_FORCE_INLINE void radix_sort_indirect_i64(
        int32_t* BOLT_RESTRICT perm, const int64_t* BOLT_RESTRICT keys,
        int32_t* BOLT_RESTRICT perm_scratch,
        int64_t* BOLT_RESTRICT key_scratch, int64_t n) noexcept {
    assert(n >= 0);
    assert((perm != nullptr && keys != nullptr) || n == 0);
    assert((perm_scratch != nullptr && key_scratch != nullptr) || n < 2);
    for (int64_t i = 0; i < n; ++i) perm[i] = static_cast<int32_t>(i);
    if (n < 2) return;  // identity perm is already sorted.
    // Bias every key once, stored by original row id (never reordered) so
    // each pass's digit read is bkeys[perm[i]].
    uint64_t* bkeys = reinterpret_cast<uint64_t*>(key_scratch);
    for (int64_t i = 0; i < n; ++i) bkeys[i] = radix_bias_i64(keys[i]);
    // Combined histogram over the SEQUENTIAL biased keys (cheap, cache-
    // friendly) — used both to skip constant bytes and to prefix-sum offsets.
    int64_t hists[kRadixPasses * kRadixBuckets] = {0};
    radix_histogram_all(bkeys, n, hists);
    int32_t* pa = perm;          // live perm
    int32_t* pb = perm_scratch;  // ping-pong destination
    for (int pass = 0; pass < kRadixPasses; ++pass) {
        int64_t* hist = hists + pass * kRadixBuckets;
        if (radix_pass_is_constant(hist, n)) continue;  // identity pass
        radix_prefix_sum(hist);
        radix_scatter_indirect(pa, bkeys, pb, n, pass * kRadixBits, hist);
        int32_t* tmp = pa; pa = pb; pb = tmp;
    }
    // Result is in `pa`; if that is perm_scratch, copy back into perm.
    if (pa != perm) {
        for (int64_t i = 0; i < n; ++i) perm[i] = pa[i];
    }
}

}  // namespace kernels
}  // namespace bolt
