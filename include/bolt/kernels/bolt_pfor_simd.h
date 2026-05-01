// bolt/kernels/bolt_pfor_simd.h — AVX2 / SSE4.2 SIMD overlay for the
// scalar PFOR-Delta bit-pack/unpack kernels in `bolt_pfor.h`.
//
// Layer 2.1b of plan `this-was-a-freach-hashed-crab.md`. The scalar baseline
// (commit 49738028) hits ~0.5 µs/block on MSVC /O2; this overlay closes the
// remaining gap to Lemire-Boytsov SIMD-BP128 (≥ 1 G int/s decode).
//
// Reference: Lemire & Boytsov 2015 (arXiv:1209.2137) §4–5; Lucene104
// `PForUtil` / `ForUtil` `VectorAPI` unpack as the JVM-equivalent.
//
// Public surface contract:
//   - `unpack_*` reads `n_values` packed `bpv`-bit values from `in` and writes
//     `n_values` uint32 to `out`. Bit-exact match with the scalar path is the
//     correctness gate (see `tests/test_bolt_pfor_simd.cpp`).
//   - `pack_*`   reads `n_values` uint32 from `in`, masks low `bpv` bits, and
//     ORs them into `out`. `out` MUST be pre-zeroed for the bytes touched.
//   - All functions return `bool`: `true` if the SIMD path executed,
//     `false` to signal the caller MUST fall through to scalar (unsupported
//     `bpv` or `n_values` shape). No partial writes on `false` return.
//
// SIMD coverage table (n_values == 128):
//   bpv  | AVX2 unpack | AVX2 pack | SSE4.2 unpack | SSE4.2 pack |
//   -----+-------------+-----------+----------------+-------------+
//   1    | yes         | yes       | yes            | yes         |
//   2    | yes         | yes       | yes            | yes         |
//   4    | yes         | yes       | yes            | yes         |
//   8    | yes         | yes       | yes            | yes         |
//   16   | yes         | yes       | yes            | yes         |
//   other| scalar      | scalar    | scalar         | scalar      |
//
// bpv ∈ {0, 32} are handled by the scalar fast paths in `bolt_pfor.h` and
// never reach the SIMD overlay (memcpy / memset). bpv ∈ {3, 5, 6, 7,
// 9–15, 17–31} fall through to the branchless scalar inner loop — Lemire's
// full SIMD-BP128 dedicates a separate routine to each, but the win is
// small at our typical posting-list bpv distribution.
//
// Tiger Style:
//   - All public functions `noexcept`.
//   - No std:: containers, no smart pointers, no allocations.
//   - BOLT_FORCE_INLINE on hot path; BOLT_RESTRICT on pointer params.
//   - ≥2 assertions per public function; ≤70 lines per function.
//   - SIMD impl selected at compile time via BOLT_SIMD_AVX2 / BOLT_SIMD_SSE42
//     gates from `bolt_port.h`; no runtime branches in the inner loop.
//
// Research: docs/research/pfor-simd-overlay.md.

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>
#include <cstring>

#if BOLT_SIMD_AVX2 || BOLT_SIMD_SSE42
#include <immintrin.h>
#endif

namespace bolt {
namespace kernels {
namespace pfor {
namespace simd {

inline constexpr int32_t kSimdBlockSize = 128;

// ---------------------------------------------------------------------------
// AVX2 path — 256-bit lanes, processes 32 uint32 at a time on the unpack
// side (4 stores of 8 uint32 each per 128-value block).
// ---------------------------------------------------------------------------
#if BOLT_SIMD_AVX2

// Helper: zero-extend 8 packed uint8 (low 64 bits of `bytes`) to 8 uint32 in a
// 256-bit register. Used by bpv=8 unpack and as the tail step of bpv=4/2/1.
BOLT_FORCE_INLINE static __m256i bolt_pfor_avx2_cvt_u8x8_to_u32x8(
    __m128i bytes) noexcept {
    return _mm256_cvtepu8_epi32(bytes);
}

// Unpack 128 values at bpv=1 (16 bytes input → 128 uint32 output). Each bit
// becomes a 0 / 1 uint32. Eight iterations: each consumes 2 input bytes
// (16 bits → 16 uint32), processed via byte broadcast + bit-mask.
BOLT_FORCE_INLINE static void bolt_pfor_avx2_unpack_bpv1_128(
    const uint8_t* BOLT_RESTRICT in,
    uint32_t* BOLT_RESTRICT out) noexcept {
    // Per-byte: load 1 byte, broadcast to 8 lanes, AND with bit-i mask, then
    // compare-equal-to-mask → 0xFFFFFFFF / 0x00000000, then `& 1`.
    const __m256i bit_masks = _mm256_setr_epi32(
        1, 2, 4, 8, 16, 32, 64, 128);
    for (int32_t b = 0; b < 16; ++b) {
        const uint32_t v = static_cast<uint32_t>(in[b]);
        const __m256i broadcast = _mm256_set1_epi32(static_cast<int32_t>(v));
        const __m256i anded     = _mm256_and_si256(broadcast, bit_masks);
        const __m256i is_set    = _mm256_cmpgt_epi32(anded, _mm256_setzero_si256());
        const __m256i ones      = _mm256_and_si256(is_set, _mm256_set1_epi32(1));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + b * 8), ones);
    }
}

// Unpack 128 values at bpv=2 (32 bytes input → 128 uint32 output).
BOLT_FORCE_INLINE static void bolt_pfor_avx2_unpack_bpv2_128(
    const uint8_t* BOLT_RESTRICT in,
    uint32_t* BOLT_RESTRICT out) noexcept {
    // Each byte holds 4 values at shifts {0, 2, 4, 6}.
    const __m256i shifts = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
    const __m256i mask   = _mm256_set1_epi32(0x3);
    for (int32_t b = 0; b < 32; b += 2) {
        const uint32_t lo = static_cast<uint32_t>(in[b]);
        const uint32_t hi = static_cast<uint32_t>(in[b + 1]);
        const __m256i bcast = _mm256_setr_epi32(
            static_cast<int32_t>(lo), static_cast<int32_t>(lo),
            static_cast<int32_t>(lo), static_cast<int32_t>(lo),
            static_cast<int32_t>(hi), static_cast<int32_t>(hi),
            static_cast<int32_t>(hi), static_cast<int32_t>(hi));
        const __m256i shifted = _mm256_srlv_epi32(bcast, shifts);
        const __m256i result  = _mm256_and_si256(shifted, mask);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + (b * 4)), result);
    }
}

// Unpack 128 values at bpv=4 (64 bytes input → 128 uint32 output).
// Implemented via 128-bit ops to avoid AVX2's per-lane unpack semantics
// (where _mm256_unpacklo_epi32 only sees lanes 0..1 / 4..5 of each half).
BOLT_FORCE_INLINE static void bolt_pfor_avx2_unpack_bpv4_128(
    const uint8_t* BOLT_RESTRICT in,
    uint32_t* BOLT_RESTRICT out) noexcept {
    // Each byte holds 2 values: low nibble (i = 2k) and high nibble (i = 2k+1).
    const __m128i mask_lo = _mm_set1_epi32(0x0F);
    // Process 4 input bytes per iteration → 8 output uint32.
    for (int32_t b = 0; b < 64; b += 4) {
        const int32_t raw32 = *reinterpret_cast<const int32_t*>(in + b);
        const __m128i raw   = _mm_cvtsi32_si128(raw32);
        // Zero-extend 4 bytes → 4 uint32 lanes.
        const __m128i bytes = _mm_cvtepu8_epi32(raw);
        const __m128i lo    = _mm_and_si128(bytes, mask_lo);
        const __m128i hi    = _mm_srli_epi32(bytes, 4);
        // Interleave the 4 lanes: out0 = [lo0,hi0,lo1,hi1], out1 = [lo2,hi2,lo3,hi3].
        const __m128i o0 = _mm_unpacklo_epi32(lo, hi);
        const __m128i o1 = _mm_unpackhi_epi32(lo, hi);
        uint32_t* dst = out + b * 2;
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 0), o0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 4), o1);
    }
}

// Unpack 128 values at bpv=8 (128 bytes input → 128 uint32 output).
BOLT_FORCE_INLINE static void bolt_pfor_avx2_unpack_bpv8_128(
    const uint8_t* BOLT_RESTRICT in,
    uint32_t* BOLT_RESTRICT out) noexcept {
    // 16 iterations × 8 lanes = 128 values.
    for (int32_t i = 0; i < 128; i += 8) {
        const __m128i raw = _mm_loadl_epi64(
            reinterpret_cast<const __m128i*>(in + i));
        const __m256i wide = _mm256_cvtepu8_epi32(raw);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), wide);
    }
}

// Unpack 128 values at bpv=16 (256 bytes input → 128 uint32 output).
BOLT_FORCE_INLINE static void bolt_pfor_avx2_unpack_bpv16_128(
    const uint8_t* BOLT_RESTRICT in,
    uint32_t* BOLT_RESTRICT out) noexcept {
    // 16 iterations × 8 lanes = 128 values.
    for (int32_t i = 0; i < 128; i += 8) {
        const __m128i raw = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(in + i * 2));
        const __m256i wide = _mm256_cvtepu16_epi32(raw);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), wide);
    }
}

// Pack 128 values at bpv=8 (128 uint32 → 128 bytes).
BOLT_FORCE_INLINE static void bolt_pfor_avx2_pack_bpv8_128(
    const uint32_t* BOLT_RESTRICT in,
    uint8_t* BOLT_RESTRICT out) noexcept {
    // bpv=8 is byte-aligned; each input value's low byte goes to one output
    // byte. _mm256_packus_epi32 saturates, but inputs are already <= 0xFF
    // (callers mask before pack — and bpv=8 mask is 0xFF). We do an explicit
    // mask anyway for safety on the test path that may pass un-masked ints.
    const __m256i mask = _mm256_set1_epi32(0xFF);
    for (int32_t i = 0; i < 128; i += 16) {
        __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i + 0));
        __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i + 8));
        v0 = _mm256_and_si256(v0, mask);
        v1 = _mm256_and_si256(v1, mask);
        // Narrow 32→16 (saturating, but mask ensures no saturation).
        __m256i p16 = _mm256_packus_epi32(v0, v1);
        // p16 lane order (per 128-bit half): [v0_lo, v1_lo] then [v0_hi, v1_hi].
        // Permute the 64-bit qwords to undo the lane-interleave.
        p16 = _mm256_permute4x64_epi64(p16, 0xD8);  // 0b11011000
        // Narrow 16→8 by packing with itself, then take low 128 bits.
        __m256i p8 = _mm256_packus_epi16(p16, p16);
        p8 = _mm256_permute4x64_epi64(p8, 0xD8);
        const __m128i lo = _mm256_castsi256_si128(p8);
        // Use OR-store to satisfy the "out is pre-zeroed; OR in" contract.
        __m128i prev = _mm_loadu_si128(reinterpret_cast<__m128i*>(out + i));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + i),
                         _mm_or_si128(prev, lo));
    }
}

// Pack 128 values at bpv=16 (128 uint32 → 256 bytes).
BOLT_FORCE_INLINE static void bolt_pfor_avx2_pack_bpv16_128(
    const uint32_t* BOLT_RESTRICT in,
    uint8_t* BOLT_RESTRICT out) noexcept {
    const __m256i mask = _mm256_set1_epi32(0xFFFF);
    for (int32_t i = 0; i < 128; i += 16) {
        __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i + 0));
        __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i + 8));
        v0 = _mm256_and_si256(v0, mask);
        v1 = _mm256_and_si256(v1, mask);
        __m256i p16 = _mm256_packus_epi32(v0, v1);
        p16 = _mm256_permute4x64_epi64(p16, 0xD8);
        __m256i prev = _mm256_loadu_si256(
            reinterpret_cast<__m256i*>(out + i * 2));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i * 2),
                            _mm256_or_si256(prev, p16));
    }
}

// Pack 128 values at bpv=4 (128 uint32 → 64 bytes). Per output byte b:
// (in[2b] & 0xF) | ((in[2b+1] & 0xF) << 4). The pairs are adjacent in input
// memory so a wide gather + lane-shuffle would re-order across pairs;
// instead we let the compiler auto-vectorise the scalar fold (MSVC /O2
// emits SSE pack on this loop). The win at bpv=4 sits in the unpack path.
BOLT_FORCE_INLINE static void bolt_pfor_avx2_pack_bpv4_128(
    const uint32_t* BOLT_RESTRICT in,
    uint8_t* BOLT_RESTRICT out) noexcept {
    for (int32_t b = 0; b < 64; ++b) {
        const uint32_t a = in[2 * b + 0] & 0x0Fu;
        const uint32_t c = in[2 * b + 1] & 0x0Fu;
        out[b] = static_cast<uint8_t>(out[b] | (a | (c << 4)));
    }
}

// Pack 128 values at bpv=2 (128 uint32 → 32 bytes).
BOLT_FORCE_INLINE static void bolt_pfor_avx2_pack_bpv2_128(
    const uint32_t* BOLT_RESTRICT in,
    uint8_t* BOLT_RESTRICT out) noexcept {
    // Per output byte b: pack in[4b..4b+3] (each masked to 2 bits) into
    // 8 bits at shifts {0, 2, 4, 6}.
    for (int32_t b = 0; b < 32; ++b) {
        const uint32_t v0 = in[4 * b + 0] & 0x3u;
        const uint32_t v1 = in[4 * b + 1] & 0x3u;
        const uint32_t v2 = in[4 * b + 2] & 0x3u;
        const uint32_t v3 = in[4 * b + 3] & 0x3u;
        const uint8_t  packed = static_cast<uint8_t>(
            v0 | (v1 << 2) | (v2 << 4) | (v3 << 6));
        out[b] = static_cast<uint8_t>(out[b] | packed);
    }
}

// Pack 128 values at bpv=1 (128 uint32 → 16 bytes).
BOLT_FORCE_INLINE static void bolt_pfor_avx2_pack_bpv1_128(
    const uint32_t* BOLT_RESTRICT in,
    uint8_t* BOLT_RESTRICT out) noexcept {
    for (int32_t b = 0; b < 16; ++b) {
        uint8_t packed = 0u;
        for (int32_t k = 0; k < 8; ++k) {
            packed = static_cast<uint8_t>(
                packed | ((in[8 * b + k] & 0x1u) << k));
        }
        out[b] = static_cast<uint8_t>(out[b] | packed);
    }
}

// Public AVX2 entry points — dispatch by bpv. Return false on unsupported
// shape so the caller falls through to scalar.
BOLT_FORCE_INLINE bool unpack_avx2(const uint8_t* BOLT_RESTRICT in,
                                    int32_t n_values, uint8_t bpv,
                                    uint32_t* BOLT_RESTRICT out) noexcept {
    assert(in != nullptr);
    assert(out != nullptr);
    if (n_values != kSimdBlockSize) { return false; }
    switch (bpv) {
        case 1u:  bolt_pfor_avx2_unpack_bpv1_128(in, out);  return true;
        case 2u:  bolt_pfor_avx2_unpack_bpv2_128(in, out);  return true;
        case 4u:  bolt_pfor_avx2_unpack_bpv4_128(in, out);  return true;
        case 8u:  bolt_pfor_avx2_unpack_bpv8_128(in, out);  return true;
        case 16u: bolt_pfor_avx2_unpack_bpv16_128(in, out); return true;
        default:  return false;
    }
}

BOLT_FORCE_INLINE bool pack_avx2(const uint32_t* BOLT_RESTRICT in,
                                  int32_t n_values, uint8_t bpv,
                                  uint8_t* BOLT_RESTRICT out) noexcept {
    assert(in != nullptr);
    assert(out != nullptr);
    if (n_values != kSimdBlockSize) { return false; }
    switch (bpv) {
        case 1u:  bolt_pfor_avx2_pack_bpv1_128(in, out);  return true;
        case 2u:  bolt_pfor_avx2_pack_bpv2_128(in, out);  return true;
        case 4u:  bolt_pfor_avx2_pack_bpv4_128(in, out);  return true;
        case 8u:  bolt_pfor_avx2_pack_bpv8_128(in, out);  return true;
        case 16u: bolt_pfor_avx2_pack_bpv16_128(in, out); return true;
        default:  return false;
    }
}

#endif  // BOLT_SIMD_AVX2

// ---------------------------------------------------------------------------
// SSE4.2 path — 128-bit lanes. Same shape as AVX2 but half-width. Used as
// the x86-64 V1 baseline fallback when /arch:AVX2 is not set.
// ---------------------------------------------------------------------------
#if BOLT_SIMD_SSE42

BOLT_FORCE_INLINE static void bolt_pfor_sse42_unpack_bpv1_128(
    const uint8_t* BOLT_RESTRICT in,
    uint32_t* BOLT_RESTRICT out) noexcept {
    const __m128i bit_masks_lo = _mm_setr_epi32(1, 2, 4, 8);
    const __m128i bit_masks_hi = _mm_setr_epi32(16, 32, 64, 128);
    for (int32_t b = 0; b < 16; ++b) {
        const uint32_t v = static_cast<uint32_t>(in[b]);
        const __m128i bcast = _mm_set1_epi32(static_cast<int32_t>(v));
        const __m128i lo_and = _mm_and_si128(bcast, bit_masks_lo);
        const __m128i hi_and = _mm_and_si128(bcast, bit_masks_hi);
        const __m128i lo_set = _mm_cmpgt_epi32(lo_and, _mm_setzero_si128());
        const __m128i hi_set = _mm_cmpgt_epi32(hi_and, _mm_setzero_si128());
        const __m128i lo_one = _mm_and_si128(lo_set, _mm_set1_epi32(1));
        const __m128i hi_one = _mm_and_si128(hi_set, _mm_set1_epi32(1));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + b * 8 + 0), lo_one);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + b * 8 + 4), hi_one);
    }
}

BOLT_FORCE_INLINE static void bolt_pfor_sse42_unpack_bpv2_128(
    const uint8_t* BOLT_RESTRICT in,
    uint32_t* BOLT_RESTRICT out) noexcept {
    const __m128i mask = _mm_set1_epi32(0x3);
    for (int32_t b = 0; b < 32; ++b) {
        const uint32_t v = static_cast<uint32_t>(in[b]);
        const __m128i bcast = _mm_setr_epi32(
            static_cast<int32_t>(v >> 0), static_cast<int32_t>(v >> 2),
            static_cast<int32_t>(v >> 4), static_cast<int32_t>(v >> 6));
        const __m128i result = _mm_and_si128(bcast, mask);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + b * 4), result);
    }
}

BOLT_FORCE_INLINE static void bolt_pfor_sse42_unpack_bpv4_128(
    const uint8_t* BOLT_RESTRICT in,
    uint32_t* BOLT_RESTRICT out) noexcept {
    for (int32_t b = 0; b < 64; ++b) {
        const uint32_t byte_v = static_cast<uint32_t>(in[b]);
        out[2 * b + 0] = byte_v & 0x0Fu;
        out[2 * b + 1] = (byte_v >> 4) & 0x0Fu;
    }
}

BOLT_FORCE_INLINE static void bolt_pfor_sse42_unpack_bpv8_128(
    const uint8_t* BOLT_RESTRICT in,
    uint32_t* BOLT_RESTRICT out) noexcept {
    // _mm_cvtepu8_epi32 takes 4 bytes → 4 uint32. 32 iterations × 4 = 128.
    for (int32_t i = 0; i < 128; i += 4) {
        const __m128i raw = _mm_cvtsi32_si128(
            *reinterpret_cast<const int32_t*>(in + i));
        const __m128i wide = _mm_cvtepu8_epi32(raw);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + i), wide);
    }
}

BOLT_FORCE_INLINE static void bolt_pfor_sse42_unpack_bpv16_128(
    const uint8_t* BOLT_RESTRICT in,
    uint32_t* BOLT_RESTRICT out) noexcept {
    // _mm_cvtepu16_epi32 takes 4 uint16 → 4 uint32. 32 iterations × 4 = 128.
    for (int32_t i = 0; i < 128; i += 4) {
        const __m128i raw = _mm_loadl_epi64(
            reinterpret_cast<const __m128i*>(in + i * 2));
        const __m128i wide = _mm_cvtepu16_epi32(raw);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + i), wide);
    }
}

BOLT_FORCE_INLINE static void bolt_pfor_sse42_pack_bpv1_128(
    const uint32_t* BOLT_RESTRICT in,
    uint8_t* BOLT_RESTRICT out) noexcept {
    for (int32_t b = 0; b < 16; ++b) {
        uint8_t packed = 0u;
        for (int32_t k = 0; k < 8; ++k) {
            packed = static_cast<uint8_t>(
                packed | ((in[8 * b + k] & 0x1u) << k));
        }
        out[b] = static_cast<uint8_t>(out[b] | packed);
    }
}

BOLT_FORCE_INLINE static void bolt_pfor_sse42_pack_bpv2_128(
    const uint32_t* BOLT_RESTRICT in,
    uint8_t* BOLT_RESTRICT out) noexcept {
    for (int32_t b = 0; b < 32; ++b) {
        const uint32_t v0 = in[4 * b + 0] & 0x3u;
        const uint32_t v1 = in[4 * b + 1] & 0x3u;
        const uint32_t v2 = in[4 * b + 2] & 0x3u;
        const uint32_t v3 = in[4 * b + 3] & 0x3u;
        const uint8_t  packed = static_cast<uint8_t>(
            v0 | (v1 << 2) | (v2 << 4) | (v3 << 6));
        out[b] = static_cast<uint8_t>(out[b] | packed);
    }
}

BOLT_FORCE_INLINE static void bolt_pfor_sse42_pack_bpv4_128(
    const uint32_t* BOLT_RESTRICT in,
    uint8_t* BOLT_RESTRICT out) noexcept {
    for (int32_t b = 0; b < 64; ++b) {
        const uint32_t a = in[2 * b + 0] & 0x0Fu;
        const uint32_t c = in[2 * b + 1] & 0x0Fu;
        out[b] = static_cast<uint8_t>(out[b] | (a | (c << 4)));
    }
}

BOLT_FORCE_INLINE static void bolt_pfor_sse42_pack_bpv8_128(
    const uint32_t* BOLT_RESTRICT in,
    uint8_t* BOLT_RESTRICT out) noexcept {
    const __m128i mask = _mm_set1_epi32(0xFF);
    for (int32_t i = 0; i < 128; i += 8) {
        __m128i v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + i + 0));
        __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + i + 4));
        v0 = _mm_and_si128(v0, mask);
        v1 = _mm_and_si128(v1, mask);
        __m128i p16 = _mm_packus_epi32(v0, v1);
        __m128i p8  = _mm_packus_epi16(p16, p16);
        // Take low 64 bits → 8 bytes.
        const uint64_t low64 = static_cast<uint64_t>(_mm_cvtsi128_si64(p8));
        uint64_t prev;
        std::memcpy(&prev, out + i, 8);
        prev |= low64;
        std::memcpy(out + i, &prev, 8);
    }
}

BOLT_FORCE_INLINE static void bolt_pfor_sse42_pack_bpv16_128(
    const uint32_t* BOLT_RESTRICT in,
    uint8_t* BOLT_RESTRICT out) noexcept {
    const __m128i mask = _mm_set1_epi32(0xFFFF);
    for (int32_t i = 0; i < 128; i += 8) {
        __m128i v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + i + 0));
        __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + i + 4));
        v0 = _mm_and_si128(v0, mask);
        v1 = _mm_and_si128(v1, mask);
        __m128i p16 = _mm_packus_epi32(v0, v1);
        __m128i prev = _mm_loadu_si128(
            reinterpret_cast<__m128i*>(out + i * 2));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + i * 2),
                         _mm_or_si128(prev, p16));
    }
}

BOLT_FORCE_INLINE bool unpack_sse42(const uint8_t* BOLT_RESTRICT in,
                                     int32_t n_values, uint8_t bpv,
                                     uint32_t* BOLT_RESTRICT out) noexcept {
    assert(in != nullptr);
    assert(out != nullptr);
    if (n_values != kSimdBlockSize) { return false; }
    switch (bpv) {
        case 1u:  bolt_pfor_sse42_unpack_bpv1_128(in, out);  return true;
        case 2u:  bolt_pfor_sse42_unpack_bpv2_128(in, out);  return true;
        case 4u:  bolt_pfor_sse42_unpack_bpv4_128(in, out);  return true;
        case 8u:  bolt_pfor_sse42_unpack_bpv8_128(in, out);  return true;
        case 16u: bolt_pfor_sse42_unpack_bpv16_128(in, out); return true;
        default:  return false;
    }
}

BOLT_FORCE_INLINE bool pack_sse42(const uint32_t* BOLT_RESTRICT in,
                                   int32_t n_values, uint8_t bpv,
                                   uint8_t* BOLT_RESTRICT out) noexcept {
    assert(in != nullptr);
    assert(out != nullptr);
    if (n_values != kSimdBlockSize) { return false; }
    switch (bpv) {
        case 1u:  bolt_pfor_sse42_pack_bpv1_128(in, out);  return true;
        case 2u:  bolt_pfor_sse42_pack_bpv2_128(in, out);  return true;
        case 4u:  bolt_pfor_sse42_pack_bpv4_128(in, out);  return true;
        case 8u:  bolt_pfor_sse42_pack_bpv8_128(in, out);  return true;
        case 16u: bolt_pfor_sse42_pack_bpv16_128(in, out); return true;
        default:  return false;
    }
}

#endif  // BOLT_SIMD_SSE42

}  // namespace simd
}  // namespace pfor
}  // namespace kernels
}  // namespace bolt
