#pragma once
// bolt_quantized_distance.h — Quantized + low-precision distance kernels.
// Header-only, scalar-first. AVX2 specialisations are deferred (see Wave
// 19 plan B). Mirrors the bolt_vector_distance.h shape but works on
// uint8 / uint16 / bit-packed payloads.
//
// Functions:
//   - l2_vertical_sq8_accumulate   8-bit scalar quantized vertical L2.
//   - l2_pair_rabitq               1-bit RaBitQ pair Hamming distance.
//   - l2_vertical_f16_accumulate   IEEE binary16 vertical L2 (scalar).
//   - f32_pack_sq8                 build-side f32 → SQ8 quantize.
//   - f32_pack_rabitq              build-side f32 → 1-bit RaBitQ pack.
//
// Usage convention:
//   * Caller resets the accumulator buffer to 0 before the first dim.
//   * Callers of `l2_vertical_sq8_accumulate` dequantize the squared
//     accumulator with `scale*scale` once at the end. The kernel itself
//     keeps everything in int32 lattice space (max bound: D * 255^2,
//     comfortably below 2^31 for any sensible D ≤ 32k).
//   * `l2_pair_rabitq` returns the raw Hamming popcount; the caller
//     applies the published RaBitQ → L2 correction term. We deliberately
//     keep this kernel as a pure popcount so it stays composable with
//     the asymmetric / two-step variants.
//
// Tiger Style: noexcept, ≥2 asserts per function, BOLT_RESTRICT on every
// pointer parameter, no STL containers, no exceptions.

#include "bolt/bolt_port.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace bolt {

namespace detail {

// Bit-cast helpers (memcpy-based, MSVC friendly, no UB).
BOLT_FORCE_INLINE float qd_bits_to_f32(uint32_t bits) noexcept {
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// Scalar IEEE binary16 → float32 expander. Branchless over the common
// path; subnormals fall through a tiny normalize loop. F16C intrinsic
// specialisation is deferred to a future wave.
BOLT_FORCE_INLINE float qd_f16_to_f32(uint16_t h) noexcept {
    const uint32_t sign     = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp_bits = static_cast<uint32_t>(h & 0x7C00u) >> 10;
    const uint32_t mant     = static_cast<uint32_t>(h & 0x03FFu);
    uint32_t out;
    if (exp_bits == 0u) {
        if (mant == 0u) {
            out = sign;  // +/-0.
        } else {
            // Subnormal: normalize.
            uint32_t m = mant;
            int32_t  e = -1;
            do {
                m <<= 1;
                e  -= 1;
            } while ((m & 0x0400u) == 0u);
            m &= 0x03FFu;
            const uint32_t f32_exp = static_cast<uint32_t>(127 - 15 + e + 1);
            out = sign | (f32_exp << 23) | (m << 13);
        }
    } else if (exp_bits == 0x1Fu) {
        // Inf / NaN — preserve mantissa payload.
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        const uint32_t f32_exp = exp_bits + (127u - 15u);
        out = sign | (f32_exp << 23) | (mant << 13);
    }
    return qd_bits_to_f32(out);
}

}  // namespace detail

// ---------------------------------------------------------------------------
// l2_vertical_sq8_accumulate
// dists_sq[i] += (col[i] - q)^2, col & q in [0,255].
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE void l2_vertical_sq8_accumulate(
    const uint8_t* BOLT_RESTRICT dim_col, uint8_t q_quantized,
    int32_t* BOLT_RESTRICT dists_sq, size_t n_vectors) noexcept {
    assert(dim_col  != nullptr || n_vectors == 0);
    assert(dists_sq != nullptr || n_vectors == 0);
    const int32_t q = static_cast<int32_t>(q_quantized);
    for (size_t i = 0; i < n_vectors; ++i) {
        const int32_t d = static_cast<int32_t>(dim_col[i]) - q;
        dists_sq[i] += d * d;
    }
}

// ---------------------------------------------------------------------------
// l2_pair_rabitq
// *out_hamming = popcount(a XOR b) over n_words u64s.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE void l2_pair_rabitq(
    const uint64_t* BOLT_RESTRICT a_bits,
    const uint64_t* BOLT_RESTRICT b_bits,
    size_t n_words, int32_t* BOLT_RESTRICT out_hamming) noexcept {
    assert(a_bits      != nullptr || n_words == 0);
    assert(b_bits      != nullptr || n_words == 0);
    assert(out_hamming != nullptr);
    uint32_t acc = 0;
    for (size_t w = 0; w < n_words; ++w) {
        acc += bolt_popcount64(a_bits[w] ^ b_bits[w]);
    }
    *out_hamming = static_cast<int32_t>(acc);
}

// ---------------------------------------------------------------------------
// l2_vertical_f16_accumulate
// dists[i] += (q - col[i])^2, col & q are IEEE binary16 (u16).
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE void l2_vertical_f16_accumulate(
    const uint16_t* BOLT_RESTRICT dim_col, uint16_t q_broadcast,
    float* BOLT_RESTRICT dists, size_t n_vectors) noexcept {
    assert(dim_col != nullptr || n_vectors == 0);
    assert(dists   != nullptr || n_vectors == 0);
    const float q = detail::qd_f16_to_f32(q_broadcast);
    for (size_t i = 0; i < n_vectors; ++i) {
        const float x = detail::qd_f16_to_f32(dim_col[i]);
        const float d = q - x;
        dists[i] += d * d;
    }
}

// ---------------------------------------------------------------------------
// f32_pack_sq8 — build-side. Quantize f32 → uint8 with affine map.
// `min` is the per-cluster (or global) minimum; `scale` is the
// dequantization multiplier so that `f ≈ min + q * scale`.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE void f32_pack_sq8(
    const float* BOLT_RESTRICT in, size_t n,
    float min, float scale,
    uint8_t* BOLT_RESTRICT out) noexcept {
    assert(in  != nullptr || n == 0);
    assert(out != nullptr || n == 0);
    assert(scale > 0.0f);
    const float inv = 1.0f / scale;
    for (size_t i = 0; i < n; ++i) {
        float q = std::floor((in[i] - min) * inv + 0.5f);
        if (q < 0.0f)   q = 0.0f;
        if (q > 255.0f) q = 255.0f;
        out[i] = static_cast<uint8_t>(q);
    }
}

// ---------------------------------------------------------------------------
// f32_pack_rabitq — build-side. Pack bit i = (in[i] >= threshold).
// Output length is ceil(n/64) u64 words. Trailing bits beyond `n` in the
// final word are written as 0.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE void f32_pack_rabitq(
    const float* BOLT_RESTRICT in, size_t n,
    float threshold,
    uint64_t* BOLT_RESTRICT out_bits) noexcept {
    assert(in       != nullptr || n == 0);
    assert(out_bits != nullptr || n == 0);
    const size_t n_words = (n + 63) / 64;
    for (size_t w = 0; w < n_words; ++w) {
        out_bits[w] = 0;
    }
    for (size_t i = 0; i < n; ++i) {
        const uint64_t bit = (in[i] >= threshold) ? 1ull : 0ull;
        out_bits[i >> 6] |= (bit << (i & 63));
    }
}

}  // namespace bolt
