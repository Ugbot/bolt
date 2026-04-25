#pragma once
#include "bolt/bolt_port.h"
#include <cstdint>
#include <cstddef>
#include <cassert>
#include <cmath>
#include <cstring>

namespace bolt {

// Apply Householder QR rotation: x_out = R @ x_in where R is stored
// as a sequence of n_reflectors Householder reflectors, each of length
// D, packed contiguously in `reflectors[r * D + d]`. Each reflector is
// a unit-norm vector v; the apply is x <- x - 2 (v.x) v.
//
// Aliasing: x_in and x_out may not overlap. First step copies; later
// steps update in place on x_out.
//
// Scalar implementation; AVX2 specialisation deferred to a follow-up.
BOLT_FORCE_INLINE void apply_qr_rotation_f32(
    const float* BOLT_RESTRICT reflectors,
    size_t D, size_t n_reflectors,
    const float* BOLT_RESTRICT x_in,
    float* BOLT_RESTRICT x_out
) noexcept {
    assert(reflectors != nullptr || (D == 0 || n_reflectors == 0));
    assert(x_in != nullptr || D == 0);
    assert(x_out != nullptr || D == 0);
    if (D == 0) return;
    // Copy input to output once; reflector loop updates output in place.
    std::memcpy(x_out, x_in, D * sizeof(float));
    for (size_t r = 0; r < n_reflectors; ++r) {
        const float* v = reflectors + r * D;
        // dot = v . x_out
        float dot = 0.0f;
        for (size_t d = 0; d < D; ++d) dot += v[d] * x_out[d];
        const float two_dot = 2.0f * dot;
        // x_out -= 2 * dot * v
        for (size_t d = 0; d < D; ++d) x_out[d] -= two_dot * v[d];
    }
}

// Apply DCT-II rotation with random +/-1 signs. O(D log D). For D >= 512
// in production, but correctness works for any power-of-2 D.
//
// `signs[d]` is +1 or -1 (stored as int8_t). Pre-multiplies x by signs,
// then performs an in-place radix-2 DIT DCT-II on a scratch buffer of
// length D, writes the result to x_out.
//
// Implementation: simple O(D^2) reference DCT-II (cosine sum). For
// D=512 that's 256K ops per query - fine for the first-cut correctness
// pass. Lee 1984 O(D log D) variant is a follow-up. We document
// the perf trade-off inline.
BOLT_FORCE_INLINE void apply_dct_rotation_f32(
    const int8_t* BOLT_RESTRICT signs,
    size_t D,
    const float* BOLT_RESTRICT x_in,
    float* BOLT_RESTRICT x_out
) noexcept {
    assert(signs != nullptr || D == 0);
    assert(x_in != nullptr || D == 0);
    assert(x_out != nullptr || D == 0);
    if (D == 0) return;
    constexpr float kPi = 3.14159265358979323846f;
    const float scale = std::sqrt(2.0f / static_cast<float>(D));
    for (size_t k = 0; k < D; ++k) {
        float sum = 0.0f;
        for (size_t n = 0; n < D; ++n) {
            const float xn = x_in[n] * static_cast<float>(signs[n]);
            const float angle = kPi * (2.0f * static_cast<float>(n) + 1.0f)
                              * static_cast<float>(k) / (2.0f * static_cast<float>(D));
            sum += xn * std::cos(angle);
        }
        const float kk = (k == 0) ? (1.0f / std::sqrt(2.0f)) : 1.0f;
        x_out[k] = scale * kk * sum;
    }
}

}  // namespace bolt
