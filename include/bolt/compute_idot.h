// compute_idot.h -- int8-ACTIVATION quantized GEMV for bolt::compute
// (tracker BLLM-14). The specialized ultra-hot decode primitive an LLM
// engine needs: a matrix-vector product where BOTH the activation vector
// AND the weight rows are int8/int4-quantized, dotted in pure integer
// arithmetic with a single final float rescale per output row.
//
// This is a DIFFERENT numerics tier from bolt_compute.h's matmul (BLLM-13),
// which dots a PLAIN FLOAT activation against a dequant-on-the-fly weight
// row (the weight-only-quant scheme). Here the activation is ALSO already
// int8-quantized -- the shape that lets VPMADDUBSW (AVX2) / VPDPBUSD
// (AVX-512VNNI) do the dot product entirely in the integer pipeline, ~19-21x
// faster than a scalar float dot (142-155 GFLOPS measured; see
// docs/research/idot-runtime-dispatch.md). That same float-activation vs
// int8-activation split reappears on the GPU (a caller wanting the GPU tier
// uses a separate device path), which is evidence it is intrinsic to the
// problem, not an artifact of this codebase -- see the design-log entry.
//
// UNLIKE the rest of bolt::compute (compile-time SIMD-tier selection, header-
// only), this capability RUNTIME-dispatches over a CPUID probe: the VNNI/AVX2
// kernels are in a compiled lib (bolt::compute_idot) and the tier is resolved
// once via a magic-static function-pointer table. That runtime probe is a
// deliberate exception to bolt's usual "no runtime dispatch" rule (io/
// bolt_crc32c.h explicitly refuses to probe) -- justified here because the
// probe is amortized once over a heavy GEMV, not per-call over a 3ns
// primitive. See docs/research/idot-runtime-dispatch.md for the reconciliation.
//
// Quant format: bolt::Tensor's Int8/Int4 per-row-scale convention
// (bolt_tensor.h), which mirrors boltllm::quant::QTensor's format
// (Colibri glm.c, Apache-2.0 -- cited in bolt_tensor.h). Int4 nibble order
// (low-nibble-first) + offset (stored = value + 8) reuse
// bolt::compute::detail::kInt4DequantOffset so there is ONE bolt source of
// truth for the constant.
//
// This header carries ONLY declarations plus a scalar (no-intrinsics)
// activation quantizer, so it is safe to #include ANYWHERE regardless of the
// consumer's compiled ISA tier -- the intrinsics live entirely in the
// bolt::compute_idot compiled TUs.

#pragma once

#include "bolt/bolt_tensor.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace bolt::compute {

// out[r] = float(sum_i act_q[i] * weight_q[i]) * act_scale * weight.scale(r),
// for r in [0, rows). `act_q` is `weight.cols()` already-int8-quantized
// activation values (produce them with quantize_act_row_int8 below, or any
// equivalent per-row-scale int8 quantizer). `weight` must be a CPU Int8
// (matmul_i8a_int8) or Int4 (matmul_i8a_int4) Tensor with a valid row_scales
// array. `out` receives `rows` floats. Blocking, noexcept, no allocation.
// Runtime-dispatches to AVX-512VNNI > AVX2 > scalar based on the CPUID probe.
// Precondition: act_q/out non-null, act_scale > 0, rows in (0, weight.rows()].
void matmul_i8a_int8(const int8_t* act_q, float act_scale, const Tensor& weight, float* out,
                     int32_t rows) noexcept;
void matmul_i8a_int4(const int8_t* act_q, float act_scale, const Tensor& weight, float* out,
                     int32_t rows) noexcept;

// True iff the CPUID probe selected the AVX2 / AVX-512VNNI kernels on this
// machine. For tests that self-skip when the fast tier is unavailable, and
// for callers that want to know which tier is live. Safe to call repeatedly
// (the probe is a one-time magic-static). Always false on non-x86 builds.
bool idot_avx2_available() noexcept;
bool idot_avx512vnni_available() noexcept;

// ---------------------------------------------------------------------------
// Scalar activation quantizer -- turns one f32 activation row into `cols`
// int8 values + a single per-row f32 scale, using the SAME per-row-scale +
// ties-to-even convention bolt::Tensor's weight quantization uses (max(|a|)/
// 127 floored at 1e-8, round-half-even, clamp [-128, 127]). Restated inline
// here (no intrinsics, no bolt-lib dependency) so this header stays
// universally includable; a caller feeds its output as matmul_i8a_*'s
// `act_q`/`act_scale`. Ported from boltllm::quant::ActQuant.h /
// QuantScalar.h (which derive the formula from Colibri glm.c, Apache-2.0),
// restated rather than #included per bolt's one-way-dependency rule.
// ---------------------------------------------------------------------------

namespace detail {

inline int32_t idot_round_half_even_i32(float x) noexcept {
    const float f = std::floor(x);
    const float diff = x - f;
    const int32_t fi = static_cast<int32_t>(f);
    if (diff < 0.5f) return fi;
    if (diff > 0.5f) return fi + 1;
    return (fi % 2 == 0) ? fi : fi + 1;  // exact tie -> even
}

inline int32_t idot_clamp_i32(int32_t v, int32_t lo, int32_t hi) noexcept {
    assert(lo <= hi);
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

inline constexpr int32_t kIdotInt8QMax = 127;
inline constexpr int32_t kIdotInt8QMin = -128;
inline constexpr float   kIdotMinRowScale = 1e-8f;

}  // namespace detail

inline void quantize_act_row_int8(const float* a, int32_t cols, int8_t* q_out,
                                   float* scale_out) noexcept {
    assert(a != nullptr && q_out != nullptr && scale_out != nullptr);
    assert(cols > 0);

    float amax = 0.0f;
    for (int32_t i = 0; i < cols; ++i) {
        const float av = std::fabs(a[i]);
        if (av > amax) amax = av;
    }
    float s = amax / static_cast<float>(detail::kIdotInt8QMax);
    if (s < detail::kIdotMinRowScale) s = detail::kIdotMinRowScale;
    *scale_out = s;

    for (int32_t i = 0; i < cols; ++i) {
        const int32_t v = detail::idot_round_half_even_i32(a[i] / s);
        q_out[i] = static_cast<int8_t>(
            detail::idot_clamp_i32(v, detail::kIdotInt8QMin, detail::kIdotInt8QMax));
    }
}

}  // namespace bolt::compute
