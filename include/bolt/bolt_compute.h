// bolt_compute.h — Device-generic compute dispatch table (BLLM-13), layered
// on top of bolt_tensor.h's `bolt::Tensor`/`bolt::Device` (BLLM-12, N0
// substrate). This is boltllm's forward-model compute ops (matmul/rmsnorm/
// rope/softmax/swiglu/sampling), generalized to a `Device` axis so a later
// Vulkan/CUDA backend can slot in without boltllm growing a second, private
// dispatch mechanism -- exactly the role bolt_tensor.h's own banner describes
// for itself ("QTensor's eventual migration onto this type is a separate,
// later ticket (BLLM-14)").
//
// SHAPE: this mirrors boltllm::quant::IdotDispatch.h/.cpp precisely --
//   - lazy, thread-safe (C++11 magic-static / function-local `static`)
//     one-time table construction;
//   - the hot path is ONE indirect call through a resolved function pointer,
//     no per-call branching;
//   - a `Device` axis widens the table: CPU is genuinely implemented today,
//     Vulkan/CUDA reserve the slot but point at a hard-failing stub (see
//     `device_not_implemented` below) rather than silently doing nothing or
//     silently falling back to CPU.
// Unlike IdotDispatch.h/.cpp, this needs no separate .cpp translation unit:
// there is no CPUID/intrinsics probe to isolate (see the "Architectural
// choices" section below for why), so -- matching bolt_tensor.h's own
// file organization (BLLM-12: header-only, no .cpp) -- this is header-only.
//
// ─────────────────────────────────────────────────────────────────────────
// Architectural choices (explained here since each is a real design call,
// not an arbitrary default):
//
// 1. matmul's CPU tier is a WEIGHT-DTYPE-GENERIC scalar GEMV (F32 direct;
//    Int8/Int4/Int2 dequantized on the fly via each row's `Tensor::scale()`),
//    dotted against a PLAIN FLOAT activation vector. This is deliberately
//    NOT a port of boltllm::quant::IdotDispatch.h's kernels: those take an
//    ALREADY INT8-QUANTIZED activation (boltllm's ActQuant.h has no Bolt
//    equivalent) and are a pure-integer, AVX2/AVX-512VNNI-accelerated hot
//    path that is boltllm-specific and stays in boltllm (this ticket's
//    explicit scope boundary). The natural generalization at BOLT's level --
//    where `Tensor` has no "pre-quantized activation" concept at all -- is
//    instead the WEIGHT-only-quantization shape boltllm's own
//    QuantScalar.h::dot_row_f32/dot_row_f32_int8/int4/int2 already uses
//    (float activation, optionally-quantized weight row, dequant-then-dot).
//    `detail::matmul_row_dot_f32` below is a from-scratch reimplementation of
//    that exact shape over `bolt::Tensor` instead of `boltllm::quant::QTensor`
//    (same nibble/2-bit-code unpacking convention bolt_tensor.h's own banner
//    already documents Tensor must carry forward) -- not a duplicate of
//    IdotDispatch.h's kernels, which solve a different problem (activation
//    quantization) this header does not model.
//    SEAM for BLLM-14: boltllm's own IDOT kernels (int8-activation, AVX2/
//    VNNI) remain boltllm's specialized ultra-hot decode path. Whether
//    BLLM-14 registers them as an alternate entry in this same table (e.g. a
//    second Tensor dtype/activation-format tag) or keeps them as boltllm's
//    own separate API is an open decision for that ticket -- this header
//    only needs to make that choice possible, not make it. TODO(BLLM-14).
//
// 2. rmsnorm/softmax/swiglu/rope are straightforward CPU-scalar ports of
//    boltllm::moe::MoeMath.h's existing reference logic, restated here (not
//    `#include`d -- Bolt must not depend on boltllm) with identical
//    numerics (same f64 reduction accumulators, same stability tricks). Two
//    RoPE conventions are implemented, per this ticket's brief:
//      - `rope_interleave` -- MoeMath.h's own convention (GLM-5.2/DeepSeek-V3
//        family): adjacent-pair (x[2i],x[2i+1]) input, split-half
//        (x[i],x[half+i]) output deposit. Ported verbatim from MoeMath.h
//        (same citations: Colibri c/glm.c + HF transformers
//        apply_rotary_pos_emb_interleave, fetched 2026-07-15).
//      - `rope_half_split` -- the OTHER convention MoeMath.h's own banner
//        flags but does not implement ("a different, non-MLA dense GLM
//        family... keeps output in the original interleaved slots" --
//        MoeMath.h lines 82-87). This is the widely-used GPT-NeoX/Llama-
//        family "rotate_half" convention: split-half PAIRING (x[j],
//        x[half+j]) for j in [0,half), rotated and written back to the SAME
//        positions (no permutation, unlike rope_interleave's deposit).
//        Verified verbatim against HF transformers
//        `src/transformers/models/llama/modeling_llama.py`
//        (raw.githubusercontent.com/huggingface/transformers/main/...,
//        fetched 2026-07-15):
//          rotate_half(x): x1,x2 = x[:half], x[half:]; return cat(-x2, x1)
//          apply_rotary_pos_emb: q_embed = q*cos + rotate_half(q)*sin
//        Expanding elementwise for pair index j (with cos[j]==cos[half+j],
//        sin[j]==sin[half+j] -- the standard "freqs repeated twice" rotary
//        embedding construction both Llama and MoeMath.h's own frequency
//        formula share):
//          out[j]      = x[j]*cos(angle_j)      - x[half+j]*sin(angle_j)
//          out[half+j] = x[half+j]*cos(angle_j) + x[j]*sin(angle_j)
//        This is the SAME 2D-rotation arithmetic as rope_interleave -- only
//        the pairing/deposit layout differs -- so `rope_half_split` needs no
//        scratch buffer (its pairing and deposit positions coincide: j and
//        half+j are both read before either is overwritten).
//
// 3. sample_topk_topp_temp is NET-NEW capability, not a port: this codebase
//    is confirmed greedy-only everywhere that exists today (boltllm::Engine,
//    per its own file-header banner and BLLM-8's Engine facade work --
//    "Decoding is GREEDY ONLY... GenerateParams therefore has no sampling
//    knobs"). Bolt's own `bolt::kernels::bolt_topk.h` (bolt_topk.h) is a
//    DIFFERENT primitive with a different contract -- a compile-time-K
//    smallest-K-distance max-heap for nearest-neighbor search (KNN), not a
//    runtime-sized largest-K-of-N selection over logits with nucleus
//    filtering and a categorical draw -- so it is not reused here (a fixed
//    template K cannot express an LLM's runtime top_k parameter, and "keep
//    the K smallest distances" is the opposite selection direction from
//    "keep the K largest logits"). The implementation below is the standard
//    top-k -> temperature -> renormalize -> top-p(nucleus) -> categorical-
//    sample pipeline (the same algorithm shape as llama.cpp's/HF
//    `generate()`'s sampler chain), built fresh and covered by
//    tests/test_bolt_compute.cpp.
// ─────────────────────────────────────────────────────────────────────────

#pragma once

#include "bolt/bolt_tensor.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace bolt::compute {

// ---------------------------------------------------------------------------
// sample_topk_topp_temp config.
// ---------------------------------------------------------------------------
struct SampleParams {
    float   temperature = 1.0f;  // must be > 0. 1.0 == no rescaling.
    int32_t top_k       = 0;     // 0 (or >= n) disables top-k filtering.
    float   top_p       = 1.0f;  // in (0, 1]. 1.0 disables nucleus filtering.
};

// ---------------------------------------------------------------------------
// Reserved-device trap. Fires unconditionally (NOT compiled out under
// NDEBUG, unlike assert()) -- calling a Vulkan/CUDA entry point before that
// backend exists is a caller bug (should have checked device support first),
// never something to silently no-op.
// ---------------------------------------------------------------------------
[[noreturn]] inline void device_not_implemented(const char* op, Device device) noexcept {
    const char* device_name = (device == Device::Vulkan) ? "Vulkan"
                            : (device == Device::CUDA)   ? "CUDA"
                                                          : "<unrecognized>";
    std::fprintf(stderr,
                 "bolt::compute: '%s' has no implementation for Device::%s "
                 "(reserved slot -- G-stage work, tracker BLLM-13). This is a "
                 "caller bug: check device support before dispatching.\n",
                 op, device_name);
    assert(false && "bolt::compute: reserved/unimplemented device slot called");
    std::abort();
}

// =============================================================================
// matmul -- GEMV: out[r] = dot(weight.row(r), activation), r in [0, out_rows).
// =============================================================================

namespace detail {

// Sub-byte packed-format dequant offsets. Restated (not `#include`d) from
// boltllm::quant::QuantTypes.h's kInt4Offset/kInt2Offset -- Bolt must not
// depend on boltllm; see bolt_tensor.h's field-mapping banner for why these
// values must stay in lock-step with QuantTypes.h if either ever changes.
inline constexpr int32_t kInt4DequantOffset = 8;
inline constexpr int32_t kInt2DequantOffset = 2;

// One row's dot product against a plain float activation vector, dequanting
// on the fly per bolt::Tensor's dtype (see architectural choice #1 above).
inline float matmul_row_dot_f32(const Tensor& weight, int32_t r, const float* activation) noexcept {
    assert(activation != nullptr);
    assert(r >= 0 && r < weight.rows());
    const uint8_t* row = weight.row_ptr(r);
    const int32_t cols = weight.cols();
    switch (weight.dtype) {
        case DType::F32: {
            const float* f = reinterpret_cast<const float*>(row);
            float acc = 0.0f;
            for (int32_t i = 0; i < cols; ++i) acc += activation[i] * f[i];
            return acc;
        }
        case DType::Int8: {
            const int8_t* q = reinterpret_cast<const int8_t*>(row);
            float acc = 0.0f;
            for (int32_t i = 0; i < cols; ++i) acc += activation[i] * static_cast<float>(q[i]);
            return acc * weight.scale(r);
        }
        case DType::Int4: {
            float acc = 0.0f;
            int32_t i = 0;
            for (; i + 1 < cols; i += 2) {
                const uint8_t byte = row[i >> 1];
                const int32_t lo = static_cast<int32_t>(byte & 0x0F) - kInt4DequantOffset;
                const int32_t hi = static_cast<int32_t>(byte >> 4) - kInt4DequantOffset;
                acc += activation[i] * static_cast<float>(lo) + activation[i + 1] * static_cast<float>(hi);
            }
            if (i < cols) {
                const uint8_t byte = row[i >> 1];
                const int32_t lo = static_cast<int32_t>(byte & 0x0F) - kInt4DequantOffset;
                acc += activation[i] * static_cast<float>(lo);
            }
            return acc * weight.scale(r);
        }
        case DType::Int2: {
            float acc = 0.0f;
            for (int32_t i = 0; i < cols; ++i) {
                const uint8_t byte = row[i >> 2];
                const int32_t sh = (i & 3) * 2;
                const int32_t v = static_cast<int32_t>((byte >> sh) & 0x03) - kInt2DequantOffset;
                acc += activation[i] * static_cast<float>(v);
            }
            return acc * weight.scale(r);
        }
        case DType::F16:
        case DType::BF16:
            // Tensor reserves these dtypes (bolt_tensor.h) but nothing
            // constructs them yet; assert loudly rather than silently
            // reinterpreting bytes as if they were something else.
            assert(false && "bolt::compute::matmul: F16/BF16 CPU path not implemented yet");
            return 0.0f;
    }
    assert(false && "bolt::compute::matmul: unrecognized DType");
    return 0.0f;
}

inline void matmul_cpu_scalar(const Tensor& weight, const float* activation, float* out,
                               int32_t out_rows) noexcept {
    assert(activation != nullptr && out != nullptr);
    assert(weight.device == Device::CPU);
    assert(weight.cols() > 0);
    assert(out_rows > 0 && out_rows <= weight.rows());
    for (int32_t r = 0; r < out_rows; ++r) {
        out[r] = matmul_row_dot_f32(weight, r, activation);
    }
}

[[noreturn]] inline void matmul_vulkan_stub(const Tensor&, const float*, float*, int32_t) noexcept {
    device_not_implemented("matmul", Device::Vulkan);
}
[[noreturn]] inline void matmul_cuda_stub(const Tensor&, const float*, float*, int32_t) noexcept {
    device_not_implemented("matmul", Device::CUDA);
}

}  // namespace detail

// =============================================================================
// rmsnorm -- y = x / sqrt(mean(x^2) + eps) * w. Direct port of
// boltllm::moe::MoeMath.h::rmsnorm (same f64 accumulator; see that file's
// header-comment numerics note for why).
// =============================================================================

namespace detail {

inline void rmsnorm_cpu_scalar(const float* x, const float* w, float* y, int32_t n, float eps) noexcept {
    assert(x != nullptr && w != nullptr && y != nullptr);
    assert(n > 0);
    assert(eps > 0.0f);

    double sum_sq = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        const double xi = static_cast<double>(x[i]);
        sum_sq += xi * xi;
    }
    const double mean_sq = sum_sq / static_cast<double>(n);
    const double inv_rms = 1.0 / std::sqrt(mean_sq + static_cast<double>(eps));
    assert(std::isfinite(inv_rms) && inv_rms > 0.0);

    const float scale = static_cast<float>(inv_rms);
    for (int32_t i = 0; i < n; ++i) y[i] = x[i] * scale * w[i];
}

[[noreturn]] inline void rmsnorm_vulkan_stub(const float*, const float*, float*, int32_t, float) noexcept {
    device_not_implemented("rmsnorm", Device::Vulkan);
}
[[noreturn]] inline void rmsnorm_cuda_stub(const float*, const float*, float*, int32_t, float) noexcept {
    device_not_implemented("rmsnorm", Device::CUDA);
}

}  // namespace detail

// =============================================================================
// rope -- two conventions. See architectural choice #2 above for the full
// derivation/citations.
// =============================================================================

namespace detail {

// Interleaved convention (GLM-5.2/DeepSeek-V3 family): pairing (x[2j],
// x[2j+1]); rotated pair deposited at (x[j], x[half+j]). Direct port of
// boltllm::moe::MoeMath.h::rope_interleave.
inline void rope_interleave_cpu_scalar(float* x, int32_t head_dim_rope, int32_t position,
                                        float theta) noexcept {
    assert(x != nullptr);
    assert(head_dim_rope > 0 && head_dim_rope <= 256);
    assert(head_dim_rope % 2 == 0);
    assert(position >= 0);
    assert(theta > 0.0f);

    const int32_t half = head_dim_rope / 2;

    // Local copy: the deposit layout (j, half+j) overlaps the pairing layout
    // (2j, 2j+1) for j < half, so in-place read/write without a copy would
    // clobber not-yet-read input (same rationale as MoeMath.h's own copy).
    float in[256];
    for (int32_t i = 0; i < head_dim_rope; ++i) in[i] = x[i];

    for (int32_t j = 0; j < half; ++j) {
        const float exponent = -2.0f * static_cast<float>(j) / static_cast<float>(head_dim_rope);
        const float inv_freq = std::pow(theta, exponent);
        const float angle = static_cast<float>(position) * inv_freq;
        const float cs = std::cos(angle);
        const float sn = std::sin(angle);

        const float a = in[2 * j];
        const float b = in[2 * j + 1];
        x[j] = a * cs - b * sn;
        x[half + j] = b * cs + a * sn;
    }
}

// Half-split ("rotate_half"/GPT-NeoX/Llama-family) convention: pairing
// (x[j], x[half+j]) for j in [0, half); rotated pair written back to the
// SAME positions (no permutation -- see architectural choice #2 above for
// the HF transformers `rotate_half`/`apply_rotary_pos_emb` derivation this
// implements). No scratch buffer needed: j and half+j are both read before
// either is overwritten.
inline void rope_half_split_cpu_scalar(float* x, int32_t head_dim_rope, int32_t position,
                                        float theta) noexcept {
    assert(x != nullptr);
    assert(head_dim_rope > 0 && head_dim_rope <= 256);
    assert(head_dim_rope % 2 == 0);
    assert(position >= 0);
    assert(theta > 0.0f);

    const int32_t half = head_dim_rope / 2;

    for (int32_t j = 0; j < half; ++j) {
        const float exponent = -2.0f * static_cast<float>(j) / static_cast<float>(head_dim_rope);
        const float inv_freq = std::pow(theta, exponent);
        const float angle = static_cast<float>(position) * inv_freq;
        const float cs = std::cos(angle);
        const float sn = std::sin(angle);

        const float a = x[j];
        const float b = x[half + j];
        x[j] = a * cs - b * sn;
        x[half + j] = b * cs + a * sn;
    }
}

[[noreturn]] inline void rope_vulkan_stub(float*, int32_t, int32_t, float) noexcept {
    device_not_implemented("rope", Device::Vulkan);
}
[[noreturn]] inline void rope_cuda_stub(float*, int32_t, int32_t, float) noexcept {
    device_not_implemented("rope", Device::CUDA);
}

}  // namespace detail

// =============================================================================
// softmax_inplace -- max-subtracted, sums to 1. Direct port of
// boltllm::moe::MoeMath.h::softmax_inplace.
// =============================================================================

namespace detail {

inline void softmax_inplace_cpu_scalar(float* x, int32_t n) noexcept {
    assert(x != nullptr);
    assert(n > 0);

    float max_val = x[0];
    for (int32_t i = 1; i < n; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }

    double sum = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        const float e = std::exp(x[i] - max_val);
        x[i] = e;
        sum += static_cast<double>(e);
    }
    assert(sum > 0.0);

    const float inv_sum = static_cast<float>(1.0 / sum);
    for (int32_t i = 0; i < n; ++i) x[i] *= inv_sum;
}

[[noreturn]] inline void softmax_vulkan_stub(float*, int32_t) noexcept {
    device_not_implemented("softmax", Device::Vulkan);
}
[[noreturn]] inline void softmax_cuda_stub(float*, int32_t) noexcept {
    device_not_implemented("softmax", Device::CUDA);
}

}  // namespace detail

// =============================================================================
// swiglu_row -- out[i] = silu(gate[i]) * up[i]. Direct port of
// boltllm::moe::MoeMath.h::swiglu_row (+ its sigmoid/silu helpers).
// =============================================================================

namespace detail {

inline float sigmoid_scalar(float x) noexcept {
    assert(!std::isnan(x));
    return (x >= 0.0f) ? 1.0f / (1.0f + std::exp(-x)) : std::exp(x) / (1.0f + std::exp(x));
}

inline float silu_scalar(float x) noexcept {
    assert(!std::isnan(x));
    return x * sigmoid_scalar(x);
}

inline void swiglu_row_cpu_scalar(const float* gate, const float* up, float* out, int32_t n) noexcept {
    assert(gate != nullptr && up != nullptr && out != nullptr);
    assert(n > 0);
    for (int32_t i = 0; i < n; ++i) out[i] = silu_scalar(gate[i]) * up[i];
}

[[noreturn]] inline void swiglu_vulkan_stub(const float*, const float*, float*, int32_t) noexcept {
    device_not_implemented("swiglu", Device::Vulkan);
}
[[noreturn]] inline void swiglu_cuda_stub(const float*, const float*, float*, int32_t) noexcept {
    device_not_implemented("swiglu", Device::CUDA);
}

}  // namespace detail

// =============================================================================
// sample_topk_topp_temp -- NET-NEW (see architectural choice #3 above).
// Standard top-k -> temperature -> renormalize -> top-p(nucleus) ->
// categorical-sample pipeline over `logits[0, n)`. Returns the sampled
// index in [0, n). `uniform_draw` is a caller-supplied draw from U[0, 1)
// (this function is deliberately RNG-agnostic: it consumes one uniform
// sample rather than owning/seeding a generator, so callers can use
// whatever RNG/seeding policy the engine needs -- reproducibility,
// per-request seeds, etc. -- without this dispatch layer caring).
// =============================================================================

namespace detail {

// Builds a descending-by-logit-value index permutation of [0, n). Ties
// broken by lowest index first (this codebase's established tie-break
// convention -- see boltllm::moe::MoeRouter.h's route(), fact 5).
inline void sample_sort_indices_desc(const float* logits, int32_t n,
                                      std::vector<int32_t>& indices) noexcept {
    indices.resize(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i) indices[static_cast<size_t>(i)] = i;
    std::sort(indices.begin(), indices.end(), [logits](int32_t a, int32_t b) {
        if (logits[a] != logits[b]) return logits[a] > logits[b];
        return a < b;
    });
}

inline int32_t sample_topk_topp_temp_cpu(const float* logits, int32_t n, const SampleParams& params,
                                          float uniform_draw) noexcept {
    assert(logits != nullptr);
    assert(n > 0);
    assert(params.temperature > 0.0f);
    assert(params.top_p > 0.0f && params.top_p <= 1.0f);
    assert(uniform_draw >= 0.0f && uniform_draw < 1.0f);

    // 1) Rank all candidates descending; top-k keeps a prefix of this order
    //    (0 or >= n means "no limit").
    std::vector<int32_t> indices;
    sample_sort_indices_desc(logits, n, indices);
    int32_t keep_k = n;
    if (params.top_k > 0 && params.top_k < n) keep_k = params.top_k;

    // 2) Temperature-scale the retained logits, then a numerically stable
    //    softmax over exactly that retained set (NOT over all n -- matches
    //    the standard "filter first, renormalize over the filtered set"
    //    top-k/top-p sampler shape).
    std::vector<float> probs(static_cast<size_t>(keep_k));
    float max_scaled = -std::numeric_limits<float>::infinity();
    for (int32_t i = 0; i < keep_k; ++i) {
        const float scaled = logits[static_cast<size_t>(indices[static_cast<size_t>(i)])] / params.temperature;
        probs[static_cast<size_t>(i)] = scaled;
        if (scaled > max_scaled) max_scaled = scaled;
    }
    double sum = 0.0;
    for (int32_t i = 0; i < keep_k; ++i) {
        const double e = std::exp(static_cast<double>(probs[static_cast<size_t>(i)] - max_scaled));
        probs[static_cast<size_t>(i)] = static_cast<float>(e);
        sum += e;
    }
    assert(sum > 0.0);
    for (int32_t i = 0; i < keep_k; ++i) {
        probs[static_cast<size_t>(i)] = static_cast<float>(static_cast<double>(probs[static_cast<size_t>(i)]) / sum);
    }

    // 3) Nucleus (top-p) truncation: keep the smallest prefix (already
    //    sorted descending) whose cumulative probability >= top_p.
    int32_t nucleus_count = keep_k;
    if (params.top_p < 1.0f) {
        double cum = 0.0;
        for (int32_t i = 0; i < keep_k; ++i) {
            cum += static_cast<double>(probs[static_cast<size_t>(i)]);
            if (cum >= static_cast<double>(params.top_p)) {
                nucleus_count = i + 1;
                break;
            }
        }
    }

    // 4) Categorical sample via inverse-CDF over the (renormalized) nucleus.
    double nucleus_sum = 0.0;
    for (int32_t i = 0; i < nucleus_count; ++i) nucleus_sum += static_cast<double>(probs[static_cast<size_t>(i)]);
    assert(nucleus_sum > 0.0);

    const double target = static_cast<double>(uniform_draw) * nucleus_sum;
    double running = 0.0;
    for (int32_t i = 0; i < nucleus_count; ++i) {
        running += static_cast<double>(probs[static_cast<size_t>(i)]);
        if (running >= target || i == nucleus_count - 1) {
            return indices[static_cast<size_t>(i)];
        }
    }
    assert(false && "sample_topk_topp_temp_cpu: unreachable");
    return indices[0];
}

[[noreturn]] inline int32_t sample_vulkan_stub(const float*, int32_t, const SampleParams&, float) noexcept {
    device_not_implemented("sample_topk_topp_temp", Device::Vulkan);
}
[[noreturn]] inline int32_t sample_cuda_stub(const float*, int32_t, const SampleParams&, float) noexcept {
    device_not_implemented("sample_topk_topp_temp", Device::CUDA);
}

}  // namespace detail

// =============================================================================
// Dispatch table -- one function pointer per op, resolved once per Device
// (function-local `static` == C++11 magic static, thread-safe one-time
// init). Mirrors boltllm::quant::IdotDispatch's DispatchTable shape.
// =============================================================================

using MatmulFn  = void (*)(const Tensor&, const float*, float*, int32_t) noexcept;
using RmsNormFn = void (*)(const float*, const float*, float*, int32_t, float) noexcept;
using RopeFn    = void (*)(float*, int32_t, int32_t, float) noexcept;
using SoftmaxFn = void (*)(float*, int32_t) noexcept;
using SwigluFn  = void (*)(const float*, const float*, float*, int32_t) noexcept;
using SampleFn  = int32_t (*)(const float*, int32_t, const SampleParams&, float) noexcept;

struct ComputeTable {
    MatmulFn matmul = nullptr;
    RmsNormFn rmsnorm = nullptr;
    RopeFn rope_interleave = nullptr;
    RopeFn rope_half_split = nullptr;
    SoftmaxFn softmax_inplace = nullptr;
    SwigluFn swiglu_row = nullptr;
    SampleFn sample_topk_topp_temp = nullptr;
};

// Returns the resolved table for `device`. CPU's tier is scalar-only today
// (see architectural choice #1); Vulkan/CUDA resolve to hard-failing stubs,
// never a silent CPU fallback.
inline const ComputeTable& table_for(Device device) noexcept {
    static const ComputeTable cpu_table = [] {
        ComputeTable t;
        t.matmul = &detail::matmul_cpu_scalar;
        t.rmsnorm = &detail::rmsnorm_cpu_scalar;
        t.rope_interleave = &detail::rope_interleave_cpu_scalar;
        t.rope_half_split = &detail::rope_half_split_cpu_scalar;
        t.softmax_inplace = &detail::softmax_inplace_cpu_scalar;
        t.swiglu_row = &detail::swiglu_row_cpu_scalar;
        t.sample_topk_topp_temp = &detail::sample_topk_topp_temp_cpu;
        return t;
    }();
    static const ComputeTable vulkan_table = [] {
        ComputeTable t;
        t.matmul = &detail::matmul_vulkan_stub;
        t.rmsnorm = &detail::rmsnorm_vulkan_stub;
        t.rope_interleave = &detail::rope_vulkan_stub;
        t.rope_half_split = &detail::rope_vulkan_stub;
        t.softmax_inplace = &detail::softmax_vulkan_stub;
        t.swiglu_row = &detail::swiglu_vulkan_stub;
        t.sample_topk_topp_temp = &detail::sample_vulkan_stub;
        return t;
    }();
    static const ComputeTable cuda_table = [] {
        ComputeTable t;
        t.matmul = &detail::matmul_cuda_stub;
        t.rmsnorm = &detail::rmsnorm_cuda_stub;
        t.rope_interleave = &detail::rope_cuda_stub;
        t.rope_half_split = &detail::rope_cuda_stub;
        t.softmax_inplace = &detail::softmax_cuda_stub;
        t.swiglu_row = &detail::swiglu_cuda_stub;
        t.sample_topk_topp_temp = &detail::sample_cuda_stub;
        return t;
    }();

    switch (device) {
        case Device::CPU:    return cpu_table;
        case Device::Vulkan: return vulkan_table;
        case Device::CUDA:   return cuda_table;
    }
    device_not_implemented("table_for", device);  // unrecognized enum value.
}

// ---- Convenience wrappers (one indirect call each; no per-call branching) -

inline void matmul(Device device, const Tensor& weight, const float* activation, float* out,
                    int32_t out_rows) noexcept {
    table_for(device).matmul(weight, activation, out, out_rows);
}

inline void rmsnorm(Device device, const float* x, const float* w, float* y, int32_t n,
                     float eps) noexcept {
    table_for(device).rmsnorm(x, w, y, n, eps);
}

inline void rope_interleave(Device device, float* x, int32_t head_dim_rope, int32_t position,
                             float theta) noexcept {
    table_for(device).rope_interleave(x, head_dim_rope, position, theta);
}

inline void rope_half_split(Device device, float* x, int32_t head_dim_rope, int32_t position,
                             float theta) noexcept {
    table_for(device).rope_half_split(x, head_dim_rope, position, theta);
}

inline void softmax_inplace(Device device, float* x, int32_t n) noexcept {
    table_for(device).softmax_inplace(x, n);
}

inline void swiglu_row(Device device, const float* gate, const float* up, float* out,
                        int32_t n) noexcept {
    table_for(device).swiglu_row(gate, up, out, n);
}

inline int32_t sample_topk_topp_temp(Device device, const float* logits, int32_t n,
                                      const SampleParams& params, float uniform_draw) noexcept {
    return table_for(device).sample_topk_topp_temp(logits, n, params, uniform_draw);
}

}  // namespace bolt::compute
