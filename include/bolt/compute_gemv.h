// compute_gemv.h -- f32/f16-WEIGHT float-activation SIMD GEMV for bolt::compute
// (perf pass). The dense-family decode hot path: a matrix-vector product
// out[r] = dot(weight_row_r, x) where the weight rows are plain f32 (or f16)
// and the activation `x` is plain f32. This is the SIMD counterpart to
// bolt_compute.h's scalar `matmul_row_dot_f32` (its F32 case), and a DIFFERENT
// numerics tier from compute_idot.h (which dots an int8-quantized activation
// against an int8/int4 weight in the integer pipeline).
//
// Why this exists: LLM decode is memory-bandwidth-bound -- every weight is
// read once per token. The scalar f32 dot cannot even saturate DRAM (~5 GF/s
// measured); an AVX-512 FMA GEMV hits the CPU's real memory wall (~4x on real
// model shapes). This is the broad, low-risk win: every dense attention
// projection (Q/K/V/O), dense FFN, and lm_head across an LLM engine's model
// families routes through here.
//
// SHAPE: mirrors compute_idot.h precisely -- declarations + `*_available()`
// probes only, so this header is safe to #include ANYWHERE regardless of the
// consumer's compiled ISA tier. The intrinsics live entirely in the compiled
// bolt::compute_idot TUs (gemv_avx2.cpp / gemv_avx512.cpp); the CPUID probe +
// magic-static function-pointer table (gemv_dispatch.cpp) resolves the tier
// ONCE (AVX-512F > AVX2+FMA+F16C > scalar) -- amortized over a heavy GEMV, the
// same justified exception to bolt's "no runtime dispatch" rule that
// compute_idot.h documents.
//
// Numerics: the SIMD tiers use 4 independent f32 accumulators (hide FMA
// latency), tree-reduce, then a final f64 horizontal sum + f64 scalar tail.
// The scalar tier accumulates the whole row in f64. SIMD reorders the f32
// reduction, so it agrees with the scalar reference to ~1e-6 relative (NOT
// bit-exact, unlike the integer IDOT kernels) -- inside any 1e-3 oracle
// tolerance, preserving greedy token-exactness.

#pragma once

#include <cassert>
#include <cstdint>

namespace bolt::compute {

// out[r] = sum_i weight[r*cols + i] * x[i], for r in [0, rows). `weight` is a
// row-major [rows, cols] f32 (matmul_f32) or IEEE-754 binary16 (matmul_f16,
// bit container `uint16_t`) matrix; `x` is `cols` f32 activations; `out`
// receives `rows` f32. Blocking, noexcept, ZERO allocation. Runtime-dispatches
// to AVX-512F > AVX2+FMA+F16C > scalar via the CPUID probe.
// Precondition: weight/x/out non-null, rows > 0, cols > 0.
void matmul_f32(const float* weight, const float* x, float* out, int32_t rows,
                int32_t cols) noexcept;
void matmul_f16(const uint16_t* weight, const float* x, float* out, int32_t rows,
                int32_t cols) noexcept;

// True iff the CPUID probe selected the AVX2 / AVX-512F GEMV kernels on this
// machine. For tests that self-skip when the fast tier is unavailable, and for
// callers that want to know which tier is live. Safe to call repeatedly (the
// probe is a one-time magic-static). Always false on non-x86 builds.
bool gemv_avx2_available() noexcept;
bool gemv_avx512_available() noexcept;

}  // namespace bolt::compute
