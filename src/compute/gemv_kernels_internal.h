// gemv_kernels_internal.h -- INTERNAL declarations of bolt::compute's f32/f16
// float-activation GEMV kernel tiers (perf pass). Not installed, not a public
// header: the intrinsics live in the .cpp files, and only gemv_dispatch.cpp
// references these symbols (inside its #if BOLT_ARCH_X86 block, so the
// _avx2/_avx512 symbols are never referenced -- and never needed -- on non-x86
// builds).
//
// Every kernel has the same signature: (weight, x, out, rows, cols). The
// _scalar tier exists on ALL architectures; the _avx2/_avx512 tiers are
// defined only when BOLT_ARCH_X86 (empty TUs otherwise). Mirrors
// idot_kernels_internal.h exactly.

#pragma once

#include "bolt/bolt_port.h"

#include <cstdint>

namespace bolt::compute::detail {

// Scalar tier -- always available (gemv_scalar.cpp, compiled at baseline ISA
// so it is a SAFE runtime fallback on any CPU). Accumulates each row in f64.
void gemv_f32_scalar(const float* weight, const float* x, float* out, int32_t rows,
                     int32_t cols) noexcept;
void gemv_f16_scalar(const uint16_t* weight, const float* x, float* out, int32_t rows,
                     int32_t cols) noexcept;

#if BOLT_ARCH_X86
// AVX2 tier (gemv_avx2.cpp, compiled at /arch:AVX2 or -mavx2 -mfma -mf16c).
// f32: 4x __m256 FMA accumulators; f16: same over _mm256_cvtph_ps upcasts.
void gemv_f32_avx2(const float* weight, const float* x, float* out, int32_t rows,
                   int32_t cols) noexcept;
void gemv_f16_avx2(const uint16_t* weight, const float* x, float* out, int32_t rows,
                   int32_t cols) noexcept;

// AVX-512F tier (gemv_avx512.cpp, compiled at /arch:AVX512 or
// -mavx512f). f32: 4x __m512 FMA accumulators; f16: same over _mm512_cvtph_ps
// (native to AVX-512F, no separate F16C needed).
void gemv_f32_avx512(const float* weight, const float* x, float* out, int32_t rows,
                     int32_t cols) noexcept;
void gemv_f16_avx512(const uint16_t* weight, const float* x, float* out, int32_t rows,
                     int32_t cols) noexcept;
#endif  // BOLT_ARCH_X86

}  // namespace bolt::compute::detail
