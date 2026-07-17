// idot_kernels_internal.h -- INTERNAL declarations of bolt::compute's IDOT
// GEMV kernel tiers (BLLM-14). Not installed, not a public header: the
// intrinsics live in the .cpp files, and only idot_dispatch.cpp references
// these symbols (inside its #if BOLT_ARCH_X86 block, so the _avx2/_avx512vnni
// symbols are never referenced -- and never needed -- on non-x86 builds).
//
// Every kernel has the same signature: (act_q, act_scale, weight, out, rows).
// The _scalar tier exists on ALL architectures; the _avx2/_avx512vnni tiers
// are defined only when BOLT_ARCH_X86 (empty TUs otherwise).

#pragma once

#include "bolt/bolt_tensor.h"

#include <cstdint>

namespace bolt::compute::detail {

// Scalar tier -- always available (idot_scalar.cpp, compiled at baseline ISA
// so it is a SAFE runtime fallback on any CPU, unlike a whole-target-AVX512
// build where even the fallback could emit AVX-512 and SIGILL on old CPUs).
void idot_i8a_int8_scalar(const int8_t* act_q, float act_scale, const Tensor& weight, float* out,
                          int32_t rows) noexcept;
void idot_i8a_int4_scalar(const int8_t* act_q, float act_scale, const Tensor& weight, float* out,
                          int32_t rows) noexcept;

#if BOLT_ARCH_X86
// AVX2 tier (idot_avx2.cpp, compiled at /arch:AVX2 or -mavx2 -mfma).
void idot_i8a_int8_avx2(const int8_t* act_q, float act_scale, const Tensor& weight, float* out,
                        int32_t rows) noexcept;
void idot_i8a_int4_avx2(const int8_t* act_q, float act_scale, const Tensor& weight, float* out,
                        int32_t rows) noexcept;

// AVX-512VNNI tier (idot_avx512vnni.cpp, compiled at /arch:AVX512 or
// -mavx2 -mfma -mavx512f -mavx512bw -mavx512vnni).
void idot_i8a_int8_avx512vnni(const int8_t* act_q, float act_scale, const Tensor& weight,
                              float* out, int32_t rows) noexcept;
void idot_i8a_int4_avx512vnni(const int8_t* act_q, float act_scale, const Tensor& weight,
                              float* out, int32_t rows) noexcept;
#endif  // BOLT_ARCH_X86

}  // namespace bolt::compute::detail
