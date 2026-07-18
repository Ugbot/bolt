// gemv_dispatch.cpp -- runtime CPUID probe + magic-static function-pointer
// table for bolt::compute's f32/f16 float-activation GEMV (perf pass).
// Compiled at BASELINE ISA (no /arch flag): this TU contains NO intrinsics of
// its own -- only a CPUID probe and function-pointer indirection -- so it is
// safe to run on any CPU, and it is the CPU-gate that ensures the AVX2/AVX-512
// kernels are only ever CALLED on hardware that supports them.
//
// Same justified exception to bolt's "no runtime dispatch" rule that
// idot_dispatch.cpp documents: the probe resolves ONCE via a C++11 magic
// static and is amortized over a heavy GEMV, not paid per call. The hot path
// is one indirect call through a resolved function pointer, no per-call
// branching. Structure mirrors idot_dispatch.cpp exactly.

#include "bolt/compute_gemv.h"

#include "gemv_kernels_internal.h"

#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>

#if BOLT_ARCH_X86 && defined(_MSC_VER)
#include <intrin.h>
#endif

namespace bolt::compute {

namespace {

#if BOLT_ARCH_X86

// The f32/f16 SIMD GEMV needs AVX2 + FMA (the fmadd) + F16C (the f16 upcast in
// the AVX2 tier). All three shipped together on Haswell (2013) and every AVX2
// CPU since, but we probe each bit so a partial-feature CPU degrades safely.
bool detect_avx2_fma_f16c() noexcept {
#if defined(_MSC_VER)
    int info1[4] = {0, 0, 0, 0};
    __cpuid(info1, 1);
    const bool osxsave = (info1[2] & (1 << 27)) != 0;
    const bool avx = (info1[2] & (1 << 28)) != 0;
    const bool fma = (info1[2] & (1 << 12)) != 0;
    const bool f16c = (info1[2] & (1 << 29)) != 0;
    if (!osxsave || !avx || !fma || !f16c) return false;
    const unsigned long long xcr0 = _xgetbv(0);
    if ((xcr0 & 0x6ULL) != 0x6ULL) return false;  // XMM (1) + YMM (2)
    int info7[4] = {0, 0, 0, 0};
    __cpuidex(info7, 7, 0);
    return (info7[1] & (1 << 5)) != 0;  // AVX2
#else
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
           __builtin_cpu_supports("f16c");
#endif
}

// AVX-512F alone suffices for this tier (native fmadd + cvtph). Requires the
// wider XCR0 mask (bits 1|2|5|6|7) for OS ZMM/opmask save-restore.
bool detect_avx512f() noexcept {
#if defined(_MSC_VER)
    int info1[4] = {0, 0, 0, 0};
    __cpuid(info1, 1);
    const bool osxsave = (info1[2] & (1 << 27)) != 0;
    const bool avx = (info1[2] & (1 << 28)) != 0;
    if (!osxsave || !avx) return false;
    const unsigned long long xcr0 = _xgetbv(0);
    const unsigned long long kAvx512StateMask = 0xE6ULL;  // XMM|YMM|opmask|ZMM_Hi256|Hi16_ZMM
    if ((xcr0 & kAvx512StateMask) != kAvx512StateMask) return false;
    int info7[4] = {0, 0, 0, 0};
    __cpuidex(info7, 7, 0);
    return (info7[1] & (1 << 16)) != 0;  // AVX512F
#else
    return __builtin_cpu_supports("avx512f");
#endif
}

#endif  // BOLT_ARCH_X86

using GemvF32Fn = void (*)(const float*, const float*, float*, int32_t, int32_t) noexcept;
using GemvF16Fn = void (*)(const uint16_t*, const float*, float*, int32_t, int32_t) noexcept;

struct DispatchTable {
    bool avx2 = false;
    bool avx512 = false;
    GemvF32Fn f32 = nullptr;
    GemvF16Fn f16 = nullptr;

    DispatchTable() noexcept {
#if BOLT_ARCH_X86
        avx2 = detect_avx2_fma_f16c();
        avx512 = detect_avx512f();
        if (avx512) {
            f32 = &detail::gemv_f32_avx512;
            f16 = &detail::gemv_f16_avx512;
        } else if (avx2) {
            f32 = &detail::gemv_f32_avx2;
            f16 = &detail::gemv_f16_avx2;
        } else {
            f32 = &detail::gemv_f32_scalar;
            f16 = &detail::gemv_f16_scalar;
        }
#else
        // Non-x86: scalar only. The _avx2/_avx512 symbols are never referenced
        // here, so those TUs are empty and nothing links against instructions
        // this CPU lacks.
        f32 = &detail::gemv_f32_scalar;
        f16 = &detail::gemv_f16_scalar;
#endif
    }
};

const DispatchTable& table() noexcept {
    static const DispatchTable t;  // C++11 magic static: thread-safe one-time init
    return t;
}

}  // namespace

bool gemv_avx2_available() noexcept { return table().avx2; }

bool gemv_avx512_available() noexcept { return table().avx512; }

void matmul_f32(const float* weight, const float* x, float* out, int32_t rows,
                int32_t cols) noexcept {
    assert(weight != nullptr && x != nullptr && out != nullptr);
    assert(rows > 0 && cols > 0);
    table().f32(weight, x, out, rows, cols);
}

void matmul_f16(const uint16_t* weight, const float* x, float* out, int32_t rows,
                int32_t cols) noexcept {
    assert(weight != nullptr && x != nullptr && out != nullptr);
    assert(rows > 0 && cols > 0);
    table().f16(weight, x, out, rows, cols);
}

}  // namespace bolt::compute
