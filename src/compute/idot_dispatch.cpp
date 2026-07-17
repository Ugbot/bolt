// idot_dispatch.cpp -- runtime CPUID probe + magic-static function-pointer
// table for bolt::compute's IDOT GEMV (BLLM-14). Compiled at BASELINE ISA
// (no /arch flag): this TU contains NO intrinsics of its own -- only a CPUID
// probe and function-pointer indirection -- so it is safe to run on any CPU,
// and it is the CPU-gate that ensures the AVX2/VNNI kernels are only ever
// CALLED on hardware that supports them.
//
// This is bolt's FIRST runtime CPUID dispatch (io/bolt_crc32c.h deliberately
// refuses to probe -- a 3ns primitive can't amortize a probe). It is a
// justified exception: the probe resolves ONCE via a C++11 magic static, and
// is amortized over a heavy int8 GEMV, not paid per-call. See
// docs/research/idot-runtime-dispatch.md and the design-log entry.
//
// Ported from boltllm::quant::IdotDispatch.cpp, with the MSVC-only CPUID
// probe EXTENDED with GCC/Clang x86 arms (__builtin_cpu_supports) and an
// ARM/non-x86 arm (-> false, scalar-only), so bolt's linux-gcc/clang CI lanes
// keep the fast path instead of silently degrading to scalar the way a
// verbatim MSVC-only port would.

#include "bolt/compute_idot.h"

#include "idot_kernels_internal.h"

#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>

#if BOLT_ARCH_X86 && defined(_MSC_VER)
#include <intrin.h>
#endif

namespace bolt::compute {

namespace {

#if BOLT_ARCH_X86

// AVX2 requires: OSXSAVE + AVX (CPUID.1:ECX), XCR0 bits 1|2 (OS saves
// XMM/YMM), and AVX2 (CPUID.7.0:EBX bit 5). See boltllm's IdotDispatch.cpp
// for the per-bit rationale.
bool detect_avx2() noexcept {
#if defined(_MSC_VER)
    int info1[4] = {0, 0, 0, 0};
    __cpuid(info1, 1);
    const bool osxsave = (info1[2] & (1 << 27)) != 0;
    const bool avx = (info1[2] & (1 << 28)) != 0;
    if (!osxsave || !avx) return false;
    const unsigned long long xcr0 = _xgetbv(0);
    if ((xcr0 & 0x6ULL) != 0x6ULL) return false;  // XMM (1) + YMM (2)
    int info7[4] = {0, 0, 0, 0};
    __cpuidex(info7, 7, 0);
    return (info7[1] & (1 << 5)) != 0;  // AVX2
#else
    // GCC/Clang: __builtin_cpu_supports checks the CPU feature bits (and, on
    // modern libgcc, is safe against absent OS state for these features). The
    // full XGETBV dance the MSVC arm does is not exposed as portably here;
    // this is the idiomatic GCC/Clang runtime-feature check and matches what
    // the ISA multiversioning machinery uses internally.
    return __builtin_cpu_supports("avx2");
#endif
}

// AVX-512VNNI requires F + BW + VNNI together (bundled -- every VNNI CPU
// shipped to date also has BW), plus the wider XCR0 mask (bits 1|2|5|6|7).
bool detect_avx512vnni() noexcept {
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
    const bool avx512f = (info7[1] & (1 << 16)) != 0;
    const bool avx512bw = (info7[1] & (1 << 30)) != 0;
    const bool avx512vnni = (info7[2] & (1 << 11)) != 0;
    return avx512f && avx512bw && avx512vnni;
#else
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512vnni");
#endif
}

#endif  // BOLT_ARCH_X86

using IdotFn = void (*)(const int8_t*, float, const Tensor&, float*, int32_t) noexcept;

struct DispatchTable {
    bool   avx2 = false;
    bool   avx512vnni = false;
    IdotFn i8 = nullptr;
    IdotFn i4 = nullptr;

    DispatchTable() noexcept {
#if BOLT_ARCH_X86
        avx2 = detect_avx2();
        avx512vnni = detect_avx512vnni();
        if (avx512vnni) {
            i8 = &detail::idot_i8a_int8_avx512vnni;
            i4 = &detail::idot_i8a_int4_avx512vnni;
        } else if (avx2) {
            i8 = &detail::idot_i8a_int8_avx2;
            i4 = &detail::idot_i8a_int4_avx2;
        } else {
            i8 = &detail::idot_i8a_int8_scalar;
            i4 = &detail::idot_i8a_int4_scalar;
        }
#else
        // Non-x86 (e.g. macos-arm): scalar only. The _avx2/_avx512vnni
        // symbols are never referenced here, so those TUs are empty and
        // nothing links against instructions this CPU lacks.
        i8 = &detail::idot_i8a_int8_scalar;
        i4 = &detail::idot_i8a_int4_scalar;
#endif
    }
};

const DispatchTable& table() noexcept {
    static const DispatchTable t;  // C++11 magic static: thread-safe one-time init
    return t;
}

}  // namespace

bool idot_avx2_available() noexcept { return table().avx2; }

bool idot_avx512vnni_available() noexcept { return table().avx512vnni; }

void matmul_i8a_int8(const int8_t* act_q, float act_scale, const Tensor& weight, float* out,
                     int32_t rows) noexcept {
    assert(act_q != nullptr && out != nullptr);
    table().i8(act_q, act_scale, weight, out, rows);
}

void matmul_i8a_int4(const int8_t* act_q, float act_scale, const Tensor& weight, float* out,
                     int32_t rows) noexcept {
    assert(act_q != nullptr && out != nullptr);
    table().i4(act_q, act_scale, weight, out, rows);
}

}  // namespace bolt::compute
