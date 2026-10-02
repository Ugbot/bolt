// bolt_crc32c.h — CRC32C (Castagnoli, poly 0x1EDC6F41) hardware-accelerated.
//
// I/O-adjacent primitives that are pure compute (checksums, bit-twiddling,
// varint). No actual I/O happens inside `bolt::io::*` — keeps Bolt's
// no-IO invariant intact. MarbleDB uses these on the SSTable write path
// (block checksum) and read path (integrity verification) but all bytes
// are already in memory by the time we hit these functions.
//
// Three-path compile-time dispatch (no runtime branch):
//   - x86_64 with SSE4.2:  `_mm_crc32_u64` / `_mm_crc32_u32` / `_mm_crc32_u8`
//   - ARM64 with CRC ext:  `__crc32cd` / `__crc32cw` / `__crc32cb`
//   - Portable fallback:   Slicing-by-8 software table (in the .cpp).
//
// Detection uses `BOLT_SIMD_SSE42` / `BOLT_ARCH_X86` from bolt_port.h.
//
// SSE4.2 baseline policy (x86_64):
//   Bolt declares SSE4.2 a baseline requirement for `bolt::io` on x86_64
//   targets. Any x64 CPU shipping in 2009 or later (Nehalem and newer)
//   supports it; pre-Nehalem x64 hardware is NOT supported and will
//   SIGILL on the `_mm_crc32_u64` instructions. We do not runtime-probe
//   via `__cpuid`, because:
//     (a) runtime dispatch on a 3-ns-per-call primitive would dominate
//         the cost of the primitive itself;
//     (b) the set of affected machines is empty in practice for our
//         target audience (datacenter / developer workstation);
//     (c) gcc / clang users who build for older silicon compile without
//         `-msse4.2` and land on the software path automatically, which
//         is the correct fallback.
//   MSVC does not auto-define `__SSE4_2__` even though `_mm_crc32_u64`
//   is always available on x64 — we therefore key off `_M_X64` via
//   `BOLT_COMPILER_MSVC` directly.
//
// On ARM64 we key off `__ARM_FEATURE_CRC32` — clang / gcc set it when
// the target supports the CRC extension; MSVC ARM64 does not define it,
// so those builds go through the software table too.
//
// Tiger Style: noexcept, no exceptions, no RTTI, no heap, ≥2 asserts on
// public entry points. API is POD + free functions.
//
// API:
//   uint32_t crc32c(data, len, seed = 0)      — core; seed = 0 for fresh.
//                                                >= 3 lanes: crc32c_3way.
//   uint32_t crc32c_3way(data, len, seed)     — interleaved hardware path.
//   uint32_t crc32c_extend_zeros(crc, n)      — crc continued over n zeros.
//   uint32_t page_crc32c(page, len, padded)   — page CRC incl. zero padding.
//   uint32_t crc32c_update(crc, data, len)    — streaming continue; same
//                                                as crc32c(data, len, crc).
//
// The update variant is provided so call sites document their intent —
// "this is a continuation" vs "this is a fresh CRC" — without a state
// struct. No difference on the wire; CRC32C is associative in the usual
// "seed = prior result" sense.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/io/bolt_io_limits.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

// ---------------------------------------------------------------------------
// Path selection
// ---------------------------------------------------------------------------
//
// BOLT_CRC32C_HW_X86 — SSE4.2 CRC32 intrinsics are available (x86_64 build
//                       compiled with SSE4.2 support). Auto-set whenever
//                       BOLT_SIMD_SSE42 is on; also auto-set on MSVC x64
//                       (where _mm_crc32_u64 exists without any /arch flag
//                       because SSE4.2 is a baseline intrinsic on MSVC x64
//                       even though __SSE4_2__ is not auto-defined).
// BOLT_CRC32C_HW_ARM — __crc32cd / __crc32cw / __crc32cb intrinsics available.

// `_mm_crc32_u64` is x86_64-only. Gate on the 64-bit macros so a 32-bit
// x86 build falls through to the software path instead of failing to
// compile.
#if (defined(__x86_64__) || defined(_M_X64)) \
    && (BOLT_SIMD_SSE42 || BOLT_SIMD_AVX2 || BOLT_SIMD_AVX512 \
        || BOLT_COMPILER_MSVC)
    // MSVC x64: SSE4.2 intrinsics (incl. _mm_crc32_u64) are always available
    // without /arch:SSE4.2 — there is no such flag, and SSE4.2 is baseline
    // for x64 CRT. __SSE4_2__ is NOT auto-defined though, so we key off
    // _M_X64 directly via BOLT_COMPILER_MSVC here.
    #define BOLT_CRC32C_HW_X86 1
    #include <nmmintrin.h>
#else
    #define BOLT_CRC32C_HW_X86 0
#endif

#if defined(__ARM_FEATURE_CRC32) && BOLT_ARCH_ARM
    #define BOLT_CRC32C_HW_ARM 1
    #include <arm_acle.h>
#else
    #define BOLT_CRC32C_HW_ARM 0
#endif

namespace bolt {
namespace io {

// ---------------------------------------------------------------------------
// Software slicing-by-8 table (shared between fallback path and test
// oracle). Declared here; defined in bolt_crc32c.cpp so the 8 KB of
// tables link once per binary, not per consumer.
// ---------------------------------------------------------------------------

// Exposed so tests can force-exercise the software path on every host.
uint32_t crc32c_software(const void* BOLT_RESTRICT data,
                          size_t len,
                          uint32_t seed) noexcept;

// ---------------------------------------------------------------------------
// Hardware fast paths — inline, noexcept, no branches on the hot loop.
// ---------------------------------------------------------------------------

#if BOLT_CRC32C_HW_X86

BOLT_FORCE_INLINE uint32_t crc32c_hw_x86(const void* BOLT_RESTRICT data,
                                         size_t len,
                                         uint32_t seed) noexcept {
    assert(data != nullptr || len == 0);
    assert(len <= (static_cast<size_t>(1) << 48));  // sanity upper bound
    // Initial inversion: CRC32C uses ~seed as the running state, final
    // output is ~state. `_mm_crc32_u*` does not pre/post-invert; we do
    // it here so the API matches the `crc32c("123456789") == 0xE3069283`
    // reference vector.
    uint64_t crc = ~static_cast<uint64_t>(seed);
    const uint8_t* p = static_cast<const uint8_t*>(data);

    // Align to 8 bytes for the u64 inner loop (unaligned u64 loads work
    // on x86 but cost a few cycles on cache-line crossings; the u8
    // prefix keeps the main loop hot).
    while (len > 0 && (reinterpret_cast<uintptr_t>(p) & 7u) != 0u) {
        crc = _mm_crc32_u8(static_cast<uint32_t>(crc), *p);
        ++p;
        --len;
    }
    // Main 8-byte loop. memcpy → mov: strict-aliasing-clean, single insn.
    while (len >= 8u) {
        uint64_t v;
        std::memcpy(&v, p, sizeof(v));
        crc = _mm_crc32_u64(crc, v);
        p   += 8;
        len -= 8u;
    }
    if (len >= 4u) {
        uint32_t v;
        std::memcpy(&v, p, sizeof(v));
        crc = _mm_crc32_u32(static_cast<uint32_t>(crc), v);
        p   += 4;
        len -= 4u;
    }
    while (len > 0u) {
        crc = _mm_crc32_u8(static_cast<uint32_t>(crc), *p);
        ++p;
        --len;
    }
    return ~static_cast<uint32_t>(crc);
}

#endif  // BOLT_CRC32C_HW_X86

#if BOLT_CRC32C_HW_ARM

BOLT_FORCE_INLINE uint32_t crc32c_hw_arm(const void* BOLT_RESTRICT data,
                                         size_t len,
                                         uint32_t seed) noexcept {
    assert(data != nullptr || len == 0);
    assert(len <= (static_cast<size_t>(1) << 48));  // sanity upper bound
    uint32_t crc = ~seed;
    const uint8_t* p = static_cast<const uint8_t*>(data);

    while (len > 0 && (reinterpret_cast<uintptr_t>(p) & 7u) != 0u) {
        crc = __crc32cb(crc, *p);
        ++p;
        --len;
    }
    while (len >= 8u) {
        uint64_t v;
        std::memcpy(&v, p, sizeof(v));
        crc = __crc32cd(crc, v);
        p   += 8;
        len -= 8u;
    }
    if (len >= 4u) {
        uint32_t v;
        std::memcpy(&v, p, sizeof(v));
        crc = __crc32cw(crc, v);
        p   += 4;
        len -= 4u;
    }
    while (len > 0u) {
        crc = __crc32cb(crc, *p);
        ++p;
        --len;
    }
    return ~crc;
}

#endif  // BOLT_CRC32C_HW_ARM

// ---------------------------------------------------------------------------
// GF(2) arithmetic modulo the Castagnoli polynomial, reflected (bit 31 is the
// x^0 coefficient). A raw CRC state s followed by n zero bytes becomes
// s * x^(8n) mod P, and raw CRCs are linear, so the CRC of A||B is
// shift(crc(A), |B|) ^ crc_raw(B). Used to recombine interleaved lanes and to
// extend a CRC over zero padding without materialising it.
// ---------------------------------------------------------------------------

namespace detail {

inline constexpr uint32_t kCrc32cPolyReflected = 0x82F63B78u;

constexpr uint32_t crc32c_multmodp(uint32_t a, uint32_t b) noexcept {
    uint32_t p = 0;
    for (int i = 0; i < 32; ++i) {
        if (a & (0x80000000u >> i)) p ^= b;
        b = (b & 1u) ? ((b >> 1) ^ kCrc32cPolyReflected) : (b >> 1);
    }
    return p;
}

// x^(2^k) mod P for k = 0..63.
struct Crc32cPow2 { uint32_t v[64]; };
constexpr Crc32cPow2 crc32c_make_pow2() noexcept {
    Crc32cPow2 t{};
    uint32_t x = 0x40000000u;  // x^1
    for (int k = 0; k < 64; ++k) { t.v[k] = x; x = crc32c_multmodp(x, x); }
    return t;
}
inline constexpr Crc32cPow2 kCrc32cPow2 = crc32c_make_pow2();

// x^(8 * n) mod P.
constexpr uint32_t crc32c_x8n(uint64_t n) noexcept {
    uint32_t r = 0x80000000u;  // x^0
    uint64_t e = n;
    for (int k = 3; e != 0 && k < 64; ++k, e >>= 1)
        if (e & 1u) r = crc32c_multmodp(kCrc32cPow2.v[k], r);
    return r;
}

// Byte-sliced table of "shift by one lane": shift(s) = xor of T[k][byte k].
struct Crc32cShiftTable { uint32_t t[4][256]; };
constexpr Crc32cShiftTable crc32c_make_shift(uint64_t n_bytes) noexcept {
    Crc32cShiftTable s{};
    const uint32_t m = crc32c_x8n(n_bytes);
    for (int k = 0; k < 4; ++k)
        for (uint32_t b = 0; b < 256u; ++b) s.t[k][b] = crc32c_multmodp(m, b << (8 * k));
    return s;
}
inline constexpr Crc32cShiftTable kCrc32cLaneShift = crc32c_make_shift(kCrc32cLaneBytes);

BOLT_FORCE_INLINE uint32_t crc32c_lane_shift(uint32_t s) noexcept {
    return kCrc32cLaneShift.t[0][s & 0xFFu] ^ kCrc32cLaneShift.t[1][(s >> 8) & 0xFFu] ^
           kCrc32cLaneShift.t[2][(s >> 16) & 0xFFu] ^ kCrc32cLaneShift.t[3][s >> 24];
}

}  // namespace detail

/// CRC32C of `crc`'s stream followed by `n` zero bytes, in O(log n) without
/// touching memory. Equal to crc32c_update(crc, zeros, n).
inline uint32_t crc32c_extend_zeros(uint32_t crc, uint64_t n) noexcept {
    assert(n <= (uint64_t(1) << 60));
    const uint32_t raw = detail::crc32c_multmodp(detail::crc32c_x8n(n), ~crc);
    assert(n != 0 || raw == ~crc);
    return ~raw;
}

// ---------------------------------------------------------------------------
// 3-way interleaved hardware CRC: three independent lanes hide the crc
// instruction's latency; lanes are recombined with the shift table. Same
// result as the serial loop for every input.
// ---------------------------------------------------------------------------

#if BOLT_CRC32C_HW_X86 || BOLT_CRC32C_HW_ARM

namespace detail {
BOLT_FORCE_INLINE uint32_t crc32c_raw_u64(uint32_t s, const uint8_t* p) noexcept {
    uint64_t v;
    std::memcpy(&v, p, sizeof(v));
#if BOLT_CRC32C_HW_X86
    return static_cast<uint32_t>(_mm_crc32_u64(s, v));
#else
    return __crc32cd(s, v);
#endif
}
}  // namespace detail

inline uint32_t crc32c_3way(const void* BOLT_RESTRICT data, size_t len,
                            uint32_t seed = 0u) noexcept {
    assert(data != nullptr || len == 0);
    assert(len <= (static_cast<size_t>(1) << 48));
    constexpr size_t L = kCrc32cLaneBytes;
    const uint8_t* p = static_cast<const uint8_t*>(data);
    uint32_t s = ~seed;  // raw running state
    while (len >= 3 * L) {
        uint32_t a = s, b = 0, c = 0;
        for (size_t i = 0; i < L; i += 8) {
            a = detail::crc32c_raw_u64(a, p + i);
            b = detail::crc32c_raw_u64(b, p + L + i);
            c = detail::crc32c_raw_u64(c, p + 2 * L + i);
        }
        s = detail::crc32c_lane_shift(detail::crc32c_lane_shift(a) ^ b) ^ c;
        p += 3 * L;
        len -= 3 * L;
    }
#if BOLT_CRC32C_HW_X86
    return crc32c_hw_x86(p, len, ~s);
#else
    return crc32c_hw_arm(p, len, ~s);
#endif
}

#endif  // HW

// ---------------------------------------------------------------------------
// Public entry — picks the hardware path at compile time.
// ---------------------------------------------------------------------------

BOLT_FORCE_INLINE uint32_t crc32c(const void* BOLT_RESTRICT data,
                                  size_t len,
                                  uint32_t seed = 0u) noexcept {
    assert(data != nullptr || len == 0);
    assert(len <= (static_cast<size_t>(1) << 48));  // sanity upper bound
#if BOLT_CRC32C_HW_X86 || BOLT_CRC32C_HW_ARM
    if (len >= kCrc32c3WayMinBytes) return crc32c_3way(data, len, seed);
#endif
#if BOLT_CRC32C_HW_X86
    return crc32c_hw_x86(data, len, seed);
#elif BOLT_CRC32C_HW_ARM
    return crc32c_hw_arm(data, len, seed);
#else
    return crc32c_software(data, len, seed);
#endif
}

// Streaming variant. Identical to `crc32c(data, len, crc)`; split out for
// documentation value at call sites (caller is continuing a stream, not
// computing fresh).
BOLT_FORCE_INLINE uint32_t crc32c_update(uint32_t crc,
                                         const void* BOLT_RESTRICT data,
                                         size_t len) noexcept {
    assert(data != nullptr || len == 0);
    assert(len <= (static_cast<size_t>(1) << 48));  // sanity upper bound
    return crc32c(data, len, crc);
}

/// Page CRC (MSEG §6.1.1): CRC32C over the page's `payload_len` bytes and
/// the zero padding up to `padded_len` (the page's end on its alignment
/// unit). The padding is not read: a page held unpadded in memory has the
/// same CRC as its padded bytes on disk.
inline uint32_t page_crc32c(const void* BOLT_RESTRICT page, size_t payload_len,
                            size_t padded_len, uint32_t seed = 0u) noexcept {
    assert(page != nullptr || payload_len == 0);
    assert(padded_len >= payload_len);
    const uint32_t c = crc32c(page, payload_len, seed);
    return crc32c_extend_zeros(c, padded_len - payload_len);
}

}  // namespace io
}  // namespace bolt
