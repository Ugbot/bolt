#pragma once
// bolt_port.h — portability macros.
// Compiles cleanly on MSVC, clang, clang-cl, gcc. No POSIX in the core.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <bit>
#include <cassert>
#include <thread>

// POSIX mmap/munmap/madvise are used by the huge-page allocator further down
// (bolt_aligned_alloc_huge / _free_huge). The platform thread includes below pull
// <sys/mman.h> in too late — include it here, before first use, so the huge-page
// path compiles on Linux/macOS when BOLT_ENABLE_HUGE_PAGES is set.
#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#endif

// ---------------------------------------------------------------------------
// Platform headers (POSIX / Win32). Hoisted above the huge-page allocator
// below, which calls mmap / VirtualAlloc — those must be declared before
// first use, otherwise enabling huge pages (Linux default) fails to compile.
// bolt_port.h is Bolt's single platform shim; the rest of the core stays
// POSIX-free.
// ---------------------------------------------------------------------------
#if defined(_WIN32)
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#elif defined(__linux__)
    #include <pthread.h>
    #include <sched.h>
    #include <sys/mman.h>
    #include <sys/syscall.h>
    #include <unistd.h>
#elif defined(__APPLE__)
    #include <mach/mach_init.h>
    #include <mach/thread_act.h>
    #include <mach/thread_policy.h>
#endif

// ---------------------------------------------------------------------------
// Compiler detection
// ---------------------------------------------------------------------------
#if defined(_MSC_VER) && !defined(__clang__)
    #define BOLT_COMPILER_MSVC 1
#else
    #define BOLT_COMPILER_MSVC 0
#endif

#if defined(__clang__)
    #define BOLT_COMPILER_CLANG 1
#else
    #define BOLT_COMPILER_CLANG 0
#endif

#if defined(__GNUC__) && !defined(__clang__)
    #define BOLT_COMPILER_GCC 1
#else
    #define BOLT_COMPILER_GCC 0
#endif

// ---------------------------------------------------------------------------
// Architecture detection
// ---------------------------------------------------------------------------
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define BOLT_ARCH_X86 1
#else
    #define BOLT_ARCH_X86 0
#endif

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON)
    #define BOLT_ARCH_ARM 1
#else
    #define BOLT_ARCH_ARM 0
#endif

// ---------------------------------------------------------------------------
// Keyword / attribute shims
// ---------------------------------------------------------------------------
#if BOLT_COMPILER_MSVC
    #define BOLT_RESTRICT     __restrict
    #define BOLT_FORCE_INLINE __forceinline
    #define BOLT_NOINLINE     __declspec(noinline)
    #define BOLT_LIKELY(x)    (x)
    #define BOLT_UNLIKELY(x)  (x)
#else
    #define BOLT_RESTRICT     __restrict__
    #define BOLT_FORCE_INLINE inline __attribute__((always_inline))
    #define BOLT_NOINLINE     __attribute__((noinline))
    #define BOLT_LIKELY(x)    (__builtin_expect(!!(x), 1))
    #define BOLT_UNLIKELY(x)  (__builtin_expect(!!(x), 0))
#endif

// ---------------------------------------------------------------------------
// CPU pause / prefetch
// ---------------------------------------------------------------------------
#if BOLT_ARCH_X86
    #include <immintrin.h>
    #define BOLT_PAUSE() _mm_pause()
#elif BOLT_ARCH_ARM
    #if BOLT_COMPILER_MSVC
        #include <intrin.h>
        #define BOLT_PAUSE() __yield()
    #else
        #define BOLT_PAUSE() asm volatile("yield" ::: "memory")
    #endif
#else
    #define BOLT_PAUSE() ((void)0)
#endif

#if BOLT_COMPILER_MSVC
    // MSVC: _mm_prefetch with locality hint; T0 = L1
    #define BOLT_PREFETCH_READ(p)  _mm_prefetch((const char*)(p), _MM_HINT_T0)
    #define BOLT_PREFETCH_WRITE(p) _mm_prefetch((const char*)(p), _MM_HINT_T0)
#else
    #define BOLT_PREFETCH_READ(p)  __builtin_prefetch((const void*)(p), 0, 1)
    #define BOLT_PREFETCH_WRITE(p) __builtin_prefetch((const void*)(p), 1, 1)
#endif

// ---------------------------------------------------------------------------
// Bit ops (C++20 <bit> is portable, but keep tight names)
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE int bolt_ctz32(uint32_t x) noexcept {
    return x ? std::countr_zero(x) : 32;
}
BOLT_FORCE_INLINE int bolt_ctz64(uint64_t x) noexcept {
    return x ? std::countr_zero(x) : 64;
}
BOLT_FORCE_INLINE int bolt_clz32(uint32_t x) noexcept {
    return x ? std::countl_zero(x) : 32;
}
BOLT_FORCE_INLINE int bolt_clz64(uint64_t x) noexcept {
    return x ? std::countl_zero(x) : 64;
}
BOLT_FORCE_INLINE uint32_t bolt_popcount32(uint32_t x) noexcept {
    return static_cast<uint32_t>(std::popcount(x));
}
BOLT_FORCE_INLINE uint32_t bolt_popcount64(uint64_t x) noexcept {
    return static_cast<uint32_t>(std::popcount(x));
}

// ---------------------------------------------------------------------------
// SWAR (SIMD-Within-A-Register) helpers — find a byte in a 64-bit word.
// Mycroft/Alcorn pattern. Useful for sub-SIMD-width text parsing where
// loading a full SIMD vector would be wasteful (delimiter scan in CSV
// fields, line-end search in small buffers).
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE uint64_t bolt_swar_has_zero_u64(uint64_t v) noexcept {
    return (v - 0x0101010101010101ull) & ~v & 0x8080808080808080ull;
}

BOLT_FORCE_INLINE uint64_t bolt_swar_has_byte_u64(uint64_t word, uint8_t b) noexcept {
    const uint64_t mask = static_cast<uint64_t>(b) * 0x0101010101010101ull;
    return bolt_swar_has_zero_u64(word ^ mask);
}

// Returns lane index [0,8) of the first matching byte, or -1 if none.
BOLT_FORCE_INLINE int bolt_swar_find_byte_u64(uint64_t word, uint8_t b) noexcept {
    const uint64_t hits = bolt_swar_has_byte_u64(word, b);
    if (hits == 0) return -1;
    return static_cast<int>(bolt_ctz64(hits) >> 3);
}

// ---------------------------------------------------------------------------
// Aligned allocation (free-function form, noexcept)
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE void* bolt_aligned_alloc(size_t alignment, size_t size) noexcept {
#if BOLT_COMPILER_MSVC
    return _aligned_malloc(size, alignment);
#else
    // C11 aligned_alloc requires size be a multiple of alignment.
    size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);
    return std::aligned_alloc(alignment, aligned_size);
#endif
}

BOLT_FORCE_INLINE void bolt_aligned_free(void* p) noexcept {
#if BOLT_COMPILER_MSVC
    _aligned_free(p);
#else
    std::free(p);
#endif
}

// ---------------------------------------------------------------------------
// Huge-page aware allocator (D1).
//
// Opt-in via `BOLT_ENABLE_HUGE_PAGES` (see `BoltCompileOptions.cmake`).
// Even when the macro is set, callers should reserve these for *large*
// buffers (≥ 16 MB) where TLB pressure matters — a typical page is 4 KB
// so a 16 MB buffer spans 4096 TLB entries.  2 MB pages cut that to 8.
//
// Per-platform behaviour:
//   * Windows: `VirtualAlloc(MEM_LARGE_PAGES | MEM_COMMIT | MEM_RESERVE)`.
//     Requires the `SeLockMemoryPrivilege` privilege on the process
//     token — NOT granted by default; admin needs to add the user via
//     secpol.msc → "Lock pages in memory".  Without the privilege the
//     call fails and we fall back to `bolt_aligned_alloc`.
//   * Linux: `mmap(..., MAP_HUGETLB)` — requires vm.nr_hugepages > 0 or
//     `vm.nr_overcommit_hugepages` > 0.  On failure we fall back to a
//     plain mmap + `madvise(MADV_HUGEPAGE)` (transparent huge pages).
//   * macOS / other: falls back to `bolt_aligned_alloc` (macOS has no
//     general-purpose huge-page knob for userspace).
//   * When `BOLT_ENABLE_HUGE_PAGES` is 0: always falls back to
//     `bolt_aligned_alloc` so the symbol is still callable.
//
// `size_out` receives the actual rounded-up allocation size (huge-page
// paths round up to 2 MB).  Callers pair with `bolt_huge_free(p, size)`
// — Windows `VirtualFree` ignores size but Linux `munmap` needs it.
//
// Returns nullptr on OOM (same contract as `bolt_aligned_alloc`).
// ---------------------------------------------------------------------------

inline constexpr size_t kBoltHugePageBytes = 2ull * 1024ull * 1024ull;  // 2 MB

BOLT_FORCE_INLINE void* bolt_aligned_alloc_huge(size_t size,
                                                size_t* size_out) noexcept {
    assert(size > 0);
    assert(size_out != nullptr);
    size_t rounded = (size + kBoltHugePageBytes - 1) &
                     ~(kBoltHugePageBytes - 1);

#if !defined(BOLT_ENABLE_HUGE_PAGES) || BOLT_ENABLE_HUGE_PAGES == 0
    // Compile-time off: fall back immediately, report the caller-requested
    // size cache-line rounded (64-byte aligned via bolt_aligned_alloc).
    (void)rounded;
    *size_out = size;
    return bolt_aligned_alloc(64, size);
#elif defined(_WIN32)
    void* p = VirtualAlloc(nullptr, rounded,
                           MEM_LARGE_PAGES | MEM_RESERVE | MEM_COMMIT,
                           PAGE_READWRITE);
    if (p != nullptr) { *size_out = rounded; return p; }
    // Fallback: plain aligned alloc at the caller-requested size.
    *size_out = size;
    return bolt_aligned_alloc(64, size);
#elif defined(__linux__)
    // MAP_HUGETLB first; fall back to regular mmap + MADV_HUGEPAGE on fail.
    #if defined(MAP_HUGETLB)
    void* p = ::mmap(nullptr, rounded, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (p != MAP_FAILED) { *size_out = rounded; return p; }
    #endif
    void* q = ::mmap(nullptr, rounded, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (q == MAP_FAILED) { *size_out = size; return bolt_aligned_alloc(64, size); }
    #if defined(MADV_HUGEPAGE)
    (void)::madvise(q, rounded, MADV_HUGEPAGE);
    #endif
    *size_out = rounded;
    return q;
#else
    *size_out = size;
    return bolt_aligned_alloc(64, size);
#endif
}

// Paired free.  Needs the size returned from bolt_aligned_alloc_huge for
// the Linux munmap path; Windows and the fallback ignore it.
BOLT_FORCE_INLINE void bolt_aligned_free_huge(void* p, size_t size) noexcept {
    if (p == nullptr) return;
#if !defined(BOLT_ENABLE_HUGE_PAGES) || BOLT_ENABLE_HUGE_PAGES == 0
    (void)size;
    bolt_aligned_free(p);
#elif defined(_WIN32)
    (void)size;
    // If the allocation came from VirtualAlloc we must VirtualFree; if it
    // fell back to _aligned_malloc we must _aligned_free. Distinguish via
    // VirtualQuery: pages from VirtualAlloc have a nonzero AllocationBase.
    MEMORY_BASIC_INFORMATION info{};
    if (VirtualQuery(p, &info, sizeof(info)) != 0 &&
        info.AllocationBase == p &&
        (info.Type & MEM_PRIVATE) != 0 &&
        (info.State & MEM_COMMIT) != 0) {
        VirtualFree(p, 0, MEM_RELEASE);
    } else {
        bolt_aligned_free(p);
    }
#elif defined(__linux__)
    if (size > 0) ::munmap(p, size);
    else bolt_aligned_free(p);
#else
    (void)size;
    bolt_aligned_free(p);
#endif
}

// Platform headers (POSIX / Win32) are included near the top of this file —
// hoisted there so the huge-page allocator above (mmap / VirtualAlloc) sees
// them. The thread-affinity / NUMA shims below rely on the same includes.

// ---------------------------------------------------------------------------
// Thread affinity, NUMA, hardware concurrency
//
// Best-effort, header-only, zero-dependency shims. All return codes only, no
// exceptions. Platform-specific headers are included inside their own
// #ifdef branches to keep the common public surface clean.
//
// Per-platform semantics:
//   * Windows: thread affinity — processor-group aware. CPUs 0..63 pin
//     via `SetThreadAffinityMask` (single-group fast path); CPUs 64+
//     use `SetThreadGroupAffinity` (group_idx = cpu >> 6, mask within
//     group = 1 << (cpu & 63)). Supports the full 4096-CPU space.
//   * Linux:   pthread_setaffinity_np; NUMA via raw set_mempolicy syscall
//     to stay libnuma-free.
//   * macOS:   thread_policy_set(THREAD_AFFINITY_POLICY) — Darwin treats
//     affinity as a *hint*, not a hard pin. No NUMA on Darwin.
//   * Other:   return false.
// ---------------------------------------------------------------------------

// Logical-CPU count, with a safe fallback of 1 when the stdlib reports 0.
BOLT_FORCE_INLINE uint32_t bolt_get_hardware_concurrency() noexcept {
    const unsigned n = std::thread::hardware_concurrency();
    return n == 0u ? 1u : static_cast<uint32_t>(n);
}

// Pin the calling thread to `logical_cpu`. Returns true on success.
// On macOS this is only a scheduler hint. On Windows this routes to
// `SetThreadAffinityMask` (group 0) or `SetThreadGroupAffinity` (group > 0)
// based on `logical_cpu >> 6`, covering the full 4096-CPU space.
BOLT_FORCE_INLINE bool bolt_pin_current_thread(uint32_t logical_cpu) noexcept {
    assert(logical_cpu < 4096u && "suspiciously large logical CPU index");
#if defined(_WIN32)
    HANDLE         h            = GetCurrentThread();
    const uint32_t group_idx    = logical_cpu >> 6;
    const uint32_t cpu_in_group = logical_cpu & 63u;
    if (group_idx == 0u) {
        // Single-group fast path — preserves the classic affinity mask.
        DWORD_PTR mask = static_cast<DWORD_PTR>(1ull << cpu_in_group);
        DWORD_PTR prev = SetThreadAffinityMask(h, mask);
        return prev != 0;
    }
    // Multi-group: groups 0..n-1 each hold up to 64 logical CPUs.
    // SetThreadGroupAffinity is the Win7+ API that crosses the group boundary.
    GROUP_AFFINITY ga;
    ga.Mask        = static_cast<KAFFINITY>(1ull << cpu_in_group);
    ga.Group       = static_cast<WORD>(group_idx);
    ga.Reserved[0] = 0;
    ga.Reserved[1] = 0;
    ga.Reserved[2] = 0;
    const BOOL ok = SetThreadGroupAffinity(h, &ga, nullptr);
    return ok != 0;
#elif defined(__linux__)
    assert(logical_cpu < static_cast<uint32_t>(CPU_SETSIZE) && "cpu index exceeds CPU_SETSIZE");
    if (logical_cpu >= static_cast<uint32_t>(CPU_SETSIZE)) {
        return false;
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(static_cast<int>(logical_cpu), &set);
    const int err = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    return err == 0;
#elif defined(__APPLE__)
    // Darwin: hint only. Threads sharing a tag are scheduled on the same L2.
    thread_affinity_policy_data_t policy;
    policy.affinity_tag = static_cast<integer_t>(logical_cpu + 1u);
    const kern_return_t kr = thread_policy_set(
        mach_thread_self(),
        THREAD_AFFINITY_POLICY,
        reinterpret_cast<thread_policy_t>(&policy),
        THREAD_AFFINITY_POLICY_COUNT);
    return kr == KERN_SUCCESS;
#else
    (void)logical_cpu;
    return false;
#endif
}

// Set NUMA preferred node for subsequent allocations by this thread.
// Windows: best-effort — maps the first logical CPU on `node` to
// SetThreadIdealProcessor. Full page-placement control wants
// VirtualAllocExNuma, which is out of scope for this wave.
// macOS: no NUMA, always returns false.
BOLT_FORCE_INLINE bool bolt_set_numa_preferred(int node) noexcept {
    assert(node >= 0 && "NUMA node must be non-negative");
    assert(node < 1024 && "suspiciously large NUMA node index");
    if (node < 0) {
        return false;
    }
#if defined(_WIN32)
    GROUP_AFFINITY ga;
    ga.Mask     = 0;
    ga.Group    = 0;
    ga.Reserved[0] = 0;
    ga.Reserved[1] = 0;
    ga.Reserved[2] = 0;
    if (!GetNumaNodeProcessorMaskEx(static_cast<USHORT>(node), &ga)) {
        return false;
    }
    if (ga.Mask == 0) {
        return false;
    }
    // First logical CPU in the node mask.
    unsigned long idx = 0;
    KAFFINITY m = ga.Mask;
    while ((m & 1u) == 0u && idx < 63u) {
        m >>= 1;
        ++idx;
    }
    const DWORD prev = SetThreadIdealProcessor(GetCurrentThread(), static_cast<DWORD>(idx));
    return prev != static_cast<DWORD>(-1);
#elif defined(__linux__)
    // Stay libnuma-free: use raw numbers.
    //   MPOL_PREFERRED == 1
    //   nodemask is a bitmask of allowed nodes; max_nodes counts bits.
    constexpr int    MPOL_PREFERRED = 1;
    constexpr unsigned long MAX_NODES = 1024;
    unsigned long nodemask[MAX_NODES / (sizeof(unsigned long) * 8)] = {};
    const int word = node / static_cast<int>(sizeof(unsigned long) * 8);
    const int bit  = node % static_cast<int>(sizeof(unsigned long) * 8);
    if (static_cast<unsigned long>(word) >= (sizeof(nodemask) / sizeof(nodemask[0]))) {
        return false;
    }
    nodemask[word] = 1ul << bit;
    const long rc = syscall(SYS_set_mempolicy, MPOL_PREFERRED, nodemask, MAX_NODES);
    return rc == 0;
#elif defined(__APPLE__)
    (void)node;
    return false;
#else
    (void)node;
    return false;
#endif
}

// ---------------------------------------------------------------------------
// cglm-inspired SIMD compile-time dispatch (bmm_* wrappers)
//
// One macro family selects the SIMD backend at compile time. Kernel authors
// write `bmm_cmpgt_i32(a, b)` once; the preprocessor picks SSE4.2, AVX2,
// NEON, or a scalar fallback. No runtime dispatch, no function pointers.
//
// MSVC note: `__AVX2__` is auto-defined only when `/arch:AVX2` is passed;
// likewise x64 MSVC does NOT define `__SSE4_2__` automatically (SSE2 is
// the baseline). On a stock MSVC Release build we therefore fall through
// to the scalar path, which is correct-but-slow. The kernels still build.
//
// Borrowed from cglm: compile-time macro dispatch, `*_ALL_UNALIGNED`
// toggle, per-type natural alignment (see docs/research/cglm.md).
// ---------------------------------------------------------------------------
#if defined(__AVX512F__)
    // AVX-512 host. We provide a native 512-bit block under
    // `#if BOLT_SIMD_AVX512` (bmm_vec_i32_x16 / bmm_vec_i64_x8 /
    // bmm_vec_f64_x8 and their load/store/mask/compare-to-mask/
    // compressstore/conflict ops). AVX-512 is a strict superset of AVX2,
    // so we *also* set BOLT_SIMD_AVX2: the 256-bit bmm_vec_i32 etc.
    // symbols remain the default and execute correctly on AVX-512
    // silicon. Kernels opt in to 512-bit by naming the _x16/_x8 types.
    // See docs/research/avx512-status.md.
    #define BOLT_SIMD_AVX512 1
    #define BOLT_SIMD_AVX2   1
    #define BOLT_SIMD_WIDTH  64
#elif defined(__AVX2__)
    #define BOLT_SIMD_AVX2  1
    #define BOLT_SIMD_WIDTH 32
#elif defined(__SSE4_2__)
    #define BOLT_SIMD_SSE42 1
    #define BOLT_SIMD_WIDTH 16
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define BOLT_SIMD_NEON  1
    #define BOLT_SIMD_WIDTH 16
#else
    #define BOLT_SIMD_SCALAR 1
    #define BOLT_SIMD_WIDTH  0
#endif

// Ensure all flavors are defined to 0 when not selected (makes `#if` safe).
#ifndef BOLT_SIMD_AVX512
    #define BOLT_SIMD_AVX512 0
#endif
#ifndef BOLT_SIMD_AVX2
    #define BOLT_SIMD_AVX2 0
#endif
#ifndef BOLT_SIMD_SSE42
    #define BOLT_SIMD_SSE42 0
#endif
#ifndef BOLT_SIMD_NEON
    #define BOLT_SIMD_NEON 0
#endif
#ifndef BOLT_SIMD_SCALAR
    #define BOLT_SIMD_SCALAR 0
#endif

// ARMv8.2-A FP16 vector arithmetic (vfmaq_f16 + float16x8_t native FMA).
// Detect via the ACLE feature macro; Apple Silicon (M1/M2/M3) and recent
// ARM Cortex-A75+ set this. AVX2 falls back via _mm256_cvtph_ps in the
// f16 kernels; this macro guards the NEON-native FP16 arms only.
#if BOLT_SIMD_NEON && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && \
    __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    #define BOLT_SIMD_NEON_FP16 1
#else
    #define BOLT_SIMD_NEON_FP16 0
#endif

// ARMv8.4-A DOTPROD (vdotq_u32 + vdotq_s32 — 4-way u8/i8 dot product
// into 32-bit accumulator). M1/M2/M3 set this; older NEON cores don't.
// Required for the u8 vector kernels' fast path.
#if BOLT_SIMD_NEON && defined(__ARM_FEATURE_DOTPROD) && __ARM_FEATURE_DOTPROD
    #define BOLT_SIMD_NEON_DOTPROD 1
#else
    #define BOLT_SIMD_NEON_DOTPROD 0
#endif

// Headers for each ISA. immintrin.h on x86 already pulled in above.
#if BOLT_SIMD_NEON
    #include <arm_neon.h>
#endif
#if BOLT_SIMD_SSE42 && !BOLT_ARCH_X86
    // Safety: SSE path only makes sense on x86. If we somehow got here on
    // a non-x86 target, fall back to scalar.
    #undef  BOLT_SIMD_SSE42
    #define BOLT_SIMD_SSE42 0
    #undef  BOLT_SIMD_SCALAR
    #define BOLT_SIMD_SCALAR 1
#endif

#include <array>

namespace bolt {
namespace simd {

// ---- Vector typedefs -------------------------------------------------------
#if BOLT_SIMD_AVX2
    using bmm_vec_i32 = __m256i;
    using bmm_vec_i64 = __m256i;
    using bmm_vec_f64 = __m256d;
    inline constexpr int bmm_lanes_i32 = 8;
    inline constexpr int bmm_lanes_i64 = 4;
    inline constexpr int bmm_lanes_f64 = 4;
#elif BOLT_SIMD_SSE42
    using bmm_vec_i32 = __m128i;
    using bmm_vec_i64 = __m128i;
    using bmm_vec_f64 = __m128d;
    inline constexpr int bmm_lanes_i32 = 4;
    inline constexpr int bmm_lanes_i64 = 2;
    inline constexpr int bmm_lanes_f64 = 2;
#elif BOLT_SIMD_NEON
    using bmm_vec_i32 = int32x4_t;
    using bmm_vec_i64 = int64x2_t;
    using bmm_vec_f64 = float64x2_t;
    inline constexpr int bmm_lanes_i32 = 4;
    inline constexpr int bmm_lanes_i64 = 2;
    inline constexpr int bmm_lanes_f64 = 2;
#else  // SCALAR
    struct bmm_vec_i32 { std::array<int32_t, 4> v; };
    struct bmm_vec_i64 { std::array<int64_t, 4> v; };
    struct bmm_vec_f64 { std::array<double,  4> v; };
    inline constexpr int bmm_lanes_i32 = 4;
    inline constexpr int bmm_lanes_i64 = 4;
    inline constexpr int bmm_lanes_f64 = 4;
#endif

// ---- Per-ISA implementation -----------------------------------------------
#if BOLT_SIMD_AVX2

BOLT_FORCE_INLINE bmm_vec_i32 bmm_load_i32(const int32_t* p) noexcept {
#ifdef BOLT_ALL_UNALIGNED
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
#else
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(p));
#endif
}
BOLT_FORCE_INLINE bmm_vec_i32 bmm_loadu_i32(const int32_t* p) noexcept {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
}
BOLT_FORCE_INLINE void bmm_store_i32(int32_t* p, bmm_vec_i32 v) noexcept {
#ifdef BOLT_ALL_UNALIGNED
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
#else
    _mm256_store_si256(reinterpret_cast<__m256i*>(p), v);
#endif
}
BOLT_FORCE_INLINE void bmm_storeu_i32(int32_t* p, bmm_vec_i32 v) noexcept {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
}
BOLT_FORCE_INLINE bmm_vec_i32 bmm_set1_i32(int32_t x) noexcept { return _mm256_set1_epi32(x); }
BOLT_FORCE_INLINE bmm_vec_i32 bmm_cmpgt_i32(bmm_vec_i32 a, bmm_vec_i32 b) noexcept { return _mm256_cmpgt_epi32(a, b); }
BOLT_FORCE_INLINE bmm_vec_i32 bmm_cmplt_i32(bmm_vec_i32 a, bmm_vec_i32 b) noexcept { return _mm256_cmpgt_epi32(b, a); }
BOLT_FORCE_INLINE bmm_vec_i32 bmm_cmpeq_i32(bmm_vec_i32 a, bmm_vec_i32 b) noexcept { return _mm256_cmpeq_epi32(a, b); }
BOLT_FORCE_INLINE bmm_vec_i32 bmm_and(bmm_vec_i32 a, bmm_vec_i32 b) noexcept { return _mm256_and_si256(a, b); }
BOLT_FORCE_INLINE bmm_vec_i32 bmm_or (bmm_vec_i32 a, bmm_vec_i32 b) noexcept { return _mm256_or_si256(a, b); }
BOLT_FORCE_INLINE int bmm_movemask_i32(bmm_vec_i32 v) noexcept { return _mm256_movemask_ps(_mm256_castsi256_ps(v)); }

BOLT_FORCE_INLINE bmm_vec_i64 bmm_cmpgt_i64(bmm_vec_i64 a, bmm_vec_i64 b) noexcept { return _mm256_cmpgt_epi64(a, b); }
BOLT_FORCE_INLINE bmm_vec_i64 bmm_cmpeq_i64(bmm_vec_i64 a, bmm_vec_i64 b) noexcept { return _mm256_cmpeq_epi64(a, b); }
BOLT_FORCE_INLINE bmm_vec_i64 bmm_loadu_i64(const int64_t* p) noexcept {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
}
BOLT_FORCE_INLINE void bmm_storeu_i64(int64_t* p, bmm_vec_i64 v) noexcept {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
}
BOLT_FORCE_INLINE bmm_vec_i64 bmm_set1_i64(int64_t x) noexcept { return _mm256_set1_epi64x(x); }
BOLT_FORCE_INLINE bmm_vec_i64 bmm_add_i64(bmm_vec_i64 a, bmm_vec_i64 b) noexcept { return _mm256_add_epi64(a, b); }
BOLT_FORCE_INLINE bmm_vec_i64 bmm_setzero_i64() noexcept { return _mm256_setzero_si256(); }
BOLT_FORCE_INLINE int bmm_movemask_i64(bmm_vec_i64 v) noexcept { return _mm256_movemask_pd(_mm256_castsi256_pd(v)); }
BOLT_FORCE_INLINE bmm_vec_f64 bmm_loadu_f64(const double* p) noexcept { return _mm256_loadu_pd(p); }
BOLT_FORCE_INLINE bmm_vec_f64 bmm_set1_f64(double x) noexcept { return _mm256_set1_pd(x); }
BOLT_FORCE_INLINE bmm_vec_f64 bmm_cmpgt_f64(bmm_vec_f64 a, bmm_vec_f64 b) noexcept { return _mm256_cmp_pd(a, b, _CMP_GT_OQ); }
BOLT_FORCE_INLINE bmm_vec_f64 bmm_cmpeq_f64(bmm_vec_f64 a, bmm_vec_f64 b) noexcept { return _mm256_cmp_pd(a, b, _CMP_EQ_OQ); }
BOLT_FORCE_INLINE int bmm_movemask_f64(bmm_vec_f64 v) noexcept { return _mm256_movemask_pd(v); }

#elif BOLT_SIMD_SSE42

BOLT_FORCE_INLINE bmm_vec_i32 bmm_load_i32(const int32_t* p) noexcept {
#ifdef BOLT_ALL_UNALIGNED
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
#else
    return _mm_load_si128(reinterpret_cast<const __m128i*>(p));
#endif
}
BOLT_FORCE_INLINE bmm_vec_i32 bmm_loadu_i32(const int32_t* p) noexcept {
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
}
BOLT_FORCE_INLINE void bmm_store_i32(int32_t* p, bmm_vec_i32 v) noexcept {
#ifdef BOLT_ALL_UNALIGNED
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v);
#else
    _mm_store_si128(reinterpret_cast<__m128i*>(p), v);
#endif
}
BOLT_FORCE_INLINE void bmm_storeu_i32(int32_t* p, bmm_vec_i32 v) noexcept {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v);
}
BOLT_FORCE_INLINE bmm_vec_i32 bmm_set1_i32(int32_t x) noexcept { return _mm_set1_epi32(x); }
BOLT_FORCE_INLINE bmm_vec_i32 bmm_cmpgt_i32(bmm_vec_i32 a, bmm_vec_i32 b) noexcept { return _mm_cmpgt_epi32(a, b); }
BOLT_FORCE_INLINE bmm_vec_i32 bmm_cmplt_i32(bmm_vec_i32 a, bmm_vec_i32 b) noexcept { return _mm_cmplt_epi32(a, b); }
BOLT_FORCE_INLINE bmm_vec_i32 bmm_cmpeq_i32(bmm_vec_i32 a, bmm_vec_i32 b) noexcept { return _mm_cmpeq_epi32(a, b); }
BOLT_FORCE_INLINE bmm_vec_i32 bmm_and(bmm_vec_i32 a, bmm_vec_i32 b) noexcept { return _mm_and_si128(a, b); }
BOLT_FORCE_INLINE bmm_vec_i32 bmm_or (bmm_vec_i32 a, bmm_vec_i32 b) noexcept { return _mm_or_si128(a, b); }
BOLT_FORCE_INLINE int bmm_movemask_i32(bmm_vec_i32 v) noexcept { return _mm_movemask_ps(_mm_castsi128_ps(v)); }

BOLT_FORCE_INLINE bmm_vec_i64 bmm_cmpgt_i64(bmm_vec_i64 a, bmm_vec_i64 b) noexcept { return _mm_cmpgt_epi64(a, b); }
BOLT_FORCE_INLINE bmm_vec_i64 bmm_cmpeq_i64(bmm_vec_i64 a, bmm_vec_i64 b) noexcept { return _mm_cmpeq_epi64(a, b); }
BOLT_FORCE_INLINE bmm_vec_i64 bmm_loadu_i64(const int64_t* p) noexcept {
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
}
BOLT_FORCE_INLINE void bmm_storeu_i64(int64_t* p, bmm_vec_i64 v) noexcept {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v);
}
BOLT_FORCE_INLINE bmm_vec_i64 bmm_set1_i64(int64_t x) noexcept { return _mm_set1_epi64x(x); }
BOLT_FORCE_INLINE bmm_vec_i64 bmm_add_i64(bmm_vec_i64 a, bmm_vec_i64 b) noexcept { return _mm_add_epi64(a, b); }
BOLT_FORCE_INLINE bmm_vec_i64 bmm_setzero_i64() noexcept { return _mm_setzero_si128(); }
BOLT_FORCE_INLINE int bmm_movemask_i64(bmm_vec_i64 v) noexcept { return _mm_movemask_pd(_mm_castsi128_pd(v)); }
BOLT_FORCE_INLINE bmm_vec_f64 bmm_loadu_f64(const double* p) noexcept { return _mm_loadu_pd(p); }
BOLT_FORCE_INLINE bmm_vec_f64 bmm_set1_f64(double x) noexcept { return _mm_set1_pd(x); }
BOLT_FORCE_INLINE bmm_vec_f64 bmm_cmpgt_f64(bmm_vec_f64 a, bmm_vec_f64 b) noexcept { return _mm_cmpgt_pd(a, b); }
BOLT_FORCE_INLINE bmm_vec_f64 bmm_cmpeq_f64(bmm_vec_f64 a, bmm_vec_f64 b) noexcept { return _mm_cmpeq_pd(a, b); }
BOLT_FORCE_INLINE int bmm_movemask_f64(bmm_vec_f64 v) noexcept { return _mm_movemask_pd(v); }

#elif BOLT_SIMD_NEON

BOLT_FORCE_INLINE bmm_vec_i32 bmm_load_i32 (const int32_t* p) noexcept { return vld1q_s32(p); }
BOLT_FORCE_INLINE bmm_vec_i32 bmm_loadu_i32(const int32_t* p) noexcept { return vld1q_s32(p); }
BOLT_FORCE_INLINE void bmm_store_i32 (int32_t* p, bmm_vec_i32 v) noexcept { vst1q_s32(p, v); }
BOLT_FORCE_INLINE void bmm_storeu_i32(int32_t* p, bmm_vec_i32 v) noexcept { vst1q_s32(p, v); }
BOLT_FORCE_INLINE bmm_vec_i32 bmm_set1_i32(int32_t x) noexcept { return vdupq_n_s32(x); }
BOLT_FORCE_INLINE bmm_vec_i32 bmm_cmpgt_i32(bmm_vec_i32 a, bmm_vec_i32 b) noexcept { return vreinterpretq_s32_u32(vcgtq_s32(a, b)); }
BOLT_FORCE_INLINE bmm_vec_i32 bmm_cmplt_i32(bmm_vec_i32 a, bmm_vec_i32 b) noexcept { return vreinterpretq_s32_u32(vcltq_s32(a, b)); }
BOLT_FORCE_INLINE bmm_vec_i32 bmm_cmpeq_i32(bmm_vec_i32 a, bmm_vec_i32 b) noexcept { return vreinterpretq_s32_u32(vceqq_s32(a, b)); }
BOLT_FORCE_INLINE bmm_vec_i32 bmm_and(bmm_vec_i32 a, bmm_vec_i32 b) noexcept { return vandq_s32(a, b); }
BOLT_FORCE_INLINE bmm_vec_i32 bmm_or (bmm_vec_i32 a, bmm_vec_i32 b) noexcept { return vorrq_s32(a, b); }
// NEON movemask polyfill: extract high bit from each lane, pack to 4-bit mask.
BOLT_FORCE_INLINE int bmm_movemask_i32(bmm_vec_i32 v) noexcept {
    uint32x4_t u = vreinterpretq_u32_s32(v);
    // Shift each lane to put the sign bit at bit position = lane_index.
    alignas(16) static const int32_t kShift[4] = {0, 1, 2, 3};
    int32x4_t shift = vld1q_s32(kShift);
    uint32x4_t bits = vshlq_u32(vshrq_n_u32(u, 31), shift);
    return static_cast<int>(vaddvq_u32(bits));
}

BOLT_FORCE_INLINE bmm_vec_i64 bmm_cmpgt_i64(bmm_vec_i64 a, bmm_vec_i64 b) noexcept { return vreinterpretq_s64_u64(vcgtq_s64(a, b)); }
BOLT_FORCE_INLINE bmm_vec_i64 bmm_cmpeq_i64(bmm_vec_i64 a, bmm_vec_i64 b) noexcept { return vreinterpretq_s64_u64(vceqq_s64(a, b)); }
BOLT_FORCE_INLINE bmm_vec_i64 bmm_loadu_i64(const int64_t* p) noexcept { return vld1q_s64(p); }
BOLT_FORCE_INLINE void bmm_storeu_i64(int64_t* p, bmm_vec_i64 v) noexcept { vst1q_s64(p, v); }
BOLT_FORCE_INLINE bmm_vec_i64 bmm_set1_i64(int64_t x) noexcept { return vdupq_n_s64(x); }
BOLT_FORCE_INLINE bmm_vec_i64 bmm_add_i64(bmm_vec_i64 a, bmm_vec_i64 b) noexcept { return vaddq_s64(a, b); }
BOLT_FORCE_INLINE bmm_vec_i64 bmm_setzero_i64() noexcept { return vdupq_n_s64(0); }
// NEON movemask polyfill for 2-lane i64: pack two sign bits.
BOLT_FORCE_INLINE int bmm_movemask_i64(bmm_vec_i64 v) noexcept {
    uint64x2_t u = vreinterpretq_u64_s64(v);
    uint64_t lo = vgetq_lane_u64(u, 0);
    uint64_t hi = vgetq_lane_u64(u, 1);
    return static_cast<int>(((lo >> 63) & 1u) | (((hi >> 63) & 1u) << 1));
}
BOLT_FORCE_INLINE bmm_vec_f64 bmm_loadu_f64(const double* p) noexcept { return vld1q_f64(p); }
BOLT_FORCE_INLINE bmm_vec_f64 bmm_set1_f64(double x) noexcept { return vdupq_n_f64(x); }
BOLT_FORCE_INLINE bmm_vec_f64 bmm_cmpgt_f64(bmm_vec_f64 a, bmm_vec_f64 b) noexcept { return vreinterpretq_f64_u64(vcgtq_f64(a, b)); }
BOLT_FORCE_INLINE bmm_vec_f64 bmm_cmpeq_f64(bmm_vec_f64 a, bmm_vec_f64 b) noexcept { return vreinterpretq_f64_u64(vceqq_f64(a, b)); }
BOLT_FORCE_INLINE int bmm_movemask_f64(bmm_vec_f64 v) noexcept {
    uint64x2_t u = vreinterpretq_u64_f64(v);
    uint64_t lo = vgetq_lane_u64(u, 0);
    uint64_t hi = vgetq_lane_u64(u, 1);
    return static_cast<int>(((lo >> 63) & 1u) | (((hi >> 63) & 1u) << 1));
}

#else  // BOLT_SIMD_SCALAR — tight, correct, slow.

BOLT_FORCE_INLINE bmm_vec_i32 bmm_load_i32 (const int32_t* p) noexcept { bmm_vec_i32 r; for (int i=0;i<4;++i) r.v[i]=p[i]; return r; }
BOLT_FORCE_INLINE bmm_vec_i32 bmm_loadu_i32(const int32_t* p) noexcept { return bmm_load_i32(p); }
BOLT_FORCE_INLINE void bmm_store_i32 (int32_t* p, bmm_vec_i32 v) noexcept { for (int i=0;i<4;++i) p[i]=v.v[i]; }
BOLT_FORCE_INLINE void bmm_storeu_i32(int32_t* p, bmm_vec_i32 v) noexcept { bmm_store_i32(p, v); }
BOLT_FORCE_INLINE bmm_vec_i32 bmm_set1_i32(int32_t x) noexcept { return bmm_vec_i32{{x,x,x,x}}; }
BOLT_FORCE_INLINE bmm_vec_i32 bmm_cmpgt_i32(bmm_vec_i32 a, bmm_vec_i32 b) noexcept { bmm_vec_i32 r; for (int i=0;i<4;++i) r.v[i] = (a.v[i] >  b.v[i]) ? -1 : 0; return r; }
BOLT_FORCE_INLINE bmm_vec_i32 bmm_cmplt_i32(bmm_vec_i32 a, bmm_vec_i32 b) noexcept { bmm_vec_i32 r; for (int i=0;i<4;++i) r.v[i] = (a.v[i] <  b.v[i]) ? -1 : 0; return r; }
BOLT_FORCE_INLINE bmm_vec_i32 bmm_cmpeq_i32(bmm_vec_i32 a, bmm_vec_i32 b) noexcept { bmm_vec_i32 r; for (int i=0;i<4;++i) r.v[i] = (a.v[i] == b.v[i]) ? -1 : 0; return r; }
BOLT_FORCE_INLINE bmm_vec_i32 bmm_and(bmm_vec_i32 a, bmm_vec_i32 b) noexcept { bmm_vec_i32 r; for (int i=0;i<4;++i) r.v[i] = a.v[i] & b.v[i]; return r; }
BOLT_FORCE_INLINE bmm_vec_i32 bmm_or (bmm_vec_i32 a, bmm_vec_i32 b) noexcept { bmm_vec_i32 r; for (int i=0;i<4;++i) r.v[i] = a.v[i] | b.v[i]; return r; }
BOLT_FORCE_INLINE int bmm_movemask_i32(bmm_vec_i32 v) noexcept {
    int m = 0;
    for (int i = 0; i < 4; ++i) m |= ((static_cast<uint32_t>(v.v[i]) >> 31) & 1u) << i;
    return m;
}

BOLT_FORCE_INLINE bmm_vec_i64 bmm_cmpgt_i64(bmm_vec_i64 a, bmm_vec_i64 b) noexcept { bmm_vec_i64 r; for (int i=0;i<4;++i) r.v[i] = (a.v[i] >  b.v[i]) ? -1 : 0; return r; }
BOLT_FORCE_INLINE bmm_vec_i64 bmm_cmpeq_i64(bmm_vec_i64 a, bmm_vec_i64 b) noexcept { bmm_vec_i64 r; for (int i=0;i<4;++i) r.v[i] = (a.v[i] == b.v[i]) ? -1 : 0; return r; }
BOLT_FORCE_INLINE bmm_vec_i64 bmm_loadu_i64(const int64_t* p) noexcept { bmm_vec_i64 r; for (int i=0;i<4;++i) r.v[i] = p[i]; return r; }
BOLT_FORCE_INLINE void bmm_storeu_i64(int64_t* p, bmm_vec_i64 v) noexcept { for (int i=0;i<4;++i) p[i] = v.v[i]; }
BOLT_FORCE_INLINE bmm_vec_i64 bmm_set1_i64(int64_t x) noexcept { return bmm_vec_i64{{x,x,x,x}}; }
BOLT_FORCE_INLINE bmm_vec_i64 bmm_add_i64(bmm_vec_i64 a, bmm_vec_i64 b) noexcept { bmm_vec_i64 r; for (int i=0;i<4;++i) r.v[i] = a.v[i] + b.v[i]; return r; }
BOLT_FORCE_INLINE bmm_vec_i64 bmm_setzero_i64() noexcept { return bmm_vec_i64{{0,0,0,0}}; }
BOLT_FORCE_INLINE int bmm_movemask_i64(bmm_vec_i64 v) noexcept {
    int m = 0;
    for (int i = 0; i < 4; ++i) m |= ((static_cast<uint64_t>(v.v[i]) >> 63) & 1u) << i;
    return m;
}
BOLT_FORCE_INLINE bmm_vec_f64 bmm_loadu_f64(const double* p) noexcept { bmm_vec_f64 r; for (int i=0;i<4;++i) r.v[i] = p[i]; return r; }
BOLT_FORCE_INLINE bmm_vec_f64 bmm_set1_f64(double x) noexcept { return bmm_vec_f64{{x,x,x,x}}; }
BOLT_FORCE_INLINE bmm_vec_f64 bmm_cmpgt_f64(bmm_vec_f64 a, bmm_vec_f64 b) noexcept { bmm_vec_f64 r; for (int i=0;i<4;++i) { uint64_t bits = (a.v[i] > b.v[i]) ? 0xFFFFFFFFFFFFFFFFull : 0ull; std::memcpy(&r.v[i], &bits, sizeof(double)); } return r; }
BOLT_FORCE_INLINE bmm_vec_f64 bmm_cmpeq_f64(bmm_vec_f64 a, bmm_vec_f64 b) noexcept { bmm_vec_f64 r; for (int i=0;i<4;++i) { uint64_t bits = (a.v[i] == b.v[i]) ? 0xFFFFFFFFFFFFFFFFull : 0ull; std::memcpy(&r.v[i], &bits, sizeof(double)); } return r; }
BOLT_FORCE_INLINE int bmm_movemask_f64(bmm_vec_f64 v) noexcept {
    int m = 0;
    for (int i = 0; i < 4; ++i) {
        uint64_t bits; std::memcpy(&bits, &v.v[i], sizeof(double));
        m |= ((bits >> 63) & 1u) << i;
    }
    return m;
}

#endif  // ISA dispatch

// ---------------------------------------------------------------------------
// Wave D2 extensions — additional SIMD ops
//
// Per-ISA mapping summary:
//   compressstore_i32 : AVX2 = permutevar8x32 + 256-entry LUT + maskstore;
//                       SSE4.2 = pshufb + 16-entry LUT (4-lane chunks);
//                       NEON   = per-lane branch-free store via ctz;
//                       Scalar = loop.
//   gather_i32/i64    : AVX2 = _mm256_i32gather_{epi32,epi64};
//                       SSE/NEON/Scalar = per-lane scalar loads.
//   hadd_i64/f64      : AVX2 = 128-lane split + add + shuffle;
//                       SSE  = shuffle + add;
//                       NEON = vaddvq_{s64,f64};
//                       Scalar = loop.
//   f32 ops           : AVX2 = __m256 ; SSE = __m128 ; NEON = float32x4_t.
//   i8  ops           : AVX2 = __m256i (32 lanes) ; SSE/NEON = 16 lanes ;
//                       movemask_i8 via _mm{,256}_movemask_epi8 / NEON shr+shl reduce.
//   shuffle_i32       : AVX2 = permutevar8x32 ; SSE = pshufb(byte-LUT) ;
//                       NEON = vqtbl1q_u8 (byte-level tbl).
//   blend_i32         : AVX2 = blendv_epi8 ; SSE = blendv_epi8 ;
//                       NEON = vbslq_s32 ; Scalar = ternary.
// ---------------------------------------------------------------------------

// ---- Compressstore LUT (used by AVX2 and SSE42 paths) ---------------------
// Each entry encodes the permutation that gathers selected lanes to the low
// end. AVX2 uses the 8-lane table (indexed 0..255 by movemask_i32).
// SSE42 uses the 4-lane byte-shuffle table (indexed 0..15 by movemask).
#if BOLT_SIMD_AVX2
// 4-lane (i64/f64) compress LUT on AVX2: 16 entries × 8 int32 lane indices.
// Each 64-bit lane maps to two adjacent 32-bit halves (indices 2k, 2k+1), so
// we implement compress-i64 as permutevar8x32 on the i64-as-i32 view.
inline const int32_t* bmm_detail_avx2_compress_lut4() noexcept {
    static const auto tbl = []() noexcept {
        std::array<std::array<int32_t, 8>, 16> t{};
        for (int m = 0; m < 16; ++m) {
            int pos = 0;
            for (int lane = 0; lane < 4; ++lane) {
                if (m & (1 << lane)) {
                    t[m][pos * 2 + 0] = lane * 2 + 0;
                    t[m][pos * 2 + 1] = lane * 2 + 1;
                    ++pos;
                }
            }
            for (int rem = pos * 2; rem < 8; ++rem) t[m][rem] = 0;
        }
        return t;
    }();
    return reinterpret_cast<const int32_t*>(tbl.data());
}

inline const int32_t* bmm_detail_avx2_compress_lut() noexcept {
    // 256 entries × 8 int32 = 8 KiB. Built at first call (const, so thread-safe
    // under C++11 magic-statics on MSVC/clang/gcc).
    static const auto tbl = []() noexcept {
        std::array<std::array<int32_t, 8>, 256> t{};
        for (int m = 0; m < 256; ++m) {
            int pos = 0;
            for (int i = 0; i < 8; ++i) {
                if (m & (1 << i)) t[m][pos++] = i;
            }
            for (; pos < 8; ++pos) t[m][pos] = 0;
        }
        return t;
    }();
    return reinterpret_cast<const int32_t*>(tbl.data());
}
#endif

#if BOLT_SIMD_SSE42
// 2-lane i64/f64 compress LUT on SSE: 4 entries × 16 byte shuffle control.
inline const uint8_t* bmm_detail_sse_compress_lut2() noexcept {
    static const auto tbl = []() noexcept {
        std::array<std::array<uint8_t, 16>, 4> t{};
        for (int m = 0; m < 4; ++m) {
            int pos = 0;
            for (int lane = 0; lane < 2; ++lane) {
                if (m & (1 << lane)) {
                    for (int b = 0; b < 8; ++b) t[m][pos * 8 + b] = static_cast<uint8_t>(lane * 8 + b);
                    ++pos;
                }
            }
            for (int rem = pos * 8; rem < 16; ++rem) t[m][rem] = 0x80;
        }
        return t;
    }();
    return reinterpret_cast<const uint8_t*>(tbl.data());
}

inline const uint8_t* bmm_detail_sse_compress_lut() noexcept {
    static const auto tbl = []() noexcept {
        std::array<std::array<uint8_t, 16>, 16> t{};
        for (int m = 0; m < 16; ++m) {
            int pos = 0;
            for (int lane = 0; lane < 4; ++lane) {
                if (m & (1 << lane)) {
                    for (int b = 0; b < 4; ++b) t[m][pos * 4 + b] = static_cast<uint8_t>(lane * 4 + b);
                    ++pos;
                }
            }
            for (int rem = pos * 4; rem < 16; ++rem) t[m][rem] = 0x80; // zero byte
        }
        return t;
    }();
    return reinterpret_cast<const uint8_t*>(tbl.data());
}
#endif

// ===========================================================================
// AVX2 implementations
// ===========================================================================
#if BOLT_SIMD_AVX2

// f32 vector type + basics
using bmm_vec_f32 = __m256;
inline constexpr int bmm_lanes_f32 = 8;

BOLT_FORCE_INLINE bmm_vec_f32 bmm_load_f32 (const float* p) noexcept { return _mm256_load_ps(p); }
BOLT_FORCE_INLINE bmm_vec_f32 bmm_loadu_f32(const float* p) noexcept { return _mm256_loadu_ps(p); }
BOLT_FORCE_INLINE void bmm_store_f32 (float* p, bmm_vec_f32 v) noexcept { _mm256_store_ps(p, v); }
BOLT_FORCE_INLINE void bmm_storeu_f32(float* p, bmm_vec_f32 v) noexcept { _mm256_storeu_ps(p, v); }
BOLT_FORCE_INLINE bmm_vec_f32 bmm_set1_f32(float x) noexcept { return _mm256_set1_ps(x); }
BOLT_FORCE_INLINE bmm_vec_f32 bmm_cmpgt_f32(bmm_vec_f32 a, bmm_vec_f32 b) noexcept { return _mm256_cmp_ps(a, b, _CMP_GT_OQ); }
BOLT_FORCE_INLINE bmm_vec_f32 bmm_cmpeq_f32(bmm_vec_f32 a, bmm_vec_f32 b) noexcept { return _mm256_cmp_ps(a, b, _CMP_EQ_OQ); }
BOLT_FORCE_INLINE int bmm_movemask_f32(bmm_vec_f32 v) noexcept { return _mm256_movemask_ps(v); }

// f32 accumulator primitives (A5 — sum kernel support)
BOLT_FORCE_INLINE bmm_vec_f32 bmm_setzero_f32() noexcept { return _mm256_setzero_ps(); }
BOLT_FORCE_INLINE bmm_vec_f32 bmm_add_f32(bmm_vec_f32 a, bmm_vec_f32 b) noexcept { return _mm256_add_ps(a, b); }
BOLT_FORCE_INLINE float bmm_hadd_f32(bmm_vec_f32 v) noexcept {
    // 8-lane → 4-lane reduce via hi+lo halves, then scalar-pair reduce.
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s4 = _mm_add_ps(lo, hi);
    __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 0x1));
    return _mm_cvtss_f32(s1);
}

// i8 vector (32 lanes on AVX2)
using bmm_vec_i8 = __m256i;
inline constexpr int bmm_lanes_i8 = 32;
BOLT_FORCE_INLINE bmm_vec_i8 bmm_loadu_i8(const int8_t* p) noexcept {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
}
BOLT_FORCE_INLINE bmm_vec_i8 bmm_set1_i8(int8_t x) noexcept { return _mm256_set1_epi8(x); }
BOLT_FORCE_INLINE bmm_vec_i8 bmm_cmpeq_i8(bmm_vec_i8 a, bmm_vec_i8 b) noexcept { return _mm256_cmpeq_epi8(a, b); }
BOLT_FORCE_INLINE int bmm_movemask_i8(bmm_vec_i8 v) noexcept { return _mm256_movemask_epi8(v); }

// Gather
BOLT_FORCE_INLINE bmm_vec_i32 bmm_gather_i32(const int32_t* base, bmm_vec_i32 idx) noexcept {
    return _mm256_i32gather_epi32(base, idx, 4);
}
BOLT_FORCE_INLINE bmm_vec_i64 bmm_gather_i64(const int64_t* base, bmm_vec_i32 idx) noexcept {
    // Only the low 128 bits (4 int32 indices) are consumed.
    __m128i lo = _mm256_castsi256_si128(idx);
    return _mm256_i32gather_epi64(reinterpret_cast<const long long*>(base), lo, 8);
}

// Horizontal add
BOLT_FORCE_INLINE int64_t bmm_hadd_i64(bmm_vec_i64 v) noexcept {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i sum2 = _mm_add_epi64(lo, hi);
    __m128i sh = _mm_unpackhi_epi64(sum2, sum2);
    __m128i s = _mm_add_epi64(sum2, sh);
    alignas(16) int64_t out[2];
    _mm_store_si128(reinterpret_cast<__m128i*>(out), s);
    return out[0];
}
BOLT_FORCE_INLINE double bmm_hadd_f64(bmm_vec_f64 v) noexcept {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d s2 = _mm_add_pd(lo, hi);
    __m128d sh = _mm_unpackhi_pd(s2, s2);
    __m128d s  = _mm_add_sd(s2, sh);
    return _mm_cvtsd_f64(s);
}

// Shuffle / blend
BOLT_FORCE_INLINE bmm_vec_i32 bmm_shuffle_i32(bmm_vec_i32 a, bmm_vec_i32 idx) noexcept {
    return _mm256_permutevar8x32_epi32(a, idx);
}
BOLT_FORCE_INLINE bmm_vec_i32 bmm_blend_i32(bmm_vec_i32 a, bmm_vec_i32 b, bmm_vec_i32 cond) noexcept {
    return _mm256_blendv_epi8(a, b, cond);
}

// Compress-store
BOLT_FORCE_INLINE uint32_t bmm_compressstore_i32(int32_t* out, bmm_vec_i32 data, uint32_t mask) noexcept {
    const uint32_t m = mask & 0xffu;
#if defined(__AVX512F__) && defined(__AVX512VL__)
    _mm256_mask_compressstoreu_epi32(out, static_cast<__mmask8>(m), data);
#else
    const int32_t* lut = bmm_detail_avx2_compress_lut();
    __m256i perm = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lut + m * 8));
    __m256i compact = _mm256_permutevar8x32_epi32(data, perm);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), compact);
#endif
    return bolt_popcount32(m);
}

// AVX2 compress-store for 4-lane i64/f64: permutevar8x32 on the 32-bit view
// using a 16-entry LUT (mask ∈ [0,16)).
BOLT_FORCE_INLINE uint32_t bmm_compressstore_i64(int64_t* out, bmm_vec_i64 data, uint32_t mask) noexcept {
    assert(out != nullptr);
    const uint32_t m = mask & 0xfu;
    const int32_t* lut = bmm_detail_avx2_compress_lut4();
    __m256i perm = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lut + m * 8));
    __m256i compact = _mm256_permutevar8x32_epi32(data, perm);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), compact);
    assert(m < 16u);
    return bolt_popcount32(m);
}
BOLT_FORCE_INLINE uint32_t bmm_compressstore_f64(double* out, bmm_vec_f64 data, uint32_t mask) noexcept {
    assert(out != nullptr);
    const uint32_t m = mask & 0xfu;
    const int32_t* lut = bmm_detail_avx2_compress_lut4();
    __m256i perm = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lut + m * 8));
    __m256i compact = _mm256_permutevar8x32_epi32(_mm256_castpd_si256(data), perm);
    _mm256_storeu_pd(out, _mm256_castsi256_pd(compact));
    assert(m < 16u);
    return bolt_popcount32(m);
}

// ---------------------------------------------------------------------------
// AVX-512 native 512-bit specializations.
//
// Coexistence model:
//   - The 256-bit bmm_vec_i32 / bmm_lanes_i32 = 8 wrappers remain the
//     default on AVX-512 silicon (AVX-512 is a strict superset of AVX2,
//     so the AVX2 wrappers compile and execute correctly).
//   - The new bmm_vec_i32_x16 / bmm_lanes_i32_x16 = 16 symbols are opt-in
//     512-bit types for kernels that want the wider path.
//
// Native 512-bit ops provided here:
//   - Typedefs:       bmm_vec_i32_x16, bmm_vec_i64_x8, bmm_vec_f64_x8.
//   - Load/store:     bmm_loadu_i32_x16, bmm_storeu_i32_x16.
//   - Masked:         bmm_maskz_loadu_i32_x16, bmm_mask_store_i32_x16.
//   - Compare->mask:  bmm_cmpgt_i32_x16_mask, bmm_cmpeq_i32_x16_mask,
//                     bmm_cmpgt_i64_x8_mask (native __mmask16/__mmask8,
//                     no movemask round-trip).
//   - Compressstore:  bmm_compressstore_i32_x16, bmm_compressstore_i64_x8
//                     (single-instruction _mm512_mask_compressstoreu_*).
//   - Conflict:       bmm_conflict_i32_x16 (AVX-512CD; SwissTable probe
//                     WAW hazard detection).
//
// Everything else (shuffle/blend/gather i64/f64, hadd_i64/f64, i8 ops,
// f32 ops, all 256-bit bmm_* symbols) falls through to the AVX2 wrappers
// above. Those remain correct on AVX-512 silicon via the superset
// guarantee; they simply execute as 256-bit ops. Future waves widen
// them as warranted by kernel demand.
// ---------------------------------------------------------------------------
#if BOLT_SIMD_AVX512

using bmm_vec_i32_x16 = __m512i;
using bmm_vec_i64_x8  = __m512i;
using bmm_vec_f64_x8  = __m512d;
inline constexpr int bmm_lanes_i32_x16 = 16;
inline constexpr int bmm_lanes_i64_x8  = 8;
inline constexpr int bmm_lanes_f64_x8  = 8;

BOLT_FORCE_INLINE bmm_vec_i32_x16 bmm_loadu_i32_x16(const int32_t* p) noexcept {
    assert(p != nullptr);
    return _mm512_loadu_si512(reinterpret_cast<const void*>(p));
}
BOLT_FORCE_INLINE void bmm_storeu_i32_x16(int32_t* p, bmm_vec_i32_x16 v) noexcept {
    assert(p != nullptr);
    _mm512_storeu_si512(reinterpret_cast<void*>(p), v);
}

// Masked load/store: kill scalar tail loops on ragged morsel boundaries.
BOLT_FORCE_INLINE bmm_vec_i32_x16 bmm_maskz_loadu_i32_x16(const int32_t* p, uint16_t mask) noexcept {
    assert(p != nullptr);
    return _mm512_maskz_loadu_epi32(static_cast<__mmask16>(mask), p);
}
BOLT_FORCE_INLINE void bmm_mask_store_i32_x16(int32_t* p, uint16_t mask, bmm_vec_i32_x16 v) noexcept {
    assert(p != nullptr);
    _mm512_mask_storeu_epi32(p, static_cast<__mmask16>(mask), v);
}

// Compare-to-mask: native __mmaskN — no movemask round-trip.
BOLT_FORCE_INLINE uint16_t bmm_cmpgt_i32_x16_mask(bmm_vec_i32_x16 a, bmm_vec_i32_x16 b) noexcept {
    return static_cast<uint16_t>(_mm512_cmpgt_epi32_mask(a, b));
}
BOLT_FORCE_INLINE uint16_t bmm_cmpeq_i32_x16_mask(bmm_vec_i32_x16 a, bmm_vec_i32_x16 b) noexcept {
    return static_cast<uint16_t>(_mm512_cmpeq_epi32_mask(a, b));
}
BOLT_FORCE_INLINE uint8_t bmm_cmpgt_i64_x8_mask(bmm_vec_i64_x8 a, bmm_vec_i64_x8 b) noexcept {
    return static_cast<uint8_t>(_mm512_cmpgt_epi64_mask(a, b));
}

// Native single-instruction compressstore. Returns popcount of the mask.
BOLT_FORCE_INLINE uint32_t bmm_compressstore_i32_x16(int32_t* out, bmm_vec_i32_x16 data, uint16_t mask) noexcept {
    assert(out != nullptr);
    _mm512_mask_compressstoreu_epi32(out, static_cast<__mmask16>(mask), data);
    const uint32_t n = bolt_popcount32(static_cast<uint32_t>(mask));
    assert(n <= 16u);
    return n;
}
BOLT_FORCE_INLINE uint32_t bmm_compressstore_i64_x8(int64_t* out, bmm_vec_i64_x8 data, uint8_t mask) noexcept {
    assert(out != nullptr);
    _mm512_mask_compressstoreu_epi64(out, static_cast<__mmask8>(mask), data);
    const uint32_t n = bolt_popcount32(static_cast<uint32_t>(mask));
    assert(n <= 8u);
    return n;
}

// Conflict detection (AVX-512CD): per-lane mask of earlier lanes holding
// the same value. Used by vectorized SwissTable / groupby to detect WAW
// hazards within a single probe vector.
BOLT_FORCE_INLINE bmm_vec_i32_x16 bmm_conflict_i32_x16(bmm_vec_i32_x16 v) noexcept {
    return _mm512_conflict_epi32(v);
}

#endif  // BOLT_SIMD_AVX512 native block

// ===========================================================================
// SSE4.2 implementations
// ===========================================================================
#elif BOLT_SIMD_SSE42

using bmm_vec_f32 = __m128;
inline constexpr int bmm_lanes_f32 = 4;
BOLT_FORCE_INLINE bmm_vec_f32 bmm_load_f32 (const float* p) noexcept { return _mm_load_ps(p); }
BOLT_FORCE_INLINE bmm_vec_f32 bmm_loadu_f32(const float* p) noexcept { return _mm_loadu_ps(p); }
BOLT_FORCE_INLINE void bmm_store_f32 (float* p, bmm_vec_f32 v) noexcept { _mm_store_ps(p, v); }
BOLT_FORCE_INLINE void bmm_storeu_f32(float* p, bmm_vec_f32 v) noexcept { _mm_storeu_ps(p, v); }
BOLT_FORCE_INLINE bmm_vec_f32 bmm_set1_f32(float x) noexcept { return _mm_set1_ps(x); }
BOLT_FORCE_INLINE bmm_vec_f32 bmm_cmpgt_f32(bmm_vec_f32 a, bmm_vec_f32 b) noexcept { return _mm_cmpgt_ps(a, b); }
BOLT_FORCE_INLINE bmm_vec_f32 bmm_cmpeq_f32(bmm_vec_f32 a, bmm_vec_f32 b) noexcept { return _mm_cmpeq_ps(a, b); }
BOLT_FORCE_INLINE int bmm_movemask_f32(bmm_vec_f32 v) noexcept { return _mm_movemask_ps(v); }
BOLT_FORCE_INLINE bmm_vec_f32 bmm_setzero_f32() noexcept { return _mm_setzero_ps(); }
BOLT_FORCE_INLINE bmm_vec_f32 bmm_add_f32(bmm_vec_f32 a, bmm_vec_f32 b) noexcept { return _mm_add_ps(a, b); }
BOLT_FORCE_INLINE float bmm_hadd_f32(bmm_vec_f32 v) noexcept {
    __m128 s2 = _mm_add_ps(v, _mm_movehl_ps(v, v));
    __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 0x1));
    return _mm_cvtss_f32(s1);
}

using bmm_vec_i8 = __m128i;
inline constexpr int bmm_lanes_i8 = 16;
BOLT_FORCE_INLINE bmm_vec_i8 bmm_loadu_i8(const int8_t* p) noexcept { return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p)); }
BOLT_FORCE_INLINE bmm_vec_i8 bmm_set1_i8(int8_t x) noexcept { return _mm_set1_epi8(x); }
BOLT_FORCE_INLINE bmm_vec_i8 bmm_cmpeq_i8(bmm_vec_i8 a, bmm_vec_i8 b) noexcept { return _mm_cmpeq_epi8(a, b); }
BOLT_FORCE_INLINE int bmm_movemask_i8(bmm_vec_i8 v) noexcept { return _mm_movemask_epi8(v); }

BOLT_FORCE_INLINE bmm_vec_i32 bmm_gather_i32(const int32_t* base, bmm_vec_i32 idx) noexcept {
    alignas(16) int32_t ix[4]; _mm_store_si128(reinterpret_cast<__m128i*>(ix), idx);
    return _mm_set_epi32(base[ix[3]], base[ix[2]], base[ix[1]], base[ix[0]]);
}
BOLT_FORCE_INLINE bmm_vec_i64 bmm_gather_i64(const int64_t* base, bmm_vec_i32 idx) noexcept {
    alignas(16) int32_t ix[4]; _mm_store_si128(reinterpret_cast<__m128i*>(ix), idx);
    return _mm_set_epi64x(base[ix[1]], base[ix[0]]);
}

BOLT_FORCE_INLINE int64_t bmm_hadd_i64(bmm_vec_i64 v) noexcept {
    alignas(16) int64_t tmp[2]; _mm_store_si128(reinterpret_cast<__m128i*>(tmp), v);
    return tmp[0] + tmp[1];
}
BOLT_FORCE_INLINE double bmm_hadd_f64(bmm_vec_f64 v) noexcept {
    __m128d sh = _mm_unpackhi_pd(v, v);
    __m128d s  = _mm_add_sd(v, sh);
    return _mm_cvtsd_f64(s);
}

BOLT_FORCE_INLINE bmm_vec_i32 bmm_shuffle_i32(bmm_vec_i32 a, bmm_vec_i32 idx) noexcept {
    // idx has 4 int32 lane indices (0..3). Build a byte-shuffle control.
    alignas(16) int32_t lane_idx[4]; _mm_store_si128(reinterpret_cast<__m128i*>(lane_idx), idx);
    alignas(16) uint8_t ctrl[16];
    for (int i = 0; i < 4; ++i) {
        int li = lane_idx[i] & 3;
        for (int b = 0; b < 4; ++b) ctrl[i * 4 + b] = static_cast<uint8_t>(li * 4 + b);
    }
    __m128i c = _mm_load_si128(reinterpret_cast<const __m128i*>(ctrl));
    return _mm_shuffle_epi8(a, c);
}
BOLT_FORCE_INLINE bmm_vec_i32 bmm_blend_i32(bmm_vec_i32 a, bmm_vec_i32 b, bmm_vec_i32 cond) noexcept {
    return _mm_blendv_epi8(a, b, cond);
}

BOLT_FORCE_INLINE uint32_t bmm_compressstore_i32(int32_t* out, bmm_vec_i32 data, uint32_t mask) noexcept {
    const uint32_t m = mask & 0xfu;
    const uint8_t* lut = bmm_detail_sse_compress_lut();
    __m128i ctrl = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lut + m * 16));
    __m128i compact = _mm_shuffle_epi8(data, ctrl);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out), compact);
    return bolt_popcount32(m);
}

// SSE compress-store for 2-lane i64/f64: pshufb on a 4-entry byte-shuffle LUT.
BOLT_FORCE_INLINE uint32_t bmm_compressstore_i64(int64_t* out, bmm_vec_i64 data, uint32_t mask) noexcept {
    assert(out != nullptr);
    const uint32_t m = mask & 0x3u;
    const uint8_t* lut = bmm_detail_sse_compress_lut2();
    __m128i ctrl = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lut + m * 16));
    __m128i compact = _mm_shuffle_epi8(data, ctrl);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out), compact);
    assert(m < 4u);
    return bolt_popcount32(m);
}
BOLT_FORCE_INLINE uint32_t bmm_compressstore_f64(double* out, bmm_vec_f64 data, uint32_t mask) noexcept {
    assert(out != nullptr);
    const uint32_t m = mask & 0x3u;
    const uint8_t* lut = bmm_detail_sse_compress_lut2();
    __m128i ctrl = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lut + m * 16));
    __m128i compact = _mm_shuffle_epi8(_mm_castpd_si128(data), ctrl);
    _mm_storeu_pd(out, _mm_castsi128_pd(compact));
    assert(m < 4u);
    return bolt_popcount32(m);
}

// ===========================================================================
// NEON implementations
// ===========================================================================
#elif BOLT_SIMD_NEON

using bmm_vec_f32 = float32x4_t;
inline constexpr int bmm_lanes_f32 = 4;
BOLT_FORCE_INLINE bmm_vec_f32 bmm_load_f32 (const float* p) noexcept { return vld1q_f32(p); }
BOLT_FORCE_INLINE bmm_vec_f32 bmm_loadu_f32(const float* p) noexcept { return vld1q_f32(p); }
BOLT_FORCE_INLINE void bmm_store_f32 (float* p, bmm_vec_f32 v) noexcept { vst1q_f32(p, v); }
BOLT_FORCE_INLINE void bmm_storeu_f32(float* p, bmm_vec_f32 v) noexcept { vst1q_f32(p, v); }
BOLT_FORCE_INLINE bmm_vec_f32 bmm_set1_f32(float x) noexcept { return vdupq_n_f32(x); }
BOLT_FORCE_INLINE bmm_vec_f32 bmm_cmpgt_f32(bmm_vec_f32 a, bmm_vec_f32 b) noexcept { return vreinterpretq_f32_u32(vcgtq_f32(a, b)); }
BOLT_FORCE_INLINE bmm_vec_f32 bmm_cmpeq_f32(bmm_vec_f32 a, bmm_vec_f32 b) noexcept { return vreinterpretq_f32_u32(vceqq_f32(a, b)); }
BOLT_FORCE_INLINE int bmm_movemask_f32(bmm_vec_f32 v) noexcept {
    uint32x4_t u = vreinterpretq_u32_f32(v);
    alignas(16) static const int32_t kShift[4] = {0, 1, 2, 3};
    int32x4_t shift = vld1q_s32(kShift);
    uint32x4_t bits = vshlq_u32(vshrq_n_u32(u, 31), shift);
    return static_cast<int>(vaddvq_u32(bits));
}
BOLT_FORCE_INLINE bmm_vec_f32 bmm_setzero_f32() noexcept { return vdupq_n_f32(0.0f); }
BOLT_FORCE_INLINE bmm_vec_f32 bmm_add_f32(bmm_vec_f32 a, bmm_vec_f32 b) noexcept { return vaddq_f32(a, b); }
BOLT_FORCE_INLINE float bmm_hadd_f32(bmm_vec_f32 v) noexcept { return vaddvq_f32(v); }

using bmm_vec_i8 = int8x16_t;
inline constexpr int bmm_lanes_i8 = 16;
BOLT_FORCE_INLINE bmm_vec_i8 bmm_loadu_i8(const int8_t* p) noexcept { return vld1q_s8(p); }
BOLT_FORCE_INLINE bmm_vec_i8 bmm_set1_i8(int8_t x) noexcept { return vdupq_n_s8(x); }
BOLT_FORCE_INLINE bmm_vec_i8 bmm_cmpeq_i8(bmm_vec_i8 a, bmm_vec_i8 b) noexcept { return vreinterpretq_s8_u8(vceqq_s8(a, b)); }
BOLT_FORCE_INLINE int bmm_movemask_i8(bmm_vec_i8 v) noexcept {
    // Extract top bit of each of 16 bytes into a 16-bit mask.
    uint8x16_t u = vreinterpretq_u8_s8(v);
    alignas(16) static const int8_t kShift[16] = {0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7};
    int8x16_t shift = vld1q_s8(kShift);
    uint8x16_t bits = vshlq_u8(vshrq_n_u8(u, 7), shift);
    uint8x8_t lo = vget_low_u8(bits);
    uint8x8_t hi = vget_high_u8(bits);
    uint32_t lo8 = vaddv_u8(lo);
    uint32_t hi8 = vaddv_u8(hi);
    return static_cast<int>(lo8 | (hi8 << 8));
}

BOLT_FORCE_INLINE bmm_vec_i32 bmm_gather_i32(const int32_t* base, bmm_vec_i32 idx) noexcept {
    alignas(16) int32_t ix[4]; vst1q_s32(ix, idx);
    int32_t r[4] = { base[ix[0]], base[ix[1]], base[ix[2]], base[ix[3]] };
    return vld1q_s32(r);
}
BOLT_FORCE_INLINE bmm_vec_i64 bmm_gather_i64(const int64_t* base, bmm_vec_i32 idx) noexcept {
    alignas(16) int32_t ix[4]; vst1q_s32(ix, idx);
    int64_t r[2] = { base[ix[0]], base[ix[1]] };
    return vld1q_s64(r);
}

BOLT_FORCE_INLINE int64_t bmm_hadd_i64(bmm_vec_i64 v) noexcept { return vaddvq_s64(v); }
BOLT_FORCE_INLINE double  bmm_hadd_f64(bmm_vec_f64 v) noexcept { return vaddvq_f64(v); }

BOLT_FORCE_INLINE bmm_vec_i32 bmm_shuffle_i32(bmm_vec_i32 a, bmm_vec_i32 idx) noexcept {
    alignas(16) int32_t lane_idx[4]; vst1q_s32(lane_idx, idx);
    alignas(16) uint8_t ctrl[16];
    for (int i = 0; i < 4; ++i) {
        int li = lane_idx[i] & 3;
        for (int b = 0; b < 4; ++b) ctrl[i * 4 + b] = static_cast<uint8_t>(li * 4 + b);
    }
    uint8x16_t c = vld1q_u8(ctrl);
    uint8x16_t abytes = vreinterpretq_u8_s32(a);
    return vreinterpretq_s32_u8(vqtbl1q_u8(abytes, c));
}
BOLT_FORCE_INLINE bmm_vec_i32 bmm_blend_i32(bmm_vec_i32 a, bmm_vec_i32 b, bmm_vec_i32 cond) noexcept {
    uint32x4_t m = vreinterpretq_u32_s32(cond);
    return vbslq_s32(m, b, a);
}

BOLT_FORCE_INLINE uint32_t bmm_compressstore_i32(int32_t* out, bmm_vec_i32 data, uint32_t mask) noexcept {
    alignas(16) int32_t tmp[4]; vst1q_s32(tmp, data);
    uint32_t m = mask & 0xfu;
    uint32_t count = 0;
    while (m) {
        int p = bolt_ctz32(m);
        out[count++] = tmp[p];
        m &= m - 1;
    }
    return count;
}

// NEON compress-store for 2-lane i64/f64: trivial ctz loop (2 lanes).
BOLT_FORCE_INLINE uint32_t bmm_compressstore_i64(int64_t* out, bmm_vec_i64 data, uint32_t mask) noexcept {
    assert(out != nullptr);
    alignas(16) int64_t tmp[2]; vst1q_s64(tmp, data);
    uint32_t m = mask & 0x3u;
    uint32_t count = 0;
    while (m) {
        int p = bolt_ctz32(m);
        out[count++] = tmp[p];
        m &= m - 1;
    }
    assert(count <= 2u);
    return count;
}
BOLT_FORCE_INLINE uint32_t bmm_compressstore_f64(double* out, bmm_vec_f64 data, uint32_t mask) noexcept {
    assert(out != nullptr);
    alignas(16) double tmp[2]; vst1q_f64(tmp, data);
    uint32_t m = mask & 0x3u;
    uint32_t count = 0;
    while (m) {
        int p = bolt_ctz32(m);
        out[count++] = tmp[p];
        m &= m - 1;
    }
    assert(count <= 2u);
    return count;
}

// ===========================================================================
// Scalar implementations
// ===========================================================================
#else  // BOLT_SIMD_SCALAR

struct bmm_vec_f32 { std::array<float, 4> v; };
inline constexpr int bmm_lanes_f32 = 4;
BOLT_FORCE_INLINE bmm_vec_f32 bmm_load_f32 (const float* p) noexcept { bmm_vec_f32 r; for (int i=0;i<4;++i) r.v[i]=p[i]; return r; }
BOLT_FORCE_INLINE bmm_vec_f32 bmm_loadu_f32(const float* p) noexcept { return bmm_load_f32(p); }
BOLT_FORCE_INLINE void bmm_store_f32 (float* p, bmm_vec_f32 v) noexcept { for (int i=0;i<4;++i) p[i]=v.v[i]; }
BOLT_FORCE_INLINE void bmm_storeu_f32(float* p, bmm_vec_f32 v) noexcept { bmm_store_f32(p, v); }
BOLT_FORCE_INLINE bmm_vec_f32 bmm_set1_f32(float x) noexcept { return bmm_vec_f32{{x,x,x,x}}; }
BOLT_FORCE_INLINE bmm_vec_f32 bmm_cmpgt_f32(bmm_vec_f32 a, bmm_vec_f32 b) noexcept {
    bmm_vec_f32 r;
    for (int i=0;i<4;++i) {
        uint32_t bits = (a.v[i] > b.v[i]) ? 0xFFFFFFFFu : 0u;
        std::memcpy(&r.v[i], &bits, sizeof(float));
    }
    return r;
}
BOLT_FORCE_INLINE bmm_vec_f32 bmm_cmpeq_f32(bmm_vec_f32 a, bmm_vec_f32 b) noexcept {
    bmm_vec_f32 r;
    for (int i=0;i<4;++i) {
        uint32_t bits = (a.v[i] == b.v[i]) ? 0xFFFFFFFFu : 0u;
        std::memcpy(&r.v[i], &bits, sizeof(float));
    }
    return r;
}
BOLT_FORCE_INLINE int bmm_movemask_f32(bmm_vec_f32 v) noexcept {
    int m = 0;
    for (int i = 0; i < 4; ++i) {
        uint32_t bits; std::memcpy(&bits, &v.v[i], sizeof(float));
        m |= ((bits >> 31) & 1u) << i;
    }
    return m;
}
BOLT_FORCE_INLINE bmm_vec_f32 bmm_setzero_f32() noexcept { return bmm_vec_f32{{0.0f,0.0f,0.0f,0.0f}}; }
BOLT_FORCE_INLINE bmm_vec_f32 bmm_add_f32(bmm_vec_f32 a, bmm_vec_f32 b) noexcept {
    bmm_vec_f32 r; for (int i=0;i<4;++i) r.v[i] = a.v[i] + b.v[i]; return r;
}
BOLT_FORCE_INLINE float bmm_hadd_f32(bmm_vec_f32 v) noexcept { return v.v[0]+v.v[1]+v.v[2]+v.v[3]; }

struct bmm_vec_i8 { std::array<int8_t, 16> v; };
inline constexpr int bmm_lanes_i8 = 16;
BOLT_FORCE_INLINE bmm_vec_i8 bmm_loadu_i8(const int8_t* p) noexcept { bmm_vec_i8 r; for (int i=0;i<16;++i) r.v[i]=p[i]; return r; }
BOLT_FORCE_INLINE bmm_vec_i8 bmm_set1_i8(int8_t x) noexcept { bmm_vec_i8 r; for (int i=0;i<16;++i) r.v[i]=x; return r; }
BOLT_FORCE_INLINE bmm_vec_i8 bmm_cmpeq_i8(bmm_vec_i8 a, bmm_vec_i8 b) noexcept {
    bmm_vec_i8 r; for (int i=0;i<16;++i) r.v[i] = (a.v[i] == b.v[i]) ? static_cast<int8_t>(-1) : 0; return r;
}
BOLT_FORCE_INLINE int bmm_movemask_i8(bmm_vec_i8 v) noexcept {
    int m = 0;
    for (int i = 0; i < 16; ++i) m |= ((static_cast<uint8_t>(v.v[i]) >> 7) & 1u) << i;
    return m;
}

BOLT_FORCE_INLINE bmm_vec_i32 bmm_gather_i32(const int32_t* base, bmm_vec_i32 idx) noexcept {
    bmm_vec_i32 r; for (int i=0;i<4;++i) r.v[i] = base[idx.v[i]]; return r;
}
BOLT_FORCE_INLINE bmm_vec_i64 bmm_gather_i64(const int64_t* base, bmm_vec_i32 idx) noexcept {
    bmm_vec_i64 r; for (int i=0;i<4;++i) r.v[i] = base[idx.v[i]]; return r;
}

BOLT_FORCE_INLINE int64_t bmm_hadd_i64(bmm_vec_i64 v) noexcept {
    int64_t s = 0; for (int i=0;i<4;++i) s += v.v[i]; return s;
}
BOLT_FORCE_INLINE double bmm_hadd_f64(bmm_vec_f64 v) noexcept {
    double s = 0.0; for (int i=0;i<4;++i) s += v.v[i]; return s;
}

BOLT_FORCE_INLINE bmm_vec_i32 bmm_shuffle_i32(bmm_vec_i32 a, bmm_vec_i32 idx) noexcept {
    bmm_vec_i32 r; for (int i=0;i<4;++i) r.v[i] = a.v[idx.v[i] & 3]; return r;
}
BOLT_FORCE_INLINE bmm_vec_i32 bmm_blend_i32(bmm_vec_i32 a, bmm_vec_i32 b, bmm_vec_i32 cond) noexcept {
    bmm_vec_i32 r; for (int i=0;i<4;++i) r.v[i] = cond.v[i] ? b.v[i] : a.v[i]; return r;
}

BOLT_FORCE_INLINE uint32_t bmm_compressstore_i32(int32_t* out, bmm_vec_i32 data, uint32_t mask) noexcept {
    uint32_t m = mask & 0xfu;
    uint32_t count = 0;
    while (m) {
        int p = bolt_ctz32(m);
        out[count++] = data.v[p];
        m &= m - 1;
    }
    return count;
}

// Scalar compress-store for 4-lane i64/f64 (mirrors scalar i32).
BOLT_FORCE_INLINE uint32_t bmm_compressstore_i64(int64_t* out, bmm_vec_i64 data, uint32_t mask) noexcept {
    assert(out != nullptr);
    uint32_t m = mask & 0xfu;
    uint32_t count = 0;
    while (m) {
        int p = bolt_ctz32(m);
        out[count++] = data.v[p];
        m &= m - 1;
    }
    assert(count <= 4u);
    return count;
}
BOLT_FORCE_INLINE uint32_t bmm_compressstore_f64(double* out, bmm_vec_f64 data, uint32_t mask) noexcept {
    assert(out != nullptr);
    uint32_t m = mask & 0xfu;
    uint32_t count = 0;
    while (m) {
        int p = bolt_ctz32(m);
        out[count++] = data.v[p];
        m &= m - 1;
    }
    assert(count <= 4u);
    return count;
}

#endif  // Wave D2 extensions ISA dispatch

// ---- ISA-independent helpers ----------------------------------------------
BOLT_FORCE_INLINE uint32_t bmm_popcount_mask(uint32_t m) noexcept {
    return bolt_popcount32(m);
}

} // namespace simd

// ---------------------------------------------------------------------------
// bolt_memcpy_nt — non-temporal streaming copy for buffers larger than L3.
//
// Standalone primitive kept alongside plain `memcpy`.  The default COW /
// clone_into paths still use `memcpy` (cache-friendly for buffers that
// the caller plans to read back).  `bolt_memcpy_nt` is for one-shot
// large transfers (persisted materialisations, spill buffers, arena
// evictions) where the destination will NOT be re-read soon — streaming
// stores bypass the cache and save L2/L3 eviction cost.
//
// Default threshold: 8 MB (~half of a modern-laptop L3).  Below that,
// caller should prefer `memcpy` — NT stores beat memcpy only when the
// destination isn't about to be reloaded.  Callers override by measuring.
//
// AVX2: `_mm256_stream_si256` on 64-byte-aligned chunks, scalar tail.
// Non-AVX2 / non-x86: falls back to plain `memcpy` (memcpy is already
// well-optimised on those paths).
// ---------------------------------------------------------------------------

inline constexpr size_t kBoltMemcpyNtThreshold = 8u * 1024u * 1024u;

BOLT_FORCE_INLINE void bolt_memcpy_nt(void* BOLT_RESTRICT dst,
                                      const void* BOLT_RESTRICT src,
                                      size_t bytes) noexcept {
    assert(dst != nullptr || bytes == 0);
    assert(src != nullptr || bytes == 0);
#if BOLT_SIMD_AVX2 || BOLT_SIMD_AVX512
    // Require 32-byte alignment for NT stores; if callers can't meet it
    // fall back to memcpy (hot-path asserts would be wrong here — the
    // fallback is a correctness, not a perf, requirement).
    const uintptr_t a = reinterpret_cast<uintptr_t>(dst);
    if ((a & 31u) != 0u) { memcpy(dst, src, bytes); return; }

    uint8_t*       d = static_cast<uint8_t*>(dst);
    const uint8_t* s = static_cast<const uint8_t*>(src);
    size_t i = 0;
    // 128-byte chunks → 4 × _mm256_stream_si256 per iter (one cache line each).
    for (; i + 128u <= bytes; i += 128u) {
        __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i +  0));
        __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i + 32));
        __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i + 64));
        __m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i + 96));
        _mm256_stream_si256(reinterpret_cast<__m256i*>(d + i +  0), v0);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(d + i + 32), v1);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(d + i + 64), v2);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(d + i + 96), v3);
    }
    // Scalar tail for sub-128-byte remainder.
    if (i < bytes) memcpy(d + i, s + i, bytes - i);
    _mm_sfence();   // order NT stores before any subsequent loads
#else
    memcpy(dst, src, bytes);
#endif
}

} // namespace bolt

// ---------------------------------------------------------------------------
// Not-yet-implemented marker: noisy at runtime, greppable at build time
// ---------------------------------------------------------------------------
#include <cassert>
#define BOLT_NYI(msg) assert(false && "BOLT_NYI: " msg)

// ---------------------------------------------------------------------------
// extern "C" convenience (for C-callable wire / arrow surfaces)
// ---------------------------------------------------------------------------
#ifdef __cplusplus
    #define BOLT_C_API extern "C"
#else
    #define BOLT_C_API
#endif
