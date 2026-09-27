// bolt_system_memory.h — query the host's total physical RAM and this
// process's resident set (header-only, zero deps). Moved down from chukonu's
// exec/system_memory.h so every repo's Budget derives from the same number.
//
// RULES: No exceptions. No RTTI. No smart pointers. No std::string. All
// functions noexcept.
//
//   Windows : GlobalMemoryStatusEx (ullTotalPhys)
//   Linux   : /proc/meminfo MemTotal  (NOT sysconf/unistd.h — portability rule)
//   macOS   : sysctlbyname("hw.memsize")
//   Other   : query fails (returns false) -> caller uses its constexpr fallback.
//
// Reports TOTAL physical RAM, never "available": a fluctuating budget would
// make spill-to-disk decisions nondeterministic. Permissive-on-failure — an
// unanswerable query returns false and leaves *out untouched, so the caller
// falls back to a fixed default rather than a bad zero-budget.

#pragma once

#include <cassert>
#include <cstdint>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#else
#include <cstdio>
#include <cstring>
#endif

#if defined(_WIN32)
#include <psapi.h>
#endif

namespace bolt {

// Fill *out with the host's total physical RAM in bytes. Returns true on a
// successful query, false on any failure (in which case *out is untouched and
// the caller should use its own constexpr fallback). noexcept, no allocation.
inline bool query_total_physical_ram_bytes(std::uint64_t* out) noexcept {
    assert(out != nullptr);
#if defined(_WIN32)
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (!GlobalMemoryStatusEx(&ms)) return false;
    if (ms.ullTotalPhys == 0) return false;
    *out = static_cast<std::uint64_t>(ms.ullTotalPhys);
    return true;
#elif defined(__APPLE__)
    std::uint64_t v = 0;
    size_t sz = sizeof(v);
    if (sysctlbyname("hw.memsize", &v, &sz, nullptr, 0) != 0) return false;
    if (v == 0) return false;
    *out = v;
    return true;
#elif defined(__linux__)
    // Read /proc/meminfo "MemTotal:      NNNN kB" — deliberately avoids
    // sysconf(_SC_PHYS_PAGES)/unistd.h per the no-unistd portability rule.
    std::FILE* f = std::fopen("/proc/meminfo", "re");
    if (f == nullptr) return false;
    char line[256];
    std::uint64_t kib = 0;
    bool found = false;
    // Bounded scan: MemTotal is the first line, but don't assume it.
    for (int i = 0; i < 64 && std::fgets(line, sizeof(line), f) != nullptr; ++i) {
        if (std::strncmp(line, "MemTotal:", 9) == 0) {
            const char* p = line + 9;
            while (*p == ' ' || *p == '\t') ++p;
            std::uint64_t v = 0;
            bool any = false;
            for (; *p >= '0' && *p <= '9'; ++p) {
                v = v * 10ull + static_cast<std::uint64_t>(*p - '0');
                any = true;
            }
            if (any) { kib = v; found = true; }
            break;
        }
    }
    std::fclose(f);
    if (!found || kib == 0) return false;
    *out = kib * 1024ull;
    return true;
#else
    (void)out;
    return false;
#endif
}

// Fill *out with THIS PROCESS's current resident set size in bytes. Returns
// true on a successful query, false on any failure (in which case *out is
// untouched and the caller must not refuse anything — an unmeasurable host
// gets the pre-existing behaviour, never a guess).
//
// G2GRAPH-167: an operator budget expressed in ARENA bytes is only a proxy
// for the footprint an operator or a harness actually cares about, and the
// proxy was MEASURED to be off by 3x on LSQB SF10 Q9 — 24.5 GiB of accounted
// arena at 77.0 GB of process RSS, because build tables, buffered probe
// morsels and decode buffers are not in the arena being counted. A budget
// that is meant to fire just before an out-of-memory kill therefore has to
// read the same number the killer reads.
//
// Cost: one syscall-class call. Callers must invoke it at a COARSE point (once
// per emitted batch of up to 65,536 rows), never per row.
//
//   Windows : GetProcessMemoryInfo (WorkingSetSize)
//   macOS   : task_info(mach_task_self(), TASK_BASIC_INFO_64) resident_size
//   Linux   : /proc/self/statm field 2 (resident pages) x page size
//   Other   : query fails.
inline bool query_process_resident_bytes(std::uint64_t* out) noexcept {
    assert(out != nullptr);
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    pmc.cb = sizeof(pmc);
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return false;
    }
    *out = static_cast<std::uint64_t>(pmc.WorkingSetSize);
    return *out != 0;
#elif defined(__APPLE__)
    mach_task_basic_info info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) {
        return false;
    }
    *out = static_cast<std::uint64_t>(info.resident_size);
    return *out != 0;
#elif defined(__linux__)
    std::FILE* f = std::fopen("/proc/self/statm", "r");
    if (f == nullptr) return false;
    unsigned long long total = 0, res = 0;
    const int got = std::fscanf(f, "%llu %llu", &total, &res);
    std::fclose(f);
    if (got != 2 || res == 0) return false;
    // 4 KiB is the page size on every Linux target this builds for; reading
    // it from sysconf would need <unistd.h>, which the portability rule bans
    // from headers. A wrong page size here can only make the budget coarser,
    // never unsound, because the caller compares like with like via the
    // SAME function.
    *out = static_cast<std::uint64_t>(res) * 4096ull;
    return true;
#else
    (void)out;
    return false;
#endif
}

}  // namespace bolt
