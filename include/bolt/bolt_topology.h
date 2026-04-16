// bolt_topology.h — Host CPU topology detection (header-only, zero deps).
//
// RULES: No exceptions. No RTTI. No smart pointers. No std::string. No heap
// outside the caller-provided struct (and a stack-only alloca for the Windows
// query buffer). All functions noexcept.
//
// Provides a POD CpuTopology + platform-specialised detection:
//   Windows : GetLogicalProcessorInformationEx (+ EfficiencyClass for hybrid)
//   Linux   : /sys/devices/system/cpu parsing (no libnuma)
//   macOS   : sysctlbyname (hw.perflevelN.physicalcpu)
//   Other   : fallback to a single-node, uniform P-core topology.

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>
#include <cstring>

namespace bolt {

inline constexpr uint32_t kTopologyMaxCpus = 256;

struct CpuTopology {
    uint32_t logical_cpus;       // total hardware threads
    uint32_t physical_cores;     // physical cores (0 if detection failed)
    uint32_t performance_cores;  // P-cores on hybrid CPUs; 0 if not detected / not hybrid
    uint32_t efficiency_cores;   // E-cores on hybrid CPUs; 0 if not detected / not hybrid
    uint32_t numa_nodes;         // always >= 1
    uint32_t cpu_to_node [kTopologyMaxCpus]; // CPU index -> NUMA node
    uint8_t  cpu_is_perf [kTopologyMaxCpus]; // 1 if P-core or uniform, 0 if E-core
};

// Forward decl for fallback (used by per-platform detectors on failure).
inline void bolt_topology_fallback(CpuTopology* out) noexcept;

namespace detail {

inline void topology_zero(CpuTopology* out) noexcept {
    assert(out != nullptr);
    out->logical_cpus      = 0;
    out->physical_cores    = 0;
    out->performance_cores = 0;
    out->efficiency_cores  = 0;
    out->numa_nodes        = 0;
    std::memset(out->cpu_to_node, 0, sizeof(out->cpu_to_node));
    std::memset(out->cpu_is_perf, 0, sizeof(out->cpu_is_perf));
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Fallback: single NUMA node, uniform P-cores.
// ---------------------------------------------------------------------------
inline void bolt_topology_fallback(CpuTopology* out) noexcept {
    assert(out != nullptr);
    assert(kTopologyMaxCpus >= 1);
    detail::topology_zero(out);

    uint32_t n = bolt_get_hardware_concurrency();
    if (n == 0) n = 1;
    if (n > kTopologyMaxCpus) n = kTopologyMaxCpus;

    out->logical_cpus      = n;
    out->physical_cores    = n;
    out->performance_cores = n;
    out->efficiency_cores  = 0;
    out->numa_nodes        = 1;
    for (uint32_t i = 0; i < n; ++i) {
        out->cpu_to_node[i] = 0;
        out->cpu_is_perf[i] = 1;
    }
}

// ---------------------------------------------------------------------------
// Windows
// ---------------------------------------------------------------------------
#ifdef _WIN32

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <malloc.h>  // _alloca

namespace detail {

// Iterate SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX records in a buffer.
// For each record of `rel`, invoke F(record).
template <typename F>
inline void walk_slpi_ex(const BYTE* buf, DWORD bytes,
                         LOGICAL_PROCESSOR_RELATIONSHIP rel, F&& fn) noexcept {
    DWORD off = 0;
    while (off < bytes) {
        const auto* info =
            reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf + off);
        if (info->Size == 0) break;  // safety
        if (info->Relationship == rel) fn(info);
        off += info->Size;
    }
}

inline bool detect_windows(CpuTopology* out) noexcept {
    assert(out != nullptr);
    detail::topology_zero(out);

    // --- hardware concurrency -------------------------------------------------
    uint32_t hc = bolt_get_hardware_concurrency();
    if (hc == 0) hc = 1;
    if (hc > kTopologyMaxCpus) hc = kTopologyMaxCpus;
    out->logical_cpus = hc;

    // --- cores + efficiency class --------------------------------------------
    DWORD need = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &need);
    if (need == 0) return false;
    BYTE* buf = static_cast<BYTE*>(_alloca(need));
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf), &need)) {
        return false;
    }

    uint32_t physical = 0;
    uint32_t max_eff  = 0;  // observed max EfficiencyClass
    // First pass: count cores, find max EfficiencyClass.
    detail::walk_slpi_ex(buf, need, RelationProcessorCore,
        [&](const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* info) noexcept {
            ++physical;
            // EfficiencyClass field is at info->Processor.EfficiencyClass;
            // on older SDKs (<22000) the field still exists in
            // PROCESSOR_RELATIONSHIP since Win10 1709. Safe to read.
            uint32_t eff = static_cast<uint32_t>(info->Processor.EfficiencyClass);
            if (eff > max_eff) max_eff = eff;
        });
    out->physical_cores = physical;

    // Second pass: assign cpu_is_perf + count P/E cores.
    uint32_t perf_cores = 0;
    uint32_t eff_cores  = 0;
    detail::walk_slpi_ex(buf, need, RelationProcessorCore,
        [&](const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* info) noexcept {
            const PROCESSOR_RELATIONSHIP& pr = info->Processor;
            // Hybrid detection: if max_eff == 0, treat all as P-cores.
            bool is_perf = (max_eff == 0)
                ? true
                : (static_cast<uint32_t>(pr.EfficiencyClass) == max_eff);
            if (is_perf) ++perf_cores; else ++eff_cores;
            // Stamp all logical CPUs covered by this core's group masks.
            for (WORD g = 0; g < pr.GroupCount; ++g) {
                const GROUP_AFFINITY& ga = pr.GroupMask[g];
                KAFFINITY m = ga.Mask;
                // Approximate absolute CPU index: group * 64 + bit.
                const uint32_t group_base = static_cast<uint32_t>(ga.Group) * 64u;
                for (uint32_t b = 0; b < 64u && m != 0; ++b) {
                    if (m & (KAFFINITY{1} << b)) {
                        uint32_t cpu = group_base + b;
                        if (cpu < kTopologyMaxCpus) {
                            out->cpu_is_perf[cpu] = is_perf ? 1u : 0u;
                        }
                        m &= ~(KAFFINITY{1} << b);
                    }
                }
            }
        });

    if (max_eff == 0) {
        out->performance_cores = physical;
        out->efficiency_cores  = 0;
    } else {
        out->performance_cores = perf_cores;
        out->efficiency_cores  = eff_cores;
    }

    // --- NUMA nodes -----------------------------------------------------------
    DWORD need2 = 0;
    GetLogicalProcessorInformationEx(RelationNumaNode, nullptr, &need2);
    if (need2 == 0) {
        out->numa_nodes = 1;
        for (uint32_t i = 0; i < kTopologyMaxCpus; ++i) out->cpu_to_node[i] = 0;
        return true;
    }
    BYTE* buf2 = static_cast<BYTE*>(_alloca(need2));
    if (!GetLogicalProcessorInformationEx(RelationNumaNode, reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf2), &need2)) {
        out->numa_nodes = 1;
        return true;
    }

    uint32_t nodes = 0;
    detail::walk_slpi_ex(buf2, need2, RelationNumaNode,
        [&](const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* info) noexcept {
            const NUMA_NODE_RELATIONSHIP& nn = info->NumaNode;
            uint32_t node_id = nn.NodeNumber;
            if (node_id + 1 > nodes) nodes = node_id + 1;
            const GROUP_AFFINITY& ga = nn.GroupMask;
            const uint32_t group_base = static_cast<uint32_t>(ga.Group) * 64u;
            KAFFINITY m = ga.Mask;
            for (uint32_t b = 0; b < 64u && m != 0; ++b) {
                if (m & (KAFFINITY{1} << b)) {
                    uint32_t cpu = group_base + b;
                    if (cpu < kTopologyMaxCpus) out->cpu_to_node[cpu] = node_id;
                    m &= ~(KAFFINITY{1} << b);
                }
            }
        });
    out->numa_nodes = nodes == 0 ? 1 : nodes;
    return true;
}

}  // namespace detail

inline bool bolt_detect_topology(CpuTopology* out) noexcept {
    assert(out != nullptr);
    assert(kTopologyMaxCpus >= 1);
    if (!detail::detect_windows(out)) {
        bolt_topology_fallback(out);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Linux
// ---------------------------------------------------------------------------
#elif defined(__linux__)

#include <cstdio>
#include <cstdlib>

namespace detail {

// Read an entire small file into `buf` (size cap). Returns bytes read or -1.
inline int read_file_small(const char* path, char* buf, int cap) noexcept {
    assert(path != nullptr);
    assert(buf != nullptr);
    assert(cap > 0);
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return -1;
    int n = static_cast<int>(std::fread(buf, 1, static_cast<size_t>(cap - 1), f));
    std::fclose(f);
    if (n < 0) n = 0;
    buf[n] = '\0';
    return n;
}

// Parse unsigned integer starting at *p, advance p past digits. Returns false
// if no digit found.
inline bool parse_u32(const char*& p, uint32_t& out) noexcept {
    if (!p) return false;
    if (*p < '0' || *p > '9') return false;
    uint32_t v = 0;
    while (*p >= '0' && *p <= '9') {
        v = v * 10u + static_cast<uint32_t>(*p - '0');
        ++p;
    }
    out = v;
    return true;
}

// Count CPUs listed in a kernel "cpu list" (e.g. "0-3,6,8-9"), and for each
// CPU in range [0, kTopologyMaxCpus) call `fn(cpu_id)`.
template <typename F>
inline uint32_t foreach_cpu_in_list(const char* s, F&& fn) noexcept {
    uint32_t count = 0;
    const char* p = s;
    while (*p) {
        while (*p == ' ' || *p == ',' || *p == '\n' || *p == '\t') ++p;
        if (!*p) break;
        uint32_t a = 0;
        if (!parse_u32(p, a)) break;
        uint32_t b = a;
        if (*p == '-') {
            ++p;
            parse_u32(p, b);
        }
        for (uint32_t c = a; c <= b && c < kTopologyMaxCpus; ++c) {
            fn(c);
            ++count;
        }
    }
    return count;
}

inline bool detect_linux(CpuTopology* out) noexcept {
    assert(out != nullptr);
    detail::topology_zero(out);

    char buf[1024];

    // --- present CPUs --------------------------------------------------------
    int n = read_file_small("/sys/devices/system/cpu/present", buf, sizeof(buf));
    if (n <= 0) return false;

    // Collect present CPU ids.
    uint8_t present[kTopologyMaxCpus];
    std::memset(present, 0, sizeof(present));
    uint32_t total = foreach_cpu_in_list(buf, [&](uint32_t c) noexcept {
        if (c < kTopologyMaxCpus) present[c] = 1;
    });
    if (total == 0) return false;

    uint32_t logical = 0;
    for (uint32_t i = 0; i < kTopologyMaxCpus; ++i) if (present[i]) ++logical;
    out->logical_cpus = logical;

    // --- physical cores via unique (pkg, core) ------------------------------
    // Cap at modest sizes; use stacked arrays.
    uint32_t pkg_ids [kTopologyMaxCpus];
    uint32_t core_ids[kTopologyMaxCpus];
    uint32_t freqs   [kTopologyMaxCpus];
    bool has_freq = false;
    for (uint32_t i = 0; i < kTopologyMaxCpus; ++i) {
        pkg_ids[i] = 0; core_ids[i] = i; freqs[i] = 0;
    }

    char path[256];
    for (uint32_t c = 0; c < kTopologyMaxCpus; ++c) {
        if (!present[c]) continue;

        std::snprintf(path, sizeof(path),
            "/sys/devices/system/cpu/cpu%u/topology/physical_package_id", c);
        if (read_file_small(path, buf, sizeof(buf)) > 0) {
            const char* p = buf; uint32_t v = 0;
            if (parse_u32(p, v)) pkg_ids[c] = v;
        }

        std::snprintf(path, sizeof(path),
            "/sys/devices/system/cpu/cpu%u/topology/core_id", c);
        if (read_file_small(path, buf, sizeof(buf)) > 0) {
            const char* p = buf; uint32_t v = 0;
            if (parse_u32(p, v)) core_ids[c] = v;
        }

        std::snprintf(path, sizeof(path),
            "/sys/devices/system/cpu/cpu%u/cpufreq/cpuinfo_max_freq", c);
        if (read_file_small(path, buf, sizeof(buf)) > 0) {
            const char* p = buf; uint32_t v = 0;
            if (parse_u32(p, v)) { freqs[c] = v; has_freq = true; }
        }
    }

    // Count unique (pkg, core) pairs among present CPUs.
    uint32_t physical = 0;
    {
        uint64_t seen[kTopologyMaxCpus];  // sentinel array; max distinct == logical
        for (uint32_t i = 0; i < kTopologyMaxCpus; ++i) seen[i] = ~uint64_t{0};
        for (uint32_t c = 0; c < kTopologyMaxCpus; ++c) {
            if (!present[c]) continue;
            uint64_t key = (uint64_t(pkg_ids[c]) << 32) | core_ids[c];
            bool dup = false;
            for (uint32_t j = 0; j < physical; ++j) {
                if (seen[j] == key) { dup = true; break; }
            }
            if (!dup) { seen[physical++] = key; }
        }
    }
    out->physical_cores = physical;

    // --- P/E heuristic: above-median max freq = P, at/below = E -------------
    if (has_freq) {
        // Compute median of present CPUs' freqs (ignore zeros).
        uint32_t tmp[kTopologyMaxCpus];
        uint32_t m = 0;
        for (uint32_t c = 0; c < kTopologyMaxCpus; ++c) {
            if (present[c] && freqs[c] > 0) tmp[m++] = freqs[c];
        }
        // Insertion sort (m is small).
        for (uint32_t i = 1; i < m; ++i) {
            uint32_t v = tmp[i]; uint32_t j = i;
            while (j > 0 && tmp[j - 1] > v) { tmp[j] = tmp[j - 1]; --j; }
            tmp[j] = v;
        }
        uint32_t median = (m == 0) ? 0 : tmp[m / 2];
        uint32_t perf = 0, eff = 0;
        for (uint32_t c = 0; c < kTopologyMaxCpus; ++c) {
            if (!present[c]) continue;
            if (freqs[c] > median) { out->cpu_is_perf[c] = 1; ++perf; }
            else                   { out->cpu_is_perf[c] = 0; ++eff;  }
        }
        // If all freqs equal, "perf == 0, eff == logical". That's uniform, not
        // hybrid — mark all as perf and report non-hybrid.
        bool uniform = true;
        for (uint32_t c = 0; c < kTopologyMaxCpus; ++c) {
            if (present[c] && freqs[c] != median) { uniform = false; break; }
        }
        if (uniform) {
            for (uint32_t c = 0; c < kTopologyMaxCpus; ++c) {
                if (present[c]) out->cpu_is_perf[c] = 1;
            }
            out->performance_cores = 0;
            out->efficiency_cores  = 0;
        } else {
            out->performance_cores = perf;
            out->efficiency_cores  = eff;
        }
    } else {
        for (uint32_t c = 0; c < kTopologyMaxCpus; ++c) {
            if (present[c]) out->cpu_is_perf[c] = 1;
        }
        out->performance_cores = 0;
        out->efficiency_cores  = 0;
    }

    // --- NUMA nodes ---------------------------------------------------------
    uint32_t nodes = 0;
    for (uint32_t node = 0; node < kTopologyMaxCpus; ++node) {
        std::snprintf(path, sizeof(path),
            "/sys/devices/system/node/node%u/cpulist", node);
        int r = read_file_small(path, buf, sizeof(buf));
        if (r <= 0) {
            // First missing node terminates scan.
            if (node == 0) { nodes = 0; break; }
            break;
        }
        nodes = node + 1;
        foreach_cpu_in_list(buf, [&](uint32_t c) noexcept {
            if (c < kTopologyMaxCpus) out->cpu_to_node[c] = node;
        });
    }
    out->numa_nodes = nodes == 0 ? 1 : nodes;
    return true;
}

}  // namespace detail

inline bool bolt_detect_topology(CpuTopology* out) noexcept {
    assert(out != nullptr);
    assert(kTopologyMaxCpus >= 1);
    if (!detail::detect_linux(out)) {
        bolt_topology_fallback(out);
    }
    return true;
}

// ---------------------------------------------------------------------------
// macOS
// ---------------------------------------------------------------------------
#elif defined(__APPLE__)

#include <sys/sysctl.h>

namespace detail {

inline bool sysctl_u32(const char* name, uint32_t& out) noexcept {
    assert(name != nullptr);
    uint32_t v = 0;
    size_t sz = sizeof(v);
    if (sysctlbyname(name, &v, &sz, nullptr, 0) != 0) return false;
    out = v;
    return true;
}

inline bool detect_macos(CpuTopology* out) noexcept {
    assert(out != nullptr);
    detail::topology_zero(out);

    uint32_t logical = 0;
    if (!sysctl_u32("hw.logicalcpu", logical) || logical == 0) return false;
    if (logical > kTopologyMaxCpus) logical = kTopologyMaxCpus;
    out->logical_cpus = logical;

    uint32_t physical = 0;
    sysctl_u32("hw.physicalcpu", physical);
    out->physical_cores = physical;

    uint32_t perf = 0, eff = 0;
    sysctl_u32("hw.perflevel0.physicalcpu", perf);
    sysctl_u32("hw.perflevel1.physicalcpu", eff);
    out->performance_cores = perf;
    out->efficiency_cores  = eff;

    // Darwin: no NUMA.
    out->numa_nodes = 1;
    for (uint32_t c = 0; c < kTopologyMaxCpus; ++c) out->cpu_to_node[c] = 0;

    // Heuristic mapping: first `perf` logical CPUs are P, rest are E.
    uint32_t perf_logical = (perf == 0) ? logical : perf;
    if (perf_logical > logical) perf_logical = logical;
    for (uint32_t c = 0; c < kTopologyMaxCpus; ++c) {
        out->cpu_is_perf[c] = (c < perf_logical) ? 1u : 0u;
    }
    return true;
}

}  // namespace detail

inline bool bolt_detect_topology(CpuTopology* out) noexcept {
    assert(out != nullptr);
    assert(kTopologyMaxCpus >= 1);
    if (!detail::detect_macos(out)) {
        bolt_topology_fallback(out);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Other platforms
// ---------------------------------------------------------------------------
#else

inline bool bolt_detect_topology(CpuTopology* out) noexcept {
    assert(out != nullptr);
    assert(kTopologyMaxCpus >= 1);
    bolt_topology_fallback(out);
    return true;
}

#endif

}  // namespace bolt
