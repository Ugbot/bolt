# CPU topology APIs

How Bolt discovers physical cores, NUMA nodes, and P/E-core classes at
`scheduler_init` without pulling a portability shim.

## Windows

`GetLogicalProcessorInformationEx`
(https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/nf-sysinfoapi-getlogicalprocessorinformationex)
with:

- `RelationProcessorCore` — one `PROCESSOR_RELATIONSHIP` per physical
  core; the `EfficiencyClass` byte distinguishes hybrid cores. On Intel
  12th gen+ and Windows 11+, P-cores report `EfficiencyClass = 1`,
  E-cores `0` (higher = more performant per the MSDN definition).
- `RelationNumaNodeEx` — one entry per NUMA node with full affinity
  masks across processor groups (the new API; `RelationNumaNode`
  truncates to the primary group since Server 2022 21H2).

Header: `sysinfoapi.h` / `Windows.h`. Link: `Kernel32.lib`. Requires
`_WIN32_WINNT >= 0x0601`.

## Linux

sysfs layout (stable kernel ABI, no library needed):

```text
/sys/devices/system/cpu/cpu<N>/topology/physical_package_id
/sys/devices/system/cpu/cpu<N>/topology/core_id
/sys/devices/system/cpu/cpu<N>/topology/thread_siblings_list
/sys/devices/system/cpu/cpu<N>/cpufreq/cpuinfo_max_freq
/sys/devices/system/cpu/cpu<N>/node<M>        (symlink → NUMA node)
```

Hybrid cores (Alder Lake and later) are exposed from kernel 5.15
onward; the `cpufreq/cpuinfo_max_freq` value is the portable P/E
heuristic — P-cores have a measurably higher max frequency than E-cores
on the same package. NUMA memory policy via `numa(7)`
(https://man7.org/linux/man-pages/man7/numa.7.html): `set_mempolicy`,
`mbind`, `move_pages`. Higher-level `libnuma` (`-lnuma`,
`<numaif.h>`) is the recommended interface but is not in the base C
library — Bolt reads sysfs directly to stay dependency-free and only
dlopens libnuma when the user opts into NUMA binding.

## macOS

`sysctlbyname` with hierarchical `hw.perflevel<N>.*` keys (undocumented
but stable since macOS 12):

```text
hw.nperflevels                    // number of perf classes
hw.perflevel0.physicalcpu         // P-cores  (on Apple Silicon)
hw.perflevel0.logicalcpu
hw.perflevel1.physicalcpu         // E-cores
hw.perflevel1.logicalcpu
```

On Apple Silicon perflevel 0 is always the performance tier. Apple
hardware is single-socket, so NUMA is a non-issue. Thread pinning uses
`thread_policy_set` with `THREAD_AFFINITY_POLICY` (advisory only — the
kernel may ignore it).

## Intel Hybrid Architecture reference

The canonical spec for Thread Director and P/E-core dispatch is the
Intel Architecture Optimization Reference Manual, "Hybrid Architecture"
chapter (Intel order no. 248966). [source?] — the public HTML landing
page 403'd during research; the PDF is mirrored under Intel's developer
downloads. Key load-bearing fact for Bolt: `EfficiencyClass` values on
Windows, and `capacity` in Linux `sched_domain` topology, both derive
from the same hardware CPUID leaf 0x1A (Native Model ID and Core Type),
so a single detection pass at `scheduler_init` can populate the
cross-platform view.
