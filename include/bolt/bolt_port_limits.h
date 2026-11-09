// Compile-time subsystem limits; kinds A/B/C follow bolt_limits.h.
// No runtime registry dependency; these retain their existing fixed bounds.
#pragma once

#include <cstdint>

// Kind A: affinity CPU-index encoding ceiling, independent of online CPUs.
inline constexpr uint32_t kBoltThreadAffinityCpuCapacity = 4096u;

// Kind A: named process-lifetime memory owners listed for /metrics (bolt_mem_owner.h).
// An owner past the cap still counts toward its totals; it is only not listed.
inline constexpr uint32_t kBoltMemOwnerRegistryCapacity = 64u;
