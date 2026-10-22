// Compile-time subsystem limits; kinds A/B/C follow bolt_limits.h.
// No runtime registry dependency; these retain their existing fixed bounds.
#pragma once

#include <cstdint>

// Kind A: affinity CPU-index encoding ceiling, independent of online CPUs.
inline constexpr uint32_t kBoltThreadAffinityCpuCapacity = 4096u;
