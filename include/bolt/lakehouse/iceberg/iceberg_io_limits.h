// Compile-time subsystem limits; kinds A/B/C follow bolt_limits.h.
// No runtime registry dependency; these retain their existing fixed bounds.
#pragma once

#include <cstdint>

namespace bolt::lakehouse::iceberg {

// Kind C: maximum individual immutable-metadata read request budget.
inline constexpr uint32_t kMetadataLoadMaxRequestBytes = 1u << 20;

// Kind B: fixed staging-path storage capacity, including the terminator.
inline constexpr uint32_t kIcebergStagedPathCap = 4096;

}  // namespace bolt::lakehouse::iceberg
