// Compile-time subsystem limits; kinds A/B/C follow bolt_limits.h.
// No runtime registry dependency; these retain their existing fixed bounds.
#pragma once

#include <cstdint>

namespace bolt::ingest::parquet {

// Kind C: maximum individual pinned-source read request budget.
inline constexpr uint32_t kParquetSourceMaxReadBytes = 1024u * 1024u;

}  // namespace bolt::ingest::parquet
