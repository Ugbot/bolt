// Compile-time limits for nested parquet schema trees and record assembly;
// kinds A/B/C follow bolt_limits.h.
#pragma once

#include <cstdint>

namespace bolt::ingest::parquet {

// Kind A: SchemaElements one footer may declare (root included). Bounds the
// arena tree and every walk over it against a hostile list count.
inline constexpr uint32_t kPqMaxSchemaElems = 1u << 16;

// Kind A: group nesting depth below the root. Matches the leaf walk's frame
// stack in pq_parse_file_meta, so a file it accepts is never refused here.
inline constexpr uint32_t kPqMaxNestDepth = 16;

}  // namespace bolt::ingest::parquet
