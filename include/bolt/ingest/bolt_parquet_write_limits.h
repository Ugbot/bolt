// Compile-time parquet writer limits; kinds A/B/C follow bolt_limits.h.
#pragma once

#include <cstdint>

namespace bolt::ingest::parquet {

// Kind A: nodes in one nested column's ParquetNestedNode table.
inline constexpr std::uint32_t kPwMaxNestedNodes = 64;
// Kind A: nodes on one root-to-leaf path of a nested column.
inline constexpr std::uint32_t kPwMaxNestedDepth = 16;
// Kind A: repeated (LIST / MAP) nodes on one path; the reader's kPqMaxRepLevels.
inline constexpr std::uint32_t kPwMaxNestedRep = 8;

}  // namespace bolt::ingest::parquet
