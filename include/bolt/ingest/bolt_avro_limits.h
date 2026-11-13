// Compile-time limits for Avro schemas; kinds A/B/C follow bolt_limits.h.
#pragma once

#include <cstdint>

namespace bolt::ingest {

// Kind B: enum fields one parsed schema reports symbols for (AvroEnumTable).
// Past it the parse with a table fails, never truncates.
inline constexpr uint32_t kAvroMaxEnums = 32u;
// Kind B: enum symbols across those fields, and the bytes of their names.
inline constexpr uint32_t kAvroMaxEnumSymbols = 1024u;
inline constexpr uint32_t kAvroEnumPoolBytes = 16384u;

}  // namespace bolt::ingest
