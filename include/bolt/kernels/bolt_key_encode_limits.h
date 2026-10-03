// bolt_key_encode_limits.h — bounds of one canonical key (bolt_key_encode.h,
// MSEG §6.5). Kind A: invariants, reported, never set.
#pragma once

#include <cstdint>

namespace bolt {

// Encoded bytes of one key, all cells: a longer key is refused (kNoRoom).
inline constexpr uint32_t kKeyEncodeMaxBytes = 1u << 20;
// Cells (pk columns) in one key.
inline constexpr uint32_t kKeyEncodeMaxCells = 64;
// Overflow bytes of one key_encode_column call: StringView offsets are u32.
inline constexpr uint64_t kKeyEncodeMaxOverflowBytes = UINT32_MAX;
static_assert(kKeyEncodeMaxOverflowBytes <= UINT32_MAX);

}  // namespace bolt
