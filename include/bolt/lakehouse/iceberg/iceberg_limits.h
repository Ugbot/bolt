// Iceberg reader caps (kind A layout invariants / kind B allocation-time bounds). A cap lives here, not inline.
#pragma once
#include <cstdint>

namespace bolt::lakehouse::iceberg::limits {
inline constexpr std::uint32_t kNameMapMaxNames  = 8u;              // alias names kept per name-mapping entry
inline constexpr std::uint32_t kEqKeyMax         = 4096u;           // bytes of one encoded equality-delete key
inline constexpr std::uint64_t kMaxDvObjectBytes = 256ull << 20;    // one deletion-vector object
inline constexpr std::uint32_t kMaxEqualityIds   = 16u;             // columns in one equality-delete key
inline constexpr std::uint32_t kMaxEqualitySets  = 64u;             // distinct equality-id sets in one scan
}  // namespace bolt::lakehouse::iceberg::limits
