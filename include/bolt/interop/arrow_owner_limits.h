// Compile-time subsystem limits; kinds A/B/C follow bolt_limits.h.
// No runtime registry dependency; these retain their existing fixed bounds.
#pragma once

#include <cstdint>

namespace bolt::arrow {

// Kind B: ceiling for the field shape admitted when an owner is created.
inline constexpr std::uint32_t kOwnerMaxFields = 4096;

// Kind B: ceiling for lease storage admitted when an owner is created.
inline constexpr std::uint32_t kOwnerMaxLeases = 64;

}  // namespace bolt::arrow
