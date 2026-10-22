// Compile-time subsystem limits; kinds A/B/C follow bolt_limits.h.
// No runtime registry dependency; these retain their existing fixed bounds.
#pragma once

#include <cstdint>

namespace bolt::net {

// Kind C: bounded response-head parsing byte budget.
inline constexpr std::uint32_t kHttpResponseHeadCap = 8192;

// Kind B: fixed response-header descriptor storage capacity.
inline constexpr std::uint32_t kHttpResponseHeaderCap = 32;

}  // namespace bolt::net
