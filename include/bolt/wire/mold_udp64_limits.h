// Compile-time subsystem limits; kinds A/B/C follow bolt_limits.h.
// No runtime registry dependency; these retain their existing fixed bounds.
#pragma once

#include <cstdint>

namespace bolt::wire::mold_udp64 {

// Kind A: maximum IPv4 UDP payload after the IP and UDP headers.
inline constexpr std::uint32_t kUdpPayloadMax = 65507;

}  // namespace bolt::wire::mold_udp64
