// Compile-time subsystem limits; kinds A/B/C follow bolt_limits.h.
// No runtime registry dependency; these retain their existing fixed bounds.
#pragma once

#include <cstdint>

namespace bolt::net {

// Kind A: maintained replica transport primary-index ceiling.
inline constexpr std::uint8_t kReplicaDataMaxPrimaries = 8;

// Kind A: maintained proof transport payload ceiling within the WAL wrapper.
inline constexpr std::uint32_t kReplicaProofMaxPayload = 4u * 1024u * 1024u - 512u;

// Kind A: maintained proof transport entry-count ceiling.
inline constexpr std::uint64_t kReplicaProofMaxEntries = 65536;

}  // namespace bolt::net
