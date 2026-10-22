// Compile-time subsystem limits; kinds A/B/C follow bolt_limits.h.
// No runtime registry dependency; these retain their existing fixed bounds.
#pragma once

#include <cstdint>

namespace bolt::crypto {

// Kind A: the OS getentropy contract permits at most 256 bytes per call.
inline constexpr uint32_t kRandomBytesCap = 256;

// Kind B: fixed request-field storage capacity for the signed If-Match value.
static constexpr uint32_t kSigV4MaxIfMatch = 256u;

}  // namespace bolt::crypto
