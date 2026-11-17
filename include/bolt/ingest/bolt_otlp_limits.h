// Compile-time limits of the OTLP decoder; kinds A/B/C follow bolt_limits.h.
#pragma once

#include <cstdint>

namespace bolt::ingest::otlp {

// Kind A: AnyValue nesting a decode follows. Deeper input is refused.
inline constexpr std::uint32_t kValueDepth = 16;
// Kind A: attributes one resource's identity may hold. More is refused.
inline constexpr std::uint32_t kIdentityAttrs = 128;

}  // namespace bolt::ingest::otlp
