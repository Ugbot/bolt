// Compile-time limits of the OTLP decoder; kinds A/B/C follow bolt_limits.h.
#pragma once

#include <cstdint>

namespace bolt::ingest::otlp {

// Kind A: AnyValue nesting a decode follows. Deeper input is refused.
inline constexpr std::uint32_t kValueDepth = 16;
// Kind A: attributes one resource's identity may hold. More is refused.
inline constexpr std::uint32_t kIdentityAttrs = 128;

// Kind A: the finest exponential-histogram scale kept; a finer one is merged down.
inline constexpr std::int32_t kExpHistogramScale = 8;
// Kind A: buckets one side of a native histogram may hold. More is refused.
inline constexpr std::uint32_t kHistBucketsPerSide = 128;
// Kind A: words one translated metric name may hold. More is refused.
inline constexpr std::uint32_t kNameTokens = 64;
// Kind A: labels one series may carry before sorting (point + resource + reserved). More is refused.
inline constexpr std::uint32_t kSeriesLabels = 512;

}  // namespace bolt::ingest::otlp
