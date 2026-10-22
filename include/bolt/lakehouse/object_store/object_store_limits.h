// Compile-time subsystem limits; kinds A/B/C follow bolt_limits.h.
// No runtime registry dependency; these retain their existing fixed bounds.
#pragma once

#include <cstdint>

namespace bolt::lakehouse {

// Kind B: ceiling for the immutable directory shape admitted at startup.
inline constexpr uint32_t kOsPinnedMaxEntries = 65536u;

// Kind C: maximum individual pinned-origin read request budget.
inline constexpr uint32_t kOsPinnedMaxRequestBytes = 1u << 20;

// Kind B: ceiling for the cache page count fixed at initialization.
inline constexpr uint32_t kObjectRangeMaxPages = 64;

// Kind C: this adapter's per-part byte budget; smaller than S3's protocol maximum.
inline constexpr uint32_t kS3MaxPartBytes = 16u * 1024u * 1024u;

// Kind A: S3 multipart protocol permits at most 10,000 parts.
inline constexpr uint32_t kS3MaxUploadParts = 10000u;

// Kind B: fixed upload-identifier storage capacity, including the terminator.
inline constexpr uint32_t kS3UploadIdCap = 1024u;

// Kind B: fixed per-part ETag storage capacity, including the terminator.
inline constexpr uint32_t kS3PartEtagCap = 128u;

// Kind B: fixed copied strong-validator storage capacity, including the terminator.
inline constexpr uint32_t kOsStrongEtagCap = 256u;

// Kind C: maximum individual versioned-range request budget.
inline constexpr uint32_t kOsVersionedRangeMaxBytes = 8u << 20;

}  // namespace bolt::lakehouse
