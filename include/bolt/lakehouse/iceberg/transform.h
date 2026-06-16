// bolt/lakehouse/iceberg/transform.h — Iceberg partition transforms (W4).
//
// Identity, year/month/day/hour from int64 epoch micros, bucket(N,x) with
// Iceberg's Murmur3-32 seed-0 mod N (sign-stripped), truncate(W,x). PODs.
// Tiger Style: noexcept, fixed caps, no exceptions.

#pragma once

#include <cassert>
#include <cstdint>

namespace bolt {
namespace lakehouse {
namespace iceberg {

enum class TransformKind : uint8_t {
    kIdentity = 0,
    kYear     = 1,
    kMonth    = 2,
    kDay      = 3,
    kHour     = 4,
    kBucket   = 5,
    kTruncate = 6,
    kVoid     = 7,
    kUnknown  = 8,
};

struct Transform {
    TransformKind kind;
    uint8_t       _pad[3];
    int32_t       param;  // bucket N, truncate W; 0 otherwise
};

// Parse `"identity"`, `"year"`, `"bucket[16]"`, `"truncate[10]"`, ...
// Returns kUnknown on failure (still safe to use).
Transform transform_parse(const char* s, uint32_t len) noexcept;

// Murmur3 32-bit (Iceberg uses seed 0). Public so callers can hash partition
// values for bucket() outside of full row evaluation.
uint32_t murmur3_32(const void* key, uint32_t len, uint32_t seed) noexcept;

// Iceberg's `bucket()` over a 64-bit int (little-endian 8 bytes).
int32_t bucket_i64(int64_t value, int32_t N) noexcept;

// `bucket()` over a string.
int32_t bucket_str(const char* s, uint32_t len, int32_t N) noexcept;

// Truncate W on integer.
int64_t truncate_i64(int64_t value, int32_t W) noexcept;

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
