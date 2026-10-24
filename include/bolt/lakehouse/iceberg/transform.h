// bolt/lakehouse/iceberg/transform.h — Iceberg partition transforms (W4).
//
// Transform parsing/formatting plus a value kernel for every (transform,
// source type) pair the spec defines (G2ICE-292). PODs.
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

// Longest spelling is `truncate[-2147483648]` (21 chars) + NUL; rounded up.
static constexpr uint32_t kIcebergMaxTransformName = 32u;

// Parse `"identity"`, `"year"`, `"bucket[16]"`, `"truncate[10]"`, ...
// Returns kUnknown on failure (still safe to use).
Transform transform_parse(const char* s, uint32_t len) noexcept;

// The inverse: render a Transform in Iceberg's spec spelling, including the
// parameter (`bucket[16]`, `truncate[10]`). Needed to SERIALIZE a partition
// spec -- without it a writer can only hardcode one transform name, which
// silently rewrites a bucketed or truncated spec into an identity one.
// Writes a NUL-terminated string; returns the length written, or 0 if `cap`
// is too small (leaving `dst` untouched) so a caller cannot emit a truncated
// transform name that would parse back as something else.
uint32_t transform_format(Transform t, char* dst, uint32_t cap) noexcept;

// Murmur3 32-bit (Iceberg uses seed 0). Public so callers can hash partition
// values for bucket() outside of full row evaluation.
uint32_t murmur3_32(const void* key, uint32_t len, uint32_t seed) noexcept;

// Iceberg's `bucket()` over a 64-bit int (little-endian 8 bytes).
int32_t bucket_i64(int64_t value, int32_t N) noexcept;

// `bucket()` over a string.
int32_t bucket_str(const char* s, uint32_t len, int32_t N) noexcept;

// Truncate W on integer.
int64_t truncate_i64(int64_t value, int32_t W) noexcept;

// ---------------------------------------------------------------------------
// Value kernels for every (transform, source type) pair the spec defines
// (spec: "Partition Transforms", "Bucket Transform Details", "Truncate
// Transform Details", Appendix B). G2ICE-292. All are pure and branch-light;
// a null source maps to a null result and is the caller's to handle.
// ---------------------------------------------------------------------------

// Appendix B hashes, as Java's signed int.
int32_t hash_long(int64_t v) noexcept;                     // int, long, date, time, timestamp*
int32_t hash_bytes(const void* p, uint32_t len) noexcept;  // string, uuid (16 BE bytes), fixed, binary
int32_t hash_double(double v) noexcept;                    // float/double: canonical NaN, -0 == 0 (not bucketable)
int32_t hash_bool(bool v) noexcept;                        // hashInt(0|1) (not bucketable)
// Decimal: the unscaled value as a 128-bit two's complement (hi:lo).
int32_t hash_decimal(int64_t hi, uint64_t lo) noexcept;
// Minimum big-endian two's complement bytes of a 128-bit unscaled value
// (BigInteger.toByteArray); 1..16 bytes written to `out`.
uint32_t decimal_min_be_bytes(int64_t hi, uint64_t lo, uint8_t out[16]) noexcept;

// (hash & INT_MAX) % N.
int32_t bucket_of_hash(int32_t h, int32_t N) noexcept;
int32_t bucket_i32(int32_t value, int32_t N) noexcept;     // promoted to long (Appendix B note 1)
int32_t bucket_bytes(const void* p, uint32_t len, int32_t N) noexcept;
int32_t bucket_decimal(int64_t hi, uint64_t lo, int32_t N) noexcept;

// Nanoseconds to microseconds, flooring (Appendix B note 3).
int64_t micros_from_nanos(int64_t ns) noexcept;

// Truncate W on int, with Java's 32-bit wrap; on a decimal's unscaled value.
int32_t truncate_i32(int32_t value, int32_t W) noexcept;
void truncate_decimal(int64_t hi, uint64_t lo, int32_t W,
                      int64_t* out_hi, uint64_t* out_lo) noexcept;
// Byte length of the first L code points of UTF-8 `s` (spec note 3); a
// truncated or invalid sequence counts each byte as one code point.
uint32_t truncate_utf8_len(const char* s, uint32_t len, int32_t L) noexcept;
uint32_t truncate_binary_len(uint32_t len, int32_t L) noexcept;

// Time transforms. The source is days (date), micros (time/timestamp*) or
// nanos (*_ns). Results count from 1970-01-01 (00:00), flooring before it.
enum class TimeUnit : uint8_t { kDays = 0, kMicros = 1, kNanos = 2 };
int32_t transform_year(int64_t v, TimeUnit u) noexcept;
int32_t transform_month(int64_t v, TimeUnit u) noexcept;
int32_t transform_day(int64_t v, TimeUnit u) noexcept;
int32_t transform_hour(int64_t v, TimeUnit u) noexcept;   // undefined for kDays (asserts)

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
