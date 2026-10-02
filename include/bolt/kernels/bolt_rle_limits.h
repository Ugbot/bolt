// bolt_rle_limits.h — invariants of ColumnFormat::RLE (bolt_rle.h, MSEG K3).
// Kind A: fixed by the layout, reported, never set.
#pragma once

#include <cstddef>
#include <cstdint>

namespace bolt {
namespace rle {

// Run ends and selections are int32 row ids, so one RLE column (one page)
// holds at most INT32_MAX rows; a longer column is split by the writer.
inline constexpr int64_t kRleMaxRows = INT32_MAX;

// IN-set size up to which membership is a linear scan; past it, a binary
// search over the caller's sorted set. A tuning point, not a capacity.
inline constexpr int32_t kRleInLinearMax = 8;

// Decode writes runs in whole chunks of this many bytes (a run's last chunk
// may spill into the next run, never past the page). A tuning point.
inline constexpr size_t kRleFillBytes = 64;

}  // namespace rle
}  // namespace bolt
