// bolt_wire_limits.h — alignment units and caps of the wire frame container
// (MSEG B2, layout decision §6.1.1, U36).
//
// The RULE is in the bytes: every frame file records the units it was written
// with, and a reader validates them and never assumes its own. The DEFAULTS
// below are per-build constants: override with -DBOLT_WIRE_CHUNK_ALIGN=4096
// etc. A file written by one build is readable by every build.
//
//   unit    default            valid range        applies to
//   buf     64 B               [64 B, 4 KiB]      every buffer, prelude, frame start
//   chunk   16 KiB             [4 KiB, 1 MiB]     index region start, footer block
//   stripe  64 KiB             [16 KiB, 8 MiB]    segment data region / extents
//   io      16 KiB mac, 4 KiB  [512 B, 1 MiB]     informational: writer's I/O unit
//
// buf <= chunk <= stripe is required.

#pragma once

#include <cstdint>

#include "bolt/bolt_column_limits.h"

#ifndef BOLT_WIRE_BUF_ALIGN
#define BOLT_WIRE_BUF_ALIGN 64u
#endif
#ifndef BOLT_WIRE_CHUNK_ALIGN
#define BOLT_WIRE_CHUNK_ALIGN 16384u
#endif
#ifndef BOLT_WIRE_STRIPE_ALIGN
#define BOLT_WIRE_STRIPE_ALIGN 65536u
#endif
#ifndef BOLT_WIRE_IO_ALIGN
#if defined(__APPLE__)
#define BOLT_WIRE_IO_ALIGN 16384u
#else
#define BOLT_WIRE_IO_ALIGN 4096u
#endif
#endif

namespace bolt {
namespace wire {

inline constexpr uint32_t kFrameBufAlign    = BOLT_WIRE_BUF_ALIGN;
inline constexpr uint32_t kFrameChunkAlign  = BOLT_WIRE_CHUNK_ALIGN;
inline constexpr uint32_t kFrameStripeAlign = BOLT_WIRE_STRIPE_ALIGN;
inline constexpr uint32_t kFrameIoAlign     = BOLT_WIRE_IO_ALIGN;

// Valid ranges, as log2 (a recorded unit outside its range is refused).
inline constexpr uint8_t kFrameBufAlignLog2Lo    = 6;    // 64 B
inline constexpr uint8_t kFrameBufAlignLog2Hi    = 12;   // 4 KiB
inline constexpr uint8_t kFrameChunkAlignLog2Lo  = 12;   // 4 KiB
inline constexpr uint8_t kFrameChunkAlignLog2Hi  = 20;   // 1 MiB
inline constexpr uint8_t kFrameStripeAlignLog2Lo = 14;   // 16 KiB
inline constexpr uint8_t kFrameStripeAlignLog2Hi = 23;   // 8 MiB
inline constexpr uint8_t kFrameIoAlignLog2Lo     = 9;    // 512 B
inline constexpr uint8_t kFrameIoAlignLog2Hi     = 20;   // 1 MiB

// A frame's zone maps: at most one per column (n_zones is a u32 on disk).
inline constexpr uint32_t kFrameMaxZones = kMaxColumns;
// A frame's wire payload length is a u32 on disk.
inline constexpr uint64_t kFrameMaxPayloadBytes = 0xFFFFFFFFull;
// Nesting depth of a wire blob: Nested children and Dictionary values are
// serialised as sub-blobs, recursively (bounded recursion).
inline constexpr uint32_t kWireMaxNestDepth = 16;
// Frames in one container (FrameIndexEntry count is a u32 on disk).
inline constexpr uint32_t kFrameFileMaxFrames = 0xFFFFFFFFu;

namespace detail {
constexpr bool frame_pow2(uint32_t v) { return v != 0 && (v & (v - 1)) == 0; }
}  // namespace detail

static_assert(detail::frame_pow2(kFrameBufAlign) && detail::frame_pow2(kFrameChunkAlign) &&
              detail::frame_pow2(kFrameStripeAlign) && detail::frame_pow2(kFrameIoAlign),
              "alignment units are powers of two");
static_assert(kFrameBufAlign >= (1u << kFrameBufAlignLog2Lo) &&
              kFrameBufAlign <= (1u << kFrameBufAlignLog2Hi), "BOLT_WIRE_BUF_ALIGN range");
static_assert(kFrameChunkAlign >= (1u << kFrameChunkAlignLog2Lo) &&
              kFrameChunkAlign <= (1u << kFrameChunkAlignLog2Hi), "BOLT_WIRE_CHUNK_ALIGN range");
static_assert(kFrameStripeAlign >= (1u << kFrameStripeAlignLog2Lo) &&
              kFrameStripeAlign <= (1u << kFrameStripeAlignLog2Hi), "BOLT_WIRE_STRIPE_ALIGN range");
static_assert(kFrameIoAlign >= (1u << kFrameIoAlignLog2Lo) &&
              kFrameIoAlign <= (1u << kFrameIoAlignLog2Hi), "BOLT_WIRE_IO_ALIGN range");
static_assert(kFrameBufAlign <= kFrameChunkAlign && kFrameChunkAlign <= kFrameStripeAlign,
              "buf <= chunk <= stripe");
static_assert(kFrameBufAlign % 64u == 0, "the wire format's own 64 B buffer alignment");

}  // namespace wire
}  // namespace bolt
