// bolt_wire_frame.h — checksummed wire frames and the multi-frame container
// (MSEG B2; layout decision §6.0, §6.1.1, §8.0; storage contract §3.2).
//
// One container serves a sealed WAL segment (which IS the L0 segment), a
// chukonu spill run and a shuffle spool. Everything is pointer-free,
// little-endian, and 64 B aligned relative to the file start, so an mmap of
// the file hands out bolt_wire_view batches with no copy.
//
// Frame (starts on a 64 B boundary):
//
//   +0                 FrameHeader      64 B  crc32c, op kind, flags, lsn,
//                                             commit_ts, rows, payload bytes,
//                                             txn_id, first_pos, key prefixes
//   +64                wire payload     bolt_wire_serialize bytes, zero
//                                       padded to 64
//   +64+align64(bytes) FrameTrailer     64 B  magic, op kind, rows, n_zones,
//                                             lsn range, commit_ts range
//                      ZoneMap[n_zones] 32 B each, zone i = column i
//                      kind[n_zones]    1 B each (FrameZoneKind)
//                      zero pad to 64
//
//   crc32c covers bytes [4, frame_len): the rest of the header, the payload
//   with its padding, and the trailer. The checksum lives in the block that
//   is read first, so a torn or misdirected write fails before any byte of
//   the payload is interpreted. It is seeded: a frame inside a container is
//   checksummed with the container's seed (its header CRC, which covers a
//   per-file incarnation), so a stale frame left in a recycled, non-zeroed
//   file fails its CRC instead of being replayed. A standalone frame uses
//   seed 0.
//
// Container (FrameFile, bolt_wire_frame_file.h):
//
//   0                  FrameFileHeader  64 B  magic "BWFF", version, purpose,
//                                             alignment units (log2) the file
//                                             was written with, ids, crc
//   frames_off (64)    frames, back to back, each on a 64 B boundary; a
//                      zero header (kind 0, bytes 0) ends the run (a
//                      preallocated segment is zero filled)
//   index_off          FrameIndexEntry[n]  64 B each; index_off is on the
//                      chunk unit
//                      rollup ZoneMap[n_cols] + kind[n_cols], zero pad to 64
//   file_len - 64      FrameFileFooter  64 B; file_len is a multiple of the
//                      chunk unit, so the tail is one aligned read
//
//   A file with no valid footer (a crash before seal) is recovered by
//   walking frames from frames_off, stopping at the first zero header or the
//   first frame whose CRC fails (frame_file_recover). Frames start on the
//   recorded buf unit: the walk advances by the frame length rounded up to
//   it (this writer's unit is 64 B, so frames are back to back).
//
//   Readers take a buffer that starts on a 64 B boundary (an mmap, an
//   aligned cache slot); the zone, index and rollup arrays are handed out as
//   typed pointers into it.
//
// Alignment (§6.1.1, U36): the header records buf/chunk/stripe/io units; a
// reader validates them (powers of two in range, buf <= chunk <= stripe) and
// refuses a file whose units fail, naming the unit. It never assumes its own
// build's constants. frame_file_unit_mappable says whether a region written
// on a unit can be mapped at a non-zero file offset on this host; a whole
// file mapped from offset 0 always is.
//
// Tiger Style: POD + free functions, noexcept, no allocation, bounded loops.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_column.h"
#include "bolt/bolt_port.h"
#include "bolt/bolt_zonemap.h"
#include "bolt/io/bolt_crc32c.h"
#include "bolt/wire/bolt_wire.h"
#include "bolt/wire/bolt_wire_limits.h"

namespace bolt {
namespace wire {

// The wire payload's own buffers are 64 B aligned (kWireAlign); frames and
// regions in this container use the same unit, so a recorded buf unit of 64
// is exactly what a reader may assume about every buffer.
static_assert(kFrameBufAlign == kWireAlign,
              "this writer emits 64 B frame/buffer alignment (the wire unit)");

// ---------------------------------------------------------------------------
// On-disk structs
// ---------------------------------------------------------------------------

enum class FrameOpKind : uint16_t {
    kNone        = 0,   // never written; a zero header ends a frame run
    kBatch       = 1,   // rows (a put, a spill run chunk, a shuffle block)
    kDeleteKeys  = 2,   // rows are the keys to delete
    kDeleteRange = 3,   // rows are [lo, hi) key bounds
    kTxnMarker   = 4,   // transaction commit / abort marker
};

enum class FrameZoneKind : uint8_t {
    kNone = 0,          // no bounds (type without an order, or no valid row)
    kI64  = 1,          // signed ints, Bool, Date32/64, Timestamp, Duration, Decimal64
    kU64  = 2,          // unsigned ints
    kF32  = 3,          // Float32 (low 32 bits)
    kF64  = 4,          // Float64
    kStr8 = 5,          // Utf8 / Binary / Symbol: 8-byte prefix (zone_str8_*)
};

struct FrameHeader {                // 64 B
    uint32_t crc32c;                //  0 over [4, frame_len)
    uint16_t kind;                  //  4 FrameOpKind
    uint16_t flags;                 //  6 caller bits (pk_sorted, append_clean, prepared, ...)
    uint64_t lsn;                   //  8 = trailer lsn_min
    int64_t  commit_ts;             // 16 = trailer ts_min
    uint32_t rows;                  // 24
    uint32_t bytes;                 // 28 wire payload bytes (unpadded)
    uint64_t txn_id;                // 32
    uint64_t first_pos;             // 40 row position of the first row in its file
    uint8_t  key_min[8];            // 48 key prefix bounds (caller supplied)
    uint8_t  key_max[8];            // 56
};

struct FrameTrailer {               // 64 B, then ZoneMap[n_zones], kind[n_zones]
    uint32_t magic;                 //  0 kFrameTrailerMagic
    uint16_t kind;                  //  4 FrameOpKind (= header kind)
    uint16_t flags;                 //  6 (= header flags)
    uint32_t rows;                  //  8
    uint32_t n_zones;               // 12 0 or the batch's column count
    uint64_t lsn_min, lsn_max;      // 16, 24
    int64_t  ts_min, ts_max;        // 32, 40 commit_ts range
    uint64_t payload_bytes;         // 48 (= header bytes)
    uint32_t trailer_bytes;         // 56 this trailer incl. zones, kinds, pad
    uint32_t _pad;                  // 60
};

struct FrameFileHeader {            // 64 B at file offset 0
    uint8_t  magic[4];              //  0 "BWFF"
    uint16_t version;               //  4 kFrameFileVersion
    uint16_t purpose;               //  6 FrameFilePurpose
    uint8_t  buf_align_log2;        //  8 units this file was written with
    uint8_t  chunk_align_log2;      //  9
    uint8_t  stripe_align_log2;     // 10
    uint8_t  io_align_log2;         // 11 informational
    uint32_t flags;                 // 12
    uint64_t id0, id1;              // 16, 24 caller ids (table / segment / run)
    uint64_t frames_off;            // 32 first frame
    uint64_t incarnation;           // 40 differs per file creation (seeds every CRC)
    uint8_t  _reserved[12];         // 48
    uint32_t header_crc32c;         // 60 over [0, 60); the seed of every frame,
                                    //    index and footer CRC in the file
};

struct FrameIndexEntry {            // 64 B
    uint64_t off;                   //  0 frame offset in the file
    uint64_t len;                   //  8 frame length
    uint32_t rows;                  // 16
    uint16_t kind;                  // 20
    uint16_t flags;                 // 22
    uint64_t lsn_min, lsn_max;      // 24, 32
    int64_t  ts_min, ts_max;        // 40, 48
    uint32_t crc32c;                // 56 the frame's own crc
    uint32_t n_zones;               // 60
};

struct FrameFileFooter {            // last 64 B of a sealed file
    uint32_t magic;                 //  0 kFrameFooterMagic
    uint16_t version;               //  4
    uint16_t purpose;               //  6
    uint64_t index_off;             //  8
    uint64_t index_len;             // 16 index entries + rollup (+ pad)
    uint64_t total_rows;            // 24
    uint64_t lsn_min, lsn_max;      // 32, 40
    uint32_t n_frames;              // 48
    uint32_t n_rollup;              // 52 rollup zones (0 = none: schemas differed)
    uint32_t index_crc32c;          // 56 over [index_off, index_off + index_len)
    uint32_t footer_crc32c;         // 60 over [0, 60)
};

static_assert(sizeof(FrameHeader) == 64 && sizeof(FrameTrailer) == 64 &&
              sizeof(FrameFileHeader) == 64 && sizeof(FrameIndexEntry) == 64 &&
              sizeof(FrameFileFooter) == 64, "frame structs are 64 B");
static_assert(offsetof(FrameHeader, bytes) == 28 && offsetof(FrameHeader, key_max) == 56,
              "FrameHeader layout (decision §8.0)");
static_assert(offsetof(FrameTrailer, trailer_bytes) == 56, "FrameTrailer layout");
static_assert(offsetof(FrameFileHeader, header_crc32c) == 60 &&
              offsetof(FrameFileHeader, incarnation) == 40, "FrameFileHeader layout");
static_assert(offsetof(FrameIndexEntry, n_zones) == 60, "FrameIndexEntry layout");
static_assert(offsetof(FrameFileFooter, footer_crc32c) == 60, "FrameFileFooter layout");

inline constexpr uint32_t kFrameTrailerMagic = 0x52544642u;   // "BFTR"
inline constexpr uint32_t kFrameFooterMagic  = 0x45465742u;   // "BWFE"
inline constexpr uint16_t kFrameFileVersion  = 1;

enum class FrameFilePurpose : uint16_t {
    kGeneric = 0, kWalSegment = 1, kL0Run = 2, kSpill = 3, kShuffle = 4,
};

enum class FrameStatus : uint8_t {
    kOk = 0,
    kEnd,             // zero header: no more frames
    kTruncated,       // a length points past the buffer
    kBadMagic,
    kBadVersion,
    kBadCrc,
    kBadLayout,       // inconsistent lengths / counts / kinds
    kBadBufAlign,     // recorded unit fails validation (named per unit)
    kBadChunkAlign,
    kBadStripeAlign,
    kBadIoAlign,
    kCapacity,        // caller buffer too small
    kUnsupported,     // batch the wire format refuses
    kUnaligned,       // reader buffer does not start on a 64 B boundary
};

inline const char* frame_status_name(FrameStatus s) noexcept {
    switch (s) {
        case FrameStatus::kOk:             return "ok";
        case FrameStatus::kEnd:            return "end of frames";
        case FrameStatus::kTruncated:      return "truncated";
        case FrameStatus::kBadMagic:       return "bad magic";
        case FrameStatus::kBadVersion:     return "unsupported version";
        case FrameStatus::kBadCrc:         return "checksum mismatch";
        case FrameStatus::kBadLayout:      return "inconsistent layout";
        case FrameStatus::kBadBufAlign:    return "invalid buf_align unit";
        case FrameStatus::kBadChunkAlign:  return "invalid chunk_align unit";
        case FrameStatus::kBadStripeAlign: return "invalid stripe_align unit";
        case FrameStatus::kBadIoAlign:     return "invalid io_align unit";
        case FrameStatus::kCapacity:       return "buffer too small";
        case FrameStatus::kUnsupported:    return "unsupported batch";
        case FrameStatus::kUnaligned:      return "buffer not 64 B aligned";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// Alignment units
// ---------------------------------------------------------------------------

struct FrameAlign {
    uint8_t buf_log2, chunk_log2, stripe_log2, io_log2;
};

namespace detail {
constexpr uint8_t frame_log2(uint32_t v) {
    uint8_t r = 0;
    while (v > 1u) { v >>= 1; ++r; }
    return r;
}
}  // namespace detail

/// The units this build writes with (bolt_wire_limits.h).
inline constexpr FrameAlign frame_align_default() noexcept {
    return FrameAlign{detail::frame_log2(kFrameBufAlign), detail::frame_log2(kFrameChunkAlign),
                      detail::frame_log2(kFrameStripeAlign), detail::frame_log2(kFrameIoAlign)};
}

/// Validate recorded units: each in range, buf <= chunk <= stripe. The status
/// names the first failing unit.
inline FrameStatus frame_align_validate(const FrameAlign& a) noexcept {
    assert(kFrameBufAlignLog2Lo <= kFrameBufAlignLog2Hi);
    assert(kFrameChunkAlignLog2Lo <= kFrameChunkAlignLog2Hi);
    if (a.buf_log2 < kFrameBufAlignLog2Lo || a.buf_log2 > kFrameBufAlignLog2Hi)
        return FrameStatus::kBadBufAlign;
    if (a.chunk_log2 < kFrameChunkAlignLog2Lo || a.chunk_log2 > kFrameChunkAlignLog2Hi ||
        a.chunk_log2 < a.buf_log2)
        return FrameStatus::kBadChunkAlign;
    if (a.stripe_log2 < kFrameStripeAlignLog2Lo || a.stripe_log2 > kFrameStripeAlignLog2Hi ||
        a.stripe_log2 < a.chunk_log2)
        return FrameStatus::kBadStripeAlign;
    if (a.io_log2 < kFrameIoAlignLog2Lo || a.io_log2 > kFrameIoAlignLog2Hi)
        return FrameStatus::kBadIoAlign;
    return FrameStatus::kOk;
}

/// Can a region that starts on a `unit_log2` boundary be mapped zero-copy at
/// its own file offset on a host whose mapping granularity is `host_gran`
/// (the OS page; 64 KiB for Windows MapViewOfFile offsets)? When false the
/// caller copies / preads the region into an aligned slot.
inline bool frame_file_unit_mappable(uint8_t unit_log2, uint32_t host_gran) noexcept {
    assert(host_gran != 0 && (host_gran & (host_gran - 1)) == 0);
    assert(unit_log2 < 32);
    return (1u << unit_log2) % host_gran == 0;
}

// ---------------------------------------------------------------------------
// Zone maps
// ---------------------------------------------------------------------------

namespace detail {

BOLT_FORCE_INLINE size_t frame_align64(size_t x) noexcept {
    assert(x <= (static_cast<size_t>(1) << 62));
    return (x + 63u) & ~static_cast<size_t>(63u);
}

inline FrameZoneKind frame_zone_kind_for(const BoltColumn& c) noexcept {
    assert(c.length >= 0);
    const BoltType t = c.type;
    if (c.format == ColumnFormat::VarBinary) return FrameZoneKind::kStr8;
    if (c.format != ColumnFormat::Flat && c.format != ColumnFormat::View)
        return FrameZoneKind::kNone;
    switch (t) {
        case BoltType::Bool: case BoltType::Int8: case BoltType::Int16:
        case BoltType::Int32: case BoltType::Int64: case BoltType::Date32:
        case BoltType::Date64: case BoltType::Timestamp: case BoltType::Duration:
        case BoltType::Decimal64:
            return FrameZoneKind::kI64;
        case BoltType::UInt8: case BoltType::UInt16: case BoltType::UInt32:
        case BoltType::UInt64:
            return FrameZoneKind::kU64;
        case BoltType::Float32: return FrameZoneKind::kF32;
        case BoltType::Float64: return FrameZoneKind::kF64;
        case BoltType::Utf8: case BoltType::Binary: case BoltType::Symbol:
            return FrameZoneKind::kStr8;
        default:                return FrameZoneKind::kNone;
    }
}

// Value of row r as a signed / unsigned 64-bit integer (Flat integer types).
inline int64_t frame_read_i64(const BoltColumn& c, int64_t r) noexcept {
    assert(r >= 0 && r < c.length);
    const uint8_t* p = static_cast<const uint8_t*>(c.data) + r * c.type_size_bytes;
    switch (c.type_size_bytes) {
        case 1: { int8_t v;  memcpy(&v, p, 1); return v; }
        case 2: { int16_t v; memcpy(&v, p, 2); return v; }
        case 4: { int32_t v; memcpy(&v, p, 4); return v; }
        default: { int64_t v; memcpy(&v, p, 8); return v; }
    }
}

inline uint64_t frame_read_u64(const BoltColumn& c, int64_t r) noexcept {
    assert(r >= 0 && r < c.length);
    const uint8_t* p = static_cast<const uint8_t*>(c.data) + r * c.type_size_bytes;
    switch (c.type_size_bytes) {
        case 1: return p[0];
        case 2: { uint16_t v; memcpy(&v, p, 2); return v; }
        case 4: { uint32_t v; memcpy(&v, p, 4); return v; }
        default: { uint64_t v; memcpy(&v, p, 8); return v; }
    }
}

// Fold row r (known valid) into z. False when the value carries no bound
// (NaN), so the caller does not treat the zone as seeded.
inline bool frame_zone_observe(ZoneMap* z, FrameZoneKind k, const BoltColumn& c,
                               int64_t r, bool first) noexcept {
    assert(z != nullptr);
    assert(r >= 0 && r < c.length);
    if (k == FrameZoneKind::kI64) {
        const int64_t v = frame_read_i64(c, r);
        if (first || v < z->min_value) z->min_value = v;
        if (first || v > z->max_value) z->max_value = v;
    } else if (k == FrameZoneKind::kU64) {
        const uint64_t v = frame_read_u64(c, r);
        if (first || v < static_cast<uint64_t>(z->min_value)) memcpy(&z->min_value, &v, 8);
        if (first || v > static_cast<uint64_t>(z->max_value)) memcpy(&z->max_value, &v, 8);
    } else if (k == FrameZoneKind::kF32) {
        float v; memcpy(&v, static_cast<const float*>(c.data) + r, 4);
        if (v != v) return false;                 // NaN carries no bound
        if (first || v < zone_min_f32(z)) zone_set_min_f32(z, v);
        if (first || v > zone_max_f32(z)) zone_set_max_f32(z, v);
    } else if (k == FrameZoneKind::kF64) {
        double v; memcpy(&v, static_cast<const double*>(c.data) + r, 8);
        if (v != v) return false;
        if (first || v < zone_min_f64(z)) memcpy(&z->min_value, &v, 8);
        if (first || v > zone_max_f64(z)) memcpy(&z->max_value, &v, 8);
    } else {
        const uint8_t* p = nullptr;
        int32_t n = 0;
        c.utf8_at(r, &p, &n);
        zone_str8_observe(z, p, n);
    }
    return true;
}

}  // namespace detail

/// Scalar reference zone map for one column: min/max over valid rows,
/// null_count, all-valid / has-nulls flags. Returns the kind; kNone (and
/// min = max = 0) when the type has no order or no row is valid.
inline FrameZoneKind frame_zone_for_column(const BoltColumn& c, ZoneMap* z) noexcept {
    assert(z != nullptr);
    assert(c.length >= 0);
    *z = zone_make_empty_i64();
    z->min_value = 0;
    z->max_value = 0;
    FrameZoneKind k = detail::frame_zone_kind_for(c);
    if (k == FrameZoneKind::kStr8) *z = zone_make_empty_str8();
    uint32_t nulls = 0;
    bool seen = false;
    for (int64_t r = 0; r < c.length; ++r) {          // bounded: c.length
        if (c.is_null(r)) { ++nulls; continue; }
        if (k == FrameZoneKind::kNone) continue;
        if (detail::frame_zone_observe(z, k, c, r, !seen)) seen = true;
    }
    if (!seen && k != FrameZoneKind::kNone) {
        k = FrameZoneKind::kNone;
        z->min_value = 0;
        z->max_value = 0;
    }
    z->null_count = nulls;
    z->flags = nulls ? kZoneFlagHasNulls : kZoneFlagAllValid;
    return k;
}

namespace detail {

// Widen `dst` to cover `src` under kind k (both already bounded).
inline void frame_zone_merge(ZoneMap* dst, const ZoneMap& src, FrameZoneKind k) noexcept {
    assert(dst != nullptr);
    assert(k != FrameZoneKind::kNone);
    const uint32_t nulls = dst->null_count + src.null_count;
    if (k == FrameZoneKind::kI64) {
        zone_merge_i64(dst, &src);
    } else if (k == FrameZoneKind::kU64) {
        uint64_t a, b;
        memcpy(&a, &dst->min_value, 8); memcpy(&b, &src.min_value, 8);
        if (b < a) dst->min_value = src.min_value;
        memcpy(&a, &dst->max_value, 8); memcpy(&b, &src.max_value, 8);
        if (b > a) dst->max_value = src.max_value;
    } else if (k == FrameZoneKind::kF32) {
        zone_merge_f32(dst, &src);
    } else if (k == FrameZoneKind::kF64) {
        if (zone_min_f64(&src) < zone_min_f64(dst)) dst->min_value = src.min_value;
        if (zone_max_f64(&src) > zone_max_f64(dst)) dst->max_value = src.max_value;
    } else {
        if (memcmp(&src.min_value, &dst->min_value, 8) < 0) dst->min_value = src.min_value;
        if (memcmp(&src.max_value, &dst->max_value, 8) > 0) dst->max_value = src.max_value;
    }
    dst->null_count = nulls;
    dst->flags = nulls ? kZoneFlagHasNulls : kZoneFlagAllValid;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Single frames
// ---------------------------------------------------------------------------

/// What the caller knows about the frame that the batch does not carry.
struct FrameMeta {
    FrameOpKind kind;
    uint16_t    flags;
    uint32_t    crc_seed;   // 0 standalone; a container's seed inside a file
    uint64_t    lsn_min, lsn_max;
    int64_t     ts_min, ts_max;
    uint64_t    txn_id;
    uint64_t    first_pos;
    uint8_t     key_min[8], key_max[8];
};

enum class FrameZones : uint8_t {
    kNone     = 0,   // no zone maps
    kCompute  = 1,   // frame_zone_for_column per column (scalar reference)
    kProvided = 2,   // caller's zones[] / kinds[] (one per column)
};

/// Trailer bytes for n zones.
inline size_t frame_trailer_size(uint32_t n_zones) noexcept {
    assert(n_zones <= kFrameMaxZones);
    return detail::frame_align64(sizeof(FrameTrailer) + 33u * static_cast<size_t>(n_zones));
}

/// Total frame bytes for `b` with zones on (n_zones = num_cols) or off, or 0
/// when the batch cannot be serialised.
inline size_t frame_size(const BoltBatch* b, bool with_zones) noexcept {
    assert(b != nullptr);
    assert(b->num_cols <= kWireMaxCols);
    const size_t payload = bolt_wire_size(b);
    if (payload == 0 || payload > kFrameMaxPayloadBytes) return 0;
    return sizeof(FrameHeader) + detail::frame_align64(payload) +
           frame_trailer_size(with_zones ? b->num_cols : 0u);
}

namespace detail {

inline void frame_fill_header(FrameHeader* h, const FrameMeta& m, uint32_t rows,
                              uint32_t bytes) noexcept {
    assert(h != nullptr);
    assert(m.lsn_min <= m.lsn_max);
    memset(h, 0, sizeof(*h));
    h->kind = static_cast<uint16_t>(m.kind);
    h->flags = m.flags;
    h->lsn = m.lsn_min;
    h->commit_ts = m.ts_min;
    h->rows = rows;
    h->bytes = bytes;
    h->txn_id = m.txn_id;
    h->first_pos = m.first_pos;
    memcpy(h->key_min, m.key_min, 8);
    memcpy(h->key_max, m.key_max, 8);
}

inline void frame_fill_trailer(FrameTrailer* t, const FrameMeta& m, uint32_t rows,
                               uint32_t n_zones, uint64_t bytes) noexcept {
    assert(t != nullptr);
    assert(n_zones <= kFrameMaxZones);
    memset(t, 0, sizeof(*t));
    t->magic = kFrameTrailerMagic;
    t->kind = static_cast<uint16_t>(m.kind);
    t->flags = m.flags;
    t->rows = rows;
    t->n_zones = n_zones;
    t->lsn_min = m.lsn_min;
    t->lsn_max = m.lsn_max;
    t->ts_min = m.ts_min;
    t->ts_max = m.ts_max;
    t->payload_bytes = bytes;
    t->trailer_bytes = static_cast<uint32_t>(frame_trailer_size(n_zones));
}

}  // namespace detail

/// Write one frame for `b` into out[0, cap) (out on a 64 B boundary). Returns
/// the frame length, or 0 with *st set (kUnsupported, kCapacity, kBadLayout,
/// kUnaligned). Every padding byte is
/// zero, so the same batch and meta give the same bytes (deterministic
/// writer, G11).
inline size_t frame_write(const BoltBatch* b, const FrameMeta& m, FrameZones zm,
                          const ZoneMap* zones, const uint8_t* kinds,
                          void* out, size_t cap, FrameStatus* st) noexcept {
    assert(b != nullptr && st != nullptr);
    assert(out != nullptr || cap == 0);
    *st = FrameStatus::kBadLayout;
    if (m.kind == FrameOpKind::kNone || m.lsn_min > m.lsn_max || m.ts_min > m.ts_max) return 0;
    if (zm == FrameZones::kProvided && (zones == nullptr || kinds == nullptr)) return 0;
    if (b->num_rows < 0 || b->num_rows > static_cast<int64_t>(UINT32_MAX)) return 0;
    *st = FrameStatus::kUnaligned;
    if ((reinterpret_cast<uintptr_t>(out) & 63u) != 0) return 0;
    const bool with_zones = zm != FrameZones::kNone;
    const size_t total = frame_size(b, with_zones);
    *st = total == 0 ? FrameStatus::kUnsupported : FrameStatus::kCapacity;
    if (total == 0 || total > cap) return 0;
    uint8_t* p = static_cast<uint8_t*>(out);
    const size_t payload =
        bolt_wire_serialize(b, p + sizeof(FrameHeader), total - sizeof(FrameHeader));
    *st = FrameStatus::kUnsupported;
    if (payload == 0) return 0;
    const size_t tr_off = sizeof(FrameHeader) + detail::frame_align64(payload);
    memset(p + sizeof(FrameHeader) + payload, 0, tr_off - sizeof(FrameHeader) - payload);
    const uint32_t nz = with_zones ? b->num_cols : 0u;
    const size_t tr_len = frame_trailer_size(nz);
    memset(p + tr_off, 0, tr_len);
    const uint32_t rows = static_cast<uint32_t>(b->num_rows);
    detail::frame_fill_trailer(reinterpret_cast<FrameTrailer*>(p + tr_off), m, rows, nz, payload);
    auto* zout = reinterpret_cast<ZoneMap*>(p + tr_off + sizeof(FrameTrailer));
    uint8_t* kout = p + tr_off + sizeof(FrameTrailer) + 32u * static_cast<size_t>(nz);
    for (uint32_t i = 0; i < nz; ++i) {               // bounded: num_cols
        if (zm == FrameZones::kProvided) {
            zout[i] = zones[i];
            kout[i] = kinds[i];
        } else {
            kout[i] = static_cast<uint8_t>(frame_zone_for_column(b->col(i), &zout[i]));
        }
    }
    detail::frame_fill_header(reinterpret_cast<FrameHeader*>(p), m, rows,
                              static_cast<uint32_t>(payload));
    const uint32_t crc = io::crc32c(p + 4, total - 4, m.crc_seed);
    memcpy(p, &crc, 4);
    *st = FrameStatus::kOk;
    assert(tr_off + tr_len == total);
    return total;
}

/// A parsed frame. Pointers alias the caller's buffer.
struct FrameView {
    FrameHeader         header;     // copy
    FrameTrailer        trailer;    // copy
    const uint8_t*      payload;    // bolt wire bytes: bolt_wire_view(payload, payload_len, ...)
    size_t              payload_len;
    const ZoneMap*      zones;      // n_zones, 32 B aligned when the frame is 64 B aligned
    const uint8_t*      zone_kinds;
    uint32_t            n_zones;
    uint32_t            _pad;
    size_t              frame_len;
};

namespace detail {

// Lengths and trailer consistency (no CRC).
inline FrameStatus frame_parse_shape(const uint8_t* p, size_t len, FrameView* v) noexcept {
    assert(p != nullptr && v != nullptr);
    assert(len >= sizeof(FrameHeader));
    memcpy(&v->header, p, sizeof(FrameHeader));
    const FrameHeader& h = v->header;
    if (h.kind == 0 && h.bytes == 0) return FrameStatus::kEnd;
    const size_t tr_off = sizeof(FrameHeader) + frame_align64(h.bytes);
    if (h.bytes < kWireHeaderSize || tr_off + sizeof(FrameTrailer) > len)
        return FrameStatus::kTruncated;
    memcpy(&v->trailer, p + tr_off, sizeof(FrameTrailer));
    const FrameTrailer& t = v->trailer;
    if (t.magic != kFrameTrailerMagic) return FrameStatus::kBadMagic;
    if (t.n_zones > kFrameMaxZones || t.trailer_bytes != frame_trailer_size(t.n_zones))
        return FrameStatus::kBadLayout;
    if (tr_off + t.trailer_bytes > len) return FrameStatus::kTruncated;
    if (t.kind != h.kind || t.flags != h.flags || t.rows != h.rows ||
        t.payload_bytes != h.bytes || t.lsn_min != h.lsn || t.ts_min != h.commit_ts ||
        t.lsn_min > t.lsn_max || t.ts_min > t.ts_max)
        return FrameStatus::kBadLayout;
    v->payload = p + sizeof(FrameHeader);
    v->payload_len = h.bytes;
    v->zones = reinterpret_cast<const ZoneMap*>(p + tr_off + sizeof(FrameTrailer));
    v->zone_kinds = p + tr_off + sizeof(FrameTrailer) + 32u * static_cast<size_t>(t.n_zones);
    v->n_zones = t.n_zones;
    v->frame_len = tr_off + t.trailer_bytes;
    return FrameStatus::kOk;
}

}  // namespace detail

/// Parse the frame at p[0, len) (p on a 64 B boundary). With verify_crc the
/// checksum (seeded with `crc_seed`, the writer's FrameMeta::crc_seed) is
/// checked before anything is returned. kEnd for a zero header.
inline FrameStatus frame_parse(const void* buf, size_t len, bool verify_crc,
                               FrameView* out, uint32_t crc_seed = 0) noexcept {
    assert(out != nullptr);
    assert(buf != nullptr || len == 0);
    memset(out, 0, sizeof(*out));
    if (len < sizeof(FrameHeader)) return FrameStatus::kTruncated;
    if ((reinterpret_cast<uintptr_t>(buf) & 63u) != 0) return FrameStatus::kUnaligned;
    const uint8_t* p = static_cast<const uint8_t*>(buf);
    const FrameStatus s = detail::frame_parse_shape(p, len, out);
    if (s != FrameStatus::kOk) return s;
    if (verify_crc && io::crc32c(p + 4, out->frame_len - 4, crc_seed) != out->header.crc32c)
        return FrameStatus::kBadCrc;
    return FrameStatus::kOk;
}

}  // namespace wire
}  // namespace bolt
