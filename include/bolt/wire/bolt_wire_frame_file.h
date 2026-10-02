// bolt_wire_frame_file.h — the multi-frame container: header, frames, index,
// per-column rollup and footer (MSEG B2). Layout: bolt_wire_frame.h.
//
// Writer: frame_file_begin over a caller buffer (a preallocated mmap'd WAL
// segment, a spill buffer), frame_file_append per batch, frame_file_seal to
// append the index region and footer in place. A crashed, unsealed file is
// reopened with frame_file_resume, which walks the frames (CRC-checked),
// rebuilds the index and rollup, and can then be sealed.
//
// Reader: frame_file_open validates the header (including the recorded
// alignment units) and, when present, the footer and index CRC; then
// frame_file_frame parses frame i through the index. An unsealed file opens
// with sealed == false; frame_file_recover lists its frames.
//
// Every frame, index and footer CRC in a file is seeded with the header's
// CRC (frame_file_seed), and the header covers a caller-chosen incarnation,
// so bytes left by an earlier file in a recycled, non-zeroed buffer (an old
// footer, an old frame past the new tail) never validate. The caller must
// pass a different incarnation for every file it creates (a counter, a clock
// reading, random bits). All offsets read from the file are bounds-checked
// without overflow: the file is untrusted input.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "bolt/wire/bolt_wire_frame.h"

namespace bolt {
namespace wire {

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

namespace detail {

inline uint32_t frame_file_header_crc(const FrameFileHeader& h) noexcept {
    static_assert(offsetof(FrameFileHeader, header_crc32c) == 60, "crc span");
    assert(h.version != 0 || h.magic[0] == 0);
    return io::crc32c(&h, offsetof(FrameFileHeader, header_crc32c));
}

inline size_t frame_align_up(size_t x, uint32_t unit) noexcept {
    assert(unit != 0 && (unit & (unit - 1)) == 0);
    assert(x <= (static_cast<size_t>(1) << 62));
    return (x + unit - 1) & ~static_cast<size_t>(unit - 1);
}

}  // namespace detail

/// The CRC seed of every frame, the index and the footer of a file.
inline uint32_t frame_file_seed(const FrameFileHeader& h) noexcept {
    assert(h.version != 0 || h.magic[0] == 0);
    assert(h.buf_align_log2 < 32);
    return h.header_crc32c;
}

/// Read and validate a container header: magic, version, CRC, alignment
/// units (the status names the failing unit), frames_off.
inline FrameStatus frame_file_read_header(const void* buf, size_t len,
                                          FrameFileHeader* out) noexcept {
    assert(out != nullptr);
    assert(buf != nullptr || len == 0);
    if (len < sizeof(FrameFileHeader)) return FrameStatus::kTruncated;
    if ((reinterpret_cast<uintptr_t>(buf) & 63u) != 0) return FrameStatus::kUnaligned;
    memcpy(out, buf, sizeof(*out));
    if (memcmp(out->magic, "BWFF", 4) != 0) return FrameStatus::kBadMagic;
    if (detail::frame_file_header_crc(*out) != out->header_crc32c) return FrameStatus::kBadCrc;
    if (out->version != kFrameFileVersion) return FrameStatus::kBadVersion;
    const FrameAlign a{out->buf_align_log2, out->chunk_align_log2, out->stripe_align_log2,
                       out->io_align_log2};
    const FrameStatus s = frame_align_validate(a);
    if (s != FrameStatus::kOk) return s;
    const uint32_t buf_unit = 1u << out->buf_align_log2;
    if (out->frames_off < sizeof(FrameFileHeader) || out->frames_off % buf_unit != 0 ||
        out->frames_off > len)
        return FrameStatus::kBadLayout;
    return FrameStatus::kOk;
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

struct FrameFileWriter {
    uint8_t*         buf;
    size_t           cap;
    size_t           pos;            // end of the last frame
    FrameIndexEntry* index;          // caller array, index_cap entries
    ZoneMap*         rollup;         // caller array, rollup_cap entries (may be null)
    uint8_t*         rollup_kind;
    uint32_t         index_cap;
    uint32_t         n_frames;
    uint32_t         rollup_cap;
    uint32_t         n_rollup;
    uint8_t          rollup_state;   // 0 no frame yet, 1 active, 2 off
    uint8_t          sealed;
    uint8_t          _pad[6];
    FrameFileHeader  header;
    uint64_t         total_rows;
    uint64_t         lsn_min, lsn_max;
    uint32_t         seed;           // frame_file_seed(header)
    uint32_t         _pad2;
};

namespace detail {

inline void frame_file_writer_reset(FrameFileWriter* w, uint8_t* buf, size_t cap,
                                    FrameIndexEntry* index, uint32_t index_cap,
                                    ZoneMap* rollup, uint8_t* rollup_kind,
                                    uint32_t rollup_cap) noexcept {
    assert(w != nullptr);
    assert(index != nullptr || index_cap == 0);
    memset(w, 0, sizeof(*w));
    w->buf = buf;
    w->cap = cap;
    w->index = index;
    w->index_cap = index_cap;
    w->rollup = (rollup != nullptr && rollup_kind != nullptr) ? rollup : nullptr;
    w->rollup_kind = w->rollup ? rollup_kind : nullptr;
    w->rollup_cap = w->rollup ? rollup_cap : 0u;
    w->lsn_min = UINT64_MAX;
}

// Fold one frame's zones into the file rollup. A column that has no bounds
// in one frame (all null) takes the other's; differing kinds or column
// counts mean the schema changed, and the rollup is dropped.
inline void frame_file_fold(FrameFileWriter* w, const FrameView& v) noexcept {
    assert(w != nullptr);
    assert(v.n_zones <= kFrameMaxZones);
    if (w->rollup_state == 2 || w->rollup == nullptr) { w->rollup_state = 2; return; }
    if (w->rollup_state == 0) {
        if (v.n_zones > w->rollup_cap) { w->rollup_state = 2; return; }
        memcpy(w->rollup, v.zones, 32u * static_cast<size_t>(v.n_zones));
        memcpy(w->rollup_kind, v.zone_kinds, v.n_zones);
        w->n_rollup = v.n_zones;
        w->rollup_state = 1;
        return;
    }
    if (v.n_zones != w->n_rollup) { w->rollup_state = 2; w->n_rollup = 0; return; }
    for (uint32_t i = 0; i < v.n_zones; ++i) {        // bounded: n_zones
        const FrameZoneKind dk = static_cast<FrameZoneKind>(w->rollup_kind[i]);
        const FrameZoneKind sk = static_cast<FrameZoneKind>(v.zone_kinds[i]);
        const uint32_t nulls = w->rollup[i].null_count + v.zones[i].null_count;
        if (dk == FrameZoneKind::kNone) {
            w->rollup[i] = v.zones[i];
            w->rollup_kind[i] = v.zone_kinds[i];
        } else if (sk == dk) {
            frame_zone_merge(&w->rollup[i], v.zones[i], dk);
        } else if (sk != FrameZoneKind::kNone) {
            w->rollup_state = 2; w->n_rollup = 0; return;
        }
        w->rollup[i].null_count = nulls;
        w->rollup[i].flags = nulls ? kZoneFlagHasNulls : kZoneFlagAllValid;
    }
}

// Zero the header slot after the last frame so a walk stops there even when
// the buffer held older bytes (a torn tail, a reused buffer).
inline void frame_file_terminate(FrameFileWriter* w) noexcept {
    assert(w != nullptr);
    assert(w->pos <= w->cap);
    const size_t n = w->cap - w->pos < sizeof(FrameHeader) ? w->cap - w->pos
                                                           : sizeof(FrameHeader);
    memset(w->buf + w->pos, 0, n);
}

inline FrameStatus frame_file_record(FrameFileWriter* w, const FrameView& v,
                                     size_t off) noexcept {
    assert(w != nullptr);
    assert(off % (1u << w->header.buf_align_log2) == 0);
    if (w->n_frames >= w->index_cap) return FrameStatus::kCapacity;
    FrameIndexEntry& e = w->index[w->n_frames];
    memset(&e, 0, sizeof(e));
    e.off = off;
    e.len = v.frame_len;
    e.rows = v.header.rows;
    e.kind = v.header.kind;
    e.flags = v.header.flags;
    e.lsn_min = v.trailer.lsn_min;
    e.lsn_max = v.trailer.lsn_max;
    e.ts_min = v.trailer.ts_min;
    e.ts_max = v.trailer.ts_max;
    e.crc32c = v.header.crc32c;
    e.n_zones = v.n_zones;
    ++w->n_frames;
    w->total_rows += v.header.rows;
    if (v.trailer.lsn_min < w->lsn_min) w->lsn_min = v.trailer.lsn_min;
    if (v.trailer.lsn_max > w->lsn_max) w->lsn_max = v.trailer.lsn_max;
    frame_file_fold(w, v);
    return FrameStatus::kOk;
}

}  // namespace detail

/// Start a container in buf[0, cap) recording alignment units `a` (validated;
/// the buf unit must be the wire's 64 B). `index` holds up to index_cap frame
/// entries; `rollup`/`rollup_kind` (nullable) hold the per-column rollup.
inline FrameStatus frame_file_begin_with(FrameFileWriter* w, void* buf, size_t cap,
                                         const FrameAlign& a, FrameFilePurpose purpose,
                                         uint64_t id0, uint64_t id1, uint64_t incarnation,
                                         FrameIndexEntry* index, uint32_t index_cap,
                                         ZoneMap* rollup, uint8_t* rollup_kind,
                                         uint32_t rollup_cap) noexcept {
    assert(w != nullptr);
    assert(buf != nullptr || cap == 0);
    detail::frame_file_writer_reset(w, static_cast<uint8_t*>(buf), cap, index, index_cap,
                                    rollup, rollup_kind, rollup_cap);
    const FrameStatus vs = frame_align_validate(a);
    if (vs != FrameStatus::kOk) return vs;
    if ((1u << a.buf_log2) != kFrameBufAlign) return FrameStatus::kBadBufAlign;
    if (buf == nullptr || cap < 2 * sizeof(FrameFileHeader)) return FrameStatus::kCapacity;
    FrameFileHeader& h = w->header;
    memcpy(h.magic, "BWFF", 4);
    h.version = kFrameFileVersion;
    h.purpose = static_cast<uint16_t>(purpose);
    h.buf_align_log2 = a.buf_log2;
    h.chunk_align_log2 = a.chunk_log2;
    h.stripe_align_log2 = a.stripe_log2;
    h.io_align_log2 = a.io_log2;
    h.id0 = id0;
    h.id1 = id1;
    h.incarnation = incarnation;
    h.frames_off = detail::frame_align_up(sizeof(FrameFileHeader), kFrameBufAlign);
    h.header_crc32c = detail::frame_file_header_crc(h);
    w->seed = frame_file_seed(h);
    memcpy(w->buf, &h, sizeof(h));
    memset(w->buf + sizeof(h), 0, h.frames_off - sizeof(h));
    w->pos = h.frames_off;
    detail::frame_file_terminate(w);
    return FrameStatus::kOk;
}

/// frame_file_begin_with this build's units (bolt_wire_limits.h).
inline FrameStatus frame_file_begin(FrameFileWriter* w, void* buf, size_t cap,
                                    FrameFilePurpose purpose, uint64_t id0, uint64_t id1,
                                    uint64_t incarnation,
                                    FrameIndexEntry* index, uint32_t index_cap,
                                    ZoneMap* rollup, uint8_t* rollup_kind,
                                    uint32_t rollup_cap) noexcept {
    assert(w != nullptr);
    assert(buf != nullptr || cap == 0);
    return frame_file_begin_with(w, buf, cap, frame_align_default(), purpose, id0, id1,
                                 incarnation, index, index_cap, rollup, rollup_kind, rollup_cap);
}

/// Append one frame (see frame_write for zone modes).
inline FrameStatus frame_file_append(FrameFileWriter* w, const BoltBatch* b,
                                     const FrameMeta& m, FrameZones zm,
                                     const ZoneMap* zones, const uint8_t* kinds) noexcept {
    assert(w != nullptr && b != nullptr);
    assert(w->pos % 64u == 0);
    if (w->sealed) return FrameStatus::kBadLayout;
    if (w->n_frames >= w->index_cap) return FrameStatus::kCapacity;
    FrameStatus st = FrameStatus::kOk;
    FrameMeta sm = m;
    sm.crc_seed = w->seed;
    const size_t len = frame_write(b, sm, zm, zones, kinds, w->buf + w->pos, w->cap - w->pos, &st);
    if (len == 0) return st;
    FrameView v;
    st = frame_parse(w->buf + w->pos, len, false, &v, w->seed);
    assert(st == FrameStatus::kOk);
    if (st != FrameStatus::kOk) return st;
    st = detail::frame_file_record(w, v, w->pos);
    if (st != FrameStatus::kOk) return st;
    w->pos += len;
    detail::frame_file_terminate(w);
    return FrameStatus::kOk;
}

/// Bytes frame_file_seal will add after the last frame (worst case).
inline size_t frame_file_seal_bytes(const FrameFileWriter* w) noexcept {
    assert(w != nullptr);
    const uint32_t chunk = 1u << w->header.chunk_align_log2;
    const size_t idx = sizeof(FrameIndexEntry) * w->n_frames +
                       detail::frame_align64(33u * static_cast<size_t>(w->n_rollup));
    return detail::frame_align_up(w->pos, chunk) - w->pos +
           detail::frame_align_up(idx + sizeof(FrameFileFooter), chunk);
}

/// Append the index region and footer; returns the sealed file length (a
/// multiple of the chunk unit), or 0 when the buffer is too small.
inline size_t frame_file_seal(FrameFileWriter* w) noexcept {
    assert(w != nullptr);
    assert(w->pos % 64u == 0);
    if (w->sealed || w->pos + frame_file_seal_bytes(w) > w->cap) return 0;
    const uint32_t chunk = 1u << w->header.chunk_align_log2;
    const uint32_t n_roll = w->rollup_state == 1 ? w->n_rollup : 0u;
    const size_t index_off = detail::frame_align_up(w->pos, chunk);
    const size_t ent = sizeof(FrameIndexEntry) * w->n_frames;
    const size_t index_len = ent + detail::frame_align64(33u * static_cast<size_t>(n_roll));
    const size_t file_len =
        detail::frame_align_up(index_off + index_len + sizeof(FrameFileFooter), chunk);
    uint8_t* p = w->buf;
    memset(p + w->pos, 0, file_len - w->pos);
    if (ent != 0) memcpy(p + index_off, w->index, ent);
    if (n_roll != 0) {
        memcpy(p + index_off + ent, w->rollup, 32u * static_cast<size_t>(n_roll));
        memcpy(p + index_off + ent + 32u * n_roll, w->rollup_kind, n_roll);
    }
    FrameFileFooter f;
    memset(&f, 0, sizeof(f));
    f.magic = kFrameFooterMagic;
    f.version = kFrameFileVersion;
    f.purpose = w->header.purpose;
    f.index_off = index_off;
    f.index_len = index_len;
    f.total_rows = w->total_rows;
    f.lsn_min = w->n_frames ? w->lsn_min : 0u;
    f.lsn_max = w->lsn_max;
    f.n_frames = w->n_frames;
    f.n_rollup = n_roll;
    f.index_crc32c = io::crc32c(p + index_off, index_len, w->seed);
    f.footer_crc32c = io::crc32c(&f, offsetof(FrameFileFooter, footer_crc32c), w->seed);
    memcpy(p + file_len - sizeof(f), &f, sizeof(f));
    w->sealed = 1;
    assert(file_len % chunk == 0);
    return file_len;
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

struct FrameFileView {
    FrameFileHeader        header;
    FrameFileFooter        footer;       // zero when !sealed
    const uint8_t*         base;
    size_t                 len;
    const FrameIndexEntry* index;        // sealed only
    const ZoneMap*         rollup;       // sealed only
    const uint8_t*         rollup_kind;
    uint32_t               n_frames;
    uint32_t               n_rollup;
    uint32_t               seed;         // frame_file_seed(header)
    uint8_t                sealed;
    uint8_t                _pad[3];
};

namespace detail {

inline FrameStatus frame_file_read_footer(FrameFileView* v) noexcept {
    assert(v != nullptr);
    assert(v->len >= 2 * sizeof(FrameFileHeader));
    FrameFileFooter& f = v->footer;
    memcpy(&f, v->base + v->len - sizeof(f), sizeof(f));
    if (f.magic != kFrameFooterMagic) { memset(&f, 0, sizeof(f)); return FrameStatus::kEnd; }
    if (io::crc32c(&f, offsetof(FrameFileFooter, footer_crc32c), v->seed) != f.footer_crc32c)
        return FrameStatus::kBadCrc;
    if (f.version != kFrameFileVersion) return FrameStatus::kBadVersion;
    if (f.n_rollup > kFrameMaxZones) return FrameStatus::kBadLayout;
    const uint32_t chunk = 1u << v->header.chunk_align_log2;
    const size_t ent = sizeof(FrameIndexEntry) * static_cast<size_t>(f.n_frames);
    const size_t need = ent + frame_align64(33u * static_cast<size_t>(f.n_rollup));
    const size_t body = v->len - sizeof(f);      // len >= 128 (asserted above)
    if (f.index_off % chunk != 0 || f.index_len != need ||
        f.index_off < v->header.frames_off || f.index_off > body ||
        f.index_len > body - f.index_off)
        return FrameStatus::kBadLayout;
    if (io::crc32c(v->base + f.index_off, f.index_len, v->seed) != f.index_crc32c)
        return FrameStatus::kBadCrc;
    v->index = reinterpret_cast<const FrameIndexEntry*>(v->base + f.index_off);
    v->rollup = reinterpret_cast<const ZoneMap*>(v->base + f.index_off + ent);
    v->rollup_kind = v->base + f.index_off + ent + 32u * static_cast<size_t>(f.n_rollup);
    v->n_frames = f.n_frames;
    v->n_rollup = f.n_rollup;
    v->sealed = 1;
    return FrameStatus::kOk;
}

}  // namespace detail

/// Open a container. kOk with sealed == 0 means no footer (recover it). A
/// footer that fails validation is an error, but v->base is still set (the
/// header is good), so a recovery path may walk the frames: a footer that
/// fails its seeded CRC is most often one left by an earlier file in a
/// recycled buffer.
inline FrameStatus frame_file_open(const void* buf, size_t len, FrameFileView* v) noexcept {
    assert(v != nullptr);
    assert(buf != nullptr || len == 0);
    memset(v, 0, sizeof(*v));
    const FrameStatus hs = frame_file_read_header(buf, len, &v->header);
    if (hs != FrameStatus::kOk) return hs;
    v->base = static_cast<const uint8_t*>(buf);
    v->len = len;
    v->seed = frame_file_seed(v->header);
    if (len < 2 * sizeof(FrameFileHeader)) return FrameStatus::kOk;
    const FrameStatus fs = detail::frame_file_read_footer(v);
    return fs == FrameStatus::kEnd ? FrameStatus::kOk : fs;
}

/// Parse frame i of a sealed file through its index entry.
inline FrameStatus frame_file_frame(const FrameFileView& v, uint32_t i, bool verify_crc,
                                    FrameView* out) noexcept {
    assert(out != nullptr);
    assert(v.sealed);
    if (i >= v.n_frames) return FrameStatus::kBadLayout;
    const FrameIndexEntry& e = v.index[i];
    const uint64_t unit = 1u << v.header.buf_align_log2;
    if (e.off % unit != 0 || e.off < v.header.frames_off || e.off > v.footer.index_off ||
        e.len > v.footer.index_off - e.off)
        return FrameStatus::kBadLayout;
    const FrameStatus s = frame_parse(v.base + e.off, e.len, verify_crc, out, v.seed);
    if (s != FrameStatus::kOk) return s;
    if (out->frame_len != e.len || out->header.crc32c != e.crc32c) return FrameStatus::kBadLayout;
    return FrameStatus::kOk;
}

/// Walk the frames of an (unsealed or sealed) file from frames_off,
/// verifying each CRC. Writes up to `cap` entries (offsets relative to the
/// file), sets *end to the byte after the last good frame and *why to the
/// reason the walk stopped (kEnd for a zero header or the end of the data
/// region, else the first bad frame's status). Returns entries written, or -1
/// when more frames exist than `cap`.
inline int64_t frame_file_recover(const FrameFileView& v, FrameIndexEntry* out, uint32_t cap,
                                  size_t* end, FrameStatus* why) noexcept {
    assert(end != nullptr && why != nullptr);
    assert(out != nullptr || cap == 0);
    const size_t limit = v.sealed ? static_cast<size_t>(v.footer.index_off) : v.len;
    const uint32_t unit = 1u << v.header.buf_align_log2;
    size_t off = v.header.frames_off;
    uint32_t n = 0;
    *why = FrameStatus::kEnd;
    while (off <= limit && limit - off >= sizeof(FrameHeader)) {   // advances >= 128 B per frame
        FrameView fv;
        const FrameStatus s = frame_parse(v.base + off, limit - off, true, &fv, v.seed);
        if (s != FrameStatus::kOk) { *why = s; break; }
        if (n >= cap) { *end = off; return -1; }
        FrameIndexEntry& e = out[n++];
        memset(&e, 0, sizeof(e));
        e.off = off; e.len = fv.frame_len; e.rows = fv.header.rows;
        e.kind = fv.header.kind; e.flags = fv.header.flags;
        e.lsn_min = fv.trailer.lsn_min; e.lsn_max = fv.trailer.lsn_max;
        e.ts_min = fv.trailer.ts_min; e.ts_max = fv.trailer.ts_max;
        e.crc32c = fv.header.crc32c; e.n_zones = fv.n_zones;
        off = detail::frame_align_up(off + fv.frame_len, unit);
    }
    *end = off < limit ? off : limit;
    return n;
}

/// Reopen an unsealed container (a crash before seal) for appending and
/// sealing: validates the header, walks and CRC-checks the frames, rebuilds
/// the index and rollup, and positions the writer after the last good frame
/// (a torn tail is overwritten by the next append or the seal). A footer
/// that does not validate under this file's seed (a stale one from an
/// earlier file in the same buffer) is ignored; a valid footer means the
/// file is sealed and is refused. The writer only appends at a 64 B unit.
inline FrameStatus frame_file_resume(FrameFileWriter* w, void* buf, size_t cap,
                                     FrameIndexEntry* index, uint32_t index_cap,
                                     ZoneMap* rollup, uint8_t* rollup_kind,
                                     uint32_t rollup_cap) noexcept {
    assert(w != nullptr);
    assert(buf != nullptr || cap == 0);
    detail::frame_file_writer_reset(w, static_cast<uint8_t*>(buf), cap, index, index_cap,
                                    rollup, rollup_kind, rollup_cap);
    FrameFileView v;
    const FrameStatus os = frame_file_open(buf, cap, &v);
    if (os != FrameStatus::kOk && v.base == nullptr) return os;   // header refused
    if (os == FrameStatus::kOk && v.sealed) return FrameStatus::kBadLayout;
    if ((1u << v.header.buf_align_log2) != kFrameBufAlign) return FrameStatus::kBadBufAlign;
    w->header = v.header;
    w->seed = v.seed;
    size_t off = v.header.frames_off;
    for (uint64_t guard = 0; guard <= cap / 128u; ++guard) {   // each frame >= 128 B
        FrameView fv;
        if (off > cap || cap - off < sizeof(FrameHeader) ||
            frame_parse(w->buf + off, cap - off, true, &fv, w->seed) != FrameStatus::kOk) break;
        const FrameStatus rs = detail::frame_file_record(w, fv, off);
        if (rs != FrameStatus::kOk) return rs;
        off += fv.frame_len;
    }
    w->pos = off;
    detail::frame_file_terminate(w);
    return FrameStatus::kOk;
}

}  // namespace wire
}  // namespace bolt
