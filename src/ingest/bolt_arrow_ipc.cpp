// bolt_arrow_ipc.cpp — Arrow IPC stream writer (G2ARROW-10; nested
// List/Struct G2ARROW-21).
// See include/bolt/ingest/bolt_arrow_ipc.h for the contract.
//
// The message headers are flatbuffers encoded FROM THE SPEC
// (https://flatbuffers.dev/internals/ + arrow/format/{Message,Schema}.fbs)
// by a minimal bottom-up builder: the buffer is filled back-to-front,
// positions are measured from the buffer END (so start-relative
// alignment holds once the total is a multiple of the max alignment),
// vtables carry [u16 vtable_bytes][u16 table_bytes][u16 field_offsets…],
// and every reference is a u32 forward offset from the referencing
// field to its target.

#include "bolt/ingest/bolt_arrow_ipc.h"

#include <cassert>
#include <cstdio>
#include <cstring>

#include "bolt/bolt_arrow.h"     // bolt::arrow::detail::var_at — the ONE
                                 // per-value Utf8 resolution, shared with
                                 // the C Data export so the two transports
                                 // cannot drift.
#include "bolt/bolt_column.h"

namespace bolt::ingest {

namespace {

// ---- Arrow flatbuffer schema constants (Message.fbs / Schema.fbs) ----
constexpr std::int16_t kMetadataV5     = 4;   // MetadataVersion::V5
constexpr std::uint8_t kHeaderSchema   = 1;   // MessageHeader union
constexpr std::uint8_t kHeaderBatch    = 3;
// Type union member ordinals — 1-indexed by flatbuffer declaration order
// (format/Schema.fbs `union Type`, fetched live and re-verified for
// G2ARROW-21: Null, Int, FloatingPoint, Binary, Utf8, Bool, Decimal, Date,
// Time, Timestamp, Interval, List, Struct_, Union, FixedSizeBinary,
// FixedSizeList, Map, ...).
constexpr std::uint8_t kTypeInt        = 2;
constexpr std::uint8_t kTypeFloat      = 3;
constexpr std::uint8_t kTypeBinary     = 4;
constexpr std::uint8_t kTypeUtf8       = 5;
constexpr std::uint8_t kTypeBool       = 6;
constexpr std::uint8_t kTypeDecimal    = 7;
constexpr std::uint8_t kTypeDate       = 8;
constexpr std::uint8_t kTypeTimestamp  = 10;
constexpr std::uint8_t kTypeList       = 12;
constexpr std::uint8_t kTypeStruct     = 13;
constexpr std::int16_t kPrecisionDouble = 2;  // FloatingPoint::Precision
constexpr std::int16_t kDateUnitDay     = 0;  // DateUnit::DAY
constexpr std::int16_t kTimeUnitMicro   = 2;  // TimeUnit::MICROSECOND
constexpr std::int32_t kDecimal128BitWidth  = 128;
constexpr std::int32_t kDecimal128Precision = 38;  // bolt tracks no
                                                    // narrower precision;
                                                    // matches bolt_arrow.h.

constexpr std::uint32_t kContinuation  = 0xFFFFFFFFu;

// ---- minimal bottom-up flatbuffer builder --------------------------------

constexpr std::uint16_t kFbMaxSlots = 8;

struct Fb {
    std::uint8_t* buf;        // capacity kIpcFbCap, filled from the end
    std::uint32_t cap;
    std::uint32_t used;       // bytes written, measured from the end
    std::uint32_t minalign;
    bool          overflow;
    // in-flight table state
    std::uint32_t object_start;
    std::uint32_t field_loc[kFbMaxSlots];   // pos-from-end; 0 = absent
};

void fb_init(Fb* b, std::uint8_t* storage, std::uint32_t cap) noexcept {
    assert(b != nullptr);
    assert(storage != nullptr);
    b->buf = storage; b->cap = cap; b->used = 0;
    b->minalign = 1;  b->overflow = false;
    b->object_start = 0;
    std::memset(b->field_loc, 0, sizeof(b->field_loc));
}

void fb_push(Fb* b, const void* p, std::uint32_t n) noexcept {
    assert(b != nullptr);
    if (b->used + n > b->cap) { b->overflow = true; return; }
    b->used += n;
    std::memcpy(b->buf + (b->cap - b->used), p, n);
}

void fb_pad(Fb* b, std::uint32_t n) noexcept {
    static const std::uint8_t zeros[8] = {0};
    while (n > 0 && !b->overflow) {
        const std::uint32_t take = (n > 8) ? 8u : n;
        fb_push(b, zeros, take);
        n -= take;
    }
}

// Pad so that, after `len` more bytes are pushed, `used` is a multiple
// of `a` (spec PreAlign). Tracks minalign so fb_finish can align the
// whole buffer.
void fb_prealign(Fb* b, std::uint32_t len, std::uint32_t a) noexcept {
    assert(a != 0 && (a & (a - 1)) == 0);
    if (a > b->minalign) b->minalign = a;
    const std::uint32_t rem = (b->used + len) & (a - 1);
    if (rem != 0) fb_pad(b, a - rem);
}

template <typename T>
void fb_push_scalar(Fb* b, T v) noexcept {
    fb_prealign(b, sizeof(T), sizeof(T));
    fb_push(b, &v, sizeof(T));
}

// Reference to an object at pos-from-end `target`: forward u32 offset.
void fb_push_uoffset(Fb* b, std::uint32_t target) noexcept {
    fb_prealign(b, 4, 4);
    assert(target != 0);
    assert(target <= b->used);
    const std::uint32_t v = (b->used + 4) - target;
    fb_push(b, &v, 4);
}

void fb_start_table(Fb* b) noexcept {
    b->object_start = b->used;
    std::memset(b->field_loc, 0, sizeof(b->field_loc));
}

template <typename T>
void fb_field_scalar(Fb* b, std::uint16_t slot, T v) noexcept {
    assert(slot < kFbMaxSlots);
    fb_push_scalar<T>(b, v);
    b->field_loc[slot] = b->used;
}

void fb_field_offset(Fb* b, std::uint16_t slot, std::uint32_t target) noexcept {
    assert(slot < kFbMaxSlots);
    fb_push_uoffset(b, target);
    b->field_loc[slot] = b->used;
}

// Close the table: soffset placeholder + vtable; returns the table's
// pos-from-end (0 on overflow).
std::uint32_t fb_end_table(Fb* b) noexcept {
    fb_prealign(b, 4, 4);
    const std::uint32_t zero = 0;
    fb_push(b, &zero, 4);
    const std::uint32_t table_pos = b->used;

    std::uint16_t n_slots = 0;
    for (std::uint16_t s = 0; s < kFbMaxSlots; ++s) {
        if (b->field_loc[s] != 0) n_slots = static_cast<std::uint16_t>(s + 1);
    }
    std::uint16_t vt[2 + kFbMaxSlots];
    vt[0] = static_cast<std::uint16_t>(4 + 2 * n_slots);      // vtable bytes
    vt[1] = static_cast<std::uint16_t>(table_pos - b->object_start);
    for (std::uint16_t s = 0; s < n_slots; ++s) {
        vt[2 + s] = (b->field_loc[s] != 0)
            ? static_cast<std::uint16_t>(table_pos - b->field_loc[s])
            : std::uint16_t{0};
    }
    fb_prealign(b, static_cast<std::uint32_t>(vt[0]), 2);
    fb_push(b, vt, vt[0]);
    const std::uint32_t vtable_pos = b->used;
    if (b->overflow) return 0;

    const std::int32_t soffset =
        static_cast<std::int32_t>(vtable_pos - table_pos);
    std::memcpy(b->buf + (b->cap - table_pos), &soffset, 4);
    return table_pos;
}

// [u32 len][bytes][NUL]; returns pos-from-end.
std::uint32_t fb_string(Fb* b, const char* s) noexcept {
    assert(s != nullptr);
    const std::uint32_t n = static_cast<std::uint32_t>(std::strlen(s));
    fb_prealign(b, 4 + n + 1, 4);
    const std::uint8_t nul = 0;
    fb_push(b, &nul, 1);
    fb_push(b, s, n);
    fb_push(b, &n, 4);
    return b->used;
}

// Vector of table references: elements pushed in reverse so element 0
// lands at the lowest address. Returns pos-from-end.
std::uint32_t fb_offset_vector(Fb* b, const std::uint32_t* targets,
                               std::uint32_t n) noexcept {
    fb_prealign(b, 4 + n * 4, 4);
    for (std::uint32_t i = n; i > 0; --i) {
        const std::uint32_t v = (b->used + 4) - targets[i - 1];
        fb_push(b, &v, 4);
    }
    fb_push(b, &n, 4);
    return b->used;
}

// Vector of 16-byte structs (FieldNode / Buffer): two int64s each,
// already little-endian in `pairs`. Returns pos-from-end.
std::uint32_t fb_struct16_vector(Fb* b, const std::int64_t* pairs,
                                 std::uint32_t n) noexcept {
    const std::uint32_t nbytes = n * 16;
    fb_prealign(b, 4 + nbytes, 4);
    fb_prealign(b, nbytes, 8);
    fb_push(b, pairs, nbytes);
    fb_push(b, &n, 4);
    return b->used;
}

// Root uoffset; returns the finished size. Buffer bytes then live at
// buf + cap - size.
std::uint32_t fb_finish(Fb* b, std::uint32_t root) noexcept {
    fb_prealign(b, 4, b->minalign);
    fb_push_uoffset(b, root);
    return b->overflow ? 0 : b->used;
}

// ---- schema message ------------------------------------------------------

// Build the Type union table for one BoltType. Writes the union tag to
// *out_tag. `decimal_scale` is used only for Decimal128. Returns table
// pos, 0 for an unsupported type (fail closed). List/Struct union tables
// are themselves EMPTY (format/Schema.fbs `table List {}` / `table
// Struct_ {}`) — the element/field type(s) live on the enclosing Field's
// `children` vector, built by the caller (build_all_fields below), not
// here.
std::uint32_t build_type_table(Fb* b, BoltType t, std::uint8_t decimal_scale,
                               std::uint8_t* out_tag) noexcept {
    assert(out_tag != nullptr);
    if (t == BoltType::Int64) {
        *out_tag = kTypeInt;
        fb_start_table(b);
        fb_field_scalar<std::int32_t>(b, 0, 64);        // bitWidth
        fb_field_scalar<std::uint8_t>(b, 1, 1);         // is_signed
        return fb_end_table(b);
    }
    if (t == BoltType::Float64) {
        *out_tag = kTypeFloat;
        fb_start_table(b);
        fb_field_scalar<std::int16_t>(b, 0, kPrecisionDouble);
        return fb_end_table(b);
    }
    if (t == BoltType::Utf8) {
        *out_tag = kTypeUtf8;
        fb_start_table(b);                               // empty table
        return fb_end_table(b);
    }
    if (t == BoltType::Binary) {
        *out_tag = kTypeBinary;
        fb_start_table(b);                               // empty table
        return fb_end_table(b);
    }
    if (t == BoltType::Bool) {
        *out_tag = kTypeBool;
        fb_start_table(b);                               // empty table
        return fb_end_table(b);
    }
    if (t == BoltType::Date32) {
        *out_tag = kTypeDate;
        fb_start_table(b);
        fb_field_scalar<std::int16_t>(b, 0, kDateUnitDay);  // unit = DAY
        return fb_end_table(b);
    }
    if (t == BoltType::Timestamp) {
        // bolt's Timestamp is epoch micros, no zone (timestamp[us]).
        *out_tag = kTypeTimestamp;
        fb_start_table(b);
        fb_field_scalar<std::int16_t>(b, 0, kTimeUnitMicro);  // unit
        return fb_end_table(b);
    }
    if (t == BoltType::Decimal128) {
        *out_tag = kTypeDecimal;
        fb_start_table(b);
        fb_field_scalar<std::int32_t>(b, 0, kDecimal128Precision);
        fb_field_scalar<std::int32_t>(
            b, 1, static_cast<std::int32_t>(decimal_scale));
        fb_field_scalar<std::int32_t>(b, 2, kDecimal128BitWidth);
        return fb_end_table(b);
    }
    if (t == BoltType::List) {
        *out_tag = kTypeList;
        fb_start_table(b);                               // empty table
        return fb_end_table(b);
    }
    if (t == BoltType::Struct) {
        *out_tag = kTypeStruct;
        fb_start_table(b);                               // empty table
        return fb_end_table(b);
    }
    return 0;                                            // unsupported
}

// ---- open()-time schema flattening (G2ARROW-21) --------------------------
//
// A FieldSpec tree is flattened into ArrowIpcWriter::desc[] BREADTH-FIRST
// (not the wire format's own depth-first pre-order — that numbering is
// produced separately, per batch, by layout_field/write_field_body below).
// Breadth-first is what keeps a node's own direct children CONTIGUOUS at
// `first_child..first_child+n_children-1` regardless of how large a
// SIBLING's own subtree turns out to be: a depth-first numbering does not
// have that property once a Struct has more than one child (a first
// child's own descendants would land between the first and second
// sibling's indices).
//
// `name_buf[i]` (out param, caller-allocated, transient — used only
// during open(), never stored on the writer) receives field i's final
// name: `spec->name` verbatim if non-null/non-empty, else the synthesized
// default (top-level "cN"; a List's sole child "item"; a Struct's Kth
// child "fK").
bool flatten_fields(ArrowIpcWriter* w, const FieldSpec* top,
                    std::uint16_t n_cols,
                    char name_buf[kIpcMaxFields][kIpcNameCap]) noexcept {
    assert(w != nullptr && top != nullptr && name_buf != nullptr);
    if (n_cols == 0 || n_cols > kIpcMaxCols) return false;

    const FieldSpec* src[kIpcMaxFields];
    std::uint16_t depth[kIpcMaxFields];
    w->n_desc = n_cols;
    for (std::uint16_t i = 0; i < n_cols; ++i) {
        src[i] = &top[i];
        depth[i] = 0;
        if (top[i].name != nullptr && top[i].name[0] != '\0') {
            std::snprintf(name_buf[i], kIpcNameCap, "%s", top[i].name);
        } else {
            std::snprintf(name_buf[i], kIpcNameCap, "c%u",
                          static_cast<unsigned>(i));
        }
    }

    for (std::uint16_t i = 0; i < w->n_desc; ++i) {
        const FieldSpec* s = src[i];
        const bool is_list = (s->type == BoltType::List);
        const bool is_struct = (s->type == BoltType::Struct);
        if (is_list) {
            if (s->n_children != 1 || s->children == nullptr) return false;
        } else if (is_struct) {
            if (s->n_children == 0 || s->n_children > kIpcMaxCols ||
                s->children == nullptr) {
                return false;
            }
        } else if (s->n_children != 0 || s->children != nullptr) {
            return false;   // a leaf type must carry no children
        }
        if (s->type == BoltType::Decimal128 &&
            s->decimal_scale > kDecimal128Precision) {
            return false;
        }

        IpcFieldDesc& fd = w->desc[i];
        fd.type = static_cast<std::uint16_t>(s->type);
        fd.n_children = static_cast<std::uint8_t>(s->n_children);
        fd.decimal_scale =
            (s->type == BoltType::Decimal128) ? s->decimal_scale : 0;
        fd.first_child = 0;
        if (s->n_children == 0) continue;

        if (static_cast<std::uint32_t>(depth[i]) + 1 > kIpcMaxNestDepth) {
            return false;
        }
        if (static_cast<std::uint32_t>(w->n_desc) + s->n_children >
                kIpcMaxFields) {
            return false;
        }
        fd.first_child = w->n_desc;
        for (std::uint16_t k = 0; k < s->n_children; ++k) {
            const std::uint16_t ci = w->n_desc++;
            const FieldSpec& child = s->children[k];
            src[ci] = &child;
            depth[ci] = static_cast<std::uint16_t>(depth[i] + 1);
            if (child.name != nullptr && child.name[0] != '\0') {
                std::snprintf(name_buf[ci], kIpcNameCap, "%s", child.name);
            } else if (is_list) {
                std::snprintf(name_buf[ci], kIpcNameCap, "item");
            } else {
                std::snprintf(name_buf[ci], kIpcNameCap, "f%u",
                              static_cast<unsigned>(k));
            }
        }
    }
    return true;
}

// Phase 2: build every desc[] entry's flatbuffer Field table, HIGHEST
// index first. flatten_fields()'s breadth-first append order guarantees a
// node's children always have a strictly greater index than the node
// itself, so walking indices n_desc-1 downto 0 always builds a node's
// children (and their own children, transitively) before the node that
// references them — exactly the bottom-up order this flatbuffer builder
// requires. `field_pos[i]` (out param) receives field i's finished table
// position. Returns false (fail closed) on flatbuffer overflow or an
// unsupported type.
bool build_all_fields(Fb* b, const ArrowIpcWriter* w,
                      const char name_buf[kIpcMaxFields][kIpcNameCap],
                      std::uint32_t empty_children,
                      std::uint32_t field_pos[kIpcMaxFields]) noexcept {
    assert(b != nullptr && w != nullptr);
    for (std::uint16_t ii = 0; ii < w->n_desc; ++ii) {
        const std::uint16_t i =
            static_cast<std::uint16_t>(w->n_desc - 1 - ii);
        const IpcFieldDesc& fd = w->desc[i];
        std::uint32_t children_vec = empty_children;
        if (fd.n_children > 0) {
            if (fd.n_children > kIpcMaxCols) return false;  // defensive
            std::uint32_t kids[kIpcMaxCols];
            for (std::uint16_t k = 0; k < fd.n_children; ++k) {
                kids[k] = field_pos[fd.first_child + k];
                if (kids[k] == 0) return false;
            }
            children_vec = fb_offset_vector(b, kids, fd.n_children);
        }
        std::uint8_t tag = 0;
        const std::uint32_t type_pos = build_type_table(
            b, static_cast<BoltType>(fd.type), fd.decimal_scale, &tag);
        if (type_pos == 0) return false;
        const std::uint32_t name_pos = fb_string(b, name_buf[i]);
        fb_start_table(b);
        fb_field_offset(b, 0, name_pos);                 // name
        fb_field_scalar<std::uint8_t>(b, 1, 1);          // nullable = true
        fb_field_scalar<std::uint8_t>(b, 2, tag);        // type_type
        fb_field_offset(b, 3, type_pos);                 // type
        fb_field_offset(b, 5, children_vec);              // children
        field_pos[i] = fb_end_table(b);
        if (field_pos[i] == 0) return false;
    }
    return true;
}

// Message table wrapping a header union + bodyLength; finishes the fb.
std::uint32_t build_message(Fb* b, std::uint8_t header_tag,
                            std::uint32_t header_pos,
                            std::int64_t body_len) noexcept {
    fb_start_table(b);
    fb_field_scalar<std::int16_t>(b, 0, kMetadataV5);    // version
    fb_field_scalar<std::uint8_t>(b, 1, header_tag);     // header_type
    fb_field_offset(b, 2, header_pos);                   // header
    if (body_len != 0) fb_field_scalar<std::int64_t>(b, 3, body_len);
    const std::uint32_t msg = fb_end_table(b);
    return fb_finish(b, msg);
}

// Encapsulate: <0xFFFFFFFF><u32 padded_size><flatbuffer><pad to 8>.
bool write_framed(std::FILE* f, const std::uint8_t* fb_bytes,
                  std::uint32_t fb_size) noexcept {
    assert(f != nullptr);
    assert(fb_size != 0);
    const std::uint32_t padded = (fb_size + 7u) & ~7u;
    if (std::fwrite(&kContinuation, 4, 1, f) != 1) return false;
    if (std::fwrite(&padded, 4, 1, f) != 1) return false;
    if (std::fwrite(fb_bytes, 1, fb_size, f) != fb_size) return false;
    static const std::uint8_t zeros[8] = {0};
    const std::uint32_t pad = padded - fb_size;
    if (pad != 0 && std::fwrite(zeros, 1, pad, f) != pad) return false;
    return true;
}

// ---- record-batch body measurement / writing -----------------------------

std::int64_t count_nulls(const std::uint8_t* validity,
                         std::int64_t n) noexcept {
    if (validity == nullptr) return 0;
    std::int64_t nulls = 0;
    for (std::int64_t i = 0; i < n; ++i) {
        if ((validity[i >> 3] & (std::uint8_t{1} << (i & 7))) == 0) ++nulls;
    }
    return nulls;
}

// Total packed Utf8/Binary bytes via the shared var_at resolution — the
// same physical layouts (inline/spilled StringView, or VarBinary offsets)
// back both logical types, and var_at itself does not branch on
// col.type, so one helper covers both. False when the offsets would
// overflow int32 (Arrow "u"/"z" limit — fail closed).
bool varlen_total_bytes(const BoltColumn& col, std::int64_t n,
                        std::int64_t* out_total) noexcept {
    assert(out_total != nullptr);
    std::int64_t total = 0;
    for (std::int64_t i = 0; i < n; ++i) {
        const char* p = nullptr; std::int32_t len = 0;
        if (!bolt::arrow::detail::var_at(col, i, &p, &len)) len = 0;
        total += (len > 0) ? len : 0;
        if (total > INT32_MAX) return false;
    }
    *out_total = total;
    return true;
}

bool write_padded(std::FILE* f, const void* p, std::int64_t n) noexcept {
    assert(f != nullptr);
    assert(n >= 0);
    if (n > 0 &&
        std::fwrite(p, 1, static_cast<std::size_t>(n), f)
            != static_cast<std::size_t>(n)) {
        return false;
    }
    static const std::uint8_t zeros[8] = {0};
    const std::int64_t pad = ((n + 7) & ~std::int64_t{7}) - n;
    if (pad > 0 &&
        std::fwrite(zeros, 1, static_cast<std::size_t>(pad), f)
            != static_cast<std::size_t>(pad)) {
        return false;
    }
    return true;
}

// Stream one column's varlen (Utf8 or Binary) offsets then data, both
// 8-padded. No UTF-8 validity assumption is made anywhere here — bytes
// are copied verbatim via var_at, same as the Utf8 case always was.
bool write_varlen_buffers(std::FILE* f, const BoltColumn& col,
                          std::int64_t n, std::int64_t total) noexcept {
    std::int32_t off = 0;
    if (std::fwrite(&off, 4, 1, f) != 1) return false;
    for (std::int64_t i = 0; i < n; ++i) {
        const char* p = nullptr; std::int32_t len = 0;
        if (!bolt::arrow::detail::var_at(col, i, &p, &len)) len = 0;
        off += (len > 0) ? len : 0;
        if (std::fwrite(&off, 4, 1, f) != 1) return false;
    }
    static const std::uint8_t zeros[8] = {0};
    const std::int64_t obytes = (n + 1) * 4;
    const std::int64_t opad = ((obytes + 7) & ~std::int64_t{7}) - obytes;
    if (opad > 0 &&
        std::fwrite(zeros, 1, static_cast<std::size_t>(opad), f) !=
            static_cast<std::size_t>(opad)) {
        return false;
    }
    for (std::int64_t i = 0; i < n; ++i) {
        const char* p = nullptr; std::int32_t len = 0;
        if (!bolt::arrow::detail::var_at(col, i, &p, &len) || len <= 0) {
            continue;
        }
        if (std::fwrite(p, 1, static_cast<std::size_t>(len), f) !=
                static_cast<std::size_t>(len)) {
            return false;
        }
    }
    const std::int64_t dpad = ((total + 7) & ~std::int64_t{7}) - total;
    if (dpad > 0 &&
        std::fwrite(zeros, 1, static_cast<std::size_t>(dpad), f) !=
            static_cast<std::size_t>(dpad)) {
        return false;
    }
    return true;
}

// Bit-pack bolt's byte-packed Bool column (one byte/row, nonzero=true)
// into Arrow's LSB-first bit-packed "b" buffer, streamed one output byte
// at a time so no full-column scratch buffer is needed (mirrors the
// offsets loop above). `col.data` must be non-null when n > 0 — checked
// by the caller (layout_field), same precondition every other fixed-width
// branch enforces.
bool write_bool_bits(std::FILE* f, const BoltColumn& col,
                     std::int64_t n) noexcept {
    assert(f != nullptr);
    assert(n >= 0);
    const auto* src = static_cast<const std::uint8_t*>(col.data);
    std::uint8_t byte = 0;
    for (std::int64_t i = 0; i < n; ++i) {
        if (src[i] != 0) byte |= static_cast<std::uint8_t>(1u << (i & 7));
        if ((i & 7) == 7) {
            if (std::fwrite(&byte, 1, 1, f) != 1) return false;
            byte = 0;
        }
    }
    if ((n & 7) != 0 && std::fwrite(&byte, 1, 1, f) != 1) return false;
    static const std::uint8_t zeros[8] = {0};
    const std::int64_t nbytes = (n + 7) / 8;
    const std::int64_t pad = ((nbytes + 7) & ~std::int64_t{7}) - nbytes;
    if (pad > 0 &&
        std::fwrite(zeros, 1, static_cast<std::size_t>(pad), f) !=
            static_cast<std::size_t>(pad)) {
        return false;
    }
    return true;
}

struct BatchLayout {
    std::int64_t nodes[kIpcMaxFields * 2];        // (length, null_count), preorder
    std::int64_t buffers[kIpcMaxFields * 3 * 2];  // (offset, length), preorder
    std::uint32_t n_nodes;
    std::uint32_t n_buffers;
    std::int64_t  varlen_total[kIpcMaxFields];    // Utf8/Binary leaf packed-byte
                                                   // total, indexed by the SAME
                                                   // preorder position n_nodes
                                                   // assigned that leaf.
    std::int64_t  body_len;
};

bool is_varlen_type(BoltType t) noexcept {
    return t == BoltType::Utf8 || t == BoltType::Binary;
}

// Depth-first pre-order walk over the schema tree (ArrowIpcWriter::desc,
// rooted at `desc_idx`) IN LOCKSTEP with the runtime column tree (`col`,
// `n` rows at this level): measures this field's own (length, null_count)
// into L->nodes and each of its buffers' (offset, length) into
// L->buffers, exactly the order Message.fbs requires ("Nodes/Buffers
// correspond to the pre-ordered flattened logical schema/buffer tree").
// L->n_nodes IS that pre-order position at entry — write_field_body below
// is an independent second walk over the identical tree and lands on the
// same position for the same field, which is what lets it reuse
// L->varlen_total[] without re-deriving the mapping.
//
// Fails closed (false) on any type/shape drift from the schema (a caller
// handed a batch that doesn't match what arrow_ipc_open[_nested] declared),
// a malformed offsets array, or exceeding kIpcMaxNestDepth/kIpcMaxFields.
bool layout_field(const ArrowIpcWriter* w, std::uint16_t desc_idx,
                  const BoltColumn& col, std::int64_t n,
                  BatchLayout* L, std::uint16_t depth) noexcept {
    assert(w != nullptr && L != nullptr);
    if (depth > kIpcMaxNestDepth) return false;
    if (desc_idx >= w->n_desc) return false;
    if (n < 0 || col.length < n) return false;
    const IpcFieldDesc& fd = w->desc[desc_idx];
    if (static_cast<std::uint16_t>(col.type) != fd.type) return false;
    if (L->n_nodes >= kIpcMaxFields) return false;

    const std::int64_t nulls = count_nulls(col.validity, n);
    const std::uint32_t node_idx = L->n_nodes++;
    L->nodes[node_idx * 2] = n;
    L->nodes[node_idx * 2 + 1] = nulls;

    auto add_buf = [&](std::int64_t len) noexcept -> bool {
        if (L->n_buffers >= kIpcMaxFields * 3) return false;
        L->buffers[L->n_buffers * 2]     = L->body_len;
        L->buffers[L->n_buffers * 2 + 1] = len;
        L->n_buffers++;
        L->body_len += (len + 7) & ~std::int64_t{7};
        return true;
    };

    // Validity is every field's OWN first buffer — List, Struct, and leaf
    // alike (Message.fbs: "most primitive arrays will have 2 buffers, 1
    // for the validity bitmap and 1 for the values... For struct arrays,
    // there will only be a single buffer for the validity bitmap"; a List
    // is the same plus its offsets buffer). Measured once here, common to
    // all three branches below, so it can't again go missing from exactly
    // one of them the way the leaf case's got dropped during this
    // refactor (write_field_body's matching bug, caught by the pyarrow
    // oracle on the plain Int64/Float64/Utf8 fixture: injection-verified,
    // dropping just this line reproduces the exact original "buffer_index
    // out of range" pyarrow error byte-for-byte).
    if (!add_buf(nulls > 0 ? (n + 7) / 8 : 0)) return false;

    if (col.type == BoltType::List) {
        if (fd.n_children != 1 || !col.is_nested()) return false;
        if (!add_buf((n + 1) * 4)) return false;                  // offsets
        const std::int32_t* offs = col.list_offsets();
        const BoltColumn* elem = col.list_element();
        if (offs == nullptr || elem == nullptr) return false;
        if (offs[0] != 0) return false;   // make_list()'s own contract
        const std::int64_t elem_n = offs[n];
        if (elem_n < 0) return false;
        return layout_field(w, fd.first_child, *elem, elem_n, L,
                            static_cast<std::uint16_t>(depth + 1));
    }
    if (col.type == BoltType::Struct) {
        if (!col.is_nested() || col.child_count() != fd.n_children) {
            return false;
        }
        for (std::int64_t k = 0; k < fd.n_children; ++k) {
            const BoltColumn* fc = col.child_at(k);
            if (fc == nullptr) return false;
            if (!layout_field(w, static_cast<std::uint16_t>(fd.first_child + k),
                              *fc, n, L, static_cast<std::uint16_t>(depth + 1))) {
                return false;
            }
        }
        return true;
    }
    // Leaf.
    if (is_varlen_type(col.type)) {
        if (!varlen_total_bytes(col, n, &L->varlen_total[node_idx])) return false;
        if (!add_buf((n + 1) * 4)) return false;              // int32 offsets
        if (!add_buf(L->varlen_total[node_idx])) return false; // packed bytes
    } else if (col.type == BoltType::Bool) {
        if (col.format != ColumnFormat::Flat &&
            col.format != ColumnFormat::View) {
            return false;
        }
        if (col.data == nullptr && n > 0) return false;
        L->varlen_total[node_idx] = 0;
        if (!add_buf((n + 7) / 8)) return false;              // bit-packed
    } else {
        if (col.format != ColumnFormat::Flat &&
            col.format != ColumnFormat::View) {
            return false;
        }
        if (col.data == nullptr && n > 0) return false;
        L->varlen_total[node_idx] = 0;
        const std::size_t width = bolt::type_size(col.type);
        if (width == 0) return false;                          // no fixed-width mapping
        if (!add_buf(n * static_cast<std::int64_t>(width))) return false;
    }
    return true;
}

bool layout_batch(const ArrowIpcWriter* w, const BoltBatch* batch,
                  BatchLayout* L) noexcept {
    assert(w != nullptr && batch != nullptr && L != nullptr);
    const std::int64_t n = batch->num_rows;
    for (std::uint16_t c = 0; c < w->n_cols; ++c) {
        if (!layout_field(w, c, batch->col(c), n, L, 0)) return false;
    }
    return true;
}

// Writes one field's own buffer bytes then recurses into its children, in
// the SAME pre-order layout_field walked (see that function's comment) —
// `*node_idx` threads the shared position counter across both this
// function's own recursive calls and the top-level loop in
// write_batch_body, landing on the identical L->varlen_total[]/L->nodes[]
// slot layout_field populated for the same field.
bool write_field_body(std::FILE* f, const ArrowIpcWriter* w,
                      std::uint16_t desc_idx, const BoltColumn& col,
                      std::int64_t n, const BatchLayout* L,
                      std::uint32_t* node_idx) noexcept {
    assert(f != nullptr && w != nullptr && L != nullptr && node_idx != nullptr);
    const IpcFieldDesc& fd = w->desc[desc_idx];
    const std::int64_t nulls = L->nodes[(*node_idx) * 2 + 1];
    const std::uint32_t my_node = (*node_idx)++;

    // Validity is every field's OWN first buffer (List/Struct/leaf alike —
    // see layout_field's matching unconditional `add_buf(nulls > 0 ? ... :
    // 0)` call at the top of each of the three branches there). Written
    // once here, common to all three, rather than duplicated per branch
    // (a per-branch copy is exactly how this was dropped for the leaf case
    // during the G2ARROW-21 refactor — caught by the pyarrow oracle
    // failing on the plain Int64/Float64/Utf8 fixture, which has no
    // List/Struct column at all: injection-verified, this line dropped
    // reproducibly shrinks the file by 32 bytes [2 batches x the Utf8
    // column's 16-byte padded validity buffer] and pyarrow refuses it
    // with "Expected to read N metadata bytes, but only read M").
    if (nulls > 0 && !write_padded(f, col.validity, (n + 7) / 8)) {
        return false;
    }

    if (col.type == BoltType::List) {
        const std::int32_t* offs = col.list_offsets();
        if (offs == nullptr) return false;
        if (!write_padded(f, offs, (n + 1) * 4)) return false;
        const BoltColumn* elem = col.list_element();
        if (elem == nullptr) return false;
        const std::int64_t elem_n = offs[n];
        return write_field_body(f, w, fd.first_child, *elem, elem_n, L,
                                node_idx);
    }
    if (col.type == BoltType::Struct) {
        for (std::int64_t k = 0; k < fd.n_children; ++k) {
            const BoltColumn* fc = col.child_at(k);
            if (fc == nullptr) return false;
            if (!write_field_body(f, w,
                                  static_cast<std::uint16_t>(fd.first_child + k),
                                  *fc, n, L, node_idx)) {
                return false;
            }
        }
        return true;
    }
    // Leaf.
    if (is_varlen_type(col.type)) {
        return write_varlen_buffers(f, col, n, L->varlen_total[my_node]);
    }
    if (col.type == BoltType::Bool) {
        return write_bool_bits(f, col, n);
    }
    const std::size_t width = bolt::type_size(col.type);
    return write_padded(f, col.data, n * static_cast<std::int64_t>(width));
}

bool write_batch_body(const ArrowIpcWriter* w, const BoltBatch* batch,
                      const BatchLayout* L) noexcept {
    const std::int64_t n = batch->num_rows;
    std::uint32_t node_idx = 0;
    for (std::uint16_t c = 0; c < w->n_cols; ++c) {
        if (!write_field_body(w->f, w, c, batch->col(c), n, L, &node_idx)) {
            return false;
        }
    }
    return true;
}

}  // namespace

// ---- public API ----------------------------------------------------------

bool arrow_ipc_open_nested(ArrowIpcWriter* w, std::FILE* f,
                           const FieldSpec* fields,
                           std::uint16_t n_cols) noexcept {
    assert(w != nullptr);
    if (f == nullptr || fields == nullptr) return false;
    if (n_cols == 0 || n_cols > kIpcMaxCols) return false;
    std::memset(w, 0, sizeof(*w));
    w->f = f;
    w->n_cols = n_cols;

    // Transient — used only during this call, never retained on `w`.
    // ~16 KB: bounded, and this runs once per stream open, never per-row.
    char name_buf[kIpcMaxFields][kIpcNameCap];
    if (!flatten_fields(w, fields, n_cols, name_buf)) return false;

    Fb b{};
    fb_init(&b, w->fb, kIpcFbCap);
    // One shared empty children vector (pyarrow wants children present
    // even for a leaf field).
    const std::uint32_t zero = 0;
    fb_prealign(&b, 4, 4);
    fb_push(&b, &zero, 4);
    const std::uint32_t empty_children = b.used;

    std::uint32_t field_pos[kIpcMaxFields];
    if (!build_all_fields(&b, w, name_buf, empty_children, field_pos)) {
        return false;
    }

    const std::uint32_t fields_vec = fb_offset_vector(&b, field_pos, n_cols);
    fb_start_table(&b);                        // Schema table
    fb_field_offset(&b, 1, fields_vec);        // endianness Little = default
    const std::uint32_t schema_pos = fb_end_table(&b);
    const std::uint32_t size = build_message(&b, kHeaderSchema,
                                             schema_pos, 0);
    if (size == 0) return false;
    if (!write_framed(f, w->fb + (kIpcFbCap - size), size)) return false;
    w->open = 1;
    return true;
}

bool arrow_ipc_open(ArrowIpcWriter* w, std::FILE* f,
                    const BoltType* types, const char* const* names,
                    std::uint16_t n_cols,
                    const std::uint8_t* decimal_scales) noexcept {
    assert(w != nullptr);
    if (types == nullptr) return false;
    if (n_cols == 0 || n_cols > kIpcMaxCols) return false;
    // Schema-time-only knowledge, same as the nested entry point's
    // FieldSpec::decimal_scale field, but the FLAT signature's scale
    // array is itself optional (most callers have no Decimal128 column
    // at all) — so, unlike FieldSpec (which always carries an explicit
    // scale), a missing array must fail closed here rather than silently
    // becoming scale 0 for every Decimal128 column, which would
    // misrepresent every value on read exactly as before G2ARROW-20.
    for (std::uint16_t c = 0; c < n_cols; ++c) {
        if (types[c] == BoltType::Decimal128 && decimal_scales == nullptr) {
            return false;
        }
    }
    FieldSpec specs[kIpcMaxCols];
    for (std::uint16_t c = 0; c < n_cols; ++c) {
        specs[c].type = types[c];
        specs[c].name = (names != nullptr) ? names[c] : nullptr;
        specs[c].decimal_scale =
            (decimal_scales != nullptr) ? decimal_scales[c] : 0;
        specs[c].n_children = 0;
        specs[c].children = nullptr;
    }
    return arrow_ipc_open_nested(w, f, specs, n_cols);
}

bool arrow_ipc_write_batch(ArrowIpcWriter* w,
                           const BoltBatch* batch) noexcept {
    assert(w != nullptr);
    if (w->open == 0 || w->failed != 0) return false;
    if (batch == nullptr || batch->num_rows < 0 ||
        batch->num_cols != w->n_cols) {
        w->failed = 1;
        return false;
    }

    // ~18.5 KB: bounded, well under any thread's stack budget (matches
    // the original ~4.6 KB local's own reasoning at kIpcMaxCols scale).
    BatchLayout L{};
    if (!layout_batch(w, batch, &L)) { w->failed = 1; return false; }

    Fb b{};
    fb_init(&b, w->fb, kIpcFbCap);
    const std::uint32_t bufs_vec =
        fb_struct16_vector(&b, L.buffers, L.n_buffers);
    const std::uint32_t nodes_vec =
        fb_struct16_vector(&b, L.nodes, L.n_nodes);
    fb_start_table(&b);                        // RecordBatch table
    if (batch->num_rows != 0) {
        fb_field_scalar<std::int64_t>(&b, 0, batch->num_rows);
    }
    fb_field_offset(&b, 1, nodes_vec);
    fb_field_offset(&b, 2, bufs_vec);
    const std::uint32_t rb_pos = fb_end_table(&b);
    const std::uint32_t size = build_message(&b, kHeaderBatch, rb_pos,
                                             L.body_len);
    if (size == 0) { w->failed = 1; return false; }

    if (!write_framed(w->f, w->fb + (kIpcFbCap - size), size) ||
        !write_batch_body(w, batch, &L)) {
        w->failed = 1;
        return false;
    }
    return true;
}

bool arrow_ipc_close(ArrowIpcWriter* w) noexcept {
    assert(w != nullptr);
    const bool was_ok = (w->open == 1 && w->failed == 0);
    w->open = 0;
    if (!was_ok) return false;
    const std::uint32_t zero = 0;
    if (std::fwrite(&kContinuation, 4, 1, w->f) != 1) return false;
    if (std::fwrite(&zero, 4, 1, w->f) != 1) return false;
    return std::fflush(w->f) == 0;
}

}  // namespace bolt::ingest
