// bolt_arrow_ipc.cpp — Arrow IPC stream writer (G2ARROW-10).
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
constexpr std::uint8_t kTypeInt        = 2;   // Type union
constexpr std::uint8_t kTypeFloat      = 3;
constexpr std::uint8_t kTypeUtf8       = 5;
constexpr std::int16_t kPrecisionDouble = 2;  // FloatingPoint::Precision

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
// *out_tag. Returns table pos, 0 for an unsupported type (fail closed).
std::uint32_t build_type_table(Fb* b, BoltType t,
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
    return 0;                                            // unsupported
}

// Field table: name / nullable / type union / empty children vector.
std::uint32_t build_field(Fb* b, const char* name, BoltType t,
                          std::uint32_t empty_children) noexcept {
    std::uint8_t tag = 0;
    const std::uint32_t type_pos = build_type_table(b, t, &tag);
    if (type_pos == 0) return 0;
    const std::uint32_t name_pos = fb_string(b, name);
    fb_start_table(b);
    fb_field_offset(b, 0, name_pos);                     // name
    fb_field_scalar<std::uint8_t>(b, 1, 1);              // nullable = true
    fb_field_scalar<std::uint8_t>(b, 2, tag);            // type_type
    fb_field_offset(b, 3, type_pos);                     // type
    fb_field_offset(b, 5, empty_children);               // children = []
    return fb_end_table(b);
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

// Total packed Utf8 bytes via the shared var_at resolution; false when
// the offsets would overflow int32 (Arrow "u" limit — fail closed).
bool utf8_total_bytes(const BoltColumn& col, std::int64_t n,
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

// Stream one column's Utf8 offsets then data, both 8-padded.
bool write_utf8_buffers(std::FILE* f, const BoltColumn& col,
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

struct BatchLayout {
    std::int64_t nodes[kIpcMaxCols * 2];       // (length, null_count) pairs
    std::int64_t buffers[kIpcMaxCols * 3 * 2]; // (offset, length) pairs
    std::uint32_t n_buffers;
    std::int64_t  utf8_total[kIpcMaxCols];
    std::int64_t  body_len;
};

// Measure the body: per-column validity/[offsets]/data buffer entries in
// spec order, each 8-padded. Fails closed on unsupported shape.
bool layout_batch(const ArrowIpcWriter* w, const BoltBatch* batch,
                  BatchLayout* L) noexcept {
    assert(w != nullptr && batch != nullptr && L != nullptr);
    const std::int64_t n = batch->num_rows;
    std::int64_t off = 0;
    L->n_buffers = 0;
    auto add_buf = [&](std::int64_t len) noexcept {
        L->buffers[L->n_buffers * 2]     = off;
        L->buffers[L->n_buffers * 2 + 1] = len;
        L->n_buffers++;
        off += (len + 7) & ~std::int64_t{7};
    };
    for (std::uint16_t c = 0; c < w->n_cols; ++c) {
        const BoltColumn& col = batch->col(c);
        if (static_cast<std::uint16_t>(col.type) != w->col_types[c]) {
            return false;                      // schema drift: fail closed
        }
        if (col.length < n) return false;
        const std::int64_t nulls = count_nulls(col.validity, n);
        L->nodes[c * 2] = n;
        L->nodes[c * 2 + 1] = nulls;
        add_buf(nulls > 0 ? (n + 7) / 8 : 0);  // validity
        if (col.type == BoltType::Utf8) {
            if (!utf8_total_bytes(col, n, &L->utf8_total[c])) return false;
            add_buf((n + 1) * 4);              // int32 offsets
            add_buf(L->utf8_total[c]);         // packed bytes
        } else {
            if (col.format != ColumnFormat::Flat &&
                col.format != ColumnFormat::View) {
                return false;
            }
            if (col.data == nullptr && n > 0) return false;
            L->utf8_total[c] = 0;
            add_buf(n * 8);                    // Int64 / Float64 payload
        }
    }
    L->body_len = off;
    return true;
}

bool write_batch_body(const ArrowIpcWriter* w, const BoltBatch* batch,
                      const BatchLayout* L) noexcept {
    const std::int64_t n = batch->num_rows;
    for (std::uint16_t c = 0; c < w->n_cols; ++c) {
        const BoltColumn& col = batch->col(c);
        const std::int64_t nulls = L->nodes[c * 2 + 1];
        if (nulls > 0 &&
            !write_padded(w->f, col.validity, (n + 7) / 8)) {
            return false;
        }
        if (col.type == BoltType::Utf8) {
            if (!write_utf8_buffers(w->f, col, n, L->utf8_total[c])) {
                return false;
            }
        } else {
            if (!write_padded(w->f, col.data, n * 8)) return false;
        }
    }
    return true;
}

}  // namespace

// ---- public API ----------------------------------------------------------

bool arrow_ipc_open(ArrowIpcWriter* w, std::FILE* f,
                    const BoltType* types, const char* const* names,
                    std::uint16_t n_cols) noexcept {
    assert(w != nullptr);
    if (f == nullptr || types == nullptr) return false;
    if (n_cols == 0 || n_cols > kIpcMaxCols) return false;
    std::memset(w, 0, sizeof(*w));
    w->f = f;
    w->n_cols = n_cols;
    for (std::uint16_t c = 0; c < n_cols; ++c) {
        // Honest scope: reject unsupported types AT OPEN.
        if (types[c] != BoltType::Int64 && types[c] != BoltType::Float64 &&
            types[c] != BoltType::Utf8) {
            return false;
        }
        w->col_types[c] = static_cast<std::uint16_t>(types[c]);
        const char* nm = (names != nullptr) ? names[c] : nullptr;
        if (nm != nullptr && nm[0] != '\0') {
            std::snprintf(w->names[c], kIpcNameCap, "%s", nm);
        } else {
            std::snprintf(w->names[c], kIpcNameCap, "c%u",
                          static_cast<unsigned>(c));
        }
    }

    Fb b{};
    fb_init(&b, w->fb, kIpcFbCap);
    // One shared empty children vector (pyarrow wants children present).
    const std::uint32_t zero = 0;
    fb_prealign(&b, 4, 4);
    fb_push(&b, &zero, 4);
    const std::uint32_t empty_children = b.used;

    std::uint32_t field_pos[kIpcMaxCols];
    for (std::uint16_t c = 0; c < n_cols; ++c) {
        field_pos[c] = build_field(&b, w->names[c],
                                   static_cast<BoltType>(w->col_types[c]),
                                   empty_children);
        if (field_pos[c] == 0) return false;
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

bool arrow_ipc_write_batch(ArrowIpcWriter* w,
                           const BoltBatch* batch) noexcept {
    assert(w != nullptr);
    if (w->open == 0 || w->failed != 0) return false;
    if (batch == nullptr || batch->num_rows < 0 ||
        batch->num_cols != w->n_cols) {
        w->failed = 1;
        return false;
    }

    // ~4.6 KB local: bounded, well under any thread's stack budget.
    BatchLayout L{};
    if (!layout_batch(w, batch, &L)) { w->failed = 1; return false; }

    Fb b{};
    fb_init(&b, w->fb, kIpcFbCap);
    const std::uint32_t bufs_vec =
        fb_struct16_vector(&b, L.buffers, L.n_buffers);
    const std::uint32_t nodes_vec =
        fb_struct16_vector(&b, L.nodes, w->n_cols);
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
