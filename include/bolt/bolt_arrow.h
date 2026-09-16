// bolt_arrow.h — Arrow C Data Interface export that a real Arrow
// implementation can actually consume.
//
// WHY THIS REPLACES BoltColumn::fill_arrow_{schema,array}
// ------------------------------------------------------
// The previous export was audited against pyarrow 21 and could not be
// consumed at all: its release callback was an empty body (the spec REQUIRES
// setting `release = NULL` to mark the struct released, and pyarrow aborts the
// process without it), its buffer array was `static thread_local` so every
// exported column aliased the last one, Utf8 declared `"vu"` while exporting
// two buffers and never exporting `str_overflow_base`, Bool handed byte-packed
// data to a bit-packed format, Decimal hardcoded `"d:38,10"` regardless of the
// column's real scale, and `null_count` came from a `uint32_t` stat that is
// zero unless someone called `compute_stats_numeric()`.
//
// OWNERSHIP: THIS EXPORT OWNS ITS BUFFERS.
// The old design tried to be zero-copy by pointing at arena memory with a
// no-op release. That is unfixable rather than buggy: a consumer has no way to
// extend the lifetime, and bolt's own per-morsel `Arena::reset()` recycles
// memory a live ArrowArray still points at. So every buffer here is copied
// into malloc'd storage owned by the export and freed by the release callback.
// The resulting ArrowArray is entirely self-contained — it outlives the
// column, the batch, and the arena. For a result-egress path (which is what
// this interface is for) one memcpy per column is the right trade; the hot
// path never touches this header.
//
// Bool, Utf8 and Binary are genuinely CONVERTED, not pointer-handed:
//   - Bool: bolt is byte-packed (one byte per value, for SIMD); Arrow "b" is
//     bit-packed LSB-first.
//   - Utf8/Binary: exported as Arrow "u"/"z" (validity, int32 offsets, packed
//     bytes) from EITHER bolt layout — Flat/View (StringView + optional
//     `str_overflow_base` for >12-byte values) or VarBinary (packed bytes with
//     int32 offsets on `dict_child`).
//
// Unsupported types and column formats FAIL CLOSED (return false) rather than
// exporting something a consumer will misread.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"

namespace bolt::arrow {

// Every heap block an export owns. Freed by the release callback, which then
// marks the struct released per the spec.
struct ExportState {
    static constexpr int kMaxOwned = 8;
    void*   owned[kMaxOwned];
    int     n_owned;
    const void* buffers[3];        // validity, [offsets], data
    char    format[24];            // owned: decimal scale is dynamic
    char    name[64];              // owned: caller's pointer may not outlive us
    // Batch export only.
    ArrowSchema* child_schemas;
    ArrowArray*  child_arrays;
    int64_t      n_children;
};

namespace detail {

inline void* xalloc(ExportState* st, std::size_t bytes) noexcept {
    if (bytes == 0) bytes = 1;
    if (st->n_owned >= ExportState::kMaxOwned) return nullptr;
    void* p = std::malloc(bytes);
    if (p == nullptr) return nullptr;
    st->owned[st->n_owned++] = p;
    return p;
}

inline void state_free(ExportState* st) noexcept {
    if (st == nullptr) return;
    for (int i = 0; i < st->n_owned; ++i) std::free(st->owned[i]);
    std::free(st->child_schemas);
    std::free(st->child_arrays);
    std::free(st);
}

// Spec: the release callback MUST mark the structure released by setting
// `release` to NULL, and must be safe against a double call.
inline void release_schema(ArrowSchema* s) noexcept {
    if (s == nullptr || s->release == nullptr) return;
    auto* st = static_cast<ExportState*>(s->private_data);
    if (st != nullptr) {
        for (int64_t i = 0; i < s->n_children; ++i) {
            ArrowSchema* c = s->children[i];
            if (c != nullptr && c->release != nullptr) c->release(c);
        }
        state_free(st);
    }
    s->private_data = nullptr;
    s->release = nullptr;
}

inline void release_array(ArrowArray* a) noexcept {
    if (a == nullptr || a->release == nullptr) return;
    auto* st = static_cast<ExportState*>(a->private_data);
    if (st != nullptr) {
        for (int64_t i = 0; i < a->n_children; ++i) {
            ArrowArray* c = a->children[i];
            if (c != nullptr && c->release != nullptr) c->release(c);
        }
        state_free(st);
    }
    a->private_data = nullptr;
    a->release = nullptr;
}

inline ExportState* new_state() noexcept {
    auto* st = static_cast<ExportState*>(std::calloc(1, sizeof(ExportState)));
    return st;
}

/// Copy bolt's validity bitmap (LSB-first, 1 = valid — same convention as
/// Arrow) and count the nulls. Returns the real null count so the export never
/// claims zero nulls for a column that has them.
inline const void* copy_validity(ExportState* st, const uint8_t* validity,
                                 int64_t length, int64_t* out_nulls) noexcept {
    *out_nulls = 0;
    if (validity == nullptr || length <= 0) return nullptr;
    const std::size_t nbytes = static_cast<std::size_t>((length + 7) / 8);
    void* dst = xalloc(st, nbytes);
    if (dst == nullptr) return nullptr;
    std::memcpy(dst, validity, nbytes);
    for (int64_t i = 0; i < length; ++i) {
        if ((validity[i >> 3] & (uint8_t{1} << (i & 7))) == 0) ++(*out_nulls);
    }
    return dst;
}

/// bolt Bool is one BYTE per value; Arrow "b" is one BIT, LSB-first.
inline const void* pack_bool(ExportState* st, const uint8_t* src,
                             int64_t length) noexcept {
    const std::size_t nbytes = static_cast<std::size_t>((length + 7) / 8);
    auto* dst = static_cast<uint8_t*>(xalloc(st, nbytes));
    if (dst == nullptr) return nullptr;
    std::memset(dst, 0, nbytes);
    for (int64_t i = 0; i < length; ++i) {
        if (src[i] != 0) dst[i >> 3] |= static_cast<uint8_t>(1u << (i & 7));
    }
    return dst;
}

/// Resolve row `i` of a Utf8/Binary column to (bytes, len) for EITHER layout.
inline bool var_at(const BoltColumn& col, int64_t i,
                   const char** out_p, int32_t* out_len) noexcept {
    if (col.format == ColumnFormat::VarBinary) {
        if (col.dict_child == nullptr || col.dict_child->data == nullptr) return false;
        const auto* offs = static_cast<const int32_t*>(col.dict_child->data);
        const int32_t s = offs[i], e = offs[i + 1];
        if (e < s) return false;
        *out_p = static_cast<const char*>(col.data) + s;
        *out_len = e - s;
        return true;
    }
    if (col.format != ColumnFormat::Flat && col.format != ColumnFormat::View) {
        return false;
    }
    const auto* views = static_cast<const StringView*>(col.data);
    if (views == nullptr) return false;
    const StringView& v = views[i];
    *out_len = static_cast<int32_t>(v.length);
    if (v.is_inline()) {
        // Inline: 4 prefix bytes then up to 8 more, contiguous in the view.
        *out_p = v.prefix;
        return true;
    }
    const auto* base = static_cast<const char*>(col.str_overflow_base);
    if (base == nullptr) return false;      // unresolvable spilled value
    *out_p = base + v.ref.offset;
    return true;
}

/// Build Arrow "u"/"z": int32 offsets + packed bytes, from either layout.
inline bool build_varlen(ExportState* st, const BoltColumn& col,
                         int64_t length) noexcept {
    auto* offs = static_cast<int32_t*>(
        xalloc(st, static_cast<std::size_t>(length + 1) * sizeof(int32_t)));
    if (offs == nullptr) return false;
    int64_t total = 0;
    offs[0] = 0;
    for (int64_t i = 0; i < length; ++i) {
        const char* p = nullptr; int32_t n = 0;
        if (!var_at(col, i, &p, &n)) n = 0;      // null / unresolvable => empty
        total += n;
        if (total > INT32_MAX) return false;     // "u" offsets are int32
        offs[i + 1] = static_cast<int32_t>(total);
    }
    auto* bytes = static_cast<char*>(xalloc(st, static_cast<std::size_t>(total)));
    if (bytes == nullptr) return false;
    for (int64_t i = 0; i < length; ++i) {
        const char* p = nullptr; int32_t n = 0;
        if (!var_at(col, i, &p, &n) || n == 0) continue;
        std::memcpy(bytes + offs[i], p, static_cast<std::size_t>(n));
    }
    st->buffers[1] = offs;
    st->buffers[2] = bytes;
    return true;
}

/// Fixed-width payload copy for the primitive lane.
inline bool copy_fixed(ExportState* st, const BoltColumn& col,
                       int64_t length, std::size_t width) noexcept {
    if (length == 0) { st->buffers[1] = nullptr; return true; }
    if (col.data == nullptr) return false;
    void* dst = xalloc(st, static_cast<std::size_t>(length) * width);
    if (dst == nullptr) return false;
    std::memcpy(dst, col.data, static_cast<std::size_t>(length) * width);
    st->buffers[1] = dst;
    return true;
}

/// Format string for a type, honouring the column's REAL decimal scale.
/// Returns false for types with no valid Arrow format — fail closed.
inline bool format_for(const BoltColumn& col, char* out,
                       std::size_t cap) noexcept {
    if (col.type == BoltType::Decimal128) {
        std::snprintf(out, cap, "d:38,%u", static_cast<unsigned>(col.decimal_scale));
        return true;
    }
    if (col.type == BoltType::Utf8)   { std::snprintf(out, cap, "u"); return true; }
    if (col.type == BoltType::Binary) { std::snprintf(out, cap, "z"); return true; }
    const char* f = arrow_format_string(col.type);
    if (f == nullptr || f[0] == '\0') return false;   // unsupported: fail closed
    std::snprintf(out, cap, "%s", f);
    return true;
}

/// Reverse of format_for(): parse an Arrow format string back to a BoltType
/// (+ decimal scale, when applicable). Only recognizes formats export_column
/// itself can produce -- an unrecognized format fails closed rather than
/// being guessed at, same philosophy as the export side.
///
/// NOTE: "I" is ambiguous on export (both UInt32 and IPv4 map to it via
/// arrow_format_string) -- a pre-existing export collision, not introduced
/// here. Import resolves it to UInt32; an IPv4 column round-trips as UInt32,
/// which is exactly what export already does to it (same physical 4-byte
/// layout), so nothing is lost that wasn't already lost by the export side.
inline bool type_for_format(const char* fmt, BoltType* out_type,
                            uint8_t* out_scale) noexcept {
    if (fmt == nullptr || out_type == nullptr || out_scale == nullptr) return false;
    *out_scale = 0;
    if (fmt[0] == 'd' && fmt[1] == ':') {
        unsigned prec = 0, scale = 0;
        if (std::sscanf(fmt, "d:%u,%u", &prec, &scale) != 2) return false;
        if (prec == 0 || prec > 38 || scale > 38) return false;  // Decimal128 only
        *out_type = BoltType::Decimal128;
        *out_scale = static_cast<uint8_t>(scale);
        return true;
    }
    if (std::strncmp(fmt, "tsu:", 4) == 0) { *out_type = BoltType::Timestamp; return true; }
    if (std::strcmp(fmt, "tdD") == 0) { *out_type = BoltType::Date32; return true; }
    if (std::strcmp(fmt, "tdm") == 0) { *out_type = BoltType::Date64; return true; }
    if (std::strcmp(fmt, "tDu") == 0) { *out_type = BoltType::Duration; return true; }
    if (std::strcmp(fmt, "w:16") == 0) { *out_type = BoltType::UUID; return true; }
    if (fmt[0] != '\0' && fmt[1] == '\0') {
        switch (fmt[0]) {
            case 'b': *out_type = BoltType::Bool;    return true;
            case 'c': *out_type = BoltType::Int8;    return true;
            case 's': *out_type = BoltType::Int16;   return true;
            case 'i': *out_type = BoltType::Int32;   return true;
            case 'l': *out_type = BoltType::Int64;   return true;
            case 'C': *out_type = BoltType::UInt8;   return true;
            case 'S': *out_type = BoltType::UInt16;  return true;
            case 'I': *out_type = BoltType::UInt32;  return true;
            case 'L': *out_type = BoltType::UInt64;  return true;
            case 'e': *out_type = BoltType::Float16; return true;
            case 'f': *out_type = BoltType::Float32; return true;
            case 'g': *out_type = BoltType::Float64; return true;
            case 'u': *out_type = BoltType::Utf8;    return true;
            case 'z': *out_type = BoltType::Binary;  return true;
            default: return false;
        }
    }
    return false;
}

/// Bit-unpack Arrow "b" (LSB-first bit-packed) into bolt's byte-packed Bool.
/// Reverse of pack_bool(). Returns nullptr on OOM; length==0 is handled by
/// the caller (never invoked with length<=0).
inline void* unpack_bool(Arena* arena, const uint8_t* bits,
                         int64_t length) noexcept {
    auto* dst = static_cast<uint8_t*>(
        arena->allocate(static_cast<std::size_t>(length)));
    if (dst == nullptr) return nullptr;
    for (int64_t i = 0; i < length; ++i) {
        dst[i] = static_cast<uint8_t>((bits[i >> 3] >> (i & 7)) & 1u);
    }
    return dst;
}

/// Copy the validity buffer verbatim (Arrow and bolt agree: LSB-first, bit
/// set = valid) into arena storage. Present iff the producer supplied a
/// buffer at all -- not merely when there happen to be no nulls, since a
/// spec-conformant producer may legally ship an all-1s validity buffer.
inline bool import_validity(Arena* arena, const ArrowArray& array,
                            int64_t length, uint8_t** out_validity) noexcept {
    *out_validity = nullptr;
    if (array.n_buffers < 1 || array.buffers[0] == nullptr) return true;
    const std::size_t nbytes = static_cast<std::size_t>((length + 7) / 8);
    void* dst = arena->copy_into(array.buffers[0], (nbytes > 0) ? nbytes : 1);
    if (dst == nullptr) return false;
    *out_validity = static_cast<uint8_t*>(dst);
    return true;
}

/// Import Utf8/Binary ("u"/"z": validity, int32 offsets, packed bytes) into a
/// fresh arena-owned column. Utf8 materializes as the Flat/View StringView
/// layout (`make_utf8_from_packed`) -- the layout every compute/filter/sort/
/// join/hash-agg operator in the engine actually reads -- while Binary uses
/// `make_var_binary` (no StringView equivalent exists for raw bytes). Every
/// byte is copied: the offsets and payload arena allocations are distinct
/// from the source ArrowArray's buffers, never the same pointer handed back.
inline bool import_varlen(Arena* arena, const ArrowArray& array,
                          BoltType type, int64_t length, uint8_t* validity,
                          BoltColumn* out) noexcept {
    if (array.n_buffers != 3) return false;
    if (length > 0 && array.buffers[1] == nullptr) return false;
    const auto* src_offs = static_cast<const int32_t*>(array.buffers[1]);
    int64_t total = 0;
    if (length > 0) {
        total = src_offs[length];
        if (total < 0 || src_offs[0] != 0) return false;
    }
    if (total > 0 && array.buffers[2] == nullptr) return false;

    auto* offs = static_cast<int32_t*>(arena->copy_into(
        src_offs, static_cast<std::size_t>(length + 1) * sizeof(int32_t)));
    if (offs == nullptr) return false;
    auto* bytes = static_cast<char*>(
        arena->copy_into(array.buffers[2], (total > 0) ? static_cast<std::size_t>(total) : 1));
    if (bytes == nullptr) return false;

    *out = (type == BoltType::Utf8)
               ? BoltColumn::make_utf8_from_packed(bytes, validity, offs, length, arena)
               : BoltColumn::make_var_binary(bytes, validity, offs, length,
                                             BoltType::Binary, arena);
    if (length > 0 && out->data == nullptr) return false;
    out->arena = arena;
    return true;
}

}  // namespace detail

/// Import a self-contained (ArrowSchema, ArrowArray) pair -- as produced by
/// `export_column`, or any spec-conformant producer emitting a format this
/// function recognizes -- into a fresh, arena-owned BoltColumn.
///
/// OWNERSHIP: THIS IMPORT COPIES, mirroring export_column's "owns its
/// buffers" stance from the other direction. Every buffer is copied into
/// `arena`-owned storage; the imported column never aliases `array`'s
/// buffers. The caller is therefore free to invoke the producer's
/// `array.release(&array)` / `schema.release(&schema)` immediately after this
/// call returns -- as the C Data Interface consumer contract expects them to
/// -- and the imported column remains fully valid afterward. This function
/// does not itself call release on either struct: reading the data and
/// releasing the struct are the two independent halves of the consumer's
/// side of the contract, and a caller that wants to inspect the struct again
/// (or hand it to a second consumer) must not have it released out from
/// under them as a side effect of importing.
///
/// Fails closed (returns false, leaves *out zeroed) on: null arguments, an
/// already-released struct (release == nullptr -- nothing left to read), a
/// sliced array (offset != 0 -- unsupported, matching export which never
/// produces one), a format this function does not recognize, an n_buffers
/// count inconsistent with the resolved type, or a malformed offsets array.
inline bool import_column(const ArrowSchema& schema, const ArrowArray& array,
                          Arena* arena, BoltColumn* out) noexcept {
    if (arena == nullptr || out == nullptr) return false;
    *out = BoltColumn::make_empty();
    if (schema.release == nullptr || array.release == nullptr) return false;
    if (array.length < 0 || array.offset != 0) return false;
    if (array.n_buffers > 0 && array.buffers == nullptr) return false;

    BoltType type;
    uint8_t decimal_scale = 0;
    if (!detail::type_for_format(schema.format, &type, &decimal_scale)) return false;

    const int64_t length = array.length;
    uint8_t* validity = nullptr;
    if (!detail::import_validity(arena, array, length, &validity)) return false;

    if (type == BoltType::Utf8 || type == BoltType::Binary) {
        return detail::import_varlen(arena, array, type, length, validity, out);
    }
    if (type == BoltType::Bool) {
        if (array.n_buffers != 2) return false;
        if (length > 0 && array.buffers[1] == nullptr) return false;
        void* dst = (length > 0)
            ? detail::unpack_bool(arena, static_cast<const uint8_t*>(array.buffers[1]), length)
            : arena->allocate(1);
        if (dst == nullptr) return false;
        *out = BoltColumn::make_flat(dst, validity, length, BoltType::Bool);
        out->arena = arena;
        return true;
    }

    // Fixed-width primitive lane (Int*/UInt*/Float*/Date*/Timestamp/Duration/
    // Decimal128/UUID).
    if (array.n_buffers != 2) return false;
    const std::size_t w = (type == BoltType::Decimal128)
                              ? 16u : static_cast<std::size_t>(kTypeSize[
                                    static_cast<int>(type)]);
    if (w == 0) return false;
    void* dst = nullptr;
    if (length > 0) {
        if (array.buffers[1] == nullptr) return false;
        dst = arena->copy_into(array.buffers[1], static_cast<std::size_t>(length) * w);
    } else {
        dst = arena->allocate(1);
    }
    if (dst == nullptr) return false;
    *out = BoltColumn::make_flat(dst, validity, length, type);
    out->arena = arena;
    if (type == BoltType::Decimal128) out->decimal_scale = decimal_scale;
    return true;
}

/// Export one column as a self-contained (ArrowSchema, ArrowArray) pair.
///
/// On success both structs own their memory and must be released by the
/// consumer exactly once. On failure nothing is allocated and both structs are
/// left zeroed, so a caller that ignores the return value cannot hand a
/// half-built struct to a consumer.
inline bool export_column(const BoltColumn& col, int64_t length,
                          const char* name, ArrowSchema* out_schema,
                          ArrowArray* out_array) noexcept {
    if (out_schema == nullptr || out_array == nullptr || length < 0) return false;
    std::memset(out_schema, 0, sizeof(ArrowSchema));
    std::memset(out_array, 0, sizeof(ArrowArray));

    // Only layouts whose values we can actually resolve. Dictionary /
    // Constant / Sequence / RLE etc. must be materialized by the caller —
    // exporting them as if they were Flat is a heap over-read.
    const bool varlen = (col.type == BoltType::Utf8 || col.type == BoltType::Binary);
    if (!varlen && col.format != ColumnFormat::Flat &&
        col.format != ColumnFormat::View) {
        return false;
    }

    ExportState* st = detail::new_state();
    if (st == nullptr) return false;
    if (!detail::format_for(col, st->format, sizeof(st->format))) {
        detail::state_free(st); return false;
    }
    std::snprintf(st->name, sizeof(st->name), "%s", (name != nullptr) ? name : "");

    int64_t nulls = 0;
    st->buffers[0] = detail::copy_validity(st, col.validity, length, &nulls);

    int64_t n_buffers = 2;
    bool ok = true;
    if (varlen) {
        n_buffers = 3;
        ok = detail::build_varlen(st, col, length);
    } else if (col.type == BoltType::Bool) {
        const auto* src = static_cast<const uint8_t*>(col.data);
        st->buffers[1] = (length > 0 && src != nullptr)
                             ? detail::pack_bool(st, src, length) : nullptr;
        ok = (length == 0 || st->buffers[1] != nullptr);
    } else {
        const std::size_t w = (col.type == BoltType::Decimal128)
                                  ? 16u : static_cast<std::size_t>(kTypeSize[
                                        static_cast<int>(col.type)]);
        ok = (w > 0) && detail::copy_fixed(st, col, length, w);
    }
    if (!ok) {
        detail::state_free(st);
        std::memset(out_schema, 0, sizeof(ArrowSchema));
        std::memset(out_array, 0, sizeof(ArrowArray));
        return false;
    }

    // The schema and the array are released INDEPENDENTLY by the consumer, in
    // either order, so they must not share owned state — otherwise whichever
    // is released first frees memory the other still points at. The schema
    // gets its own small state holding just the strings it exposes.
    ExportState* sst = detail::new_state();
    if (sst == nullptr) {
        detail::state_free(st);
        std::memset(out_schema, 0, sizeof(ArrowSchema));
        std::memset(out_array, 0, sizeof(ArrowArray));
        return false;
    }
    std::memcpy(sst->format, st->format, sizeof(sst->format));
    std::memcpy(sst->name, st->name, sizeof(sst->name));

    out_schema->format       = sst->format;
    out_schema->name         = sst->name;
    out_schema->flags        = (col.validity != nullptr) ? 2 : 0;  // NULLABLE
    out_schema->release      = &detail::release_schema;
    out_schema->private_data = sst;

    out_array->length       = length;
    out_array->null_count   = nulls;
    out_array->n_buffers    = n_buffers;
    out_array->buffers      = st->buffers;
    out_array->release      = &detail::release_array;
    out_array->private_data = st;
    return true;
}

}  // namespace bolt::arrow
