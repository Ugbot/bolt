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

}  // namespace detail

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
