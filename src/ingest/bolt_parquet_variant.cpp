// bolt_parquet_variant.cpp — see bolt/ingest/bolt_parquet_variant.h (G2ICE-310).

#include "bolt/ingest/bolt_parquet_variant.h"

#include <cassert>
#include <cstring>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"

namespace bolt {
namespace ingest {
namespace parquet {

namespace {

constexpr uint32_t kMaxObjectFields = 1u << 16;   // keys of one rebuilt object
constexpr int32_t kConvUtf8 = 0, kConvDecimal = 5, kConvDate = 6;   // parquet ConvertedType
constexpr uint32_t kMaxDepth = 2u * kPqMaxNestDepth + 2u;

// Variant primitive type ids (VariantEncoding.md, basic type 0).
enum : uint8_t {
    kVNull = 0, kVTrue = 1, kVFalse = 2, kVInt8 = 3, kVInt16 = 4, kVInt32 = 5, kVInt64 = 6, kVDouble = 7, kVDec4 = 8,
    kVDec8 = 9, kVDec16 = 10, kVDate = 11, kVTsTz = 12, kVTsNtz = 13, kVFloat = 14, kVBinary = 15, kVString = 16,
    kVTimeNtz = 17, kVTsTzNs = 18, kVTsNtzNs = 19, kVUuid = 20,
};

struct Buf {
    uint8_t* p = nullptr;
    uint64_t n = 0, cap = 0;
    Arena*   a = nullptr;
};

bool reserve(Buf* b, uint64_t extra) noexcept {
    assert(b != nullptr && b->a != nullptr);
    if (b->n + extra <= b->cap) return true;
    if (b->n + extra > kPqVariantValueBytes) return false;
    uint64_t cap = b->cap < 4096u ? 4096u : b->cap * 2u;
    while (cap < b->n + extra) cap *= 2u;                              // bounded: kPqVariantValueBytes
    if (cap > kPqVariantValueBytes) cap = kPqVariantValueBytes;
    auto* p = static_cast<uint8_t*>(b->a->allocate(static_cast<size_t>(cap), 8));
    if (p == nullptr) return false;
    if (b->n > 0u) std::memcpy(p, b->p, static_cast<size_t>(b->n));
    b->p = p;
    b->cap = cap;
    return true;
}

bool put(Buf* b, const void* v, uint64_t n) noexcept {
    if (!reserve(b, n)) return false;
    if (n > 0u) std::memcpy(b->p + b->n, v, static_cast<size_t>(n));
    b->n += n;
    return true;
}
bool put_u8(Buf* b, uint8_t v) noexcept { return put(b, &v, 1u); }
bool put_le(Buf* b, uint64_t v, uint32_t width) noexcept {
    uint8_t t[8];
    for (uint32_t i = 0; i < width; ++i) t[i] = static_cast<uint8_t>(v >> (8u * i));   // bounded: 8
    return put(b, t, width);
}
uint64_t get_le(const uint8_t* p, uint32_t width) noexcept {
    uint64_t v = 0;
    for (uint32_t i = 0; i < width; ++i) v |= static_cast<uint64_t>(p[i]) << (8u * i);   // bounded: 8
    return v;
}
uint32_t width_for(uint64_t v) noexcept { return v <= 0xFFu ? 1u : v <= 0xFFFFu ? 2u : v <= 0xFFFFFFu ? 3u : 4u; }

// Insert `n` bytes at `at`, moving [at, b->n) up.
bool insert(Buf* b, uint64_t at, const uint8_t* v, uint64_t n) noexcept {
    assert(at <= b->n);
    if (!reserve(b, n)) return false;
    std::memmove(b->p + at + n, b->p + at, static_cast<size_t>(b->n - at));
    std::memcpy(b->p + at, v, static_cast<size_t>(n));
    b->n += n;
    return true;
}

// ---- the metadata dictionary (VariantEncoding.md "Metadata encoding") ------------------------------------------------
struct Dict {
    const uint8_t* p = nullptr;
    uint32_t len = 0, size = 0, osz = 0;
    uint64_t base = 0;          // start of the string bytes
};

bool dict_open(const uint8_t* p, uint32_t len, Dict* d) noexcept {
    if (p == nullptr || len < 1u || (p[0] & 0x0Fu) != 1u) return false;
    d->p = p;
    d->len = len;
    d->osz = ((p[0] >> 6) & 0x3u) + 1u;
    if (1u + d->osz > len) return false;
    d->size = static_cast<uint32_t>(get_le(p + 1, d->osz));
    d->base = 1u + static_cast<uint64_t>(d->osz) * (d->size + 2u);
    return d->base <= len;
}

bool dict_key(const Dict& d, uint32_t id, const uint8_t** k, uint32_t* kl) noexcept {
    if (id >= d.size) return false;
    const uint64_t o0 = get_le(d.p + 1 + d.osz * (1u + id), d.osz), o1 = get_le(d.p + 1 + d.osz * (2u + id), d.osz);
    if (o1 < o0 || d.base + o1 > d.len) return false;
    *k = d.p + d.base + o0;
    *kl = static_cast<uint32_t>(o1 - o0);
    return true;
}

bool dict_find(const Dict& d, const char* name, uint32_t nl, uint32_t* id) noexcept {
    for (uint32_t i = 0; i < d.size; ++i) {                            // bounded by the dictionary
        const uint8_t* k = nullptr;
        uint32_t kl = 0;
        if (dict_key(d, i, &k, &kl) && kl == nl && std::memcmp(k, name, nl) == 0) { *id = i; return true; }
    }
    return false;
}

// ---- the shredded tree -------------------------------------------------------------------------------------------------
struct Ctx {
    const PqSchemaTree* t = nullptr;
    const PqMeta*       meta = nullptr;   // the original footer (time units)
    Buf*                out = nullptr;
    Dict                dict;
};

// Child of struct field `f` named `name` and its position (the column's child index), or -1.
int32_t child_named(const PqSchemaTree* t, int32_t f, const char* name, int64_t* pos) noexcept {
    int32_t c = t->fields[f].first_child;
    for (int64_t k = 0; c >= 0 && k < static_cast<int64_t>(t->fields[f].n_children); ++k) {   // bounded by the children
        if (std::strcmp(t->fields[c].name, name) == 0) { *pos = k; return c; }
        c = t->fields[c].next_sibling;
    }
    return -1;
}

bool col_null(const BoltColumn* c, int64_t i) noexcept { return c == nullptr || c->is_null(i); }

bool bin_at(const BoltColumn& c, int64_t i, const uint8_t** p, uint32_t* n) noexcept {
    int32_t l = 0;
    c.utf8_at_or_empty(i, p, &l);
    *n = l > 0 ? static_cast<uint32_t>(l) : 0u;
    return l >= 0;
}

int64_t lane_i64(const BoltColumn& c, int64_t i) noexcept {
    const auto* d = static_cast<const uint8_t*>(c.data) + i * c.type_size_bytes;
    switch (c.type_size_bytes) {
        case 1: { int8_t v; std::memcpy(&v, d, 1); return v; }
        case 2: { int16_t v; std::memcpy(&v, d, 2); return v; }
        case 4: { int32_t v; std::memcpy(&v, d, 4); return v; }
        default: { int64_t v; std::memcpy(&v, d, 8); return v; }
    }
}

bool put_decimal(Buf* b, const PqColumn& pc, const BoltColumn& c, int64_t i) noexcept {
    const uint8_t tag = pc.precision <= 9 ? kVDec4 : pc.precision <= 18 ? kVDec8 : kVDec16;
    if (!put_u8(b, static_cast<uint8_t>(tag << 2)) || !put_u8(b, static_cast<uint8_t>(pc.scale))) return false;
    if (c.type_size_bytes == 16u) {                                        // Decimal128 {lo, hi}
        const auto* d = static_cast<const uint8_t*>(c.data) + i * 16;
        return tag == kVDec16 && put(b, d, 16u);
    }
    const int64_t v = lane_i64(c, i);
    if (tag == kVDec4) return put_le(b, static_cast<uint64_t>(v), 4u);
    if (tag == kVDec8) return put_le(b, static_cast<uint64_t>(v), 8u);
    return put_le(b, static_cast<uint64_t>(v), 8u) && put_le(b, v < 0 ? ~uint64_t{0} : 0u, 8u);
}

bool put_bytes_value(Buf* b, const BoltColumn& c, int64_t i, bool is_string) noexcept {
    const uint8_t* p = nullptr;
    uint32_t n = 0;
    if (!bin_at(c, i, &p, &n)) return false;
    if (is_string && n < 64u) return put_u8(b, static_cast<uint8_t>((n << 2) | 1u)) && put(b, p, n);   // short string
    return put_u8(b, static_cast<uint8_t>((is_string ? kVString : kVBinary) << 2)) && put_le(b, n, 4u) && put(b, p, n);
}

bool put_primitive(const Ctx& x, int32_t f, const BoltColumn& c, int64_t i) noexcept {
    const PqColumn& pc = x.meta->columns[x.t->fields[f].column];
    const auto lg = static_cast<PqLogical>(pc.logical);
    Buf* b = x.out;
    const auto tag = [&](uint8_t id) { return put_u8(b, static_cast<uint8_t>(id << 2)); };
    if (lg == PqLogical::Decimal || pc.converted == kConvDecimal) return put_decimal(b, pc, c, i);
    switch (pc.physical) {
        case PqType::Boolean: return tag(lane_i64(c, i) != 0 ? kVTrue : kVFalse);
        case PqType::Int32:
            if (lg == PqLogical::Date || pc.converted == kConvDate) return tag(kVDate) && put_le(b, static_cast<uint64_t>(lane_i64(c, i)), 4u);
            if (lg == PqLogical::Int && pc.int_bits == 8u) return tag(kVInt8) && put_le(b, static_cast<uint64_t>(lane_i64(c, i)), 1u);
            if (lg == PqLogical::Int && pc.int_bits == 16u) return tag(kVInt16) && put_le(b, static_cast<uint64_t>(lane_i64(c, i)), 2u);
            if (lg == PqLogical::Int && pc.int_signed == 0u) return false;       // no unsigned variant primitive
            return tag(kVInt32) && put_le(b, static_cast<uint64_t>(lane_i64(c, i)), 4u);
        case PqType::Int64: {
            const uint64_t v = static_cast<uint64_t>(lane_i64(c, i));
            if (lg == PqLogical::Timestamp && pc.time_unit == 2) return tag(pc.ts_utc != 0u ? kVTsTz : kVTsNtz) && put_le(b, v, 8u);
            if (lg == PqLogical::Timestamp && pc.time_unit == 3) return tag(pc.ts_utc != 0u ? kVTsTzNs : kVTsNtzNs) && put_le(b, v, 8u);
            if (lg == PqLogical::Time && pc.time_unit == 2 && pc.ts_utc == 0u) return tag(kVTimeNtz) && put_le(b, v, 8u);
            if (lg == PqLogical::Timestamp || lg == PqLogical::Time || (lg == PqLogical::Int && pc.int_signed == 0u)) return false;
            return tag(kVInt64) && put_le(b, v, 8u);
        }
        case PqType::Float: {
            float v = static_cast<float>(static_cast<const double*>(c.data)[i]);   // the reader widens FLOAT to a double lane
            uint32_t bits = 0;
            std::memcpy(&bits, &v, 4);
            return tag(kVFloat) && put_le(b, bits, 4u);
        }
        case PqType::Double: {
            uint64_t bits = 0;
            std::memcpy(&bits, static_cast<const double*>(c.data) + i, 8);
            return tag(kVDouble) && put_le(b, bits, 8u);
        }
        case PqType::ByteArray: return put_bytes_value(b, c, i, lg == PqLogical::String || pc.converted == kConvUtf8);
        case PqType::FixedLenByteArray:
            if (lg == PqLogical::Uuid && c.type_size_bytes == 16u) return tag(kVUuid) && put(b, static_cast<const uint8_t*>(c.data) + i * 16, 16u);
            return false;
        default: return false;
    }
}

bool put_value(const Ctx& x, int32_t g, const BoltColumn& gc, int64_t i, uint32_t depth, bool* present) noexcept;

// An element / field group {value, typed_value} at position i; a missing value (both null) is a variant null.
bool put_present(const Ctx& x, int32_t g, const BoltColumn& gc, int64_t i, uint32_t depth) noexcept {
    bool present = false;
    if (!put_value(x, g, gc, i, depth, &present)) return false;
    return present || put_u8(x.out, kVNull);
}

bool put_array(const Ctx& x, int32_t lf, const BoltColumn& lc, int64_t i, uint32_t depth) noexcept {
    const int32_t* offs = lc.list_offsets();
    const BoltColumn* ec = lc.list_element();
    const int32_t ef = x.t->fields[lf].first_child;
    if (offs == nullptr || ec == nullptr || ef < 0 || x.t->fields[ef].kind != PqFieldKind::Struct) return false;
    const int64_t n = offs[i + 1] - offs[i];
    Buf* b = x.out;
    const uint64_t start = b->n;
    auto* at = static_cast<uint64_t*>(b->a->allocate(static_cast<size_t>(n + 1) * sizeof(uint64_t), 8));
    if (at == nullptr) return false;
    for (int64_t j = 0; j < n; ++j) {                                      // bounded by the list
        at[j] = b->n - start;
        if (!put_present(x, ef, *ec, offs[i] + j, depth + 1u)) return false;
    }
    at[n] = b->n - start;
    const bool large = n > 255;
    const uint32_t osz = width_for(at[n]);
    uint8_t hdr[16];
    uint32_t h = 0;
    hdr[h++] = static_cast<uint8_t>(((osz - 1u) | (large ? 4u : 0u)) << 2 | 3u);
    for (uint32_t k = 0; k < (large ? 4u : 1u); ++k) hdr[h++] = static_cast<uint8_t>(static_cast<uint64_t>(n) >> (8u * k));
    Buf head{nullptr, 0, 0, b->a};
    if (!put(&head, hdr, h)) return false;
    for (int64_t j = 0; j <= n; ++j) if (!put_le(&head, at[j], osz)) return false;   // bounded by the list
    return insert(b, start, head.p, head.n);
}

struct Entry {
    const uint8_t* key;
    uint32_t       klen;
    uint32_t       id;
    uint64_t       off, len;   // into the object's value region
};

bool entry_less(const Entry& a, const Entry& b) noexcept {
    const uint32_t n = a.klen < b.klen ? a.klen : b.klen;
    const int c = n > 0u ? std::memcmp(a.key, b.key, n) : 0;
    return c != 0 ? c < 0 : a.klen < b.klen;
}

void sort_entries(Entry* e, uint32_t n) noexcept {
    for (uint32_t i = 1; i < n; ++i) {                                     // insertion sort: bounded by the fields
        const Entry k = e[i];
        uint32_t j = i;
        while (j > 0u && entry_less(k, e[j - 1u])) { e[j] = e[j - 1u]; --j; }
        e[j] = k;
    }
}

// The residual object `v` (shredded keys moved out): its fields appended to the value region, recorded in `e`.
bool add_residual(const Ctx& x, const uint8_t* v, uint32_t vl, uint64_t start, Entry* e, uint32_t* n) noexcept {
    if (vl < 1u || (v[0] & 3u) != 2u) return false;                        // must be an object
    const uint32_t vh = v[0] >> 2;
    const uint32_t osz = (vh & 3u) + 1u, isz = ((vh >> 2) & 3u) + 1u, nsz = (vh & 16u) != 0u ? 4u : 1u;
    if (1u + nsz > vl) return false;
    const uint32_t cnt = static_cast<uint32_t>(get_le(v + 1, nsz));
    const uint64_t ids = 1u + nsz, offs = ids + static_cast<uint64_t>(cnt) * isz, data = offs + static_cast<uint64_t>(cnt + 1u) * osz;
    if (data > vl || cnt > kMaxObjectFields) return false;
    const uint64_t total = get_le(v + offs + static_cast<uint64_t>(cnt) * osz, osz);
    if (data + total > vl) return false;
    for (uint32_t k = 0; k < cnt; ++k) {                                   // bounded by the object
        const uint32_t id = static_cast<uint32_t>(get_le(v + ids + static_cast<uint64_t>(k) * isz, isz));
        const uint64_t o = get_le(v + offs + static_cast<uint64_t>(k) * osz, osz);
        uint64_t end = total;
        for (uint32_t q = 0; q < cnt; ++q) {                               // the next value start after o
            const uint64_t oq = get_le(v + offs + static_cast<uint64_t>(q) * osz, osz);
            if (oq > o && oq < end) end = oq;
        }
        Entry& en = e[(*n)++];
        if (o > end || !dict_key(x.dict, id, &en.key, &en.klen)) return false;
        en.id = id;
        en.off = x.out->n - start;
        en.len = end - o;
        if (!put(x.out, v + data + o, end - o)) return false;
    }
    return true;
}

bool object_header(const Ctx& x, Entry* e, uint32_t n, uint64_t start) noexcept {
    sort_entries(e, n);
    uint32_t max_id = 0;
    for (uint32_t k = 0; k < n; ++k) {                                     // bounded by the fields
        if (k > 0u && e[k].klen == e[k - 1u].klen && std::memcmp(e[k].key, e[k - 1u].key, e[k].klen) == 0) return false;
        if (e[k].id > max_id) max_id = e[k].id;
    }
    const uint64_t total = x.out->n - start;
    const uint32_t isz = width_for(max_id), osz = width_for(total);
    const bool large = n > 255u;
    Buf head{nullptr, 0, 0, x.out->a};
    if (!put_u8(&head, static_cast<uint8_t>((((osz - 1u) | ((isz - 1u) << 2) | (large ? 16u : 0u)) << 2) | 2u)) ||
        !put_le(&head, n, large ? 4u : 1u)) return false;
    for (uint32_t k = 0; k < n; ++k) if (!put_le(&head, e[k].id, isz)) return false;      // bounded by the fields
    for (uint32_t k = 0; k < n; ++k) if (!put_le(&head, e[k].off, osz)) return false;
    return put_le(&head, total, osz) && insert(x.out, start, head.p, head.n);
}

// Fields of the residual object `v` (0 when it is not an object: add_residual refuses it).
uint32_t residual_count(const uint8_t* v, uint32_t vl) noexcept {
    if (vl < 2u || (v[0] & 3u) != 2u) return 0u;
    const uint32_t nsz = ((v[0] >> 2) & 16u) != 0u ? 4u : 1u;
    return 1u + nsz <= vl ? static_cast<uint32_t>(get_le(v + 1, nsz)) : 0u;
}

bool put_object(const Ctx& x, int32_t of, const BoltColumn& oc, int64_t i, const BoltColumn* vc, uint32_t depth) noexcept {
    const PqField& f = x.t->fields[of];
    const uint8_t* rv = nullptr;
    uint32_t rl = 0;
    const bool residual = !col_null(vc, i);
    if (residual && !bin_at(*vc, i, &rv, &rl)) return false;
    const uint32_t extra = residual ? residual_count(rv, rl) : 0u;
    if (extra > kMaxObjectFields) return false;
    auto* e = static_cast<Entry*>(x.out->a->allocate(sizeof(Entry) * (f.n_children + extra + 1u), 8));
    if (e == nullptr) return false;
    const uint64_t start = x.out->n;
    uint32_t n = 0;
    int32_t c = f.first_child;
    for (int64_t k = 0; c >= 0 && k < static_cast<int64_t>(f.n_children); ++k) {   // bounded by the shredded keys
        const BoltColumn* fc = oc.child_at(k);
        const uint64_t at = x.out->n - start;
        bool present = false;
        if (fc == nullptr || x.t->fields[c].kind != PqFieldKind::Struct || !put_value(x, c, *fc, i, depth + 1u, &present)) return false;
        if (present) {
            Entry& en = e[n++];
            en.key = reinterpret_cast<const uint8_t*>(x.t->fields[c].name);
            en.klen = static_cast<uint32_t>(std::strlen(x.t->fields[c].name));
            if (!dict_find(x.dict, x.t->fields[c].name, en.klen, &en.id)) return false;   // a shredded key the dictionary lacks
            en.off = at;
            en.len = x.out->n - start - at;
        }
        c = x.t->fields[c].next_sibling;
    }
    if (residual && !add_residual(x, rv, rl, start, e, &n)) return false;
    return object_header(x, e, n, start);
}

bool put_value(const Ctx& x, int32_t g, const BoltColumn& gc, int64_t i, uint32_t depth, bool* present) noexcept {
    *present = false;
    if (depth > kMaxDepth) return false;
    int64_t vp = -1, tp = -1;
    const int32_t vf = child_named(x.t, g, "value", &vp);
    const int32_t tf = child_named(x.t, g, "typed_value", &tp);
    const BoltColumn* vc = vf >= 0 ? gc.child_at(vp) : nullptr;
    const BoltColumn* tc = tf >= 0 ? gc.child_at(tp) : nullptr;
    if (gc.is_null(i)) return true;                                        // a null field group: missing
    if (!col_null(tc, i)) {
        *present = true;
        const PqFieldKind k = x.t->fields[tf].kind;
        if (k == PqFieldKind::Struct) return put_object(x, tf, *tc, i, vc, depth);
        if (!col_null(vc, i)) return false;                                // value beside a non-object typed_value
        if (k == PqFieldKind::List) return put_array(x, tf, *tc, i, depth);
        return k == PqFieldKind::Leaf && put_primitive(x, tf, *tc, i);
    }
    if (col_null(vc, i)) return true;
    const uint8_t* p = nullptr;
    uint32_t n = 0;
    *present = true;
    return bin_at(*vc, i, &p, &n) && n > 0u && put(x.out, p, n);
}

bool is_variant_group(const PqSchemaTree* t, int32_t f) noexcept;

// Every NANOS leaf under a VARIANT group of field `f` left raw (the reader rescales TIMESTAMP(NANOS) to micros otherwise); millis
// under a variant group has no variant primitive and fails.
bool raw_variant_leaves(const PqSchemaTree* t, int32_t f, PqColumn* cols, uint32_t depth) noexcept {
    if (depth > kMaxDepth) return false;
    const PqField& fd = t->fields[f];
    if (is_variant_group(t, f)) {
        for (uint32_t c = fd.col_begin; c < fd.col_end; ++c) {            // bounded: the group's leaves
            if (cols[c].time_unit == 3) cols[c].time_unit = 0;
            if (cols[c].time_unit == 1) return false;
        }
        return true;
    }
    int32_t c = fd.first_child;
    for (uint32_t k = 0; c >= 0 && k < fd.n_children; ++k) {              // bounded by the children
        if (!raw_variant_leaves(t, c, cols, depth + 1u)) return false;
        c = t->fields[c].next_sibling;
    }
    return true;
}

bool raw_time_meta(const PqMeta* meta, const PqSchemaTree* t, int32_t f, Arena* a, PqMeta* out) noexcept {
    *out = *meta;
    auto* cols = static_cast<PqColumn*>(a->allocate(sizeof(PqColumn) * (meta->n_columns > 0u ? meta->n_columns : 1u), 8));
    if (cols == nullptr) return false;
    std::memcpy(cols, meta->columns, sizeof(PqColumn) * meta->n_columns);
    out->columns = cols;
    return raw_variant_leaves(t, f, cols, 0u);
}

// The VARIANT group `gf` assembled as `gc` ({metadata, value?, typed_value?}) as make_variant({metadata, value}).
bool unshred_group(const PqSchemaTree* tree, const PqMeta* meta, int32_t gf, const BoltColumn& gc, Arena* arena, BoltColumn* out) noexcept {
    const int64_t rows = gc.length;
    const BoltColumn* md = gc.child_at(0);
    if (md == nullptr || md->length != rows) return false;
    Buf b{nullptr, 0, 0, arena};
    auto* offs = static_cast<int32_t*>(arena->allocate(sizeof(int32_t) * static_cast<size_t>(rows + 1), 8));
    if (offs == nullptr) return false;
    Ctx x;
    x.t = tree;
    x.meta = meta;
    x.out = &b;
    for (int64_t i = 0; i < rows; ++i) {                                   // bounded by the column
        offs[i] = static_cast<int32_t>(b.n);
        if (gc.is_null(i)) continue;
        const uint8_t* mp = nullptr;
        uint32_t ml = 0;
        if (!bin_at(*md, i, &mp, &ml) || !dict_open(mp, ml, &x.dict)) return false;
        if (!put_present(x, gf, gc, i, 0u) || b.n > static_cast<uint64_t>(INT32_MAX)) return false;
    }
    offs[rows] = static_cast<int32_t>(b.n);
    uint8_t* data = b.p != nullptr ? b.p : static_cast<uint8_t*>(arena->allocate(1, 8));
    if (data == nullptr) return false;
    const BoltColumn value = BoltColumn::make_var_binary(data, nullptr, offs, rows, BoltType::Binary, arena);
    *out = BoltColumn::make_variant(md, &value, rows, gc.validity, arena);
    return out->data != nullptr;
}

// Replace every VARIANT group under field `f` (assembled as `col`) by its unshredded column, in place.
bool unshred_tree(const PqSchemaTree* t, const PqMeta* meta, int32_t f, BoltColumn* col, Arena* a, uint32_t depth) noexcept {
    if (depth > kMaxDepth) return false;
    if (is_variant_group(t, f)) return unshred_group(t, meta, f, *col, a, col);
    const PqField& fd = t->fields[f];
    if (fd.kind == PqFieldKind::Leaf) return true;
    if (!col->is_nested() || col->data == nullptr) return false;
    auto* kids = static_cast<BoltColumn*>(const_cast<void*>(col->data));
    int32_t c = fd.first_child;
    for (int64_t k = 0; c >= 0 && k < static_cast<int64_t>(fd.n_children); ++k) {   // bounded by the children
        if (k >= col->child_count() || !unshred_tree(t, meta, c, &kids[k], a, depth + 1u)) return false;
        c = t->fields[c].next_sibling;
    }
    return true;
}

bool has_variant_group(const PqSchemaTree* t, int32_t f, uint32_t depth) noexcept {
    if (depth > kMaxDepth) return false;
    if (is_variant_group(t, f)) return true;
    int32_t c = t->fields[f].first_child;
    for (uint32_t k = 0; c >= 0 && k < t->fields[f].n_children; ++k) {   // bounded by the children
        if (has_variant_group(t, c, depth + 1u)) return true;
        c = t->fields[c].next_sibling;
    }
    return false;
}

}  // namespace

bool pq_variant_group_ok(const PqSchemaTree* tree, int32_t f) noexcept {
    if (tree == nullptr || f < 0 || static_cast<uint32_t>(f) >= tree->n_fields) return false;
    const PqField& g = tree->fields[f];
    if (g.kind != PqFieldKind::Struct || g.n_children < 2u || g.n_children > 3u || g.first_child < 0) return false;
    const PqField& m = tree->fields[g.first_child];
    if (m.kind != PqFieldKind::Leaf || std::strcmp(m.name, "metadata") != 0) return false;
    int64_t pos = 0;
    const int32_t v = child_named(tree, f, "value", &pos);
    const int32_t t = child_named(tree, f, "typed_value", &pos);
    if (v >= 0 && tree->fields[v].kind != PqFieldKind::Leaf) return false;
    return (v >= 0 || t >= 0) && 1u + (v >= 0 ? 1u : 0u) + (t >= 0 ? 1u : 0u) == g.n_children;
}

namespace {
bool is_variant_group(const PqSchemaTree* t, int32_t f) noexcept {
    const PqField& fd = t->fields[f];
    return fd.node >= 0 && static_cast<uint32_t>(fd.node) < t->n_nodes &&
           t->nodes[fd.node].logical == static_cast<int32_t>(PqLogical::Variant) && pq_variant_group_ok(t, f);
}
}  // namespace

bool pq_variant_group_shredded(const PqSchemaTree* tree, int32_t f) noexcept {
    int64_t pos = 0;
    return pq_variant_group_ok(tree, f) && child_named(tree, f, "typed_value", &pos) >= 0;
}

bool parquet_read_variant_column(const uint8_t* buf, uint64_t len, const PqMeta* meta, const PqSchemaTree* tree,
                                 uint32_t row_group, uint32_t top_field, Arena* arena, BoltColumn* out,
                                 int64_t* out_rows) noexcept {
    if (buf == nullptr || meta == nullptr || tree == nullptr || arena == nullptr || out == nullptr || out_rows == nullptr) return false;
    if (top_field >= tree->n_top || !pq_variant_group_ok(tree, tree->top[top_field])) return false;
    const int32_t gf = tree->top[top_field];
    PqMeta raw{};
    BoltColumn gc;
    int64_t rows = 0;
    // The top-level group is taken by its shape alone (a writer may leave the VARIANT annotation off it).
    if (!raw_time_meta(meta, tree, gf, arena, &raw)) return false;
    for (uint32_t c = tree->fields[gf].col_begin; c < tree->fields[gf].col_end; ++c) {   // bounded: the group's leaves
        if (raw.columns[c].time_unit == 1) return false;
        if (raw.columns[c].time_unit == 3) raw.columns[c].time_unit = 0;
    }
    if (!parquet_read_nested_column_tree(buf, len, &raw, tree, row_group, top_field, arena, &gc, &rows)) return false;
    if (!unshred_group(tree, meta, gf, gc, arena, out)) return false;
    *out_rows = rows;
    return true;
}

bool parquet_read_nested_column_variants(const uint8_t* buf, uint64_t len, const PqMeta* meta, const PqSchemaTree* tree,
                                         uint32_t row_group, uint32_t top_field, Arena* arena, BoltColumn* out,
                                         int64_t* out_rows) noexcept {
    if (buf == nullptr || meta == nullptr || tree == nullptr || arena == nullptr || out == nullptr || out_rows == nullptr) return false;
    if (top_field >= tree->n_top) return false;
    const int32_t f = tree->top[top_field];
    if (!has_variant_group(tree, f, 0u)) return parquet_read_nested_column_tree(buf, len, meta, tree, row_group, top_field, arena, out, out_rows);
    PqMeta raw{};
    if (!raw_time_meta(meta, tree, f, arena, &raw) ||
        !parquet_read_nested_column_tree(buf, len, &raw, tree, row_group, top_field, arena, out, out_rows)) return false;
    return unshred_tree(tree, meta, f, out, arena, 0u);
}

}  // namespace parquet
}  // namespace ingest
}  // namespace bolt
