// bolt/kernels/bolt_variant_spark_schema.h — Spark's schema_of_variant(v) (G2CHK-963): the SQL type of one variant value.
//
// Spark 4.0 SchemaOfVariant: an object is OBJECT<k: T, ...> over its keys (byte order; "A" and "a" stay two fields), an array
// ARRAY<T> with T the merge of its elements, a decimal DECIMAL(max(digits, scale), scale) of its own value, every integer
// width BIGINT, null VOID. Merging (arrays, and objects field by field): equal types stay; VOID gives way; integers merge
// with decimals as DECIMAL(20,0) by integer digits and scale (past precision 38: DOUBLE); BIGINT + FLOAT is FLOAT; FLOAT or a
// decimal with DOUBLE, and FLOAT with a decimal, are DOUBLE; DATE + TIMESTAMP is TIMESTAMP, TIMESTAMP + TIMESTAMP_NTZ is
// TIMESTAMP, DATE + TIMESTAMP_NTZ is TIMESTAMP_NTZ; objects take the union of their keys; arrays merge their elements;
// anything else is VARIANT. A field name is printed as is when it is [A-Za-z_][A-Za-z0-9_]*, else in backticks (` doubled).
#pragma once

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "bolt/kernels/bolt_decimal.h"
#include "bolt/kernels/bolt_variant_spark_get.h"
#include "bolt/kernels/bolt_variant_spark_json.h"

namespace bolt {
namespace kernels {
namespace variantspark {

enum class SchemaKind : uint8_t {
    Void = 0, Boolean, Long, Decimal, Float, Double, String, Date, Timestamp, TimestampNtz, Binary, Variant, Object, Array,
};

// One type in the caller's pool. Object: `child` is its first field (a linked list in key order, `key` / `key_n` on each
// field node, the field's type in that node's `child`). Array: `child` is the element type.
struct SchemaNode {
    const uint8_t* key;
    uint32_t       key_n;
    uint32_t       child;   // node index + 1; 0 = none
    uint32_t       next;    // next field (node index + 1); 0 = last
    SchemaKind     kind;
    uint8_t        precision;
    uint8_t        scale;
    uint8_t        _pad;
};

struct SchemaPool {
    SchemaNode* nodes;
    uint32_t    cap;
    uint32_t    n;
};

enum class SchemaStatus : uint8_t { Ok = 0, Malformed, Budget };

namespace schema_detail {

inline uint32_t alloc(SchemaPool* p, SchemaKind k) noexcept {
    assert(p != nullptr);
    if (p->n >= p->cap) return 0;
    SchemaNode& s = p->nodes[p->n++];
    s = SchemaNode{nullptr, 0, 0, 0, k, 0, 0, 0};
    return p->n;   // index + 1
}
inline SchemaNode& at(SchemaPool* p, uint32_t i) noexcept {
    assert(i != 0 && i <= p->n);
    return p->nodes[i - 1u];
}

// Spark's DECIMAL type of a variant decimal (a two's-complement little-endian unscaled value of `w` bytes, scale `scale`):
// trailing zeros of the fraction stripped as Variant.getDecimal does, then (max(digits, scale), scale).
inline void decimal_type(const uint8_t* p, uint32_t w, uint32_t scale, uint8_t* prec, uint8_t* out_scale) noexcept {
    assert(w >= 1u && w <= 16u);
    uint64_t lo = 0, hi = 0;
    for (uint32_t i = 0; i < w; ++i) {
        if (i < 8u) lo |= uint64_t{p[i]} << (8u * i);
        else hi |= uint64_t{p[i]} << (8u * (i - 8u));
    }
    if ((p[w - 1u] & 0x80u) != 0) {                                  // sign-extend, then the magnitude
        if (w < 8u) { lo |= ~uint64_t{0} << (8u * w); hi = ~uint64_t{0}; }
        else if (w == 8u) hi = ~uint64_t{0};
        else if (w < 16u) hi |= ~uint64_t{0} << (8u * (w - 8u));
        lo = ~lo;
        hi = ~hi;
        if (++lo == 0) ++hi;
    }
    uint64_t qh = 0, ql = 0, r = 0;
    while (scale > 0 && (hi != 0 || lo != 0)) {                      // bounded: scale <= 38
        decimal::div_u128_u64(hi, lo, 10u, &qh, &ql, &r);
        if (r != 0) break;
        hi = qh;
        lo = ql;
        --scale;
    }
    if (hi == 0 && lo == 0) scale = 0;
    uint32_t digits = 0;
    for (uint32_t k = 0; k < 40u && (hi != 0 || lo != 0); ++k) {    // bounded: 39 digits in 128 bits
        decimal::div_u128_u64(hi, lo, 10u, &qh, &ql, &r);
        hi = qh;
        lo = ql;
        ++digits;
    }
    if (digits == 0) digits = 1;
    *prec = static_cast<uint8_t>(digits > scale ? digits : scale);
    *out_scale = static_cast<uint8_t>(scale);
}

inline SchemaStatus of(SchemaPool* p, const uint8_t* m, uint32_t mn, const uint8_t* v, uint32_t n, uint32_t depth,
                       uint32_t* out) noexcept;
inline uint32_t merge(SchemaPool* p, uint32_t a, uint32_t b, uint32_t depth) noexcept;

inline int key_cmp(const SchemaNode& x, const SchemaNode& y) noexcept {
    const uint32_t k = x.key_n < y.key_n ? x.key_n : y.key_n;
    const int c = k == 0 ? 0 : std::memcmp(x.key, y.key, k);
    return c != 0 ? c : (x.key_n < y.key_n ? -1 : x.key_n > y.key_n ? 1 : 0);
}

// Object fields `a` and `b` (sorted by key) merged into `a`'s list, reusing `b`'s nodes. Returns the merged list head.
inline uint32_t merge_fields(SchemaPool* p, uint32_t a, uint32_t b, uint32_t depth, bool* failed) noexcept {
    uint32_t head = 0, tail = 0;
    auto push = [&](uint32_t f) noexcept {
        at(p, f).next = 0;
        if (tail == 0) head = f; else at(p, tail).next = f;
        tail = f;
    };
    while (a != 0 || b != 0) {                                       // bounded: each step consumes a field
        if (b == 0 || (a != 0 && key_cmp(at(p, a), at(p, b)) < 0)) { const uint32_t nx = at(p, a).next; push(a); a = nx; continue; }
        if (a == 0 || key_cmp(at(p, a), at(p, b)) > 0) { const uint32_t nx = at(p, b).next; push(b); b = nx; continue; }
        const uint32_t na = at(p, a).next, nb = at(p, b).next;
        const uint32_t t = merge(p, at(p, a).child, at(p, b).child, depth + 1u);
        if (t == 0) { *failed = true; return 0; }
        at(p, a).child = t;
        push(a);
        a = na;
        b = nb;
    }
    return head;
}

inline bool numeric(SchemaKind k) noexcept {
    return k == SchemaKind::Long || k == SchemaKind::Decimal || k == SchemaKind::Float || k == SchemaKind::Double;
}

// The merge of types `a` and `b` (in place over `a` where possible); 0 when the pool is exhausted.
inline uint32_t merge(SchemaPool* p, uint32_t a, uint32_t b, uint32_t depth) noexcept {
    if (depth > jsonspark::kMaxNesting + 2u) return 0;
    SchemaNode& x = at(p, a);
    const SchemaNode& y = at(p, b);
    if (x.kind == SchemaKind::Void) return b;
    if (y.kind == SchemaKind::Void) return a;
    if (x.kind == SchemaKind::Object && y.kind == SchemaKind::Object) {
        bool failed = false;
        x.child = merge_fields(p, x.child, y.child, depth, &failed);
        return failed ? 0 : a;
    }
    if (x.kind == SchemaKind::Array && y.kind == SchemaKind::Array) {
        const uint32_t e = merge(p, x.child, y.child, depth + 1u);
        if (e == 0) return 0;
        at(p, a).child = e;
        return a;
    }
    if (x.kind == y.kind && (x.kind != SchemaKind::Decimal || (x.precision == y.precision && x.scale == y.scale))) return a;
    if (numeric(x.kind) && numeric(y.kind)) {
        const bool xd = x.kind == SchemaKind::Long || x.kind == SchemaKind::Decimal;
        const bool yd = y.kind == SchemaKind::Long || y.kind == SchemaKind::Decimal;
        if (xd && yd) {
            const uint32_t xp = x.kind == SchemaKind::Long ? 20u : x.precision, xs = x.kind == SchemaKind::Long ? 0u : x.scale;
            const uint32_t yp = y.kind == SchemaKind::Long ? 20u : y.precision, ys = y.kind == SchemaKind::Long ? 0u : y.scale;
            const uint32_t whole = (xp - xs) > (yp - ys) ? (xp - xs) : (yp - ys), sc = xs > ys ? xs : ys;
            if (whole + sc > 38u) { x.kind = SchemaKind::Double; return a; }
            x.kind = SchemaKind::Decimal;
            x.precision = static_cast<uint8_t>(whole + sc);
            x.scale = static_cast<uint8_t>(sc);
            return a;
        }
        const bool long_float = (x.kind == SchemaKind::Long && y.kind == SchemaKind::Float) ||
                                (x.kind == SchemaKind::Float && y.kind == SchemaKind::Long);
        x.kind = long_float ? SchemaKind::Float : SchemaKind::Double;
        return a;
    }
    auto pair = [&](SchemaKind u, SchemaKind w) noexcept {
        return (x.kind == u && y.kind == w) || (x.kind == w && y.kind == u);
    };
    if (pair(SchemaKind::Date, SchemaKind::Timestamp) || pair(SchemaKind::Timestamp, SchemaKind::TimestampNtz)) {
        x.kind = SchemaKind::Timestamp;
        return a;
    }
    if (pair(SchemaKind::Date, SchemaKind::TimestampNtz)) {
        x.kind = SchemaKind::TimestampNtz;
        return a;
    }
    x.kind = SchemaKind::Variant;
    x.child = 0;
    return a;
}

inline SchemaStatus container(SchemaPool* p, const uint8_t* m, uint32_t mn, const uint8_t* v, uint32_t n, uint32_t depth,
                              bool obj, uint32_t* out) noexcept {
    const uint8_t vh = static_cast<uint8_t>(v[0] >> 2);
    const bool large = obj ? ((vh >> 4) & 1u) != 0 : ((vh >> 2) & 1u) != 0;
    const uint32_t osize = (vh & 3u) + 1u, idsize = obj ? ((vh >> 2) & 3u) + 1u : 0u, nsize = large ? 4u : 1u;
    if (n < 1u + nsize) return SchemaStatus::Malformed;
    const uint64_t cnt = detail::get_le(v + 1, nsize);
    const uint64_t hdr = 1u + nsize + cnt * idsize + (cnt + 1u) * osize;
    if (hdr > n) return SchemaStatus::Malformed;
    const uint8_t* ids = v + 1 + nsize;
    const uint8_t* offs = ids + cnt * idsize;
    const uint8_t* data = v + hdr;
    const uint64_t dn = n - hdr;
    const uint32_t self = alloc(p, obj ? SchemaKind::Object : SchemaKind::Array);
    if (self == 0) return SchemaStatus::Budget;
    uint32_t elem = 0, tail = 0;
    const uint32_t ko = mn == 0 ? 1u : ((m[0] >> 6) & 3u) + 1u;
    if (obj && mn < 1u + ko) return SchemaStatus::Malformed;
    const uint64_t nkeys = obj ? detail::get_le(m + 1, ko) : 0u;
    const uint8_t* koff = m + 1 + ko;
    const uint8_t* kbytes = koff + (nkeys + 1u) * ko;
    for (uint64_t i = 0; i < cnt; ++i) {                             // bounded by the container
        const uint64_t off = detail::get_le(offs + i * osize, osize);
        if (off >= dn) return SchemaStatus::Malformed;
        uint32_t t = 0;
        const SchemaStatus s = of(p, m, mn, data + off, static_cast<uint32_t>(dn - off), depth + 1u, &t);
        if (s != SchemaStatus::Ok) return s;
        if (!obj) {
            elem = elem == 0 ? t : merge(p, elem, t, depth + 1u);
            if (elem == 0) return SchemaStatus::Budget;
            continue;
        }
        const uint64_t id = detail::get_le(ids + i * idsize, idsize);
        if (id >= nkeys) return SchemaStatus::Malformed;
        const uint64_t ka = detail::get_le(koff + id * ko, ko), kb = detail::get_le(koff + (id + 1u) * ko, ko);
        if (kb < ka || kbytes + kb > m + mn) return SchemaStatus::Malformed;
        const uint32_t f = alloc(p, SchemaKind::Void);
        if (f == 0) return SchemaStatus::Budget;
        SchemaNode& fn = at(p, f);
        fn.key = kbytes + ka;
        fn.key_n = static_cast<uint32_t>(kb - ka);
        fn.child = t;
        if (tail != 0 && key_cmp(at(p, tail), fn) >= 0) return SchemaStatus::Malformed;   // the spec sorts object keys
        if (tail == 0) at(p, self).child = f; else at(p, tail).next = f;
        tail = f;
    }
    if (!obj) {
        if (elem == 0) elem = alloc(p, SchemaKind::Void);
        if (elem == 0) return SchemaStatus::Budget;
        at(p, self).child = elem;
    }
    *out = self;
    return SchemaStatus::Ok;
}

inline SchemaStatus of(SchemaPool* p, const uint8_t* m, uint32_t mn, const uint8_t* v, uint32_t n, uint32_t depth,
                       uint32_t* out) noexcept {
    if (n == 0 || depth > jsonspark::kMaxNesting) return SchemaStatus::Malformed;
    const uint8_t basic = v[0] & 3u, h = static_cast<uint8_t>(v[0] >> 2);
    if (basic == kObject || basic == kArray) return container(p, m, mn, v, n, depth, basic == kObject, out);
    SchemaKind k = SchemaKind::String;
    uint8_t prec = 0, scale = 0;
    if (basic == kPrimitive) {
        switch (h) {
            case kNull: k = SchemaKind::Void; break;
            case kTrue: case kFalse: k = SchemaKind::Boolean; break;
            case kInt8: case kInt16: case kInt32: case kInt64: k = SchemaKind::Long; break;
            case kDouble: k = SchemaKind::Double; break;
            case kDecimal4: case kDecimal8: case kDecimal16: {
                const uint32_t w = h == kDecimal4 ? 4u : h == kDecimal8 ? 8u : 16u;
                if (n < 2u + w || v[1] > 38u) return SchemaStatus::Malformed;
                k = SchemaKind::Decimal;
                decimal_type(v + 2, w, v[1], &prec, &scale);
                if (prec > 38u) return SchemaStatus::Malformed;
                break;
            }
            case kDate: k = SchemaKind::Date; break;
            case kTimestamp: case kTimestampNanos: k = SchemaKind::Timestamp; break;
            case kTimestampNtz: case kTimestampNtzNanos: k = SchemaKind::TimestampNtz; break;
            case kFloat: k = SchemaKind::Float; break;
            case kBinary: k = SchemaKind::Binary; break;
            case kLongString: k = SchemaKind::String; break;
            default: return SchemaStatus::Malformed;                  // time / uuid: no Spark 4.0 SQL type
        }
    }
    const uint32_t self = alloc(p, k);
    if (self == 0) return SchemaStatus::Budget;
    at(p, self).precision = prec;
    at(p, self).scale = scale;
    *out = self;
    return SchemaStatus::Ok;
}

inline void put_name(jsonspark::Out* o, const uint8_t* s, uint32_t n) noexcept {
    bool plain = n != 0;
    for (uint32_t i = 0; i < n && plain; ++i) {
        const uint8_t c = s[i];
        const bool alpha = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
        plain = alpha || (i != 0 && c >= '0' && c <= '9');
    }
    if (plain) { o->put(reinterpret_cast<const char*>(s), n); return; }
    o->put('`');
    for (uint32_t i = 0; i < n; ++i) {
        if (s[i] == '`') o->put('`');
        o->put(static_cast<char>(s[i]));
    }
    o->put('`');
}

inline void render(SchemaPool* p, uint32_t t, jsonspark::Out* o, uint32_t depth) noexcept {
    if (depth > jsonspark::kMaxNesting + 2u) { o->over = true; return; }
    const SchemaNode& s = at(p, t);
    static constexpr const char* kNames[] = {"VOID", "BOOLEAN", "BIGINT", "DECIMAL", "FLOAT", "DOUBLE", "STRING", "DATE",
                                             "TIMESTAMP", "TIMESTAMP_NTZ", "BINARY", "VARIANT"};
    switch (s.kind) {
        case SchemaKind::Object: {
            o->put("OBJECT<", 7);
            bool first = true;
            for (uint32_t f = s.child; f != 0 && !o->over; f = at(p, f).next) {   // bounded by the pool
                if (!first) o->put(", ", 2);
                first = false;
                put_name(o, at(p, f).key, at(p, f).key_n);
                o->put(": ", 2);
                render(p, at(p, f).child, o, depth + 1u);
            }
            o->put('>');
            return;
        }
        case SchemaKind::Array:
            o->put("ARRAY<", 6);
            render(p, s.child, o, depth + 1u);
            o->put('>');
            return;
        case SchemaKind::Decimal: {
            char b[24];
            const int k = std::snprintf(b, sizeof b, "DECIMAL(%u,%u)", static_cast<unsigned>(s.precision),
                                        static_cast<unsigned>(s.scale));
            o->put(b, static_cast<uint32_t>(k));
            return;
        }
        default: {
            const char* nm = kNames[static_cast<uint8_t>(s.kind)];
            o->put(nm, static_cast<uint32_t>(std::strlen(nm)));
            return;
        }
    }
}

}  // namespace schema_detail

// schema_of_variant(v) into `o`, with type nodes from `pool` (reset here; a budget: past it, Budget).
inline SchemaStatus schema_of(const uint8_t* value, uint32_t vn, const uint8_t* metadata, uint32_t mn, SchemaPool* pool,
                              jsonspark::Out* o) noexcept {
    assert(pool != nullptr && o != nullptr);
    if (value == nullptr || metadata == nullptr || mn == 0 || (metadata[0] & 0x0Fu) != 1u) return SchemaStatus::Malformed;
    pool->n = 0;
    uint32_t t = 0;
    const SchemaStatus s = schema_detail::of(pool, metadata, mn, value, vn, 0, &t);
    if (s != SchemaStatus::Ok) return s;
    schema_detail::render(pool, t, o, 0);
    return o->over ? SchemaStatus::Budget : SchemaStatus::Ok;
}

}  // namespace variantspark
}  // namespace kernels
}  // namespace bolt
