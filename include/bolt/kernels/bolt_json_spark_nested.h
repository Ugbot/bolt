// bolt/kernels/bolt_json_spark_nested.h — Spark 4 schema_of_json, to_json and from_json over bolt nested values.
//
// Nested values are bolt_nested_row encodings read and built through a type tree (TreeView) plus per-node metadata
// taken from the BNT1 wire (names, decimal precision). Semantics follow Spark's JacksonGenerator / JacksonParser /
// JsonInferSchema under the default options (the options map is refused by the frontend, never approximated).
//
// Tiger Style: noexcept, no allocation (caller regions / pools), bounded loops and recursion (type depth <= kMaxDepth,
// JSON nesting <= kMaxNesting), >=2 asserts per non-trivial function.
#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_nested_row.h"
#include "bolt/bolt_nested_value.h"
#include "bolt/datetime/bolt_tz.h"
#include "bolt/kernels/bolt_json_spark.h"

namespace bolt {
namespace kernels {
namespace jsonspark {

namespace nrv = ::bolt::nested_row;

// ---- node metadata (names, decimal precision) from a BNT1 wire --------------------------------------------------------

struct NodeMeta {
    const char* name;
    uint8_t     name_len;
    uint8_t     precision;
};

// BNT1 header (8 bytes) then per node 25 fixed bytes + name. False on a malformed or too-long wire.
inline bool wire_meta(const uint8_t* wire, uint32_t len, NodeMeta* out, uint32_t cap, uint32_t* n) noexcept {
    assert(wire != nullptr && out != nullptr && n != nullptr);
    if (len < 8 || std::memcmp(wire, "BNT1", 4) != 0) return false;
    const uint32_t count = static_cast<uint32_t>(wire[4]) | (static_cast<uint32_t>(wire[5]) << 8);
    if (count > cap) return false;
    uint32_t at = 8;
    for (uint32_t i = 0; i < count; ++i) {
        if (at + 25u > len) return false;
        const uint32_t nl = wire[at + 24];
        if (at + 25u + nl > len) return false;
        out[i] = NodeMeta{reinterpret_cast<const char*>(wire + at + 25), static_cast<uint8_t>(nl), wire[at + 20]};
        at += 25u + nl;
    }
    *n = count;
    return true;
}

// ---- numbers as Java prints them --------------------------------------------------------------------------------------

// Java's Float.toString (JDK 19+: the shortest decimal that rounds to the float).
inline uint32_t java_float_chars(char* out, uint32_t cap, float v) noexcept {
    assert(out != nullptr && cap >= 32);
    if (v != v) { std::memcpy(out, "NaN", 3); return 3; }
    uint32_t w = 0;
    if (std::signbit(v)) { out[w++] = '-'; v = -v; }
    if (v == 0.0f) { std::memcpy(out + w, "0.0", 3); return w + 3; }
    if (std::isinf(v)) { std::memcpy(out + w, "Infinity", 8); return w + 8; }
    char t[48];
    uint32_t nd = 0;
    int e10 = 0;
    for (int prec = 0; prec <= 8; ++prec) {                          // d.ddd e+XX with prec fraction digits
        std::snprintf(t, sizeof t, "%.*e", prec, static_cast<double>(v));
        if (std::strtof(t, nullptr) == v) { nd = static_cast<uint32_t>(prec) + 1u; break; }
    }
    if (nd == 0) { std::snprintf(t, sizeof t, "%.8e", static_cast<double>(v)); nd = 9; }
    if (nd == 1) {                                                   // JDK renders >= 2 digits: the closest 2-digit one
        char t2[48];
        std::snprintf(t2, sizeof t2, "%.1e", static_cast<double>(v));
        if (std::strtof(t2, nullptr) == v) { std::memcpy(t, t2, sizeof t2); nd = 2; }
    }
    char dig[12];
    uint32_t k = 0;
    for (const char* q = t; *q != 'e' && *q != '\0'; ++q) if (*q >= '0' && *q <= '9') dig[k++] = *q;
    e10 = std::atoi(std::strchr(t, 'e') + 1);
    while (k > 1 && dig[k - 1] == '0') --k;                          // trailing zeros are not digits
    nd = k;
    const int32_t kk = e10 + 1;                                      // v ~= 0.d1..dn * 10^kk
    if (v >= 1e-3f && v < 1e7f) {
        if (kk <= 0) {
            out[w++] = '0'; out[w++] = '.';
            for (int32_t z = 0; z < -kk; ++z) out[w++] = '0';
            for (uint32_t d = 0; d < nd; ++d) out[w++] = dig[d];
        } else if (static_cast<uint32_t>(kk) >= nd) {
            for (uint32_t d = 0; d < nd; ++d) out[w++] = dig[d];
            for (uint32_t z = nd; z < static_cast<uint32_t>(kk); ++z) out[w++] = '0';
            out[w++] = '.'; out[w++] = '0';
        } else {
            for (int32_t d = 0; d < kk; ++d) out[w++] = dig[d];
            out[w++] = '.';
            for (uint32_t d = static_cast<uint32_t>(kk); d < nd; ++d) out[w++] = dig[d];
        }
        return w;
    }
    out[w++] = dig[0]; out[w++] = '.';
    if (nd == 1) out[w++] = '0';
    for (uint32_t d = 1; d < nd; ++d) out[w++] = dig[d];
    out[w++] = 'E';
    w += static_cast<uint32_t>(std::snprintf(out + w, cap - w, "%d", e10));
    return w;
}

// java.math.BigDecimal.toString of unscaled * 10^-scale (scale >= 0).
inline uint32_t java_decimal_chars(char* out, uint32_t cap, __int128 unscaled, uint32_t scale) noexcept {
    assert(out != nullptr && cap >= 64 && scale <= 38);
    uint32_t w = 0;
    unsigned __int128 u = unscaled < 0 ? static_cast<unsigned __int128>(0) - static_cast<unsigned __int128>(unscaled)
                                       : static_cast<unsigned __int128>(unscaled);
    if (unscaled < 0) out[w++] = '-';
    char d[48];
    uint32_t nd = 0;
    do { d[nd++] = static_cast<char>('0' + static_cast<int>(u % 10u)); u /= 10u; } while (u != 0 && nd < sizeof d);
    for (uint32_t a = 0, b = nd - 1; a < b; ++a, --b) { const char t = d[a]; d[a] = d[b]; d[b] = t; }
    if (scale == 0) { std::memcpy(out + w, d, nd); return w + nd; }
    const int32_t adjusted = static_cast<int32_t>(nd) - 1 - static_cast<int32_t>(scale);
    if (adjusted >= -6) {
        if (nd > scale) {
            std::memcpy(out + w, d, nd - scale); w += nd - scale;
            out[w++] = '.';
            std::memcpy(out + w, d + (nd - scale), scale); w += scale;
        } else {
            out[w++] = '0'; out[w++] = '.';
            for (uint32_t z = nd; z < scale; ++z) out[w++] = '0';
            std::memcpy(out + w, d, nd); w += nd;
        }
        return w;
    }
    out[w++] = d[0];
    if (nd > 1) { out[w++] = '.'; std::memcpy(out + w, d + 1, nd - 1); w += nd - 1; }
    out[w++] = 'E';
    w += static_cast<uint32_t>(std::snprintf(out + w, cap - w, "%d", adjusted));
    return w;
}

// ---- schema_of_json -----------------------------------------------------------------------------------------------------

enum class IK : uint8_t { Null = 0, Bool, Long, Decimal, Double, String, Array, Struct };

struct IType {
    IK       kind;
    uint8_t  prec, scale;
    uint8_t  _pad;
    uint32_t first;           // Array: element type; Struct: head of the sorted field list (0 = none)
};
struct IField {
    const uint8_t* name;
    uint32_t       name_len;
    uint32_t       type;
    uint32_t       next;
};
// Caller-provided pools (budgets json_infer_types / json_infer_fields / json_infer_name_bytes). Index 0 is the
// Null type / the empty list.
struct InferPool {
    IType*   types;  uint32_t ntypes, cap_types;
    IField*  fields; uint32_t nfields, cap_fields;
    uint8_t* chars;  uint32_t nchars, cap_chars;
    bool     over;
};

namespace inferd {

inline uint32_t mk(InferPool* p, IK k, uint8_t prec, uint8_t scale, uint32_t first) noexcept {
    if (p->ntypes >= p->cap_types) { p->over = true; return 0; }
    const uint32_t i = p->ntypes++;
    p->types[i] = IType{k, prec, scale, 0, first};
    return i;
}
inline uint32_t mkfield(InferPool* p, const uint8_t* name, uint32_t len, uint32_t type) noexcept {
    if (p->nfields >= p->cap_fields) { p->over = true; return 0; }
    const uint32_t i = p->nfields++;
    p->fields[i] = IField{name, len, type, 0};
    return i;
}

// Java String.compareTo over UTF-16 code units of two UTF-8 names.
inline uint32_t unit_at(const uint8_t* s, uint32_t n, uint32_t* i, uint32_t* low) noexcept {
    if (*low != 0) { const uint32_t u = *low; *low = 0; return u; }
    const uint8_t c = s[*i];
    if (c < 0x80) { ++*i; return c; }
    if (c < 0xE0) { const uint32_t u = ((c & 0x1Fu) << 6) | (s[*i + 1] & 0x3Fu); *i += 2; return u; }
    if (c < 0xF0) { const uint32_t u = ((c & 0x0Fu) << 12) | ((s[*i + 1] & 0x3Fu) << 6) | (s[*i + 2] & 0x3Fu); *i += 3; return u; }
    const uint32_t cp = ((c & 0x07u) << 18) | ((s[*i + 1] & 0x3Fu) << 12) | ((s[*i + 2] & 0x3Fu) << 6) | (s[*i + 3] & 0x3Fu);
    *i += 4;
    (void)n;
    *low = 0xDC00u + ((cp - 0x10000u) & 0x3FFu);
    return 0xD800u + ((cp - 0x10000u) >> 10);
}
inline int name_cmp(const IField& a, const IField& b) noexcept {
    uint32_t i = 0, j = 0, la = 0, lb = 0;
    while ((i < a.name_len || la != 0) && (j < b.name_len || lb != 0)) {
        const uint32_t x = unit_at(a.name, a.name_len, &i, &la), y = unit_at(b.name, b.name_len, &j, &lb);
        if (x != y) return x < y ? -1 : 1;
    }
    const bool ea = !(i < a.name_len || la != 0), eb = !(j < b.name_len || lb != 0);
    return ea && eb ? 0 : (ea ? -1 : 1);
}

inline uint32_t compat(InferPool* p, uint32_t a, uint32_t b, uint32_t depth) noexcept;

inline bool same_type(const InferPool* p, uint32_t a, uint32_t b) noexcept {
    const IType &x = p->types[a], &y = p->types[b];
    return x.kind == y.kind && x.prec == y.prec && x.scale == y.scale && x.kind != IK::Struct && x.kind != IK::Array;
}

// The common type of two struct types: fields merged by name, sorted, same names reduced pairwise.
inline uint32_t merge_structs(InferPool* p, uint32_t a, uint32_t b, uint32_t depth) noexcept {
    uint32_t fa = p->types[a].first, fb = p->types[b].first, head = 0, tail = 0;
    for (uint64_t guard = 0; fa != 0 || fb != 0; ++guard) {
        if (guard > (uint64_t{1} << 26)) { p->over = true; return 0; }
        uint32_t src;
        if (fa != 0 && fb != 0) src = name_cmp(p->fields[fa], p->fields[fb]) <= 0 ? fa : fb;
        else src = fa != 0 ? fa : fb;
        const IField f = p->fields[src];
        if (src == fa) fa = p->fields[fa].next; else fb = p->fields[fb].next;
        if (tail != 0 && p->fields[tail].name_len == f.name_len &&
            (f.name_len == 0 || std::memcmp(p->fields[tail].name, f.name, f.name_len) == 0)) {
            const uint32_t t = compat(p, p->fields[tail].type, f.type, depth + 1);
            if (p->over) return 0;
            p->fields[tail].type = t;
            continue;
        }
        const uint32_t nf = mkfield(p, f.name, f.name_len, f.type);
        if (nf == 0) return 0;
        if (tail == 0) head = nf; else p->fields[tail].next = nf;
        tail = nf;
    }
    return mk(p, IK::Struct, 0, 0, head);
}

inline uint32_t decimal_of_long(InferPool* p) noexcept { return mk(p, IK::Decimal, 20, 0, 0); }

// Spark's JsonInferSchema.compatibleType.
inline uint32_t compat(InferPool* p, uint32_t a, uint32_t b, uint32_t depth) noexcept {
    assert(p != nullptr);
    if (depth > kMaxNesting + 2u) { p->over = true; return 0; }
    const IType x = p->types[a], y = p->types[b];
    if (x.kind == IK::Null) return b;
    if (y.kind == IK::Null) return a;
    if (same_type(p, a, b)) return a;
    auto str = [&]() noexcept { return mk(p, IK::String, 0, 0, 0); };
    auto dbl = [&]() noexcept { return mk(p, IK::Double, 0, 0, 0); };
    const bool xn = x.kind == IK::Long || x.kind == IK::Decimal || x.kind == IK::Double;
    const bool yn = y.kind == IK::Long || y.kind == IK::Decimal || y.kind == IK::Double;
    if (xn && yn) {
        if (x.kind == IK::Double || y.kind == IK::Double) return dbl();
        if (x.kind == IK::Long && y.kind == IK::Decimal) {          // a decimal wider than long keeps itself
            if (y.prec - y.scale >= 20) return b;
            const uint32_t d = decimal_of_long(p);
            return d == 0 ? 0 : compat(p, d, b, depth + 1);
        }
        if (x.kind == IK::Decimal && y.kind == IK::Long) {
            if (x.prec - x.scale >= 20) return a;
            const uint32_t d = decimal_of_long(p);
            return d == 0 ? 0 : compat(p, a, d, depth + 1);
        }
        const uint32_t scale = x.scale > y.scale ? x.scale : y.scale;
        const uint32_t xr = static_cast<uint32_t>(x.prec) - x.scale, yr = static_cast<uint32_t>(y.prec) - y.scale;
        const uint32_t range = xr > yr ? xr : yr;
        if (range + scale > 38) return dbl();
        return mk(p, IK::Decimal, static_cast<uint8_t>(range + scale), static_cast<uint8_t>(scale), 0);
    }
    if (x.kind == IK::Struct && y.kind == IK::Struct) return merge_structs(p, a, b, depth);
    if (x.kind == IK::Array && y.kind == IK::Array) {
        const uint32_t e = compat(p, x.first, y.first, depth + 1);
        return p->over ? 0 : mk(p, IK::Array, 0, 0, e);
    }
    return str();
}

// Decode the current FieldName token into the pool's char region.
inline bool take_name(InferPool* p, const Parser& ps, const uint8_t** name, uint32_t* len) noexcept {
    Out o{reinterpret_cast<char*>(p->chars) + p->nchars, p->cap_chars - p->nchars, 0, false};
    token_utf8(ps, &o, false);
    if (o.over) { p->over = true; return false; }
    *name = p->chars + p->nchars;
    *len = o.n;
    p->nchars += o.n;
    return true;
}

// JsonInferSchema.inferField for the value at the parser's current token. ~0u on a syntax error / budget.
inline uint32_t infer(Parser* ps, InferPool* p, uint32_t depth) noexcept {
    assert(ps != nullptr && p != nullptr);
    if (depth > kMaxNesting) return ~0u;
    switch (ps->cur) {
        case Tok::Null: return 0;
        case Tok::True: case Tok::False: return mk(p, IK::Bool, 0, 0, 0);
        case Tok::String: return ps->tb == ps->te ? 0u : mk(p, IK::String, 0, 0, 0);
        case Tok::Float: return mk(p, IK::Double, 0, 0, 0);
        case Tok::Int: {
            const char* t = reinterpret_cast<const char*>(ps->s + ps->tb);
            const uint32_t len = ps->te - ps->tb;
            const uint32_t digits = len - (t[0] == '-' ? 1u : 0u);
            if (digits < 19 || (digits == 19 && (t[0] == '-' ? std::strncmp(t, "-9223372036854775808", 20) <= 0 || len < 20
                                                             : std::strncmp(t, "9223372036854775807", 19) <= 0))) {
                // fits int64 when <= the extreme of the same length
                if (digits < 19) return mk(p, IK::Long, 0, 0, 0);
                if (t[0] == '-' ? (len == 20 ? std::strncmp(t, "-9223372036854775808", 20) <= 0 : true)
                                : std::strncmp(t, "9223372036854775807", 19) <= 0) return mk(p, IK::Long, 0, 0, 0);
            }
            if (digits <= 38) return mk(p, IK::Decimal, static_cast<uint8_t>(digits), 0, 0);
            return mk(p, IK::Double, 0, 0, 0);
        }
        case Tok::StartObject: {
            uint32_t head = 0;
            for (uint64_t guard = 0;; ++guard) {
                const Tok t = next(ps);
                if (t == Tok::Error || t == Tok::None) return ~0u;
                if (t == Tok::EndObject) break;
                if (guard > ps->n) return ~0u;
                const uint8_t* nm = nullptr;
                uint32_t nl = 0;
                if (!take_name(p, *ps, &nm, &nl)) return ~0u;
                const Tok v = next(ps);
                if (v == Tok::Error || v == Tok::None) return ~0u;
                const uint32_t ty = infer(ps, p, depth + 1);
                if (ty == ~0u || p->over) return ~0u;
                const uint32_t f = mkfield(p, nm, nl, ty);
                if (f == 0) return ~0u;
                uint32_t prev = 0, cur = head;                      // stable insertion by name
                while (cur != 0 && name_cmp(p->fields[cur], p->fields[f]) <= 0) { prev = cur; cur = p->fields[cur].next; }
                p->fields[f].next = cur;
                if (prev == 0) head = f; else p->fields[prev].next = f;
            }
            const uint32_t st = mk(p, IK::Struct, 0, 0, head);
            return st == 0 ? ~0u : st;
        }
        case Tok::StartArray: {
            uint32_t elem = 0;
            for (uint64_t guard = 0;; ++guard) {
                const Tok t = next(ps);
                if (t == Tok::Error || t == Tok::None) return ~0u;
                if (t == Tok::EndArray) break;
                if (guard > ps->n) return ~0u;
                const uint32_t ty = infer(ps, p, depth + 1);
                if (ty == ~0u || p->over) return ~0u;
                elem = compat(p, elem, ty, 0);
                if (p->over) return ~0u;
            }
            const uint32_t at = mk(p, IK::Array, 0, 0, elem);
            return at == 0 ? ~0u : at;
        }
        default: return ~0u;
    }
}

// canonicalizeType: false when the type vanishes (an empty struct, or an array of one).
inline bool vanishes(const InferPool* p, uint32_t t, uint32_t depth) noexcept {
    const IType& x = p->types[t];
    if (depth > kMaxNesting + 2u) return false;
    if (x.kind == IK::Array) return vanishes(p, x.first, depth + 1);
    if (x.kind != IK::Struct) return false;
    for (uint32_t f = x.first; f != 0; f = p->fields[f].next) {
        if (p->fields[f].name_len != 0 && !vanishes(p, p->fields[f].type, depth + 1)) return false;
    }
    return true;
}

inline void put_name(Out* o, const IField& f) noexcept {
    bool plain = f.name_len != 0, digits = true;
    for (uint32_t i = 0; i < f.name_len; ++i) {
        const uint8_t c = f.name[i];
        const bool al = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
        const bool dg = c >= '0' && c <= '9';
        if (!al && !dg) plain = false;
        if (!dg) digits = false;
    }
    if (plain && !digits) { o->put(reinterpret_cast<const char*>(f.name), f.name_len); return; }
    o->put('`');
    for (uint32_t i = 0; i < f.name_len; ++i) { if (f.name[i] == '`') o->put('`'); o->put(static_cast<char>(f.name[i])); }
    o->put('`');
}

inline void render(const InferPool* p, uint32_t t, Out* o, uint32_t depth) noexcept {
    const IType& x = p->types[t];
    if (depth > kMaxNesting + 2u) { o->over = true; return; }
    switch (x.kind) {
        case IK::Null: case IK::String: o->put("STRING", 6); return;
        case IK::Bool: o->put("BOOLEAN", 7); return;
        case IK::Long: o->put("BIGINT", 6); return;
        case IK::Double: o->put("DOUBLE", 6); return;
        case IK::Decimal: {
            char b[24];
            const int n = std::snprintf(b, sizeof b, "DECIMAL(%u,%u)", static_cast<unsigned>(x.prec), static_cast<unsigned>(x.scale));
            o->put(b, static_cast<uint32_t>(n));
            return;
        }
        case IK::Array:
            o->put("ARRAY<", 6); render(p, x.first, o, depth + 1); o->put('>');
            return;
        case IK::Struct: {
            o->put("STRUCT<", 7);
            bool first = true;
            for (uint32_t f = x.first; f != 0; f = p->fields[f].next) {
                if (p->fields[f].name_len == 0 || vanishes(p, p->fields[f].type, depth + 1)) continue;
                if (!first) o->put(", ", 2);
                first = false;
                put_name(o, p->fields[f]);
                o->put(": ", 2);
                render(p, p->fields[f].type, o, depth + 1);
            }
            o->put('>');
            return;
        }
    }
}

}  // namespace inferd

// schema_of_json(json) as DDL text into o. 1 ok, 0 malformed input, -1 a pool / output budget.
inline int schema_of_json(const uint8_t* json, uint32_t len, InferPool* pool, Out* o) noexcept {
    assert(pool != nullptr && o != nullptr);
    pool->ntypes = 1; pool->nfields = 1; pool->nchars = 0; pool->over = false;
    pool->types[0] = IType{IK::Null, 0, 0, 0, 0};
    Parser ps;
    parser_init(&ps, json, len, false, true);
    const Tok t = next(&ps);
    if (t == Tok::Error) return 0;
    uint32_t ty = 0;
    if (t != Tok::None) {
        ty = inferd::infer(&ps, pool, 0);
        if (pool->over) return -1;
        if (ty == ~0u) return 0;
    }
    if (inferd::vanishes(pool, ty, 0)) {
        // the root erased to nothing: a struct stays an empty struct, an array of structs keeps its element, else STRING
        const IType& r = pool->types[ty];
        if (r.kind == IK::Struct) o->put("STRUCT<>", 8);
        else if (r.kind == IK::Array && pool->types[r.first].kind == IK::Struct) o->put("ARRAY<STRUCT<>>", 15);
        else o->put("STRING", 6);
    } else {
        inferd::render(pool, ty, o, 0);
    }
    return o->over ? -1 : 1;
}

// ---- to_json ----------------------------------------------------------------------------------------------------------

// Renders one variant {metadata, value} as JSON (variantspark::nested_variant_json): 1 ok, -1 budget, -2 unsupported.
using VariantJsonFn = int (*)(const uint8_t* metadata, uint32_t mn, const uint8_t* value, uint32_t vn, Out* o, tz::ZoneId zone);

struct Tree {
    nrv::TreeView     t;
    const NodeMeta*   meta;     // per node (BNT1 names / decimal precision)
    bolt::tz::ZoneId  tz;       // session time zone (TIMESTAMP rendering)
    VariantJsonFn     variant_json = nullptr;   // a Variant-annotated {metadata, value} node; null refuses it (-2)
};

namespace tojd {

inline bool is_null(const nrv::TypeNode& nd, const nrv::Span& v) noexcept { return nd.nullable != 0 && (v.len == 0 || v.p[0] == 0); }

// Jacksongenerator escaping over UTF-16 chars: quote, backslash, controls; everything else (supplementary too) raw.
inline void put_escaped(Out* o, const char* s, uint32_t n) noexcept {
    o->put('"');
    for (uint32_t i = 0; i < n; ++i) {
        const uint8_t c = static_cast<uint8_t>(s[i]);
        if (c >= 0x20 && c != '"' && c != '\\') { o->put(static_cast<char>(c)); continue; }
        put_unit_escaped(o, c);
    }
    o->put('"');
}

inline void civil(int64_t days, int64_t* y, uint32_t* m, uint32_t* d) noexcept {   // days since 1970-01-01 -> y-m-d
    int64_t z = days + 719468;
    const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    const uint32_t doe = static_cast<uint32_t>(z - era * 146097);
    const uint32_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const int64_t yy = static_cast<int64_t>(yoe) + era * 400;
    const uint32_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const uint32_t mp = (5 * doy + 2) / 153;
    *d = doy - (153 * mp + 2) / 5 + 1;
    *m = mp < 10 ? mp + 3 : mp - 9;
    *y = yy + (*m <= 2 ? 1 : 0);
}

// A proleptic-ISO year as Spark's yyyy prints it: four digits, a leading '-' below zero, a leading '+' past 9999.
inline uint32_t put_year(char* b, size_t cap, int64_t y) noexcept {
    assert(b != nullptr && cap >= 24);
    if (y >= 0 && y <= 9999) return static_cast<uint32_t>(std::snprintf(b, cap, "%04d", static_cast<int>(y)));
    if (y > 9999) return static_cast<uint32_t>(std::snprintf(b, cap, "+%lld", static_cast<long long>(y)));
    return static_cast<uint32_t>(std::snprintf(b, cap, "-%04lld", static_cast<long long>(-y)));
}

inline bool put_date(Out* o, int64_t days) noexcept {
    int64_t y; uint32_t m, d;
    civil(days, &y, &m, &d);
    char b[48];
    uint32_t n = put_year(b, sizeof b, y);
    n += static_cast<uint32_t>(std::snprintf(b + n, sizeof b - n, "-%02u-%02u", m, d));
    o->put(b, n);
    return true;
}

// yyyy-MM-dd'T'HH:mm:ss.SSS[XXX]; XXX prints seconds of an offset that has them (LMT zones).
inline bool put_timestamp(Out* o, int64_t utc_us, bolt::tz::ZoneId z, bool ntz) noexcept {
    const int64_t local = ntz ? utc_us : bolt::tz::utc_micros_to_local(z, utc_us);
    const int64_t secs = bolt::tz::floor_div(local, bolt::tz::kMicrosPerSec);
    const int64_t frac = local - secs * bolt::tz::kMicrosPerSec;
    const int64_t days = bolt::tz::floor_div(secs, 86400);
    const int64_t sod = secs - days * 86400;
    int64_t y; uint32_t m, d;
    civil(days, &y, &m, &d);
    char b[96];
    b[0] = '"';
    uint32_t n = 1 + put_year(b + 1, sizeof b - 1, y);
    n += static_cast<uint32_t>(std::snprintf(b + n, sizeof b - n, "-%02u-%02uT%02d:%02d:%02d.%03d", m, d,
                                             static_cast<int>(sod / 3600), static_cast<int>((sod / 60) % 60),
                                             static_cast<int>(sod % 60), static_cast<int>(frac / 1000)));
    if (!ntz) {
        const int64_t off = bolt::tz::floor_div(local - utc_us, bolt::tz::kMicrosPerSec);
        if (off == 0) { b[n++] = 'Z'; }
        else {
            const int64_t a = off < 0 ? -off : off;
            n += static_cast<uint32_t>(std::snprintf(b + n, sizeof b - n, "%c%02d:%02d", off < 0 ? '-' : '+',
                                                     static_cast<int>(a / 3600), static_cast<int>((a / 60) % 60)));
            if (a % 60 != 0) n += static_cast<uint32_t>(std::snprintf(b + n, sizeof b - n, ":%02d", static_cast<int>(a % 60)));
        }
    }
    b[n++] = '"';
    o->put(b, n);
    return true;
}

inline void put_base64(Out* o, const uint8_t* p, uint32_t n) noexcept {
    static const char a[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    o->put('"');
    uint32_t i = 0;
    for (; i + 2 < n; i += 3) {
        const uint32_t v = (static_cast<uint32_t>(p[i]) << 16) | (static_cast<uint32_t>(p[i + 1]) << 8) | p[i + 2];
        o->put(a[v >> 18]); o->put(a[(v >> 12) & 63]); o->put(a[(v >> 6) & 63]); o->put(a[v & 63]);
    }
    if (i + 1 == n) { const uint32_t v = static_cast<uint32_t>(p[i]) << 16; o->put(a[v >> 18]); o->put(a[(v >> 12) & 63]); o->put("==", 2); }
    else if (i + 2 == n) {
        const uint32_t v = (static_cast<uint32_t>(p[i]) << 16) | (static_cast<uint32_t>(p[i + 1]) << 8);
        o->put(a[v >> 18]); o->put(a[(v >> 12) & 63]); o->put(a[(v >> 6) & 63]); o->put('=');
    }
    o->put('"');
}

inline int64_t read_int(const uint8_t* b, uint32_t len) noexcept {
    int64_t i = 0;
    std::memcpy(&i, b, len);
    if (len < 8) { const int s = 64 - 8 * static_cast<int>(len); i = static_cast<int64_t>(static_cast<uint64_t>(i) << s) >> s; }
    return i;
}

inline void put_double(Out* o, double d, bool as_key) noexcept {
    char b[48];
    if (d != d) { o->put(as_key ? "NaN" : "\"NaN\"", as_key ? 3 : 5); return; }
    if (std::isinf(d)) {
        const char* t = d > 0 ? (as_key ? "Infinity" : "\"Infinity\"") : (as_key ? "-Infinity" : "\"-Infinity\"");
        o->put(t, static_cast<uint32_t>(std::strlen(t)));
        return;
    }
    o->put(b, java_double_chars(b, sizeof b, d));
}

// One leaf. 1 ok, -2 unsupported.
inline int put_leaf(const Tree& c, uint32_t ni, const nrv::LeafView& lv, Out* o) noexcept {
    const nrv::TypeNode& nd = c.t.nodes[ni];
    char b[80];
    switch (nd.type) {
        case BoltType::Bool: o->put(lv.bytes[0] != 0 ? "true" : "false", lv.bytes[0] != 0 ? 4 : 5); return 1;
        case BoltType::Int8: case BoltType::Int16: case BoltType::Int32: case BoltType::Int64: {
            const int64_t iv = read_int(lv.bytes, lv.len);
            if (nd.scale != 0) {                                     // a narrow decimal rides the integer lane with its scale
                o->put(b, java_decimal_chars(b, sizeof b, static_cast<__int128>(iv), nd.scale));
                return 1;
            }
            const int n = std::snprintf(b, sizeof b, "%lld", static_cast<long long>(iv));
            o->put(b, static_cast<uint32_t>(n));
            return 1;
        }
        case BoltType::Float64: { double d; std::memcpy(&d, lv.bytes, 8); put_double(o, d, false); return 1; }
        case BoltType::Float32: {
            float f; std::memcpy(&f, lv.bytes, 4);
            if (f != f || std::isinf(f)) { put_double(o, static_cast<double>(f), false); return 1; }
            o->put(b, java_float_chars(b, sizeof b, f));
            return 1;
        }
        case BoltType::Decimal128: {
            __int128 u; std::memcpy(&u, lv.bytes, 16);
            o->put(b, java_decimal_chars(b, sizeof b, u, nd.scale));
            return 1;
        }
        case BoltType::Date32: { int32_t d; std::memcpy(&d, lv.bytes, 4); o->put('"'); const bool ok = put_date(o, d); o->put('"'); return ok ? 1 : -2; }
        case BoltType::Timestamp: case BoltType::TimestampNtz: {
            int64_t us; std::memcpy(&us, lv.bytes, 8);
            return put_timestamp(o, us, c.tz, nd.type == BoltType::TimestampNtz) ? 1 : -2;
        }
        case BoltType::Utf8: put_escaped(o, reinterpret_cast<const char*>(lv.bytes), lv.len); return 1;
        case BoltType::Binary: put_base64(o, lv.bytes, lv.len); return 1;
        default: return -2;
    }
}

inline int put_key(const Tree& c, uint32_t ni, const nrv::LeafView& lv, Out* o) noexcept {
    const nrv::TypeNode& nd = c.t.nodes[ni];
    char b[48];
    switch (nd.type) {
        case BoltType::Utf8: put_escaped(o, reinterpret_cast<const char*>(lv.bytes), lv.len); return 1;
        case BoltType::Bool: o->put(lv.bytes[0] != 0 ? "\"true\"" : "\"false\"", lv.bytes[0] != 0 ? 6 : 7); return 1;
        case BoltType::Int8: case BoltType::Int16: case BoltType::Int32: case BoltType::Int64: {
            const int n = std::snprintf(b, sizeof b, "\"%lld\"", static_cast<long long>(read_int(lv.bytes, lv.len)));
            o->put(b, static_cast<uint32_t>(n));
            return 1;
        }
        default: return -2;
    }
}

inline int value(const Tree& c, uint32_t ni, nrv::Span v, Out* o, uint32_t depth) noexcept;

inline int null_or(const Tree& c, uint32_t ni, nrv::Span v, Out* o, bool* was_null) noexcept {
    (void)c; (void)o;
    *was_null = is_null(c.t.nodes[ni], v);
    return 1;
}

inline int put_struct(const Tree& c, uint32_t ni, nrv::Span v, Out* o, uint32_t depth) noexcept {
    const nrv::TypeNode& nd = c.t.nodes[ni];
    o->put('{');
    bool first = true;
    for (uint32_t k = 0; k < nd.n_children; ++k) {
        nrv::Span fs{};
        if (nrv::struct_field(c.t, ni, v, k, &fs) != 1) return -2;
        const uint32_t ci = nd.first_child + k;
        if (is_null(c.t.nodes[ci], fs)) continue;                    // ignoreNullFields (the default)
        if (!first) o->put(',');
        first = false;
        put_escaped(o, c.meta[ci].name, c.meta[ci].name_len);
        o->put(':');
        const int r = value(c, ci, fs, o, depth + 1);
        if (r != 1) return r;
    }
    o->put('}');
    return 1;
}

inline int put_list(const Tree& c, uint32_t ni, nrv::Span v, Out* o, uint32_t depth) noexcept {
    nrv::ListCursor cur{};
    if (nrv::list_open(c.t, ni, v, &cur) != 1) return -2;
    const uint32_t ei = c.t.nodes[ni].first_child;
    o->put('[');
    nrv::Span e{};
    for (bool first = true; nrv::list_next(&cur, &e); first = false) {
        if (!first) o->put(',');
        if (is_null(c.t.nodes[ei], e)) { o->put("null", 4); continue; }
        const int r = value(c, ei, e, o, depth + 1);
        if (r != 1) return r;
    }
    o->put(']');
    return 1;
}

inline int put_map(const Tree& c, uint32_t ni, nrv::Span v, Out* o, uint32_t depth) noexcept {
    nrv::ListCursor cur{};
    if (nrv::list_open(c.t, ni, v, &cur) != 1) return -2;
    const uint32_t en = c.t.nodes[ni].first_child, kn = c.t.nodes[en].first_child, vn = kn + 1;
    o->put('{');
    nrv::Span e{};
    for (bool first = true; nrv::list_next(&cur, &e); first = false) {
        nrv::Span ks{}, vs{};
        if (nrv::struct_field(c.t, en, e, 0, &ks) != 1 || nrv::struct_field(c.t, en, e, 1, &vs) != 1) return -2;
        nrv::LeafView kl{};
        if (!nrv::leaf_view(c.t, kn, ks, &kl) || kl.is_null) return -2;
        if (!first) o->put(',');
        int r = put_key(c, kn, kl, o);
        if (r != 1) return r;
        o->put(':');
        if (is_null(c.t.nodes[vn], vs)) { o->put("null", 4); continue; }
        r = value(c, vn, vs, o, depth + 1);
        if (r != 1) return r;
    }
    o->put('}');
    return 1;
}

// A Variant-annotated {metadata, value} struct node (G2CHK-984): the variant's own JSON.
inline int put_variant(const Tree& c, uint32_t ni, nrv::Span v, Out* o) noexcept {
    const nrv::TypeNode& nd = c.t.nodes[ni];
    if (c.variant_json == nullptr || nd.n_children != 2) return -2;
    nrv::Span ms{}, vs{};
    nrv::LeafView ml{}, vl{};
    if (nrv::struct_field(c.t, ni, v, 0, &ms) != 1 || nrv::struct_field(c.t, ni, v, 1, &vs) != 1 ||
        !nrv::leaf_view(c.t, nd.first_child, ms, &ml) || !nrv::leaf_view(c.t, nd.first_child + 1u, vs, &vl) ||
        ml.is_null || vl.is_null) return -2;
    return c.variant_json(ml.bytes, ml.len, vl.bytes, vl.len, o, c.tz);
}

inline int value(const Tree& c, uint32_t ni, nrv::Span v, Out* o, uint32_t depth) noexcept {
    assert(ni < c.t.n);
    if (depth > nrv::kMaxDepth + 1u) return -2;
    const nrv::TypeNode& nd = c.t.nodes[ni];
    if (nd.type == BoltType::Struct && nd.logical == BoltLogical::Variant) return put_variant(c, ni, v, o);
    if (nd.type == BoltType::Struct) return put_struct(c, ni, v, o, depth);
    if (nd.type == BoltType::List) return put_list(c, ni, v, o, depth);
    if (nd.type == BoltType::Map) return put_map(c, ni, v, o, depth);
    nrv::LeafView lv{};
    if (!nrv::leaf_view(c.t, ni, v, &lv)) return -2;
    if (lv.is_null) { o->put("null", 4); return 1; }
    return put_leaf(c, ni, lv, o);
}

}  // namespace tojd

// to_json of the value `v` of node `root`. 1 ok, 0 the value is NULL, -1 output budget, -2 unsupported value
// (a type or a year this kernel does not render).
inline int to_json(const Tree& c, uint32_t root, nrv::Span v, Out* o) noexcept {
    assert(o != nullptr && root < c.t.n);
    if (tojd::is_null(c.t.nodes[root], v)) return 0;
    const int r = tojd::value(c, root, v, o, 0);
    if (o->over) return -1;
    return r;
}

// ---- from_json --------------------------------------------------------------------------------------------------------

struct Region {
    uint8_t* base;
    uint32_t cap;
    uint32_t used;
    bool     over;
    uint8_t* take(uint32_t n) noexcept {
        if (n > cap - used) { over = true; return nullptr; }
        uint8_t* p = base + used;
        used += n;
        return p;
    }
    void u8(uint8_t v) noexcept { uint8_t* p = take(1); if (p != nullptr) *p = v; }
    void u32(uint32_t v) noexcept { uint8_t* p = take(4); if (p != nullptr) std::memcpy(p, &v, 4); }
};

enum class Cv : uint8_t { Ok = 0, Fail = 1, Err = 2, Budget = 3 };

namespace fromd {

inline Cv bad(const Region& r) noexcept { return r.over ? Cv::Budget : Cv::Err; }

// Skip to the end of the current value's container context: afterwards the parser depth is <= target.
inline bool skip_until(Parser* ps, uint32_t target) noexcept {
    for (uint64_t guard = 0; ps->depth > target; ++guard) {
        const Tok t = next(ps);
        if (t == Tok::Error || t == Tok::None || guard > uint64_t{ps->n} + 2u) return false;
    }
    return true;
}

inline Cv put_null_value(const nrv::TypeNode& nd, Region* r) noexcept {
    if (!nd.nullable) return Cv::Fail;
    r->u8(0);
    return r->over ? Cv::Budget : Cv::Ok;
}

inline Cv convert(const Tree& c, Parser* ps, uint32_t ni, Region* r, uint32_t depth) noexcept;

// An integer token's value (digits only; false when it does not fit int64).
inline bool token_int64(const Parser& ps, int64_t* out) noexcept {
    const char* t = reinterpret_cast<const char*>(ps.s + ps.tb);
    const uint32_t len = ps.te - ps.tb;
    uint32_t i = 0;
    const bool neg = t[0] == '-';
    if (neg) ++i;
    uint64_t v = 0;
    for (; i < len; ++i) {
        const uint64_t d = static_cast<uint64_t>(t[i] - '0');
        if (v > (UINT64_MAX - d) / 10u) return false;
        v = v * 10u + d;
    }
    if (neg) { if (v > uint64_t{1} << 63) return false; *out = static_cast<int64_t>(0 - v); return true; }
    if (v > static_cast<uint64_t>(INT64_MAX)) return false;
    *out = static_cast<int64_t>(v);
    return true;
}

// Java's Character.digit(cp, 10): the value of a Unicode decimal digit (general category Nd, Unicode 15), else -1.
inline int unicode_decimal_digit(uint32_t cp) noexcept {
    static const uint32_t kZero[] = {
        0x30, 0x660, 0x6F0, 0x7C0, 0x966, 0x9E6, 0xA66, 0xAE6, 0xB66, 0xBE6, 0xC66, 0xCE6, 0xD66, 0xDE6, 0xE50, 0xED0, 0xF20,
        0x1040, 0x1090, 0x17E0, 0x1810, 0x1946, 0x19D0, 0x1A80, 0x1A90, 0x1B50, 0x1BB0, 0x1C40, 0x1C50, 0xA620, 0xA8D0,
        0xA900, 0xA9D0, 0xA9F0, 0xAA50, 0xABF0, 0xFF10, 0x104A0, 0x10D30, 0x11066, 0x110F0, 0x11136, 0x111D0, 0x112F0,
        0x11450, 0x114D0, 0x11650, 0x116C0, 0x11730, 0x118E0, 0x11950, 0x11C50, 0x11D50, 0x11DA0, 0x11F50, 0x16A60,
        0x16AC0, 0x16B50, 0x1D7CE, 0x1D7D8, 0x1D7E2, 0x1D7EC, 0x1D7F6, 0x1E140, 0x1E2F0, 0x1E4F0, 0x1E950, 0x1FBF0};
    uint32_t lo = 0, hi = static_cast<uint32_t>(sizeof kZero / sizeof kZero[0]);
    while (lo < hi) {                                                // the last zero <= cp
        const uint32_t mid = (lo + hi) / 2;
        if (kZero[mid] <= cp) lo = mid + 1; else hi = mid;
    }
    return lo != 0 && cp - kZero[lo - 1] < 10u ? static_cast<int>(cp - kZero[lo - 1]) : -1;
}

// The next character of valid UTF-8 text as a decimal digit: its value and byte length; -1 when it is not one.
inline int text_digit(const char* t, uint32_t len, uint32_t i, uint32_t* width) noexcept {
    const uint8_t c = static_cast<uint8_t>(t[i]);
    if (c < 0x80) { *width = 1; return c >= '0' && c <= '9' ? c - '0' : -1; }
    uint32_t w = c >= 0xF0 ? 4u : c >= 0xE0 ? 3u : c >= 0xC0 ? 2u : 0u;
    if (w == 0 || i + w > len) return -1;
    uint32_t cp = c & (0xFFu >> (w + 1));
    for (uint32_t k = 1; k < w; ++k) cp = (cp << 6) | (static_cast<uint8_t>(t[i + k]) & 0x3Fu);
    *width = w;
    return unicode_decimal_digit(cp);
}

// BigDecimal text -> unscaled value at `scale` (HALF_UP), within `prec` digits. false: not a number / out of range.
inline bool decimal_from_text(const char* t, uint32_t len, uint32_t prec, uint32_t scale, __int128* out) noexcept {
    uint32_t i = 0;
    bool neg = false;
    if (i < len && (t[i] == '+' || t[i] == '-')) { neg = t[i] == '-'; ++i; }
    char dig[1100];
    uint32_t nd = 0;
    int64_t frac = 0;
    bool any = false, dot = false;
    for (uint32_t w = 1; i < len; i += w) {
        const int dv = text_digit(t, len, i, &w);
        if (dv >= 0) {
            any = true;
            if (nd < sizeof dig) dig[nd++] = static_cast<char>('0' + dv); else return false;
            if (dot) ++frac;
        } else if (t[i] == '.' && !dot) {
            dot = true;
            w = 1;
        } else break;
    }
    if (!any) return false;
    int64_t exp10 = 0;
    if (i < len && (t[i] == 'e' || t[i] == 'E')) {
        ++i;
        bool eneg = false;
        if (i < len && (t[i] == '+' || t[i] == '-')) { eneg = t[i] == '-'; ++i; }
        if (i >= len) return false;
        for (uint32_t w = 1; i < len; i += w) {
            const int dv = text_digit(t, len, i, &w);
            if (dv < 0) return false;
            exp10 = exp10 * 10 + dv;
            if (exp10 > 100000) return false;
        }
        if (eneg) exp10 = -exp10;
    }
    if (i != len) return false;
    uint32_t lead = 0;
    while (lead < nd && dig[lead] == '0') ++lead;
    unsigned __int128 u = 0;
    if (lead < nd) {
        const uint32_t L = nd - lead;
        const int64_t shift = exp10 - frac + static_cast<int64_t>(scale);   // unscaled = D * 10^shift
        if (shift >= 0) {
            if (static_cast<int64_t>(L) + shift > 38) return false;
            for (uint32_t k = 0; k < L; ++k) u = u * 10u + static_cast<unsigned>(dig[lead + k] - '0');
            for (int64_t k = 0; k < shift; ++k) u *= 10u;
        } else {
            const int64_t keep = static_cast<int64_t>(L) + shift;
            if (keep > 38) return false;
            if (keep > 0) {
                for (int64_t k = 0; k < keep; ++k) u = u * 10u + static_cast<unsigned>(dig[lead + static_cast<uint32_t>(k)] - '0');
                if (dig[lead + static_cast<uint32_t>(keep)] >= '5') ++u;
            } else if (keep == 0 && dig[lead] >= '5') {
                u = 1;
            }
        }
    }
    unsigned __int128 lim = 1;
    for (uint32_t k = 0; k < prec; ++k) lim *= 10u;
    if (u >= lim) return false;
    *out = neg ? -static_cast<__int128>(u) : static_cast<__int128>(u);
    return true;
}

inline bool special_float_text(const char* t, uint32_t len, double* d) noexcept {
    if (len == 3 && std::memcmp(t, "NaN", 3) == 0) { *d = std::nan(""); return true; }
    if (len == 8 && std::memcmp(t, "Infinity", 8) == 0) { *d = HUGE_VAL; return true; }
    if (len == 9 && std::memcmp(t, "+Infinity", 9) == 0) { *d = HUGE_VAL; return true; }
    if (len == 9 && std::memcmp(t, "-Infinity", 9) == 0) { *d = -HUGE_VAL; return true; }
    return false;
}

inline Cv leaf(const Tree& c, Parser* ps, uint32_t ni, Region* r) noexcept {
    const nrv::TypeNode& nd = c.t.nodes[ni];
    const Tok t = ps->cur;
    const char* tx = reinterpret_cast<const char*>(ps->s + ps->tb);
    const uint32_t tl = ps->te - ps->tb;
    if (nd.nullable) r->u8(1);
    switch (nd.type) {
        case BoltType::Bool: {
            if (t != Tok::True && t != Tok::False) return Cv::Fail;
            r->u8(t == Tok::True ? 1 : 0);
            break;
        }
        case BoltType::Int8: case BoltType::Int16: case BoltType::Int32: case BoltType::Int64: {
            int64_t v = 0;
            if (t != Tok::Int || !token_int64(*ps, &v)) return Cv::Fail;
            if (nd.type == BoltType::Int8) { if (v < -128 || v > 255) return Cv::Fail; v = static_cast<int8_t>(v); }
            else if (nd.type == BoltType::Int16) { if (v < -32768 || v > 32767) return Cv::Fail; }
            else if (nd.type == BoltType::Int32) { if (v < INT32_MIN || v > INT32_MAX) return Cv::Fail; }
            uint8_t* p = r->take(nd.fixed_bytes);
            if (p != nullptr) std::memcpy(p, &v, nd.fixed_bytes);
            break;
        }
        case BoltType::Float64: case BoltType::Float32: {
            double d = 0;
            if (t == Tok::Int || t == Tok::Float) d = token_double(*ps);
            else if (t == Tok::String && !ps->esc && special_float_text(tx, tl, &d)) {}
            else return Cv::Fail;
            if (nd.type == BoltType::Float32) {
                float f = static_cast<float>(d);
                if (t == Tok::Int || t == Tok::Float) {
                    if (d == d && !std::isinf(d)) { char b[1100]; if (tl < sizeof b) { std::memcpy(b, tx, tl); b[tl] = 0; f = std::strtof(b, nullptr); } }
                    if (t == Tok::Int && f == 0.0f) f = 0.0f;       // an integer token has no negative zero
                }
                uint8_t* p = r->take(4);
                if (p != nullptr) std::memcpy(p, &f, 4);
            } else {
                uint8_t* p = r->take(8);
                if (p != nullptr) std::memcpy(p, &d, 8);
            }
            break;
        }
        case BoltType::Decimal128: {
            __int128 u = 0;
            const uint32_t prec = c.meta[ni].precision != 0 ? c.meta[ni].precision : 38u;
            if (t == Tok::Int || (t == Tok::Float && (tx[0] != 'N' && tx[0] != 'I' && tx[1] != 'I'))) {
                if (!decimal_from_text(tx, tl, prec, nd.scale, &u)) return Cv::Fail;
            } else if (t == Tok::String && !ps->esc && tl >= 1) {
                if (!decimal_from_text(tx, tl, prec, nd.scale, &u)) return Cv::Fail;
            } else return Cv::Fail;
            uint8_t* p = r->take(16);
            if (p != nullptr) std::memcpy(p, &u, 16);
            break;
        }
        case BoltType::Utf8: {
            if (t == Tok::Null) return Cv::Fail;
            const uint32_t lp = r->used;
            r->u32(0);
            if (r->over) return Cv::Budget;
            Out o{reinterpret_cast<char*>(r->base) + r->used, r->cap - r->used, 0, false};
            if (t == Tok::String) {
                token_utf8(*ps, &o, false);
            } else {
                Gen g;
                gen_init(&g, &o);
                if (!copy_structure(ps, &g)) return o.over ? Cv::Budget : Cv::Err;
            }
            if (o.over) { r->over = true; return Cv::Budget; }
            std::memcpy(r->base + lp, &o.n, 4);
            r->used += o.n;
            break;
        }
        default: return Cv::Fail;
    }
    return r->over ? Cv::Budget : Cv::Ok;
}

inline Cv conv_struct(const Tree& c, Parser* ps, uint32_t ni, Region* r, uint32_t depth) noexcept {
    const nrv::TypeNode& nd = c.t.nodes[ni];
    const uint32_t S = r->used, nc = nd.n_children;
    uint32_t foff[nrv::kMaxNodes], flen[nrv::kMaxNodes];
    for (uint32_t k = 0; k < nc; ++k) foff[k] = ~0u;
    const uint32_t d_in = ps->depth;
    for (uint64_t guard = 0;; ++guard) {
        const Tok t = next(ps);
        if (t == Tok::Error || t == Tok::None) { r->used = S; return Cv::Err; }
        if (t == Tok::EndObject) break;
        if (guard > uint64_t{ps->n} + 2u) { r->used = S; return Cv::Err; }
        uint32_t k = ~0u;
        for (uint32_t j = nc; j > 0; --j) {                            // the last field of that name
            const NodeMeta& m = c.meta[nd.first_child + j - 1];
            if (token_equals(*ps, m.name, m.name_len)) { k = j - 1; break; }
        }
        const Tok v = next(ps);
        if (v == Tok::Error || v == Tok::None) { r->used = S; return Cv::Err; }
        if (k == ~0u) { if (!skip_children(ps)) { r->used = S; return Cv::Err; } continue; }
        const uint32_t start = r->used, d0 = ps->depth - ((v == Tok::StartObject || v == Tok::StartArray) ? 1u : 0u);
        const Cv st = convert(c, ps, nd.first_child + k, r, depth + 1);
        if (st == Cv::Ok) { foff[k] = start; flen[k] = r->used - start; continue; }
        r->used = start;
        if (st != Cv::Fail) { r->used = S; return st; }
        if (!skip_until(ps, d0)) { r->used = S; return Cv::Err; }
    }
    (void)d_in;
    const uint32_t base2 = r->used;
    if (nd.nullable) r->u8(1);
    for (uint32_t k = 0; k < nc; ++k) {
        if (foff[k] != ~0u) {
            uint8_t* p = r->take(flen[k]);
            if (p != nullptr) std::memmove(p, r->base + foff[k], flen[k]);
        } else {
            const Cv n = put_null_value(c.t.nodes[nd.first_child + k], r);
            if (n != Cv::Ok) { r->used = S; return n; }
        }
    }
    if (r->over) return Cv::Budget;
    const uint32_t len = r->used - base2;
    std::memmove(r->base + S, r->base + base2, len);
    r->used = S + len;
    return Cv::Ok;
}

inline Cv conv_list(const Tree& c, Parser* ps, uint32_t ni, Region* r, uint32_t depth, bool single) noexcept {
    const nrv::TypeNode& nd = c.t.nodes[ni];
    const uint32_t S = r->used, ei = nd.first_child, d_arr = ps->depth;
    if (nd.nullable) r->u8(1);
    const uint32_t cnt_at = r->used;
    r->u32(0);
    uint32_t count = 0;
    if (single) {                                                    // a root object stands for a one-element array
        const Cv st = convert(c, ps, ei, r, depth + 1);
        if (st != Cv::Ok) { r->used = S; return st; }
        count = 1;
    } else {
        for (uint64_t guard = 0;; ++guard) {
            const Tok t = next(ps);
            if (t == Tok::Error || t == Tok::None) { r->used = S; return Cv::Err; }
            if (t == Tok::EndArray) break;
            if (guard > uint64_t{ps->n} + 2u) { r->used = S; return Cv::Err; }
            const Cv st = convert(c, ps, ei, r, depth + 1);
            if (st == Cv::Ok) { ++count; continue; }
            r->used = S;
            if (st != Cv::Fail) return st;
            if (!skip_until(ps, d_arr - 1u)) return Cv::Err;
            return Cv::Fail;
        }
    }
    if (r->over) return Cv::Budget;
    std::memcpy(r->base + cnt_at, &count, 4);
    return Cv::Ok;
}

inline Cv conv_map(const Tree& c, Parser* ps, uint32_t ni, Region* r, uint32_t depth) noexcept {
    const nrv::TypeNode& nd = c.t.nodes[ni];
    const uint32_t S = r->used, en = nd.first_child, kn = c.t.nodes[en].first_child, vn = kn + 1u, d_obj = ps->depth;
    if (nd.nullable) r->u8(1);
    const uint32_t cnt_at = r->used;
    r->u32(0);
    uint32_t count = 0;
    for (uint64_t guard = 0;; ++guard) {
        const Tok t = next(ps);
        if (t == Tok::Error || t == Tok::None) { r->used = S; return Cv::Err; }
        if (t == Tok::EndObject) break;
        if (guard > uint64_t{ps->n} + 2u) { r->used = S; return Cv::Err; }
        if (c.t.nodes[en].nullable) r->u8(1);
        if (c.t.nodes[kn].nullable) r->u8(1);
        Out cnt{nullptr, 0, 0, false};
        token_utf8(*ps, &cnt, false);
        r->u32(cnt.n);
        uint8_t* kp = r->take(cnt.n);
        if (r->over) return Cv::Budget;
        if (kp != nullptr && cnt.n != 0) { Out kd{reinterpret_cast<char*>(kp), cnt.n, 0, false}; token_utf8(*ps, &kd, false); }
        const Tok v = next(ps);
        if (v == Tok::Error || v == Tok::None) { r->used = S; return Cv::Err; }
        const Cv st = convert(c, ps, vn, r, depth + 1);
        if (st == Cv::Ok) { ++count; continue; }
        r->used = S;
        if (st != Cv::Fail) return st;
        if (!skip_until(ps, d_obj - 1u)) return Cv::Err;
        return Cv::Fail;
    }
    if (r->over) return Cv::Budget;
    std::memcpy(r->base + cnt_at, &count, 4);
    return Cv::Ok;
}

inline Cv convert(const Tree& c, Parser* ps, uint32_t ni, Region* r, uint32_t depth) noexcept {
    assert(ni < c.t.n && ps != nullptr && r != nullptr);
    if (depth > nrv::kMaxDepth + 1u) return Cv::Fail;
    const nrv::TypeNode& nd = c.t.nodes[ni];
    if (ps->cur == Tok::Null) return put_null_value(nd, r);
    switch (nd.type) {
        case BoltType::Struct: if (ps->cur != Tok::StartObject) return Cv::Fail; return conv_struct(c, ps, ni, r, depth);
        case BoltType::List: if (ps->cur != Tok::StartArray) return Cv::Fail; return conv_list(c, ps, ni, r, depth, false);
        case BoltType::Map: if (ps->cur != Tok::StartObject) return Cv::Fail; return conv_map(c, ps, ni, r, depth);
        default: return leaf(c, ps, ni, r);
    }
}

// A struct whose every field is NULL (a malformed record).
inline Cv null_struct(const Tree& c, uint32_t ni, Region* r) noexcept {
    const nrv::TypeNode& nd = c.t.nodes[ni];
    if (nd.nullable) r->u8(1);
    for (uint32_t k = 0; k < nd.n_children; ++k) {
        const Cv n = put_null_value(c.t.nodes[nd.first_child + k], r);
        if (n != Cv::Ok) return n;
    }
    return r->over ? Cv::Budget : Cv::Ok;
}

}  // namespace fromd

// from_json(json, schema) under the default options (PERMISSIVE). The value is encoded into region[0, *out_len).
// 1 ok, 0 NULL, -1 the region (a budget) is too small.
inline int from_json(const Tree& c, uint32_t root, const uint8_t* json, uint32_t len, uint8_t* region, uint32_t cap,
                     uint32_t* out_len) noexcept {
    assert(region != nullptr && out_len != nullptr && root < c.t.n);
    Region r{region, cap, 0, false};
    Parser ps;
    parser_init(&ps, json, len, false, true);
    const nrv::TypeNode& nd = c.t.nodes[root];
    const Tok t = next(&ps);
    if (t == Tok::None) return 0;                                    // no token: NULL
    Cv st = Cv::Err;
    if (t != Tok::Error) {
        if (nd.type == BoltType::Struct) {
            st = ps.cur == Tok::StartObject ? fromd::conv_struct(c, &ps, root, &r, 0) : Cv::Err;
        } else if (nd.type == BoltType::List) {
            const bool single = ps.cur == Tok::StartObject && c.t.nodes[nd.first_child].type == BoltType::Struct;
            st = ps.cur == Tok::Null ? Cv::Fail
                 : (ps.cur == Tok::StartArray || single) ? fromd::conv_list(c, &ps, root, &r, 0, single) : Cv::Fail;
        } else {
            st = ps.cur == Tok::Null ? Cv::Fail : fromd::convert(c, &ps, root, &r, 0);
        }
    }
    if (st == Cv::Budget || r.over) return -1;
    if (st == Cv::Ok) { *out_len = r.used; return 1; }
    if (nd.type != BoltType::Struct) return 0;
    r.used = 0;
    if (fromd::null_struct(c, root, &r) != Cv::Ok) return r.over ? -1 : 0;
    *out_len = r.used;
    return 1;
}

}  // namespace jsonspark
}  // namespace kernels
}  // namespace bolt
