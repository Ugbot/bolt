// bolt_variant_json.cpp — VARIANT -> JSON text, Spark 4 to_json form (G2ICE-382). See bolt_variant_json.h.
#include "bolt/ingest/bolt_variant_json.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/kernels/bolt_bignum_small.h"
#include "bolt/kernels/bolt_date.h"
#include "bolt/kernels/bolt_float_chars.h"

namespace bolt {
namespace ingest {
namespace variant {

namespace {

struct Out {
    char*    p;
    uint32_t n;
    uint32_t cap;
    bool     ok;
};

void put(Out* o, char c) noexcept {
    if (o->n >= o->cap) { o->ok = false; return; }
    o->p[o->n++] = c;
}

void put(Out* o, const char* s, uint32_t len) noexcept {
    assert(o != nullptr && (s != nullptr || len == 0));
    if (len > o->cap - o->n) { o->ok = false; return; }
    std::memcpy(o->p + o->n, s, len);
    o->n += len;
}

void put(Out* o, const char* s) noexcept { put(o, s, static_cast<uint32_t>(std::strlen(s))); }

uint64_t le(const uint8_t* p, uint32_t w) noexcept {
    assert(p != nullptr && w >= 1 && w <= 8);
    uint64_t v = 0;
    for (uint32_t i = 0; i < w; ++i) v |= static_cast<uint64_t>(p[i]) << (8u * i);   // bounded: 8
    return v;
}

struct Meta {
    const uint8_t* offsets;
    const uint8_t* bytes;
    uint32_t       width;
    uint32_t       count;
    uint32_t       bytes_len;
};

bool parse_meta(const uint8_t* m, uint32_t len, Meta* out) noexcept {
    assert(out != nullptr);
    if (m == nullptr || len < 1 || (m[0] & 0x0Fu) != 1u) return false;    // version 1
    out->width = ((m[0] >> 6) & 3u) + 1u;
    if (len < 1u + out->width) return false;
    const uint64_t count = le(m + 1, out->width);
    const uint64_t head = 1u + uint64_t{out->width} * (count + 2u);
    if (head > len) return false;
    out->count = static_cast<uint32_t>(count);
    out->offsets = m + 1 + out->width;
    out->bytes = m + head;
    out->bytes_len = static_cast<uint32_t>(len - head);
    return true;
}

bool meta_key(const Meta& md, uint64_t id, const char** s, uint32_t* n) noexcept {
    assert(s != nullptr && n != nullptr);
    if (id >= md.count) return false;
    const uint64_t a = le(md.offsets + id * md.width, md.width);
    const uint64_t b = le(md.offsets + (id + 1) * md.width, md.width);
    if (a > b || b > md.bytes_len) return false;
    *s = reinterpret_cast<const char*>(md.bytes + a);
    *n = static_cast<uint32_t>(b - a);
    return true;
}

// Jackson's string escapes.
void put_json_string(Out* o, const char* s, uint32_t n) noexcept {
    static const char* const kHex = "0123456789ABCDEF";
    put(o, '"');
    for (uint32_t i = 0; i < n && o->ok; ++i) {                          // bounded: n
        const unsigned char c = static_cast<unsigned char>(s[i]);
        switch (c) {
            case '"':  put(o, "\\\"", 2); continue;
            case '\\': put(o, "\\\\", 2); continue;
            case '\b': put(o, "\\b", 2); continue;
            case '\f': put(o, "\\f", 2); continue;
            case '\n': put(o, "\\n", 2); continue;
            case '\r': put(o, "\\r", 2); continue;
            case '\t': put(o, "\\t", 2); continue;
            default: break;
        }
        if (c < 0x20u) {
            const char e[6] = {'\\', 'u', '0', '0', kHex[c >> 4], kHex[c & 15u]};
            put(o, e, 6);
        } else {
            put(o, static_cast<char>(c));
        }
    }
    put(o, '"');
}

void put_u64(Out* o, uint64_t v) noexcept {
    char b[24];
    uint32_t i = 0;
    do { b[i++] = static_cast<char>('0' + v % 10u); v /= 10u; } while (v != 0 && i < sizeof(b));   // bounded: 20 digits
    while (i > 0) put(o, b[--i]);
}

void put_i64(Out* o, int64_t v) noexcept {
    if (v < 0) { put(o, '-'); put_u64(o, static_cast<uint64_t>(0) - static_cast<uint64_t>(v)); return; }
    put_u64(o, static_cast<uint64_t>(v));
}

// Java's Double.toString / Float.toString layout of shortest digits d (n of them), v = 0.d * 10^k.
void put_java_float(Out* o, bool neg, const char* d, uint32_t n, int32_t k, bool plain) noexcept {
    assert(d != nullptr && n >= 1);
    if (neg) put(o, '-');
    if (!plain) {                                                         // d1.d2..dnE(k-1)
        put(o, d[0]);
        put(o, '.');
        if (n > 1) put(o, d + 1, n - 1); else put(o, '0');
        put(o, 'E');
        put_i64(o, k - 1);
        return;
    }
    if (k <= 0) {
        put(o, "0.", 2);
        for (int32_t i = 0; i < -k; ++i) put(o, '0');                     // bounded: |v| >= 1e-3
        put(o, d, n);
        return;
    }
    const uint32_t ku = static_cast<uint32_t>(k);
    if (ku >= n) {
        put(o, d, n);
        for (uint32_t i = n; i < ku; ++i) put(o, '0');                    // bounded: |v| < 1e7
        put(o, ".0", 2);
        return;
    }
    put(o, d, ku);
    put(o, '.');
    put(o, d + ku, n - ku);
}

void put_double(Out* o, double v) noexcept {
    if (std::isnan(v)) { put(o, "\"NaN\""); return; }
    if (std::isinf(v)) { put(o, v > 0 ? "\"Infinity\"" : "\"-Infinity\""); return; }
    if (v == 0.0) { put(o, std::signbit(v) ? "-0.0" : "0.0"); return; }
    char d[24];
    int32_t k = 0;
    const double a = std::fabs(v);
    const uint32_t n = ::bolt::kernels::float_chars::detail::shortest_digits(a, d, &k);
    put_java_float(o, v < 0, d, n, k, a >= 1e-3 && a < 1e7);
}

// Float.toString: the shortest digits that read back as this float.
void put_float(Out* o, float f) noexcept {
    if (std::isnan(f)) { put(o, "\"NaN\""); return; }
    if (std::isinf(f)) { put(o, f > 0 ? "\"Infinity\"" : "\"-Infinity\""); return; }
    if (f == 0.0f) { put(o, std::signbit(f) ? "-0.0" : "0.0"); return; }
    const float a = std::fabs(f);
    char buf[40];
    for (int p = 1; p <= 9; ++p) {                                        // bounded: 9 digits always round-trip a float
        const int w = std::snprintf(buf, sizeof(buf), "%.*e", p - 1, static_cast<double>(a));
        if (w <= 0) { o->ok = false; return; }
        if (std::strtof(buf, nullptr) != a && p < 9) continue;
        char d[16];
        uint32_t n = 0;
        const char* e = std::strchr(buf, 'e');
        if (e == nullptr) { o->ok = false; return; }
        for (const char* q = buf; q < e; ++q) if (*q >= '0' && *q <= '9') d[n++] = *q;   // bounded: p digits
        while (n > 1 && d[n - 1] == '0') --n;
        const int32_t k = static_cast<int32_t>(std::strtol(e + 1, nullptr, 10)) + 1;
        put_java_float(o, f < 0, d, n, k, a >= 1e-3f && a < 1e7f);
        return;
    }
}

// A decimal of `bytes` little-endian two's-complement unscaled bytes and `scale`, trailing fraction zeros stripped.
void put_decimal(Out* o, const uint8_t* p, uint32_t bytes, uint32_t scale) noexcept {
    assert(p != nullptr && (bytes == 4 || bytes == 8 || bytes == 16));
    namespace bn = ::bolt::kernels::bignum;
    const bool neg = (p[bytes - 1] & 0x80u) != 0;
    uint8_t mag[16];
    uint32_t carry = 1;
    for (uint32_t i = 0; i < bytes; ++i) {                                // bounded: 16
        uint32_t b = neg ? static_cast<uint32_t>(static_cast<uint8_t>(~p[i])) + carry : p[i];
        if (neg) { carry = b >> 8; b &= 0xFFu; }
        mag[i] = static_cast<uint8_t>(b);
    }
    bn::Big v;
    bn::set_u64(&v, 0);
    for (uint32_t i = bytes; i-- > 0;) {                                  // bounded: 16
        bn::shl(&v, 8);
        bn::add_small(&v, mag[i]);
    }
    char digits[48];
    uint32_t n = 0;
    do { digits[n++] = static_cast<char>('0' + bn::div_small(&v, 10)); } while (!bn::is_zero(&v) && n < sizeof(digits));
    // digits are least significant first: strip fraction zeros
    uint32_t lo = 0;
    while (scale > 0 && lo + 1 < n && digits[lo] == '0') { ++lo; --scale; }
    if (scale > 0 && n == 1 && digits[0] == '0') scale = 0;               // 0.000 -> 0
    const bool zero = n - lo == 1 && digits[lo] == '0';
    if (neg && !zero) put(o, '-');
    const uint32_t len = n - lo;
    if (scale >= len) {
        put(o, "0.", 2);
        for (uint32_t i = len; i < scale; ++i) put(o, '0');
        for (uint32_t i = n; i-- > lo;) put(o, digits[i]);
        return;
    }
    for (uint32_t i = n; i-- > lo;) {                                     // bounded: 39 digits
        put(o, digits[i]);
        if (scale > 0 && i - lo == scale) put(o, '.');
    }
}

void put2(Out* o, uint32_t v) noexcept { put(o, static_cast<char>('0' + v / 10u % 10u)); put(o, static_cast<char>('0' + v % 10u)); }

void put_date(Out* o, int64_t days) noexcept {
    if (days < INT32_MIN || days > INT32_MAX) { o->ok = false; return; }
    const auto c = ::bolt::kernels::date::civil_from_days32(static_cast<int32_t>(days));
    char b[16];
    const int w = std::snprintf(b, sizeof(b), "%04d-", static_cast<int>(c.y));
    if (w <= 0) { o->ok = false; return; }
    put(o, b, static_cast<uint32_t>(w));
    put2(o, c.m); put(o, '-'); put2(o, c.d);
}

// HH:MM:SS[.f] of `units` within a day, `per_sec` units a second (1e6 micros, 1e9 nanos), fraction trimmed.
void put_time_of_day(Out* o, int64_t units, int64_t per_sec, uint32_t frac_digits) noexcept {
    const int64_t secs = units / per_sec;
    int64_t frac = units % per_sec;
    put2(o, static_cast<uint32_t>(secs / 3600)); put(o, ':');
    put2(o, static_cast<uint32_t>(secs / 60 % 60)); put(o, ':');
    put2(o, static_cast<uint32_t>(secs % 60));
    if (frac == 0) return;
    char f[12];
    for (uint32_t i = frac_digits; i-- > 0;) { f[i] = static_cast<char>('0' + frac % 10); frac /= 10; }   // bounded: 9
    uint32_t n = frac_digits;
    while (n > 0 && f[n - 1] == '0') --n;
    put(o, '.');
    put(o, f, n);
}

void put_timestamp(Out* o, int64_t v, bool nanos, bool tz) noexcept {
    const int64_t per_sec = nanos ? 1000000000 : 1000000;
    const int64_t per_day = per_sec * 86400;
    int64_t days = v / per_day, rem = v % per_day;
    if (rem < 0) { rem += per_day; --days; }
    put(o, '"');
    put_date(o, days);
    put(o, ' ');
    put_time_of_day(o, rem, per_sec, nanos ? 9u : 6u);
    if (tz) put(o, "+00:00", 6);
    put(o, '"');
}

void put_base64(Out* o, const uint8_t* p, uint32_t n) noexcept {
    static const char* const k = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    put(o, '"');
    for (uint32_t i = 0; i < n && o->ok; i += 3) {                        // bounded: n
        const uint32_t b0 = p[i], b1 = i + 1 < n ? p[i + 1] : 0u, b2 = i + 2 < n ? p[i + 2] : 0u;
        const char q[4] = {k[b0 >> 2], k[((b0 & 3u) << 4) | (b1 >> 4)], i + 1 < n ? k[((b1 & 15u) << 2) | (b2 >> 6)] : '=',
                           i + 2 < n ? k[b2 & 63u] : '='};
        put(o, q, 4);
    }
    put(o, '"');
}

void put_uuid(Out* o, const uint8_t* p) noexcept {
    static const char* const kHex = "0123456789abcdef";
    put(o, '"');
    for (uint32_t i = 0; i < 16; ++i) {                                   // bounded: 16
        if (i == 4 || i == 6 || i == 8 || i == 10) put(o, '-');
        put(o, kHex[p[i] >> 4]);
        put(o, kHex[p[i] & 15u]);
    }
    put(o, '"');
}

int64_t sext(const uint8_t* p, uint32_t w) noexcept {
    const uint64_t u = le(p, w);
    const uint32_t bits = 8u * w;
    return bits == 64 ? static_cast<int64_t>(u) : static_cast<int64_t>(u << (64 - bits)) >> (64 - bits);
}

bool put_value(Out* o, const Meta& md, const uint8_t* v, uint32_t len, uint32_t depth) noexcept;

bool put_primitive(Out* o, uint32_t id, const uint8_t* v, uint32_t len) noexcept {
    const uint8_t* b = v + 1;
    const uint32_t avail = len - 1;
    static const uint8_t kWidth[21] = {0, 0, 0, 1, 2, 4, 8, 8, 5, 9, 17, 4, 8, 8, 4, 4, 4, 8, 8, 8, 16};
    if (id > 20 || avail < kWidth[id]) return false;
    switch (id) {
        case 0: put(o, "null"); return true;
        case 1: put(o, "true"); return true;
        case 2: put(o, "false"); return true;
        case 3: case 4: case 5: case 6: put_i64(o, sext(b, kWidth[id])); return true;
        case 7: { double d; std::memcpy(&d, b, 8); put_double(o, d); return true; }
        case 8: case 9: case 10:
            if (b[0] > 38) return false;
            put_decimal(o, b + 1, kWidth[id] - 1u, b[0]);
            return true;
        case 11: put(o, '"'); put_date(o, sext(b, 4)); put(o, '"'); return true;
        case 12: case 13: put_timestamp(o, sext(b, 8), false, id == 12); return true;
        case 14: { float f; std::memcpy(&f, b, 4); put_float(o, f); return true; }
        case 15: case 16: {
            const uint64_t n = le(b, 4);
            if (n > avail - 4u) return false;
            if (id == 15) put_base64(o, b + 4, static_cast<uint32_t>(n));
            else put_json_string(o, reinterpret_cast<const char*>(b + 4), static_cast<uint32_t>(n));
            return true;
        }
        case 17: {
            const int64_t us = sext(b, 8);
            if (us < 0 || us >= INT64_C(86400000000)) return false;
            put(o, '"'); put_time_of_day(o, us, 1000000, 6u); put(o, '"');
            return true;
        }
        case 18: case 19: put_timestamp(o, sext(b, 8), true, id == 18); return true;
        case 20: put_uuid(o, b); return true;
        default: return false;
    }
}

bool put_object(Out* o, const Meta& md, const uint8_t* v, uint32_t len, uint32_t depth) noexcept {
    const uint32_t vh = v[0] >> 2;
    const uint32_t ow = (vh & 3u) + 1u, iw = ((vh >> 2) & 3u) + 1u, nw = ((vh >> 4) & 1u) ? 4u : 1u;
    if (len < 1u + nw) return false;
    const uint64_t num = le(v + 1, nw);
    const uint64_t ids = 1u + nw, offs = ids + num * iw, vals = offs + (num + 1u) * ow;
    if (vals > len) return false;
    put(o, '{');
    for (uint64_t i = 0; i < num && o->ok; ++i) {                         // bounded: num fits the value
        const char* k = nullptr;
        uint32_t kn = 0;
        if (!meta_key(md, le(v + ids + i * iw, iw), &k, &kn)) return false;
        const uint64_t at = vals + le(v + offs + i * ow, ow);
        if (at >= len) return false;
        if (i != 0) put(o, ',');
        put_json_string(o, k, kn);
        put(o, ':');
        if (!put_value(o, md, v + at, static_cast<uint32_t>(len - at), depth + 1)) return false;
    }
    put(o, '}');
    return true;
}

bool put_array(Out* o, const Meta& md, const uint8_t* v, uint32_t len, uint32_t depth) noexcept {
    const uint32_t vh = v[0] >> 2;
    const uint32_t ow = (vh & 3u) + 1u, nw = ((vh >> 2) & 1u) ? 4u : 1u;
    if (len < 1u + nw) return false;
    const uint64_t num = le(v + 1, nw);
    const uint64_t offs = 1u + nw, vals = offs + (num + 1u) * ow;
    if (vals > len) return false;
    put(o, '[');
    for (uint64_t i = 0; i < num && o->ok; ++i) {                         // bounded: num fits the value
        const uint64_t at = vals + le(v + offs + i * ow, ow);
        if (at >= len) return false;
        if (i != 0) put(o, ',');
        if (!put_value(o, md, v + at, static_cast<uint32_t>(len - at), depth + 1)) return false;
    }
    put(o, ']');
    return true;
}

bool put_value(Out* o, const Meta& md, const uint8_t* v, uint32_t len, uint32_t depth) noexcept {
    assert(o != nullptr);
    if (v == nullptr || len == 0 || depth > kVariantJsonMaxDepth) return false;
    const uint32_t basic = v[0] & 3u, vh = v[0] >> 2;
    switch (basic) {
        case 0: return put_primitive(o, vh, v, len);
        case 1:
            if (vh > len - 1u) return false;
            put_json_string(o, reinterpret_cast<const char*>(v + 1), vh);
            return true;
        case 2: return put_object(o, md, v, len, depth);
        default: return put_array(o, md, v, len, depth);
    }
}

}  // namespace

bool variant_to_json(const uint8_t* metadata, uint32_t metadata_len, const uint8_t* value, uint32_t value_len, char* out,
                     uint32_t cap, uint32_t* out_len) noexcept {
    assert(out != nullptr && out_len != nullptr);
    Meta md{};
    if (!parse_meta(metadata, metadata_len, &md)) return false;
    Out o{out, 0, cap, true};
    if (!put_value(&o, md, value, value_len, 0) || !o.ok) return false;
    *out_len = o.n;
    assert(o.n <= cap);
    return true;
}

bool variant_column_to_json(const BoltColumn& variant, int64_t rows, Arena* arena, BoltColumn* out) noexcept {
    assert(arena != nullptr && out != nullptr);
    assert(rows >= 0);
    const BoltColumn* m = variant.variant_metadata();
    const BoltColumn* v = variant.variant_value();
    if (m == nullptr || v == nullptr) return false;
    auto* offsets = arena->allocate_array<int32_t>(static_cast<size_t>(rows) + 1u);
    if (offsets == nullptr) return false;
    uint64_t bound = 64, total = 0;
    for (int64_t r = 0; r < rows; ++r) {                                  // bounded: rows
        const uint8_t* p = nullptr;
        int32_t l = 0;
        v->utf8_at_or_empty(r, &p, &l);
        const uint64_t b = variant_json_bound(static_cast<uint32_t>(l < 0 ? 0 : l));
        bound = b > bound ? b : bound;
    }
    if (bound > kVariantJsonColumnBytes) return false;
    char* scratch = static_cast<char*>(arena->allocate(static_cast<size_t>(bound), 1));
    if (scratch == nullptr) return false;
    // Pass 1: each row's text length (rendered into scratch); pass 2: render into the packed buffer.
    for (int64_t r = 0; r < rows; ++r) {                                  // bounded: rows
        offsets[r] = static_cast<int32_t>(total);
        if (variant.is_null(r)) continue;
        const uint8_t *mp = nullptr, *vp = nullptr;
        int32_t ml = 0, vl = 0;
        m->utf8_at_or_empty(r, &mp, &ml);
        v->utf8_at_or_empty(r, &vp, &vl);
        uint32_t n = 0;
        if (!variant_to_json(mp, static_cast<uint32_t>(ml), vp, static_cast<uint32_t>(vl), scratch,
                             static_cast<uint32_t>(bound), &n)) return false;
        total += n;
        if (total > kVariantJsonColumnBytes) return false;
    }
    offsets[rows] = static_cast<int32_t>(total);
    char* bytes = static_cast<char*>(arena->allocate(static_cast<size_t>(total) + 1u, 1));
    if (bytes == nullptr) return false;
    for (int64_t r = 0; r < rows; ++r) {                                  // bounded: rows
        if (variant.is_null(r)) continue;
        const uint8_t *mp = nullptr, *vp = nullptr;
        int32_t ml = 0, vl = 0;
        m->utf8_at_or_empty(r, &mp, &ml);
        v->utf8_at_or_empty(r, &vp, &vl);
        uint32_t n = 0;
        const uint32_t room = static_cast<uint32_t>(offsets[r + 1] - offsets[r]);
        if (!variant_to_json(mp, static_cast<uint32_t>(ml), vp, static_cast<uint32_t>(vl), bytes + offsets[r], room, &n) ||
            n != room) return false;
    }
    uint8_t* validity = nullptr;
    if (variant.validity != nullptr) {
        validity = static_cast<uint8_t*>(arena->allocate_zeroed((static_cast<size_t>(rows) + 7u) / 8u + 1u));
        if (validity == nullptr) return false;
        for (int64_t r = 0; r < rows; ++r)                                // bounded: rows
            if (!variant.is_null(r)) validity[r >> 3] = static_cast<uint8_t>(validity[r >> 3] | (1u << (r & 7)));
    }
    *out = BoltColumn::make_utf8_from_packed(bytes, validity, offsets, rows, arena);
    if (validity != nullptr) out->stats.all_valid = false;
    return out->type == BoltType::Utf8 && (rows == 0 || out->data != nullptr);
}

}  // namespace variant
}  // namespace ingest
}  // namespace bolt
