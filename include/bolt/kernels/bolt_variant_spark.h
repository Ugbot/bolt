// bolt/kernels/bolt_variant_spark.h — Spark 4 parse_json / to_json over the variant binary encoding
// (G2CHK-963). The encoding is Parquet's VariantEncoding.md, the same bytes Iceberg variant columns carry
// (bolt_parquet_variant.h reads them; BoltColumn::make_variant holds them as {metadata, value}).
//
// parse_json is Spark's VariantBuilder.parseJson over Jackson's default factory: the root value only
// (anything after it is not read), no single quotes / raw control characters / NaN, a lone surrogate
// becomes '?', duplicate object keys are an error. Numbers: an integer token is the smallest of
// int8 / int16 / int32 / int64, past int64 a decimal(<=38, 0), past that a double; a fraction without
// an exponent is decimal4 / 8 / 16 by precision (scale kept: 1.0 is 10 at scale 1), past 38 digits a
// double; an exponent is a double. Strings under 64 bytes are short strings. The metadata dictionary
// lists keys in first-seen order; an object's field ids and offsets are sorted by key, its values stay
// in input order (as VariantBuilder writes them).
//
// to_json is Spark's VariantUtil.toJson: keys in sorted order, Jackson string escaping, decimals as
// stripTrailingZeros().toPlainString(), doubles / floats as Java's toString with non-finite values
// quoted, dates / timestamps / binary / uuid as quoted text.
//
// Caller buffers only, no allocation; every bound reports Status::Budget, never truncates.
// Tiger Style: noexcept, bounded loops, >= 2 asserts per non-trivial function.
#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_hash.h"
#include "bolt/kernels/bolt_json_spark.h"
#include "bolt/kernels/bolt_u128.h"

namespace bolt {
namespace kernels {
namespace variantspark {

inline constexpr uint32_t kSizeLimit = 16u * 1024u * 1024u;   // Spark VARIANT_SIZE_LIMIT, value and metadata

enum class Status : uint8_t { Ok = 0, Malformed, DuplicateKey, TooLarge, Budget };

// Basic types (low 2 bits of a value header) and primitive type ids (Parquet VariantEncoding.md).
enum : uint8_t { kPrimitive = 0, kShortString = 1, kObject = 2, kArray = 3 };
enum : uint8_t {
    kNull = 0, kTrue = 1, kFalse = 2, kInt8 = 3, kInt16 = 4, kInt32 = 5, kInt64 = 6, kDouble = 7,
    kDecimal4 = 8, kDecimal8 = 9, kDecimal16 = 10, kDate = 11, kTimestamp = 12, kTimestampNtz = 13,
    kFloat = 14, kBinary = 15, kLongString = 16, kTime = 17, kTimestampNanos = 18, kTimestampNtzNanos = 19,
    kUuid = 20,
};

// Scratch for one parse_json call, sized by the caller once (all capacities are budgets).
struct Workspace {
    uint8_t*  key_bytes;  uint32_t key_bytes_cap;   // dictionary key text, first-seen order
    uint32_t* key_off;    uint32_t keys_cap;        // key_off[i] .. key_off[i + 1] (keys_cap + 1 entries)
    uint32_t* table;      uint32_t table_cap;       // open addressing, id + 1, power of two >= 2 * keys_cap
    uint32_t* ent_id;     uint32_t* ent_off;        // pending container entries (field id, value offset)
    uint32_t  ent_cap;
    uint32_t* frame_start;                          // per open container: value offset, first entry, object?
    uint32_t* frame_ent;
    uint8_t*  frame_obj;
    uint32_t  frame_cap;
};

struct Encoded {
    uint8_t* value;    uint32_t value_cap;    uint32_t value_n;
    uint8_t* metadata; uint32_t metadata_cap; uint32_t metadata_n;
};

namespace detail {

inline uint32_t int_size(uint64_t v) noexcept { return v <= 0xFFu ? 1u : v <= 0xFFFFu ? 2u : v <= 0xFFFFFFu ? 3u : 4u; }

inline void put_le(uint8_t* p, uint64_t v, uint32_t n) noexcept {
    for (uint32_t i = 0; i < n; ++i) p[i] = static_cast<uint8_t>(v >> (8u * i));
}
inline uint64_t get_le(const uint8_t* p, uint32_t n) noexcept {
    uint64_t v = 0;
    for (uint32_t i = 0; i < n; ++i) v |= uint64_t{p[i]} << (8u * i);
    return v;
}

struct Enc {
    Workspace* w;
    Encoded*   out;
    uint32_t   nkeys;
    uint32_t   kbytes;
    uint32_t   nent;
    uint32_t   nframe;
    Status     st;
};

inline bool room(Enc* e, uint32_t n) noexcept {
    if (e->out->value_n + uint64_t{n} > e->out->value_cap) { e->st = Status::Budget; return false; }
    if (e->out->value_n + uint64_t{n} > kSizeLimit) { e->st = Status::TooLarge; return false; }
    return true;
}
inline bool put_bytes(Enc* e, const void* p, uint32_t n) noexcept {
    if (!room(e, n)) return false;
    std::memcpy(e->out->value + e->out->value_n, p, n);
    e->out->value_n += n;
    return true;
}
inline bool put_byte(Enc* e, uint8_t b) noexcept { return put_bytes(e, &b, 1); }
inline uint8_t prim(uint8_t type_id) noexcept { return static_cast<uint8_t>(type_id << 2); }

// A new pending entry of the innermost container (its value starts now).
inline bool add_entry(Enc* e, uint32_t id) noexcept {
    if (e->nframe == 0) return true;
    if (e->nent >= e->w->ent_cap) { e->st = Status::Budget; return false; }
    e->w->ent_id[e->nent] = id;
    e->w->ent_off[e->nent] = e->out->value_n - e->w->frame_start[e->nframe - 1];
    ++e->nent;
    return true;
}

// The dictionary id of key[0..len), added on first sight.
inline bool key_id(Enc* e, const uint8_t* key, uint32_t len, uint32_t* id) noexcept {
    Workspace* w = e->w;
    assert(w->table_cap != 0 && (w->table_cap & (w->table_cap - 1)) == 0);
    const uint32_t mask = w->table_cap - 1;
    uint32_t h = static_cast<uint32_t>(bolt::hash_bytes(key, len)) & mask;
    for (uint32_t probe = 0; probe < w->table_cap; ++probe, h = (h + 1) & mask) {
        const uint32_t slot = w->table[h];
        if (slot == 0) break;
        const uint32_t k = slot - 1;
        const uint32_t kl = w->key_off[k + 1] - w->key_off[k];
        if (kl == len && std::memcmp(w->key_bytes + w->key_off[k], key, len) == 0) { *id = k; return true; }
    }
    if (e->nkeys >= w->keys_cap || e->kbytes + uint64_t{len} > w->key_bytes_cap || 2u * (e->nkeys + 1) > w->table_cap) {
        e->st = Status::Budget;
        return false;
    }
    if (e->kbytes + uint64_t{len} > kSizeLimit) { e->st = Status::TooLarge; return false; }
    std::memcpy(w->key_bytes + e->kbytes, key, len);
    w->key_off[e->nkeys] = e->kbytes;
    e->kbytes += len;
    w->key_off[e->nkeys + 1] = e->kbytes;
    w->table[h] = e->nkeys + 1;
    *id = e->nkeys++;
    return true;
}

inline bool append_long(Enc* e, int64_t v) noexcept {
    uint8_t b[9];
    uint32_t n = 0;
    if (v == static_cast<int8_t>(v)) { b[0] = prim(kInt8); n = 1; }
    else if (v == static_cast<int16_t>(v)) { b[0] = prim(kInt16); n = 2; }
    else if (v == static_cast<int32_t>(v)) { b[0] = prim(kInt32); n = 4; }
    else { b[0] = prim(kInt64); n = 8; }
    put_le(b + 1, static_cast<uint64_t>(v), n);
    return put_bytes(e, b, n + 1);
}

inline bool append_double(Enc* e, double d) noexcept {
    uint8_t b[9];
    b[0] = prim(kDouble);
    uint64_t bits = 0;
    std::memcpy(&bits, &d, 8);
    put_le(b + 1, bits, 8);
    return put_bytes(e, b, 9);
}

// BigDecimal(text) for text without an exponent: the 128-bit unscaled value, its scale and precision. False
// past 38 digits (Spark then appends a double).
inline bool parse_decimal(const char* t, uint32_t len, bool* neg, uint64_t* hi, uint64_t* lo, uint32_t* scale,
                          uint32_t* prec) noexcept {
    assert(t != nullptr && len > 0);
    uint32_t i = 0;
    *neg = t[0] == '-';
    if (*neg) ++i;
    u128::U128 u{};
    uint32_t digits = 0, frac = 0;
    bool dot = false, lead = true;
    for (; i < len; ++i) {
        const char c = t[i];
        if (c == '.') { dot = true; continue; }
        if (c < '0' || c > '9') return false;
        if (dot) ++frac;
        if (lead && c == '0') continue;
        lead = false;
        if (++digits > 38) return false;
        if (!u128::mul10_add(&u, static_cast<uint32_t>(c - '0'))) return false;
    }
    *scale = frac;
    *prec = digits == 0 ? 1u : digits;
    if (*scale > 38) return false;
    *hi = u.hi;
    *lo = u.lo;
    return true;
}

inline bool append_decimal(Enc* e, bool neg, uint64_t hi, uint64_t lo, uint32_t scale, uint32_t prec) noexcept {
    u128::U128 u{hi, lo};
    if (neg) u = u128::negate(u);                                    // two's complement of the unscaled value
    uint8_t b[18];
    uint32_t n = 16;
    if (scale <= 9 && prec <= 9) { b[0] = prim(kDecimal4); n = 4; }
    else if (scale <= 18 && prec <= 18) { b[0] = prim(kDecimal8); n = 8; }
    else { b[0] = prim(kDecimal16); n = 16; }
    b[1] = static_cast<uint8_t>(scale);
    put_le(b + 2, u.lo, n < 8 ? n : 8);
    if (n == 16) put_le(b + 10, u.hi, 8);
    return put_bytes(e, b, n + 2);
}

inline bool append_number(Enc* e, const jsonspark::Parser& p) noexcept {
    const char* t = reinterpret_cast<const char*>(p.s + p.tb);
    const uint32_t len = p.te - p.tb;
    bool exp = false;
    for (uint32_t i = 0; i < len; ++i) exp |= t[i] == 'e' || t[i] == 'E';
    if (p.cur == jsonspark::Tok::Int) {
        // getLongValue, else (BIG_INTEGER) a decimal, else a double.
        bool neg = false;
        uint64_t hi = 0, lo = 0;
        uint32_t scale = 0, prec = 0;
        if (parse_decimal(t, len, &neg, &hi, &lo, &scale, &prec)) {
            const uint64_t lim = neg ? (uint64_t{1} << 63) : (uint64_t{1} << 63) - 1u;
            if (hi == 0 && lo <= lim) return append_long(e, neg ? static_cast<int64_t>(0u - lo) : static_cast<int64_t>(lo));
            return append_decimal(e, neg, hi, lo, scale, prec);
        }
        return append_double(e, jsonspark::token_double(p));
    }
    if (!exp) {
        bool neg = false;
        uint64_t hi = 0, lo = 0;
        uint32_t scale = 0, prec = 0;
        if (parse_decimal(t, len, &neg, &hi, &lo, &scale, &prec)) return append_decimal(e, neg, hi, lo, scale, prec);
    }
    return append_double(e, jsonspark::token_double(p));
}

inline bool append_string(Enc* e, const uint8_t* s, uint32_t n) noexcept {
    if (n < 64) {
        if (!put_byte(e, static_cast<uint8_t>((n << 2) | kShortString))) return false;
    } else {
        uint8_t b[5] = {prim(kLongString), 0, 0, 0, 0};
        put_le(b + 1, n, 4);
        if (!put_bytes(e, b, 5)) return false;
    }
    return n == 0 || put_bytes(e, s, n);
}

// The current string token's text (escapes decoded, a lone surrogate as '?').
inline bool token_text(Enc* e, const jsonspark::Parser& p, const uint8_t** s, uint32_t* n, uint8_t* buf,
                       uint32_t cap) noexcept {
    if (!p.esc) { *s = p.s + p.tb; *n = p.te - p.tb; return true; }
    jsonspark::Out o{reinterpret_cast<char*>(buf), cap, 0, false};
    (void)jsonspark::token_utf8(p, &o, false);
    if (o.over) { e->st = Status::Budget; return false; }
    *s = buf;
    *n = o.n;
    return true;
}

// Insertion sort of the frame's entries by key text (objects are small; Spark sorts the same way, then rejects
// equal neighbours).
inline bool sort_fields(Enc* e, uint32_t first, uint32_t n) noexcept {
    Workspace* w = e->w;
    auto key_less = [w](uint32_t a, uint32_t b) noexcept {
        const uint32_t al = w->key_off[a + 1] - w->key_off[a], bl = w->key_off[b + 1] - w->key_off[b];
        const int c = std::memcmp(w->key_bytes + w->key_off[a], w->key_bytes + w->key_off[b], al < bl ? al : bl);
        return c < 0 || (c == 0 && al < bl);
    };
    for (uint32_t i = 1; i < n; ++i) {
        const uint32_t id = w->ent_id[first + i], off = w->ent_off[first + i];
        uint32_t j = i;
        while (j > 0 && key_less(id, w->ent_id[first + j - 1])) {
            w->ent_id[first + j] = w->ent_id[first + j - 1];
            w->ent_off[first + j] = w->ent_off[first + j - 1];
            --j;
        }
        w->ent_id[first + j] = id;
        w->ent_off[first + j] = off;
    }
    for (uint32_t i = 1; i < n; ++i) {
        if (w->ent_id[first + i] == w->ent_id[first + i - 1]) { e->st = Status::DuplicateKey; return false; }
    }
    return true;
}

// Close the innermost container: move its values right and write its header in front.
inline bool finish_container(Enc* e) noexcept {
    assert(e->nframe > 0);
    Workspace* w = e->w;
    const uint32_t f = e->nframe - 1;
    const uint32_t start = w->frame_start[f], first = w->frame_ent[f];
    const bool obj = w->frame_obj[f] != 0;
    const uint32_t n = e->nent - first;
    const uint32_t data = e->out->value_n - start;
    if (obj && !sort_fields(e, first, n)) return false;
    const bool large = n > 0xFFu;
    const uint32_t nsize = large ? 4u : 1u;
    const uint32_t osize = int_size(data);
    uint32_t idsize = 0;
    if (obj) {
        uint32_t maxid = 0;
        for (uint32_t i = 0; i < n; ++i) maxid = w->ent_id[first + i] > maxid ? w->ent_id[first + i] : maxid;
        idsize = int_size(maxid);
    }
    const uint32_t hdr = 1u + nsize + n * idsize + (n + 1u) * osize;
    if (!room(e, hdr)) return false;
    uint8_t* base = e->out->value + start;
    std::memmove(base + hdr, base, data);
    uint8_t* q = base;
    *q++ = obj ? static_cast<uint8_t>((((large ? 1u : 0u) << 4 | (idsize - 1u) << 2 | (osize - 1u)) << 2) | kObject)
               : static_cast<uint8_t>((((large ? 1u : 0u) << 2 | (osize - 1u)) << 2) | kArray);
    put_le(q, n, nsize);
    q += nsize;
    for (uint32_t i = 0; obj && i < n; ++i, q += idsize) put_le(q, w->ent_id[first + i], idsize);
    for (uint32_t i = 0; i < n; ++i, q += osize) put_le(q, w->ent_off[first + i], osize);
    put_le(q, data, osize);
    e->out->value_n += hdr;
    e->nent = first;
    --e->nframe;
    return true;
}

inline bool open_container(Enc* e, bool obj, uint32_t id) noexcept {
    if (!add_entry(e, id)) return false;
    if (e->nframe >= e->w->frame_cap) { e->st = Status::Budget; return false; }
    e->w->frame_start[e->nframe] = e->out->value_n;
    e->w->frame_ent[e->nframe] = e->nent;
    e->w->frame_obj[e->nframe] = obj ? 1 : 0;
    ++e->nframe;
    return true;
}

inline bool write_metadata(Enc* e) noexcept {
    const uint32_t nk = e->nkeys, bytes = e->kbytes;
    const uint32_t osize = int_size(nk > bytes ? nk : bytes);
    const uint64_t need = 1u + uint64_t{osize} * (nk + 2u) + bytes;
    if (need > e->out->metadata_cap) { e->st = Status::Budget; return false; }
    if (need > kSizeLimit) { e->st = Status::TooLarge; return false; }
    uint8_t* q = e->out->metadata;
    *q++ = static_cast<uint8_t>(1u | ((osize - 1u) << 6));          // version 1, keys not flagged sorted
    put_le(q, nk, osize);
    q += osize;
    for (uint32_t i = 0; i <= nk; ++i, q += osize) put_le(q, e->w->key_off[i], osize);
    if (bytes != 0) std::memcpy(q, e->w->key_bytes, bytes);
    e->out->metadata_n = static_cast<uint32_t>(need);
    return true;
}

}  // namespace detail

// Spark parse_json(json): the root value's variant encoding into `out`. `text_buf` (cap >= json length) decodes
// escaped strings. Status::Malformed is Spark's MALFORMED_RECORD_IN_PARSING (try_parse_json: NULL).
inline Status parse_json(const uint8_t* json, uint32_t len, Workspace* w, Encoded* out, uint8_t* text_buf,
                         uint32_t text_cap) noexcept {
    assert(w != nullptr && out != nullptr && (json != nullptr || len == 0));
    assert(w->table_cap != 0 && w->frame_cap != 0);
    using jsonspark::Tok;
    std::memset(w->table, 0, sizeof(uint32_t) * w->table_cap);
    w->key_off[0] = 0;
    out->value_n = 0;
    out->metadata_n = 0;
    detail::Enc e{w, out, 0, 0, 0, 0, Status::Ok};
    jsonspark::Parser p;
    jsonspark::parser_init(&p, json, len, /*ctrl_ok=*/false, /*nan_ok=*/false);
    uint32_t field = UINT32_MAX;                                      // the pending field id inside an object
    for (uint64_t guard = 0; guard <= uint64_t{len} + 1u; ++guard) {
        const Tok t = jsonspark::next(&p);
        if (t == Tok::Error || t == Tok::None) return t == Tok::None && guard > 0 ? e.st : Status::Malformed;
        if ((t == Tok::String || t == Tok::FieldName) && p.quote == '\'') return Status::Malformed;
        const uint32_t id = field;
        field = UINT32_MAX;
        bool ok = true;
        switch (t) {
            case Tok::FieldName: {
                const uint8_t* s = nullptr;
                uint32_t n = 0;
                ok = detail::token_text(&e, p, &s, &n, text_buf, text_cap) && detail::key_id(&e, s, n, &field);
                break;
            }
            case Tok::StartObject: case Tok::StartArray:
                ok = detail::open_container(&e, t == Tok::StartObject, id);
                break;
            case Tok::EndObject: case Tok::EndArray:
                ok = detail::finish_container(&e);
                break;
            case Tok::String: {
                const uint8_t* s = nullptr;
                uint32_t n = 0;
                ok = detail::add_entry(&e, id) && detail::token_text(&e, p, &s, &n, text_buf, text_cap) &&
                     detail::append_string(&e, s, n);
                break;
            }
            case Tok::Int: case Tok::Float:
                ok = detail::add_entry(&e, id) && detail::append_number(&e, p);
                break;
            case Tok::True: case Tok::False: case Tok::Null:
                ok = detail::add_entry(&e, id) &&
                     detail::put_byte(&e, detail::prim(t == Tok::True ? kTrue : t == Tok::False ? kFalse : kNull));
                break;
            default:
                return Status::Malformed;
        }
        if (!ok) return e.st == Status::Ok ? Status::Malformed : e.st;
        if (e.nframe == 0 && t != Tok::FieldName) return detail::write_metadata(&e) ? Status::Ok : e.st;
    }
    return Status::Malformed;
}

}  // namespace variantspark
}  // namespace kernels
}  // namespace bolt

#include "bolt/kernels/bolt_variant_spark_json.h"
#include "bolt/kernels/bolt_variant_spark_get.h"
