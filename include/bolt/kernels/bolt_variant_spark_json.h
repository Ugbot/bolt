// bolt/kernels/bolt_variant_spark_json.h — Spark 4 to_json(variant) (VariantUtil.toJson), G2CHK-963.
// Included by bolt_variant_spark.h. Objects print their fields in key order (the encoding's field-id order),
// strings and keys with Jackson's escaping, decimals as stripTrailingZeros().toPlainString(), doubles as
// Java's Double.toString with a non-finite value quoted, dates as quoted yyyy-MM-dd, binary as quoted base64,
// uuid as quoted text, floats as Java's Float.toString, a timestamp as quoted "yyyy-MM-dd HH:mm:ss[.ffffff]+HH:MM"
// in the session zone and a timestamp_ntz without the offset (fraction trailing zeros trimmed). TIME and the
// nanosecond timestamps (Spark 4.1 types) are refused (JsonStatus::Unsupported).
#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

#include "bolt/datetime/bolt_tz.h"
#include "bolt/kernels/bolt_json_spark_nested.h"

namespace bolt {
namespace kernels {
namespace variantspark {

enum class JsonStatus : uint8_t { Ok = 0, Malformed, Unsupported, Budget };

namespace detail {

struct Dec {
    const uint8_t* m;      // metadata
    uint32_t       mn;
    uint32_t       osize;  // metadata offset size
    uint32_t       nkeys;
    jsonspark::Out* o;
    tz::ZoneId     zone;   // session zone of TIMESTAMP values
};

inline void put_escaped(jsonspark::Out* o, const uint8_t* s, uint32_t n) noexcept {
    static const char hexd[] = "0123456789ABCDEF";
    o->put('"');
    for (uint32_t i = 0; i < n; ++i) {
        const uint8_t c = s[i];
        if (c == '"') o->put("\\\"", 2);
        else if (c == '\\') o->put("\\\\", 2);
        else if (c < 0x20) {
            switch (c) {
                case 8: o->put("\\b", 2); break;
                case 9: o->put("\\t", 2); break;
                case 10: o->put("\\n", 2); break;
                case 12: o->put("\\f", 2); break;
                case 13: o->put("\\r", 2); break;
                default: o->put("\\u00", 4); o->put(hexd[c >> 4]); o->put(hexd[c & 15]); break;
            }
        } else {
            o->put(static_cast<char>(c));
        }
    }
    o->put('"');
}

// unscaled (two's complement, `n` bytes) at `scale`, stripTrailingZeros().toPlainString().
inline void put_decimal(jsonspark::Out* o, const uint8_t* p, uint32_t n, uint32_t scale) noexcept {
    unsigned __int128 u = 0;
    for (uint32_t i = 0; i < n; ++i) u |= static_cast<unsigned __int128>(p[i]) << (8u * i);
    const bool neg = (p[n - 1] & 0x80u) != 0;
    if (neg) {
        if (n < 16) u |= ~static_cast<unsigned __int128>(0) << (8u * n);   // sign-extend
        u = ~u + 1u;
    }
    while (scale > 0 && u != 0 && u % 10u == 0) { u /= 10u; --scale; }
    if (u == 0) scale = 0;
    char dig[48];
    uint32_t nd = 0;
    do { dig[nd++] = static_cast<char>('0' + static_cast<int>(u % 10u)); u /= 10u; } while (u != 0 && nd < 40);
    if (neg) o->put('-');
    if (scale >= nd) {
        o->put("0.", 2);
        for (uint32_t z = nd; z < scale; ++z) o->put('0');
        for (uint32_t i = nd; i-- > 0;) o->put(dig[i]);
        return;
    }
    for (uint32_t i = nd; i-- > 0;) {
        o->put(dig[i]);
        if (i == scale && scale != 0) o->put('.');
    }
}

inline void put_base64(jsonspark::Out* o, const uint8_t* p, uint32_t n) noexcept {
    static const char t[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    o->put('"');
    for (uint32_t i = 0; i < n; i += 3) {
        const uint32_t v = (uint32_t{p[i]} << 16) | (i + 1 < n ? uint32_t{p[i + 1]} << 8 : 0u) | (i + 2 < n ? p[i + 2] : 0u);
        o->put(t[(v >> 18) & 63]);
        o->put(t[(v >> 12) & 63]);
        o->put(i + 1 < n ? t[(v >> 6) & 63] : '=');
        o->put(i + 2 < n ? t[v & 63] : '=');
    }
    o->put('"');
}

inline void put_date(jsonspark::Out* o, int32_t days) noexcept {
    // civil_from_days (proleptic Gregorian), as Spark prints a DATE.
    int64_t z = int64_t{days} + 719468;
    const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    const int64_t doe = z - era * 146097;
    const int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    int64_t y = yoe + era * 400;
    const int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const int64_t mp = (5 * doy + 2) / 153;
    const int64_t d = doy - (153 * mp + 2) / 5 + 1;
    const int64_t m = mp < 10 ? mp + 3 : mp - 9;
    if (m <= 2) ++y;
    char b[32];
    const int w = std::snprintf(b, sizeof b, "\"%04lld-%02lld-%02lld\"", static_cast<long long>(y),
                                static_cast<long long>(m), static_cast<long long>(d));
    o->put(b, static_cast<uint32_t>(w));
}

// A TIMESTAMP (UTC micros) in `zone` with its offset, or a TIMESTAMP_NTZ (wall micros) without one.
inline void put_timestamp(jsonspark::Out* o, int64_t micros, bool ntz, tz::ZoneId zone) noexcept {
    int64_t sec = micros >= 0 ? micros / 1000000 : -((-micros + 999999) / 1000000);
    int64_t frac = micros - sec * 1000000;
    int32_t off = 0;
    if (!ntz) {
        tz::OffsetSpan sp{};
        tz::offset_span(zone, sec, &sp);
        off = sp.off;
    }
    const int64_t local = sec + off;
    const int64_t days = local >= 0 ? local / 86400 : -((-local + 86399) / 86400);
    const int64_t tod = local - days * 86400;
    char date[32];
    jsonspark::Out tmp{date, sizeof date, 0, false};
    put_date(&tmp, static_cast<int32_t>(days));                      // "yyyy-MM-dd" with quotes
    o->put(date, tmp.n - 1);                                         // keep the opening quote, drop the closing one
    char b[64];
    int w = std::snprintf(b, sizeof b, " %02lld:%02lld:%02lld", static_cast<long long>(tod / 3600),
                          static_cast<long long>((tod / 60) % 60), static_cast<long long>(tod % 60));
    o->put(b, static_cast<uint32_t>(w));
    if (frac != 0) {
        w = std::snprintf(b, sizeof b, ".%06lld", static_cast<long long>(frac));
        while (w > 1 && b[w - 1] == '0') --w;
        o->put(b, static_cast<uint32_t>(w));
    }
    if (!ntz) {
        const int32_t a = off < 0 ? -off : off;
        w = std::snprintf(b, sizeof b, "%c%02d:%02d", off < 0 ? '-' : '+', a / 3600, (a / 60) % 60);
        o->put(b, static_cast<uint32_t>(w));
    }
    o->put('"');
}

inline JsonStatus value_json(const Dec& d, const uint8_t* v, uint32_t n, uint32_t depth) noexcept;

inline JsonStatus container_json(const Dec& d, const uint8_t* v, uint32_t n, uint32_t depth, bool obj) noexcept {
    const uint8_t vh = static_cast<uint8_t>(v[0] >> 2);
    const bool large = obj ? ((vh >> 4) & 1u) != 0 : ((vh >> 2) & 1u) != 0;
    const uint32_t osize = (vh & 3u) + 1u;
    const uint32_t idsize = obj ? ((vh >> 2) & 3u) + 1u : 0u;
    const uint32_t nsize = large ? 4u : 1u;
    if (n < 1u + nsize) return JsonStatus::Malformed;
    const uint64_t cnt = get_le(v + 1, nsize);
    const uint64_t hdr = 1u + nsize + cnt * idsize + (cnt + 1u) * osize;
    if (hdr > n) return JsonStatus::Malformed;
    const uint8_t* ids = v + 1 + nsize;
    const uint8_t* offs = ids + cnt * idsize;
    const uint8_t* data = v + hdr;
    const uint64_t dn = n - hdr;
    d.o->put(obj ? '{' : '[');
    for (uint64_t i = 0; i < cnt; ++i) {
        if (i != 0) d.o->put(',');
        if (obj) {
            const uint64_t id = get_le(ids + i * idsize, idsize);
            if (id >= d.nkeys) return JsonStatus::Malformed;
            const uint8_t* ko = d.m + 1 + d.osize;   // past the header and the dictionary size
            const uint64_t a = get_le(ko + id * d.osize, d.osize), b = get_le(ko + (id + 1) * d.osize, d.osize);
            const uint8_t* kb = ko + (uint64_t{d.nkeys} + 1u) * d.osize;
            if (b < a || kb + b > d.m + d.mn) return JsonStatus::Malformed;
            put_escaped(d.o, kb + a, static_cast<uint32_t>(b - a));
            d.o->put(':');
        }
        const uint64_t off = get_le(offs + i * osize, osize);
        if (off >= dn) return JsonStatus::Malformed;
        const JsonStatus s = value_json(d, data + off, static_cast<uint32_t>(dn - off), depth + 1u);
        if (s != JsonStatus::Ok) return s;
    }
    d.o->put(obj ? '}' : ']');
    return JsonStatus::Ok;
}

inline JsonStatus value_json(const Dec& d, const uint8_t* v, uint32_t n, uint32_t depth) noexcept {
    assert(d.o != nullptr);
    if (n == 0 || depth > jsonspark::kMaxNesting) return JsonStatus::Malformed;
    const uint8_t basic = v[0] & 3u, h = static_cast<uint8_t>(v[0] >> 2);
    if (basic == kShortString) {
        if (1u + h > n) return JsonStatus::Malformed;
        put_escaped(d.o, v + 1, h);
        return JsonStatus::Ok;
    }
    if (basic == kObject || basic == kArray) return container_json(d, v, n, depth, basic == kObject);
    auto need = [n](uint32_t k) noexcept { return 1u + k <= n; };
    char b[64];
    switch (h) {
        case kNull: d.o->put("null", 4); return JsonStatus::Ok;
        case kTrue: d.o->put("true", 4); return JsonStatus::Ok;
        case kFalse: d.o->put("false", 5); return JsonStatus::Ok;
        case kInt8: case kInt16: case kInt32: case kInt64: {
            const uint32_t k = h == kInt8 ? 1u : h == kInt16 ? 2u : h == kInt32 ? 4u : 8u;
            if (!need(k)) return JsonStatus::Malformed;
            uint64_t u = get_le(v + 1, k);
            if (k < 8 && (u >> (8u * k - 1u)) != 0) u |= ~uint64_t{0} << (8u * k);   // sign-extend
            const int w = std::snprintf(b, sizeof b, "%lld", static_cast<long long>(static_cast<int64_t>(u)));
            d.o->put(b, static_cast<uint32_t>(w));
            return JsonStatus::Ok;
        }
        case kDouble: {
            if (!need(8)) return JsonStatus::Malformed;
            const uint64_t bits = get_le(v + 1, 8);
            double x = 0;
            std::memcpy(&x, &bits, 8);
            const uint32_t w = jsonspark::java_double_chars(b, sizeof b, x);
            const bool finite = x == x && x != HUGE_VAL && x != -HUGE_VAL;
            if (!finite) d.o->put('"');
            d.o->put(b, w);
            if (!finite) d.o->put('"');
            return JsonStatus::Ok;
        }
        case kDecimal4: case kDecimal8: case kDecimal16: {
            const uint32_t k = h == kDecimal4 ? 4u : h == kDecimal8 ? 8u : 16u;
            if (!need(1u + k)) return JsonStatus::Malformed;
            put_decimal(d.o, v + 2, k, v[1]);
            return JsonStatus::Ok;
        }
        case kFloat: {
            if (!need(4)) return JsonStatus::Malformed;
            const uint32_t bits = static_cast<uint32_t>(get_le(v + 1, 4));
            float x = 0;
            std::memcpy(&x, &bits, 4);
            const uint32_t w = jsonspark::java_float_chars(b, sizeof b, x);
            const bool finite = x == x && x != HUGE_VALF && x != -HUGE_VALF;
            if (!finite) d.o->put('"');
            d.o->put(b, w);
            if (!finite) d.o->put('"');
            return JsonStatus::Ok;
        }
        case kTimestamp: case kTimestampNtz: {
            if (!need(8)) return JsonStatus::Malformed;
            put_timestamp(d.o, static_cast<int64_t>(get_le(v + 1, 8)), h == kTimestampNtz, d.zone);
            return JsonStatus::Ok;
        }
        case kDate: {
            if (!need(4)) return JsonStatus::Malformed;
            put_date(d.o, static_cast<int32_t>(static_cast<uint32_t>(get_le(v + 1, 4))));
            return JsonStatus::Ok;
        }
        case kBinary: case kLongString: {
            if (!need(4)) return JsonStatus::Malformed;
            const uint64_t len = get_le(v + 1, 4);
            if (5u + len > n) return JsonStatus::Malformed;
            if (h == kBinary) put_base64(d.o, v + 5, static_cast<uint32_t>(len));
            else put_escaped(d.o, v + 5, static_cast<uint32_t>(len));
            return JsonStatus::Ok;
        }
        case kUuid: {
            if (!need(16)) return JsonStatus::Malformed;
            static const char hx[] = "0123456789abcdef";
            d.o->put('"');
            for (uint32_t i = 0; i < 16; ++i) {
                if (i == 4 || i == 6 || i == 8 || i == 10) d.o->put('-');
                d.o->put(hx[v[1 + i] >> 4]);
                d.o->put(hx[v[1 + i] & 15]);
            }
            d.o->put('"');
            return JsonStatus::Ok;
        }
        default:
            return JsonStatus::Unsupported;                           // TIME, nanosecond timestamps (Spark 4.1)
    }
}

}  // namespace detail

// Spark to_json(variant) of {value, metadata} into `o`.
inline JsonStatus to_json(const uint8_t* value, uint32_t vn, const uint8_t* metadata, uint32_t mn,
                          jsonspark::Out* o, tz::ZoneId zone = 0) noexcept {
    assert(o != nullptr);
    assert(value != nullptr || vn == 0);
    if (mn < 1 || metadata == nullptr || (metadata[0] & 0x0Fu) != 1u) return JsonStatus::Malformed;
    detail::Dec d{metadata, mn, ((metadata[0] >> 6) & 3u) + 1u, 0, o, zone};
    if (mn < 1u + d.osize * 2u) return JsonStatus::Malformed;
    d.nkeys = static_cast<uint32_t>(detail::get_le(metadata + 1, d.osize));
    if (1u + uint64_t{d.osize} * (d.nkeys + 2u) > mn) return JsonStatus::Malformed;
    const JsonStatus s = detail::value_json(d, value, vn, 0);
    if (s == JsonStatus::Ok && o->over) return JsonStatus::Budget;
    return s;
}

}  // namespace variantspark
}  // namespace kernels
}  // namespace bolt
