// bolt/kernels/bolt_variant_spark_get.h — Spark variant_get paths and casts (G2CHK-963). Included by
// bolt_variant_spark.h. A path is Spark's VariantPathParser grammar: `$` then any of `.name`, `['name']`,
// `["name"]`, `[n]`. A sub-variant is a slice of the value bytes under the same metadata.
#pragma once

#include <cassert>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace bolt {
namespace kernels {
namespace variantspark {

enum class GetStatus : uint8_t { Found = 0, Missing, BadPath, Malformed };
enum class CastStatus : uint8_t { Ok = 0, Null, Invalid, Malformed };

namespace detail {

// Bytes of the value starting at v (0 when malformed).
inline uint64_t value_size(const uint8_t* v, uint64_t n) noexcept {
    if (n == 0) return 0;
    const uint8_t basic = v[0] & 3u, h = static_cast<uint8_t>(v[0] >> 2);
    if (basic == kShortString) return 1u + h;
    if (basic == kObject || basic == kArray) {
        const bool obj = basic == kObject;
        const bool large = obj ? ((h >> 4) & 1u) != 0 : ((h >> 2) & 1u) != 0;
        const uint32_t osize = (h & 3u) + 1u, idsize = obj ? ((h >> 2) & 3u) + 1u : 0u, nsize = large ? 4u : 1u;
        if (n < 1u + nsize) return 0;
        const uint64_t cnt = get_le(v + 1, nsize);
        const uint64_t hdr = 1u + nsize + cnt * idsize + (cnt + 1u) * osize;
        if (hdr > n) return 0;
        const uint64_t total = hdr + get_le(v + 1 + nsize + cnt * idsize + cnt * osize, osize);
        return total <= n ? total : 0;
    }
    switch (h) {
        case kNull: case kTrue: case kFalse: return 1;
        case kInt8: return 2;
        case kInt16: return 3;
        case kInt32: case kDate: case kFloat: return 5;
        case kInt64: case kDouble: case kTimestamp: case kTimestampNtz: case kTime:
        case kTimestampNanos: case kTimestampNtzNanos: return 9;
        case kDecimal4: return 6;
        case kDecimal8: return 10;
        case kDecimal16: return 18;
        case kUuid: return 17;
        case kBinary: case kLongString: return n >= 5 ? 5u + get_le(v + 1, 4) : 0;
        default: return 0;
    }
}

// The child of object `v` named key[0..klen) or element `idx` of array `v`.
inline GetStatus child(const uint8_t* v, uint64_t n, const uint8_t* m, uint32_t mn, const char* key, uint32_t klen,
                       int64_t idx, const uint8_t** out, uint64_t* out_n) noexcept {
    const uint8_t basic = v[0] & 3u, h = static_cast<uint8_t>(v[0] >> 2);
    const bool want_obj = key != nullptr;
    if ((want_obj && basic != kObject) || (!want_obj && basic != kArray)) return GetStatus::Missing;
    const bool obj = basic == kObject;
    const bool large = obj ? ((h >> 4) & 1u) != 0 : ((h >> 2) & 1u) != 0;
    const uint32_t osize = (h & 3u) + 1u, idsize = obj ? ((h >> 2) & 3u) + 1u : 0u, nsize = large ? 4u : 1u;
    const uint64_t cnt = get_le(v + 1, nsize);
    const uint8_t* ids = v + 1 + nsize;
    const uint8_t* offs = ids + cnt * idsize;
    const uint8_t* data = offs + (cnt + 1u) * osize;
    if (data > v + n) return GetStatus::Malformed;
    uint64_t pick = cnt;
    if (!obj) {
        if (idx < 0 || static_cast<uint64_t>(idx) >= cnt) return GetStatus::Missing;
        pick = static_cast<uint64_t>(idx);
    } else {
        const uint32_t ko = ((m[0] >> 6) & 3u) + 1u;
        const uint64_t nkeys = get_le(m + 1, ko);
        const uint8_t* koff = m + 1 + ko;
        const uint8_t* kb = koff + (nkeys + 1u) * ko;
        for (uint64_t i = 0; i < cnt; ++i) {                          // fields in key order: a linear probe is enough
            const uint64_t id = get_le(ids + i * idsize, idsize);
            if (id >= nkeys) return GetStatus::Malformed;
            const uint64_t a = get_le(koff + id * ko, ko), b = get_le(koff + (id + 1) * ko, ko);
            if (kb + b > m + mn || b < a) return GetStatus::Malformed;
            if (b - a == klen && std::memcmp(kb + a, key, klen) == 0) { pick = i; break; }
        }
        if (pick == cnt) return GetStatus::Missing;
    }
    const uint64_t off = get_le(offs + pick * osize, osize);
    const uint64_t left = static_cast<uint64_t>((v + n) - (data + off));
    if (data + off >= v + n) return GetStatus::Malformed;
    const uint64_t sz = value_size(data + off, left);
    if (sz == 0) return GetStatus::Malformed;
    *out = data + off;
    *out_n = sz;
    return GetStatus::Found;
}

}  // namespace detail

// Is path[0..plen) a valid Spark variant path?
inline bool path_valid(const char* path, uint32_t plen) noexcept;

// The sub-variant at `path` (a slice of value; same metadata).
inline GetStatus get_path(const uint8_t* value, uint32_t vn, const uint8_t* metadata, uint32_t mn, const char* path,
                          uint32_t plen, const uint8_t** out, uint32_t* out_n) noexcept {
    assert(out != nullptr && out_n != nullptr);
    assert(path != nullptr || plen == 0);
    if (plen == 0 || path[0] != '$') return GetStatus::BadPath;
    const uint8_t* v = value;
    uint64_t n = detail::value_size(value, vn);
    if (n == 0) return GetStatus::Malformed;
    bool missing = false;
    uint32_t i = 1;
    for (uint32_t guard = 0; i < plen && guard < plen; ++guard) {    // each step consumes >= 2 bytes
        const char* key = nullptr;
        uint32_t klen = 0;
        int64_t idx = -1;
        if (path[i] == '.') {
            const uint32_t b = ++i;
            while (i < plen && path[i] != '.' && path[i] != '[') ++i;
            if (i == b) return GetStatus::BadPath;
            key = path + b;
            klen = i - b;
        } else if (path[i] == '[') {
            ++i;
            if (i < plen && (path[i] == '\'' || path[i] == '"')) {
                const char q = path[i];
                const uint32_t b = ++i;
                while (i < plen && path[i] != q) ++i;
                if (i + 1 >= plen || path[i + 1] != ']') return GetStatus::BadPath;
                key = path + b;
                klen = i - b;
                i += 2;
            } else {
                const uint32_t b = i;
                int64_t x = 0;
                while (i < plen && path[i] >= '0' && path[i] <= '9' && i - b < 18) x = x * 10 + (path[i++] - '0');
                if (i == b || i >= plen || path[i] != ']') return GetStatus::BadPath;
                ++i;
                idx = x;
            }
        } else {
            return GetStatus::BadPath;
        }
        if (missing) continue;                                         // keep validating the rest of the path
        const uint8_t* c = nullptr;
        uint64_t cn = 0;
        const GetStatus st = detail::child(v, n, metadata, mn, key, klen, idx, &c, &cn);
        if (st == GetStatus::Malformed) return st;
        if (st == GetStatus::Missing) { missing = true; continue; }
        v = c;
        n = cn;
    }
    if (missing) return GetStatus::Missing;
    *out = v;
    *out_n = static_cast<uint32_t>(n);
    return GetStatus::Found;
}

inline bool path_valid(const char* path, uint32_t plen) noexcept {
    static const uint8_t kNullValue[1] = {0};
    static const uint8_t kEmptyMeta[3] = {1, 0, 0};
    const uint8_t* o = nullptr;
    uint32_t on = 0;
    return get_path(kNullValue, 1, kEmptyMeta, 3, path, plen, &o, &on) != GetStatus::BadPath;
}

// Spark's cast of a variant to an integral (truncating a decimal / double, parsing a string, a boolean as 0 / 1).
inline CastStatus to_long(const uint8_t* v, uint32_t n, int64_t* out) noexcept {
    assert(out != nullptr);
    if (n == 0) return CastStatus::Malformed;
    const uint8_t basic = v[0] & 3u, h = static_cast<uint8_t>(v[0] >> 2);
    const uint8_t* s = nullptr;
    uint64_t sn = 0;
    if (basic == kShortString) { s = v + 1; sn = h; }
    else if (basic != kPrimitive) return CastStatus::Invalid;
    else if (h == kLongString) { s = v + 5; sn = detail::get_le(v + 1, 4); }
    if (s != nullptr) {
        char buf[32];
        uint64_t b = 0, e = sn;
        while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\n')) ++b;
        while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\n')) --e;
        if (e == b || e - b >= sizeof buf) return CastStatus::Invalid;
        std::memcpy(buf, s + b, e - b);
        buf[e - b] = '\0';
        char* end = nullptr;
        errno = 0;
        const long long x = std::strtoll(buf, &end, 10);
        if (end != buf + (e - b) || errno == ERANGE) return CastStatus::Invalid;
        *out = x;
        return CastStatus::Ok;
    }
    switch (h) {
        case kNull: return CastStatus::Null;
        case kTrue: *out = 1; return CastStatus::Ok;
        case kFalse: *out = 0; return CastStatus::Ok;
        case kInt8: case kInt16: case kInt32: case kInt64: {
            const uint32_t k = h == kInt8 ? 1u : h == kInt16 ? 2u : h == kInt32 ? 4u : 8u;
            if (n < 1u + k) return CastStatus::Malformed;
            uint64_t u = detail::get_le(v + 1, k);
            if (k < 8 && (u >> (8u * k - 1u)) != 0) u |= ~uint64_t{0} << (8u * k);
            *out = static_cast<int64_t>(u);
            return CastStatus::Ok;
        }
        case kDouble: case kFloat: {
            double d = 0;
            if (h == kDouble) { const uint64_t b = detail::get_le(v + 1, 8); std::memcpy(&d, &b, 8); }
            else { const uint32_t b = static_cast<uint32_t>(detail::get_le(v + 1, 4)); float f; std::memcpy(&f, &b, 4); d = f; }
            if (!(d >= -9223372036854775808.0 && d < 9223372036854775808.0)) return CastStatus::Invalid;
            *out = static_cast<int64_t>(d);
            return CastStatus::Ok;
        }
        case kDecimal4: case kDecimal8: case kDecimal16: {
            const uint32_t k = h == kDecimal4 ? 4u : h == kDecimal8 ? 8u : 16u;
            if (n < 2u + k) return CastStatus::Malformed;
            unsigned __int128 u = 0;
            for (uint32_t i = 0; i < k; ++i) u |= static_cast<unsigned __int128>(v[2 + i]) << (8u * i);
            if (k < 16 && (v[1 + k] & 0x80u) != 0) u |= ~static_cast<unsigned __int128>(0) << (8u * k);
            __int128 x = static_cast<__int128>(u);
            for (uint32_t sc = 0; sc < v[1]; ++sc) x /= 10;              // truncate toward zero
            if (x > INT64_MAX || x < INT64_MIN) return CastStatus::Invalid;
            *out = static_cast<int64_t>(x);
            return CastStatus::Ok;
        }
        default: return CastStatus::Invalid;
    }
}

// The text of a string variant (short or long), else false.
inline bool string_of(const uint8_t* v, uint32_t n, const uint8_t** s, uint32_t* sn) noexcept {
    if (n == 0) return false;
    const uint8_t basic = v[0] & 3u, h = static_cast<uint8_t>(v[0] >> 2);
    if (basic == kShortString && 1u + h <= n) { *s = v + 1; *sn = h; return true; }
    if (basic == kPrimitive && h == kLongString && n >= 5) {
        const uint64_t len = detail::get_le(v + 1, 4);
        if (5u + len > n) return false;
        *s = v + 5;
        *sn = static_cast<uint32_t>(len);
        return true;
    }
    return false;
}

// Spark's cast of a variant to a double (any number, a numeric string).
inline CastStatus to_double(const uint8_t* v, uint32_t n, double* out) noexcept {
    assert(out != nullptr);
    if (n == 0) return CastStatus::Malformed;
    const uint8_t basic = v[0] & 3u, h = static_cast<uint8_t>(v[0] >> 2);
    const uint8_t* s = nullptr;
    uint32_t sn = 0;
    if (string_of(v, n, &s, &sn)) {
        char buf[64];
        uint32_t b = 0, e = sn;
        while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\n')) ++b;
        while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\n')) --e;
        if (e == b || e - b >= sizeof buf) return CastStatus::Invalid;
        std::memcpy(buf, s + b, e - b);
        buf[e - b] = '\0';
        char* end = nullptr;
        const double d = std::strtod(buf, &end);
        if (end != buf + (e - b)) return CastStatus::Invalid;
        *out = d;
        return CastStatus::Ok;
    }
    if (basic != kPrimitive) return CastStatus::Invalid;
    switch (h) {
        case kNull: return CastStatus::Null;
        case kDouble: { const uint64_t b = detail::get_le(v + 1, 8); std::memcpy(out, &b, 8); return CastStatus::Ok; }
        case kFloat: {
            const uint32_t b = static_cast<uint32_t>(detail::get_le(v + 1, 4));
            float f;
            std::memcpy(&f, &b, 4);
            *out = f;
            return CastStatus::Ok;
        }
        case kDecimal4: case kDecimal8: case kDecimal16: {
            char t[64];
            jsonspark::Out o{t, sizeof t, 0, false};
            const uint32_t k = h == kDecimal4 ? 4u : h == kDecimal8 ? 8u : 16u;
            if (n < 2u + k) return CastStatus::Malformed;
            detail::put_decimal(&o, v + 2, k, v[1]);
            t[o.n < sizeof t ? o.n : sizeof t - 1] = '\0';
            *out = std::strtod(t, nullptr);
            return CastStatus::Ok;
        }
        default: {
            int64_t x = 0;
            const CastStatus st = to_long(v, n, &x);
            if (st == CastStatus::Ok) *out = static_cast<double>(x);
            return st;
        }
    }
}

// Spark's cast of a variant to a boolean (a boolean, a number != 0, "true" / "false" text).
inline CastStatus to_bool(const uint8_t* v, uint32_t n, bool* out) noexcept {
    assert(out != nullptr);
    if (n == 0) return CastStatus::Malformed;
    const uint8_t h = static_cast<uint8_t>(v[0] >> 2);
    const uint8_t* s = nullptr;
    uint32_t sn = 0;
    if (string_of(v, n, &s, &sn)) {
        auto ieq = [&](const char* w) noexcept {
            const uint32_t wl = static_cast<uint32_t>(std::strlen(w));
            if (wl != sn) return false;
            for (uint32_t i = 0; i < wl; ++i) if ((s[i] | 0x20) != w[i]) return false;
            return true;
        };
        if (ieq("true") || ieq("t") || ieq("yes") || ieq("y") || ieq("1")) { *out = true; return CastStatus::Ok; }
        if (ieq("false") || ieq("f") || ieq("no") || ieq("n") || ieq("0")) { *out = false; return CastStatus::Ok; }
        return CastStatus::Invalid;
    }
    if ((v[0] & 3u) != kPrimitive) return CastStatus::Invalid;
    if (h == kNull) return CastStatus::Null;
    if (h == kTrue || h == kFalse) { *out = h == kTrue; return CastStatus::Ok; }
    double d = 0;
    const CastStatus st = to_double(v, n, &d);
    if (st == CastStatus::Ok) *out = d != 0.0;
    return st;
}

// ---- cast(x AS VARIANT): one primitive value under the empty metadata (01 00 00) ----------------------------------------
// Each writes the value bytes into out[0..cap) and returns their length (0 when cap is too small). The width
// follows the source type, as Spark's cast does: a TINYINT is int8, an INT is int32 whatever its value.
inline constexpr uint8_t kEmptyMetadata[3] = {1, 0, 0};

inline uint32_t encode_int(uint8_t* out, uint32_t cap, int64_t v, uint32_t width_bytes) noexcept {
    assert(width_bytes == 1 || width_bytes == 2 || width_bytes == 4 || width_bytes == 8);
    if (cap < 1u + width_bytes) return 0;
    out[0] = detail::prim(width_bytes == 1 ? kInt8 : width_bytes == 2 ? kInt16 : width_bytes == 4 ? kInt32 : kInt64);
    detail::put_le(out + 1, static_cast<uint64_t>(v), width_bytes);
    return 1u + width_bytes;
}
inline uint32_t encode_double(uint8_t* out, uint32_t cap, double d) noexcept {
    if (cap < 9) return 0;
    uint64_t b = 0;
    std::memcpy(&b, &d, 8);
    out[0] = detail::prim(kDouble);
    detail::put_le(out + 1, b, 8);
    return 9;
}
inline uint32_t encode_float(uint8_t* out, uint32_t cap, float f) noexcept {
    if (cap < 5) return 0;
    uint32_t b = 0;
    std::memcpy(&b, &f, 4);
    out[0] = detail::prim(kFloat);
    detail::put_le(out + 1, b, 4);
    return 5;
}
inline uint32_t encode_bool(uint8_t* out, uint32_t cap, bool v) noexcept {
    if (cap < 1) return 0;
    out[0] = detail::prim(v ? kTrue : kFalse);
    return 1;
}
inline uint32_t encode_null(uint8_t* out, uint32_t cap) noexcept {
    if (cap < 1) return 0;
    out[0] = detail::prim(kNull);
    return 1;
}
// type_id: kDate (days, 4 bytes), kTimestamp / kTimestampNtz (micros, 8 bytes).
inline uint32_t encode_temporal(uint8_t* out, uint32_t cap, uint8_t type_id, int64_t v) noexcept {
    const uint32_t w = type_id == kDate ? 4u : 8u;
    if (cap < 1u + w) return 0;
    out[0] = detail::prim(type_id);
    detail::put_le(out + 1, static_cast<uint64_t>(v), w);
    return 1u + w;
}
inline uint32_t encode_string(uint8_t* out, uint32_t cap, const uint8_t* s, uint32_t n) noexcept {
    const uint32_t hdr = n < 64 ? 1u : 5u;
    if (uint64_t{cap} < uint64_t{hdr} + n) return 0;
    if (n < 64) out[0] = static_cast<uint8_t>((n << 2) | kShortString);
    else { out[0] = detail::prim(kLongString); detail::put_le(out + 1, n, 4); }
    if (n != 0) std::memcpy(out + hdr, s, n);
    return hdr + n;
}
inline uint32_t encode_binary(uint8_t* out, uint32_t cap, const uint8_t* s, uint32_t n) noexcept {
    if (uint64_t{cap} < 5u + uint64_t{n}) return 0;
    out[0] = detail::prim(kBinary);
    detail::put_le(out + 1, n, 4);
    if (n != 0) std::memcpy(out + 5, s, n);
    return 5u + n;
}
// A DECIMAL(precision, scale) of unscaled two's-complement (hi, lo): decimal4 / 8 / 16 by the declared precision.
inline uint32_t encode_decimal(uint8_t* out, uint32_t cap, int64_t hi, uint64_t lo, uint32_t precision,
                               uint32_t scale) noexcept {
    const uint32_t k = precision <= 9 ? 4u : precision <= 18 ? 8u : 16u;
    if (cap < 2u + k) return 0;
    out[0] = detail::prim(k == 4 ? kDecimal4 : k == 8 ? kDecimal8 : kDecimal16);
    out[1] = static_cast<uint8_t>(scale);
    detail::put_le(out + 2, lo, k < 8 ? k : 8);
    if (k == 16) detail::put_le(out + 10, static_cast<uint64_t>(hi), 8);
    return 2u + k;
}

// Is the variant a variant null (not an SQL NULL)?
inline bool is_null(const uint8_t* v, uint32_t n) noexcept { return n != 0 && v[0] == 0; }

}  // namespace variantspark
}  // namespace kernels
}  // namespace bolt
