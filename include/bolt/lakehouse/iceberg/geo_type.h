// bolt/lakehouse/iceberg/geo_type.h — Iceberg v3 geometry / geography type names (G2ICE-327).
//
// The grammar is Iceberg-Java's Types.fromTypeName (1.12), case-insensitive, whitespace-tolerant, parameters optionally quoted:
//   geometry                      geometry(<crs>)          <crs>: no ')' , non-empty
//   geography                     geography(<crs>)         geography(<crs>, <algorithm>)   <crs>: no ','
// <algorithm> is one of spherical / vincenty / thomas / andoyer / karney (EdgeAlgorithm; Parquet's
// EdgeInterpolationAlgorithm uses the same order, 0..4). An omitted crs is OGC:CRS84, an omitted algorithm spherical.
// The parsed crs is a view into the name (trimmed); `crs_len == 0` means the default.
//
// RULES: No exceptions. No RTTI. No heap. All fns noexcept. Bounded loops.

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

namespace bolt {
namespace lakehouse {
namespace iceberg {

enum class GeoKind : uint8_t { kNone = 0, kGeometry = 1, kGeography = 2 };

struct GeoTypeInfo {
    GeoKind     kind = GeoKind::kNone;
    uint8_t     algorithm = 0;        // geography: 0 spherical .. 4 karney
    uint32_t    crs_len = 0;          // 0: OGC:CRS84
    const char* crs = nullptr;        // into the parsed name
};

inline constexpr const char* kGeoDefaultCrs = "OGC:CRS84";
inline constexpr const char* kGeoAlgorithms[5] = {"spherical", "vincenty", "thomas", "andoyer", "karney"};

namespace geo_detail {

inline char lower(char c) noexcept { return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c; }

inline bool ieq(const char* a, uint32_t an, const char* b) noexcept {
    const uint32_t bn = static_cast<uint32_t>(std::strlen(b));
    if (an != bn) return false;
    for (uint32_t i = 0; i < an; ++i) if (lower(a[i]) != lower(b[i])) return false;   // bounded: an
    return true;
}

inline bool space(char c) noexcept { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }

inline void trim(const char** p, uint32_t* n) noexcept {
    while (*n > 0u && space(**p)) { ++*p; --*n; }
    while (*n > 0u && space((*p)[*n - 1u])) --*n;
}

// pyiceberg (0.12) writes the parameters quoted, geometry('srid:4326'); Iceberg-Java and the spec do not. A matching pair of
// quotes around a parameter is dropped, so both spellings name the same type.
inline void unquote(const char** p, uint32_t* n) noexcept {
    trim(p, n);
    if (*n >= 2u && ((**p == '\'' && (*p)[*n - 1u] == '\'') || (**p == '"' && (*p)[*n - 1u] == '"'))) { ++*p; *n -= 2u; }
}

}  // namespace geo_detail

// Is `n[0..len)` a geometry / geography type name? False (kind kNone) for any other name, and for a malformed
// geometry / geography one (an empty crs, an unknown algorithm, trailing text): the caller names the refusal.
inline bool geo_type_parse(const char* name, uint32_t len, GeoTypeInfo* out) noexcept {
    assert(out != nullptr);
    *out = GeoTypeInfo{};
    if (name == nullptr) return false;
    const char* p = name;
    uint32_t n = len;
    geo_detail::trim(&p, &n);
    GeoKind k = GeoKind::kNone;
    uint32_t kw = 0;
    if (n >= 8u && geo_detail::ieq(p, 8u, "geometry")) { k = GeoKind::kGeometry; kw = 8u; }
    else if (n >= 9u && geo_detail::ieq(p, 9u, "geography")) { k = GeoKind::kGeography; kw = 9u; }
    else return false;
    p += kw;
    n -= kw;
    geo_detail::trim(&p, &n);
    if (n == 0u) { out->kind = k; return true; }
    if (p[0] != '(' || p[n - 1u] != ')') return false;
    const char* in = p + 1;
    uint32_t in_n = n - 2u;
    const char* alg = nullptr;
    uint32_t alg_n = 0;
    for (uint32_t i = 0; i < in_n; ++i) {               // bounded: the parenthesised text
        if (in[i] == ')') return false;
        if (in[i] == ',') {
            if (k == GeoKind::kGeometry || alg != nullptr) return false;
            alg = in + i + 1u;
            alg_n = in_n - i - 1u;
            in_n = i;
            break;
        }
    }
    geo_detail::unquote(&in, &in_n);
    if (in_n == 0u) return false;                       // Java: "Invalid CRS: (empty string)"
    out->kind = k;
    if (!geo_detail::ieq(in, in_n, kGeoDefaultCrs)) { out->crs = in; out->crs_len = in_n; }
    if (alg != nullptr) {
        geo_detail::unquote(&alg, &alg_n);
        uint32_t a = 0;
        while (a < 5u && !geo_detail::ieq(alg, alg_n, kGeoAlgorithms[a])) ++a;   // bounded: 5
        if (a == 5u) { *out = GeoTypeInfo{}; return false; }
        out->algorithm = static_cast<uint8_t>(a);
    }
    return true;
}

// The canonical name: "geometry" / "geometry(<crs>)", "geography" / "geography(<crs>)" / "geography(<crs>, <algorithm>)".
// The default crs and algorithm are left out. False when it does not fit `cap` (NUL included).
inline bool geo_type_name(const GeoTypeInfo& g, char* out, uint32_t cap) noexcept {
    assert(out != nullptr && cap > 0u);
    if (g.kind == GeoKind::kNone || g.algorithm > 4u) return false;
    const char* kw = g.kind == GeoKind::kGeometry ? "geometry" : "geography";
    const bool alg = g.kind == GeoKind::kGeography && g.algorithm != 0u;
    const char* crs = g.crs_len != 0u ? g.crs : kGeoDefaultCrs;
    const uint32_t crs_n = g.crs_len != 0u ? g.crs_len : 9u;
    const uint32_t kwn = static_cast<uint32_t>(std::strlen(kw));
    uint64_t need = kwn + 1u;
    if (g.crs_len != 0u || alg) need += 2u + crs_n + (alg ? 2u + std::strlen(kGeoAlgorithms[g.algorithm]) : 0u);
    if (need > cap) return false;
    uint32_t o = 0;
    std::memcpy(out, kw, kwn);
    o += kwn;
    if (g.crs_len != 0u || alg) {
        out[o++] = '(';
        std::memcpy(out + o, crs, crs_n);
        o += crs_n;
        if (alg) {
            out[o++] = ',';
            out[o++] = ' ';
            const uint32_t an = static_cast<uint32_t>(std::strlen(kGeoAlgorithms[g.algorithm]));
            std::memcpy(out + o, kGeoAlgorithms[g.algorithm], an);
            o += an;
        }
        out[o++] = ')';
    }
    out[o] = '\0';
    assert(o + 1u == need);
    return true;
}

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
