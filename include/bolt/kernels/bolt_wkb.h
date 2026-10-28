// bolt_wkb.h — bounding boxes and geometry types of ISO WKB values, header-only.
//
// The Parquet GEOMETRY / GEOGRAPHY logical types and Iceberg's geometry / geography types store values as ISO WKB
// (OGC 06-103r4 / ISO 13249-3: a byte-order byte, a uint32 type = dimension * 1000 + base type, then the body).
// One fold serves both consumers:
//   - Iceberg data-file bounds (Iceberg-Java 1.12 GeometryBoundsBuilder): x / y ranges only; NaN ordinates (empty
//     points) are skipped; one malformed value makes the whole fold incomplete, and then no bounds are written;
//   - Parquet GeospatialStatistics: x / y / z / m ranges and the distinct top-level type codes.
// Validation follows GeometryBoundsBuilder exactly: base types 1..7, dimension groups 0..3 (XY, XYZ, XYM, XYZM); the
// members of a multi-geometry must be the member type, every member of any collection shares its parent's dimension;
// counts may not exceed the remaining bytes; a non-empty ring needs >= 4 points and must be closed on x / y; no
// trailing bytes. Nesting deeper than k_max_depth collections is malformed (bounded recursion).
//
// RULES: No exceptions. No RTTI. No heap. All fns noexcept. Bounded loops.

#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace bolt {
namespace kernels {
namespace wkb {

constexpr uint32_t k_max_depth = 64u;     // nested geometry collections
constexpr uint32_t k_max_types = 28u;     // 7 base types x 4 dimension groups

struct Range {
    double lo = 0.0;
    double hi = 0.0;
    bool   have = false;
};

struct Bounds {
    Range    x, y, z, m;
    bool     incomplete = false;          // a malformed value was folded: no bounds, no types
    uint32_t n_types = 0;
    uint32_t types[k_max_types]{};        // distinct ISO type codes of the folded values, ascending
};

namespace detail {

// Java's Math.min / Math.max order -0.0 below +0.0; NaN is never folded.
inline void range_add(Range* r, double v) noexcept {
    assert(r != nullptr);
    if (std::isnan(v)) return;
    if (!r->have) { r->lo = v; r->hi = v; r->have = true; return; }
    if (v < r->lo || (v == r->lo && std::signbit(v) && !std::signbit(r->lo))) r->lo = v;
    if (v > r->hi || (v == r->hi && !std::signbit(v) && std::signbit(r->hi))) r->hi = v;
}

struct Cursor {
    const uint8_t* p = nullptr;
    size_t         n = 0;
    size_t         at = 0;
    bool           le = true;
    size_t remaining() const noexcept { return n - at; }
};

inline bool u32(Cursor* c, uint32_t* out) noexcept {
    if (c->remaining() < 4u) return false;
    const uint8_t* b = c->p + c->at;
    *out = c->le ? (uint32_t{b[0]} | uint32_t{b[1]} << 8 | uint32_t{b[2]} << 16 | uint32_t{b[3]} << 24)
                 : (uint32_t{b[3]} | uint32_t{b[2]} << 8 | uint32_t{b[1]} << 16 | uint32_t{b[0]} << 24);
    c->at += 4u;
    return true;
}

inline double f64(Cursor* c) noexcept {                 // caller checked the bytes remain
    assert(c->remaining() >= 8u);
    uint8_t b[8];
    std::memcpy(b, c->p + c->at, 8u);
    if (!c->le) for (int i = 0; i < 4; ++i) { const uint8_t t = b[i]; b[i] = b[7 - i]; b[7 - i] = t; }
    double v;
    std::memcpy(&v, b, 8u);
    c->at += 8u;
    return v;
}

inline uint32_t dims_of(uint32_t group) noexcept { return group == 0u ? 2u : group == 3u ? 4u : 3u; }

// One coordinate of `dims` ordinates into the bounds; x / y returned for the ring-closure check.
inline void coord(Cursor* c, uint32_t group, Bounds* b, double* x, double* y) noexcept {
    *x = f64(c);
    *y = f64(c);
    range_add(&b->x, *x);
    range_add(&b->y, *y);
    if (group == 1u || group == 3u) range_add(&b->z, f64(c));
    if (group == 2u) range_add(&b->m, f64(c));
    if (group == 3u) range_add(&b->m, f64(c));
}

// A count, which may not exceed the bytes left (GeometryBoundsBuilder.readCount).
inline bool count(Cursor* c, uint32_t* out) noexcept {
    return u32(c, out) && uint64_t{*out} <= c->remaining();
}

inline bool sequence(Cursor* c, uint32_t group, Bounds* b) noexcept {
    uint32_t k = 0;
    if (!count(c, &k)) return false;
    const uint64_t need = uint64_t{k} * dims_of(group) * 8u;
    if (need > c->remaining()) return false;
    double x, y;
    for (uint32_t i = 0; i < k; ++i) coord(c, group, b, &x, &y);   // bounded: k <= remaining / 8
    return true;
}

inline bool ring(Cursor* c, uint32_t group, Bounds* b) noexcept {
    uint32_t k = 0;
    if (!count(c, &k)) return false;
    if (uint64_t{k} * dims_of(group) * 8u > c->remaining()) return false;
    if (k == 0u) return true;
    if (k < 4u) return false;
    double x0, y0, x, y;
    coord(c, group, b, &x0, &y0);
    for (uint32_t i = 1; i < k; ++i) coord(c, group, b, &x, &y);   // bounded: k
    return x0 == x && y0 == y;                         // closed on x / y (NaN is never closed, as in Java)
}

inline bool geometry(Cursor* c, uint32_t expect_base, int32_t expect_group, uint32_t depth, Bounds* b, uint32_t* type_out) noexcept;

inline bool collection(Cursor* c, uint32_t member_base, uint32_t group, uint32_t depth, Bounds* b) noexcept {
    uint32_t k = 0;
    if (!count(c, &k)) return false;
    for (uint32_t i = 0; i < k; ++i) {                 // bounded: each member consumes >= 5 bytes of the remaining
        uint32_t t = 0;
        if (!geometry(c, member_base, static_cast<int32_t>(group), depth + 1u, b, &t)) return false;
    }
    return true;
}

inline bool geometry(Cursor* c, uint32_t expect_base, int32_t expect_group, uint32_t depth, Bounds* b, uint32_t* type_out) noexcept {
    if (depth > k_max_depth || c->remaining() < 5u) return false;
    const uint8_t order = c->p[c->at++];
    if (order > 1u) return false;
    c->le = order == 1u;
    uint32_t code = 0;
    if (!u32(c, &code)) return false;
    const uint32_t group = code / 1000u, base = code % 1000u;
    if (base < 1u || base > 7u || group > 3u) return false;
    if (expect_base != 0u && base != expect_base) return false;
    if (expect_group >= 0 && group != static_cast<uint32_t>(expect_group)) return false;
    *type_out = code;
    switch (base) {
        case 1: {
            if (uint64_t{dims_of(group)} * 8u > c->remaining()) return false;
            double x, y;
            coord(c, group, b, &x, &y);
            return true;
        }
        case 2: return sequence(c, group, b);
        case 3: {
            uint32_t k = 0;
            if (!count(c, &k)) return false;
            for (uint32_t i = 0; i < k; ++i) if (!ring(c, group, b)) return false;   // bounded: each ring >= 4 bytes
            return true;
        }
        case 4: return collection(c, 1u, group, depth, b);
        case 5: return collection(c, 2u, group, depth, b);
        case 6: return collection(c, 3u, group, depth, b);
        default: return collection(c, 0u, group, depth, b);
    }
}

inline void note_type(Bounds* b, uint32_t code) noexcept {
    uint32_t i = 0;
    while (i < b->n_types && b->types[i] < code) ++i;  // bounded: k_max_types
    if (i < b->n_types && b->types[i] == code) return;
    if (b->n_types >= k_max_types) return;             // cannot happen: codes are 1..3007 over 28 values
    for (uint32_t j = b->n_types; j > i; --j) b->types[j] = b->types[j - 1u];
    b->types[i] = code;
    ++b->n_types;
}

}  // namespace detail

// Fold one WKB value. False (and `incomplete` set, every later value ignored) when it is malformed.
inline bool bounds_add(Bounds* b, const uint8_t* p, size_t n) noexcept {
    assert(b != nullptr);
    if (b->incomplete) return false;
    if (p == nullptr && n != 0u) { b->incomplete = true; return false; }
    Bounds t = *b;                                     // a malformed value leaves nothing behind
    detail::Cursor c{p, n, 0u, true};
    uint32_t code = 0;
    if (!detail::geometry(&c, 0u, -1, 0u, &t, &code) || c.remaining() != 0u) { b->incomplete = true; return false; }
    detail::note_type(&t, code);
    *b = t;
    return true;
}

// Iceberg's single-value encoding of a geometry bound (spec Appendix D): x, y as little-endian doubles. Lower =
// (xmin, ymin), upper = (xmax, ymax). False when the fold proves no bounds (incomplete, or no finite x / y).
inline bool iceberg_xy_bounds(const Bounds& b, uint8_t lower[16], uint8_t upper[16]) noexcept {
    if (b.incomplete || !b.x.have || !b.y.have) return false;
    const double lo[2] = {b.x.lo, b.y.lo}, hi[2] = {b.x.hi, b.y.hi};
    std::memcpy(lower, lo, 16u);                       // little-endian hosts (bolt's supported targets)
    std::memcpy(upper, hi, 16u);
    return true;
}

}  // namespace wkb
}  // namespace kernels
}  // namespace bolt
