// bolt/kernels/numeric_key.h — the value-canonical key contract.
//
// A hash/equality key is the VALUE, not the storage bits:
//   * Float64: -0.0 == +0.0, and every NaN is one value (the largest, as in
//     DuckDB, Postgres and Spark ordering).
//   * Int32 / Date32 widen by sign extension, so a key compares like the
//     number it is when mixed with Int64 lanes.
//   * NULL is a separate flag, never a reserved value.
//   * Anything wider than 8 bytes (Decimal128, Utf8 views) does not fit an
//     8-byte slot; slot8_pack refuses it so the caller takes a 16-byte or
//     content-addressed key instead of truncating.

#pragma once

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "bolt/bolt_port.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"

namespace bolt::kernels::numeric_key {

inline constexpr std::uint64_t kCanonQNaN = 0x7FF8000000000000ULL;

BOLT_FORCE_INLINE std::int64_t canon_f64_bits(std::int64_t raw) noexcept {
    constexpr std::uint64_t kSignMask = 0x8000000000000000ULL;
    constexpr std::uint64_t kExpMask  = 0x7FF0000000000000ULL;
    constexpr std::uint64_t kFracMask = 0x000FFFFFFFFFFFFFULL;
    const std::uint64_t b = static_cast<std::uint64_t>(raw);
    const bool is_nan = ((b & kExpMask) == kExpMask) && ((b & kFracMask) != 0);
    std::uint64_t out = (b == kSignMask) ? 0ULL : b;
    out = is_nan ? kCanonQNaN : out;
    return static_cast<std::int64_t>(out);
}

BOLT_FORCE_INLINE std::int64_t canon_f64_bits_of(double v) noexcept {
    std::int64_t raw = 0;
    static_assert(sizeof(raw) == sizeof(v), "ieee754 double is 8 bytes");
    std::memcpy(&raw, &v, sizeof(raw));
    return canon_f64_bits(raw);
}

BOLT_FORCE_INLINE double canon_f64(double v) noexcept {
    const std::int64_t b = canon_f64_bits_of(v);
    double out = 0.0;
    std::memcpy(&out, &b, sizeof(out));
    return out;
}

// Order-preserving signed key: -inf < ... < -0.0 == 0.0 < ... < inf < NaN.
BOLT_FORCE_INLINE std::int64_t f64_sortable_key(double v) noexcept {
    const std::uint64_t u = static_cast<std::uint64_t>(canon_f64_bits_of(v));
    const std::uint64_t mask =
        (static_cast<std::uint64_t>(-static_cast<std::int64_t>(u >> 63))) &
        0x7FFFFFFFFFFFFFFFULL;
    return static_cast<std::int64_t>(u ^ mask);
}

// SQL/Spark total order for MIN/MAX over doubles: NaN is the largest value.
BOLT_FORCE_INLINE bool f64_total_less(double a, double b) noexcept {
    return f64_sortable_key(a) < f64_sortable_key(b);
}

// Exact decimal literal (two's-complement 128-bit mantissa hi:lo, `scale`
// fractional digits) as the correctly rounded double. A mantissa within 2^53
// and a scale within 22 are both exact doubles, so one IEEE division rounds
// once; anything else goes through strtod on the decimal text. Literal-time
// only (not a per-row kernel).
inline double dec_lit_to_f64(std::int64_t hi, std::int64_t lo,
                             std::uint8_t scale) noexcept {
    assert(scale <= 38);
    const bool neg = hi < 0;
    std::uint64_t ulo = static_cast<std::uint64_t>(lo);
    std::uint64_t uhi = static_cast<std::uint64_t>(hi);
    if (neg) {
        ulo = ~ulo + 1ULL;
        uhi = ~uhi + ((ulo == 0ULL) ? 1ULL : 0ULL);
    }
    constexpr std::uint64_t k2p53 = 1ULL << 53;
    if (uhi == 0 && ulo <= k2p53 && scale <= 22) {
        double p = 1.0;
        for (std::uint8_t i = 0; i < scale; ++i) p *= 10.0;   // exact <= 1e22
        const double m = static_cast<double>(ulo);
        return neg ? -(m / p) : (m / p);
    }
    char digits[48];
    int nd = 0;
    for (int guard = 0; guard < 40 && (uhi != 0 || ulo != 0 || nd == 0); ++guard) {
        // 128-bit / 10 via 32-bit limbs.
        const std::uint64_t qh = uhi / 10, r1 = uhi % 10;
        const std::uint64_t mid = (r1 << 32) | (ulo >> 32);
        const std::uint64_t qm = mid / 10, r2 = mid % 10;
        const std::uint64_t low = (r2 << 32) | (ulo & 0xFFFFFFFFULL);
        const std::uint64_t ql = low / 10, r3 = low % 10;
        uhi = qh;
        ulo = (qm << 32) | ql;
        digits[nd++] = static_cast<char>('0' + r3);
    }
    char buf[64];
    int n = 0;
    if (neg) buf[n++] = '-';
    while (nd > 0) buf[n++] = digits[--nd];
    buf[n++] = 'e';
    buf[n++] = '-';
    if (scale >= 10) buf[n++] = static_cast<char>('0' + scale / 10);
    buf[n++] = static_cast<char>('0' + scale % 10);
    buf[n] = '\0';
    assert(n < static_cast<int>(sizeof(buf)));
    return std::strtod(buf, nullptr);
}

BOLT_FORCE_INLINE double dec_lit_to_f64(std::int64_t mantissa,
                                        std::uint8_t scale) noexcept {
    return dec_lit_to_f64(mantissa < 0 ? -1 : 0, mantissa, scale);
}

// Can a value of type `t` ride an 8-byte key slot exactly?
BOLT_FORCE_INLINE bool slot8_type_ok(BoltType t) noexcept {
    switch (t) {
        case BoltType::Bool:
        case BoltType::Int8:   case BoltType::Int16:
        case BoltType::Int32:  case BoltType::Int64:
        case BoltType::UInt8:  case BoltType::UInt16:
        case BoltType::UInt32: case BoltType::UInt64:
        case BoltType::Float32: case BoltType::Float64:
        case BoltType::Date32: case BoltType::Date64:
        case BoltType::Timestamp: case BoltType::Duration:
        case BoltType::Decimal64:
            return true;
        default:
            return false;
    }
}

// One value of type `t` stored at `p` (`width` bytes) as its canonical slot.
// The slot round-trips: writing its low `width` bytes back (or the double for
// Float32 promotion callers) reproduces the canonical value. Float64 keeps
// canonical IEEE bits rather than the sortable key for that reason.
BOLT_FORCE_INLINE bool slot8_from_bytes(BoltType t, const std::uint8_t* p,
                                        std::uint32_t width,
                                        std::int64_t* slot) noexcept {
    assert(p != nullptr && slot != nullptr);
    assert(width >= 1 && width <= 8);
    switch (t) {
        case BoltType::Int8: {
            std::int8_t v; std::memcpy(&v, p, 1); *slot = v; return true;
        }
        case BoltType::Int16: {
            std::int16_t v; std::memcpy(&v, p, 2); *slot = v; return true;
        }
        case BoltType::Int32: case BoltType::Date32: {
            std::int32_t v; std::memcpy(&v, p, 4); *slot = v; return true;
        }
        case BoltType::Float64: {
            std::int64_t v; std::memcpy(&v, p, 8);
            *slot = canon_f64_bits(v);
            return true;
        }
        case BoltType::Float32: {
            float f; std::memcpy(&f, p, 4);
            if (f == 0.0f) f = 0.0f;                       // -0.0f -> +0.0f
            if (f != f) {
                constexpr std::uint32_t kQ = 0x7FC00000u;  // one quiet NaN
                std::memcpy(&f, &kQ, 4);
            }
            std::uint32_t u; std::memcpy(&u, &f, 4);
            *slot = static_cast<std::int64_t>(u);
            return true;
        }
        default: {
            // Unsigned / temporal / Decimal64 / opaque fixed-width: raw bits.
            std::uint64_t u = 0;
            std::memcpy(&u, p, width);                     // zero-extend
            *slot = static_cast<std::int64_t>(u);
            return true;
        }
    }
}

// Pack row `r` of `c`. *is_null carries the validity; a NULL row's slot is 0
// so the caller's null flag (not the bits) decides its identity. False for a
// type wider than 8 bytes: take a 16-byte / content key instead.
BOLT_FORCE_INLINE bool slot8_pack(const BoltColumn& c, std::int64_t r,
                                  std::int64_t* slot, bool* is_null) noexcept {
    assert(slot != nullptr && is_null != nullptr);
    assert(r >= 0);
    if (c.data == nullptr || c.type == BoltType::Utf8) return false;
    const std::uint32_t w = c.type_size_bytes;
    if (w == 0 || w > 8) return false;
    if (c.validity != nullptr) {
        const std::int64_t bit = c.validity_offset + r;
        if (((c.validity[bit >> 3] >> (bit & 7)) & 1) == 0) {
            *is_null = true;
            *slot = 0;
            return true;
        }
    }
    *is_null = false;
    const auto* p = static_cast<const std::uint8_t*>(c.data) +
                    static_cast<std::size_t>(r) * w;
    return slot8_from_bytes(c.type, p, w, slot);
}

}  // namespace bolt::kernels::numeric_key
