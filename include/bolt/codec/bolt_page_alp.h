// bolt/codec/bolt_page_alp.h — ALP for float64 pages (Afroozeh, Kuffo, Boncz, "ALP: Adaptive Lossless
// floating-Point Compression", SIGMOD 2024): a value v is stored as the integer d = round(v * 10^e / 10^f)
// when d * 10^f / 10^e reproduces v bit for bit, else as an exception {row, raw bits}. The integers are
// frame-of-reference packed (bolt_page_int.h). Exact for every bit pattern: NaN payloads, infinities, -0.0 and
// denormals that do not round-trip are exceptions.
//
// (e, f) comes from a fixed sample of the page (every k-th value, at most kAlpSample), scored by a closed-form
// size; nothing is encoded to choose it. Encode and decode use the same two multiplies (no add, so no FMA
// contraction), and IEEE multiplication is deterministic, so a value accepted by the encoder decodes to the
// same bits on any host.
//
// Decoders treat input as untrusted; no allocation; whole-page kernels.

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_port.h"
#include "bolt/codec/bolt_page_int.h"

namespace bolt {
namespace pagecodec {

inline constexpr uint32_t kAlpTopExponent = 18;
inline constexpr uint32_t kAlpSample = 64;

namespace detail {

inline constexpr double kAlpF10[kAlpTopExponent + 1] = {
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18};
inline constexpr double kAlpIF10[kAlpTopExponent + 1] = {
    1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16,
    1e-17, 1e-18};
inline constexpr double kAlpIntLimit = 4503599627370496.0;   // 2^52: every integer below is exact in a double

BOLT_FORCE_INLINE double alp_decode_one(int64_t d, double f10, double ie10) noexcept {
    return static_cast<double>(d) * f10 * ie10;
}

// d for v at (e, f); false when v is an exception there.
BOLT_FORCE_INLINE bool alp_try(double v, double e10, double if10, double f10, double ie10, int64_t* d) noexcept {
    const double x = v * e10 * if10;
    const bool in = x > -kAlpIntLimit && x < kAlpIntLimit;           // false for NaN
    const double xs = in ? x : 0.0;
    const int64_t k = static_cast<int64_t>(xs + (xs >= 0.0 ? 0.5 : -0.5));
    const double back = alp_decode_one(k, f10, ie10);
    uint64_t a, b;
    std::memcpy(&a, &v, 8);
    std::memcpy(&b, &back, 8);
    *d = k;
    return in && a == b;
}

}  // namespace detail

struct AlpPlan {
    uint8_t  e, f;
    uint32_t exceptions;
    ForPlan  ints;        // frame of the integers (signed order)
    uint64_t bytes;       // packed integers + exceptions; UINT64_MAX when not encodable
};

/// Chooses (e, f) from the sample, then fills ints[n] (an exception's slot repeats the last accepted integer,
/// or 0 before the first) and counts exceptions. The plan is not encodable when exceptions exceed
/// n * max_exc_permille / 1000.
inline AlpPlan alp_plan(const double* BOLT_RESTRICT v, uint32_t n, int64_t* BOLT_RESTRICT ints,
                        uint32_t max_exc_permille) noexcept {
    assert(v != nullptr && ints != nullptr && n > 0);
    using namespace detail;
    const uint32_t step = n > kAlpSample ? n / kAlpSample : 1u;
    const uint32_t ns = n > kAlpSample ? kAlpSample : n;
    uint32_t be = 0, bf = 0;
    uint64_t best = UINT64_MAX;
    for (uint32_t e = 0; e <= kAlpTopExponent; ++e) {               // bounded: 19
        for (uint32_t f = 0; f <= e; ++f) {                         // bounded: 19
            int64_t lo = INT64_MAX, hi = INT64_MIN;
            uint32_t ok = 0;
            for (uint32_t s = 0; s < ns; ++s) {                     // bounded: kAlpSample
                int64_t d;
                const bool good = alp_try(v[s * step], kAlpF10[e], kAlpIF10[f], kAlpF10[f], kAlpIF10[e], &d);
                ok += good;
                lo = good && d < lo ? d : lo;
                hi = good && d > hi ? d : hi;
            }
            const uint32_t width = ok ? fastlanes::width_of(static_cast<uint64_t>(hi) - static_cast<uint64_t>(lo)) : 64u;
            const uint64_t score = uint64_t{ok} * width + uint64_t{ns - ok} * (64u + 64u + width);   // bits
            if (score < best) { best = score; be = e; bf = f; }     // ties: the smaller exponent, then factor
        }
    }
    AlpPlan p;
    p.e = static_cast<uint8_t>(be);
    p.f = static_cast<uint8_t>(bf);
    uint32_t exc = 0;
    int64_t last = 0;
    const double e10 = kAlpF10[be], if10 = kAlpIF10[bf], f10 = kAlpF10[bf], ie10 = kAlpIF10[be];
    for (uint32_t i = 0; i < n; ++i) {                              // bounded: n
        int64_t d;
        const bool good = alp_try(v[i], e10, if10, f10, ie10, &d);
        last = good ? d : last;
        ints[i] = last;
        exc += !good;
    }
    // slots before the first accepted value hold 0: repeat the first accepted integer there instead
    if (exc != 0 && exc != n) {
        uint32_t first = 0;
        int64_t d;
        while (first < n && !alp_try(v[first], e10, if10, f10, ie10, &d)) ++first;   // bounded: n
        for (uint32_t i = 0; i < first; ++i) ints[i] = ints[first]; // bounded: n
    }
    p.exceptions = exc;
    p.ints = for_plan(ints, n, 8, true);
    const bool fits = uint64_t{exc} * 1000u <= uint64_t{n} * max_exc_permille;
    p.bytes = fits ? p.ints.bytes + uint64_t{exc} * sizeof(Exception) : UINT64_MAX;
    return p;
}

/// packed: plan.ints.bytes; exc: plan.exceptions entries (ascending row).
inline void alp_encode(const double* BOLT_RESTRICT v, uint32_t n, const int64_t* BOLT_RESTRICT ints,
                       const AlpPlan& p, void* packed, Exception* BOLT_RESTRICT exc) noexcept {
    assert(p.bytes != UINT64_MAX && packed != nullptr);
    using namespace detail;
    for_encode(ints, n, 8, p.ints, packed);
    if (p.exceptions == 0) return;
    const double e10 = kAlpF10[p.e], if10 = kAlpIF10[p.f], f10 = kAlpF10[p.f], ie10 = kAlpIF10[p.e];
    uint32_t k = 0;
    for (uint32_t i = 0; i < n; ++i) {                              // bounded: n
        int64_t d;
        if (alp_try(v[i], e10, if10, f10, ie10, &d)) continue;
        uint64_t bits;
        std::memcpy(&bits, &v[i], 8);
        exc[k++] = Exception{i, 0u, bits};
    }
    assert(k == p.exceptions);
}

[[nodiscard]] inline bool alp_decode(const void* packed, uint64_t packed_len, uint32_t n, uint32_t e, uint32_t f,
                                     uint64_t ref, uint32_t width, const Exception* exc, uint64_t exc_len,
                                     double* BOLT_RESTRICT out) noexcept {
    if (e > kAlpTopExponent || f > e || out == nullptr || exc_len % sizeof(Exception) != 0) return false;
    if (exc_len / sizeof(Exception) > n) return false;
    // integers decode in place: out is 8 B per row
    if (!for_decode(packed, packed_len, n, 8, ref, width, out)) return false;
    const double f10 = detail::kAlpF10[f], ie10 = detail::kAlpIF10[e];
    for (uint32_t i = 0; i < n; ++i) {                              // bounded: n
        int64_t d;
        std::memcpy(&d, &out[i], 8);
        out[i] = detail::alp_decode_one(d, f10, ie10);
    }
    const uint32_t ne = static_cast<uint32_t>(exc_len / sizeof(Exception));
    for (uint32_t k = 0; k < ne; ++k) {                             // bounded: exceptions
        if (exc[k].pos >= n || (k != 0 && exc[k].pos <= exc[k - 1].pos)) return false;
        std::memcpy(&out[exc[k].pos], &exc[k].value, 8);
    }
    return true;
}

}  // namespace pagecodec
}  // namespace bolt
