// bolt/kernels/bolt_float_parse.h — locale-independent, correctly rounded
// text -> double with std::from_chars(first, last, v) (general format)
// semantics: optional '-', no leading blanks or '+', "inf"/"infinity"/"nan"
// /"nan(chars)" case-insensitive, decimal digits with an optional exponent.
// Replaces the <charconv> overload Apple gates behind macOS 26.0.
//
// Exact small values (mantissa <= 2^53, |exp10| <= 22) take one IEEE
// multiply or divide; everything else is an exact big-integer quotient
// rounded half-to-even. Past 800 significant digits the rest only matters
// as a sticky bit (a midpoint never needs more than 767 digits).
//
// Tiger Style: noexcept, no allocation, bounded loops, >=2 asserts.

#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

#include "bolt/kernels/bolt_bignum_small.h"

namespace bolt {
namespace kernels {
namespace float_chars {

enum class ParseStatus : uint8_t { ok, invalid, out_of_range };

struct ParseResult {
    const char* ptr;       // one past the match; `first` when invalid
    ParseStatus status;
};

namespace pdetail {

namespace bn = ::bolt::kernels::bignum;

inline constexpr uint32_t k_max_digits = 800;

struct Decimal {
    uint8_t digit[k_max_digits + 1];
    uint32_t nd;           // significant digits kept (plus the sticky one)
    int32_t exp10;         // value = D * 10^exp10
};

inline char lower(char c) noexcept {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

inline bool match_ci(const char* p, const char* e, const char* w) noexcept {
    assert(p != nullptr && w != nullptr);
    assert(p <= e);
    for (; *w != '\0'; ++w, ++p) {
        if (p == e || lower(*p) != *w) return false;
    }
    return true;
}

// inf / infinity / nan / nan(n-char-seq). Returns the end, or nullptr.
inline const char* scan_special(const char* p, const char* e, double* v) noexcept {
    assert(p != nullptr && v != nullptr);
    assert(p <= e);
    if (match_ci(p, e, "inf")) {
        *v = std::numeric_limits<double>::infinity();
        return match_ci(p, e, "infinity") ? p + 8 : p + 3;
    }
    if (!match_ci(p, e, "nan")) return nullptr;
    *v = std::numeric_limits<double>::quiet_NaN();
    const char* q = p + 3;
    if (q == e || *q != '(') return q;
    const char* c = q + 1;
    while (c != e && ((*c >= '0' && *c <= '9') || (lower(*c) >= 'a' &&
                      lower(*c) <= 'z') || *c == '_')) ++c;
    return (c != e && *c == ')') ? c + 1 : q;
}

inline void push_digit(Decimal* d, uint32_t v, bool frac, bool* sticky) noexcept {
    assert(d != nullptr && v <= 9);
    assert(d->nd <= k_max_digits);
    if (d->nd == 0 && v == 0) { if (frac) --d->exp10; return; }
    if (d->nd < k_max_digits) {
        d->digit[d->nd++] = static_cast<uint8_t>(v);
        if (frac) --d->exp10;
        return;
    }
    if (!frac) ++d->exp10;
    if (v != 0) *sticky = true;
}

// Mantissa digits and exponent; returns the end, or nullptr if no digit.
inline const char* scan_decimal(const char* p, const char* e, Decimal* d) noexcept {
    assert(p != nullptr && d != nullptr);
    assert(p <= e);
    d->nd = 0;
    d->exp10 = 0;
    bool sticky = false, any = false, frac = false;
    for (; p != e; ++p) {
        if (*p == '.' && !frac) { frac = true; continue; }
        if (*p < '0' || *p > '9') break;
        any = true;
        push_digit(d, static_cast<uint32_t>(*p - '0'), frac, &sticky);
    }
    if (!any) return nullptr;
    if (sticky) { d->digit[d->nd++] = 1; --d->exp10; }
    if (p != e && (*p == 'e' || *p == 'E')) {
        const char* q = p + 1;
        bool neg = false;
        if (q != e && (*q == '+' || *q == '-')) { neg = *q == '-'; ++q; }
        if (q != e && *q >= '0' && *q <= '9') {
            int32_t x = 0;
            for (; q != e && *q >= '0' && *q <= '9'; ++q) {
                if (x < 100000) x = x * 10 + (*q - '0');
            }
            d->exp10 += neg ? -x : x;
            p = q;
        }
    }
    assert(d->nd <= k_max_digits + 1);
    return p;
}

// Correctly rounded D * 10^exp10 via q = floor(D * 10^exp10 * 2^k) with
// 63-64 bits. Returns false on overflow or a result that rounds to zero.
inline bool exact_quotient(const Decimal* d, double* out) noexcept {
    assert(d != nullptr && out != nullptr);
    assert(d->nd > 0 && d->exp10 >= -1200 && d->exp10 <= 400);
    bn::Big num, den, t;
    bn::set_u64(&num, 0);
    bn::set_u64(&den, 1);
    for (uint32_t i = 0; i < d->nd; ++i) {
        bn::mul_small(&num, 10);
        bn::add_small(&num, d->digit[i]);
    }
    if (d->exp10 >= 0) bn::mul_pow10(&num, static_cast<uint32_t>(d->exp10));
    else bn::mul_pow10(&den, static_cast<uint32_t>(-d->exp10));
    const int32_t k = 63 - (static_cast<int32_t>(bn::bit_length(&num)) -
                            static_cast<int32_t>(bn::bit_length(&den)));
    if (k >= 0) bn::shl(&num, static_cast<uint32_t>(k));
    else bn::shl(&den, static_cast<uint32_t>(-k));
    bn::copy(&t, &den);
    bn::shl(&t, 63);
    assert(!num.overflow && !t.overflow);
    uint64_t q = 0;
    for (int32_t bit = 63; bit >= 0; --bit) {
        if (bn::compare(&num, &t) >= 0) {
            bn::sub(&num, &t);
            q |= uint64_t{1} << bit;
        }
        bn::shr1(&t);
    }
    const bool sticky = !bn::is_zero(&num);
    uint32_t len = 0;
    for (uint64_t x = q; x != 0; x >>= 1) ++len;
    assert(len >= 62 && len <= 64);
    const int32_t e2 = static_cast<int32_t>(len) - 1 - k;
    if (e2 > 1023) return false;
    const int32_t p = e2 >= -1022 ? 53 : e2 + 1075;
    if (p < 0) return false;
    const uint32_t drop = len - static_cast<uint32_t>(p);
    const uint64_t m = p == 0 ? 0 : q >> drop;
    const uint64_t rem = drop == 64 ? q : q & ((uint64_t{1} << drop) - 1u);
    const uint64_t half = uint64_t{1} << (drop - 1);
    const bool up = rem > half || (rem == half && (sticky || (m & 1u)));
    const double r = std::ldexp(static_cast<double>(m + (up ? 1u : 0u)),
                                e2 - p + 1);
    if (r == 0.0 || std::isinf(r)) return false;
    *out = r;
    return true;
}

inline bool to_double(const Decimal* d, double* out) noexcept {
    assert(d != nullptr && out != nullptr);
    assert(d->nd <= k_max_digits + 1);
    if (d->nd == 0) { *out = 0.0; return true; }
    const int32_t mag = d->exp10 + static_cast<int32_t>(d->nd);
    if (mag > 310 || mag <= -324) return false;
    if (d->nd <= 16 && d->exp10 >= -22 && d->exp10 <= 22) {
        static constexpr double p10[23] = {
            1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11,
            1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22};
        uint64_t m = 0;
        for (uint32_t i = 0; i < d->nd; ++i) m = m * 10u + d->digit[i];
        if (m <= (uint64_t{1} << 53)) {
            const double x = static_cast<double>(m);
            *out = d->exp10 >= 0 ? x * p10[d->exp10] : x / p10[-d->exp10];
            return true;
        }
    }
    return exact_quotient(d, out);
}

}  // namespace pdetail

// Parses like std::from_chars(first, last, *out) (general format). On
// invalid, ptr == first; on out_of_range *out is left unmodified.
inline ParseResult f64_from_chars(const char* first, const char* last,
                                  double* out) noexcept {
    assert(first != nullptr && out != nullptr);
    assert(first <= last);
    const char* p = first;
    const bool neg = p != last && *p == '-';
    if (neg) ++p;
    double v = 0.0;
    const char* end = pdetail::scan_special(p, last, &v);
    if (end == nullptr) {
        pdetail::Decimal d;
        end = pdetail::scan_decimal(p, last, &d);
        if (end == nullptr) return ParseResult{first, ParseStatus::invalid};
        if (!pdetail::to_double(&d, &v))
            return ParseResult{end, ParseStatus::out_of_range};
    }
    *out = neg ? -v : v;
    return ParseResult{end, ParseStatus::ok};
}

// JSON number text -> double, locale-independent. The caller's scanner owns
// the JSON grammar; this only refuses what from_chars accepts beyond it
// (inf/nan) and saturates out-of-range values as strtod does (+-inf on
// overflow, signed zero on underflow). The whole [first, last) must match.
inline bool f64_from_json_chars(const char* first, const char* last,
                                double* out) noexcept {
    assert(first != nullptr && out != nullptr);
    assert(first <= last);
    const bool neg = first != last && *first == '-';
    const char* p = neg ? first + 1 : first;
    if (p == last || *p < '0' || *p > '9') return false;
    const ParseResult r = f64_from_chars(first, last, out);
    if (r.ptr != last) return false;
    if (r.status == ParseStatus::ok) return true;
    assert(r.status == ParseStatus::out_of_range);
    pdetail::Decimal d;
    const char* e = pdetail::scan_decimal(p, last, &d);
    assert(e == last && d.nd > 0);
    (void)e;
    const bool overflow = d.exp10 + static_cast<int32_t>(d.nd) > 0;
    const double mag = overflow ? std::numeric_limits<double>::infinity() : 0.0;
    *out = neg ? -mag : mag;
    return true;
}

}  // namespace float_chars
}  // namespace kernels
}  // namespace bolt
