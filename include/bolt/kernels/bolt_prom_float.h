// bolt/kernels/bolt_prom_float.h: a double in Prometheus's text spelling (Go's
// strconv 'g' with shortest round-trip digits): NaN, +Inf, -Inf, 0, 0.1, 1e+06
// is 1000000, exponent form below 1e-6 and from 1e21. Used for `le` and
// `quantile` label values and PromQL result text.
#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "bolt/kernels/bolt_float_chars.h"

namespace bolt::kernels {

inline constexpr std::uint32_t k_prom_float_chars = 48;

// Writes v to out[0, k_prom_float_chars) and returns the length (no terminator).
inline std::uint32_t prom_float_chars(double v, char* out) noexcept {
    assert(out != nullptr);
    if (std::isnan(v)) { std::memcpy(out, "NaN", 3); return 3; }
    if (std::isinf(v)) { std::memcpy(out, v < 0.0 ? "-Inf" : "+Inf", 4); return 4; }
    std::uint32_t n = 0;
    if (std::signbit(v)) out[n++] = '-';
    const double a = std::fabs(v);
    if (a == 0.0) { out[n++] = '0'; return n; }
    char d[20];
    std::int32_t k = 0;   // a ~= 0.d1..dn * 10^k
    const std::uint32_t nd = float_chars::detail::shortest_digits(a, d, &k);
    assert(nd >= 1 && nd <= 17);
    if (a < 1e-6 || a >= 1e21) {
        out[n++] = d[0];
        if (nd > 1) { out[n++] = '.'; std::memcpy(out + n, d + 1, nd - 1); n += nd - 1; }
        const std::int32_t x = k - 1;
        const int e = std::snprintf(out + n, 8, "e%c%02d", x < 0 ? '-' : '+', x < 0 ? -x : x);
        n += e > 0 ? static_cast<std::uint32_t>(e) : 0u;
    } else if (k <= 0) {
        out[n++] = '0'; out[n++] = '.';
        std::memset(out + n, '0', static_cast<std::size_t>(-k)); n += static_cast<std::uint32_t>(-k);
        std::memcpy(out + n, d, nd); n += nd;
    } else if (static_cast<std::uint32_t>(k) >= nd) {
        std::memcpy(out + n, d, nd); n += nd;
        std::memset(out + n, '0', static_cast<std::size_t>(k) - nd); n += static_cast<std::uint32_t>(k) - nd;
    } else {
        std::memcpy(out + n, d, static_cast<std::size_t>(k)); n += static_cast<std::uint32_t>(k);
        out[n++] = '.';
        std::memcpy(out + n, d + k, nd - static_cast<std::uint32_t>(k)); n += nd - static_cast<std::uint32_t>(k);
    }
    assert(n > 0 && n < k_prom_float_chars);
    return n;
}

}  // namespace bolt::kernels
