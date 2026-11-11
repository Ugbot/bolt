// Bit-exact C++ port of the Sun fdlibm routines as java.lang.FdLibm (JDK 21) runs them behind
// StrictMath, which Spark's math expressions call. Verbatim port of that reference (kept function-for-function
// so a diff against FdLibm.java stays readable; the long routines are exempt from the 70-line rule for that
// reason). Copyright (C) 1993 by Sun Microsystems, Inc. Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice is preserved.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace bolt::kernels::fdlibm {

inline constexpr double k_pi = 3.141592653589793;
inline constexpr double INFINITY_D = std::numeric_limits<double>::infinity();
inline constexpr double TWO24 = 0x1.0p24;
inline constexpr double TWO54 = 0x1.0p54;
inline constexpr double HUGE_D = 1.0e+300;
inline constexpr int32_t EXP_BITS = 0x7ff00000;
inline constexpr int32_t EXP_SIGNIF_BITS = 0x7fffffff;
inline constexpr int32_t SIGN_BIT = static_cast<int32_t>(0x80000000u);

inline int32_t lo_w(double x) noexcept {
    std::uint64_t u;
    std::memcpy(&u, &x, 8);
    return static_cast<int32_t>(static_cast<std::uint32_t>(u));
}
inline double lo_w(double x, int32_t low) noexcept {
    std::uint64_t u;
    std::memcpy(&u, &x, 8);
    u = (u & 0xFFFFFFFF00000000ull) | static_cast<std::uint32_t>(low);
    std::memcpy(&x, &u, 8);
    return x;
}
inline int32_t hi_w(double x) noexcept {
    std::uint64_t u;
    std::memcpy(&u, &x, 8);
    return static_cast<int32_t>(static_cast<std::uint32_t>(u >> 32));
}
inline double hi_w(double x, int32_t high) noexcept {
    std::uint64_t u;
    std::memcpy(&u, &x, 8);
    u = (u & 0x00000000FFFFFFFFull) | (static_cast<std::uint64_t>(static_cast<std::uint32_t>(high)) << 32);
    std::memcpy(&x, &u, 8);
    return x;
}
inline double hi_lo(int32_t high, int32_t low) noexcept {
    const std::uint64_t u = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(high)) << 32) |
                            static_cast<std::uint32_t>(low);
    double x;
    std::memcpy(&x, &u, 8);
    return x;
}
inline double pow2d(int32_t n) noexcept { return std::ldexp(1.0, n); }

}  // namespace bolt::kernels::fdlibm
