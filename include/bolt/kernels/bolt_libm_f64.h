// Scalar f64 kernels with C library (not fdlibm) semantics: the trig / hyperbolic family and the
// civil-calendar parts of a Unix-seconds value, as Prometheus and Go's math package define them.
// Reached from a Compute MathFn Unary through F64Op values >= k_libm_f64_base (bolt_math_spark.h).
#pragma once

#include <cmath>
#include <cstdint>

#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace bolt::kernels::libm_f64 {

enum class LibmOp : std::uint8_t {
    Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Tanh, Asinh, Acosh, Atanh,
    Radians, Degrees,
    UnixYear, UnixMonth, UnixDayOfMonth, UnixDayOfYear, UnixDayOfWeek, UnixDaysInMonth,
    UnixHour, UnixMinute,
    Count_
};

// Days since 1970-01-01 -> civil (y, m 1..12, d 1..31): Hinnant's exact proleptic Gregorian algorithm.
inline void civil_from_days(std::int64_t z, std::int64_t* y, std::int32_t* m, std::int32_t* d) noexcept {
    z += 719468;
    const std::int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    const std::int64_t doe = z - era * 146097;
    const std::int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const std::int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const std::int64_t mp = (5 * doy + 2) / 153;
    const std::int64_t mm = mp + (mp < 10 ? 3 : -9);
    *d = static_cast<std::int32_t>(doy - (153 * mp + 2) / 5 + 1);
    *m = static_cast<std::int32_t>(mm);
    *y = yoe + era * 400 + (mm <= 2 ? 1 : 0);
}

inline std::int64_t days_from_civil(std::int64_t y, std::int32_t m, std::int32_t d) noexcept {
    y -= (m <= 2) ? 1 : 0;
    const std::int64_t era = (y >= 0 ? y : y - 399) / 400;
    const std::int64_t yoe = y - era * 400;
    const std::int64_t doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    return era * 146097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719468;
}

// One civil part of `secs` read as Unix seconds (truncated toward zero, as Go's time.Unix(int64(v), 0)).
// NaN, +-Inf and |secs| >= 2^62 answer NaN.
inline double unix_part(LibmOp op, double secs) noexcept {
    if (std::isnan(secs) || std::isinf(secs)) return std::nan("");
    constexpr double k_max = 4611686018427387904.0;
    if (!(secs > -k_max && secs < k_max)) return std::nan("");
    const std::int64_t t = static_cast<std::int64_t>(std::trunc(secs));
    std::int64_t days = t / 86400;
    if (t % 86400 != 0 && t < 0) --days;
    const std::int64_t sod = t - days * 86400;
    if (op == LibmOp::UnixHour) return static_cast<double>(sod / 3600);
    if (op == LibmOp::UnixMinute) return static_cast<double>((sod / 60) % 60);
    if (op == LibmOp::UnixDayOfWeek) return static_cast<double>(((days % 7) + 11) % 7);
    std::int64_t y = 0;
    std::int32_t m = 0, d = 0;
    civil_from_days(days, &y, &m, &d);
    switch (op) {
        case LibmOp::UnixYear: return static_cast<double>(y);
        case LibmOp::UnixMonth: return static_cast<double>(m);
        case LibmOp::UnixDayOfMonth: return static_cast<double>(d);
        case LibmOp::UnixDayOfYear: return static_cast<double>(days - days_from_civil(y, 1, 1) + 1);
        default: break;
    }
    static const std::int32_t k_len[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    const bool leap = (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
    return (m == 2 && leap) ? 29.0 : static_cast<double>(k_len[m - 1]);
}

inline double libm_f64(LibmOp op, double x) noexcept {
    switch (op) {
        case LibmOp::Sin: return std::sin(x);
        case LibmOp::Cos: return std::cos(x);
        case LibmOp::Tan: return std::tan(x);
        case LibmOp::Asin: return std::asin(x);
        case LibmOp::Acos: return std::acos(x);
        case LibmOp::Atan: return std::atan(x);
        case LibmOp::Sinh: return std::sinh(x);
        case LibmOp::Cosh: return std::cosh(x);
        case LibmOp::Tanh: return std::tanh(x);
        case LibmOp::Asinh: return std::asinh(x);
        case LibmOp::Acosh: return std::acosh(x);
        case LibmOp::Atanh: return std::atanh(x);
        case LibmOp::Radians: return x * 0.017453292519943295769;
        case LibmOp::Degrees: return x * 57.295779513082320877;
        default: return unix_part(op, x);
    }
}

}  // namespace bolt::kernels::libm_f64
