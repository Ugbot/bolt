// bolt::cal — proleptic-Gregorian calendar arithmetic on epoch days and
// epoch microseconds: month arithmetic, months_between, truncation, and the
// split of an instant into local fields. Header-only, allocation-free.
//
// Semantics follow Spark 3+/java.time (the reference the engine is checked
// against): add_months clamps the day to the target month's length;
// months_between counts whole months when the days of month agree or both
// are month ends, else adds the (day, second) difference over a 31-day month.
#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>

#include "bolt/datetime/bolt_tz.h"
#include "bolt/kernels/bolt_date.h"

namespace bolt::cal {

inline constexpr std::int64_t kSecPerDay = 86400;
inline constexpr std::int64_t kMicrosPerDay = kSecPerDay * 1000000;
// Days representable as Date32 and still convertible to micros without overflow.
inline constexpr std::int64_t kMaxAbsDays = 106751991LL;

struct LocalFields {
    std::int32_t year;
    std::int32_t month;     // 1..12
    std::int32_t day;       // 1..31
    std::int32_t hour;
    std::int32_t minute;
    std::int32_t second;
    std::int32_t micros;    // 0..999999
};

constexpr bool is_leap(std::int64_t y) noexcept {
    return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
}

constexpr std::int32_t month_days(std::int64_t y, std::int32_t m) noexcept {
    constexpr std::int8_t k[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    return k[m - 1] + ((m == 2 && is_leap(y)) ? 1 : 0);
}

inline std::int64_t days_from_ymd(std::int64_t y, std::int32_t m, std::int32_t d) noexcept {
    assert(m >= 1 && m <= 12);
    assert(d >= 1 && d <= 31);
    return bolt::kernels::date::days_from_civil(static_cast<std::int32_t>(y),
                                                static_cast<std::uint32_t>(m),
                                                static_cast<std::uint32_t>(d));
}

inline void ymd_from_days(std::int64_t days, std::int32_t* y, std::int32_t* m, std::int32_t* d) noexcept {
    assert(y != nullptr && m != nullptr && d != nullptr);
    assert(days >= -kMaxAbsDays && days <= kMaxAbsDays);
    const auto r = bolt::kernels::date::civil_from_days32(static_cast<std::int32_t>(days));
    *y = r.y;
    *m = static_cast<std::int32_t>(r.m);
    *d = static_cast<std::int32_t>(r.d);
}

// ISO day of week, Monday = 1 .. Sunday = 7.
inline std::int32_t iso_dow(std::int64_t days) noexcept {
    std::int64_t r = (days + 3) % 7;
    r += r < 0 ? 7 : 0;
    return static_cast<std::int32_t>(r + 1);
}

inline std::int64_t add_months(std::int64_t days, std::int64_t months) noexcept {
    std::int32_t y = 0, m = 0, d = 0;
    ymd_from_days(days, &y, &m, &d);
    const std::int64_t idx = static_cast<std::int64_t>(y) * 12 + (m - 1) + months;
    const std::int64_t ny = tz::floor_div(idx, 12);
    const std::int32_t nm = static_cast<std::int32_t>(idx - ny * 12) + 1;
    const std::int32_t md = month_days(ny, nm);
    assert(nm >= 1 && nm <= 12);
    return days_from_ymd(ny, nm, d < md ? d : md);
}

inline std::int64_t last_day(std::int64_t days) noexcept {
    std::int32_t y = 0, m = 0, d = 0;
    ymd_from_days(days, &y, &m, &d);
    return days_from_ymd(y, m, month_days(y, m));
}

// First date strictly after `days` falling on ISO weekday `dow` (1..7).
inline std::int64_t next_day(std::int64_t days, std::int32_t dow) noexcept {
    assert(dow >= 1 && dow <= 7);
    const std::int32_t cur = iso_dow(days);
    const std::int32_t delta = ((dow - cur) + 7) % 7;
    return days + (delta == 0 ? 7 : delta);
}

// Truncation units shared by trunc(date) and date_trunc(timestamp).
enum class TruncUnit : std::uint8_t {
    Year = 1, Quarter, Month, Week, Day, Hour, Minute, Second, Millisecond, Microsecond,
};

inline std::int64_t trunc_days(std::int64_t days, TruncUnit u) noexcept {
    std::int32_t y = 0, m = 0, d = 0;
    ymd_from_days(days, &y, &m, &d);
    switch (u) {
        case TruncUnit::Year: return days_from_ymd(y, 1, 1);
        case TruncUnit::Quarter: return days_from_ymd(y, ((m - 1) / 3) * 3 + 1, 1);
        case TruncUnit::Month: return days_from_ymd(y, m, 1);
        case TruncUnit::Week: return days - (iso_dow(days) - 1);
        default: return days;
    }
}

// Local wall-clock micros -> truncated local wall-clock micros.
inline std::int64_t trunc_local_micros(std::int64_t local_us, TruncUnit u) noexcept {
    const std::int64_t days = tz::floor_div(local_us, kMicrosPerDay);
    const std::int64_t tod = local_us - days * kMicrosPerDay;
    assert(tod >= 0 && tod < kMicrosPerDay);
    switch (u) {
        case TruncUnit::Hour: return days * kMicrosPerDay + tod - tod % 3600000000LL;
        case TruncUnit::Minute: return days * kMicrosPerDay + tod - tod % 60000000LL;
        case TruncUnit::Second: return days * kMicrosPerDay + tod - tod % 1000000LL;
        case TruncUnit::Millisecond: return days * kMicrosPerDay + tod - tod % 1000LL;
        case TruncUnit::Microsecond: return local_us;
        default: return trunc_days(days, u) * kMicrosPerDay;
    }
}

inline void fields_from_local_micros(std::int64_t local_us, LocalFields* f) noexcept {
    assert(f != nullptr);
    const std::int64_t days = tz::floor_div(local_us, kMicrosPerDay);
    const std::int64_t tod = local_us - days * kMicrosPerDay;
    assert(tod >= 0 && tod < kMicrosPerDay);
    ymd_from_days(days, &f->year, &f->month, &f->day);
    const std::int64_t sec = tod / 1000000;
    f->hour = static_cast<std::int32_t>(sec / 3600);
    f->minute = static_cast<std::int32_t>((sec / 60) % 60);
    f->second = static_cast<std::int32_t>(sec % 60);
    f->micros = static_cast<std::int32_t>(tod % 1000000);
}

inline std::int64_t local_micros_from_fields(const LocalFields& f) noexcept {
    assert(f.month >= 1 && f.month <= 12);
    assert(f.day >= 1 && f.day <= month_days(f.year, f.month));
    const std::int64_t days = days_from_ymd(f.year, f.month, f.day);
    const std::int64_t sec = static_cast<std::int64_t>(f.hour) * 3600 + f.minute * 60 + f.second;
    return days * kMicrosPerDay + sec * 1000000 + f.micros;
}

// Spark months_between over two LOCAL wall-clock instants (micros).
inline double months_between(std::int64_t local1, std::int64_t local2, bool round_off) noexcept {
    LocalFields a{}, b{};
    fields_from_local_micros(local1, &a);
    fields_from_local_micros(local2, &b);
    const std::int64_t month_diff = (static_cast<std::int64_t>(a.year) * 12 + a.month) -
                                    (static_cast<std::int64_t>(b.year) * 12 + b.month);
    const bool a_end = a.day == month_days(a.year, a.month);
    const bool b_end = b.day == month_days(b.year, b.month);
    if (a.day == b.day || (a_end && b_end)) return static_cast<double>(month_diff);
    const std::int64_t sa = static_cast<std::int64_t>(a.hour) * 3600 + a.minute * 60 + a.second;
    const std::int64_t sb = static_cast<std::int64_t>(b.hour) * 3600 + b.minute * 60 + b.second;
    const std::int64_t sec_diff = static_cast<std::int64_t>(a.day - b.day) * kSecPerDay + sa - sb;
    const double diff = static_cast<double>(month_diff) +
                        static_cast<double>(sec_diff) / static_cast<double>(31 * kSecPerDay);
    if (!round_off) return diff;
    return std::floor(diff * 1e8 + 0.5) / 1e8;   // java Math.round
}

}  // namespace bolt::cal
