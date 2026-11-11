// bolt::cal — java.time-exact timestampadd / timestampdiff arithmetic and the
// English short names, on local wall-clock micros. Header-only, allocation-free.
//
// add_*: java ZonedDateTime / LocalDateTime plusX. diff_local_us: java
// LocalDateTime.until (the end date is pulled back a day when its time of day
// has not reached the start's), so DAY..YEAR count whole units, truncating
// toward zero. Every function returns false when Spark would raise
// DATETIME_OVERFLOW (the result leaves the micros range), never wraps.
#pragma once

#include <cassert>
#include <cstdint>

#include "bolt/datetime/bolt_calendar.h"

namespace bolt::cal {

enum class AddUnit : std::uint8_t {
    Microsecond = 1, Millisecond, Second, Minute, Hour, Day, Week, Month, Quarter, Year,
};

namespace detail {

inline char up(char c) noexcept { return (c >= 'a' && c <= 'z') ? static_cast<char>(c - 'a' + 'A') : c; }

inline bool word_eq(const char* s, std::uint32_t n, const char* w) noexcept {
    std::uint32_t i = 0;
    for (; i < n && w[i] != '\0'; ++i) {
        if (up(s[i]) != w[i]) return false;
    }
    return i == n && w[i] == '\0';
}

}  // namespace detail

// timestampadd / timestampdiff unit spelling (case-insensitive; DAYOFYEAR is DAY).
inline bool add_unit_parse(const char* s, std::uint32_t n, AddUnit* out) noexcept {
    assert(out != nullptr);
    struct W { const char* w; AddUnit u; };
    static constexpr W kW[] = {
        {"MICROSECOND", AddUnit::Microsecond}, {"MILLISECOND", AddUnit::Millisecond},
        {"SECOND", AddUnit::Second}, {"MINUTE", AddUnit::Minute}, {"HOUR", AddUnit::Hour},
        {"DAY", AddUnit::Day}, {"DAYOFYEAR", AddUnit::Day}, {"WEEK", AddUnit::Week},
        {"MONTH", AddUnit::Month}, {"QUARTER", AddUnit::Quarter}, {"YEAR", AddUnit::Year},
    };
    if (s == nullptr) return false;
    for (const W& w : kW) {
        if (detail::word_eq(s, n, w.w)) { *out = w.u; return true; }
    }
    return false;
}

constexpr bool is_clock_unit(AddUnit u) noexcept { return u <= AddUnit::Hour; }

constexpr std::int64_t clock_unit_micros(AddUnit u) noexcept {
    switch (u) {
        case AddUnit::Microsecond: return 1;
        case AddUnit::Millisecond: return 1000;
        case AddUnit::Second: return 1000000;
        case AddUnit::Minute: return 60000000LL;
        default: return 3600000000LL;
    }
}

// us + qty * unit (SECOND and below, HOUR): java Math.addExact / multiplyExact.
inline bool add_clock_us(AddUnit u, std::int64_t qty, std::int64_t us, std::int64_t* out) noexcept {
    assert(is_clock_unit(u) && out != nullptr);
    std::int64_t d = 0;
    if (__builtin_mul_overflow(qty, clock_unit_micros(u), &d)) return false;
    return !__builtin_add_overflow(us, d, out);
}

// Wall-clock micros plus DAY / WEEK / MONTH / QUARTER / YEAR (calendar units).
inline bool add_calendar_local_us(AddUnit u, std::int64_t qty, std::int64_t local_us,
                                  std::int64_t* out) noexcept {
    assert(!is_clock_unit(u) && out != nullptr);
    const std::int64_t days = tz::floor_div(local_us, kMicrosPerDay);
    const std::int64_t tod = local_us - days * kMicrosPerDay;
    if (days < -kMaxAbsDays || days > kMaxAbsDays) return false;
    std::int64_t nd = 0;
    if (u == AddUnit::Day || u == AddUnit::Week) {
        std::int64_t d = 0;
        if (__builtin_mul_overflow(qty, u == AddUnit::Week ? std::int64_t{7} : std::int64_t{1}, &d)) return false;
        if (d > 2 * kMaxAbsDays || d < -2 * kMaxAbsDays) return false;
        nd = days + d;
    } else {
        std::int64_t months = qty;
        if (u == AddUnit::Quarter && __builtin_mul_overflow(qty, std::int64_t{3}, &months)) return false;
        if (u == AddUnit::Year && __builtin_mul_overflow(qty, std::int64_t{12}, &months)) return false;
        if (months > 12 * 400000LL || months < -12 * 400000LL) return false;
        nd = add_months(days, months);
    }
    if (nd < -kMaxAbsDays || nd > kMaxAbsDays) return false;
    std::int64_t r = 0;
    if (__builtin_mul_overflow(nd, kMicrosPerDay, &r) || __builtin_add_overflow(r, tod, out)) return false;
    return true;
}

// Instant plus `qty` units in `zone` (ZonedDateTime.plusX): clock units are
// absolute, calendar units move the wall clock and keep the instant's offset
// when it is still valid there (overlap), else resolve like a local->UTC cast.
inline bool add_zoned_us(tz::ZoneId zone, AddUnit u, std::int64_t qty, std::int64_t utc_us,
                         std::int64_t* out) noexcept {
    assert(out != nullptr);
    if (is_clock_unit(u)) return add_clock_us(u, qty, utc_us, out);
    const std::int64_t local = tz::utc_micros_to_local(zone, utc_us);
    std::int64_t nl = 0;
    if (!add_calendar_local_us(u, qty, local, &nl)) return false;
    const std::int64_t off = local - utc_us;
    const std::int64_t cand = nl - off;
    *out = tz::utc_micros_to_local(zone, cand) == nl ? cand : tz::local_micros_to_utc(zone, nl);
    return true;
}

// Wall clock plus units (TIMESTAMP_NTZ): LocalDateTime.plusX.
inline bool add_local_us(AddUnit u, std::int64_t qty, std::int64_t local_us, std::int64_t* out) noexcept {
    assert(out != nullptr);
    return is_clock_unit(u) ? add_clock_us(u, qty, local_us, out) : add_calendar_local_us(u, qty, local_us, out);
}

inline std::int64_t months_until(std::int64_t d1, std::int64_t d2) noexcept {
    std::int32_t y1 = 0, m1 = 0, a1 = 0, y2 = 0, m2 = 0, a2 = 0;
    ymd_from_days(d1, &y1, &m1, &a1);
    ymd_from_days(d2, &y2, &m2, &a2);
    const std::int64_t p1 = (static_cast<std::int64_t>(y1) * 12 + (m1 - 1)) * 32 + a1;
    const std::int64_t p2 = (static_cast<std::int64_t>(y2) * 12 + (m2 - 1)) * 32 + a2;
    return (p2 - p1) / 32;
}

// LocalDateTime.until(end, unit): whole units from `a` to `b`, toward zero.
inline bool diff_local_us(AddUnit u, std::int64_t a, std::int64_t b, std::int64_t* out) noexcept {
    assert(out != nullptr);
    if (is_clock_unit(u)) {
        std::int64_t d = 0;
        if (__builtin_sub_overflow(b, a, &d)) return false;
        *out = d / clock_unit_micros(u);
        return true;
    }
    const std::int64_t da = tz::floor_div(a, kMicrosPerDay);
    std::int64_t db = tz::floor_div(b, kMicrosPerDay);
    const std::int64_t ta = a - da * kMicrosPerDay;
    const std::int64_t tb = b - tz::floor_div(b, kMicrosPerDay) * kMicrosPerDay;
    if (db > da && tb < ta) {
        db -= 1;
    } else if (db < da && tb > ta) {
        db += 1;
    }
    if (da < -kMaxAbsDays || da > kMaxAbsDays || db < -kMaxAbsDays || db > kMaxAbsDays) return false;
    switch (u) {
        case AddUnit::Day: *out = db - da; break;
        case AddUnit::Week: *out = (db - da) / 7; break;
        case AddUnit::Month: *out = months_until(da, db); break;
        case AddUnit::Quarter: *out = months_until(da, db) / 3; break;
        default: *out = months_until(da, db) / 12; break;
    }
    return true;
}

// English short names (java TextStyle.SHORT, Locale.ENGLISH).
inline const char* short_day_name(std::int32_t iso_dow) noexcept {   // 1 = Monday
    static constexpr const char* kD[7] = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"};
    assert(iso_dow >= 1 && iso_dow <= 7);
    return kD[iso_dow - 1];
}

inline const char* short_month_name(std::int32_t month) noexcept {   // 1..12
    static constexpr const char* kM[12] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
    assert(month >= 1 && month <= 12);
    return kM[month - 1];
}

// Year of the ISO week containing `days` (the year of that week's Thursday),
// and that week's number 1..53; exact for proleptic years <= 0 too.
inline void iso_week_year(std::int64_t days, std::int32_t* year, std::int32_t* week) noexcept {
    assert(year != nullptr && week != nullptr);
    const std::int64_t thu = days - (iso_dow(days) - 1) + 3;
    std::int32_t y = 0, m = 0, d = 0;
    ymd_from_days(thu, &y, &m, &d);
    *year = y;
    *week = static_cast<std::int32_t>((thu - days_from_ymd(y, 1, 1)) / 7 + 1);
}

}  // namespace bolt::cal
