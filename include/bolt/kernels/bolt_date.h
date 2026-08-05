// bolt_date.h — Date32 extract kernel family.
//
// Date32 = signed int32 days since 1970-01-01 (Arrow / Parquet semantics).
// EXTRACT(year|month|day|quarter|week|dow FROM date) for TPC-H Q7/Q8/Q9 and
// similar analytical workloads where the predecessor used Arrow compute.
//
// Algorithm: Howard Hinnant's civil_from_days (public domain; chrono WG21
// paper "date.h"). All-integer, branch-free. Correct on the proleptic
// Gregorian calendar across the full int32 day range.
//
// Tiger Style: noexcept, ≥2 asserts/fn, ≤70 LOC/fn, BOLT_RESTRICT on every
// array param, no allocations, no exceptions, no RTTI. Compiler auto-
// vectorizes the inner loops on AVX2 (8x int32 lanes) / NEON (4x).
//
// API:
//   date32_extract_{year, month, day, quarter, week, day_of_week, day_of_year}
//   date32_to_ymd                         — joint Y/M/D in one pass
//   date32_filter_{gt,ge,lt,le,eq,ne}     — thin int32 filter wrappers
//
// Day-of-week convention: ISO 8601, Monday = 1 .. Sunday = 7.
// Week convention: ISO 8601 week number (1..53). Year-spanning weeks are
// attributed to whichever year contains the Thursday of that week.

#pragma once

#include <cassert>
#include <cstdint>

#include "bolt/bolt_port.h"
#include "bolt/kernels/bolt_numeric.h"  // filter_gt<int32_t> etc.

namespace bolt {
namespace kernels {
namespace date {

// ---------------------------------------------------------------------------
// Joint Y/M/D record. 8-byte payload so an array of these vectorises cleanly.
// ---------------------------------------------------------------------------
struct YMD32 {
    int32_t y;
    uint8_t m;
    uint8_t d;
    uint8_t _pad[2];
};

// ---------------------------------------------------------------------------
// Hinnant civil_from_days specialised for int32 input. Returns Y, M, D plus
// the days-of-year ([0,365]) so callers wanting quarter/week/dow don't have
// to re-run the full algorithm.
// ---------------------------------------------------------------------------
struct YMDDoy { int32_t y; uint32_t m; uint32_t d; uint32_t doy; };

BOLT_FORCE_INLINE YMDDoy civil_from_days32(int32_t z) noexcept {
    // Shift epoch to 0000-03-01. For any z >= -719468 (year 0 AD onwards),
    // zz is non-negative — we use the unsigned fast path. The negative-z
    // branch is preserved for completeness but covers no realistic dates.
    // Using uint32 throughout lets MSVC/clang synthesize divide-by-constant
    // as multiply-high; key to auto-vectorising on AVX2 (8x i32 lanes).
    int32_t zz = z + 719468;
    int32_t era;
    uint32_t doe;
    if (zz >= 0) {
        era = static_cast<int32_t>(static_cast<uint32_t>(zz) / 146097u);
        doe = static_cast<uint32_t>(zz) - static_cast<uint32_t>(era) * 146097u;
    } else {
        // Cold path: pre-0000-03-01 dates (year < 0). Use signed math.
        int64_t zz64 = static_cast<int64_t>(zz);
        int64_t e = (zz64 - 146096) / 146097;
        era = static_cast<int32_t>(e);
        doe = static_cast<uint32_t>(zz64 - e * 146097);
    }
    const uint32_t yoe = (doe - doe/1460u + doe/36524u - doe/146096u) / 365u;   // [0,399]
    const int32_t  y   = static_cast<int32_t>(yoe) + era * 400;
    const uint32_t doy = doe - (365u*yoe + yoe/4u - yoe/100u);                  // [0,365]
    const uint32_t mp  = (5u*doy + 2u) / 153u;                                  // [0,11]
    const uint32_t d   = doy - (153u*mp + 2u)/5u + 1u;                          // [1,31]
    const uint32_t m   = (mp < 10u) ? mp + 3u : mp - 9u;                        // [1,12]
    YMDDoy r;
    r.y   = y + static_cast<int32_t>(m <= 2u);
    r.m   = m;
    r.d   = d;
    r.doy = doy;
    return r;
}

// ---------------------------------------------------------------------------
// Day-of-week from Date32 (days since 1970-01-01). 1970-01-01 was a Thursday.
// Returns ISO 8601 (Mon=1..Sun=7). Branchless modulo via signed adjust.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE int32_t dow_from_days32(int32_t z) noexcept {
    // (z + 3) mod 7 in [0,6] with 0=Mon ; +1 to map Mon=1..Sun=7.
    int32_t r = (z + 3) % 7;
    r += (r < 0) ? 7 : 0;
    return r + 1;
}

// ---------------------------------------------------------------------------
// Civil-day serial for (y, m, d) — inverse of civil_from_days, used by the
// ISO-week kernel to find the ordinal of the year's first Thursday.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE int64_t days_from_civil(int32_t y, uint32_t m, uint32_t d) noexcept {
    assert(m >= 1 && m <= 12);
    assert(d >= 1 && d <= 31);
    int64_t yy = static_cast<int64_t>(y) - (m <= 2 ? 1 : 0);
    const int64_t era = (yy >= 0 ? yy : yy - 399) / 400;
    const uint64_t yoe = static_cast<uint64_t>(yy - era * 400);                 // [0,399]
    const uint64_t doy = (153 * (m > 2 ? m - 3 : m + 9) + 2) / 5 + d - 1;       // [0,365]
    const uint64_t doe = yoe * 365 + yoe/4 - yoe/100 + doy;                     // [0,146096]
    return era * 146097 + static_cast<int64_t>(doe) - 719468;
}

// ---------------------------------------------------------------------------
// EXTRACT(year FROM date)
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE void date32_extract_year(
        const int32_t* BOLT_RESTRICT days_in,
        int32_t*       BOLT_RESTRICT year_out,
        int64_t        n) noexcept {
    assert(days_in  != nullptr || n == 0);
    assert(year_out != nullptr || n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) {
        YMDDoy r = civil_from_days32(days_in[i]);
        year_out[i] = r.y;
    }
}

// ---------------------------------------------------------------------------
// EXTRACT(month FROM date) — 1..12
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE void date32_extract_month(
        const int32_t* BOLT_RESTRICT days_in,
        int32_t*       BOLT_RESTRICT month_out,
        int64_t        n) noexcept {
    assert(days_in   != nullptr || n == 0);
    assert(month_out != nullptr || n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) {
        YMDDoy r = civil_from_days32(days_in[i]);
        month_out[i] = static_cast<int32_t>(r.m);
    }
}

// ---------------------------------------------------------------------------
// EXTRACT(day FROM date) — 1..31
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE void date32_extract_day(
        const int32_t* BOLT_RESTRICT days_in,
        int32_t*       BOLT_RESTRICT day_out,
        int64_t        n) noexcept {
    assert(days_in != nullptr || n == 0);
    assert(day_out != nullptr || n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) {
        YMDDoy r = civil_from_days32(days_in[i]);
        day_out[i] = static_cast<int32_t>(r.d);
    }
}

// ---------------------------------------------------------------------------
// EXTRACT(quarter FROM date) — 1..4. Branchless via (month - 1) / 3 + 1.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE void date32_extract_quarter(
        const int32_t* BOLT_RESTRICT days_in,
        int32_t*       BOLT_RESTRICT quarter_out,
        int64_t        n) noexcept {
    assert(days_in     != nullptr || n == 0);
    assert(quarter_out != nullptr || n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) {
        YMDDoy r = civil_from_days32(days_in[i]);
        quarter_out[i] = static_cast<int32_t>((r.m - 1) / 3 + 1);
    }
}

// ---------------------------------------------------------------------------
// EXTRACT(day_of_week FROM date) — ISO 8601, Mon=1 .. Sun=7.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE void date32_extract_day_of_week(
        const int32_t* BOLT_RESTRICT days_in,
        int32_t*       BOLT_RESTRICT dow_out,
        int64_t        n) noexcept {
    assert(days_in != nullptr || n == 0);
    assert(dow_out != nullptr || n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) dow_out[i] = dow_from_days32(days_in[i]);
}

// ---------------------------------------------------------------------------
// EXTRACT(day_of_year FROM date) — 1..366. Re-derived properly so leap-year
// edge cases at Jan 1 / Feb 29 land correctly (the shifted-civil doy in
// civil_from_days uses Mar 1 = 0 internally, so we recompute from Jan 1).
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE void date32_extract_day_of_year(
        const int32_t* BOLT_RESTRICT days_in,
        int32_t*       BOLT_RESTRICT doy_out,
        int64_t        n) noexcept {
    assert(days_in != nullptr || n == 0);
    assert(doy_out != nullptr || n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) {
        YMDDoy r = civil_from_days32(days_in[i]);
        int64_t jan1 = days_from_civil(r.y, 1, 1);
        doy_out[i] = static_cast<int32_t>(static_cast<int64_t>(days_in[i]) - jan1 + 1);
    }
}

// ---------------------------------------------------------------------------
// EXTRACT(week FROM date) — ISO 8601 week number (1..53). Algorithm:
//   1. Compute the date's ordinal (1..366).
//   2. ISO week = (ordinal - dow + 10) / 7 with dow Mon=1..Sun=7.
//   3. Handle week-0 → last week of prior year, week-53 → week-1 of next year.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE int32_t iso_week_from_components(int32_t y, int32_t doy1, int32_t dow) noexcept {
    // doy1 = day-of-year (1-based); dow = ISO Mon=1..Sun=7.
    int32_t w = (doy1 - dow + 10) / 7;
    if (w == 0) {
        // Belongs to last week of previous year.
        int32_t py = y - 1;
        // Number of days in py: 366 if leap, else 365.
        bool leap = (py % 4 == 0 && py % 100 != 0) || (py % 400 == 0);
        int32_t pydays = leap ? 366 : 365;
        int32_t pdow = dow_from_days32(static_cast<int32_t>(days_from_civil(py, 12, 31)));
        int32_t last_doy = pydays;
        return (last_doy - pdow + 10) / 7;
    }
    if (w == 53) {
        // Verify it really is week 53 (year must contain Jan 1 or Dec 31 as Thu).
        int32_t jan1_dow = dow_from_days32(static_cast<int32_t>(days_from_civil(y, 1, 1)));
        bool leap = (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
        // ISO week 53 exists iff Jan 1 is Thursday, or it's a leap year and Jan 1 is Wed.
        if (jan1_dow == 4 || (leap && jan1_dow == 3)) return 53;
        return 1;
    }
    return w;
}

BOLT_FORCE_INLINE void date32_extract_week(
        const int32_t* BOLT_RESTRICT days_in,
        int32_t*       BOLT_RESTRICT week_out,
        int64_t        n) noexcept {
    assert(days_in  != nullptr || n == 0);
    assert(week_out != nullptr || n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) {
        int32_t z = days_in[i];
        YMDDoy r = civil_from_days32(z);
        int64_t jan1 = days_from_civil(r.y, 1, 1);
        int32_t doy1 = static_cast<int32_t>(static_cast<int64_t>(z) - jan1 + 1);
        int32_t dow  = dow_from_days32(z);
        week_out[i]  = iso_week_from_components(r.y, doy1, dow);
    }
}

// ---------------------------------------------------------------------------
// Joint Y/M/D extract — single pass, three outputs.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE void date32_to_ymd(
        const int32_t* BOLT_RESTRICT days_in,
        YMD32*         BOLT_RESTRICT out,
        int64_t        n) noexcept {
    assert(days_in != nullptr || n == 0);
    assert(out     != nullptr || n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) {
        YMDDoy r = civil_from_days32(days_in[i]);
        out[i].y      = r.y;
        out[i].m      = static_cast<uint8_t>(r.m);
        out[i].d      = static_cast<uint8_t>(r.d);
        out[i]._pad[0] = 0;
        out[i]._pad[1] = 0;
    }
}

// ---------------------------------------------------------------------------
// Compare/filter wrappers — Date32 is int32 underneath, so we route through
// the existing numeric filter kernels. Provided for API completeness so
// callers don't have to reach into bolt_numeric.h directly.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE int64_t date32_filter_gt(
        const int32_t* BOLT_RESTRICT data, int64_t n, int32_t scalar,
        int32_t* BOLT_RESTRICT sel_out) noexcept {
    assert(data    != nullptr || n == 0);
    assert(sel_out != nullptr || n == 0);
    return bolt::kernels::filter_gt_i32_simd(data, n, scalar, sel_out);
}
BOLT_FORCE_INLINE int64_t date32_filter_lt(
        const int32_t* BOLT_RESTRICT data, int64_t n, int32_t scalar,
        int32_t* BOLT_RESTRICT sel_out) noexcept {
    assert(data    != nullptr || n == 0);
    assert(sel_out != nullptr || n == 0);
    return bolt::kernels::filter_lt<int32_t>(data, n, scalar, sel_out);
}
BOLT_FORCE_INLINE int64_t date32_filter_ge(
        const int32_t* BOLT_RESTRICT data, int64_t n, int32_t scalar,
        int32_t* BOLT_RESTRICT sel_out) noexcept {
    assert(data    != nullptr || n == 0);
    assert(sel_out != nullptr || n == 0);
    return bolt::kernels::filter_ge<int32_t>(data, n, scalar, sel_out);
}
BOLT_FORCE_INLINE int64_t date32_filter_le(
        const int32_t* BOLT_RESTRICT data, int64_t n, int32_t scalar,
        int32_t* BOLT_RESTRICT sel_out) noexcept {
    assert(data    != nullptr || n == 0);
    assert(sel_out != nullptr || n == 0);
    return bolt::kernels::filter_le<int32_t>(data, n, scalar, sel_out);
}
BOLT_FORCE_INLINE int64_t date32_filter_eq(
        const int32_t* BOLT_RESTRICT data, int64_t n, int32_t scalar,
        int32_t* BOLT_RESTRICT sel_out) noexcept {
    assert(data    != nullptr || n == 0);
    assert(sel_out != nullptr || n == 0);
    return bolt::kernels::filter_eq<int32_t>(data, n, scalar, sel_out);
}
BOLT_FORCE_INLINE int64_t date32_filter_ne(
        const int32_t* BOLT_RESTRICT data, int64_t n, int32_t scalar,
        int32_t* BOLT_RESTRICT sel_out) noexcept {
    assert(data    != nullptr || n == 0);
    assert(sel_out != nullptr || n == 0);
    return bolt::kernels::filter_ne<int32_t>(data, n, scalar, sel_out);
}

// ---------------------------------------------------------------------------
// Unified int64-output EXTRACT — for the column-at-a-time expression evaluator
// (chukonu represents EXTRACT results as int64). One pass; `field` selects:
// 0=year 1=month(1..12) 2=day(1..31) 3=quarter(1..4). ~5 ns/row. Caller owns
// `out`. Other fields (week/dow) keep their dedicated int32 kernels above.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE void date32_extract_i64(
        const int32_t* BOLT_RESTRICT days_in, int64_t n, int field,
        int64_t* BOLT_RESTRICT out) noexcept {
    assert((days_in != nullptr && out != nullptr) || n == 0);
    assert(n >= 0 && field >= 0 && field <= 3);
    for (int64_t i = 0; i < n; ++i) {
        const YMDDoy r = civil_from_days32(days_in[i]);
        int64_t v;
        switch (field) {
            case 0:  v = r.y; break;
            case 1:  v = r.m; break;
            case 2:  v = r.d; break;
            default: v = (static_cast<int64_t>(r.m) - 1) / 3 + 1; break;  // quarter
        }
        out[i] = v;
    }
}

// ---------------------------------------------------------------------------
// Time-of-day extraction from an Int64 NANOSECOND epoch timestamp (distinct
// from the Date32-day-count family above). Matches chukonu's existing
// date_trunc/time_bucket nanosecond convention. Wall-clock UTC components —
// no timezone conversion (chukonu timestamps are timezone-naive throughout).
// G2FEAT-283 ClickBench follow-up (EXTRACT(minute/hour/second FROM ts)).
// Branch-free floor-division-then-modulo; correct for negative epoch
// nanoseconds too (pre-1970 timestamps), unlike a plain C `%`.
// ---------------------------------------------------------------------------
inline constexpr int64_t kNsPerSecond = 1000000000LL;
inline constexpr int64_t kNsPerMinute = 60LL * kNsPerSecond;
inline constexpr int64_t kNsPerHour   = 60LL * kNsPerMinute;
inline constexpr int64_t kNsPerDay    = 24LL * kNsPerHour;

// Floor-mod: result always in [0, m), even for negative `a` (C's `%` would
// return a negative remainder for negative `a`, wrong for epoch-before-1970
// timestamps).
BOLT_FORCE_INLINE int64_t floor_mod_i64(int64_t a, int64_t m) noexcept {
    assert(m > 0);
    const int64_t r = a % m;
    return (r < 0) ? (r + m) : r;
}

BOLT_FORCE_INLINE int32_t hour_from_ns64(int64_t ns) noexcept {
    return static_cast<int32_t>(floor_mod_i64(ns, kNsPerDay) / kNsPerHour);
}

BOLT_FORCE_INLINE int32_t minute_from_ns64(int64_t ns) noexcept {
    return static_cast<int32_t>(
        floor_mod_i64(ns, kNsPerHour) / kNsPerMinute);
}

BOLT_FORCE_INLINE int32_t second_from_ns64(int64_t ns) noexcept {
    return static_cast<int32_t>(
        floor_mod_i64(ns, kNsPerMinute) / kNsPerSecond);
}

// Batch int64-output form, mirroring date32_extract_i64's shape. `field`:
// 0=hour(0..23) 1=minute(0..59) 2=second(0..59). ~2 ns/row (pure integer
// div/mod, no civil-calendar algorithm needed for time-of-day).
BOLT_FORCE_INLINE void ns64_extract_i64(
        const int64_t* BOLT_RESTRICT ns_in, int64_t n, int field,
        int64_t* BOLT_RESTRICT out) noexcept {
    assert((ns_in != nullptr && out != nullptr) || n == 0);
    assert(n >= 0 && field >= 0 && field <= 2);
    for (int64_t i = 0; i < n; ++i) {
        int32_t v;
        switch (field) {
            case 0:  v = hour_from_ns64(ns_in[i]);   break;
            case 1:  v = minute_from_ns64(ns_in[i]); break;
            default: v = second_from_ns64(ns_in[i]); break;
        }
        out[i] = static_cast<int64_t>(v);
    }
}

// ---------------------------------------------------------------------------
// G2FEAT-133 — CIVIL DATE FIELDS FROM AN INT64 NANOSECOND EPOCH.
//
// The date32_extract_* family above answers year/month/day/quarter/week for a
// value that COUNTS DAYS. `ns64_extract_i64` answers hour/minute/second for a
// value that COUNTS NANOSECONDS. Nothing answered the date fields for a
// nanosecond epoch, so a caller holding one had to divide down to days itself
// -- and the obvious spelling, C's `/`, TRUNCATES TOWARD ZERO. For any instant
// before 1970 that is off by a whole day: -1 ns is 1969-12-31, but -1/86400e9
// is 0, i.e. 1970-01-01. Exactly the class of error `floor_mod_i64` already
// exists to prevent on the time-of-day side, one field family over.
//
// So the floor is done HERE, once, and the existing civil algorithm is reused
// unchanged -- these are thin adapters, not a second calendar implementation.
//
// RANGE. int64 nanoseconds span 1677-09-21 .. 2262-04-11, so the day count is
// within +/-106,752 and always fits int32; the assert states that rather than
// assuming it.
// ---------------------------------------------------------------------------

// Floor-div: rounds toward NEGATIVE INFINITY, unlike C's `/`. Companion to
// floor_mod_i64 above (a == floor_div_i64(a,m)*m + floor_mod_i64(a,m)).
BOLT_FORCE_INLINE int64_t floor_div_i64(int64_t a, int64_t m) noexcept {
    assert(m > 0);
    const int64_t q = a / m;
    const int64_t r = a % m;
    assert(r > -m && r < m);
    return (r < 0) ? (q - 1) : q;
}

// Whole days since 1970-01-01 for a nanosecond epoch. Floor semantics: every
// instant inside 1969-12-31 maps to -1, never to 0.
BOLT_FORCE_INLINE int32_t days_from_ns64(int64_t ns) noexcept {
    const int64_t d = floor_div_i64(ns, kNsPerDay);
    assert(d >= -3000000LL && d <= 3000000LL);   // int64-ns span is ~+/-106752
    assert(d * kNsPerDay <= ns);                 // floor, not truncate
    return static_cast<int32_t>(d);
}

// Batch int64-output form, mirroring date32_extract_i64's shape and field
// numbering so the two are drop-in alternatives for the same caller.
// `field`: 0=year 1=month(1..12) 2=day(1..31) 3=quarter(1..4) 4=ISO week(1..53).
BOLT_FORCE_INLINE void ns64_extract_date_i64(
        const int64_t* BOLT_RESTRICT ns_in, int64_t n, int field,
        int64_t* BOLT_RESTRICT out) noexcept {
    assert((ns_in != nullptr && out != nullptr) || n == 0);
    assert(n >= 0 && field >= 0 && field <= 4);
    for (int64_t i = 0; i < n; ++i) {
        const int32_t z = days_from_ns64(ns_in[i]);
        const YMDDoy  r = civil_from_days32(z);
        int64_t v;
        switch (field) {
            case 0:  v = r.y; break;
            case 1:  v = r.m; break;
            case 2:  v = r.d; break;
            case 3:  v = (static_cast<int64_t>(r.m) - 1) / 3 + 1; break;
            default: {
                const int64_t jan1 = days_from_civil(r.y, 1, 1);
                const int32_t doy1 =
                    static_cast<int32_t>(static_cast<int64_t>(z) - jan1 + 1);
                v = iso_week_from_components(r.y, doy1, dow_from_days32(z));
                break;
            }
        }
        out[i] = v;
    }
}

} // namespace date
} // namespace kernels
} // namespace bolt
