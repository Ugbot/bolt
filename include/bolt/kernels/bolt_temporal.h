// bolt_temporal.h — timestamp / date kernels over int64 Unix-epoch seconds.
//
// RULES: No exceptions. No RTTI. No <time.h>. Pure integer math — portable
// on MSVC without tzdata. BOLT_RESTRICT on array params. >=2 asserts/fn,
// <=70 LOC/fn. All fns noexcept.
//
// Date math: Howard Hinnant's civil_from_days (public domain; chrono WG21
// paper "date.h"). Correct for the proleptic Gregorian calendar across the
// full int64 range that matters here. Unix epoch day 0 == 1970-01-01; days
// are computed via floor-division of seconds by 86400, with negative inputs
// handled explicitly so we always floor (not truncate-toward-zero).

#pragma once

#include <cassert>
#include <cstdint>

#include "bolt/bolt_port.h"

namespace bolt {
namespace kernels {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int64_t kSecondsPerDay = 86400;

// ---------------------------------------------------------------------------
// Floor-divide two int64s (truncation toward -inf).
// C++ integer division truncates toward zero; we need floor for negatives.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE int64_t fdiv_i64(int64_t a, int64_t b) noexcept {
    assert(b > 0);
    int64_t q = a / b;
    int64_t r = a % b;
    // If signs of r and b disagree, adjust down.
    q -= (r != 0) && ((r ^ b) < 0) ? 1 : 0;
    return q;
}

BOLT_FORCE_INLINE int64_t fmod_i64(int64_t a, int64_t b) noexcept {
    assert(b > 0);
    int64_t r = a % b;
    r += (r != 0 && (r ^ b) < 0) ? b : 0;
    return r;
}

// ---------------------------------------------------------------------------
// Hinnant civil_from_days: given serial day (days since 1970-01-01, may be
// negative), return (y, m, d) on the proleptic Gregorian calendar.
// Reference: http://howardhinnant.github.io/date_algorithms.html
// ---------------------------------------------------------------------------
struct YMD { int32_t y; uint32_t m; uint32_t d; };

BOLT_FORCE_INLINE YMD civil_from_days(int64_t z) noexcept {
    z += 719468;  // Shift epoch to 0000-03-01.
    const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    const uint64_t doe = static_cast<uint64_t>(z - era * 146097);              // [0, 146096]
    const uint64_t yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365;      // [0, 399]
    const int64_t  y   = static_cast<int64_t>(yoe) + era * 400;
    const uint64_t doy = doe - (365*yoe + yoe/4 - yoe/100);                    // [0, 365]
    const uint64_t mp  = (5*doy + 2) / 153;                                    // [0, 11]
    const uint32_t d   = static_cast<uint32_t>(doy - (153*mp + 2)/5 + 1);      // [1, 31]
    const uint32_t m   = static_cast<uint32_t>(mp < 10 ? mp + 3 : mp - 9);      // [1, 12]
    YMD r;
    r.y = static_cast<int32_t>(y + (m <= 2 ? 1 : 0));
    r.m = m;
    r.d = d;
    return r;
}

// ---------------------------------------------------------------------------
// timestamp_add_seconds: out[i] = ts[i] + seconds.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE void timestamp_add_seconds(
        const int64_t* BOLT_RESTRICT ts, int64_t n,
        int64_t seconds,
        int64_t* BOLT_RESTRICT out) noexcept {
    assert(ts  != nullptr || n == 0);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) out[i] = ts[i] + seconds;
}

// ---------------------------------------------------------------------------
// timestamp_diff_seconds: out[i] = a[i] - b[i].
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE void timestamp_diff_seconds(
        const int64_t* BOLT_RESTRICT a,
        const int64_t* BOLT_RESTRICT b, int64_t n,
        int64_t* BOLT_RESTRICT out) noexcept {
    assert((a != nullptr && b != nullptr) || n == 0);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) out[i] = a[i] - b[i];
}

// ---------------------------------------------------------------------------
// timestamp_extract_year: Gregorian year from Unix-epoch seconds.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE void timestamp_extract_year(
        const int64_t* BOLT_RESTRICT ts, int64_t n,
        int32_t* BOLT_RESTRICT out) noexcept {
    assert(ts  != nullptr || n == 0);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) {
        int64_t days = fdiv_i64(ts[i], kSecondsPerDay);
        YMD r = civil_from_days(days);
        out[i] = r.y;
    }
}

// ---------------------------------------------------------------------------
// timestamp_extract_month: 1..12.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE void timestamp_extract_month(
        const int64_t* BOLT_RESTRICT ts, int64_t n,
        int32_t* BOLT_RESTRICT out) noexcept {
    assert(ts  != nullptr || n == 0);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) {
        int64_t days = fdiv_i64(ts[i], kSecondsPerDay);
        YMD r = civil_from_days(days);
        out[i] = static_cast<int32_t>(r.m);
    }
}

// ---------------------------------------------------------------------------
// timestamp_extract_day: 1..31.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE void timestamp_extract_day(
        const int64_t* BOLT_RESTRICT ts, int64_t n,
        int32_t* BOLT_RESTRICT out) noexcept {
    assert(ts  != nullptr || n == 0);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) {
        int64_t days = fdiv_i64(ts[i], kSecondsPerDay);
        YMD r = civil_from_days(days);
        out[i] = static_cast<int32_t>(r.d);
    }
}

// ---------------------------------------------------------------------------
// timestamp_date_trunc_day: floor to start-of-day (midnight UTC).
// Handles negatives correctly: ts - fmod(ts, 86400).
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE void timestamp_date_trunc_day(
        const int64_t* BOLT_RESTRICT ts, int64_t n,
        int64_t* BOLT_RESTRICT out) noexcept {
    assert(ts  != nullptr || n == 0);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) {
        out[i] = ts[i] - fmod_i64(ts[i], kSecondsPerDay);
    }
}

} // namespace kernels
} // namespace bolt
