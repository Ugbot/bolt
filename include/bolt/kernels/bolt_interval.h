// bolt_interval.h — SQL INTERVAL arithmetic over Date32 (int32 days since
// 1970-01-01).
//
// Why this exists: TPC-H is full of `l_shipdate <= DATE '1998-12-01' -
// INTERVAL '90' DAY`, `o_orderdate >= DATE '...' - INTERVAL '1' YEAR`, etc.
// Tiny semantic, ubiquitous in the workload. PostgreSQL/Arrow split an
// interval into (months, days, microseconds) because month length varies
// — TPC-H only ever needs the (months, days) subset, so we ship a compact
// 8-byte IntervalDate plus the full 16-byte Interval for completeness.
//
// Hot path: months==0 (day-only) which is a trivial int32 add — auto-
// vectorizes 8-wide AVX2, 4-wide NEON. Slow path (months != 0) decomposes
// each date via Hinnant civil_from_days, adds months with year carry,
// clamps the day to month length, then recomposes. We pre-scan the
// interval array OR'ing the month words; if all zero, dispatch is the
// fast path with no per-row branch.
//
// RULES: noexcept, no exceptions, no RTTI, BOLT_RESTRICT on array params,
// >=2 asserts/fn, <=70 LOC/fn.

#pragma once

#include <cassert>
#include <cstdint>

#include "bolt/bolt_port.h"
#include "bolt/kernels/bolt_temporal.h"

namespace bolt {
namespace kernels {
namespace interval {

// ---------------------------------------------------------------------------
// Representations
// ---------------------------------------------------------------------------

// Full PostgreSQL/Arrow MonthDayNano-style interval. 16 bytes.
struct alignas(8) Interval {
    int32_t months;
    int32_t days;
    int64_t microseconds;
};
static_assert(sizeof(Interval) == 16, "Interval must be 16 bytes");

// Compact day/month interval used by TPC-H. 8 bytes.
struct alignas(8) IntervalDate {
    int32_t months;
    int32_t days;
};
static_assert(sizeof(IntervalDate) == 8, "IntervalDate must be 8 bytes");

// ---------------------------------------------------------------------------
// days_from_civil — inverse of civil_from_days. Hinnant public domain.
// Returns the serial day for proleptic Gregorian (y, m, d). Caller is
// responsible for valid month/day ranges; we assert.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE int64_t days_from_civil(int32_t y, uint32_t m, uint32_t d) noexcept {
    assert(m >= 1 && m <= 12);
    assert(d >= 1 && d <= 31);
    int64_t yy = static_cast<int64_t>(y) - (m <= 2 ? 1 : 0);
    const int64_t era = (yy >= 0 ? yy : yy - 399) / 400;
    const uint64_t yoe = static_cast<uint64_t>(yy - era * 400);                  // [0, 399]
    const uint64_t doy =
        (153u * (m > 2 ? m - 3 : m + 9) + 2) / 5 + d - 1;                        // [0, 365]
    const uint64_t doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;                  // [0, 146096]
    return era * 146097 + static_cast<int64_t>(doe) - 719468;
}

// ---------------------------------------------------------------------------
// Month length, clamped to last day on overflow. Branchless leap-year check.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE uint32_t days_in_month(int32_t y, uint32_t m) noexcept {
    assert(m >= 1 && m <= 12);
    assert(y > -32768 && y < 32768);
    // Lookup table for non-Feb months. Feb gets the leap branch.
    static constexpr uint8_t kMonthLen[13] = {
        0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
    };
    if (m != 2) return kMonthLen[m];
    // Leap rule: divisible by 4, not 100, or by 400.
    const bool leap = ((y % 4) == 0 && (y % 100) != 0) || ((y % 400) == 0);
    return leap ? 29u : 28u;
}

// ---------------------------------------------------------------------------
// add_months_clamp — add a (possibly negative) month delta to (y, m, d) with
// year carry and day clamp to the resulting month length.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE void add_months_clamp(int32_t* BOLT_RESTRICT y_io,
                                        uint32_t* BOLT_RESTRICT m_io,
                                        uint32_t* BOLT_RESTRICT d_io,
                                        int32_t delta_months) noexcept {
    assert(y_io && m_io && d_io);
    assert(*m_io >= 1 && *m_io <= 12);
    // Convert to absolute month index (1-based across years), apply delta,
    // unpack.
    int64_t total = static_cast<int64_t>(*y_io) * 12 +
                    static_cast<int64_t>(*m_io) - 1 +
                    static_cast<int64_t>(delta_months);
    int64_t new_y = total / 12;
    int64_t new_m = total % 12;
    if (new_m < 0) { new_m += 12; new_y -= 1; }
    *y_io = static_cast<int32_t>(new_y);
    *m_io = static_cast<uint32_t>(new_m + 1);
    uint32_t dim = days_in_month(*y_io, *m_io);
    if (*d_io > dim) *d_io = dim;
}

// ---------------------------------------------------------------------------
// add_interval_one — scalar helper: date + (months, days) -> date.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE int32_t add_interval_one(int32_t date, int32_t months,
                                           int32_t days) noexcept {
    if (months == 0) return date + days;  // hot path
    YMD r = civil_from_days(static_cast<int64_t>(date));
    int32_t  y = r.y;
    uint32_t m = r.m;
    uint32_t d = r.d;
    add_months_clamp(&y, &m, &d, months);
    int64_t out = days_from_civil(y, m, d) + static_cast<int64_t>(days);
    assert(out >= INT32_MIN && out <= INT32_MAX);
    return static_cast<int32_t>(out);
}

// ---------------------------------------------------------------------------
// pre_scan_months_zero — OR-reduce the month words. If 0, every interval
// is day-only and we can use the fast path with no per-row branch.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE bool all_months_zero(const IntervalDate* BOLT_RESTRICT iv,
                                       int64_t n) noexcept {
    assert(iv != nullptr || n == 0);
    assert(n >= 0);
    int32_t acc = 0;
    for (int64_t i = 0; i < n; ++i) acc |= iv[i].months;
    return acc == 0;
}

// ---------------------------------------------------------------------------
// date32_add_interval: out[i] = date_in[i] + iv_in[i].
// Fast path on all months==0.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE void date32_add_interval(
        const int32_t*      BOLT_RESTRICT date_in,
        const IntervalDate* BOLT_RESTRICT iv_in,
        int32_t*            BOLT_RESTRICT date_out,
        int64_t             n) noexcept {
    assert((date_in != nullptr && iv_in != nullptr) || n == 0);
    assert(date_out != nullptr || n == 0);
    assert(n >= 0);
    if (all_months_zero(iv_in, n)) {
        // Interleave 4-wide so the AoS stride doesn't stall the issue
        // window; MSVC/GCC both vectorise this on AVX2/NEON.
        int64_t i = 0;
        for (; i + 4 <= n; i += 4) {
            int32_t d0 = iv_in[i + 0].days;
            int32_t d1 = iv_in[i + 1].days;
            int32_t d2 = iv_in[i + 2].days;
            int32_t d3 = iv_in[i + 3].days;
            date_out[i + 0] = date_in[i + 0] + d0;
            date_out[i + 1] = date_in[i + 1] + d1;
            date_out[i + 2] = date_in[i + 2] + d2;
            date_out[i + 3] = date_in[i + 3] + d3;
        }
        for (; i < n; ++i) date_out[i] = date_in[i] + iv_in[i].days;
        return;
    }
    for (int64_t i = 0; i < n; ++i) {
        date_out[i] = add_interval_one(date_in[i], iv_in[i].months,
                                       iv_in[i].days);
    }
}

// ---------------------------------------------------------------------------
// date32_add_interval_scalar: broadcast iv across the date array. The
// common TPC-H shape `date - INTERVAL '90' DAY`.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE void date32_add_interval_scalar(
        const int32_t* BOLT_RESTRICT date_in,
        IntervalDate                 iv,
        int32_t*       BOLT_RESTRICT date_out,
        int64_t                      n) noexcept {
    assert(date_in  != nullptr || n == 0);
    assert(date_out != nullptr || n == 0);
    assert(n >= 0);
    if (iv.months == 0) {
        const int32_t d = iv.days;
        for (int64_t i = 0; i < n; ++i) date_out[i] = date_in[i] + d;
        return;
    }
    for (int64_t i = 0; i < n; ++i) {
        date_out[i] = add_interval_one(date_in[i], iv.months, iv.days);
    }
}

// ---------------------------------------------------------------------------
// date32_sub_date32: out[i] = (months=0, days=a[i] - b[i]).
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE void date32_sub_date32(
        const int32_t*      BOLT_RESTRICT a,
        const int32_t*      BOLT_RESTRICT b,
        IntervalDate*       BOLT_RESTRICT out,
        int64_t             n) noexcept {
    assert((a != nullptr && b != nullptr) || n == 0);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) {
        out[i].months = 0;
        out[i].days   = a[i] - b[i];
    }
}

// ---------------------------------------------------------------------------
// interval_mul_scalar: out[i] = iv_in[i] * factor (both fields).
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE void interval_mul_scalar(
        const IntervalDate* BOLT_RESTRICT iv_in,
        int32_t                           factor,
        IntervalDate*       BOLT_RESTRICT iv_out,
        int64_t                           n) noexcept {
    assert(iv_in  != nullptr || n == 0);
    assert(iv_out != nullptr || n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) {
        iv_out[i].months = iv_in[i].months * factor;
        iv_out[i].days   = iv_in[i].days   * factor;
    }
}

// ---------------------------------------------------------------------------
// Normalised total-days view of an interval for comparison. Assumes 30-day
// months — matches PostgreSQL's `justify_interval` semantics for ordering.
// Real TPC-H only ever compares like-shaped intervals so the approximation
// is exact in practice; it only matters when months and days disagree in
// sign, which the SQL standard leaves implementation-defined.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE int64_t interval_to_days_for_cmp(IntervalDate iv) noexcept {
    return static_cast<int64_t>(iv.months) * 30 +
           static_cast<int64_t>(iv.days);
}

// ---------------------------------------------------------------------------
// interval_filter_gt: write indices i where data[i] > scalar; return count.
// Branchless selection-vector emit (Pirk DaMoN 2014).
// ---------------------------------------------------------------------------
inline int64_t interval_filter_gt(
        const IntervalDate* BOLT_RESTRICT data,
        int64_t                           n,
        IntervalDate                      scalar,
        int32_t*            BOLT_RESTRICT sel_out) noexcept {
    assert(data    != nullptr || n == 0);
    assert(sel_out != nullptr || n == 0);
    assert(n >= 0);
    const int64_t s = interval_to_days_for_cmp(scalar);
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t d = interval_to_days_for_cmp(data[i]);
        sel_out[count] = static_cast<int32_t>(i);
        count += (d > s) ? 1 : 0;
    }
    return count;
}

} // namespace interval
} // namespace kernels
} // namespace bolt
