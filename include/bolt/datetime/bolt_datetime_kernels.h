// bolt::dtk — column-at-a-time date/time kernels (G2LAUNCH-74).
//
// Every kernel runs over a contiguous [0, n) slice and allocates nothing.
// Zone lookups go through cursors that cache the span over which the UTC
// offset is constant (bolt::tz::offset_span): a run of nearby instants — the
// normal shape of a time column — pays one binary search per span, not per
// row, and the per-row work is an add plus a predictable range check.
// Calendar decomposition is Hinnant's branch-free civil_from_days.
//
// Kernels that can meet an input with no value (text that does not parse)
// return false at the first such row; callers fall back to their per-row
// path, which owns NULL / ANSI-error semantics.
#pragma once

#include <cstdint>

#include "bolt/bolt_types.h"
#include "bolt/datetime/bolt_calendar.h"
#include "bolt/datetime/bolt_calendar_arith.h"
#include "bolt/datetime/bolt_datetime_fmt.h"
#include "bolt/datetime/bolt_tz.h"

namespace bolt::dtk {

// Field codes; numbering is shared with chukonu's canonical DtField.
enum class Field : std::uint8_t {
    Year = 1, Month, Day, Hour, Minute, Second, DayOfWeek /* 1 = Sunday */,
    WeekDay /* 0 = Monday */, DayOfYear, WeekOfYear /* ISO */, Quarter,
    YearOfWeek /* ISO week-numbering year */, DowIso /* 1 = Monday */,
    SecondMicros /* second * 1e6 + micros: Spark's decimal(8,6) SECOND */,
};

// UTC seconds -> offset, cached over the constant-offset span.
struct UtcCursor {
    tz::ZoneId   zone;
    std::int64_t lo;
    std::int64_t hi;
    std::int32_t off;
};

inline void utc_cursor_init(UtcCursor* c, tz::ZoneId z) noexcept {
    c->zone = z;
    c->lo = 1;
    c->hi = 0;   // empty: the first lookup refreshes
    c->off = 0;
}

inline std::int32_t utc_cursor_offset(UtcCursor* c, std::int64_t utc_sec) noexcept {
    if (utc_sec < c->lo || utc_sec >= c->hi) {
        tz::OffsetSpan s{};
        tz::offset_span(c->zone, utc_sec, &s);
        c->lo = s.lo;
        c->hi = s.hi;
        c->off = s.off;
    }
    return c->off;
}

// Local seconds -> UTC, cached over the local window that maps through one
// offset unambiguously (no gap, no overlap); anything else takes the exact
// java.time resolution in bolt::tz::local_to_utc.
struct LocalCursor {
    tz::ZoneId   zone;
    std::int64_t safe_lo;
    std::int64_t safe_hi;
    std::int32_t off;
};

void local_cursor_init(LocalCursor* c, tz::ZoneId z) noexcept;
std::int64_t local_cursor_refresh(LocalCursor* c, std::int64_t local_sec) noexcept;

inline std::int64_t local_cursor_to_utc(LocalCursor* c, std::int64_t local_sec) noexcept {
    if (local_sec >= c->safe_lo && local_sec < c->safe_hi) return local_sec - c->off;
    return local_cursor_refresh(c, local_sec);
}

// ---- column kernels -------------------------------------------------------

void utc_to_local_us(tz::ZoneId z, const std::int64_t* BOLT_RESTRICT in, std::int64_t n,
                     std::int64_t* BOLT_RESTRICT out) noexcept;
void local_to_utc_us(tz::ZoneId z, const std::int64_t* BOLT_RESTRICT in, std::int64_t n,
                     std::int64_t* BOLT_RESTRICT out) noexcept;
// Local calendar date of each instant.
void utc_us_to_days(tz::ZoneId z, const std::int64_t* BOLT_RESTRICT in, std::int64_t n,
                    std::int32_t* BOLT_RESTRICT out) noexcept;
// Instant of each date's local midnight.
void days_to_utc_us(tz::ZoneId z, const std::int32_t* BOLT_RESTRICT in, std::int64_t n,
                    std::int64_t* BOLT_RESTRICT out) noexcept;
void trunc_utc_us(tz::ZoneId z, cal::TruncUnit u, const std::int64_t* BOLT_RESTRICT in,
                  std::int64_t n, std::int64_t* BOLT_RESTRICT out) noexcept;
void trunc_days(cal::TruncUnit u, const std::int32_t* BOLT_RESTRICT in, std::int64_t n,
                std::int32_t* BOLT_RESTRICT out) noexcept;
// Field of local wall-clock micros (a DATE is days * kMicrosPerDay).
// timestampadd over columns: out[i] = in[i] + qty[i] of `u`, in zone `z` (instants) or
// as a wall clock (_local). Return false on the first overflow (Spark: DATETIME_OVERFLOW).
bool timestamp_add_utc_us(tz::ZoneId z, cal::AddUnit u, const std::int64_t* BOLT_RESTRICT qty,
                          const std::int64_t* BOLT_RESTRICT in, std::int64_t n,
                          std::int64_t* BOLT_RESTRICT out) noexcept;
bool timestamp_add_local_us(cal::AddUnit u, const std::int64_t* BOLT_RESTRICT qty,
                            const std::int64_t* BOLT_RESTRICT in, std::int64_t n,
                            std::int64_t* BOLT_RESTRICT out) noexcept;
// timestampdiff over wall-clock micros columns (java LocalDateTime.until).
bool timestamp_diff_local_us(cal::AddUnit u, const std::int64_t* BOLT_RESTRICT a,
                             const std::int64_t* BOLT_RESTRICT b, std::int64_t n,
                             std::int64_t* BOLT_RESTRICT out) noexcept;

void field_local_us(Field f, const std::int64_t* BOLT_RESTRICT local, std::int64_t n,
                    std::int64_t* BOLT_RESTRICT out) noexcept;
void field_days(Field f, const std::int32_t* BOLT_RESTRICT days, std::int64_t n,
                std::int64_t* BOLT_RESTRICT out) noexcept;
// Spark CAST(text AS DATE); false at the first row with no value.
bool text_to_days(const StringView* BOLT_RESTRICT sv, const char* base, std::int64_t n,
                  std::int32_t* BOLT_RESTRICT out) noexcept;

// ---- formatting -------------------------------------------------------------

// Patterns made only of fixed-width numbers (yyyy MM dd HH mm ss S..S, plus
// literals) render through a digit-pair table with no per-token dispatch;
// format_fixed_width() is their exact output length, 0 when not eligible.
std::uint32_t format_fixed_width(const dtfmt::Pattern& p) noexcept;
void format_fixed(const dtfmt::Pattern& p, std::int64_t local_us, char* BOLT_RESTRICT out) noexcept;

}  // namespace bolt::dtk
