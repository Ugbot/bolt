// bolt::tz — IANA time-zone rules, embedded (no system tzdata, no library).
//
// The database is generated from compiled TZif files by
// scripts/gen_tzdb.py into src/datetime/bolt_tzdb_data.inc: per zone a
// transition table (UTC seconds -> local-time type) plus the POSIX rule that
// governs every instant after the last transition. All tables are constexpr
// arrays; nothing allocates and every lookup is a bounded binary search.
//
// A ZoneId is a 32-bit handle, cheap to carry in an expression node:
//   * bit 31 clear: (index into the sorted zone-name table) + 1, so the
//     region id the caller named can be printed back ('VV' pattern letter);
//   * bit 31 set:   a fixed offset, (seconds + kFixedBias) in the low bits.
// kUtc (fixed +00:00) is the zero-offset handle.
//
// Local -> UTC follows java.time.ZonedDateTime.ofLocal (what Spark uses): a
// local time inside a gap moves forward by the gap's length; a local time
// inside an overlap takes the earlier offset.
#pragma once

#include <cstdint>

namespace bolt::tz {

using ZoneId = std::uint32_t;

inline constexpr std::uint32_t kFixedFlag = 0x80000000u;
inline constexpr std::int32_t  kFixedBias = 1 << 20;
inline constexpr std::int32_t  kMaxFixedOffsetSec = 18 * 3600;   // java.time bound
inline constexpr ZoneId        kUtc = kFixedFlag | static_cast<std::uint32_t>(kFixedBias);
inline constexpr ZoneId        kInvalidZone = 0;
inline constexpr std::uint32_t kMaxZoneIdChars = 64;

constexpr bool zone_is_fixed(ZoneId z) noexcept { return (z & kFixedFlag) != 0; }

constexpr ZoneId zone_fixed(std::int32_t offset_sec) noexcept {
    return kFixedFlag | static_cast<std::uint32_t>(offset_sec + kFixedBias);
}

constexpr std::int32_t zone_fixed_offset(ZoneId z) noexcept {
    return static_cast<std::int32_t>(z & ~kFixedFlag) - kFixedBias;
}

// Parse a Spark / java.time zone id: a region ("America/New_York", exact,
// then ASCII case-insensitive), a java SHORT_ID alias ("PST"), "UTC" / "Z" /
// "GMT" / "UT", or an offset "+08:00", "-0530", "+8", "GMT+8", "UTC-05:00".
// Returns false (and *out = kInvalidZone) on anything else.
bool zone_parse(const char* s, std::uint32_t len, ZoneId* out) noexcept;

// UTC offset in seconds in effect at `utc_sec` (seconds since the epoch).
std::int32_t offset_at(ZoneId z, std::int64_t utc_sec) noexcept;

// The instant whose local wall clock in `z` is `local_sec` (java.time rules).
std::int64_t local_to_utc(ZoneId z, std::int64_t local_sec) noexcept;

// Writes the zone id text ("America/New_York", "+08:00", "UTC") into buf.
// Returns the length, or 0 if `cap` is too small.
std::uint32_t zone_name(ZoneId z, char* buf, std::uint32_t cap) noexcept;

// Writes the short abbreviation in effect at `utc_sec` ("EST", "+08", "UTC").
std::uint32_t zone_abbr(ZoneId z, std::int64_t utc_sec, char* buf,
                        std::uint32_t cap) noexcept;

const char* tzdb_version() noexcept;

// ---- microsecond helpers (Spark TimestampType is epoch micros) ---------

inline constexpr std::int64_t kMicrosPerSec = 1000000;

constexpr std::int64_t floor_div(std::int64_t a, std::int64_t b) noexcept {
    return (a / b) - ((a % b != 0) && ((a < 0) != (b < 0)) ? 1 : 0);
}

inline std::int64_t utc_micros_to_local(ZoneId z, std::int64_t us) noexcept {
    const std::int64_t sec = floor_div(us, kMicrosPerSec);
    return us + static_cast<std::int64_t>(offset_at(z, sec)) * kMicrosPerSec;
}

inline std::int64_t local_micros_to_utc(ZoneId z, std::int64_t local_us) noexcept {
    const std::int64_t sec = floor_div(local_us, kMicrosPerSec);
    const std::int64_t frac = local_us - sec * kMicrosPerSec;
    return local_to_utc(z, sec) * kMicrosPerSec + frac;
}

}  // namespace bolt::tz
