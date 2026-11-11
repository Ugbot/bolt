// bolt::dtfmt — Spark / java.time datetime patterns ("yyyy-MM-dd HH:mm:ss")
// and Spark's string -> DATE / TIMESTAMP cast grammar.
//
// A pattern is compiled once into a fixed-capacity token array, then used to
// format or parse any number of values without allocation. Letters follow
// Spark 3+'s "Datetime Patterns" page: G y u M L d D Q q E a h K k H m s S V
// z O X x Z, quoting with '...' ('' is a quote) and optional sections [...].
// Week-based letters (Y w W e c F) are rejected by name, as Spark rejects
// them. Parsing is ASCII case-insensitive and must consume the whole input.
#pragma once

#include <cstdint>

#include "bolt/datetime/bolt_calendar.h"
#include "bolt/datetime/bolt_tz.h"

namespace bolt::dtfmt {

inline constexpr std::uint32_t kMaxPatternBytes = 128;
inline constexpr std::uint32_t kMaxTokens = 64;
inline constexpr std::uint32_t kMaxLiteralBytes = 128;
inline constexpr std::uint32_t kMaxFormattedBytes = 256;

struct Token {
    char          letter;      // 0 = literal run
    std::uint8_t  count;
    std::uint8_t  optional;    // 1 inside a [...] section
    std::uint8_t  lit_len;
    std::uint16_t lit_off;
};

struct Pattern {
    Token         tok[kMaxTokens];
    char          lit[kMaxLiteralBytes];
    std::uint16_t n;
    std::uint16_t lit_used;
};

enum class CompileStatus : std::uint8_t {
    Ok = 0, TooLong, TooManyTokens, UnterminatedQuote, UnbalancedSection,
    IllegalLetter, TooManyLetters,
};

CompileStatus compile(const char* s, std::uint32_t len, Pattern* out) noexcept;
const char* compile_status_text(CompileStatus st) noexcept;

// Format the instant `utc_us` as seen in `zone`. Returns the byte length, or
// 0 if `cap` is too small or the value is out of range.
std::uint32_t format_instant(const Pattern& p, std::int64_t utc_us, tz::ZoneId zone,
                             char* buf, std::uint32_t cap) noexcept;

// Parse `s` with pattern `p`. A zone/offset in the text wins over `zone`, unless
// `ignore_text_zone` (TIMESTAMP_NTZ: the zone is read and dropped, the result is
// the wall clock as epoch micros when `zone` is UTC).
// Missing fields default as Spark does (1970-01-01 00:00:00).
bool parse_instant(const Pattern& p, const char* s, std::uint32_t len, tz::ZoneId zone,
                   std::int64_t* utc_us, bool ignore_text_zone = false) noexcept;

// Spark CAST(string AS DATE): "[+-]yyyy[-[m]m[-[d]d[( |T)...]]]", trimmed.
bool parse_date_text(const char* s, std::uint32_t len, std::int64_t* days) noexcept;

// Spark CAST(string AS TIMESTAMP): date part as above, then optional
// "( |T)[h]h[:[m]m[:[s]s[.fraction]]]" and an optional zone id / offset.
// `ignore_text_zone` as parse_instant (CAST(string AS TIMESTAMP_NTZ)).
bool parse_timestamp_text(const char* s, std::uint32_t len, tz::ZoneId zone,
                          std::int64_t* utc_us, bool ignore_text_zone = false) noexcept;

// "yyyy-MM-dd" / "yyyy-MM-dd HH:mm:ss[.ffffff]" (Spark's default rendering).
std::uint32_t format_date_default(std::int64_t days, char* buf, std::uint32_t cap) noexcept;
std::uint32_t format_timestamp_default(std::int64_t local_us, char* buf, std::uint32_t cap) noexcept;

}  // namespace bolt::dtfmt
