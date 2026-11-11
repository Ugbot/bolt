// bolt::dtfmt — parsing with a compiled pattern, and Spark's string -> DATE /
// TIMESTAMP cast grammar. See include/bolt/datetime/bolt_datetime_fmt.h.
#include <cassert>
#include <cstring>

#include "bolt/datetime/bolt_datetime_fmt.h"
#include "bolt_datetime_internal.h"

namespace bolt::dtfmt {
namespace {

constexpr std::int32_t kUnset = -1;

struct Parsed {
    std::int32_t year = kUnset, yoe = kUnset, month = kUnset, day = kUnset, doy = kUnset;
    std::int32_t hour24 = kUnset, hour12 = kUnset, ampm = kUnset, minute = 0, second = 0;
    std::int32_t micros = 0, dow = kUnset, era = 1;
    bool has_zone = false;
    tz::ZoneId zone = tz::kUtc;
};

struct Cursor {
    const char*   s;
    std::uint32_t len;
    std::uint32_t pos;
};

char lower(char c) noexcept { return (c >= 'A' && c <= 'Z') ? static_cast<char>(c + 32) : c; }
bool is_digit(char c) noexcept { return c >= '0' && c <= '9'; }

bool read_digits(Cursor* c, std::uint32_t min_w, std::uint32_t max_w, std::int64_t* out) noexcept {
    assert(c != nullptr && out != nullptr);
    assert(min_w >= 1 && max_w <= 18);
    std::uint32_t n = 0;
    std::int64_t v = 0;
    while (n < max_w && c->pos + n < c->len && is_digit(c->s[c->pos + n])) {
        v = v * 10 + (c->s[c->pos + n] - '0');
        ++n;
    }
    if (n < min_w) return false;
    c->pos += n;
    *out = v;
    return true;
}

bool match_ci(Cursor* c, const char* word, std::uint32_t n) noexcept {
    if (n > c->len - c->pos) return false;
    for (std::uint32_t i = 0; i < n; ++i) {
        if (lower(c->s[c->pos + i]) != lower(word[i])) return false;
    }
    c->pos += n;
    return true;
}

// Index (0-based) of the name in `names` matched at the cursor, full or 3-letter.
std::int32_t match_name(Cursor* c, const char* const* names, std::int32_t count, bool full) noexcept {
    for (std::int32_t i = 0; i < count; ++i) {
        const std::uint32_t n = full ? static_cast<std::uint32_t>(std::strlen(names[i])) : 3u;
        if (match_ci(c, names[i], n)) return i;
    }
    return kUnset;
}

// [+-]HH[[:]MM[[:]SS]] or 'Z' / "GMT" / "UTC" with an optional offset.
bool read_offset(Cursor* c, std::int32_t* off) noexcept {
    assert(off != nullptr);
    if (c->pos < c->len && (c->s[c->pos] == 'Z' || c->s[c->pos] == 'z')) {
        ++c->pos;
        *off = 0;
        return true;
    }
    const bool gmt = match_ci(c, "GMT", 3) || match_ci(c, "UTC", 3);
    if (c->pos >= c->len || (c->s[c->pos] != '+' && c->s[c->pos] != '-')) {
        *off = 0;
        return gmt;
    }
    const std::int32_t sign = c->s[c->pos] == '-' ? -1 : 1;
    ++c->pos;
    std::int64_t h = 0, m = 0, s = 0;
    if (!read_digits(c, 1, 2, &h)) return false;
    const bool colon = c->pos < c->len && c->s[c->pos] == ':';
    if (colon) ++c->pos;
    if (c->pos < c->len && is_digit(c->s[c->pos])) {
        if (!read_digits(c, 2, 2, &m)) return false;
        if (c->pos < c->len && c->s[c->pos] == ':' && colon) ++c->pos;
        if (c->pos < c->len && is_digit(c->s[c->pos]) && !read_digits(c, 2, 2, &s)) return false;
    } else if (colon) {
        return false;
    }
    if (h > 18 || m > 59 || s > 59) return false;
    *off = sign * static_cast<std::int32_t>(h * 3600 + m * 60 + s);
    return true;
}

bool read_zone_id(Cursor* c, tz::ZoneId* out) noexcept {
    const std::uint32_t start = c->pos;
    std::uint32_t n = 0;
    while (start + n < c->len && n < tz::kMaxZoneIdChars) {
        const char ch = c->s[start + n];
        const bool ok = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || is_digit(ch) ||
                        ch == '/' || ch == '_' || ch == '+' || ch == '-' || ch == ':';
        if (!ok) break;
        ++n;
    }
    if (n == 0 || !tz::zone_parse(c->s + start, n, out)) return false;
    c->pos = start + n;
    return true;
}

bool parse_zone_token(Cursor* c, const Token& t, Parsed* p) noexcept {
    std::int32_t off = 0;
    if (t.letter == 'V' || t.letter == 'z') {
        if (!read_zone_id(c, &p->zone)) return false;
    } else {
        if (!read_offset(c, &off)) return false;
        p->zone = tz::zone_fixed(off);
    }
    p->has_zone = true;
    return true;
}

// `adjacent`: a numeric field follows with no literal between, so a
// variable-width year takes exactly its pattern width (java "adjacent value
// parsing": yyyyMMdd).
bool parse_number(Cursor* c, const Token& t, Parsed* p, bool adjacent = false) noexcept {
    const std::uint8_t n = t.count;
    std::int64_t v = 0;
    std::uint32_t lo = n, hi = n;
    if (n == 1) { lo = 1; hi = t.letter == 'y' ? 10 : (t.letter == 'D' ? 3 : 2); }
    if (t.letter == 'y' && n >= 3) { lo = n; hi = 10; }
    if (t.letter == 'y' && n != 2 && adjacent) { lo = n < 4 ? 4 : n; hi = lo; }
    if (t.letter == 'y' && n == 4 && c->pos < c->len && c->s[c->pos] == '+') ++c->pos;
    if (!read_digits(c, lo, hi, &v)) return false;
    const std::int32_t x = static_cast<std::int32_t>(v);
    switch (t.letter) {
        case 'y': p->yoe = n == 2 ? 2000 + x : x; return true;
        case 'M': case 'L': p->month = x; return true;
        case 'd': p->day = x; return true;
        case 'D': p->doy = x; return true;
        case 'H': p->hour24 = x; return x <= 23;
        case 'k': p->hour24 = x == 24 ? 0 : x; return x >= 1 && x <= 24;
        case 'K': p->hour12 = x; return x <= 11;
        case 'h': p->hour12 = x == 12 ? 0 : x; return x >= 1 && x <= 12;
        case 'm': p->minute = x; return x <= 59;
        case 's': p->second = x; return x <= 59;
        case 'Q': case 'q': return x >= 1 && x <= 4;
        default: return false;
    }
}

bool parse_fraction(Cursor* c, const Token& t, Parsed* p) noexcept {
    std::int64_t v = 0;
    const std::uint32_t start = c->pos;
    if (!read_digits(c, 1, t.count, &v)) return false;
    std::uint32_t digits = c->pos - start;
    while (digits < 6) { v *= 10; ++digits; }
    while (digits > 6) { v /= 10; --digits; }
    p->micros = static_cast<std::int32_t>(v);
    return true;
}

bool parse_text(Cursor* c, const Token& t, Parsed* p) noexcept {
    const std::uint8_t n = t.count;
    switch (t.letter) {
        case 'M': case 'L': {
            if (n <= 2) return parse_number(c, t, p);
            const std::int32_t i = match_name(c, detail::kMonthLong, 12, n == 4);
            if (i == kUnset) return false;
            p->month = i + 1;
            return true;
        }
        case 'E': {
            const std::int32_t i = match_name(c, detail::kDowLong, 7, n == 4);
            if (i == kUnset) return false;
            p->dow = i + 1;
            return true;
        }
        case 'a':
            if (match_ci(c, "AM", 2)) { p->ampm = 0; return true; }
            if (match_ci(c, "PM", 2)) { p->ampm = 1; return true; }
            return false;
        case 'G':
            if (match_ci(c, n == 4 ? "Anno Domini" : "AD", n == 4 ? 11 : 2)) { p->era = 1; return true; }
            if (match_ci(c, n == 4 ? "Before Christ" : "BC", n == 4 ? 13 : 2)) { p->era = 0; return true; }
            return false;
        case 'Q': case 'q':
            if (n <= 2) return parse_number(c, t, p);
            if (n == 3) return match_ci(c, "Q", 1) && parse_number(c, Token{'Q', 1, 0, 0, 0}, p);
            return false;
        default: return parse_number(c, t, p);
    }
}

bool numeric_letter(const Token& t) noexcept {
    switch (t.letter) {
        case 'd': case 'D': case 'H': case 'h': case 'k': case 'K': case 'm': case 's': case 'S':
        case 'y': return true;
        case 'M': case 'L': return t.count <= 2;
        default: return false;
    }
}

bool parse_token(Cursor* c, const Token& t, const Pattern& pat, Parsed* p,
                 const Token* next) noexcept {
    if (t.letter == 'y') return parse_number(c, t, p, next != nullptr && numeric_letter(*next));
    switch (t.letter) {
        case 0: return match_ci(c, pat.lit + t.lit_off, t.lit_len);
        case 'S': return parse_fraction(c, t, p);
        case 'V': case 'z': case 'O': case 'X': case 'x': case 'Z': return parse_zone_token(c, t, p);
        default: return parse_text(c, t, p);
    }
}

bool resolve(const Parsed& p, tz::ZoneId zone, std::int64_t* utc_us, bool ignore_text_zone) noexcept {
    cal::LocalFields f{};
    const std::int32_t yoe = p.yoe == kUnset ? 1970 : p.yoe;
    f.year = p.era == 1 ? yoe : 1 - yoe;
    if (p.doy != kUnset && p.month == kUnset && p.day == kUnset) {
        if (p.doy < 1 || p.doy > (cal::is_leap(f.year) ? 366 : 365)) return false;
        const std::int64_t days = cal::days_from_ymd(f.year, 1, 1) + p.doy - 1;
        cal::ymd_from_days(days, &f.year, &f.month, &f.day);
    } else {
        f.month = p.month == kUnset ? 1 : p.month;
        f.day = p.day == kUnset ? 1 : p.day;
        if (f.month < 1 || f.month > 12 || f.day < 1 || f.day > cal::month_days(f.year, f.month))
            return false;
    }
    if (p.hour24 != kUnset) f.hour = p.hour24;
    else if (p.hour12 != kUnset) f.hour = p.hour12 + (p.ampm == 1 ? 12 : 0);
    else f.hour = 0;
    f.minute = p.minute;
    f.second = p.second;
    f.micros = p.micros;
    const std::int64_t days = cal::days_from_ymd(f.year, f.month, f.day);
    if (p.dow != kUnset && cal::iso_dow(days) != p.dow) return false;
    *utc_us = tz::local_micros_to_utc(p.has_zone && !ignore_text_zone ? p.zone : zone,
                                         cal::local_micros_from_fields(f));
    return true;
}

// End index (exclusive) of the optional section starting at token i.
std::uint16_t section_end(const Pattern& pat, std::uint16_t i) noexcept {
    while (i < pat.n && pat.tok[i].optional != 0) ++i;
    return i;
}

}  // namespace

bool parse_instant(const Pattern& pat, const char* s, std::uint32_t len, tz::ZoneId zone,
                   std::int64_t* utc_us, bool ignore_text_zone) noexcept {
    assert(utc_us != nullptr);
    assert(pat.n <= kMaxTokens);
    Cursor c{s, len, 0};
    Parsed p{};
    std::uint16_t i = 0;
    for (std::uint32_t guard = 0; i < pat.n && guard < kMaxTokens * 2; ++guard) {
        const Token& t = pat.tok[i];
        if (t.optional == 0) {
            const Token* next = i + 1 < pat.n ? &pat.tok[i + 1] : nullptr;
            if (!parse_token(&c, t, pat, &p, next)) return false;
            ++i;
            continue;
        }
        const std::uint16_t end = section_end(pat, i);
        const Cursor saved = c;
        const Parsed saved_p = p;
        bool ok = true;
        for (std::uint16_t k = i; k < end && ok; ++k) {
            const Token* next = k + 1 < end ? &pat.tok[k + 1] : nullptr;
            ok = parse_token(&c, pat.tok[k], pat, &p, next);
        }
        if (!ok) { c = saved; p = saved_p; }
        i = end;
    }
    if (c.pos != len) return false;
    return resolve(p, zone, utc_us, ignore_text_zone);
}

namespace {

bool is_space(char ch) noexcept { return static_cast<unsigned char>(ch) <= ' '; }

void trim(const char** s, std::uint32_t* len) noexcept {
    while (*len > 0 && is_space((*s)[0])) { ++*s; --*len; }
    while (*len > 0 && is_space((*s)[*len - 1])) --*len;
}

// "[+-]yyyy*[-[m]m[-[d]d]]": stops at the first byte that is not part of it.
bool read_date_part(Cursor* c, std::uint32_t max_year_digits, std::int64_t* days) noexcept {
    std::int32_t sign = 1;
    if (c->pos < c->len && (c->s[c->pos] == '+' || c->s[c->pos] == '-')) {
        sign = c->s[c->pos] == '-' ? -1 : 1;
        ++c->pos;
    }
    std::int64_t y = 0, m = 1, d = 1;
    if (!read_digits(c, 4, max_year_digits, &y)) return false;
    if (c->pos < c->len && c->s[c->pos] == '-') {
        ++c->pos;
        if (!read_digits(c, 1, 2, &m)) return false;
        if (c->pos < c->len && c->s[c->pos] == '-') {
            ++c->pos;
            if (!read_digits(c, 1, 2, &d)) return false;
        }
    }
    const std::int64_t year = sign * y;
    if (m < 1 || m > 12 || d < 1 || d > cal::month_days(year, static_cast<std::int32_t>(m))) return false;
    *days = cal::days_from_ymd(year, static_cast<std::int32_t>(m), static_cast<std::int32_t>(d));
    return true;
}

}  // namespace

bool parse_date_text(const char* s, std::uint32_t len, std::int64_t* days) noexcept {
    assert(days != nullptr);
    assert(s != nullptr || len == 0);
    trim(&s, &len);
    Cursor c{s, len, 0};
    if (!read_date_part(&c, 7, days)) return false;
    if (c.pos == len) return true;
    return c.s[c.pos] == ' ' || c.s[c.pos] == 'T';   // anything may follow
}

namespace {

bool read_time_part(Cursor* c, cal::LocalFields* f) noexcept {
    std::int64_t h = 0, m = 0, sec = 0;
    if (!read_digits(c, 1, 2, &h)) return false;
    if (c->pos < c->len && c->s[c->pos] == ':') {
        ++c->pos;
        if (!read_digits(c, 1, 2, &m)) return false;
        if (c->pos < c->len && c->s[c->pos] == ':') {
            ++c->pos;
            if (!read_digits(c, 1, 2, &sec)) return false;
            if (c->pos < c->len && c->s[c->pos] == '.') {
                ++c->pos;
                std::int64_t frac = 0;
                const std::uint32_t start = c->pos;
                if (!read_digits(c, 1, 9, &frac)) return false;
                std::uint32_t digits = c->pos - start;
                while (digits < 6) { frac *= 10; ++digits; }
                while (digits > 6) { frac /= 10; --digits; }
                f->micros = static_cast<std::int32_t>(frac);
            }
        }
    }
    if (h > 23 || m > 59 || sec > 59) return false;
    f->hour = static_cast<std::int32_t>(h);
    f->minute = static_cast<std::int32_t>(m);
    f->second = static_cast<std::int32_t>(sec);
    return true;
}

}  // namespace

bool parse_timestamp_text(const char* s, std::uint32_t len, tz::ZoneId zone,
                          std::int64_t* utc_us, bool ignore_text_zone) noexcept {
    assert(utc_us != nullptr);
    assert(s != nullptr || len == 0);
    trim(&s, &len);
    Cursor c{s, len, 0};
    std::int64_t days = 0;
    if (!read_date_part(&c, 6, &days)) return false;
    cal::LocalFields f{};
    cal::ymd_from_days(days, &f.year, &f.month, &f.day);
    if (c.pos < len && (c.s[c.pos] == ' ' || c.s[c.pos] == 'T')) {
        ++c.pos;
        if (c.pos < len && is_digit(c.s[c.pos]) && !read_time_part(&c, &f)) return false;
    }
    tz::ZoneId z = zone;
    while (c.pos < len && c.s[c.pos] == ' ') ++c.pos;
    if (c.pos < len) {
        std::int32_t off = 0;
        Cursor probe = c;
        tz::ZoneId text_zone = tz::kInvalidZone;
        if (read_offset(&probe, &off) && probe.pos == len) {
            text_zone = tz::zone_fixed(off);
        } else if (!tz::zone_parse(c.s + c.pos, len - c.pos, &text_zone)) {
            return false;
        }
        if (!ignore_text_zone) z = text_zone;
    }
    *utc_us = tz::local_micros_to_utc(z, cal::local_micros_from_fields(f));
    return true;
}

}  // namespace bolt::dtfmt
