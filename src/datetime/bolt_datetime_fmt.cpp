// bolt::dtfmt — pattern compile / format. Parsing lives in
// bolt_datetime_parse.cpp. See include/bolt/datetime/bolt_datetime_fmt.h.
#include "bolt/datetime/bolt_datetime_fmt.h"

#include <cassert>
#include <cstring>

#include "bolt_datetime_internal.h"

namespace bolt::dtfmt {
namespace detail {

const char* const kMonthLong[12] = {"January", "February", "March", "April", "May", "June",
                                    "July", "August", "September", "October", "November",
                                    "December"};
const char* const kDowLong[7] = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                                 "Saturday", "Sunday"};

std::uint8_t letter_max(char c) noexcept {
    switch (c) {
        case 'G': return 5;
        case 'y': return 10;
        case 'M': case 'L': case 'Q': case 'q': case 'E': case 'X': case 'x': case 'Z': return 5;
        case 'd': case 'h': case 'K': case 'k': case 'H': case 'm': case 's': case 'V': return 2;
        case 'D': return 3;
        case 'a': return 1;
        case 'S': return 9;
        case 'z': case 'O': return 4;
        default: return 0;
    }
}

}  // namespace detail

namespace {

using detail::Buf;

bool push_literal(Pattern* p, char c, std::uint8_t optional) noexcept {
    assert(p != nullptr);
    if (p->lit_used >= kMaxLiteralBytes) return false;
    Token* last = p->n > 0 ? &p->tok[p->n - 1] : nullptr;
    const bool extend = last != nullptr && last->letter == 0 && last->optional == optional &&
                        last->lit_off + last->lit_len == p->lit_used && last->lit_len < 255;
    if (!extend) {
        if (p->n >= kMaxTokens) return false;
        p->tok[p->n++] = Token{0, 0, optional, 0, p->lit_used};
        last = &p->tok[p->n - 1];
    }
    p->lit[p->lit_used++] = c;
    ++last->lit_len;
    return true;
}

CompileStatus compile_quote(const char* s, std::uint32_t len, std::uint32_t* i, Pattern* p,
                            std::uint8_t optional) noexcept {
    assert(s[*i] == '\'');
    std::uint32_t j = *i + 1;
    if (j < len && s[j] == '\'') {   // '' is a literal quote
        *i = j + 1;
        return push_literal(p, '\'', optional) ? CompileStatus::Ok : CompileStatus::TooManyTokens;
    }
    for (; j < len; ++j) {
        if (s[j] == '\'') {
            if (j + 1 < len && s[j + 1] == '\'') {
                if (!push_literal(p, '\'', optional)) return CompileStatus::TooManyTokens;
                ++j;
                continue;
            }
            *i = j + 1;
            return CompileStatus::Ok;
        }
        if (!push_literal(p, s[j], optional)) return CompileStatus::TooManyTokens;
    }
    return CompileStatus::UnterminatedQuote;
}

CompileStatus compile_letter(const char* s, std::uint32_t len, std::uint32_t* i, Pattern* p,
                             std::uint8_t optional) noexcept {
    const char c = s[*i];
    std::uint32_t j = *i;
    while (j < len && s[j] == c && j - *i < 255) ++j;
    const std::uint32_t count = j - *i;
    *i = j;
    const std::uint8_t mx = detail::letter_max(c);
    if (mx == 0) return CompileStatus::IllegalLetter;
    if (count > mx) return CompileStatus::TooManyLetters;
    if ((c == 'V' && count != 2) || (c == 'O' && count != 1 && count != 4))
        return CompileStatus::TooManyLetters;
    if (p->n >= kMaxTokens) return CompileStatus::TooManyTokens;
    p->tok[p->n++] = Token{c, static_cast<std::uint8_t>(count), optional, 0, 0};
    return CompileStatus::Ok;
}

}  // namespace

CompileStatus compile(const char* s, std::uint32_t len, Pattern* out) noexcept {
    assert(out != nullptr);
    assert(s != nullptr || len == 0);
    out->n = 0;
    out->lit_used = 0;
    if (len > kMaxPatternBytes) return CompileStatus::TooLong;
    std::uint8_t optional = 0;
    std::uint32_t i = 0;
    for (std::uint32_t guard = 0; i < len && guard < kMaxPatternBytes * 2; ++guard) {
        const char c = s[i];
        CompileStatus st = CompileStatus::Ok;
        if (c == '\'') {
            st = compile_quote(s, len, &i, out, optional);
        } else if (c == '[') {
            if (optional != 0) return CompileStatus::UnbalancedSection;
            optional = 1;
            ++i;
        } else if (c == ']') {
            if (optional == 0) return CompileStatus::UnbalancedSection;
            optional = 0;
            ++i;
        } else if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
            st = compile_letter(s, len, &i, out, optional);
        } else {
            st = push_literal(out, c, optional) ? CompileStatus::Ok : CompileStatus::TooManyTokens;
            ++i;
        }
        if (st != CompileStatus::Ok) return st;
    }
    if (optional != 0) return CompileStatus::UnbalancedSection;
    assert(i == len);
    return CompileStatus::Ok;
}

const char* compile_status_text(CompileStatus st) noexcept {
    switch (st) {
        case CompileStatus::Ok: return "ok";
        case CompileStatus::TooLong: return "pattern longer than 128 bytes";
        case CompileStatus::TooManyTokens: return "pattern has more than 64 fields";
        case CompileStatus::UnterminatedQuote: return "unterminated quote in pattern";
        case CompileStatus::UnbalancedSection: return "unbalanced or nested [ ] section";
        case CompileStatus::IllegalLetter:
            return "illegal pattern letter (week-based and unsupported letters are rejected)";
        case CompileStatus::TooManyLetters: return "too many pattern letters";
    }
    return "unknown";
}

namespace {

// +HH[MM[SS]] / +HH[:MM[:SS]]: minutes always or only when non-zero,
// seconds only when non-zero and allowed.
bool put_offset(Buf* b, std::int32_t off, bool colon, bool minutes_always,
                bool seconds_allowed) noexcept {
    const std::int32_t a = off < 0 ? -off : off;
    const std::int32_t h = a / 3600, m = (a / 60) % 60, s = a % 60;
    bool ok = b->ch(off < 0 ? '-' : '+') && b->num(h, 2);
    const bool secs = seconds_allowed && s != 0;
    if (minutes_always || m != 0 || secs) ok = ok && (!colon || b->ch(':')) && b->num(m, 2);
    if (secs) ok = ok && (!colon || b->ch(':')) && b->num(s, 2);
    return ok;
}

bool put_localized_offset(Buf* b, std::int32_t off, bool full) noexcept {
    if (!b->str("GMT", 3)) return false;
    if (off == 0) return true;
    const std::int32_t a = off < 0 ? -off : off;
    const std::int32_t h = a / 3600, m = (a / 60) % 60, s = a % 60;
    bool ok = b->ch(off < 0 ? '-' : '+') && b->num(h, full ? 2 : 1);
    if (full || m != 0 || s != 0) ok = ok && b->ch(':') && b->num(m, 2);
    if (s != 0) ok = ok && b->ch(':') && b->num(s, 2);
    return ok;
}

bool put_zone(Buf* b, const Token& t, tz::ZoneId zone, std::int32_t off, std::int64_t utc_sec) noexcept {
    char tmp[tz::kMaxZoneIdChars];
    switch (t.letter) {
        case 'V': {
            const std::uint32_t n = tz::zone_name(zone, tmp, sizeof(tmp));
            return n != 0 && b->str(tmp, n);
        }
        case 'z': {
            const std::uint32_t n = t.count == 4 ? tz::zone_name(zone, tmp, sizeof(tmp))
                                                 : tz::zone_abbr(zone, utc_sec, tmp, sizeof(tmp));
            return n != 0 && b->str(tmp, n);
        }
        case 'O': return put_localized_offset(b, off, t.count == 4);
        case 'X': case 'x': {
            if (t.letter == 'X' && off == 0) return b->ch('Z');
            const std::uint8_t c = t.count;
            return put_offset(b, off, c == 3 || c == 5, c != 1, c >= 4);
        }
        case 'Z':
            if (t.count <= 3) return put_offset(b, off, false, true, false);
            if (t.count == 4) return put_localized_offset(b, off, true);
            if (off == 0) return b->ch('Z');
            return put_offset(b, off, true, true, true);
        default: return false;
    }
}

bool put_text_field(Buf* b, const Token& t, const cal::LocalFields& f, std::int32_t dow) noexcept {
    const std::uint8_t c = t.count;
    switch (t.letter) {
        case 'M': case 'L': {
            if (c <= 2) return b->num(f.month, c);
            const char* name = detail::kMonthLong[f.month - 1];
            if (c == 3) return b->str(name, 3);
            if (c == 5) return b->ch(name[0]);
            return b->str(name, static_cast<std::uint32_t>(std::strlen(name)));
        }
        case 'E': {
            const char* name = detail::kDowLong[dow - 1];
            if (c <= 3) return b->str(name, 3);
            if (c == 5) return b->ch(name[0]);
            return b->str(name, static_cast<std::uint32_t>(std::strlen(name)));
        }
        case 'Q': case 'q': {
            const std::int32_t q = (f.month - 1) / 3 + 1;
            if (c <= 2) return b->num(q, c);
            if (c == 3) return b->ch('Q') && b->num(q, 1);
            static const char* const kOrd[4] = {"1st", "2nd", "3rd", "4th"};
            return b->str(kOrd[q - 1], 3) && b->str(" quarter", 8);
        }
        case 'G': {
            const bool ad = f.year > 0;
            if (c <= 3) return b->str(ad ? "AD" : "BC", 2);
            if (c == 4) return ad ? b->str("Anno Domini", 11) : b->str("Before Christ", 13);
            return b->ch(ad ? 'A' : 'B');
        }
        case 'a': return b->str(f.hour < 12 ? "AM" : "PM", 2);
        default: return false;
    }
}

bool put_year(Buf* b, std::int32_t year, std::uint8_t c) noexcept {
    const std::int32_t yoe = year > 0 ? year : 1 - year;   // year-of-era
    if (c == 2) return b->num(yoe % 100, 2);
    if (c == 4 && yoe > 9999 && !b->ch('+')) return false;
    return b->num(yoe, c);
}

bool put_token(Buf* b, const Token& t, const Pattern& p, const cal::LocalFields& f,
               std::int32_t doy, std::int32_t dow, tz::ZoneId zone, std::int32_t off,
               std::int64_t utc_sec) noexcept {
    const std::uint8_t c = t.count;
    switch (t.letter) {
        case 0: return b->str(p.lit + t.lit_off, t.lit_len);
        case 'y': return put_year(b, f.year, c);
        case 'd': return b->num(f.day, c);
        case 'D': return b->num(doy, c);
        case 'H': return b->num(f.hour, c);
        case 'k': return b->num(f.hour == 0 ? 24 : f.hour, c);
        case 'K': return b->num(f.hour % 12, c);
        case 'h': return b->num(f.hour % 12 == 0 ? 12 : f.hour % 12, c);
        case 'm': return b->num(f.minute, c);
        case 's': return b->num(f.second, c);
        case 'S': {
            std::int64_t nanos = static_cast<std::int64_t>(f.micros) * 1000;
            char digs[9];
            for (int i = 8; i >= 0; --i) { digs[i] = static_cast<char>('0' + nanos % 10); nanos /= 10; }
            return b->str(digs, c);
        }
        case 'V': case 'z': case 'O': case 'X': case 'x': case 'Z':
            return put_zone(b, t, zone, off, utc_sec);
        default: return put_text_field(b, t, f, dow);
    }
}

}  // namespace

std::uint32_t format_instant(const Pattern& p, std::int64_t utc_us, tz::ZoneId zone,
                             char* buf, std::uint32_t cap) noexcept {
    assert(buf != nullptr);
    assert(p.n <= kMaxTokens);
    const std::int64_t utc_sec = tz::floor_div(utc_us, tz::kMicrosPerSec);
    const std::int32_t off = tz::offset_at(zone, utc_sec);
    const std::int64_t local = utc_us + static_cast<std::int64_t>(off) * tz::kMicrosPerSec;
    if (local / cal::kMicrosPerDay > cal::kMaxAbsDays || local / cal::kMicrosPerDay < -cal::kMaxAbsDays)
        return 0;
    cal::LocalFields f{};
    cal::fields_from_local_micros(local, &f);
    const std::int64_t days = tz::floor_div(local, cal::kMicrosPerDay);
    const std::int32_t doy = static_cast<std::int32_t>(days - cal::days_from_ymd(f.year, 1, 1)) + 1;
    const std::int32_t dow = cal::iso_dow(days);
    Buf b{buf, cap, 0};
    for (std::uint16_t i = 0; i < p.n; ++i) {
        if (!put_token(&b, p.tok[i], p, f, doy, dow, zone, off, utc_sec)) return 0;
    }
    assert(b.len <= cap);
    return b.len;
}

std::uint32_t format_date_default(std::int64_t days, char* buf, std::uint32_t cap) noexcept {
    assert(buf != nullptr);
    if (days > cal::kMaxAbsDays || days < -cal::kMaxAbsDays) return 0;
    std::int32_t y = 0, m = 0, d = 0;
    cal::ymd_from_days(days, &y, &m, &d);
    Buf b{buf, cap, 0};
    bool ok = true;
    if (y < 0) ok = b.ch('-');
    else if (y > 9999) ok = b.ch('+');
    ok = ok && b.num(y < 0 ? -y : y, 4) && b.ch('-') && b.num(m, 2) && b.ch('-') && b.num(d, 2);
    return ok ? b.len : 0;
}

std::uint32_t format_timestamp_default(std::int64_t local_us, char* buf, std::uint32_t cap) noexcept {
    assert(buf != nullptr);
    const std::int64_t days = tz::floor_div(local_us, cal::kMicrosPerDay);
    if (days > cal::kMaxAbsDays || days < -cal::kMaxAbsDays) return 0;
    const std::uint32_t dl = format_date_default(days, buf, cap);
    if (dl == 0) return 0;
    cal::LocalFields f{};
    cal::fields_from_local_micros(local_us, &f);
    Buf b{buf, cap, dl};
    bool ok = b.ch(' ') && b.num(f.hour, 2) && b.ch(':') && b.num(f.minute, 2) && b.ch(':') &&
              b.num(f.second, 2);
    if (ok && f.micros != 0) {
        char digs[6];
        std::int32_t us = f.micros;
        for (int i = 5; i >= 0; --i) { digs[i] = static_cast<char>('0' + us % 10); us /= 10; }
        std::uint32_t n = 6;
        while (n > 1 && digs[n - 1] == '0') --n;
        ok = b.ch('.') && b.str(digs, n);
    }
    return ok ? b.len : 0;
}

}  // namespace bolt::dtfmt
