// bolt::dtk — column-at-a-time date/time kernels. See
// include/bolt/datetime/bolt_datetime_kernels.h.
#include "bolt/datetime/bolt_datetime_kernels.h"

#include <cassert>
#include <cstring>

#include "bolt/kernels/bolt_date.h"

namespace bolt::dtk {
namespace {

constexpr std::int64_t kUsPerSec = tz::kMicrosPerSec;
constexpr std::int64_t kUsPerDay = cal::kMicrosPerDay;
constexpr std::int64_t kSpanEdge = INT64_MAX / 4;   // saturating window bound

std::int64_t sat_add(std::int64_t a, std::int32_t b) noexcept {
    if (a >= kSpanEdge) return INT64_MAX;
    if (a <= -kSpanEdge) return INT64_MIN;
    return a + b;
}

// Floor division by a positive constant; the compiler strength-reduces it.
inline std::int64_t fdiv(std::int64_t a, std::int64_t m) noexcept {
    const std::int64_t q = a / m;
    return q - ((a % m) < 0 ? 1 : 0);
}

}  // namespace

void local_cursor_init(LocalCursor* c, tz::ZoneId z) noexcept {
    assert(c != nullptr);
    c->zone = z;
    c->safe_lo = 1;
    c->safe_hi = 0;
    c->off = 0;
}

std::int64_t local_cursor_refresh(LocalCursor* c, std::int64_t local_sec) noexcept {
    assert(c != nullptr);
    const std::int64_t utc = tz::local_to_utc(c->zone, local_sec);
    tz::OffsetSpan s{};
    tz::offset_span(c->zone, utc, &s);
    const std::int32_t prev = s.lo == INT64_MIN ? s.off : tz::offset_at(c->zone, s.lo - 1);
    const std::int32_t next = s.hi == INT64_MAX ? s.off : tz::offset_at(c->zone, s.hi);
    c->off = s.off;
    c->safe_lo = sat_add(s.lo, s.off > prev ? s.off : prev);
    c->safe_hi = sat_add(s.hi, s.off < next ? s.off : next);
    assert(c->safe_lo <= c->safe_hi || utc != local_sec - s.off);
    return utc;
}

void utc_to_local_us(tz::ZoneId z, const std::int64_t* BOLT_RESTRICT in, std::int64_t n,
                     std::int64_t* BOLT_RESTRICT out) noexcept {
    assert(n >= 0);
    assert(in != nullptr || n == 0);
    if (tz::zone_is_fixed(z)) {
        const std::int64_t d = static_cast<std::int64_t>(tz::zone_fixed_offset(z)) * kUsPerSec;
        for (std::int64_t i = 0; i < n; ++i) out[i] = in[i] + d;
        return;
    }
    UtcCursor c{};
    utc_cursor_init(&c, z);
    for (std::int64_t i = 0; i < n; ++i) {
        const std::int64_t off = utc_cursor_offset(&c, fdiv(in[i], kUsPerSec));
        out[i] = in[i] + off * kUsPerSec;
    }
}

void local_to_utc_us(tz::ZoneId z, const std::int64_t* BOLT_RESTRICT in, std::int64_t n,
                     std::int64_t* BOLT_RESTRICT out) noexcept {
    assert(n >= 0);
    assert(in != nullptr || n == 0);
    if (tz::zone_is_fixed(z)) {
        const std::int64_t d = static_cast<std::int64_t>(tz::zone_fixed_offset(z)) * kUsPerSec;
        for (std::int64_t i = 0; i < n; ++i) out[i] = in[i] - d;
        return;
    }
    LocalCursor c{};
    local_cursor_init(&c, z);
    for (std::int64_t i = 0; i < n; ++i) {
        const std::int64_t sec = fdiv(in[i], kUsPerSec);
        out[i] = local_cursor_to_utc(&c, sec) * kUsPerSec + (in[i] - sec * kUsPerSec);
    }
}

void utc_us_to_days(tz::ZoneId z, const std::int64_t* BOLT_RESTRICT in, std::int64_t n,
                    std::int32_t* BOLT_RESTRICT out) noexcept {
    assert(n >= 0);
    UtcCursor c{};
    utc_cursor_init(&c, z);
    for (std::int64_t i = 0; i < n; ++i) {
        const std::int64_t off = utc_cursor_offset(&c, fdiv(in[i], kUsPerSec));
        out[i] = static_cast<std::int32_t>(fdiv(in[i] + off * kUsPerSec, kUsPerDay));
    }
}

void days_to_utc_us(tz::ZoneId z, const std::int32_t* BOLT_RESTRICT in, std::int64_t n,
                    std::int64_t* BOLT_RESTRICT out) noexcept {
    assert(n >= 0);
    LocalCursor c{};
    local_cursor_init(&c, z);
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = local_cursor_to_utc(&c, static_cast<std::int64_t>(in[i]) * cal::kSecPerDay) * kUsPerSec;
    }
}

void trunc_utc_us(tz::ZoneId z, cal::TruncUnit u, const std::int64_t* BOLT_RESTRICT in,
                  std::int64_t n, std::int64_t* BOLT_RESTRICT out) noexcept {
    assert(n >= 0);
    UtcCursor uc{};
    LocalCursor lc{};
    utc_cursor_init(&uc, z);
    local_cursor_init(&lc, z);
    for (std::int64_t i = 0; i < n; ++i) {
        const std::int64_t off = utc_cursor_offset(&uc, fdiv(in[i], kUsPerSec));
        const std::int64_t t = cal::trunc_local_micros(in[i] + off * kUsPerSec, u);
        const std::int64_t sec = fdiv(t, kUsPerSec);
        out[i] = local_cursor_to_utc(&lc, sec) * kUsPerSec + (t - sec * kUsPerSec);
    }
}

void trunc_days(cal::TruncUnit u, const std::int32_t* BOLT_RESTRICT in, std::int64_t n,
                std::int32_t* BOLT_RESTRICT out) noexcept {
    assert(n >= 0);
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = static_cast<std::int32_t>(cal::trunc_days(in[i], u));
    }
}

namespace {

inline std::int64_t field_of(Field f, std::int64_t days, std::int64_t tod_us) noexcept {
    const auto r = bolt::kernels::date::civil_from_days32(static_cast<std::int32_t>(days));
    switch (f) {
        case Field::Year: return r.y;
        case Field::Month: return r.m;
        case Field::Day: return r.d;
        case Field::Hour: return tod_us / 3600000000LL;
        case Field::Minute: return (tod_us / 60000000LL) % 60;
        case Field::Second: return (tod_us / 1000000LL) % 60;
        case Field::DayOfWeek: return cal::iso_dow(days) % 7 + 1;
        case Field::WeekDay: return cal::iso_dow(days) - 1;
        case Field::DayOfYear: return days - cal::days_from_ymd(r.y, 1, 1) + 1;   // r.doy counts from March
        case Field::Quarter: return (r.m - 1) / 3 + 1;
        case Field::WeekOfYear:
            return bolt::kernels::date::iso_week_from_components(
                r.y, static_cast<std::int32_t>(days - cal::days_from_ymd(r.y, 1, 1) + 1),
                cal::iso_dow(days));
    }
    return 0;
}

}  // namespace

void field_local_us(Field f, const std::int64_t* BOLT_RESTRICT local, std::int64_t n,
                    std::int64_t* BOLT_RESTRICT out) noexcept {
    assert(n >= 0);
    for (std::int64_t i = 0; i < n; ++i) {
        const std::int64_t d = fdiv(local[i], kUsPerDay);
        out[i] = field_of(f, d, local[i] - d * kUsPerDay);
    }
}

void field_days(Field f, const std::int32_t* BOLT_RESTRICT days, std::int64_t n,
                std::int64_t* BOLT_RESTRICT out) noexcept {
    assert(n >= 0);
    for (std::int64_t i = 0; i < n; ++i) out[i] = field_of(f, days[i], 0);
}

namespace {

// "yyyy-MM-dd" exactly: the common case, without the general grammar.
inline bool iso_date10(const char* p, std::int32_t* out) noexcept {
    std::uint32_t bad = static_cast<std::uint32_t>(p[4] != '-') | static_cast<std::uint32_t>(p[7] != '-');
    std::int32_t v[8];
    static constexpr std::uint8_t kPos[8] = {0, 1, 2, 3, 5, 6, 8, 9};
    for (int k = 0; k < 8; ++k) {
        const std::int32_t d = static_cast<std::int32_t>(static_cast<unsigned char>(p[kPos[k]])) - '0';
        bad |= static_cast<std::uint32_t>(d < 0 || d > 9);
        v[k] = d;
    }
    if (bad != 0) return false;
    const std::int32_t y = v[0] * 1000 + v[1] * 100 + v[2] * 10 + v[3];
    const std::int32_t m = v[4] * 10 + v[5];
    const std::int32_t d = v[6] * 10 + v[7];
    if (m < 1 || m > 12 || d < 1 || d > cal::month_days(y, m)) return false;
    *out = static_cast<std::int32_t>(cal::days_from_ymd(y, m, d));
    return true;
}

}  // namespace

bool text_to_days(const StringView* BOLT_RESTRICT sv, const char* base, std::int64_t n,
                  std::int32_t* BOLT_RESTRICT out) noexcept {
    assert(n >= 0);
    assert(sv != nullptr || n == 0);
    for (std::int64_t i = 0; i < n; ++i) {
        const StringView& v = sv[i];
        if (v.length == 10u && iso_date10(v.prefix, &out[i])) continue;   // inline: 4 + 8 bytes contiguous
        const char* p = v.length <= 12u ? v.prefix : (base == nullptr ? nullptr : base + v.ref.offset);
        if (p == nullptr) return false;
        std::int64_t d = 0;
        if (!dtfmt::parse_date_text(p, v.length, &d) || d < INT32_MIN || d > INT32_MAX) return false;
        out[i] = static_cast<std::int32_t>(d);
    }
    return true;
}

namespace {

constexpr char kPairs[] =
    "00010203040506070809101112131415161718192021222324252627282930313233343536373839"
    "40414243444546474849505152535455565758596061626364656667686970717273747576777879"
    "8081828384858687888990919293949596979899";

inline void put2(char* o, std::int64_t v) noexcept {
    assert(v >= 0 && v < 100);
    std::memcpy(o, kPairs + 2 * v, 2);
}

}  // namespace

std::uint32_t format_fixed_width(const dtfmt::Pattern& p) noexcept {
    std::uint32_t w = 0;
    for (std::uint16_t i = 0; i < p.n; ++i) {
        const dtfmt::Token& t = p.tok[i];
        if (t.optional != 0) return 0;
        switch (t.letter) {
            case 0: w += t.lit_len; break;
            case 'y': if (t.count != 4 && t.count != 2) return 0; w += t.count; break;
            case 'M': case 'd': case 'H': case 'm': case 's':
                if (t.count != 2) return 0;
                w += 2;
                break;
            case 'S': w += t.count; break;
            default: return 0;
        }
    }
    return w;
}

// Caller guarantees format_fixed_width(p) != 0 and a year in [0, 9999].
void format_fixed(const dtfmt::Pattern& p, std::int64_t local_us, char* BOLT_RESTRICT out) noexcept {
    const std::int64_t days = fdiv(local_us, kUsPerDay);
    const std::int64_t tod = local_us - days * kUsPerDay;
    const auto r = bolt::kernels::date::civil_from_days32(static_cast<std::int32_t>(days));
    assert(r.y >= 0 && r.y <= 9999);
    const std::int64_t sec = tod / kUsPerSec;
    std::uint32_t o = 0;
    for (std::uint16_t i = 0; i < p.n; ++i) {
        const dtfmt::Token& t = p.tok[i];
        switch (t.letter) {
            case 0: std::memcpy(out + o, p.lit + t.lit_off, t.lit_len); o += t.lit_len; break;
            case 'y':
                if (t.count == 4) { put2(out + o, r.y / 100); o += 2; }
                put2(out + o, r.y % 100);
                o += 2;
                break;
            case 'M': put2(out + o, r.m); o += 2; break;
            case 'd': put2(out + o, r.d); o += 2; break;
            case 'H': put2(out + o, sec / 3600); o += 2; break;
            case 'm': put2(out + o, (sec / 60) % 60); o += 2; break;
            case 's': put2(out + o, sec % 60); o += 2; break;
            case 'S': {
                std::int64_t frac = (tod % kUsPerSec) * 1000;   // nanoseconds
                char digs[9];
                for (int k = 8; k >= 0; --k) { digs[k] = static_cast<char>('0' + frac % 10); frac /= 10; }
                std::memcpy(out + o, digs, t.count);
                o += t.count;
                break;
            }
            default: break;
        }
    }
}

}  // namespace bolt::dtk
