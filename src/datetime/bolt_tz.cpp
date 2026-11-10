// bolt::tz — embedded IANA zone rules. See include/bolt/datetime/bolt_tz.h.
#include "bolt/datetime/bolt_tz.h"

#include <cassert>
#include <cstring>

#include "bolt/kernels/bolt_date.h"

namespace bolt::tz {
namespace data {

struct ZoneNameRec { std::uint32_t off; std::uint16_t len; std::uint16_t zone; };
struct ZoneRec {
    std::uint32_t trans_begin; std::uint32_t trans_count;
    std::uint32_t type_begin;  std::uint8_t  type_count;
    std::uint8_t  initial;     std::uint16_t rule;
};
struct TypeRec { std::int32_t utoff; std::uint8_t isdst; std::uint16_t abbr; };
struct AbbrRec { std::uint32_t off; std::uint16_t len; };
// kind: 1 = Mm.w.d, 2 = Jn (1..365, no Feb 29), 3 = n (0..365).
struct RuleDate {
    std::uint8_t kind; std::uint8_t month; std::uint8_t week; std::uint8_t dow;
    std::uint16_t day; std::int32_t secs;
};
struct RuleRec {
    std::int32_t std_off; std::int32_t dst_off;
    std::uint16_t std_abbr; std::uint16_t dst_abbr;   // dst_abbr 0xFFFF: no DST
    RuleDate start; RuleDate end;
};

#include "bolt_tzdb_data.inc"

}  // namespace data

namespace {

constexpr std::uint16_t kNoRule = 0xFFFF;
constexpr std::int64_t kSecPerDay = 86400;

const data::ZoneRec* named_zone(ZoneId z) noexcept {
    assert(!zone_is_fixed(z));
    if (z == 0 || z > data::kZoneNameCount) return nullptr;
    const data::ZoneNameRec& n = data::kZoneNames[z - 1];
    assert(n.zone < data::kZoneCount);
    return &data::kZones[n.zone];
}

// Epoch day of the POSIX rule date `d` in `year`.
std::int64_t rule_day(const data::RuleDate& d, std::int32_t year) noexcept {
    namespace dk = bolt::kernels::date;
    assert(d.kind >= 1 && d.kind <= 3);
    const std::int64_t jan1 = dk::days_from_civil(year, 1, 1);
    const bool leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    if (d.kind == 2) {
        const std::int64_t n = d.day;
        return jan1 + n - 1 + ((leap && n >= 60) ? 1 : 0);
    }
    if (d.kind == 3) return jan1 + d.day;
    assert(d.month >= 1 && d.month <= 12 && d.week >= 1 && d.week <= 5);
    const std::int64_t first = dk::days_from_civil(year, d.month, 1);
    const std::int32_t iso = dk::dow_from_days32(static_cast<std::int32_t>(first));
    const std::int32_t dow0 = iso % 7;   // POSIX: 0 = Sunday
    std::int64_t day = first + ((d.dow - dow0 + 7) % 7) + 7 * (d.week - 1);
    static constexpr std::uint8_t kMonthDays[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    const std::int64_t mdays = kMonthDays[d.month - 1] + ((leap && d.month == 2) ? 1 : 0);
    if (d.week == 5) {
        while (day >= first + mdays) day -= 7;
    }
    return day;
}

bool rule_in_dst(const data::RuleRec& r, std::int64_t utc_sec) noexcept {
    assert(r.dst_abbr != kNoRule);
    const std::int64_t local_std_days =
        floor_div(utc_sec + r.std_off, kSecPerDay);
    const auto ymd = bolt::kernels::date::civil_from_days32(
        static_cast<std::int32_t>(local_std_days));
    const std::int64_t start = rule_day(r.start, ymd.y) * kSecPerDay + r.start.secs - r.std_off;
    const std::int64_t end = rule_day(r.end, ymd.y) * kSecPerDay + r.end.secs - r.dst_off;
    if (start < end) return utc_sec >= start && utc_sec < end;
    return !(utc_sec >= end && utc_sec < start);
}

// Index into kTypes of the local-time type at `utc_sec`, or -1 when the
// POSIX rule governs (`*rule_dst` then says which half of the rule).
std::int64_t type_at(const data::ZoneRec& zr, std::int64_t utc_sec, bool* rule_dst) noexcept {
    assert(rule_dst != nullptr);
    assert(zr.type_count > 0);
    const std::int64_t* t = data::kTransUtc + zr.trans_begin;
    const std::uint32_t n = zr.trans_count;
    if (n == 0 || utc_sec < t[0]) {
        if (n == 0 && zr.rule != kNoRule) {
            const data::RuleRec& r = data::kRules[zr.rule];
            *rule_dst = r.dst_abbr != kNoRule && rule_in_dst(r, utc_sec);
            return -1;
        }
        return zr.type_begin + zr.initial;
    }
    if (utc_sec >= t[n - 1] && zr.rule != kNoRule) {
        const data::RuleRec& r = data::kRules[zr.rule];
        *rule_dst = r.dst_abbr != kNoRule && rule_in_dst(r, utc_sec);
        return -1;
    }
    std::uint32_t lo = 0, hi = n;   // invariant: t[lo] <= utc_sec < t[hi]
    for (std::uint32_t guard = 0; hi - lo > 1 && guard < 64; ++guard) {
        const std::uint32_t mid = lo + (hi - lo) / 2;
        if (t[mid] <= utc_sec) lo = mid; else hi = mid;
    }
    assert(hi - lo == 1);
    const std::uint8_t ti = data::kTransType[zr.trans_begin + lo];
    assert(ti < zr.type_count);
    return zr.type_begin + ti;
}

int name_cmp(const char* a, std::uint32_t al, const char* b, std::uint32_t bl) noexcept {
    const std::uint32_t n = al < bl ? al : bl;
    const int c = n == 0 ? 0 : std::memcmp(a, b, n);
    if (c != 0) return c;
    return al < bl ? -1 : (al > bl ? 1 : 0);
}

char ascii_lower(char c) noexcept { return (c >= 'A' && c <= 'Z') ? static_cast<char>(c + 32) : c; }

bool find_region(const char* s, std::uint32_t len, ZoneId* out) noexcept {
    assert(out != nullptr);
    std::uint32_t lo = 0, hi = data::kZoneNameCount;
    for (std::uint32_t guard = 0; lo < hi && guard < 64; ++guard) {
        const std::uint32_t mid = lo + (hi - lo) / 2;
        const auto& r = data::kZoneNames[mid];
        const int c = name_cmp(data::kNameBlob + r.off, r.len, s, len);
        if (c == 0) { *out = mid + 1; return true; }
        if (c < 0) lo = mid + 1; else hi = mid;
    }
    for (std::uint32_t i = 0; i < data::kZoneNameCount; ++i) {   // cold: case-insensitive
        const auto& r = data::kZoneNames[i];
        if (r.len != len) continue;
        std::uint32_t k = 0;
        while (k < len && ascii_lower(data::kNameBlob[r.off + k]) == ascii_lower(s[k])) ++k;
        if (k == len) { *out = i + 1; return true; }
    }
    return false;
}

bool digits(const char* s, std::uint32_t n, std::int32_t* v) noexcept {
    std::int32_t x = 0;
    for (std::uint32_t i = 0; i < n; ++i) {
        if (s[i] < '0' || s[i] > '9') return false;
        x = x * 10 + (s[i] - '0');
    }
    *v = x;
    return n > 0;
}

// "+h", "+hh", "+hh:mm", "+hhmm", "+hh:mm:ss", "+hhmmss".
bool parse_offset(const char* s, std::uint32_t len, std::int32_t* out) noexcept {
    assert(out != nullptr);
    if (len < 2 || (s[0] != '+' && s[0] != '-')) return false;
    const std::int32_t sign = s[0] == '-' ? -1 : 1;
    const char* p = s + 1;
    const std::uint32_t n = len - 1;
    std::int32_t h = 0, m = 0, sec = 0;
    if (n <= 2) {
        if (!digits(p, n, &h)) return false;
    } else if (p[2] == ':') {
        if (!digits(p, 2, &h) || !(n == 5 || n == 8) || !digits(p + 3, 2, &m)) return false;
        if (n == 8 && (p[5] != ':' || !digits(p + 6, 2, &sec))) return false;
    } else {
        if (!(n == 4 || n == 6) || !digits(p, 2, &h) || !digits(p + 2, 2, &m)) return false;
        if (n == 6 && !digits(p + 4, 2, &sec)) return false;
    }
    if (h > 18 || m > 59 || sec > 59) return false;
    const std::int32_t total = sign * (h * 3600 + m * 60 + sec);
    if (total > kMaxFixedOffsetSec || total < -kMaxFixedOffsetSec) return false;
    *out = total;
    return true;
}

bool prefix_ieq(const char* s, std::uint32_t len, const char* p) noexcept {
    const std::uint32_t pl = static_cast<std::uint32_t>(std::strlen(p));
    if (len < pl) return false;
    for (std::uint32_t i = 0; i < pl; ++i) {
        if (ascii_lower(s[i]) != ascii_lower(p[i])) return false;
    }
    return true;
}

std::uint32_t write_offset(std::int32_t off, char* buf, std::uint32_t cap) noexcept {
    if (off == 0) {
        if (cap < 3) return 0;
        std::memcpy(buf, "UTC", 3);
        return 3;
    }
    const std::int32_t a = off < 0 ? -off : off;
    const std::int32_t h = a / 3600, m = (a / 60) % 60, s = a % 60;
    const std::uint32_t need = s != 0 ? 9u : 6u;
    if (cap < need) return 0;
    buf[0] = off < 0 ? '-' : '+';
    buf[1] = static_cast<char>('0' + h / 10); buf[2] = static_cast<char>('0' + h % 10);
    buf[3] = ':';
    buf[4] = static_cast<char>('0' + m / 10); buf[5] = static_cast<char>('0' + m % 10);
    if (s != 0) {
        buf[6] = ':';
        buf[7] = static_cast<char>('0' + s / 10); buf[8] = static_cast<char>('0' + s % 10);
    }
    return need;
}

}  // namespace

bool zone_parse(const char* s, std::uint32_t len, ZoneId* out) noexcept {
    assert(out != nullptr);
    *out = kInvalidZone;
    if (s == nullptr || len == 0 || len > kMaxZoneIdChars) return false;
    std::int32_t off = 0;
    if ((len == 1 && (s[0] == 'Z' || s[0] == 'z'))) { *out = kUtc; return true; }
    if (parse_offset(s, len, &off)) { *out = zone_fixed(off); return true; }
    static const char* const kPrefixes[3] = {"UTC", "GMT", "UT"};
    for (const char* pre : kPrefixes) {
        const std::uint32_t pl = static_cast<std::uint32_t>(std::strlen(pre));
        if (!prefix_ieq(s, len, pre)) continue;
        if (len == pl) { *out = kUtc; return true; }
        if (parse_offset(s + pl, len - pl, &off)) { *out = zone_fixed(off); return true; }
    }
    if (find_region(s, len, out)) return true;
    *out = kInvalidZone;
    return false;
}

std::int32_t offset_at(ZoneId z, std::int64_t utc_sec) noexcept {
    if (zone_is_fixed(z)) return zone_fixed_offset(z);
    const data::ZoneRec* zr = named_zone(z);
    if (zr == nullptr) return 0;
    bool rule_dst = false;
    const std::int64_t ti = type_at(*zr, utc_sec, &rule_dst);
    if (ti < 0) {
        const data::RuleRec& r = data::kRules[zr->rule];
        return rule_dst ? r.dst_off : r.std_off;
    }
    assert(ti < static_cast<std::int64_t>(data::kTypeCount));
    return data::kTypes[ti].utoff;
}

std::int64_t local_to_utc(ZoneId z, std::int64_t local_sec) noexcept {
    if (zone_is_fixed(z)) return local_sec - zone_fixed_offset(z);
    // UTC offsets are within +-26h, so local +-2 days brackets the instant;
    // zone transitions are never closer than that apart.
    const std::int32_t before = offset_at(z, local_sec - 2 * kSecPerDay);
    const std::int32_t after = offset_at(z, local_sec + 2 * kSecPerDay);
    if (before == after) {
        const std::int64_t u = local_sec - before;
        const std::int32_t o = offset_at(z, u);
        return o == before ? u : local_sec - o;
    }
    const std::int64_t ub = local_sec - before;
    if (offset_at(z, ub) == before) return ub;        // valid, or the earlier overlap offset
    const std::int64_t ua = local_sec - after;
    if (offset_at(z, ua) == after) return ua;
    return ub;                                         // gap: shift forward by its length
}

namespace {

// The rule-governed span around `utc_sec`: between the nearest rule
// transitions of the neighbouring years, never before `floor`.
void rule_span(const data::RuleRec& r, std::int64_t floor, std::int64_t utc_sec,
               OffsetSpan* out) noexcept {
    assert(out != nullptr);
    out->lo = floor;
    out->hi = INT64_MAX;
    out->off = r.std_off;
    if (r.dst_abbr == kNoRule) return;
    out->off = rule_in_dst(r, utc_sec) ? r.dst_off : r.std_off;
    const std::int64_t days = floor_div(utc_sec + r.std_off, kSecPerDay);
    const std::int32_t year = bolt::kernels::date::civil_from_days32(static_cast<std::int32_t>(days)).y;
    for (std::int32_t y = year - 1; y <= year + 1; ++y) {
        const std::int64_t pts[2] = {
            rule_day(r.start, y) * kSecPerDay + r.start.secs - r.std_off,
            rule_day(r.end, y) * kSecPerDay + r.end.secs - r.dst_off};
        for (const std::int64_t t : pts) {
            if (t <= utc_sec && t > out->lo) out->lo = t;
            if (t > utc_sec && t < out->hi) out->hi = t;
        }
    }
}

}  // namespace

void offset_span(ZoneId z, std::int64_t utc_sec, OffsetSpan* out) noexcept {
    assert(out != nullptr);
    out->lo = INT64_MIN;
    out->hi = INT64_MAX;
    out->off = 0;
    if (zone_is_fixed(z)) { out->off = zone_fixed_offset(z); return; }
    const data::ZoneRec* zr = named_zone(z);
    if (zr == nullptr) return;
    const std::int64_t* t = data::kTransUtc + zr->trans_begin;
    const std::uint32_t n = zr->trans_count;
    if (n == 0 && zr->rule != kNoRule) { rule_span(data::kRules[zr->rule], INT64_MIN, utc_sec, out); return; }
    if (n == 0 || utc_sec < t[0]) {
        out->off = data::kTypes[zr->type_begin + zr->initial].utoff;
        out->hi = n == 0 ? INT64_MAX : t[0];
        return;
    }
    if (utc_sec >= t[n - 1] && zr->rule != kNoRule) {
        rule_span(data::kRules[zr->rule], t[n - 1], utc_sec, out);
        return;
    }
    std::uint32_t lo = 0, hi = n;   // t[lo] <= utc_sec < t[hi]
    for (std::uint32_t guard = 0; hi - lo > 1 && guard < 64; ++guard) {
        const std::uint32_t mid = lo + (hi - lo) / 2;
        if (t[mid] <= utc_sec) lo = mid; else hi = mid;
    }
    out->lo = t[lo];
    out->hi = hi < n ? t[hi] : INT64_MAX;
    out->off = data::kTypes[zr->type_begin + data::kTransType[zr->trans_begin + lo]].utoff;
    assert(out->lo <= utc_sec && utc_sec < out->hi);
}

std::uint32_t zone_name(ZoneId z, char* buf, std::uint32_t cap) noexcept {
    assert(buf != nullptr || cap == 0);
    if (zone_is_fixed(z)) return write_offset(zone_fixed_offset(z), buf, cap);
    if (z == 0 || z > data::kZoneNameCount) return 0;
    const auto& r = data::kZoneNames[z - 1];
    if (r.len > cap) return 0;
    std::memcpy(buf, data::kNameBlob + r.off, r.len);
    return r.len;
}

std::uint32_t zone_abbr(ZoneId z, std::int64_t utc_sec, char* buf, std::uint32_t cap) noexcept {
    assert(buf != nullptr || cap == 0);
    if (zone_is_fixed(z)) return write_offset(zone_fixed_offset(z), buf, cap);
    const data::ZoneRec* zr = named_zone(z);
    if (zr == nullptr) return 0;
    bool rule_dst = false;
    const std::int64_t ti = type_at(*zr, utc_sec, &rule_dst);
    std::uint16_t a = 0;
    if (ti < 0) {
        const data::RuleRec& r = data::kRules[zr->rule];
        a = rule_dst ? r.dst_abbr : r.std_abbr;
    } else {
        a = data::kTypes[ti].abbr;
    }
    assert(a < data::kAbbrCount);
    const auto& ar = data::kAbbrs[a];
    if (ar.len > cap) return 0;
    std::memcpy(buf, data::kAbbrBlob + ar.off, ar.len);
    return ar.len;
}

const char* tzdb_version() noexcept { return data::kTzdbVersion; }

}  // namespace bolt::tz
