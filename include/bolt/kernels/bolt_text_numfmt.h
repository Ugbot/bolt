// bolt_text_numfmt.h — number <-> text kernels for Spark: format_number
// (java.text.DecimalFormat "#,###,##0.00" HALF_EVEN) and, below it, the
// to_char / to_number format-model mini-language.
//
// format_number works on a decimal digit string and a decimal exponent, so one
// routine serves integers, doubles (shortest round-trip digits, as the JVM's
// DecimalFormat sees them) and exact decimals.
//
// RULES: No exceptions, no heap, all fns noexcept, bounded loops.

#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "bolt/kernels/bolt_float_chars.h"
#include "bolt/kernels/bolt_text_fn.h"
#include "bolt/kernels/bolt_text_spark.h"
#include "bolt/kernels/bolt_unicode_class.h"
#include "bolt/kernels/bolt_utf8_cp.h"

namespace bolt {
namespace kernels {
namespace text {

// Upper bound of format_number's output for `int_digits` integer digits and `decimals`
// fraction digits (grouping commas, a sign, a carry digit and the decimal point).
constexpr uint64_t format_number_bound(uint32_t int_digits, uint32_t decimals) noexcept {
    return static_cast<uint64_t>(int_digits) + int_digits / 3u + decimals + 8u;
}

// d[0..nd): decimal digits, value = 0.d1...dn * 10^k. Rounds HALF_EVEN to `decimals`
// fraction digits, groups the integer part by three. Returns bytes or kTextOverflow.
inline int64_t format_number_digits(bool neg, const char* d, uint32_t nd, int32_t k,
                                    uint32_t decimals, char* out, uint32_t cap) noexcept {
    assert(d != nullptr && nd >= 1 && out != nullptr);
    const int32_t int_digits = k > 0 ? k : 0;
    const uint64_t total = static_cast<uint64_t>(int_digits) + decimals;     // kept digits
    if (format_number_bound(static_cast<uint32_t>(int_digits) + 1u, decimals) > cap) return kTextOverflow;
    const int64_t shift = static_cast<int64_t>(int_digits) - k;
    auto digit_at = [&](uint64_t i) noexcept -> int {
        const int64_t src = static_cast<int64_t>(i) - shift;
        return (src >= 0 && src < static_cast<int64_t>(nd)) ? d[src] - '0' : 0;
    };
    // The kept digits sit at the tail of the buffer; the result is composed at the front. The
    // front write position stays behind the unread digits (cap - total >= int_digits / 3 + 8).
    char* kept = out + (cap - total);
    for (uint64_t i = 0; i < total; ++i) kept[i] = static_cast<char>('0' + digit_at(i));
    const int drop = digit_at(total);
    bool tail_nonzero = false;
    for (int64_t s = static_cast<int64_t>(total) + 1 - shift; s < static_cast<int64_t>(nd); ++s) {
        if (s >= 0 && d[s] != '0') { tail_nonzero = true; break; }
    }
    const int prev = total > 0 ? kept[total - 1] - '0' : 0;
    bool carry = drop > 5 || (drop == 5 && (tail_nonzero || (prev & 1) != 0));
    for (int64_t i = static_cast<int64_t>(total) - 1; carry && i >= 0; --i) {
        if (kept[i] == '9') kept[i] = '0'; else { ++kept[i]; carry = false; }
    }
    // carry still set: the integer part gained a leading 1.
    const uint32_t idig = static_cast<uint32_t>(int_digits) + (carry ? 1u : 0u);
    uint32_t w = 0;
    if (neg) out[w++] = '-';
    if (idig == 0) out[w++] = '0';
    for (uint32_t i = 0; i < idig; ++i) {
        const char c = (carry && i == 0) ? '1' : kept[carry ? i - 1u : i];
        if (i != 0 && (idig - i) % 3u == 0) out[w++] = ',';
        out[w++] = c;
    }
    if (decimals > 0) {
        out[w++] = '.';
        for (uint32_t i = 0; i < decimals; ++i) out[w++] = kept[static_cast<uint32_t>(int_digits) + i];
    }
    return w;
}

// The exact decimal expansion of a finite positive double as digits: a = 0.d1...dn * 10^k,
// leading and trailing zeros trimmed. `d` holds at least 1400 chars. Returns n.
inline uint32_t exact_digits(double a, char* d, int32_t* k) noexcept {
    assert(d != nullptr && k != nullptr);
    char buf[1500];
    const int n = std::snprintf(buf, sizeof(buf), "%.1080f", a);
    assert(n > 0 && static_cast<size_t>(n) < sizeof(buf));
    int32_t int_len = 0;
    while (int_len < n && buf[int_len] != '.') ++int_len;
    uint32_t nd = 0;
    for (int32_t i = 0; i < n; ++i) if (buf[i] != '.') d[nd++] = buf[i];
    int32_t kk = int_len;
    uint32_t lead = 0;
    while (lead + 1 < nd && d[lead] == '0') { ++lead; --kk; }
    if (lead != 0) { std::memmove(d, d + lead, nd - lead); nd -= lead; }
    while (nd > 1 && d[nd - 1] == '0') --nd;
    *k = kk;
    return nd;
}

// format_number over a double: NaN, ±∞ (U+221E) as java.text.DecimalFormat prints them. The
// digits are the shortest round-trip ones, except that an exact tie in those digits is decided
// by the double's exact value (the JVM's DecimalFormat does the same).
inline int64_t format_number_f64(double v, uint32_t decimals, char* out, uint32_t cap) noexcept {
    assert(out != nullptr);
    if (v != v) { if (cap < 3) return kTextOverflow; std::memcpy(out, "NaN", 3); return 3; }
    const bool neg = std::signbit(v);
    if (std::isinf(v)) {
        if (cap < 4) return kTextOverflow;
        uint32_t w = 0;
        if (neg) out[w++] = '-';
        out[w++] = static_cast<char>(0xE2); out[w++] = static_cast<char>(0x88); out[w++] = static_cast<char>(0x9E);
        return w;
    }
    char d[40] = {};
    int32_t k = 0;
    uint32_t nd = 1;
    const double a = neg ? -v : v;
    if (a == 0.0) { d[0] = '0'; k = 1; }
    else nd = float_chars::detail::shortest_digits(a, d, &k);
    const int64_t drop_at = static_cast<int64_t>(decimals) + k;     // index of the first dropped digit
    if (a != 0.0 && drop_at == static_cast<int64_t>(nd) - 1 && d[nd - 1] == '5') {
        char ex[1400];                                              // an exact tie in the shortest digits
        const uint32_t en = exact_digits(a, ex, &k);
        return format_number_digits(neg, ex, en, k, decimals, out, cap);
    }
    return format_number_digits(neg, d, nd, k, decimals, out, cap);
}

// format_number over an integer.
inline int64_t format_number_i64(int64_t v, uint32_t decimals, char* out, uint32_t cap) noexcept {
    assert(out != nullptr);
    char d[24];
    uint64_t u = v < 0 ? ~static_cast<uint64_t>(v) + 1u : static_cast<uint64_t>(v);
    uint32_t nd = 0;
    char rev[24];
    do { rev[nd++] = static_cast<char>('0' + u % 10u); u /= 10u; } while (u != 0);
    for (uint32_t i = 0; i < nd; ++i) d[i] = rev[nd - 1 - i];
    return format_number_digits(v < 0, d, nd, static_cast<int32_t>(nd), decimals, out, cap);
}

// format_number over an exact decimal: hi:lo is the two's-complement mantissa, value = m * 10^-scale.
inline int64_t format_number_dec(int64_t hi, uint64_t lo, uint32_t scale, uint32_t decimals,
                                 char* out, uint32_t cap) noexcept {
    assert(out != nullptr);
    assert(scale <= 38u);
    using u128 = unsigned __int128;
    const bool neg = hi < 0;
    u128 m = (static_cast<u128>(static_cast<uint64_t>(hi)) << 64) | lo;
    if (neg) m = ~m + 1u;
    char rev[48], d[48];
    uint32_t nd = 0;
    do { rev[nd++] = static_cast<char>('0' + static_cast<uint32_t>(m % 10u)); m /= 10u; } while (m != 0 && nd < 40u);
    for (uint32_t i = 0; i < nd; ++i) d[i] = rev[nd - 1 - i];
    return format_number_digits(neg, d, nd, static_cast<int32_t>(nd) - static_cast<int32_t>(scale),
                                decimals, out, cap);
}


// ---- to_char / to_number format model ---------------------------------------
//
// Spark's number format string: digits 0 / 9, thousands separator , or G, decimal point . or D,
// a leading currency $, a sign S or MI before the number (and after it), and a trailing PR.
// The same parser serves both functions; its error order is Spark's (observed on 4.0.1).

enum class NumFmtErr : int {
    Ok = 0, Empty, UnexpectedToken, WrongNumToken, ContThousandsSeps, ThousandsSepsMustBeforeDec,
    CurMustBeforeDigit, WrongNumDigit, TooLong,
};

constexpr uint32_t kNumFmtMaxItems = 96;       // format items (digits, separators, the point)

struct NumFmt {
    uint8_t  kind[kNumFmtMaxItems];            // 0 = '9', 1 = '0', 2 = separator, 3 = decimal point
    uint8_t  sep_g[kNumFmtMaxItems];           // separator spelled G / g (matches ',' in an input); point spelled D
    uint32_t n_items;
    uint32_t n_int;                            // digit placeholders before the point
    uint32_t n_frac;                           // digit placeholders after it
    int      lead;                             // 0 none, 1 S, 2 MI
    int      trail;                            // 0 none, 1 S, 2 MI, 3 PR
    bool     cur;                              // $
    bool     has_point;
};

namespace detail {
inline char lc(char c) noexcept { return (c >= 'A' && c <= 'Z') ? static_cast<char>(c + 32) : c; }
}  // namespace detail

inline NumFmtErr numfmt_parse(const char* f, uint32_t len, NumFmt* out) noexcept {
    assert(out != nullptr && (f != nullptr || len == 0));
    std::memset(out, 0, sizeof(*out));
    if (len == 0) return NumFmtErr::Empty;
    // Tokens: 'd' digit 9/0, ',' sep, '.' point, '$', 'S', 'M' (MI), 'P' (PR), '?' unexpected.
    char tok[kNumFmtMaxItems + 8];
    uint8_t spell[kNumFmtMaxItems + 8];
    uint32_t nt = 0;
    for (uint32_t i = 0; i < len;) {                             // bounded: len
        if (nt >= sizeof(tok)) return NumFmtErr::TooLong;
        const char c = f[i];
        const char c1 = detail::lc(c);
        char t = '?';
        uint8_t sp = 0;
        uint32_t adv = 1;
        if (c == '9' || c == '0') { t = 'd'; sp = static_cast<uint8_t>(c == '0'); }
        else if (c == ',' || c1 == 'g') { t = ','; sp = static_cast<uint8_t>(c1 == 'g'); }
        else if (c == '.' || c1 == 'd') { t = '.'; sp = static_cast<uint8_t>(c1 == 'd'); }
        else if (c == '$') t = '$';
        else if (c1 == 's') t = 'S';
        else if (c1 == 'm' && i + 1 < len && detail::lc(f[i + 1]) == 'i') { t = 'M'; adv = 2; }
        else if (c1 == 'p' && i + 1 < len && detail::lc(f[i + 1]) == 'r') { t = 'P'; adv = 2; }
        tok[nt] = t;
        spell[nt++] = sp;
        i += adv;
    }
    bool after_point = false;
    for (uint32_t i = 0; i < nt; ++i) {                          // thousands separators, left to right
        if (tok[i] == '.') after_point = true;
        if (tok[i] != ',') continue;
        if (after_point) return NumFmtErr::ThousandsSepsMustBeforeDec;
        if (i == 0 || tok[i - 1] != 'd' || i + 1 >= nt || tok[i + 1] != 'd') return NumFmtErr::ContThousandsSeps;
    }
    uint32_t cnt_pt = 0, cnt_cur = 0, cnt_s = 0, cnt_mi = 0, cnt_pr = 0, ndig = 0;
    for (uint32_t i = 0; i < nt; ++i) {
        cnt_pt += tok[i] == '.'; cnt_cur += tok[i] == '$'; cnt_s += tok[i] == 'S';
        cnt_mi += tok[i] == 'M'; cnt_pr += tok[i] == 'P'; ndig += tok[i] == 'd';
    }
    if (cnt_pt > 1 || cnt_cur > 1 || cnt_s > 1 || cnt_mi > 1 || cnt_pr > 1) return NumFmtErr::WrongNumToken;
    bool seen_body = false;
    for (uint32_t i = 0; i < nt; ++i) {                          // '$' only before the digits
        if (tok[i] == 'd' || tok[i] == ',' || tok[i] == '.') seen_body = true;
        if (tok[i] == '$' && seen_body) return NumFmtErr::CurMustBeforeDigit;
    }
    if (ndig == 0) return NumFmtErr::WrongNumDigit;
    uint32_t i = 0, e = nt;
    if (tok[i] == 'S' || tok[i] == 'M') { out->lead = tok[i] == 'S' ? 1 : 2; ++i; }
    if (i < e && tok[i] == '$') { out->cur = true; ++i; }
    if (e > i && (tok[e - 1] == 'S' || tok[e - 1] == 'M' || tok[e - 1] == 'P')) {
        out->trail = tok[e - 1] == 'S' ? 1 : (tok[e - 1] == 'M' ? 2 : 3);
        --e;
    }
    if (i >= e) return NumFmtErr::UnexpectedToken;
    for (uint32_t k = i; k < e; ++k) {                           // the body: digits, separators, the point
        const char t = tok[k];
        if (t != 'd' && t != ',' && t != '.') return NumFmtErr::UnexpectedToken;
        out->kind[out->n_items] = t == 'd' ? static_cast<uint8_t>(spell[k]) : (t == ',' ? 2u : 3u);
        out->sep_g[out->n_items] = spell[k];
        ++out->n_items;
        if (t == 'd') { if (out->has_point) ++out->n_frac; else ++out->n_int; }
        if (t == '.') out->has_point = true;
    }
    assert(out->n_int + out->n_frac == ndig);
    return NumFmtErr::Ok;
}

// ---- to_char ----------------------------------------------------------------

// A number as to_char sees it: |value| digits split at the point, trailing zeros of the fraction
// and leading zeros of the integer part removed (BigDecimal.stripTrailingZeros).
struct NumDigits {
    bool    neg;
    char    ip[64];  uint32_t ni;            // integer digits, no leading zeros ("" for 0 < |v| < 1)
    char    fp[64];  uint32_t nf;            // fraction digits, no trailing zeros
};

// d[0..nd): 0.d1...dn * 10^k.
inline bool numdigits_from(bool neg, const char* d, uint32_t nd, int32_t k, NumDigits* o) noexcept {
    assert(d != nullptr && o != nullptr);
    o->neg = neg; o->ni = 0; o->nf = 0;
    int32_t at_int = 0;
    for (int32_t i = 0; i < static_cast<int32_t>(nd); ++i) {
        const int32_t pos = i - k;                               // >= 0: fraction index; < 0: integer index
        if (pos < 0) {
            if (d[i] == '0' && o->ni == 0) continue;             // leading zero
            if (o->ni >= sizeof(o->ip)) return false;
            o->ip[o->ni++] = d[i];
        } else {
            while (static_cast<int32_t>(o->nf) < pos) {          // zeros between the point and the first digit
                if (o->nf >= sizeof(o->fp)) return false;
                o->fp[o->nf++] = '0';
            }
            if (o->nf >= sizeof(o->fp)) return false;
            o->fp[o->nf++] = d[i];
        }
        at_int = 1;
    }
    (void)at_int;
    if (k > static_cast<int32_t>(nd)) {                          // integer part padded with zeros: 12 * 10^k
        for (int32_t i = static_cast<int32_t>(nd); i < k; ++i) {
            if (o->ni >= sizeof(o->ip)) return false;
            o->ip[o->ni++] = '0';
        }
    }
    while (o->nf > 0 && o->fp[o->nf - 1] == '0') --o->nf;
    if (o->ni == 0 && o->nf == 0) o->neg = false;
    return true;
}

constexpr uint64_t to_char_bound(uint32_t items) noexcept { return 8u + 2u * items; }

// to_char(number, format): writes the result, returns its bytes (kTextOverflow when cap is short).
// Layout as observed on Spark 4.0.1: each digit placeholder takes a value digit (right-aligned)
// or pads with '0' when its run (split by separators) starts with '0', else ' '; a zero integer
// part prints '0' before a decimal point unless a sign or '<' attaches in front of it; a separator
// prints only after a digit; the point prints only before fraction digits; a too-wide value is
// all '#'. Signs float next to the number (the leading one not past the point).
inline int64_t to_char(const NumFmt& f, const NumDigits& v, char* out, uint32_t cap) noexcept {
    assert(out != nullptr);
    assert(f.n_items >= 1u && f.n_items <= kNumFmtMaxItems);
    if (cap < to_char_bound(f.n_items)) return kTextOverflow;
    const bool over = v.ni > f.n_int || v.nf > f.n_frac;
    char sign_lead = 0, sign_trail = 0;
    if (f.lead == 1) sign_lead = v.neg ? '-' : '+';
    if (f.lead == 2) sign_lead = v.neg ? '-' : ' ';
    if (f.trail == 1) sign_trail = v.neg ? '-' : '+';
    if (f.trail == 2) sign_trail = v.neg ? '-' : ' ';
    const bool pr_neg = f.trail == 3 && v.neg;
    // A sign or '<' that would sit right against a lone integer '0' takes its place ("-.5" for -0.5).
    const bool drop_zero = !f.cur && ((sign_lead != 0 && sign_lead != ' ') || pr_neg);
    char body[kNumFmtMaxItems + 4];
    uint32_t nb = 0, nbi = 0;
    int32_t int_pos = static_cast<int32_t>(f.n_int);             // integer placeholders still to emit
    bool run_zero = false, run_start = true, any_digit = false, run_digit = false;
    uint32_t item = 0;
    for (; item < f.n_items && f.kind[item] != 3; ++item) {
        const uint8_t kd = f.kind[item];
        if (kd == 2) { body[nb++] = run_digit && !over ? ',' : ' '; run_start = true; run_digit = false; continue; }
        if (run_start) { run_zero = kd == 1; run_start = false; }
        --int_pos;
        char c;
        if (over) c = '#';
        else if (static_cast<int32_t>(v.ni) > int_pos) c = v.ip[v.ni - 1u - static_cast<uint32_t>(int_pos)];
        else if (v.ni == 0 && int_pos == 0 && f.has_point && !(drop_zero && !any_digit)) c = '0';
        else c = run_zero ? '0' : ' ';
        any_digit = any_digit || c != ' ';
        run_digit = run_digit || c != ' ';
        body[nb++] = c;
    }
    nbi = nb;
    if (item < f.n_items) {                                      // the point and the fraction
        body[nb++] = f.n_frac > 0 ? '.' : ' ';
        uint32_t fi = 0;
        for (++item; item < f.n_items; ++item) body[nb++] = over ? '#' : (fi < v.nf ? v.fp[fi++] : '0');
    }
    uint32_t lead_at = 0;                                        // leading blanks of the integer part
    while (lead_at < nbi && body[lead_at] == ' ') ++lead_at;
    uint32_t trail_at = nb;                                      // after the last non-blank
    while (trail_at > 0 && body[trail_at - 1] == ' ') --trail_at;
    char res[kNumFmtMaxItems + 16];
    uint32_t w = 0;
    auto lead_marks = [&]() noexcept {
        if (pr_neg) res[w++] = '<';
        if (sign_lead != 0) res[w++] = sign_lead;
    };
    auto trail_marks = [&]() noexcept {
        if (f.trail == 3) { res[w++] = pr_neg ? '>' : ' '; if (!pr_neg) res[w++] = ' '; }
        else if (sign_trail != 0) res[w++] = sign_trail;
    };
    if (f.cur) { lead_marks(); res[w++] = '$'; }
    for (uint32_t k = 0; k <= nb; ++k) {
        if (!f.cur && k == lead_at) lead_marks();
        if (k == trail_at) trail_marks();
        if (k < nb) res[w++] = body[k];
    }
    if (w > cap) return kTextOverflow;
    std::memcpy(out, res, w);
    return w;
}

// ---- to_number --------------------------------------------------------------

constexpr int64_t kNumMismatch = kTextInvalid;   // the input does not match the format

namespace detail {
// Java's Character.digit value of a code point that isDigit, or -1.
inline int digit_cp_value(uint32_t cp) noexcept {
    if (cp >= '0' && cp <= '9') return static_cast<int>(cp - '0');
    if (cp < 0x80u) return -1;
    uint32_t lo = 0, hi = unicode::kDigitCount;
    for (uint32_t it = 0; it < 16u && lo < hi; ++it) {
        const uint32_t mid = (lo + hi) >> 1;
        if (cp < unicode::kDigit[mid].lo) hi = mid;
        else if (cp > unicode::kDigit[mid].hi) lo = mid + 1u;
        else return static_cast<int>((cp - unicode::kDigit[mid].lo) % 10u);
    }
    return -1;
}

struct Group { uint32_t min, max; };
struct Runs {
    uint8_t  dig[96];      // all integer digits, in order
    uint32_t len[40];      // digits per run
    uint32_t nruns;
    uint32_t ndig;
};

// The integer part of the format as digit groups. A group starting with '0' needs exactly its
// width (zero padded); one starting with '9' takes 0..width digits and, before a separator, may
// be left out with that separator. Returns the separator count, or ~0u when there are too many.
inline uint32_t fmt_groups(const NumFmt& f, Group* g) noexcept {
    bool start0[40] = {};
    uint32_t k = 0;
    g[0] = {0, 0};
    for (uint32_t i = 0; i < f.n_items && f.kind[i] != 3; ++i) {     // bounded: n_items
        if (f.kind[i] == 2) { if (k + 1 >= 40u) return 0xFFFFFFFFu; g[++k] = {0, 0}; }
        else { if (g[k].max == 0) start0[k] = f.kind[i] == 1u; ++g[k].max; }
    }
    for (uint32_t t = 0; t <= k; ++t) g[t].min = start0[t] ? g[t].max : 0u;
    return k;
}

// Can runs[j..] match the optional (group, separator) units t >= i and the final group?
inline bool runs_match(const Runs& r, const Group* g, uint32_t k, uint32_t j, uint32_t i) noexcept {
    if (j + 1 == r.nruns) {
        for (uint32_t t = i; t < k; ++t) if (g[t].min != 0) return false;   // skipped units must be skippable
        return r.len[j] >= g[k].min && r.len[j] <= g[k].max;
    }
    for (uint32_t t = i; t < k; ++t) {                           // bounded: k
        if (r.len[j] >= g[t].min && r.len[j] <= g[t].max && runs_match(r, g, k, j + 1, t + 1)) return true;
        if (g[t].min != 0) break;                                // a unit with a required '0' cannot be skipped
    }
    return false;
}
}  // namespace detail

// to_number(input, format): on success the signed mantissa digits go to digs (n_int_used + n_frac
// digits, fraction zero-padded) and *neg is set; returns the digit count. kNumMismatch otherwise.
inline int64_t to_number(const NumFmt& f, const char* s, uint32_t len, bool* neg, char* digs) noexcept {
    assert(neg != nullptr && digs != nullptr);
    uint32_t b = 0, e = len;
    if (f.trail != 3) {                                          // Java String.trim
        while (b < e && static_cast<uint8_t>(s[b]) <= ' ') ++b;
        while (e > b && static_cast<uint8_t>(s[e - 1]) <= ' ') --e;
    }
    bool pr_neg = false, sign_neg = false;
    if (f.trail == 3) {
        if (e > b && s[b] == '<') { if (e - b < 2 || s[e - 1] != '>') return kNumMismatch; ++b; --e; pr_neg = true; }
        else if (e > b && s[e - 1] == '>') return kNumMismatch;
    }
    if (f.lead != 0 && b < e && (s[b] == '-' || (f.lead == 1 && s[b] == '+'))) { sign_neg = s[b] == '-'; ++b; }
    if (f.cur) {
        if (b >= e || s[b] != '$') return kNumMismatch;
        ++b;
    }
    if (f.trail == 1 && e > b && (s[e - 1] == '+' || s[e - 1] == '-')) { sign_neg = sign_neg || s[e - 1] == '-'; --e; }
    else if (f.trail == 2 && e > b && s[e - 1] == '-') { sign_neg = true; --e; }
    // Groups of the integer format.
    detail::Group g[40];
    const uint32_t k = detail::fmt_groups(f, g);
    if (k == 0xFFFFFFFFu) return kNumMismatch;
    detail::Runs r;
    r.nruns = 1; r.len[0] = 0; r.ndig = 0;
    uint32_t nfrac = 0;
    bool in_frac = false;
    char fd[48];
    for (uint32_t i = b; i < e;) {                               // bounded: len
        const uint8_t c = static_cast<uint8_t>(s[i]);
        if (c == ' ') { ++i; continue; }                         // spaces between digits are ignored
        if (c == ',' && !in_frac) {
            if (r.nruns >= 40u) return kNumMismatch;
            r.len[r.nruns++] = 0;
            ++i;
            continue;
        }
        if (c == '.') {
            if (in_frac || !f.has_point) return kNumMismatch;
            in_frac = true;
            ++i;
            continue;
        }
        uint32_t cp = c, u = 1;
        if (c >= 0x80u) u = utf8::utf8_cp_decode(s, e, i, &cp);
        const int v = cp == utf8::kUtf8Invalid ? -1 : detail::digit_cp_value(cp);
        if (v < 0) return kNumMismatch;
        if (in_frac) {
            if (nfrac >= f.n_frac || nfrac >= sizeof(fd)) return kNumMismatch;
            fd[nfrac++] = static_cast<char>('0' + v);
        } else {
            if (r.ndig >= sizeof(r.dig)) return kNumMismatch;
            r.dig[r.ndig++] = static_cast<uint8_t>(v);
            ++r.len[r.nruns - 1u];
        }
        i += u;
    }
    if (r.nruns - 1u > k) return kNumMismatch;
    if (r.ndig == 0 && nfrac == 0 && !in_frac) return kNumMismatch;      // nothing to read
    if (!detail::runs_match(r, g, k, 0, 0)) return kNumMismatch;
    uint32_t w = 0;
    for (uint32_t i = 0; i < r.ndig; ++i) digs[w++] = static_cast<char>('0' + r.dig[i]);
    for (uint32_t i = 0; i < f.n_frac; ++i) digs[w++] = i < nfrac ? fd[i] : '0';
    *neg = sign_neg != pr_neg;
    return w;
}

}  // namespace text
}  // namespace kernels
}  // namespace bolt
