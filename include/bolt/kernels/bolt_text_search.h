// bolt_text_search.h — word / phrase containment and unanchored regex search
// over a Utf8 column, selection-vector in and out (G2CHK-1198, logs).
//
// WORD RULE (one tokenizer, documented): a word byte is [A-Za-z0-9_]. Every
// other byte, including every byte >= 0x80, separates words. This is the
// VictoriaLogs word rule restricted to ASCII.
//
// token_compile(pattern): `pattern` is matched as an EXACT byte string whose
// ends sit on word boundaries: where the pattern's first (last) byte is a word
// byte, the byte before (after) the match must not be one. So
//   has_word:   pattern is one word      "gif"        matches ".gif" "a gif b", not "gifs"
//   has_phrase: pattern spans words      "images/hm"  matches "x images/hm_bg" ? no: "hm"
//               is followed by '_' (a word byte), so it does not; "images/hm" matches
//               "images/hm ok". Separators must be the same bytes, as in VictoriaLogs.
// Case-insensitive = ASCII only (A-Z fold to a-z); non-ASCII bytes compare equal
// or not as bytes.
//
// regex_search_compile(pattern): unanchored search over the re2_* Thompson NFA
// (alternation, classes, {m,n}, (?i)); linear time, no ReDoS. When the pattern
// has a required literal (found at top level, no alternation, no (?i)), the
// column scan first rejects rows lacking it with bytes_find, so the NFA runs
// only on candidates.
//
// All filters take an optional input selection vector (sel_in, n rows) and write
// the surviving row ids to sel_out (may alias sel_in). No allocation.
#pragma once

#include "bolt/bolt_kernel_limits.h"
#include <cassert>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_types.h"
#include "bolt/kernels/bolt_regex.h"
#include "bolt/kernels/bolt_utf8.h"

namespace bolt {
namespace kernels {
namespace text {


BOLT_FORCE_INLINE bool is_word_byte(uint8_t b) noexcept {
    return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || (b >= '0' && b <= '9') || b == '_';
}

BOLT_FORCE_INLINE uint8_t ascii_lower(uint8_t b) noexcept {
    return (b >= 'A' && b <= 'Z') ? static_cast<uint8_t>(b + 32) : b;
}

struct TokenPattern {
    char     needle[kTokenMaxNeedle + 1];   // folded to lower case when ci
    uint32_t len;
    uint8_t  ci;
    uint8_t  need_left;     // first byte is a word byte: left boundary required
    uint8_t  need_right;
    uint8_t  _pad;
};

// False when the pattern is empty or longer than kTokenMaxNeedle.
BOLT_FORCE_INLINE bool token_compile(const char* p, uint32_t n, bool ci,
                                     TokenPattern* out) noexcept {
    assert(out != nullptr);
    assert(p != nullptr || n == 0);
    if (n == 0 || n > kTokenMaxNeedle) return false;
    out->len = n;
    out->ci = ci ? 1 : 0;
    for (uint32_t i = 0; i < n; ++i)
        out->needle[i] = ci ? static_cast<char>(ascii_lower(static_cast<uint8_t>(p[i]))) : p[i];
    out->needle[n] = '\0';
    out->need_left = is_word_byte(static_cast<uint8_t>(p[0])) ? 1 : 0;
    out->need_right = is_word_byte(static_cast<uint8_t>(p[n - 1])) ? 1 : 0;
    out->_pad = 0;
    return true;
}

namespace detail {

BOLT_FORCE_INLINE bool token_bounds_ok(const TokenPattern* pt, const char* hay, uint32_t hlen,
                                       uint32_t at) noexcept {
    if (pt->need_left && at > 0 && is_word_byte(static_cast<uint8_t>(hay[at - 1]))) return false;
    const uint32_t end = at + pt->len;
    if (pt->need_right && end < hlen && is_word_byte(static_cast<uint8_t>(hay[end]))) return false;
    return true;
}

// Case-sensitive: SIMD candidate search, then the boundary check.
BOLT_FORCE_INLINE bool token_match_cs(const TokenPattern* pt, const char* hay, uint32_t hlen) noexcept {
    uint32_t from = 0;
    while (from + pt->len <= hlen) {
        const int32_t k = utf8::bytes_find(hay + from, hlen - from, pt->needle, pt->len);
        if (k < 0) return false;
        const uint32_t at = from + static_cast<uint32_t>(k);
        if (token_bounds_ok(pt, hay, hlen, at)) return true;
        from = at + 1u;
    }
    return false;
}

// ASCII case-insensitive: first-byte filter on both cases, then a folded compare.
BOLT_FORCE_INLINE bool token_match_ci(const TokenPattern* pt, const char* hay, uint32_t hlen) noexcept {
    const uint8_t f = static_cast<uint8_t>(pt->needle[0]);
    const uint8_t fu = (f >= 'a' && f <= 'z') ? static_cast<uint8_t>(f - 32) : f;
    if (hlen < pt->len) return false;
    const uint32_t last = hlen - pt->len;
    for (uint32_t i = 0; i <= last; ++i) {
        const uint8_t h = static_cast<uint8_t>(hay[i]);
        if (h != f && h != fu) continue;
        uint32_t j = 1;
        while (j < pt->len && ascii_lower(static_cast<uint8_t>(hay[i + j])) ==
                                  static_cast<uint8_t>(pt->needle[j])) ++j;
        if (j == pt->len && token_bounds_ok(pt, hay, hlen, i)) return true;
    }
    return false;
}

}  // namespace detail

BOLT_FORCE_INLINE bool token_match_one(const TokenPattern* pt, const char* hay, uint32_t hlen) noexcept {
    assert(pt != nullptr && pt->len > 0);
    assert(hay != nullptr || hlen == 0);
    return pt->ci ? detail::token_match_ci(pt, hay, hlen) : detail::token_match_cs(pt, hay, hlen);
}

// Rows [0,n) (sel_in == nullptr) or the rows named by sel_in[0..n); returns the
// survivor count, ids in sel_out in input order.
BOLT_FORCE_INLINE int64_t utf8_filter_token(const StringView* data, const int32_t* sel_in, int64_t n,
                                            const TokenPattern* pt, int32_t* sel_out,
                                            const char* spilled_base = nullptr) noexcept {
    assert(data != nullptr || n == 0);
    assert(pt != nullptr && (sel_out != nullptr || n == 0));
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        const int32_t r = sel_in != nullptr ? sel_in[i] : static_cast<int32_t>(i);
        const char* p = utf8::sv_bytes(data[r], spilled_base);
        if (token_match_one(pt, p, data[r].length)) sel_out[count++] = r;
    }
    return count;
}

// ------------------------------ regex search ---------------------------

struct RegexSearchPattern {
    regex::Re2Program prog;
    char     literal[kRegexLiteralMax];
    uint32_t literal_len;     // 0: no prefilter
};

// Longest literal every match must contain, from the pattern text. Conservative:
// 0 when the pattern has a top-level `|` or an inline flag (`(?`), because then
// the literal is not required (or is case-folded).
BOLT_FORCE_INLINE uint32_t regex_required_literal(const char* pat, uint32_t n, char* out,
                                                  uint32_t cap) noexcept {
    assert(pat != nullptr || n == 0);
    assert(out != nullptr && cap > 0);
    char cur[kRegexLiteralMax];
    uint32_t cl = 0, best = 0;
    int32_t depth = 0;
    auto flush = [&]() noexcept {
        if (cl > best) { best = cl; std::memcpy(out, cur, cl > cap ? cap : cl); }
        cl = 0;
    };
    for (uint32_t i = 0; i < n; ++i) {
        const char c = pat[i];
        if (c == '\\') {
            if (i + 1 >= n) return 0;
            const char e = pat[i + 1];
            i += 1;
            const bool lit = !((e >= 'a' && e <= 'z') || (e >= 'A' && e <= 'Z') || (e >= '0' && e <= '9'));
            const bool quant = i + 1 < n && (pat[i + 1] == '*' || pat[i + 1] == '?' || pat[i + 1] == '{');
            if (depth == 0 && lit && !quant && cl < kRegexLiteralMax) { cur[cl++] = e; } else flush();
            if (quant || !lit) flush();
            continue;
        }
        if (c == '|') { if (depth == 0) return 0; flush(); continue; }
        if (c == '(') {
            if (i + 1 < n && pat[i + 1] == '?') return 0;
            ++depth; flush(); continue;
        }
        if (c == ')') { if (depth > 0) --depth; flush(); continue; }
        if (c == '[') {
            flush();
            uint32_t j = i + 1;
            if (j < n && pat[j] == '^') ++j;
            if (j < n && pat[j] == ']') ++j;
            while (j < n && pat[j] != ']') { if (pat[j] == '\\') ++j; ++j; }
            i = j;
            continue;
        }
        if (c == '{') { flush(); while (i < n && pat[i] != '}') ++i; continue; }
        const bool meta = c == '.' || c == '*' || c == '+' || c == '?' || c == '{' || c == '}' ||
                          c == '^' || c == '$' || c == ']';
        if (meta) { flush(); continue; }
        const bool quant = i + 1 < n && (pat[i + 1] == '*' || pat[i + 1] == '?' || pat[i + 1] == '{');
        if (depth == 0 && !quant && static_cast<uint8_t>(c) < 0x80u && cl < kRegexLiteralMax) cur[cl++] = c;
        else flush();
        if (quant) flush();
    }
    flush();
    return best > cap ? cap : best;
}

BOLT_FORCE_INLINE bool regex_search_compile(const char* pat, uint32_t n, RegexSearchPattern* out) noexcept {
    assert(out != nullptr);
    assert(pat != nullptr || n == 0);
    out->literal_len = 0;
    if (!regex::re2_compile(pat, n, &out->prog)) return false;
    out->literal_len = regex_required_literal(pat, n, out->literal, kRegexLiteralMax);
    return true;
}

BOLT_FORCE_INLINE bool regex_search_one(const RegexSearchPattern* rp, const char* hay, uint32_t hlen) noexcept {
    assert(rp != nullptr);
    assert(hay != nullptr || hlen == 0);
    if (rp->literal_len != 0 && utf8::bytes_find(hay, hlen, rp->literal, rp->literal_len) < 0) return false;
    return regex::re2_search(&rp->prog, hay, static_cast<int32_t>(hlen));
}

BOLT_FORCE_INLINE int64_t utf8_filter_regex_search(const StringView* data, const int32_t* sel_in, int64_t n,
                                                   const RegexSearchPattern* rp, int32_t* sel_out,
                                                   const char* spilled_base = nullptr) noexcept {
    assert(data != nullptr || n == 0);
    assert(rp != nullptr && (sel_out != nullptr || n == 0));
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        const int32_t r = sel_in != nullptr ? sel_in[i] : static_cast<int32_t>(i);
        const char* p = utf8::sv_bytes(data[r], spilled_base);
        if (regex_search_one(rp, p, data[r].length)) sel_out[count++] = r;
    }
    return count;
}

}  // namespace text
}  // namespace kernels
}  // namespace bolt
