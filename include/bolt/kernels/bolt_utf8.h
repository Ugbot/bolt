// bolt_utf8.h — UTF-8 native operator kernels for SQL VARCHAR / TEXT.
//
// Companion to bolt_string.h (which holds the prefix-equality / starts-with
// kernels). This header ships the operator family Chukonu needs to drive
// VARCHAR/TEXT through the executor without falling back to Arrow compute:
//
//   utf8_compare, utf8_filter_eq, utf8_filter_ne, utf8_hash,
//   utf8_like_compile + utf8_filter_like / utf8_filter_not_like,
//   utf8_substring, utf8_trim / utf8_ltrim / utf8_rtrim,
//   utf8_upper / utf8_lower (ASCII fast path),
//   utf8_byte_length, utf8_char_length, utf8_position,
//   utf8_concat, utf8_replace.
//
// RULES: No exceptions. No RTTI. No heap outside Arena. All fns noexcept.
// BOLT_RESTRICT on array params; >=2 asserts per fn; <=70 LOC per fn.
//
// Byte-resolution model:
//   StringView carries (length, prefix[4], { inline_data[8] | ref }) in 16
//   bytes.  For inline views (length <= 12) the bytes live contiguously at
//   &sv.prefix[0].  For spilled views (length > 12) the kernel must be told
//   where the underlying byte buffer lives.  Every kernel that needs the
//   tail accepts a `const char* spilled_base` parameter; spilled view i's
//   bytes are at `spilled_base + sv.ref.offset`.  Pass nullptr when all
//   inputs are inline.  Output-producing kernels write inline StringViews
//   when the result fits (<= 12 bytes) and otherwise emit spilled views
//   pointing into the caller-supplied Arena (ref.buf_idx=0, ref.offset =
//   byte offset within the arena-tracked output buffer).  Callers wire the
//   arena's contiguous bytes back to the column-side spilled buffer.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_hash.h"
#include "bolt/bolt_port.h"
#include "bolt/bolt_types.h"

namespace bolt {
namespace kernels {
namespace utf8 {

// ---------------------------------------------------------------------------
// Internal helpers — byte resolution + StringView build.
// ---------------------------------------------------------------------------

BOLT_FORCE_INLINE const char* sv_bytes(
        const StringView& s, const char* spilled_base) noexcept {
    // Inline: prefix[4] + inline_data[8] are contiguous in the union layout,
    // so &s.prefix[0] is the canonical byte pointer for inline views.
    if (s.length <= 12u) return s.prefix;
    assert(spilled_base != nullptr);
    return spilled_base + s.ref.offset;
}

BOLT_FORCE_INLINE StringView sv_make_inline(
        const char* BOLT_RESTRICT bytes, uint32_t len) noexcept {
    assert(len <= 12u);
    StringView r;
    memset(&r, 0, sizeof(r));
    r.length = len;
    const uint32_t p = (len < 4u) ? len : 4u;
    if (p > 0) memcpy(r.prefix, bytes, p);
    if (len > 4u) memcpy(r.inline_data, bytes + 4, len - 4u);
    return r;
}

// Allocate `len` bytes from `arena`, copy from `bytes`, and produce a
// spilled StringView with `ref.offset = byte_offset_from_arena_anchor`.
// `arena_anchor` is the base pointer the caller will resolve against;
// typically the first allocation pointer of this batch's output buffer.
BOLT_FORCE_INLINE StringView sv_make_spilled(
        const char* BOLT_RESTRICT bytes, uint32_t len,
        Arena* arena, const char* arena_anchor) noexcept {
    assert(arena != nullptr);
    assert(arena_anchor != nullptr);
    assert(len > 12u);
    char* dst = static_cast<char*>(arena->allocate(len, 1));
    assert(dst != nullptr);
    memcpy(dst, bytes, len);
    StringView r;
    memset(&r, 0, sizeof(r));
    r.length = len;
    memcpy(r.prefix, bytes, 4);
    r.ref.buf_idx = 0;
    r.ref.offset  = static_cast<uint32_t>(dst - arena_anchor);
    return r;
}

// Build a view from freshly-arena'd bytes, dispatching inline vs spilled.
BOLT_FORCE_INLINE StringView sv_make(
        const char* BOLT_RESTRICT bytes, uint32_t len,
        Arena* arena, const char* arena_anchor) noexcept {
    return (len <= 12u) ? sv_make_inline(bytes, len)
                        : sv_make_spilled(bytes, len, arena, arena_anchor);
}

// SIMD-accelerated bytewise memcmp(==) over `n` bytes.  Returns true iff
// every byte matches.  Used by utf8_compare / utf8_filter_eq fast path.
BOLT_FORCE_INLINE bool bytes_equal_simd(
        const char* BOLT_RESTRICT a, const char* BOLT_RESTRICT b,
        uint32_t n) noexcept {
#if BOLT_SIMD_AVX2 || BOLT_SIMD_SSE42 || BOLT_SIMD_NEON
    using namespace bolt::simd;
    uint32_t i = 0;
    constexpr int LANES = bmm_lanes_i8;
    while (i + static_cast<uint32_t>(LANES) <= n) {
        bmm_vec_i8 va = bmm_loadu_i8(reinterpret_cast<const int8_t*>(a + i));
        bmm_vec_i8 vb = bmm_loadu_i8(reinterpret_cast<const int8_t*>(b + i));
        bmm_vec_i8 vc = bmm_cmpeq_i8(va, vb);
        const unsigned mask = static_cast<unsigned>(bmm_movemask_i8(vc));
        constexpr unsigned all = (LANES == 32) ? 0xFFFFFFFFu : 0xFFFFu;
        if (mask != all) return false;
        i += static_cast<uint32_t>(LANES);
    }
    if (i < n) return memcmp(a + i, b + i, n - i) == 0;
    return true;
#else
    return memcmp(a, b, n) == 0;
#endif
}

// Bytewise memcmp returning -1/0/+1.
BOLT_FORCE_INLINE int32_t bytes_compare(
        const char* BOLT_RESTRICT a, uint32_t alen,
        const char* BOLT_RESTRICT b, uint32_t blen) noexcept {
    const uint32_t n = (alen < blen) ? alen : blen;
    if (n > 0) {
        const int c = memcmp(a, b, n);
        if (c < 0) return -1;
        if (c > 0) return  1;
    }
    if (alen < blen) return -1;
    if (alen > blen) return  1;
    return 0;
}

// memmem byte search (naive; small needles dominate SQL workloads).
// Returns 0-based byte index or -1.
BOLT_FORCE_INLINE int32_t bytes_find(
        const char* BOLT_RESTRICT hay, uint32_t hlen,
        const char* BOLT_RESTRICT ndl, uint32_t nlen) noexcept {
    if (nlen == 0) return 0;
    if (nlen > hlen) return -1;
    const uint32_t last = hlen - nlen;
    const char first = ndl[0];
    for (uint32_t k = 0; k <= last; ++k) {
        if (hay[k] == first && memcmp(hay + k, ndl, nlen) == 0) {
            return static_cast<int32_t>(k);
        }
    }
    return -1;
}

// ===========================================================================
// 1. utf8_compare — pairwise sign(memcmp).
// ===========================================================================
BOLT_FORCE_INLINE void utf8_compare(
        const StringView* BOLT_RESTRICT a,
        const StringView* BOLT_RESTRICT b,
        int32_t*          BOLT_RESTRICT out,
        int64_t           n,
        const char*       spilled_base_a = nullptr,
        const char*       spilled_base_b = nullptr) noexcept {
    assert(a != nullptr || n == 0);
    assert(b != nullptr || n == 0);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) {
        const char* pa = sv_bytes(a[i], spilled_base_a);
        const char* pb = sv_bytes(b[i], spilled_base_b);
        out[i] = bytes_compare(pa, a[i].length, pb, b[i].length);
    }
}

// ===========================================================================
// 2. utf8_filter_eq / utf8_filter_ne — equality selection vector.
// ===========================================================================
BOLT_FORCE_INLINE int64_t utf8_filter_eq(
        const StringView* BOLT_RESTRICT data, int64_t n,
        StringView scalar,
        int32_t*   BOLT_RESTRICT sel_out,
        const char* spilled_base_data = nullptr,
        const char* spilled_base_scalar = nullptr) noexcept {
    assert(data    != nullptr || n == 0);
    assert(sel_out != nullptr || n == 0);
    assert(n >= 0);
    const char* sb = sv_bytes(scalar, spilled_base_scalar);
    const uint32_t slen = scalar.length;
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        const StringView& s = data[i];
        bool match = (s.length == slen);
        if (match && slen > 0) {
            // 4-byte prefix screen first (cheap, branch-predicted).
            match = (memcmp(s.prefix, scalar.prefix, (slen < 4u) ? slen : 4u) == 0);
            if (match && slen > 4u) {
                const char* p = sv_bytes(s, spilled_base_data);
                match = bytes_equal_simd(p + 4, sb + 4, slen - 4u);
            }
        }
        sel_out[count] = static_cast<int32_t>(i);
        count += match ? 1 : 0;
    }
    return count;
}

BOLT_FORCE_INLINE int64_t utf8_filter_ne(
        const StringView* BOLT_RESTRICT data, int64_t n,
        StringView scalar,
        int32_t*   BOLT_RESTRICT sel_out,
        const char* spilled_base_data = nullptr,
        const char* spilled_base_scalar = nullptr) noexcept {
    assert(data    != nullptr || n == 0);
    assert(sel_out != nullptr || n == 0);
    assert(n >= 0);
    const char* sb = sv_bytes(scalar, spilled_base_scalar);
    const uint32_t slen = scalar.length;
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        const StringView& s = data[i];
        bool eq = (s.length == slen);
        if (eq && slen > 0) {
            eq = (memcmp(s.prefix, scalar.prefix, (slen < 4u) ? slen : 4u) == 0);
            if (eq && slen > 4u) {
                const char* p = sv_bytes(s, spilled_base_data);
                eq = bytes_equal_simd(p + 4, sb + 4, slen - 4u);
            }
        }
        sel_out[count] = static_cast<int32_t>(i);
        count += eq ? 0 : 1;
    }
    return count;
}

// ===========================================================================
// 3. utf8_hash — 8-byte-stride wyhash3-family mixer.  Result is suitable for
//    SwissTable seeding (same finalizer family as `swiss_mix_wyhash3`).
// ===========================================================================
BOLT_FORCE_INLINE uint64_t utf8_hash_one(
        const char* BOLT_RESTRICT bytes, uint32_t len) noexcept {
    uint64_t h = 0x9E3779B97F4A7C15ULL ^ static_cast<uint64_t>(len);
    uint32_t i = 0;
    while (i + 8u <= len) {
        uint64_t w;
        memcpy(&w, bytes + i, 8);
        h ^= w;
        h = swiss_mix_wyhash3(h);
        i += 8u;
    }
    if (i < len) {
        uint64_t tail = 0;
        memcpy(&tail, bytes + i, len - i);
        h ^= tail;
        h = swiss_mix_wyhash3(h);
    }
    return h;
}

BOLT_FORCE_INLINE void utf8_hash(
        const StringView* BOLT_RESTRICT data, int64_t n,
        uint64_t*         BOLT_RESTRICT out,
        const char* spilled_base = nullptr) noexcept {
    assert(data != nullptr || n == 0);
    assert(out  != nullptr || n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) {
        const char* p = sv_bytes(data[i], spilled_base);
        out[i] = utf8_hash_one(p, data[i].length);
    }
}

// ===========================================================================
// 4. LIKE pattern — compile once, match many.
// ===========================================================================

enum class LikeKind : uint8_t {
    Exact    = 0,
    Prefix   = 1,
    Suffix   = 2,
    Contains = 3,
    General  = 4,
};

// Small POD; passes by pointer.  dfa_buf holds the literalized pattern for
// the General case: alternating run-length-encoded segments of either
// (LIT, len, bytes...) or (PCT, ) or (UND, ).  Matched by a tiny recursive
// matcher (bounded by dfa_len, no heap).
struct CompiledLike {
    LikeKind kind;
    uint8_t  _pad[3];
    uint32_t literal_len;       // for Exact/Prefix/Suffix/Contains
    char     literal[224];      // inline buffer (TPC-H patterns are tiny)
    uint8_t  dfa_buf[256];      // tokenized pattern for General
    uint16_t dfa_len;
    uint16_t _pad2;
};
static_assert(sizeof(CompiledLike) <= 512, "CompiledLike too large");

// Pattern tokens used by General matcher.
enum : uint8_t {
    kLikeTokLit = 1, // followed by uint8 len, len bytes
    kLikeTokPct = 2, // % — match any (greedy)
    kLikeTokUnd = 3, // _ — match exactly one byte
};

// Compile a SQL LIKE pattern.  Escape via doubled '%' / '_' not supported
// (SQL standard uses ESCAPE clause; chukonu lowers that to a different rule).
BOLT_FORCE_INLINE bool utf8_like_compile(
        StringView pattern, CompiledLike* out,
        const char* spilled_base = nullptr) noexcept {
    assert(out != nullptr);
    memset(out, 0, sizeof(*out));
    const char* p = sv_bytes(pattern, spilled_base);
    const uint32_t n = pattern.length;
    // Fast-path classification.
    if (n == 0) {
        out->kind = LikeKind::Exact;
        out->literal_len = 0;
        return true;
    }
    // Count % and _ to detect simple shapes.
    uint32_t pct = 0, und = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (p[i] == '%') ++pct;
        else if (p[i] == '_') ++und;
    }
    if (pct == 0 && und == 0) {
        if (n > sizeof(out->literal)) return false;
        out->kind = LikeKind::Exact;
        out->literal_len = n;
        memcpy(out->literal, p, n);
        return true;
    }
    if (und == 0 && pct == 1) {
        if (p[0] == '%') {
            const uint32_t L = n - 1;
            if (L > sizeof(out->literal)) return false;
            out->kind = LikeKind::Suffix;
            out->literal_len = L;
            memcpy(out->literal, p + 1, L);
            return true;
        }
        if (p[n - 1] == '%') {
            const uint32_t L = n - 1;
            if (L > sizeof(out->literal)) return false;
            out->kind = LikeKind::Prefix;
            out->literal_len = L;
            memcpy(out->literal, p, L);
            return true;
        }
    }
    if (und == 0 && pct == 2 && p[0] == '%' && p[n - 1] == '%') {
        const uint32_t L = n - 2;
        if (L > sizeof(out->literal)) return false;
        out->kind = LikeKind::Contains;
        out->literal_len = L;
        memcpy(out->literal, p + 1, L);
        return true;
    }
    // General case — tokenize into dfa_buf.
    out->kind = LikeKind::General;
    uint32_t w = 0;
    uint32_t i = 0;
    while (i < n) {
        if (p[i] == '%') {
            if (w + 1u > sizeof(out->dfa_buf)) return false;
            out->dfa_buf[w++] = kLikeTokPct;
            ++i;
        } else if (p[i] == '_') {
            if (w + 1u > sizeof(out->dfa_buf)) return false;
            out->dfa_buf[w++] = kLikeTokUnd;
            ++i;
        } else {
            uint32_t j = i;
            while (j < n && p[j] != '%' && p[j] != '_') ++j;
            const uint32_t L = j - i;
            if (L > 255u) return false;
            if (w + 2u + L > sizeof(out->dfa_buf)) return false;
            out->dfa_buf[w++] = kLikeTokLit;
            out->dfa_buf[w++] = static_cast<uint8_t>(L);
            memcpy(out->dfa_buf + w, p + i, L);
            w += L;
            i = j;
        }
    }
    out->dfa_len = static_cast<uint16_t>(w);
    return true;
}

// General-case LIKE matcher.  Bounded by dfa_len + hlen.  Iterative
// backtracking — no recursion.  Returns true iff the entire haystack
// matches the tokenized pattern.
BOLT_FORCE_INLINE bool like_match_general(
        const uint8_t* BOLT_RESTRICT tok, uint16_t toklen,
        const char* BOLT_RESTRICT hay, uint32_t hlen) noexcept {
    assert(tok != nullptr || toklen == 0);
    assert(hay != nullptr || hlen == 0);
    uint16_t ti = 0;        // token cursor
    uint32_t hi = 0;        // haystack cursor
    uint16_t star_ti = 0xFFFFu;
    uint32_t star_hi = 0;
    while (hi <= hlen) {
        if (ti < toklen) {
            const uint8_t op = tok[ti];
            if (op == kLikeTokLit) {
                const uint8_t L = tok[ti + 1];
                if (hi + L <= hlen &&
                    memcmp(hay + hi, tok + ti + 2, L) == 0) {
                    ti += 2u + L;
                    hi += L;
                    continue;
                }
            } else if (op == kLikeTokUnd) {
                if (hi < hlen) { ti += 1u; hi += 1u; continue; }
            } else { // kLikeTokPct
                star_ti = static_cast<uint16_t>(ti + 1u);
                star_hi = hi;
                ti += 1u;
                continue;
            }
        } else if (hi == hlen) {
            return true;
        }
        if (star_ti != 0xFFFFu && star_hi < hlen) {
            ti = star_ti;
            hi = ++star_hi;
            continue;
        }
        return false;
    }
    return false;
}

BOLT_FORCE_INLINE bool utf8_like_match_one(
        const CompiledLike* BOLT_RESTRICT pat,
        const char* BOLT_RESTRICT hay, uint32_t hlen) noexcept {
    assert(pat != nullptr);
    switch (pat->kind) {
    case LikeKind::Exact:
        return hlen == pat->literal_len &&
               (pat->literal_len == 0 ||
                memcmp(hay, pat->literal, pat->literal_len) == 0);
    case LikeKind::Prefix:
        return hlen >= pat->literal_len &&
               memcmp(hay, pat->literal, pat->literal_len) == 0;
    case LikeKind::Suffix:
        return hlen >= pat->literal_len &&
               memcmp(hay + (hlen - pat->literal_len),
                      pat->literal, pat->literal_len) == 0;
    case LikeKind::Contains:
        return bytes_find(hay, hlen, pat->literal, pat->literal_len) >= 0;
    case LikeKind::General:
        return like_match_general(pat->dfa_buf, pat->dfa_len, hay, hlen);
    }
    return false;
}

BOLT_FORCE_INLINE int64_t utf8_filter_like(
        const StringView*   BOLT_RESTRICT data, int64_t n,
        const CompiledLike* pattern,
        int32_t*            BOLT_RESTRICT sel_out,
        const char* spilled_base = nullptr) noexcept {
    assert(data    != nullptr || n == 0);
    assert(pattern != nullptr);
    assert(sel_out != nullptr || n == 0);
    assert(n >= 0);
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        const char* p = sv_bytes(data[i], spilled_base);
        const bool match = utf8_like_match_one(pattern, p, data[i].length);
        sel_out[count] = static_cast<int32_t>(i);
        count += match ? 1 : 0;
    }
    return count;
}

BOLT_FORCE_INLINE int64_t utf8_filter_not_like(
        const StringView*   BOLT_RESTRICT data, int64_t n,
        const CompiledLike* pattern,
        int32_t*            BOLT_RESTRICT sel_out,
        const char* spilled_base = nullptr) noexcept {
    assert(data    != nullptr || n == 0);
    assert(pattern != nullptr);
    assert(sel_out != nullptr || n == 0);
    assert(n >= 0);
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        const char* p = sv_bytes(data[i], spilled_base);
        const bool match = utf8_like_match_one(pattern, p, data[i].length);
        sel_out[count] = static_cast<int32_t>(i);
        count += match ? 0 : 1;
    }
    return count;
}

// ===========================================================================
// 5. SUBSTRING(s, start_1based, length).  SQL semantics:
//      start_1based <= 0  → treated as 1 (clamped).
//      length      < 0   → empty result.
//      length saturates at remaining bytes.
//    NOTE: byte-level slicing.  Char-level slicing would require a
//    codepoint walk; SQL standard permits either for VARCHAR.
// ===========================================================================
BOLT_FORCE_INLINE void utf8_substring(
        const StringView* BOLT_RESTRICT data, int64_t n,
        int32_t start_1based, int32_t length,
        StringView*       BOLT_RESTRICT out,
        Arena* arena,
        const char* arena_anchor,
        const char* spilled_base = nullptr) noexcept {
    assert(data != nullptr || n == 0);
    assert(out  != nullptr || n == 0);
    assert(arena != nullptr);
    assert(arena_anchor != nullptr);
    assert(n >= 0);
    const int32_t s0 = (start_1based < 1) ? 0 : (start_1based - 1);
    const int32_t L  = (length < 0) ? 0 : length;
    for (int64_t i = 0; i < n; ++i) {
        const uint32_t slen = data[i].length;
        const char* p = sv_bytes(data[i], spilled_base);
        uint32_t start_u = (static_cast<uint32_t>(s0) >= slen)
                          ? slen : static_cast<uint32_t>(s0);
        uint32_t rem = slen - start_u;
        uint32_t take = (static_cast<uint32_t>(L) < rem)
                       ? static_cast<uint32_t>(L) : rem;
        out[i] = sv_make(p + start_u, take, arena, arena_anchor);
    }
}

// ===========================================================================
// 6. TRIM / LTRIM / RTRIM — whitespace by default (space/tab/LF/CR/VT/FF).
// ===========================================================================
BOLT_FORCE_INLINE bool is_ws_ascii(unsigned char c) noexcept {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0x0B || c == 0x0C;
}

BOLT_FORCE_INLINE void utf8_ltrim(
        const StringView* BOLT_RESTRICT in, int64_t n,
        StringView*       BOLT_RESTRICT out,
        Arena* arena, const char* arena_anchor,
        const char* spilled_base = nullptr) noexcept {
    assert(in  != nullptr || n == 0);
    assert(out != nullptr || n == 0);
    assert(arena != nullptr);
    assert(arena_anchor != nullptr);
    for (int64_t i = 0; i < n; ++i) {
        const uint32_t L = in[i].length;
        const char* p = sv_bytes(in[i], spilled_base);
        uint32_t k = 0;
        while (k < L && is_ws_ascii(static_cast<unsigned char>(p[k]))) ++k;
        out[i] = sv_make(p + k, L - k, arena, arena_anchor);
    }
}

BOLT_FORCE_INLINE void utf8_rtrim(
        const StringView* BOLT_RESTRICT in, int64_t n,
        StringView*       BOLT_RESTRICT out,
        Arena* arena, const char* arena_anchor,
        const char* spilled_base = nullptr) noexcept {
    assert(in  != nullptr || n == 0);
    assert(out != nullptr || n == 0);
    assert(arena != nullptr);
    assert(arena_anchor != nullptr);
    for (int64_t i = 0; i < n; ++i) {
        const uint32_t L = in[i].length;
        const char* p = sv_bytes(in[i], spilled_base);
        uint32_t k = L;
        while (k > 0 && is_ws_ascii(static_cast<unsigned char>(p[k - 1]))) --k;
        out[i] = sv_make(p, k, arena, arena_anchor);
    }
}

BOLT_FORCE_INLINE void utf8_trim(
        const StringView* BOLT_RESTRICT in, int64_t n,
        StringView*       BOLT_RESTRICT out,
        Arena* arena, const char* arena_anchor,
        const char* spilled_base = nullptr) noexcept {
    assert(in  != nullptr || n == 0);
    assert(out != nullptr || n == 0);
    assert(arena != nullptr);
    assert(arena_anchor != nullptr);
    for (int64_t i = 0; i < n; ++i) {
        const uint32_t L = in[i].length;
        const char* p = sv_bytes(in[i], spilled_base);
        uint32_t a = 0;
        while (a < L && is_ws_ascii(static_cast<unsigned char>(p[a]))) ++a;
        uint32_t b = L;
        while (b > a && is_ws_ascii(static_cast<unsigned char>(p[b - 1]))) --b;
        out[i] = sv_make(p + a, b - a, arena, arena_anchor);
    }
}

// ===========================================================================
// 7. UPPER / LOWER — ASCII fast path.  Non-ASCII bytes pass through unchanged.
//    Full Unicode case-folding is TODO (requires ICU-style tables).
// ===========================================================================
BOLT_FORCE_INLINE void utf8_upper(
        const StringView* BOLT_RESTRICT in, int64_t n,
        StringView*       BOLT_RESTRICT out,
        Arena* arena, const char* arena_anchor,
        const char* spilled_base = nullptr) noexcept {
    assert(in  != nullptr || n == 0);
    assert(out != nullptr || n == 0);
    assert(arena != nullptr);
    assert(arena_anchor != nullptr);
    for (int64_t i = 0; i < n; ++i) {
        const uint32_t L = in[i].length;
        const char* p = sv_bytes(in[i], spilled_base);
        if (L == 0) { out[i] = sv_make_inline(p, 0); continue; }
        char inline_buf[12];
        char* dst;
        char* anchor_dst = nullptr;
        if (L <= 12u) {
            dst = inline_buf;
        } else {
            dst = static_cast<char*>(arena->allocate(L, 1));
            assert(dst != nullptr);
            anchor_dst = dst;
        }
        for (uint32_t k = 0; k < L; ++k) {
            unsigned char c = static_cast<unsigned char>(p[k]);
            dst[k] = (c >= 'a' && c <= 'z') ? static_cast<char>(c - 32) : static_cast<char>(c);
        }
        if (L <= 12u) {
            out[i] = sv_make_inline(dst, L);
        } else {
            StringView r;
            memset(&r, 0, sizeof(r));
            r.length = L;
            memcpy(r.prefix, dst, 4);
            r.ref.buf_idx = 0;
            r.ref.offset  = static_cast<uint32_t>(anchor_dst - arena_anchor);
            out[i] = r;
        }
    }
}

BOLT_FORCE_INLINE void utf8_lower(
        const StringView* BOLT_RESTRICT in, int64_t n,
        StringView*       BOLT_RESTRICT out,
        Arena* arena, const char* arena_anchor,
        const char* spilled_base = nullptr) noexcept {
    assert(in  != nullptr || n == 0);
    assert(out != nullptr || n == 0);
    assert(arena != nullptr);
    assert(arena_anchor != nullptr);
    for (int64_t i = 0; i < n; ++i) {
        const uint32_t L = in[i].length;
        const char* p = sv_bytes(in[i], spilled_base);
        if (L == 0) { out[i] = sv_make_inline(p, 0); continue; }
        char inline_buf[12];
        char* dst;
        char* anchor_dst = nullptr;
        if (L <= 12u) {
            dst = inline_buf;
        } else {
            dst = static_cast<char*>(arena->allocate(L, 1));
            assert(dst != nullptr);
            anchor_dst = dst;
        }
        for (uint32_t k = 0; k < L; ++k) {
            unsigned char c = static_cast<unsigned char>(p[k]);
            dst[k] = (c >= 'A' && c <= 'Z') ? static_cast<char>(c + 32) : static_cast<char>(c);
        }
        if (L <= 12u) {
            out[i] = sv_make_inline(dst, L);
        } else {
            StringView r;
            memset(&r, 0, sizeof(r));
            r.length = L;
            memcpy(r.prefix, dst, 4);
            r.ref.buf_idx = 0;
            r.ref.offset  = static_cast<uint32_t>(anchor_dst - arena_anchor);
            out[i] = r;
        }
    }
}

// ===========================================================================
// 8. LENGTH — byte count and codepoint count.
// ===========================================================================
BOLT_FORCE_INLINE void utf8_byte_length(
        const StringView* BOLT_RESTRICT in, int64_t n,
        int32_t*          BOLT_RESTRICT out) noexcept {
    assert(in  != nullptr || n == 0);
    assert(out != nullptr || n == 0);
    for (int64_t i = 0; i < n; ++i) {
        out[i] = static_cast<int32_t>(in[i].length);
    }
}

BOLT_FORCE_INLINE int32_t utf8_count_codepoints(
        const char* BOLT_RESTRICT p, uint32_t L) noexcept {
    int32_t c = 0;
    for (uint32_t k = 0; k < L; ++k) {
        // Count bytes whose top two bits aren't '10' (i.e. not continuation).
        if ((static_cast<unsigned char>(p[k]) & 0xC0u) != 0x80u) ++c;
    }
    return c;
}

BOLT_FORCE_INLINE void utf8_char_length(
        const StringView* BOLT_RESTRICT in, int64_t n,
        int32_t*          BOLT_RESTRICT out,
        const char* spilled_base = nullptr) noexcept {
    assert(in  != nullptr || n == 0);
    assert(out != nullptr || n == 0);
    for (int64_t i = 0; i < n; ++i) {
        const char* p = sv_bytes(in[i], spilled_base);
        out[i] = utf8_count_codepoints(p, in[i].length);
    }
}

// ===========================================================================
// 9. POSITION — 1-based byte index of needle within each haystack; 0 if missing.
// ===========================================================================
BOLT_FORCE_INLINE void utf8_position(
        const StringView* BOLT_RESTRICT haystack, int64_t n,
        StringView        needle,
        int32_t*          BOLT_RESTRICT out_pos,
        const char* spilled_base_h = nullptr,
        const char* spilled_base_n = nullptr) noexcept {
    assert(haystack != nullptr || n == 0);
    assert(out_pos  != nullptr || n == 0);
    const char* nb = sv_bytes(needle, spilled_base_n);
    const uint32_t nlen = needle.length;
    for (int64_t i = 0; i < n; ++i) {
        const char* hb = sv_bytes(haystack[i], spilled_base_h);
        const int32_t at = bytes_find(hb, haystack[i].length, nb, nlen);
        out_pos[i] = (at < 0) ? 0 : (at + 1);
    }
}

// ===========================================================================
// 10. CONCAT (binary).  Output bytes materialized into the arena.
// ===========================================================================
BOLT_FORCE_INLINE void utf8_concat(
        const StringView* BOLT_RESTRICT a,
        const StringView* BOLT_RESTRICT b,
        int64_t n,
        StringView*       BOLT_RESTRICT out,
        Arena* arena, const char* arena_anchor,
        const char* spilled_base_a = nullptr,
        const char* spilled_base_b = nullptr) noexcept {
    assert(a != nullptr || n == 0);
    assert(b != nullptr || n == 0);
    assert(out != nullptr || n == 0);
    assert(arena != nullptr);
    assert(arena_anchor != nullptr);
    for (int64_t i = 0; i < n; ++i) {
        const uint32_t la = a[i].length;
        const uint32_t lb = b[i].length;
        const uint32_t L  = la + lb;
        const char* pa = sv_bytes(a[i], spilled_base_a);
        const char* pb = sv_bytes(b[i], spilled_base_b);
        if (L <= 12u) {
            char buf[12];
            if (la > 0) memcpy(buf, pa, la);
            if (lb > 0) memcpy(buf + la, pb, lb);
            out[i] = sv_make_inline(buf, L);
        } else {
            char* dst = static_cast<char*>(arena->allocate(L, 1));
            assert(dst != nullptr);
            if (la > 0) memcpy(dst, pa, la);
            if (lb > 0) memcpy(dst + la, pb, lb);
            StringView r;
            memset(&r, 0, sizeof(r));
            r.length = L;
            memcpy(r.prefix, dst, 4);
            r.ref.buf_idx = 0;
            r.ref.offset  = static_cast<uint32_t>(dst - arena_anchor);
            out[i] = r;
        }
    }
}

// ===========================================================================
// 11. REPLACE — every occurrence of `from` with `to`.  Allocates output into
//     arena.  If `from` is empty, the input is passed through unchanged.
// ===========================================================================
BOLT_FORCE_INLINE void utf8_replace(
        const StringView* BOLT_RESTRICT data, int64_t n,
        StringView from, StringView to,
        StringView*       BOLT_RESTRICT out,
        Arena* arena, const char* arena_anchor,
        const char* spilled_base_data = nullptr,
        const char* spilled_base_from = nullptr,
        const char* spilled_base_to   = nullptr) noexcept {
    assert(data != nullptr || n == 0);
    assert(out  != nullptr || n == 0);
    assert(arena != nullptr);
    assert(arena_anchor != nullptr);
    const char* fb = sv_bytes(from, spilled_base_from);
    const char* tb = sv_bytes(to,   spilled_base_to);
    const uint32_t flen = from.length;
    const uint32_t tlen = to.length;
    for (int64_t i = 0; i < n; ++i) {
        const uint32_t L = data[i].length;
        const char* p = sv_bytes(data[i], spilled_base_data);
        if (flen == 0 || L < flen) {
            out[i] = sv_make(p, L, arena, arena_anchor);
            continue;
        }
        // Worst-case output size: every byte is a match start.
        // upper_bound = L * tlen / flen + tlen.  Use 2-pass to keep it tight:
        // pass 1 — count matches, pass 2 — write.
        uint32_t matches = 0;
        uint32_t k = 0;
        while (k + flen <= L) {
            if (p[k] == fb[0] && memcmp(p + k, fb, flen) == 0) {
                ++matches; k += flen;
            } else { ++k; }
        }
        if (matches == 0) {
            out[i] = sv_make(p, L, arena, arena_anchor);
            continue;
        }
        const uint64_t outL64 = static_cast<uint64_t>(L)
                              + static_cast<uint64_t>(matches)
                              * (static_cast<int64_t>(tlen) - static_cast<int64_t>(flen));
        const uint32_t outL = static_cast<uint32_t>(outL64);
        if (outL <= 12u) {
            char buf[12];
            uint32_t r = 0, w = 0;
            while (r < L) {
                if (r + flen <= L && p[r] == fb[0] &&
                    memcmp(p + r, fb, flen) == 0) {
                    if (tlen > 0) memcpy(buf + w, tb, tlen);
                    w += tlen; r += flen;
                } else { buf[w++] = p[r++]; }
            }
            out[i] = sv_make_inline(buf, outL);
        } else {
            char* dst = static_cast<char*>(arena->allocate(outL, 1));
            assert(dst != nullptr);
            uint32_t r = 0, w = 0;
            while (r < L) {
                if (r + flen <= L && p[r] == fb[0] &&
                    memcmp(p + r, fb, flen) == 0) {
                    if (tlen > 0) memcpy(dst + w, tb, tlen);
                    w += tlen; r += flen;
                } else { dst[w++] = p[r++]; }
            }
            StringView rv;
            memset(&rv, 0, sizeof(rv));
            rv.length = outL;
            memcpy(rv.prefix, dst, 4);
            rv.ref.buf_idx = 0;
            rv.ref.offset  = static_cast<uint32_t>(dst - arena_anchor);
            out[i] = rv;
        }
    }
}

} // namespace utf8
} // namespace kernels
} // namespace bolt
