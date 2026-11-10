// Column-at-a-time string kernels with a bounded byte sink (G2LAUNCH-152).
//
// Every input is a StringView column plus the base its spilled (>12-byte)
// rows resolve against. Every string output goes to an SvSink: a caller-owned
// byte range that grows DOWNWARD from `top` towards `floor` (so a caller can
// bump-allocate fixed-width intermediates upward from the same block). Output
// views spill against `sink.base`. A kernel that runs out of sink returns
// false and leaves the caller to take its exact per-row path.
#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_port.h"
#include "bolt/bolt_types.h"
#include "bolt/kernels/bolt_utf8.h"
#include "bolt/kernels/bolt_utf8_pycase.h"
#include "bolt/kernels/bolt_pystr.h"

namespace bolt::kernels::utf8 {

struct SvSink {
    char*    base;
    uint32_t floor;   // lowest byte offset the sink may use
    uint32_t top;     // next allocation ends here (exclusive)
};

BOLT_FORCE_INLINE char* sv_sink_alloc(SvSink* s, uint32_t n) noexcept {
    assert(s != nullptr && s->top >= s->floor);
    if (s->top - s->floor < n) return nullptr;
    s->top -= n;
    return s->base + s->top;
}

// A view of `len` bytes at `p`: inline, or copied into the sink.
BOLT_FORCE_INLINE bool sv_put(SvSink* s, const char* p, uint32_t len, StringView* out) noexcept {
    if (len <= 12u) { *out = sv_make_inline(p, len); return true; }
    char* dst = sv_sink_alloc(s, len);
    if (dst == nullptr) return false;
    memcpy(dst, p, len);
    StringView r;
    memset(&r, 0, sizeof(r));
    r.length = len;
    memcpy(r.prefix, p, 4);
    r.ref.buf_idx = 0;
    r.ref.offset = static_cast<uint32_t>(dst - s->base);
    *out = r;
    return true;
}

// Re-anchor a column's spilled rows into the sink (inline rows unchanged).
inline bool sv_rebase_col(const StringView* BOLT_RESTRICT in, const char* base, int64_t n,
                          SvSink* sink, StringView* BOLT_RESTRICT out) noexcept {
    assert(n >= 0 && sink != nullptr);
    for (int64_t i = 0; i < n; ++i) {
        if (in[i].length <= 12u) { out[i] = in[i]; continue; }
        if (!sv_put(sink, sv_bytes(in[i], base), in[i].length, &out[i])) return false;
    }
    return true;
}

// out[i] = mask[i] ? a[i] : b[i]; a and b resolve against one base.
inline void sv_select_col(const int64_t* BOLT_RESTRICT mask, const StringView* a,
                          const StringView* b, int64_t n, StringView* BOLT_RESTRICT out) noexcept {
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) out[i] = mask[i] != 0 ? a[i] : b[i];
}

// Character (code point) count per row.
inline void utf8_length_col(const StringView* BOLT_RESTRICT in, const char* base, int64_t n,
                            int64_t* BOLT_RESTRICT out) noexcept {
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) {
        out[i] = static_cast<int64_t>(utf8_count_codepoints(sv_bytes(in[i], base), in[i].length));
    }
}

enum class SvMatch : uint8_t { Prefix, Suffix, Contains };

// 1 where the row starts with / ends with / contains `needle`, else 0.
inline void utf8_match_col(const StringView* BOLT_RESTRICT in, const char* base, int64_t n,
                           const char* needle, uint32_t nlen, SvMatch how,
                           int64_t* BOLT_RESTRICT out) noexcept {
    assert(n >= 0 && (needle != nullptr || nlen == 0));
    for (int64_t i = 0; i < n; ++i) {
        const uint32_t L = in[i].length;
        const char* p = sv_bytes(in[i], base);
        int64_t hit = 0;
        if (L >= nlen) {
            if (how == SvMatch::Prefix) hit = memcmp(p, needle, nlen) == 0;
            else if (how == SvMatch::Suffix) hit = memcmp(p + (L - nlen), needle, nlen) == 0;
            else if (nlen == 0) hit = 1;
            else {
                for (uint32_t k = 0; k + nlen <= L && hit == 0; ++k) {   // bounded by L
                    hit = (p[k] == needle[0] && memcmp(p + k, needle, nlen) == 0) ? 1 : 0;
                }
            }
        }
        out[i] = hit;
    }
}

// Python str.lower / str.upper (utf8_case_map_python) per row.
inline bool utf8_case_python_col(const StringView* BOLT_RESTRICT in, const char* base, int64_t n,
                                 bool upper, SvSink* sink, StringView* BOLT_RESTRICT out) noexcept {
    assert(n >= 0 && sink != nullptr);
    char small[12 * 6];
    for (int64_t i = 0; i < n; ++i) {
        const char* p = sv_bytes(in[i], base);
        const uint32_t L = in[i].length;
        const uint32_t M = utf8_is_ascii(p, L) ? L : utf8_case_map_python(p, L, upper, nullptr);
        if (M <= 12u) {
            if (L <= 12u) {
                const uint32_t w = utf8_case_map_python(p, L, upper, small);
                out[i] = sv_make_inline(small, w);
                continue;
            }
        }
        char* dst = M <= 12u ? small : sv_sink_alloc(sink, M);
        if (dst == nullptr) return false;
        const uint32_t w = utf8_case_map_python(p, L, upper, dst);
        assert(w == M);
        if (M <= 12u) { out[i] = sv_make_inline(dst, w); continue; }
        StringView r;
        memset(&r, 0, sizeof(r));
        r.length = w;
        memcpy(r.prefix, dst, 4);
        r.ref.buf_idx = 0;
        r.ref.offset = static_cast<uint32_t>(dst - sink->base);
        out[i] = r;
    }
    return true;
}

// a[i] || b[i].
inline bool utf8_concat_col(const StringView* BOLT_RESTRICT a, const char* abase,
                            const StringView* BOLT_RESTRICT b, const char* bbase, int64_t n,
                            SvSink* sink, StringView* BOLT_RESTRICT out) noexcept {
    assert(n >= 0 && sink != nullptr);
    for (int64_t i = 0; i < n; ++i) {
        const uint32_t la = a[i].length, lb = b[i].length;
        const uint32_t L = la + lb;
        char tmp[12];
        char* dst = L <= 12u ? tmp : sv_sink_alloc(sink, L);
        if (dst == nullptr) return false;
        if (la) memcpy(dst, sv_bytes(a[i], abase), la);
        if (lb) memcpy(dst + la, sv_bytes(b[i], bbase), lb);
        if (L <= 12u) { out[i] = sv_make_inline(dst, L); continue; }
        StringView r;
        memset(&r, 0, sizeof(r));
        r.length = L;
        memcpy(r.prefix, dst, 4);
        r.ref.buf_idx = 0;
        r.ref.offset = static_cast<uint32_t>(dst - sink->base);
        out[i] = r;
    }
    return true;
}

// Decimal text of each int64 (at most 20 bytes).
inline bool i64_to_utf8_col(const int64_t* BOLT_RESTRICT in, int64_t n, SvSink* sink,
                            StringView* BOLT_RESTRICT out) noexcept {
    assert(n >= 0 && sink != nullptr);
    for (int64_t i = 0; i < n; ++i) {
        char buf[24];
        uint64_t u = in[i] < 0 ? 0u - static_cast<uint64_t>(in[i]) : static_cast<uint64_t>(in[i]);
        uint32_t k = 24;
        do { buf[--k] = static_cast<char>('0' + u % 10u); u /= 10u; } while (u != 0u);   // <= 20 digits
        if (in[i] < 0) buf[--k] = '-';
        if (!sv_put(sink, buf + k, 24 - k, &out[i])) return false;
    }
    return true;
}

// Every non-overlapping occurrence of `from` (non-empty) replaced by `to`.
inline bool utf8_replace_col(const StringView* BOLT_RESTRICT in, const char* base, int64_t n,
                             const char* from, uint32_t flen, const char* to, uint32_t tlen,
                             SvSink* sink, StringView* BOLT_RESTRICT out) noexcept {
    assert(n >= 0 && flen > 0 && sink != nullptr);
    for (int64_t i = 0; i < n; ++i) {
        const char* p = sv_bytes(in[i], base);
        const uint32_t L = in[i].length;
        uint32_t hits = 0;
        for (uint32_t k = 0; k + flen <= L;) {   // count, bounded by L
            const void* hit = memchr(p + k, from[0], L - flen + 1 - k);
            if (hit == nullptr) break;
            k = static_cast<uint32_t>(static_cast<const char*>(hit) - p);
            if (memcmp(p + k, from, flen) == 0) { ++hits; k += flen; } else { ++k; }
        }
        if (hits == 0) {
            if (!sv_put(sink, p, L, &out[i])) return false;
            continue;
        }
        const uint64_t M64 = static_cast<uint64_t>(L) - static_cast<uint64_t>(hits) * flen +
                             static_cast<uint64_t>(hits) * tlen;
        if (M64 > 0xFFFFFFFFull) return false;
        const uint32_t M = static_cast<uint32_t>(M64);
        char tmp[12];
        char* dst = M <= 12u ? tmp : sv_sink_alloc(sink, M);
        if (dst == nullptr) return false;
        uint32_t w = 0;
        for (uint32_t k = 0; k < L;) {   // copy runs between matches
            const void* hit = k + flen <= L ? memchr(p + k, from[0], L - flen + 1 - k) : nullptr;
            uint32_t at = hit == nullptr ? L : static_cast<uint32_t>(static_cast<const char*>(hit) - p);
            if (at < L && memcmp(p + at, from, flen) != 0) at += 1;   // a false start: copy through it
            else if (at < L) {
                memcpy(dst + w, p + k, at - k);
                w += at - k;
                if (tlen) memcpy(dst + w, to, tlen);
                w += tlen;
                k = at + flen;
                continue;
            }
            memcpy(dst + w, p + k, at - k);
            w += at - k;
            k = at;
        }
        assert(w == M);
        if (M <= 12u) { out[i] = sv_make_inline(dst, M); continue; }
        StringView r;
        memset(&r, 0, sizeof(r));
        r.length = M;
        memcpy(r.prefix, dst, 4);
        r.ref.buf_idx = 0;
        r.ref.offset = static_cast<uint32_t>(dst - sink->base);
        out[i] = r;
    }
    return true;
}

// Python str.strip / lstrip / rstrip per row (chars == nullptr: whitespace).
// The result is a sub-range of the input, re-anchored in the sink.
inline bool py_strip_col(const StringView* BOLT_RESTRICT in, const char* base, int64_t n,
                         PyStrip mode, const char* chars, uint32_t clen, SvSink* sink,
                         StringView* BOLT_RESTRICT out) noexcept {
    assert(n >= 0 && sink != nullptr);
    for (int64_t i = 0; i < n; ++i) {
        const char* p = sv_bytes(in[i], base);
        uint32_t lo = 0, hi = in[i].length;
        py_strip(p, in[i].length, mode, chars, clen, &lo, &hi);
        if (!sv_put(sink, p + lo, hi - lo, &out[i])) return false;
    }
    return true;
}

#if defined(__SIZEOF_INT128__)
// Python round(x, n) per row, 1 <= n <= 22.
inline void f64_round_python_col(const double* BOLT_RESTRICT in, int64_t n, int32_t digits,
                                 double* BOLT_RESTRICT out) noexcept {
    assert(n >= 0 && digits >= 1 && digits <= 22);
    for (int64_t i = 0; i < n; ++i) out[i] = f64_round_python(in[i], digits);
}
#endif

// repr(float) per row.
inline bool f64_python_repr_col(const double* BOLT_RESTRICT in, int64_t n, SvSink* sink,
                                StringView* BOLT_RESTRICT out) noexcept {
    assert(n >= 0 && sink != nullptr);
    for (int64_t i = 0; i < n; ++i) {
        char buf[40];
        if (!sv_put(sink, buf, f64_python_repr(in[i], buf), &out[i])) return false;
    }
    return true;
}

}  // namespace bolt::kernels::utf8
