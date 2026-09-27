// bolt_string.h — UTF-8 / string kernels operating on StringView arrays.
//
// RULES: No exceptions. No RTTI. No heap outside Arena. All fns noexcept.
// BOLT_RESTRICT on array params; >=2 asserts per fn; <=70 LOC per fn.
//
// Fast path: StringView carries a 4-byte prefix + length inline. For equality
// and prefix tests, the prefix resolves the vast majority of comparisons
// without a pointer chase into the spilled data buffer.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_port.h"
#include "bolt/bolt_types.h"
#include "bolt/kernels/bolt_utf8.h"   // sv_bytes / sv_make_inline (substr)

namespace bolt {
namespace kernels {

// ---------------------------------------------------------------------------
// Internal: resolve raw pointer to a StringView's bytes.
//
// Inline views: bytes live in the 12-byte inline buffer (prefix + inline_data).
// Spilled views: caller must provide the backing buffer base (out of scope for
// a pure StringView API). For the equality/prefix kernels below we compare
// inline-against-inline cheaply; for spilled-vs-spilled we fall through to a
// full-length prefix memcmp spanning the 12-byte window that both views carry
// in their own headers. A full spilled comparison requires the data buffer,
// which is carried by BoltColumn — those kernels go elsewhere.
// ---------------------------------------------------------------------------

BOLT_FORCE_INLINE bool sv_eq_headers(const StringView& a, const StringView& b) noexcept {
    // Only valid when BOTH inline (length <= 12). Checks length + 12 bytes.
    assert(a.is_inline());
    assert(b.is_inline());
    if (a.length != b.length) return false;
    // prefix[4] + inline_data[8] are contiguous in the union layout.
    return memcmp(a.prefix, b.prefix, a.length) == 0;
}

// ---------------------------------------------------------------------------
// utf8_equals: selection-vector filter where data[i] == needle.
//
// Fast path: if needle is inline, every candidate can be resolved from the
// StringView header alone (length + 12-byte inline buffer). If needle is
// spilled, we first filter by length + 4-byte prefix, then the caller is
// expected to follow up with a full memcmp pass once the spilled buffer is
// available. For the inline-vs-spilled mismatch case, lengths differ in the
// common case; when lengths match we conservatively accept on prefix match
// (rare, same-length inline vs spilled is impossible: inline <= 12, spilled
// > 12). So prefix match on equal length is a true-equal *unless* both are
// spilled; there, caller resolves.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE int64_t utf8_equals(
        const StringView* BOLT_RESTRICT data, int64_t n,
        StringView needle,
        int32_t* BOLT_RESTRICT out) noexcept {
    assert(data != nullptr || n == 0);
    assert(out  != nullptr || n == 0);
    assert(n >= 0);

    int64_t count = 0;
    const bool needle_inline = needle.is_inline();
    for (int64_t i = 0; i < n; ++i) {
        const StringView& s = data[i];
        bool match = false;
        if (s.length == needle.length) {
            if (needle_inline && s.is_inline()) {
                match = (memcmp(s.prefix, needle.prefix,
                                (s.length < 12u) ? s.length : 12u) == 0);
            } else {
                // Prefix-only test; if both spilled, caller must verify tail.
                match = (memcmp(s.prefix, needle.prefix,
                                (s.length < 4u) ? s.length : 4u) == 0);
            }
        }
        out[count] = static_cast<int32_t>(i);
        count += match ? 1 : 0;
    }
    return count;
}

// ---------------------------------------------------------------------------
// utf8_starts_with: selection-vector filter where data[i] begins with prefix.
// Uses the 4-byte inline prefix of the StringView (no data-buffer chase) for
// prefixes <= 4 bytes; longer prefixes only work when the candidate itself is
// inline (whole candidate accessible from the header).
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE int64_t utf8_starts_with(
        const StringView* BOLT_RESTRICT data, int64_t n,
        StringView prefix,
        int32_t* BOLT_RESTRICT out) noexcept {
    assert(data != nullptr || n == 0);
    assert(out  != nullptr || n == 0);
    assert(n >= 0);

    int64_t count = 0;
    const uint32_t plen = prefix.length;
    for (int64_t i = 0; i < n; ++i) {
        const StringView& s = data[i];
        bool match = false;
        if (s.length >= plen) {
            if (plen <= 4u) {
                match = (memcmp(s.prefix, prefix.prefix, plen) == 0);
            } else if (s.is_inline() && prefix.is_inline()) {
                // Both fully in the header's 12-byte window.
                match = (memcmp(s.prefix, prefix.prefix, plen) == 0);
            } else {
                // Head matches on 4 bytes; caller must verify tail via buffer.
                match = (memcmp(s.prefix, prefix.prefix, 4) == 0);
            }
        }
        out[count] = static_cast<int32_t>(i);
        count += match ? 1 : 0;
    }
    return count;
}

// ---------------------------------------------------------------------------
// utf8_contains: selection-vector filter where data[i] contains needle.
//
// Inline-only: operates on inline StringViews (header-resident bytes). For
// spilled views, callers should use a BoltColumn-aware kernel. Straight
// memmem-style loop; no SIMD here on purpose.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE int64_t utf8_contains(
        const StringView* BOLT_RESTRICT data, int64_t n,
        StringView needle,
        int32_t* BOLT_RESTRICT out) noexcept {
    assert(data != nullptr || n == 0);
    assert(out  != nullptr || n == 0);
    assert(n >= 0);
    assert(needle.is_inline());  // Only inline needles supported here.

    int64_t count = 0;
    const uint32_t nlen = needle.length;
    for (int64_t i = 0; i < n; ++i) {
        const StringView& s = data[i];
        bool match = false;
        if (nlen == 0) {
            match = true;
        } else if (s.is_inline() && s.length >= nlen) {
            const char* hay = s.prefix;
            const uint32_t slen = s.length;
            const uint32_t last = slen - nlen;
            for (uint32_t k = 0; k <= last; ++k) {
                if (memcmp(hay + k, needle.prefix, nlen) == 0) {
                    match = true;
                    break;
                }
            }
        }
        out[count] = static_cast<int32_t>(i);
        count += match ? 1 : 0;
    }
    return count;
}

// ---------------------------------------------------------------------------
// utf8_substr_const: vectorized SUBSTRING(col, start1, len) with COMPILE-TIME
// CONSTANT 1-based `start1` and CHARACTER count `take` (utf8_cp_advance),
// producing one inline output StringView per row. The hot case in OLAP —
// extracting a fixed prefix/field (TPC-H Q22's substring(c_phone,1,2)
// country code) — is a tight, branch-light column loop.
//
// A spilled source whose window lies in its 4 inline prefix bytes, all
// ASCII, is answered without chasing the spilled buffer; any other window
// resolves the bytes through `spilled_base`. Returns false (the caller falls
// back to its per-row path, `out` is then unspecified) when a row's window
// is wider than 12 bytes, which only non-ASCII text can produce for
// take <= 12.
//
// Semantics match utf8_substring: start1 < 1 clamps to 1; a start past the
// string yields empty; `take` saturates at the remaining characters. NULL
// handling is the caller's.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE bool utf8_substr_const(
        const StringView* BOLT_RESTRICT data, int64_t n,
        const char* spilled_base, uint32_t start1, uint32_t take,
        StringView* BOLT_RESTRICT out) noexcept {
    assert(data != nullptr || n == 0);
    assert(out  != nullptr || n == 0);
    assert(n >= 0);
    assert(take <= 12u && "utf8_substr_const targets inline (<=12B) results");
    const uint32_t s0 = (start1 < 1u) ? 0u : (start1 - 1u);
    for (int64_t i = 0; i < n; ++i) {
        const StringView& s = data[i];
        const uint32_t slen = s.length;
        const char* p = s.prefix;
        uint32_t b, e;
        if (slen > 12u && !(s0 + take <= 4u && utf8::utf8_is_ascii(p, s0 + take))) {
            assert(spilled_base != nullptr);
            p = utf8::sv_bytes(s, spilled_base);
        }
        if (slen > 12u && p == s.prefix) {
            b = s0;
            e = s0 + take;
        } else {
            b = utf8::utf8_cp_advance(p, slen, 0, s0);
            e = utf8::utf8_cp_advance(p, slen, b, take);
        }
        if (e - b > 12u) return false;
        out[i] = utf8::sv_make_inline(p + b, e - b);
    }
    return true;
}

// ---------------------------------------------------------------------------
// utf8_length_bytes: extract StringView.length into a parallel int32 array.
// Trivially vectorizable (gather 4-byte field at stride 16).
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE void utf8_length_bytes(
        const StringView* BOLT_RESTRICT data, int64_t n,
        int32_t* BOLT_RESTRICT out_lens) noexcept {
    assert(data != nullptr || n == 0);
    assert(out_lens != nullptr || n == 0);
    assert(n >= 0);

    for (int64_t i = 0; i < n; ++i) {
        out_lens[i] = static_cast<int32_t>(data[i].length);
    }
}

// ===========================================================================
// VarBinary kernels — operate directly on the offsets+data shape of
// `BoltColumn::Format::VarBinary`. They do NOT require a parallel
// StringView header array; the column itself is the canonical layout.
// Inputs:
//   data    — flat payload bytes (concatenated rows)
//   offsets — Int32 array of length n+1; row i spans [offsets[i],offsets[i+1])
//   n       — row count
// Outputs:
//   out     — selection vector (caller-sized for at least n int32_t)
// All kernels: noexcept, branchless write into `out`, ≥2 asserts.
// ===========================================================================

BOLT_FORCE_INLINE int64_t varbinary_equals(
        const uint8_t* BOLT_RESTRICT data,
        const int32_t* BOLT_RESTRICT offsets, int64_t n,
        const uint8_t* BOLT_RESTRICT needle, int32_t nlen,
        int32_t* BOLT_RESTRICT out) noexcept {
    assert(offsets != nullptr || n == 0);
    assert(out     != nullptr || n == 0);
    assert(nlen >= 0);
    assert(n >= 0);
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        const int32_t start = offsets[i];
        const int32_t len   = offsets[i + 1] - start;
        bool match = (len == nlen);
        if (match && nlen > 0) {
            match = (memcmp(data + start, needle, static_cast<size_t>(nlen)) == 0);
        }
        out[count] = static_cast<int32_t>(i);
        count += match ? 1 : 0;
    }
    return count;
}

BOLT_FORCE_INLINE int64_t varbinary_starts_with(
        const uint8_t* BOLT_RESTRICT data,
        const int32_t* BOLT_RESTRICT offsets, int64_t n,
        const uint8_t* BOLT_RESTRICT prefix, int32_t plen,
        int32_t* BOLT_RESTRICT out) noexcept {
    assert(offsets != nullptr || n == 0);
    assert(out     != nullptr || n == 0);
    assert(plen >= 0);
    assert(n >= 0);
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        const int32_t start = offsets[i];
        const int32_t len   = offsets[i + 1] - start;
        bool match = (len >= plen);
        if (match && plen > 0) {
            match = (memcmp(data + start, prefix, static_cast<size_t>(plen)) == 0);
        }
        out[count] = static_cast<int32_t>(i);
        count += match ? 1 : 0;
    }
    return count;
}

BOLT_FORCE_INLINE int64_t varbinary_contains(
        const uint8_t* BOLT_RESTRICT data,
        const int32_t* BOLT_RESTRICT offsets, int64_t n,
        const uint8_t* BOLT_RESTRICT needle, int32_t nlen,
        int32_t* BOLT_RESTRICT out) noexcept {
    assert(offsets != nullptr || n == 0);
    assert(out     != nullptr || n == 0);
    assert(nlen >= 0);
    assert(n >= 0);
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        const int32_t start = offsets[i];
        const int32_t len   = offsets[i + 1] - start;
        bool match = false;
        if (nlen == 0) {
            match = true;
        } else if (len >= nlen) {
            const uint8_t* hay = data + start;
            const int32_t  last = len - nlen;
            for (int32_t k = 0; k <= last; ++k) {
                if (memcmp(hay + k, needle, static_cast<size_t>(nlen)) == 0) {
                    match = true;
                    break;
                }
            }
        }
        out[count] = static_cast<int32_t>(i);
        count += match ? 1 : 0;
    }
    return count;
}

BOLT_FORCE_INLINE int64_t varbinary_length_in_range(
        const int32_t* BOLT_RESTRICT offsets, int64_t n,
        int32_t lo_incl, int32_t hi_incl,
        int32_t* BOLT_RESTRICT out) noexcept {
    assert(offsets != nullptr || n == 0);
    assert(out     != nullptr || n == 0);
    assert(n >= 0);
    assert(lo_incl <= hi_incl);
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        const int32_t len = offsets[i + 1] - offsets[i];
        const bool match = (len >= lo_incl) & (len <= hi_incl);
        out[count] = static_cast<int32_t>(i);
        count += match ? 1 : 0;
    }
    return count;
}

BOLT_FORCE_INLINE void varbinary_lengths(
        const int32_t* BOLT_RESTRICT offsets, int64_t n,
        int32_t* BOLT_RESTRICT out_lens) noexcept {
    assert(offsets  != nullptr || n == 0);
    assert(out_lens != nullptr || n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) {
        out_lens[i] = offsets[i + 1] - offsets[i];
    }
}

} // namespace kernels
} // namespace bolt
