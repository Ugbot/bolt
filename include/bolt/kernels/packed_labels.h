// bolt/kernels/packed_labels.h: read and cut a packed key/value set held in one Utf8 column.
//
// The set is `name 0x1f value 0x1e` pairs (a series' labels, a row's sparse attributes). One leading
// 0x1e is tolerated (a set a pair was cut from). Names are compared bytewise; the pairs need not be
// sorted. Two column kernels:
//
//   packed_label_get   the value of one name per row, "" when the row has no such pair. A value of
//                      more than 12 bytes is a view of the INPUT's bytes (same spilled base): no copy.
//   packed_labels_drop the set without the listed names, in the stored form (no leading separator,
//                      "" when nothing is left), written to a caller buffer.
//
// NULL handling is the caller's (a NULL set reads as the empty set).
#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_port.h"
#include "bolt/bolt_types.h"
#include "bolt/kernels/bolt_utf8.h"

namespace bolt::kernels::packed_labels {

inline constexpr char k_pair_sep = '\x1e';
inline constexpr char k_kv_sep = '\x1f';

// One pair of the set starting at p[i]: name [i, *sep), value (*sep, *end). False at the end of the set
// or on a tail without a key/value separator (ignored, as the result edge does).
BOLT_FORCE_INLINE bool next_pair(const char* BOLT_RESTRICT p, uint32_t n, uint32_t i, uint32_t* sep,
                                 uint32_t* end) noexcept {
    assert(p != nullptr || n == 0);
    assert(i <= n);
    if (i >= n) return false;
    const void* e = memchr(p + i, k_pair_sep, n - i);
    const uint32_t pe = e != nullptr ? static_cast<uint32_t>(static_cast<const char*>(e) - p) : n;
    const void* s = memchr(p + i, k_kv_sep, pe - i);
    if (s == nullptr) { *sep = pe; *end = pe; return true; }   // a pair without a value separator: no name
    *sep = static_cast<uint32_t>(static_cast<const char*>(s) - p);
    *end = pe;
    return true;
}

// The value of `name` in the set p[0, n): [*voff, *voff + *vlen). False when the set has no such pair.
inline bool packed_label_find(const char* BOLT_RESTRICT p, uint32_t n, const char* BOLT_RESTRICT name,
                              uint32_t nlen, uint32_t* voff, uint32_t* vlen) noexcept {
    assert(p != nullptr || n == 0);
    assert(name != nullptr && voff != nullptr && vlen != nullptr);
    uint32_t i = (n != 0 && p[0] == k_pair_sep) ? 1u : 0u;
    uint32_t sep = 0, end = 0;
    while (next_pair(p, n, i, &sep, &end)) {   // bounded: one step per pair, i advances
        if (sep < end && sep - i == nlen && memcmp(p + i, name, nlen) == 0) {
            *voff = sep + 1u;
            *vlen = end - (sep + 1u);
            return true;
        }
        i = end < n ? end + 1u : end;
    }
    return false;
}

// out[i] = the value of `name` in in[i], "" when absent.
inline void packed_label_get(const StringView* BOLT_RESTRICT in, int64_t n, const char* spilled_base,
                             const char* BOLT_RESTRICT name, uint32_t nlen,
                             StringView* BOLT_RESTRICT out) noexcept {
    assert(in != nullptr || n == 0);
    assert(out != nullptr || n == 0);
    assert(n >= 0 && name != nullptr);
    for (int64_t r = 0; r < n; ++r) {
        const StringView& s = in[r];
        const char* p = utf8::sv_bytes(s, spilled_base);
        uint32_t voff = 0, vlen = 0;
        if (!packed_label_find(p, s.length, name, nlen, &voff, &vlen)) { memset(&out[r], 0, sizeof(StringView)); continue; }
        if (vlen <= 12u) { out[r] = utf8::sv_make_inline(p + voff, vlen); continue; }
        StringView v;
        memset(&v, 0, sizeof(v));
        v.length = vlen;
        memcpy(v.prefix, p + voff, 4);
        v.ref.buf_idx = s.ref.buf_idx;
        v.ref.offset = s.ref.offset + voff;   // a long value sits in a spilled set: same buffer
        out[r] = v;
    }
}

// True when name [p, p + len) is one of `names` (0x1f-joined).
BOLT_FORCE_INLINE bool name_listed(const char* BOLT_RESTRICT p, uint32_t len, const char* BOLT_RESTRICT names,
                                   uint32_t names_len) noexcept {
    assert(names != nullptr || names_len == 0);
    uint32_t i = 0;
    while (i <= names_len) {   // bounded: one step per listed name
        const void* e = i < names_len ? memchr(names + i, k_kv_sep, names_len - i) : nullptr;
        const uint32_t ne = e != nullptr ? static_cast<uint32_t>(static_cast<const char*>(e) - names) : names_len;
        if (ne - i == len && memcmp(names + i, p, len) == 0) return true;
        i = ne + 1u;
    }
    return false;
}

// The set p[0, n) without the pairs named in `names` (0x1f-joined), written to out; returns its length
// (<= n + 1: a last pair without its separator gains one). `out` holds at least n + 1 bytes. Pairs keep
// their order; each ends with 0x1e.
inline uint32_t packed_labels_drop(const char* BOLT_RESTRICT p, uint32_t n, const char* BOLT_RESTRICT names,
                                   uint32_t names_len, char* BOLT_RESTRICT out) noexcept {
    assert(p != nullptr || n == 0);
    assert(out != nullptr || n == 0);
    uint32_t i = (n != 0 && p[0] == k_pair_sep) ? 1u : 0u;
    uint32_t o = 0, sep = 0, end = 0;
    while (next_pair(p, n, i, &sep, &end)) {   // bounded: one step per pair
        if (sep < end && !name_listed(p + i, sep - i, names, names_len)) {
            memcpy(out + o, p + i, end - i);
            o += end - i;
            out[o++] = k_pair_sep;
        }
        i = end < n ? end + 1u : end;
    }
    assert(o <= n + 1u);
    return o;
}

// Bytes the column form of packed_labels_drop needs at most for rows [0, n): each row's length + 1.
inline int64_t packed_labels_drop_bound(const StringView* BOLT_RESTRICT in, int64_t n) noexcept {
    assert(in != nullptr || n == 0);
    assert(n >= 0);
    int64_t total = 0;
    for (int64_t r = 0; r < n; ++r) total += static_cast<int64_t>(in[r].length) + 1;
    return total;
}

// out[i] = in[i] without the pairs named in `names`. Results past 12 bytes are written to out_bytes
// (cap bytes; ref.offset is relative to out_bytes, buf_idx 0). Returns the bytes used, or -1 when cap is
// too small (out is then unspecified). cap >= packed_labels_drop_bound(in, n) always fits.
inline int64_t packed_labels_drop(const StringView* BOLT_RESTRICT in, int64_t n, const char* spilled_base,
                                  const char* BOLT_RESTRICT names, uint32_t names_len,
                                  StringView* BOLT_RESTRICT out, char* BOLT_RESTRICT out_bytes,
                                  int64_t cap) noexcept {
    assert(in != nullptr || n == 0);
    assert(out != nullptr || n == 0);
    assert(n >= 0 && cap >= 0 && (out_bytes != nullptr || cap == 0));
    int64_t used = 0;
    for (int64_t r = 0; r < n; ++r) {
        const StringView& s = in[r];
        const char* p = utf8::sv_bytes(s, spilled_base);
        if (used + static_cast<int64_t>(s.length) + 1 > cap) return -1;
        char* dst = out_bytes + used;
        const uint32_t len = packed_labels_drop(p, s.length, names, names_len, dst);
        if (len <= 12u) { out[r] = utf8::sv_make_inline(dst, len); continue; }
        StringView v;
        memset(&v, 0, sizeof(v));
        v.length = len;
        memcpy(v.prefix, dst, 4);
        v.ref.offset = static_cast<uint32_t>(used);
        out[r] = v;
        used += len;
    }
    assert(used <= cap);
    return used;
}

}  // namespace bolt::kernels::packed_labels
