// bolt_kv_packed.h — lookups in a packed key-value set, a whole column at a time.
//
// A packed set is one byte string of pairs, each `key KV value PAIR`: every
// pair ends with the pair separator, so a non-empty set ends with one. A key
// holds neither separator, a value never the pair separator. (A PromQL label
// set: KV = 0x1f, PAIR = 0x1e.)
//
// kv_packed_get: the value of one key per row, as a view of the row's own
// bytes (no copy: a value over 12 bytes keeps the row's buffer and offset).
// Absent, or a row that is not whole pairs, gives the empty view. When a key
// repeats the last pair wins.
#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_port.h"
#include "bolt/bolt_types.h"
#include "bolt/kernels/bolt_kv_packed_limits.h"
#include "bolt/kernels/bolt_utf8.h"

namespace bolt {
namespace kernels {
namespace kv_packed {

// Byte range of `key`'s value inside one set, or false. `ndl` = PAIR key KV.
BOLT_FORCE_INLINE bool find_value(const char* BOLT_RESTRICT p, uint32_t len,
                                  const char* BOLT_RESTRICT ndl, uint32_t nlen,
                                  char pair_sep, uint32_t* vs, uint32_t* vl) noexcept {
    assert(nlen >= 3u && vs != nullptr && vl != nullptr);
    if (len < nlen || p[len - 1u] != pair_sep) return false;
    uint32_t at = UINT32_MAX;
    if (memcmp(p, ndl + 1, nlen - 1u) == 0) at = nlen - 1u;           // the first pair has no separator before it
    for (uint32_t pos = 0; pos + nlen <= len;) {                       // bounded: len
        const int32_t i = utf8::bytes_find(p + pos, len - pos, ndl, nlen);
        if (i < 0) break;
        at = pos + static_cast<uint32_t>(i) + nlen;
        pos = at;
    }
    if (at == UINT32_MAX) return false;
    const void* e = memchr(p + at, static_cast<unsigned char>(pair_sep), len - at);
    assert(e != nullptr);                                              // the set ends with a separator
    *vs = at;
    *vl = static_cast<uint32_t>(static_cast<const char*>(e) - (p + at));
    return true;
}

// out[i] = value of `key` in in[i], viewing in[i]'s bytes (resolve against the
// same `base`). Returns the rows that hold the key; -1 for a key that is
// empty, too long or holds a separator (nothing written).
inline int64_t kv_packed_get(const StringView* BOLT_RESTRICT in, const char* base, int64_t n,
                             const char* key, uint32_t key_len, char kv_sep, char pair_sep,
                             StringView* BOLT_RESTRICT out) noexcept {
    assert(in != nullptr && out != nullptr && key != nullptr && n >= 0);
    assert(kv_sep != pair_sep);
    if (key_len == 0u || key_len > kKeyMaxBytes) return -1;
    if (memchr(key, kv_sep, key_len) != nullptr || memchr(key, pair_sep, key_len) != nullptr) return -1;
    char ndl[kKeyMaxBytes + 2u];
    ndl[0] = pair_sep;
    memcpy(ndl + 1, key, key_len);
    ndl[key_len + 1u] = kv_sep;
    const uint32_t nlen = key_len + 2u;
    int64_t found = 0;
    for (int64_t i = 0; i < n; ++i) {
        const StringView& s = in[i];
        const char* p = utf8::sv_bytes(s, base);
        uint32_t vs = 0, vl = 0;
        if (!find_value(p, s.length, ndl, nlen, pair_sep, &vs, &vl)) {
            memset(&out[i], 0, sizeof(StringView));
            continue;
        }
        ++found;
        if (vl <= 12u) { out[i] = utf8::sv_make_inline(p + vs, vl); continue; }
        StringView r;
        r.length = vl;
        memcpy(r.prefix, p + vs, 4);
        r.ref.buf_idx = s.ref.buf_idx;
        r.ref.offset = s.ref.offset + vs;
        out[i] = r;
    }
    return found;
}

}  // namespace kv_packed
}  // namespace kernels
}  // namespace bolt
