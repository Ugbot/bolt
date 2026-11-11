// bolt_listagg.h — ordered grouped string / binary concatenation (Spark
// listagg / string_agg, SQL LISTAGG ... WITHIN GROUP (ORDER BY k)).
//
// A group's state is a list of items, each `[u32 klen][key bytes][value bytes]`:
// the key is the order key's canonical order-preserving bytes (bolt_key_encode.h)
// behind a NULL-placement marker, complemented for DESC, so one memcmp orders
// items as the WITHIN GROUP clause does. An item list is mergeable as-is: the
// partial of a split aggregate is its items, the final sorts the union.
//
// finish: a stable sort of the items by key (and value, for DISTINCT, which
// then drops adjacent equal values), concatenated with the delimiter into a
// caller buffer bounded by a byte budget (kNoRoom past it, never truncated).
//
// Tiger Style: POD + free functions, noexcept, no allocation (the caller
// passes every buffer); every loop bounded by an item count or length.

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_column.h"
#include "bolt/kernels/bolt_key_encode.h"

namespace bolt::listagg {

// Order flags (one byte).
inline constexpr uint8_t kHasOrder   = 1u;   // a WITHIN GROUP key
inline constexpr uint8_t kDesc       = 2u;
inline constexpr uint8_t kNullsFirst = 4u;
inline constexpr uint8_t kDistinct   = 8u;
inline constexpr uint8_t kBinary     = 16u;  // the values are binary (result type)

inline constexpr uint8_t kNullFirstMark = 0x00;
inline constexpr uint8_t kValueMark     = 0x01;
inline constexpr uint8_t kNullLastMark  = 0x02;

enum class Status : uint8_t { kOk = 0, kNoRoom, kUnsupported };

struct Item {
    const uint8_t* bytes;   // [u32 klen][key][value]
    uint32_t       len;
};

BOLT_FORCE_INLINE uint32_t item_key_len(const Item& it) noexcept {
    uint32_t k = 0;
    std::memcpy(&k, it.bytes, 4);
    return k;
}
BOLT_FORCE_INLINE const uint8_t* item_key(const Item& it) noexcept { return it.bytes + 4; }
BOLT_FORCE_INLINE const uint8_t* item_value(const Item& it) noexcept { return it.bytes + 4 + item_key_len(it); }
BOLT_FORCE_INLINE uint32_t item_value_len(const Item& it) noexcept { return it.len - 4u - item_key_len(it); }

// Order key of row r of `c` under `flags` into dst[0..cap); *len = bytes.
inline Status encode_key(const BoltColumn& c, int64_t r, uint8_t flags, uint8_t* dst, uint32_t cap,
                         uint32_t* len) noexcept {
    assert(dst != nullptr && len != nullptr);
    assert(r >= 0 && r < c.length);
    *len = 0;
    if (cap == 0) return Status::kNoRoom;
    if (c.is_null(r)) {
        dst[0] = (flags & kNullsFirst) != 0 ? kNullFirstMark : kNullLastMark;
        *len = 1;
        return Status::kOk;
    }
    dst[0] = kValueMark;
    uint32_t n = 0;
    const KeyEncodeStatus s = key_encode_cell(c, r, dst + 1, cap - 1u, &n);
    if (s == KeyEncodeStatus::kNoRoom) return Status::kNoRoom;
    if (s != KeyEncodeStatus::kOk) return Status::kUnsupported;
    if ((flags & kDesc) != 0) {
        for (uint32_t i = 0; i < n; ++i) dst[1 + i] = static_cast<uint8_t>(~dst[1 + i]);   // bounded: n
    }
    *len = n + 1u;
    return Status::kOk;
}

// Bytes of an item with a `klen`-byte key and a `vlen`-byte value.
BOLT_FORCE_INLINE uint64_t item_bytes(uint32_t klen, uint32_t vlen) noexcept {
    return 4ull + klen + vlen;
}

// Assemble an item into dst (item_bytes(klen, vlen) long).
inline void put_item(uint8_t* dst, const uint8_t* key, uint32_t klen, const uint8_t* val, uint32_t vlen) noexcept {
    assert(dst != nullptr);
    assert((key != nullptr || klen == 0) && (val != nullptr || vlen == 0));
    std::memcpy(dst, &klen, 4);
    if (klen != 0) std::memcpy(dst + 4, key, klen);
    if (vlen != 0) std::memcpy(dst + 4 + klen, val, vlen);
}

BOLT_FORCE_INLINE int cmp_bytes(const uint8_t* a, uint32_t an, const uint8_t* b, uint32_t bn) noexcept {
    const uint32_t m = an < bn ? an : bn;
    const int c = m != 0 ? std::memcmp(a, b, m) : 0;
    if (c != 0) return c;
    return an < bn ? -1 : an > bn ? 1 : 0;
}

// Item a orders before-or-equal item b: by key, then (distinct) by value.
BOLT_FORCE_INLINE bool item_le(const Item& a, const Item& b, bool by_value) noexcept {
    const int k = cmp_bytes(item_key(a), item_key_len(a), item_key(b), item_key_len(b));
    if (k != 0 || !by_value) return k <= 0;
    return cmp_bytes(item_value(a), item_value_len(a), item_value(b), item_value_len(b)) <= 0;
}

// Stable bottom-up merge sort of perm[0..n) (indexes into items); scratch holds n.
inline void sort_items(const Item* items, uint32_t* perm, uint32_t* scratch, uint32_t n, bool by_value) noexcept {
    assert(perm != nullptr && scratch != nullptr);
    assert(items != nullptr || n == 0);
    for (uint32_t i = 0; i < n; ++i) perm[i] = i;
    uint32_t* src = perm;
    uint32_t* dst = scratch;
    for (uint32_t w = 1; w < n; w = w > 0x7FFFFFFFu ? n : w * 2u) {   // bounded: log2(n) passes
        for (uint32_t lo = 0; lo < n; lo += 2u * w) {
            const uint32_t mid = lo + w < n ? lo + w : n;
            const uint32_t hi = mid + w < n ? mid + w : n;
            uint32_t i = lo, j = mid, o = lo;
            while (i < mid && j < hi) dst[o++] = item_le(items[src[i]], items[src[j]], by_value) ? src[i++] : src[j++];
            while (i < mid) dst[o++] = src[i++];
            while (j < hi) dst[o++] = src[j++];
        }
        uint32_t* t = src;
        src = dst;
        dst = t;
    }
    if (src != perm) std::memcpy(perm, src, sizeof(uint32_t) * n);
}

// The ordered values joined by `delim` into out[0..cap); *len = bytes. With
// kDistinct an item equal in value to the previous kept one is dropped.
inline Status concat(const Item* items, const uint32_t* perm, uint32_t n, uint8_t flags, const uint8_t* delim,
                     uint32_t dlen, uint8_t* out, uint64_t cap, uint64_t* len) noexcept {
    assert(len != nullptr && (perm != nullptr || n == 0));
    assert(out != nullptr || cap == 0);
    uint64_t o = 0;
    const Item* prev = nullptr;
    for (uint32_t i = 0; i < n; ++i) {
        const Item& it = items[perm[i]];
        const uint32_t vl = item_value_len(it);
        if ((flags & kDistinct) != 0 && prev != nullptr &&
            cmp_bytes(item_value(*prev), item_value_len(*prev), item_value(it), vl) == 0) continue;
        const uint64_t need = (prev != nullptr ? dlen : 0u) + static_cast<uint64_t>(vl);
        if (o + need > cap) return Status::kNoRoom;
        if (prev != nullptr && dlen != 0) { std::memcpy(out + o, delim, dlen); o += dlen; }
        if (vl != 0) { std::memcpy(out + o, item_value(it), vl); o += vl; }
        prev = &it;
    }
    *len = o;
    return Status::kOk;
}

}  // namespace bolt::listagg
