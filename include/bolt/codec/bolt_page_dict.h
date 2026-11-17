// bolt/codec/bolt_page_dict.h — dictionary pages for Flat StringView columns: a sorted dictionary of the
// page's distinct strings plus one code per row (ColumnFormat::Dictionary on disk).
//
//   dictionary buffer: u32 offs[count + 1] (offs[0] = 0, non-decreasing), then offs[count] bytes;
//                      entry k is bytes [offs[k], offs[k + 1]), entries ascend in byte order, no duplicates.
//   codes buffer:      rows codes of code_bytes (1 for count <= 256, 2 for <= 65,536, else 4).
//
// The distinct count is exact: rows are hashed (hash_bytes_sv_wyhash3_v1), grouped by hash and each group is
// byte-compared; two different strings with one hash make the plan not encodable rather than wrong. The
// plan stops as soon as the count passes max_count. Size is closed form from the count and byte totals.
//
// Decode rebuilds a plain page: 16 B views (strings <= 12 B inline) and an overflow buffer holding each long
// dictionary entry once, in dictionary order. Input is untrusted; no allocation (caller scratch).

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_port.h"
#include "bolt/bolt_types.h"
#include "bolt/kernels/bolt_hash_sv.h"

namespace bolt {
namespace pagecodec {

struct DictPlan {
    uint32_t count;        // distinct strings
    uint32_t code_bytes;   // 1, 2 or 4
    uint64_t dict_bytes;   // offsets + string bytes
    uint64_t long_bytes;   // bytes of entries longer than 12 (the decoded overflow)
    uint64_t bytes;        // codes + dictionary; UINT64_MAX when not encodable
};

/// Scratch for dict_plan / dict_encode over n rows, 8-byte aligned.
constexpr uint64_t dict_scratch_bytes(uint32_t n) noexcept { return uint64_t{n} * (8u + 4u * 4u); }

struct DictScratch {
    uint64_t* hash;    // [n]
    uint32_t* order;   // [n] rows by (hash, row); then groups by string
    uint32_t* group;   // [n] row -> group, then row -> code
    uint32_t* rep;     // [n] group -> a row holding its string
    uint32_t* rank;    // [n] group -> code
};

inline DictScratch dict_scratch_at(void* mem, uint32_t n) noexcept {
    assert(mem != nullptr && reinterpret_cast<uintptr_t>(mem) % 8 == 0);
    DictScratch s;
    s.hash = static_cast<uint64_t*>(mem);
    s.order = reinterpret_cast<uint32_t*>(s.hash + n);
    s.group = s.order + n;
    s.rep = s.group + n;
    s.rank = s.rep + n;
    return s;
}

namespace detail {

BOLT_FORCE_INLINE const uint8_t* dict_sv_bytes(const StringView& v, const void* overflow) noexcept {
    return v.length <= 12u ? reinterpret_cast<const uint8_t*>(v.prefix)
                           : static_cast<const uint8_t*>(overflow) + v.ref.offset;
}

BOLT_FORCE_INLINE int dict_sv_cmp(const StringView& a, const StringView& b, const void* overflow) noexcept {
    const uint32_t m = a.length < b.length ? a.length : b.length;
    const int c = m ? std::memcmp(dict_sv_bytes(a, overflow), dict_sv_bytes(b, overflow), m) : 0;
    return c != 0 ? c : (a.length < b.length ? -1 : (a.length > b.length ? 1 : 0));
}

}  // namespace detail

/// After an encodable plan: sc.group[row] is the row's code and sc.order[0, count) lists one row per
/// dictionary entry in dictionary order.
inline DictPlan dict_plan(const StringView* BOLT_RESTRICT rows, uint32_t n, const void* overflow,
                          uint32_t max_count, const DictScratch& sc) noexcept {
    assert(rows != nullptr && n > 0 && max_count > 0);
    DictPlan p{0, 0, 0, 0, UINT64_MAX};
    hash_bytes_sv_wyhash3_v1(rows, n, overflow, sc.hash);
    for (uint32_t i = 0; i < n; ++i) sc.order[i] = i;               // bounded: n
    const uint64_t* h = sc.hash;
    std::sort(sc.order, sc.order + n, [h](uint32_t a, uint32_t b) noexcept {
        return h[a] != h[b] ? h[a] < h[b] : a < b;
    });
    uint32_t count = 0;
    for (uint32_t i = 0; i < n; ++i) {                              // bounded: n
        const uint32_t r = sc.order[i];
        const bool fresh = i == 0 || h[r] != h[sc.order[i - 1]];
        if (fresh) {
            if (count == max_count) return p;
            sc.rep[count++] = r;
        } else if (detail::dict_sv_cmp(rows[r], rows[sc.rep[count - 1]], overflow) != 0) {
            return p;                                               // two strings, one hash
        }
        sc.group[r] = count - 1;
    }
    // groups in byte order of their string; order[] is free now
    for (uint32_t g = 0; g < count; ++g) sc.order[g] = g;           // bounded: count
    const uint32_t* rep = sc.rep;
    std::sort(sc.order, sc.order + count, [rows, rep, overflow](uint32_t a, uint32_t b) noexcept {
        return detail::dict_sv_cmp(rows[rep[a]], rows[rep[b]], overflow) < 0;
    });
    uint64_t bytes = 0, long_bytes = 0;
    for (uint32_t k = 0; k < count; ++k) {                          // bounded: count
        const uint32_t g = sc.order[k];
        sc.rank[g] = k;
        const uint32_t len = rows[rep[g]].length;
        bytes += len;
        long_bytes += len > 12u ? len : 0u;
        sc.order[k] = rep[g];                                       // entry k's row
    }
    for (uint32_t i = 0; i < n; ++i) sc.group[i] = sc.rank[sc.group[i]];   // bounded: n
    if (bytes > UINT32_MAX) return p;
    p.count = count;
    p.code_bytes = count <= 256u ? 1u : (count <= 65536u ? 2u : 4u);
    p.dict_bytes = (uint64_t{count} + 1u) * sizeof(uint32_t) + bytes;
    p.long_bytes = long_bytes;
    p.bytes = uint64_t{n} * p.code_bytes + p.dict_bytes;
    return p;
}

/// codes: n * plan.code_bytes; dict: plan.dict_bytes.
inline void dict_encode(const StringView* BOLT_RESTRICT rows, uint32_t n, const void* overflow, const DictPlan& p,
                        const DictScratch& sc, uint8_t* BOLT_RESTRICT codes, uint8_t* BOLT_RESTRICT dict) noexcept {
    assert(p.bytes != UINT64_MAX && codes != nullptr && dict != nullptr);
    auto* offs = reinterpret_cast<uint32_t*>(dict);
    uint8_t* bytes = dict + (static_cast<size_t>(p.count) + 1u) * sizeof(uint32_t);
    uint32_t at = 0;
    for (uint32_t k = 0; k < p.count; ++k) {                        // bounded: count
        const StringView& v = rows[sc.order[k]];
        std::memcpy(&offs[k], &at, sizeof(at));
        if (v.length) std::memcpy(bytes + at, detail::dict_sv_bytes(v, overflow), v.length);
        at += v.length;
    }
    std::memcpy(&offs[p.count], &at, sizeof(at));
    if (p.code_bytes == 1) {
        for (uint32_t i = 0; i < n; ++i) codes[i] = static_cast<uint8_t>(sc.group[i]);          // bounded: n
    } else if (p.code_bytes == 2) {
        auto* c = reinterpret_cast<uint16_t*>(codes);
        for (uint32_t i = 0; i < n; ++i) c[i] = static_cast<uint16_t>(sc.group[i]);             // bounded: n
    } else {
        std::memcpy(codes, sc.group, static_cast<size_t>(n) * sizeof(uint32_t));
    }
}

/// views_scratch: count StringViews. out_views: n; out_overflow: exactly overflow_len bytes (the plan's
/// long_bytes). False on any inconsistency.
[[nodiscard]] inline bool dict_decode(const uint8_t* BOLT_RESTRICT codes, uint64_t codes_len, uint32_t code_bytes,
                                      const uint8_t* BOLT_RESTRICT dict, uint64_t dict_len, uint32_t count,
                                      uint32_t n, StringView* BOLT_RESTRICT views_scratch,
                                      StringView* BOLT_RESTRICT out_views, uint8_t* BOLT_RESTRICT out_overflow,
                                      uint64_t overflow_len) noexcept {
    if (count == 0 || n == 0 || (code_bytes != 1 && code_bytes != 2 && code_bytes != 4)) return false;
    if (codes_len != uint64_t{n} * code_bytes) return false;
    const uint64_t head = (uint64_t{count} + 1u) * sizeof(uint32_t);
    if (dict_len < head || views_scratch == nullptr || out_views == nullptr) return false;
    const auto* offs = reinterpret_cast<const uint32_t*>(dict);
    const uint8_t* bytes = dict + head;
    const uint64_t payload = dict_len - head;
    if (offs[0] != 0 || offs[count] != payload) return false;
    uint64_t at = 0;
    for (uint32_t k = 0; k < count; ++k) {                          // bounded: count
        const uint32_t lo = offs[k], hi = offs[k + 1];
        if (hi < lo || hi > payload) return false;
        const uint32_t len = hi - lo;
        StringView v;
        std::memset(&v, 0, sizeof(v));
        v.length = len;
        if (len <= 12u) {
            if (len) std::memcpy(v.prefix, bytes + lo, len);
        } else {
            if (at + len > overflow_len || at > UINT32_MAX) return false;
            std::memcpy(v.prefix, bytes + lo, 4);
            v.ref.buf_idx = 0;
            v.ref.offset = static_cast<uint32_t>(at);
            std::memcpy(out_overflow + at, bytes + lo, len);
            at += len;
        }
        views_scratch[k] = v;
    }
    if (at != overflow_len) return false;
    uint32_t top = 0;
    if (code_bytes == 1) {
        for (uint32_t i = 0; i < n; ++i) top = codes[i] > top ? codes[i] : top;                  // bounded: n
    } else if (code_bytes == 2) {
        const auto* c = reinterpret_cast<const uint16_t*>(codes);
        for (uint32_t i = 0; i < n; ++i) top = c[i] > top ? c[i] : top;                          // bounded: n
    } else {
        const auto* c = reinterpret_cast<const uint32_t*>(codes);
        for (uint32_t i = 0; i < n; ++i) top = c[i] > top ? c[i] : top;                          // bounded: n
    }
    if (top >= count) return false;
    if (code_bytes == 1) {
        for (uint32_t i = 0; i < n; ++i) out_views[i] = views_scratch[codes[i]];                 // bounded: n
    } else if (code_bytes == 2) {
        const auto* c = reinterpret_cast<const uint16_t*>(codes);
        for (uint32_t i = 0; i < n; ++i) out_views[i] = views_scratch[c[i]];                     // bounded: n
    } else {
        const auto* c = reinterpret_cast<const uint32_t*>(codes);
        for (uint32_t i = 0; i < n; ++i) out_views[i] = views_scratch[c[i]];                     // bounded: n
    }
    return true;
}

}  // namespace pagecodec
}  // namespace bolt
