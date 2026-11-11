// bolt_value_counts.h — per-group value frequencies (Spark mode).
//
// One table per aggregate holds a count for every distinct (group id, value)
// it has seen: a SwissTableGrowable from a 64-bit hash of the pair to the head
// of that hash's entry chain, entries in an arena array. Memory is O(distinct
// pairs), bounded by the caller's `max_entries` budget; past it vc_add fails
// and the caller reports the knob. Partials merge by adding counts.
//
// Value identity is Spark Mode's OpenHashMap key equality: the bytes of a
// string, every NaN one key, -0.0 and 0.0 distinct. vc_best picks, per group,
// the highest count and among tied counts the smallest value in Spark's
// ordering (bytes unsigned, NaN above every double, -0.0 before 0.0 where
// Spark's ordering calls them equal), which is mode(x, true) exactly.

#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_hash.h"
#include "bolt/bolt_types.h"
#include "bolt/join/bolt_swiss_growable.h"
#include "bolt/kernels/bolt_utf8.h"

namespace bolt::kernels::value_counts {

inline constexpr uint32_t kVcNone = 0xFFFFFFFFu;
inline constexpr uint64_t kVcCanonNaN = 0x7FF8000000000000ULL;

struct VcEntry {
    int64_t  a;       // value cell: int / double bits / d128 low / StringView
    int64_t  b;       //             d128 high / StringView tail
    uint64_t count;
    uint32_t gid;
    uint32_t next;    // next entry with the same pair hash, kVcNone ends
};
static_assert(sizeof(VcEntry) == 32, "VcEntry layout");

struct ValueCounts {
    SwissTableGrowable idx;
    VcEntry*  ents;
    uint32_t  n;
    uint32_t  cap;
    uint32_t  max_entries;
    BoltType  t;
    uint8_t   _pad[3];
    char*     bytes;   // owned bytes of spilled (> 12-byte) strings
    size_t    used;
    size_t    bcap;
    Arena*    arena;
};

inline bool vc_supported(BoltType t) noexcept {
    switch (t) {
        case BoltType::Bool: case BoltType::Int8: case BoltType::Int16: case BoltType::Int32:
        case BoltType::Int64: case BoltType::Date32: case BoltType::Date64: case BoltType::Timestamp:
        case BoltType::TimestampNtz: case BoltType::Float32: case BoltType::Float64:
        case BoltType::Decimal64: case BoltType::Decimal128: case BoltType::Utf8:
            return true;
        default:
            return false;
    }
}

inline bool vc_init(ValueCounts* vc, Arena* arena, BoltType t, uint32_t max_entries) noexcept {
    assert(vc != nullptr && arena != nullptr);
    assert(vc_supported(t) && max_entries > 0);
    std::memset(vc, 0, sizeof(*vc));
    vc->arena = arena;
    vc->t = t;
    vc->max_entries = max_entries;
    return SwissTableGrowable::create(&vc->idx, 64, arena);
}

// Row `r` of `c` as a value cell; floats canonical (one NaN, -0.0 kept).
inline void vc_read(const BoltColumn& c, int64_t r, int64_t* a, int64_t* b) noexcept {
    assert(c.data != nullptr && r >= 0);
    *a = 0;
    *b = 0;
    switch (c.type) {
        case BoltType::Bool: *a = static_cast<const uint8_t*>(c.data)[r] != 0; break;
        case BoltType::Int8: *a = static_cast<const int8_t*>(c.data)[r]; break;
        case BoltType::Int16: *a = static_cast<const int16_t*>(c.data)[r]; break;
        case BoltType::Int32: case BoltType::Date32: *a = static_cast<const int32_t*>(c.data)[r]; break;
        case BoltType::Float32: {
            const double d = static_cast<double>(static_cast<const float*>(c.data)[r]);
            std::memcpy(a, &d, 8);
            break;
        }
        case BoltType::Decimal128: case BoltType::Utf8:
            std::memcpy(a, static_cast<const char*>(c.data) + r * 16, 8);
            std::memcpy(b, static_cast<const char*>(c.data) + r * 16 + 8, 8);
            break;
        default:   // Int64, Date64, Timestamp(Ntz), Float64, Decimal64: 8 bytes
            std::memcpy(a, static_cast<const char*>(c.data) + r * 8, 8);
            break;
    }
    const bool f = c.type == BoltType::Float32 || c.type == BoltType::Float64;
    if (f && (static_cast<uint64_t>(*a) & 0x7FFFFFFFFFFFFFFFULL) > 0x7FF0000000000000ULL)
        *a = static_cast<int64_t>(kVcCanonNaN);
    if (c.type == BoltType::Decimal64) *b = *a >> 63;   // widened d128
}

// The entry's string bytes (an inline view's bytes live in the entry itself).
inline const char* vc_str(const ValueCounts* vc, const VcEntry& e, uint32_t* len) noexcept {
    static_assert(offsetof(VcEntry, b) == offsetof(VcEntry, a) + 8, "a, b hold one StringView");
    const auto* sv = reinterpret_cast<const StringView*>(&e.a);
    *len = sv->length;
    return utf8::sv_bytes(*sv, vc->bytes);
}

inline uint64_t vc_hash(uint32_t gid, int64_t a, int64_t b, const char* s, uint32_t len) noexcept {
    const uint64_t g = static_cast<uint64_t>(gid) * 0x9E3779B97F4A7C15ULL;
    if (s != nullptr) return hash_bytes_accumulate_v1(s, len) ^ g ^ len;
    return static_cast<uint64_t>(a) ^ (static_cast<uint64_t>(b) * 0xC2B2AE3D27D4EB4FULL) ^ g;
}

inline bool vc_grow_entries(ValueCounts* vc) noexcept {
    assert(vc->n == vc->cap);
    if (vc->cap >= vc->max_entries) return false;
    uint32_t ncap = vc->cap == 0 ? 256u : vc->cap * 2u;
    if (ncap > vc->max_entries) ncap = vc->max_entries;
    VcEntry* ne = vc->arena->allocate_array<VcEntry>(ncap);
    if (ne == nullptr) return false;
    if (vc->n != 0) std::memcpy(ne, vc->ents, sizeof(VcEntry) * vc->n);
    vc->ents = ne;
    vc->cap = ncap;
    assert(vc->cap > vc->n);
    return true;
}

inline bool vc_own_bytes(ValueCounts* vc, const char* p, uint32_t len, uint32_t* off) noexcept {
    assert(vc != nullptr && p != nullptr);
    if (vc->used + len > vc->bcap) {
        size_t ncap = vc->bcap != 0 ? vc->bcap * 2u : size_t{4096};
        for (int i = 0; i < 40 && ncap < vc->used + len; ++i) ncap *= 2u;
        if (ncap < vc->used + len || ncap > 0xFFFFFFFFull) return false;
        char* nb = static_cast<char*>(vc->arena->allocate(ncap, 16));
        if (nb == nullptr) return false;
        if (vc->used != 0) std::memcpy(nb, vc->bytes, vc->used);
        vc->bytes = nb;
        vc->bcap = ncap;
    }
    *off = static_cast<uint32_t>(vc->used);
    std::memcpy(vc->bytes + vc->used, p, len);
    vc->used += len;
    return true;
}

// Add `count` occurrences of value (a, b) to group `gid`. A Utf8 value's
// spilled bytes resolve through `base`. False when the budget or the arena
// is exhausted (the table is unchanged).
inline bool vc_add(ValueCounts* vc, uint32_t gid, int64_t a, int64_t b, const char* base, uint64_t count) noexcept {
    assert(vc != nullptr && vc->arena != nullptr);
    assert(count > 0);
    const bool str = vc->t == BoltType::Utf8;
    StringView sv{};
    const char* s = nullptr;
    if (str) {
        std::memcpy(&sv, &a, sizeof(sv));
        s = utf8::sv_bytes(sv, base);
    }
    const uint64_t h = vc_hash(gid, a, b, s, sv.length);
    const int32_t head = vc->idx.find(h);
    for (uint32_t e = head < 0 ? kVcNone : static_cast<uint32_t>(head); e != kVcNone; e = vc->ents[e].next) {
        VcEntry& x = vc->ents[e];
        if (x.gid != gid) continue;
        bool eq = x.a == a && x.b == b;
        if (str) {
            uint32_t xl = 0;
            const char* xs = vc_str(vc, x, &xl);
            eq = xl == sv.length && std::memcmp(xs, s, xl) == 0;
        }
        if (eq) { x.count += count; return true; }
    }
    if (vc->n == vc->cap && !vc_grow_entries(vc)) return false;
    if (str && sv.length > 12u) {
        uint32_t off = 0;
        if (!vc_own_bytes(vc, s, sv.length, &off)) return false;
        sv.ref.buf_idx = 0;
        sv.ref.offset = off;
        std::memcpy(&a, &sv, 8);
        std::memcpy(&b, reinterpret_cast<const char*>(&sv) + 8, 8);
    }
    const uint32_t id = vc->n;
    if (!vc->idx.insert(h, id)) return false;
    vc->ents[id] = VcEntry{a, b, count, gid, head < 0 ? kVcNone : static_cast<uint32_t>(head)};
    ++vc->n;
    return true;
}

// Row r of column c into group gid; NULLs are skipped.
inline bool vc_add_row(ValueCounts* vc, uint32_t gid, const BoltColumn& c, int64_t r) noexcept {
    assert(vc != nullptr && c.type == vc->t);
    if (c.validity != nullptr) {
        const int64_t bit = c.validity_offset + r;
        if (((c.validity[bit >> 3] >> (bit & 7)) & 1) == 0) return true;
    }
    int64_t a = 0, b = 0;
    vc_read(c, r, &a, &b);
    return vc_add(vc, gid, a, b, static_cast<const char*>(c.str_overflow_base), 1);
}

// Merge every entry of `src` (group ids remapped through `gid_map`, or kept
// when it is null) into `dst`.
inline bool vc_merge(ValueCounts* dst, const ValueCounts* src, const uint32_t* gid_map) noexcept {
    assert(dst != nullptr && src != nullptr && dst->t == src->t);
    for (uint32_t i = 0; i < src->n; ++i) {
        const VcEntry& e = src->ents[i];
        const uint32_t g = gid_map != nullptr ? gid_map[e.gid] : e.gid;
        if (!vc_add(dst, g, e.a, e.b, src->bytes, e.count)) return false;
    }
    return true;
}

// Spark's ordering of two values of type t: <0, 0, >0 (-0.0 before 0.0).
inline int vc_cmp(const ValueCounts* vc, const VcEntry& x, const VcEntry& y) noexcept {
    const BoltType t = vc->t;
    if (t == BoltType::Utf8) {
        uint32_t xl = 0, yl = 0;
        const char* xs = vc_str(vc, x, &xl);
        const char* ys = vc_str(vc, y, &yl);
        const int c = std::memcmp(xs, ys, xl < yl ? xl : yl);
        if (c != 0) return c < 0 ? -1 : 1;
        return (xl > yl) - (xl < yl);
    }
    if (t == BoltType::Float32 || t == BoltType::Float64) {
        double p, q;
        std::memcpy(&p, &x.a, 8);
        std::memcpy(&q, &y.a, 8);
        const bool pn = p != p, qn = q != q;
        if (pn || qn) return static_cast<int>(pn) - static_cast<int>(qn);
        if (p != q) return p < q ? -1 : 1;
        return static_cast<int>(std::signbit(q)) - static_cast<int>(std::signbit(p));
    }
    if (t == BoltType::Decimal128 || t == BoltType::Decimal64) {
        if (x.b != y.b) return x.b < y.b ? -1 : 1;
        const auto xa = static_cast<uint64_t>(x.a), ya = static_cast<uint64_t>(y.a);
        return (xa > ya) - (xa < ya);
    }
    return (x.a > y.a) - (x.a < y.a);
}

// best[g] = the entry of group g's mode (kVcNone: the group saw no value).
inline void vc_best(const ValueCounts* vc, uint32_t n_groups, uint32_t* best) noexcept {
    assert(vc != nullptr && (best != nullptr || n_groups == 0));
    for (uint32_t g = 0; g < n_groups; ++g) best[g] = kVcNone;
    for (uint32_t i = 0; i < vc->n; ++i) {
        const VcEntry& e = vc->ents[i];
        if (e.gid >= n_groups) continue;
        const uint32_t cur = best[e.gid];
        if (cur == kVcNone || e.count > vc->ents[cur].count ||
            (e.count == vc->ents[cur].count && vc_cmp(vc, e, vc->ents[cur]) < 0))
            best[e.gid] = i;
    }
}

// Entry `e`'s value into row r of `out` (type vc->t; Utf8 rows point into
// vc->bytes, so the output column's str_overflow_base is vc->bytes).
inline void vc_store(const ValueCounts* vc, const VcEntry& e, void* out, int64_t r) noexcept {
    assert(vc != nullptr && out != nullptr && r >= 0);
    switch (vc->t) {
        case BoltType::Bool: static_cast<uint8_t*>(out)[r] = static_cast<uint8_t>(e.a != 0); break;
        case BoltType::Int8: static_cast<int8_t*>(out)[r] = static_cast<int8_t>(e.a); break;
        case BoltType::Int16: static_cast<int16_t*>(out)[r] = static_cast<int16_t>(e.a); break;
        case BoltType::Int32: case BoltType::Date32: static_cast<int32_t*>(out)[r] = static_cast<int32_t>(e.a); break;
        case BoltType::Float32: {
            double d;
            std::memcpy(&d, &e.a, 8);
            static_cast<float*>(out)[r] = static_cast<float>(d);
            break;
        }
        case BoltType::Decimal128: case BoltType::Utf8:
            std::memcpy(static_cast<char*>(out) + r * 16, &e.a, 8);
            std::memcpy(static_cast<char*>(out) + r * 16 + 8, &e.b, 8);
            break;
        default:
            std::memcpy(static_cast<char*>(out) + r * 8, &e.a, 8);
            break;
    }
}

}  // namespace bolt::kernels::value_counts
