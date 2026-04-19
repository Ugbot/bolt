// bolt_zonemap.h — Unified 32-byte zone-map primitive.
//
// POD summary stats for a contiguous range of values in a column. One
// granularity-agnostic struct used by SSTable blocks, PDX clusters,
// BoltBatch column stats, and future scan pushdown. Consolidates four
// parallel zone-map representations (MarbleDB Wave 10.1 cross-pollination
// audit — see docs/MARBLEDB_DECISION_LOG.md §"Decision 13").
//
// RULES (Tiger Style):
//   - POD — no ctors/dtors, memcpy-safe.
//   - 32 bytes, 32-byte aligned — half a cache line.
//   - No exceptions, everything `noexcept` / `BOLT_FORCE_INLINE`.
//   - No std::* containers; no RTTI; no virtuals.
//   - Branchless predicate-pushdown helpers for the scan hot path.
//
// Type-punning: `min_value` / `max_value` are stored as int64_t bit
// patterns. Callers interpret the bits per the column's physical type
// (signed i64 direct, unsigned u64 via bit_cast, IEEE f32 via the low
// 32 bits, f64 via bit_cast<u64>, fixed-16 prefix via the raw 8 bytes).
// The per-type predicate helpers below handle the reinterpretation.
//
// For vector-search BOND / σ-bound (PDX): one ZoneMap per (cluster, dim).
// `null_count` is reused there as the Welford sample count; callers that
// don't care about nulls leave it zero.

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cfloat>
#include <climits>
#include <cstdint>
#include <cstring>

namespace bolt {

// ---------------------------------------------------------------------------
// Flag bits — `ZoneMap::flags` byte.
// ---------------------------------------------------------------------------
inline constexpr uint8_t kZoneFlagHasNulls   = 1u << 0;
inline constexpr uint8_t kZoneFlagSortedAsc  = 1u << 1;
inline constexpr uint8_t kZoneFlagMonotonic  = 1u << 2;
inline constexpr uint8_t kZoneFlagAllValid   = 1u << 3;
inline constexpr uint8_t kZoneFlagMeanValid  = 1u << 4;  // mean populated
inline constexpr uint8_t kZoneFlagVarValid   = 1u << 5;  // variance populated

// ---------------------------------------------------------------------------
// Cardinality class.
// ---------------------------------------------------------------------------
inline constexpr uint8_t kZoneCardUnknown  = 0u;
inline constexpr uint8_t kZoneCardConstant = 1u;
inline constexpr uint8_t kZoneCardLow      = 2u;
inline constexpr uint8_t kZoneCardMedium   = 3u;
inline constexpr uint8_t kZoneCardHigh     = 4u;

// NaN sentinel for f32 slots that haven't been populated.
inline constexpr uint32_t kZoneNanBits = 0x7FC00000u;

// ---------------------------------------------------------------------------
// ZoneMap — 32 bytes exactly.
// ---------------------------------------------------------------------------
struct alignas(32) ZoneMap {
    int64_t  min_value;         //  0 ..  7 : type-punned min
    int64_t  max_value;         //  8 .. 15 : type-punned max
    float    mean;              // 16 .. 19 : NaN when unset
    float    variance;          // 20 .. 23 : Welford population variance
    uint32_t null_count;        // 24 .. 27 : reused as Welford `n` by PDX
    uint16_t distinct_count;    // 28 .. 29 : 0 = unknown, capped at 65535
    uint8_t  cardinality_class; // 30       : one of kZoneCard*
    uint8_t  flags;             // 31       : kZoneFlag*
};
static_assert(sizeof(ZoneMap) == 32, "ZoneMap must be exactly 32 bytes");
static_assert(alignof(ZoneMap) >= 32, "ZoneMap must be 32-byte aligned");
static_assert(offsetof(ZoneMap, min_value) == 0, "min_value offset");
static_assert(offsetof(ZoneMap, max_value) == 8, "max_value offset");
static_assert(offsetof(ZoneMap, mean) == 16, "mean offset");
static_assert(offsetof(ZoneMap, variance) == 20, "variance offset");
static_assert(offsetof(ZoneMap, null_count) == 24, "null_count offset");
static_assert(offsetof(ZoneMap, distinct_count) == 28, "distinct_count offset");
static_assert(offsetof(ZoneMap, cardinality_class) == 30, "card offset");
static_assert(offsetof(ZoneMap, flags) == 31, "flags offset");

// ---------------------------------------------------------------------------
// Factories
// ---------------------------------------------------------------------------

// Empty zone map with NaN mean/variance and neutral extremes for signed
// i64 accumulation (min = INT64_MAX, max = INT64_MIN so first sample
// trivially wins the compare).
BOLT_FORCE_INLINE ZoneMap zone_make_empty_i64() noexcept {
    ZoneMap z;
    std::memset(&z, 0, sizeof(z));
    z.min_value = INT64_MAX;
    z.max_value = INT64_MIN;
    uint32_t nb = kZoneNanBits;
    std::memcpy(&z.mean, &nb, sizeof(nb));
    std::memcpy(&z.variance, &nb, sizeof(nb));
    return z;
}

// Empty zone map ready for f32 accumulation: min = +inf, max = -inf
// stored in the low 32 bits of min_value/max_value.
BOLT_FORCE_INLINE ZoneMap zone_make_empty_f32() noexcept {
    ZoneMap z;
    std::memset(&z, 0, sizeof(z));
    const uint32_t pinf_bits = 0x7F800000u;
    const uint32_t ninf_bits = 0xFF800000u;
    int64_t mn = 0, mx = 0;
    std::memcpy(&mn, &pinf_bits, sizeof(pinf_bits));
    std::memcpy(&mx, &ninf_bits, sizeof(ninf_bits));
    z.min_value = mn;
    z.max_value = mx;
    uint32_t nb = kZoneNanBits;
    std::memcpy(&z.mean, &nb, sizeof(nb));
    std::memcpy(&z.variance, &nb, sizeof(nb));
    return z;
}

// ---------------------------------------------------------------------------
// Bit-cast helpers to read min/max as a typed value.
// ---------------------------------------------------------------------------

BOLT_FORCE_INLINE int64_t zone_min_i64(const ZoneMap* z) noexcept {
    assert(z != nullptr);
    return z->min_value;
}

BOLT_FORCE_INLINE int64_t zone_max_i64(const ZoneMap* z) noexcept {
    assert(z != nullptr);
    return z->max_value;
}

BOLT_FORCE_INLINE uint64_t zone_min_u64(const ZoneMap* z) noexcept {
    assert(z != nullptr);
    uint64_t v;
    std::memcpy(&v, &z->min_value, sizeof(v));
    return v;
}

BOLT_FORCE_INLINE uint64_t zone_max_u64(const ZoneMap* z) noexcept {
    assert(z != nullptr);
    uint64_t v;
    std::memcpy(&v, &z->max_value, sizeof(v));
    return v;
}

BOLT_FORCE_INLINE float zone_min_f32(const ZoneMap* z) noexcept {
    assert(z != nullptr);
    uint32_t lo = static_cast<uint32_t>(
        static_cast<uint64_t>(z->min_value) & 0xFFFFFFFFu);
    float f;
    std::memcpy(&f, &lo, sizeof(f));
    return f;
}

BOLT_FORCE_INLINE float zone_max_f32(const ZoneMap* z) noexcept {
    assert(z != nullptr);
    uint32_t lo = static_cast<uint32_t>(
        static_cast<uint64_t>(z->max_value) & 0xFFFFFFFFu);
    float f;
    std::memcpy(&f, &lo, sizeof(f));
    return f;
}

BOLT_FORCE_INLINE void zone_set_min_f32(ZoneMap* z, float v) noexcept {
    assert(z != nullptr);
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    // Sign-extend to i64 slot (but keep just the low 32 bits; the high
    // bits stay zero so zone_min_f32 reads back the same value).
    uint64_t wide = static_cast<uint64_t>(bits);
    std::memcpy(&z->min_value, &wide, sizeof(wide));
}

BOLT_FORCE_INLINE void zone_set_max_f32(ZoneMap* z, float v) noexcept {
    assert(z != nullptr);
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    uint64_t wide = static_cast<uint64_t>(bits);
    std::memcpy(&z->max_value, &wide, sizeof(wide));
}

// ---------------------------------------------------------------------------
// Predicate-pushdown helpers — branchless; each returns `true` if the
// range's zone *could* contain a row satisfying the predicate.
//
// "can_have" means "pushdown cannot prove absence" — the scanner must
// still visit the block. Returning `false` is a hard skip.
// ---------------------------------------------------------------------------

BOLT_FORCE_INLINE bool zone_can_have_eq_i64(const ZoneMap* z,
                                             int64_t v) noexcept {
    assert(z != nullptr);
    return v >= z->min_value && v <= z->max_value;
}

BOLT_FORCE_INLINE bool zone_can_have_range_i64(const ZoneMap* z,
                                                int64_t lo,
                                                int64_t hi) noexcept {
    assert(z != nullptr);
    // disjoint iff lo > max OR hi < min.
    return !(lo > z->max_value || hi < z->min_value);
}

BOLT_FORCE_INLINE bool zone_can_have_eq_u64(const ZoneMap* z,
                                             uint64_t v) noexcept {
    assert(z != nullptr);
    const uint64_t mn = zone_min_u64(z);
    const uint64_t mx = zone_max_u64(z);
    return v >= mn && v <= mx;
}

BOLT_FORCE_INLINE bool zone_can_have_eq_f32(const ZoneMap* z,
                                             float v) noexcept {
    assert(z != nullptr);
    const float mn = zone_min_f32(z);
    const float mx = zone_max_f32(z);
    return v >= mn && v <= mx;
}

BOLT_FORCE_INLINE bool zone_can_have_range_f32(const ZoneMap* z,
                                                float lo,
                                                float hi) noexcept {
    assert(z != nullptr);
    const float mn = zone_min_f32(z);
    const float mx = zone_max_f32(z);
    return !(lo > mx || hi < mn);
}

// ---------------------------------------------------------------------------
// Vector-search distance bounds.
//
// For one dim of a clustered vector store with per-(cluster, dim) ZoneMap:
// given a query component `q`, the worst-case squared-L2 contribution of
// this dim across any vector in the cluster is
//   max((q − min)², (q − max)²).
// This is Weber 1998 VA-File — an exact upper envelope with no
// distributional assumption. Used by PDX σ-bound column-level early
// termination (MarbleDB Decision 11).
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE float zone_dim_l2_upper_bound_f32(const ZoneMap* z,
                                                     float q) noexcept {
    assert(z != nullptr);
    const float mn = zone_min_f32(z);
    const float mx = zone_max_f32(z);
    const float d0 = q - mn;
    const float d1 = q - mx;
    const float a  = d0 * d0;
    const float b  = d1 * d1;
    return (a > b) ? a : b;
}

// ---------------------------------------------------------------------------
// Merge — combines two zone maps (for compaction / roll-up). Follows the
// law-of-total-variance for `mean` + `variance`; min/max reduce.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE void zone_merge_i64(ZoneMap* dst,
                                       const ZoneMap* other) noexcept {
    assert(dst != nullptr);
    assert(other != nullptr);
    if (other->min_value < dst->min_value) dst->min_value = other->min_value;
    if (other->max_value > dst->max_value) dst->max_value = other->max_value;
    dst->null_count += other->null_count;
    // Distinct + cardinality become conservatively "high/unknown" after
    // merge — we can't cheaply union distinct counts.
    dst->distinct_count = 0;
    dst->cardinality_class = kZoneCardUnknown;
    // Flags: AND-intersect (sorted_asc / monotonic / all_valid survive only
    // if both sides had them); OR-union has_nulls / mean_valid / var_valid
    // are kept conservatively unset.
    const uint8_t keep = static_cast<uint8_t>(
        (dst->flags & other->flags) &
        (kZoneFlagSortedAsc | kZoneFlagMonotonic | kZoneFlagAllValid));
    dst->flags = keep;
    // Mean/variance become invalid on merge (would need the per-side n).
    uint32_t nb = kZoneNanBits;
    std::memcpy(&dst->mean, &nb, sizeof(nb));
    std::memcpy(&dst->variance, &nb, sizeof(nb));
}

BOLT_FORCE_INLINE void zone_merge_f32(ZoneMap* dst,
                                       const ZoneMap* other) noexcept {
    assert(dst != nullptr);
    assert(other != nullptr);
    const float dmn = zone_min_f32(dst);
    const float dmx = zone_max_f32(dst);
    const float omn = zone_min_f32(other);
    const float omx = zone_max_f32(other);
    if (omn < dmn) zone_set_min_f32(dst, omn);
    if (omx > dmx) zone_set_max_f32(dst, omx);
    dst->null_count += other->null_count;
    dst->distinct_count = 0;
    dst->cardinality_class = kZoneCardUnknown;
    const uint8_t keep = static_cast<uint8_t>(
        (dst->flags & other->flags) &
        (kZoneFlagSortedAsc | kZoneFlagMonotonic | kZoneFlagAllValid));
    dst->flags = keep;
    uint32_t nb = kZoneNanBits;
    std::memcpy(&dst->mean, &nb, sizeof(nb));
    std::memcpy(&dst->variance, &nb, sizeof(nb));
}

}  // namespace bolt
