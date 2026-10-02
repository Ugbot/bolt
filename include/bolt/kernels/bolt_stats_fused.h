// bolt_stats_fused.h — MSEG K7: page/stripe statistics produced by the pass
// that copies a column chunk into its page (layout decision §9, §17 K7).
//
// One pass over a fixed-width column (Int8..Int64, UInt8..UInt64, Float32,
// Float64, Date32, Date64, Timestamp, Duration, Decimal64, Decimal128) copies
// it to `dst` (null slots written as zero; dst may be null for stats only) and
// accumulates FusedStats. Feed a page in one or many update() calls; results
// do not depend on how the rows were chunked.
//
// Semantics (the spec the scalar reference in test_bolt_stats_fused.cpp
// checks):
//   order    ints by value (unsigned as unsigned); floats by IEEE total order
//            restricted to non-NaN values (-0.0 < +0.0); Decimal128 as 128-bit
//            two's complement.
//   min/max  over non-null, non-NaN values; kStatsExtremesValid says whether one
//            exists. Encoding in the 16 B slots: ints sign- (signed) or zero-
//            (unsigned) extended little-endian; f32 bits in bytes 0..3, f64
//            bits in 0..7, rest zero; Decimal128 its 16 bytes. Zero when not
//            valid.
//   sum      ints: the exact 128-bit total as (sum_hi:sum), sum = its low 64
//            bits (wraps); kStatsSumValid iff the total fits int64 (signed
//            kinds) or uint64 (unsigned). Floats: a double sum over non-null,
//            non-NaN values in kStatsSumLanes lanes (lane = row index mod 8,
//            in row order), folded ((l0+l1)+(l2+l3))+((l4+l5)+(l6+l7)); sum
//            holds its bits, sum_hi 0, valid iff finite. Decimal128: absent
//            (0, not valid).
//   sorted   asc/desc: non-null values non-decreasing/non-increasing (floats
//            numerically, so -0.0 == +0.0); vacuously true with < 2 values;
//            false for floats with any NaN.
//   constant >= 1 non-null value, all bitwise equal, no NaN. Nulls allowed
//            (null_count says whether a pure Constant page applies).
//   all_null row_count > 0 and every row null.
//
// stats_fused_merge folds two adjacent ranges (page -> stripe). It equals a
// single pass over the concatenation for every field except a float sum,
// which is (a.sum + b.sum).
//
// K4 hook: update() takes an extension policy `Ext` whose
//   template <class T> void on_rows(const T* src, uint32_t n, uint64_t row,
//                                   bool dense)
// runs inside the same pass on each sub-range just processed (still in L1;
// dense = every row valid and NaN-free; not called for Decimal128). The
// default is a no-op. Page-kind detection (constant / all-null, run
// counting) adds a policy here instead of a second pass over the page.
#pragma once

#include "bolt/bolt_port.h"
#include "bolt/bolt_stats_limits.h"
#include "bolt/bolt_types.h"
#include "bolt/bolt_zonemap.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace bolt::stats {

inline constexpr uint16_t kStatsSortedAsc   = 1u << 0;
inline constexpr uint16_t kStatsSortedDesc  = 1u << 1;
inline constexpr uint16_t kStatsConstant    = 1u << 2;
inline constexpr uint16_t kStatsAllNull     = 1u << 3;
inline constexpr uint16_t kStatsHasNan      = 1u << 4;
inline constexpr uint16_t kStatsSumValid    = 1u << 5;
// Bits 6..8 are the writer's (StripeStat exact/has_bloom/has_q8).
inline constexpr uint16_t kStatsExtremesValid = 1u << 9;

struct alignas(64) FusedStats {
    uint8_t  min[16];     //  0
    uint8_t  max[16];     // 16
    int64_t  sum;         // 32 low 64 bits of the int total; f64 bits for floats
    uint32_t null_count;  // 40
    uint32_t row_count;   // 44
    uint16_t flags;       // 48 kStats*
    uint8_t  type;        // 50 BoltType
    uint8_t  _pad;        // 51
    uint32_t nan_count;   // 52
    int64_t  sum_hi;      // 56 high 64 bits of the int total
};
static_assert(sizeof(FusedStats) == 64, "FusedStats is one cache line");
static_assert(offsetof(FusedStats, sum) == 32, "StripeStat sum offset");
static_assert(offsetof(FusedStats, null_count) == 40, "StripeStat null_count offset");
static_assert(offsetof(FusedStats, row_count) == 44, "StripeStat row_count offset");
static_assert(offsetof(FusedStats, flags) == 48, "StripeStat flags offset");
static_assert(offsetof(FusedStats, sum_hi) == 56, "sum_hi offset");

enum class StatsKind : uint8_t { kNone = 0, kSigned, kUnsigned, kF32, kF64, kD128 };
enum class StatsStatus : uint8_t { kOk = 0, kUnsupportedType, kTooManyRows, kTypeMismatch };

BOLT_FORCE_INLINE StatsKind stats_kind(BoltType t) noexcept {
    switch (t) {
        case BoltType::Int8: case BoltType::Int16: case BoltType::Int32:
        case BoltType::Int64: case BoltType::Date32: case BoltType::Date64:
        case BoltType::Timestamp: case BoltType::Duration: case BoltType::Decimal64:
            return StatsKind::kSigned;
        case BoltType::UInt8: case BoltType::UInt16: case BoltType::UInt32:
        case BoltType::UInt64:
            return StatsKind::kUnsigned;
        case BoltType::Float32: return StatsKind::kF32;
        case BoltType::Float64: return StatsKind::kF64;
        case BoltType::Decimal128: return StatsKind::kD128;
        default: return StatsKind::kNone;
    }
}

struct D128Key { uint64_t lo; int64_t hi; };

// Accumulator for one page/stripe in flight. POD; lives on the caller's
// stack. Keys are the order-preserving integers of bolt_stats_fused_impl.h.
struct StatsFusedState {
    int64_t  key_min, key_max;
    uint64_t last_raw;   // bits of the last non-null, non-NaN value
    D128Key  d_min, d_max, d_last;
    uint64_t sum_lo[kStatsSumLanes];
    int64_t  sum_hi[kStatsSumLanes];
    double   fsum[kStatsSumLanes];
    uint64_t rows;
    uint32_t nulls, nans, nvals;
    BoltType type;
    StatsKind kind;
    uint8_t  width;
    bool     asc, desc;
};

}  // namespace bolt::stats

#include "bolt/kernels/bolt_stats_fused_impl.h"

namespace bolt::stats {

inline StatsStatus stats_fused_init(StatsFusedState* st, BoltType type) noexcept {
    assert(st != nullptr);
    std::memset(st, 0, sizeof(*st));
    st->type = type;
    st->kind = stats_kind(type);
    if (st->kind == StatsKind::kNone) return StatsStatus::kUnsupportedType;
    st->width = static_cast<uint8_t>(type_size(type));
    assert(st->width == 1 || st->width == 2 || st->width == 4 || st->width == 8 ||
           st->width == 16);
    st->key_min = INT64_MAX;
    st->key_max = INT64_MIN;
    st->d_min = D128Key{UINT64_MAX, INT64_MAX};
    st->d_max = D128Key{0, INT64_MIN};
    st->asc = st->desc = true;
    return StatsStatus::kOk;
}

// Accumulate n rows of `src` (the state's type); validity bit i is at
// validity_offset + i (null validity = all valid). dst, when non-null,
// receives the copy with null slots zeroed; it may not overlap src.
template <class Ext = detail::NoExt>
inline StatsStatus stats_fused_update(StatsFusedState* st, const void* src,
                                      const uint8_t* validity, uint64_t validity_offset,
                                      uint32_t n, void* dst, Ext* ext = nullptr) noexcept {
    assert(st != nullptr);
    assert(n == 0 || src != nullptr);
    if (st->kind == StatsKind::kNone) return StatsStatus::kUnsupportedType;
    if (st->rows + n > kStatsMaxRows) return StatsStatus::kTooManyRows;
    if (n == 0) return StatsStatus::kOk;
    switch (st->kind) {
        case StatsKind::kSigned:
        case StatsKind::kUnsigned:
            detail::update_ints(*st, src, validity, validity_offset, n, dst, ext);
            break;
        case StatsKind::kF32:
            detail::update_as<float, StatsKind::kF32>(*st, src, validity, validity_offset, n, dst, ext);
            break;
        case StatsKind::kF64:
            detail::update_as<double, StatsKind::kF64>(*st, src, validity, validity_offset, n, dst, ext);
            break;
        default:
            detail::update_d128(*st, static_cast<const uint8_t*>(src), validity, validity_offset,
                                n, static_cast<uint8_t*>(dst));
            break;
    }
    return StatsStatus::kOk;
}

inline FusedStats stats_fused_finish(const StatsFusedState& st) noexcept {
    assert(st.kind != StatsKind::kNone);
    assert(st.rows == static_cast<uint64_t>(st.nulls) + st.nans + st.nvals);
    FusedStats o;
    std::memset(&o, 0, sizeof(o));
    o.type = static_cast<uint8_t>(st.type);
    o.row_count = static_cast<uint32_t>(st.rows);
    o.null_count = st.nulls;
    o.nan_count = st.nans;
    const bool d128 = st.kind == StatsKind::kD128;
    if (st.nvals > 0) {
        o.flags |= kStatsExtremesValid;
        if (d128) {
            std::memcpy(o.min, &st.d_min.lo, 8); std::memcpy(o.min + 8, &st.d_min.hi, 8);
            std::memcpy(o.max, &st.d_max.lo, 8); std::memcpy(o.max + 8, &st.d_max.hi, 8);
        } else {
            detail::write_key(st.kind, st.width, st.key_min, o.min);
            detail::write_key(st.kind, st.width, st.key_max, o.max);
        }
    }
    const bool one = d128 ? st.d_min.lo == st.d_max.lo && st.d_min.hi == st.d_max.hi
                          : st.key_min == st.key_max;
    if (st.asc && st.nans == 0) o.flags |= kStatsSortedAsc;
    if (st.desc && st.nans == 0) o.flags |= kStatsSortedDesc;
    if (one && st.nvals > 0 && st.nans == 0) o.flags |= kStatsConstant;
    if (st.rows > 0 && st.nulls == st.rows) o.flags |= kStatsAllNull;
    if (st.nans > 0) o.flags |= kStatsHasNan;
    detail::finish_sum(st, &o);
    return o;
}

// One-shot: copy + stats over a whole chunk.
inline StatsStatus stats_fused_copy(BoltType type, const void* src, const uint8_t* validity,
                                    uint64_t validity_offset, uint32_t n, void* dst,
                                    FusedStats* out) noexcept {
    assert(out != nullptr);
    StatsFusedState st;
    StatsStatus s = stats_fused_init(&st, type);
    if (s != StatsStatus::kOk) return s;
    s = stats_fused_update(&st, src, validity, validity_offset, n, dst);
    if (s != StatsStatus::kOk) return s;
    *out = stats_fused_finish(st);
    return StatsStatus::kOk;
}

// Stats of range a followed by range b.
inline StatsStatus stats_fused_merge(const FusedStats& a, const FusedStats& b,
                                     FusedStats* out) noexcept {
    assert(out != nullptr);
    if (a.type != b.type) return StatsStatus::kTypeMismatch;
    const StatsKind k = stats_kind(static_cast<BoltType>(a.type));
    if (k == StatsKind::kNone) return StatsStatus::kUnsupportedType;
    if (static_cast<uint64_t>(a.row_count) + b.row_count > kStatsMaxRows)
        return StatsStatus::kTooManyRows;
    FusedStats o;
    std::memset(&o, 0, sizeof(o));
    o.type = a.type;
    o.row_count = a.row_count + b.row_count;
    o.null_count = a.null_count + b.null_count;
    o.nan_count = a.nan_count + b.nan_count;
    const bool ha = (a.flags & kStatsExtremesValid) != 0, hb = (b.flags & kStatsExtremesValid) != 0;
    const bool both = ha && hb, nan = o.nan_count > 0;
    if (ha || hb) {
        o.flags |= kStatsExtremesValid;
        std::memcpy(o.min, (!hb || (both && detail::cmp_slot(k, a.min, b.min) <= 0)) ? a.min : b.min, 16);
        std::memcpy(o.max, (!hb || (both && detail::cmp_slot(k, a.max, b.max) >= 0)) ? a.max : b.max, 16);
        if (!nan && std::memcmp(o.min, o.max, 16) == 0) o.flags |= kStatsConstant;  // sv-memcmp-ok: min/max are fixed uint8_t[16] numeric slots, zero-padded, never a StringView
    }
    if ((a.flags & b.flags & kStatsSortedAsc) && !nan &&
        (!both || detail::cmp_sorted(k, a.max, b.min) <= 0)) o.flags |= kStatsSortedAsc;
    if ((a.flags & b.flags & kStatsSortedDesc) && !nan &&
        (!both || detail::cmp_sorted(k, a.min, b.max) >= 0)) o.flags |= kStatsSortedDesc;
    if (o.row_count > 0 && o.null_count == o.row_count) o.flags |= kStatsAllNull;
    if (nan) o.flags |= kStatsHasNan;
    detail::merge_sum(k, a, b, &o);
    assert(o.null_count <= o.row_count);
    *out = o;
    return StatsStatus::kOk;
}

// The 32 B ZoneMap for Morsel::zonemaps and frame trailers. An absent min/max
// is the empty range (prunes every comparison; null_count answers IS NULL).
inline void stats_to_zonemap(const FusedStats& s, ZoneMap* z) noexcept {
    assert(z != nullptr);
    const StatsKind k = stats_kind(static_cast<BoltType>(s.type));
    assert(k != StatsKind::kNone);
    const bool has = (s.flags & kStatsExtremesValid) != 0;
    if (k == StatsKind::kF32) *z = zone_make_empty_f32();
    else if (k == StatsKind::kF64) *z = zone_make_empty_f64();
    else *z = zone_make_empty_i64();
    if (has && k == StatsKind::kD128) {
        const D128Key mn = detail::d128_load(s.min), mx = detail::d128_load(s.max);
        const bool fit = mn.hi == (static_cast<int64_t>(mn.lo) >> 63) &&
                         mx.hi == (static_cast<int64_t>(mx.lo) >> 63);
        z->min_value = fit ? static_cast<int64_t>(mn.lo) : INT64_MIN;
        z->max_value = fit ? static_cast<int64_t>(mx.lo) : INT64_MAX;
    } else if (has) {
        std::memcpy(&z->min_value, s.min, 8);
        std::memcpy(&z->max_value, s.max, 8);
    }
    z->null_count = s.null_count;
    z->flags |= s.null_count > 0 ? kZoneFlagHasNulls : kZoneFlagAllValid;
    if (s.flags & kStatsSortedAsc) z->flags |= kZoneFlagSortedAsc;
    if (s.flags & (kStatsSortedAsc | kStatsSortedDesc)) z->flags |= kZoneFlagMonotonic;
    if (s.flags & kStatsConstant) {
        z->cardinality_class = kZoneCardConstant;
        z->distinct_count = 1;
    }
}

}  // namespace bolt::stats
