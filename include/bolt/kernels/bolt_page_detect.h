// bolt_page_detect.h — MSEG K4: constant / all-null page detection, and the
// Constant page it emits (layout decision §6.0, §6.4 encoding 1, §17 K4).
//
// A page is
//   kAllNull   row_count > 0 and every row null;
//   kConstant  >= 1 non-null row, every non-null row bitwise equal (strings:
//              equal length and bytes), and, for Float32/Float64, no NaN;
//              nulls allowed (the page keeps its validity bitmap);
//   kFlat      otherwise (also: an empty page).
// The float rule is K7's kStatsConstant: -0.0 and +0.0 differ, a NaN page
// stays Flat (its min/max would not be valid).
//
// Detection rides the copy pass that builds the page:
//   - K7 types (stats_kind != kNone): stats_fused already computes constant
//     and all-null in its pass; page_kind_from_stats reads them (no work).
//   - other fixed widths <= 16 B (Bool, Float16, UUID, IPv4, FixedSizeBinary):
//     fixed_detect_update copies (null slots zeroed) and folds
//     (value ^ first) & valid into one accumulator; purely bitwise (a
//     Float16 page of one NaN pattern is Constant). Decimal256 (32 B) has
//     no Constant page.
//   - StringView strings (Utf8/Binary/Symbol): sv_detect_update copies the
//     views (null slots zeroed) and compares length + the bytes the length
//     covers; a long string whose reference differs is compared byte-wise
//     only while the page is still a candidate.
// Each detector can be fed a page in one or many calls; the answer does not
// depend on the chunking.
//
// page_make_constant turns the detected page into a ColumnFormat::Constant
// BoltColumn (value inline; validity borrowed when the page has nulls, the
// all-zero bitmap for kAllNull) that bolt_wire writes as a v5 Constant.
#pragma once

#include "bolt/bolt_column.h"
#include "bolt/bolt_port.h"
#include "bolt/bolt_stats_limits.h"
#include "bolt/bolt_types.h"
#include "bolt/kernels/bolt_bitmap.h"
#include "bolt/kernels/bolt_stats_fused.h"

#include <cassert>
#include <cstdint>
#include <cstring>

namespace bolt::page {

enum class PageKind : uint8_t { kFlat = 0, kConstant = 1, kAllNull = 2 };
enum class DetectStatus : uint8_t { kOk = 0, kUnsupportedWidth, kTooManyRows };

inline constexpr uint32_t kPageValueBytes = sizeof(BoltColumn::inline_value);
static_assert(kPageValueBytes == 16, "Constant pages hold a 16 B value");
static_assert(sizeof(StringView) == kPageValueBytes, "a StringView fits the Constant value");

// ---- detector state ---------------------------------------------------------

struct FixedDetectState {
    uint64_t first_lo, first_hi;  // first non-null value
    uint64_t diff;                // OR of (value ^ first) over non-null rows
    uint64_t rows;
    uint32_t nulls;
    uint32_t width;
    bool     have_first;
};

struct SvDetectState {
    StringView  first;        // first non-null view (as read)
    const char* first_bytes;  // its bytes (inline: &first.prefix; long: overflow)
    uint64_t    hdr_mask;     // bytes 0..7 of a view that carry length + prefix
    uint64_t    tail_mask;    // bytes 8..15 that carry inline data (0 when long)
    uint64_t    rows;
    uint32_t    nulls;
    bool        have_first;
    bool        differs;
};

}  // namespace bolt::page

#include "bolt/kernels/bolt_page_detect_impl.h"

namespace bolt::page {

// ---- K7 types: the fused stats already decided ------------------------------

inline PageKind page_kind_from_stats(const stats::FusedStats& s) noexcept {
    assert(s.null_count <= s.row_count);
    assert(!(s.flags & stats::kStatsConstant) || (s.flags & stats::kStatsExtremesValid));
    if (s.flags & stats::kStatsAllNull) return PageKind::kAllNull;
    if (s.flags & stats::kStatsConstant) return PageKind::kConstant;
    return PageKind::kFlat;
}

// The constant's bytes: the min slot holds the value in its first width bytes.
inline void page_value_from_stats(const stats::FusedStats& s, uint8_t value[kPageValueBytes]) noexcept {
    assert(value != nullptr);
    const size_t w = type_size(static_cast<BoltType>(s.type));
    assert(w > 0 && w <= kPageValueBytes);
    std::memset(value, 0, kPageValueBytes);
    if (s.flags & stats::kStatsConstant) std::memcpy(value, s.min, w);
}

// ---- other fixed widths -----------------------------------------------------

inline DetectStatus fixed_detect_init(FixedDetectState* st, uint32_t width) noexcept {
    assert(st != nullptr);
    std::memset(st, 0, sizeof(*st));
    if (width == 0 || width > kPageValueBytes || (width & (width - 1)) != 0)
        return DetectStatus::kUnsupportedWidth;
    st->width = width;
    assert(st->diff == 0 && !st->have_first);
    return DetectStatus::kOk;
}


// Copy n rows of width `st->width` from src to dst (dst may be null; null
// slots written as zero) and fold them into the detection.
inline DetectStatus fixed_detect_update(FixedDetectState* st, const void* src,
                                        const uint8_t* validity, uint64_t validity_offset,
                                        uint32_t n, void* dst) noexcept {
    assert(st != nullptr && st->width != 0);
    assert(n == 0 || src != nullptr);
    if (st->rows + n > stats::kStatsMaxRows) return DetectStatus::kTooManyRows;
    const auto* s = static_cast<const uint8_t*>(src);
    auto* d = static_cast<uint8_t*>(dst);
    switch (st->width) {
        case 1:  detail::fixed_detect_dispatch<1>(st, s, validity, validity_offset, n, d); break;
        case 2:  detail::fixed_detect_dispatch<2>(st, s, validity, validity_offset, n, d); break;
        case 4:  detail::fixed_detect_dispatch<4>(st, s, validity, validity_offset, n, d); break;
        case 8:  detail::fixed_detect_dispatch<8>(st, s, validity, validity_offset, n, d); break;
        case 16: detail::fixed_detect_dispatch<16>(st, s, validity, validity_offset, n, d); break;
        default: return DetectStatus::kUnsupportedWidth;
    }
    st->rows += n;
    assert(st->nulls <= st->rows);
    return DetectStatus::kOk;
}

inline PageKind fixed_detect_finish(const FixedDetectState& st,
                                    uint8_t value[kPageValueBytes]) noexcept {
    assert(value != nullptr);
    assert(st.nulls <= st.rows);
    std::memset(value, 0, kPageValueBytes);
    if (st.rows > 0 && st.nulls == st.rows) return PageKind::kAllNull;
    if (!st.have_first || st.diff != 0) return PageKind::kFlat;
    std::memcpy(value, &st.first_lo, st.width < 8 ? st.width : 8);
    if (st.width > 8) std::memcpy(value + 8, &st.first_hi, st.width - 8);
    return PageKind::kConstant;
}

// ---- StringView strings -----------------------------------------------------

inline void sv_detect_init(SvDetectState* st) noexcept {
    assert(st != nullptr);
    std::memset(st, 0, sizeof(*st));
    assert(!st->have_first && !st->differs);
}


// Copy n views (null slots zeroed) and fold them into the detection. The
// overflow bytes of the first non-null value must stay readable until finish.
inline DetectStatus sv_detect_update(SvDetectState* st, const StringView* src,
                                     const uint8_t* validity, uint64_t validity_offset,
                                     uint32_t n, StringView* dst,
                                     const char* overflow) noexcept {
    assert(st != nullptr);
    assert(n == 0 || src != nullptr);
    if (st->rows + n > stats::kStatsMaxRows) return DetectStatus::kTooManyRows;
    uint32_t i = 0;
    while (!st->have_first && i < n) {  // bounded: n
        if (detail::valid_mask(validity, validity_offset + i) != 0) {
            detail::sv_set_first(st, src[i], overflow);
            break;
        }
        ++st->nulls;
        if (dst != nullptr) std::memset(dst + i, 0, sizeof(StringView));
        ++i;
    }
    if (i < n)
        detail::sv_detect_rows(st, src + i, validity, validity_offset + i, n - i,
                               dst != nullptr ? dst + i : nullptr, overflow);
    st->rows += n;
    assert(st->nulls <= st->rows);
    return DetectStatus::kOk;
}

// value receives the first non-null view (its reference resolves against the
// overflow of the update that saw it).
inline PageKind sv_detect_finish(const SvDetectState& st, StringView* value) noexcept {
    assert(value != nullptr);
    assert(st.nulls <= st.rows);
    std::memset(value, 0, sizeof(*value));
    if (st.rows > 0 && st.nulls == st.rows) return PageKind::kAllNull;
    if (!st.have_first || st.differs) return PageKind::kFlat;
    *value = st.first;
    return PageKind::kConstant;
}

// ---- emission ---------------------------------------------------------------

// The Constant column for a page detected as kConstant or kAllNull. value is
// the detector's value bytes (a StringView for strings; overflow resolves a
// long one). validity is the page's bitmap: dropped when the page has no
// nulls, borrowed otherwise (kAllNull: it is all zero).
inline BoltColumn page_make_constant(BoltType type, PageKind kind,
                                     const uint8_t value[kPageValueBytes], int64_t rows,
                                     uint8_t* validity, int64_t validity_offset,
                                     uint32_t null_count, void* overflow) noexcept {
    assert(kind == PageKind::kConstant || kind == PageKind::kAllNull);
    assert(value != nullptr && rows > 0 && null_count <= static_cast<uint64_t>(rows));
    assert(null_count == 0 || validity != nullptr);
    assert(kind != PageKind::kAllNull || null_count == static_cast<uint64_t>(rows));
    const bool str = type == BoltType::Utf8 || type == BoltType::Binary || type == BoltType::Symbol;
    const size_t w = str ? sizeof(StringView) : type_size(type);
    assert(w > 0 && w <= kPageValueBytes);
    BoltColumn c = BoltColumn::make_empty();
    c.length = rows;
    c.format = ColumnFormat::Constant;
    c.type = type;
    c.type_size_bytes = static_cast<uint16_t>(w);
    std::memcpy(c.inline_value, value, kPageValueBytes);
    c.data = c.inline_value;
    if (str) c.str_overflow_base = overflow;
    if (!str) std::memcpy(&c.stats.min_value, value, w < 8 ? w : 8);
    c.stats.max_value = c.stats.min_value;
    c.stats.null_count = null_count;
    c.stats.all_valid = null_count == 0;
    c.stats.distinct_count = kind == PageKind::kConstant ? 1u : 0u;
    c.stats.cardinality = CardinalityClass::Constant;
    c.stats.sort_order = SortOrder::Ascending;
    if (null_count > 0) {
        c.validity = validity;
        c.validity_offset = validity_offset;
    }
    assert(c.format == ColumnFormat::Constant && c.data == c.inline_value);
    return c;
}

}  // namespace bolt::page
