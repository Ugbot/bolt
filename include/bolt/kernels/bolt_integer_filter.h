#pragma once
// Exact integer column vs signed64 literal. The literal is classified before
// any conversion to the column width: negative unsigned thresholds and values
// outside a narrow lane never wrap. Existing comparison templates own the
// all-valid path; the nullable path honors byte bitmaps and sliced offsets.
#include "bolt/kernels/bolt_filter_null.h"
#include <limits>
#include <type_traits>

namespace bolt::kernels::filter {
template<class T>
inline int integer_literal_position(std::int64_t literal) noexcept {
    static_assert(std::is_integral_v<T> && !std::is_same_v<T,bool>);
    static_assert(sizeof(T) <= sizeof(std::int64_t));
    if constexpr (std::is_signed_v<T>) {
        if (literal < static_cast<std::int64_t>(std::numeric_limits<T>::min())) return -1;
        if (literal > static_cast<std::int64_t>(std::numeric_limits<T>::max())) return 1;
    } else {
        if (literal < 0) return -1;
        if (static_cast<std::uint64_t>(literal) > static_cast<std::uint64_t>(std::numeric_limits<T>::max())) return 1;
    }
    return 0;
}

// Source/bitmap immutable, destination disjoint and sized for the logical
// input length. Selection indices are relative to the (possibly sliced) data
// pointer; validity_offset is relative to the retained parent bitmap. No
// payload sentinel means null. Null payloads are not read on the bitmap path.
template<CmpOp Op, class T>
inline std::int64_t integer_literal_filter(
    const T* BOLT_RESTRICT data, std::int64_t rows,
    const std::int32_t* BOLT_RESTRICT selected, std::int64_t selected_count,
    std::int64_t literal, const std::uint8_t* BOLT_RESTRICT validity,
    std::uint64_t validity_offset, std::int32_t* BOLT_RESTRICT output) noexcept {
    assert(rows >= 0 && rows <= INT32_MAX);
    assert(selected_count >= 0 && selected_count <= rows);
    const auto count = selected == nullptr ? rows : selected_count;
    assert((data != nullptr && output != nullptr) || count == 0);
    assert(validity_offset <= UINT64_MAX-static_cast<std::uint64_t>(rows));
    if (count == 0) return 0;
    const auto position = integer_literal_position<T>(literal);
    const bool constant = position != 0;
    const bool constant_keep = cmp_one<Op,int>(-position,0);
    if (constant && !constant_keep) return 0;
    T threshold{};
    if (!constant) threshold = static_cast<T>(literal); // Only representable literals convert.
    if (validity == nullptr) {
        const auto n = static_cast<std::size_t>(rows), sn = static_cast<std::size_t>(selected_count);
        return static_cast<std::int64_t>(constant ? cmp_is_not_null_i32(output,selected,sn,n,nullptr) :
            cmp_scalar_i32<Op,T>(output,selected,sn,n,data,threshold,nullptr));
    }
    std::int64_t kept = 0;
    for (std::int64_t i = 0; i < count; ++i) {
        const auto row = selected == nullptr ? static_cast<std::int32_t>(i) : selected[i];
        assert(row >= 0 && row < rows);
        const auto bit = validity_offset+static_cast<std::uint64_t>(row);
        const bool valid = (validity[bit>>3] & (1u<<(bit&7))) != 0;
        output[kept] = row;
        kept += valid && (constant || cmp_one<Op,T>(data[row],threshold)) ? 1 : 0;
    }
    assert(kept >= 0 && kept <= count);
    return kept;
}
} // namespace bolt::kernels::filter
