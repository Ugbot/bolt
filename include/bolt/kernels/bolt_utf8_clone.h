// bolt_utf8_clone.h — checked, O(1)-scratch single-base StringView cloning.
// Valid source backing ranges are a caller precondition: BoltColumn has no
// spill capacity. Byte bound: min(referenced span, sum of valid tail lengths).
#pragma once
#include "bolt/bolt_types.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace bolt::detail {

inline bool clone_byte_extent(uint64_t count, uint64_t width,
                              size_t* bytes) noexcept {
    assert(bytes != nullptr);
    constexpr uint64_t cap = static_cast<uint64_t>(
        std::numeric_limits<ptrdiff_t>::max()) - 64u;
    if (width == 0 || count > cap / width) return false;
    *bytes = static_cast<size_t>(count * width);
    assert(*bytes <= cap);
    return true;
}

inline bool clone_validity_extent(int64_t count, int64_t offset,
                                  size_t* bytes) noexcept {
    assert(bytes != nullptr);
    if (count < 0 || offset < 0 || offset > INT64_MAX - count) return false;
    const uint64_t n = static_cast<uint64_t>(count);
    const bool ok = clone_byte_extent((n + 7u) / 8u, 1, bytes);
    assert(!ok || *bytes <= static_cast<uint64_t>(INT64_MAX));
    return ok;
}

inline bool clone_row_valid(const uint8_t* validity, int64_t offset,
                            int64_t row) noexcept {
    assert(offset >= 0 && row >= 0);
    assert(offset <= INT64_MAX - row);
    const uint64_t bit = static_cast<uint64_t>(offset + row);
    return validity == nullptr || ((validity[bit >> 3] >> (bit & 7u)) & 1u) != 0;
}

inline void clone_validity_copy(uint8_t* dst, const uint8_t* src,
                                int64_t offset, int64_t count) noexcept {
    assert(dst != nullptr && src != nullptr);
    assert(count > 0 && offset >= 0 && offset <= INT64_MAX - count);
    if ((offset & 7) == 0) {
        // Keep the common Flat bitmap clone as one memcpy; slices whose bit
        // offset is not byte-aligned use the bounded rebasing loop below.
        const auto bytes = (static_cast<uint64_t>(count) + 7u) / 8u;
        std::memcpy(dst, src + (offset >> 3), static_cast<size_t>(bytes));
        return;
    }
    for (int64_t i = 0; i < count; ++i) {
        if (clone_row_valid(src, offset, i))
            dst[i >> 3] |= static_cast<uint8_t>(1u << (i & 7));
    }
}

struct Utf8ClonePlan {
    uint64_t lo = 0;
    uint64_t hi = 0;
    uint64_t sum = 0;
    size_t bytes = 0;
    bool any = false;
    bool packed_starts_fit = true;
    bool span = true;
};

// Pure arithmetic seam permits huge-range tests without huge allocations.
inline bool utf8_clone_add_range(Utf8ClonePlan* plan, uint64_t offset,
                                 uint64_t length) noexcept {
    assert(plan != nullptr);
    assert(!plan->any || plan->lo <= plan->hi);
    if (offset > UINT64_MAX - length || plan->sum > UINT64_MAX - length)
        return false;
    const uint64_t end = offset + length;
    if (!plan->any || offset < plan->lo) plan->lo = offset;
    if (end > plan->hi) plan->hi = end;
    plan->packed_starts_fit = plan->packed_starts_fit && plan->sum <= UINT32_MAX;
    plan->sum += length;
    plan->any = true;
    return true;
}

inline bool utf8_clone_finish_plan(Utf8ClonePlan* plan) noexcept {
    assert(plan != nullptr);
    assert(!plan->any || plan->lo <= plan->hi);
    const uint64_t span = plan->hi - plan->lo;
    plan->span = span <= plan->sum; // Ties preserve sharing and overlaps.
    if (!plan->span && !plan->packed_starts_fit) return false;
    // Check SOURCE address arithmetic too, even if copying only packed tails.
    size_t checked = 0;
    if (!clone_byte_extent(plan->hi, 1, &checked)) return false;
    return clone_byte_extent(plan->span ? span : plan->sum, 1, &plan->bytes);
}

inline bool utf8_clone_plan(const StringView* rows, int64_t count,
                            const uint8_t* validity, int64_t offset,
                            const void* base, Utf8ClonePlan* plan) noexcept {
    assert(plan != nullptr);
    assert(rows != nullptr || count == 0);
    *plan = {};
    size_t checked = 0;
    if (!clone_validity_extent(count, offset, &checked) ||
        !clone_byte_extent(static_cast<uint64_t>(count), sizeof(StringView), &checked))
        return false;
    for (int64_t i = 0; i < count; ++i) {
        if (!clone_row_valid(validity, offset, i)) continue;
        if (rows[i].length <= 12u) continue;
        if (base == nullptr ||
            !utf8_clone_add_range(plan, rows[i].ref.offset, rows[i].length))
            return false;
    }
    return utf8_clone_finish_plan(plan);
}

inline void utf8_clone_copy_span(StringView* rows, int64_t count,
                                 const uint8_t* validity, int64_t offset,
                                 char* dst, const char* src,
                                 const Utf8ClonePlan& plan) noexcept {
    assert(plan.span && plan.any && dst != nullptr && src != nullptr);
    assert(count > 0 && plan.bytes == plan.hi - plan.lo);
    std::memcpy(dst, src + static_cast<size_t>(plan.lo), plan.bytes);
    for (int64_t i = 0; i < count; ++i) {
        if (!clone_row_valid(validity, offset, i) || rows[i].length <= 12u) continue;
        const uint64_t rebased = rows[i].ref.offset - plan.lo;
        assert(rebased <= UINT32_MAX);
        rows[i].ref.buf_idx = 0;
        rows[i].ref.offset = static_cast<uint32_t>(rebased);
    }
}

inline void utf8_clone_copy_packed(StringView* rows, int64_t count,
                                   const uint8_t* validity, int64_t offset,
                                   char* dst, const char* src,
                                   const Utf8ClonePlan& plan) noexcept {
    assert(!plan.span && plan.any && dst != nullptr && src != nullptr);
    assert(count > 0 && plan.bytes == plan.sum && plan.packed_starts_fit);
    size_t cursor = 0;
    for (int64_t i = 0; i < count; ++i) {
        if (!clone_row_valid(validity, offset, i) || rows[i].length <= 12u) continue;
        assert(cursor <= UINT32_MAX);
        std::memcpy(dst + cursor, src + rows[i].ref.offset, rows[i].length);
        rows[i].ref.buf_idx = 0;
        rows[i].ref.offset = static_cast<uint32_t>(cursor);
        cursor += rows[i].length;
    }
    assert(cursor == plan.bytes);
}

} // namespace bolt::detail
