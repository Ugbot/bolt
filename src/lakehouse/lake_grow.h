// lake_grow.h — arena doubling for lakehouse metadata arrays (kind C).
//
// Grows `*arr` to hold at least `need` elements: x2, in `a`, charging the new
// block to `b` (optional). The old block stays in the arena until it resets.
// On refusal the array is unchanged and ResourceExhausted is set.

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_budget.h"
#include "bolt/bolt_resource.h"

namespace bolt {
namespace lakehouse {

inline constexpr char kLakeMetadataKnob[] = "lake_metadata_budget_mb";

inline uint64_t lake_metadata_budget_bytes() noexcept {
    return bolt_limits_value(bolt_limits_id::lake_metadata_budget_mb) << 20;
}

template <typename T>
bool lake_grow(Arena* a, Budget* b, T** arr, uint32_t n, uint32_t* cap,
               uint32_t need) noexcept {
    assert(a != nullptr && arr != nullptr && cap != nullptr);
    assert(n <= *cap);
    if (need <= *cap) return true;
    uint64_t want = *cap < 8u ? 8u : static_cast<uint64_t>(*cap) * 2u;
    if (want < need) want = need;
    if (want > UINT32_MAX) {
        set_resource_exhausted(kLakeMetadataKnob, want, UINT32_MAX);
        return false;
    }
    const uint64_t bytes = want * sizeof(T);
    if (b != nullptr && !b->try_reserve(bytes)) return false;
    T* next = static_cast<T*>(a->allocate(static_cast<size_t>(bytes), alignof(T)));
    if (next == nullptr) {
        if (b != nullptr) b->release(bytes);
        set_resource_exhausted("arena_max_blocks", bytes, 0);
        return false;
    }
    if (n != 0u) std::memcpy(next, *arr, sizeof(T) * n);
    *arr = next;
    *cap = static_cast<uint32_t>(want);
    assert(*cap >= need);
    return true;
}

}  // namespace lakehouse
}  // namespace bolt
