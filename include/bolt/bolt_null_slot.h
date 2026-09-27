// bolt_null_slot.h — what a producer writes into the payload slot of a NULL row.
//
// NULL lives only in the validity bitmap (WI-2). A NULL row's payload slot is
// never data, but it is still memory a careless consumer can read, so a
// producer writes a DEFINED filler instead of leaving arena garbage: zero
// bytes, which is a zero-length inline StringView for Utf8 and 0 / 0.0 for the
// fixed-width types.
//
// Poison mode (BOLT_NULL_POISON=1, read once per process) writes 0xA5 bytes
// instead; a StringView slot becomes a well-formed 12-byte inline string of
// 0xA5 bytes, so a branchless kernel that compares every row and masks NULLs
// afterwards stays memory-safe, while a consumer that USES a NULL payload
// surfaces a visibly wrong value in the harnesses instead of a quiet zero.

#ifndef BOLT_NULL_SLOT_H
#define BOLT_NULL_SLOT_H

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "bolt/bolt_column.h"

namespace bolt {

inline constexpr uint8_t kNullPoisonByte = 0xA5;

inline bool null_poison_enabled() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("BOLT_NULL_POISON");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

inline uint8_t null_slot_byte() noexcept {
    return null_poison_enabled() ? kNullPoisonByte : uint8_t{0};
}

/// Fill one NULL row's payload slot of `width` bytes; `string_view` marks a
/// 16-byte StringView slot (Utf8 / Binary Flat layout).
inline void null_slot_fill(void* slot, size_t width, bool string_view) noexcept {
    assert(slot != nullptr || width == 0);
    assert(width <= 64 && (!string_view || width == sizeof(StringView)));
    std::memset(slot, null_slot_byte(), width);
    if (string_view && null_poison_enabled()) {
        static_cast<StringView*>(slot)->length = 12;   // inline, never spilled
    }
}

inline bool is_string_view_slot(const BoltColumn& c) noexcept {
    return (c.type == BoltType::Utf8 || c.type == BoltType::Binary) &&
           c.type_size_bytes == sizeof(StringView);
}

/// Refill every NULL row's slot of a Flat fixed-width (or StringView) column.
/// A no-op for a column without a validity bitmap or with no payload stride.
inline void null_slots_fill(BoltColumn* c) noexcept {
    assert(c != nullptr);
    assert(c->length >= 0);
    if (c->validity == nullptr || c->data == nullptr) return;
    if (c->format != ColumnFormat::Flat) return;
    const size_t w = c->type_size_bytes;
    if (w == 0 || w > 64) return;
    const bool sv = is_string_view_slot(*c);
    auto* base = static_cast<uint8_t*>(c->data);
    for (int64_t i = 0; i < c->length; ++i) {
        if (c->is_null(i)) null_slot_fill(base + static_cast<size_t>(i) * w, w, sv);
    }
}

}  // namespace bolt

#endif  // BOLT_NULL_SLOT_H
