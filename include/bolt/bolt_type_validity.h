#pragma once
#include "bolt/bolt_types.h"
namespace bolt {
// Wire/metadata admission must not depend on diagnostic names. The enum has
// reserved holes and internal refinements such as Decimal64; NUM_TYPES is a
// ceiling, not a valid type. Keep these ranges aligned with BoltType's ABI.
constexpr bool is_valid_type(BoltType type) noexcept {
    const auto v = static_cast<std::uint8_t>(type);
    return v <= 12 || (v >= 16 && v <= 19) || (v >= 25 && v <= 26) ||
           (v >= 30 && v <= 32) || (v >= 40 && v <= 44) || (v >= 50 && v <= 56);
}
static_assert(is_valid_type(BoltType::Decimal64));
static_assert(!is_valid_type(static_cast<BoltType>(13)));
static_assert(!is_valid_type(BoltType::NUM_TYPES));
} // namespace bolt
