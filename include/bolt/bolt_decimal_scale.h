// Shared fixed-decimal AVG scale contract; mantissa arithmetic stays in kernels.
#pragma once

#include <cassert>
#include <cstdint>

namespace bolt {

constexpr std::uint8_t k_decimal_avg_scale_extra = 4;
constexpr std::uint8_t k_decimal_scale_ceiling = 38;

// Preserve up to four extra fractional digits, within the decimal scale domain.
constexpr std::uint8_t decimal_avg_scale(std::uint8_t input_scale) noexcept {
    assert(input_scale <= k_decimal_scale_ceiling);
    const unsigned widened = input_scale + k_decimal_avg_scale_extra;
    const auto result = static_cast<std::uint8_t>(
        widened > k_decimal_scale_ceiling ? k_decimal_scale_ceiling : widened);
    assert(result >= input_scale && result <= k_decimal_scale_ceiling);
    return result;
}

}  // namespace bolt
