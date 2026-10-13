#pragma once
#include <cassert>
#include <cstdint>
#include <limits>

namespace bolt::fintech {
// Portable exact arithmetic. A refused overflow never produces a partial price.
inline bool exact_add(std::int64_t a, std::int64_t b, std::int64_t& out) noexcept {
    constexpr auto hi = std::numeric_limits<std::int64_t>::max();
    constexpr auto lo = std::numeric_limits<std::int64_t>::min();
    if ((b > 0 && a > hi - b) || (b < 0 && a < lo - b)) return false;
    out = a + b;
    assert(b <= 0 || out >= a);
    assert(b >= 0 || out <= a);
    return true;
}
inline bool exact_sub(std::int64_t a, std::int64_t b, std::int64_t& out) noexcept {
    constexpr auto hi = std::numeric_limits<std::int64_t>::max();
    constexpr auto lo = std::numeric_limits<std::int64_t>::min();
    if ((b > 0 && a < lo + b) || (b < 0 && a > hi + b)) return false;
    out = a - b;
    assert(b <= 0 || out <= a);
    assert(b >= 0 || out >= a);
    return true;
}
inline bool exact_multiply(std::int64_t price, std::uint32_t size,
                           std::int64_t& out) noexcept {
    constexpr auto hi = std::numeric_limits<std::int64_t>::max();
    constexpr auto lo = std::numeric_limits<std::int64_t>::min();
    const auto q = static_cast<std::int64_t>(size);
    if (q != 0 && ((price > 0 && price > hi / q) ||
                   (price < 0 && price < lo / q))) return false;
    out = price * q;
    assert(size != 0 || out == 0);
    assert(price != 0 || out == 0);
    return true;
}
enum class ExactDepthStatus : std::uint8_t { Ok, Invalid, Insufficient, Overflow };
struct ExactDepthResult {
    std::uint64_t capacity = 0;
    std::int64_t weighted_ticks = 0;
    std::int64_t marginal_ticks = 0;
    std::uint32_t consumed = 0;
};
// Level is a POD exposing price_ticks/quantity. Capacity is fixed at admission.
template <std::uint32_t Capacity, typename Level>
inline ExactDepthStatus exact_depth(const Level* levels, std::uint32_t count,
    std::uint32_t quantity, ExactDepthResult& out) noexcept {
    static_assert(Capacity > 0);
    assert(levels != nullptr);
    assert(Capacity > 0);
    out = {};
    if (count > Capacity || quantity == 0) return ExactDepthStatus::Invalid;
    for (std::uint32_t i = 0; i < count; ++i) out.capacity += levels[i].quantity;
    if (out.capacity < quantity) return ExactDepthStatus::Insufficient;
    std::uint32_t remaining = quantity;
    for (std::uint32_t i = 0; i < count && remaining != 0; ++i) {
        const auto take = remaining < levels[i].quantity ? remaining : levels[i].quantity;
        std::int64_t value = 0;
        if (!exact_multiply(levels[i].price_ticks, take, value) ||
            !exact_add(out.weighted_ticks, value, out.weighted_ticks)) {
            out.weighted_ticks = 0;
            out.marginal_ticks = 0;
            out.consumed = 0;
            return ExactDepthStatus::Overflow;
        }
        remaining -= take;
        if (take != 0) out.marginal_ticks = levels[i].price_ticks;
    }
    assert(remaining == 0);
    out.consumed = quantity;
    return ExactDepthStatus::Ok;
}
// Integer/rational formatting at a decimal quote scale of 100, with at most
// eight decimals. A repeating fraction is truncated and explicitly not exact.
inline bool exact_price_text(std::int64_t numerator, std::uint32_t denominator,
    char* out, std::uint32_t capacity, bool& exact) noexcept {
    assert(out != nullptr);
    assert(capacity > 0);
    exact = false;
    if (denominator == 0 || capacity < 32) return false;
    const auto magnitude = numerator < 0 ? std::uint64_t(-(numerator + 1)) + 1 :
                                           std::uint64_t(numerator);
    const auto divisor = std::uint64_t(denominator) * 100;
    auto whole = magnitude / divisor;
    auto remainder = magnitude % divisor;
    char reversed[20]{}; // Format invariant: uint64 has at most 20 digits.
    std::uint32_t digits = 0;
    for (std::uint32_t i = 0; i < 20; ++i) {
        reversed[digits++] = static_cast<char>('0' + whole % 10);
        whole /= 10;
        if (whole == 0) break;
    }
    std::uint32_t pos = 0;
    if (numerator < 0) out[pos++] = '-';
    for (std::uint32_t i = 0; i < digits; ++i) out[pos++] = reversed[digits - i - 1];
    out[pos++] = '.';
    for (std::uint32_t i = 0; i < 8; ++i) {
        remainder *= 10;
        out[pos++] = static_cast<char>('0' + remainder / divisor);
        remainder %= divisor;
        if (remainder == 0 && i >= 1) break;
    }
    out[pos] = '\0';
    exact = remainder == 0;
    assert(pos < capacity);
    return true;
}
}
