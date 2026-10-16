#pragma once
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include "bolt/kernels/fintech/exact_depth.h"

namespace bolt::fintech {
enum class ExactIocSide : std::uint8_t { Buy, Sell };
enum class ExactIocStatus : std::uint8_t { Ok, Invalid, Overflow };
struct ExactIocResult {
    std::uint32_t requested = 0, filled = 0, remaining = 0, levels_touched = 0;
    std::int64_t weighted_ticks = 0, marginal_ticks = 0;
    bool price_limited = false;
    std::uint8_t reserved[7]{};
};
static_assert(sizeof(ExactIocResult) == 40 && alignof(ExactIocResult) == 8);
static_assert(std::is_standard_layout_v<ExactIocResult> && std::is_trivially_copyable_v<ExactIocResult>);

namespace exact_ioc_detail {
struct Range { std::uintptr_t begin = 0, end = 0; };

template <typename T>
inline bool range(const T* pointer, std::uint32_t count, Range& out) noexcept {
    if (count == 0) { out = {}; return true; }
    if (!pointer || count > std::numeric_limits<std::size_t>::max() / sizeof(T)) return false;
    const auto bytes = static_cast<std::size_t>(count) * sizeof(T);
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (begin % alignof(T) || bytes > std::numeric_limits<std::uintptr_t>::max() - begin) return false;
    out = {begin, begin + bytes};
    assert(out.begin < out.end);
    assert(out.end - out.begin == bytes);
    return true;
}
inline bool overlaps(Range a, Range b) noexcept {
    assert(a.begin <= a.end); assert(b.begin <= b.end);
    return a.begin != a.end && b.begin != b.end && a.begin < b.end && b.begin < a.end;
}
template <std::uint32_t Capacity, typename Level>
inline bool spans(const Level* levels, std::uint32_t count,
    const std::uint64_t* consumed, std::uint32_t consumed_count,
    const std::uint32_t* takes, std::uint32_t takes_count, const ExactIocResult* out) noexcept {
    static_assert(Capacity > 0);
    static_assert(std::is_standard_layout_v<Level> && std::is_trivially_copyable_v<Level>);
    if (count > Capacity || consumed_count > Capacity || takes_count > Capacity ||
        consumed_count < count || takes_count < count) return false;
    Range ranges[4]{};
    if (!range(levels, count, ranges[0]) || !range(consumed, consumed_count, ranges[1]) ||
        !range(takes, takes_count, ranges[2]) || !range(out, out ? 1u : 0u, ranges[3])) return false;
    for (std::uint32_t i = 0; i < 4; ++i)
        for (std::uint32_t j = 0; j < i; ++j)
            if (overlaps(ranges[i], ranges[j])) return false;
    assert(count <= consumed_count); assert(count <= takes_count);
    return true;
}
template <typename Level>
inline bool quantity(const Level& level, std::uint64_t& out) noexcept {
    using Q = std::remove_cv_t<decltype(level.quantity)>;
    static_assert(std::is_same_v<std::remove_cv_t<decltype(level.price_ticks)>, std::int64_t>);
    static_assert(std::is_same_v<Q, std::uint32_t> || std::is_same_v<Q, std::uint64_t> ||
        std::is_same_v<Q, std::int64_t>);
    if constexpr (std::is_signed_v<Q>) { if (level.quantity < 0) return false; }
    if (level.quantity == 0) return false;
    out = static_cast<std::uint64_t>(level.quantity);
    assert(out > 0); assert(static_cast<Q>(out) == level.quantity);
    return true;
}
template <ExactIocSide Side, typename Level>
inline bool ladder(const Level* levels, std::uint32_t count, const std::uint64_t* consumed) noexcept {
    static_assert(Side == ExactIocSide::Buy || Side == ExactIocSide::Sell);
    assert(levels || count == 0); assert(consumed || count == 0);
    for (std::uint32_t i = 0; i < count; ++i) {
        std::uint64_t displayed = 0;
        if (!quantity(levels[i], displayed) || consumed[i] > displayed) return false;
        if (i == 0) continue;
        if constexpr (Side == ExactIocSide::Buy) {
            if (levels[i - 1].price_ticks >= levels[i].price_ticks) return false;
        } else if (levels[i - 1].price_ticks <= levels[i].price_ticks) return false;
    }
    return true;
}
template <ExactIocSide Side>
inline bool stressed(std::int64_t price, std::int64_t adverse, std::int64_t& out) noexcept {
    assert(adverse >= 0);
    static_assert(Side == ExactIocSide::Buy || Side == ExactIocSide::Sell);
    bool ok = false;
    if constexpr (Side == ExactIocSide::Buy) ok = exact_add(price, adverse, out);
    else ok = exact_sub(price, adverse, out);
    assert(!ok || (Side == ExactIocSide::Buy ? out >= price : out <= price));
    return ok;
}
template <ExactIocSide Side, typename Level>
inline ExactIocStatus fill(const Level* levels, std::uint32_t count, const std::uint64_t* consumed,
    std::int64_t limit, std::int64_t adverse, std::uint32_t* takes, ExactIocResult& result) noexcept {
    assert(result.remaining == result.requested); assert(result.requested > 0);
    for (std::uint32_t i = 0; i < count && result.remaining != 0; ++i) {
        const auto available = static_cast<std::uint64_t>(levels[i].quantity) - consumed[i];
        if (available == 0) continue;
        std::int64_t price = 0, value = 0, total = 0;
        if (!stressed<Side>(levels[i].price_ticks, adverse, price)) return ExactIocStatus::Overflow;
        bool limited = false;
        if constexpr (Side == ExactIocSide::Buy) limited = price > limit;
        else limited = price < limit;
        if (limited) {
            result.price_limited = true; break;
        }
        const auto take = available < result.remaining ? static_cast<std::uint32_t>(available) : result.remaining;
        if (!exact_multiply(price, take, value) || !exact_add(result.weighted_ticks, value, total))
            return ExactIocStatus::Overflow;
        takes[i] = take;
        result.remaining -= take; result.filled += take; ++result.levels_touched;
        result.weighted_ticks = total; result.marginal_ticks = price;
    }
    assert(result.filled <= result.requested);
    assert(result.remaining == result.requested - result.filled);
    return ExactIocStatus::Ok;
}
} // namespace exact_ioc_detail

// Plans only: the published result, levels and overlay never change on refusal.
// Takes is caller-owned unpublished scratch and may change even on Overflow.
// All declared spans must be valid, aligned, bounded and pairwise disjoint.
template <ExactIocSide Side, std::uint32_t Capacity, typename Level>
inline ExactIocStatus exact_ioc_plan(const Level* levels, std::uint32_t count,
    const std::uint64_t* consumed, std::uint32_t consumed_count, std::uint32_t requested,
    std::int64_t limit_ticks, std::int64_t adverse_ticks, std::uint32_t* takes,
    std::uint32_t takes_count, ExactIocResult* out) noexcept {
    using namespace exact_ioc_detail;
    if (!out || requested == 0 || adverse_ticks < 0 ||
        !spans<Capacity>(levels, count, consumed, consumed_count, takes, takes_count, out)) return ExactIocStatus::Invalid;
    assert(count <= Capacity); assert(requested > 0);
    for (std::uint32_t i = 0; i < count; ++i) takes[i] = 0;
    if (!ladder<Side>(levels, count, consumed)) return ExactIocStatus::Invalid;
    ExactIocResult result{}; result.requested = requested; result.remaining = requested;
    const auto status = fill<Side>(levels, count, consumed, limit_ticks, adverse_ticks, takes, result);
    if (status != ExactIocStatus::Ok) return status;
    *out = result;
    assert(out->filled + out->remaining == out->requested);
    assert(out->levels_touched <= count);
    return ExactIocStatus::Ok;
}

// Apply once per accepted plan. No order identity/idempotence lives in Bolt.
// Every current availability is validated before the first overlay mutation.
template <std::uint32_t Capacity, typename Level>
inline ExactIocStatus exact_ioc_apply(const Level* levels, std::uint32_t count,
    std::uint64_t* consumed, std::uint32_t consumed_count, const std::uint32_t* takes,
    std::uint32_t takes_count) noexcept {
    using namespace exact_ioc_detail;
    if (!spans<Capacity>(levels, count, consumed, consumed_count, takes, takes_count, nullptr))
        return ExactIocStatus::Invalid;
    assert(count <= Capacity); assert(takes_count >= count);
    for (std::uint32_t i = 0; i < count; ++i) {
        std::uint64_t displayed = 0;
        if (!quantity(levels[i], displayed) || consumed[i] > displayed || takes[i] > displayed - consumed[i])
            return ExactIocStatus::Invalid;
    }
    for (std::uint32_t i = 0; i < count; ++i) {
        assert(consumed[i] <= UINT64_MAX - takes[i]);
        consumed[i] += takes[i];
        assert(consumed[i] >= takes[i]);
    }
    return ExactIocStatus::Ok;
}
} // namespace bolt::fintech
