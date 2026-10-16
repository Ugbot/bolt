#pragma once
#include "order_sim.h"

namespace bolt::fintech {
namespace order_sim_detail {
inline bool crosses(const OrderSimSlot& s, BookSide side, std::int64_t limit) noexcept {
    assert(book_side_valid(side));
    assert(book_side_valid(s.side));
    return s.live && s.side != side && (side == BookSide::Bid ? s.price <= limit : s.price >= limit);
}
inline bool before(const OrderSimSlot& a, const OrderSimSlot& b, BookSide side) noexcept {
    assert(book_side_valid(side));
    assert(a.priority != b.priority);
    return a.price == b.price ? a.priority < b.priority :
        (side == BookSide::Bid ? a.price < b.price : a.price > b.price);
}
inline std::uint32_t first(const OrderSim& b, BookSide side, std::int64_t limit,
    const std::uint64_t* plan, bool planned) noexcept {
    assert(order_sim_valid(b));
    assert(plan != nullptr);
    auto best = b.capacity;
    for (std::uint32_t i = 0; i < b.capacity; ++i) {
        const auto& s = b.slots[i];
        if (!crosses(s, side, limit) || (planned ? plan[i] == 0 : plan[i] >= s.qty)) continue;
        if (best == b.capacity || before(s, b.slots[best], side)) best = i;
    }
    return best;
}
inline bool buffers_valid(OrderSim* b, OrderSimFill* fills, std::uint32_t fill_cap,
    std::uint64_t* scratch, OrderSimMatchResult* result) noexcept {
    assert(b != nullptr);
    assert(order_sim_valid(*b));
    if (!scratch || !result || (fill_cap && !fills) ||
        fill_cap > UINT32_MAX / sizeof(OrderSimFill)) return false;
    const void* ptrs[] = {b, b->slots, fills, scratch, result};
    const std::size_t sizes[] = {sizeof(*b), sizeof(*b->slots) * b->capacity,
        sizeof(*fills) * fill_cap, sizeof(*scratch) * b->capacity, sizeof(*result)};
    for (std::uint32_t i = 0; i < 5; ++i)
        for (std::uint32_t j = i + 1; j < 5; ++j)
            if (order_sim_overlap(ptrs[i], sizes[i], ptrs[j], sizes[j])) return false;
    return true;
}
inline void fifo_level(const OrderSim& b, std::int64_t price, BookSide side,
    std::uint32_t& remaining, std::uint64_t* plan) noexcept {
    assert(order_sim_valid(b));
    assert(plan != nullptr);
    for (std::uint32_t step = 0; step < b.capacity && remaining; ++step) {
        const auto i = first(b, side, price, plan, false);
        if (i == b.capacity || b.slots[i].price != price) break;
        const auto available = b.slots[i].qty - static_cast<std::uint32_t>(plan[i]);
        const auto take = available < remaining ? available : remaining;
        plan[i] += take;
        remaining -= take;
    }
}
inline OrderSimStatus pro_rata_level(const OrderSim& b, std::int64_t price,
    BookSide side, std::uint32_t minimum, std::uint32_t& remaining,
    std::uint64_t* plan) noexcept {
    assert(order_sim_valid(b));
    assert(plan != nullptr);
    std::uint64_t total = 0;
    for (std::uint32_t i = 0; i < b.capacity; ++i) {
        const auto& s = b.slots[i];
        if (!s.live || s.side == side || s.price != price) continue;
        if (UINT64_MAX - total < s.qty) return OrderSimStatus::Overflow;
        total += s.qty;
    }
    assert(total > 0);
    const auto target = total < remaining ? static_cast<std::uint32_t>(total) : remaining;
    std::uint32_t allocated = 0;
    for (std::uint32_t i = 0; i < b.capacity; ++i) {
        const auto& s = b.slots[i];
        if (!s.live || s.side == side || s.price != price) continue;
        // Both factors are uint32: product is representable in uint64.
        const auto share = static_cast<std::uint32_t>((std::uint64_t{target} * s.qty) / total);
        plan[i] = share >= minimum ? share : 0;
        allocated += static_cast<std::uint32_t>(plan[i]);
    }
    remaining -= allocated;
    // Minimum applies to the proportional pass only. Residual uses FIFO,
    // including orders below minimum; no venue-specific rounding is inferred.
    fifo_level(b, price, side, remaining, plan);
    return OrderSimStatus::Ok;
}
} // namespace order_sim_detail
// Matching is IOC: any unexecuted quantity is returned, never rested. All
// mutation follows a complete plan and output-capacity validation. Scratch may
// change on failure; book, fills and result do not. Scratch must have capacity
// >= book.capacity. All supplied writable buffers must be pairwise disjoint.
template<OrderSimPolicy Policy>
inline OrderSimStatus order_sim_match(OrderSim* book, BookSide side,
    std::int64_t limit_price, std::uint32_t qty, std::uint32_t minimum,
    OrderSimFill* fills, std::uint32_t fill_capacity, std::uint64_t* scratch,
    std::uint32_t scratch_capacity, OrderSimMatchResult* result) noexcept {
    if constexpr (Policy != OrderSimPolicy::Fifo && Policy != OrderSimPolicy::ProRata)
        return OrderSimStatus::Unsupported;
    if (!book || !order_sim_valid(*book) || !book_side_valid(side) || !qty ||
        (Policy == OrderSimPolicy::ProRata && minimum == 0)) return OrderSimStatus::Invalid;
    if (scratch_capacity < book->capacity) return OrderSimStatus::Capacity;
    if (!order_sim_detail::buffers_valid(book, fills, fill_capacity, scratch, result))
        return OrderSimStatus::Invalid;
    assert(scratch_capacity >= book->capacity);
    assert(result != nullptr);
    for (std::uint32_t i = 0; i < book->capacity; ++i) scratch[i] = 0;
    auto remaining = qty;
    for (std::uint32_t level = 0; level < book->capacity && remaining; ++level) {
        const auto i = order_sim_detail::first(*book, side, limit_price, scratch, false);
        if (i == book->capacity) break;
        const auto price = book->slots[i].price;
        if constexpr (Policy == OrderSimPolicy::Fifo)
            order_sim_detail::fifo_level(*book, price, side, remaining, scratch);
        else {
            const auto status = order_sim_detail::pro_rata_level(*book, price, side, minimum, remaining, scratch);
            if (status != OrderSimStatus::Ok) return status;
        }
    }
    std::uint32_t count = 0;
    for (std::uint32_t i = 0; i < book->capacity; ++i) count += scratch[i] != 0;
    if (count > fill_capacity) return OrderSimStatus::Capacity;
    for (std::uint32_t out = 0; out < count; ++out) {
        const auto i = order_sim_detail::first(*book, side, limit_price, scratch, true);
        assert(i < book->capacity);
        auto& s = book->slots[i];
        const auto take = static_cast<std::uint32_t>(scratch[i]);
        fills[out] = {s.id, s.price, take};
        s.qty -= take;
        s.live = s.qty != 0;
        scratch[i] = 0;
    }
    *result = {count, qty - remaining, remaining};
    return OrderSimStatus::Ok;
}
} // namespace bolt::fintech
