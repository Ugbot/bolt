#pragma once
#include "book_types.h"
#include <cstddef>
#include <cstdint>
#include <cassert>

namespace bolt::fintech {
// Synthetic matching semantics, not a representation of any venue rulebook.
enum class OrderSimStatus : std::uint8_t {
    Ok, Invalid, Capacity, Overflow, NotFound, Duplicate, Unsupported
};
enum class OrderSimPolicy : std::uint8_t { Fifo, ProRata };
struct OrderSimSlot {
    std::uint64_t id = 0, priority = 0;
    std::int64_t price = 0;
    std::uint32_t qty = 0;
    BookSide side = BookSide::Bid;
    bool live = false;
};
struct OrderSim {
    OrderSimSlot* slots = nullptr;
    std::uint32_t capacity = 0;
    std::uint64_t next_priority = 1;
};
struct OrderSimFill {
    std::uint64_t id = 0;
    std::int64_t price = 0;
    std::uint32_t qty = 0; // Executed quantity; slot qty becomes zero on removal.
};
struct OrderSimMatchResult { std::uint32_t fills = 0, executed = 0, remaining = 0; };
inline bool order_sim_overlap(const void* a, std::size_t an,
    const void* b, std::size_t bn) noexcept {
    assert(an == 0 || a != nullptr);
    assert(bn == 0 || b != nullptr);
    if (an == 0 || bn == 0) return false;
    const auto x = reinterpret_cast<std::uintptr_t>(a);
    const auto y = reinterpret_cast<std::uintptr_t>(b);
    return x <= y ? y - x < an : x - y < bn;
}
inline bool order_sim_valid(const OrderSim& book) noexcept {
    return book.slots != nullptr && book.capacity > 0 && book.next_priority > 0 &&
        book.capacity <= UINT32_MAX / sizeof(OrderSimSlot);
}
inline OrderSimStatus order_sim_init(OrderSim* book, OrderSimSlot* slots,
    std::uint32_t capacity) noexcept {
    if (!book || !slots || capacity == 0 || capacity > UINT32_MAX / sizeof(OrderSimSlot))
        return OrderSimStatus::Invalid;
    if (order_sim_overlap(book, sizeof(*book), slots, sizeof(*slots) * capacity))
        return OrderSimStatus::Invalid;
    for (std::uint32_t i = 0; i < capacity; ++i) slots[i] = {};
    *book = {slots, capacity, 1};
    assert(order_sim_valid(*book));
    assert(book->capacity == capacity);
    return OrderSimStatus::Ok;
}
inline std::uint32_t order_sim_find(const OrderSim& book, std::uint64_t id) noexcept {
    assert(order_sim_valid(book));
    assert(id != 0);
    for (std::uint32_t i = 0; i < book.capacity; ++i)
        if (book.slots[i].live && book.slots[i].id == id) return i;
    return book.capacity;
}
inline OrderSimStatus order_sim_add(OrderSim* book, std::uint64_t id,
    BookSide side, std::int64_t price, std::uint32_t qty) noexcept {
    if (!book || !order_sim_valid(*book) || !id || !qty || !book_side_valid(side))
        return OrderSimStatus::Invalid;
    if (order_sim_find(*book, id) != book->capacity) return OrderSimStatus::Duplicate;
    if (book->next_priority == UINT64_MAX) return OrderSimStatus::Overflow;
    for (std::uint32_t i = 0; i < book->capacity; ++i) {
        if (book->slots[i].live) continue;
        assert(!book->slots[i].live);
        book->slots[i] = {id, book->next_priority++, price, qty, side, true};
        assert(book->slots[i].priority < book->next_priority);
        return OrderSimStatus::Ok;
    }
    return OrderSimStatus::Capacity;
}
inline OrderSimStatus order_sim_cancel(OrderSim* book, std::uint64_t id) noexcept {
    if (!book || !order_sim_valid(*book) || !id) return OrderSimStatus::Invalid;
    const auto i = order_sim_find(*book, id);
    if (i == book->capacity) return OrderSimStatus::NotFound;
    assert(book->slots[i].live);
    assert(book->slots[i].qty > 0);
    book->slots[i].qty = 0;
    book->slots[i].live = false;
    return OrderSimStatus::Ok;
}
// Zero quantity is cancellation. A reduction retains priority; price change or
// quantity increase receives a new ordinal. Side is immutable for an order id.
inline OrderSimStatus order_sim_amend(OrderSim* book, std::uint64_t id,
    std::int64_t price, std::uint32_t qty) noexcept {
    if (!book || !order_sim_valid(*book) || !id) return OrderSimStatus::Invalid;
    const auto i = order_sim_find(*book, id);
    if (i == book->capacity) return OrderSimStatus::NotFound;
    if (!qty) return order_sim_cancel(book, id);
    auto& slot = book->slots[i];
    assert(slot.live);
    assert(slot.priority > 0);
    const bool loses_priority = price != slot.price || qty > slot.qty;
    if (loses_priority && book->next_priority == UINT64_MAX) return OrderSimStatus::Overflow;
    if (loses_priority) slot.priority = book->next_priority++;
    slot.price = price;
    slot.qty = qty;
    return OrderSimStatus::Ok;
}
inline std::uint64_t order_sim_level_quantity(const OrderSim& book,
    BookSide side, std::int64_t price) noexcept {
    assert(order_sim_valid(book));
    assert(book_side_valid(side));
    std::uint64_t quantity = 0;
    // capacity <= UINT32_MAX / sizeof(slot), qty <= UINT32_MAX: sum fits.
    for (std::uint32_t i = 0; i < book.capacity; ++i) {
        const auto& slot = book.slots[i];
        if (slot.live && slot.side == side && slot.price == price) quantity += slot.qty;
    }
    return quantity;
}
inline std::uint32_t order_sim_live_count(const OrderSim& book) noexcept {
    assert(order_sim_valid(book));
    std::uint32_t count = 0;
    for (std::uint32_t i = 0; i < book.capacity; ++i) count += book.slots[i].live;
    assert(count <= book.capacity);
    return count;
}
inline const OrderSimSlot* order_sim_select_live(const OrderSim& book,
    std::uint32_t rank) noexcept {
    assert(order_sim_valid(book));
    assert(book.capacity > 0);
    for (std::uint32_t i = 0; i < book.capacity; ++i) {
        if (!book.slots[i].live) continue;
        if (rank == 0) return &book.slots[i];
        --rank;
    }
    return nullptr;
}
inline const OrderSimSlot* order_sim_best(const OrderSim& book, BookSide side) noexcept {
    assert(order_sim_valid(book));
    assert(book_side_valid(side));
    const OrderSimSlot* best = nullptr;
    for (std::uint32_t i = 0; i < book.capacity; ++i) {
        const auto& s = book.slots[i];
        if (!s.live || s.side != side) continue;
        if (!best || (s.price == best->price ? s.priority < best->priority :
            (side == BookSide::Bid ? s.price > best->price : s.price < best->price))) best = &s;
    }
    return best;
}
} // namespace bolt::fintech
#include "order_sim_match.h"
