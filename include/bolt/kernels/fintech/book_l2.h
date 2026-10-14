#pragma once
#include "bolt/kernels/fintech/book_l2_tree.h"

namespace bolt::fintech {
// One writer, no concurrent readers. All calls are bounded and allocation-free.
// Reads copy committed values (including while an event is staged). Copies may
// be handed to another thread; the mutable state/storage may not be shared.
using BookL2 = book_detail::L2State;
enum class BookL2Action : std::uint8_t { Upsert, Delete, Clear };
struct BookL2Update {
    BookSide side = BookSide::Bid;
    BookL2Action action = BookL2Action::Upsert;
    BookLevel level{};
};
inline BookStatus book_l2_storage_size(BookLimits limits, std::uint64_t& bytes) noexcept {
    return book_detail::storage_size<book_detail::L2Node>(limits, bytes);
}
inline BookStatus book_l2_init(BookL2* s, const BookDefinition& definition, BookLimits limits,
    void* storage, std::uint64_t bytes) noexcept {
    const auto status = book_detail::init(s, definition, limits, storage, bytes);
    if (status != BookStatus::Ok) return status;
    book_detail::l2_empty_bank(*s); s->committed = s->working;
    assert(s->committed.read.levels[0] == 0);
    assert(s->committed.read.levels[1] == 0);
    return BookStatus::Ok;
}
// Fresh state accepts an explicitly authorized replacement at ordinal zero.
// Completion boundaries are supplied by the caller: chunks only call stage.
inline BookStatus book_l2_begin(BookL2& s, BookCut cut, BookEventKind kind) noexcept {
    const auto status = book_detail::begin(s, cut, kind);
    if (status == BookStatus::Ok && kind == BookEventKind::Replacement) book_detail::l2_empty_bank(s);
    return status;
}
inline BookStatus book_l2_stage(BookL2& s, BookL2Update update) noexcept {
    auto status = book_detail::admit_operation(s);
    if (status != BookStatus::Ok) return status;
    if (!book_side_valid(update.side)) return book_detail::fail(s, BookStatus::Invalid);
    const unsigned side = unsigned(update.side);
    const auto& level = update.level;
    if (update.action == BookL2Action::Clear) {
        if (level.price_ticks || level.quantity || level.order_count || level.order_count_valid)
            return book_detail::fail(s, BookStatus::Invalid);
        const auto count = s.working.read.levels[side];
        for (std::uint32_t i = 0; i < count; ++i)
            if (!book_detail::erase(s, side, s.working.best[side]))
                return book_detail::fail(s, BookStatus::Capacity);
        s.working.read.quantity[side] = 0;
        return BookStatus::Ok;
    }
    const auto id = book_detail::find(s, side, level.price_ticks, false);
    const auto old = id == kBookNull ? 0 : s.bank[s.staged_bank][id].level.quantity;
    if (update.action == BookL2Action::Delete) {
        if (level.quantity || level.order_count || level.order_count_valid)
            return book_detail::fail(s, BookStatus::Invalid);
        if (id == kBookNull) return book_detail::fail(s, BookStatus::NotFound);
        if (!book_detail::erase(s, side, id)) return book_detail::fail(s, BookStatus::Capacity);
        s.working.read.quantity[side] -= old;
        return BookStatus::Ok;
    }
    if (update.action != BookL2Action::Upsert || !book_level_valid(level))
        return book_detail::fail(s, BookStatus::Invalid);
    std::uint64_t total = 0;
    if (!book_total_replace(s.working.read.quantity[side], old, level.quantity, total))
        return book_detail::fail(s, BookStatus::Overflow);
    if (id != kBookNull) {
        auto* n = book_detail::touch(s, id);
        if (!n) return book_detail::fail(s, BookStatus::Capacity);
        n->level = level;
    } else {
        if (s.working.read.levels[side] == s.limits.levels_per_side ||
            s.working.free_head == kBookNull || !book_detail::insert(s, side, level))
            return book_detail::fail(s, BookStatus::Capacity);
    }
    s.working.read.quantity[side] = total;
    assert(s.touched <= s.limits.journal_records);
    assert(s.working.read.levels[side] <= s.limits.levels_per_side);
    return BookStatus::Ok;
}
inline BookStatus book_l2_commit(BookL2& s) noexcept { return book_detail::commit(s); }
inline void book_l2_abort(BookL2& s) noexcept { book_detail::abort(s); }
inline BookStatus book_l2_invalidate(BookL2& s) noexcept {
    assert(s.touched <= s.limits.journal_records);
    assert(s.operations <= s.limits.event_operations);
    if (!s.initialized) return BookStatus::Invalid;
    if (s.event_open) return BookStatus::Busy;
    s.committed.read.valid = false;
    return BookStatus::Ok;
}
inline BookReadInfo book_l2_info(const BookL2& s) noexcept { return s.committed.read; }
inline BookStatus book_l2_lookup(const BookL2& s, BookSide side, std::int64_t price,
    BookLevel& out) noexcept {
    assert(s.touched <= s.limits.journal_records);
    assert(s.active_bank < 2);
    if (!book_detail::output_valid(s, &out, sizeof(out))) return BookStatus::Invalid;
    out = {};
    if (!s.initialized || !s.committed.read.valid || !book_side_valid(side)) return BookStatus::Invalid;
    const auto id = book_detail::find(s, unsigned(side), price, true);
    if (id == kBookNull) return BookStatus::NotFound;
    out = book_detail::committed_slot(s, id).level;
    return BookStatus::Ok;
}
// Traversal costs O(offset + returned levels); best/top is O(1).
inline BookStatus book_l2_depth(const BookL2& s, BookSide side, std::uint32_t offset,
    BookLevel* out, std::uint32_t capacity, std::uint32_t& count) noexcept {
    assert(s.active_bank < 2);
    assert(s.touched <= s.limits.journal_records);
    if (!book_detail::output_valid(s, &count, sizeof(count))) return BookStatus::Invalid;
    count = 0;
    if (!s.initialized || !s.committed.read.valid || !book_side_valid(side) ||
        capacity > s.limits.levels_per_side || offset > s.limits.levels_per_side) return BookStatus::Invalid;
    if (!book_detail::output_valid(s, out, std::uint64_t(capacity) * sizeof(BookLevel)) ||
        book_detail::overlap(&count, sizeof(count), out, std::uint64_t(capacity) * sizeof(BookLevel)))
        return BookStatus::Invalid;
    auto id = s.committed.best[unsigned(side)];
    for (std::uint32_t i = 0; i < s.limits.levels_per_side && id != kBookNull; ++i) {
        const auto& node = book_detail::committed_slot(s, id);
        if (i >= offset) { if (count == capacity) break; out[count++] = node.level; }
        id = side == BookSide::Bid ? node.prev : node.next;
    }
    assert(count <= capacity);
    return BookStatus::Ok;
}
inline BookStatus book_l2_top(const BookL2& s, BookSide side, BookLevel& out) noexcept {
    std::uint32_t count = 0;
    if (!book_detail::output_valid(s, &out, sizeof(out))) return BookStatus::Invalid;
    out = {};
    const auto status = book_l2_depth(s, side, 0, &out, 1, count);
    return status == BookStatus::Ok && count == 0 ? BookStatus::NotFound : status;
}
} // namespace bolt::fintech
