#pragma once
#include "bolt/kernels/fintech/book_detail.h"

namespace bolt::fintech {
// Ranked MBP is distinct from price-keyed L2. Rank is 1-based, bid/ask
// independently contiguous; Insert shifts down, Delete shifts up. Capacity is
// a refusal boundary, never silent worst-level eviction. Rank determines order;
// this primitive does not infer exchange price-order or queue-priority rules.
// One writer, no concurrent readers. Depth copies can be handed across threads.
using BookMBP = book_detail::State<BookLevel>;
enum class BookMBPAction : std::uint8_t { Insert, Change, Delete, Clear };
struct BookMBPUpdate {
    BookSide side = BookSide::Bid;
    BookMBPAction action = BookMBPAction::Insert;
    std::uint32_t rank = 0;
    BookLevel level{};
};
inline BookStatus book_mbp_storage_size(BookLimits limits, std::uint64_t& bytes) noexcept {
    return book_detail::storage_size<BookLevel>(limits, bytes);
}
inline BookStatus book_mbp_init(BookMBP* s, const BookDefinition& definition, BookLimits limits,
    void* storage, std::uint64_t bytes) noexcept {
    return book_detail::init(s, definition, limits, storage, bytes);
}
inline BookStatus book_mbp_begin(BookMBP& s, BookCut cut, BookEventKind kind) noexcept {
    return book_detail::begin(s, cut, kind);
}
namespace book_detail {
inline bool mbp_insert(BookMBP& s, unsigned side, std::uint32_t index, BookLevel level) noexcept {
    assert(side < 2);
    assert(index <= s.working.read.levels[side]);
    const auto base = side * s.limits.levels_per_side;
    for (auto i = s.working.read.levels[side]; i > index; --i) {
        auto* dest = touch(s, base + i); if (!dest) return false;
        *dest = s.bank[s.staged_bank][base + i - 1];
    }
    auto* dest = touch(s, base + index); if (!dest) return false;
    *dest = level; ++s.working.read.levels[side];
    return true;
}
inline bool mbp_erase(BookMBP& s, unsigned side, std::uint32_t index) noexcept {
    assert(side < 2);
    assert(index < s.working.read.levels[side]);
    const auto base = side * s.limits.levels_per_side;
    for (auto i = index; i + 1 < s.working.read.levels[side]; ++i) {
        auto* dest = touch(s, base + i); if (!dest) return false;
        *dest = s.bank[s.staged_bank][base + i + 1];
    }
    auto* last = touch(s, base + s.working.read.levels[side] - 1); if (!last) return false;
    *last = {}; --s.working.read.levels[side];
    return true;
}
inline BookStatus mbp_clear(BookMBP& s, unsigned side) noexcept {
    assert(side < 2);
    assert(s.event_open);
    const auto base = side * s.limits.levels_per_side;
    for (std::uint32_t i = 0; i < s.working.read.levels[side]; ++i) {
        auto* dest = touch(s, base + i);
        if (!dest) return fail(s, BookStatus::Capacity);
        *dest = {};
    }
    s.working.read.levels[side] = 0; s.working.read.quantity[side] = 0;
    return BookStatus::Ok;
}
} // namespace book_detail
inline BookStatus book_mbp_stage(BookMBP& s, BookMBPUpdate update) noexcept {
    const auto status = book_detail::admit_operation(s);
    if (status != BookStatus::Ok) return status;
    if (!book_side_valid(update.side)) return book_detail::fail(s, BookStatus::Invalid);
    if (update.action != BookMBPAction::Insert && update.action != BookMBPAction::Change &&
        update.action != BookMBPAction::Delete && update.action != BookMBPAction::Clear)
        return book_detail::fail(s, BookStatus::Invalid);
    const unsigned side = unsigned(update.side);
    const auto& level = update.level;
    const bool empty = !level.price_ticks && !level.quantity && !level.order_count && !level.order_count_valid;
    if (update.action == BookMBPAction::Clear) {
        if (update.rank != 0 || !empty) return book_detail::fail(s, BookStatus::Invalid);
        return book_detail::mbp_clear(s, side);
    }
    const auto count = s.working.read.levels[side];
    if (update.rank == 0 || update.rank > count + 1) return book_detail::fail(s, BookStatus::Invalid);
    const auto index = update.rank - 1, offset = side * s.limits.levels_per_side + index;
    const bool insert = update.action == BookMBPAction::Insert;
    if (!insert && index == count) return book_detail::fail(s, BookStatus::NotFound);
    const auto old = insert ? 0 : s.bank[s.staged_bank][offset].quantity;
    if (update.action == BookMBPAction::Delete) {
        if (!empty) return book_detail::fail(s, BookStatus::Invalid);
        if (!book_detail::mbp_erase(s, side, index)) return book_detail::fail(s, BookStatus::Capacity);
        s.working.read.quantity[side] -= old;
        return BookStatus::Ok;
    }
    if ((!insert && update.action != BookMBPAction::Change) || !book_level_valid(level))
        return book_detail::fail(s, BookStatus::Invalid);
    std::uint64_t total = 0;
    if (!book_total_replace(s.working.read.quantity[side], old, level.quantity, total))
        return book_detail::fail(s, BookStatus::Overflow);
    if (insert) {
        if (count == s.limits.levels_per_side || !book_detail::mbp_insert(s, side, index, level))
            return book_detail::fail(s, BookStatus::Capacity);
    } else {
        auto* dest = book_detail::touch(s, offset);
        if (!dest) return book_detail::fail(s, BookStatus::Capacity);
        *dest = level;
    }
    s.working.read.quantity[side] = total;
    assert(s.working.read.levels[side] <= s.limits.levels_per_side);
    assert(s.touched <= s.limits.journal_records);
    return BookStatus::Ok;
}
inline BookStatus book_mbp_commit(BookMBP& s) noexcept { return book_detail::commit(s); }
inline void book_mbp_abort(BookMBP& s) noexcept { book_detail::abort(s); }
inline BookStatus book_mbp_invalidate(BookMBP& s) noexcept {
    assert(s.touched <= s.limits.journal_records);
    assert(s.operations <= s.limits.event_operations);
    if (!s.initialized) return BookStatus::Invalid;
    if (s.event_open) return BookStatus::Busy;
    s.committed.read.valid = false;
    return BookStatus::Ok;
}
inline BookReadInfo book_mbp_info(const BookMBP& s) noexcept { return s.committed.read; }
inline BookStatus book_mbp_lookup(const BookMBP& s, BookSide side, std::uint32_t rank,
    BookLevel& out) noexcept {
    assert(s.active_bank < 2);
    assert(s.touched <= s.limits.journal_records);
    if (!book_detail::output_valid(s, &out, sizeof(out))) return BookStatus::Invalid;
    out = {};
    if (!s.initialized || !s.committed.read.valid || !book_side_valid(side) || rank == 0)
        return BookStatus::Invalid;
    if (rank > s.committed.read.levels[unsigned(side)]) return BookStatus::NotFound;
    out = book_detail::committed_slot(s, unsigned(side) * s.limits.levels_per_side + rank - 1);
    return BookStatus::Ok;
}
inline BookStatus book_mbp_depth(const BookMBP& s, BookSide side, std::uint32_t offset,
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
    const auto side_id = unsigned(side), base = side_id * s.limits.levels_per_side;
    for (std::uint32_t i = offset; i < s.committed.read.levels[side_id] && count < capacity; ++i)
        out[count++] = book_detail::committed_slot(s, base + i);
    assert(count <= capacity);
    return BookStatus::Ok;
}
inline BookStatus book_mbp_top(const BookMBP& s, BookSide side, BookLevel& out) noexcept {
    return book_mbp_lookup(s, side, 1, out);
}
} // namespace bolt::fintech
