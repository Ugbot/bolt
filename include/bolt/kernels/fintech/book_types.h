#pragma once
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace bolt::fintech {
// Representability bounds only. Actual depth/event/journal budgets are admitted
// by the caller; this primitive contains no workload policy depth ceiling.
inline constexpr std::uint32_t kBookMaxLevels = (UINT32_MAX - 1u) / 2u;
inline constexpr std::uint32_t kBookMaxEventOperations = UINT32_MAX;
inline constexpr std::uint32_t kBookNull = UINT32_MAX;
enum class BookStatus : std::uint8_t { Ok, Invalid, Capacity, Overflow, NotFound, Busy };
enum class BookSide : std::uint8_t { Bid, Ask };
enum class BookEventKind : std::uint8_t { Delta, Replacement };
enum class BookLiquidity : std::uint8_t { Explicit, Implied };
// Exact values are signed price_ticks / price_scale and quantity / quantity_scale.
// Scales identify the admitted representation; kernels never rescale or round.
struct BookDefinition {
    std::uint64_t book_id = 0, instrument_id = 0, definition_id = 0, tick_rule_id = 0;
    std::uint64_t price_scale = 0, quantity_scale = 0;
    BookLiquidity liquidity = BookLiquidity::Explicit;
};
struct BookLevel {
    std::int64_t price_ticks = 0;
    std::uint64_t quantity = 0, order_count = 0;
    std::uint8_t order_count_valid = 0;
};
struct BookCut { std::uint64_t epoch = 0, generation = 0, position = 0; };
struct BookLimits {
    std::uint32_t levels_per_side = 0, event_operations = 0, journal_records = 0;
};
struct BookReadInfo {
    BookCut cut{};
    std::uint64_t revision = 0, quantity[2]{};
    std::uint32_t levels[2]{};
    bool valid = false; // Kernel baseline only; never trading/source authority.
};
inline bool book_side_valid(BookSide side) noexcept {
    return side == BookSide::Bid || side == BookSide::Ask;
}
inline bool book_level_valid(const BookLevel& level) noexcept {
    return level.quantity != 0 && level.order_count_valid <= 1 &&
        (level.order_count_valid != 0 || level.order_count == 0);
}
inline bool book_definition_valid(const BookDefinition& def) noexcept {
    return def.book_id != 0 && def.instrument_id != 0 && def.definition_id != 0 &&
        def.tick_rule_id != 0 && def.price_scale != 0 && def.quantity_scale != 0 &&
        (def.liquidity == BookLiquidity::Explicit || def.liquidity == BookLiquidity::Implied);
}
inline bool book_limits_valid(const BookLimits& limits) noexcept {
    return limits.levels_per_side > 0 && limits.levels_per_side <= kBookMaxLevels &&
        limits.event_operations > 0 && limits.event_operations <= kBookMaxEventOperations &&
        limits.journal_records > 0 && limits.journal_records <= 2 * limits.levels_per_side;
}
// Epoch is opaque. Rebaseline/reset requires caller admission and a strictly
// newer execution generation; epoch magnitude never revives stale work.
// No sequence gap/replay authority is inferred from these normalized tokens.
inline bool book_cut_after(BookCut next, BookCut old, BookEventKind kind) noexcept {
    if (next.epoch == 0 || next.generation == 0) return false;
    if (old.epoch == 0 && old.generation == 0) return kind == BookEventKind::Replacement;
    if (next.epoch == old.epoch && next.generation == old.generation)
        return next.position > old.position;
    if (kind != BookEventKind::Replacement) return false;
    return next.generation > old.generation;
}
inline bool book_total_replace(std::uint64_t total, std::uint64_t old,
    std::uint64_t next, std::uint64_t& out) noexcept {
    assert(total >= old);
    assert(old <= total);
    const auto base = total - old;
    if (next > UINT64_MAX - base) return false;
    out = base + next;
    return true;
}
} // namespace bolt::fintech
