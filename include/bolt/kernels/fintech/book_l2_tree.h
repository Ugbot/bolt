#pragma once
#include "bolt/kernels/fintech/book_detail.h"

namespace bolt::fintech::book_detail {
struct L2Node {
    BookLevel level{};
    std::uint32_t left = kBookNull, right = kBookNull, parent = kBookNull;
    std::uint32_t prev = kBookNull, next = kBookNull, free_next = kBookNull, height = 1;
};
using L2State = State<L2Node>;
inline void l2_empty_bank(L2State& state) noexcept {
    assert(state.initialized);
    assert(state.staged_bank < 2);
    const auto count = state.limits.levels_per_side * 2;
    for (std::uint32_t i = 0; i < count; ++i) {
        state.bank[state.staged_bank][i] = {};
        state.bank[state.staged_bank][i].free_next = i + 1 == count ? kBookNull : i + 1;
    }
    state.working.free_head = 0;
}
inline std::uint32_t height(const L2State& s, std::uint32_t id) noexcept {
    return id == kBookNull ? 0 : s.bank[s.staged_bank][id].height;
}
inline std::int32_t balance(const L2State& s, std::uint32_t id) noexcept {
    const auto& n = s.bank[s.staged_bank][id];
    return std::int32_t(height(s, n.left)) - std::int32_t(height(s, n.right));
}
inline bool update_height(L2State& s, std::uint32_t id) noexcept {
    assert(id < s.limits.levels_per_side * 2);
    assert(s.event_open);
    const auto& old = s.bank[s.staged_bank][id];
    const auto l = height(s, old.left), r = height(s, old.right);
    const auto h = 1 + (l > r ? l : r);
    if (old.height == h) return true;
    auto* n = touch(s, id);
    if (!n) return false;
    n->height = h;
    return true;
}
inline bool link_parent(L2State& s, unsigned side, std::uint32_t parent,
    std::uint32_t old, std::uint32_t next) noexcept {
    assert(side < 2);
    assert(s.event_open);
    if (parent == kBookNull) s.working.root[side] = next;
    else {
        auto* p = touch(s, parent);
        if (!p) return false;
        if (p->left == old) p->left = next;
        else { assert(p->right == old); p->right = next; }
    }
    if (next != kBookNull) {
        auto* n = touch(s, next);
        if (!n) return false;
        n->parent = parent;
    }
    return true;
}
// Direction is compile-time: left rotation when Left=true, right otherwise.
template<bool Left> inline bool rotate(L2State& s, unsigned side,
    std::uint32_t x, std::uint32_t& out) noexcept {
    assert(s.event_open);
    assert(x < s.limits.levels_per_side * 2);
    const auto old = s.bank[s.staged_bank][x];
    const auto y = Left ? old.right : old.left;
    assert(y != kBookNull);
    const auto middle = Left ? s.bank[s.staged_bank][y].left : s.bank[s.staged_bank][y].right;
    if (!link_parent(s, side, old.parent, x, y)) return false;
    auto* nx = touch(s, x); auto* ny = touch(s, y);
    if (!nx || !ny) return false;
    if constexpr (Left) { nx->right = middle; ny->left = x; }
    else { nx->left = middle; ny->right = x; }
    nx->parent = y;
    if (middle != kBookNull) {
        auto* nm = touch(s, middle);
        if (!nm) return false;
        nm->parent = x;
    }
    if (!update_height(s, x) || !update_height(s, y)) return false;
    out = y;
    return true;
}
inline bool rebalance(L2State& s, unsigned side, std::uint32_t id) noexcept {
    assert(side < 2);
    assert(s.event_open);
    // AVL height at the representable u32 slot bound is below 64.
    for (unsigned step = 0; step < 64 && id != kBookNull; ++step) {
        if (!update_height(s, id)) return false;
        const auto delta = balance(s, id);
        std::uint32_t root = id, ignored = 0;
        if (delta > 1) {
            const auto left = s.bank[s.staged_bank][id].left;
            if (balance(s, left) < 0 && !rotate<true>(s, side, left, ignored)) return false;
            if (!rotate<false>(s, side, id, root)) return false;
        } else if (delta < -1) {
            const auto right = s.bank[s.staged_bank][id].right;
            if (balance(s, right) > 0 && !rotate<false>(s, side, right, ignored)) return false;
            if (!rotate<true>(s, side, id, root)) return false;
        }
        id = s.bank[s.staged_bank][root].parent;
    }
    assert(id == kBookNull);
    return id == kBookNull;
}
inline std::uint32_t find(const L2State& s, unsigned side, std::int64_t price,
    bool committed) noexcept {
    assert(side < 2);
    assert(s.initialized);
    auto id = (committed ? s.committed : s.working).root[side];
    for (unsigned step = 0; step < 64 && id != kBookNull; ++step) {
        const auto& n = committed ? committed_slot(s, id) : s.bank[s.staged_bank][id];
        if (price == n.level.price_ticks) return id;
        id = price < n.level.price_ticks ? n.left : n.right;
    }
    return kBookNull;
}
inline bool insert(L2State& s, unsigned side, BookLevel level) noexcept {
    assert(side < 2);
    assert(s.working.free_head != kBookNull);
    std::uint32_t parent = kBookNull, prev = kBookNull, next = kBookNull;
    auto id = s.working.root[side];
    for (unsigned step = 0; step < 64 && id != kBookNull; ++step) {
        const auto& n = s.bank[s.staged_bank][id]; parent = id;
        if (level.price_ticks < n.level.price_ticks) { next = id; id = n.left; }
        else { prev = id; id = n.right; }
    }
    assert(id == kBookNull);
    id = s.working.free_head;
    auto* n = touch(s, id);
    if (!n) return false;
    s.working.free_head = n->free_next;
    *n = {}; n->level = level; n->parent = parent; n->prev = prev; n->next = next;
    if (prev != kBookNull) { auto* p = touch(s, prev); if (!p) return false; p->next = id; }
    if (next != kBookNull) { auto* p = touch(s, next); if (!p) return false; p->prev = id; }
    if (parent == kBookNull) s.working.root[side] = id;
    else {
        auto* p = touch(s, parent); if (!p) return false;
        if (level.price_ticks < p->level.price_ticks) p->left = id; else p->right = id;
    }
    if (s.working.best[side] == kBookNull || (side == 0 ? next == kBookNull : prev == kBookNull))
        s.working.best[side] = id;
    ++s.working.read.levels[side];
    return rebalance(s, side, parent);
}
inline bool erase(L2State& s, unsigned side, std::uint32_t id) noexcept {
    assert(side < 2);
    assert(s.working.read.levels[side] > 0);
    const auto original = s.bank[s.staged_bank][id];
    if (original.left != kBookNull && original.right != kBookNull) {
        const auto successor = original.next;
        auto* n = touch(s, id); if (!n) return false;
        n->level = s.bank[s.staged_bank][successor].level;
        id = successor;
    }
    const auto old = s.bank[s.staged_bank][id];
    if (old.prev != kBookNull) { auto* n = touch(s, old.prev); if (!n) return false; n->next = old.next; }
    if (old.next != kBookNull) { auto* n = touch(s, old.next); if (!n) return false; n->prev = old.prev; }
    if (s.working.best[side] == id) s.working.best[side] = side == 0 ? old.prev : old.next;
    const auto child = old.left != kBookNull ? old.left : old.right;
    if (!link_parent(s, side, old.parent, id, child)) return false;
    auto* n = touch(s, id); if (!n) return false;
    n->free_next = s.working.free_head; s.working.free_head = id;
    --s.working.read.levels[side];
    return rebalance(s, side, old.parent);
}
} // namespace bolt::fintech::book_detail
