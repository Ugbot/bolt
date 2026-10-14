#pragma once
#include "bolt/kernels/fintech/book_types.h"
#include <cstring>
#include <new>

namespace bolt::fintech::book_detail {
struct Meta {
    BookReadInfo read{};
    std::uint32_t root[2]{kBookNull, kBookNull}, best[2]{kBookNull, kBookNull};
    std::uint32_t free_head = kBookNull;
};
template<class Slot> struct State {
    BookDefinition definition{};
    BookLimits limits{};
    Slot* bank[2]{};
    Slot* before = nullptr;
    std::uint32_t* directory = nullptr;
    std::uint32_t* dirty = nullptr;
    Meta committed{}, working{};
    std::uint32_t active_bank = 0, staged_bank = 0, touched = 0, operations = 0;
    BookStatus failure = BookStatus::Ok;
    BookEventKind kind = BookEventKind::Delta;
    bool initialized = false, event_open = false;
    std::uintptr_t storage_base = 0;
    std::uint64_t storage_bytes = 0;
};
template<class Slot> inline BookStatus storage_size(BookLimits limits, std::uint64_t& bytes) noexcept {
    bytes = 0;
    if (!book_limits_valid(limits)) return BookStatus::Invalid;
    const auto slots = std::uint64_t(limits.levels_per_side) * 2;
    bytes = slots * (2 * sizeof(Slot) + sizeof(std::uint32_t)) +
        std::uint64_t(limits.journal_records) * (sizeof(Slot) + sizeof(std::uint32_t));
    // Each region aligned independently; sizeof(Slot) is a multiple of alignment.
    bytes += 4 * alignof(Slot);
    assert(bytes > 0);
    assert(bytes < UINT64_MAX / 2);
    return BookStatus::Ok;
}
inline std::uintptr_t align_up(std::uintptr_t value, std::uintptr_t alignment) noexcept {
    return (value + alignment - 1) & ~(alignment - 1);
}
inline bool overlap(const void* a, std::uint64_t a_bytes, const void* b, std::uint64_t b_bytes) noexcept {
    if (a_bytes == 0 || b_bytes == 0) return false;
    const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
    if (a_bytes > UINTPTR_MAX - x || b_bytes > UINTPTR_MAX - y) return true;
    return x < y + b_bytes && y < x + a_bytes;
}
template<class Slot> inline BookStatus init(State<Slot>* state, const BookDefinition& def,
    BookLimits limits, void* storage, std::uint64_t bytes) noexcept {
    assert(state != nullptr);
    assert(sizeof(Slot) % alignof(Slot) == 0);
    if (state == nullptr || storage == nullptr || !book_definition_valid(def)) return BookStatus::Invalid;
    std::uint64_t needed = 0;
    const auto status = storage_size<Slot>(limits, needed);
    if (status != BookStatus::Ok) return status;
    const auto base = reinterpret_cast<std::uintptr_t>(storage);
    if (bytes < needed || bytes > UINTPTR_MAX - base) return BookStatus::Capacity;
    const auto sbase = reinterpret_cast<std::uintptr_t>(state);
    if (base < sbase + sizeof(*state) && sbase < base + bytes) return BookStatus::Invalid;
    const auto definition = def;
    *state = {};
    state->definition = definition; state->limits = limits;
    state->storage_base = base; state->storage_bytes = bytes;
    const auto count = limits.levels_per_side * 2;
    auto cursor = align_up(base, alignof(Slot));
    for (unsigned b = 0; b < 2; ++b) {
        state->bank[b] = reinterpret_cast<Slot*>(cursor);
        for (std::uint32_t i = 0; i < count; ++i) ::new (state->bank[b] + i) Slot{};
        cursor += sizeof(Slot) * std::uint64_t(count);
    }
    state->before = reinterpret_cast<Slot*>(cursor);
    for (std::uint32_t i = 0; i < limits.journal_records; ++i) ::new (state->before + i) Slot{};
    cursor += sizeof(Slot) * std::uint64_t(limits.journal_records);
    cursor = align_up(cursor, alignof(std::uint32_t));
    state->directory = reinterpret_cast<std::uint32_t*>(cursor);
    for (std::uint32_t i = 0; i < count; ++i) ::new (state->directory + i) std::uint32_t(kBookNull);
    cursor += sizeof(std::uint32_t) * std::uint64_t(count);
    state->dirty = reinterpret_cast<std::uint32_t*>(cursor);
    for (std::uint32_t i = 0; i < limits.journal_records; ++i) ::new (state->dirty + i) std::uint32_t(0);
    cursor += sizeof(std::uint32_t) * std::uint64_t(limits.journal_records);
    assert(cursor <= base + bytes);
    state->initialized = true;
    return BookStatus::Ok;
}
template<class Slot> inline Slot* touch(State<Slot>& state, std::uint32_t index) noexcept {
    assert(state.event_open);
    assert(index < 2 * state.limits.levels_per_side);
    if (state.kind == BookEventKind::Delta && state.directory[index] == kBookNull) {
        if (state.touched == state.limits.journal_records) return nullptr;
        state.directory[index] = state.touched;
        state.before[state.touched] = state.bank[state.active_bank][index];
        state.dirty[state.touched++] = index;
    }
    return state.bank[state.staged_bank] + index;
}
template<class Slot> inline const Slot& committed_slot(const State<Slot>& state,
    std::uint32_t index) noexcept {
    assert(state.initialized);
    assert(index < 2 * state.limits.levels_per_side);
    const auto record = state.directory[index];
    return record == kBookNull ? state.bank[state.active_bank][index] : state.before[record];
}
template<class Slot> inline void discard(State<Slot>& state, bool restore) noexcept {
    assert(state.initialized);
    assert(state.touched <= state.limits.journal_records);
    for (std::uint32_t i = 0; i < state.touched; ++i) {
        const auto index = state.dirty[i];
        if (restore) state.bank[state.active_bank][index] = state.before[i];
        state.directory[index] = kBookNull;
    }
    state.touched = 0;
}
template<class Slot> inline BookStatus begin(State<Slot>& state, BookCut cut, BookEventKind kind) noexcept {
    assert(state.limits.levels_per_side <= kBookMaxLevels);
    assert(state.touched <= state.limits.journal_records);
    if (!state.initialized || (kind != BookEventKind::Delta && kind != BookEventKind::Replacement))
        return BookStatus::Invalid;
    if (state.event_open) return BookStatus::Busy;
    if (kind == BookEventKind::Delta && !state.committed.read.valid) return BookStatus::Invalid;
    if (!book_cut_after(cut, state.committed.read.cut, kind)) return BookStatus::Invalid;
    if (state.committed.read.revision == UINT64_MAX) return BookStatus::Overflow;
    state.working = kind == BookEventKind::Replacement ? Meta{} : state.committed;
    state.working.read.cut = cut;
    state.working.read.revision = state.committed.read.revision + 1;
    state.working.read.valid = true;
    state.kind = kind; state.event_open = true; state.failure = BookStatus::Ok;
    state.staged_bank = state.active_bank ^ unsigned(kind == BookEventKind::Replacement);
    state.operations = 0;
    return BookStatus::Ok;
}
template<class Slot> inline BookStatus fail(State<Slot>& state, BookStatus status) noexcept {
    assert(status != BookStatus::Ok);
    assert(state.event_open);
    discard(state, true);
    state.failure = status;
    return status;
}
template<class Slot> inline BookStatus admit_operation(State<Slot>& state) noexcept {
    assert(state.operations <= state.limits.event_operations);
    assert(state.touched <= state.limits.journal_records);
    if (!state.initialized || !state.event_open) return BookStatus::Invalid;
    if (state.failure != BookStatus::Ok) return state.failure;
    if (state.operations == state.limits.event_operations) return fail(state, BookStatus::Capacity);
    ++state.operations;
    return BookStatus::Ok;
}
template<class Slot> inline BookStatus commit(State<Slot>& state) noexcept {
    assert(state.touched <= state.limits.journal_records);
    assert(state.operations <= state.limits.event_operations);
    if (!state.initialized || !state.event_open) return BookStatus::Invalid;
    if (state.failure != BookStatus::Ok) return state.failure;
    discard(state, false);
    state.active_bank = state.staged_bank;
    state.committed = state.working;
    state.event_open = false;
    return BookStatus::Ok;
}
template<class Slot> inline void abort(State<Slot>& state) noexcept {
    assert(state.initialized);
    assert(state.touched <= state.limits.journal_records);
    discard(state, true);
    state.event_open = false; state.failure = BookStatus::Ok;
}
template<class Slot> inline bool output_valid(const State<Slot>& s, const void* output,
    std::uint64_t bytes) noexcept {
    if (bytes == 0) return true;
    if (!output) return false;
    const auto base = reinterpret_cast<std::uintptr_t>(output);
    if (bytes > UINTPTR_MAX - base) return false;
    const auto state = reinterpret_cast<std::uintptr_t>(&s);
    return !(base < s.storage_base + s.storage_bytes && s.storage_base < base + bytes) &&
        !(base < state + sizeof(s) && state < base + bytes);
}
} // namespace bolt::fintech::book_detail
