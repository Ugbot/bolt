// bolt_arena_ring.h — Fixed-capacity pool of pre-allocated Arenas.
//
// RULES: No exceptions. No RTTI. Lock-free acquire/release via CAS on a
// packed slot-index+generation head. Zero allocation after `init`.
//
// Motivation. Streaming pipelines and scan iterators need per-task
// scratch memory. A fresh `bolt::Arena()` per task triggers the Arena
// constructor's initial-block allocation; a per-thread thread-local
// Arena fights the work-stealing pattern. This primitive stands up a
// bounded ring of pre-allocated Arenas and hands them out LIFO; each
// slot's Arena is placement-constructed exactly once (lazily on first
// acquire) and its cursor is `reset()` on every subsequent acquire.
//
// Semantics:
//   init(cfg)    — stores Arena config for lazy first-touch construction.
//                  Zero allocation performed here.
//   acquire()    — pops a slot. Returns its Arena pointer; the slot's
//                  arena has been reset (pointer rewind). Returns
//                  nullptr if the pool is exhausted.
//   release(a)   — returns the Arena's owning slot to the freelist.
//                  Safe from any thread.
//   destroy()    — tears down every constructed slot's Arena. Not
//                  thread-safe with concurrent acquires.
//
// Invariants:
//   - slots[0..slots_used) have their Arena constructed.
//   - slots on the freelist have `arena.cursor_` reset on next acquire
//     (we reset AT acquire rather than AT release so the fast path
//     always returns a fresh arena to the caller).
//
// ABA. Head is a 64-bit packed `(generation << 32) | slot_idx`. Each
// push bumps the generation; CAS on the packed value prevents ABA.

#pragma once

#include "bolt/bolt_arena.h"     // Arena, ArenaConfig
#include "bolt/bolt_port.h"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <new>

namespace bolt {

struct alignas(64) ArenaRingSlot {
    Arena    arena;
    uint32_t next_free_idx;   // UINT32_MAX = end-of-stack
    uint8_t  initialized;
    uint8_t  _pad[3];
};

static constexpr uint32_t kArenaRingEmptyIdx = 0xFFFFFFFFu;

BOLT_FORCE_INLINE uint32_t arena_ring_head_idx(uint64_t head) noexcept {
    return static_cast<uint32_t>(head & 0xFFFFFFFFu);
}

BOLT_FORCE_INLINE uint64_t arena_ring_pack(uint32_t idx, uint32_t gen) noexcept {
    return (static_cast<uint64_t>(gen) << 32) | static_cast<uint64_t>(idx);
}

template <uint32_t Capacity>
struct alignas(64) ArenaRing {
    using Slot = ArenaRingSlot;

    alignas(64) std::atomic<uint64_t> free_head;
    alignas(64) std::atomic<uint32_t> slots_used;
    ArenaConfig                       arena_cfg;
    Slot                              slots[Capacity];

    BOLT_FORCE_INLINE void init(ArenaConfig cfg = ArenaConfig{}) noexcept {
        arena_cfg = cfg;
        free_head.store(arena_ring_pack(kArenaRingEmptyIdx, 0u),
                        std::memory_order_relaxed);
        slots_used.store(0u, std::memory_order_relaxed);
        for (uint32_t i = 0; i < Capacity; ++i) {
            slots[i].initialized   = 0u;
            slots[i].next_free_idx = kArenaRingEmptyIdx;
        }
    }

    BOLT_FORCE_INLINE uint32_t pop_free() noexcept {
        uint64_t head = free_head.load(std::memory_order_acquire);
        for (;;) {
            const uint32_t idx = arena_ring_head_idx(head);
            if (idx == kArenaRingEmptyIdx) return kArenaRingEmptyIdx;
            const uint32_t next = slots[idx].next_free_idx;
            const uint32_t gen  = static_cast<uint32_t>(head >> 32);
            const uint64_t new_head = arena_ring_pack(next, gen + 1u);
            if (free_head.compare_exchange_weak(
                    head, new_head,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                return idx;
            }
        }
    }

    BOLT_FORCE_INLINE void push_free(uint32_t idx) noexcept {
        assert(idx < Capacity);
        uint64_t head = free_head.load(std::memory_order_acquire);
        for (;;) {
            const uint32_t cur = arena_ring_head_idx(head);
            slots[idx].next_free_idx = cur;
            const uint32_t gen = static_cast<uint32_t>(head >> 32);
            const uint64_t new_head = arena_ring_pack(idx, gen + 1u);
            if (free_head.compare_exchange_weak(
                    head, new_head,
                    std::memory_order_release,
                    std::memory_order_acquire)) {
                return;
            }
        }
    }

    // Acquire. Returns a pointer to a reset Arena, or nullptr on
    // capacity exhaustion.
    BOLT_FORCE_INLINE Arena* acquire() noexcept {
        const uint32_t freed = pop_free();
        Slot* s = nullptr;
        if (freed != kArenaRingEmptyIdx) {
            s = &slots[freed];
            s->arena.reset();
        } else {
            const uint32_t idx = slots_used.fetch_add(1u, std::memory_order_acq_rel);
            if (idx >= Capacity) {
                slots_used.fetch_sub(1u, std::memory_order_relaxed);
                return nullptr;
            }
            s = &slots[idx];
            assert(s->initialized == 0u);
            new (&s->arena) Arena(arena_cfg);
            s->initialized = 1u;
        }
        s->next_free_idx = kArenaRingEmptyIdx;
        return &s->arena;
    }

    // Release an Arena previously returned by `acquire`. The Arena is
    // the first field of its slot, so the cast back is well-defined.
    BOLT_FORCE_INLINE void release(Arena* a) noexcept {
        if (a == nullptr) return;
        auto* s = reinterpret_cast<Slot*>(a);
        assert(s >= &slots[0] && s < &slots[Capacity]);
        const uint32_t idx = static_cast<uint32_t>(s - &slots[0]);
        push_free(idx);
    }

    BOLT_FORCE_INLINE void destroy() noexcept {
        const uint32_t n = slots_used.load(std::memory_order_acquire);
        for (uint32_t i = 0; i < n && i < Capacity; ++i) {
            if (slots[i].initialized != 0u) {
                slots[i].arena.~Arena();
                slots[i].initialized = 0u;
            }
        }
        slots_used.store(0u, std::memory_order_release);
        free_head.store(arena_ring_pack(kArenaRingEmptyIdx, 0u),
                        std::memory_order_release);
    }
};

}  // namespace bolt
