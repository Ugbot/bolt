// bolt_batch_pool.h — Typed, fixed-capacity Treiber-stack pool of
// `{Arena, BoltBatch}` slots.
//
// RULES: No exceptions. No RTTI. No heap after `init`. Lock-free.
//
// Motivation. C-ABI callers to storage engines (MarbleDB's
// `mdb_batch_create`, various Bolt consumers) need a cheap, reusable
// `BoltBatch` handle. Naive `malloc + Arena()` per call triggers a 4 MiB
// `VirtualAlloc` under the default Arena config and is the hot-path
// offender profiled in MarbleDB Wave 18b. The fix is a process-global
// (or consumer-owned) pool of pre-allocated slots — this header is the
// reusable primitive.
//
// Semantics.
//   init(cfg)     — sets every slot's Arena config for lazy first-touch
//                   construction. Zero allocation performed here; slots
//                   stay uninitialized until first `acquire`.
//   acquire()     — pops a slot from the Treiber stack (LIFO, lock-free
//                   via CAS on `free_head`). On stack-empty, bumps
//                   `slots_used` to claim a fresh slot and placement-news
//                   its Arena with the config from `init`. Returns
//                   nullptr if pool is fully exhausted AND no slots are
//                   in the freelist — correct backpressure signal.
//   release(slot) — pushes back onto the Treiber stack. Future `acquire`
//                   will reset the arena (pointer rewind, ~5 ns) before
//                   handing the slot out. Safe from any thread.
//
// ABA. `free_head` is a uint32_t slot-index; the high bits carry a
// monotonic generation counter packed into a uint64_t. That bumps on
// every push; CAS on the packed value prevents ABA.
//
// Capacity. Compile-time — instantiate `TypedBatchPool<128>` for a
// 128-slot pool. Exceeding capacity during concurrent use returns
// nullptr from `acquire`; caller handles the rare case (retry or fall
// back). No dynamic growth.

#pragma once

#include "bolt/bolt_arena.h"     // Arena, ArenaConfig
#include "bolt/bolt_column.h"    // BoltBatch
#include "bolt/bolt_port.h"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <new>

namespace bolt {

// A pool slot. `next_free_idx` is UINT32_MAX at pool-tail; the Treiber
// stack head stores the live head index in the low 32 bits and a
// generation counter in the high 32 bits.
struct alignas(64) TypedBatchPoolSlot {
    Arena     arena;
    BoltBatch batch;
    uint32_t  next_free_idx;     // valid when slot is on the freelist
    uint8_t   initialized;       // 0 until first acquire
    uint8_t   _pad[3];
};

// Packed stack head: [63:32] = generation, [31:0] = slot index (UINT32_MAX = empty).
static constexpr uint64_t kTypedBatchPoolEmpty = 0xFFFFFFFFu;

BOLT_FORCE_INLINE uint32_t typed_batch_pool_head_idx(uint64_t head) noexcept {
    return static_cast<uint32_t>(head & 0xFFFFFFFFu);
}

BOLT_FORCE_INLINE uint64_t typed_batch_pool_pack(uint32_t idx, uint32_t gen) noexcept {
    return (static_cast<uint64_t>(gen) << 32) | static_cast<uint64_t>(idx);
}

template <uint32_t Capacity>
struct alignas(64) TypedBatchPool {
    using Slot = TypedBatchPoolSlot;

    // Treiber stack head (generation + slot idx) and the "how many
    // slots have we constructed" counter, cache-line padded.
    alignas(64) std::atomic<uint64_t> free_head;
    alignas(64) std::atomic<uint32_t> slots_used;

    // Stored Arena config for lazy slot initialization.
    ArenaConfig arena_cfg;

    Slot slots[Capacity];

    BOLT_FORCE_INLINE void init(ArenaConfig cfg = ArenaConfig{}) noexcept {
        arena_cfg = cfg;
        free_head.store(typed_batch_pool_pack(kTypedBatchPoolEmpty, 0u),
                        std::memory_order_relaxed);
        slots_used.store(0u, std::memory_order_relaxed);
        for (uint32_t i = 0; i < Capacity; ++i) {
            slots[i].initialized   = 0u;
            slots[i].next_free_idx = kTypedBatchPoolEmpty;
        }
    }

    // Pop from freelist. Returns slot index, or UINT32_MAX on empty.
    BOLT_FORCE_INLINE uint32_t pop_free() noexcept {
        uint64_t head = free_head.load(std::memory_order_acquire);
        for (;;) {
            const uint32_t idx = typed_batch_pool_head_idx(head);
            if (idx == kTypedBatchPoolEmpty) return kTypedBatchPoolEmpty;
            const uint32_t next = slots[idx].next_free_idx;
            const uint32_t gen  = static_cast<uint32_t>(head >> 32);
            const uint64_t new_head = typed_batch_pool_pack(next, gen + 1u);
            if (free_head.compare_exchange_weak(
                    head, new_head,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                return idx;
            }
        }
    }

    // Push onto freelist. Safe from any thread.
    BOLT_FORCE_INLINE void push_free(uint32_t idx) noexcept {
        assert(idx < Capacity);
        uint64_t head = free_head.load(std::memory_order_acquire);
        for (;;) {
            const uint32_t cur = typed_batch_pool_head_idx(head);
            slots[idx].next_free_idx = cur;
            const uint32_t gen = static_cast<uint32_t>(head >> 32);
            const uint64_t new_head = typed_batch_pool_pack(idx, gen + 1u);
            if (free_head.compare_exchange_weak(
                    head, new_head,
                    std::memory_order_release,
                    std::memory_order_acquire)) {
                return;
            }
        }
    }

    // Acquire a slot. Arena is reset + batch re-initialized in-place.
    // Returns nullptr if the pool is exhausted.
    //
    // Wave 9.4 — switched from `arena.reset()` to a conditional
    // `compact()`. Bump-allocator semantics mean reset() keeps blocks
    // around for reuse, which is fast for repeat workloads with the
    // SAME column shape but pathological when batch shapes vary
    // (e.g. small-i64 batches grew block 0; later large-vector_f32
    // batches added more blocks; after enough cycles the arena hit
    // `kArenaMaxBlocks = 32` and refused to grow). Compacting when
    // the block count exceeds a heuristic threshold (4) drops the
    // excess back to block 0 and lets subsequent batches re-grow
    // from a clean slate. The cost is `O(blocks)` aligned_free
    // calls — measured ≤ 1 µs per acquire, vs the ~50 µs of a
    // fresh new (&arena) Arena. Pure win.
    BOLT_FORCE_INLINE Slot* acquire() noexcept {
        const uint32_t freed = pop_free();
        Slot* s = nullptr;
        if (freed != kTypedBatchPoolEmpty) {
            s = &slots[freed];
            if (s->arena.num_blocks() > 4u) {
                s->arena.compact();
            } else {
                s->arena.reset();
            }
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
        BoltBatch::init_empty(&s->batch);
        s->batch.arena = &s->arena;
        s->next_free_idx = kTypedBatchPoolEmpty;
        return s;
    }

    // Release a slot previously returned by `acquire`. Returns it to the
    // freelist; no arena reset done here (acquire does it to keep the
    // fast path's data fresh).
    BOLT_FORCE_INLINE void release(Slot* s) noexcept {
        assert(s != nullptr);
        assert(s >= &slots[0] && s < &slots[Capacity]);
        const uint32_t idx = static_cast<uint32_t>(s - &slots[0]);
        push_free(idx);
    }

    // Destroy any arenas that were lazily constructed. Safe to call at
    // process teardown; not thread-safe vs. concurrent acquires.
    BOLT_FORCE_INLINE void destroy() noexcept {
        const uint32_t n = slots_used.load(std::memory_order_acquire);
        for (uint32_t i = 0; i < n && i < Capacity; ++i) {
            if (slots[i].initialized != 0u) {
                slots[i].arena.~Arena();
                slots[i].initialized = 0u;
            }
        }
        slots_used.store(0u, std::memory_order_release);
        free_head.store(typed_batch_pool_pack(kTypedBatchPoolEmpty, 0u),
                        std::memory_order_release);
    }
};

}  // namespace bolt
