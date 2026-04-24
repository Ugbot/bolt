// bolt_lock_free_clock_lru.h — Sharded CLOCK-approximation LRU.
//
// RULES: No exceptions. No RTTI. No heap after `init`. Lock-free.
//
// Motivation. True LRU needs a doubly-linked list protected by a
// mutex (every hit rewires the list). CLOCK-LRU approximates LRU
// without the list: each slot has a `ref_bit`; hits set the bit;
// eviction scans the key's probe chain and evicts the first slot with
// `ref_bit == 0 && pin_count == 0`, clearing bits it passes.
//
// Sharded: N independent rings, chosen by key hash. Contention is
// split across shards. The pattern Pebble and RocksDB use for their
// block caches.
//
// Addressing: BOTH `pin` and `evict_and_pin` probe from
// `bucket(key) = (uint32_t)key & kSlotMask` with the same
// `kProbeDistance` horizon. This guarantees round-trip correctness
// (`pin(k)` finds whatever `evict_and_pin(k)` installed) at the cost
// of per-probe-chain (rather than global) CLOCK fairness — the
// ref-bit sweep happens within each key's probe window, giving local
// LRU approximation with correct lookup semantics.
//
// Hit path is `ref_bit.store(1, relaxed)` — single atomic write, no
// mutex. Miss path scans the key's probe chain, clearing ref bits on
// a first sweep, claiming an unref'd unpinned slot on a second.
//
// Pin protocol: hits return a `slot_id`; consumers hold the slot
// until they call `unpin(slot_id)`. Eviction skips pinned slots.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/bolt_sequence.h"

#include <atomic>
#include <cassert>
#include <cstdint>

namespace bolt {

template <typename K, uint32_t SlotsPerShard, uint32_t Shards = 16>
struct alignas(64) LockFreeClockLru {
    static_assert((SlotsPerShard & (SlotsPerShard - 1)) == 0, "pow2");
    static_assert((Shards & (Shards - 1)) == 0, "pow2");

    struct alignas(64) Slot {
        std::atomic<uint64_t>  key_occupied;   // packed: key << 1 | occupied_bit
        std::atomic<uint32_t>  pin_count;
        std::atomic<uint8_t>   ref_bit;
        uint8_t                _pad[3];
    };

    struct alignas(64) Shard {
        Slot                   slots[SlotsPerShard];
    };

    Shard shards[Shards];
    static constexpr uint32_t kShardMask     = Shards - 1;
    static constexpr uint32_t kSlotMask      = SlotsPerShard - 1;
    // Shared probe horizon for pin + evict_and_pin — must be identical
    // on both sides so the round-trip `pin(k)` after `evict_and_pin(k)`
    // is guaranteed to find the slot.
    static constexpr uint32_t kProbeDistance = 16;

    BOLT_FORCE_INLINE void init() noexcept {
        for (uint32_t s = 0; s < Shards; ++s) {
            for (uint32_t i = 0; i < SlotsPerShard; ++i) {
                shards[s].slots[i].key_occupied.store(0, std::memory_order_relaxed);
                shards[s].slots[i].pin_count.store(0, std::memory_order_relaxed);
                shards[s].slots[i].ref_bit.store(0, std::memory_order_relaxed);
            }
        }
    }

    // Try to pin an existing slot for `key`. Returns packed slot_id
    // (shard << 24 | slot_idx) on hit, or UINT32_MAX on miss.
    BOLT_FORCE_INLINE uint32_t pin(K key) noexcept {
        const uint64_t ku = static_cast<uint64_t>(key);
        const uint32_t shard_idx = static_cast<uint32_t>(ku ^ (ku >> 32)) & kShardMask;
        Shard& sh = shards[shard_idx];
        const uint32_t start = static_cast<uint32_t>(ku) & kSlotMask;
        for (uint32_t probe = 0; probe < kProbeDistance; ++probe) {
            const uint32_t si = (start + probe) & kSlotMask;
            Slot& s = sh.slots[si];
            const uint64_t ko = s.key_occupied.load(std::memory_order_acquire);
            if ((ko & 1u) == 0u) continue;   // slot empty; continue probe
            if ((ko >> 1) != ku) continue;
            // Bump pin first, then verify key still matches (avoid ABA).
            s.pin_count.fetch_add(1, std::memory_order_acquire);
            if ((s.key_occupied.load(std::memory_order_acquire) >> 1) == ku) {
                s.ref_bit.store(1, std::memory_order_relaxed);
                return (shard_idx << 24) | si;
            }
            s.pin_count.fetch_sub(1, std::memory_order_release);
        }
        return UINT32_MAX;
    }

    // Unpin a previously pinned slot.
    BOLT_FORCE_INLINE void unpin(uint32_t slot_id) noexcept {
        if (slot_id == UINT32_MAX) return;
        const uint32_t shard_idx = slot_id >> 24;
        const uint32_t si        = slot_id & 0xFFFFFFu;
        assert(shard_idx < Shards);
        assert(si < SlotsPerShard);
        shards[shard_idx].slots[si].pin_count.fetch_sub(
            1, std::memory_order_release);
    }

    // Evict a slot for `key` and install it — lock-free two-sweep
    // scan within the key's probe chain. Sweep 1 clears ref bits on
    // active unpinned slots; sweep 2 claims the first unref'd slot.
    // If the key is already present in the probe chain, short-circuits
    // and re-pins the existing slot (idempotent insert). Caller is
    // responsible for subsequently filling the slot's associated
    // state. Returns a pinned slot_id, or UINT32_MAX if every slot in
    // the probe chain is pinned (rare — cache is saturated).
    BOLT_FORCE_INLINE uint32_t evict_and_pin(K key) noexcept {
        const uint64_t ku = static_cast<uint64_t>(key);
        const uint32_t shard_idx = static_cast<uint32_t>(ku ^ (ku >> 32)) & kShardMask;
        Shard& sh = shards[shard_idx];
        const uint32_t start = static_cast<uint32_t>(ku) & kSlotMask;
        // Two sweeps of the probe window: first clears ref bits, second
        // claims. Idempotent on duplicate insert (short-circuits to the
        // existing slot).
        for (uint32_t sweep = 0; sweep < 2 * kProbeDistance; ++sweep) {
            const uint32_t probe = sweep % kProbeDistance;
            const uint32_t h = (start + probe) & kSlotMask;
            Slot& s = sh.slots[h];
            if (s.pin_count.load(std::memory_order_acquire) > 0) continue;
            const uint64_t ko = s.key_occupied.load(std::memory_order_acquire);
            const bool     occupied = (ko & 1u) != 0u;
            // Idempotent insert: already present → re-pin.
            if (occupied && (ko >> 1) == ku) {
                s.pin_count.fetch_add(1, std::memory_order_acquire);
                if ((s.key_occupied.load(std::memory_order_acquire) >> 1) == ku) {
                    s.ref_bit.store(1, std::memory_order_relaxed);
                    return (shard_idx << 24) | h;
                }
                s.pin_count.fetch_sub(1, std::memory_order_release);
                continue;
            }
            // Empty slot — claim directly (no eviction needed).
            if (!occupied) {
                const uint64_t new_ko = (ku << 1) | 1u;
                s.key_occupied.store(new_ko, std::memory_order_release);
                s.ref_bit.store(1, std::memory_order_relaxed);
                s.pin_count.fetch_add(1, std::memory_order_acquire);
                return (shard_idx << 24) | h;
            }
            // Occupied by a different key — consult ref bit.
            const uint8_t rb = s.ref_bit.load(std::memory_order_acquire);
            if (rb != 0u) {
                s.ref_bit.store(0, std::memory_order_release);
                continue;
            }
            // Claim this slot (evict the occupant).
            const uint64_t new_ko = (ku << 1) | 1u;
            s.key_occupied.store(new_ko, std::memory_order_release);
            s.ref_bit.store(1, std::memory_order_relaxed);
            s.pin_count.fetch_add(1, std::memory_order_acquire);
            return (shard_idx << 24) | h;
        }
        return UINT32_MAX;
    }
};

}  // namespace bolt
