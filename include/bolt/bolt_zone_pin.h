// bolt_zone_pin.h — Reader-refcount pin for columnar zones / pages.
//
// RULES: No exceptions. No RTTI. No smart pointers. Single atomic field.
//
// Motivation. Columnar engines that hand out pointers into zone / block
// memory need a way to keep that memory alive while a reader is mid-walk.
// The block-cache `pin_count` pattern (in marbledb's block_cache) is the
// precedent; this header generalises it to any producer/consumer pair
// where:
//
//   - producers publish read-only memory (e.g. a sealed zone's columns);
//   - consumers hold direct pointers for the duration of a scoped op;
//   - a third party (flush / compaction / recycler) wants to reclaim the
//     memory once every reader has exited.
//
// Semantics:
//   pin()          — register one reader. Must be paired with unpin().
//                    fetch_add with acquire-release ordering so any data
//                    the reader subsequently reads is ordered after the
//                    publishing writer's release.
//   unpin()        — register reader exit. Release so the published
//                    memory cannot be reordered past unpin; the reclaimer
//                    on the other side acquire-loads.
//   wait_drained() — spin until readers count goes to zero. Caller MUST
//                    guarantee no new pin() can race (e.g. the zone is
//                    already marked sealed/retired). No allocation; pure
//                    pause + load loop.
//
// Size: one cache line (one atomic uint32_t + padding). Tiger Style: POD.
// Copies disabled — embed by value, don't move.

#pragma once

#include "bolt/bolt_port.h"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <thread>

namespace bolt {

struct alignas(64) ZonePin {
    alignas(64) std::atomic<uint32_t> readers;

    BOLT_FORCE_INLINE void init() noexcept {
        readers.store(0u, std::memory_order_relaxed);
    }

    // Pin — one extra reader. Acquire-release pairs with a producer's
    // release-store on the zone's publication atomic; any subsequent
    // load of the zone's data cannot be reordered above this.
    BOLT_FORCE_INLINE void pin() noexcept {
        readers.fetch_add(1u, std::memory_order_acq_rel);
    }

    // Unpin — one fewer reader. Release so every load the reader did
    // happens-before the reclaimer's acquire-load.
    BOLT_FORCE_INLINE void unpin() noexcept {
        const uint32_t prev = readers.fetch_sub(1u, std::memory_order_release);
        assert(prev > 0u);
        (void)prev;
    }

    // Spin until readers == 0. Caller's responsibility: ensure no new
    // pin() can race — typically by setting a sealed/retired flag first
    // (which readers check under release/acquire pairing with unpin).
    BOLT_FORCE_INLINE void wait_drained() noexcept {
        while (readers.load(std::memory_order_acquire) != 0u) {
            BOLT_PAUSE();
            std::this_thread::yield();
        }
    }

    // Non-blocking check — returns true if no reader is currently pinned.
    BOLT_FORCE_INLINE bool is_drained() const noexcept {
        return readers.load(std::memory_order_acquire) == 0u;
    }

    ZonePin()                          = default;
    ~ZonePin()                         = default;
    ZonePin(const ZonePin&)            = delete;
    ZonePin& operator=(const ZonePin&) = delete;
    ZonePin(ZonePin&&)                 = delete;
    ZonePin& operator=(ZonePin&&)      = delete;
};

}  // namespace bolt
