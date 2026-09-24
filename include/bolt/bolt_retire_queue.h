// bolt_retire_queue.h — deferred-teardown queue for EBR-managed
// objects with fat payloads.
//
// RULES: No exceptions. No RTTI. No heap after `init`. Lock-free
// MP-enqueue, SC-dequeue (single reaper thread).
//
// Motivation. `bolt::Ebr`'s per-shard retire slot is 16 bytes
// (`EbrNode = {void*, void(*)(void*)}`). Too small to carry a
// filesystem path for deferred `filesystem::remove` after an SST
// close. This primitive stores fat payloads in its own ring with an
// epoch stamp; a reaper thread dequeues entries only once
// `ebr_global_epoch() >= retire_epoch + 3`, guaranteeing no EBR
// reader still holds a pointer to the object. +3, not +2: a reader
// pinned AT the retire epoch R may hold the object, and the collector
// only checks for such a reader on the R+2 -> R+3 advance (the same
// point at which `ebr_retire`'s own slot for R is drained).
//
// Templated on the payload type T. The producer enqueues a T with
// the current epoch at retirement time. The reaper tries to dequeue;
// if the head entry's epoch hasn't drained yet, it backs off.

#pragma once

#include "bolt/bolt_disruptor.h"
#include "bolt/bolt_ebr.h"

#include <cstdint>

namespace bolt {

// Advances past the retire epoch before an entry is reclaimable; matches
// `ebr_try_advance_and_collect`'s drain of slot (ge + 1) mod 3.
inline constexpr uint64_t kRetireQueueDrainAdvances = 3u;

template <typename T, uint32_t Capacity>
struct alignas(64) RetireQueue {
    struct Entry {
        T        payload;
        uint64_t retire_epoch;
    };
    Disruptor<Entry, Capacity> ring;
    Sequence                   consumer_cursor;
    uint32_t                   consumer_id;

    BOLT_FORCE_INLINE void init() noexcept {
        ring.init();
        consumer_id = ring.register_consumer(&consumer_cursor);
    }

    // MP-enqueue — any thread can call. Grabs an epoch number via the
    // passed-in Ebr (caller's responsibility to hand in the right
    // one).
    BOLT_FORCE_INLINE bool enqueue(T&& payload, const Ebr* e) noexcept {
        const uint64_t ge = e->global_epoch.load(std::memory_order_acquire);
        const uint64_t seq = ring.claim(1);
        Entry* slot = ring.slot(seq);
        slot->payload      = std::move(payload);
        slot->retire_epoch = ge;
        ring.publish(seq);
        return true;
    }

    // SC-dequeue — only the reaper thread calls this. Returns true
    // and writes into `*out` if an entry is available AND its
    // retire_epoch + 3 has been reached by the global epoch. Returns
    // false if the queue is empty or the head entry isn't drained yet.
    BOLT_FORCE_INLINE bool try_dequeue(Ebr* e, T* out) noexcept {
        const uint64_t next = consumer_cursor.load_acquire();
        const uint64_t pub  = ring.published.load_acquire();
        if (next >= pub) return false;   // empty
        const Entry* head = ring.slot(next);
        const uint64_t ge = e->global_epoch.load(std::memory_order_acquire);
        if (ge < head->retire_epoch + kRetireQueueDrainAdvances) {
            return false;  // not drained yet — try to advance the epoch
        }
        *out = head->payload;
        ring.mark_consumed(consumer_id, next);
        return true;
    }

    // True if there are no pending entries at all (for a graceful
    // shutdown the reaper can drain to empty before exiting).
    BOLT_FORCE_INLINE bool empty() const noexcept {
        return consumer_cursor.load_acquire() >= ring.published.load_acquire();
    }
};

}  // namespace bolt
