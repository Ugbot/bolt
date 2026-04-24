// bolt_disruptor.h — LMAX-style MP-claim Disruptor ring.
//
// RULES: No exceptions. No RTTI. No heap after `init`. Lock-free.
//
// What this is: the real LMAX Disruptor. A single fixed-capacity ring
// with (a) a producer-side `cursor` that multiple threads CAS-claim
// via `fetch_add`, (b) a producer-side `published` counter that
// advances when each producer has finished writing its claimed slot
// AND every predecessor has also published, (c) a set of consumer
// cursors registered at init time, each of which waits on `published`
// and marks slots consumed so the producer can recycle them.
//
// Why we need this in MarbleDB: the WAL reserve/commit discipline is
// exactly this pattern. Producer claims an LSN by fetch_add on a seq
// counter; writes its entry into the ring slot; publishes in
// predecessor-order so consumers (memtable subscriber, flusher) see
// contiguous committed entries. Today we fake it with `put_mu`; this
// primitive makes it lock-free and multi-producer safe.
//
// Semantics in detail:
//
//   Producer (any thread):
//     const uint64_t seq = d.claim(1);          // fetch_add on cursor
//     T* p = d.slot(seq);                        // pointer into ring
//     *p = ...populate your entry...
//     d.publish(seq);                            // wait for pred, CAS-advance
//
//   Consumer (registered at init with id in [0, num_consumers)):
//     uint64_t next = 0;
//     while (running) {
//         const uint64_t upto = d.wait_for(next);     // parks until published>=next
//         for (uint64_t s = next; s <= upto; ++s) {
//             const T& entry = *d.slot(s);
//             ... process entry ...
//         }
//         d.mark_consumed(consumer_id, upto);
//         next = upto + 1;
//     }
//
// Backpressure: producers cannot claim more than `Capacity` slots
// ahead of the slowest consumer. If `cursor - min_consumer_cursor >=
// Capacity`, `claim` spin-yields until consumers catch up.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/bolt_sequence.h"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <thread>

namespace bolt {

static constexpr uint32_t kDisruptorMaxConsumers = 16;

template <typename T, uint32_t Capacity>
struct alignas(64) Disruptor {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    static_assert(Capacity >= 16, "Capacity too small");

    // Producer cursor — atomically fetch_add-claimed. Monotonically
    // increasing forever (mod wraps onto `slots` via `& kMask`).
    Sequence cursor;
    // Highest published sequence. Only advances once all predecessors
    // have published. Consumers wait on this.
    Sequence published;
    // Per-consumer high-water marks. Producers read the minimum across
    // all registered consumers for backpressure.
    Sequence* consumer_cursors[kDisruptorMaxConsumers];
    uint32_t  num_consumers;
    uint32_t  _pad_consumers;
    // The ring. Cache-line-padded between cursor/published and slots
    // so producer/consumer cursor traffic doesn't trash slot lines.
    alignas(64) T slots[Capacity];

    static constexpr uint32_t kMask = Capacity - 1;

    // Init — zero the cursors, register no consumers yet.
    BOLT_FORCE_INLINE void init() noexcept {
        cursor.store_relaxed(0);
        published.store_relaxed(0);
        num_consumers = 0;
        for (uint32_t i = 0; i < kDisruptorMaxConsumers; ++i) {
            consumer_cursors[i] = nullptr;
        }
    }

    // Register a consumer — returns its id, or UINT32_MAX if full.
    // The passed-in Sequence must live as long as this Disruptor.
    BOLT_FORCE_INLINE uint32_t register_consumer(Sequence* cursor_out) noexcept {
        assert(cursor_out != nullptr);
        if (num_consumers >= kDisruptorMaxConsumers) return UINT32_MAX;
        const uint32_t id = num_consumers;
        consumer_cursors[id] = cursor_out;
        cursor_out->store_relaxed(0);
        ++num_consumers;
        return id;
    }

    // Min consumer cursor — the slowest consumer. Used for producer
    // backpressure: we can't overwrite slots the slowest consumer
    // hasn't drained.
    BOLT_FORCE_INLINE uint64_t min_consumer_cursor() const noexcept {
        if (num_consumers == 0) {
            // No consumers — producer can run freely up to published.
            return published.load_acquire();
        }
        uint64_t m = UINT64_MAX;
        for (uint32_t i = 0; i < num_consumers; ++i) {
            const uint64_t v = consumer_cursors[i]->load_acquire();
            if (v < m) m = v;
        }
        return m;
    }

    // Claim N slots atomically. Returns the starting sequence. Spin-
    // yields if the ring is full (producer ahead of slowest consumer
    // by >= Capacity).
    BOLT_FORCE_INLINE uint64_t claim(uint32_t count = 1) noexcept {
        assert(count > 0);
        assert(count <= Capacity);
        const uint64_t my_end = cursor.fetch_add_acq_rel(count) + count;
        // Wait for consumers to catch up if needed.
        for (;;) {
            const uint64_t mc = min_consumer_cursor();
            if (my_end - mc <= Capacity) break;
            BOLT_PAUSE();
            std::this_thread::yield();
        }
        return my_end - count;
    }

    BOLT_FORCE_INLINE T* slot(uint64_t seq) noexcept {
        return &slots[seq & kMask];
    }
    BOLT_FORCE_INLINE const T* slot(uint64_t seq) const noexcept {
        return &slots[seq & kMask];
    }

    // Publish a single slot — this is the MP-safe commit step. We can
    // only advance `published` past `seq` when all predecessors are
    // also published. Producer spin-waits for predecessors under
    // contention; this is PostgreSQL's insert-LSN discipline.
    BOLT_FORCE_INLINE void publish(uint64_t seq) noexcept {
        // Wait for predecessor to publish first.
        while (published.load_acquire() < seq) {
            BOLT_PAUSE();
        }
        // We are the next in line; advance.
        published.store_release(seq + 1);
        published.notify_all();
    }

    // Batch-publish a claimed range [lo, hi] (inclusive).
    BOLT_FORCE_INLINE void publish_range(uint64_t lo, uint64_t hi) noexcept {
        assert(hi >= lo);
        while (published.load_acquire() < lo) {
            BOLT_PAUSE();
        }
        published.store_release(hi + 1);
        published.notify_all();
    }

    // Consumer: wait until published >= min_seq. Returns the max
    // available seq (>= min_seq). Parks via atomic::wait.
    BOLT_FORCE_INLINE uint64_t wait_for(uint64_t min_seq) noexcept {
        uint64_t p = published.load_acquire();
        while (p < min_seq + 1) {
            published.wait(p);
            p = published.load_acquire();
        }
        return p - 1;
    }

    // Consumer: mark slots up to `seq` (inclusive) as processed. The
    // producer can recycle them.
    BOLT_FORCE_INLINE void mark_consumed(uint32_t consumer_id,
                                          uint64_t seq) noexcept {
        assert(consumer_id < num_consumers);
        assert(consumer_cursors[consumer_id] != nullptr);
        consumer_cursors[consumer_id]->store_release(seq + 1);
    }
};

}  // namespace bolt
