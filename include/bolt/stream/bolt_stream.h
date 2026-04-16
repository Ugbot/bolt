// bolt_stream.h — v0 streaming pipeline core
//
// RULES: No exceptions. No RTTI. No smart pointers. No std::vector / std::string.
// All noexcept. Tiger-Style: pre-allocate everything at create(), fixed bounds,
// >=2 asserts per hot-path function, functions <=70 LOC.
//
// Scope: a fixed-shape, single linear chain of stages. The user wires it at
// startup, then pushes BoltBatches through pipeline_push. v0 is synchronous:
// the caller's thread walks the stage chain, and the bolt::Scheduler parallelizes
// work INSIDE each stage kernel (not across batches). This keeps the public
// surface stable while leaving room for an async dispatcher in v1.
//
// Backpressure: an ArenaRing of fixed capacity owns per-in-flight scratch
// arenas. Acquiring a slot spin-yields when the ring is full, giving
// Tiger-Style implicit backpressure — callers push_wait rather than unbounded
// enqueue.

#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <thread>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_channel.h"   // for bolt::cpu_pause()
#include "bolt/bolt_column.h"
#include "bolt/bolt_port.h"
#include "bolt/bolt_scheduler.h"

namespace bolt {
namespace stream {

// ---------------------------------------------------------------------------
// Hard bounds (Tiger-Style). Reasoning:
//   kStreamMaxStages = 16 — streaming pipelines in practice are shallow
//   (source -> filter -> project -> aggregate -> sink is 5). 16 gives 3x
//   headroom while keeping Pipeline one cache-line-padded struct with a
//   predictable size (no dynamic growth, no std::vector).
// ---------------------------------------------------------------------------
inline constexpr uint32_t kStreamMaxStages        = 16;
inline constexpr uint32_t kStreamMaxInFlight      = 1024;   // arena ring cap
inline constexpr uint32_t kStreamDefaultInFlight  = 16;

// ---------------------------------------------------------------------------
// ArenaRing — bounded pool of per-batch scratch arenas.
//
// v0: push is synchronous, so acquire/release are LIFO-free serial calls from
// the caller's thread — head == tail most of the time. The atomic head/tail
// design is picked so the ring continues to function if v1 introduces a
// dispatcher thread (one producer, one consumer of slots).
// ---------------------------------------------------------------------------
struct ArenaRing {
    Arena*   slots;                        // capacity Arenas, contiguous block
    uint32_t capacity;                     // power of two
    uint32_t mask;                         // capacity - 1
    alignas(kCacheLine) std::atomic<uint64_t> head;  // claimed count
    alignas(kCacheLine) std::atomic<uint64_t> tail;  // released count

    static bool create(ArenaRing* out, uint32_t capacity_hint) noexcept;
    void destroy() noexcept;

    Arena* acquire() noexcept;             // spin-yield on full; never null
    void   release(Arena* a) noexcept;     // returns slot to the ring
};

// Round up to next power of two, clamp into [1, kStreamMaxInFlight].
inline uint32_t stream_round_pow2(uint32_t v) noexcept {
    assert(kStreamMaxInFlight >= 1);
    if (v == 0) v = 1;
    if (v > kStreamMaxInFlight) v = kStreamMaxInFlight;
    uint32_t p = 1;
    while (p < v) p <<= 1;
    assert(p >= v && (p & (p - 1)) == 0);
    return p;
}

inline bool ArenaRing::create(ArenaRing* out, uint32_t capacity_hint) noexcept {
    assert(out != nullptr);
    assert(capacity_hint > 0);

    const uint32_t cap = stream_round_pow2(capacity_hint);
    void* block = bolt_aligned_alloc(alignof(Arena), sizeof(Arena) * cap);
    if (!block) return false;

    out->slots    = static_cast<Arena*>(block);
    out->capacity = cap;
    out->mask     = cap - 1;
    out->head.store(0, std::memory_order_relaxed);
    out->tail.store(0, std::memory_order_relaxed);

    for (uint32_t i = 0; i < cap; ++i) {
        new (&out->slots[i]) Arena();      // default ArenaConfig
    }
    assert(out->slots != nullptr);
    assert((out->capacity & out->mask) == 0);
    return true;
}

inline void ArenaRing::destroy() noexcept {
    assert(this != nullptr);
    if (!slots) return;
    for (uint32_t i = 0; i < capacity; ++i) {
        slots[i].~Arena();
    }
    bolt_aligned_free(slots);
    slots    = nullptr;
    capacity = 0;
    mask     = 0;
    head.store(0, std::memory_order_relaxed);
    tail.store(0, std::memory_order_relaxed);
}

inline Arena* ArenaRing::acquire() noexcept {
    assert(slots != nullptr);
    assert(capacity > 0);

    // Spin-yield while ring is full (head - tail == capacity).
    for (;;) {
        const uint64_t h = head.load(std::memory_order_relaxed);
        const uint64_t t = tail.load(std::memory_order_acquire);
        if ((h - t) < capacity) {
            head.store(h + 1, std::memory_order_release);
            const uint32_t idx = static_cast<uint32_t>(h) & mask;
            Arena* a = &slots[idx];
            a->reset();                    // per-batch reset when reused
            return a;
        }
        for (int i = 0; i < 32; ++i) cpu_pause();
        std::this_thread::yield();
    }
}

inline void ArenaRing::release(Arena* a) noexcept {
    assert(slots != nullptr);
    assert(a != nullptr);
    (void)a;  // v0 push is synchronous => FIFO order is guaranteed by caller.
    // We just bump tail; the ring itself doesn't need to locate `a`.
    tail.fetch_add(1, std::memory_order_release);
}

// ---------------------------------------------------------------------------
// Stage + Pipeline
// ---------------------------------------------------------------------------

// Sink: consumes the final stage's output batch. Implementation-defined.
typedef void (*SinkFn)(const BoltBatch* result, void* user_data) noexcept;

// Stage operator. Returns true on success; false signals "drop this batch"
// (pipeline halts this batch, increments batches_dropped, releases the arena).
typedef bool (*StageFn)(const BoltBatch* in_batch,
                        BoltBatch*       out_batch,
                        Scheduler*       sched,
                        Arena*           arena,
                        void*            stage_state) noexcept;

struct Stage {
    StageFn fn;
    void*   state;
    char    name[32];
};

struct PipelineConfig {
    Scheduler* sched              = nullptr;   // required
    SinkFn     sink               = nullptr;   // required
    void*      sink_user          = nullptr;
    uint32_t   in_flight_capacity = kStreamDefaultInFlight;
};

struct Pipeline {
    Scheduler* sched;                         // not owned
    Stage      stages[kStreamMaxStages];
    uint32_t   num_stages;
    ArenaRing  arena_ring;                    // owned
    SinkFn     sink;
    void*      sink_user;

    alignas(kCacheLine) std::atomic<uint64_t> batches_pushed;
    alignas(kCacheLine) std::atomic<uint64_t> batches_drained;
    alignas(kCacheLine) std::atomic<uint64_t> batches_dropped;
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------
inline bool pipeline_create(Pipeline* out, const PipelineConfig& cfg) noexcept {
    assert(out != nullptr);
    assert(cfg.sched != nullptr);
    assert(cfg.sink  != nullptr);
    if (!out || !cfg.sched || !cfg.sink) return false;

    out->sched      = cfg.sched;
    out->num_stages = 0;
    out->sink       = cfg.sink;
    out->sink_user  = cfg.sink_user;
    memset(out->stages, 0, sizeof(out->stages));
    out->batches_pushed.store(0,  std::memory_order_relaxed);
    out->batches_drained.store(0, std::memory_order_relaxed);
    out->batches_dropped.store(0, std::memory_order_relaxed);

    if (!ArenaRing::create(&out->arena_ring, cfg.in_flight_capacity)) {
        return false;
    }
    assert(out->arena_ring.capacity > 0);
    return true;
}

inline void pipeline_destroy(Pipeline* p) noexcept {
    assert(p != nullptr);
    if (!p) return;
    p->arena_ring.destroy();
    p->num_stages = 0;
    p->sched      = nullptr;
    p->sink       = nullptr;
}

inline bool pipeline_add_stage(Pipeline* p, StageFn fn, void* state,
                               const char* name) noexcept {
    assert(p  != nullptr);
    assert(fn != nullptr);
    if (!p || !fn) return false;
    if (p->num_stages >= kStreamMaxStages) return false;

    Stage& s = p->stages[p->num_stages];
    s.fn    = fn;
    s.state = state;
    s.name[0] = '\0';
    if (name) {
        const size_t cap = sizeof(s.name) - 1;
        size_t n = 0;
        while (n < cap && name[n] != '\0') { s.name[n] = name[n]; ++n; }
        s.name[n] = '\0';
    }
    p->num_stages++;
    assert(p->num_stages <= kStreamMaxStages);
    return true;
}

// ---------------------------------------------------------------------------
// Push (v0 synchronous). Flow:
//   1. acquire an arena from the ring (implicit backpressure)
//   2. allocate ping-pong BoltBatch buffers in the arena
//   3. walk stages; on drop return false + bump counter
//   4. invoke sink on the final batch
//   5. release arena
// ---------------------------------------------------------------------------
inline bool pipeline_push(Pipeline* p, const BoltBatch* batch) noexcept {
    assert(p     != nullptr);
    assert(batch != nullptr);
    if (!p || !batch) return false;

    p->batches_pushed.fetch_add(1, std::memory_order_relaxed);

    Arena* arena = p->arena_ring.acquire();
    assert(arena != nullptr);

    // Ping-pong output batches in the arena. Two slots are enough because each
    // stage reads from `in` and writes to `out`; we swap roles after the call.
    BoltBatch* buf_a = static_cast<BoltBatch*>(arena->allocate(sizeof(BoltBatch), alignof(BoltBatch)));
    BoltBatch* buf_b = static_cast<BoltBatch*>(arena->allocate(sizeof(BoltBatch), alignof(BoltBatch)));
    if (!buf_a || !buf_b) {
        p->arena_ring.release(arena);
        p->batches_dropped.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    BoltBatch::init_empty(buf_a);
    BoltBatch::init_empty(buf_b);

    const BoltBatch* in_batch  = batch;
    BoltBatch*       out_batch = buf_a;
    BoltBatch*       spare     = buf_b;

    for (uint32_t i = 0; i < p->num_stages; ++i) {
        Stage& s = p->stages[i];
        assert(s.fn != nullptr);
        const bool ok = s.fn(in_batch, out_batch, p->sched, arena, s.state);
        if (!ok) {
            p->arena_ring.release(arena);
            p->batches_dropped.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        in_batch  = out_batch;
        BoltBatch* tmp = out_batch;
        out_batch = spare;
        spare     = tmp;
    }

    // Final result = last-written batch = current `in_batch` (post-swap).
    p->sink(in_batch, p->sink_user);

    p->arena_ring.release(arena);
    p->batches_drained.fetch_add(1, std::memory_order_relaxed);
    return true;
}

// v0 is synchronous: every push has already completed by the time it returns,
// so drain is a no-op. Kept in the surface so v1 (async dispatcher) doesn't
// break callers.
inline void pipeline_drain(Pipeline* p) noexcept {
    assert(p != nullptr);
    (void)p;
}

}  // namespace stream
}  // namespace bolt
