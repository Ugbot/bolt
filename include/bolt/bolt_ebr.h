// bolt_ebr.h — Epoch-Based Reclamation (EBR) primitive.
//
// 3-epoch scheme (Harris / Fraser / McKenney). Readers pin the current
// global epoch; producers retire objects into the current epoch's retire
// queue; the collector advances global_epoch and drains epoch (E-2 mod 3)
// once no reader is still pinned to it.
//
// RULES: No exceptions. No RTTI. No smart pointers. No std::vector/map
// on hot paths. Fixed-capacity arrays. Cache-line-padded atomics. All
// noexcept. No TLS — shard_id is passed explicitly by the caller.
//
// Consumer of this primitive (MarbleDB): safely free closed SSTable
// handles and memtable zones while concurrent readers hold raw pointers.
//
// CALLER RESPONSIBILITY — free_fn callbacks:
//   `free_fn` runs on the collector thread that called
//   `ebr_try_advance_and_collect`.  Heavy work (malloc/free, disk I/O,
//   locking) in `free_fn` stalls the collector and pushes back pressure
//   onto every shard.  Keep free_fn small: `std::free` / arena-pool
//   release / a single `delete` is fine.  Anything heavier should push
//   the obj onto a separate deferred queue and return immediately.
//
// Canonical EBR references:
//   Fraser, "Practical Lock-Freedom" (PhD thesis, Cambridge 2004)
//   McKenney, "Is Parallel Programming Hard?" (perfbook, RCU chapters)
//   Harris, "Pragmatic Implementation of Non-Blocking Linked-Lists" (DISC 2001)

#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>

#include "bolt/bolt_port.h"

namespace bolt {

// ---------------------------------------------------------------------------
// Compile-time bounds. No runtime growth — every loop, every queue is
// capped at init time (Tiger Style).
// ---------------------------------------------------------------------------
constexpr uint32_t kEbrMaxShards      = 64;   // one slot per worker thread
constexpr uint32_t kEbrRetirePerEpoch = 256;  // bounded retire queue / epoch
constexpr uint64_t kEbrUnpinned       = UINT64_MAX;

static_assert(kEbrMaxShards      > 0, "at least one shard");
static_assert(kEbrRetirePerEpoch > 0, "at least one retire slot");

// ---------------------------------------------------------------------------
// Retired-object descriptor. POD. No dtor runs — all cleanup goes through
// the function pointer stored here.
// ---------------------------------------------------------------------------
struct EbrNode {
    void* obj;
    void (*free_fn)(void*);
};

// ---------------------------------------------------------------------------
// Per-shard state. One per worker thread (MarbleDB maps worker_id to
// shard_id). Pad to cache-line to avoid false sharing between shards.
//
// Layout rationale (false-sharing avoidance):
//   `local_epoch` is WRITTEN by the reader on enter/exit and READ by the
//   collector during `ebr_min_pinned_slot` / `ebr_try_advance_and_collect`.
//   `head[]`/`tail[]` and `nodes[]` are producer-owned (written on
//   retire, read by the collector only during drain, and drain already
//   implies the epoch is quiesced). Keep `local_epoch` on its own
//   64-byte cache line so the collector's scan of all shards does not
//   bounce the producer's cursor lines.
//
// Ring depth per epoch is `kEbrRetirePerEpoch`; `head - tail` always fits
// in uint32_t because we bounds-check on push.
// ---------------------------------------------------------------------------
struct alignas(64) EbrShard {
    // --- cache line 0: cross-thread pin slot, isolated from producer data ---
    alignas(64) std::atomic<uint64_t> local_epoch;  // 0|1|2 or kEbrUnpinned
    // Pad the rest of this cache line so the collector's scan of
    // `local_epoch` does not fetch any producer-owned bytes.
    char _pad_pin[64 - sizeof(std::atomic<uint64_t>)];

    // --- cache line 1: producer-owned cursors ---
    alignas(64) uint32_t head[3];                   // write cursor / epoch
    uint32_t              tail[3];                  // read cursor / epoch
    uint32_t              _pad_hdr;                 // legacy slot; keeps ABI

    // --- payload storage, explicit cache-line boundary ---
    alignas(64) EbrNode nodes[kEbrRetirePerEpoch * 3];  // [epoch][slot]
};

// ---------------------------------------------------------------------------
// Global EBR context. Cache-line pad every atomic that crosses threads.
// `global_epoch` is monotone — only the collector increments it, and only
// via CAS so two concurrent collectors cannot stomp each other.
// ---------------------------------------------------------------------------
struct alignas(64) Ebr {
    alignas(64) std::atomic<uint64_t> global_epoch;
    alignas(64) std::atomic<uint32_t> collector_busy;  // 0 = free, 1 = draining
    uint32_t                          num_shards;
    uint32_t                          _pad_hdr;
    EbrShard                          shards[kEbrMaxShards];
};

// ---------------------------------------------------------------------------
// Helpers — internal, kept short so primary fns stay ≤70 lines.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE uint32_t ebr_epoch_slot(uint64_t epoch) noexcept {
    return static_cast<uint32_t>(epoch % 3u);
}

// Return the smallest local_epoch across shards, mapped onto [0,2] as the
// mod-3 slot still in use. Returns UINT32_MAX if no shard is pinned.
BOLT_FORCE_INLINE uint32_t ebr_min_pinned_slot(const Ebr* e) noexcept {
    assert(e != nullptr);
    assert(e->num_shards <= kEbrMaxShards);

    uint32_t seen_mask = 0;  // bit i set ⇒ some shard pinned to slot i
    for (uint32_t s = 0; s < e->num_shards; ++s) {
        const uint64_t le = e->shards[s].local_epoch.load(std::memory_order_acquire);
        if (le == kEbrUnpinned) continue;
        seen_mask |= (1u << ebr_epoch_slot(le));
    }
    if (seen_mask == 0) return UINT32_MAX;  // nobody pinned
    // Return any slot that IS pinned — callers only care "is slot X free?".
    return bolt_ctz32(seen_mask);
}

// Drain a specific epoch's retire ring across all shards. Calls free_fn
// for every queued object. Returns number freed. Bounded work: at most
// `num_shards * kEbrRetirePerEpoch`.
BOLT_FORCE_INLINE uint32_t ebr_drain_epoch(Ebr* e, uint32_t slot) noexcept {
    assert(e != nullptr);
    assert(slot < 3);

    uint32_t freed = 0;
    for (uint32_t s = 0; s < e->num_shards; ++s) {
        EbrShard* sh = &e->shards[s];
        while (sh->tail[slot] != sh->head[slot]) {
            const uint32_t idx = slot * kEbrRetirePerEpoch
                               + (sh->tail[slot] % kEbrRetirePerEpoch);
            EbrNode n = sh->nodes[idx];
            assert(n.free_fn != nullptr);
            n.free_fn(n.obj);
            sh->tail[slot]++;
            freed++;
        }
    }
    return freed;
}

// ---------------------------------------------------------------------------
// Init / destroy.
// ---------------------------------------------------------------------------
inline void ebr_init(Ebr* e, uint32_t num_shards) noexcept {
    assert(e != nullptr);
    assert(num_shards > 0 && num_shards <= kEbrMaxShards);

    e->global_epoch.store(0, std::memory_order_relaxed);
    e->collector_busy.store(0, std::memory_order_relaxed);
    e->num_shards = num_shards;
    e->_pad_hdr   = 0;

    for (uint32_t s = 0; s < kEbrMaxShards; ++s) {
        EbrShard* sh = &e->shards[s];
        sh->local_epoch.store(kEbrUnpinned, std::memory_order_relaxed);
        for (uint32_t k = 0; k < 3; ++k) { sh->head[k] = 0; sh->tail[k] = 0; }
        sh->_pad_hdr = 0;
        for (uint32_t i = 0; i < kEbrRetirePerEpoch * 3; ++i) {
            sh->nodes[i].obj = nullptr;
            sh->nodes[i].free_fn = nullptr;
        }
    }
}

// Drain every remaining retire slot. Caller guarantees no readers remain.
inline void ebr_destroy(Ebr* e) noexcept {
    assert(e != nullptr);
    assert(e->num_shards > 0 && e->num_shards <= kEbrMaxShards);

    // Unpin everyone so the collector path isn't blocked on a stale reader.
    for (uint32_t s = 0; s < e->num_shards; ++s) {
        e->shards[s].local_epoch.store(kEbrUnpinned, std::memory_order_release);
    }
    // Drain all three slots unconditionally.
    for (uint32_t slot = 0; slot < 3; ++slot) {
        (void)ebr_drain_epoch(e, slot);
    }
    e->num_shards = 0;
}

// ---------------------------------------------------------------------------
// Reader side. `ebr_enter` pins the current global epoch into the shard's
// local slot; `ebr_exit` unpins. The happens-before chain is:
//
//   producer_publish(obj) ──release──► global_epoch advances eventually
//                                             │
//                                             ▼
//                                  reader sees advanced epoch
//   reader's ebr_enter acquire-fences any subsequent load of `obj` to
//   happen after the pin store is visible to the collector. Therefore a
//   reader that loads `obj` after pinning cannot race with the drain of
//   the epoch the object was retired in.
//
// Invariant (unobserved-retire): if a reader pins epoch E after the
// retire is enqueued, it still sees a valid `obj` because retire-to-free
// crosses TWO epoch advances (E → E+1 → E+2 drains E-1 mod 3 = E). The
// reader pinned at E can still observe the object; the drain of E cannot
// begin until every reader pinned to E has exited.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE void ebr_enter(Ebr* e, uint32_t shard_id) noexcept {
    assert(e != nullptr);
    assert(shard_id < e->num_shards);

    // Read the current global epoch with acquire — we need to observe
    // any prior collector advance before we commit our pin.
    const uint64_t ge = e->global_epoch.load(std::memory_order_acquire);
    // Publish the pin with release: the collector's acquire load of
    // local_epoch in ebr_try_advance_and_collect / ebr_min_pinned_slot
    // synchronises-with this store.
    e->shards[shard_id].local_epoch.store(ge, std::memory_order_release);
    // Acquire fence: subsequent loads of the protected object cannot be
    // reordered before the pin store from this thread's point of view.
    // Combined with the release above this gives the StoreLoad barrier
    // EBR requires on weak memory models (ARM/POWER).
    std::atomic_thread_fence(std::memory_order_acquire);
}

BOLT_FORCE_INLINE void ebr_exit(Ebr* e, uint32_t shard_id) noexcept {
    assert(e != nullptr);
    assert(shard_id < e->num_shards);

    // Release: any read the reader did of the protected object happens
    // before the collector observes the unpinned slot.
    e->shards[shard_id].local_epoch.store(kEbrUnpinned, std::memory_order_release);
}

// ---------------------------------------------------------------------------
// Producer side. Enqueues `obj` into the retire queue for the current
// global epoch on `shard_id`. Returns false if that shard's queue is
// full — caller either backs off or forces a collect.
// ---------------------------------------------------------------------------
inline bool ebr_retire(Ebr* e,
                       uint32_t shard_id,
                       void* obj,
                       void (*free_fn)(void*)) noexcept {
    assert(e != nullptr);
    assert(shard_id < e->num_shards);
    assert(obj != nullptr);
    assert(free_fn != nullptr);

    const uint64_t ge   = e->global_epoch.load(std::memory_order_acquire);
    const uint32_t slot = ebr_epoch_slot(ge);
    EbrShard*      sh   = &e->shards[shard_id];

    const uint32_t used = sh->head[slot] - sh->tail[slot];
    if (used >= kEbrRetirePerEpoch) return false;

    const uint32_t idx = slot * kEbrRetirePerEpoch
                       + (sh->head[slot] % kEbrRetirePerEpoch);
    sh->nodes[idx].obj     = obj;
    sh->nodes[idx].free_fn = free_fn;
    sh->head[slot]++;
    return true;
}

// ---------------------------------------------------------------------------
// Collector. Intended to run periodically from a background task.
//
// Algorithm:
//   1) CAS collector_busy 0→1. If another collector is draining, return 0.
//   2) Read ge = global_epoch.
//   3) Target drain slot is (ge + 1) mod 3 — the slot two epochs behind
//      the current (ge → ge-1 → drain ge-2; mod 3 that is ge+1).
//   4) Scan every shard's local_epoch. If any shard is pinned to that
//      slot, abort (try again later).
//   5) Drain that slot across all shards.
//   6) CAS-advance global_epoch from ge to ge+1. Failure means another
//      collector beat us — still fine, we drained safely.
//   7) Release collector_busy.
// Returns number of objects freed.
// ---------------------------------------------------------------------------
inline uint32_t ebr_try_advance_and_collect(Ebr* e) noexcept {
    assert(e != nullptr);
    assert(e->num_shards > 0 && e->num_shards <= kEbrMaxShards);

    uint32_t expected = 0;
    if (!e->collector_busy.compare_exchange_strong(
            expected, 1,
            std::memory_order_acquire,
            std::memory_order_relaxed)) {
        return 0;
    }

    const uint64_t ge         = e->global_epoch.load(std::memory_order_acquire);
    const uint32_t drain_slot = ebr_epoch_slot(ge + 1);  // i.e. ge - 2 mod 3

    // If any shard is pinned to the slot we want to drain, abort.
    for (uint32_t s = 0; s < e->num_shards; ++s) {
        const uint64_t le = e->shards[s].local_epoch.load(std::memory_order_acquire);
        if (le == kEbrUnpinned) continue;
        if (ebr_epoch_slot(le) == drain_slot) {
            e->collector_busy.store(0, std::memory_order_release);
            return 0;
        }
    }

    const uint32_t freed = ebr_drain_epoch(e, drain_slot);

    // Release so readers entering after this point see the advance.
    uint64_t expected_ge = ge;
    (void)e->global_epoch.compare_exchange_strong(
        expected_ge, ge + 1,
        std::memory_order_release,
        std::memory_order_relaxed);

    e->collector_busy.store(0, std::memory_order_release);
    return freed;
}

}  // namespace bolt
