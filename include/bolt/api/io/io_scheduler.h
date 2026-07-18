#pragma once
// io_scheduler — the I/O scheduler + reactor QoS layer (STATION-85).
//
// A REUSABLE, STANDALONE scheduler primitive that belongs BETWEEN an async I/O
// API (bolt::api::io::file_io, the socket layer, …) and the kernel ring
// (IOCP / io_uring / epoll / kqueue). It is the backend-agnostic QoS layer that
// is the load-bearing value borrowed from Seastar (reference: seastar
// src/core/io_queue.cc + include/seastar/core/fair_queue.hh). Rather than
// submitting each `co_await` straight into the ring, a caller ENQUEUES the
// request here; the reactor loop then calls poll_io_queue() once per tick to
// decide how many/which requests to hand to the kernel this tick.
//
// What it gives you (the four Seastar core ideas, v1):
//   1. fair_queue selection — priority CLASSES with shares; the class that has
//      consumed the least capacity relative to its shares is served next, so
//      device bandwidth is shared proportionally and one workload cannot
//      starve another. (fair_queue.h)
//   2. Bounded in-flight depth — a hard `max_in_flight` cap on concurrently
//      dispatched requests. Requests beyond the cap WAIT in the fair_queue
//      (backpressure); the device queue is never overflowed. THE core value.
//   3. io_intent cancellation — enqueue() returns an io_intent; because a
//      request sits in the queue before the kernel, io_intent::cancel() drops a
//      still-queued request for FREE (an O(1) fair_queue unlink). The natural
//      fit for coroutine cancellation / timeouts.
//   4. poll_io_queue() dispatch step — the reactor loop calls it each tick; it
//      pulls from the fair_queue up to the in-flight (and optional token)
//      budget and hands each request to a caller-supplied submit functor.
//      complete() accounts a finished op and frees depth so the next tick can
//      dispatch more.
//
// Token-bucket device budget (Seastar idea #2, optionally enabled): when
// `tokens_per_tick > 0`, poll_io_queue also meters total dispatched COST per
// tick against a refilling bucket (`bucket_capacity`), so a burst cannot flood
// the device even under the in-flight cap. Cost is the caller-supplied scalar
// passed to enqueue() (map a weight+size model onto it — see fair_queue.h). The
// flow-ratio EMA backpressure / adaptive stall detector (Seastar idea #4) is
// NOT implemented — see the TODO below for the interface seam.
//
// Explicitly OUT OF SCOPE for v1 (per the ticket): shard-per-core. This is a
// single shared/global scheduler; a per-shard split is a later refactor that
// does not change this interface.
//
// ---------------------------------------------------------------------------
// WIRING SEAM — how file_io / async_io route THROUGH this later (follow-on).
// ---------------------------------------------------------------------------
// This layer is deliberately NOT wired into file.cpp / async_io.cpp yet (those
// TUs are freshly shipped and concurrently edited; forcing the wiring now risks
// destabilizing them). The intended integration, when it lands:
//
//   • file_io::submit_read/write/fsync currently submit straight to the backend
//     / worker pool. Instead they would: (a) enqueue() a request on a shared
//     io_scheduler with a priority class + a cost derived from the byte size,
//     stashing the OVERLAPPED/op token as the `cookie`; (b) store the returned
//     io_intent on the awaitable so a cancelled coroutine drops a still-queued
//     read for free.
//   • The reactor run loop (file_io::poll / EventLoop) would, each tick, call
//     io_scheduler::poll_io_queue([](io_request_ref r){ /* real ReadFile /
//     io_uring_prep_* using r.cookie() */ }) BEFORE reaping completions, so the
//     "reap → dispatch → submit → reap" poller order reuses freed depth in the
//     same tick.
//   • On completion reap, the backend maps the completion back to its request
//     and calls io_scheduler::complete(request_id) (carried alongside the
//     OVERLAPPED), freeing one unit of in-flight depth.
//
// TODO(STATION-85 follow-on): flow-ratio EMA backpressure + adaptive stall
// detector. Seam: add a hook after complete() that feeds (dispatched, completed,
// latency) samples into an EMA and lowers the effective max_in_flight when the
// completion/dispatch flow ratio drops — a self-tuning "why is I/O slow" signal.
// It composes on top of the bounded-depth cap without changing enqueue/poll.
//
// Tiger-Style: noexcept, bounded (fixed request pool), no allocation on the
// dispatch hot path, no RAII-per-thread. Not thread-safe — owned and driven by
// a single reactor loop thread, like the rest of the reactor's per-loop state.

#include "bolt/api/io/fair_queue.h"

#include <cassert>
#include <cstddef>   // offsetof
#include <cstdint>
#include <vector>

namespace bolt {
namespace api {
namespace io {

class io_scheduler;

// -----------------------------------------------------------------------------
// io_request_ref — the handle handed to the submit functor in poll_io_queue.
// -----------------------------------------------------------------------------
// Carries everything the caller needs to actually issue the I/O and to account
// its completion. Cheap value type (four words); copyable.
class io_request_ref {
public:
    // Internal: constructed by io_scheduler::poll_io_queue.
    io_request_ref(io_scheduler* sched, void* cookie, uint64_t id,
                   fair_queue::class_id klass) noexcept
        : sched_(sched), cookie_(cookie), id_(id), klass_(klass) {}

    // The opaque caller context stashed at enqueue() (e.g. an OVERLAPPED* or an
    // op token). The scheduler never dereferences it.
    void* cookie() const noexcept { return cookie_; }

    // Stable id for this dispatched request; pass it to io_scheduler::complete()
    // when the op finishes. Also usable to correlate a completion.
    uint64_t request_id() const noexcept { return id_; }

    // The priority class this request was enqueued against.
    fair_queue::class_id priority_class() const noexcept { return klass_; }

    // Convenience: account this request complete (frees one unit of in-flight
    // depth). Equivalent to sched.complete(request_id()).
    bool complete() const noexcept;

private:
    io_scheduler*        sched_;
    void*                cookie_;
    uint64_t             id_;
    fair_queue::class_id klass_;
};

// -----------------------------------------------------------------------------
// io_intent — the cancellation handle returned by enqueue().
// -----------------------------------------------------------------------------
// Because the request sits in the fair_queue before the kernel, cancelling a
// still-queued request is free. Once dispatched (in flight) it can no longer be
// cancelled — cancel() then returns false.
class io_intent {
public:
    io_intent() noexcept = default;
    io_intent(io_scheduler* sched, uint64_t id) noexcept : sched_(sched), id_(id) {}

    // True iff this handle refers to a request that was successfully enqueued.
    bool valid() const noexcept { return sched_ != nullptr && id_ != 0; }

    // The request's stable id (0 if invalid).
    uint64_t id() const noexcept { return id_; }

    // Cancel the request if it is still queued (not yet dispatched). Returns
    // true iff it was queued and is now cancelled; false if already dispatched,
    // already completed/cancelled, or this handle is invalid/stale.
    bool cancel() noexcept;

private:
    io_scheduler* sched_ = nullptr;
    uint64_t      id_    = 0;
};

// -----------------------------------------------------------------------------
// io_scheduler
// -----------------------------------------------------------------------------
class io_scheduler {
public:
    using class_id   = fair_queue::class_id;
    using capacity_t = fair_queue::capacity_t;

    struct config {
        // Hard cap on concurrently dispatched (in-flight) requests. Requests
        // beyond it wait in the fair_queue. This is the device-queue depth.
        uint32_t max_in_flight = 128;

        // Request-slot pool capacity — the maximum number of simultaneously
        // outstanding (queued + in-flight) requests. enqueue() fails (returns
        // an invalid io_intent) when the pool is exhausted; that is admission
        // backpressure and the caller should retry after progress. Must be >= 1.
        uint32_t max_requests = 4096;

        // Max requests dispatched per poll_io_queue() call (0 = unlimited).
        // Bounds the work a single reactor tick spends submitting, so other
        // pollers get a turn.
        uint32_t max_dispatch_per_tick = 0;

        // Token-bucket device budget (Seastar #2). 0 disables metering (only
        // the in-flight cap applies). When > 0, poll_io_queue refills `tokens`
        // by tokens_per_tick each call (capped at bucket_capacity) and each
        // dispatched request consumes its enqueue() cost.
        capacity_t tokens_per_tick  = 0;
        capacity_t bucket_capacity  = 0;

        // fair_queue balancing config (forgiving factor).
        fair_queue::config fq{};
    };

    explicit io_scheduler(const config& cfg = {});
    ~io_scheduler() = default;

    io_scheduler(const io_scheduler&)            = delete;
    io_scheduler& operator=(const io_scheduler&) = delete;

    // ---- class registration (delegates to the fair_queue) -------------------
    void register_priority_class(class_id id, uint32_t shares) {
        fq_.register_priority_class(id, shares);
    }
    void update_shares(class_id id, uint32_t shares) noexcept {
        fq_.update_shares(id, shares);
    }
    void unregister_priority_class(class_id id) noexcept {
        fq_.unregister_priority_class(id);
    }

    // ---- enqueue ------------------------------------------------------------

    // Enqueue an I/O request against priority class `klass`, stashing `cookie`
    // for the submit functor to pick up at dispatch, with dispatch `cost` (the
    // scalar the fair_queue and token bucket meter — default 1 = pure IOPS
    // fairness). Returns an io_intent for cancellation; the intent is invalid
    // (valid()==false) iff the request pool is exhausted.
    io_intent enqueue(class_id klass, void* cookie, capacity_t cost = 1) noexcept;

    // ---- dispatch (call once per reactor tick) ------------------------------

    // Pull from the fair_queue and dispatch up to the in-flight (and optional
    // token / per-tick) budget, handing each request to `submit`, invoked as
    //     submit(io_request_ref r)
    // The functor performs the real kernel submission (ReadFile, io_uring_prep_*,
    // …) using r.cookie(), and must NOT complete the request inline — completion
    // is reported later via complete()/io_request_ref::complete() when the op
    // actually finishes. Returns the number of requests dispatched this call.
    template <typename SubmitFn>
    uint32_t poll_io_queue(SubmitFn&& submit) noexcept {
        replenish_tokens();
        uint32_t dispatched = 0;
        while (max_dispatch_per_tick_ == 0 || dispatched < max_dispatch_per_tick_) {
            // Bounded in-flight depth: the core backpressure guarantee.
            if (in_flight_ >= max_in_flight_) {
                break;
            }
            fair_queue::entry* fe = fq_.top();
            if (fe == nullptr) {
                break;  // nothing queued
            }
            const capacity_t cost = fe->capacity();
            if (!tokens_ok(cost)) {
                break;  // device token budget exhausted for this tick
            }

            request* r = request_of(fe);
            assert(r->st == request::state::queued && "fair_queue top not queued");

            fq_.pop_front();
            consume_tokens(cost);
            r->st = request::state::in_flight;
            ++in_flight_;
            ++stat_dispatched_;
            assert(in_flight_ <= max_in_flight_ && "in-flight cap violated");

            const uint64_t id = make_id(slot_of(r), r->gen);
            submit(io_request_ref{this, r->cookie, id, r->klass});
            ++dispatched;
        }
        return dispatched;
    }

    // ---- completion / cancellation ------------------------------------------

    // Account a dispatched request complete, freeing one unit of in-flight
    // depth so poll_io_queue can dispatch more. Returns true iff `id` named a
    // live in-flight request. Idempotent-safe against stale ids (returns false).
    bool complete(uint64_t id) noexcept;

    // Cancel a still-queued request. Returns true iff it was queued and is now
    // cancelled. Used by io_intent::cancel().
    bool cancel(uint64_t id) noexcept;

    // ---- introspection ------------------------------------------------------

    uint32_t in_flight() const noexcept { return in_flight_; }
    uint32_t max_in_flight() const noexcept { return max_in_flight_; }
    size_t   queued() const noexcept { return fq_.size(); }
    capacity_t tokens() const noexcept { return tokens_; }

    struct stats {
        uint64_t enqueued   = 0;
        uint64_t dispatched = 0;
        uint64_t completed  = 0;
        uint64_t cancelled  = 0;
        uint64_t rejected   = 0;  // enqueue() calls that hit an exhausted pool
    };
    stats get_stats() const noexcept {
        return stats{stat_enqueued_, stat_dispatched_, stat_completed_,
                     stat_cancelled_, stat_rejected_};
    }

private:
    // A pooled request slot. fq_entry MUST be the first member so request_of()
    // can recover the slot from a fair_queue::entry* by offset.
    struct request {
        enum class state : uint8_t { free, queued, in_flight };
        fair_queue::entry fq_entry;
        void*             cookie    = nullptr;
        class_id          klass     = 0;
        uint32_t          gen       = 1;   // generation; ids stay nonzero
        state             st        = state::free;
        uint32_t          next_free = kNoSlot;
    };

    static constexpr uint32_t kNoSlot = UINT32_MAX;

    // id layout: high 32 bits generation (>=1), low 32 bits slot index. gen>=1
    // guarantees a live id is never 0 (the invalid sentinel).
    static uint64_t make_id(uint32_t slot, uint32_t gen) noexcept {
        return (static_cast<uint64_t>(gen) << 32) | slot;
    }
    static uint32_t slot_of_id(uint64_t id) noexcept {
        return static_cast<uint32_t>(id & 0xFFFFFFFFu);
    }
    static uint32_t gen_of_id(uint64_t id) noexcept {
        return static_cast<uint32_t>(id >> 32);
    }

    uint32_t slot_of(const request* r) const noexcept {
        return static_cast<uint32_t>(r - pool_.data());
    }
    request* request_of(fair_queue::entry* fe) noexcept {
        return reinterpret_cast<request*>(
            reinterpret_cast<char*>(fe) - offsetof(request, fq_entry));
    }

    void free_slot(request& r, uint32_t slot) noexcept {
        r.st     = request::state::free;
        r.cookie = nullptr;
        ++r.gen;
        if (r.gen == 0) {
            r.gen = 1;  // never reuse gen 0 — keeps ids nonzero
        }
        r.next_free = free_head_;
        free_head_  = slot;
    }

    // ---- token bucket -------------------------------------------------------
    void replenish_tokens() noexcept {
        if (tokens_per_tick_ == 0) {
            return;  // metering disabled
        }
        tokens_ += tokens_per_tick_;
        if (tokens_ > bucket_capacity_) {
            tokens_ = bucket_capacity_;
        }
    }
    bool tokens_ok(capacity_t cost) const noexcept {
        if (tokens_per_tick_ == 0) {
            return true;
        }
        if (tokens_ >= cost) {
            return true;
        }
        // A single request larger than the whole bucket must still make
        // progress once the bucket is full, else it would deadlock forever.
        return tokens_ == bucket_capacity_;
    }
    void consume_tokens(capacity_t cost) noexcept {
        if (tokens_per_tick_ == 0) {
            return;
        }
        tokens_ = (tokens_ > cost) ? (tokens_ - cost) : 0;
    }

    fair_queue            fq_;
    std::vector<request>  pool_;
    uint32_t              free_head_ = kNoSlot;

    uint32_t   max_in_flight_;
    uint32_t   max_dispatch_per_tick_;
    uint32_t   in_flight_ = 0;

    capacity_t tokens_per_tick_;
    capacity_t bucket_capacity_;
    capacity_t tokens_ = 0;

    uint64_t stat_enqueued_   = 0;
    uint64_t stat_dispatched_ = 0;
    uint64_t stat_completed_  = 0;
    uint64_t stat_cancelled_  = 0;
    uint64_t stat_rejected_   = 0;
};

// ---- out-of-line handle methods (need the complete io_scheduler type) -------
inline bool io_request_ref::complete() const noexcept {
    return sched_->complete(id_);
}
inline bool io_intent::cancel() noexcept {
    return sched_ != nullptr && sched_->cancel(id_);
}

}  // namespace io
}  // namespace api
}  // namespace bolt
