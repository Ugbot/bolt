// io_scheduler.cpp — enqueue / complete / cancel + construction for the
// STATION-85 I/O scheduler. The hot dispatch step (poll_io_queue) and the token
// bucket are header-inline (io_scheduler.h) so submission stays allocation-free
// and the submit functor is not type-erased; this TU holds the pool lifecycle
// and the request state transitions.

#include "bolt/api/io/io_scheduler.h"

#include <algorithm>

namespace bolt {
namespace api {
namespace io {

io_scheduler::io_scheduler(const config& cfg)
    : fq_(cfg.fq),
      max_in_flight_(cfg.max_in_flight != 0 ? cfg.max_in_flight : 1),
      max_dispatch_per_tick_(cfg.max_dispatch_per_tick),
      tokens_per_tick_(cfg.tokens_per_tick),
      bucket_capacity_(cfg.bucket_capacity) {
    const uint32_t cap = cfg.max_requests != 0 ? cfg.max_requests : 1;

    // A full bucket to start so the first tick can dispatch immediately (and so
    // a request larger than tokens_per_tick can make progress). If the caller
    // left bucket_capacity 0 while enabling metering, fall back to one tick's
    // worth so the budget is usable rather than a permanent zero.
    if (tokens_per_tick_ != 0 && bucket_capacity_ == 0) {
        bucket_capacity_ = tokens_per_tick_;
    }
    tokens_ = bucket_capacity_;

    // Build the fixed request pool and thread the free-list. Stable addresses:
    // pool_ is sized once here and never resized, so the fair_queue's entry
    // pointers (which alias slot storage) stay valid for the scheduler's life.
    pool_.resize(cap);
    for (uint32_t i = 0; i < cap; ++i) {
        pool_[i].st        = request::state::free;
        pool_[i].gen       = 1;
        pool_[i].next_free = (i + 1 < cap) ? (i + 1) : kNoSlot;
    }
    free_head_ = 0;
}

io_intent io_scheduler::enqueue(class_id klass, void* cookie, capacity_t cost) noexcept {
    assert(fq_.class_registered(klass) && "enqueue against unregistered class");

    if (free_head_ == kNoSlot) {
        ++stat_rejected_;
        return io_intent{};  // pool exhausted — admission backpressure
    }

    const uint32_t slot = free_head_;
    request& r = pool_[slot];
    assert(r.st == request::state::free && "free-list head not free");
    free_head_ = r.next_free;

    r.cookie = cookie;
    r.klass  = klass;
    r.st     = request::state::queued;
    r.fq_entry.set_capacity(cost);
    fq_.queue(klass, r.fq_entry);

    ++stat_enqueued_;
    return io_intent{this, make_id(slot, r.gen)};
}

bool io_scheduler::complete(uint64_t id) noexcept {
    const uint32_t slot = slot_of_id(id);
    if (slot >= pool_.size()) {
        return false;
    }
    request& r = pool_[slot];
    if (r.gen != gen_of_id(id) || r.st != request::state::in_flight) {
        return false;  // stale id, or not actually in flight
    }

    assert(in_flight_ > 0 && "complete with zero in-flight");
    --in_flight_;
    free_slot(r, slot);
    ++stat_completed_;
    return true;
}

bool io_scheduler::cancel(uint64_t id) noexcept {
    const uint32_t slot = slot_of_id(id);
    if (slot >= pool_.size()) {
        return false;
    }
    request& r = pool_[slot];
    if (r.gen != gen_of_id(id) || r.st != request::state::queued) {
        return false;  // stale id, or already dispatched/completed (too late)
    }

    fq_.cancel(r.fq_entry);   // O(1) unlink — the "cancel while queued is free"
    free_slot(r, slot);
    ++stat_cancelled_;
    return true;
}

}  // namespace io
}  // namespace api
}  // namespace bolt
