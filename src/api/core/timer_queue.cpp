// timer_queue — see header. A binary min-heap of deadlines with lazy
// cancellation via a live-id set. STATION-82.

#include "bolt/api/core/timer_queue.h"

#include <algorithm>
#include <cassert>
#include <utility>

namespace bolt {
namespace api {
namespace core {

namespace {
// Timer registries are per-reactor-loop (one instance, not per-connection),
// so a small initial block is enough; growth headroom mirrors FdRegistry's.
bolt::ArenaConfig timer_ids_arena_config() noexcept {
    bolt::ArenaConfig cfg;
    cfg.initial_block_size = 16u * 1024;
    cfg.max_block_size = 16u * 1024 * 1024;
    return cfg;
}
}  // namespace

timer_queue::timer_queue() noexcept
    : live_ids_arena_(timer_ids_arena_config()) {
    const bool ok = bolt::SwissTableGrowable::create(
        &live_ids_, 64, &live_ids_arena_, kMaxLiveTimers);
    assert(ok && "timer_queue live-id table creation cannot fail at 64 slots");
    assert(live_ids_.size == 0);
    (void)ok;
}

timer_queue::timer_id timer_queue::add(duration timeout, timer_callback cb,
                                       bool repeating) noexcept {
    assert(cb && "timer_queue::add requires a callback");
    // A repeating timer must have a positive interval or it would busy-loop.
    const duration interval = repeating ? std::max(timeout, duration(1)) : duration::zero();
    const time_point deadline = clock::now() + (timeout > duration::zero() ? timeout : duration::zero());

    const timer_id id = next_id_++;
    if (next_id_ == INVALID_ID) ++next_id_;  // never hand out INVALID_ID on wrap
    // Value is unused (set-only usage); a failed insert only happens at the
    // kMaxLiveTimers hard ceiling, in which case the id is simply never
    // "live" — drop_dead_top()/fire_expired() already treat that as an
    // already-cancelled timer, so this degrades exactly like a lazy cancel.
    const bool inserted = live_ids_.insert(id, 1u);
    assert((inserted || live_ids_.size >= kMaxLiveTimers) &&
           "unexpected timer_queue insert failure below the ceiling");
    (void)inserted;
    heap_.push_back(entry{deadline, interval, id, std::move(cb), repeating});
    std::push_heap(heap_.begin(), heap_.end(), later_first{});
    assert(!heap_.empty());
    return id;
}

bool timer_queue::cancel(timer_id id) noexcept {
    // Lazy: forget the id now; the stale heap entry is dropped when it surfaces.
    return live_ids_.erase(id);
}

void timer_queue::drop_dead_top() noexcept {
    while (!heap_.empty() && live_ids_.find(heap_.front().id) < 0) {
        std::pop_heap(heap_.begin(), heap_.end(), later_first{});
        heap_.pop_back();
    }
}

uint64_t timer_queue::next_timeout_us(time_point now) noexcept {
    drop_dead_top();
    if (heap_.empty()) return NO_TIMEOUT;
    const time_point deadline = heap_.front().deadline;
    if (deadline <= now) return 0;
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(deadline - now).count();
    return us < 0 ? 0 : static_cast<uint64_t>(us);
}

void timer_queue::fire_expired(time_point now) {
    // Bound the loop by the entry count captured up front: a repeating timer
    // re-arms with a future deadline, so it cannot fire twice in one pass, and
    // callbacks that add near-future timers won't be serviced until next pass.
    size_t budget = heap_.size();
    while (budget-- > 0) {
        drop_dead_top();
        if (heap_.empty() || heap_.front().deadline > now) break;

        std::pop_heap(heap_.begin(), heap_.end(), later_first{});
        entry due = std::move(heap_.back());
        heap_.pop_back();

        // Re-check liveness: a prior callback in this pass may have cancelled it.
        if (live_ids_.find(due.id) < 0) continue;

        if (due.repeating) {
            // Re-arm relative to the intended deadline; if we've fallen behind
            // more than one interval, skip forward to now+interval to avoid a
            // catch-up storm (matches libuv/Seastar behaviour).
            time_point next = due.deadline + due.interval;
            if (next <= now) next = now + due.interval;
            heap_.push_back(entry{next, due.interval, due.id, due.cb, true});
            std::push_heap(heap_.begin(), heap_.end(), later_first{});
        } else {
            live_ids_.erase(due.id);
        }
        due.cb();  // fire last so a re-arm is already in the heap
    }
}

}  // namespace core
}  // namespace api
}  // namespace bolt
