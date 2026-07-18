#pragma once
// timer_queue — backend-agnostic timer facility for the bolt reactor
// (STATION-82). async_io's per-OS backends (IOCP/epoll/io_uring/kqueue) have
// no native timer surface (io_op::timer is declared but unimplemented), so
// timers live one layer up, exactly as libuv/Seastar do: a min-heap of
// deadlines that the run loop consults to bound its wait, then drains.
//
// Usage in a reactor run loop (e.g. bolt EventLoop / llm-station EventLoop):
//     for (;;) {
//         auto now = timer_queue::clock::now();
//         tq.fire_expired(now);                     // run due callbacks
//         uint64_t us = tq.next_timeout_us(now);    // NO_TIMEOUT if empty
//         engine.poll(us == timer_queue::NO_TIMEOUT ? default_us
//                                                   : static_cast<uint32_t>(us));
//         // ... drain ready I/O / coroutines ...
//     }
//
// A coroutine `co_await sleep_for(tq, 50ms)` registers a one-shot timer whose
// callback resumes the awaiting coroutine on the loop thread.
//
// Not thread-safe: like the rest of the reactor's per-loop state, a
// timer_queue is owned and driven by a single loop thread. Cross-thread
// scheduling goes through async_io::wake() + the loop's own queue.

#include <chrono>
#include <coroutine>
#include <cstdint>
#include <functional>
#include <unordered_set>
#include <vector>

namespace bolt {
namespace api {
namespace core {

class timer_queue {
public:
    using clock          = std::chrono::steady_clock;
    using time_point     = clock::time_point;
    using duration       = clock::duration;
    using timer_id       = uint64_t;
    using timer_callback = std::function<void()>;

    // Returned by next_timeout_us() when no timer is armed.
    static constexpr uint64_t NO_TIMEOUT = UINT64_MAX;
    // A timer_id that never denotes a live timer (add() never returns it).
    static constexpr timer_id INVALID_ID = 0;

    timer_queue() = default;
    timer_queue(const timer_queue&)            = delete;
    timer_queue& operator=(const timer_queue&) = delete;

    // Arm a timer to fire `timeout` from now. If `repeating`, it re-arms every
    // `timeout` after each firing. Returns a live id (never INVALID_ID).
    timer_id add(duration timeout, timer_callback cb, bool repeating = false) noexcept;

    // Cancel an armed timer (lazy: dropped when it reaches the heap top).
    // Returns true if the id was live and not already cancelled.
    bool cancel(timer_id id) noexcept;

    // Microseconds until the next live timer is due, saturating so that an
    // already-due timer yields 0. NO_TIMEOUT when no timer is armed. Prunes
    // cancelled timers off the top as a side effect (hence non-const).
    uint64_t next_timeout_us(time_point now) noexcept;

    // Invoke the callback of every timer due at/before `now`, re-arming
    // repeaters. Cancelled timers are skipped. A callback may add/cancel
    // timers re-entrantly.
    void fire_expired(time_point now);

    size_t size() const noexcept { return live_ids_.size(); }
    bool   empty() const noexcept { return live_ids_.empty(); }

private:
    struct entry {
        time_point     deadline;
        duration       interval;   // zero for one-shot
        timer_id       id;
        timer_callback cb;
        bool           repeating;
    };
    // Min-heap by deadline: `entry a` is "greater" when it is due LATER, so
    // std::*_heap (a max-heap) surfaces the soonest deadline at the top.
    struct later_first {
        bool operator()(const entry& a, const entry& b) const noexcept {
            return a.deadline > b.deadline;
        }
    };

    void drop_dead_top() noexcept;  // pop entries whose id is no longer live

    std::vector<entry>            heap_;
    std::unordered_set<timer_id>  live_ids_;  // ids currently armed
    timer_id                      next_id_ = 1;
};

// co_await sleep_for(tq, 50ms) — suspends the coroutine and resumes it once the
// loop driving `tq` fires the timer. A non-positive duration never suspends.
struct sleep_awaitable {
    timer_queue&           tq;
    timer_queue::duration  d;

    bool await_ready() const noexcept { return d <= timer_queue::duration::zero(); }
    void await_suspend(std::coroutine_handle<> h) noexcept {
        tq.add(d, [h]() noexcept { h.resume(); });
    }
    void await_resume() const noexcept {}
};

inline sleep_awaitable sleep_for(timer_queue& tq, timer_queue::duration d) noexcept {
    return sleep_awaitable{tq, d};
}

}  // namespace core
}  // namespace api
}  // namespace bolt
