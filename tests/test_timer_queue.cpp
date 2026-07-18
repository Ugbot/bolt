// Unit tests for bolt::api::core::timer_queue (STATION-82).
// Self-contained (assert-based); compile with timer_queue.cpp.

#include "bolt/api/core/timer_queue.h"

#include <cassert>
#include <chrono>
#include <coroutine>
#include <cstdio>
#include <vector>

using namespace std::chrono_literals;
using bolt::api::core::timer_queue;
using bolt::api::core::sleep_for;

namespace {

using clock = timer_queue::clock;

void test_fire_order() {
    timer_queue tq;
    std::vector<int> fired;
    tq.add(30ms, [&] { fired.push_back(30); });
    tq.add(10ms, [&] { fired.push_back(10); });
    tq.add(20ms, [&] { fired.push_back(20); });
    assert(tq.size() == 3);
    tq.fire_expired(clock::now() + 100ms);
    assert((fired == std::vector<int>{10, 20, 30}));  // soonest-first
    assert(tq.empty());
}

void test_next_timeout() {
    timer_queue tq;
    assert(tq.next_timeout_us(clock::now()) == timer_queue::NO_TIMEOUT);
    tq.add(50ms, [] {});
    const auto now = clock::now();
    const uint64_t us = tq.next_timeout_us(now);
    assert(us != timer_queue::NO_TIMEOUT);
    assert(us > 0 && us <= 60'000);          // ~50ms, generous upper bound
    // A due timer yields 0, not a huge number.
    assert(tq.next_timeout_us(now + 100ms) == 0);
}

void test_cancel() {
    timer_queue tq;
    bool fired = false;
    auto id = tq.add(10ms, [&] { fired = true; });
    assert(tq.cancel(id));
    assert(!tq.cancel(id));                    // second cancel is a no-op
    tq.fire_expired(clock::now() + 100ms);
    assert(!fired);
    assert(tq.empty());
}

void test_one_shot_fires_once() {
    timer_queue tq;
    int count = 0;
    tq.add(10ms, [&] { ++count; });
    tq.fire_expired(clock::now() + 50ms);
    tq.fire_expired(clock::now() + 100ms);
    assert(count == 1);
    assert(tq.empty());
}

void test_repeating() {
    timer_queue tq;
    int count = 0;
    auto id = tq.add(10ms, [&] { ++count; }, /*repeating=*/true);
    const auto t0 = clock::now();
    tq.fire_expired(t0 + 15ms);   // one interval elapsed -> fires once, re-arms
    assert(count == 1);
    assert(tq.size() == 1);       // still armed
    tq.fire_expired(t0 + 100ms);  // fires again (skips catch-up storm)
    assert(count == 2);
    assert(tq.cancel(id));
    tq.fire_expired(t0 + 1s);
    assert(count == 2);           // cancelled -> no more
}

void test_cancel_from_callback() {
    timer_queue tq;
    int a = 0, b = 0;
    timer_queue::timer_id idb = 0;
    tq.add(10ms, [&] { ++a; tq.cancel(idb); });   // fires first, cancels b
    idb = tq.add(20ms, [&] { ++b; });
    tq.fire_expired(clock::now() + 100ms);
    assert(a == 1 && b == 0);
}

// Minimal eager coroutine for the sleep_awaitable path.
struct eager_task {
    struct promise_type {
        eager_task get_return_object() noexcept { return {}; }
        std::suspend_never initial_suspend() noexcept { return {}; }
        std::suspend_never final_suspend() noexcept { return {}; }
        void return_void() noexcept {}
        void unhandled_exception() noexcept {}
    };
};

void test_sleep_awaitable() {
    timer_queue tq;
    bool done = false;
    auto co = [&]() -> eager_task {
        co_await sleep_for(tq, 10ms);
        done = true;
    };
    co();
    assert(!done);                          // suspended on the timer
    assert(tq.size() == 1);
    tq.fire_expired(clock::now() + 50ms);   // fires -> resumes coroutine
    assert(done);
    assert(tq.empty());
}

}  // namespace

int main() {
    test_fire_order();
    test_next_timeout();
    test_cancel();
    test_one_shot_fires_once();
    test_repeating();
    test_cancel_from_callback();
    test_sleep_awaitable();
    std::printf("timer_queue: all 7 tests passed\n");
    return 0;
}
