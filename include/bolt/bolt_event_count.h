// bolt_event_count.h — park/notify beside a lock-free queue, without lost wakeups.
//
//   consumer                            producer
//   --------                            --------
//   ec.await(spins, [&] {               queue.try_push(x);   // publish first
//       return queue.try_pop(&x);       ec.notify_one();
//   });
//
// The consumer announces itself, fences and re-checks before it sleeps; the
// producer publishes, fences and looks for sleepers. One of the two fences is
// first in the seq_cst order, so either the producer sees the sleeper or the
// sleeper sees the item. With nobody asleep notify costs a fence and a load:
// no syscall and no shared write.
//
// A notify can wake a thread that finds nothing (another consumer took the
// item, or an earlier slot of an MPMC ring is claimed and not yet published).
// That thread parks again, so a consumer that did get an item and can see more
// must notify_one() before it runs it, or the spare item waits for it.

#pragma once

#include "bolt/bolt_port.h"

#include <atomic>
#include <cassert>
#include <cstdint>

namespace bolt {

// The word the standard library hands straight to the kernel (__ulock 64-bit,
// futex 32-bit, WaitOnAddress any). Other widths wait on a shared hash bucket,
// where notify_one degrades to waking every waiter of the bucket.
#if defined(__APPLE__) || (defined(__FreeBSD__) && defined(__LP64__))
using ParkWord = int64_t;
#else
using ParkWord = int32_t;
#endif

struct EventCount {
    using Key = ParkWord;

    EventCount() noexcept : epoch_(0), sleepers_(0) {}
    EventCount(const EventCount&)            = delete;
    EventCount& operator=(const EventCount&) = delete;

    // Announce, then re-check the condition; follow with park(key) or cancel().
    [[nodiscard]] Key prepare() noexcept {
        const uint32_t before = sleepers_.fetch_add(1, std::memory_order_seq_cst);
        assert(before < UINT32_MAX);
        (void)before;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        return epoch_.load(std::memory_order_acquire);
    }

    void cancel() noexcept {
        const uint32_t before = sleepers_.fetch_sub(1, std::memory_order_seq_cst);
        assert(before > 0);
        (void)before;
    }

    // Returns once a notify after prepare() was observed (or spuriously).
    void park(Key key) noexcept {
        epoch_.wait(key, std::memory_order_acquire);
        cancel();
    }

    void notify_one() noexcept {
        if (bump()) epoch_.notify_one();
    }

    void notify_all() noexcept {
        if (bump()) epoch_.notify_all();
    }

    // Returns when `ready()` does. ready() is the pop itself, so it runs once
    // per success; `spins` bounds the poll before each park.
    template <typename Ready>
    void await(uint32_t spins, Ready&& ready) noexcept(noexcept(ready())) {
        for (;;) {
            for (uint32_t i = 0; i <= spins; ++i) {
                if (ready()) return;
                BOLT_PAUSE();
            }
            const Key key = prepare();
            if (ready()) {
                cancel();
                return;
            }
            park(key);
        }
    }

    [[nodiscard]] uint32_t sleepers() const noexcept {
        return sleepers_.load(std::memory_order_relaxed);
    }

private:
    // True when someone may be asleep; the epoch moves only then.
    [[nodiscard]] bool bump() noexcept {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (sleepers_.load(std::memory_order_relaxed) == 0) return false;
        epoch_.fetch_add(1, std::memory_order_seq_cst);
        return true;
    }

    alignas(config::kCacheIsolationBytes) std::atomic<ParkWord> epoch_;
    alignas(config::kCacheIsolationBytes) std::atomic<uint32_t> sleepers_;
};

}  // namespace bolt
