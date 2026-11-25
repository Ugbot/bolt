// bolt_park_idle.h — park::IdleSet: idle workers of one pool (G2CHK-1295).
//
// Interchangeable workers wait for "the queue is not empty". Three rules keep
// wakes to the ones that can run:
//   - spinner rule: while a worker is in its spin phase it will find the work,
//     so notify() wakes nobody;
//   - LIFO: notify() wakes the most recently parked worker (warm caches); cold
//     ones stay asleep;
//   - chained wake: a worker that takes an item and sees more calls notify()
//     before it runs it (the caller's job).
// notify_all() is for shutdown only. Each worker parks on its own park::Slot,
// so a wake is one CAS and at most one kernel wake.
//
//   worker                                     producer
//   ------                                     --------
//   spin (spinning++ .. spinning--)            push item
//   enlist in the idle stack                   fence(seq_cst)
//   park_once: kParked, fence, re-check        if spinning: done
//                                              else pop the newest idle, wake it
#pragma once

#include "bolt/bolt_park.h"

#include <atomic>
#include <cassert>
#include <cstdint>

namespace bolt::park {

template <uint32_t Cap>
struct alignas(config::kCacheIsolationBytes) IdleSetT {
    static_assert(Cap > 0 && Cap <= 65536, "idle set capacity");

    alignas(config::kCacheIsolationBytes) std::atomic<uint32_t> spinning{0};
    alignas(config::kCacheIsolationBytes) std::atomic<uint32_t> nidle{0};
    std::atomic_flag latch = ATOMIC_FLAG_INIT;
    uint32_t top = 0;        // stack size (latch held)
    Slot* stack[Cap] = {};   // idle workers, newest last (latch held)
    std::atomic<uint64_t> wakes{0};
    std::atomic<uint64_t> skipped_for_spinner{0};

    // Call after publishing work.
    void notify() noexcept {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (spinning.load(std::memory_order_relaxed) != 0) {
            skipped_for_spinner.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (nidle.load(std::memory_order_relaxed) == 0) return;
        lock();
        Slot* s = top != 0 ? stack[--top] : nullptr;
        nidle.store(top, std::memory_order_relaxed);
        unlock();
        if (s == nullptr) return;
        s->lease.registered.store(false, std::memory_order_release);  // "taken off the stack"
        wakes.fetch_add(1, std::memory_order_relaxed);
        (void)wake(s);
    }

    void notify_all() noexcept {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        for (;;) {
            lock();
            Slot* s = top != 0 ? stack[--top] : nullptr;
            nidle.store(top, std::memory_order_relaxed);
            unlock();
            if (s == nullptr) return;
            s->lease.registered.store(false, std::memory_order_release);
            (void)wake(s);
        }
    }

    // Returns when ready() does, or at the deadline (absolute ns; 0 = the
    // policy's class deadline, or none). ready() may take the item itself.
    template <typename Ready>
    WaitResult wait(Slot* self, Ready&& ready, const WaitPolicy& p, uint64_t deadline_ns = 0) noexcept {
        assert(self != nullptr);
        if (ready()) return WaitResult::kReady;
        if (deadline_ns == 0 && p.deadline_ms != 0) deadline_ns = now_ns() + uint64_t{p.deadline_ms} * 1000000u;
        for (;;) {
            if (p.mode != WaitMode::kPark && spin(ready, p, deadline_ns)) return WaitResult::kReady;
            enlist(self);
            const uint64_t now = now_ns();
            const uint64_t t = deadline_ns == 0 ? 0 : (deadline_ns > now ? deadline_ns - now : 1);
            // A notify that popped us before we announced kParked could not wake
            // us; being off the stack is itself a reason not to block.
            bool got = false;
            (void)park_once(self, [&]() noexcept {
                got = ready();
                return got || !self->lease.registered.load(std::memory_order_acquire);
            }, [self](uint64_t tn) noexcept {
                if (self->word.load(std::memory_order_acquire) == kParked) native_wait(&self->word, kParked, tn);
            }, t);
            delist(self);
            if (got || ready()) return WaitResult::kReady;
            if (deadline_ns != 0 && now_ns() >= deadline_ns) return WaitResult::kDeadline;
            ++self->spurious;
        }
    }

private:
    template <typename Ready>
    bool spin(Ready& ready, const WaitPolicy& p, uint64_t deadline_ns) noexcept {
        if (!spin_enter()) return false;
        spinning.fetch_add(1, std::memory_order_relaxed);
        const uint64_t until = p.mode == WaitMode::kBusy ? deadline_ns : now_ns() + p.spin_ns;
        bool hit = false;
        for (uint32_t i = 1;; ++i) {
            if (ready()) { hit = true; break; }
            BOLT_PAUSE();
            if ((i & 63u) == 0 && until != 0 && now_ns() >= until) break;
        }
        // Leave the spinner count before enlisting: a producer that still sees
        // us spinning skips its wake, and our park_once re-check catches it.
        spinning.fetch_sub(1, std::memory_order_relaxed);
        spin_leave();
        return hit;
    }

    void enlist(Slot* s) noexcept {
        lock();
        assert(top < Cap && "idle set smaller than the pool");
        s->lease.registered.store(true, std::memory_order_relaxed);
        stack[top++] = s;
        nidle.store(top, std::memory_order_relaxed);
        unlock();
    }

    void delist(Slot* s) noexcept {
        if (!s->lease.registered.load(std::memory_order_acquire)) return;
        lock();
        for (uint32_t i = 0; i < top; ++i) {
            if (stack[i] != s) continue;
            for (uint32_t j = i + 1; j < top; ++j) stack[j - 1] = stack[j];
            --top;
            break;
        }
        nidle.store(top, std::memory_order_relaxed);
        s->lease.registered.store(false, std::memory_order_release);
        unlock();
    }

    void lock() noexcept {
        while (latch.test_and_set(std::memory_order_acquire)) BOLT_PAUSE();
    }
    void unlock() noexcept { latch.clear(std::memory_order_release); }
};

}  // namespace bolt::park
