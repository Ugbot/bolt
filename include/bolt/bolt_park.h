// bolt_park.h — one park protocol for every waiting thread (G2CHK-1293).
//
// A waiter owns a park::Slot: one 32-bit word on its own cache line, waited on
// with the OS's native wait-on-address (futex, __ulock, WaitOnAddress), or, for
// a reactor (SlotKind::kReactor), by its I/O backend's poll.
//
//   waiter                                      waker
//   ------                                      -----
//   spin per WaitPolicy, word = kRunning        publish (push / store), release
//   word.store(kParked); fence(seq_cst)         fence(seq_cst)
//   if (ready()) { word = kRunning; done }      if (word == kParked &&
//   block while word == kParked (deadline)          cas(kParked -> kWoken))
//   word = kRunning; re-check ready()               kernel wake  (only the CAS winner)
//
// One of the two fences is first in the seq_cst order: either the waiter's
// re-check sees the publish, or the waker sees kParked. Producers never make a
// syscall while the waiter is running or spinning, and of several producers
// only the one that moves kParked -> kWoken pays for the wake. A stale kernel
// wake costs one spurious return, never a lost one.
//
// Policy is data: WaitMode and the spin budget come from the wait class's
// Limits rows (bolt_park_limits.h), resolved per wait, so a running process can
// change them. park_max_spinners caps the threads spinning at once.
#pragma once

#include "bolt/bolt_config.h"
#include "bolt/bolt_park_limits.h"
#include "bolt/bolt_port.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <thread>

#if defined(__linux__)
#include <climits>
#include <ctime>
#include <linux/futex.h>
#endif

#if defined(__APPLE__)
extern "C" int __ulock_wait2(uint32_t operation, void* addr, uint64_t value, uint64_t timeout_ns,
                             uint64_t value2);
extern "C" int __ulock_wake(uint32_t operation, void* addr, uint64_t wake_value);
#endif

namespace bolt::park {

inline constexpr uint32_t kRunning = 0;
inline constexpr uint32_t kParked = 1;
inline constexpr uint32_t kWoken = 2;

enum class WaitMode : uint8_t { kBusy = 0, kSpinThenPark = 1, kPark = 2 };
enum class WaitClass : uint8_t { kReactor = 0, kAck, kPool, kFlusher, kBackground, kCount };
enum class SlotKind : uint8_t { kAddress = 0, kReactor = 1 };
enum class WaitResult : uint8_t { kReady = 0, kDeadline = 1 };

struct WaitPolicy {
    WaitMode mode = WaitMode::kSpinThenPark;
    uint32_t spin_ns = 0;
    uint32_t poll_every_ns = 0;  // reactor only: zero-timeout poll interval while spinning
    uint32_t deadline_ms = 0;    // 0 = no deadline for the class
};

inline uint64_t now_ns() noexcept {
    using namespace std::chrono;
    return static_cast<uint64_t>(duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
}

// ---------------------------------------------------------------------------
// Native 32-bit wait-on-address.
// ---------------------------------------------------------------------------

// Blocks while *word == expect, for at most timeout_ns (0 = no timeout).
// Spurious returns are allowed; the caller re-checks.
inline void native_wait(std::atomic<uint32_t>* word, uint32_t expect, uint64_t timeout_ns) noexcept {
    static_assert(sizeof(std::atomic<uint32_t>) == 4, "native wait needs a plain 32-bit word");
    assert(word != nullptr);
#if defined(__linux__)
    struct timespec ts;
    struct timespec* tsp = nullptr;
    if (timeout_ns != 0) {
        ts.tv_sec = static_cast<time_t>(timeout_ns / 1000000000u);
        ts.tv_nsec = static_cast<long>(timeout_ns % 1000000000u);
        tsp = &ts;
    }
    (void)syscall(SYS_futex, reinterpret_cast<uint32_t*>(word), FUTEX_WAIT_PRIVATE, expect, tsp,
                  nullptr, 0);
#elif defined(__APPLE__)
    constexpr uint32_t kCompareAndWait = 1;      // UL_COMPARE_AND_WAIT
    constexpr uint32_t kNoErrno = 0x01000000;    // ULF_NO_ERRNO
    (void)__ulock_wait2(kCompareAndWait | kNoErrno, word, expect, timeout_ns, 0);
#elif defined(_WIN32)
    uint32_t cmp = expect;
    const DWORD ms = timeout_ns == 0 ? INFINITE
                                     : static_cast<DWORD>((timeout_ns + 999999u) / 1000000u);
    (void)WaitOnAddress(word, &cmp, sizeof(cmp), ms);
#else
    (void)timeout_ns;
    word->wait(expect, std::memory_order_acquire);
#endif
}

inline void native_wake_one(std::atomic<uint32_t>* word) noexcept {
    assert(word != nullptr);
#if defined(__linux__)
    (void)syscall(SYS_futex, reinterpret_cast<uint32_t*>(word), FUTEX_WAKE_PRIVATE, 1, nullptr,
                  nullptr, 0);
#elif defined(__APPLE__)
    constexpr uint32_t kCompareAndWait = 1;
    constexpr uint32_t kNoErrno = 0x01000000;
    (void)__ulock_wake(kCompareAndWait | kNoErrno, word, 0);
#elif defined(_WIN32)
    WakeByAddressSingle(word);
#else
    word->notify_one();
#endif
}

// ---------------------------------------------------------------------------
// Slot.
// ---------------------------------------------------------------------------

using KernelWake = void (*)(void* ctx) noexcept;

struct alignas(config::kCacheIsolationBytes) Slot {
    std::atomic<uint32_t> word{kRunning};
    SlotKind kind = SlotKind::kAddress;
    WaitClass wait_class = WaitClass::kPool;
    // kReactor: the backend's raw wake (EVFILT_USER, eventfd, MSG_RING ...).
    KernelWake kernel_wake = nullptr;
    void* kernel_ctx = nullptr;
    // Owner-written counters (no shared writes on the wait path).
    uint64_t announces = 0;  // kParked stores (park cycles); kernel_wakes <= announces
    uint64_t parks = 0;      // cycles that blocked
    uint64_t spurious = 0;
    uint64_t spin_hits = 0;
    // Waker-written.
    std::atomic<uint64_t> kernel_wakes{0};
};

// Lever for the falsifier tests: each value models one forbidden reordering
// and must make the lost-wake stress report LOST_WAKE. Never set in a build
// that ships.
//   1: the waker skips its fence
//   2: the waiter checks ready() before it announces kParked
//   3: the waiter announces kParked but does not re-check
// Levers 2 and 3 widen the window with a sleep, so the stress hits it.
#ifndef BOLT_PARK_FALSIFY
#define BOLT_PARK_FALSIFY 0
#endif

// Waker half. Call after publishing. True when this call paid a kernel wake.
inline bool wake(Slot* s) noexcept {
    assert(s != nullptr);
#if BOLT_PARK_FALSIFY != 1
    std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
    if (s->word.load(std::memory_order_relaxed) != kParked) return false;
    uint32_t expected = kParked;
    if (!s->word.compare_exchange_strong(expected, kWoken, std::memory_order_acq_rel,
                                         std::memory_order_relaxed)) {
        return false;  // another waker won, or the waiter left
    }
    s->kernel_wakes.fetch_add(1, std::memory_order_relaxed);
    if (s->kind == SlotKind::kReactor) {
        assert(s->kernel_wake != nullptr);
        s->kernel_wake(s->kernel_ctx);
    } else {
        native_wake_one(&s->word);
    }
    return true;
}

// Waiter half, steps 2-4: announce, re-check, block. Returns true when ready()
// held at the re-check; otherwise the caller has blocked once and re-checks.
// `block(timeout_ns)` waits for at most timeout_ns (reactor: backend poll).
template <typename Ready, typename Block>
inline bool park_once(Slot* s, Ready&& ready, Block&& block, uint64_t timeout_ns) noexcept {
    assert(s != nullptr);
#if BOLT_PARK_FALSIFY == 2
    const bool early = ready();
    std::this_thread::sleep_for(std::chrono::microseconds(20));
    s->word.store(kParked, std::memory_order_relaxed);
#elif BOLT_PARK_FALSIFY == 3
    std::this_thread::sleep_for(std::chrono::microseconds(20));
    s->word.store(kParked, std::memory_order_relaxed);
    const bool early = false;
    (void)ready;
#else
    s->word.store(kParked, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const bool early = ready();
#endif
    ++s->announces;
    if (early) {
        s->word.store(kRunning, std::memory_order_relaxed);
        return true;
    }
    ++s->parks;
    block(timeout_ns);
    s->word.store(kRunning, std::memory_order_relaxed);
    assert(s->word.load(std::memory_order_relaxed) == kRunning);
    return false;
}

// ---------------------------------------------------------------------------
// Policy.
// ---------------------------------------------------------------------------

namespace detail {
struct ClassRows {
    bolt_park_limits_id mode, spin, deadline;
};
inline ClassRows class_rows(WaitClass c) noexcept {
    using id = bolt_park_limits_id;
    switch (c) {
        case WaitClass::kReactor: return {id::park_reactor_mode, id::park_reactor_spin_ns, id::park_reactor_deadline_ms};
        case WaitClass::kAck: return {id::park_ack_mode, id::park_ack_spin_ns, id::park_ack_deadline_ms};
        case WaitClass::kPool: return {id::park_pool_mode, id::park_pool_spin_ns, id::park_pool_deadline_ms};
        case WaitClass::kFlusher: return {id::park_flusher_mode, id::park_flusher_spin_ns, id::park_flusher_deadline_ms};
        case WaitClass::kBackground:
        case WaitClass::kCount: break;
    }
    return {id::park_background_mode, id::park_background_spin_ns, id::park_background_deadline_ms};
}

inline WaitMode profile_mode(WaitClass c, uint64_t profile) noexcept {
    assert(profile <= 2);
    if (c == WaitClass::kBackground) return WaitMode::kPark;
    if (profile == 0) return WaitMode::kPark;
    if (profile == 2 && c != WaitClass::kPool) return WaitMode::kBusy;
    return WaitMode::kSpinThenPark;
}

inline std::atomic<uint32_t>& spinners() noexcept {
    alignas(config::kCacheIsolationBytes) static std::atomic<uint32_t> n{0};
    return n;
}

inline uint32_t max_spinners() noexcept {
    const uint64_t v = bolt_park_limits_value(bolt_park_limits_id::park_max_spinners);
    if (v != 0) return static_cast<uint32_t>(v);
    const uint32_t cores = bolt_get_hardware_concurrency();
    return cores >= 8 ? cores / 4 : 2;
}
}  // namespace detail

// The class's current policy (Limits rows, resolved now).
inline WaitPolicy policy(WaitClass c) noexcept {
    const detail::ClassRows r = detail::class_rows(c);
    WaitPolicy p;
    const uint64_t mode = bolt_park_limits_value(r.mode);
    p.mode = mode == 3 ? detail::profile_mode(c, bolt_park_limits_value(bolt_park_limits_id::park_profile))
                       : static_cast<WaitMode>(mode);
    p.spin_ns = static_cast<uint32_t>(bolt_park_limits_value(r.spin));
    p.deadline_ms = static_cast<uint32_t>(bolt_park_limits_value(r.deadline));
    if (c == WaitClass::kReactor) {
        p.poll_every_ns = static_cast<uint32_t>(
            bolt_park_limits_value(bolt_park_limits_id::park_reactor_poll_every_ns));
    }
    assert(p.mode == WaitMode::kBusy || p.mode == WaitMode::kSpinThenPark || p.mode == WaitMode::kPark);
    return p;
}

// Admission to a spin phase under park_max_spinners. Pair with spin_leave().
inline bool spin_enter() noexcept {
    const uint32_t before = detail::spinners().fetch_add(1, std::memory_order_relaxed);
    if (before < detail::max_spinners()) return true;
    detail::spinners().fetch_sub(1, std::memory_order_relaxed);
    return false;
}
inline void spin_leave() noexcept {
    const uint32_t before = detail::spinners().fetch_sub(1, std::memory_order_relaxed);
    assert(before > 0);
    (void)before;
}

// Full wait on an address slot: spin per policy, then park until ready() or the
// deadline (absolute, steady clock ns; 0 = the class deadline, or none).
template <typename Ready>
inline WaitResult wait(Slot* s, Ready&& ready, uint64_t deadline_ns, const WaitPolicy& p) noexcept {
    assert(s != nullptr && s->kind == SlotKind::kAddress);
    if (deadline_ns == 0 && p.deadline_ms != 0) deadline_ns = now_ns() + uint64_t{p.deadline_ms} * 1000000u;
    for (;;) {
        if (ready()) return WaitResult::kReady;
        if (p.mode != WaitMode::kPark && spin_enter()) {
            const uint64_t until = p.mode == WaitMode::kBusy ? deadline_ns : now_ns() + p.spin_ns;
            bool hit = false;
            for (uint32_t i = 1;; ++i) {
                if (ready()) { hit = true; break; }
                BOLT_PAUSE();
                if ((i & 63u) == 0 && until != 0 && now_ns() >= until) break;
            }
            spin_leave();
            if (hit) { ++s->spin_hits; return WaitResult::kReady; }
        }
        const uint64_t now = now_ns();
        if (deadline_ns != 0 && now >= deadline_ns) return WaitResult::kDeadline;
        const uint64_t timeout = deadline_ns == 0 ? 0 : deadline_ns - now;
        if (park_once(s, ready, [s](uint64_t t) noexcept {
                while (s->word.load(std::memory_order_acquire) == kParked) {
                    native_wait(&s->word, kParked, t);
                    if (t != 0) break;  // timed: one native wait, then the caller re-checks
                }
            }, timeout)) {
            return WaitResult::kReady;
        }
        if (ready()) return WaitResult::kReady;
        ++s->spurious;
    }
}

}  // namespace bolt::park
