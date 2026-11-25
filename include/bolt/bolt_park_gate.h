// bolt_park_gate.h — park::Gate: exact-target waits with leases (G2CHK-1294).
//
// A gate holds a monotone progress value (an LSN, a durable offset, a
// completion count) and the leases of the threads waiting for it to reach their
// target. advance(v) stores v, fences, and when anyone is registered pops every
// lease with target <= v and wakes only those threads, once each, after the
// latch is released. A waiter spins on the value per its WaitPolicy, then
// registers its lease (in its park::Slot) and parks on its own slot.
//
//   writer                               waiter (target t)
//   ------                               -----------------
//   value.store(v, release)              register lease (latch), nwaiters++
//   fence(seq_cst)                       park_once: kParked, fence, value >= t ?
//   if nwaiters: pop <= v, wake each     block on the slot word
//
// Leases never expire on their own: a waiter's deadline ends its wait with
// kDeadline and removes the lease. The stuck report (for_each_lease) only
// names old leases. A full gate refuses the next waiter with kExhausted
// (Limits B: kGateWaitersDefault or the gate's own capacity), never queues
// unboundedly. A waker may deposit a result in lease.handoff before the wake
// (direct handoff: the woken thread need not re-read shared state).
#pragma once

#include "bolt/bolt_park.h"

#include <atomic>
#include <cassert>
#include <cstdint>

namespace bolt::park {

enum class GateWait : uint8_t { kReady = 0, kDeadline = 1, kExhausted = 2 };

template <uint32_t Cap = kGateWaitersDefault>
struct alignas(config::kCacheIsolationBytes) GateT {
    static_assert(Cap > 0 && Cap <= 65536, "gate capacity");

    std::atomic<uint64_t> value{0};
    alignas(config::kCacheIsolationBytes) std::atomic<uint32_t> nwaiters{0};
    std::atomic_flag latch = ATOMIC_FLAG_INIT;
    uint32_t count = 0;   // heap size (latch held)
    Slot* heap[Cap] = {};  // min-heap by lease.target (latch held)
    // Counters (latch held): advances that woke someone, threads woken, peak waiters.
    uint64_t wake_rounds = 0;
    uint64_t woken = 0;
    uint32_t peak = 0;

    // Store v (monotone) and wake every lease with target <= v. `handoff`
    // nonzero is deposited in each woken lease first.
    void advance(uint64_t v, uint64_t handoff = 0) noexcept {
        assert(v >= value.load(std::memory_order_relaxed) && "gate values are monotone");
        value.store(v, std::memory_order_release);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (nwaiters.load(std::memory_order_relaxed) == 0) return;
        Slot* ready[Cap];
        uint32_t n = 0;
        lock();
        while (count != 0 && heap[0]->lease.target <= v) {
            Slot* s = pop_min();
            s->lease.handoff.store(handoff, std::memory_order_relaxed);
            s->lease.registered.store(false, std::memory_order_release);
            ready[n++] = s;
        }
        nwaiters.store(count, std::memory_order_relaxed);
        if (n != 0) { ++wake_rounds; woken += n; }
        unlock();
        assert(n <= Cap);
        for (uint32_t i = 0; i < n; ++i) (void)wake(ready[i]);
    }

    // Wait until value >= target or the deadline (absolute ns; 0 = the class
    // deadline). `site` names the call site in reports. *handoff_out gets the
    // waker's deposit when the wait ended by an advance.
    GateWait wait(uint64_t target, uint64_t deadline_ns, const WaitPolicy& p, uint16_t site = 0,
                  uint64_t* handoff_out = nullptr) noexcept {
        auto reached = [&]() noexcept { return value.load(std::memory_order_acquire) >= target; };
        if (reached()) return GateWait::kReady;
        if (deadline_ns == 0 && p.deadline_ms != 0) deadline_ns = now_ns() + uint64_t{p.deadline_ms} * 1000000u;
        if (p.mode != WaitMode::kPark && spin(reached, p, deadline_ns)) return GateWait::kReady;
        Slot local;
        Slot* s = slot_self();
        if (s == nullptr) s = &local;  // registry full: still correct, just not listed
        if (!enlist(s, target, deadline_ns, site)) return GateWait::kExhausted;
        GateWait r = GateWait::kReady;
        for (;;) {
            if (!s->lease.registered.load(std::memory_order_acquire) || reached()) break;
            const uint64_t now = now_ns();
            if (deadline_ns != 0 && now >= deadline_ns) { r = GateWait::kDeadline; break; }
            const uint64_t t = deadline_ns == 0 ? 0 : deadline_ns - now;
            (void)park_once(s, [&]() noexcept {
                return !s->lease.registered.load(std::memory_order_acquire) || reached();
            }, [s](uint64_t tn) noexcept {
                if (s->word.load(std::memory_order_acquire) == kParked) native_wait(&s->word, kParked, tn);
            }, t);
        }
        delist(s);
        if (r == GateWait::kReady && !reached()) r = GateWait::kDeadline;
        if (handoff_out != nullptr) *handoff_out = s->lease.handoff.load(std::memory_order_relaxed);
        return r;
    }

    uint32_t waiters() const noexcept { return nwaiters.load(std::memory_order_relaxed); }

private:
    template <typename Reached>
    static bool spin(Reached& reached, const WaitPolicy& p, uint64_t deadline_ns) noexcept {
        if (!spin_enter()) return false;
        const uint64_t until = p.mode == WaitMode::kBusy ? deadline_ns : now_ns() + p.spin_ns;
        bool hit = false;
        for (uint32_t i = 1;; ++i) {
            if (reached()) { hit = true; break; }
            BOLT_PAUSE();
            if ((i & 63u) == 0 && until != 0 && now_ns() >= until) break;
        }
        spin_leave();
        return hit;
    }

    bool enlist(Slot* s, uint64_t target, uint64_t deadline_ns, uint16_t site) noexcept {
        s->lease_seq.fetch_add(1, std::memory_order_acq_rel);  // odd: writing
        s->lease.gate = this;
        s->lease.target = target;
        s->lease.deadline_ns = deadline_ns;
        s->lease.since_ns = now_ns();
        s->lease.site = site;
        s->lease.handoff.store(0, std::memory_order_relaxed);
        s->lease.reported.store(false, std::memory_order_relaxed);
        lock();
        const bool ok = count < Cap;
        if (ok) {
            s->lease.registered.store(true, std::memory_order_release);
            push(s);
            nwaiters.store(count, std::memory_order_relaxed);
            if (count > peak) peak = count;
        }
        unlock();
        s->lease_seq.fetch_add(1, std::memory_order_release);  // even: published
        return ok;
    }

    // Remove the lease if an advance has not already taken it.
    void delist(Slot* s) noexcept {
        if (s->lease.registered.load(std::memory_order_acquire)) {
            lock();
            if (s->lease.registered.load(std::memory_order_relaxed)) {
                erase(s);
                s->lease.registered.store(false, std::memory_order_release);
                nwaiters.store(count, std::memory_order_relaxed);
            }
            unlock();
        }
        s->lease_seq.fetch_add(1, std::memory_order_acq_rel);
        s->lease.gate = nullptr;
        s->lease_seq.fetch_add(1, std::memory_order_release);
        assert(!s->lease.registered.load(std::memory_order_relaxed));
    }

    void lock() noexcept {
        while (latch.test_and_set(std::memory_order_acquire)) BOLT_PAUSE();
    }
    void unlock() noexcept { latch.clear(std::memory_order_release); }

    void push(Slot* s) noexcept {
        assert(count < Cap);
        uint32_t i = count++;
        heap[i] = s;
        sift_up(i);
    }
    Slot* pop_min() noexcept {
        assert(count > 0);
        Slot* top = heap[0];
        heap[0] = heap[--count];
        if (count != 0) sift_down(0);
        return top;
    }
    void erase(Slot* s) noexcept {
        uint32_t i = 0;
        while (i < count && heap[i] != s) ++i;
        assert(i < count && "registered lease missing from its gate");
        if (i == count) return;
        heap[i] = heap[--count];
        if (i < count) { sift_down(i); sift_up(i); }
    }
    void sift_up(uint32_t i) noexcept {
        while (i > 0) {
            const uint32_t parent = (i - 1) / 2;
            if (heap[parent]->lease.target <= heap[i]->lease.target) break;
            Slot* t = heap[parent]; heap[parent] = heap[i]; heap[i] = t;
            i = parent;
        }
    }
    void sift_down(uint32_t i) noexcept {
        for (;;) {
            const uint32_t l = 2 * i + 1, r = l + 1;
            uint32_t m = i;
            if (l < count && heap[l]->lease.target < heap[m]->lease.target) m = l;
            if (r < count && heap[r]->lease.target < heap[m]->lease.target) m = r;
            if (m == i) return;
            Slot* t = heap[m]; heap[m] = heap[i]; heap[i] = t;
            i = m;
        }
    }
};

using Gate = GateT<>;

// Walk every live lease in the slot table (admin.debug.waiters). `fn(const
// Lease&, uint32_t slot_index)` gets a consistent copy; leases being written
// are skipped this pass.
template <typename Fn>
inline void for_each_lease(Fn&& fn) noexcept {
    detail::SlotTable& t = detail::slot_table();
    for (uint32_t i = 0; i < t.n; ++i) {
        if (t.slots[i].in_use.load(std::memory_order_acquire) == 0) continue;
        Lease copy;
        if (!lease_snapshot(t.slots[i], &copy) || copy.gate == nullptr) continue;
        fn(static_cast<const Lease&>(copy), i);
    }
}

// Stuck-waiter report: calls fn once per lease older than park_stuck_report_ms
// that has not been reported yet. Never wakes anyone (correctness stays with
// the waiter's own deadline). Run it from an existing tick.
template <typename Fn>
inline uint32_t report_stuck(uint64_t now, Fn&& fn) noexcept {
    const uint64_t age_ns = bolt_park_limits_value(bolt_park_limits_id::park_stuck_report_ms) * 1000000u;
    detail::SlotTable& t = detail::slot_table();
    uint32_t n = 0;
    for (uint32_t i = 0; i < t.n; ++i) {
        Slot& s = t.slots[i];
        if (s.in_use.load(std::memory_order_acquire) == 0) continue;
        Lease copy;
        if (!lease_snapshot(s, &copy) || copy.gate == nullptr) continue;
        if (!copy.registered.load(std::memory_order_relaxed) || now - copy.since_ns < age_ns) continue;
        if (s.lease.reported.exchange(true, std::memory_order_acq_rel)) continue;
        fn(static_cast<const Lease&>(copy), i);
        ++n;
    }
    return n;
}

}  // namespace bolt::park
