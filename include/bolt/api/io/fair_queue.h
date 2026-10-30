#pragma once
// fair_queue — priority-class fair scheduling primitive for bolt's I/O layer
// (STATION-85). This is the "which request goes next" half of the I/O
// scheduler that sits between the async file/socket API and the kernel ring —
// the backend-agnostic QoS layer that is the load-bearing value borrowed from
// Seastar (reference: seastar/include/seastar/core/fair_queue.hh +
// src/core/fair_queue.cc). Do NOT submit each co_await straight into the ring;
// enqueue here, and let io_scheduler::poll_io_queue() decide how many/which.
//
// Model. Multiple producers register priority CLASSES, each with a `shares`
// weight. Requests (entries) are queued against a class. When the scheduler
// asks for the next entry, the class that has consumed the LEAST capacity
// relative to its shares is served — so over time each class receives device
// bandwidth proportional to its shares, while a class may still race ahead when
// the others are idle. Balancing uses a virtual-time accumulator per class
// (Seastar's `_accumulated`): dispatching a request of cost `c` from a class of
// `shares` s charges `c * kCostScale / s` to that class's accumulator, and the
// active class with the smallest accumulator is always served next. Higher
// shares ⇒ slower accumulation ⇒ served more often. When an idle class rejoins
// it is fast-forwarded to the current frontier (minus a `forgiving_factor`
// allowance) so a long-idle class cannot monopolize the device with a stale-low
// accumulator (Seastar's `push_from_idle`).
//
// Intrusive & allocation-free on the hot path. An `entry` is an intrusive
// doubly-linked list node; the caller owns the storage (io_scheduler pools it).
// queue()/pop_front()/cancel() are all O(1) with no allocation. In particular a
// still-QUEUED request cancels for free — an O(1) unlink — which is exactly the
// io_intent property Seastar's cancellable_queue provides (io_queue.cc): a
// request sitting in the scheduler before the kernel can be dropped at no cost,
// the natural fit for coroutine cancellation/timeouts.
//
// Not thread-safe. Like the rest of the reactor's per-loop state a fair_queue
// is owned and driven by a single loop thread; cross-thread submission goes
// through the reactor's wake path, never a lock here. No RAII-per-thread.
//
// Cost model note (v1). `entry` cost is a caller-supplied scalar. A caller that
// wants Seastar's weight+size cost model maps (IOPS weight, byte size) → one
// scalar (e.g. `weight*W + size*S`) before queueing; the token-bucket device
// budget that meters those costs per tick lives in io_scheduler. The fixed-point
// `kCostScale` makes proportionality hold even for unit costs (cost=1), so a
// caller that just wants IOPS fairness can pass cost=1 everywhere.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <queue>
#include <vector>

namespace bolt {
namespace api {
namespace io {

class fair_queue {
public:
    using class_id   = uint32_t;
    using capacity_t = uint64_t;   // virtual-time / cost units (unsigned)

    // Not a live class id; also the "unlinked" marker inside an entry.
    static constexpr class_id kInvalidClass = UINT32_MAX;

    // Fixed-point scale applied to per-request cost before dividing by shares.
    // Keeps `cost * kCostScale / shares` non-zero (and thus proportionality
    // meaningful) even when cost is 1 and shares is large — the pitfall called
    // out in Seastar's pop_front ("zero cost ... monopolize the queue").
    static constexpr capacity_t kCostScale = 1024;

    struct config {
        // How far behind (in accumulator units, pre-scaled) a rejoining idle
        // class is allowed to remain relative to the current frontier. 0 =
        // strict (a rejoining class starts exactly at the frontier). A non-zero
        // value gives briefly-idle classes a small head start (Seastar's tau).
        capacity_t forgiving_factor = 0;
    };

    // ---- entry: an intrusive queue node owned by the caller -----------------

    // One queued request. The caller embeds this in its own request object and
    // keeps it alive until it is popped or cancelled. Cost defaults to 1 (pure
    // IOPS fairness); set_capacity() before queue() for a weight+size cost.
    class entry {
        friend class fair_queue;
    public:
        entry() noexcept = default;
        explicit entry(capacity_t cost) noexcept
            : capacity_(cost != 0 ? cost : 1) {}

        capacity_t capacity() const noexcept { return capacity_; }

        // Set the dispatch cost. Only valid while the entry is NOT linked.
        void set_capacity(capacity_t cost) noexcept {
            assert(klass_ == kInvalidClass && "set_capacity on a queued entry");
            capacity_ = (cost != 0 ? cost : 1);
        }

        // True while this entry sits in a class queue (queued, not yet popped
        // or cancelled).
        bool linked() const noexcept { return klass_ != kInvalidClass; }

    private:
        capacity_t capacity_ = 1;
        entry*     prev_ = nullptr;
        entry*     next_ = nullptr;
        class_id   klass_ = kInvalidClass;   // owning class while linked
    };

    explicit fair_queue(config cfg = {}) noexcept : config_(cfg) {}

    fair_queue(const fair_queue&)            = delete;
    fair_queue& operator=(const fair_queue&) = delete;

    // ---- class registration -------------------------------------------------

    // Register class `id` with `shares` (>=1; 0 is clamped to 1). Ids are dense
    // indices — registering id N sizes storage to N+1. Re-registering a live
    // class is a no-op error in debug (asserts).
    void register_priority_class(class_id id, uint32_t shares) {
        if (id >= classes_.size()) {
            classes_.resize(static_cast<size_t>(id) + 1);
        }
        pclass& pc = classes_[id];
        assert(!pc.registered && "class already registered");
        pc.registered   = true;
        pc.shares       = shares != 0 ? shares : 1;
        pc.accumulated  = 0;
        pc.activations  = 0;
        pc.queued       = false;
        pc.count        = 0;
        pc.head = pc.tail = nullptr;
    }

    // Change a registered class's shares (takes effect on the next dispatch).
    void update_shares(class_id id, uint32_t shares) noexcept {
        assert(id < classes_.size() && classes_[id].registered);
        classes_[id].shares = shares != 0 ? shares : 1;
    }

    // Unregister a class. It must have no queued entries.
    void unregister_priority_class(class_id id) noexcept {
        assert(id < classes_.size() && classes_[id].registered);
        assert(classes_[id].count == 0 && "unregister with queued entries");
        classes_[id].registered = false;
    }

    bool class_registered(class_id id) const noexcept {
        return id < classes_.size() && classes_[id].registered;
    }

    // ---- queueing -----------------------------------------------------------

    // Queue `e` against class `id` (FIFO within the class). `e` must be
    // unlinked and its owning storage must outlive the entry's time in queue.
    void queue(class_id id, entry& e) noexcept {
        assert(id < classes_.size() && classes_[id].registered);
        assert(!e.linked() && "entry already queued");
        pclass& pc = classes_[id];

        // Link at the tail (FIFO).
        e.prev_  = pc.tail;
        e.next_  = nullptr;
        e.klass_ = id;
        if (pc.tail != nullptr) {
            pc.tail->next_ = &e;
        } else {
            pc.head = &e;
        }
        pc.tail = &e;
        ++pc.count;
        ++total_;

        if (!pc.queued) {
            push_from_idle(id, pc);
        }
    }

    // Cancel a still-queued entry: O(1) unlink, no dispatch. Safe to call on an
    // entry that is not linked (no-op). This is the "cancel-while-queued is
    // free" property. The owning class stays in the active heap until top()
    // lazily reaps it, so cancellation never touches the heap.
    void cancel(entry& e) noexcept {
        if (!e.linked()) {
            return;
        }
        unlink(e);
    }

    // ---- dispatch selection -------------------------------------------------

    // Peek the next entry to dispatch: the head of the active class with the
    // smallest accumulator. Returns nullptr when nothing is queued. Reaps
    // now-empty classes off the heap lazily (hence non-const).
    entry* top() noexcept {
        while (!active_.empty()) {
            class_id id = active_.top();
            pclass&  pc = classes_[id];
            if (pc.head != nullptr) {
                return pc.head;
            }
            // Class drained (by pops and/or cancels) — remove from the heap.
            assert(pc.queued);
            pc.queued = false;
            active_.pop();
        }
        return nullptr;
    }

    // Pop the entry top() just returned, charging its cost to its class's
    // accumulator. Must be preceded by a top() that returned non-null.
    void pop_front() noexcept {
        assert(!active_.empty());
        class_id id = active_.top();
        pclass&  pc = classes_[id];
        assert(pc.head != nullptr && "pop_front with empty class at heap top");

        entry& e = *pc.head;
        const capacity_t cost = e.capacity();
        unlink(e);

        // Advance the frontier, then charge this class. Charging AFTER reading
        // the frontier mirrors Seastar (uses pre-increment accumulated).
        if (pc.accumulated > frontier_) {
            frontier_ = pc.accumulated;
        }
        const capacity_t charge = cost_for(cost, pc.shares);
        // Saturate rather than wrap — a wrapped accumulator would corrupt the
        // ordering. Over-long runs are handled by rebase() below.
        if (pc.accumulated > UINT64_MAX - charge) {
            pc.accumulated = UINT64_MAX;
        } else {
            pc.accumulated += charge;
        }

        active_.pop();
        if (pc.head != nullptr) {
            active_.push(id);   // still has work — re-heap with new accumulator
        } else {
            pc.queued = false;
        }

        // Keep accumulators from marching to overflow on very long runs by
        // periodically rebasing all live classes down by the frontier.
        if (frontier_ >= kRebaseThreshold) {
            rebase();
        }
    }

    // ---- introspection ------------------------------------------------------

    // Total live (non-cancelled) queued entries across all classes.
    size_t size() const noexcept { return total_; }
    bool empty() const noexcept { return total_ == 0; }

    size_t class_size(class_id id) const noexcept {
        assert(id < classes_.size());
        return classes_[id].count;
    }
    capacity_t accumulated(class_id id) const noexcept {
        assert(id < classes_.size());
        return classes_[id].accumulated;
    }
    uint32_t activations(class_id id) const noexcept {
        assert(id < classes_.size());
        return classes_[id].activations;
    }

private:
    struct pclass {
        uint32_t   shares      = 1;
        capacity_t accumulated = 0;
        uint32_t   activations = 0;
        size_t     count       = 0;      // live entries linked in this class
        entry*     head        = nullptr;
        entry*     tail        = nullptr;
        bool       registered  = false;
        bool       queued      = false;  // present in the active heap
    };

    // Rebase the frontier once accumulators near this magnitude (well below the
    // uint64 ceiling) so a long-lived scheduler never approaches overflow.
    static constexpr capacity_t kRebaseThreshold =
        capacity_t{1} << 60;

    // Min-accumulator-first ordering. std::priority_queue is a max-heap, so the
    // comparator returns true when `a` should sink BELOW `b` — i.e. when a's
    // accumulator is the larger — leaving the smallest accumulator on top.
    struct class_compare {
        const std::vector<pclass>* classes = nullptr;
        bool operator()(class_id a, class_id b) const noexcept {
            return (*classes)[a].accumulated > (*classes)[b].accumulated;
        }
    };

    static capacity_t cost_for(capacity_t cost, uint32_t shares) noexcept {
        // Fixed-point so unit costs still divide meaningfully by shares.
        const capacity_t scaled = cost * kCostScale;   // cost bounded by caller
        const capacity_t per    = scaled / shares;
        return per != 0 ? per : 1;
    }

    void unlink(entry& e) noexcept {
        assert(e.klass_ < classes_.size());
        pclass& pc = classes_[e.klass_];
        assert(pc.count > 0 && total_ > 0);
        if (e.prev_ != nullptr) {
            e.prev_->next_ = e.next_;
        } else {
            pc.head = e.next_;
        }
        if (e.next_ != nullptr) {
            e.next_->prev_ = e.prev_;
        } else {
            pc.tail = e.prev_;
        }
        e.prev_ = e.next_ = nullptr;
        e.klass_ = kInvalidClass;
        --pc.count;
        --total_;
    }

    void push_from_idle(class_id id, pclass& pc) noexcept {
        // Fast-forward a rejoining class to the frontier (less its forgiving
        // allowance) so it cannot monopolize with a stale-low accumulator, but
        // never move it backward. Mirrors Seastar's push_from_idle.
        capacity_t floor = frontier_;
        const capacity_t allow = config_.forgiving_factor / pc.shares;
        floor = (floor > allow) ? (floor - allow) : 0;
        if (pc.accumulated < floor) {
            pc.accumulated = floor;
        }
        pc.queued = true;
        ++pc.activations;
        active_.push(id);
    }

    // Subtract the frontier from every live accumulator (and the frontier
    // itself), preserving all relative orderings. Called rarely.
    void rebase() noexcept {
        const capacity_t base = frontier_;
        for (auto& pc : classes_) {
            if (pc.registered) {
                pc.accumulated = (pc.accumulated > base) ? (pc.accumulated - base) : 0;
            }
        }
        frontier_ = 0;
    }

    config                 config_;
    std::vector<pclass>    classes_;
    capacity_t             frontier_ = 0;   // max accumulator dispatched so far
    size_t                 total_    = 0;   // live queued entries

    // Active classes ordered by accumulator; comparator reads classes_.
    std::priority_queue<class_id, std::vector<class_id>, class_compare>
        active_{class_compare{&classes_}};
};

}  // namespace io
}  // namespace api
}  // namespace bolt
