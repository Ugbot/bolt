// bolt_scheduler.h — Lock-free task scheduler with configurable spin policy
//
// Combines Venus job system patterns (SPMC ring, CAS claim, spin-then-yield,
// job pools, range subdivision, phase barriers) with FasterAPI's I/O dispatch.
//
// RULES: No exceptions. No RTTI. No smart pointers. No heap on hot path.
// All task data from pre-allocated pools. Ring buffer pre-allocated at init.

#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/bolt_channel.h"
#include "bolt/bolt_config.h"
#include "bolt/bolt_port.h"
#include "bolt/bolt_topology.h"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <thread>

namespace bolt {

// ============================================================================
// Spin Policy
// ============================================================================

enum class SpinPolicy : uint8_t {
    BusySpin,     // Pure spin. Pin to core. Latency-critical (< 1μs).
    SpinYield,    // Spin N then yield(). Default for compute workers.
    ParkWait,     // Condvar-backed park. Sleeps on the scheduler's
                  // idle_cv_ when the ring is empty; woken by
                  // scheduler_notify_submit() after a push. Real impl
                  // requires a Scheduler*, so callers should prefer
                  // scheduler_worker_idle(sched, worker_id) — this
                  // enum slot is still accepted by spin_wait() which
                  // falls back to yield for back-compat.
};

static constexpr uint32_t kDefaultSpinCount = config::kDefaultSpinCount;

// Scheduler-less spin_wait. Keeps back-compat for callers that don't
// have a Scheduler* handy (submit_wait, wait_all). The ParkWait path
// here falls back to yield — real parking lives in scheduler_worker_idle.
inline void spin_wait(SpinPolicy policy, uint32_t& spin_count, uint32_t max_spins) noexcept {
    switch (policy) {
        case SpinPolicy::BusySpin:
            cpu_pause();
            break;
        case SpinPolicy::SpinYield:
            if (++spin_count > max_spins) {
                std::this_thread::yield();
                spin_count = 0;
            } else {
                cpu_pause();
            }
            break;
        case SpinPolicy::ParkWait:
            // No Scheduler* here — worst case a yield. Hot path uses
            // scheduler_worker_idle which has the real cv-backed wait.
            std::this_thread::yield();
            break;
    }
}

// ============================================================================
// Tuning Profiles & SchedulerConfig
// ============================================================================

enum class TuneProfile : uint8_t {
    Latency,     // HFT / streaming: lowest tail; burn CPU to get it.
    Balanced,    // Default. Analytics-friendly, predictable.
    Throughput,  // Bulk analytics: biggest grain, coop scheduling.
};

/// Compute the number of rows that fit into a grain of `grain_bytes`, given
/// per-row element size. Never returns 0 (element > budget collapses to 1 row).
BOLT_FORCE_INLINE uint32_t bolt_grain_rows(uint32_t grain_bytes, size_t elem_size) noexcept {
    assert(elem_size > 0);
    assert(grain_bytes > 0);
    if (elem_size == 0) return 1;
    uint32_t g = static_cast<uint32_t>(grain_bytes / elem_size);
    if (g == 0) g = 1;
    return g;
}

struct SchedulerConfig {
    TuneProfile tune           = TuneProfile::Balanced;
    uint32_t    num_workers    = 0;              // 0 -> bolt_get_hardware_concurrency()
    SpinPolicy  spin           = SpinPolicy::SpinYield;
    bool        pin_workers    = false;
    bool        numa_bind      = false;
    bool        prefer_p_cores = true;
    uint32_t    grain_bytes    = 256u * 1024u;   // morsel = 256 KB of column payload
    uint32_t    dispatch_batch = 1;              // reserved for coalesced submit (Wave G)

    // Factory: return a config populated from a TuneProfile preset.
    static SchedulerConfig with_profile(TuneProfile p) noexcept;
};

inline SchedulerConfig SchedulerConfig::with_profile(TuneProfile p) noexcept {
    SchedulerConfig c{};
    c.tune = p;
    switch (p) {
        case TuneProfile::Latency:
            c.spin           = SpinPolicy::BusySpin;
            c.pin_workers    = true;
            c.numa_bind      = true;
            c.prefer_p_cores = true;
            c.grain_bytes    = config::kLatencyGrainBytes;
            c.dispatch_batch = config::kLatencyDispatchBatch;
            break;
        case TuneProfile::Balanced:
            c.spin           = SpinPolicy::SpinYield;
            c.pin_workers    = false;
            c.numa_bind      = false;
            c.prefer_p_cores = true;
            c.grain_bytes    = config::kBalancedGrainBytes;
            c.dispatch_batch = config::kBalancedDispatchBatch;
            break;
        case TuneProfile::Throughput:
            c.spin           = SpinPolicy::SpinYield;
            c.pin_workers    = false;
            c.numa_bind      = true;
            c.prefer_p_cores = false;
            c.grain_bytes    = config::kThroughputGrainBytes;
            c.dispatch_batch = config::kThroughputDispatchBatch;
            break;
    }
    return c;
}

// ============================================================================
// Task — function pointer + argument, no std::function, no heap
// ============================================================================

/// Task function signature. Takes opaque arg, returns nothing.
/// All state passed via arg pointer (arena-allocated or pool-allocated).
typedef void (*TaskFn)(void* arg) noexcept;

/// Range task function signature. Operates on [start, end) of a data range.
/// thread_id for per-thread resources (arena, counters).
typedef void (*RangeTaskFn)(void* user_data, uint32_t start, uint32_t end,
                            uint32_t thread_id) noexcept;

/// Column task function signature. Operates on BoltBatch columns [start, end).
/// read/write buffers for Venus tick-tock COW pattern.
typedef void (*ColumnTaskFn)(const void* read_batch, void* write_batch,
                             uint32_t start, uint32_t end,
                             float delta_time, uint32_t thread_id) noexcept;

struct Task {
    TaskFn fn;
    void*  arg;
};

// ============================================================================
// Task Pool — Treiber stack, lock-free, zero-alloc in steady state
// ============================================================================

/// Lock-free object pool using Treiber stack (same as Venus job_pool).
/// Pre-populates on init. acquire() pops, release() pushes. No malloc
/// after warmup.
struct TaskPool {
    struct Node { Node* next; };

    std::atomic<uintptr_t> head;
    size_t obj_size;

    static TaskPool* create(size_t data_size, uint32_t capacity) noexcept {
        TaskPool* p = static_cast<TaskPool*>(malloc(sizeof(TaskPool)));
        if (!p) return nullptr;
        p->obj_size = (data_size < sizeof(Node)) ? sizeof(Node) : data_size;
        p->head.store(0, std::memory_order_relaxed);

        // Pre-populate
        for (uint32_t i = 0; i < capacity; ++i) {
            void* obj = calloc(1, p->obj_size);
            if (obj) p->release(obj);
        }
        return p;
    }

    void destroy() noexcept {
        Node* n;
        while ((n = pop()) != nullptr) free(n);
        free(this);
    }

    void* acquire() noexcept {
        Node* n = pop();
        if (n) return static_cast<void*>(n);
        return calloc(1, obj_size);  // Pool exhausted, fallback
    }

    void release(void* obj) noexcept {
        push(static_cast<Node*>(obj));
    }

private:
    Node* pop() noexcept {
        uintptr_t old = head.load(std::memory_order_acquire);
        Node* ptr = reinterpret_cast<Node*>(old);
        while (ptr && !head.compare_exchange_weak(
                old, reinterpret_cast<uintptr_t>(ptr->next),
                std::memory_order_release, std::memory_order_relaxed)) {
            ptr = reinterpret_cast<Node*>(old);
        }
        return ptr;
    }

    void push(Node* n) noexcept {
        uintptr_t old = head.load(std::memory_order_relaxed);
        do {
            n->next = reinterpret_cast<Node*>(old);
        } while (!head.compare_exchange_weak(
                old, reinterpret_cast<uintptr_t>(n),
                std::memory_order_release, std::memory_order_relaxed));
    }
};

// ============================================================================
// Task Ring — SPMC ring buffer (same pattern as Venus jobs.c)
// ============================================================================

static constexpr uint32_t kTaskRingSize = config::kTaskRingSize;
static constexpr uint32_t kTaskRingMask = kTaskRingSize - 1;

struct alignas(64) TaskRing {
    Task ring[kTaskRingSize];

    alignas(64) std::atomic<uint64_t> head;   // Next slot to publish
    alignas(64) std::atomic<uint64_t> tail;   // Last claimed slot

    void init() noexcept {
        memset(ring, 0, sizeof(ring));
        head.store(0, std::memory_order_relaxed);
        tail.store(0, std::memory_order_relaxed);
    }

    /// Submit a task (producer side). Returns false if ring is full.
    bool submit(TaskFn fn, void* arg) noexcept {
        uint64_t seq = head.load(std::memory_order_relaxed);

        // Backpressure: check if ring is full
        if (seq - tail.load(std::memory_order_acquire) >= kTaskRingSize)
            return false;

        ring[seq & kTaskRingMask] = Task{fn, arg};
        std::atomic_thread_fence(std::memory_order_release);
        head.store(seq + 1, std::memory_order_release);
        return true;
    }

    /// Submit with backpressure wait (spins until space available).
    void submit_wait(TaskFn fn, void* arg, SpinPolicy policy) noexcept {
        uint32_t spins = 0;
        while (!submit(fn, arg)) {
            spin_wait(policy, spins, kDefaultSpinCount);
        }
    }

    /// Claim and execute one task (consumer/worker side).
    /// Returns true if a task was executed.
    ///
    /// COPY-THEN-CLAIM (the ordering is load-bearing; do not "simplify" it
    /// back to claim-then-read). `submit()`'s only backpressure is
    /// `seq - tail >= kTaskRingSize`, and `tail` advances at CLAIM time — so
    /// the instant a worker bumps tail from t to t+1, the producer is free to
    /// overwrite `ring[t & kTaskRingMask]` with sequence t + kTaskRingSize.
    /// Reading the slot AFTER the claim therefore races the producer for one
    /// full lap of the ring, and on a saturated ring (producer faster than the
    /// workers, which is the normal state for a long-running fan-out) that
    /// window is open on essentially every claim: the worker runs the
    /// producer's NEW task and the task it actually claimed is never run at
    /// all. The failure is silent — exactly one duplicate execution per lost
    /// one, so task-count bookkeeping still balances and `wait_all()` is
    /// satisfied. Measured on this ring at 26,364 tasks / grain 1 / 6 workers:
    /// up to 19,710 of 26,364 tasks NOT RUN with an exactly equal number of
    /// duplicate executions (see test_bolt_scheduler_ring_exactly_once).
    ///
    /// Copying first closes it with no extra synchronization: the producer may
    /// only overwrite slot `t` once tail > t, and a SUCCESSFUL
    /// compare_exchange with expected == t proves tail was still t at claim
    /// time — hence the copy, sequenced before the release CAS, read the slot
    /// while it was still ours. A FAILED CAS means some other worker owns t;
    /// our copy is simply discarded unused. The release CAS also forbids
    /// sinking the copy below it.
    bool try_claim_and_execute() noexcept {
        uint64_t t = tail.load(std::memory_order_acquire);
        const uint64_t h = head.load(std::memory_order_acquire);
        if (t >= h) return false;  // Empty
        // NOTE: do NOT assert (h - t <= kTaskRingSize) here. `t` and `h` are
        // loaded at different instants; another consumer can advance tail
        // between the two loads, so a stale `t` legitimately trails `h` by more
        // than one ring. Measured: that assert fires within a second under a
        // saturated multi-consumer ring.

        // Copy BEFORE claiming — see the contract above.
        const Task task = ring[t & kTaskRingMask];

        if (!tail.compare_exchange_weak(t, t + 1,
                std::memory_order_release, std::memory_order_relaxed))
            return false;  // Another worker claimed it; discard the copy.

        if (task.fn) task.fn(task.arg);
        return true;
    }

    /// Wait until all submitted tasks have been claimed.
    void wait_all(SpinPolicy policy) noexcept {
        uint64_t target = head.load(std::memory_order_acquire);
        uint32_t spins = 0;
        while (tail.load(std::memory_order_acquire) < target) {
            spin_wait(policy, spins, kDefaultSpinCount);
        }
    }
};

// ============================================================================
// Worker Configuration
// ============================================================================

struct WorkerConfig {
    uint32_t   worker_id;
    int        cpu_affinity;    // -1 = no affinity
    SpinPolicy spin_policy;
    uint32_t   spin_count;      // Max spins before escalation
};

static constexpr uint32_t kMaxWorkers = config::kMaxWorkers;

// ============================================================================
// Scheduler — owns workers, ring, pools, arenas
// ============================================================================

/// Bolt task scheduler. Init once, submit tasks, shutdown.
///
/// Workers spin on the TaskRing, claim tasks via CAS, execute.
/// Each worker has its own Arena for per-task temporaries.
///
/// Usage:
///   Scheduler sched;
///   sched.init(num_cores);
///   sched.submit(my_fn, my_arg);
///   sched.submit_range(range_fn, data, count, grain_size);
///   sched.wait_all();
///   sched.shutdown();
struct Scheduler {
    TaskRing ring;

    // Worker threads (std::thread — portable across MSVC / gcc / clang).
    std::thread workers[kMaxWorkers];
    uint32_t    num_workers;
    std::atomic<bool> shutdown_flag;

    // Per-worker arenas
    Arena*     worker_arenas[kMaxWorkers];

    // Per-worker config
    WorkerConfig worker_configs[kMaxWorkers];

    // Applied scheduler-level config (from init()).
    SchedulerConfig cfg;

    // Observed placement: UINT32_MAX means floating / not bound.
    uint32_t worker_cpus[kMaxWorkers];
    uint32_t worker_numa[kMaxWorkers];

    // Task pools for range/column tasks
    TaskPool*  range_task_pool;
    TaskPool*  column_task_pool;

    // Stats
    struct {
        std::atomic<uint64_t> tasks_submitted;
        std::atomic<uint64_t> tasks_completed;
    } stats;

    // ParkWait machinery (Wave 18a). Lock-free via std::atomic::wait /
    // notify (C++20), which maps to WaitOnAddress on Windows and futex
    // on Linux — no std::mutex or condition_variable. `submit_seq_` is
    // bumped on every task submission (and on shutdown) so parked
    // workers observing the old value are woken.
    alignas(64) std::atomic<uint64_t> submit_seq_;

    // E2 — adaptive morsel sizing (opt-in feedback).
    //
    // Callers may record the observed ns-per-row of their most recent
    // morsel; `recommended_grain_bytes()` uses an exponentially-weighted
    // average of those observations to suggest a grain within the
    // profile's bounded range [grain_bytes/4, grain_bytes*4]. The
    // default `submit_range` DOES NOT read the recommendation — callers
    // opt in by passing `recommended_grain_bytes(elem_size)` instead of
    // the static `grain_bytes()`. Keeps the hot path unchanged.
    //
    // Reset by `init()`; thread-safe via atomic<double> on platforms
    // that provide it (all supported compilers do for IEEE 64-bit).
    struct {
        std::atomic<double>  ns_per_row_ewma;  // 0.0 until first observation
        std::atomic<uint64_t> samples;         // monotonic observation count
    } adaptive;

    // Record one observation. Typical call site: wrap a kernel invocation
    // in a Clock::now() pair, compute `(t1-t0).count() / rows`, call this
    // once per morsel. alpha = 0.25 → fast enough to track shape changes
    // across 4-8 morsels, slow enough not to chase single-morsel noise.
    void record_morsel_ns_per_row(double observed) noexcept {
        assert(observed >= 0.0);
        if (!(observed > 0.0)) return;  // guard NaN / zero
        double cur = adaptive.ns_per_row_ewma.load(std::memory_order_relaxed);
        double next = (cur > 0.0) ? (0.75 * cur + 0.25 * observed) : observed;
        adaptive.ns_per_row_ewma.store(next, std::memory_order_relaxed);
        adaptive.samples.fetch_add(1u, std::memory_order_relaxed);
    }

    // Suggest a morsel grain size (bytes) for the next submit_range. Uses
    // the EWMA observed ns/row to keep each morsel in the 1-10 ms band:
    //   target_rows = clamp(10 ms / ns_per_row, 1024, grain_bytes * 64 / elem_size)
    // and returns target_rows * elem_size clamped to
    // [grain_bytes/4, grain_bytes*4]. Returns the profile static if no
    // observations exist yet.
    uint32_t recommended_grain_bytes(size_t elem_size) const noexcept {
        assert(elem_size > 0);
        const uint32_t base = cfg.grain_bytes;
        const uint64_t samples = adaptive.samples.load(std::memory_order_relaxed);
        if (samples == 0u) return base;
        const double ns_per_row = adaptive.ns_per_row_ewma.load(std::memory_order_relaxed);
        if (!(ns_per_row > 0.0)) return base;

        // Target band: 1 ms lower bound (avoid per-morsel overhead
        // dominating), 10 ms upper bound (keep scheduler responsive).
        constexpr double kLowerNs = 1'000'000.0;
        constexpr double kUpperNs = 10'000'000.0;
        const double current_rows = static_cast<double>(base) / static_cast<double>(elem_size);
        const double current_ns   = current_rows * ns_per_row;

        double target_rows = current_rows;
        if (current_ns < kLowerNs) target_rows *= 2.0;
        else if (current_ns > kUpperNs) target_rows *= 0.5;

        const double min_rows = static_cast<double>(base / 4u) / static_cast<double>(elem_size);
        const double max_rows = static_cast<double>(base * 4u) / static_cast<double>(elem_size);
        if (target_rows < min_rows) target_rows = min_rows;
        if (target_rows > max_rows) target_rows = max_rows;

        const uint64_t grain = static_cast<uint64_t>(target_rows) * elem_size;
        // Keep within the caller's 32-bit grain contract.
        if (grain > 0xFFFFFFFFu) return 0xFFFFFFFFu;
        if (grain < 64u) return 64u;
        return static_cast<uint32_t>(grain);
    }

    // =====================================================================
    // Init / Shutdown
    // =====================================================================

    bool init(uint32_t num_threads, SpinPolicy default_policy = SpinPolicy::SpinYield) noexcept;
    bool init(const SchedulerConfig& cfg) noexcept;
    void shutdown() noexcept;

    // =====================================================================
    // Submit (all noexcept, never allocate on hot path)
    // =====================================================================

    /// Fire-and-forget task.
    void submit(TaskFn fn, void* arg) noexcept {
        ring.submit_wait(fn, arg, SpinPolicy::SpinYield);
        stats.tasks_submitted.fetch_add(1, std::memory_order_relaxed);
        // Wave 18a — wake any ParkWait worker that may be sleeping on
        // `submit_seq_`. Release-order bump ensures the ring push above
        // happens-before the wake observer's acquire-load in the worker.
        submit_seq_.fetch_add(1, std::memory_order_release);
        submit_seq_.notify_one();
    }

    /// Range task — auto-subdivides [0, count) across workers.
    void submit_range(RangeTaskFn fn, void* user_data,
                      uint32_t count, uint32_t grain_size = 512) noexcept;

    /// Column task — range task specialized for (BoltBatch, start, end).
    void submit_column_task(ColumnTaskFn fn, const void* read_batch,
                            void* write_batch, uint32_t entity_count,
                            float delta_time, uint32_t grain_size = 512) noexcept;

    /// Wait for all submitted tasks to complete. Uses the tasks_submitted /
    /// tasks_completed counters (incremented by the range/column trampolines)
    /// for a precise post-execution barrier — ring tail advances at claim time,
    /// which is not sufficient to guarantee the task body has finished.
    void wait_all() noexcept {
        const uint64_t target = stats.tasks_submitted.load(std::memory_order_acquire);
        uint32_t spins = 0;
        while (stats.tasks_completed.load(std::memory_order_acquire) < target) {
            spin_wait(SpinPolicy::SpinYield, spins, kDefaultSpinCount);
        }
        // Also fence the ring to be safe for raw submit() consumers.
        ring.wait_all(SpinPolicy::SpinYield);
    }

    /// Get the arena for a specific worker thread.
    Arena* worker_arena(uint32_t worker_id) noexcept {
        assert(worker_id < num_workers);
        return worker_arenas[worker_id];
    }

    /// Reset (bump-pointer rewind, not free) every worker arena for reuse.
    /// MUST be called only when NO tasks are in flight — it does not
    /// synchronize. A long-lived warm pool otherwise only reclaims per-task
    /// worker-arena allocations at shutdown(); a borrowing operator that
    /// allocates output from tl_arena would grow memory per call. Callers that
    /// reset between work units (e.g. per query, between scheduler-free points)
    /// keep worker-arena usage bounded. (G2FEAT-146)
    /// Retained warm bytes per worker arena across queries. Covers the full
    /// natural doubling progression (4+8+16+32+64+64+64 MB ≈ 252 MB) so a
    /// steady workload never re-mallocs, while bespoke oversize blocks from a
    /// heavy query are returned. Without trimming, the 32-slot block table
    /// ratchets monotonically for the process lifetime — measured saturation
    /// in ~3 heavy queries, after which allocations FAIL and, before that,
    /// throughput quietly degrades ~2.8x (most reserved bytes stranded).
    static constexpr size_t kWorkerArenaKeepBytes = 256ull * 1024 * 1024;

    void reset_worker_arenas() noexcept {
        for (uint32_t i = 0; i < num_workers; ++i) {
            if (worker_arenas[i] != nullptr) {
                worker_arenas[i]->reset_keep(kWorkerArenaKeepBytes);
            }
        }
    }

    uint32_t thread_count() const noexcept { return num_workers; }

    uint32_t worker_cpu (uint32_t worker_id) const noexcept {
        assert(worker_id < num_workers);
        return worker_cpus[worker_id];
    }
    uint32_t worker_node(uint32_t worker_id) const noexcept {
        assert(worker_id < num_workers);
        return worker_numa[worker_id];
    }
    uint32_t grain_bytes() const noexcept { return cfg.grain_bytes; }
    const SchedulerConfig& config() const noexcept { return cfg; }
};

// ============================================================================
// Scheduler implementation
// ============================================================================

/// Payload stored in range_task_pool. Also reused by submit_column_task via a
/// discriminated layout — column tasks carry extra pointers but fit in the
/// same pool node size (128 bytes). We keep two pools to avoid cross-type
/// contention on the Treiber stack.
struct RangeTaskPayload {
    // First field must be TaskPool::Node-compatible (pointer-sized) for pool reuse.
    void*         pool_next_slot;  // Unused after acquire; preserved for layout.
    RangeTaskFn   fn;
    void*         user_data;
    uint32_t      start;
    uint32_t      end;
    uint32_t      thread_id;
    uint32_t      _pad0;
    TaskPool*     owning_pool;
    Scheduler*    owning_sched;
};
static_assert(sizeof(RangeTaskPayload) <= 128, "RangeTaskPayload must fit pool slot");

struct ColumnTaskPayload {
    void*         pool_next_slot;
    ColumnTaskFn  fn;
    const void*   read_batch;
    void*         write_batch;
    uint32_t      start;
    uint32_t      end;
    float         delta_time;
    uint32_t      thread_id;
    TaskPool*     owning_pool;
    Scheduler*    owning_sched;
};
static_assert(sizeof(ColumnTaskPayload) <= 128, "ColumnTaskPayload must fit pool slot");

static constexpr size_t   kSchedulerPoolSlot     = config::kSchedulerPoolSlotBytes;
static constexpr uint32_t kSchedulerPoolCapacity = config::kSchedulerPoolCapacity;

/// Worker index of the current scheduler thread (UINT32_MAX off-pool).
/// Set by scheduler_worker_loop alongside tl_arena; the task trampolines
/// pass it as the RangeTaskFn/ColumnTaskFn `thread_id` so tasks can index
/// per-worker resources. Before this existed every task received the
/// payload's constant 0 ("not per-worker yet") — callers that indexed
/// per-thread scratch by thread_id silently shared slot 0 across ALL
/// workers (measured: chukonu W-J5 parallel-probe pair-buffer race -> AV).
inline thread_local uint32_t tl_worker_index = UINT32_MAX;

inline void scheduler_range_trampoline(void* arg) noexcept {
    assert(arg != nullptr);
    RangeTaskPayload* p = static_cast<RangeTaskPayload*>(arg);
    assert(p->fn != nullptr);
    assert(p->end >= p->start);
    const uint32_t tid =
        (tl_worker_index != UINT32_MAX) ? tl_worker_index : p->thread_id;
    p->fn(p->user_data, p->start, p->end, tid);
    Scheduler* sched = p->owning_sched;
    TaskPool* pool = p->owning_pool;
    pool->release(p);
    if (sched) {
        // RELEASE, not relaxed: wait_all() acquire-loads this counter and then
        // reads what the task WROTE. A relaxed increment publishes the count
        // without publishing the task's stores, so the barrier would let the
        // waiter observe a completed task's memory as it was before the task
        // ran. Release here + acquire there is the only synchronizes-with edge
        // between a worker finishing and the submitter proceeding (the ring's
        // tail CAS happens at CLAIM time, before the body, so it publishes
        // nothing about the result). Cost is one release RMW per task.
        sched->stats.tasks_completed.fetch_add(1, std::memory_order_release);
    }
}

inline void scheduler_column_trampoline(void* arg) noexcept {
    assert(arg != nullptr);
    ColumnTaskPayload* p = static_cast<ColumnTaskPayload*>(arg);
    assert(p->fn != nullptr);
    assert(p->end >= p->start);
    const uint32_t tid =
        (tl_worker_index != UINT32_MAX) ? tl_worker_index : p->thread_id;
    p->fn(p->read_batch, p->write_batch, p->start, p->end, p->delta_time, tid);
    Scheduler* sched = p->owning_sched;
    TaskPool* pool = p->owning_pool;
    pool->release(p);
    if (sched) {
        // RELEASE, not relaxed: wait_all() acquire-loads this counter and then
        // reads what the task WROTE. A relaxed increment publishes the count
        // without publishing the task's stores, so the barrier would let the
        // waiter observe a completed task's memory as it was before the task
        // ran. Release here + acquire there is the only synchronizes-with edge
        // between a worker finishing and the submitter proceeding (the ring's
        // tail CAS happens at CLAIM time, before the body, so it publishes
        // nothing about the result). Cost is one release RMW per task.
        sched->stats.tasks_completed.fetch_add(1, std::memory_order_release);
    }
}

inline void scheduler_worker_loop(Scheduler* sched, uint32_t worker_id) noexcept {
    assert(sched != nullptr);
    assert(worker_id < kMaxWorkers);

    // ------------------------------------------------------------------------
    // Pinning + NUMA binding: MUST happen before tl_arena is set so that the
    // arena's first-touch pages land on the correct NUMA node (first-touch
    // policy on Linux; Windows uses the preferred node hint for VirtualAlloc).
    // ------------------------------------------------------------------------
    const uint32_t intended_cpu = sched->worker_cpus[worker_id];
    if (intended_cpu != UINT32_MAX) {
        if (!bolt_pin_current_thread(intended_cpu)) {
            // Honest diagnostics: record the failure.
            sched->worker_cpus[worker_id] = UINT32_MAX;
        } else if (sched->cfg.numa_bind) {
            uint32_t node = 0;
            if (intended_cpu < kTopologyMaxCpus) {
                // cpu_to_node was captured into worker_numa during init as a
                // pre-computed hint; fall back to 0 if unset.
                uint32_t n = sched->worker_numa[worker_id];
                if (n != UINT32_MAX) node = n;
            }
            if (!bolt_set_numa_preferred(static_cast<int>(node))) {
                sched->worker_numa[worker_id] = UINT32_MAX;
            }
        }
    }

    tl_arena = sched->worker_arenas[worker_id];
    tl_worker_index = worker_id;   // per-worker task thread_id (see trampolines)
    const SpinPolicy policy = sched->worker_configs[worker_id].spin_policy;
    const uint32_t max_spins = sched->worker_configs[worker_id].spin_count;
    uint32_t spins = 0;

    if (policy == SpinPolicy::ParkWait) {
        // Wave 18a — real park/unpark via std::atomic::wait. No mutex,
        // no condition_variable. Submitters bump `submit_seq_` + call
        // notify_one / notify_all; shutdown() calls notify_all.
        // Re-check ring after wake to cover the race where notify fires
        // between the ring check and the wait() call.
        //
        // Spin-then-park (2026-07-10): a bounded spin grace before each
        // park. Pure ParkWait measured two-faced on the TAQ board: long
        // idle stretches parked (idle spin was 57-75% of process CPU under
        // SpinYield) but fork-join-heavy queries paid a futex wake per
        // task wave (+15% on small queries — the same effect as the W-J
        // neutral result on TPC-H). The grace (max_spins cpu_pause ~30 us,
        // then kParkGraceYields yields) bridges intra-query task gaps at
        // spin cost while whole-process idle still parks.
        constexpr uint32_t kParkGraceYields = 16;
        uint32_t grace = 0;
        const uint32_t grace_max = max_spins + kParkGraceYields;
        while (!sched->shutdown_flag.load(std::memory_order_acquire)) {
            if (sched->ring.try_claim_and_execute()) { grace = 0; continue; }
            if (grace < grace_max) {
                if (grace < max_spins) cpu_pause();
                else                   std::this_thread::yield();
                ++grace;
                continue;
            }
            const uint64_t seen = sched->submit_seq_.load(std::memory_order_acquire);
            // Re-check once after sampling the seq so we don't miss work
            // that arrived between the first try and the sample.
            if (sched->ring.try_claim_and_execute()) { grace = 0; continue; }
            if (sched->shutdown_flag.load(std::memory_order_acquire)) break;
            sched->submit_seq_.wait(seen, std::memory_order_acquire);
            grace = 0;
        }
    } else {
        while (!sched->shutdown_flag.load(std::memory_order_acquire)) {
            if (sched->ring.try_claim_and_execute()) {
                spins = 0;
            } else {
                spin_wait(policy, spins, max_spins);
            }
        }
    }

    // Drain any remaining tasks so wait_all() callers see completion even if
    // shutdown races with submission. Bounded drain; no new work can land after
    // shutdown_flag observe because callers must quiesce before shutdown().
    while (sched->ring.try_claim_and_execute()) { /* drain */ }

    tl_arena = nullptr;
    tl_worker_index = UINT32_MAX;
}

// Build the CPU assignment list: prefer P-cores first when requested, then
// E-cores, then any remaining logical CPU. Unused slots are UINT32_MAX.
// Writes exactly `want` entries into `out[0..want)`.
inline void scheduler_assign_cpus(const CpuTopology& topo,
                                  uint32_t want, bool prefer_p_cores,
                                  uint32_t* out) noexcept {
    assert(out != nullptr);
    assert(want <= kMaxWorkers);

    const uint32_t ncpu = (topo.logical_cpus < kTopologyMaxCpus)
        ? topo.logical_cpus : kTopologyMaxCpus;
    uint32_t filled = 0;

    if (prefer_p_cores && topo.performance_cores > 0) {
        for (uint32_t c = 0; c < ncpu && filled < want; ++c) {
            if (topo.cpu_is_perf[c]) out[filled++] = c;
        }
        for (uint32_t c = 0; c < ncpu && filled < want; ++c) {
            if (!topo.cpu_is_perf[c]) out[filled++] = c;
        }
    } else {
        for (uint32_t c = 0; c < ncpu && filled < want; ++c) {
            out[filled++] = c;
        }
    }
    while (filled < want) out[filled++] = UINT32_MAX;
}

inline bool Scheduler::init(uint32_t num_threads, SpinPolicy default_policy) noexcept {
    // Every num_threads value is legal here: init(SchedulerConfig) maps 0 to
    // bolt_get_hardware_concurrency() (the documented auto-size sentinel) and
    // clamps anything above kMaxWorkers. A precondition assert on num_threads
    // would abort on inputs this function is specified to accept.
    SchedulerConfig c{};
    c.num_workers = num_threads;
    c.spin        = default_policy;
    return init(c);
}

inline bool Scheduler::init(const SchedulerConfig& in_cfg) noexcept {
    assert(in_cfg.grain_bytes > 0);
    assert(in_cfg.dispatch_batch > 0);

    cfg = in_cfg;

    uint32_t num_threads = cfg.num_workers;
    if (num_threads == 0) num_threads = bolt_get_hardware_concurrency();
    if (num_threads == 0) num_threads = 1;
    if (num_threads > kMaxWorkers) num_threads = kMaxWorkers;

    // Defensive: clear owned pointers in case caller didn't value-init.
    range_task_pool = nullptr;
    column_task_pool = nullptr;
    for (uint32_t i = 0; i < kMaxWorkers; ++i) {
        worker_arenas[i] = nullptr;
        worker_cpus[i]   = UINT32_MAX;
        worker_numa[i]   = UINT32_MAX;
    }

    ring.init();
    shutdown_flag.store(false, std::memory_order_relaxed);
    stats.tasks_submitted.store(0, std::memory_order_relaxed);
    stats.tasks_completed.store(0, std::memory_order_relaxed);
    submit_seq_.store(0, std::memory_order_relaxed);
    adaptive.ns_per_row_ewma.store(0.0, std::memory_order_relaxed);
    adaptive.samples.store(0, std::memory_order_relaxed);
    num_workers = 0;

    // Detect topology on the stack (~2KB): not persisted after init.
    CpuTopology topo;
    bolt_detect_topology(&topo);

    // Pre-compute CPU assignments.
    uint32_t planned_cpus[kMaxWorkers];
    for (uint32_t i = 0; i < kMaxWorkers; ++i) planned_cpus[i] = UINT32_MAX;
    if (cfg.pin_workers) {
        scheduler_assign_cpus(topo, num_threads, cfg.prefer_p_cores, planned_cpus);
    }

    // Allocate per-worker arenas (startup allocation — permitted).
    for (uint32_t i = 0; i < num_threads; ++i) {
        // Deliberate config, not the default: a pool worker arena backs whole
        // OPERATOR states (join builds, aggregate tables) for the process
        // lifetime, and with the default 64 MB max block the 32-slot table
        // caps total capacity near ~1.8 GB — measured as spurious allocation
        // failures on 60M-row aggregates whose memory budget said 51 GiB was
        // fine. A 512 MB max block raises the ceiling to ~14 GB per worker;
        // reset_keep() (kWorkerArenaKeepBytes) still trims the warm set back
        // between queries, so steady-state footprint is unchanged.
        ArenaConfig wa_cfg{};
        wa_cfg.max_block_size = 512ull * 1024 * 1024;
        worker_arenas[i] = new (std::nothrow) Arena(wa_cfg);
        if (!worker_arenas[i]) {
            for (uint32_t j = 0; j < i; ++j) { delete worker_arenas[j]; worker_arenas[j] = nullptr; }
            return false;
        }
        worker_configs[i] = WorkerConfig{ i, -1, cfg.spin, kDefaultSpinCount };

        // Stamp intended placement. The worker thread will attempt to apply it
        // and overwrite with UINT32_MAX on failure for honest diagnostics.
        worker_cpus[i] = planned_cpus[i];
        if (cfg.numa_bind && planned_cpus[i] != UINT32_MAX
                          && planned_cpus[i] < kTopologyMaxCpus) {
            worker_numa[i] = topo.cpu_to_node[planned_cpus[i]];
        } else {
            worker_numa[i] = UINT32_MAX;
        }
    }

    range_task_pool  = TaskPool::create(kSchedulerPoolSlot, kSchedulerPoolCapacity);
    column_task_pool = TaskPool::create(kSchedulerPoolSlot, kSchedulerPoolCapacity);
    if (!range_task_pool || !column_task_pool) {
        if (range_task_pool)  { range_task_pool->destroy();  range_task_pool  = nullptr; }
        if (column_task_pool) { column_task_pool->destroy(); column_task_pool = nullptr; }
        for (uint32_t i = 0; i < num_threads; ++i) { delete worker_arenas[i]; worker_arenas[i] = nullptr; }
        return false;
    }

    // Spawn workers last so they see fully-initialised state.
    num_workers = num_threads;
    for (uint32_t i = 0; i < num_threads; ++i) {
        workers[i] = std::thread(scheduler_worker_loop, this, i);
    }
    return true;
}

inline void Scheduler::shutdown() noexcept {
    assert(num_workers <= kMaxWorkers);

    shutdown_flag.store(true, std::memory_order_release);
    // Wave 18a — wake any ParkWait workers sleeping on submit_seq_ so
    // they observe shutdown_flag. Bump + notify_all handles all workers
    // including those that sampled a stale seq value.
    submit_seq_.fetch_add(1, std::memory_order_release);
    submit_seq_.notify_all();
    for (uint32_t i = 0; i < num_workers; ++i) {
        if (workers[i].joinable()) workers[i].join();
    }
    num_workers = 0;

    if (range_task_pool)  { range_task_pool->destroy();  range_task_pool  = nullptr; }
    if (column_task_pool) { column_task_pool->destroy(); column_task_pool = nullptr; }

    for (uint32_t i = 0; i < kMaxWorkers; ++i) {
        if (worker_arenas[i]) { delete worker_arenas[i]; worker_arenas[i] = nullptr; }
    }
}

inline void Scheduler::submit_range(RangeTaskFn fn, void* user_data,
                                    uint32_t count, uint32_t grain_size) noexcept {
    assert(fn != nullptr);
    assert(grain_size > 0);
    assert(range_task_pool != nullptr);

    if (count == 0) return;

    for (uint32_t start = 0; start < count; start += grain_size) {
        uint32_t end = start + grain_size;
        if (end > count) end = count;

        void* slot = range_task_pool->acquire();
        if (!slot) return;  // Pool & fallback exhausted; caller's count stats will diverge.

        RangeTaskPayload* p = static_cast<RangeTaskPayload*>(slot);
        p->fn            = fn;
        p->user_data     = user_data;
        p->start         = start;
        p->end           = end;
        p->thread_id     = 0;  // Set by worker via tl_arena; not per-worker yet.
        p->owning_pool   = range_task_pool;
        p->owning_sched  = this;

        ring.submit_wait(&scheduler_range_trampoline, p, SpinPolicy::SpinYield);
        stats.tasks_submitted.fetch_add(1, std::memory_order_relaxed);
        // Wave 18a — wake any ParkWait worker that may be sleeping on
        // `submit_seq_`. Release-order bump ensures the ring push above
        // happens-before the wake observer's acquire-load in the worker.
        submit_seq_.fetch_add(1, std::memory_order_release);
        submit_seq_.notify_one();
    }
}

inline void Scheduler::submit_column_task(ColumnTaskFn fn, const void* read_batch,
                                          void* write_batch, uint32_t entity_count,
                                          float delta_time, uint32_t grain_size) noexcept {
    assert(fn != nullptr);
    assert(grain_size > 0);
    assert(column_task_pool != nullptr);

    if (entity_count == 0) return;

    for (uint32_t start = 0; start < entity_count; start += grain_size) {
        uint32_t end = start + grain_size;
        if (end > entity_count) end = entity_count;

        void* slot = column_task_pool->acquire();
        if (!slot) return;

        ColumnTaskPayload* p = static_cast<ColumnTaskPayload*>(slot);
        p->fn            = fn;
        p->read_batch    = read_batch;
        p->write_batch   = write_batch;
        p->start         = start;
        p->end           = end;
        p->delta_time    = delta_time;
        p->thread_id     = 0;
        p->owning_pool   = column_task_pool;
        p->owning_sched  = this;

        ring.submit_wait(&scheduler_column_trampoline, p, SpinPolicy::SpinYield);
        stats.tasks_submitted.fetch_add(1, std::memory_order_relaxed);
        // Wave 18a — wake any ParkWait worker that may be sleeping on
        // `submit_seq_`. Release-order bump ensures the ring push above
        // happens-before the wake observer's acquire-load in the worker.
        submit_seq_.fetch_add(1, std::memory_order_release);
        submit_seq_.notify_one();
    }
}

// ============================================================================
// Phase Barrier — multi-phase execution (Venus pattern)
// ============================================================================

/// Execution phases for structured pipeline processing.
/// Same pattern as Venus JOB_PHASE_PRE_SIM / SIM / POST_SIM / RENDER.
enum class Phase : uint8_t {
    Scan      = 0,   // Read from sources (Parquet, CDC, MarbleDB)
    Compute   = 1,   // Filter, project, join, aggregate
    Emit      = 2,   // Write to sinks (MarbleDB, Flight, Parquet)
    Housekeep = 3,   // Stats update, arena compact, GC
    NUM_PHASES
};

/// Phase barrier: submit tasks tagged with a phase, wait for phase completion.
struct PhaseBarrier {
    std::atomic<uint64_t> phase_submitted[static_cast<int>(Phase::NUM_PHASES)];
    std::atomic<uint64_t> phase_completed[static_cast<int>(Phase::NUM_PHASES)];

    void init() noexcept {
        for (int i = 0; i < static_cast<int>(Phase::NUM_PHASES); ++i) {
            phase_submitted[i].store(0, std::memory_order_relaxed);
            phase_completed[i].store(0, std::memory_order_relaxed);
        }
    }

    void submit(Phase p) noexcept {
        phase_submitted[static_cast<int>(p)].fetch_add(1, std::memory_order_relaxed);
    }

    void complete(Phase p) noexcept {
        phase_completed[static_cast<int>(p)].fetch_add(1, std::memory_order_relaxed);
    }

    void wait_phase(Phase p) noexcept {
        uint64_t target = phase_submitted[static_cast<int>(p)].load(std::memory_order_acquire);
        uint32_t spins = 0;
        while (phase_completed[static_cast<int>(p)].load(std::memory_order_acquire) < target) {
            spin_wait(SpinPolicy::SpinYield, spins, kDefaultSpinCount);
        }
    }

    void reset_all() noexcept {
        for (int i = 0; i < static_cast<int>(Phase::NUM_PHASES); ++i) {
            phase_submitted[i].store(0, std::memory_order_relaxed);
            phase_completed[i].store(0, std::memory_order_relaxed);
        }
    }
};

}  // namespace bolt
