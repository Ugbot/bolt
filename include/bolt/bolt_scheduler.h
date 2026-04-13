// bolt_scheduler.h — Lock-free task scheduler with configurable spin policy
//
// Combines Venus job system patterns (SPMC ring, CAS claim, spin-then-yield,
// job pools, range subdivision, phase barriers) with FasterAPI's I/O dispatch.
//
// RULES: No exceptions. No RTTI. No smart pointers. No heap on hot path.
// All task data from pre-allocated pools. Ring buffer pre-allocated at init.

#pragma once

#include "bolt_arena.h"
#include "bolt_channel.h"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>

// Platform threads (no libuv dependency)
#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

namespace chukonu {
namespace bolt {

// ============================================================================
// Spin Policy
// ============================================================================

enum class SpinPolicy : uint8_t {
    BusySpin,     // Pure spin. Pin to core. Latency-critical (< 1μs).
    SpinYield,    // Spin N then yield(). Default for compute workers.
    ParkWait,     // futex/condvar. I/O-bound or infrequent work.
};

static constexpr uint32_t kDefaultSpinCount = 1000;

inline void spin_wait(SpinPolicy policy, uint32_t& spin_count, uint32_t max_spins) noexcept {
    switch (policy) {
        case SpinPolicy::BusySpin:
            cpu_pause();
            break;
        case SpinPolicy::SpinYield:
            if (++spin_count > max_spins) {
#ifdef _WIN32
                SwitchToThread();
#else
                sched_yield();
#endif
                spin_count = 0;
            } else {
                cpu_pause();
            }
            break;
        case SpinPolicy::ParkWait:
            // For proper park/unpark we'd use futex or condition_variable.
            // Fallback to yield for now — ParkWait workers should use
            // the IOBridge path with FasterAPI's event loop instead.
#ifdef _WIN32
            Sleep(0);
#else
            usleep(0);
#endif
            break;
    }
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

static constexpr uint32_t kTaskRingSize = 16384;  // 2^14, same as Venus
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
    bool try_claim_and_execute() noexcept {
        uint64_t t = tail.load(std::memory_order_acquire);
        uint64_t h = head.load(std::memory_order_acquire);
        if (t >= h) return false;  // Empty

        // CAS claim
        uint64_t next = t + 1;
        if (!tail.compare_exchange_weak(t, next,
                std::memory_order_release, std::memory_order_relaxed))
            return false;  // Another worker claimed it

        Task task = ring[next & kTaskRingMask];
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

static constexpr uint32_t kMaxWorkers = 64;

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

    // Worker threads
#ifdef _WIN32
    HANDLE     workers[kMaxWorkers];
#else
    pthread_t  workers[kMaxWorkers];
#endif
    uint32_t   num_workers;
    std::atomic<bool> shutdown_flag;

    // Per-worker arenas
    Arena*     worker_arenas[kMaxWorkers];

    // Per-worker config
    WorkerConfig worker_configs[kMaxWorkers];

    // Task pools for range/column tasks
    TaskPool*  range_task_pool;
    TaskPool*  column_task_pool;

    // Stats
    struct {
        std::atomic<uint64_t> tasks_submitted;
        std::atomic<uint64_t> tasks_completed;
    } stats;

    // =====================================================================
    // Init / Shutdown
    // =====================================================================

    bool init(uint32_t num_threads, SpinPolicy default_policy = SpinPolicy::SpinYield) noexcept;
    void shutdown() noexcept;

    // =====================================================================
    // Submit (all noexcept, never allocate on hot path)
    // =====================================================================

    /// Fire-and-forget task.
    void submit(TaskFn fn, void* arg) noexcept {
        ring.submit_wait(fn, arg, SpinPolicy::SpinYield);
        stats.tasks_submitted.fetch_add(1, std::memory_order_relaxed);
    }

    /// Range task — auto-subdivides [0, count) across workers.
    void submit_range(RangeTaskFn fn, void* user_data,
                      uint32_t count, uint32_t grain_size = 512) noexcept;

    /// Column task — range task specialized for (BoltBatch, start, end).
    void submit_column_task(ColumnTaskFn fn, const void* read_batch,
                            void* write_batch, uint32_t entity_count,
                            float delta_time, uint32_t grain_size = 512) noexcept;

    /// Wait for all submitted tasks to complete.
    void wait_all() noexcept {
        ring.wait_all(SpinPolicy::SpinYield);
    }

    /// Get the arena for a specific worker thread.
    Arena* worker_arena(uint32_t worker_id) noexcept {
        assert(worker_id < num_workers);
        return worker_arenas[worker_id];
    }

    uint32_t thread_count() const noexcept { return num_workers; }
};

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
}  // namespace chukonu
