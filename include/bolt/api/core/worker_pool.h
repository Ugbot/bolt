// Worker Thread Pool for Coroutine Execution
//
// Fixed-size thread pool that executes coroutines from a work queue.
// Designed for the coroutine-on-threadpool architecture where:
// - 1-2 event loops dispatch I/O events
// - N worker threads execute coroutines from the pool
//
// Features:
// - Lock-free MPMC task queue
// - Work stealing for load balancing (future)
// - Blocking work offload to separate pool
// - Integration with CoroPool for zero-allocation execution

#pragma once

// NOTE: core/coro_pool.h was cut from the fork (unused by the engine — no
// CoroPool/PooledPromise symbols are referenced here). Include severed.
#include "bolt/api/core/coro_task.h"
#include "bolt/api/core/stacked_thread.h"
#include "bolt/bolt_event_count.h"
#include "bolt/bolt_park_idle.h"
#include <atomic>
#include <cassert>
#include <coroutine>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

namespace bolt::api {
namespace core {

// =============================================================================
// MPMC Bounded Queue (for coroutine handle submission)
// =============================================================================

/// Bounded MPMC queue for any copyable type
/// Uses sequence numbers for lock-free operation
template <typename T, size_t Capacity = 4096>
class MPMCQueue {
    static_assert(Capacity > 0, "Capacity must be positive");
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");

public:
    using ValueType = T;

    MPMCQueue() : head_(0), tail_(0) {
        for (size_t i = 0; i < Capacity; ++i) {
            buffer_[i].sequence.store(i, std::memory_order_relaxed);
        }
    }

    ~MPMCQueue() = default;

    // Non-copyable, non-movable
    MPMCQueue(const MPMCQueue&) = delete;
    MPMCQueue& operator=(const MPMCQueue&) = delete;

    /// Try to push (multiple producers)
    bool try_push(ValueType handle) noexcept {
        Cell* cell;
        size_t pos = tail_.load(std::memory_order_relaxed);

        for (;;) {
            cell = &buffer_[pos & kMask];
            size_t seq = cell->sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);

            if (diff == 0) {
                if (tail_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false;  // Queue full
            } else {
                pos = tail_.load(std::memory_order_relaxed);
            }
        }

        cell->data = handle;
        cell->sequence.store(pos + 1, std::memory_order_release);
        return true;
    }

    /// Try to pop (multiple consumers)
    std::optional<ValueType> try_pop() noexcept {
        Cell* cell;
        size_t pos = head_.load(std::memory_order_relaxed);

        for (;;) {
            cell = &buffer_[pos & kMask];
            size_t seq = cell->sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);

            if (diff == 0) {
                if (head_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return std::nullopt;  // Queue empty
            } else {
                pos = head_.load(std::memory_order_relaxed);
            }
        }

        ValueType data = cell->data;
        cell->sequence.store(pos + Capacity, std::memory_order_release);
        return data;
    }

    /// Check if empty (approximate)
    bool empty() const noexcept {
        size_t head = head_.load(std::memory_order_relaxed);
        size_t tail = tail_.load(std::memory_order_relaxed);
        return head >= tail;
    }

    /// Get approximate size
    size_t size() const noexcept {
        size_t head = head_.load(std::memory_order_relaxed);
        size_t tail = tail_.load(std::memory_order_relaxed);
        return tail > head ? tail - head : 0;
    }

private:
    static constexpr size_t kMask = Capacity - 1;

    struct Cell {
        std::atomic<size_t> sequence;
        ValueType data;
    };

    alignas(kCacheLineSize) std::atomic<size_t> head_;
    alignas(kCacheLineSize) std::atomic<size_t> tail_;
    alignas(kCacheLineSize) Cell buffer_[Capacity];
};

// =============================================================================
// Worker Thread Pool
// =============================================================================

/// Configuration for WorkerThreadPool
struct WorkerPoolConfig {
    size_t num_workers = 0;           // 0 = hardware_concurrency
    size_t task_queue_size = 4096;    // Bounded queue capacity
    size_t blocking_workers = 2;      // Threads for blocking operations
    bool enable_work_stealing = false; // Future: work stealing between workers
    // Polls an idle worker makes before it parks (about 2 us); 0 parks at once.
#if defined(__aarch64__) || defined(_M_ARM64)
    uint32_t idle_spins = 1024;  // yield is a nop here
#else
    uint32_t idle_spins = 64;    // pause is 40-140 cycles
#endif
    // Stack size for every spawned worker (main + blocking) thread. These
    // threads resume request-handling coroutines and therefore run arbitrary
    // handler code (SQL/PromQL/etc. compilation, deeply recursive plan
    // building) — see stacked_thread.h for why this must not be the platform
    // pthread default. G2ICE-111/G2ICE-112.
    size_t stack_size_bytes = kDefaultStackBytes;
};

/// Thread pool that executes coroutine handles
class WorkerThreadPool {
public:
    /// Create pool with configuration
    explicit WorkerThreadPool(const WorkerPoolConfig& config = {});

    /// Create pool with simple worker count
    explicit WorkerThreadPool(size_t num_workers);

    ~WorkerThreadPool();

    // Non-copyable, non-movable
    WorkerThreadPool(const WorkerThreadPool&) = delete;
    WorkerThreadPool& operator=(const WorkerThreadPool&) = delete;

    /// Start the worker threads
    void start();

    /// Stop the pool (waits for workers to finish)
    void stop();

    /// Submit a coroutine handle for execution
    /// Returns true if submitted, false if queue full
    bool submit(std::coroutine_handle<> handle) noexcept;

    /// Submit and block until space available
    void submit_wait(std::coroutine_handle<> handle);

    /// One-shot callback admission. False on a stopped pool, full queue or a
    /// stop transition; never waits for a worker dequeue or another submitter.
    /// Accepted callbacks run exactly once, including stop's final queue drain.
    bool try_submit_blocking(std::function<void()> work) noexcept;

    /// Submit blocking work (file I/O, DNS lookup, etc.)
    /// Returns a future that will hold the result
    template <typename F>
    auto submit_blocking(F&& func) -> std::future<std::invoke_result_t<F>>;

    /// Get number of worker threads
    size_t num_workers() const noexcept { return workers_.size(); }

    /// Get number of pending tasks (approximate)
    size_t pending_tasks() const noexcept { return task_queue_.size(); }

    /// Check if pool is running
    bool is_running() const noexcept {
        return running_.load(std::memory_order_relaxed);
    }

    /// Statistics
    struct Stats {
        uint64_t tasks_executed;
        uint64_t tasks_submitted;
        uint64_t blocking_tasks_executed;
    };

    Stats get_stats() const noexcept;

private:
    void worker_loop(size_t worker_id);
    void blocking_worker_loop(size_t worker_id);

    // Next item, parking while the queue is empty; false once stop() asked.
    using IdleSet = ::bolt::park::IdleSetT<BOLT_MAX_WORKERS>;
    template <typename Queue>
    bool next_item(Queue& queue, IdleSet& ready, bolt::EventCount& space,
                   typename Queue::ValueType* out);
    void run_task(std::coroutine_handle<> handle);
    void run_blocking(std::function<void()>& work);

    WorkerPoolConfig config_;
    std::vector<StackedThread> workers_;
    std::vector<StackedThread> blocking_workers_;

    MPMCQueue<std::coroutine_handle<>, 4096> task_queue_;

    // Blocking task queue (lock-free, aligns with "No Mutexes" principle)
    // Uses a function wrapper queue for blocking operations (file I/O, DNS, etc.)
    MPMCQueue<std::function<void()>, 1024> blocking_task_queue_;

    // Idle workers park in *_ready_ (park::IdleSet: spinner rule, LIFO, chained
    // wake; G2CHK-1295), producers facing a full queue on *_space_.
    IdleSet task_ready_;
    bolt::EventCount task_space_;
    IdleSet blocking_ready_;
    bolt::EventCount blocking_space_;

    // Control
    std::atomic<bool> running_{false};
    std::atomic<bool> shutdown_requested_{false};

    // Stats
    alignas(kCacheLineSize) std::atomic<uint64_t> tasks_executed_{0};
    alignas(kCacheLineSize) std::atomic<uint64_t> tasks_submitted_{0};
    alignas(kCacheLineSize) std::atomic<uint64_t> blocking_tasks_executed_{0};
    // Seq-cst entry/exit pairs with running_'s stop exchange: a late submitter
    // must see stopped; an accepted publisher is included in stop's wait.
    alignas(kCacheLineSize) std::atomic<uint32_t> blocking_admissions_{0};
};

// =============================================================================
// Implementation
// =============================================================================

inline WorkerThreadPool::WorkerThreadPool(const WorkerPoolConfig& config)
    : config_(config) {
    if (config_.num_workers == 0) {
        config_.num_workers = std::thread::hardware_concurrency();
        if (config_.num_workers == 0) {
            config_.num_workers = 4;  // Fallback
        }
    }
}

inline WorkerThreadPool::WorkerThreadPool(size_t num_workers)
    : WorkerThreadPool(WorkerPoolConfig{.num_workers = num_workers}) {}

inline WorkerThreadPool::~WorkerThreadPool() {
    stop();
}

inline void WorkerThreadPool::start() {
    if (running_.exchange(true)) {
        return;  // Already running
    }

    shutdown_requested_.store(false, std::memory_order_relaxed);

    // Start main workers. Explicit stack size (see stacked_thread.h) — these
    // threads resume request-handling coroutines and run arbitrary handler
    // code, so they must not inherit the platform pthread default.
    workers_.reserve(config_.num_workers);
    for (size_t i = 0; i < config_.num_workers; ++i) {
        workers_.emplace_back(config_.stack_size_bytes,
                               &WorkerThreadPool::worker_loop, this, i);
    }

    // Start blocking workers (same stack-size requirement — offloaded
    // blocking work runs arbitrary handler code too).
    blocking_workers_.reserve(config_.blocking_workers);
    for (size_t i = 0; i < config_.blocking_workers; ++i) {
        blocking_workers_.emplace_back(config_.stack_size_bytes,
                                        &WorkerThreadPool::blocking_worker_loop, this, i);
    }
}

inline void WorkerThreadPool::stop() {
    if (!running_.exchange(false)) {
        return;  // Not running
    }

    // Prevent a successful nonblocking admission after workers' final drain.
    while (blocking_admissions_.load(std::memory_order_seq_cst) != 0)
        std::this_thread::yield();
    shutdown_requested_.store(true, std::memory_order_release);

    task_ready_.notify_all();
    task_space_.notify_all();
    blocking_ready_.notify_all();
    blocking_space_.notify_all();

    // Join all workers
    for (auto& t : workers_) {
        if (t.joinable()) {
            t.join();
        }
    }
    workers_.clear();

    for (auto& t : blocking_workers_) {
        if (t.joinable()) {
            t.join();
        }
    }
    blocking_workers_.clear();
}

inline bool WorkerThreadPool::try_submit_blocking(std::function<void()> work) noexcept {
    if (!work || config_.blocking_workers == 0) return false;
    const uint32_t before = blocking_admissions_.fetch_add(1, std::memory_order_seq_cst);
    assert(before < UINT32_MAX - 1);
    (void)before;
    const bool allowed = running_.load(std::memory_order_seq_cst) &&
                         !shutdown_requested_.load(std::memory_order_acquire);
    const bool accepted = allowed && blocking_task_queue_.try_push(std::move(work));
    assert(!accepted || allowed);
    if (accepted) blocking_ready_.notify();  // inside the admission stop() waits for
    blocking_admissions_.fetch_sub(1, std::memory_order_seq_cst);
    return accepted;
}

inline bool WorkerThreadPool::submit(std::coroutine_handle<> handle) noexcept {
    if (!handle || shutdown_requested_.load(std::memory_order_relaxed)) {
        return false;
    }

    if (task_queue_.try_push(handle)) {
        tasks_submitted_.fetch_add(1, std::memory_order_relaxed);
        task_ready_.notify();
        return true;
    }
    return false;
}

inline void WorkerThreadPool::submit_wait(std::coroutine_handle<> handle) {
    if (!handle) return;
    task_space_.await(config_.idle_spins, [&] {
        return submit(handle) || shutdown_requested_.load(std::memory_order_acquire);
    });
}

template <typename F>
auto WorkerThreadPool::submit_blocking(F&& func) -> std::future<std::invoke_result_t<F>> {
    using ReturnType = std::invoke_result_t<F>;
    auto task = std::make_shared<std::packaged_task<ReturnType()>>(std::forward<F>(func));
    auto future = task->get_future();

    // Use lock-free queue (no mutex, aligns with "No Mutexes" principle)
    std::function<void()> wrapped = [task]() { (*task)(); };

    blocking_space_.await(config_.idle_spins, [&] {
        if (blocking_task_queue_.try_push(wrapped)) {
            blocking_ready_.notify();
            return true;
        }
        return shutdown_requested_.load(std::memory_order_acquire);
    });

    return future;
}

inline WorkerThreadPool::Stats WorkerThreadPool::get_stats() const noexcept {
    return Stats{
        .tasks_executed = tasks_executed_.load(std::memory_order_relaxed),
        .tasks_submitted = tasks_submitted_.load(std::memory_order_relaxed),
        .blocking_tasks_executed = blocking_tasks_executed_.load(std::memory_order_relaxed),
    };
}

template <typename Queue>
bool WorkerThreadPool::next_item(Queue& queue, IdleSet& ready, bolt::EventCount& space,
                                 typename Queue::ValueType* out) {
    assert(out != nullptr);
    static thread_local ::bolt::park::Slot fallback;
    ::bolt::park::Slot* self = ::bolt::park::slot_self();
    if (self == nullptr) self = &fallback;
    bool stopping = false;
    for (;;) {
        const ::bolt::park::WaitResult r = ready.wait(self, [&]() noexcept {
            stopping = shutdown_requested_.load(std::memory_order_acquire);
            if (stopping) return true;
            auto item = queue.try_pop();
            if (!item) return false;
            *out = std::move(*item);
            return true;
        }, ::bolt::park::policy(::bolt::park::WaitClass::kPool));
        if (r == ::bolt::park::WaitResult::kReady) break;  // kDeadline: idle past the class deadline, wait again
    }
    if (stopping) return false;
    space.notify_one();
    // Chained wake: work left behind, or a slot ahead that was claimed but not
    // yet published when this worker looked.
    if (!queue.empty()) ready.notify();
    return true;
}

inline void WorkerThreadPool::run_task(std::coroutine_handle<> handle) {
    if (!handle || handle.done()) return;
    handle.resume();
    tasks_executed_.fetch_add(1, std::memory_order_relaxed);
}

inline void WorkerThreadPool::run_blocking(std::function<void()>& work) {
    if (!work) return;
    work();
    work = nullptr;
    blocking_tasks_executed_.fetch_add(1, std::memory_order_relaxed);
}

inline void WorkerThreadPool::worker_loop(size_t /*worker_id*/) {
    std::coroutine_handle<> handle;
    while (next_item(task_queue_, task_ready_, task_space_, &handle)) run_task(handle);
    // Drain remaining tasks on shutdown
    while (auto left = task_queue_.try_pop()) run_task(*left);
}

inline void WorkerThreadPool::blocking_worker_loop(size_t /*worker_id*/) {
    std::function<void()> work;
    while (next_item(blocking_task_queue_, blocking_ready_, blocking_space_, &work)) run_blocking(work);
    // Drain remaining tasks on shutdown
    while (auto left = blocking_task_queue_.try_pop()) run_blocking(*left);
}

// =============================================================================
// Global Pool Access
// =============================================================================

/// Get the global worker pool instance
WorkerThreadPool& global_worker_pool();

/// Initialize global worker pool
/// Must be called before submitting work
void init_global_worker_pool(const WorkerPoolConfig& config = {});

}  // namespace core
}  // namespace bolt::api
