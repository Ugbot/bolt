// I/O Dispatcher Implementation
//
// Bridges async I/O callbacks to coroutine awaitables using the
// worker thread pool architecture.

#include "bolt/api/net/io_dispatcher.h"
#include "bolt/bolt_channel.h"
#include "bolt/bolt_port.h"        // bolt_pin_current_thread
#include "bolt/bolt_topology.h"    // CpuTopology, bolt_detect_topology
#include "bolt/bolt_scheduler.h"   // scheduler_assign_cpus
#include <cassert>
#include <cerrno>
#include <cstdint>
#include <mutex>
#include <iostream>
#include <thread>
#include <utility>

namespace bolt::api {
namespace net {

// Per-IO-thread inbox: a bounded MPSC ring of coroutine handles to resume on the
// owning IO thread. Producers (any thread) post via post_to_io_thread; the single
// consumer is the IO thread, which drains at the top of each loop iteration.
// Capacity bounds startup posts and accepted-connection handoffs. Refusal sheds
// new connections instead of waiting on another reactor's progress.
struct IoInbox {
    static constexpr size_t kCapacity = 256;
    bolt::MPSCChannel<std::coroutine_handle<>, kCapacity> queue;
};

namespace {
// Resume every handle currently posted to this thread's inbox. Bounded by the
// ring capacity so a flood can't starve the poll loop. Runs on the owning IO
// thread (single consumer); each resume runs the coroutine up to its next
// co_await (pinned to this thread).
void drain_inbox(IoInbox* inbox) noexcept {
    if (inbox == nullptr) {
        return;
    }
    std::coroutine_handle<> h;
    size_t drained = 0;
    while (drained < IoInbox::kCapacity && inbox->queue.try_pop(&h)) {
        ++drained;
        if (h && !h.done()) {
            h.resume();
        }
    }
}
}  // namespace

// The engine the CURRENT thread polls, and the dispatcher that owns it. Set by
// io_thread_loop on each IO thread; cleared on exit. A single thread is an IO
// thread for at most one dispatcher at a time, so a thread_local pair is enough
// to route an inline-resumed coroutine's next op to its own thread's engine.
// The owner tag guards the (unusual) case of an op submitted on this thread for a
// DIFFERENT dispatcher — that falls back to the other dispatcher's engine 0.
namespace {
thread_local IODispatcher*    t_current_owner = nullptr;
thread_local core::async_io*  t_current_io    = nullptr;
thread_local size_t           t_current_index = SIZE_MAX;
}  // namespace

// =============================================================================
// Awaitable Implementations
// =============================================================================

void ReadAwaitable::await_suspend(std::coroutine_handle<> handle) noexcept {
    dispatcher_->reads_started_.fetch_add(1, std::memory_order_relaxed);

    // Capture state for callback
    ReadAwaitable* self = this;

    // Submit async read to the I/O engine
    int rc = dispatcher_->current_io()->read_async(
        fd_, buf_, len_,
        [self, handle](const core::io_event& event) {
            // Store result
            self->result_ = event.result;
            self->dispatcher_->reads_completed_.fetch_add(1, std::memory_order_relaxed);

            // Resume coroutine on worker thread
            self->dispatcher_->resume_on_worker(handle);
        },
        nullptr
    );

    if (rc != 0) {
        // Error submitting - set error result and resume immediately
        result_ = -1;
        dispatcher_->resume_on_worker(handle);
    }
}

void WriteAwaitable::await_suspend(std::coroutine_handle<> handle) noexcept {
    dispatcher_->writes_started_.fetch_add(1, std::memory_order_relaxed);

    WriteAwaitable* self = this;

    int rc = dispatcher_->current_io()->write_async(
        fd_, buf_, len_,
        [self, handle](const core::io_event& event) {
            self->result_ = event.result;
            self->dispatcher_->writes_completed_.fetch_add(1, std::memory_order_relaxed);
            self->dispatcher_->resume_on_worker(handle);
        },
        nullptr
    );

    if (rc != 0) {
        result_ = -1;
        dispatcher_->resume_on_worker(handle);
    }
}

void RecvFromAwaitable::await_suspend(std::coroutine_handle<> handle) noexcept {
    dispatcher_->reads_started_.fetch_add(1, std::memory_order_relaxed);

    RecvFromAwaitable* self = this;

    // Forward the caller-owned peer-address out-params straight to the backend.
    // They (and `buf_`) must outlive the op — the awaitable temporary lives
    // across the suspension point, and the caller guarantees `src_`/`srclen_`
    // storage outlives the co_await.
    int rc = dispatcher_->current_io()->recvfrom_async(
        fd_, buf_, len_, src_, srclen_,
        [self, handle](const core::io_event& event) {
            self->result_ = event.result;
            self->dispatcher_->reads_completed_.fetch_add(1, std::memory_order_relaxed);
            self->dispatcher_->resume_on_worker(handle);
        },
        nullptr
    );

    if (rc != 0) {
        result_ = -1;
        dispatcher_->resume_on_worker(handle);
    }
}

void SendToAwaitable::await_suspend(std::coroutine_handle<> handle) noexcept {
    dispatcher_->writes_started_.fetch_add(1, std::memory_order_relaxed);

    SendToAwaitable* self = this;

    int rc = dispatcher_->current_io()->sendto_async(
        fd_, buf_, len_, dst_, dstlen_,
        [self, handle](const core::io_event& event) {
            self->result_ = event.result;
            self->dispatcher_->writes_completed_.fetch_add(1, std::memory_order_relaxed);
            self->dispatcher_->resume_on_worker(handle);
        },
        nullptr
    );

    if (rc != 0) {
        result_ = -1;
        dispatcher_->resume_on_worker(handle);
    }
}

void AcceptAwaitable::await_suspend(std::coroutine_handle<> handle) noexcept {
    dispatcher_->accepts_started_.fetch_add(1, std::memory_order_relaxed);

    AcceptAwaitable* self = this;

    auto* engine = dispatcher_->accept_io(retirement_);
    int rc = engine == nullptr ? -1 : engine->accept_async(
        listen_fd_,
        [self, handle](const core::io_event& event) {
            self->result_fd_ = static_cast<int>(event.result);
            self->dispatcher_->accepts_completed_.fetch_add(1, std::memory_order_relaxed);
            // Tracked listener accepts are reactor control work. Keep their
            // registration/cancellation serialized even when client I/O opts
            // into worker resumption (inline_resume=false).
            if (self->retirement_ && self->dispatcher_->per_thread_engines()) handle.resume();
            else self->dispatcher_->resume_on_worker(handle);
        },
        nullptr, retirement_
    );

    if (rc != 0) {
#ifdef _WIN32
        const int admission_error = engine == nullptr ? WSAEINVAL : WSAGetLastError();
#else
        const int admission_error = engine == nullptr ? EINVAL : errno;
#endif
        result_fd_ = -1;
        if (retirement_) retirement_->admission_error.store(admission_error != 0 ? admission_error : EIO, std::memory_order_release);
        if (retirement_ && dispatcher_->per_thread_engines() && dispatcher_->current_io_thread_index() == retirement_->reactor) handle.resume();
        else dispatcher_->resume_on_worker(handle);
    }
}

core::AcceptRetirementStatus IODispatcher::try_retire_accepts(size_t reactor,
    core::AcceptRetirement& record, core::AcceptRetirementDeadline deadline) noexcept {
    if (record.complete.load(std::memory_order_acquire)) return core::AcceptRetirementStatus::Complete;
    if (!is_running() || reactor >= config_.num_io_threads) return core::AcceptRetirementStatus::Unavailable;
    if (per_thread_engines_ && reactor != record.reactor) return core::AcceptRetirementStatus::Error;
    const size_t index = per_thread_engines_ ? reactor : 0u;
    if (index >= ios_.size() || !ios_[index]) return core::AcceptRetirementStatus::Error;
    return ios_[index]->try_retire_accepts(record, deadline);
}

void ConnectAwaitable::await_suspend(std::coroutine_handle<> handle) noexcept {
    dispatcher_->connects_started_.fetch_add(1, std::memory_order_relaxed);

    ConnectAwaitable* self = this;

    int rc = dispatcher_->current_io()->connect_async(
        fd_, addr_, addrlen_,
        [self, handle](const core::io_event& event) {
            self->result_ = static_cast<int>(event.result);
            self->dispatcher_->connects_completed_.fetch_add(1, std::memory_order_relaxed);
            self->dispatcher_->resume_on_worker(handle);
        },
        nullptr
    );

    if (rc != 0) {
        result_ = -1;
        dispatcher_->resume_on_worker(handle);
    }
}

// =============================================================================
// IODispatcher Implementation
// =============================================================================

IODispatcher::IODispatcher(const IODispatcherConfig& config)
    : config_(config)
    , worker_pool_(config.worker_pool) {

    // Ensure at least 1 I/O thread (engine count is derived from this).
    if (config_.num_io_threads == 0) {
        config_.num_io_threads = 1;
    }

    // Create the async I/O engine(s). Thread-per-core: for a readiness/per-poll
    // backend (epoll/kqueue/io_uring) give EACH IO thread its OWN engine so there
    // is no shared poll set — ios_[i] is polled only by IO thread i. For a shared-
    // completion backend (IOCP) one engine is correct (the kernel distributes
    // completions across all threads polling the single completion port), so we
    // keep ONE engine that every IO thread polls — preserving the proven Windows
    // model. We probe the first engine's backend to decide.
    auto first = core::async_io::create(config_.io_config);
    per_thread_engines_ =
        first && (first->backend() == core::io_backend::epoll ||
                  first->backend() == core::io_backend::kqueue ||
                  first->backend() == core::io_backend::io_uring);
    ios_.reserve(per_thread_engines_ ? config_.num_io_threads : 1);
    ios_.emplace_back(std::move(first));
    if (per_thread_engines_) {
        for (size_t i = 1; i < config_.num_io_threads; ++i) {
            ios_.emplace_back(core::async_io::create(config_.io_config));
        }
    }

    // One inbox per IO thread (post_to_io_thread targets a thread, not an engine —
    // for the shared backend all threads share ios_[0] but still have own inboxes).
    inboxes_.reserve(config_.num_io_threads);
    for (size_t i = 0; i < config_.num_io_threads; ++i) {
        inboxes_.emplace_back(std::make_unique<IoInbox>());
    }
    // Invariant: at least one engine, and either one (shared) or exactly one per
    // IO thread (per-core). Both are valid for the io_thread_loop index math.
    assert(!ios_.empty() && "IODispatcher must have at least one async_io engine");
    assert((ios_.size() == 1 || ios_.size() == config_.num_io_threads) &&
           "engine count must be 1 (shared) or one-per-IO-thread");

    // If no worker pool provided, create one
    if (!worker_pool_) {
        owns_worker_pool_ = true;
        // Worker pool will be created and started in start()
    }
}

IODispatcher::IODispatcher(core::WorkerThreadPool* pool)
    : IODispatcher(IODispatcherConfig{.worker_pool = pool}) {}

IODispatcher::~IODispatcher() {
    stop();

    if (owns_worker_pool_ && worker_pool_) {
        delete worker_pool_;
    }
}

void IODispatcher::start() {
    if (running_.exchange(true)) {
        return;  // Already running
    }

    // Preserve a legacy pre-start publisher that overlaps first start. Owner
    // serializes start/stop/restart; clearing flags must never erase its count.
    post_state_.fetch_and(kPostCountMask, std::memory_order_acq_rel);
    shutdown_requested_.store(false, std::memory_order_relaxed);

    // Create worker pool if needed
    if (owns_worker_pool_ && !worker_pool_) {
        auto* pool = new core::WorkerThreadPool(core::WorkerPoolConfig{});
        pool->start();
        worker_pool_ = pool;
    }

    // Plan IO-thread CPU pinning (thread-per-core: each thread gets a dedicated
    // logical CPU so its working set stays in L1/L2). Topology detect is one-shot
    // on the stack — bolt_scheduler's standard pattern. UINT32_MAX entries mean
    // "no pin" (e.g. when pin_io_threads is false, or topology has fewer CPUs).
    planned_cpus_.assign(config_.num_io_threads, UINT32_MAX);
    if (config_.pin_io_threads) {
        bolt::CpuTopology topo;
        if (bolt::bolt_detect_topology(&topo)) {
            // Writes exactly num_io_threads entries; CPUs past the machine
            // stay UINT32_MAX (unpinned).
            bolt::scheduler_assign_cpus(
                topo, static_cast<uint32_t>(config_.num_io_threads),
                config_.prefer_p_cores, planned_cpus_.data());
        }
        // Topology-detect failure: planned_cpus_ stays UINT32_MAX → io_thread_loop
        // skips pinning, runtime degrades to OS scheduling (still correct).
    }

    // Start I/O threads. Explicit stack size (see stacked_thread.h) — with
    // inline_resume (the default), these threads resume request-handling
    // coroutines directly and run arbitrary handler code, so they must not
    // inherit the platform pthread default (512 KiB on macOS).
    io_threads_.reserve(config_.num_io_threads);
    for (size_t i = 0; i < config_.num_io_threads; ++i) {
        io_threads_.emplace_back(config_.stack_size_bytes,
                                  &IODispatcher::io_thread_loop, this, i);
    }

    std::cout << "IODispatcher started with " << config_.num_io_threads
              << " I/O thread(s) and "
              << (worker_pool_ ? worker_pool_->num_workers() : 0)
              << " worker threads" << std::endl;
}

void IODispatcher::stop() {
    if (!running_.exchange(false)) {
        return;  // Not running
    }

    // Close before final drain. Publishers admitted before this RMW retain a
    // count through both queue publication and engine wake; later entries refuse.
    post_state_.fetch_or(kPostsClosed, std::memory_order_acq_rel);
    while ((post_state_.load(std::memory_order_acquire) & kPostCountMask) != 0)
        std::this_thread::yield();
    shutdown_requested_.store(true, std::memory_order_release);

    // Stop the async I/O engine(s) (this will wake any blocked poll())
    for (auto& io : ios_) {
        if (io) {
            io->stop();
        }
    }

    // Join I/O threads
    for (auto& t : io_threads_) {
        if (t.joinable()) {
            t.join();
        }
    }
    io_threads_.clear();

    // Stop worker pool if we own it
    if (owns_worker_pool_ && worker_pool_) {
        worker_pool_->stop();
    }

    std::cout << "IODispatcher stopped" << std::endl;
}

void IODispatcher::io_thread_loop(size_t thread_id) {
    // Pin to the planned CPU FIRST so every subsequent alloc/runtime touch lands
    // on the right NUMA node / L2 (best-effort: failure is non-fatal — see
    // start()). UINT32_MAX means "no pin" (pin_io_threads off or unknown topo).
    if (thread_id < planned_cpus_.size() && planned_cpus_[thread_id] != UINT32_MAX) {
        // bolt_pin_current_thread is at GLOBAL scope (bolt_port.h declares it
        // outside namespace bolt — see line 321 of bolt_port.h).
        const bool pinned = ::bolt_pin_current_thread(planned_cpus_[thread_id]);
        std::cout << "IODispatcher I/O thread " << thread_id
                  << (pinned ? " pinned to CPU " : " pin FAILED for CPU ")
                  << planned_cpus_[thread_id] << std::endl;
    } else {
        std::cout << "IODispatcher I/O thread " << thread_id
                  << " started (no CPU pin)" << std::endl;
    }

    // Pick THIS thread's engine. Per-core backends have one engine per IO thread
    // (poll only ours); the shared backend (IOCP) has a single engine that every
    // IO thread polls. Both collapse to the same index math.
    const size_t engine_idx = (thread_id < ios_.size()) ? thread_id : 0;
    core::async_io* my_io = ios_[engine_idx].get();
    assert(my_io != nullptr && "io_thread_loop: null engine");

    // Publish the per-thread engine so awaitables resumed inline on this thread
    // (and ops submitted by coroutines pinned here) register on OUR engine — the
    // completion is then reaped by THIS thread (no cross-thread handoff). Tagged
    // with the owning dispatcher so a stray op for a different dispatcher on this
    // thread does not pick up our engine (it falls back to that dispatcher's #0).
    t_current_owner = this;
    t_current_io = my_io;
    t_current_index = thread_id;

    IoInbox* inbox = (thread_id < inboxes_.size()) ? inboxes_[thread_id].get() : nullptr;

    while (!shutdown_requested_.load(std::memory_order_acquire)) {
        // Drain handles posted to THIS thread (e.g. accept-loop startup), resuming
        // each up to its next co_await — they run pinned to this thread thereafter.
        drain_inbox(inbox);
        // Poll for I/O events with timeout
        // 1ms timeout to check shutdown periodically without burning CPU
        my_io->poll(1000);  // 1000 microseconds = 1ms
    }

    // Final drain of pending events
    drain_inbox(inbox);
    my_io->poll(0);

    t_current_io = nullptr;
    t_current_owner = nullptr;
    t_current_index = SIZE_MAX;

    std::cout << "IODispatcher I/O thread " << thread_id << " stopped" << std::endl;
}

void IODispatcher::resume_on_worker(std::coroutine_handle<> handle) noexcept {
    if (!handle || handle.done()) {
        return;
    }

    // Inline resume (default): run the coroutine on THIS event-loop thread up to
    // its next co_await — no per-request worker handoff. This is the hot path; the
    // coroutine suspends back to the poll loop at its next async op, so the stack is
    // bounded (no recursion). The worker pool is used only when inline_resume is
    // disabled (handlers that do blocking work inline).
    if (config_.inline_resume || worker_pool_ == nullptr) {
        handle.resume();
        return;
    }

    // Worker-pool handoff (opt-out path): submit; block-spin only if the queue is full.
    if (!worker_pool_->submit(handle)) {
        worker_pool_->submit_wait(handle);
    }
}

size_t IODispatcher::current_io_thread_index() const noexcept {
    const size_t index = t_current_owner == this ? t_current_index : SIZE_MAX;
    assert(t_current_owner != this || t_current_io != nullptr);
    assert(index == SIZE_MAX || index < config_.num_io_threads);
    return index;
}

core::async_io* IODispatcher::current_io() noexcept {
    // On our own IO thread: the engine that thread polls (ops register where they
    // complete — no cross-thread handoff). Otherwise engine 0 (e.g. an op from a
    // worker before the connection is pinned to an IO thread).
    if (t_current_owner == this && t_current_io != nullptr) {
        return t_current_io;
    }
    assert(!ios_.empty() && "current_io: no engine");
    return ios_[0].get();
}

core::async_io* IODispatcher::accept_io(core::AcceptRetirement* retirement) noexcept {
    if (retirement == nullptr) return current_io();
    const size_t index = per_thread_engines_ ? retirement->reactor : 0u;
    if (index >= ios_.size()) return nullptr;
    if (per_thread_engines_ && current_io_thread_index() != index) return nullptr;
    assert(retirement->fd >= 0);
    assert(ios_[index] != nullptr);
    return ios_[index].get();
}

int IODispatcher::accept_admission_error(const core::AcceptRetirement& record) const noexcept {
    const size_t index = per_thread_engines_ ? record.reactor : 0u;
    if (record.fd < 0 || index >= ios_.size() || !ios_[index]) return EINVAL;
    const auto backend = ios_[index]->backend();
    if ((backend == core::io_backend::epoll || backend == core::io_backend::kqueue) &&
        record.fd >= core::AcceptRetirement::kReadinessFdLimit) return EOVERFLOW;
    return 0;
}

bool IODispatcher::try_post_impl(size_t index, std::coroutine_handle<> h,
                                  bool allow_before_start) noexcept {
    assert(index < inboxes_.size());
    assert(h != nullptr);
    const uint64_t before = post_state_.fetch_add(1, std::memory_order_acq_rel);
    assert((before & kPostCountMask) < kPostCountMask);
    const bool initial = (before & kPostsBeforeStart) != 0;
    const bool allowed = (before & kPostsClosed) == 0 &&
        (initial ? allow_before_start : running_.load(std::memory_order_acquire));
    std::coroutine_handle<> item = h;
    const bool accepted = allowed && inboxes_[index]->queue.try_push_nowait(std::move(item));
    if (accepted || (allowed && !initial)) {
        const size_t engine = index < ios_.size() ? index : 0;
        assert(engine < ios_.size());
        ios_[engine]->wake();  // retain admission through final engine access
    }
    post_state_.fetch_sub(1, std::memory_order_release);
    return accepted;
}

bool IODispatcher::try_post_to_io_thread(size_t index,
                                        std::coroutine_handle<> h) noexcept {
    if (index >= inboxes_.size() || !h) return false;
    assert(index < config_.num_io_threads);
    assert(inboxes_[index] != nullptr);
    if (try_post_impl(index, h, false)) return true;
    inbox_posts_rejected_.fetch_add(1, std::memory_order_relaxed);
    return false;
}

bool IODispatcher::post_to_io_thread(size_t index,
                                     std::coroutine_handle<> h) noexcept {
    if (index >= inboxes_.size() || !h) return false;
    assert(index < config_.num_io_threads);
    assert(inboxes_[index] != nullptr);
    const bool on_target = t_current_owner == this && t_current_index == index;
    // Legacy capacity waiting stays outside admission so stop never waits for a
    // publisher blocked on the reactor whose final drain it is about to run.
    for (;;) {
        if (try_post_impl(index, h, true)) return true;
        if (on_target || !running_.load(std::memory_order_acquire) ||
            shutdown_requested_.load(std::memory_order_acquire)) {
            inbox_posts_rejected_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        std::this_thread::yield();
    }
}

void IODispatcher::async_close(int fd) noexcept {
    // Close on the engine the calling thread is pinned to: that is where this
    // fd's operations were registered. Generic io_uring close does not prove
    // pending read/write/UDP completion; tracked ACCEPT retirement is separate.
    if (core::async_io* io = current_io()) {
        io->close_async(fd);
    }
}

// =============================================================================
// Coroutine Task Wrappers
// =============================================================================

core::coro_task<ssize_t> IODispatcher::read(int fd, void* buf, size_t len) {
    ssize_t result = co_await async_read(fd, buf, len);
    co_return result;
}

core::coro_task<ssize_t> IODispatcher::write(int fd, const void* buf, size_t len) {
    ssize_t result = co_await async_write(fd, buf, len);
    co_return result;
}

core::coro_task<int> IODispatcher::accept(int listen_fd) {
    int result = co_await async_accept(listen_fd);
    co_return result;
}

core::coro_task<int> IODispatcher::connect(int fd, const struct sockaddr* addr, socklen_t addrlen) {
    int result = co_await async_connect(fd, addr, addrlen);
    co_return result;
}

// =============================================================================
// Statistics
// =============================================================================

IODispatcher::Stats IODispatcher::get_stats() const noexcept {
    return Stats{
        .reads_started = reads_started_.load(std::memory_order_relaxed),
        .reads_completed = reads_completed_.load(std::memory_order_relaxed),
        .writes_started = writes_started_.load(std::memory_order_relaxed),
        .writes_completed = writes_completed_.load(std::memory_order_relaxed),
        .accepts_started = accepts_started_.load(std::memory_order_relaxed),
        .accepts_completed = accepts_completed_.load(std::memory_order_relaxed),
        .connects_started = connects_started_.load(std::memory_order_relaxed),
        .connects_completed = connects_completed_.load(std::memory_order_relaxed),
    };
}

// =============================================================================
// Global Instance
// =============================================================================

static std::unique_ptr<IODispatcher> g_io_dispatcher;
static std::once_flag g_io_dispatcher_init_flag;

IODispatcher& global_io_dispatcher() {
    std::call_once(g_io_dispatcher_init_flag, []() {
        if (!g_io_dispatcher) {
            // Create with default config and global worker pool
            g_io_dispatcher = std::make_unique<IODispatcher>(
                &core::global_worker_pool()
            );
            g_io_dispatcher->start();
        }
    });
    return *g_io_dispatcher;
}

void init_global_io_dispatcher(core::WorkerThreadPool* pool) {
    std::call_once(g_io_dispatcher_init_flag, [pool]() {
        g_io_dispatcher = std::make_unique<IODispatcher>(pool);
        g_io_dispatcher->start();
    });
}

void init_global_io_dispatcher(const IODispatcherConfig& config) {
    std::call_once(g_io_dispatcher_init_flag, [&config]() {
        g_io_dispatcher = std::make_unique<IODispatcher>(config);
        g_io_dispatcher->start();
    });
}

} // namespace net
} // namespace bolt::api
