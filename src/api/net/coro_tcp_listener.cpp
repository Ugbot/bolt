// Coroutine TCP Listener Implementation

#include "bolt/api/net/coro_tcp_listener.h"
#include "bolt/api/net/event_loop.h"
#include "bolt/api/core/logger.h"
#include "bolt/api/net/sys_compat.h"
#include <cassert>
#include <cerrno>
#include <cstring>
#include <iostream>

namespace bolt::api {
namespace net {

namespace {

// Binds the port once WITHOUT SO_REUSEPORT and releases it. SO_REUSEPORT on the
// real sockets is for this process's per-thread accept loops; if another
// process already holds the port with it set, our own binds would silently
// join its kernel load-balancing group. This bind fails instead.
int probe_port_exclusive(uint16_t port) {
    assert(port != 0);
    sys::startup();

    int fd = static_cast<int>(socket(AF_INET, SOCK_STREAM, 0));
    if (fd < 0) {
        return -1;
    }
#ifndef _WIN32
    // Tolerate our own TIME_WAIT leftovers; Windows SO_REUSEADDR would let the
    // probe steal a live port, so it is POSIX-only.
    int opt = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR,
                   reinterpret_cast<const char*>(&opt), sizeof(opt)) < 0) {
        const int saved = errno;
        sys::close_socket(fd);
        errno = saved;
        return -1;
    }
#endif
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    const int rc = bind(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr));
    const int saved = errno;
    sys::close_socket(fd);
    errno = saved;
    assert(rc == 0 || errno != 0);
    return rc < 0 ? -1 : 0;
}

} // namespace

// =============================================================================
// Constructor / Destructor
// =============================================================================

CoroTcpListener::CoroTcpListener(const CoroTcpListenerConfig& config, CoroConnectionHandler handler)
    : config_(config)
    , handler_(std::move(handler)) {

    // Auto-detect number of workers
    if (config_.num_workers == 0) {
        config_.num_workers = std::thread::hardware_concurrency();
        if (config_.num_workers == 0) {
            config_.num_workers = 4;
        }
    }

    // Update worker config
    config_.worker_config.num_workers = config_.num_workers;
}

CoroTcpListener::CoroTcpListener(const CoroTcpListenerConfig& config, CoroConnectionHandler handler,
                                 IODispatcher* shared_io_dispatcher, core::WorkerThreadPool* shared_worker_pool)
    : config_(config)
    , handler_(std::move(handler))
    , shared_io_dispatcher_(shared_io_dispatcher)
    , shared_worker_pool_(shared_worker_pool)
    , owns_resources_(false) {
    // Using shared resources - no need to create our own
}

CoroTcpListener::CoroTcpListener(uint16_t port, CoroConnectionHandler handler)
    : CoroTcpListener(CoroTcpListenerConfig{.port = port}, std::move(handler)) {}

CoroTcpListener::~CoroTcpListener() {
    stop();
}

// =============================================================================
// Start / Stop
// =============================================================================

int CoroTcpListener::start() {
    if (start_background() != 0) {
        return -1;
    }

    // Block until stop requested
    while (!stop_requested_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return 0;
}

int CoroTcpListener::start_background() {
    if (accept_record_count_ != 0u) { errno = EBUSY; return -1; }
    accepts_retired_ = false;
    if (running_.exchange(true)) {
        return -1;  // Already running
    }

    stop_requested_.store(false, std::memory_order_relaxed);
    accept_error_.store(0, std::memory_order_release);

    LOG_INFO("CORO_TCP", "Starting on %s:%d (shared_resources: %s)",
             config_.host.c_str(), config_.port,
             owns_resources_ ? "no" : "yes");

    if (config_.port != 0 && probe_port_exclusive(config_.port) != 0) {
        const int saved = errno;
        LOG_ERROR("CORO_TCP", "Port %d is already in use (%s); refusing to share it "
                  "with another listener", config_.port, strerror(saved));
        running_.store(false);
        accept_error_.store(saved, std::memory_order_release);
        errno = saved;
        return -1;
    }

    // Get pointers to worker pool and I/O dispatcher
    core::WorkerThreadPool* worker_pool = nullptr;
    IODispatcher* io_dispatcher = nullptr;

    if (owns_resources_) {
        // Create our own resources
        LOG_INFO("CORO_TCP", "Creating own resources (I/O threads: %zu, workers: %zu)",
                 config_.num_io_threads, config_.num_workers);

        // Create worker thread pool
        worker_pool_ = std::make_unique<core::WorkerThreadPool>(config_.worker_config);
        worker_pool_->start();

        // Create I/O dispatcher
        IODispatcherConfig io_config{
            .num_io_threads = config_.num_io_threads,
            .io_config = config_.io_config,
            .worker_pool = worker_pool_.get()
        };
        io_dispatcher_ = std::make_unique<IODispatcher>(io_config);
        io_dispatcher_->start();

        worker_pool = worker_pool_.get();
        io_dispatcher = io_dispatcher_.get();
    } else {
        // Use shared resources
        LOG_INFO("CORO_TCP", "Using shared I/O dispatcher and worker pool");

        if (!shared_io_dispatcher_ || !shared_worker_pool_) {
            LOG_ERROR("CORO_TCP", "Shared resources not provided");
            running_.store(false);
            accept_error_.store(EINVAL, std::memory_order_release); errno = EINVAL;
            return -1;
        }

        worker_pool = shared_worker_pool_;
        io_dispatcher = shared_io_dispatcher_;
    }

    // Linux per-core engines use one SO_REUSEPORT listener per reactor. macOS
    // the observed wildcard topology does not fan out these binds: accept once on
    // reactor 0 and transfer each fresh fd before its first I/O. IOCP is shared.
    const bool per_thread = io_dispatcher->per_thread_engines();
#ifdef __APPLE__
    const size_t accept_loops = 1;
#else
    const size_t accept_loops = per_thread ? io_dispatcher->io_thread_count() : 1;
#endif
    if (accept_loops > core::AcceptRetirement::kCapacity) {
        accept_error_.store(EINVAL, std::memory_order_release); stop(); errno = EINVAL; return -1;
    }
    accept_records_ = std::make_unique<core::AcceptRetirement[]>(accept_loops);
    accept_record_count_ = accept_loops;
    for (size_t i = 0u; i < accept_loops; ++i)
        accept_records_[i].loop_exited.store(true, std::memory_order_relaxed);

    auto unwind = [&]() {
        stop(); // published accept ownership uses the same retirement path
    };

    listen_fds_.reserve(accept_loops);
    for (size_t i = 0; i < accept_loops; ++i) {
        int fd = create_listen_socket();
        if (fd < 0) {
            const int saved = errno;
            LOG_ERROR("CORO_TCP", "Failed to create listen socket %zu", i);
            unwind();
            accept_error_.store(saved, std::memory_order_release);
            errno = saved;
            return -1;
        }
        listen_fds_.push_back(fd);
        accept_records_[i].fd = fd;
        accept_records_[i].reactor = static_cast<uint32_t>(i);
        const int admission_error = io_dispatcher->accept_admission_error(accept_records_[i]);
        if (admission_error != 0) {
            accept_records_[i].admission_error.store(admission_error, std::memory_order_release);
            accept_error_.store(admission_error, std::memory_order_release);
            unwind(); errno = admission_error; return -1;
        }
    }
    LOG_INFO("CORO_TCP", "Listening on %zu socket(s) (per-thread accept: %s)",
             listen_fds_.size(), per_thread ? "yes" : "no");

    // Start the accept loop(s). Each is a coroutine; release its handle (the
    // worker pool / IO thread owns it thereafter) and start it.
    for (size_t i = 0; i < accept_loops; ++i) {
        auto& record = accept_records_[i];
        record.loop_exited.store(false, std::memory_order_release); // before queue publication
        auto accept_task = accept_loop(&record);
        auto handle = accept_task.release();
        const bool ok = per_thread
            ? io_dispatcher->try_post_to_io_thread(i, handle)   // bounded startup admission
            : worker_pool->submit(handle);                  // shared: any worker
        if (!ok) {
            LOG_ERROR("CORO_TCP", "Failed to start accept loop %zu", i);
            if (handle) handle.destroy();
            record.loop_exited.store(true, std::memory_order_release);
            unwind();
            accept_error_.store(ENOBUFS, std::memory_order_release);
            errno = ENOBUFS;
            return -1;
        }
    }

    return 0;
}

void CoroTcpListener::stop() {
    if (!running_.load(std::memory_order_acquire) && accept_record_count_ == 0u) return;
    // Legacy cold owner-progress wrapper; no universal wall-clock bound. Must
    // run outside owned reactors/workers and before shared progress is stopped.
    while (try_retire_accepts(std::chrono::steady_clock::now() + std::chrono::milliseconds(10)) !=
           core::AcceptRetirementStatus::Complete) std::this_thread::yield();

    // Only stop resources if we own them
    if (owns_resources_) {
        // Stop I/O dispatcher first (stops I/O event loops)
        if (io_dispatcher_) {
            io_dispatcher_->stop();
        }

        // Then stop worker pool (waits for pending coroutines)
        if (worker_pool_) {
            worker_pool_->stop();
        }
    }

    LOG_INFO("CORO_TCP", "Stopped");
    listen_fds_.clear(); accept_records_.reset(); accept_record_count_ = 0u; accepts_retired_ = false;
}

core::AcceptRetirementStatus CoroTcpListener::try_retire_accepts(core::AcceptRetirementDeadline deadline) noexcept {
    if (accepts_retired_) return core::AcceptRetirementStatus::Complete;
    stop_requested_.store(true, std::memory_order_release);
    running_.store(false, std::memory_order_release);
    auto* io = dispatcher();
    bool pending = false;
    for (size_t i = 0u; i < accept_record_count_; ++i) {
        if (accept_records_[i].fd < 0) continue; // never published startup slot
        if (std::chrono::steady_clock::now() >= deadline) return core::AcceptRetirementStatus::Pending;
        if (io == nullptr) return core::AcceptRetirementStatus::Error;
        const auto result = io->try_retire_accepts(i, accept_records_[i], deadline);
        if (result == core::AcceptRetirementStatus::Error || result == core::AcceptRetirementStatus::Unavailable) return result;
        pending |= result != core::AcceptRetirementStatus::Complete;
    }
    if (pending) return core::AcceptRetirementStatus::Pending;
    accepts_retired_ = true; // storage reclamation belongs to legacy stop/start, not this bounded call
    return core::AcceptRetirementStatus::Complete;
}

// =============================================================================
// Accept Loop Coroutine
// =============================================================================

core::detached_task CoroTcpListener::accept_loop(core::AcceptRetirement* record) {
    const int listen_fd = record->fd;
    assert(listen_fd >= 0);
    assert(dispatcher() != nullptr);
    LOG_INFO("CORO_TCP", "Accept loop started on fd %d", listen_fd);

    IODispatcher* io_disp = dispatcher();

    while (!stop_requested_.load(std::memory_order_acquire)) {
        // Async accept - yields until connection ready
        int client_fd = co_await io_disp->async_accept(listen_fd, record);

        if (client_fd < 0) {
            if (stop_requested_.load(std::memory_order_relaxed)) {
                break;  // Clean shutdown
            }
            if (const int error = record->admission_error.load(std::memory_order_acquire); error != 0) {
                int empty = 0;
                accept_error_.compare_exchange_strong(empty, error, std::memory_order_acq_rel);
                stop_requested_.store(true, std::memory_order_release);
                running_.store(false, std::memory_order_release);
                LOG_ERROR("CORO_TCP", "Accept submission refused on fd %d: %s", listen_fd, strerror(error));
                break; // no registered operation; retain fd for checked retirement
            }
            // Accept error - continue trying
            LOG_ERROR("CORO_TCP", "Accept error: %s", strerror(errno));
            continue;
        }

        connections_accepted_.fetch_add(1, std::memory_order_relaxed);
        const uint64_t sequence = connections_total_.fetch_add(1, std::memory_order_relaxed);

        // Spawn connection handler
        spawn_connection(client_fd, sequence);
    }

    LOG_INFO("CORO_TCP", "Accept loop stopped");
    record->loop_exited.store(true, std::memory_order_release); // final listener/record access
}

// =============================================================================
// Connection Spawning
// =============================================================================

// Helper coroutine that stores its parameters in the coroutine frame
// This avoids the lambda capture lifetime issue.
// detached_task: self-destroys when the connection ends (closes the latent
// per-connection frame leak — frames previously suspended at final_suspend
// forever, masked by keep-alive's bounded connection count).
static core::detached_task connection_handler_coro(
    CoroConnectionHandler& handler,
    IODispatcher& io,
    int fd,
    std::atomic<uint64_t>& active_counter
) {
    // A queued start remains active but has no registered client I/O yet. The
    // dispatcher's final inbox drain must close it instead of starting fresh I/O
    // after stop. This does not cancel a child handler that has already started.
    if (io.is_running()) co_await handler(io, fd);
    else sys::close_socket(fd);
    active_counter.fetch_sub(1, std::memory_order_relaxed);
}

void CoroTcpListener::spawn_connection(int client_fd, uint64_t sequence) {
    assert(client_fd >= 0);
    assert(dispatcher() != nullptr);
    if (stop_requested_.load(std::memory_order_acquire) ||
        sys::set_nonblocking(client_fd) != 0) {
        sys::close_socket(client_fd);
        return;
    }
#ifndef __APPLE__
    (void)sequence;
#endif

    // Track active connections
    connections_active_.fetch_add(1, std::memory_order_relaxed);

    // Get the dispatcher (works for both owned and shared)
    IODispatcher* io_disp = dispatcher();

    // Create connection handler coroutine
    // Using a proper coroutine function ensures parameters are stored in the frame
    auto conn_task = connection_handler_coro(
        handler_, *io_disp, client_fd, connections_active_);

    // Release ownership from coro_task (the handle is detached; lifetime is the
    // same as before — driven to completion by inline I/O resumes).
    auto handle = conn_task.release();

    if (io_disp->per_thread_engines()) {
#ifdef __APPLE__
        const size_t reactors = io_disp->io_thread_count();
        assert(reactors > 0);
        const size_t target = static_cast<size_t>(sequence % reactors);
        if (target != io_disp->current_io_thread_index()) {
            // No client-fd operation has registered yet. Success transfers both
            // frame and fd ownership; never touch the frame after publication.
            if (!io_disp->try_post_to_io_thread(target, handle)) {
                handle.destroy();
                sys::close_socket(client_fd);
                connections_active_.fetch_sub(1, std::memory_order_relaxed);
            }
            return;
        }
#endif
        // We are running ON the accepting IO thread. Resume inline so the
        // connection's reads/writes register on THIS thread's engine — the
        // connection is PINNED here for its lifetime (no worker handoff, best
        // cache locality). Runs up to the first co_await, then returns to the
        // accept loop. Bounded depth (the connection suspends on its first read).
        handle.resume();
    } else {
        // Shared backend (IOCP): hand to the worker pool; the kernel's completion
        // port distributes the connection's I/O across IO threads (proven model).
        core::WorkerThreadPool* wp = worker_pool();
        if (!wp->submit(handle)) {
            wp->submit_wait(handle);  // Queue full - blocking submit
        }
    }
}

// =============================================================================
// Listen Socket Creation
// =============================================================================

int CoroTcpListener::create_listen_socket() {
    // Ensure Winsock is initialized before any socket call (no-op on POSIX).
    sys::startup();

    int fd = static_cast<int>(socket(AF_INET, SOCK_STREAM, 0));
    if (fd < 0) {
        LOG_ERROR("CORO_TCP", "Failed to create socket: %s", strerror(errno));
        return -1;
    }

    // SO_REUSEADDR
    int opt = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR,
                   reinterpret_cast<const char*>(&opt), sizeof(opt)) < 0) {
        LOG_ERROR("CORO_TCP", "Failed to set SO_REUSEADDR: %s", strerror(errno));
        sys::close_socket(fd);
        return -1;
    }

    // SO_REUSEPORT (Linux/BSD only; not available on Windows)
#ifdef SO_REUSEPORT
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEPORT,
                   reinterpret_cast<const char*>(&opt), sizeof(opt)) < 0) {
        // Not fatal - continue anyway
        LOG_WARN("CORO_TCP", "Failed to set SO_REUSEPORT: %s", strerror(errno));
    }
#endif

    // Non-blocking
    if (sys::set_nonblocking(fd) < 0) {
        LOG_ERROR("CORO_TCP", "Failed to set non-blocking: %s", strerror(errno));
        sys::close_socket(fd);
        return -1;
    }

    // Bind
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;  // TODO: parse config_.host
    addr.sin_port = htons(config_.port);

    if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        LOG_ERROR("CORO_TCP", "Failed to bind to port %d: %s", config_.port, strerror(errno));
        sys::close_socket(fd);
        return -1;
    }

    // Listen
    if (listen(fd, config_.backlog) < 0) {
        LOG_ERROR("CORO_TCP", "Failed to listen: %s", strerror(errno));
        sys::close_socket(fd);
        return -1;
    }

    return fd;
}

// =============================================================================
// Statistics
// =============================================================================

CoroTcpListener::Stats CoroTcpListener::get_stats() const noexcept {
    return Stats{
        .connections_accepted = connections_accepted_.load(std::memory_order_relaxed),
        .connections_active = connections_active_.load(std::memory_order_relaxed),
        .connections_total = connections_total_.load(std::memory_order_relaxed),
    };
}

} // namespace net
} // namespace bolt::api
