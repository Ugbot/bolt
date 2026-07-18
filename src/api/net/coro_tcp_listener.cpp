// Coroutine TCP Listener Implementation

#include "bolt/api/net/coro_tcp_listener.h"
#include "bolt/api/net/event_loop.h"
#include "bolt/api/core/logger.h"
#include "bolt/api/net/sys_compat.h"
#include <cstring>
#include <iostream>

namespace bolt::api {
namespace net {

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
    if (running_.exchange(true)) {
        return -1;  // Already running
    }

    stop_requested_.store(false, std::memory_order_relaxed);

    LOG_INFO("CORO_TCP", "Starting on %s:%d (shared_resources: %s)",
             config_.host.c_str(), config_.port,
             owns_resources_ ? "no" : "yes");

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
            return -1;
        }

        worker_pool = shared_worker_pool_;
        io_dispatcher = shared_io_dispatcher_;
    }

    // Decide the accept topology. Per-thread-engine backends (epoll/kqueue/
    // io_uring) get ONE SO_REUSEPORT listen socket + ONE accept loop PER IO
    // thread, each started ON its thread so its async_accept registers on that
    // thread's own engine (kernel load-balances connections across the sockets →
    // accepts distribute, connections pin to the accepting thread). The shared
    // backend (IOCP) keeps a single socket + a single worker-pool accept loop.
    const bool per_thread = io_dispatcher->per_thread_engines();
    const size_t accept_loops = per_thread ? io_dispatcher->io_thread_count() : 1;

    auto unwind = [&]() {
        for (int fd : listen_fds_) {
            if (fd >= 0) sys::close_socket(fd);
        }
        listen_fds_.clear();
        if (owns_resources_) {
            io_dispatcher_->stop();
            worker_pool_->stop();
        }
        running_.store(false);
    };

    listen_fds_.reserve(accept_loops);
    for (size_t i = 0; i < accept_loops; ++i) {
        int fd = create_listen_socket();
        if (fd < 0) {
            LOG_ERROR("CORO_TCP", "Failed to create listen socket %zu", i);
            unwind();
            return -1;
        }
        listen_fds_.push_back(fd);
    }
    LOG_INFO("CORO_TCP", "Listening on %zu socket(s) (per-thread accept: %s)",
             listen_fds_.size(), per_thread ? "yes" : "no");

    // Start the accept loop(s). Each is a coroutine; release its handle (the
    // worker pool / IO thread owns it thereafter) and start it.
    for (size_t i = 0; i < accept_loops; ++i) {
        auto accept_task = accept_loop(listen_fds_[i]);
        auto handle = accept_task.release();
        const bool ok = per_thread
            ? io_dispatcher->post_to_io_thread(i, handle)   // run ON IO thread i
            : worker_pool->submit(handle);                  // shared: any worker
        if (!ok) {
            LOG_ERROR("CORO_TCP", "Failed to start accept loop %zu", i);
            if (handle) handle.destroy();
            unwind();
            return -1;
        }
    }

    return 0;
}

void CoroTcpListener::stop() {
    if (!running_.exchange(false)) {
        return;  // Not running
    }

    stop_requested_.store(true, std::memory_order_release);

    // Close listen socket(s) to break the accept loop(s).
    for (int fd : listen_fds_) {
        if (fd >= 0) {
            sys::close_socket(fd);
        }
    }
    listen_fds_.clear();

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
}

// =============================================================================
// Accept Loop Coroutine
// =============================================================================

core::detached_task CoroTcpListener::accept_loop(int listen_fd) {
    LOG_INFO("CORO_TCP", "Accept loop started on fd %d", listen_fd);

    IODispatcher* io_disp = dispatcher();

    while (!stop_requested_.load(std::memory_order_acquire)) {
        // Async accept - yields until connection ready
        int client_fd = co_await io_disp->async_accept(listen_fd);

        if (client_fd < 0) {
            if (stop_requested_.load(std::memory_order_relaxed)) {
                break;  // Clean shutdown
            }
            // Accept error - continue trying
            LOG_ERROR("CORO_TCP", "Accept error: %s", strerror(errno));
            continue;
        }

        connections_accepted_.fetch_add(1, std::memory_order_relaxed);
        connections_total_.fetch_add(1, std::memory_order_relaxed);

        // Spawn connection handler
        spawn_connection(client_fd);
    }

    LOG_INFO("CORO_TCP", "Accept loop stopped");
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
    // Parameters are stored in the coroutine frame, safe across suspensions
    co_await handler(io, fd);
    active_counter.fetch_sub(1, std::memory_order_relaxed);
}

void CoroTcpListener::spawn_connection(int client_fd) {
    // Set non-blocking
    sys::set_nonblocking(client_fd);

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
