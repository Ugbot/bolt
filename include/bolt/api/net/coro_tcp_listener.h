// Coroutine-based TCP Listener
//
// Implements the Seastar-inspired architecture:
// - 1-2 event loops (IODispatcher) handle I/O dispatch
// - N worker threads (WorkerThreadPool) execute coroutines
// - Coroutines yield on I/O, workers pick up other work
//
// This is the new architecture that replaces the old N-threads-N-event-loops model.

#pragma once

#include "bolt/api/core/coro_task.h"
#include "bolt/api/core/worker_pool.h"
#include "bolt/api/net/io_dispatcher.h"
#include "bolt/api/net/tcp_socket.h"
#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace bolt::api {
namespace net {

// =============================================================================
// Coroutine-based Connection Handler
// =============================================================================

/// Connection handler coroutine type
/// Takes the accepted fd and IODispatcher, returns a task that handles the connection
using CoroConnectionHandler = std::function<core::coro_task<void>(IODispatcher&, int fd)>;

// =============================================================================
// Configuration
// =============================================================================

struct CoroTcpListenerConfig {
    std::string host = "0.0.0.0";     // Bind address
    uint16_t port = 8080;             // Bind port
    int backlog = 1024;               // Listen backlog
    size_t num_io_threads = 1;        // Number of I/O dispatch threads (1-2 recommended)
    size_t num_workers = 0;           // Worker threads (0 = auto)

    // Configuration for underlying components
    core::WorkerPoolConfig worker_config;
    core::async_io_config io_config;
};

// =============================================================================
// Coroutine TCP Listener
// =============================================================================

/// High-performance TCP listener using coroutine-on-threadpool architecture
class CoroTcpListener {
public:
    /// Create listener with configuration and connection handler
    CoroTcpListener(const CoroTcpListenerConfig& config, CoroConnectionHandler handler);

    /// Create with configuration, handler, and optional shared IODispatcher
    /// If shared_io_dispatcher is provided, the listener will use it instead of creating its own
    CoroTcpListener(const CoroTcpListenerConfig& config, CoroConnectionHandler handler,
                    IODispatcher* shared_io_dispatcher, core::WorkerThreadPool* shared_worker_pool);

    /// Create with simple parameters
    CoroTcpListener(uint16_t port, CoroConnectionHandler handler);

    ~CoroTcpListener();

    // Non-copyable, non-movable
    CoroTcpListener(const CoroTcpListener&) = delete;
    CoroTcpListener& operator=(const CoroTcpListener&) = delete;

    /// Start listening and accepting connections
    /// This starts the I/O dispatcher and worker pool, then blocks until stop()
    int start();

    /// Start in background (non-blocking)
    /// Returns immediately, listener runs in background threads
    int start_background();

    /// Stop the listener
    /// Thread-safe. Can be called from any thread.
    void stop();

    /// Check if listener is running
    bool is_running() const noexcept {
        return running_.load(std::memory_order_relaxed);
    }

    /// Get the IODispatcher (works for both owned and shared)
    IODispatcher* dispatcher() noexcept {
        return owns_resources_ ? io_dispatcher_.get() : shared_io_dispatcher_;
    }

    /// Get the WorkerThreadPool (works for both owned and shared)
    core::WorkerThreadPool* worker_pool() noexcept {
        return owns_resources_ ? worker_pool_.get() : shared_worker_pool_;
    }

    /// Get configuration
    const CoroTcpListenerConfig& config() const noexcept { return config_; }

    // =========================================================================
    // Statistics
    // =========================================================================

    struct Stats {
        uint64_t connections_accepted;
        uint64_t connections_active;
        uint64_t connections_total;
    };

    Stats get_stats() const noexcept;

private:
    // Accept loop coroutine — detached_task self-destroys on completion, so
    // the frame is reclaimed when the loop exits (listener stop or fatal accept
    // error) — no leaked frame for the lifetime of the process.
    core::detached_task accept_loop(int listen_fd);

    // Create and configure the listen socket
    int create_listen_socket();

    // Spawn connection handler on worker pool
    void spawn_connection(int client_fd);

    CoroTcpListenerConfig config_;
    CoroConnectionHandler handler_;

    std::unique_ptr<core::WorkerThreadPool> worker_pool_;
    std::unique_ptr<IODispatcher> io_dispatcher_;

    // Pointers to shared resources (if using shared mode)
    IODispatcher* shared_io_dispatcher_ = nullptr;
    core::WorkerThreadPool* shared_worker_pool_ = nullptr;
    bool owns_resources_ = true;  // Track if we own the resources

    // Listen sockets. Per-thread-engine backends (epoll/etc): ONE SO_REUSEPORT
    // socket per IO thread, each accepted on by that thread's own accept loop
    // (the kernel load-balances connections across them). Shared backend (IOCP):
    // a single socket with one accept loop. Closed in stop().
    std::vector<int> listen_fds_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};

    // Stats
    alignas(core::kCacheLineSize) std::atomic<uint64_t> connections_accepted_{0};
    alignas(core::kCacheLineSize) std::atomic<uint64_t> connections_active_{0};
    alignas(core::kCacheLineSize) std::atomic<uint64_t> connections_total_{0};
};

// =============================================================================
// Example Connection Handlers
// =============================================================================

/// Simple echo handler (for testing)
inline core::coro_task<void> echo_handler(IODispatcher& io, int fd) {
    char buffer[4096];

    while (true) {
        ssize_t n = co_await io.async_read(fd, buffer, sizeof(buffer));
        if (n <= 0) {
            break;  // Connection closed or error
        }

        ssize_t written = co_await io.async_write(fd, buffer, n);
        if (written < 0) {
            break;  // Write error
        }
    }

    io.async_close(fd);
}

/// HTTP/1.1 style request-response handler skeleton
inline core::coro_task<void> http1_skeleton_handler(IODispatcher& io, int fd) {
    char buffer[8192];

    // Read request
    ssize_t n = co_await io.async_read(fd, buffer, sizeof(buffer) - 1);
    if (n <= 0) {
        io.async_close(fd);
        co_return;
    }
    buffer[n] = '\0';

    // Simple response
    const char* response =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/plain\r\n"
        "Content-Length: 13\r\n"
        "Connection: close\r\n"
        "\r\n"
        "Hello, World!";

    co_await io.async_write(fd, response, strlen(response));
    io.async_close(fd);
}

} // namespace net
} // namespace bolt::api
