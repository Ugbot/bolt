/**
 * Bolt API Native Event Loop - Abstract Interface
 *
 * Platform-agnostic event loop interface for high-performance I/O multiplexing.
 * Implementations:
 * - macOS/BSD: kqueue (EV_CLEAR for edge-triggered)
 * - Linux: epoll (EPOLLET for edge-triggered) or io_uring
 * - Windows: IOCP (future)
 *
 * Design principles:
 * - Zero-copy where possible
 * - Edge-triggered for maximum throughput
 * - Non-blocking I/O only
 * - Direct syscalls, no wrappers
 * - Lock-free in hot path
 */

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include "bolt/api/net/sys_compat.h"

namespace bolt::api {
namespace net {

/**
 * I/O event types
 */
enum class IOEvent {
    READ = 1 << 0,      // Socket readable
    WRITE = 1 << 1,     // Socket writable
    ERROR = 1 << 2,     // Socket error
    HUP = 1 << 3,       // Connection closed
    EDGE = 1 << 4       // Edge-triggered mode
};

inline IOEvent operator|(IOEvent a, IOEvent b) {
    return static_cast<IOEvent>(static_cast<int>(a) | static_cast<int>(b));
}

inline bool operator&(IOEvent a, IOEvent b) {
    return (static_cast<int>(a) & static_cast<int>(b)) != 0;
}

/**
 * Event handler callback
 *
 * Args:
 * - fd: File descriptor that triggered event
 * - events: IOEvent flags (READ, WRITE, ERROR, HUP)
 * - user_data: User-provided pointer
 */
using EventHandler = std::function<void(int fd, IOEvent events, void* user_data)>;

/**
 * Abstract event loop interface
 *
 * All platform-specific implementations must inherit from this.
 */
class EventLoop {
public:
    virtual ~EventLoop() = default;

    /**
     * Add file descriptor to event loop
     *
     * @param fd File descriptor to monitor
     * @param events IOEvent flags (READ, WRITE, EDGE)
     * @param handler Callback when event occurs
     * @param user_data User pointer passed to handler
     * @return 0 on success, -1 on error (check errno)
     */
    virtual int add_fd(int fd, IOEvent events, EventHandler handler, void* user_data = nullptr) = 0;

    /**
     * Modify events for existing file descriptor
     *
     * @param fd File descriptor
     * @param events New IOEvent flags
     * @return 0 on success, -1 on error
     */
    virtual int modify_fd(int fd, IOEvent events) = 0;

    /**
     * Remove file descriptor from event loop
     *
     * @param fd File descriptor to remove
     * @return 0 on success, -1 on error
     */
    virtual int remove_fd(int fd) = 0;

    /**
     * Run one iteration of the event loop
     *
     * Waits for events (up to timeout_ms) and dispatches handlers.
     *
     * @param timeout_ms Timeout in milliseconds (-1 = infinite, 0 = non-blocking)
     * @return Number of events processed, or -1 on error
     */
    virtual int poll(int timeout_ms = -1) = 0;

    /**
     * Run the event loop continuously
     *
     * Loops until stop() is called.
     */
    virtual void run() = 0;

    /**
     * Stop the event loop
     *
     * Thread-safe. Can be called from any thread.
     */
    virtual void stop() = 0;

    /**
     * Check if event loop is running
     */
    virtual bool is_running() const = 0;

    /**
     * Get platform name
     *
     * @return "kqueue", "epoll", "io_uring", "iocp", etc.
     */
    virtual const char* platform_name() const = 0;

    // Helper to set socket non-blocking
    static int set_nonblocking(int fd);

    // Helper to set TCP_NODELAY (disable Nagle's algorithm)
    static int set_tcp_nodelay(int fd);

    // Helper to set SO_REUSEADDR
    static int set_reuseaddr(int fd);

    // Helper to set SO_REUSEPORT (Linux only)
    static int set_reuseport(int fd);
};

/**
 * Factory function to create platform-specific event loop
 *
 * Automatically selects best available implementation:
 * - macOS/BSD: kqueue
 * - Linux: io_uring (if available), else epoll
 * - Windows: IOCP (future)
 *
 * @return Unique pointer to event loop, or nullptr on error
 */
std::unique_ptr<EventLoop> create_event_loop();

/**
 * Get recommended number of worker threads
 *
 * Returns hardware_concurrency - 2 (leave cores for OS and other tasks)
 */
uint32_t recommended_worker_count();

} // namespace net
} // namespace bolt::api
