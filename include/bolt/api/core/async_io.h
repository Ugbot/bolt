/**
 * Async I/O - Unified interface for kqueue, epoll, io_uring, and IOCP
 * 
 * Provides a high-performance, platform-agnostic async I/O layer.
 * 
 * Backends:
 * - macOS/BSD: kqueue
 * - Linux:     epoll or io_uring (configurable)
 * - Windows:   IOCP
 */

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <atomic>

// Platform-specific includes
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>   // socklen_t
#include <BaseTsd.h>
#ifndef BOLTAPI_HAVE_SSIZE_T
#define BOLTAPI_HAVE_SSIZE_T 1
using ssize_t = SSIZE_T;
#endif
#else
#include <sys/socket.h>
#endif

namespace bolt::api {
namespace core {

// Forward declarations
class async_io;
class async_socket;

/**
 * I/O operation types
 */
enum class io_op : uint8_t {
    accept,    // Accept new connection
    read,      // Read from socket
    write,     // Write to socket
    connect,   // Connect to remote
    close,     // Close socket
    timer,     // Timer expiration
    recvfrom,  // Datagram receive (UDP) — fills caller's sockaddr out-params
    sendto     // Datagram send (UDP)
};

/**
 * I/O event - passed to callbacks
 */
struct io_event {
    io_op operation;
    int fd;
    void* user_data;
    ssize_t result;      // Bytes transferred or error code
    uint32_t flags;
};

/**
 * I/O callback function type
 */
using io_callback = std::function<void(const io_event&)>;

/**
 * Async I/O backend type
 */
enum class io_backend : uint8_t {
    auto_detect,  // Choose best for platform
    kqueue,       // macOS/BSD kqueue
    epoll,        // Linux epoll
    io_uring,     // Linux io_uring (kernel 5.1+)
    iocp          // Windows IOCP
};

/**
 * Async I/O configuration
 */
struct async_io_config {
    io_backend backend = io_backend::auto_detect;
    uint32_t max_events = 1024;        // Max events per poll
    uint32_t queue_depth = 4096;       // Queue depth (io_uring)
    bool zero_copy = true;             // Enable zero-copy where possible
    bool poll_busy = false;            // Busy-poll mode (low latency)
    uint32_t poll_timeout_us = 1000;   // Poll timeout (microseconds)
};

/**
 * Async I/O engine - platform-agnostic interface
 */
class async_io {
public:
    /**
     * Create async I/O engine
     */
    static std::unique_ptr<async_io> create(const async_io_config& config = {});
    
    virtual ~async_io() = default;
    
    /**
     * Get backend type
     */
    virtual io_backend backend() const noexcept = 0;
    
    /**
     * Get backend name
     */
    virtual const char* backend_name() const noexcept = 0;
    
    /**
     * Submit async accept operation
     * 
     * @param listen_fd Listening socket FD
     * @param callback Callback when connection accepted
     * @param user_data User data passed to callback
     * @return 0 on success, error code otherwise
     */
    virtual int accept_async(
        int listen_fd,
        io_callback callback,
        void* user_data = nullptr
    ) noexcept = 0;
    
    /**
     * Submit async read operation
     * 
     * @param fd Socket FD
     * @param buffer Buffer to read into
     * @param size Buffer size
     * @param callback Callback when read completes
     * @param user_data User data
     * @return 0 on success
     */
    virtual int read_async(
        int fd,
        void* buffer,
        size_t size,
        io_callback callback,
        void* user_data = nullptr
    ) noexcept = 0;
    
    /**
     * Submit async write operation
     * 
     * @param fd Socket FD
     * @param buffer Buffer to write from
     * @param size Buffer size
     * @param callback Callback when write completes
     * @param user_data User data
     * @return 0 on success
     */
    virtual int write_async(
        int fd,
        const void* buffer,
        size_t size,
        io_callback callback,
        void* user_data = nullptr
    ) noexcept = 0;
    
    /**
     * Submit async connect operation
     * 
     * @param fd Socket FD
     * @param addr Address to connect to
     * @param addrlen Address length
     * @param callback Callback when connected
     * @param user_data User data
     * @return 0 on success
     */
    virtual int connect_async(
        int fd,
        const struct sockaddr* addr,
        socklen_t addrlen,
        io_callback callback,
        void* user_data = nullptr
    ) noexcept = 0;
    
    /**
     * Submit async datagram receive (UDP recvfrom).
     *
     * Waits for the socket to become readable, then performs a single
     * recvfrom into `buffer`, filling the caller-owned `src`/`srclen`
     * out-params with the peer address. The callback's io_event.result is
     * the byte count (>=0) or -1 on error.
     *
     * IMPORTANT (lifetime): `buffer`, `src` and `srclen` MUST outlive the
     * operation (until the callback fires). The awaitable/op owner is
     * responsible for this — the backend does NOT copy them.
     *
     * @param fd      UDP socket FD
     * @param buffer  Buffer to receive into
     * @param size    Buffer capacity
     * @param src     [out] peer address (filled on completion); may be null
     * @param srclen  [in/out] capacity of src on entry, actual len on exit
     * @param callback Completion callback
     * @param user_data User data
     * @return 0 on success (op submitted), -1 on immediate error
     */
    virtual int recvfrom_async(
        int fd,
        void* buffer,
        size_t size,
        struct sockaddr* src,
        socklen_t* srclen,
        io_callback callback,
        void* user_data = nullptr
    ) noexcept = 0;

    /**
     * Submit async datagram send (UDP sendto).
     *
     * UDP sendto rarely blocks, so backends may perform the sendto directly
     * in the submit path and still deliver a completion callback. The
     * callback's io_event.result is the byte count (>=0) or -1 on error.
     *
     * `buffer` and `dst` need only be valid for the duration of the call when
     * the backend sends synchronously; backends that defer MUST keep them
     * alive until completion (the awaitable owner guarantees this).
     *
     * @param fd      UDP socket FD
     * @param buffer  Buffer to send from
     * @param size    Bytes to send
     * @param dst     Destination address
     * @param dstlen  Destination address length
     * @param callback Completion callback
     * @param user_data User data
     * @return 0 on success, -1 on immediate error
     */
    virtual int sendto_async(
        int fd,
        const void* buffer,
        size_t size,
        const struct sockaddr* dst,
        socklen_t dstlen,
        io_callback callback,
        void* user_data = nullptr
    ) noexcept = 0;

    /**
     * Close socket asynchronously
     *
     * @param fd Socket FD
     * @return 0 on success
     */
    virtual int close_async(int fd) noexcept = 0;
    
    /**
     * Poll for I/O events
     * 
     * Processes pending I/O operations and invokes callbacks.
     * 
     * @param timeout_us Timeout in microseconds (0 = non-blocking)
     * @return Number of events processed, or -1 on error
     */
    virtual int poll(uint32_t timeout_us = 0) noexcept = 0;
    
    /**
     * Run event loop until stopped
     * 
     * Continuously polls for events and processes them.
     */
    virtual void run() noexcept = 0;
    
    /**
     * Stop event loop
     */
    virtual void stop() noexcept = 0;
    
    /**
     * Check if running
     */
    virtual bool is_running() const noexcept = 0;

    /**
     * Wake up the event loop from another thread
     *
     * Thread-safe. Wakes the event loop if it's blocked in poll().
     * Use this to signal that work is ready from another thread.
     */
    virtual void wake() noexcept = 0;

    /**
     * Set wake callback - called when wake() is triggered
     *
     * The callback is invoked from the event loop thread when
     * wake() is called from another thread.
     *
     * @param callback Function to call on wake event
     */
    using wake_callback = std::function<void()>;
    virtual void set_wake_callback(wake_callback callback) noexcept = 0;

    /**
     * Get statistics
     */
    struct stats {
        uint64_t accepts{0};
        uint64_t reads{0};
        uint64_t writes{0};
        uint64_t connects{0};
        uint64_t closes{0};
        uint64_t polls{0};
        uint64_t events{0};
        uint64_t errors{0};
        uint64_t wakes{0};
    };

    virtual stats get_stats() const noexcept = 0;

protected:
    async_io() = default;
};

/**
 * Backend implementations
 */

#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__)
/**
 * kqueue-based async I/O (macOS/BSD)
 * 
 * High-performance async I/O using kqueue.
 * Supports level and edge-triggered events.
 */
class kqueue_io : public async_io {
public:
    explicit kqueue_io(const async_io_config& config);
    ~kqueue_io() override;

    io_backend backend() const noexcept override { return io_backend::kqueue; }
    const char* backend_name() const noexcept override { return "kqueue"; }

    int accept_async(int listen_fd, io_callback callback, void* user_data) noexcept override;
    int read_async(int fd, void* buffer, size_t size, io_callback callback, void* user_data) noexcept override;
    int write_async(int fd, const void* buffer, size_t size, io_callback callback, void* user_data) noexcept override;
    int connect_async(int fd, const struct sockaddr* addr, socklen_t addrlen, io_callback callback, void* user_data) noexcept override;
    int recvfrom_async(int fd, void* buffer, size_t size, struct sockaddr* src, socklen_t* srclen, io_callback callback, void* user_data) noexcept override;
    int sendto_async(int fd, const void* buffer, size_t size, const struct sockaddr* dst, socklen_t dstlen, io_callback callback, void* user_data) noexcept override;
    int close_async(int fd) noexcept override;

    int poll(uint32_t timeout_us) noexcept override;
    void run() noexcept override;
    void stop() noexcept override;
    bool is_running() const noexcept override;

    void wake() noexcept override;
    void set_wake_callback(wake_callback callback) noexcept override;

    stats get_stats() const noexcept override;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};
#endif

#ifdef __linux__
/**
 * epoll-based async I/O (Linux)
 * 
 * High-performance async I/O using epoll.
 */
class epoll_io : public async_io {
public:
    explicit epoll_io(const async_io_config& config);
    ~epoll_io() override;

    io_backend backend() const noexcept override { return io_backend::epoll; }
    const char* backend_name() const noexcept override { return "epoll"; }

    int accept_async(int listen_fd, io_callback callback, void* user_data) noexcept override;
    int read_async(int fd, void* buffer, size_t size, io_callback callback, void* user_data) noexcept override;
    int write_async(int fd, const void* buffer, size_t size, io_callback callback, void* user_data) noexcept override;
    int connect_async(int fd, const struct sockaddr* addr, socklen_t addrlen, io_callback callback, void* user_data) noexcept override;
    int recvfrom_async(int fd, void* buffer, size_t size, struct sockaddr* src, socklen_t* srclen, io_callback callback, void* user_data) noexcept override;
    int sendto_async(int fd, const void* buffer, size_t size, const struct sockaddr* dst, socklen_t dstlen, io_callback callback, void* user_data) noexcept override;
    int close_async(int fd) noexcept override;

    int poll(uint32_t timeout_us) noexcept override;
    void run() noexcept override;
    void stop() noexcept override;
    bool is_running() const noexcept override;

    void wake() noexcept override;
    void set_wake_callback(wake_callback callback) noexcept override;

    stats get_stats() const noexcept override;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

/**
 * io_uring-based async I/O (Linux 5.1+)
 * 
 * Next-generation async I/O using io_uring.
 * Zero-copy, true async operations.
 */
class io_uring_io : public async_io {
public:
    explicit io_uring_io(const async_io_config& config);
    ~io_uring_io() override;

    io_backend backend() const noexcept override { return io_backend::io_uring; }
    const char* backend_name() const noexcept override { return "io_uring"; }

    int accept_async(int listen_fd, io_callback callback, void* user_data) noexcept override;
    int read_async(int fd, void* buffer, size_t size, io_callback callback, void* user_data) noexcept override;
    int write_async(int fd, const void* buffer, size_t size, io_callback callback, void* user_data) noexcept override;
    int connect_async(int fd, const struct sockaddr* addr, socklen_t addrlen, io_callback callback, void* user_data) noexcept override;
    int recvfrom_async(int fd, void* buffer, size_t size, struct sockaddr* src, socklen_t* srclen, io_callback callback, void* user_data) noexcept override;
    int sendto_async(int fd, const void* buffer, size_t size, const struct sockaddr* dst, socklen_t dstlen, io_callback callback, void* user_data) noexcept override;
    int close_async(int fd) noexcept override;

    int poll(uint32_t timeout_us) noexcept override;
    void run() noexcept override;
    void stop() noexcept override;
    bool is_running() const noexcept override;

    void wake() noexcept override;
    void set_wake_callback(wake_callback callback) noexcept override;

    stats get_stats() const noexcept override;

    /// Probe whether io_uring is usable on this kernel/sandbox: true iff
    /// io_uring_setup succeeded AND the required features (IORING_FEAT_EXT_ARG,
    /// kernel ≥5.11) are present. The factory checks this and falls back to
    /// epoll when false (Docker default seccomp blocks the io_uring syscalls →
    /// ENOSYS/EPERM; old kernels also miss EXT_ARG). NEVER aborts.
    bool usable() const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};
#endif

#ifdef _WIN32
/**
 * IOCP-based async I/O (Windows)
 * 
 * High-performance async I/O using I/O Completion Ports.
 */
class iocp_io : public async_io {
public:
    explicit iocp_io(const async_io_config& config);
    ~iocp_io() override;

    io_backend backend() const noexcept override { return io_backend::iocp; }
    const char* backend_name() const noexcept override { return "IOCP"; }

    int accept_async(int listen_fd, io_callback callback, void* user_data) noexcept override;
    int read_async(int fd, void* buffer, size_t size, io_callback callback, void* user_data) noexcept override;
    int write_async(int fd, const void* buffer, size_t size, io_callback callback, void* user_data) noexcept override;
    int connect_async(int fd, const struct sockaddr* addr, socklen_t addrlen, io_callback callback, void* user_data) noexcept override;
    int recvfrom_async(int fd, void* buffer, size_t size, struct sockaddr* src, socklen_t* srclen, io_callback callback, void* user_data) noexcept override;
    int sendto_async(int fd, const void* buffer, size_t size, const struct sockaddr* dst, socklen_t dstlen, io_callback callback, void* user_data) noexcept override;
    int close_async(int fd) noexcept override;

    int poll(uint32_t timeout_us) noexcept override;
    void run() noexcept override;
    void stop() noexcept override;
    bool is_running() const noexcept override;

    void wake() noexcept override;
    void set_wake_callback(wake_callback callback) noexcept override;

    stats get_stats() const noexcept override;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};
#endif

} // namespace core
} // namespace bolt::api

