/**
 * Bolt API Native Event Loop - Common Implementation
 *
 * Factory functions and platform-agnostic helpers.
 *
 * Platform support:
 * - macOS/BSD: kqueue
 * - Linux: epoll
 * - Windows: IOCP (I/O Completion Ports)
 */

#include "bolt/api/net/event_loop.h"
#include <thread>
#include <cstring>

#if defined(_WIN32)
    #define HAVE_IOCP 1
    #include <winsock2.h>
    #include <ws2tcpip.h>
#else
    #include <fcntl.h>
    #include <netinet/in.h>
    #include <netinet/tcp.h>
    #include <sys/socket.h>
    #include <errno.h>
    #if defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
        #define HAVE_KQUEUE 1
    #elif defined(__linux__)
        #define HAVE_EPOLL 1
    #endif
#endif

namespace bolt::api {
namespace net {

// Forward declarations for platform-specific implementations
#ifdef HAVE_KQUEUE
std::unique_ptr<EventLoop> create_kqueue_event_loop();
#endif

#ifdef HAVE_EPOLL
std::unique_ptr<EventLoop> create_epoll_event_loop();
#endif

#ifdef HAVE_IOCP
std::unique_ptr<EventLoop> create_iocp_event_loop();
#endif

/**
 * Factory function to create platform-specific event loop
 */
std::unique_ptr<EventLoop> create_event_loop() {
#ifdef HAVE_KQUEUE
    return create_kqueue_event_loop();
#elif defined(HAVE_EPOLL)
    return create_epoll_event_loop();
#elif defined(HAVE_IOCP)
    return create_iocp_event_loop();
#else
    #error "Unsupported platform: no kqueue, epoll, or IOCP available"
#endif
}

/**
 * Get recommended number of worker threads
 */
uint32_t recommended_worker_count() {
    unsigned int hw_threads = std::thread::hardware_concurrency();

    // Leave 2 cores for OS and other tasks
    if (hw_threads <= 2) {
        return 1;  // Minimum 1 worker thread
    }

    return hw_threads - 2;
}

/**
 * Helper: Set socket to non-blocking mode
 */
int EventLoop::set_nonblocking(int fd) {
#if defined(_WIN32)
    u_long mode = 1;
    if (ioctlsocket(static_cast<SOCKET>(fd), FIONBIO, &mode) != 0) {
        return -1;
    }
    return 0;
#else
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) {
        return -1;
    }

    if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
        return -1;
    }

    return 0;
#endif
}

/**
 * Helper: Disable Nagle's algorithm (enable TCP_NODELAY)
 */
int EventLoop::set_tcp_nodelay(int fd) {
#if defined(_WIN32)
    BOOL enable = TRUE;
    if (setsockopt(static_cast<SOCKET>(fd), IPPROTO_TCP, TCP_NODELAY,
                   reinterpret_cast<const char*>(&enable), sizeof(enable)) == SOCKET_ERROR) {
        return -1;
    }
    return 0;
#else
    int enable = 1;
    if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &enable, sizeof(enable)) < 0) {
        return -1;
    }
    return 0;
#endif
}

/**
 * Helper: Enable SO_REUSEADDR
 */
int EventLoop::set_reuseaddr(int fd) {
#if defined(_WIN32)
    BOOL enable = TRUE;
    if (setsockopt(static_cast<SOCKET>(fd), SOL_SOCKET, SO_REUSEADDR,
                   reinterpret_cast<const char*>(&enable), sizeof(enable)) == SOCKET_ERROR) {
        return -1;
    }
    return 0;
#else
    int enable = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable)) < 0) {
        return -1;
    }
    return 0;
#endif
}

/**
 * Helper: Enable SO_REUSEPORT (Linux-specific, not available on Windows)
 */
int EventLoop::set_reuseport(int fd) {
#if defined(_WIN32)
    // Windows does not support SO_REUSEPORT
    // Use SO_EXCLUSIVEADDRUSE for similar behavior if needed
    (void)fd;
    return -1;
#elif defined(SO_REUSEPORT)
    int enable = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &enable, sizeof(enable)) < 0) {
        return -1;
    }
    return 0;
#else
    // Not supported on this platform
    (void)fd;
    errno = ENOTSUP;
    return -1;
#endif
}

} // namespace net
} // namespace bolt::api
