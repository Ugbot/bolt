/**
 * Bolt API Native Event Loop - epoll Implementation (Linux)
 *
 * Direct epoll syscalls for maximum performance.
 * Features:
 * - Edge-triggered mode (EPOLLET)
 * - Zero-copy event delivery
 * - Support for 10K+ concurrent connections
 */

#if defined(__linux__)

#include "bolt/api/net/event_loop.h"
#include <sys/epoll.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <netinet/tcp.h>
#include <errno.h>
#include <cstring>
#include <unordered_map>
#include <atomic>
#include <iostream>
#include <cstdlib>   // std::abort (no exceptions in this build)

namespace bolt::api {
namespace net {

/**
 * Event handler storage
 */
struct EventHandlerData {
    EventHandler handler;
    void* user_data;
    IOEvent events;  // Registered events
};

/**
 * epoll-based event loop implementation
 */
class EpollEventLoop : public EventLoop {
public:
    EpollEventLoop()
        : epoll_fd_(-1)
        , running_(false)
    {
        // Create epoll file descriptor
        epoll_fd_ = epoll_create1(EPOLL_CLOEXEC);
        if (epoll_fd_ < 0) {
            // -fno-exceptions build: a failed epoll fd at startup is fatal and
            // unrecoverable from a constructor — log and abort rather than throw.
            std::cerr << "epoll_create1() failed: " << strerror(errno) << std::endl;
            std::abort();
        }

        // Pre-allocate event array for epoll_wait()
        events_.resize(256);  // Start with 256, will grow if needed
    }

    ~EpollEventLoop() override {
        if (epoll_fd_ >= 0) {
            close(epoll_fd_);
        }
    }

    int add_fd(int fd, IOEvent events, EventHandler handler, void* user_data) override {
        if (fd < 0 || !handler) {
            errno = EINVAL;
            return -1;
        }

        // Store handler
        EventHandlerData data;
        data.handler = std::move(handler);
        data.user_data = user_data;
        data.events = events;
        handlers_[fd] = std::move(data);

        // Register with epoll
        return update_epoll_events(fd, events, false);
    }

    int modify_fd(int fd, IOEvent events) override {
        auto it = handlers_.find(fd);
        if (it == handlers_.end()) {
            errno = ENOENT;
            return -1;
        }

        // Update stored events
        it->second.events = events;

        // Re-register with epoll
        return update_epoll_events(fd, events, true);
    }

    int remove_fd(int fd) override {
        auto it = handlers_.find(fd);
        if (it == handlers_.end()) {
            errno = ENOENT;
            return -1;
        }

        // Remove from epoll
        struct epoll_event ev;
        if (epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, fd, &ev) < 0) {
            // Ignore ENOENT (already removed) and EBADF (fd already closed)
            if (errno != ENOENT && errno != EBADF) {
                return -1;
            }
        }

        // Remove handler
        handlers_.erase(it);
        return 0;
    }

    int poll(int timeout_ms) override {
        if (epoll_fd_ < 0) {
            errno = EBADF;
            return -1;
        }

        // Wait for events
        int n_events = epoll_wait(epoll_fd_, events_.data(), events_.size(), timeout_ms);

        if (n_events < 0) {
            if (errno == EINTR) {
                return 0;  // Interrupted, not an error
            }
            return -1;
        }

        // Dispatch events
        for (int i = 0; i < n_events; i++) {
            struct epoll_event& ev = events_[i];
            int fd = ev.data.fd;

            // Look up handler
            auto it = handlers_.find(fd);
            if (it == handlers_.end()) {
                continue;  // Handler was removed
            }

            // Determine event type
            IOEvent event_type = static_cast<IOEvent>(0);

            if (ev.events & EPOLLIN) {
                event_type = IOEvent::READ;
            }
            if (ev.events & EPOLLOUT) {
                event_type = event_type | IOEvent::WRITE;
            }
            if (ev.events & (EPOLLHUP | EPOLLRDHUP)) {
                event_type = event_type | IOEvent::HUP;
            }
            if (ev.events & EPOLLERR) {
                event_type = event_type | IOEvent::ERROR;
            }

            // Invoke handler directly — handlers are noexcept in this
            // exception-free (-fno-exceptions) build.
            it->second.handler(fd, event_type, it->second.user_data);
        }

        // Grow event array if needed
        if (n_events == static_cast<int>(events_.size())) {
            events_.resize(events_.size() * 2);
        }

        return n_events;
    }

    void run() override {
        running_.store(true, std::memory_order_release);

        while (running_.load(std::memory_order_acquire)) {
            int result = poll(100);  // 100ms timeout for responsiveness
            if (result < 0 && errno != EINTR) {
                std::cerr << "poll() error: " << strerror(errno) << std::endl;
                break;
            }
        }
    }

    void stop() override {
        running_.store(false, std::memory_order_release);
    }

    bool is_running() const override {
        return running_.load(std::memory_order_acquire);
    }

    const char* platform_name() const override {
        return "epoll";
    }

private:
    int update_epoll_events(int fd, IOEvent events, bool modify) {
        struct epoll_event ev;
        ev.events = 0;
        ev.data.fd = fd;

        // Set event flags
        if (events & IOEvent::READ) {
            ev.events |= EPOLLIN;
        }
        if (events & IOEvent::WRITE) {
            ev.events |= EPOLLOUT;
        }
        if (events & IOEvent::EDGE) {
            ev.events |= EPOLLET;  // Edge-triggered mode
        }

        // Always enable EPOLLRDHUP to detect peer shutdown
        ev.events |= EPOLLRDHUP;

        // Add or modify
        int op = modify ? EPOLL_CTL_MOD : EPOLL_CTL_ADD;
        if (epoll_ctl(epoll_fd_, op, fd, &ev) < 0) {
            return -1;
        }

        return 0;
    }

    int epoll_fd_;                                       // epoll file descriptor
    std::unordered_map<int, EventHandlerData> handlers_; // fd → handler mapping
    std::vector<struct epoll_event> events_;             // Event array for epoll_wait()
    std::atomic<bool> running_;                          // Running flag
};

// Factory function for epoll
std::unique_ptr<EventLoop> create_epoll_event_loop() {
    return std::make_unique<EpollEventLoop>();
}

} // namespace net
} // namespace bolt::api

#endif // __linux__
