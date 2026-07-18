/**
 * Bolt API Native Event Loop - kqueue Implementation (macOS/BSD)
 *
 * Direct kqueue syscalls for maximum performance.
 * Features:
 * - Edge-triggered mode (EV_CLEAR)
 * - Zero-copy event delivery
 * - Support for 10K+ concurrent connections
 */

#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)

#include "bolt/api/net/event_loop.h"
#include "bolt/api/core/logger.h"
#include <sys/event.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <netinet/tcp.h>
#include <errno.h>
#include <cstring>
#include <unordered_map>
#include <atomic>
#include <iostream>

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
 * kqueue-based event loop implementation
 */
class KqueueEventLoop : public EventLoop {
public:
    KqueueEventLoop()
        : kq_fd_(-1)
        , running_(false)
    {
        // Create kqueue file descriptor
        kq_fd_ = kqueue();
        if (kq_fd_ < 0) {
            std::cerr << "kqueue() failed: " << strerror(errno) << std::endl;
            std::abort();  // Fatal error, cannot continue without event loop
        }

        // Set kqueue fd to close-on-exec
        fcntl(kq_fd_, F_SETFD, FD_CLOEXEC);

        // Pre-allocate event array for poll()
        events_.resize(256);  // Start with 256, will grow if needed
    }

    ~KqueueEventLoop() override {
        if (kq_fd_ >= 0) {
            close(kq_fd_);
        }
    }

    int add_fd(int fd, IOEvent events, EventHandler handler, void* user_data) override {
        if (fd < 0 || !handler) {
            errno = EINVAL;
            LOG_DEBUG("KQUEUE", "add_fd INVALID fd=%d handler=%s", fd, handler ? "valid" : "null");
            return -1;
        }

        LOG_DEBUG("KQUEUE", "add_fd fd=%d events=%d kq_fd=%d", fd, static_cast<int>(events), kq_fd_);

        // Store handler
        EventHandlerData data;
        data.handler = std::move(handler);
        data.user_data = user_data;
        data.events = events;
        handlers_[fd] = std::move(data);

        // Register with kqueue
        int result = update_kqueue_events(fd, events, false);
        LOG_DEBUG("KQUEUE", "add_fd result=%d errno=%d", result, errno);
        return result;
    }

    int modify_fd(int fd, IOEvent events) override {
        auto it = handlers_.find(fd);
        if (it == handlers_.end()) {
            errno = ENOENT;
            return -1;
        }

        // Update stored events
        it->second.events = events;

        // Re-register with kqueue
        return update_kqueue_events(fd, events, true);
    }

    int remove_fd(int fd) override {
        auto it = handlers_.find(fd);
        if (it == handlers_.end()) {
            errno = ENOENT;
            return -1;
        }

        // Remove from kqueue
        struct kevent changes[2];
        int n_changes = 0;

        // Remove READ filter if present
        if (it->second.events & IOEvent::READ) {
            EV_SET(&changes[n_changes++], fd, EVFILT_READ, EV_DELETE, 0, 0, nullptr);
        }

        // Remove WRITE filter if present
        if (it->second.events & IOEvent::WRITE) {
            EV_SET(&changes[n_changes++], fd, EVFILT_WRITE, EV_DELETE, 0, 0, nullptr);
        }

        if (n_changes > 0) {
            if (kevent(kq_fd_, changes, n_changes, nullptr, 0, nullptr) < 0) {
                // Ignore ENOENT (filter was already removed)
                if (errno != ENOENT) {
                    return -1;
                }
            }
        }

        // Remove handler
        handlers_.erase(it);
        return 0;
    }

    int poll(int timeout_ms) override {
        if (kq_fd_ < 0) {
            errno = EBADF;
            return -1;
        }

        // Convert timeout to timespec
        struct timespec* timeout_ptr = nullptr;
        struct timespec timeout_ts;
        if (timeout_ms >= 0) {
            timeout_ts.tv_sec = timeout_ms / 1000;
            timeout_ts.tv_nsec = (timeout_ms % 1000) * 1000000;
            timeout_ptr = &timeout_ts;
        }

        // Wait for events
        int n_events = kevent(kq_fd_, nullptr, 0, events_.data(), events_.size(), timeout_ptr);

        if (n_events < 0) {
            if (errno == EINTR) {
                return 0;  // Interrupted, not an error
            }
            LOG_DEBUG("KQUEUE", "poll error errno=%d", errno);
            return -1;
        }

        if (n_events > 0) {
            LOG_DEBUG("KQUEUE", "poll got %d events", n_events);
        }

        // Dispatch events
        for (int i = 0; i < n_events; i++) {
            struct kevent& ev = events_[i];
            int fd = static_cast<int>(ev.ident);

            LOG_DEBUG("KQUEUE", "event fd=%d filter=%d flags=%d", fd, ev.filter, ev.flags);

            // Look up handler
            auto it = handlers_.find(fd);
            if (it == handlers_.end()) {
                LOG_DEBUG("KQUEUE", "no handler for fd=%d", fd);
                continue;  // Handler was removed
            }

            // Determine event type
            IOEvent event_type = static_cast<IOEvent>(0);

            if (ev.filter == EVFILT_READ) {
                event_type = IOEvent::READ;

                // Check for EOF/HUP
                if (ev.flags & EV_EOF) {
                    event_type = event_type | IOEvent::HUP;
                }
            } else if (ev.filter == EVFILT_WRITE) {
                event_type = IOEvent::WRITE;
            }

            // Check for errors
            if (ev.flags & EV_ERROR) {
                event_type = event_type | IOEvent::ERROR;
            }

            LOG_DEBUG("KQUEUE", "dispatching to handler fd=%d event_type=%d", fd, static_cast<int>(event_type));

            // Invoke handler
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
                LOG_ERROR("KQUEUE", "poll() error: %s", strerror(errno));
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
        return "kqueue";
    }

private:
    int update_kqueue_events(int fd, IOEvent events, bool modify) {
        // Determine flags for ADD operations
        uint16_t flags = EV_ADD;
        if (events & IOEvent::EDGE) {
            flags |= EV_CLEAR;  // Edge-triggered mode
        }

        // First, apply ADD operations (these should always succeed)
        struct kevent add_changes[2];
        int n_add = 0;

        if (events & IOEvent::READ) {
            EV_SET(&add_changes[n_add++], fd, EVFILT_READ, flags, 0, 0, nullptr);
        }
        if (events & IOEvent::WRITE) {
            EV_SET(&add_changes[n_add++], fd, EVFILT_WRITE, flags, 0, 0, nullptr);
        }

        if (n_add > 0) {
            if (kevent(kq_fd_, add_changes, n_add, nullptr, 0, nullptr) < 0) {
                return -1;
            }
        }

        // Then, apply DELETE operations for filters not requested
        // We do these separately and ignore ENOENT errors (filter not found)
        if (modify) {
            struct kevent del_change;

            if (!(events & IOEvent::READ)) {
                EV_SET(&del_change, fd, EVFILT_READ, EV_DELETE, 0, 0, nullptr);
                if (kevent(kq_fd_, &del_change, 1, nullptr, 0, nullptr) < 0) {
                    // Ignore ENOENT - filter was never added
                    if (errno != ENOENT) {
                        return -1;
                    }
                }
            }

            if (!(events & IOEvent::WRITE)) {
                EV_SET(&del_change, fd, EVFILT_WRITE, EV_DELETE, 0, 0, nullptr);
                if (kevent(kq_fd_, &del_change, 1, nullptr, 0, nullptr) < 0) {
                    // Ignore ENOENT - filter was never added
                    if (errno != ENOENT) {
                        return -1;
                    }
                }
            }
        }

        return 0;
    }

    int kq_fd_;                                      // kqueue file descriptor
    std::unordered_map<int, EventHandlerData> handlers_;  // fd → handler mapping
    std::vector<struct kevent> events_;              // Event array for kevent()
    std::atomic<bool> running_;                      // Running flag
};

// Factory function for kqueue
std::unique_ptr<EventLoop> create_kqueue_event_loop() {
    return std::make_unique<KqueueEventLoop>();
}

} // namespace net
} // namespace bolt::api

#endif // __APPLE__ || __FreeBSD__ || __OpenBSD__ || __NetBSD__
