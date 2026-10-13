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
#include "event_handler_registry.h"
#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <iostream>

namespace bolt::api {
namespace net {

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

        // Store handler (insert-or-overwrite; slot addresses are stable)
        EventHandlerData* data = handlers_.insert(fd);
        if (!data) {
            errno = ENOMEM;  // registry hard cap — honest fail, never silent
            return -1;
        }
        data->handler = std::move(handler);
        data->user_data = user_data;
        data->events = events;

        // Register with kqueue
        int result = update_kqueue_events(fd, events, false);
        LOG_DEBUG("KQUEUE", "add_fd result=%d errno=%d", result, errno);
        return result;
    }

    int modify_fd(int fd, IOEvent events) override {
        EventHandlerData* data = handlers_.find(fd);
        if (!data) {
            errno = ENOENT;
            return -1;
        }

        // Update stored events
        data->events = events;

        // Re-register with kqueue
        return update_kqueue_events(fd, events, true);
    }

    int remove_fd(int fd) override {
        EventHandlerData* data = handlers_.find(fd);
        if (!data) {
            errno = ENOENT;
            return -1;
        }

        // Remove from kqueue
        struct kevent changes[2];
        int n_changes = 0;

        // Remove READ filter if present
        if (data->events & IOEvent::READ) {
            EV_SET(&changes[n_changes++], fd, EVFILT_READ, EV_DELETE, 0, 0, nullptr);
        }

        // Remove WRITE filter if present
        if (data->events & IOEvent::WRITE) {
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
        const bool removed = handlers_.remove(fd);
        assert(removed);
        (void)removed;
        return 0;
    }

    int poll(int timeout_ms) override {
        if (kq_fd_ < 0) {
            errno = EBADF;
            return -1;
        }
        if (poll_active_) {
            errno = EBUSY;
            return -1;
        }
        poll_active_ = true;

        // Convert timeout to timespec
        struct timespec* timeout_ptr = nullptr;
        struct timespec timeout_ts;
        if (timeout_ms >= 0) {
            timeout_ts.tv_sec = timeout_ms / 1000;
            timeout_ts.tv_nsec = (timeout_ms % 1000) * 1000000;
            timeout_ptr = &timeout_ts;
        }

        // Wait for events
        int n_events = kevent(
            kq_fd_, nullptr, 0, events_.data(),
            static_cast<int>(kEventBatchCapacity), timeout_ptr);

        if (n_events < 0) {
            poll_active_ = false;
            if (errno == EINTR) {
                return 0;  // Interrupted, not an error
            }
            LOG_DEBUG("KQUEUE", "poll error errno=%d", errno);
            return -1;
        }

        if (n_events > 0) {
            LOG_DEBUG("KQUEUE", "poll got %d events", n_events);
        }

        snapshot_event_tokens(n_events);

        HandlerDispatchResult dispatch_error = HandlerDispatchResult::dispatched;
        for (int i = 0; i < n_events; i++) {
            const HandlerDispatchResult result =
                dispatch_event(static_cast<uint32_t>(i));
            if (result == HandlerDispatchResult::missing ||
                result == HandlerDispatchResult::stale) {
                continue;
            }
            if (result != HandlerDispatchResult::dispatched) {
                dispatch_error = result;
                break;
            }
        }

        poll_active_ = false;
        if (dispatch_error == HandlerDispatchResult::busy) {
            errno = EBUSY;
            return -1;
        }
        if (dispatch_error != HandlerDispatchResult::dispatched) {
            errno = ENOBUFS;
            return -1;
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
    void snapshot_event_tokens(int n_events) noexcept {
        assert(n_events >= 0);
        assert(n_events <= static_cast<int>(events_.size()));
        for (int i = 0; i < n_events; ++i) {
            const int fd = static_cast<int>(events_[static_cast<size_t>(i)].ident);
            EventHandlerData* const data = handlers_.find(fd);
            event_tokens_[static_cast<size_t>(i)] =
                data != nullptr ? data->token : 0;
        }
    }

    HandlerDispatchResult dispatch_event(uint32_t index) noexcept {
        assert(index < events_.size());
        const struct kevent& ev = events_[index];
        const int fd = static_cast<int>(ev.ident);
        assert(fd >= 0);
        const uint64_t token = event_tokens_[index];
        if (token == 0) return HandlerDispatchResult::missing;

        LOG_DEBUG("KQUEUE", "event fd=%d filter=%d flags=%d", fd, ev.filter, ev.flags);
        IOEvent event_type = static_cast<IOEvent>(0);
        if (ev.filter == EVFILT_READ) {
            event_type = IOEvent::READ;
            if (ev.flags & EV_EOF) event_type = event_type | IOEvent::HUP;
        } else if (ev.filter == EVFILT_WRITE) {
            event_type = IOEvent::WRITE;
        }
        if (ev.flags & EV_ERROR) event_type = event_type | IOEvent::ERROR;
        LOG_DEBUG("KQUEUE", "dispatching fd=%d event_type=%d", fd,
                  static_cast<int>(event_type));
        return handlers_.dispatch(fd, token, event_type);
    }

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

    static constexpr size_t kEventBatchCapacity = 256;

    int kq_fd_;
    EventHandlerRegistry handlers_;
    // A fixed batch lets us snapshot every token before callbacks mutate the
    // registry. The kernel retains excess readiness for a later poll.
    std::array<struct kevent, kEventBatchCapacity> events_{};
    std::array<uint64_t, kEventBatchCapacity> event_tokens_{};
    std::atomic<bool> running_;
    bool poll_active_{false};
};

// Factory function for kqueue
std::unique_ptr<EventLoop> create_kqueue_event_loop() {
    return std::make_unique<KqueueEventLoop>();
}

} // namespace net
} // namespace bolt::api

#endif // __APPLE__ || __FreeBSD__ || __OpenBSD__ || __NetBSD__
