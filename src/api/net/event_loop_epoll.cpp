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
#include "event_handler_registry.h"
#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <cstdlib>   // std::abort (no exceptions in this build)

namespace bolt::api {
namespace net {

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

        // Store handler (insert-or-overwrite; slot addresses are stable)
        EventHandlerData* data = handlers_.insert(fd);
        if (!data) {
            errno = ENOMEM;  // registry hard cap — honest fail, never silent
            return -1;
        }
        data->handler = std::move(handler);
        data->user_data = user_data;
        data->events = events;

        // Register with epoll
        return update_epoll_events(fd, events, false);
    }

    int modify_fd(int fd, IOEvent events) override {
        EventHandlerData* data = handlers_.find(fd);
        if (!data) {
            errno = ENOENT;
            return -1;
        }

        // Update stored events
        data->events = events;

        // Re-register with epoll
        return update_epoll_events(fd, events, true);
    }

    int remove_fd(int fd) override {
        if (!handlers_.find(fd)) {
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
        const bool removed = handlers_.remove(fd);
        assert(removed);
        (void)removed;
        return 0;
    }

    int poll(int timeout_ms) override {
        if (epoll_fd_ < 0) {
            errno = EBADF;
            return -1;
        }
        if (poll_active_) {
            errno = EBUSY;
            return -1;
        }
        poll_active_ = true;

        // Wait for events
        int n_events = epoll_wait(
            epoll_fd_, events_.data(),
            static_cast<int>(kEventBatchCapacity), timeout_ms);

        if (n_events < 0) {
            poll_active_ = false;
            if (errno == EINTR) {
                return 0;  // Interrupted, not an error
            }
            return -1;
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
    void snapshot_event_tokens(int n_events) noexcept {
        assert(n_events >= 0);
        assert(n_events <= static_cast<int>(events_.size()));
        for (int i = 0; i < n_events; ++i) {
            const int fd = events_[static_cast<size_t>(i)].data.fd;
            EventHandlerData* const data = handlers_.find(fd);
            event_tokens_[static_cast<size_t>(i)] =
                data != nullptr ? data->token : 0;
        }
    }

    HandlerDispatchResult dispatch_event(uint32_t index) noexcept {
        assert(index < events_.size());
        const struct epoll_event& ev = events_[index];
        const int fd = ev.data.fd;
        assert(fd >= 0);
        const uint64_t token = event_tokens_[index];
        if (token == 0) return HandlerDispatchResult::missing;

        IOEvent event_type = static_cast<IOEvent>(0);
        if (ev.events & EPOLLIN) event_type = IOEvent::READ;
        if (ev.events & EPOLLOUT) event_type = event_type | IOEvent::WRITE;
        if (ev.events & (EPOLLHUP | EPOLLRDHUP)) {
            event_type = event_type | IOEvent::HUP;
        }
        if (ev.events & EPOLLERR) event_type = event_type | IOEvent::ERROR;
        return handlers_.dispatch(fd, token, event_type);
    }

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

    static constexpr size_t kEventBatchCapacity = 256;

    int epoll_fd_;
    EventHandlerRegistry handlers_;
    // A fixed batch lets us snapshot every token before callbacks mutate the
    // registry. The kernel retains excess readiness for a later poll.
    std::array<struct epoll_event, kEventBatchCapacity> events_{};
    std::array<uint64_t, kEventBatchCapacity> event_tokens_{};
    std::atomic<bool> running_;
    bool poll_active_{false};
};

// Factory function for epoll
std::unique_ptr<EventLoop> create_epoll_event_loop() {
    return std::make_unique<EpollEventLoop>();
}

} // namespace net
} // namespace bolt::api

#endif // __linux__
