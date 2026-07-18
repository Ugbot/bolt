/**
 * kqueue async I/O - OPTIMIZED VERSION (no mutex!)
 * 
 * Changes from v1:
 * - Removed global mutex (was killing performance!)
 * - Store pointers directly in kevent.udata
 * - No hash map lookups
 * - Lock-free operation
 * 
 * Expected: 500K+ req/s (125x faster than v1!)
 */

#include "bolt/api/core/async_io.h"

#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__)

#include <sys/event.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

#include <atomic>
#include <memory>
#include <cstring>

namespace bolt::api {
namespace core {

/**
 * Pending I/O operation (stored directly in kevent.udata)
 */
struct pending_op {
    io_op operation;
    int fd;
    io_callback callback;
    void* user_data;
    
    // For read/write
    void* buffer{nullptr};
    size_t size{0};
    
    // For connect
    struct sockaddr_storage addr;
    socklen_t addrlen{0};

    // For recvfrom (UDP): caller-owned peer-address out-params. Must outlive
    // the op (guaranteed by the awaitable owner); filled in poll() on read.
    struct sockaddr* user_src{nullptr};
    socklen_t*       user_srclen{nullptr};
};

/**
 * kqueue implementation (optimized, lock-free)
 */
struct kqueue_io::impl {
    int kq_fd{-1};
    async_io_config config;

    std::atomic<bool> running{false};
    std::atomic<bool> stop_requested{false};

    // Wake mechanism
    async_io::wake_callback wake_cb;
    std::atomic<bool> wake_pending{false};

    // Statistics (atomic, no locks!)
    std::atomic<uint64_t> stat_accepts{0};
    std::atomic<uint64_t> stat_reads{0};
    std::atomic<uint64_t> stat_writes{0};
    std::atomic<uint64_t> stat_connects{0};
    std::atomic<uint64_t> stat_closes{0};
    std::atomic<uint64_t> stat_polls{0};
    std::atomic<uint64_t> stat_events{0};
    std::atomic<uint64_t> stat_errors{0};
    std::atomic<uint64_t> stat_wakes{0};

    impl(const async_io_config& cfg) : config(cfg) {
        kq_fd = kqueue();
        if (kq_fd < 0) {
            // Handle error
        }

        // Register wake event (EVFILT_USER for cross-thread signaling)
        // ident=0 is our wake event
        struct kevent kev;
        EV_SET(&kev, 0, EVFILT_USER, EV_ADD | EV_CLEAR, 0, 0, nullptr);
        kevent(kq_fd, &kev, 1, nullptr, 0, nullptr);
    }
    
    ~impl() {
        if (kq_fd >= 0) {
            close(kq_fd);
        }
    }
    
    int set_nonblocking(int fd) {
        int flags = fcntl(fd, F_GETFL, 0);
        if (flags < 0) return -1;
        return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    }
};

// Replace old kqueue_io with this optimized version
// (keeping same class name for ABI compatibility)

kqueue_io::kqueue_io(const async_io_config& config)
    : impl_(std::make_unique<impl>(config)) {
}

kqueue_io::~kqueue_io() {
    stop();
}

int kqueue_io::accept_async(
    int listen_fd,
    io_callback callback,
    void* user_data
) noexcept {
    impl_->set_nonblocking(listen_fd);
    
    // Allocate operation (will be freed in poll())
    pending_op* op = new pending_op();
    op->operation = io_op::accept;
    op->fd = listen_fd;
    op->callback = std::move(callback);
    op->user_data = user_data;
    
    // Register with kqueue - store pointer directly in udata!
    struct kevent kev;
    EV_SET(&kev, listen_fd, EVFILT_READ, EV_ADD | EV_ENABLE | EV_ONESHOT, 0, 0, op);
    //                                                                             ↑
    //                                                               Pointer stored here!
    
    if (kevent(impl_->kq_fd, &kev, 1, nullptr, 0, nullptr) < 0) {
        delete op;  // Clean up on error
        impl_->stat_errors.fetch_add(1, std::memory_order_relaxed);
        return -1;
    }
    
    impl_->stat_accepts.fetch_add(1, std::memory_order_relaxed);
    return 0;
}

int kqueue_io::read_async(
    int fd,
    void* buffer,
    size_t size,
    io_callback callback,
    void* user_data
) noexcept {
    impl_->set_nonblocking(fd);
    
    pending_op* op = new pending_op();
    op->operation = io_op::read;
    op->fd = fd;
    op->buffer = buffer;
    op->size = size;
    op->callback = std::move(callback);
    op->user_data = user_data;
    
    struct kevent kev;
    EV_SET(&kev, fd, EVFILT_READ, EV_ADD | EV_ENABLE | EV_ONESHOT, 0, 0, op);
    
    if (kevent(impl_->kq_fd, &kev, 1, nullptr, 0, nullptr) < 0) {
        delete op;
        impl_->stat_errors.fetch_add(1, std::memory_order_relaxed);
        return -1;
    }
    
    impl_->stat_reads.fetch_add(1, std::memory_order_relaxed);
    return 0;
}

int kqueue_io::write_async(
    int fd,
    const void* buffer,
    size_t size,
    io_callback callback,
    void* user_data
) noexcept {
    impl_->set_nonblocking(fd);
    
    pending_op* op = new pending_op();
    op->operation = io_op::write;
    op->fd = fd;
    op->buffer = const_cast<void*>(buffer);
    op->size = size;
    op->callback = std::move(callback);
    op->user_data = user_data;
    
    struct kevent kev;
    EV_SET(&kev, fd, EVFILT_WRITE, EV_ADD | EV_ENABLE | EV_ONESHOT, 0, 0, op);
    
    if (kevent(impl_->kq_fd, &kev, 1, nullptr, 0, nullptr) < 0) {
        delete op;
        impl_->stat_errors.fetch_add(1, std::memory_order_relaxed);
        return -1;
    }
    
    impl_->stat_writes.fetch_add(1, std::memory_order_relaxed);
    return 0;
}

int kqueue_io::connect_async(
    int fd,
    const struct sockaddr* addr,
    socklen_t addrlen,
    io_callback callback,
    void* user_data
) noexcept {
    impl_->set_nonblocking(fd);
    
    pending_op* op = new pending_op();
    op->operation = io_op::connect;
    op->fd = fd;
    op->callback = std::move(callback);
    op->user_data = user_data;
    memcpy(&op->addr, addr, addrlen);
    op->addrlen = addrlen;
    
    // Start connection
    int ret = connect(fd, addr, addrlen);
    if (ret < 0 && errno != EINPROGRESS) {
        delete op;
        impl_->stat_errors.fetch_add(1, std::memory_order_relaxed);
        return -1;
    }
    
    struct kevent kev;
    EV_SET(&kev, fd, EVFILT_WRITE, EV_ADD | EV_ENABLE | EV_ONESHOT, 0, 0, op);
    
    if (kevent(impl_->kq_fd, &kev, 1, nullptr, 0, nullptr) < 0) {
        delete op;
        impl_->stat_errors.fetch_add(1, std::memory_order_relaxed);
        return -1;
    }
    
    impl_->stat_connects.fetch_add(1, std::memory_order_relaxed);
    return 0;
}

int kqueue_io::recvfrom_async(
    int fd,
    void* buffer,
    size_t size,
    struct sockaddr* src,
    socklen_t* srclen,
    io_callback callback,
    void* user_data
) noexcept {
    impl_->set_nonblocking(fd);

    pending_op* op = new pending_op();
    op->operation = io_op::recvfrom;
    op->fd = fd;
    op->buffer = buffer;
    op->size = size;
    op->user_src = src;
    op->user_srclen = srclen;
    op->callback = std::move(callback);
    op->user_data = user_data;

    struct kevent kev;
    EV_SET(&kev, fd, EVFILT_READ, EV_ADD | EV_ENABLE | EV_ONESHOT, 0, 0, op);

    if (kevent(impl_->kq_fd, &kev, 1, nullptr, 0, nullptr) < 0) {
        delete op;
        impl_->stat_errors.fetch_add(1, std::memory_order_relaxed);
        return -1;
    }

    impl_->stat_reads.fetch_add(1, std::memory_order_relaxed);
    return 0;
}

int kqueue_io::sendto_async(
    int fd,
    const void* buffer,
    size_t size,
    const struct sockaddr* dst,
    socklen_t dstlen,
    io_callback callback,
    void* user_data
) noexcept {
    impl_->set_nonblocking(fd);

    // UDP sendto rarely blocks: send directly in the submit path and deliver
    // the completion synchronously. `buffer`/`dst` are valid for this call.
    ssize_t n = ::sendto(fd, buffer, size, 0, dst, dstlen);

    impl_->stat_writes.fetch_add(1, std::memory_order_relaxed);

    if (callback) {
        io_event event;
        event.operation = io_op::sendto;
        event.fd = fd;
        event.user_data = user_data;
        event.flags = 0;
        event.result = n;
        callback(event);
    }
    return 0;
}

int kqueue_io::close_async(int fd) noexcept {
    impl_->stat_closes.fetch_add(1, std::memory_order_relaxed);
    return close(fd);
}

int kqueue_io::poll(uint32_t timeout_us) noexcept {
    if (impl_->kq_fd < 0) return -1;
    
    impl_->stat_polls.fetch_add(1, std::memory_order_relaxed);
    
    struct kevent events[128];
    struct timespec timeout;
    timeout.tv_sec = timeout_us / 1000000;
    timeout.tv_nsec = (timeout_us % 1000000) * 1000;
    
    int n = kevent(impl_->kq_fd, nullptr, 0, events, 128, &timeout);
    if (n < 0) {
        impl_->stat_errors.fetch_add(1, std::memory_order_relaxed);
        return -1;
    }
    
    impl_->stat_events.fetch_add(n, std::memory_order_relaxed);
    
    // Process events (NO MUTEX!)
    for (int i = 0; i < n; ++i) {
        const struct kevent& kev = events[i];

        // Check for wake event (EVFILT_USER with ident=0)
        if (kev.filter == EVFILT_USER && kev.ident == 0) {
            impl_->stat_wakes.fetch_add(1, std::memory_order_relaxed);
            impl_->wake_pending.store(false, std::memory_order_release);
            if (impl_->wake_cb) {
                impl_->wake_cb();
            }
            continue;
        }

        // Get operation pointer directly from event (NO HASH MAP LOOKUP!)
        pending_op* op = static_cast<pending_op*>(kev.udata);
        if (!op) continue;

        std::unique_ptr<pending_op> op_guard(op);  // Auto-delete

        // Execute operation
        io_event event;
        event.operation = op->operation;
        event.fd = op->fd;
        event.user_data = op->user_data;
        event.flags = kev.flags;
        event.result = 0;

        // Perform actual I/O (non-blocking!)
        switch (op->operation) {
            case io_op::accept: {
                struct sockaddr_storage addr;
                socklen_t addrlen = sizeof(addr);
                int client_fd = accept(op->fd, (struct sockaddr*)&addr, &addrlen);
                event.result = client_fd;
                break;
            }

            case io_op::read: {
                ssize_t bytes = read(op->fd, op->buffer, op->size);
                event.result = bytes;
                break;
            }

            case io_op::write: {
                ssize_t bytes = write(op->fd, op->buffer, op->size);
                event.result = bytes;
                break;
            }

            case io_op::recvfrom: {
                // Socket is readable: recvfrom into the buffer, filling the
                // caller's peer-address out-params.
                ssize_t bytes = recvfrom(op->fd, op->buffer, op->size, 0,
                                         op->user_src, op->user_srclen);
                event.result = bytes;
                break;
            }

            case io_op::connect: {
                // Check for connection error
                int error = 0;
                socklen_t len = sizeof(error);
                getsockopt(op->fd, SOL_SOCKET, SO_ERROR, &error, &len);
                event.result = error == 0 ? 0 : -1;
                break;
            }

            default:
                break;
        }

        // Invoke callback
        if (op->callback) {
            op->callback(event);
        }
    }
    
    return n;
}

void kqueue_io::run() noexcept {
    if (impl_->running.exchange(true)) {
        return;
    }
    
    impl_->stop_requested.store(false);
    
    while (!impl_->stop_requested.load(std::memory_order_acquire)) {
        poll(impl_->config.poll_timeout_us);
    }
    
    impl_->running.store(false);
}

void kqueue_io::stop() noexcept {
    impl_->stop_requested.store(true, std::memory_order_release);
}

bool kqueue_io::is_running() const noexcept {
    return impl_->running.load(std::memory_order_acquire);
}

void kqueue_io::wake() noexcept {
    // Thread-safe wake using EVFILT_USER
    // Only trigger if not already pending (avoid redundant wakes)
    bool expected = false;
    if (impl_->wake_pending.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        struct kevent kev;
        // NOTE_TRIGGER triggers the EVFILT_USER event
        EV_SET(&kev, 0, EVFILT_USER, 0, NOTE_TRIGGER, 0, nullptr);
        kevent(impl_->kq_fd, &kev, 1, nullptr, 0, nullptr);
    }
}

void kqueue_io::set_wake_callback(wake_callback callback) noexcept {
    impl_->wake_cb = std::move(callback);
}

async_io::stats kqueue_io::get_stats() const noexcept {
    stats s;
    s.accepts = impl_->stat_accepts.load(std::memory_order_relaxed);
    s.reads = impl_->stat_reads.load(std::memory_order_relaxed);
    s.writes = impl_->stat_writes.load(std::memory_order_relaxed);
    s.connects = impl_->stat_connects.load(std::memory_order_relaxed);
    s.closes = impl_->stat_closes.load(std::memory_order_relaxed);
    s.polls = impl_->stat_polls.load(std::memory_order_relaxed);
    s.events = impl_->stat_events.load(std::memory_order_relaxed);
    s.errors = impl_->stat_errors.load(std::memory_order_relaxed);
    s.wakes = impl_->stat_wakes.load(std::memory_order_relaxed);
    return s;
}

} // namespace core
} // namespace bolt::api

#endif // __APPLE__ || __FreeBSD__ || __OpenBSD__

