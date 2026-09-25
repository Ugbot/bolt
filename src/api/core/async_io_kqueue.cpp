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
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

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

    // Stable slot in the fixed op pool; the op is never freed or relocated.
    std::uint32_t pool_index{0};

    void reset() noexcept {
        operation = io_op::read;
        fd = -1;
        callback = nullptr;
        user_data = nullptr;
        buffer = nullptr;
        size = 0;
        addrlen = 0;
        user_src = nullptr;
        user_srclen = nullptr;
    }
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

    // Fixed op pool + tagged Treiber free list (mirrors epoll_io). kevent.udata
    // carries pool_index+1. kqueue arms EVFILT_READ and EVFILT_WRITE on one fd
    // independently, so each fd has a read slot and a write slot; close_async
    // claims them to complete in-flight ops.
    static constexpr std::uint32_t kOpPoolSize = 8192;
    static constexpr int           kMaxFds     = 1 << 16;
    std::vector<pending_op>                 op_pool{std::vector<pending_op>(kOpPoolSize)};
    std::vector<std::atomic<std::uint32_t>> free_next{std::vector<std::atomic<std::uint32_t>>(kOpPoolSize)};
    std::atomic<std::uint64_t>              free_head{0};  // (tag<<32)|(index+1); 0=empty
    std::vector<std::atomic<std::uint32_t>> fd_slot{std::vector<std::atomic<std::uint32_t>>(2 * kMaxFds)};

    void init_op_pool() noexcept {
        for (std::uint32_t i = 0; i < kOpPoolSize; ++i) {
            op_pool[i].pool_index = i;
            free_next[i].store((i + 1 < kOpPoolSize) ? (i + 2) : 0, std::memory_order_relaxed);
        }
        free_head.store(1, std::memory_order_relaxed);
        for (auto& slot : fd_slot) slot.store(0, std::memory_order_relaxed);
        assert(op_pool.size() == kOpPoolSize);
        assert(fd_slot.size() == 2u * kMaxFds);
    }

    std::uint32_t alloc_index() noexcept {
        std::uint64_t head = free_head.load(std::memory_order_acquire);
        for (;;) {
            const auto idx_p1 = static_cast<std::uint32_t>(head);
            if (idx_p1 == 0) return UINT32_MAX;  // exhausted: submit fails
            assert(idx_p1 <= kOpPoolSize);
            const std::uint64_t nxt = (((head >> 32) + 1) << 32) |
                free_next[idx_p1 - 1].load(std::memory_order_relaxed);
            if (free_head.compare_exchange_weak(head, nxt, std::memory_order_acq_rel,
                                                std::memory_order_acquire)) {
                return idx_p1 - 1;
            }
        }
    }

    void free_index(std::uint32_t idx) noexcept {
        assert(idx < kOpPoolSize);
        op_pool[idx].callback = nullptr;  // release captured state now
        std::uint64_t head = free_head.load(std::memory_order_acquire);
        for (;;) {
            free_next[idx].store(static_cast<std::uint32_t>(head), std::memory_order_relaxed);
            const std::uint64_t nxt = (((head >> 32) + 1) << 32) | (idx + 1);
            if (free_head.compare_exchange_weak(head, nxt, std::memory_order_acq_rel,
                                                std::memory_order_acquire)) return;
        }
    }

    pending_op* alloc_op() noexcept {
        const std::uint32_t idx = alloc_index();
        if (idx == UINT32_MAX) {
            stat_errors.fetch_add(1, std::memory_order_relaxed);
            return nullptr;
        }
        pending_op& op = op_pool[idx];
        op.reset();
        assert(op.pool_index == idx);
        return &op;
    }

    std::atomic<std::uint32_t>* slot_for(int fd, int16_t filter) noexcept {
        if (fd < 0 || fd >= kMaxFds) return nullptr;  // beyond table: untracked
        const std::size_t base = static_cast<std::size_t>(fd) * 2;
        return &fd_slot[base + (filter == EVFILT_WRITE ? 1 : 0)];
    }

    // Publish the slot, then arm (a completion on another thread claims via the
    // slot). Re-arming a filter replaces the knote's udata, so the displaced op
    // can never complete and is returned to the pool.
    int register_op(pending_op* op, int16_t filter) noexcept {
        assert(op != nullptr);
        assert(filter == EVFILT_READ || filter == EVFILT_WRITE);
        const std::uint32_t idx_p1 = op->pool_index + 1;
        std::atomic<std::uint32_t>* slot = slot_for(op->fd, filter);
        if (slot) {
            const std::uint32_t displaced = slot->exchange(idx_p1, std::memory_order_acq_rel);
            if (displaced != 0) free_index(displaced - 1);
        }
        struct kevent kev;
        EV_SET(&kev, op->fd, filter, EV_ADD | EV_ENABLE | EV_ONESHOT, 0, 0,
               reinterpret_cast<void*>(static_cast<std::uintptr_t>(idx_p1)));
        if (kevent(kq_fd, &kev, 1, nullptr, 0, nullptr) < 0) {
            std::uint32_t expected = idx_p1;
            const bool owned = slot == nullptr ||
                slot->compare_exchange_strong(expected, 0, std::memory_order_acq_rel);
            if (owned) free_index(op->pool_index);
            stat_errors.fetch_add(1, std::memory_order_relaxed);
            return -1;
        }
        return 0;
    }

    // Complete a claimed in-flight op with -1 (closed / cancelled).
    void cancel_slot(int fd, int16_t filter) noexcept {
        std::atomic<std::uint32_t>* slot = slot_for(fd, filter);
        if (slot == nullptr) return;
        const std::uint32_t idx_p1 = slot->exchange(0, std::memory_order_acq_rel);
        if (idx_p1 == 0) return;
        assert(idx_p1 <= kOpPoolSize);
        pending_op& op = op_pool[idx_p1 - 1];
        assert(op.fd == fd);
        if (op.callback) {
            io_event event;
            event.operation = op.operation;
            event.fd        = op.fd;
            event.user_data = op.user_data;
            event.flags     = 0;
            event.result    = -1;
            op.callback(event);
        }
        free_index(idx_p1 - 1);
    }

    impl(const async_io_config& cfg) : config(cfg) {
        init_op_pool();
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
    
    pending_op* op = impl_->alloc_op();
    if (op == nullptr) return -1;
    op->operation = io_op::accept;
    op->fd = listen_fd;
    op->callback = std::move(callback);
    op->user_data = user_data;

    impl_->stat_accepts.fetch_add(1, std::memory_order_relaxed);
    return impl_->register_op(op, EVFILT_READ);
}

int kqueue_io::read_async(
    int fd,
    void* buffer,
    size_t size,
    io_callback callback,
    void* user_data
) noexcept {
    impl_->set_nonblocking(fd);
    
    pending_op* op = impl_->alloc_op();
    if (op == nullptr) return -1;
    op->operation = io_op::read;
    op->fd = fd;
    op->buffer = buffer;
    op->size = size;
    op->callback = std::move(callback);
    op->user_data = user_data;

    impl_->stat_reads.fetch_add(1, std::memory_order_relaxed);
    return impl_->register_op(op, EVFILT_READ);
}

int kqueue_io::write_async(
    int fd,
    const void* buffer,
    size_t size,
    io_callback callback,
    void* user_data
) noexcept {
    impl_->set_nonblocking(fd);
    
    pending_op* op = impl_->alloc_op();
    if (op == nullptr) return -1;
    op->operation = io_op::write;
    op->fd = fd;
    op->buffer = const_cast<void*>(buffer);
    op->size = size;
    op->callback = std::move(callback);
    op->user_data = user_data;

    impl_->stat_writes.fetch_add(1, std::memory_order_relaxed);
    return impl_->register_op(op, EVFILT_WRITE);
}

int kqueue_io::connect_async(
    int fd,
    const struct sockaddr* addr,
    socklen_t addrlen,
    io_callback callback,
    void* user_data
) noexcept {
    impl_->set_nonblocking(fd);
    
    pending_op* op = impl_->alloc_op();
    if (op == nullptr) return -1;
    op->operation = io_op::connect;
    op->fd = fd;
    op->callback = std::move(callback);
    op->user_data = user_data;
    memcpy(&op->addr, addr, addrlen);
    op->addrlen = addrlen;

    int ret = connect(fd, addr, addrlen);
    if (ret < 0 && errno != EINPROGRESS) {
        impl_->free_index(op->pool_index);
        impl_->stat_errors.fetch_add(1, std::memory_order_relaxed);
        return -1;
    }

    impl_->stat_connects.fetch_add(1, std::memory_order_relaxed);
    return impl_->register_op(op, EVFILT_WRITE);
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

    pending_op* op = impl_->alloc_op();
    if (op == nullptr) return -1;
    op->operation = io_op::recvfrom;
    op->fd = fd;
    op->buffer = buffer;
    op->size = size;
    op->user_src = src;
    op->user_srclen = srclen;
    op->callback = std::move(callback);
    op->user_data = user_data;

    impl_->stat_reads.fetch_add(1, std::memory_order_relaxed);
    return impl_->register_op(op, EVFILT_READ);
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
    // Complete in-flight ops with -1 before close() silently drops their
    // knotes (epoll_io/IOCP parity: UdpTransport::stop() waits on them).
    impl_->cancel_slot(fd, EVFILT_READ);
    impl_->cancel_slot(fd, EVFILT_WRITE);
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

        const auto idx_p1 = static_cast<std::uint32_t>(
            reinterpret_cast<std::uintptr_t>(kev.udata));
        if (idx_p1 == 0 || idx_p1 > impl::kOpPoolSize) continue;
        // Claim: close_async or a re-arm may have taken this op already.
        const int kev_fd = static_cast<int>(kev.ident);
        if (auto* slot = impl_->slot_for(kev_fd, kev.filter)) {
            std::uint32_t expected = idx_p1;
            if (!slot->compare_exchange_strong(expected, 0, std::memory_order_acq_rel)) {
                continue;
            }
        }
        pending_op* op = &impl_->op_pool[idx_p1 - 1];
        assert(op->fd == kev_fd);

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

        if (op->callback) {
            op->callback(event);
        }
        impl_->free_index(op->pool_index);
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

