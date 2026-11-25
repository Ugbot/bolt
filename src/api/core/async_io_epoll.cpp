/**
 * epoll-based async I/O implementation (Linux)
 * 
 * High-performance async I/O using epoll.
 */

#include "bolt/api/core/async_io.h"

#ifdef __linux__

#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/eventfd.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <cstring>

#include "readiness_fast_path.h"

namespace bolt::api {
namespace core {

/**
 * Pending I/O operation
 */
struct pending_op {
    io_op operation{io_op::read};
    int fd{-1};
    io_callback callback;
    void* user_data{nullptr};
    AcceptRetirement* retirement{nullptr};

    // For read/write
    void* buffer{nullptr};
    size_t size{0};

    // For connect
    struct sockaddr_storage addr;
    socklen_t addrlen{0};

    // For recvfrom (UDP): caller-owned out-params for the peer address. They
    // must outlive the op (guaranteed by the awaitable owner); we fill them in
    // poll() when the socket is readable.
    struct sockaddr* user_src{nullptr};
    socklen_t*       user_srclen{nullptr};

    // Event flags
    uint32_t events{0};

    // Result of a try made on the submit path, delivered from the deferred ring.
    ssize_t result{0};

    // Slot index in the fixed op pool (stable; the op is NEVER freed/relocated,
    // so reading its fields in the close/complete race can't use-after-free).
    std::uint32_t pool_index{0};

    // Reset for reuse from the pool (drops the prior callback + captured state).
    void reset() noexcept {
        operation = io_op::read;
        fd = -1;
        callback = nullptr;
        user_data = nullptr;
        retirement = nullptr;
        buffer = nullptr;
        size = 0;
        addrlen = 0;
        user_src = nullptr;
        user_srclen = nullptr;
        events = 0;
        result = 0;
    }
};

/**
 * epoll implementation
 */
struct epoll_io::impl {
    int epoll_fd{-1};
    int wake_fd{-1};
    async_io_config config;

    std::atomic<bool> running{false};
    std::atomic<bool> stop_requested{false};

    // Wake mechanism
    async_io::wake_callback wake_cb;

    // --- Lock-free op management (no per-op alloc, no mutex) ----------------
    // Fixed pool of ops (stable storage). The hot path carries the op via
    // epoll_event.data.u64 = (pool_index+1)<<32 | fd, so completion needs no map.
    // A tagged free-list of indices hands ops out/back lock-free; an atomic
    // fd->index slot lets close_async claim+cancel an in-flight op without a map.
    static constexpr std::uint32_t kOpPoolSize = 8192;   // max concurrent ops
    static constexpr int           kMaxFds     = AcceptRetirement::kReadinessFdLimit;
    std::vector<pending_op>             op_pool{std::vector<pending_op>(kOpPoolSize)};
    std::vector<std::uint32_t>          free_next{std::vector<std::uint32_t>(kOpPoolSize)};
    std::atomic<std::uint64_t>          free_head{0};   // (tag<<32)|(index+1); 0=empty
    // Two slots per fd (read side, write side), so a read and a write can be in
    // flight on one fd at once.
    std::vector<std::atomic<std::uint32_t>> fd_slot{std::vector<std::atomic<std::uint32_t>>(2 * kMaxFds)};

    // Register-once / try-first path (G2CHK-1269), owner thread only. A
    // persistent registration carries kPersistTag | fd in data.u64.
    static constexpr std::uint64_t kPersistTag = std::uint64_t{1} << 63;
    detail::EngineOwner owner;
    std::vector<detail::FdFast> fast{std::vector<detail::FdFast>(kMaxFds)};
    detail::DeferredRing<kOpPoolSize> deferred;

    void init_op_pool() noexcept {
        // Chain: index 0 is head, each free_next points to the next index+1.
        for (std::uint32_t i = 0; i < kOpPoolSize; ++i) {
            op_pool[i].pool_index = i;
            free_next[i] = (i + 1 < kOpPoolSize) ? (i + 2) : 0;  // next index+1, 0=end
        }
        free_head.store(1, std::memory_order_relaxed);  // index 0 (idx+1=1)
        for (auto& s : fd_slot) s.store(0, std::memory_order_relaxed);
    }
    // Pop a free op index (tagged Treiber stack). Returns UINT32_MAX if exhausted.
    std::uint32_t alloc_index() noexcept {
        std::uint64_t head = free_head.load(std::memory_order_acquire);
        for (;;) {
            std::uint32_t idx_p1 = static_cast<std::uint32_t>(head);
            if (idx_p1 == 0) return UINT32_MAX;  // pool exhausted (overload)
            std::uint64_t nxt = (((head >> 32) + 1) << 32) | free_next[idx_p1 - 1];
            if (free_head.compare_exchange_weak(head, nxt, std::memory_order_acq_rel,
                                                std::memory_order_acquire)) {
                return idx_p1 - 1;
            }
        }
    }
    void free_index(std::uint32_t idx) noexcept {
        auto* retirement = op_pool[idx].retirement;
        op_pool[idx].retirement = nullptr;
        if (retirement) op_pool[idx].callback = nullptr;
        std::uint64_t head = free_head.load(std::memory_order_acquire);
        for (;;) {
            free_next[idx] = static_cast<std::uint32_t>(head);  // next = old head idx+1
            std::uint64_t nxt = (((head >> 32) + 1) << 32) | (idx + 1);
            if (free_head.compare_exchange_weak(head, nxt, std::memory_order_acq_rel,
                                                std::memory_order_acquire)) {
                if (retirement) retirement->operations.fetch_sub(1u, std::memory_order_release);
                return;
            }
        }
    }
    static int dir_of(uint32_t events) noexcept { return (events & EPOLLOUT) != 0 ? 1 : 0; }

    std::atomic<std::uint32_t>* slot_for(int fd, int dir) noexcept {
        assert(dir == 0 || dir == 1);
        if (fd < 0 || fd >= kMaxFds) return nullptr;  // beyond table: not tracked
        return &fd_slot[static_cast<std::size_t>(fd) * 2 + static_cast<std::size_t>(dir)];
    }

    // Statistics
    std::atomic<uint64_t> stat_accepts{0};
    std::atomic<uint64_t> stat_reads{0};
    std::atomic<uint64_t> stat_writes{0};
    std::atomic<uint64_t> stat_connects{0};
    std::atomic<uint64_t> stat_closes{0};
    std::atomic<uint64_t> stat_polls{0};
    std::atomic<uint64_t> stat_events{0};
    std::atomic<uint64_t> stat_errors{0};
    std::atomic<uint64_t> stat_wakes{0};
    std::atomic<uint64_t> stat_registrations{0};
    std::atomic<uint64_t> stat_inline{0};
    std::atomic<uint64_t> stat_deferred{0};

    impl(const async_io_config& cfg) : config(cfg) {
        init_op_pool();
        epoll_fd = epoll_create1(EPOLL_CLOEXEC);
        if (epoll_fd < 0) {
            // Handle error
        }

        // Create eventfd for wake mechanism
        wake_fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
        if (wake_fd >= 0) {
            // Register wake_fd with epoll. data.u64 = 0 is the wake sentinel
            // (op events carry (index+1)<<32 | fd, so their high word is never 0).
            struct epoll_event ev;
            ev.events = EPOLLIN | EPOLLET;  // Edge-triggered
            ev.data.u64 = 0;
            epoll_ctl(epoll_fd, EPOLL_CTL_ADD, wake_fd, &ev);
        }
    }
    
    ~impl() {
        if (wake_fd >= 0) {
            close(wake_fd);
        }
        if (epoll_fd >= 0) {
            close(epoll_fd);
        }
    }
    
    int set_nonblocking(int fd) {
        int flags = fcntl(fd, F_GETFL, 0);
        if (flags < 0) return -1;
        return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    }
    
    // Acquire a pooled op (no allocation after warmup). Returns nullptr on
    // pool exhaustion (treated as a submit failure / overload).
    pending_op* alloc_op() noexcept {
        std::uint32_t idx = alloc_index();
        if (idx == UINT32_MAX) return nullptr;
        pending_op& op = op_pool[idx];
        op.reset();
        op.pool_index = idx;
        return &op;
    }

    // Register a pooled op with epoll. The op is carried in data.u64 (no map),
    // and indexed in fd_slot for close_async cancellation. epoll_ctl is
    // thread-safe in the kernel, so no user-space lock is needed.
    int register_op(pending_op* op, uint32_t events) noexcept {
        op->events = events;
        struct epoll_event ev;
        ev.events = events | EPOLLET | EPOLLONESHOT;  // edge-triggered, one-shot
        ev.data.u64 = (static_cast<std::uint64_t>(op->pool_index + 1) << 32) |
                      static_cast<std::uint32_t>(op->fd);
        // Publish the fd->index slot BEFORE arming the fd in epoll. The completion
        // can fire on another IO thread the instant epoll_ctl arms it, and poll
        // claims the op via this slot — if it weren't set yet, poll would read a
        // stale slot, fail its claim check, and DROP the completion (the coroutine
        // would hang). Ordering: slot first, then arm.
        std::atomic<std::uint32_t>* s = slot_for(op->fd, dir_of(events));
        if (s) s->store(op->pool_index + 1, std::memory_order_release);
        // A one-shot registration replaces a persistent one on this fd.
        if (op->fd >= 0 && op->fd < kMaxFds) detail::fd_epoch_bump(op->fd);
        stat_registrations.fetch_add(1, std::memory_order_relaxed);
        if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, op->fd, &ev) < 0) {
            if (errno != EEXIST ||
                epoll_ctl(epoll_fd, EPOLL_CTL_MOD, op->fd, &ev) < 0) {
                stat_errors.fetch_add(1, std::memory_order_relaxed);
                if (s) s->store(0, std::memory_order_release);
                free_index(op->pool_index);
                return -1;
            }
        }
        return 0;
    }

    bool fast_ok(int fd) const noexcept {
        return fd >= 0 && fd < kMaxFds && owner.on_owner();
    }

    // One edge-triggered registration for both directions, kept until close.
    int ensure_registered(int fd, detail::FdFast& st) noexcept {
        if ((st.flags & detail::kFdRegRead) != 0) return 0;
        struct epoll_event ev;
        ev.events = EPOLLIN | EPOLLOUT | EPOLLRDHUP | EPOLLET;
        ev.data.u64 = kPersistTag | static_cast<std::uint32_t>(fd);
        stat_registrations.fetch_add(1, std::memory_order_relaxed);
        if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, fd, &ev) < 0 &&
            (errno != EEXIST || epoll_ctl(epoll_fd, EPOLL_CTL_MOD, fd, &ev) < 0)) {
            stat_errors.fetch_add(1, std::memory_order_relaxed);
            return -1;
        }
        st.flags |= static_cast<std::uint8_t>(detail::kFdRegRead | detail::kFdRegWrite);
        return 0;
    }

    int arm_fast(pending_op* op, int dir, detail::FdFast& st) noexcept {
        assert(op != nullptr);
        std::atomic<std::uint32_t>* slot = slot_for(op->fd, dir);
        assert(slot != nullptr);
        const std::uint32_t displaced = slot->exchange(op->pool_index + 1, std::memory_order_acq_rel);
        if (displaced != 0) free_index(displaced - 1);
        if (ensure_registered(op->fd, st) == 0) return 0;
        std::uint32_t mine = op->pool_index + 1;
        if (slot->compare_exchange_strong(mine, 0, std::memory_order_acq_rel)) free_index(op->pool_index);
        return -1;
    }

    // A try on the submit path finished the op: hand the result back inline, or
    // queue it for the next poll when the caller cannot resume inline.
    int finish_early(io_op operation, int fd, ssize_t result, io_callback& cb,
                     void* user_data, ssize_t* inline_result) noexcept {
        if (inline_result != nullptr) {
            *inline_result = result;
            stat_inline.fetch_add(1, std::memory_order_relaxed);
            return 1;
        }
        pending_op* op = alloc_op();
        if (op == nullptr) return -1;
        op->operation = operation;
        op->fd = fd;
        op->callback = std::move(cb);
        op->user_data = user_data;
        op->result = result;
        const bool pushed = deferred.push(op->pool_index);
        assert(pushed && "deferred ring is sized to the op pool");
        (void)pushed;
        stat_deferred.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }

    void run_deferred() noexcept {
        std::uint32_t budget = deferred.count;
        std::uint32_t index = 0;
        while (budget-- > 0 && deferred.pop(&index)) {
            pending_op& op = op_pool[index];
            const io_event event{op.operation, op.fd, op.user_data, op.result, 0u};
            if (op.callback) op.callback(event);
            free_index(index);
        }
    }

    // One-shot path: another thread's submit, a non-socket fd, or an fd past the table.
    int oneshot(io_op operation, int fd, void* buffer, size_t size, struct sockaddr* src,
                socklen_t* srclen, io_callback& callback, void* user_data, uint32_t events) noexcept {
        set_nonblocking(fd);
        pending_op* op = alloc_op();
        if (op == nullptr) return -1;
        op->operation = operation;
        op->fd = fd;
        op->buffer = buffer;
        op->size = size;
        op->user_src = src;
        op->user_srclen = srclen;
        op->callback = std::move(callback);
        op->user_data = user_data;
        return register_op(op, events);
    }

    void on_persistent_side(int fd, int dir, uint32_t events) noexcept;

    static void kernel_wake(void* ctx) noexcept;
};

epoll_io::epoll_io(const async_io_config& config)
    : impl_(std::make_unique<impl>(config)) {
    park_slot_.kernel_wake = &impl::kernel_wake;
    park_slot_.kernel_ctx = impl_.get();
}

epoll_io::~epoll_io() {
    stop();
}

int epoll_io::accept_async(
    int listen_fd,
    io_callback callback,
    void* user_data, AcceptRetirement* retirement
) noexcept {
    if (retirement && retirement->requested.load(std::memory_order_acquire)) { errno = ECANCELED; return -1; }
    if (retirement && impl_->slot_for(listen_fd, 0) == nullptr) { errno = EOVERFLOW; return -1; }
    impl_->set_nonblocking(listen_fd);
    
    pending_op* op = impl_->alloc_op();
    if (op == nullptr) { errno = ENOBUFS; return -1; }
    op->operation = io_op::accept;
    op->fd = listen_fd;
    op->callback = std::move(callback);
    op->user_data = user_data;
    op->retirement = retirement;
    if (retirement) retirement->operations.fetch_add(1u, std::memory_order_acq_rel);

    impl_->stat_accepts.fetch_add(1, std::memory_order_relaxed);
    return impl_->register_op(op, EPOLLIN);
}

int epoll_io::read_async(
    int fd,
    void* buffer,
    size_t size,
    io_callback callback,
    void* user_data,
    ssize_t* inline_result
) noexcept {
    impl_->stat_reads.fetch_add(1, std::memory_order_relaxed);
    if (!impl_->fast_ok(fd)) {
        return impl_->oneshot(io_op::read, fd, buffer, size, nullptr, nullptr, callback, user_data, EPOLLIN);
    }
    detail::FdFast& st = detail::fd_fast_current(impl_->fast.data(), fd);
    detail::ensure_nonblocking(fd, st);
    if ((st.flags & detail::kFdReadDrained) == 0 && (st.flags & detail::kFdNotSocket) == 0) {
        ssize_t n = 0;
        const detail::TryResult t = detail::try_recv(fd, buffer, size, &n);
        if (t == detail::TryResult::done) {
            if (n > 0 && static_cast<size_t>(n) < size) st.flags |= detail::kFdReadDrained;
            return impl_->finish_early(io_op::read, fd, n, callback, user_data, inline_result);
        }
        if (t == detail::TryResult::not_socket) st.flags |= detail::kFdNotSocket;
        else st.flags |= detail::kFdReadDrained;
    }
    if ((st.flags & detail::kFdNotSocket) != 0) {
        return impl_->oneshot(io_op::read, fd, buffer, size, nullptr, nullptr, callback, user_data, EPOLLIN);
    }
    pending_op* op = impl_->alloc_op();
    if (op == nullptr) return -1;
    op->operation = io_op::read;
    op->fd = fd;
    op->buffer = buffer;
    op->size = size;
    op->callback = std::move(callback);
    op->user_data = user_data;
    return impl_->arm_fast(op, 0, st);
}

int epoll_io::write_async(
    int fd,
    const void* buffer,
    size_t size,
    io_callback callback,
    void* user_data,
    ssize_t* inline_result
) noexcept {
    impl_->stat_writes.fetch_add(1, std::memory_order_relaxed);
    const bool fast = impl_->fast_ok(fd);
    detail::FdFast* st = fast ? &detail::fd_fast_current(impl_->fast.data(), fd) : nullptr;
    if (st != nullptr) detail::ensure_nonblocking(fd, *st);
    if (st != nullptr && (st->flags & detail::kFdNotSocket) == 0) {
        ssize_t n = 0;
        const detail::TryResult t = detail::try_send(fd, buffer, size, &n);
        if (t == detail::TryResult::done) {
            return impl_->finish_early(io_op::write, fd, n, callback, user_data, inline_result);
        }
        if (t == detail::TryResult::not_socket) st->flags |= detail::kFdNotSocket;
    }
    if (st == nullptr || (st->flags & detail::kFdNotSocket) != 0) {
        return impl_->oneshot(io_op::write, fd, const_cast<void*>(buffer), size, nullptr, nullptr,
                              callback, user_data, EPOLLOUT);
    }
    pending_op* op = impl_->alloc_op();
    if (op == nullptr) return -1;
    op->operation = io_op::write;
    op->fd = fd;
    op->buffer = const_cast<void*>(buffer);
    op->size = size;
    op->callback = std::move(callback);
    op->user_data = user_data;
    return impl_->arm_fast(op, 1, *st);
}

int epoll_io::connect_async(
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

    // Start connection
    int ret = connect(fd, addr, addrlen);
    if (ret < 0 && errno != EINPROGRESS) {
        impl_->stat_errors.fetch_add(1, std::memory_order_relaxed);
        impl_->free_index(op->pool_index);
        return -1;
    }

    impl_->stat_connects.fetch_add(1, std::memory_order_relaxed);
    return impl_->register_op(op, EPOLLOUT);
}

int epoll_io::recvfrom_async(
    int fd,
    void* buffer,
    size_t size,
    struct sockaddr* src,
    socklen_t* srclen,
    io_callback callback,
    void* user_data,
    ssize_t* inline_result
) noexcept {
    impl_->stat_reads.fetch_add(1, std::memory_order_relaxed);
    if (!impl_->fast_ok(fd)) {
        return impl_->oneshot(io_op::recvfrom, fd, buffer, size, src, srclen, callback, user_data, EPOLLIN);
    }
    detail::FdFast& st = detail::fd_fast_current(impl_->fast.data(), fd);
    detail::ensure_nonblocking(fd, st);
    if ((st.flags & detail::kFdReadDrained) == 0) {
        ssize_t n = 0;
        const detail::TryResult t = detail::try_recvfrom(fd, buffer, size, src, srclen, &n);
        if (t == detail::TryResult::done) {
            return impl_->finish_early(io_op::recvfrom, fd, n, callback, user_data, inline_result);
        }
        st.flags |= detail::kFdReadDrained;  // a datagram read is drained only on EAGAIN
    }
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
    return impl_->arm_fast(op, 0, st);
}

int epoll_io::sendto_async(
    int fd,
    const void* buffer,
    size_t size,
    const struct sockaddr* dst,
    socklen_t dstlen,
    io_callback callback,
    void* user_data
) noexcept {
    if (impl_->fast_ok(fd)) detail::ensure_nonblocking(fd, detail::fd_fast_current(impl_->fast.data(), fd));
    else impl_->set_nonblocking(fd);

    // UDP sendto rarely blocks: perform it directly in the submit path and
    // deliver the completion synchronously. `buffer`/`dst` are valid for the
    // duration of this call (the awaitable owner guarantees it). MSG_DONTWAIT
    // replaces the per-datagram fcntl pair (G2CHK-1269).
    ssize_t n = ::sendto(fd, buffer, size, MSG_DONTWAIT | MSG_NOSIGNAL, dst, dstlen);

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

int epoll_io::readable_async(int fd, io_callback callback, void* user_data) noexcept {
    if (!impl_->fast_ok(fd)) return impl_->oneshot(io_op::readable, fd, nullptr, 0, nullptr, nullptr, callback, user_data, EPOLLIN);
    detail::FdFast& st = detail::fd_fast_current(impl_->fast.data(), fd);
    detail::ensure_nonblocking(fd, st);
    pending_op* op = impl_->alloc_op();
    if (op == nullptr) return -1;
    op->operation = io_op::readable;
    op->fd = fd;
    op->callback = std::move(callback);
    op->user_data = user_data;
    return impl_->arm_fast(op, 0, st);
}

int epoll_io::close_async(int fd) noexcept {
    // Remove from epoll
    epoll_ctl(impl_->epoll_fd, EPOLL_CTL_DEL, fd, nullptr);

    // Complete any STILL-PENDING op for this fd with an error result so its
    // callback runs (mirrors IOCP, where closing the handle completes in-flight
    // operations). Without this a readiness backend would silently drop the op:
    // UdpTransport::stop() closes the socket and then SPINS waiting for its
    // recvfrom completion to decrement in_flight_ — if the callback never fires
    // it hangs forever. Firing it (the callback observes the transport's stop
    // flag, decrements, and does NOT re-arm) unblocks the drain.
    // Claim the in-flight op for this fd (atomic exchange — only one of close /
    // poll-completion wins; the op lives in the fixed pool so reading it is never
    // a use-after-free). The slot holds pool_index+1 (0 = none).
    for (int dir = 0; dir < 2; ++dir) {
        auto* s = impl_->slot_for(fd, dir);
        if (s == nullptr) break;
        std::uint32_t idx_p1 = s->exchange(0, std::memory_order_acq_rel);
        if (idx_p1 != 0) {
            pending_op& op = impl_->op_pool[idx_p1 - 1];
            if (op.callback) {
                io_event event;
                event.operation = op.operation;
                event.fd        = op.fd;
                event.user_data = op.user_data;
                event.flags     = 0;
                event.result    = -1;  // closed / cancelled
                op.callback(event);
            }
            impl_->free_index(idx_p1 - 1);
        }
    }
    if (fd >= 0 && fd < impl::kMaxFds) detail::fd_epoch_bump(fd);

    impl_->stat_closes.fetch_add(1, std::memory_order_relaxed);
    return close(fd);
}

void epoll_io::impl::on_persistent_side(int fd, int dir, uint32_t events) noexcept {
    assert(dir == 0 || dir == 1);
    std::atomic<std::uint32_t>* slot = slot_for(fd, dir);
    if (slot == nullptr) return;
    const std::uint32_t idx_p1 = slot->exchange(0, std::memory_order_acq_rel);
    detail::FdFast& st = detail::fd_fast_current(fast.data(), fd);
    if (idx_p1 == 0) {
        if (dir == 0) st.flags &= static_cast<std::uint8_t>(~detail::kFdReadDrained);
        return;
    }
    pending_op& op = op_pool[idx_p1 - 1];
    assert(op.fd == fd);
    ssize_t n = 0;
    detail::TryResult t = detail::TryResult::would_block;
    if (op.operation == io_op::readable) t = detail::TryResult::done;  // n = 0: readable
    else if (op.operation == io_op::read) t = detail::try_recv(fd, op.buffer, op.size, &n);
    else if (op.operation == io_op::recvfrom) t = detail::try_recvfrom(fd, op.buffer, op.size, op.user_src, op.user_srclen, &n);
    else if (op.operation == io_op::write) t = detail::try_send(fd, op.buffer, op.size, &n);
    if (t == detail::TryResult::would_block) {
        slot->store(idx_p1, std::memory_order_release);  // spurious edge: keep waiting
        return;
    }
    if (op.operation == io_op::read && n > 0 && static_cast<size_t>(n) < op.size) {
        st.flags |= detail::kFdReadDrained;
    } else if (op.operation != io_op::write) {
        // A full read, or one datagram: more may be queued behind it.
        st.flags &= static_cast<std::uint8_t>(~detail::kFdReadDrained);
    }
    if (t == detail::TryResult::not_socket) n = -1;
    const io_event event{op.operation, op.fd, op.user_data, n, events};
    if (op.callback) op.callback(event);
    free_index(idx_p1 - 1);
}

int epoll_io::poll(uint32_t timeout_us) noexcept {
    progress_accept_retirements();
    if (impl_->epoll_fd < 0) return -1;
    
    impl_->stat_polls.fetch_add(1, std::memory_order_relaxed);
    impl_->owner.note_poll();
    const bool fast = impl_->owner.on_owner();
    if (fast) impl_->run_deferred();
    if (fast && impl_->deferred.count != 0) timeout_us = 0;

    struct epoll_event events[128];
    int timeout_ms = timeout_us / 1000;

    int n = epoll_wait(impl_->epoll_fd, events, 128, timeout_ms);
    if (fast) park_slot_.word.store(::bolt::park::kRunning, std::memory_order_relaxed);
    if (n < 0 && errno == EINTR) return 0;
    if (n < 0) {
        if (errno != EINTR) {
            impl_->stat_errors.fetch_add(1, std::memory_order_relaxed);
        }
        return -1;
    }
    
    impl_->stat_events.fetch_add(n, std::memory_order_relaxed);
    
    // Process events
    for (int i = 0; i < n; ++i) {
        const struct epoll_event& ev = events[i];
        const std::uint64_t d = ev.data.u64;
        const std::uint32_t idx_p1 = static_cast<std::uint32_t>(d >> 32);

        // Wake event sentinel (data.u64 == 0; op events always have idx_p1 != 0).
        if (idx_p1 == 0) {
            // The eventfd is edge-triggered and never drained: every write is a
            // new edge (G2CHK-1270; the old clear-then-drain lost wakes, C1).
            impl_->stat_wakes.fetch_add(1, std::memory_order_relaxed);
            if (impl_->wake_cb) impl_->wake_cb();
            continue;
        }

        const int fd = static_cast<int>(static_cast<std::uint32_t>(d));
        if ((d & impl::kPersistTag) != 0) {
            if ((ev.events & (EPOLLIN | EPOLLERR | EPOLLHUP | EPOLLRDHUP)) != 0) {
                impl_->on_persistent_side(fd, 0, ev.events);
            }
            if ((ev.events & (EPOLLOUT | EPOLLERR | EPOLLHUP)) != 0) {
                impl_->on_persistent_side(fd, 1, ev.events);
            }
            continue;
        }

        // Claim the op (atomic — close_async may race; only one wins). The op
        // lives in the fixed pool, so reading it is never a use-after-free.
        pending_op* op = &impl_->op_pool[idx_p1 - 1];
        if (auto* s = impl_->slot_for(fd, impl::dir_of(op->events))) {
            std::uint32_t expected = idx_p1;
            if (!s->compare_exchange_strong(expected, 0, std::memory_order_acq_rel)) {
                continue;  // close_async (or a re-arm) already took this op
            }
        }

        // Execute operation
        io_event event;
        event.operation = op->operation;
        event.fd = op->fd;
        event.user_data = op->user_data;
        event.flags = ev.events;
        event.result = 0;
        
        // Check for errors
        if (ev.events & (EPOLLERR | EPOLLHUP)) {
            event.result = -1;
        } else {
            switch (op->operation) {
                case io_op::accept: {
                    struct sockaddr_storage addr;
                    socklen_t addrlen = sizeof(addr);
                    int client_fd = accept(op->fd, (struct sockaddr*)&addr, &addrlen);
                    if (client_fd >= 0 && client_fd < impl::kMaxFds) detail::fd_epoch_bump(client_fd);
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
                    // Socket is readable: do the recvfrom, filling the caller's
                    // peer-address out-params.
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
        }
        
        // Invoke callback, then return the op to the pool (no deallocation).
        if (op->callback) {
            op->callback(event);
        }
        impl_->free_index(op->pool_index);
    }

    return n;
}

AcceptRetirementStatus epoll_io::advance_accept_retirement(AcceptRetirement& record) noexcept {
    if (auto* slot = impl_->slot_for(record.fd, 0)) {
        const auto index = slot->load(std::memory_order_acquire);
        if (index != 0u) {
            auto& op = impl_->op_pool[index - 1u];
            if (op.operation != io_op::accept || op.retirement != &record) {
                record.error.store(EINVAL, std::memory_order_release); return AcceptRetirementStatus::Error;
            }
            if (epoll_ctl(impl_->epoll_fd, EPOLL_CTL_DEL, record.fd, nullptr) != 0) {
                record.error.store(errno, std::memory_order_release); return AcceptRetirementStatus::Error;
            }
            const auto claimed = slot->exchange(0u, std::memory_order_acq_rel);
            if (claimed == index) {
                const io_event event{io_op::accept, record.fd, op.user_data, -1, 0u};
                if (op.callback) op.callback(event);
                impl_->free_index(index - 1u);
            }
        }
    }
    if (record.operations.load(std::memory_order_acquire) != 0u ||
        !record.loop_exited.load(std::memory_order_acquire)) return AcceptRetirementStatus::Pending;
    if (::close(record.fd) != 0) {
        record.error.store(errno, std::memory_order_release); return AcceptRetirementStatus::Error;
    }
    return AcceptRetirementStatus::Complete;
}

void epoll_io::run() noexcept {
    if (impl_->running.exchange(true)) {
        return;  // Already running
    }
    
    impl_->stop_requested.store(false);
    
    while (!impl_->stop_requested.load(std::memory_order_acquire)) {
        poll(impl_->config.poll_timeout_us);
    }
    
    impl_->running.store(false);
}

void epoll_io::stop() noexcept {
    impl_->stop_requested.store(true, std::memory_order_release);
}

bool epoll_io::is_running() const noexcept {
    return impl_->running.load(std::memory_order_acquire);
}

async_io::stats epoll_io::get_stats() const noexcept {
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
    s.registrations = impl_->stat_registrations.load(std::memory_order_relaxed);
    s.inline_completions = impl_->stat_inline.load(std::memory_order_relaxed);
    s.deferred_completions = impl_->stat_deferred.load(std::memory_order_relaxed);
    return s;
}

// Raw backend wake; only the park::wake CAS winner calls it (G2CHK-1270).
void epoll_io::impl::kernel_wake(void* ctx) noexcept {
    auto* self = static_cast<epoll_io::impl*>(ctx);
    assert(self != nullptr);
    if (self->wake_fd < 0) return;
    const uint64_t one = 1;
    (void)write(self->wake_fd, &one, sizeof(one));
}

void epoll_io::wake() noexcept {
    (void)::bolt::park::wake(&park_slot_);
}

bool epoll_io::has_ready() const noexcept {
    return impl_->deferred.count != 0;
}

void epoll_io::set_wake_callback(wake_callback callback) noexcept {
    impl_->wake_cb = std::move(callback);
}

} // namespace core
} // namespace bolt::api

#endif // __linux__









