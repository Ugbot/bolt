// io_uring-based async I/O implementation (Linux 5.11+)
//
// Ring-per-thread design: each IODispatcher IO thread owns its own io_uring
// instance (slice A: per-thread async_io engines). Single-owner ring → plain
// free-list (assert owner thread); keep an ATOMIC fd-slot because close_async
// is the lone cross-thread caller (UdpTransport::stop / connection teardown).
//
// Built via raw syscalls (no liburing) to honor the project's no-deps rule:
// io_uring_setup(2) + io_uring_enter(2). ABI types come from <linux/io_uring.h>
// directly (no hand-rolled ABI fallback; this TU is #ifdef __linux__ and
// modern distros ship the header).
//
// Robust fallback: ctor records setup failure (ENOSYS / EPERM under Docker
// default seccomp / kernel <5.11). usable() = false → async_io::create()
// transparently returns epoll. NEVER abort — the default container lanes
// run epoll; a dedicated unconfined lane exercises the io_uring path.
//
// Op→SQE: ACCEPT, RECV (read_async), SEND (write_async), CONNECT, RECVMSG
// (recvfrom_async), SENDMSG (sendto_async), POLL_ADD on the wake-eventfd.
// close_async stays SYNCHRONOUS (cancel+close) — mirrors epoll, preserving
// the UdpTransport::stop drain semantic the engine API contract relies on.

#include "bolt/api/core/async_io.h"

#ifdef __linux__

#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <poll.h>
#include <time.h>

#include <vector>
#include <atomic>
#include <mutex>
#include <cstring>
#include <cassert>
#include <thread>

// Some distros' io_uring headers lag the kernel; provide fallbacks so we can
// build on older toolchains while still using the runtime features we need.
#ifndef IORING_ENTER_EXT_ARG
#define IORING_ENTER_EXT_ARG (1U << 3)
#endif
#ifndef IORING_FEAT_EXT_ARG
#define IORING_FEAT_EXT_ARG  (1U << 8)
#endif
#ifndef IORING_FEAT_SINGLE_MMAP
#define IORING_FEAT_SINGLE_MMAP  (1U << 0)
#endif

namespace bolt::api {
namespace core {

namespace {

// Raw syscall wrappers (no liburing). Glibc lacks wrappers for these on older
// toolchains, so go through syscall(2) directly — same approach as epoll's
// existing primitives. Kernel ABI is stable.
inline int io_uring_setup(unsigned entries, struct io_uring_params* p) noexcept {
    return static_cast<int>(syscall(__NR_io_uring_setup, entries, p));
}
inline int io_uring_enter(int fd, unsigned to_submit, unsigned min_complete,
                          unsigned flags, const void* arg, std::size_t argsz) noexcept {
    return static_cast<int>(syscall(__NR_io_uring_enter, fd, to_submit,
                                    min_complete, flags, arg, argsz));
}

// Wake-fd sentinel: a CQE with user_data == 0 means the eventfd POLL_ADD fired.
// User ops carry user_data = op_pool_index + 1 (never zero).
constexpr std::uint64_t kWakeUserData = 0;

}  // namespace

// =============================================================================
// Per-op state — owned by the op pool slot for the in-flight duration. RECVMSG
// and SENDMSG need a stable msghdr+iovec+sockaddr (the kernel reads them async,
// not at submit time — the epoll "valid only during call" guarantee is void).
// =============================================================================
struct uring_pending_op {
    io_op operation{io_op::read};
    int fd{-1};
    io_callback callback;
    void* user_data{nullptr};

    // For read/write/recv/send
    const void* cbuf{nullptr};
    void*       buf{nullptr};
    std::size_t len{0};

    // For RECVMSG/SENDMSG: msghdr + iovec + addr live IN THE SLOT.
    struct msghdr msg {};
    struct iovec  iov {};
    struct sockaddr_storage addr {};
    socklen_t addrlen{0};

    // For RECVMSG completion: caller's out-params (memcpy on completion).
    struct sockaddr* user_src{nullptr};
    socklen_t*       user_srclen{nullptr};

    // Slot index in the fixed pool.
    std::uint32_t pool_index{0};

    void reset() noexcept {
        operation = io_op::read;
        fd = -1;
        callback = nullptr;
        user_data = nullptr;
        cbuf = nullptr;
        buf = nullptr;
        len = 0;
        std::memset(&msg, 0, sizeof(msg));
        std::memset(&iov, 0, sizeof(iov));
        std::memset(&addr, 0, sizeof(addr));
        addrlen = 0;
        user_src = nullptr;
        user_srclen = nullptr;
    }
};

// =============================================================================
// io_uring_io::impl
// =============================================================================
struct io_uring_io::impl {
    async_io_config config;

    // Lifecycle / probe result.
    bool   setup_ok{false};
    int    ring_fd{-1};

    // mmap regions.
    void*       sq_ring_ptr{nullptr};
    std::size_t sq_ring_sz{0};
    void*       cq_ring_ptr{nullptr};
    std::size_t cq_ring_sz{0};
    struct io_uring_sqe* sqes{nullptr};
    std::size_t sqes_sz{0};

    // Cached SQ ring pointers.
    unsigned* sq_head{nullptr};
    unsigned* sq_tail{nullptr};
    unsigned* sq_ring_mask{nullptr};
    unsigned* sq_flags{nullptr};
    unsigned* sq_array{nullptr};

    // Cached CQ ring pointers.
    unsigned* cq_head{nullptr};
    unsigned* cq_tail{nullptr};
    unsigned* cq_ring_mask{nullptr};
    struct io_uring_cqe* cqes{nullptr};

    unsigned sq_entries{0};
    unsigned cq_entries{0};

    // Wake mechanism: eventfd watched by a POLL_ADD SQE (re-armed each fire).
    int wake_fd{-1};
    bool wake_armed{false};
    async_io::wake_callback wake_cb;
    std::atomic<bool> wake_pending{false};

    // Pending SQEs not yet submitted (lazy-submit at the next poll → batching).
    unsigned pending_submits{0};

    // Submit-side mutex. Three reasons it has to exist:
    //  1. io_uring's SQ ring is SINGLE-PRODUCER by design (one writer to
    //     `*sq_tail`). Multiple threads calling submit_op concurrently must
    //     serialize the SQE write + tail bump.
    //  2. The op-pool free-list is a plain stack (no atomic Treiber stack here;
    //     keeping it simple). alloc_op + free_op need serialization.
    //  3. The pending_submits counter (queued SQEs not yet io_uring_enter'd)
    //     is read by the IO thread in do_poll and written by submitters.
    // Server-side post-slice-B: connections are pinned, ALL submits happen on
    // the same IO thread → mutex is uncontended (a few cycles per acquire,
    // dwarfed by the syscall avoided when batching). Client-side / pre-pinning
    // (e.g. HTTP client's first async_connect from the calling thread): the
    // mutex actually contends but correctness is the priority.
    std::mutex submit_mu;

    // Op pool — protected by submit_mu (mutex above).
    static constexpr std::uint32_t kOpPoolSize = 4096;
    static constexpr int           kMaxFds     = 1 << 16;
    std::vector<uring_pending_op> op_pool{std::vector<uring_pending_op>(kOpPoolSize)};
    std::vector<std::uint32_t>    free_stack{std::vector<std::uint32_t>(kOpPoolSize)};
    std::uint32_t                 free_top{0};
    std::vector<std::atomic<std::uint32_t>> fd_slot{std::vector<std::atomic<std::uint32_t>>(kMaxFds)};

    // Owner-thread guard (set on first poll); helps catch a cross-thread submit
    // regression if anyone ever takes us off-thread.
    std::atomic<std::thread::id> owner_tid{std::thread::id{}};

    // Lifecycle flags.
    std::atomic<bool> running{false};
    std::atomic<bool> stop_requested{false};

    // Statistics.
    std::atomic<std::uint64_t> stat_accepts{0};
    std::atomic<std::uint64_t> stat_reads{0};
    std::atomic<std::uint64_t> stat_writes{0};
    std::atomic<std::uint64_t> stat_connects{0};
    std::atomic<std::uint64_t> stat_closes{0};
    std::atomic<std::uint64_t> stat_polls{0};
    std::atomic<std::uint64_t> stat_events{0};
    std::atomic<std::uint64_t> stat_errors{0};
    std::atomic<std::uint64_t> stat_wakes{0};

    // -----------------------------------------------------------------------
    // Setup / teardown
    // -----------------------------------------------------------------------
    explicit impl(const async_io_config& cfg) : config(cfg) {
        // Init pool.
        for (std::uint32_t i = 0; i < kOpPoolSize; ++i) {
            op_pool[i].pool_index = i;
            free_stack[i] = kOpPoolSize - 1 - i;  // pop high-to-low for cache locality
        }
        free_top = kOpPoolSize;
        for (auto& s : fd_slot) s.store(0, std::memory_order_relaxed);

        // Sized for queue_depth (pow2). Bound: op pool >= sq_entries (assert).
        unsigned want = config.queue_depth;
        if (want == 0) want = 4096;
        // Round up to power of two as required by io_uring.
        unsigned po2 = 1;
        while (po2 < want) po2 <<= 1;
        want = po2;

        struct io_uring_params p{};
        ring_fd = io_uring_setup(want, &p);
        if (ring_fd < 0) {
            // Most likely: ENOSYS (kernel <5.1), EPERM (Docker default seccomp
            // blocks io_uring), or an old kernel without our features. Stay
            // unusable() → create() falls back to epoll.
            return;
        }
        // Require EXT_ARG (kernel 5.11+) so we can express a poll timeout without
        // burning an SQE on a linked TIMEOUT op. Below that floor: fall back.
        if ((p.features & IORING_FEAT_EXT_ARG) == 0) {
            ::close(ring_fd);
            ring_fd = -1;
            return;
        }

        sq_entries = p.sq_entries;
        cq_entries = p.cq_entries;
        assert(sq_entries <= kOpPoolSize && "io_uring SQ deeper than op pool");

        // Map the three regions (or two if SINGLE_MMAP).
        sq_ring_sz = p.sq_off.array + p.sq_entries * sizeof(unsigned);
        cq_ring_sz = p.cq_off.cqes  + p.cq_entries * sizeof(struct io_uring_cqe);
        if (p.features & IORING_FEAT_SINGLE_MMAP) {
            if (cq_ring_sz > sq_ring_sz) sq_ring_sz = cq_ring_sz;
            cq_ring_sz = sq_ring_sz;
        }
        sq_ring_ptr = mmap(nullptr, sq_ring_sz, PROT_READ | PROT_WRITE,
                           MAP_SHARED | MAP_POPULATE, ring_fd, IORING_OFF_SQ_RING);
        if (sq_ring_ptr == MAP_FAILED) { ::close(ring_fd); ring_fd = -1; return; }

        if (p.features & IORING_FEAT_SINGLE_MMAP) {
            cq_ring_ptr = sq_ring_ptr;
        } else {
            cq_ring_ptr = mmap(nullptr, cq_ring_sz, PROT_READ | PROT_WRITE,
                               MAP_SHARED | MAP_POPULATE, ring_fd, IORING_OFF_CQ_RING);
            if (cq_ring_ptr == MAP_FAILED) {
                munmap(sq_ring_ptr, sq_ring_sz);
                ::close(ring_fd);
                ring_fd = -1;
                return;
            }
        }

        sqes_sz = p.sq_entries * sizeof(struct io_uring_sqe);
        sqes = static_cast<struct io_uring_sqe*>(
            mmap(nullptr, sqes_sz, PROT_READ | PROT_WRITE,
                 MAP_SHARED | MAP_POPULATE, ring_fd, IORING_OFF_SQES));
        if (sqes == MAP_FAILED) {
            if (cq_ring_ptr != sq_ring_ptr) munmap(cq_ring_ptr, cq_ring_sz);
            munmap(sq_ring_ptr, sq_ring_sz);
            ::close(ring_fd);
            ring_fd = -1;
            sqes = nullptr;
            return;
        }

        // Compute cached ring pointers via the kernel-supplied offsets.
        auto sqbase = static_cast<char*>(sq_ring_ptr);
        sq_head      = reinterpret_cast<unsigned*>(sqbase + p.sq_off.head);
        sq_tail      = reinterpret_cast<unsigned*>(sqbase + p.sq_off.tail);
        sq_ring_mask = reinterpret_cast<unsigned*>(sqbase + p.sq_off.ring_mask);
        sq_flags     = reinterpret_cast<unsigned*>(sqbase + p.sq_off.flags);
        sq_array     = reinterpret_cast<unsigned*>(sqbase + p.sq_off.array);

        auto cqbase = static_cast<char*>(cq_ring_ptr);
        cq_head      = reinterpret_cast<unsigned*>(cqbase + p.cq_off.head);
        cq_tail      = reinterpret_cast<unsigned*>(cqbase + p.cq_off.tail);
        cq_ring_mask = reinterpret_cast<unsigned*>(cqbase + p.cq_off.ring_mask);
        cqes         = reinterpret_cast<struct io_uring_cqe*>(cqbase + p.cq_off.cqes);

        // Wake-fd: an eventfd watched via POLL_ADD. Re-armed each completion.
        wake_fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
        if (wake_fd >= 0) {
            // Single-threaded during ctor — no contention; lock for invariant only.
            std::lock_guard<std::mutex> lk(submit_mu);
            arm_wake_poll_locked();
        }

        setup_ok = true;
    }

    ~impl() {
        if (sqes && sqes != MAP_FAILED)       munmap(sqes, sqes_sz);
        if (cq_ring_ptr && cq_ring_ptr != sq_ring_ptr && cq_ring_ptr != MAP_FAILED) {
            munmap(cq_ring_ptr, cq_ring_sz);
        }
        if (sq_ring_ptr && sq_ring_ptr != MAP_FAILED) munmap(sq_ring_ptr, sq_ring_sz);
        if (wake_fd >= 0) ::close(wake_fd);
        if (ring_fd >= 0) ::close(ring_fd);
    }

    // -----------------------------------------------------------------------
    // Op pool (single-owner)
    // -----------------------------------------------------------------------
    uring_pending_op* alloc_op() noexcept {
        if (free_top == 0) return nullptr;  // pool exhausted
        const std::uint32_t idx = free_stack[--free_top];
        op_pool[idx].reset();
        op_pool[idx].pool_index = idx;
        return &op_pool[idx];
    }
    void free_op(std::uint32_t idx) noexcept {
        assert(idx < kOpPoolSize && "free_op: bad index");
        assert(free_top < kOpPoolSize && "free_op: stack overflow (double free?)");
        free_stack[free_top++] = idx;
    }
    std::atomic<std::uint32_t>* slot_for(int fd) noexcept {
        if (fd < 0 || fd >= kMaxFds) return nullptr;
        return &fd_slot[static_cast<std::size_t>(fd)];
    }

    // -----------------------------------------------------------------------
    // SQE submit helpers
    // -----------------------------------------------------------------------
    // Get the next SQE slot to write. Returns nullptr if SQ is full (caller
    // should flush via io_uring_enter and retry — we lazily submit so this
    // would only fire under sustained pressure faster than poll() drains).
    struct io_uring_sqe* next_sqe() noexcept {
        const unsigned tail = *sq_tail;
        const unsigned head = __atomic_load_n(sq_head, __ATOMIC_ACQUIRE);
        if (tail - head >= sq_entries) return nullptr;
        const unsigned index = tail & *sq_ring_mask;
        struct io_uring_sqe* s = &sqes[index];
        std::memset(s, 0, sizeof(*s));  // CRITICAL: stale fields cause -EINVAL.
        sq_array[index] = index;
        return s;
    }
    void publish_sqe() noexcept {
        // Release-store the bumped tail so the kernel sees our SQE.
        __atomic_store_n(sq_tail, *sq_tail + 1, __ATOMIC_RELEASE);
        ++pending_submits;
    }

    // Arm a single-shot POLL_ADD on wake_fd; user_data = kWakeUserData (0).
    // Called from the IO thread (do_poll) with submit_mu HELD by the caller —
    // the lock protects the SQE write + sq_tail bump shared with cross-thread
    // submitters.
    void arm_wake_poll_locked() noexcept {
        if (wake_armed || wake_fd < 0) return;
        struct io_uring_sqe* s = next_sqe();
        if (s == nullptr) return;  // SQ full — re-arm on next poll
        s->opcode = IORING_OP_POLL_ADD;
        s->fd = wake_fd;
        s->poll_events = POLLIN;
        s->user_data = kWakeUserData;
        publish_sqe();
        wake_armed = true;
    }

    // -----------------------------------------------------------------------
    // Submit dispatch (8 ops). Each takes submit_mu around alloc + SQE fill +
    // sq_tail bump (io_uring SQ ring is single-producer; cross-thread submitters
    // — e.g. HTTP client's first async_connect from the calling thread before
    // the connection pins itself to an IO thread — must serialize here).
    // -----------------------------------------------------------------------
    int submit_accept(int listen_fd, io_callback cb, void* udata) noexcept {
        std::lock_guard<std::mutex> lk(submit_mu);
        uring_pending_op* op = alloc_op();
        if (op == nullptr) return -1;
        op->operation = io_op::accept;
        op->fd = listen_fd;
        op->callback = std::move(cb);
        op->user_data = udata;
        op->addrlen = sizeof(op->addr);
        struct io_uring_sqe* s = next_sqe();
        if (s == nullptr) { free_op(op->pool_index); return -1; }
        s->opcode = IORING_OP_ACCEPT;
        s->fd = listen_fd;
        s->addr = reinterpret_cast<__u64>(&op->addr);
        s->off  = reinterpret_cast<__u64>(&op->addrlen);  // addr2
        s->accept_flags = SOCK_CLOEXEC;
        s->user_data = op->pool_index + 1;
        publish_sqe();
        stat_accepts.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }

    int submit_read(int fd, void* buffer, std::size_t size,
                    io_callback cb, void* udata) noexcept {
        std::lock_guard<std::mutex> lk(submit_mu);
        uring_pending_op* op = alloc_op();
        if (op == nullptr) return -1;
        op->operation = io_op::read;
        op->fd = fd;
        op->callback = std::move(cb);
        op->user_data = udata;
        op->buf = buffer;
        op->len = size;
        struct io_uring_sqe* s = next_sqe();
        if (s == nullptr) { free_op(op->pool_index); return -1; }
        s->opcode = IORING_OP_RECV;
        s->fd = fd;
        s->addr = reinterpret_cast<__u64>(buffer);
        s->len  = static_cast<__u32>(size);
        s->user_data = op->pool_index + 1;
        publish_sqe();
        stat_reads.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }

    int submit_write(int fd, const void* buffer, std::size_t size,
                     io_callback cb, void* udata) noexcept {
        std::lock_guard<std::mutex> lk(submit_mu);
        uring_pending_op* op = alloc_op();
        if (op == nullptr) return -1;
        op->operation = io_op::write;
        op->fd = fd;
        op->callback = std::move(cb);
        op->user_data = udata;
        op->cbuf = buffer;
        op->len = size;
        struct io_uring_sqe* s = next_sqe();
        if (s == nullptr) { free_op(op->pool_index); return -1; }
        s->opcode = IORING_OP_SEND;
        s->fd = fd;
        s->addr = reinterpret_cast<__u64>(buffer);
        s->len  = static_cast<__u32>(size);
        s->user_data = op->pool_index + 1;
        publish_sqe();
        stat_writes.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }

    int submit_connect(int fd, const struct sockaddr* addr, socklen_t addrlen,
                       io_callback cb, void* udata) noexcept {
        if (addrlen > sizeof(sockaddr_storage)) return -1;
        std::lock_guard<std::mutex> lk(submit_mu);
        uring_pending_op* op = alloc_op();
        if (op == nullptr) return -1;
        op->operation = io_op::connect;
        op->fd = fd;
        op->callback = std::move(cb);
        op->user_data = udata;
        std::memcpy(&op->addr, addr, addrlen);
        op->addrlen = addrlen;
        struct io_uring_sqe* s = next_sqe();
        if (s == nullptr) { free_op(op->pool_index); return -1; }
        s->opcode = IORING_OP_CONNECT;
        s->fd = fd;
        s->addr = reinterpret_cast<__u64>(&op->addr);
        s->off  = static_cast<__u64>(op->addrlen);
        s->user_data = op->pool_index + 1;
        publish_sqe();
        stat_connects.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }

    int submit_recvfrom(int fd, void* buffer, std::size_t size,
                        struct sockaddr* user_src, socklen_t* user_srclen,
                        io_callback cb, void* udata) noexcept {
        std::lock_guard<std::mutex> lk(submit_mu);
        uring_pending_op* op = alloc_op();
        if (op == nullptr) return -1;
        op->operation = io_op::recvfrom;
        op->fd = fd;
        op->callback = std::move(cb);
        op->user_data = udata;
        op->buf = buffer;
        op->len = size;
        op->user_src = user_src;
        op->user_srclen = user_srclen;
        // msghdr+iovec live IN the op slot (kernel reads async, the awaitable's
        // stack-bound src/srclen are not safe to pass through directly).
        op->iov.iov_base = buffer;
        op->iov.iov_len = size;
        op->msg.msg_name = &op->addr;
        op->msg.msg_namelen = sizeof(op->addr);
        op->msg.msg_iov = &op->iov;
        op->msg.msg_iovlen = 1;
        struct io_uring_sqe* s = next_sqe();
        if (s == nullptr) { free_op(op->pool_index); return -1; }
        s->opcode = IORING_OP_RECVMSG;
        s->fd = fd;
        s->addr = reinterpret_cast<__u64>(&op->msg);
        s->len  = 1;
        s->user_data = op->pool_index + 1;
        publish_sqe();
        stat_reads.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }

    int submit_sendto(int fd, const void* buffer, std::size_t size,
                      const struct sockaddr* dst, socklen_t dstlen,
                      io_callback cb, void* udata) noexcept {
        if (dstlen > sizeof(sockaddr_storage)) return -1;
        std::lock_guard<std::mutex> lk(submit_mu);
        uring_pending_op* op = alloc_op();
        if (op == nullptr) return -1;
        op->operation = io_op::sendto;
        op->fd = fd;
        op->callback = std::move(cb);
        op->user_data = udata;
        op->cbuf = buffer;
        op->len = size;
        std::memcpy(&op->addr, dst, dstlen);
        op->addrlen = dstlen;
        // memcpy dst into the slot — the epoll "valid only during call"
        // guarantee on async_io.h:225-227 is VOID for io_uring (async read).
        op->iov.iov_base = const_cast<void*>(buffer);  // SENDMSG won't mutate
        op->iov.iov_len = size;
        op->msg.msg_name = &op->addr;
        op->msg.msg_namelen = op->addrlen;
        op->msg.msg_iov = &op->iov;
        op->msg.msg_iovlen = 1;
        struct io_uring_sqe* s = next_sqe();
        if (s == nullptr) { free_op(op->pool_index); return -1; }
        s->opcode = IORING_OP_SENDMSG;
        s->fd = fd;
        s->addr = reinterpret_cast<__u64>(&op->msg);
        s->len  = 1;
        s->user_data = op->pool_index + 1;
        publish_sqe();
        stat_writes.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }

    // close_async: just ::close(fd) — DO NOT touch the op pool or fire
    // callbacks from this thread. io_uring's op pool + free-list are
    // single-owner (the IO thread that runs poll); close_async is the lone
    // cross-thread entry (UdpTransport::stop / connection teardown can call
    // from anywhere). Closing the fd causes the kernel to complete any
    // pending op on it with a negative cqe->res (-ECANCELED / -EBADF);
    // those completions land in the IO thread's next poll() and fire their
    // callbacks there (correct thread, normal free_op). The UdpTransport
    // drain on in_flight_ observes those decrements normally — just with
    // a poll-cycle of latency vs the epoll backend's synchronous fire.
    //
    // Earlier version (cancel+fire+free here, mirroring epoll) corrupted
    // the single-owner free-list under cross-thread close.
    int close_async_impl(int fd) noexcept {
        if (fd < 0) return -1;
        ::close(fd);
        stat_closes.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }

    // -----------------------------------------------------------------------
    // poll(timeout_us): flush pending submits + reap CQEs (bounded).
    // -----------------------------------------------------------------------
    int do_poll(std::uint32_t timeout_us) noexcept {
        // Owner-thread guard (set once; assert thereafter).
        if (owner_tid.load(std::memory_order_relaxed) == std::thread::id{}) {
            owner_tid.store(std::this_thread::get_id(), std::memory_order_relaxed);
        }
        assert(owner_tid.load(std::memory_order_relaxed) == std::this_thread::get_id() &&
               "io_uring_io::poll called from a thread other than its owner");

        stat_polls.fetch_add(1, std::memory_order_relaxed);

        unsigned to_submit = 0;
        {
            // Re-arm wake POLL_ADD if previous fired and we haven't re-armed yet
            // (e.g. SQ full at the prior arm). Under submit_mu — shares the SQE
            // write path with cross-thread submitters.
            std::lock_guard<std::mutex> lk(submit_mu);
            if (!wake_armed) arm_wake_poll_locked();
            to_submit = pending_submits;
            pending_submits = 0;
        }

        unsigned min_complete = 0;
        unsigned flags = 0;

        // The kernel's __kernel_timespec is a 64-bit-only struct.
        struct __kernel_timespec ts {};
        struct io_uring_getevents_arg arg{};
        const void* arg_ptr = nullptr;
        std::size_t arg_sz = 0;

        if (timeout_us > 0) {
            min_complete = 1;
            ts.tv_sec  = static_cast<__s64>(timeout_us / 1000000U);
            ts.tv_nsec = static_cast<long long>((timeout_us % 1000000U) * 1000U);
            arg.ts = reinterpret_cast<__u64>(&ts);
            arg.sigmask_sz = 0;
            arg.sigmask = 0;
            flags |= IORING_ENTER_GETEVENTS | IORING_ENTER_EXT_ARG;
            arg_ptr = &arg;
            arg_sz = sizeof(arg);
        } else if (to_submit > 0) {
            // Submit-only path (non-blocking drain).
        }

        if (to_submit > 0 || (flags & IORING_ENTER_GETEVENTS)) {
            int r = io_uring_enter(ring_fd, to_submit, min_complete, flags, arg_ptr, arg_sz);
            if (r < 0 && errno != EINTR && errno != ETIME) {
                stat_errors.fetch_add(1, std::memory_order_relaxed);
                // Don't abort — fall through to reap whatever's already in CQ.
            }
        }

        // Reap CQEs, bounded by max_events so a flood can't starve other work.
        unsigned reaped = 0;
        const unsigned max_events = config.max_events ? config.max_events : 1024;
        unsigned head = *cq_head;
        const unsigned tail = __atomic_load_n(cq_tail, __ATOMIC_ACQUIRE);

        while (head != tail && reaped < max_events) {
            const struct io_uring_cqe* cqe = &cqes[head & *cq_ring_mask];
            const std::uint64_t ud = cqe->user_data;
            const ssize_t res = cqe->res;  // negative = -errno
            ++head;
            ++reaped;

            if (ud == kWakeUserData) {
                // wake-fd POLL_ADD fired: clear the eventfd, dispatch wake_cb,
                // mark for re-arm (top of next poll). wake_armed flag is touched
                // only by the IO thread (here + arm_wake_poll_locked); but the
                // lock is cheap and consistent with the submit path.
                std::uint64_t v = 0;
                while (::read(wake_fd, &v, sizeof(v)) > 0) { /* drain */ }
                {
                    std::lock_guard<std::mutex> lk(submit_mu);
                    wake_armed = false;
                }
                stat_wakes.fetch_add(1, std::memory_order_relaxed);
                wake_pending.store(false, std::memory_order_release);
                if (wake_cb) wake_cb();
                continue;
            }

            const std::uint32_t idx_p1 = static_cast<std::uint32_t>(ud);
            if (idx_p1 == 0 || idx_p1 - 1 >= kOpPoolSize) {
                stat_errors.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            uring_pending_op& op = op_pool[idx_p1 - 1];
            // cqe->user_data is authoritative — no fd_slot match needed.
            // io_uring allows multiple ops in flight on the same fd (HTTP
            // client overlaps write/read on a keep-alive socket); the
            // earlier fd_slot-based match dropped callbacks for all but
            // the most recently submitted op on a given fd.

            // RECVMSG: write the peer address back to the caller's out-params.
            if (op.operation == io_op::recvfrom && op.user_src && op.user_srclen) {
                const socklen_t copy = (op.msg.msg_namelen < *op.user_srclen)
                    ? op.msg.msg_namelen : *op.user_srclen;
                std::memcpy(op.user_src, &op.addr, copy);
                *op.user_srclen = op.msg.msg_namelen;
            }

            // Fire callback with normalized result (epoll-compatible: -1 on error).
            io_event ev{
                op.operation,
                op.fd,
                op.user_data,
                (res < 0) ? ssize_t{-1} : res,
                0u
            };
            // Fire the callback OUTSIDE the lock — it can resume a coroutine
            // that submits a new op (which re-acquires submit_mu); holding the
            // lock around resume would self-deadlock. Then briefly lock to
            // return the slot to the pool.
            if (op.callback) op.callback(ev);
            {
                std::lock_guard<std::mutex> lk(submit_mu);
                free_op(idx_p1 - 1);
            }
            stat_events.fetch_add(1, std::memory_order_relaxed);
        }
        __atomic_store_n(cq_head, head, __ATOMIC_RELEASE);
        return static_cast<int>(reaped);
    }
};

// =============================================================================
// io_uring_io public interface
// =============================================================================
io_uring_io::io_uring_io(const async_io_config& config)
    : impl_(std::make_unique<impl>(config)) {}

io_uring_io::~io_uring_io() = default;

int io_uring_io::accept_async(int listen_fd, io_callback callback, void* user_data) noexcept {
    return impl_->submit_accept(listen_fd, std::move(callback), user_data);
}
int io_uring_io::read_async(int fd, void* buffer, std::size_t size,
                            io_callback callback, void* user_data) noexcept {
    return impl_->submit_read(fd, buffer, size, std::move(callback), user_data);
}
int io_uring_io::write_async(int fd, const void* buffer, std::size_t size,
                             io_callback callback, void* user_data) noexcept {
    return impl_->submit_write(fd, buffer, size, std::move(callback), user_data);
}
int io_uring_io::connect_async(int fd, const struct sockaddr* addr, socklen_t addrlen,
                               io_callback callback, void* user_data) noexcept {
    return impl_->submit_connect(fd, addr, addrlen, std::move(callback), user_data);
}
int io_uring_io::recvfrom_async(int fd, void* buffer, std::size_t size,
                                struct sockaddr* src, socklen_t* srclen,
                                io_callback callback, void* user_data) noexcept {
    return impl_->submit_recvfrom(fd, buffer, size, src, srclen, std::move(callback), user_data);
}
int io_uring_io::sendto_async(int fd, const void* buffer, std::size_t size,
                              const struct sockaddr* dst, socklen_t dstlen,
                              io_callback callback, void* user_data) noexcept {
    return impl_->submit_sendto(fd, buffer, size, dst, dstlen, std::move(callback), user_data);
}
int io_uring_io::close_async(int fd) noexcept {
    return impl_->close_async_impl(fd);
}
int io_uring_io::poll(std::uint32_t timeout_us) noexcept {
    return impl_->do_poll(timeout_us);
}
void io_uring_io::run() noexcept {
    impl_->running.store(true, std::memory_order_release);
    while (!impl_->stop_requested.load(std::memory_order_acquire)) {
        impl_->do_poll(impl_->config.poll_timeout_us ? impl_->config.poll_timeout_us : 1000);
    }
    impl_->running.store(false, std::memory_order_release);
}
void io_uring_io::stop() noexcept {
    impl_->stop_requested.store(true, std::memory_order_release);
    // Wake any blocked poll() so it observes stop_requested.
    wake();
}
bool io_uring_io::is_running() const noexcept {
    return impl_->running.load(std::memory_order_acquire);
}
void io_uring_io::wake() noexcept {
    // Dedupe — many concurrent wakes collapse to one eventfd write.
    bool expected = false;
    if (!impl_->wake_pending.compare_exchange_strong(expected, true,
                                                     std::memory_order_acq_rel)) {
        return;
    }
    if (impl_->wake_fd >= 0) {
        const std::uint64_t one = 1;
        const ssize_t r = ::write(impl_->wake_fd, &one, sizeof(one));
        (void)r;  // eventfd write either succeeds atomically or returns EAGAIN
                  // (already non-zero); both are fine.
    }
}
void io_uring_io::set_wake_callback(wake_callback callback) noexcept {
    impl_->wake_cb = std::move(callback);
}
async_io::stats io_uring_io::get_stats() const noexcept {
    return stats{
        impl_->stat_accepts.load(std::memory_order_relaxed),
        impl_->stat_reads.load(std::memory_order_relaxed),
        impl_->stat_writes.load(std::memory_order_relaxed),
        impl_->stat_connects.load(std::memory_order_relaxed),
        impl_->stat_closes.load(std::memory_order_relaxed),
        impl_->stat_polls.load(std::memory_order_relaxed),
        impl_->stat_events.load(std::memory_order_relaxed),
        impl_->stat_errors.load(std::memory_order_relaxed),
        impl_->stat_wakes.load(std::memory_order_relaxed),
    };
}

// Probe: did io_uring_setup succeed AND have we got the kernel features we
// need (IORING_FEAT_EXT_ARG)? The factory checks this and falls back to epoll
// when false (Docker default seccomp / kernel <5.11). Non-virtual; called only
// in the factory while the static type is io_uring_io.
bool io_uring_io::usable() const noexcept {
    return impl_ != nullptr && impl_->setup_ok;
}

}  // namespace core
}  // namespace bolt::api

#endif  // __linux__
