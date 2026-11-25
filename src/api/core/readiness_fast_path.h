// readiness_fast_path.h — per-fd state for the kqueue and epoll backends'
// register-once / try-first path (G2CHK-1269). Internal to bolt::reactor.
//
// Contract (owner thread = the one thread that polls the engine):
//  - An fd is registered with the kernel once (kqueue EV_CLEAR per filter, epoll
//    EPOLLIN|EPOLLOUT|EPOLLET), and an op only stores its index in the fd's slot.
//  - The fd is made O_NONBLOCK once per epoch (callers rely on it), not per op.
//  - Reads and writes try the syscall before arming. A read that came back short
//    or with EAGAIN marks the fd drained, so the next read arms without a try.
//  - A registration is valid only for the epoch it was made under. Every close,
//    accept and connect that goes through bolt bumps the fd's global epoch, so a
//    reused fd number is registered again instead of waiting on a dead knote.
//    An fd closed with a raw close() must be released with async_io::forget_fd.
//  - Calls from any other thread, or on an engine polled by several threads,
//    use the per-op one-shot registration.
#pragma once

#include <atomic>
#include <cassert>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <sys/socket.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>

#include "bolt/api/core/accept_retirement.h"
#include "bolt/api/net/event_loop_limits.h"

namespace bolt::api::core::detail {

inline constexpr int kFastFds = AcceptRetirement::kReadinessFdLimit;

std::uint32_t fd_epoch(int fd) noexcept;
void fd_epoch_bump(int fd) noexcept;

enum : std::uint8_t {
    kFdRegRead = 1u << 0,
    kFdRegWrite = 1u << 1,
    kFdReadDrained = 1u << 2,
    kFdNotSocket = 1u << 3,
    kFdNonblock = 1u << 4,
};

struct FdFast {
    std::uint32_t epoch = 0;
    std::uint8_t flags = 0;
};

// Owner-thread state of one fd, reset when the fd's epoch moved on.
inline FdFast& fd_fast_current(FdFast* table, int fd) noexcept {
    assert(table != nullptr);
    assert(fd >= 0 && fd < kFastFds);
    FdFast& st = table[fd];
    const std::uint32_t epoch = fd_epoch(fd);
    if (st.epoch != epoch) {
        st.epoch = epoch;
        st.flags = 0;
    }
    return st;
}

// Callers (e.g. a drain loop after a readiness callback) rely on an fd being
// O_NONBLOCK once bolt has run an op on it; set it once per fd epoch.
inline void ensure_nonblocking(int fd, FdFast& st) noexcept {
    assert(fd >= 0);
    if ((st.flags & kFdNonblock) != 0) return;
    const int flags = ::fcntl(fd, F_GETFL, 0);
    if (flags >= 0 && (flags & O_NONBLOCK) == 0) (void)::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    st.flags |= kFdNonblock;
    assert((st.flags & kFdNonblock) != 0);
}

// The single thread allowed to take the fast path. Set by the first poll; a
// second polling thread turns the fast path off for the engine's lifetime.
class EngineOwner {
public:
    void note_poll() noexcept {
        const void* me = marker();
        const void* expected = nullptr;
        if (owner_.compare_exchange_strong(expected, me, std::memory_order_acq_rel)) return;
        if (expected != me) shared_.store(true, std::memory_order_release);
        assert(owner_.load(std::memory_order_relaxed) != nullptr);
    }
    bool on_owner() const noexcept {
        if (shared_.load(std::memory_order_acquire)) return false;
        return owner_.load(std::memory_order_acquire) == marker();
    }

private:
    static const void* marker() noexcept {
        static thread_local char token = 0;
        return &token;
    }
    std::atomic<const void*> owner_{nullptr};
    std::atomic<bool> shared_{false};
};

// Ops completed by a try on the submit path when the caller cannot take an
// inline result. Owner thread only; drained at the top of the next poll.
template <std::uint32_t Cap>
struct DeferredRing {
    std::uint32_t items[Cap] = {};
    std::uint32_t head = 0;
    std::uint32_t count = 0;

    bool push(std::uint32_t index) noexcept {
        assert(count <= Cap);
        if (count == Cap) return false;
        items[(head + count) % Cap] = index;
        ++count;
        return true;
    }
    bool pop(std::uint32_t* index) noexcept {
        assert(index != nullptr);
        if (count == 0) return false;
        *index = items[head];
        head = (head + 1) % Cap;
        --count;
        return true;
    }
};

enum class TryResult : std::uint8_t { done, would_block, not_socket };

// Non-blocking socket read/write regardless of the fd's O_NONBLOCK flag.
inline TryResult try_recv(int fd, void* buf, std::size_t len, ssize_t* out) noexcept {
    assert(out != nullptr);
    assert(fd >= 0);
    const ssize_t n = ::recv(fd, buf, len, MSG_DONTWAIT);
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return TryResult::would_block;
    if (n < 0 && errno == ENOTSOCK) return TryResult::not_socket;
    if (n < 0 && errno == EINTR) return TryResult::would_block;
    *out = n;
    return TryResult::done;
}

inline TryResult try_send(int fd, const void* buf, std::size_t len, ssize_t* out) noexcept {
    assert(out != nullptr);
    assert(fd >= 0);
#if defined(MSG_NOSIGNAL)
    const int flags = MSG_DONTWAIT | MSG_NOSIGNAL;
#else
    const int flags = MSG_DONTWAIT;
#endif
    const ssize_t n = ::send(fd, buf, len, flags);
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return TryResult::would_block;
    if (n < 0 && errno == ENOTSOCK) return TryResult::not_socket;
    if (n < 0 && errno == EINTR) return TryResult::would_block;
    *out = n;
    return TryResult::done;
}

inline TryResult try_recvfrom(int fd, void* buf, std::size_t len, struct sockaddr* src,
                              socklen_t* srclen, ssize_t* out) noexcept {
    assert(out != nullptr);
    assert(fd >= 0);
    const ssize_t n = ::recvfrom(fd, buf, len, MSG_DONTWAIT, src, srclen);
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return TryResult::would_block;
    if (n < 0 && errno == ENOTSOCK) return TryResult::not_socket;
    if (n < 0 && errno == EINTR) return TryResult::would_block;
    *out = n;
    return TryResult::done;
}

}  // namespace bolt::api::core::detail
