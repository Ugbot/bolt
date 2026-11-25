// io_cork.cpp — per-reactor write corking (G2CHK-1273); see io_cork.h.

#include "io_cork.h"

#include <cassert>
#include <cerrno>
#include <cstring>

#if !defined(_WIN32)
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>
#endif

namespace bolt::api::net {

using cork_limits::kCorkEntries;
using cork_limits::kCorkFds;

CorkFd* CorkTable::find(int fd) noexcept {
    CorkFd* free_slot = nullptr;
    for (uint32_t i = 0; i < kCorkFds; ++i) {
        if (fds_[i].fd == fd) return &fds_[i];
        if (fds_[i].fd < 0 && free_slot == nullptr) free_slot = &fds_[i];
    }
    if (free_slot != nullptr) {
        free_slot->fd = fd;
        free_slot->head = 0;
        free_slot->n = 0;
        free_slot->waiting = false;
    }
    return free_slot;
}

void CorkTable::recount() noexcept {
    uint32_t r = 0;
    for (uint32_t i = 0; i < kCorkFds; ++i) {
        if (fds_[i].fd >= 0 && fds_[i].n != 0 && !fds_[i].waiting) ++r;
    }
    ready_fds_ = r;
    assert(ready_fds_ <= kCorkFds);
}

bool CorkTable::add(int fd, const uint8_t* buf, size_t len, std::coroutine_handle<> h, ssize_t* result) noexcept {
    assert(fd >= 0 && h != nullptr && result != nullptr);
    CorkFd* c = find(fd);
    if (c == nullptr || c->n == kCorkEntries) return false;
    CorkEntry& e = c->e[(c->head + c->n) % kCorkEntries];
    e = CorkEntry{buf, len, 0, h, result};
    ++c->n;
    recount();
    return true;
}

void CorkTable::writable(int fd) noexcept {
    for (uint32_t i = 0; i < kCorkFds; ++i) {
        if (fds_[i].fd == fd) fds_[i].waiting = false;
    }
    recount();
}

uint32_t CorkTable::fail_all(CorkFd& c, std::coroutine_handle<>* done, uint32_t cap) noexcept {
    uint32_t k = 0;
    while (c.n != 0 && k < cap) {
        CorkEntry& e = c.e[c.head];
        *e.result = -1;
        done[k++] = e.handle;
        c.head = (c.head + 1) % kCorkEntries;
        --c.n;
    }
    if (c.n == 0) c.fd = -1;
    return k;
}

uint32_t CorkTable::drop(int fd, std::coroutine_handle<>* done, uint32_t cap) noexcept {
    uint32_t k = 0;
    for (uint32_t i = 0; i < kCorkFds; ++i) {
        if (fds_[i].fd == fd) k += fail_all(fds_[i], done + k, cap - k);
    }
    recount();
    return k;
}

uint32_t CorkTable::flush_one(core::async_io* io, CorkFd& c, std::coroutine_handle<>* done, uint32_t cap) noexcept {
    assert(c.fd >= 0 && c.n != 0 && !c.waiting);
#if defined(_WIN32)
    (void)io;
    return fail_all(c, done, cap);  // corking is POSIX-only; IOCP callers use write
#else
    struct iovec iov[kCorkEntries];
    for (uint32_t i = 0; i < c.n; ++i) {
        const CorkEntry& e = c.e[(c.head + i) % kCorkEntries];
        iov[i].iov_base = const_cast<uint8_t*>(e.buf + e.off);
        iov[i].iov_len = e.len - e.off;
    }
    struct msghdr m;
    std::memset(&m, 0, sizeof(m));
    m.msg_iov = iov;
    m.msg_iovlen = static_cast<decltype(m.msg_iovlen)>(c.n);
#  if defined(MSG_NOSIGNAL)
    ssize_t sent = ::sendmsg(c.fd, &m, MSG_DONTWAIT | MSG_NOSIGNAL);
#  else
    ssize_t sent = ::sendmsg(c.fd, &m, MSG_DONTWAIT);
#  endif
    if (sent < 0 && errno == ENOTSOCK) sent = ::writev(c.fd, iov, static_cast<int>(c.n));
    if (sent < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) return fail_all(c, done, cap);
    uint32_t k = 0;
    size_t left = sent > 0 ? static_cast<size_t>(sent) : 0;
    while (c.n != 0 && k < cap) {
        CorkEntry& e = c.e[c.head];
        const size_t want = e.len - e.off;
        if (left < want) { e.off += left; break; }
        left -= want;
        *e.result = static_cast<ssize_t>(e.len);
        done[k++] = e.handle;
        c.head = (c.head + 1) % kCorkEntries;
        --c.n;
    }
    if (c.n == 0) {
        c.fd = -1;
        return k;
    }
    // Short write or would-block: the rest waits for writability, in order.
    c.waiting = true;
    CorkTable* self = this;
    const int fd = c.fd;
    if (io->writable_async(fd, [self, fd](const core::io_event&) { self->writable(fd); }, nullptr) != 0) {
        c.waiting = false;  // no readiness hook (IOCP): retry next turn
    }
    return k;
#endif
}

uint32_t CorkTable::flush(core::async_io* io, std::coroutine_handle<>* done, uint32_t done_cap) noexcept {
    assert(io != nullptr && done != nullptr);
    uint32_t k = 0;
    for (uint32_t i = 0; i < kCorkFds && k < done_cap; ++i) {
        CorkFd& c = fds_[i];
        if (c.fd < 0 || c.n == 0 || c.waiting) continue;
        k += flush_one(io, c, done + k, done_cap - k);
    }
    recount();
    assert(k <= done_cap);
    return k;
}

}  // namespace bolt::api::net
