// io_cork.h — per-reactor write corking behind IODispatcher::write_corked
// (G2CHK-1273). Internal to bolt::reactor.
//
// Writes a reactor turn issues to one fd are queued, then flushed together at
// the end of the turn with one sendmsg / writev per fd. A writer resumes when
// ALL of its bytes are on the wire (result = its length), so a short write
// never reorders the stream: the rest waits for writability and the next turn.
// An error resumes every queued writer of that fd with -1.
#pragma once

#include <coroutine>
#include <cstddef>
#include <cstdint>
#include <sys/types.h>

#include "bolt/api/core/async_io.h"
#include "bolt/api/net/event_loop_limits.h"

namespace bolt::api::net {

namespace cork_limits {
inline constexpr uint32_t kCorkFds = limits::kCorkedFds;
inline constexpr uint32_t kCorkEntries = limits::kCorkedWritesPerFd;
}  // namespace cork_limits

struct CorkEntry {
    const uint8_t* buf = nullptr;
    size_t len = 0;
    size_t off = 0;
    std::coroutine_handle<> handle;
    ssize_t* result = nullptr;
};

struct CorkFd {
    int fd = -1;
    uint32_t head = 0;
    uint32_t n = 0;
    bool waiting = false;  // a writable wait is armed
    CorkEntry e[cork_limits::kCorkEntries];
};

class CorkTable {
public:
    // Queue one write; false when the fd's queue (or the fd table) is full.
    bool add(int fd, const uint8_t* buf, size_t len, std::coroutine_handle<> h, ssize_t* result) noexcept;
    // Corked writes that can go out now (queued, not waiting for writability).
    bool pending() const noexcept { return ready_fds_ != 0; }
    // Flush every ready fd once; finished writers are appended to `done`.
    uint32_t flush(core::async_io* io, std::coroutine_handle<>* done, uint32_t done_cap) noexcept;
    // Fail and remove every write queued on fd (before it is closed).
    uint32_t drop(int fd, std::coroutine_handle<>* done, uint32_t done_cap) noexcept;
    // A writable wait fired for fd.
    void writable(int fd) noexcept;

    static constexpr uint32_t kDoneMax = cork_limits::kCorkFds * cork_limits::kCorkEntries;

private:
    CorkFd* find(int fd) noexcept;
    uint32_t flush_one(core::async_io* io, CorkFd& c, std::coroutine_handle<>* done, uint32_t cap) noexcept;
    uint32_t fail_all(CorkFd& c, std::coroutine_handle<>* done, uint32_t cap) noexcept;
    void recount() noexcept;

    CorkFd fds_[cork_limits::kCorkFds];
    uint32_t ready_fds_ = 0;
};

}  // namespace bolt::api::net
