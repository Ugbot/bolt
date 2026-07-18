#pragma once
// file.h — bolt async File I/O surface on the hoisted reactor (STATION-84).
//
// A first-class async file API so higher layers (the llm-station daemon's
// config / WAL / index work) consolidate off std::filesystem/fstream/stdio and
// onto bolt. This is the async/canonical companion to MarbleDB's SYNC io layer.
//
// Per-OS async completion (the hard part):
//   • Windows: native IOCP overlapped file I/O — files opened FILE_FLAG_OVERLAPPED,
//     ReadFile/WriteFile carry the byte offset in the OVERLAPPED, completions are
//     reaped off an I/O completion port. Synchronous completions short-circuit
//     via FILE_SKIP_COMPLETION_PORT_ON_SUCCESS (a cached read pays no scheduler
//     hop — Seastar's await_ready() short-circuit, criterion #93).
//   • Linux/macOS: a regular file is ALWAYS "ready" to epoll/kqueue, so those
//     backends cannot signal file readiness (this is exactly why libuv runs fs
//     ops on a thread pool). The POSIX path therefore offloads the blocking
//     pread/pwrite/fsync onto the reactor's WorkerThreadPool. (io_uring native
//     submission is the follow-on optimization for kernels where the reactor's
//     io_uring probe reports usable — see file.cpp NOTE.)
//
// OVERLAPPED / op control blocks are pooled (no per-op malloc on the hot path),
// mirroring async_io_iocp.cpp's iocp_op free-list.
//
// Two call forms per async op, mirroring net/io_dispatcher.h:
//   • callback:   pread_async(h, buf, n, off, [](io_result r){ ... });
//   • coroutine:  io_result r = co_await fio->pread(h, buf, n, off);
//
// Threading model. On Windows the completion is reaped by whichever thread calls
// poll()/run() on this engine; in production that is a reactor I/O thread. On the
// POSIX offload path the coroutine is resumed on the worker thread that ran the
// blocking syscall (libuv semantics). Neither path uses RAII-per-thread.
//
// DMA / O_DIRECT (Seastar align-down-offset / align-up-length, criterion #93) is
// deliberately deferred: this layer does correct BUFFERED I/O first. See the
// documented TODO in file.cpp for the O_DIRECT/FILE_FLAG_NO_BUFFERING follow-on.

#include <cstddef>
#include <cstdint>
#include <coroutine>
#include <functional>
#include <memory>

namespace bolt::api {
namespace core { class WorkerThreadPool; }
namespace io {

// -----------------------------------------------------------------------------
// Handles
// -----------------------------------------------------------------------------

// OS-agnostic file handle. On Windows this holds a HANDLE reinterpreted as an
// integer; on POSIX it holds an int fd widened to intptr. Never dereferenced by
// callers — pass it back to file_io methods.
using native_handle = std::intptr_t;

// Sentinel for "no handle" — equals Windows INVALID_HANDLE_VALUE ((HANDLE)-1)
// and is < 0 so POSIX fd validity (fd >= 0) also rejects it.
inline constexpr native_handle kInvalidHandle = -1;

// True iff `h` denotes a live handle. On Windows both -1 (INVALID_HANDLE_VALUE)
// and 0 (NULL) are invalid; on POSIX a valid fd is >= 0 (0 = stdin is not opened
// by this layer, so treating >=0 as valid is safe here).
bool handle_valid(native_handle h) noexcept;

// -----------------------------------------------------------------------------
// Open flags (bitmask)
// -----------------------------------------------------------------------------

enum class open_flags : uint32_t {
    none       = 0,
    read_only  = 1u << 0,
    write_only = 1u << 1,
    read_write = 1u << 2,
    create     = 1u << 3,  // create the file if it does not exist
    truncate   = 1u << 4,  // truncate to zero length on open
    exclusive  = 1u << 5,  // with `create`: fail if the file already exists
};

constexpr open_flags operator|(open_flags a, open_flags b) noexcept {
    return static_cast<open_flags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
constexpr open_flags operator&(open_flags a, open_flags b) noexcept {
    return static_cast<open_flags>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
constexpr bool has_flag(open_flags set, open_flags f) noexcept {
    return (static_cast<uint32_t>(set) & static_cast<uint32_t>(f)) != 0;
}

// -----------------------------------------------------------------------------
// Results
// -----------------------------------------------------------------------------

// Outcome of an async read/write/fsync op.
struct io_result {
    int64_t bytes = -1;          // bytes transferred (>= 0), or -1 on error
    int32_t error = 0;           // 0 on success, else OS error (GetLastError/errno)
    bool    completed_sync = false;  // true iff the op completed without suspending
                                     // (IOCP FILE_SKIP_COMPLETION_PORT_ON_SUCCESS)
    bool ok() const noexcept { return bytes >= 0 && error == 0; }
};

// File metadata (subset the daemon needs: size + mtime).
struct file_stat {
    uint64_t size = 0;       // logical size in bytes
    int64_t  mtime_ns = 0;   // last-modified, ns since Unix epoch
};

// Completion callback for the callback-form async ops.
using completion = std::function<void(io_result)>;

// -----------------------------------------------------------------------------
// Awaitables (coroutine form)
// -----------------------------------------------------------------------------

class file_io;

// Common machinery for the three async ops. await_ready() SUBMITS the op and
// returns true when it already completed synchronously (Seastar short-circuit):
// a ready result then pays no suspension. Otherwise it returns false and
// await_suspend() attaches the continuation to the pending op.
class file_awaitable {
public:
    bool await_ready() noexcept;
    void await_suspend(std::coroutine_handle<> h) noexcept;
    io_result await_resume() const noexcept { return result_; }

protected:
    // Kind of op; drives which submit_* the engine performs in await_ready().
    enum class kind : uint8_t { read, write, fsync };

    file_awaitable(file_io* io, kind k, native_handle h,
                   void* buf, const void* cbuf, size_t size, uint64_t offset) noexcept
        : io_(io), kind_(k), handle_(h), buf_(buf), cbuf_(cbuf),
          size_(size), offset_(offset) {}

    file_io*      io_;
    kind          kind_;
    native_handle handle_;
    void*         buf_;
    const void*   cbuf_;
    size_t        size_;
    uint64_t      offset_;
    io_result     result_{};
    void*         op_ = nullptr;   // pooled op token when the op is pending
};

class read_awaitable : public file_awaitable {
public:
    read_awaitable(file_io* io, native_handle h, void* buf, size_t n, uint64_t off) noexcept
        : file_awaitable(io, kind::read, h, buf, nullptr, n, off) {}
};

class write_awaitable : public file_awaitable {
public:
    write_awaitable(file_io* io, native_handle h, const void* buf, size_t n, uint64_t off) noexcept
        : file_awaitable(io, kind::write, h, nullptr, buf, n, off) {}
};

class fsync_awaitable : public file_awaitable {
public:
    fsync_awaitable(file_io* io, native_handle h) noexcept
        : file_awaitable(io, kind::fsync, h, nullptr, nullptr, 0, 0) {}
};

// -----------------------------------------------------------------------------
// file_io — the async file engine
// -----------------------------------------------------------------------------

class file_io {
public:
    struct config {
        // Worker pool for the POSIX/offload path (POSIX pread/pwrite; fsync on all
        // platforms). If null, file_io creates and owns a small pool. On a
        // Windows-native-only workload the owned pool's blocking workers idle.
        core::WorkerThreadPool* offload_pool = nullptr;
        uint32_t max_events = 256;   // completions drained per poll() call (Windows)
    };

    static std::unique_ptr<file_io> create(const config& cfg = {});

    ~file_io();

    file_io(const file_io&)            = delete;
    file_io& operator=(const file_io&) = delete;

    // ---- metadata / path ops (synchronous fast syscalls) --------------------

    // Open `path`. Returns a live handle, or kInvalidHandle on failure (then
    // *out_err, if non-null, holds the OS error). On Windows the handle is
    // opened FILE_FLAG_OVERLAPPED and registered with this engine's completion
    // port + FILE_SKIP_COMPLETION_PORT_ON_SUCCESS.
    native_handle open(const char* path, open_flags flags, int* out_err = nullptr) noexcept;

    // Close a handle opened by open(). Returns 0 on success, -1 on error.
    int close(native_handle h) noexcept;

    // Stat a path. Returns true and fills *out on success.
    bool stat(const char* path, file_stat* out) noexcept;

    // Stat an open handle.
    bool fstat(native_handle h, file_stat* out) noexcept;

    // Atomically rename `from`→`to`, replacing an existing `to` (same volume).
    // Windows: MoveFileEx(MOVEFILE_REPLACE_EXISTING). POSIX: rename(2).
    bool atomic_rename(const char* from, const char* to) noexcept;

    // fsync the directory containing `path` so a rename/create is durable. POSIX
    // opens the dir + fsync; Windows metadata is journaled by NTFS so this is a
    // documented no-op returning true.
    bool fsync_dir(const char* path) noexcept;

    // ---- async data ops: callback form --------------------------------------

    void pread_async(native_handle h, void* buf, size_t size, uint64_t offset,
                     completion cb) noexcept;
    void pwrite_async(native_handle h, const void* buf, size_t size, uint64_t offset,
                      completion cb) noexcept;
    void fsync_async(native_handle h, completion cb) noexcept;

    // ---- async data ops: coroutine form -------------------------------------

    read_awaitable  pread(native_handle h, void* buf, size_t size, uint64_t offset) noexcept {
        return read_awaitable(this, h, buf, size, offset);
    }
    write_awaitable pwrite(native_handle h, const void* buf, size_t size, uint64_t offset) noexcept {
        return write_awaitable(this, h, buf, size, offset);
    }
    fsync_awaitable fsync(native_handle h) noexcept {
        return fsync_awaitable(this, h);
    }

    // ---- completion driving (Windows native port; no-op on offload-only) ----

    // Reap up to config.max_events completions, invoking callbacks / resuming
    // coroutines. `timeout_us` bounds the wait. Returns events processed, or -1.
    // On a pure-offload platform there is no port to poll; returns 0.
    int poll(uint32_t timeout_us = 0) noexcept;

    // Run poll() until stop(); stop() wakes a blocked poll().
    void run() noexcept;
    void stop() noexcept;

    // ---- counters -----------------------------------------------------------

    struct stats {
        uint64_t reads = 0;
        uint64_t writes = 0;
        uint64_t fsyncs = 0;
        uint64_t sync_completions = 0;  // ops that took the await_ready short-circuit
        uint64_t offloads = 0;          // ops routed to the worker pool
        uint64_t errors = 0;
    };
    stats get_stats() const noexcept;

    // ---- awaitable plumbing (advanced; used by file_awaitable) --------------
    // Submit an op. Returns true if it completed synchronously (fills *out, no
    // pending op). Returns false if pending (*out_op receives an opaque token;
    // the op publishes its result into *out on completion).
    bool submit_read(native_handle h, void* buf, size_t size, uint64_t offset,
                     io_result* out, void** out_op) noexcept;
    bool submit_write(native_handle h, const void* buf, size_t size, uint64_t offset,
                      io_result* out, void** out_op) noexcept;
    bool submit_fsync(native_handle h, io_result* out, void** out_op) noexcept;

    // Attach a coroutine continuation to a pending op (from await_suspend).
    void await_op(void* op, std::coroutine_handle<> h) noexcept;

private:
    struct impl;
    std::unique_ptr<impl> impl_;

    explicit file_io(std::unique_ptr<impl> i) noexcept;
};

}  // namespace io
}  // namespace bolt::api
