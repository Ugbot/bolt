// file.cpp — bolt async File I/O backends (STATION-84). See include/bolt/api/io/file.h.
//
// Per-OS async completion strategy:
//   • Windows  : native IOCP overlapped file I/O. Read/WriteFile carry the byte
//                offset in the OVERLAPPED; completions are reaped off a dedicated
//                completion port. Handles are opened FILE_FLAG_OVERLAPPED and
//                switched to FILE_SKIP_COMPLETION_PORT_ON_SUCCESS so a synchronous
//                completion queues NO packet and short-circuits await_ready()
//                (Seastar file-layer pattern, criterion #93). OVERLAPPED control
//                blocks are pooled (no per-op malloc), mirroring async_io_iocp.cpp.
//   • POSIX     : a regular file is always "ready" to epoll/kqueue, which therefore
//                cannot drive file readiness (the reason libuv offloads fs ops to a
//                thread pool). pread/pwrite/fsync are offloaded onto the reactor's
//                WorkerThreadPool. This is the AC-required io_uring-absent path and
//                is correct-by-construction on every kernel/sandbox.
//
// NOTE (io_uring native, deferred): on kernels where the reactor's io_uring probe
// (io_uring_io::usable()) reports usable, pread/pwrite/fsync SHOULD submit real
// IORING_OP_READ/WRITE/FSYNC SQEs on a ring instead of offloading — a synchronous
// CQE then feeds the same await_ready() short-circuit. That fast path is a
// follow-on; the offload path below is the correct fallback it falls back to, so
// shipping offload-only for POSIX is behaviourally complete (just not yet optimal).
//
// TODO (DMA / O_DIRECT, deferred per ticket): this layer does correct BUFFERED
// I/O. Seastar's align-down-offset / align-up-length + read-into-aligned-buffer
// with copy-fallback (file.cc dma_read_bulk/read_maybe_eof) maps to Windows
// FILE_FLAG_NO_BUFFERING sector rules and Linux O_DIRECT. Adding it means: align
// the offset down and the length up to the device sector size, read into a
// sector-aligned bounce buffer, then copy the requested slice out. Left as a
// follow-on so buffered correctness ships first rather than half-aligned code.

#include "bolt/api/io/file.h"
#include "bolt/api/core/worker_pool.h"

#include <atomic>
#include <cassert>
#include <cstring>
#include <mutex>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cerrno>
#include <ctime>
#endif

namespace bolt::api {
namespace io {

// -----------------------------------------------------------------------------
// handle validity
// -----------------------------------------------------------------------------

bool handle_valid(native_handle h) noexcept {
#ifdef _WIN32
    return h != kInvalidHandle && h != 0;  // reject INVALID_HANDLE_VALUE and NULL
#else
    return h >= 0;
#endif
}

// -----------------------------------------------------------------------------
// op control block + state machine
// -----------------------------------------------------------------------------

namespace {

enum class op_kind : uint8_t { read, write, fsync };

// Op lifecycle states for the completion↔continuation handshake. INIT is armed;
// exactly one of {completion, await_suspend} transitions it and the SECOND one
// observed drives the resume — so a completion racing await_suspend on another
// thread (Windows multi-poll / POSIX worker) resumes exactly once.
constexpr uint32_t kOpInit      = 0;
constexpr uint32_t kOpSuspended = 1;
constexpr uint32_t kOpCompleted = 2;

// Pooled op control block. On Windows OVERLAPPED MUST be the first member so
// CONTAINING_RECORD recovers the op from the completion's LPOVERLAPPED.
struct file_op {
#ifdef _WIN32
    OVERLAPPED ov;
#endif
    op_kind       kind = op_kind::read;
    native_handle handle = kInvalidHandle;
    void*         buf = nullptr;         // read destination
    const void*   cbuf = nullptr;        // write source
    size_t        size = 0;
    uint64_t      offset = 0;

    io_result     result{};

    // completion handshake (coroutine form)
    std::atomic<uint32_t>    state{kOpInit};
    std::coroutine_handle<>  cont{};
    io_result*               result_out = nullptr;

    // callback form
    bool          has_cb = false;
    completion    cb{};

    void reset() noexcept {
#ifdef _WIN32
        ZeroMemory(&ov, sizeof(ov));
#endif
        kind = op_kind::read;
        handle = kInvalidHandle;
        buf = nullptr;
        cbuf = nullptr;
        size = 0;
        offset = 0;
        result = io_result{};
        state.store(kOpInit, std::memory_order_relaxed);
        cont = std::coroutine_handle<>{};
        result_out = nullptr;
        has_cb = false;
        cb = nullptr;
    }
};

// launch() outcome.
enum class arm_status : uint8_t { sync_done, pending };

#ifdef _WIN32
// Distinguish wake/stop packets from real file completions (same trick iocp_io
// uses). Real file ops carry a heap file_op* via LPOVERLAPPED; these keys carry
// a NULL overlapped.
constexpr ULONG_PTR kWakeKey = 0xF11E0001;
constexpr ULONG_PTR kStopKey = 0xF11E0002;

// FILETIME (100 ns ticks since 1601-01-01) → ns since Unix epoch (1970-01-01).
int64_t filetime_to_unix_ns(const FILETIME& ft) noexcept {
    uint64_t t = (static_cast<uint64_t>(ft.dwHighDateTime) << 32) | ft.dwLowDateTime;
    constexpr uint64_t kEpochDelta = 116444736000000000ULL;  // 1601→1970 in 100ns
    if (t < kEpochDelta) return 0;
    return static_cast<int64_t>((t - kEpochDelta) * 100ULL);
}
#endif

}  // namespace

// -----------------------------------------------------------------------------
// file_io::impl
// -----------------------------------------------------------------------------

struct file_io::impl {
    config cfg;

    core::WorkerThreadPool* pool = nullptr;
    bool owns_pool = false;

#ifdef _WIN32
    HANDLE port = INVALID_HANDLE_VALUE;
    std::atomic<bool> running{false};
    std::atomic<bool> stop_requested{false};
    std::atomic<bool> wake_pending{false};
#endif

    // op free-list pool (bounded); lock only contended between submitters and the
    // completion reaper — far cheaper than malloc/free on every op.
    std::mutex             pool_mutex;
    std::vector<file_op*>  pool_free;
    static constexpr size_t kPoolCap = 256;

    // stats
    std::atomic<uint64_t> stat_reads{0};
    std::atomic<uint64_t> stat_writes{0};
    std::atomic<uint64_t> stat_fsyncs{0};
    std::atomic<uint64_t> stat_sync{0};
    std::atomic<uint64_t> stat_offloads{0};
    std::atomic<uint64_t> stat_errors{0};

    explicit impl(const config& c) noexcept : cfg(c) {}

    ~impl() {
        std::lock_guard<std::mutex> lock(pool_mutex);
        for (file_op* op : pool_free) delete op;
        pool_free.clear();
    }

    file_op* alloc_op() noexcept {
        {
            std::lock_guard<std::mutex> lock(pool_mutex);
            if (!pool_free.empty()) {
                file_op* op = pool_free.back();
                pool_free.pop_back();
                op->reset();
                return op;
            }
        }
        return new (std::nothrow) file_op();
    }

    void free_op(file_op* op) noexcept {
        if (op == nullptr) return;
        op->cb = nullptr;  // release captured state promptly
        {
            std::lock_guard<std::mutex> lock(pool_mutex);
            if (pool_free.size() < kPoolCap) {
                pool_free.push_back(op);
                return;
            }
        }
        delete op;
    }

    // Dispatch a completed op: invoke the callback (callback form) or run the
    // completion↔continuation handshake (coroutine form). op->result MUST be set.
    void complete_op(file_op* op) noexcept {
        assert(op != nullptr && "complete_op: null op");
        if (op->has_cb) {
            completion cb = std::move(op->cb);
            io_result r = op->result;
            free_op(op);
            if (cb) cb(r);
            return;
        }
        if (op->result_out) *op->result_out = op->result;
        const uint32_t prev = op->state.exchange(kOpCompleted, std::memory_order_acq_rel);
        if (prev == kOpSuspended) {
            std::coroutine_handle<> h = op->cont;
            free_op(op);
            if (h && !h.done()) h.resume();
        }
        // prev == kOpInit: await_op() will observe kOpCompleted and resume + free.
    }

    // Offload an op to the worker pool (POSIX read/write/fsync; Windows fsync).
    arm_status offload(file_op* op) noexcept {
        stat_offloads.fetch_add(1, std::memory_order_relaxed);
        impl* self = this;
        assert(pool != nullptr && "offload requires a worker pool");
        // submit_blocking enqueues onto the pool's blocking-work queue; we drop
        // the future — the lambda completes the op itself.
        (void)pool->submit_blocking([self, op]() noexcept { self->run_blocking(op); });
        return arm_status::pending;
    }

    // Perform the blocking syscall for `op` on a worker thread, then complete it.
    void run_blocking(file_op* op) noexcept {
        io_result r{};
#ifdef _WIN32
        // Windows offload is used only for fsync (read/write go native overlapped).
        HANDLE h = reinterpret_cast<HANDLE>(op->handle);
        if (op->kind == op_kind::fsync) {
            r = FlushFileBuffers(h) ? io_result{0, 0, false}
                                    : io_result{-1, static_cast<int32_t>(GetLastError()), false};
        } else {
            r = io_result{-1, static_cast<int32_t>(ERROR_INVALID_FUNCTION), false};
        }
#else
        int fd = static_cast<int>(op->handle);
        switch (op->kind) {
            case op_kind::read: {
                ssize_t n = ::pread(fd, op->buf, op->size, static_cast<off_t>(op->offset));
                r = (n >= 0) ? io_result{static_cast<int64_t>(n), 0, false}
                             : io_result{-1, errno, false};
                break;
            }
            case op_kind::write: {
                ssize_t n = ::pwrite(fd, op->cbuf, op->size, static_cast<off_t>(op->offset));
                r = (n >= 0) ? io_result{static_cast<int64_t>(n), 0, false}
                             : io_result{-1, errno, false};
                break;
            }
            case op_kind::fsync: {
                int rc = ::fsync(fd);
                r = (rc == 0) ? io_result{0, 0, false} : io_result{-1, errno, false};
                break;
            }
        }
#endif
        if (!r.ok()) stat_errors.fetch_add(1, std::memory_order_relaxed);
        op->result = r;
        complete_op(op);
    }

#ifdef _WIN32
    // Native overlapped read/write. Returns sync_done when the op completed
    // synchronously (result filled — no packet queued thanks to skip-on-success),
    // else pending (a completion packet will arrive on `port`).
    arm_status win_launch_rw(file_op* op, bool is_write) noexcept {
        ZeroMemory(&op->ov, sizeof(op->ov));
        op->ov.Offset     = static_cast<DWORD>(op->offset & 0xFFFFFFFFULL);
        op->ov.OffsetHigh = static_cast<DWORD>(op->offset >> 32);

        HANDLE h = reinterpret_cast<HANDLE>(op->handle);
        DWORD moved = 0;
        BOOL ok = is_write
            ? WriteFile(h, op->cbuf, static_cast<DWORD>(op->size), &moved, &op->ov)
            : ReadFile(h, op->buf, static_cast<DWORD>(op->size), &moved, &op->ov);

        if (ok) {
            // Synchronous completion. Handles carry FILE_SKIP_COMPLETION_PORT_ON_
            // SUCCESS (enforced at open), so NO packet is queued — finish inline.
            op->result = io_result{static_cast<int64_t>(moved), 0, /*sync*/ true};
            stat_sync.fetch_add(1, std::memory_order_relaxed);
            return arm_status::sync_done;
        }
        const DWORD e = GetLastError();
        if (e == ERROR_IO_PENDING) {
            return arm_status::pending;   // completion will arrive on the port
        }
        if (e == ERROR_HANDLE_EOF) {
            op->result = io_result{0, 0, true};  // read starting at/after EOF
            stat_sync.fetch_add(1, std::memory_order_relaxed);  // no packet, no suspend
            return arm_status::sync_done;
        }
        op->result = io_result{-1, static_cast<int32_t>(e), true};
        stat_errors.fetch_add(1, std::memory_order_relaxed);
        return arm_status::sync_done;
    }
#endif

    // Dispatch an armed op to its per-OS backend.
    arm_status launch(file_op* op) noexcept {
#ifdef _WIN32
        switch (op->kind) {
            case op_kind::read:  return win_launch_rw(op, /*is_write=*/false);
            case op_kind::write: return win_launch_rw(op, /*is_write=*/true);
            case op_kind::fsync: return offload(op);  // no overlapped FlushFileBuffers
        }
        return arm_status::sync_done;
#else
        return offload(op);  // POSIX: everything offloads (epoll can't ready a file)
#endif
    }
};

// -----------------------------------------------------------------------------
// file_io — construction
// -----------------------------------------------------------------------------

file_io::file_io(std::unique_ptr<impl> i) noexcept : impl_(std::move(i)) {}

file_io::~file_io() {
    if (!impl_) return;
    stop();
#ifdef _WIN32
    if (impl_->port != INVALID_HANDLE_VALUE) {
        CloseHandle(impl_->port);
        impl_->port = INVALID_HANDLE_VALUE;
    }
#endif
    if (impl_->owns_pool && impl_->pool) {
        impl_->pool->stop();
        delete impl_->pool;
        impl_->pool = nullptr;
    }
}

std::unique_ptr<file_io> file_io::create(const config& cfg) {
    auto im = std::make_unique<impl>(cfg);

#ifdef _WIN32
    im->port = CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, 0);
    if (im->port == nullptr) {
        im->port = INVALID_HANDLE_VALUE;  // creation failed; poll() will no-op
    }
#endif

    // Worker pool for the offload path. If the caller supplied one, use it; else
    // own a small pool (blocking workers do the fs offload; one main worker keeps
    // the pool valid). On a Windows-native-only workload it simply idles.
    if (cfg.offload_pool) {
        im->pool = cfg.offload_pool;
        im->owns_pool = false;
    } else {
        core::WorkerPoolConfig wc;
        wc.num_workers = 1;
        wc.blocking_workers = 2;
        auto* p = new core::WorkerThreadPool(wc);
        p->start();
        im->pool = p;
        im->owns_pool = true;
    }

    return std::unique_ptr<file_io>(new file_io(std::move(im)));
}

// -----------------------------------------------------------------------------
// metadata / path ops (synchronous)
// -----------------------------------------------------------------------------

native_handle file_io::open(const char* path, open_flags flags, int* out_err) noexcept {
    assert(path != nullptr && "file_io::open: null path");
    auto set_err = [out_err](int e) noexcept { if (out_err) *out_err = e; };
    set_err(0);

#ifdef _WIN32
    DWORD access = 0;
    if (has_flag(flags, open_flags::read_write))      access = GENERIC_READ | GENERIC_WRITE;
    else if (has_flag(flags, open_flags::write_only)) access = GENERIC_WRITE;
    else                                              access = GENERIC_READ;  // default/read_only

    DWORD disp;
    const bool creat = has_flag(flags, open_flags::create);
    const bool trunc = has_flag(flags, open_flags::truncate);
    const bool excl  = has_flag(flags, open_flags::exclusive);
    if (creat && excl)       disp = CREATE_NEW;
    else if (creat && trunc) disp = CREATE_ALWAYS;
    else if (creat)          disp = OPEN_ALWAYS;
    else if (trunc)          disp = TRUNCATE_EXISTING;
    else                     disp = OPEN_EXISTING;

    HANDLE h = CreateFileA(
        path, access,
        FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
        disp, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        set_err(static_cast<int>(GetLastError()));
        return kInvalidHandle;
    }

    // Associate with our completion port so overlapped read/write complete here.
    if (impl_->port != INVALID_HANDLE_VALUE) {
        if (CreateIoCompletionPort(h, impl_->port, reinterpret_cast<ULONG_PTR>(h), 0) == nullptr) {
            const int e = static_cast<int>(GetLastError());
            CloseHandle(h);
            set_err(e);
            return kInvalidHandle;
        }
    }

    // Require FILE_SKIP_COMPLETION_PORT_ON_SUCCESS: it makes a synchronous
    // ReadFile/WriteFile queue NO packet, which is what lets await_ready()
    // short-circuit a cached read safely. Without it a sync-TRUE return would
    // ALSO post a packet → double completion; rather than track skip per handle
    // we require it (Vista+, always present on the Win11 target). If it is
    // unavailable we fail the open honestly.
    if (!SetFileCompletionNotificationModes(h, FILE_SKIP_COMPLETION_PORT_ON_SUCCESS)) {
        const int e = static_cast<int>(GetLastError());
        CloseHandle(h);
        set_err(e);
        return kInvalidHandle;
    }

    return reinterpret_cast<native_handle>(h);
#else
    int oflags = 0;
    if (has_flag(flags, open_flags::read_write))      oflags |= O_RDWR;
    else if (has_flag(flags, open_flags::write_only)) oflags |= O_WRONLY;
    else                                              oflags |= O_RDONLY;
    if (has_flag(flags, open_flags::create))    oflags |= O_CREAT;
    if (has_flag(flags, open_flags::truncate))  oflags |= O_TRUNC;
    if (has_flag(flags, open_flags::exclusive)) oflags |= O_EXCL;

    int fd = ::open(path, oflags, 0644);
    if (fd < 0) {
        set_err(errno);
        return kInvalidHandle;
    }
    return static_cast<native_handle>(fd);
#endif
}

int file_io::close(native_handle h) noexcept {
    if (!handle_valid(h)) return -1;
#ifdef _WIN32
    return CloseHandle(reinterpret_cast<HANDLE>(h)) ? 0 : -1;
#else
    return ::close(static_cast<int>(h));
#endif
}

bool file_io::fstat(native_handle h, file_stat* out) noexcept {
    if (out == nullptr || !handle_valid(h)) return false;
#ifdef _WIN32
    BY_HANDLE_FILE_INFORMATION info;
    if (!GetFileInformationByHandle(reinterpret_cast<HANDLE>(h), &info)) return false;
    out->size = (static_cast<uint64_t>(info.nFileSizeHigh) << 32) | info.nFileSizeLow;
    out->mtime_ns = filetime_to_unix_ns(info.ftLastWriteTime);
    return true;
#else
    struct stat st;
    if (::fstat(static_cast<int>(h), &st) != 0) return false;
    out->size = static_cast<uint64_t>(st.st_size);
    out->mtime_ns = static_cast<int64_t>(st.st_mtime) * 1000000000LL;
    return true;
#endif
}

bool file_io::stat(const char* path, file_stat* out) noexcept {
    if (path == nullptr || out == nullptr) return false;
#ifdef _WIN32
    WIN32_FILE_ATTRIBUTE_DATA data;
    if (!GetFileAttributesExA(path, GetFileExInfoStandard, &data)) return false;
    out->size = (static_cast<uint64_t>(data.nFileSizeHigh) << 32) | data.nFileSizeLow;
    out->mtime_ns = filetime_to_unix_ns(data.ftLastWriteTime);
    return true;
#else
    struct stat st;
    if (::stat(path, &st) != 0) return false;
    out->size = static_cast<uint64_t>(st.st_size);
    out->mtime_ns = static_cast<int64_t>(st.st_mtime) * 1000000000LL;
    return true;
#endif
}

bool file_io::atomic_rename(const char* from, const char* to) noexcept {
    if (from == nullptr || to == nullptr) return false;
#ifdef _WIN32
    // MOVEFILE_REPLACE_EXISTING makes it atomic-replace on the same volume;
    // WRITE_THROUGH flushes the rename to disk before returning.
    return MoveFileExA(from, to, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    return ::rename(from, to) == 0;
#endif
}

bool file_io::fsync_dir(const char* path) noexcept {
    if (path == nullptr) return false;
#ifdef _WIN32
    (void)path;
    // NTFS journals directory metadata; there is no directory-handle flush
    // equivalent to POSIX dir fsync. Treated as a durable no-op.
    return true;
#else
    int fd = ::open(path, O_RDONLY);
    if (fd < 0) return false;
    int rc = ::fsync(fd);
    ::close(fd);
    return rc == 0;
#endif
}

// -----------------------------------------------------------------------------
// async data ops
// -----------------------------------------------------------------------------

bool file_io::submit_read(native_handle h, void* buf, size_t size, uint64_t offset,
                          io_result* out, void** out_op) noexcept {
    assert(out != nullptr && out_op != nullptr && "submit_read: null out");
    impl_->stat_reads.fetch_add(1, std::memory_order_relaxed);

    file_op* op = impl_->alloc_op();
    if (op == nullptr) {
        *out = io_result{-1, -1, true};
        impl_->stat_errors.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    op->kind = op_kind::read;
    op->handle = h;
    op->buf = buf;
    op->size = size;
    op->offset = offset;
    op->result_out = out;

    if (impl_->launch(op) == arm_status::sync_done) {
        *out = op->result;
        impl_->free_op(op);
        return true;
    }
    *out_op = op;
    return false;
}

bool file_io::submit_write(native_handle h, const void* buf, size_t size, uint64_t offset,
                           io_result* out, void** out_op) noexcept {
    assert(out != nullptr && out_op != nullptr && "submit_write: null out");
    impl_->stat_writes.fetch_add(1, std::memory_order_relaxed);

    file_op* op = impl_->alloc_op();
    if (op == nullptr) {
        *out = io_result{-1, -1, true};
        impl_->stat_errors.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    op->kind = op_kind::write;
    op->handle = h;
    op->cbuf = buf;
    op->size = size;
    op->offset = offset;
    op->result_out = out;

    if (impl_->launch(op) == arm_status::sync_done) {
        *out = op->result;
        impl_->free_op(op);
        return true;
    }
    *out_op = op;
    return false;
}

bool file_io::submit_fsync(native_handle h, io_result* out, void** out_op) noexcept {
    assert(out != nullptr && out_op != nullptr && "submit_fsync: null out");
    impl_->stat_fsyncs.fetch_add(1, std::memory_order_relaxed);

    file_op* op = impl_->alloc_op();
    if (op == nullptr) {
        *out = io_result{-1, -1, true};
        impl_->stat_errors.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    op->kind = op_kind::fsync;
    op->handle = h;
    op->result_out = out;

    if (impl_->launch(op) == arm_status::sync_done) {
        *out = op->result;
        impl_->free_op(op);
        return true;
    }
    *out_op = op;
    return false;
}

void file_io::await_op(void* p, std::coroutine_handle<> handle) noexcept {
    auto* op = static_cast<file_op*>(p);
    assert(op != nullptr && "await_op: null op");
    op->cont = handle;
    const uint32_t prev = op->state.exchange(kOpSuspended, std::memory_order_acq_rel);
    if (prev == kOpCompleted) {
        // Completion beat us; result_out already published. Resume + free here.
        impl_->free_op(op);
        if (handle && !handle.done()) handle.resume();
    }
    // prev == kOpInit: complete_op() will resume us when the op finishes.
}

void file_io::pread_async(native_handle h, void* buf, size_t size, uint64_t offset,
                          completion cb) noexcept {
    impl_->stat_reads.fetch_add(1, std::memory_order_relaxed);
    file_op* op = impl_->alloc_op();
    if (op == nullptr) {
        if (cb) cb(io_result{-1, -1, true});
        return;
    }
    op->kind = op_kind::read;
    op->handle = h;
    op->buf = buf;
    op->size = size;
    op->offset = offset;
    op->has_cb = true;
    op->cb = std::move(cb);

    if (impl_->launch(op) == arm_status::sync_done) {
        io_result r = op->result;
        completion c = std::move(op->cb);
        impl_->free_op(op);
        if (c) c(r);
    }
    // pending: complete_op() invokes the callback on completion.
}

void file_io::pwrite_async(native_handle h, const void* buf, size_t size, uint64_t offset,
                           completion cb) noexcept {
    impl_->stat_writes.fetch_add(1, std::memory_order_relaxed);
    file_op* op = impl_->alloc_op();
    if (op == nullptr) {
        if (cb) cb(io_result{-1, -1, true});
        return;
    }
    op->kind = op_kind::write;
    op->handle = h;
    op->cbuf = buf;
    op->size = size;
    op->offset = offset;
    op->has_cb = true;
    op->cb = std::move(cb);

    if (impl_->launch(op) == arm_status::sync_done) {
        io_result r = op->result;
        completion c = std::move(op->cb);
        impl_->free_op(op);
        if (c) c(r);
    }
}

void file_io::fsync_async(native_handle h, completion cb) noexcept {
    impl_->stat_fsyncs.fetch_add(1, std::memory_order_relaxed);
    file_op* op = impl_->alloc_op();
    if (op == nullptr) {
        if (cb) cb(io_result{-1, -1, true});
        return;
    }
    op->kind = op_kind::fsync;
    op->handle = h;
    op->has_cb = true;
    op->cb = std::move(cb);

    if (impl_->launch(op) == arm_status::sync_done) {
        io_result r = op->result;
        completion c = std::move(op->cb);
        impl_->free_op(op);
        if (c) c(r);
    }
}

// -----------------------------------------------------------------------------
// completion driving
// -----------------------------------------------------------------------------

int file_io::poll(uint32_t timeout_us) noexcept {
#ifdef _WIN32
    if (impl_->port == INVALID_HANDLE_VALUE) return 0;

    const uint32_t budget = impl_->cfg.max_events ? impl_->cfg.max_events : 1;
    int processed = 0;
    for (uint32_t i = 0; i < budget; ++i) {
        DWORD bytes = 0;
        ULONG_PTR key = 0;
        LPOVERLAPPED ov = nullptr;
        // First dequeue honours the caller's timeout; the rest drain non-blocking.
        const DWORD wait_ms = (i == 0) ? (timeout_us / 1000) : 0;
        BOOL ok = GetQueuedCompletionStatus(impl_->port, &bytes, &key, &ov, wait_ms);

        if (!ok && ov == nullptr) break;  // timeout / nothing ready

        if (ov == nullptr || key == kWakeKey || key == kStopKey) {
            if (key == kWakeKey) impl_->wake_pending.store(false, std::memory_order_release);
            // kStopKey: run() checks stop_requested; nothing else to do here.
            continue;
        }

        file_op* op = CONTAINING_RECORD(ov, file_op, ov);
        if (ok) {
            op->result = io_result{static_cast<int64_t>(bytes), 0, false};
        } else {
            const DWORD e = GetLastError();
            op->result = (e == ERROR_HANDLE_EOF)
                ? io_result{0, 0, false}
                : io_result{-1, static_cast<int32_t>(e), false};
            if (e != ERROR_HANDLE_EOF) impl_->stat_errors.fetch_add(1, std::memory_order_relaxed);
        }
        impl_->complete_op(op);
        ++processed;
    }
    return processed;
#else
    // POSIX: completions self-dispatch on worker threads; nothing to reap here.
    (void)timeout_us;
    return 0;
#endif
}

void file_io::run() noexcept {
#ifdef _WIN32
    if (impl_->running.exchange(true)) return;
    impl_->stop_requested.store(false, std::memory_order_release);
    const uint32_t to = impl_->cfg.max_events ? 1000u : 1000u;
    while (!impl_->stop_requested.load(std::memory_order_acquire)) {
        poll(to);
    }
    impl_->running.store(false, std::memory_order_release);
#else
    // Offload-only: run() has nothing to drive.
#endif
}

void file_io::stop() noexcept {
#ifdef _WIN32
    impl_->stop_requested.store(true, std::memory_order_release);
    if (impl_->port != INVALID_HANDLE_VALUE) {
        PostQueuedCompletionStatus(impl_->port, 0, kStopKey, nullptr);
    }
#endif
}

file_io::stats file_io::get_stats() const noexcept {
    stats s;
    s.reads  = impl_->stat_reads.load(std::memory_order_relaxed);
    s.writes = impl_->stat_writes.load(std::memory_order_relaxed);
    s.fsyncs = impl_->stat_fsyncs.load(std::memory_order_relaxed);
    s.sync_completions = impl_->stat_sync.load(std::memory_order_relaxed);
    s.offloads = impl_->stat_offloads.load(std::memory_order_relaxed);
    s.errors = impl_->stat_errors.load(std::memory_order_relaxed);
    return s;
}

// -----------------------------------------------------------------------------
// awaitable
// -----------------------------------------------------------------------------

bool file_awaitable::await_ready() noexcept {
    switch (kind_) {
        case kind::read:  return io_->submit_read(handle_, buf_, size_, offset_, &result_, &op_);
        case kind::write: return io_->submit_write(handle_, cbuf_, size_, offset_, &result_, &op_);
        case kind::fsync: return io_->submit_fsync(handle_, &result_, &op_);
    }
    return true;
}

void file_awaitable::await_suspend(std::coroutine_handle<> h) noexcept {
    io_->await_op(op_, h);
}

}  // namespace io
}  // namespace bolt::api
