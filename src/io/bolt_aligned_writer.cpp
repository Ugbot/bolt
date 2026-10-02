// bolt_aligned_writer.cpp — see bolt/io/bolt_aligned_writer.h.

#include "bolt/io/bolt_aligned_writer.h"

#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <malloc.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace bolt {
namespace io {

namespace {

constexpr intptr_t kClosedHandle = -1;

bool in_range(uint8_t v, uint8_t lo, uint8_t hi) noexcept { return v >= lo && v <= hi; }

// The recorded-unit rule of the wire container (frame_align_validate).
bool units_valid(const AlignedWriterConfig& c) noexcept {
    assert(wire::kFrameBufAlignLog2Lo <= wire::kFrameBufAlignLog2Hi);
    assert(wire::kFrameStripeAlignLog2Lo <= wire::kFrameStripeAlignLog2Hi);
    return in_range(c.buf_log2, wire::kFrameBufAlignLog2Lo, wire::kFrameBufAlignLog2Hi) &&
           in_range(c.chunk_log2, wire::kFrameChunkAlignLog2Lo, wire::kFrameChunkAlignLog2Hi) &&
           in_range(c.stripe_log2, wire::kFrameStripeAlignLog2Lo,
                    wire::kFrameStripeAlignLog2Hi) &&
           in_range(c.io_log2, wire::kFrameIoAlignLog2Lo, wire::kFrameIoAlignLog2Hi) &&
           c.buf_log2 <= c.chunk_log2 && c.chunk_log2 <= c.stripe_log2;
}

uint64_t align_up(uint64_t x, uint64_t unit) noexcept {
    assert(unit != 0 && (unit & (unit - 1)) == 0);
    assert(x <= (uint64_t(1) << 62));
    return (x + unit - 1) & ~(unit - 1);
}

// --- platform layer -------------------------------------------------------

#if defined(_WIN32)

uint8_t* alloc_aligned(size_t bytes, size_t align) noexcept {
    return static_cast<uint8_t*>(_aligned_malloc(bytes, align));
}
void free_aligned(uint8_t* p) noexcept { _aligned_free(p); }

intptr_t open_file(const char* path, bool want_direct, uint8_t* direct) noexcept {
    const DWORD flags = FILE_ATTRIBUTE_NORMAL | (want_direct ? FILE_FLAG_NO_BUFFERING : 0);
    HANDLE h = CreateFileA(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, flags, nullptr);
    *direct = (h != INVALID_HANDLE_VALUE && want_direct) ? 1 : 0;
    if (h == INVALID_HANDLE_VALUE && want_direct)
        h = CreateFileA(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL,
                        nullptr);
    return h == INVALID_HANDLE_VALUE ? kClosedHandle : reinterpret_cast<intptr_t>(h);
}

bool write_at(AlignedWriter* w, const uint8_t* p, uint64_t len, uint64_t off) noexcept {
    HANDLE h = reinterpret_cast<HANDLE>(w->handle);
    for (uint32_t tries = 0; len != 0 && tries < kAlignedWriterMaxWriteRetries; ++tries) {
        OVERLAPPED ov{};
        ov.Offset = static_cast<DWORD>(off);
        ov.OffsetHigh = static_cast<DWORD>(off >> 32);
        const DWORD chunk = len > (1u << 30) ? (1u << 30) : static_cast<DWORD>(len);
        DWORD done = 0;
        if (!WriteFile(h, p, chunk, &done, &ov) || done == 0) return false;
        p += done; off += done; len -= done;
    }
    return len == 0;
}

bool truncate_sync_close(AlignedWriter* w, uint64_t size, bool sync) noexcept {
    HANDLE h = reinterpret_cast<HANDLE>(w->handle);
    LARGE_INTEGER li;
    li.QuadPart = static_cast<LONGLONG>(size);
    bool ok = SetFilePointerEx(h, li, nullptr, FILE_BEGIN) && SetEndOfFile(h);
    if (ok && sync) ok = FlushFileBuffers(h) != 0;
    ok = (CloseHandle(h) != 0) && ok;
    w->handle = kClosedHandle;
    return ok;
}

void close_file(AlignedWriter* w) noexcept {
    CloseHandle(reinterpret_cast<HANDLE>(w->handle));
    w->handle = kClosedHandle;
}

#else  // POSIX

uint8_t* alloc_aligned(size_t bytes, size_t align) noexcept {
    void* p = nullptr;
    if (posix_memalign(&p, align, bytes) != 0) return nullptr;
    return static_cast<uint8_t*>(p);
}
void free_aligned(uint8_t* p) noexcept { std::free(p); }

intptr_t open_file(const char* path, bool want_direct, uint8_t* direct) noexcept {
    *direct = 0;
    int fd = -1;
#if defined(O_DIRECT)
    if (want_direct) {
        fd = ::open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC | O_DIRECT, 0644);
        if (fd >= 0) *direct = 1;
    }
#endif
    if (fd < 0) fd = ::open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
#if defined(__APPLE__)
    if (fd >= 0 && want_direct && ::fcntl(fd, F_NOCACHE, 1) == 0) *direct = 1;
#endif
    return fd;
}

#if defined(O_DIRECT)
// Some file systems accept O_DIRECT at open and refuse the first write.
bool drop_direct(AlignedWriter* w) noexcept {
    const int fd = static_cast<int>(w->handle);
    const int fl = ::fcntl(fd, F_GETFL);
    if (fl < 0 || ::fcntl(fd, F_SETFL, fl & ~O_DIRECT) != 0) return false;
    w->direct = 0;
    return true;
}
#endif

bool write_at(AlignedWriter* w, const uint8_t* p, uint64_t len, uint64_t off) noexcept {
    const int fd = static_cast<int>(w->handle);
    for (uint32_t tries = 0; len != 0 && tries < kAlignedWriterMaxWriteRetries; ++tries) {
        const ssize_t n = ::pwrite(fd, p, static_cast<size_t>(len), static_cast<off_t>(off));
        if (n > 0) { p += n; off += static_cast<uint64_t>(n); len -= static_cast<uint64_t>(n); continue; }
        if (n < 0 && errno == EINTR) continue;
#if defined(O_DIRECT)
        if (n < 0 && errno == EINVAL && w->direct && drop_direct(w)) continue;
#endif
        return false;
    }
    return len == 0;
}

bool truncate_sync_close(AlignedWriter* w, uint64_t size, bool sync) noexcept {
    const int fd = static_cast<int>(w->handle);
    bool ok = ::ftruncate(fd, static_cast<off_t>(size)) == 0;
    if (ok && sync) {
#if defined(__APPLE__)
        ok = ::fcntl(fd, F_FULLFSYNC) == 0 || ::fsync(fd) == 0;
#else
        ok = ::fsync(fd) == 0;
#endif
    }
    ok = (::close(fd) == 0) && ok;
    w->handle = kClosedHandle;
    return ok;
}

void close_file(AlignedWriter* w) noexcept {
    ::close(static_cast<int>(w->handle));
    w->handle = kClosedHandle;
}

#endif

// --- staging --------------------------------------------------------------

AlignedWriteStatus flush_full(AlignedWriter* w) noexcept {
    assert(w->staged == w->staging_cap);
    assert(w->flushed % w->staging_cap == 0);
    if (!write_at(w, w->staging, w->staging_cap, w->flushed)) return AlignedWriteStatus::kIoError;
    w->flushed += w->staging_cap;
    w->bytes_io += w->staging_cap;
    w->staged = 0;
    return AlignedWriteStatus::kOk;
}

// Copy `len` bytes (or zeros when src is null) into staging, flushing whole
// staging buffers as they fill.
AlignedWriteStatus stage(AlignedWriter* w, const uint8_t* src, uint64_t len) noexcept {
    assert(w->handle != kClosedHandle);
    assert(w->staged < w->staging_cap);
    const uint64_t max_iters = len / w->staging_cap + 2;
    for (uint64_t it = 0; len != 0 && it < max_iters; ++it) {
        const uint64_t room = w->staging_cap - w->staged;
        const uint64_t n = len < room ? len : room;
        if (src != nullptr) { std::memcpy(w->staging + w->staged, src, n); src += n; }
        else std::memset(w->staging + w->staged, 0, n);
        w->staged += n;
        len -= n;
        if (w->staged == w->staging_cap) {
            const AlignedWriteStatus s = flush_full(w);
            if (s != AlignedWriteStatus::kOk) return s;
        }
    }
    return len == 0 ? AlignedWriteStatus::kOk : AlignedWriteStatus::kIoError;
}

}  // namespace

AlignedWriterConfig aligned_writer_config_default() noexcept {
    AlignedWriterConfig c{};
    c.buf_log2 = static_cast<uint8_t>(bolt_ctz32(wire::kFrameBufAlign));
    c.chunk_log2 = static_cast<uint8_t>(bolt_ctz32(wire::kFrameChunkAlign));
    c.stripe_log2 = static_cast<uint8_t>(bolt_ctz32(wire::kFrameStripeAlign));
    c.io_log2 = static_cast<uint8_t>(bolt_ctz32(wire::kFrameIoAlign));
    c.direct_io = 1;
    c.staging_bytes = kAlignedWriterStagingDefault;
    assert(units_valid(c));
    return c;
}

AlignedWriteStatus aligned_writer_open(const char* path, const AlignedWriterConfig& cfg,
                                       AlignedWriter* w) noexcept {
    assert(path != nullptr && w != nullptr);
    std::memset(w, 0, sizeof(*w));
    w->handle = kClosedHandle;
    if (!units_valid(cfg)) return AlignedWriteStatus::kBadUnits;
    const uint32_t stripe = 1u << cfg.stripe_log2;
    const uint32_t io = 1u << cfg.io_log2;
    if (cfg.staging_bytes < kAlignedWriterStagingMin ||
        cfg.staging_bytes > kAlignedWriterStagingMax || cfg.staging_bytes % stripe != 0 ||
        cfg.staging_bytes % io != 0)
        return AlignedWriteStatus::kBadStaging;
    const size_t mem_align = io < 4096u ? 4096u : io;
    w->staging = alloc_aligned(cfg.staging_bytes, mem_align);
    if (w->staging == nullptr) return AlignedWriteStatus::kNoMemory;
    w->handle = open_file(path, cfg.direct_io != 0, &w->direct);
    if (w->handle == kClosedHandle) {
        free_aligned(w->staging);
        w->staging = nullptr;
        return AlignedWriteStatus::kOpenFailed;
    }
    w->staging_cap = cfg.staging_bytes;
    w->unit[0] = 1u << cfg.buf_log2;
    w->unit[1] = 1u << cfg.chunk_log2;
    w->unit[2] = stripe;
    w->io_unit = io;
    assert(w->staging_cap % w->unit[2] == 0);
    return AlignedWriteStatus::kOk;
}

AlignedWriteStatus aligned_writer_append(AlignedWriter* w, const void* data,
                                         size_t len) noexcept {
    assert(w != nullptr);
    assert(data != nullptr || len == 0);
    if (w->handle == kClosedHandle) return AlignedWriteStatus::kClosed;
    return stage(w, static_cast<const uint8_t*>(data), len);
}

AlignedWriteStatus aligned_writer_pad(AlignedWriter* w, AlignUnit unit) noexcept {
    assert(w != nullptr);
    assert(static_cast<uint32_t>(unit) <= 2u);
    if (w->handle == kClosedHandle) return AlignedWriteStatus::kClosed;
    const uint64_t pos = aligned_writer_pos(*w);
    const uint64_t pad = align_up(pos, w->unit[static_cast<uint32_t>(unit)]) - pos;
    return stage(w, nullptr, pad);
}

AlignedWriteStatus aligned_writer_page(AlignedWriter* w, const void* page, size_t len,
                                       AlignUnit unit, uint32_t seed,
                                       AlignedPage* out) noexcept {
    assert(w != nullptr && out != nullptr);
    assert(page != nullptr || len == 0);
    if (w->handle == kClosedHandle) return AlignedWriteStatus::kClosed;
    if (len > kAlignedWriterMaxPageBytes) return AlignedWriteStatus::kTooLarge;
    AlignedWriteStatus s = aligned_writer_pad(w, unit);
    if (s != AlignedWriteStatus::kOk) return s;
    const uint64_t u = w->unit[static_cast<uint32_t>(unit)];
    out->off = aligned_writer_pos(*w);
    out->len = len;
    out->padded_len = align_up(len, u);
    out->crc32c = page_crc32c(page, len, out->padded_len, seed);
    out->_pad = 0;
    s = stage(w, static_cast<const uint8_t*>(page), len);
    if (s != AlignedWriteStatus::kOk) return s;
    s = stage(w, nullptr, out->padded_len - len);
    assert(s != AlignedWriteStatus::kOk || aligned_writer_pos(*w) % u == 0);
    return s;
}

AlignedWriteStatus aligned_writer_finish(AlignedWriter* w, bool sync,
                                         uint64_t* out_size) noexcept {
    assert(w != nullptr);
    if (w->handle == kClosedHandle) return AlignedWriteStatus::kClosed;
    const uint64_t size = aligned_writer_pos(*w);
    const uint64_t tail = align_up(w->staged, w->io_unit);
    assert(tail <= w->staging_cap);
    bool ok = true;
    if (tail != 0) {
        std::memset(w->staging + w->staged, 0, tail - w->staged);
        ok = write_at(w, w->staging, tail, w->flushed);
        if (ok) w->bytes_io += tail;
    }
    ok = truncate_sync_close(w, size, sync) && ok;
    free_aligned(w->staging);
    w->staging = nullptr;
    if (out_size != nullptr) *out_size = size;
    return ok ? AlignedWriteStatus::kOk : AlignedWriteStatus::kIoError;
}

void aligned_writer_abort(AlignedWriter* w) noexcept {
    assert(w != nullptr);
    if (w->handle != kClosedHandle) close_file(w);
    if (w->staging != nullptr) free_aligned(w->staging);
    w->staging = nullptr;
    assert(w->handle == kClosedHandle);
}

}  // namespace io
}  // namespace bolt
