// bolt_file_io.h — file reads for the MSEG reader and cold path (B6; layout
// decision K9/K11, U38, §3.13): positional reads into caller slots (aligned
// for direct I/O), a read-only mmap handle with page-aligned views and
// readahead hints, a batched read over a bounded worker pool, and
// range_coalesce for sorted (off, len) lists.
//
// Direct I/O per platform: O_DIRECT (Linux), F_NOCACHE (macOS),
// FILE_FLAG_NO_BUFFERING (Windows). A direct read wants its buffer, offset
// and length on ReadFile::io_unit; file_pread_aligned enforces that, and
// io_alloc_aligned gives slots that satisfy it. When the platform or file
// system refuses direct I/O the file opens buffered (ReadFile::direct = 0).
//
// Readahead: madvise / PrefetchVirtualMemory on mappings, F_RDADVISE /
// posix_fadvise on descriptors. Hints are advisory; a platform without one
// answers kOk and does nothing.
//
// Batched reads run on a ReadPool of fixed worker threads (sized by the
// caller's scale profile); with 0 workers, or no pool, they run inline on
// the calling thread with no handoff. io_uring is not used yet: the Linux
// path is the same pool (follow-up, compile-time switch BOLT_IO_URING).
//
// Lives in bolt::io_file with the aligned writer, so bolt::io stays pure
// compute. Errors are FileIoStatus values; nothing throws.

#pragma once

#include <cstddef>
#include <cstdint>

#include "bolt/io/bolt_io_limits.h"

namespace bolt {
namespace io {

enum class FileIoStatus : uint8_t {
    kOk = 0,
    kOpenFailed,
    kIoError,
    kBadArg,
    kMisaligned,   // a direct read's buffer/offset/length not on io_unit
    kNoMemory,
    kTooLarge,     // past a cap in bolt_io_limits.h, or out of output space
    kUnsupported,  // the platform has no such operation
    kClosed,
};

// --- aligned slots -----------------------------------------------------------

/// `bytes` rounded up to `align` (a power of two); null on failure.
uint8_t* io_alloc_aligned(size_t bytes, size_t align) noexcept;
void io_free_aligned(uint8_t* p) noexcept;
/// The OS page (16 KiB on Apple Silicon, 4 KiB on x86).
uint32_t io_page_size() noexcept;
/// The mapping granularity: the page on POSIX, 64 KiB on Windows.
uint32_t io_map_granularity() noexcept;

// --- positional reads -----------------------------------------------------------

struct ReadFile {
    intptr_t handle;    // fd, or a Windows HANDLE; -1 when closed
    uint64_t size;      // file length at open
    uint32_t io_unit;   // kFrameIoAlign: direct-read alignment
    uint8_t  direct;    // 1 if open for direct I/O
    uint8_t  _pad[3];
};

FileIoStatus file_open_read(const char* path, bool want_direct, ReadFile* f) noexcept;
/// Idempotent.
void file_close(ReadFile* f) noexcept;

/// Read [off, off + len) into dst. *out_n is the bytes read: len, or fewer
/// only at end of file. Any alignment; on a direct file the caller must
/// already be aligned (use file_pread_aligned to have it checked).
FileIoStatus file_pread(const ReadFile& f, uint64_t off, void* dst, uint64_t len,
                        uint64_t* out_n) noexcept;

/// file_pread with dst, off and len required on f.io_unit (kMisaligned
/// otherwise), whether or not the file is direct.
FileIoStatus file_pread_aligned(const ReadFile& f, uint64_t off, void* dst, uint64_t len,
                                uint64_t* out_n) noexcept;

/// Ask the OS to start reading [off, off + len) (F_RDADVISE /
/// POSIX_FADV_WILLNEED). No-op on Windows.
FileIoStatus file_advise_willneed(const ReadFile& f, uint64_t off, uint64_t len) noexcept;

/// Drop the file's pages from the OS page cache (benchmarks and cold-path
/// tests): msync(MS_INVALIDATE) on a shared mapping (macOS), then
/// POSIX_FADV_DONTNEED where it exists. kUnsupported on Windows.
FileIoStatus file_evict_cache(const char* path) noexcept;

// --- mmap ------------------------------------------------------------------------

struct MappedFile {
    const uint8_t* base;    // page-aligned; null for an empty file
    uint64_t size;          // file length
    intptr_t handle;        // fd / file HANDLE
    intptr_t mapping;       // Windows mapping HANDLE; unused on POSIX
    uint32_t page;          // io_page_size()
    uint32_t _pad;
};

struct MappedView {
    const uint8_t* data;       // base + off
    uint64_t len;              // clamped to end of file
    const uint8_t* page_base;  // data rounded down to the page
    uint64_t page_len;         // [page_base, data + len) rounded up to the page
};

enum class Advice : uint8_t { kNormal = 0, kSequential, kRandom, kWillNeed, kDontNeed };

FileIoStatus mapped_file_open(const char* path, MappedFile* m) noexcept;
/// Idempotent.
void mapped_file_close(MappedFile* m) noexcept;

/// A view of [off, off + len) and its page-aligned extent. kBadArg past the
/// end of the file; len is clamped at it.
FileIoStatus mapped_view(const MappedFile& m, uint64_t off, uint64_t len,
                         MappedView* v) noexcept;

/// madvise over the page extent of [off, off + len) (PrefetchVirtualMemory
/// for kWillNeed on Windows; other advice is a no-op there).
FileIoStatus mapped_advise(const MappedFile& m, uint64_t off, uint64_t len,
                           Advice advice) noexcept;

/// Resident pages of the extent of [off, off + len) (mincore).
/// kUnsupported on Windows.
FileIoStatus mapped_resident_pages(const MappedFile& m, uint64_t off, uint64_t len,
                                   uint64_t* out_resident, uint64_t* out_pages) noexcept;

// --- range_coalesce ----------------------------------------------------------

struct IoRange {
    uint64_t off;
    uint64_t len;
};

struct CoalescedRange {
    uint64_t off;
    uint64_t len;      // <= max_request
    // [first, first + count) is the tightest index range holding every input
    // that overlaps this request; first and first + count - 1 overlap it. An
    // input inside the range may miss the request only when an earlier input
    // spans past it (nested inputs), so a scatter must clip to the request.
    uint32_t first;
    uint32_t count;
};

/// Merge `in` (sorted by off; overlaps allowed; empty ranges ignored) into
/// requests no longer than max_request, joining inputs whose gap is at most
/// max_gap. Requests come out sorted and disjoint and cover every input byte;
/// an input longer than max_request is split. kBadArg if `in` is unsorted or
/// max_request is 0 or not above max_gap; kTooLarge if out_cap is too small
/// (nothing is dropped silently; *out_n is then the count needed so far).
FileIoStatus range_coalesce(const IoRange* in, uint32_t n, uint64_t max_request,
                            uint64_t max_gap, CoalescedRange* out, uint32_t out_cap,
                            uint32_t* out_n) noexcept;

/// The output count range_coalesce would produce (sizing out).
uint32_t range_coalesce_count(const IoRange* in, uint32_t n, uint64_t max_request,
                              uint64_t max_gap) noexcept;

// --- batched reads -----------------------------------------------------------

struct ReadReq {
    uint64_t off;
    uint64_t len;
    uint8_t* dst;
    uint64_t got;          // out: bytes read
    FileIoStatus status;   // out
    uint8_t aligned;       // 1: checked as file_pread_aligned
    uint8_t _pad[6];
};

struct ReadPool;

/// `workers` threads (0..kReadPoolMaxWorkers). 0 gives a pool that runs
/// every batch inline on the caller.
FileIoStatus read_pool_create(uint32_t workers, ReadPool** out) noexcept;
/// Joins the workers. Null is a no-op.
void read_pool_destroy(ReadPool* pool) noexcept;
uint32_t read_pool_workers(const ReadPool* pool) noexcept;

/// Run every request; the caller works alongside the pool's workers. One
/// batch runs on a pool at a time (concurrent callers wait their turn).
/// Returns kOk if every request is kOk, else the first failed request's
/// status (each request carries its own). pool may be null (inline).
FileIoStatus file_read_batch(ReadPool* pool, const ReadFile& f, ReadReq* reqs,
                             uint32_t n) noexcept;

}  // namespace io
}  // namespace bolt
