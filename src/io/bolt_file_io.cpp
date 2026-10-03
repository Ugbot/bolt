// bolt_file_io.cpp — positional reads, aligned slots, mmap; see
// bolt/io/bolt_file_io.h. Batching and range_coalesce: bolt_read_batch.cpp.

#include "bolt/io/bolt_file_io.h"

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
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace bolt {
namespace io {

namespace {

constexpr intptr_t kClosedHandle = -1;

bool pow2(uint64_t x) noexcept { return x != 0 && (x & (x - 1)) == 0; }

uint64_t round_down(uint64_t x, uint64_t unit) noexcept {
    assert(pow2(unit));
    return x & ~(unit - 1);
}

uint64_t round_up(uint64_t x, uint64_t unit) noexcept {
    assert(pow2(unit));
    assert(x <= (uint64_t(1) << 62));
    return (x + unit - 1) & ~(unit - 1);
}

}  // namespace

// --- platform layer -------------------------------------------------------

#if defined(_WIN32)

uint8_t* io_alloc_aligned(size_t bytes, size_t align) noexcept {
    if (!pow2(align) || bytes == 0) return nullptr;
    return static_cast<uint8_t*>(_aligned_malloc(round_up(bytes, align), align));
}
void io_free_aligned(uint8_t* p) noexcept { _aligned_free(p); }

uint32_t io_page_size() noexcept {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwPageSize;
}

uint32_t io_map_granularity() noexcept {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwAllocationGranularity;
}

namespace {

intptr_t open_ro(const char* path, bool want_direct, uint8_t* direct) noexcept {
    const DWORD flags = FILE_ATTRIBUTE_NORMAL | (want_direct ? FILE_FLAG_NO_BUFFERING : 0);
    HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                           OPEN_EXISTING, flags, nullptr);
    *direct = (h != INVALID_HANDLE_VALUE && want_direct) ? 1 : 0;
    if (h == INVALID_HANDLE_VALUE && want_direct)
        h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    return h == INVALID_HANDLE_VALUE ? kClosedHandle : reinterpret_cast<intptr_t>(h);
}

bool file_size_of(intptr_t h, uint64_t* out) noexcept {
    LARGE_INTEGER li;
    if (!GetFileSizeEx(reinterpret_cast<HANDLE>(h), &li)) return false;
    *out = static_cast<uint64_t>(li.QuadPart);
    return true;
}

void close_handle(intptr_t h) noexcept { CloseHandle(reinterpret_cast<HANDLE>(h)); }

// One OS read call: bytes read (0 at EOF), or -1.
int64_t read_call(intptr_t h, uint8_t* p, uint64_t len, uint64_t off) noexcept {
    OVERLAPPED ov{};
    ov.Offset = static_cast<DWORD>(off);
    ov.OffsetHigh = static_cast<DWORD>(off >> 32);
    DWORD done = 0;
    if (!::ReadFile(reinterpret_cast<HANDLE>(h), p, static_cast<DWORD>(len), &done, &ov))
        return GetLastError() == ERROR_HANDLE_EOF ? 0 : -1;
    return static_cast<int64_t>(done);
}

bool retryable() noexcept { return false; }

}  // namespace

FileIoStatus file_advise_willneed(const ReadFile& f, uint64_t off, uint64_t len) noexcept {
    assert(f.handle != kClosedHandle);
    (void)off;
    (void)len;
    return FileIoStatus::kOk;
}

FileIoStatus file_evict_cache(const char* path) noexcept {
    assert(path != nullptr);
    return FileIoStatus::kUnsupported;
}

#else  // POSIX

uint8_t* io_alloc_aligned(size_t bytes, size_t align) noexcept {
    if (!pow2(align) || bytes == 0) return nullptr;
    if (align < sizeof(void*)) align = sizeof(void*);
    void* p = nullptr;
    if (posix_memalign(&p, align, round_up(bytes, align)) != 0) return nullptr;
    return static_cast<uint8_t*>(p);
}
void io_free_aligned(uint8_t* p) noexcept { std::free(p); }

uint32_t io_page_size() noexcept { return static_cast<uint32_t>(::sysconf(_SC_PAGESIZE)); }
uint32_t io_map_granularity() noexcept { return io_page_size(); }

namespace {

intptr_t open_ro(const char* path, bool want_direct, uint8_t* direct) noexcept {
    *direct = 0;
    int fd = -1;
#if defined(O_DIRECT)
    if (want_direct) {
        fd = ::open(path, O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (fd >= 0) *direct = 1;
    }
#endif
    if (fd < 0) fd = ::open(path, O_RDONLY | O_CLOEXEC);
#if defined(__APPLE__)
    if (fd >= 0 && want_direct && ::fcntl(fd, F_NOCACHE, 1) == 0) *direct = 1;
#endif
    return fd;
}

bool file_size_of(intptr_t h, uint64_t* out) noexcept {
    struct stat st;
    if (::fstat(static_cast<int>(h), &st) != 0) return false;
    *out = static_cast<uint64_t>(st.st_size);
    return true;
}

void close_handle(intptr_t h) noexcept { ::close(static_cast<int>(h)); }

int64_t read_call(intptr_t h, uint8_t* p, uint64_t len, uint64_t off) noexcept {
    const ssize_t n = ::pread(static_cast<int>(h), p, static_cast<size_t>(len),
                              static_cast<off_t>(off));
    return static_cast<int64_t>(n);
}

bool retryable() noexcept { return errno == EINTR || errno == EAGAIN; }

}  // namespace

FileIoStatus file_advise_willneed(const ReadFile& f, uint64_t off, uint64_t len) noexcept {
    assert(f.handle != kClosedHandle);
    if (off >= f.size || len == 0) return FileIoStatus::kOk;
    if (len > f.size - off) len = f.size - off;
    const uint64_t max_calls = len / kFileAdviseMaxCallBytes + 1;
    for (uint64_t c = 0; c < max_calls && len != 0; ++c) {
        const uint64_t n = len < kFileAdviseMaxCallBytes ? len : kFileAdviseMaxCallBytes;
#if defined(__APPLE__)
        struct radvisory ra;
        ra.ra_offset = static_cast<off_t>(off);
        ra.ra_count = static_cast<int>(n);
        (void)::fcntl(static_cast<int>(f.handle), F_RDADVISE, &ra);
#elif defined(POSIX_FADV_WILLNEED)
        (void)::posix_fadvise(static_cast<int>(f.handle), static_cast<off_t>(off),
                              static_cast<off_t>(n), POSIX_FADV_WILLNEED);
#endif
        off += n;
        len -= n;
    }
    assert(len == 0);
    return FileIoStatus::kOk;
}

FileIoStatus file_evict_cache(const char* path) noexcept {
    assert(path != nullptr);
    const int fd = ::open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return FileIoStatus::kOpenFailed;
    struct stat st;
    FileIoStatus s = FileIoStatus::kOk;
    if (::fstat(fd, &st) != 0) s = FileIoStatus::kIoError;
    if (s == FileIoStatus::kOk && st.st_size > 0) {
        const size_t n = static_cast<size_t>(st.st_size);
        void* p = ::mmap(nullptr, n, PROT_READ, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) {
            s = FileIoStatus::kIoError;
        } else {
            if (::msync(p, n, MS_INVALIDATE) != 0) s = FileIoStatus::kIoError;
            ::munmap(p, n);
        }
#if defined(POSIX_FADV_DONTNEED)
        (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
    }
    ::close(fd);
    return s;
}

#endif

// --- positional reads -------------------------------------------------------

FileIoStatus file_open_read(const char* path, bool want_direct, ReadFile* f) noexcept {
    assert(path != nullptr && f != nullptr);
    f->handle = kClosedHandle;
    f->size = 0;
    f->io_unit = wire::kFrameIoAlign;
    f->direct = 0;
    const intptr_t h = open_ro(path, want_direct, &f->direct);
    if (h == kClosedHandle) return FileIoStatus::kOpenFailed;
    if (!file_size_of(h, &f->size)) {
        close_handle(h);
        return FileIoStatus::kIoError;
    }
    f->handle = h;
    assert(pow2(f->io_unit));
    return FileIoStatus::kOk;
}

void file_close(ReadFile* f) noexcept {
    assert(f != nullptr);
    if (f->handle == kClosedHandle) return;
    close_handle(f->handle);
    f->handle = kClosedHandle;
    assert(f->handle == kClosedHandle);
}

FileIoStatus file_pread(const ReadFile& f, uint64_t off, void* dst, uint64_t len,
                        uint64_t* out_n) noexcept {
    assert(out_n != nullptr);
    assert(dst != nullptr || len == 0);
    *out_n = 0;
    if (f.handle == kClosedHandle) return FileIoStatus::kClosed;
    if (len > kFileReadMaxBytes) return FileIoStatus::kTooLarge;
    uint8_t* p = static_cast<uint8_t*>(dst);
    uint64_t done = 0;
    uint32_t retries = 0;
    // Each pass reads >= 1 byte or spends a retry, so len + retries bounds it.
    const uint64_t max_passes = len + kFileReadMaxRetries + 1;
    for (uint64_t pass = 0; done < len && pass < max_passes; ++pass) {
        uint64_t want = len - done;
        if (want > kFileReadMaxCallBytes) want = kFileReadMaxCallBytes;
        const int64_t n = read_call(f.handle, p + done, want, off + done);
        if (n == 0) break;  // end of file
        if (n < 0) {
            if (retryable() && ++retries < kFileReadMaxRetries) continue;
            return FileIoStatus::kIoError;
        }
        done += static_cast<uint64_t>(n);
    }
    assert(done <= len);
    if (done < len && off + done < f.size) return FileIoStatus::kIoError;
    *out_n = done;
    return FileIoStatus::kOk;
}

FileIoStatus file_pread_aligned(const ReadFile& f, uint64_t off, void* dst, uint64_t len,
                                uint64_t* out_n) noexcept {
    assert(out_n != nullptr);
    assert(pow2(f.io_unit));
    *out_n = 0;
    const uint64_t mask = f.io_unit - 1;
    if ((off & mask) != 0 || (len & mask) != 0 ||
        (reinterpret_cast<uintptr_t>(dst) & mask) != 0)
        return FileIoStatus::kMisaligned;
    return file_pread(f, off, dst, len, out_n);
}

// --- mmap ------------------------------------------------------------------

#if defined(_WIN32)

FileIoStatus mapped_file_open(const char* path, MappedFile* m) noexcept {
    assert(path != nullptr && m != nullptr);
    *m = MappedFile{nullptr, 0, kClosedHandle, kClosedHandle, io_page_size(), 0};
    uint8_t direct = 0;
    const intptr_t h = open_ro(path, false, &direct);
    if (h == kClosedHandle) return FileIoStatus::kOpenFailed;
    m->handle = h;
    if (!file_size_of(h, &m->size)) { mapped_file_close(m); return FileIoStatus::kIoError; }
    if (m->size == 0) return FileIoStatus::kOk;
    HANDLE mh = CreateFileMappingA(reinterpret_cast<HANDLE>(h), nullptr, PAGE_READONLY, 0, 0,
                                   nullptr);
    if (mh == nullptr) { mapped_file_close(m); return FileIoStatus::kIoError; }
    m->mapping = reinterpret_cast<intptr_t>(mh);
    m->base = static_cast<const uint8_t*>(MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0));
    if (m->base == nullptr) { mapped_file_close(m); return FileIoStatus::kIoError; }
    assert(m->base != nullptr);
    return FileIoStatus::kOk;
}

void mapped_file_release_handle(MappedFile* m) noexcept {
    assert(m != nullptr);
    if (m->mapping != kClosedHandle) close_handle(m->mapping);
    if (m->handle != kClosedHandle) close_handle(m->handle);
    m->mapping = kClosedHandle;
    m->handle = kClosedHandle;
    assert(m->handle == kClosedHandle);
}

void mapped_file_close(MappedFile* m) noexcept {
    assert(m != nullptr);
    if (m->base != nullptr) UnmapViewOfFile(m->base);
    if (m->mapping != kClosedHandle) close_handle(m->mapping);
    if (m->handle != kClosedHandle) close_handle(m->handle);
    m->base = nullptr;
    m->mapping = kClosedHandle;
    m->handle = kClosedHandle;
    assert(m->base == nullptr);
}

FileIoStatus mapped_advise(const MappedFile& m, uint64_t off, uint64_t len,
                           Advice advice) noexcept {
    MappedView v{};
    const FileIoStatus s = mapped_view(m, off, len, &v);
    if (s != FileIoStatus::kOk || v.page_len == 0 || advice != Advice::kWillNeed) return s;
    WIN32_MEMORY_RANGE_ENTRY e;
    e.VirtualAddress = const_cast<uint8_t*>(v.page_base);
    e.NumberOfBytes = static_cast<SIZE_T>(v.page_len);
    (void)PrefetchVirtualMemory(GetCurrentProcess(), 1, &e, 0);
    return FileIoStatus::kOk;
}

FileIoStatus mapped_resident_pages(const MappedFile& m, uint64_t off, uint64_t len,
                                   uint64_t* out_resident, uint64_t* out_pages) noexcept {
    assert(out_resident != nullptr && out_pages != nullptr);
    (void)m; (void)off; (void)len;
    *out_resident = 0;
    *out_pages = 0;
    return FileIoStatus::kUnsupported;
}

#else

FileIoStatus mapped_file_open(const char* path, MappedFile* m) noexcept {
    assert(path != nullptr && m != nullptr);
    *m = MappedFile{nullptr, 0, kClosedHandle, kClosedHandle, io_page_size(), 0};
    uint8_t direct = 0;
    const intptr_t h = open_ro(path, false, &direct);
    if (h == kClosedHandle) return FileIoStatus::kOpenFailed;
    m->handle = h;
    if (!file_size_of(h, &m->size)) { mapped_file_close(m); return FileIoStatus::kIoError; }
    if (m->size == 0) return FileIoStatus::kOk;
    void* p = ::mmap(nullptr, static_cast<size_t>(m->size), PROT_READ, MAP_SHARED,
                     static_cast<int>(h), 0);
    if (p == MAP_FAILED) { mapped_file_close(m); return FileIoStatus::kIoError; }
    m->base = static_cast<const uint8_t*>(p);
    assert((reinterpret_cast<uintptr_t>(m->base) & (m->page - 1)) == 0);
    return FileIoStatus::kOk;
}

void mapped_file_release_handle(MappedFile* m) noexcept {
    assert(m != nullptr);
    if (m->handle != kClosedHandle) close_handle(m->handle);
    m->handle = kClosedHandle;
    assert(m->handle == kClosedHandle);
}

void mapped_file_close(MappedFile* m) noexcept {
    assert(m != nullptr);
    if (m->base != nullptr) ::munmap(const_cast<uint8_t*>(m->base), static_cast<size_t>(m->size));
    if (m->handle != kClosedHandle) close_handle(m->handle);
    m->base = nullptr;
    m->handle = kClosedHandle;
    assert(m->base == nullptr);
}

FileIoStatus mapped_advise(const MappedFile& m, uint64_t off, uint64_t len,
                           Advice advice) noexcept {
    MappedView v{};
    const FileIoStatus s = mapped_view(m, off, len, &v);
    if (s != FileIoStatus::kOk || v.page_len == 0) return s;
    int a = MADV_NORMAL;
    switch (advice) {
        case Advice::kNormal: a = MADV_NORMAL; break;
        case Advice::kSequential: a = MADV_SEQUENTIAL; break;
        case Advice::kRandom: a = MADV_RANDOM; break;
        case Advice::kWillNeed: a = MADV_WILLNEED; break;
        case Advice::kDontNeed: a = MADV_DONTNEED; break;
    }
    if (::madvise(const_cast<uint8_t*>(v.page_base), static_cast<size_t>(v.page_len), a) != 0)
        return FileIoStatus::kIoError;
    return FileIoStatus::kOk;
}

FileIoStatus mapped_resident_pages(const MappedFile& m, uint64_t off, uint64_t len,
                                   uint64_t* out_resident, uint64_t* out_pages) noexcept {
    assert(out_resident != nullptr && out_pages != nullptr);
    *out_resident = 0;
    *out_pages = 0;
    MappedView v{};
    const FileIoStatus s = mapped_view(m, off, len, &v);
    if (s != FileIoStatus::kOk || v.page_len == 0) return s;
    constexpr uint64_t kVecPages = kMincoreChunkPages;
    unsigned char vec[kVecPages];
    const uint64_t pages = v.page_len / m.page;
    for (uint64_t p0 = 0; p0 < pages; p0 += kVecPages) {
        const uint64_t k = pages - p0 < kVecPages ? pages - p0 : kVecPages;
#if defined(__APPLE__)
        char* cv = reinterpret_cast<char*>(vec);
#else
        unsigned char* cv = vec;
#endif
        if (::mincore(const_cast<uint8_t*>(v.page_base) + p0 * m.page,
                      static_cast<size_t>(k * m.page), cv) != 0)
            return FileIoStatus::kIoError;
        for (uint64_t i = 0; i < k; ++i) *out_resident += vec[i] & 1u;
    }
    *out_pages = pages;
    assert(*out_resident <= *out_pages);
    return FileIoStatus::kOk;
}

#endif

FileIoStatus mapped_view(const MappedFile& m, uint64_t off, uint64_t len,
                         MappedView* v) noexcept {
    assert(v != nullptr);
    assert(pow2(m.page));
    *v = MappedView{nullptr, 0, nullptr, 0};
    if (off > m.size) return FileIoStatus::kBadArg;
    if (m.base == nullptr) return m.size == 0 ? FileIoStatus::kOk : FileIoStatus::kClosed;
    if (len > m.size - off) len = m.size - off;
    const uint64_t lo = round_down(off, m.page);
    const uint64_t hi = round_up(off + len, m.page);
    v->data = m.base + off;
    v->len = len;
    v->page_base = m.base + lo;
    v->page_len = hi - lo;
    assert(v->page_base <= v->data && v->page_base + v->page_len >= v->data + v->len);
    return FileIoStatus::kOk;
}

}  // namespace io
}  // namespace bolt
