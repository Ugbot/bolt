// Fixed writable mappings. Included in bolt_file_io.cpp; platform headers stay private.
namespace {
void writable_empty(WritableMappedFile* m) noexcept {
    assert(m != nullptr);
    *m = {nullptr, 0, kClosedHandle, kClosedHandle, io_page_size(), io_map_granularity()};
    assert(m->base == nullptr);
}

#if !defined(_WIN32)
FileIoStatus writable_allocate(int fd, uint64_t bytes) noexcept {
    assert(fd >= 0);
    assert(bytes != 0 && bytes <= static_cast<uint64_t>(INT64_MAX));
#if defined(__APPLE__)
    fstore_t request{};
    request.fst_flags = F_ALLOCATECONTIG;
    request.fst_posmode = F_PEOFPOSMODE;
    request.fst_length = static_cast<off_t>(bytes);
    if (::fcntl(fd, F_PREALLOCATE, &request) == -1) {
        request.fst_flags = F_ALLOCATEALL;
        if (::fcntl(fd, F_PREALLOCATE, &request) == -1)
            return errno == ENOTSUP ? FileIoStatus::kUnsupported : FileIoStatus::kIoError;
    }
    if (request.fst_bytesalloc < static_cast<off_t>(bytes)) return FileIoStatus::kIoError;
#else
    const int error = ::posix_fallocate(fd, 0, static_cast<off_t>(bytes));
    if (error != 0) return error == ENOSYS || error == EOPNOTSUPP ?
        FileIoStatus::kUnsupported : FileIoStatus::kIoError;
#endif
    return ::ftruncate(fd, static_cast<off_t>(bytes)) == 0 ?
        FileIoStatus::kOk : FileIoStatus::kIoError;
}
#endif
} // namespace

void mapped_write_close(WritableMappedFile* m) noexcept {
    if (m == nullptr) return;
    assert(m->base == nullptr || m->size != 0);
#if defined(_WIN32)
    if (m->base) (void)UnmapViewOfFile(m->base);
    if (m->mapping != kClosedHandle) (void)CloseHandle(reinterpret_cast<HANDLE>(m->mapping));
    if (m->handle != kClosedHandle) (void)CloseHandle(reinterpret_cast<HANDLE>(m->handle));
#else
    if (m->base) (void)::munmap(m->base, static_cast<size_t>(m->size));
    if (m->handle != kClosedHandle) (void)::close(static_cast<int>(m->handle));
#endif
    writable_empty(m);
    assert(m->handle == kClosedHandle);
}

FileIoStatus mapped_write_create(const char* path, uint64_t bytes, bool temporary,
                                 bool prefault, WritableMappedFile* out) noexcept {
    if (out == nullptr) return FileIoStatus::kBadArg;
    writable_empty(out);
    if (path == nullptr || path[0] == '\0') return FileIoStatus::kBadArg;
    const uint32_t unit = out->page > out->granularity ? out->page : out->granularity;
    if (!pow2(unit) || bytes == 0 || bytes % unit || bytes > kFileReadMaxBytes ||
        bytes > SIZE_MAX || bytes > static_cast<uint64_t>(INT64_MAX)) return FileIoStatus::kBadArg;
    assert(bytes % out->page == 0);
    FileIoStatus result = FileIoStatus::kIoError;
#if defined(_WIN32)
    const DWORD flags = FILE_ATTRIBUTE_NORMAL | (temporary ? FILE_FLAG_DELETE_ON_CLOSE : 0);
    HANDLE file = CreateFileA(path, GENERIC_READ | GENERIC_WRITE | DELETE,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, CREATE_NEW, flags, nullptr);
    if (file == INVALID_HANDLE_VALUE) return FileIoStatus::kOpenFailed;
    out->handle = reinterpret_cast<intptr_t>(file);
    FILE_ALLOCATION_INFO allocation{};
    allocation.AllocationSize.QuadPart = static_cast<LONGLONG>(bytes);
    FILE_END_OF_FILE_INFO extent{};
    extent.EndOfFile.QuadPart = static_cast<LONGLONG>(bytes);
    if (SetFileInformationByHandle(file, FileAllocationInfo, &allocation, sizeof(allocation)) &&
        SetFileInformationByHandle(file, FileEndOfFileInfo, &extent, sizeof(extent))) {
        HANDLE mapping = CreateFileMappingA(file, nullptr, PAGE_READWRITE, 0, 0, nullptr);
        if (mapping != nullptr) {
            out->mapping = reinterpret_cast<intptr_t>(mapping);
            out->base = static_cast<uint8_t*>(MapViewOfFile(mapping, FILE_MAP_WRITE | FILE_MAP_READ,
                                                           0, 0, static_cast<SIZE_T>(bytes)));
        }
    } else if (GetLastError() == ERROR_NOT_SUPPORTED || GetLastError() == ERROR_INVALID_FUNCTION) {
        result = FileIoStatus::kUnsupported;
    }
#else
    const int fd = ::open(path, O_RDWR | O_CREAT | O_EXCL, 0600);
    if (fd < 0) return FileIoStatus::kOpenFailed;
    out->handle = fd;
    result = writable_allocate(fd, bytes);
    if (result == FileIoStatus::kOk) {
        result = FileIoStatus::kIoError;
        void* base = ::mmap(nullptr, static_cast<size_t>(bytes), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (base != MAP_FAILED) out->base = static_cast<uint8_t*>(base);
    }
#endif
    out->size = bytes;
    if (!out->base) {
        mapped_write_close(out);
#if defined(_WIN32)
        (void)DeleteFileA(path);
#else
        (void)::unlink(path);
#endif
        return result;
    }
#if !defined(_WIN32)
    if (temporary && ::unlink(path) != 0) {
        mapped_write_close(out);
        (void)::unlink(path);
        return FileIoStatus::kIoError;
    }
#endif
    if (prefault) {
        volatile uint8_t* pages = out->base;
        for (uint64_t off = 0; off < bytes; off += out->page) pages[off] = pages[off];
    }
    assert(out->base != nullptr && out->size == bytes);
    return FileIoStatus::kOk;
}

FileIoStatus mapped_write_flush(WritableMappedFile* m, uint64_t off, uint64_t len) noexcept {
    if (m == nullptr || m->handle == kClosedHandle || m->base == nullptr) return FileIoStatus::kClosed;
    assert(m->size != 0);
    assert(pow2(m->page));
    if (off > m->size || len > m->size - off) return FileIoStatus::kBadArg;
    if (len == 0) return FileIoStatus::kOk;
    const uint64_t lo = off & ~(uint64_t{m->page} - 1);
    const uint64_t hi = (off + len + m->page - 1) & ~(uint64_t{m->page} - 1);
#if defined(_WIN32)
    if (!FlushViewOfFile(m->base + lo, static_cast<SIZE_T>(hi - lo)) ||
        !FlushFileBuffers(reinterpret_cast<HANDLE>(m->handle))) return FileIoStatus::kIoError;
#else
    if (::msync(m->base + lo, static_cast<size_t>(hi - lo), MS_SYNC) != 0 ||
        ::fsync(static_cast<int>(m->handle)) != 0) return FileIoStatus::kIoError;
#endif
    return FileIoStatus::kOk;
}
