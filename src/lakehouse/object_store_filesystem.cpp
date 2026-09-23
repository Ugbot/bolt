// bolt/lakehouse/object_store_filesystem.cpp — local-filesystem ObjectStore.
// Uses portable C stdio + std::filesystem for listing. No POSIX-only headers.

#include "bolt/lakehouse/object_store.h"

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <system_error>

#include "bolt/bolt_port.h"   // windows.h / unistd.h, bolt_getpid

#if !defined(_WIN32)
#include <fcntl.h>
#endif

namespace bolt {
namespace lakehouse {

namespace {

// Join root + '/' + key into out (cap bytes). Returns false on overflow.
bool join_path(const char* root, const char* key, char* out,
               uint32_t cap) noexcept {
    assert(root != nullptr);
    assert(key != nullptr);
    const size_t rl = std::strlen(root);
    const size_t kl = std::strlen(key);
    if (rl + 1u + kl + 1u > cap) return false;
    std::memcpy(out, root, rl);
    uint32_t p = static_cast<uint32_t>(rl);
    if (rl > 0 && root[rl - 1] != '/' && root[rl - 1] != '\\') out[p++] = '/';
    std::memcpy(out + p, key, kl);
    p += static_cast<uint32_t>(kl);
    out[p] = '\0';
    return true;
}

int fs_get(void* impl, const char* key, Arena* arena,
           const uint8_t** out_data, uint64_t* out_len) noexcept {
    assert(impl != nullptr);
    assert(out_data != nullptr && out_len != nullptr);
    FilesystemObjectStore* fs = static_cast<FilesystemObjectStore*>(impl);
    char path[kOsMaxRoot + kOsMaxKey + 2u];
    if (!join_path(fs->root, key, path, sizeof(path))) return kOsBadArg;
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) return kOsNotFound;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (n < 0) { std::fclose(f); return kOsIoError; }
    uint8_t* buf = arena->allocate_array<uint8_t>(n == 0 ? 1u :
                                                  static_cast<size_t>(n));
    if (buf == nullptr) { std::fclose(f); return kOsIoError; }
    const size_t got = std::fread(buf, 1, static_cast<size_t>(n), f);
    std::fclose(f);
    if (got != static_cast<size_t>(n)) return kOsIoError;
    *out_data = buf;
    *out_len = static_cast<uint64_t>(n);
    return kOsOk;
}

int fs_put(void* impl, const char* key, const uint8_t* data,
           uint64_t len) noexcept {
    assert(impl != nullptr);
    assert(data != nullptr || len == 0);
    FilesystemObjectStore* fs = static_cast<FilesystemObjectStore*>(impl);
    char path[kOsMaxRoot + kOsMaxKey + 2u];
    if (!join_path(fs->root, key, path, sizeof(path))) return kOsBadArg;
    // Ensure the parent directory exists.
    std::error_code ec;
    std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path(), ec);
    }
    std::FILE* f = std::fopen(path, "wb");
    if (f == nullptr) return kOsIoError;
    const size_t wrote = len == 0 ? 0u
                                  : std::fwrite(data, 1, static_cast<size_t>(len), f);
    std::fclose(f);
    if (wrote != static_cast<size_t>(len)) return kOsIoError;
    return kOsOk;
}

int fs_list(void* impl, const char* prefix, ObjectEntry* out, uint32_t cap,
            uint32_t* out_n) noexcept {
    assert(impl != nullptr);
    assert(out != nullptr && out_n != nullptr);
    FilesystemObjectStore* fs = static_cast<FilesystemObjectStore*>(impl);
    *out_n = 0;
    std::error_code ec;
    const size_t root_len = std::strlen(fs->root);
    // Start the walk at the deepest directory the prefix names, never at the
    // store root. The root is often a SHARED directory (the Delta sink splits
    // "<dir>/<table>" into root=<dir>, so root can be the OS temp dir), and:
    //   (1) walking unrelated subtrees is O(everything in the root); and
    //   (2) on macOS the per-user temp dir contains system dirs
    //       (com.apple.*/TemporaryItems) that EPERM even their owner. A
    //       recursive_directory_iterator increment error TERMINATES the walk
    //       (the iterator becomes end()), so the old root-wide walk returned
    //       kOsOk with a SILENTLY TRUNCATED listing — latest_version() then
    //       saw "no commits" on a populated table and Delta's concurrent-
    //       commit check failed every append (found on macOS by
    //       test_lakehouse_sink; never reproduced on Windows because that
    //       temp dir walk happened not to error).
    std::filesystem::path start(fs->root);
    if (prefix != nullptr && *prefix != '\0') {
        const char* last_slash = std::strrchr(prefix, '/');
        if (last_slash != nullptr && last_slash != prefix) {
            start /= std::string(prefix, static_cast<size_t>(last_slash - prefix));
        }
    }
    if (!std::filesystem::is_directory(start, ec) || ec) {
        return kOsOk;   // nothing under this prefix — a legitimate empty listing
    }
    std::filesystem::recursive_directory_iterator it(
        start, std::filesystem::directory_options::skip_permission_denied, ec);
    std::filesystem::recursive_directory_iterator end;
    if (ec) return kOsIoError;
    uint32_t count = 0;
    while (it != end && count < cap) {   // bounded by cap
        if (it->is_regular_file(ec)) {
            std::string full = it->path().generic_string();
            if (full.size() > root_len) {
                const char* rel = full.c_str() + root_len;
                while (*rel == '/' || *rel == '\\') ++rel;
                const size_t rl = std::strlen(rel);
                const size_t pl = (prefix != nullptr) ? std::strlen(prefix) : 0u;
                if (rl > 0 && rl < kOsMaxKey &&
                    (pl == 0 ||
                     (rl >= pl && std::memcmp(rel, prefix, pl) == 0))) {
                    std::memcpy(out[count].key, rel, rl + 1u);
                    out[count].size = static_cast<uint64_t>(it->file_size(ec));
                    ++count;
                }
            }
        }
        it.increment(ec);
        // A walk error must FAIL the listing, never silently truncate it:
        // callers use list results to pick the next Delta/Iceberg commit
        // number, and a truncated listing corrupts the table.
        if (ec) return kOsIoError;
    }
    *out_n = count;
    return kOsOk;
}

int fs_delete(void* impl, const char* key) noexcept {
    assert(impl != nullptr);
    assert(key != nullptr);
    FilesystemObjectStore* fs = static_cast<FilesystemObjectStore*>(impl);
    char path[kOsMaxRoot + kOsMaxKey + 2u];
    if (!join_path(fs->root, key, path, sizeof(path))) return kOsBadArg;
    std::error_code ec;
    const bool removed = std::filesystem::remove(path, ec);
    if (ec) return kOsIoError;
    return removed ? kOsOk : kOsNotFound;
}

int fs_head(void* impl, const char* key, ObjectMeta* out) noexcept {
    assert(impl != nullptr);
    assert(out != nullptr);
    FilesystemObjectStore* fs = static_cast<FilesystemObjectStore*>(impl);
    char path[kOsMaxRoot + kOsMaxKey + 2u];
    if (!join_path(fs->root, key, path, sizeof(path))) return kOsBadArg;
    std::memset(out, 0, sizeof(*out));
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec) return kOsNotFound;
    out->size = static_cast<uint64_t>(std::filesystem::file_size(path, ec));
    out->exists = true;
    return kOsOk;
}

// ---------------------------------------------------------------------------
// put_if_absent (G2ICE-163). The body is staged in a private temp file and
// then PUBLISHED under `key` by an operation the OS itself refuses when the
// name exists: link(2) on POSIX, MoveFileExW without REPLACE_EXISTING on
// Windows. Readers therefore never see a partial object, and of N racing
// writers exactly one wins. HEAD-then-PUT (what the Iceberg writer used to
// do) is not atomic: every racer can observe "absent" before any of them
// writes.
// ---------------------------------------------------------------------------

std::atomic<uint64_t> g_tmp_nonce{0};

// Stage data[0..len) at `tmp`. Returns kOs* code.
int stage_tmp(const char* tmp, const uint8_t* data, uint64_t len) noexcept {
    assert(tmp != nullptr);
    assert(data != nullptr || len == 0);
    std::FILE* f = std::fopen(tmp, "wb");
    if (f == nullptr) return kOsIoError;
    const size_t wrote = len == 0 ? 0u
                                  : std::fwrite(data, 1, static_cast<size_t>(len), f);
    const bool flushed = std::fflush(f) == 0;
    const bool closed  = std::fclose(f) == 0;
    if (wrote != static_cast<size_t>(len) || !flushed || !closed) {
        std::remove(tmp);
        return kOsIoError;
    }
    return kOsOk;
}

#if !defined(_WIN32)
// Fallback for filesystems without hard links: an O_EXCL create is still
// atomic with respect to existence (only the content becomes visible
// progressively).
int publish_excl_create(const char* path, const uint8_t* data,
                        uint64_t len) noexcept {
    assert(path != nullptr);
    assert(data != nullptr || len == 0);
    const int fd = ::open(path, O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd < 0) return errno == EEXIST ? kOsExists : kOsIoError;
    uint64_t off = 0;
    for (uint32_t guard = 0; off < len && guard < (1u << 30); ++guard) {
        const ssize_t w = ::write(fd, data + off, static_cast<size_t>(len - off));
        if (w <= 0) { ::close(fd); ::unlink(path); return kOsIoError; }
        off += static_cast<uint64_t>(w);
    }
    return ::close(fd) == 0 && off == len ? kOsOk : kOsIoError;
}
#endif

// Move staged `tmp` to `path` iff `path` does not exist. Consumes `tmp`.
int publish_no_replace(const char* tmp, const char* path, const uint8_t* data,
                       uint64_t len) noexcept {
    assert(tmp != nullptr && path != nullptr);
    assert(std::strcmp(tmp, path) != 0);
#if defined(_WIN32)
    (void)data; (void)len;
    const std::filesystem::path wt(tmp), wp(path);
    if (::MoveFileExW(wt.c_str(), wp.c_str(), MOVEFILE_WRITE_THROUGH))
        return kOsOk;
    const DWORD e = ::GetLastError();
    std::remove(tmp);
    return (e == ERROR_ALREADY_EXISTS || e == ERROR_FILE_EXISTS) ? kOsExists
                                                                 : kOsIoError;
#else
    int rc = kOsOk;
    if (::link(tmp, path) != 0) {
        const int e = errno;
        if (e == EEXIST) rc = kOsExists;
        else if (e == EPERM || e == ENOTSUP || e == EOPNOTSUPP || e == EXDEV ||
                 e == EMLINK || e == ENOSYS)
            rc = publish_excl_create(path, data, len);
        else rc = kOsIoError;
    }
    ::unlink(tmp);
    return rc;
#endif
}

int fs_put_if_absent(void* impl, const char* key, const uint8_t* data,
                     uint64_t len) noexcept {
    assert(impl != nullptr);
    assert(data != nullptr || len == 0);
    FilesystemObjectStore* fs = static_cast<FilesystemObjectStore*>(impl);
    char path[kOsMaxRoot + kOsMaxKey + 2u];
    if (!join_path(fs->root, key, path, sizeof(path))) return kOsBadArg;
    std::error_code ec;
    std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path(), ec);
    }
    // Private staging name: pid separates processes (read live, so a forked
    // child never shares its parent's), the nonce separates threads/calls.
    char tmp[sizeof(path) + 64u];
    const int n = std::snprintf(
        tmp, sizeof(tmp), "%s.tmp-%d-%llu", path, bolt_getpid(),
        static_cast<unsigned long long>(
            g_tmp_nonce.fetch_add(1, std::memory_order_relaxed)));
    if (n <= 0 || static_cast<size_t>(n) >= sizeof(tmp)) return kOsBadArg;
    const int st = stage_tmp(tmp, data, len);
    if (st != kOsOk) return st;
    return publish_no_replace(tmp, path, data, len);
}

const ObjectStoreVT kFilesystemVT = {
    fs_get, fs_put, fs_list, fs_delete, fs_head, fs_put_if_absent,
};

}  // namespace

bool filesystem_object_store_init(FilesystemObjectStore* fs, const char* root,
                                  ObjectStore* out) noexcept {
    assert(fs != nullptr);
    assert(out != nullptr);
    if (fs == nullptr || root == nullptr || out == nullptr) return false;
    const size_t rl = std::strlen(root);
    if (rl + 1u > kOsMaxRoot) return false;
    std::memcpy(fs->root, root, rl + 1u);
    out->vt = &kFilesystemVT;
    out->impl = fs;
    return true;
}

}  // namespace lakehouse
}  // namespace bolt
