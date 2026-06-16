// bolt/lakehouse/catalog_filesystem.cpp — filesystem-backed Catalog.
// namespace = sub-dir of root; table = sub-dir of namespace holding a Delta
// `_delta_log/` or Iceberg `metadata/` tree. Portable std::filesystem.

#include "bolt/lakehouse/catalog.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <system_error>

namespace bolt {
namespace lakehouse {

namespace {

namespace fs = std::filesystem;

// Build root/ns or root/ns/name into a std::filesystem::path. Returns false on
// a null required component.
bool ns_dir(const char* root, const char* ns, fs::path* out) noexcept {
    assert(root != nullptr);
    if (ns == nullptr || *ns == '\0') return false;
    *out = fs::path(root) / ns;
    return true;
}
bool tbl_dir(const char* root, const char* ns, const char* name,
             fs::path* out) noexcept {
    assert(root != nullptr);
    if (ns == nullptr || name == nullptr || *ns == '\0' || *name == '\0')
        return false;
    *out = fs::path(root) / ns / name;
    return true;
}

// Copy a path's UTF-8 generic string into out (cap bytes). False on overflow.
bool path_to(const fs::path& p, char* out, uint32_t cap) noexcept {
    assert(out != nullptr);
    const std::string s = p.generic_string();
    if (s.size() + 1u > cap) return false;
    std::memcpy(out, s.c_str(), s.size() + 1u);
    return true;
}

int list_dirs(const fs::path& base, CatalogName* out, uint32_t cap,
              uint32_t* out_n) noexcept {
    assert(out != nullptr && out_n != nullptr);
    *out_n = 0;
    std::error_code ec;
    if (!fs::exists(base, ec)) return kCatOk;       // empty listing, not error
    fs::directory_iterator it(base, ec), end;
    if (ec) return kCatIoError;
    uint32_t count = 0;
    for (; it != end && count < cap; it.increment(ec)) {   // bounded by cap
        if (ec) break;
        if (!it->is_directory(ec)) continue;
        const std::string nm = it->path().filename().generic_string();
        if (nm.empty() || nm.size() >= kCatMaxName) continue;
        std::memcpy(out[count].name, nm.c_str(), nm.size() + 1u);
        ++count;
    }
    *out_n = count;
    return kCatOk;
}

int cf_list_namespaces(void* impl, CatalogName* out, uint32_t cap,
                       uint32_t* out_n) noexcept {
    assert(impl != nullptr);
    FilesystemCatalog* c = static_cast<FilesystemCatalog*>(impl);
    return list_dirs(fs::path(c->root), out, cap, out_n);
}

int cf_list_tables(void* impl, const char* ns, CatalogName* out, uint32_t cap,
                   uint32_t* out_n) noexcept {
    assert(impl != nullptr);
    assert(out != nullptr);
    FilesystemCatalog* c = static_cast<FilesystemCatalog*>(impl);
    fs::path d;
    if (!ns_dir(c->root, ns, &d)) return kCatBadArg;
    return list_dirs(d, out, cap, out_n);
}

int cf_table_path(void* impl, const char* ns, const char* name, char* out,
                  uint32_t cap) noexcept {
    assert(impl != nullptr);
    assert(out != nullptr);
    FilesystemCatalog* c = static_cast<FilesystemCatalog*>(impl);
    fs::path d;
    if (!tbl_dir(c->root, ns, name, &d)) return kCatBadArg;
    std::error_code ec;
    if (!fs::exists(d, ec) || ec) return kCatNotFound;
    return path_to(d, out, cap) ? kCatOk : kCatBadArg;
}

int cf_create_table(void* impl, const char* ns, const char* name,
                    TableLayout layout) noexcept {
    assert(impl != nullptr);
    FilesystemCatalog* c = static_cast<FilesystemCatalog*>(impl);
    fs::path d;
    if (!tbl_dir(c->root, ns, name, &d)) return kCatBadArg;
    std::error_code ec;
    if (fs::exists(d, ec)) return kCatExists;
    const char* sub = layout == TableLayout::kIceberg ? "metadata"
                                                      : "_delta_log";
    if (!fs::create_directories(d / sub, ec) || ec) return kCatIoError;
    return kCatOk;
}

int cf_drop_table(void* impl, const char* ns, const char* name) noexcept {
    assert(impl != nullptr);
    FilesystemCatalog* c = static_cast<FilesystemCatalog*>(impl);
    fs::path d;
    if (!tbl_dir(c->root, ns, name, &d)) return kCatBadArg;
    std::error_code ec;
    if (!fs::exists(d, ec) || ec) return kCatNotFound;
    const auto removed = fs::remove_all(d, ec);
    if (ec) return kCatIoError;
    return removed > 0 ? kCatOk : kCatNotFound;
}

int cf_commit_metadata(void* impl, const char* ns, const char* name,
                       const char* rel_path, const uint8_t* data,
                       uint64_t len) noexcept {
    assert(impl != nullptr);
    assert(data != nullptr || len == 0);
    FilesystemCatalog* c = static_cast<FilesystemCatalog*>(impl);
    fs::path d;
    if (!tbl_dir(c->root, ns, name, &d)) return kCatBadArg;
    if (rel_path == nullptr || *rel_path == '\0') return kCatBadArg;
    const fs::path target = d / rel_path;
    std::error_code ec;
    if (target.has_parent_path()) {
        fs::create_directories(target.parent_path(), ec);
    }
    std::FILE* f = std::fopen(target.generic_string().c_str(), "wb");
    if (f == nullptr) return kCatIoError;
    const size_t wrote = len == 0 ? 0u
                                  : std::fwrite(data, 1, static_cast<size_t>(len), f);
    std::fclose(f);
    return wrote == static_cast<size_t>(len) ? kCatOk : kCatIoError;
}

const CatalogVT kFilesystemCatalogVT = {
    cf_list_namespaces, cf_list_tables, cf_table_path,
    cf_create_table, cf_drop_table, cf_commit_metadata,
};

}  // namespace

bool filesystem_catalog_init(FilesystemCatalog* fc, const char* root,
                             Catalog* out) noexcept {
    assert(fc != nullptr);
    assert(out != nullptr);
    if (fc == nullptr || root == nullptr || out == nullptr) return false;
    const size_t rl = std::strlen(root);
    if (rl + 1u > kCatMaxRoot) return false;
    std::memcpy(fc->root, root, rl + 1u);
    std::error_code ec;
    std::filesystem::create_directories(root, ec);   // ok if it already exists
    out->vt = &kFilesystemCatalogVT;
    out->impl = fc;
    return true;
}

}  // namespace lakehouse
}  // namespace bolt
