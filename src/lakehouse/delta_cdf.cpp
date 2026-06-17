// bolt/lakehouse/delta_cdf.cpp — CDF file lister (skeleton).

#include "bolt/lakehouse/delta/cdf.h"

#include <cstring>

namespace bolt {
namespace lakehouse {
namespace delta {

namespace {

bool join_rel(const char* prefix, const char* sub, char* out,
              uint32_t cap) noexcept {
    assert(out != nullptr && cap > 0u);
    const size_t pl = prefix != nullptr ? std::strlen(prefix) : 0u;
    const size_t sl = std::strlen(sub);
    if (pl + 1u + sl + 1u > cap) return false;
    uint32_t p = 0;
    if (pl > 0) { std::memcpy(out, prefix, pl); p = static_cast<uint32_t>(pl); }
    if (pl > 0 && prefix[pl - 1] != '/' && prefix[pl - 1] != '\\')
        out[p++] = '/';
    std::memcpy(out + p, sub, sl);
    p += static_cast<uint32_t>(sl);
    out[p] = '\0';
    return true;
}

bool ends_with(const char* s, const char* suf) noexcept {
    assert(s != nullptr && suf != nullptr);
    const size_t sl = std::strlen(s);
    const size_t fl = std::strlen(suf);
    if (sl < fl) return false;
    return std::strcmp(s + sl - fl, suf) == 0;
}

}  // namespace

bool delta_cdf_list(ObjectStore* os, const char* table_rel_prefix,
                    Arena* arena, CdfFileSet* out) noexcept {
    assert(os != nullptr && arena != nullptr && out != nullptr);
    std::memset(out, 0, sizeof(*out));
    out->files = arena->allocate_array<CdfFile>(kDeltaMaxCdfFiles);
    if (out->files == nullptr) return false;
    out->n_files_cap = kDeltaMaxCdfFiles;
    char prefix[kDeltaMaxPath];
    if (!join_rel(table_rel_prefix, "_change_data/", prefix, sizeof(prefix)))
        return false;
    ObjectEntry* entries = arena->allocate_array<ObjectEntry>(kDeltaMaxCdfFiles);
    if (entries == nullptr) return false;
    uint32_t n_entries = 0;
    const int rc = os_list(os, prefix, entries, kDeltaMaxCdfFiles, &n_entries);
    if (rc != kOsOk) return false;
    for (uint32_t i = 0; i < n_entries && out->n_files < out->n_files_cap; ++i) {
        if (!ends_with(entries[i].key, ".parquet")) continue;
        CdfFile* f = &out->files[out->n_files++];
        std::memset(f, 0, sizeof(*f));
        const size_t kl = std::strlen(entries[i].key);
        const uint32_t n = kl >= sizeof(f->path) ? sizeof(f->path) - 1u
                                                  : static_cast<uint32_t>(kl);
        std::memcpy(f->path, entries[i].key, n);
        f->path[n] = '\0';
        f->kind = CdfKind::kInsert;
    }
    return true;
}

}  // namespace delta
}  // namespace lakehouse
}  // namespace bolt
