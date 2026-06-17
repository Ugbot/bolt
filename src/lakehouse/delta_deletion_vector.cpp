// bolt/lakehouse/delta_deletion_vector.cpp — Roaring-bitmap DV loader.

#include "bolt/lakehouse/delta/deletion_vector.h"

#include <cstdio>
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

bool build_uuid_path(const char* uuid, char* out, uint32_t cap) noexcept {
    assert(uuid != nullptr && out != nullptr);
    const int n = std::snprintf(out, cap, "deletion_vector_%s.bin", uuid);
    return n > 0 && static_cast<uint32_t>(n) < cap;
}

}  // namespace

bool delta_dv_load(ObjectStore* os, const char* table_rel_prefix,
                   const DvDescriptor* dv, Arena* arena,
                   DeletionVector* out) noexcept {
    assert(os != nullptr && dv != nullptr && arena != nullptr && out != nullptr);
    std::memset(out, 0, sizeof(*out));
    if (dv->type == DvStorageType::kNone) return true;
    if (dv->type == DvStorageType::kInline) {
        // base85 inline decoder not in W2 — leave absent (scanner returns all rows).
        return true;
    }
    char key[kDeltaMaxPath];
    if (dv->type == DvStorageType::kUuid) {
        char fname[kDeltaMaxPath];
        if (!build_uuid_path(dv->uuid_or_path, fname, sizeof(fname)))
            return false;
        if (!join_rel(table_rel_prefix, fname, key, sizeof(key))) return false;
    } else {
        if (dv->uuid_or_path[0] == '/') {
            std::strncpy(key, dv->uuid_or_path, sizeof(key) - 1);
            key[sizeof(key) - 1] = '\0';
        } else if (!join_rel(table_rel_prefix, dv->uuid_or_path,
                              key, sizeof(key))) {
            return false;
        }
    }
    const uint8_t* body = nullptr;
    uint64_t len = 0;
    if (os_get(os, key, arena, &body, &len) != kOsOk) return false;
    if (len >= 12u && bolt::ingest::roaring_deserialize(body + 12, len - 12u,
                                                         arena, &out->bitmap)) {
        out->present = true;
        return true;
    }
    if (bolt::ingest::roaring_deserialize(body, len, arena, &out->bitmap)) {
        out->present = true;
        return true;
    }
    return false;
}

bool delta_dv_contains(const DeletionVector* dv, uint64_t row_index) noexcept {
    assert(dv != nullptr);
    if (!dv->present) return false;
    if (row_index > 0xFFFFFFFFull) return false;
    return bolt::ingest::roaring_contains(
        &dv->bitmap, static_cast<uint32_t>(row_index));
}

}  // namespace delta
}  // namespace lakehouse
}  // namespace bolt
