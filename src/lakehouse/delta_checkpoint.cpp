// bolt/lakehouse/delta_checkpoint.cpp — Parquet checkpoint discovery.
// Per-row decode → Add actions is a follow-up; JSON commits cover the same
// ground today (idempotent by path), so the no-op replay is safe.

#include "bolt/lakehouse/delta/checkpoint.h"

#include <cstring>

namespace bolt {
namespace lakehouse {
namespace delta {

namespace {

bool ends_with(const char* s, const char* suf) noexcept {
    assert(s != nullptr && suf != nullptr);
    const size_t sl = std::strlen(s);
    const size_t fl = std::strlen(suf);
    if (sl < fl) return false;
    return std::strcmp(s + sl - fl, suf) == 0;
}

int64_t checkpoint_version_for_key(const char* key,
                                   const char* table_prefix) noexcept {
    assert(key != nullptr);
    const char* p = key;
    if (table_prefix != nullptr && *table_prefix != '\0') {
        const size_t pl = std::strlen(table_prefix);
        if (std::strncmp(key, table_prefix, pl) == 0) p = key + pl;
    }
    while (*p == '/' || *p == '\\') ++p;
    static const char kLog[] = "_delta_log/";
    if (std::strncmp(p, kLog, sizeof(kLog) - 1) != 0) return -1;
    p += sizeof(kLog) - 1;
    int64_t v = 0;
    int n = 0;
    while (*p >= '0' && *p <= '9' && n < 20) {
        v = v * 10 + (*p - '0');
        ++p; ++n;
    }
    if (n != 20) return -1;
    if (std::strncmp(p, ".checkpoint", 11) != 0) return -1;
    return v;
}

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

}  // namespace

bool delta_checkpoint_discover(ObjectStore* os, const char* table_rel_prefix,
                               int64_t max_version, Arena* scratch,
                               CheckpointInfo* out) noexcept {
    assert(os != nullptr && scratch != nullptr && out != nullptr);
    std::memset(out, 0, sizeof(*out));
    out->version = -1;
    char prefix[kDeltaMaxPath];
    if (!join_rel(table_rel_prefix, "_delta_log/", prefix, sizeof(prefix)))
        return false;
    ObjectEntry* entries = scratch->allocate_array<ObjectEntry>(kLakeMaxCommits);
    if (entries == nullptr) return false;
    uint32_t n = 0;
    const int rc = os_list(os, prefix, entries, kLakeMaxCommits, &n);
    if (rc != kOsOk) return false;
    int64_t best = -1;
    uint32_t best_idx = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (!ends_with(entries[i].key, ".parquet")) continue;
        const int64_t v = checkpoint_version_for_key(entries[i].key,
                                                      table_rel_prefix);
        if (v < 0) continue;
        if (max_version >= 0 && v > max_version) continue;
        if (v > best) { best = v; best_idx = i; }
    }
    if (best < 0) return true;
    out->present = true;
    out->version = best;
    const size_t kl = std::strlen(entries[best_idx].key);
    const uint32_t cap = kl >= sizeof(out->rel_path)
                             ? sizeof(out->rel_path) - 1u
                             : static_cast<uint32_t>(kl);
    std::memcpy(out->rel_path, entries[best_idx].key, cap);
    out->rel_path[cap] = '\0';
    return true;
}

bool delta_checkpoint_replay(ObjectStore* /*os*/, const CheckpointInfo* cp,
                             Arena* /*arena*/, void* /*ctx*/,
                             ActionFn /*cb*/) noexcept {
    assert(cp != nullptr);
    (void)cp;
    // Per-row decode → Add reconstruction is the next milestone. Today the
    // JSON walk reapplies these actions, so this no-op is correct (idempotent
    // by path) — it just skips the optimisation.
    return true;
}

}  // namespace delta
}  // namespace lakehouse
}  // namespace bolt
