// bolt/lakehouse/delta_vacuum.cpp — VACUUM: list table data dir, cross-
// reference with live snapshot (and a bounded window of historical Add
// paths), delete unreferenced files older than retention_hours. Dry-run
// returns the list without deleting.

#include "bolt/lakehouse/delta/optimize.h"

#include <cstdio>
#include <cstring>

#include "bolt/bolt_arena.h"
#include "bolt/lakehouse/delta/log.h"
#include "bolt/lakehouse/delta/snapshot.h"
#include "bolt/lakehouse/handle.h"
#include "delta_write_internal.h"

namespace bolt {
namespace lakehouse {
namespace delta {

namespace {

constexpr uint32_t kVacuumHistMaxPaths = 4096u;

struct HistCtx {
    char (*paths)[kDeltaMaxPath];
    uint32_t  n;
    uint32_t  cap;
};

bool collect_path(void* raw, const DeltaAction* a) noexcept {
    assert(raw != nullptr && a != nullptr);
    auto* c = static_cast<HistCtx*>(raw);
    if (a->kind != ActionKind::kAdd) return true;
    if (c->n >= c->cap) return true;
    std::strncpy(c->paths[c->n], a->add.path, kDeltaMaxPath - 1u);
    c->paths[c->n][kDeltaMaxPath - 1u] = '\0';
    ++c->n;
    return true;
}

bool referenced(const HistCtx* h, const char* rel) noexcept {
    assert(h != nullptr && rel != nullptr);
    for (uint32_t i = 0; i < h->n; ++i)
        if (std::strcmp(h->paths[i], rel) == 0) return true;
    return false;
}

}  // namespace

bool delta_table_vacuum(TableHandle* th, uint64_t retention_hours,
                        bool dry_run,
                        char (*out_deleted)[kVacuumPathBytes],
                        uint32_t cap, uint32_t* n) noexcept {
    assert(th != nullptr && n != nullptr);
    assert(out_deleted != nullptr || cap == 0);
    *n = 0;
    Arena scratch;
    HistCtx hist{};
    hist.paths = scratch.allocate_array<char[kDeltaMaxPath]>(kVacuumHistMaxPaths);
    if (hist.paths == nullptr) return false;
    hist.cap = kVacuumHistMaxPaths;
    if (!delta_log_walk_all(&th->os, th->table_rel, -1, &scratch, &hist,
                             collect_path)) return false;

    // List all objects under <table_rel>/data/
    char prefix[kDeltaMaxPath];
    std::snprintf(prefix, sizeof(prefix), "%s/data/", th->table_rel);
    ObjectEntry* entries = scratch.allocate_array<ObjectEntry>(kLakeMaxLiveFiles);
    if (entries == nullptr) return false;
    uint32_t n_entries = 0;
    if (os_list(&th->os, prefix, entries, kLakeMaxLiveFiles, &n_entries)
        != kOsOk) return false;

    const uint64_t now_s = delta_writer_now_ms() / 1000ull;
    const uint64_t cutoff_s = retention_hours * 3600ull;

    for (uint32_t i = 0; i < n_entries; ++i) {
        // Build path relative to table_rel.
        const char* k = entries[i].key;
        const size_t pl = std::strlen(th->table_rel);
        const char* rel = k + pl;
        while (*rel == '/' || *rel == '\\') ++rel;
        if (referenced(&hist, rel)) continue;

        ObjectMeta meta{};
        if (os_head(&th->os, entries[i].key, &meta) != kOsOk) continue;
        if (cutoff_s > 0 && meta.mtime_unix > 0 &&
            now_s - meta.mtime_unix < cutoff_s) continue;

        if (!dry_run) {
            if (os_delete(&th->os, entries[i].key) != kOsOk) continue;
        }
        if (*n < cap) {
            std::strncpy(out_deleted[*n], entries[i].key,
                         kVacuumPathBytes - 1u);
            out_deleted[*n][kVacuumPathBytes - 1u] = '\0';
        }
        ++(*n);
    }
    return true;
}

}  // namespace delta
}  // namespace lakehouse
}  // namespace bolt
