// bolt/lakehouse/delta_restore.cpp — RESTORE TABLE TO VERSION N.
// Reads snapshot @target_version, diffs against current snapshot, emits a
// commit with Add for files present at target but not at current, and
// Remove for files present at current but not at target. CommitInfo carries
// operation=RESTORE.

#include "bolt/lakehouse/delta/writer.h"

#include <cstdio>
#include <cstring>

#include "bolt/bolt_arena.h"
#include "bolt/lakehouse/delta/snapshot.h"
#include "bolt/lakehouse/handle.h"
#include "delta_write_internal.h"

namespace bolt {
namespace lakehouse {
namespace delta {

namespace {

bool snap_contains(const Snapshot* s, const char* path) noexcept {
    assert(s != nullptr && path != nullptr);
    for (uint32_t i = 0; i < s->n_files; ++i)
        if (std::strcmp(s->files[i].path, path) == 0) return true;
    return false;
}

}  // namespace

bool delta_table_restore(TableHandle* th, int64_t target_version) noexcept {
    assert(th != nullptr);
    if (target_version < 0) return false;
    Arena scratch;
    Snapshot cur{};
    Snapshot tgt{};
    if (!delta_snapshot_build(&th->os, th->table_rel, -1, &scratch, &cur))
        return false;
    if (!delta_snapshot_build(&th->os, th->table_rel, target_version, &scratch,
                              &tgt))
        return false;
    if (target_version > cur.version) return false;
    int64_t base = cur.version;
    char* body = scratch.allocate_array<char>(128u * 1024u);
    if (body == nullptr) return false;
    uint32_t off = 0;
    const uint64_t ms = delta_writer_now_ms();

    // Remove files no longer in target.
    for (uint32_t i = 0; i < cur.n_files; ++i) {
        if (snap_contains(&tgt, cur.files[i].path)) continue;
        int wn = std::snprintf(body + off, 128u * 1024u - off,
            "{\"remove\":{\"path\":\"%s\",\"deletionTimestamp\":%llu,"
            "\"dataChange\":false}}\n",
            cur.files[i].path, static_cast<unsigned long long>(ms));
        if (wn < 0) break;
        off += static_cast<uint32_t>(wn);
    }
    // Re-Add files present in target but missing in current.
    for (uint32_t i = 0; i < tgt.n_files; ++i) {
        if (snap_contains(&cur, tgt.files[i].path)) continue;
        int wn = std::snprintf(body + off, 128u * 1024u - off,
            "{\"add\":{\"path\":\"%s\",\"size\":%lld,\"modificationTime\":%llu,"
            "\"dataChange\":false,\"partitionValues\":{}}}\n",
            tgt.files[i].path,
            static_cast<long long>(tgt.files[i].size),
            static_cast<unsigned long long>(ms));
        if (wn < 0) break;
        off += static_cast<uint32_t>(wn);
    }
    int wn = std::snprintf(body + off, 128u * 1024u - off,
        "{\"commitInfo\":{\"timestamp\":%llu,\"operation\":\"RESTORE\"}}\n",
        static_cast<unsigned long long>(ms));
    if (wn > 0) off += static_cast<uint32_t>(wn);
    return delta_writer_commit_raw(th, body, off, &base);
}

}  // namespace delta
}  // namespace lakehouse
}  // namespace bolt
