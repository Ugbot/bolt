// bolt/lakehouse/delta_delete.cpp — DELETE WHERE per-file rewrite (v1: emits
// Remove actions for files that may contain matching rows).

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

bool delta_table_delete(TableHandle* th, const Predicate* pred) noexcept {
    assert(th != nullptr && pred != nullptr);
    Arena scratch;
    Snapshot snap{};
    if (!delta_snapshot_build(&th->os, th->table_rel, -1, &scratch, &snap))
        return false;
    int64_t base = snap.version;
    char* body = scratch.allocate_array<char>(64u * 1024u);
    if (body == nullptr) return false;
    uint32_t off = 0;
    const uint64_t ms = delta_writer_now_ms();
    uint32_t hits = 0;
    for (uint32_t i = 0; i < snap.n_files; ++i) {
        if (!delta_file_passes(&snap.files[i], pred, 1)) continue;
        int wn = std::snprintf(body + off, 64u * 1024u - off,
            "{\"remove\":{\"path\":\"%s\",\"deletionTimestamp\":%llu,"
            "\"dataChange\":true}}\n",
            snap.files[i].path, static_cast<unsigned long long>(ms));
        if (wn < 0) break;
        off += static_cast<uint32_t>(wn);
        ++hits;
    }
    if (hits == 0) return true;
    int wn = std::snprintf(body + off, 64u * 1024u - off,
        "{\"commitInfo\":{\"timestamp\":%llu,\"operation\":\"DELETE\"}}\n",
        static_cast<unsigned long long>(ms));
    if (wn > 0) off += static_cast<uint32_t>(wn);
    return delta_writer_commit_raw(th, body, off, &base);
}

}  // namespace delta
}  // namespace lakehouse
}  // namespace bolt
