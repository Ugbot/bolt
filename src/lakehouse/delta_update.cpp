// bolt/lakehouse/delta_update.cpp — UPDATE WHERE per-file rewrite (v1: emits
// Remove for matching files; logical replacement scheduled in a follow-up
// once the row-rewriter ships). Tiger Style: PODs, asserts ≥2, no exceptions.

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

bool file_matches(const LiveFile* f, const Predicate* p) noexcept {
    assert(f != nullptr && p != nullptr);
    // Reuse the stats/partition prune: if file *cannot* contain matches we
    // skip it. delta_file_passes returns true when the file MAY match — a
    // sound (over-approximating) per-file rewrite filter.
    return delta_file_passes(f, p, 1);
}

uint32_t emit_remove(char* dst, uint32_t cap, uint32_t off, const char* path,
                     uint64_t ms) noexcept {
    assert(dst != nullptr && path != nullptr);
    int wn = std::snprintf(dst + off, cap - off,
        "{\"remove\":{\"path\":\"%s\",\"deletionTimestamp\":%llu,"
        "\"dataChange\":true}}\n",
        path, static_cast<unsigned long long>(ms));
    if (wn < 0) return off;
    return off + static_cast<uint32_t>(wn);
}

uint32_t emit_commit_info(char* dst, uint32_t cap, uint32_t off,
                          const char* op, uint64_t ms) noexcept {
    int wn = std::snprintf(dst + off, cap - off,
        "{\"commitInfo\":{\"timestamp\":%llu,\"operation\":\"%s\"}}\n",
        static_cast<unsigned long long>(ms), op);
    if (wn < 0) return off;
    return off + static_cast<uint32_t>(wn);
}

}  // namespace

bool delta_table_update(TableHandle* th, const Predicate* pred,
                        const Assignment* assigns, uint32_t n) noexcept {
    assert(th != nullptr && pred != nullptr);
    (void)assigns; (void)n;
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
        if (!file_matches(&snap.files[i], pred)) continue;
        off = emit_remove(body, 64u * 1024u, off, snap.files[i].path, ms);
        ++hits;
    }
    if (hits == 0) return true;
    off = emit_commit_info(body, 64u * 1024u, off, "UPDATE", ms);
    return delta_writer_commit_raw(th, body, off, &base);
}

}  // namespace delta
}  // namespace lakehouse
}  // namespace bolt
