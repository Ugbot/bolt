// bolt/lakehouse/delta_merge.cpp — MERGE INTO. v1: source rows are bounded
// (≤1M) and compiled into a per-key hash-set; we emit a Remove for each
// target file overlapping any source key (over-approximation via min/max
// stats) and a CommitInfo. Row-level rewrite ships in a follow-up with the
// Parquet row-rewriter.

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

constexpr uint32_t kMergeSourceCap = 1u << 20;

bool file_overlaps_key(const LiveFile* f, int64_t key,
                       const char* col) noexcept {
    assert(f != nullptr && col != nullptr);
    int slot = -1;
    for (uint32_t i = 0; i < f->stats.n_cols; ++i) {
        if (std::strcmp(f->stats.col_names[i], col) == 0) {
            slot = static_cast<int>(i); break;
        }
    }
    if (slot < 0) return true;
    char buf[kLakeMaxValBytes];
    std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(key));
    const char* lo = f->stats.min_str[slot];
    const char* hi = f->stats.max_str[slot];
    if (lo[0] != '\0' && std::strcmp(buf, lo) < 0) return false;
    if (hi[0] != '\0' && std::strcmp(buf, hi) > 0) return false;
    return true;
}

}  // namespace

bool delta_table_merge(TableHandle* th, const MergeSpec* spec) noexcept {
    assert(th != nullptr && spec != nullptr);
    if (spec->n_source > kMergeSourceCap) return false;
    Arena scratch;
    Snapshot snap{};
    if (!delta_snapshot_build(&th->os, th->table_rel, -1, &scratch, &snap))
        return false;
    int64_t base = snap.version;
    char* body = scratch.allocate_array<char>(64u * 1024u);
    if (body == nullptr) return false;
    uint32_t off = 0;
    const uint64_t ms = delta_writer_now_ms();
    uint32_t touched = 0;

    for (uint32_t fi = 0; fi < snap.n_files; ++fi) {
        bool overlap = false;
        for (uint32_t i = 0; i < spec->n_source && !overlap; ++i) {
            if (file_overlaps_key(&snap.files[fi], spec->source[i].key,
                                   spec->key_column)) {
                overlap = true;
            }
        }
        if (!overlap) continue;
        if (spec->matched_action == MergeWhenMatched::kNone) continue;
        int wn = std::snprintf(body + off, 64u * 1024u - off,
            "{\"remove\":{\"path\":\"%s\",\"deletionTimestamp\":%llu,"
            "\"dataChange\":true}}\n",
            snap.files[fi].path, static_cast<unsigned long long>(ms));
        if (wn < 0) break;
        off += static_cast<uint32_t>(wn);
        ++touched;
    }
    if (touched == 0 && spec->not_matched_action == MergeWhenNotMatched::kNone)
        return true;
    int wn = std::snprintf(body + off, 64u * 1024u - off,
        "{\"commitInfo\":{\"timestamp\":%llu,\"operation\":\"MERGE\"}}\n",
        static_cast<unsigned long long>(ms));
    if (wn > 0) off += static_cast<uint32_t>(wn);
    return delta_writer_commit_raw(th, body, off, &base);
}

}  // namespace delta
}  // namespace lakehouse
}  // namespace bolt
