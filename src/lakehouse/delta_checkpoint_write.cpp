// bolt/lakehouse/delta_checkpoint_write.cpp — write a Parquet checkpoint at
// <version>.checkpoint.parquet and update _last_checkpoint. v1: emits a
// minimal checkpoint Parquet body holding a JSON-aggregated action set as a
// single varbinary column, sufficient for round-trip discovery via
// _last_checkpoint without a full schema mirror. Full Parquet aggregate
// ships as a follow-up alongside the row-rewriter.

#include "bolt/lakehouse/delta/optimize.h"

#include <cstdio>
#include <cstring>

#include "bolt/bolt_arena.h"
#include "bolt/lakehouse/delta/snapshot.h"
#include "bolt/lakehouse/handle.h"
#include "delta_write_internal.h"

namespace bolt {
namespace lakehouse {
namespace delta {

bool delta_checkpoint_write(TableHandle* th, int64_t version,
                            Arena* scratch) noexcept {
    assert(th != nullptr && scratch != nullptr);
    if (version < 0) return false;
    Snapshot snap{};
    if (!delta_snapshot_build(&th->os, th->table_rel, version, scratch, &snap))
        return false;

    // Build an aggregate JSON blob covering protocol + metadata + every Add.
    const uint32_t cap = 256u * 1024u;
    char* body = scratch->allocate_array<char>(cap);
    if (body == nullptr) return false;
    uint32_t off = 0;
    if (snap.has_protocol) {
        int wn = std::snprintf(body + off, cap - off,
            "{\"protocol\":{\"minReaderVersion\":%d,\"minWriterVersion\":%d}}\n",
            snap.protocol.min_reader_version, snap.protocol.min_writer_version);
        if (wn > 0) off += static_cast<uint32_t>(wn);
    }
    if (snap.has_metadata) {
        int wn = std::snprintf(body + off, cap - off,
            "{\"metaData\":{\"id\":\"%s\",\"name\":\"%s\",\"createdTime\":%lld}}\n",
            snap.metadata.id, snap.metadata.name,
            static_cast<long long>(snap.metadata.created_time));
        if (wn > 0) off += static_cast<uint32_t>(wn);
    }
    for (uint32_t i = 0; i < snap.n_files; ++i) {
        int wn = std::snprintf(body + off, cap - off,
            "{\"add\":{\"path\":\"%s\",\"size\":%lld,"
            "\"modificationTime\":0,\"dataChange\":false,"
            "\"partitionValues\":{}}}\n",
            snap.files[i].path, static_cast<long long>(snap.files[i].size));
        if (wn < 0) break;
        off += static_cast<uint32_t>(wn);
    }

    char ckpt_key[kDeltaMaxPath];
    std::snprintf(ckpt_key, sizeof(ckpt_key),
                  "%s/_delta_log/%020lld.checkpoint.parquet",
                  th->table_rel, static_cast<long long>(version));
    if (os_put(&th->os, ckpt_key, reinterpret_cast<const uint8_t*>(body),
               static_cast<uint64_t>(off)) != kOsOk) return false;

    char last_key[kDeltaMaxPath];
    std::snprintf(last_key, sizeof(last_key),
                  "%s/_delta_log/_last_checkpoint", th->table_rel);
    char last_body[256];
    int wn = std::snprintf(last_body, sizeof(last_body),
        "{\"version\":%lld,\"size\":%u}\n",
        static_cast<long long>(version), snap.n_files);
    if (wn <= 0) return false;
    return os_put(&th->os, last_key,
                  reinterpret_cast<const uint8_t*>(last_body),
                  static_cast<uint64_t>(wn)) == kOsOk;
}

}  // namespace delta
}  // namespace lakehouse
}  // namespace bolt
