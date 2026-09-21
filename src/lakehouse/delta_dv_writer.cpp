// bolt/lakehouse/delta_dv_writer.cpp — attach a deletion vector to an
// already-committed `add` action without touching the physical parquet
// file (G2ICE-80). This is the writer half of the tiering-eviction use
// case: a cold Delta file that already exists in the log gets its visible
// rows narrowed by a remove(old identity)+add(same path, new DV) commit,
// upgrading the table's protocol to the `deletionVectors` reader/writer
// feature in the same commit if it hasn't been already.
//
// Tiger Style: PODs, >=2 asserts/fn, <=70-line fns, no exceptions, Arena
// allocs, one commit at the end (all-or-nothing).

#include "bolt/lakehouse/delta/writer.h"

#include <cstdio>
#include <cstring>

#include "bolt/bolt_arena.h"
#include "bolt/lakehouse/delta/deletion_vector.h"
#include "bolt/lakehouse/delta/log.h"
#include "delta_write_internal.h"

namespace bolt {
namespace lakehouse {
namespace delta {

namespace {

// One path-scoped pass over the whole log: does `target_path` currently
// have a live `add`, what raw action is it (verbatim stats/partitions), and
// does the table's current protocol already advertise `deletionVectors`.
struct FindCtx {
    const char* target_path;
    DeltaAdd      found;
    bool          alive;
    DeltaProtocol protocol;
    bool          has_protocol;
    DeltaMetadata metadata;
    bool          has_metadata;
    int64_t       version;
};

bool find_cb(void* raw, const DeltaAction* a) noexcept {
    assert(raw != nullptr && a != nullptr);
    FindCtx* ctx = static_cast<FindCtx*>(raw);
    if (a->version > ctx->version) ctx->version = a->version;
    switch (a->kind) {
        case ActionKind::kProtocol:
            ctx->protocol = a->protocol;
            ctx->has_protocol = true;
            break;
        case ActionKind::kMetadata:
            ctx->metadata = a->metadata;
            ctx->has_metadata = true;
            break;
        case ActionKind::kAdd:
            if (std::strcmp(a->add.path, ctx->target_path) == 0) {
                ctx->found = a->add;
                ctx->alive = true;
            }
            break;
        case ActionKind::kRemove:
            if (std::strcmp(a->rem.path, ctx->target_path) == 0) {
                ctx->alive = false;
            }
            break;
        default: break;
    }
    return true;
}

// Merge `old_bitmap` (may be empty/absent) with `new_rows` into a single
// strictly-increasing, deduplicated array. Deletion vectors only grow.
bool merge_sorted_unique(const bolt::ingest::RoaringBitmap* old_bitmap,
                         bool has_old, const uint32_t* new_rows,
                         uint64_t n_new, Arena* arena, uint32_t** out_vals,
                         uint64_t* out_n) noexcept {
    assert(new_rows != nullptr || n_new == 0);
    assert(arena != nullptr && out_vals != nullptr && out_n != nullptr);
    uint64_t old_n = 0;
    uint32_t* old_vals = nullptr;
    if (has_old && old_bitmap != nullptr) {
        old_n = bolt::ingest::roaring_cardinality(old_bitmap);
        old_vals = arena->allocate_array<uint32_t>(old_n == 0 ? 1u : old_n);
        if (old_vals == nullptr) return false;
        uint64_t got = 0;
        if (old_n > 0 &&
            !bolt::ingest::roaring_to_sorted_array(old_bitmap, old_vals,
                                                   old_n, &got))
            return false;
        if (old_n > 0 && got != old_n) return false;
    }
    const uint64_t cap = old_n + n_new;
    uint32_t* merged = arena->allocate_array<uint32_t>(cap == 0 ? 1u : cap);
    if (merged == nullptr) return false;
    uint64_t i = 0, j = 0, w = 0;
    while (i < old_n && j < n_new) {   // bounded: old_n + n_new
        if (old_vals[i] < new_rows[j])      merged[w++] = old_vals[i++];
        else if (old_vals[i] > new_rows[j]) merged[w++] = new_rows[j++];
        else { merged[w++] = old_vals[i]; ++i; ++j; }
    }
    while (i < old_n) merged[w++] = old_vals[i++];   // bounded: old_n
    while (j < n_new) merged[w++] = new_rows[j++];   // bounded: n_new
    *out_vals = merged;
    *out_n = w;
    return true;
}

uint32_t emit_dv_json(char* dst, uint32_t cap, uint32_t off,
                      const DvDescriptor* dv) noexcept {
    assert(dst != nullptr && dv != nullptr);
    const int wn = std::snprintf(dst + off, cap - off,
        ",\"deletionVector\":{\"storageType\":\"u\",\"pathOrInlineDv\":\"%s\","
        "\"offset\":%lld,\"sizeInBytes\":%lld,\"cardinality\":%lld}",
        dv->uuid_or_path, static_cast<long long>(dv->offset),
        static_cast<long long>(dv->size_in_bytes),
        static_cast<long long>(dv->cardinality));
    if (wn < 0) return off;
    return off + static_cast<uint32_t>(wn);
}

uint32_t emit_remove_reconcile(char* dst, uint32_t cap, uint32_t off,
                               const char* path, uint64_t ms,
                               const DeltaAdd* old_add) noexcept {
    assert(dst != nullptr && path != nullptr && old_add != nullptr);
    int wn = std::snprintf(dst + off, cap - off,
        "{\"remove\":{\"path\":\"%s\",\"deletionTimestamp\":%llu,"
        "\"dataChange\":true,\"size\":%lld",
        path, static_cast<unsigned long long>(ms),
        static_cast<long long>(old_add->size));
    if (wn < 0) return off;
    off += static_cast<uint32_t>(wn);
    if (old_add->has_dv) off = emit_dv_json(dst, cap, off, &old_add->dv);
    if (off + 3u < cap) { dst[off++] = '}'; dst[off++] = '}'; dst[off++] = '\n'; dst[off] = '\0'; }
    return off;
}

uint32_t emit_add_reconcile(char* dst, uint32_t cap, uint32_t off,
                            const DeltaAdd* old_add, uint64_t ms,
                            const DvDescriptor* new_dv) noexcept {
    assert(dst != nullptr && old_add != nullptr && new_dv != nullptr);
    int wn = std::snprintf(dst + off, cap - off,
        "{\"add\":{\"path\":\"%s\",\"size\":%lld,\"modificationTime\":%llu,"
        "\"dataChange\":true,\"partitionValues\":{}",
        old_add->path, static_cast<long long>(old_add->size),
        static_cast<unsigned long long>(ms));
    if (wn < 0) return off;
    off += static_cast<uint32_t>(wn);
    if (old_add->stats_len > 0 && off + old_add->stats_len * 2u + 16u < cap) {
        wn = std::snprintf(dst + off, cap - off, ",\"stats\":\"");
        if (wn > 0) off += static_cast<uint32_t>(wn);
        // The log parser's `eat_raw` UNESCAPES `stats` into real JSON text
        // (so callers can re-parse it) — re-embedding it as a JSON string
        // value means it must be RE-ESCAPED here, the same way
        // `delta_delete.cpp`'s `emit_add` escapes freshly-built stats.
        // Forgetting this the first time produced literal unescaped quotes
        // inside a JSON string (`"stats":"{"numRecords":8,...}"`), which
        // silently corrupted the WHOLE commit line -- the parser skipped
        // the malformed `add` action, so the reconciled file came back
        // with a `remove` and no matching `add`: 0 live files.
        off += delta_writer_escape_json(dst, cap, off, old_add->stats_json);
        if (off + 2u < cap) dst[off++] = '"';
    }
    off = emit_dv_json(dst, cap, off, new_dv);
    if (off + 3u < cap) { dst[off++] = '}'; dst[off++] = '}'; dst[off++] = '\n'; dst[off] = '\0'; }
    return off;
}

// `needs_invariants`: Spark's DeltaLog refuses a table-features-protocol
// (v3+) table whose writerFeatures omits "invariants" while its metadata
// implies one (any NOT NULL column) is in use — found verifying this
// ticket externally against real Apache Spark + delta-spark 4.0; DuckDB's
// delta-kernel-based reader did not catch this, so it must be checked
// against a real writer-side implementation, not just one reader.
uint32_t emit_protocol_upgrade(char* dst, uint32_t cap, uint32_t off,
                               bool needs_invariants) noexcept {
    assert(dst != nullptr);
    const int wn = std::snprintf(dst + off, cap - off,
        "{\"protocol\":{\"minReaderVersion\":3,\"minWriterVersion\":7,"
        "\"readerFeatures\":[\"deletionVectors\"],"
        "\"writerFeatures\":[\"deletionVectors\"%s]}}\n",
        needs_invariants ? ",\"invariants\"" : "");
    if (wn < 0) return off;
    return off + static_cast<uint32_t>(wn);
}

// Bolt's own `schema_json_build` always emits the literal `"nullable":%s`
// with `true`/`false` from `%s ? "true" : "false"` (delta_writer.cpp) — a
// deterministic substring probe over OUR OWN generated schemaString is
// exact for that format (not a general JSON scanner; this file only ever
// reads a schema string this same writer produced).
bool metadata_schema_has_not_null(const DeltaMetadata* md) noexcept {
    assert(md != nullptr);
    if (md->schema_len == 0u) return false;
    return std::strstr(md->schema_string, "\"nullable\":false") != nullptr;
}

}  // namespace

bool delta_table_mark_deleted_via_dv(TableHandle* th, const char* file_path,
                                     const uint32_t* deleted_rows,
                                     uint64_t n_deleted_rows) noexcept {
    assert(th != nullptr && file_path != nullptr);
    assert(deleted_rows != nullptr || n_deleted_rows == 0);
    if (n_deleted_rows == 0) return false;
    for (uint64_t i = 1; i < n_deleted_rows; ++i) {   // bounded: caller-sized
        if (deleted_rows[i] <= deleted_rows[i - 1]) return false;
    }

    Arena scratch;
    FindCtx ctx{};
    ctx.target_path = file_path;
    ctx.version = -1;
    if (!delta_log_walk_all(&th->os, th->table_rel, -1, &scratch, &ctx, find_cb))
        return false;
    if (!ctx.alive) return false;   // the file isn't live -- nothing to mark

    uint32_t* merged = nullptr;
    uint64_t merged_n = 0;
    if (ctx.found.has_dv) {
        DeletionVector old_dv{};
        if (!delta_dv_load(&th->os, th->table_rel, &ctx.found.dv, &scratch,
                           &old_dv))
            return false;
        if (!merge_sorted_unique(&old_dv.bitmap, old_dv.present, deleted_rows,
                                 n_deleted_rows, &scratch, &merged, &merged_n))
            return false;
    } else {
        merged = scratch.allocate_array<uint32_t>(n_deleted_rows);
        if (merged == nullptr) return false;
        std::memcpy(merged, deleted_rows, n_deleted_rows * sizeof(uint32_t));
        merged_n = n_deleted_rows;
    }
    if (merged_n == 0) return false;

    DvDescriptor new_dv{};
    if (!delta_dv_write(&th->os, th->table_rel, merged, merged_n, &scratch,
                        &new_dv))
        return false;

    const bool needs_upgrade = !ctx.has_protocol ||
        ctx.protocol.min_reader_version < 3 ||
        ctx.protocol.min_writer_version < 7;

    const uint32_t cap = 8u * 1024u + 2u * static_cast<uint32_t>(sizeof(ctx.found.stats_json));
    char* body = scratch.allocate_array<char>(cap);
    if (body == nullptr) return false;
    uint32_t off = 0;
    const uint64_t ms = delta_writer_now_ms();
    if (needs_upgrade) {
        const bool needs_invariants = ctx.has_metadata &&
            metadata_schema_has_not_null(&ctx.metadata);
        off = emit_protocol_upgrade(body, cap, off, needs_invariants);
    }
    off = emit_remove_reconcile(body, cap, off, file_path, ms, &ctx.found);
    off = emit_add_reconcile(body, cap, off, &ctx.found, ms, &new_dv);
    const int wn = std::snprintf(body + off, cap - off,
        "{\"commitInfo\":{\"timestamp\":%llu,\"operation\":\"DELETE\"}}\n",
        static_cast<unsigned long long>(ms));
    if (wn > 0) off += static_cast<uint32_t>(wn);

    int64_t base = ctx.version;
    return delta_writer_commit_raw(th, body, off, &base);
}

}  // namespace delta
}  // namespace lakehouse
}  // namespace bolt
