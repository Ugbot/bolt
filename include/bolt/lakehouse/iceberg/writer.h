// bolt/lakehouse/iceberg/writer.h — W5 Iceberg write public surface.
//
// Path-based open (caller passes a filesystem root); per-table opaque
// TableHandle. Append/overwrite/delete/rewrite/expire/refs/views — the full
// W5 surface. Bounded caps: 256 manifests/snapshot, 64 snapshots, 16 refs.
//
// Tiger Style: PODs, ≥2 asserts/fn, no exceptions, no smart pointers, all
// allocations through the caller-owned Arena, atomic commits via
// ObjectStore put_if_absent emulation (head_object then put).

#pragma once

#include <cassert>
#include <cstdint>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"
#include "bolt/lakehouse/format.h"
#include "bolt/lakehouse/iceberg/delete_file.h"
#include "bolt/lakehouse/iceberg/metadata.h"
#include "bolt/lakehouse/iceberg/partition.h"
#include "bolt/lakehouse/iceberg/snapshot.h"
#include "bolt/lakehouse/iceberg/sort_order.h"
#include "bolt/lakehouse/object_store.h"

namespace bolt {
namespace lakehouse {
namespace iceberg {

static constexpr uint32_t kIcebergMaxRefs        = 16u;
static constexpr int32_t  kIcebergWriteDefaultMb  = 128;
static constexpr int32_t  kIcebergWriteDefaultRg  = 65536;

struct WriteOptions {
    char         partition_spec_name[64];
    char         sort_order_name[64];
    Compression  compression;
    uint8_t      _pad0[3];
    int32_t      target_file_size_mb;
    int32_t      target_row_group_rows;
    bool         emit_stats;
    bool         use_v2_format;
    uint8_t      _pad1[2];
};

inline void write_options_init(WriteOptions* o) noexcept {
    assert(o != nullptr);
    if (o == nullptr) return;
    uint8_t* p = reinterpret_cast<uint8_t*>(o);
    for (uint32_t i = 0; i < sizeof(WriteOptions); ++i) p[i] = 0;
    o->compression           = Compression::kSnappy;
    o->target_file_size_mb   = kIcebergWriteDefaultMb;
    o->target_row_group_rows = kIcebergWriteDefaultRg;
    o->emit_stats            = true;
    o->use_v2_format         = true;
}

struct OptimizeOptions {
    int32_t  target_file_size_mb;
    bool     run_bin_pack;
    uint8_t  _pad[3];
};

inline void optimize_options_init(OptimizeOptions* o) noexcept {
    assert(o != nullptr);
    if (o == nullptr) return;
    o->target_file_size_mb = kIcebergWriteDefaultMb;
    o->run_bin_pack        = true;
}

// Opaque handles defined in iceberg_writer.cpp.
struct TableHandle;
struct AppendHandle;

// Create a new Iceberg table rooted at `path` (filesystem root). Writes
// metadata/v1.metadata.json + metadata/version-hint.text. Returns false on
// any IO error or capacity overflow.
bool table_create(TableHandle** out, Arena* arena, ObjectStore* os,
                  const char* path, const Schema* schema,
                  const PartitionSpec* spec, const SortOrder* sort,
                  const WriteOptions* opts) noexcept;

// Open an existing Iceberg table (reads latest metadata.json).
bool table_open(TableHandle** out, Arena* arena, ObjectStore* os,
                const char* path) noexcept;

void table_close(TableHandle* h) noexcept;

// Append path: open → write N batches → commit. Each commit emits one new
// data Parquet file + a manifest + a manifest list + a new metadata.json.
bool append_open  (AppendHandle** out, TableHandle* th) noexcept;
bool append_write (AppendHandle* ah, const BoltBatch* batch) noexcept;
bool append_commit(AppendHandle* ah) noexcept;
// As append_commit, and additionally reports the ABSOLUTE path of the data
// file the commit wrote and its row count. A caller that wants to delete rows
// from what it just appended needs both: an Iceberg positional delete names a
// (file_path, pos) pair, and the path is otherwise only recoverable by parsing
// the manifest back. `out_path` may be null; `out_rows` may be null.
bool append_commit_ex(AppendHandle* ah, char* out_path, uint32_t out_path_cap,
                      int64_t* out_rows) noexcept;
void append_close (AppendHandle* ah) noexcept;

// Commit `n` Iceberg v2 POSITIONAL deletes as a new delete manifest on top of
// the current snapshot. Each entry names a data file (the absolute path a
// commit recorded) and a 0-based row position within it; a reader drops those
// rows. Entries need not be sorted and need not target one file.
//
// Positional rather than equality deletes deliberately -- see the note in
// iceberg_writer.cpp. Late deletes are supported by construction: the delete
// commits at a strictly higher sequence number than any data already in the
// table, so it applies to that data. The caller owns finding the position of a
// row it did not just write.
//
// Returns false on: a null/empty entry list, an empty path, a negative
// position, more than kIcebergMaxManifestEntries entries, or a failed commit.
// It never partially applies: either the snapshot publishes or nothing changes.
bool table_delete_positions(TableHandle* th, const PositionDeleteEntry* dels,
                            uint32_t n) noexcept;

// Overwrite: drop files matching the predicate and append new data.
bool table_overwrite(TableHandle* th, const Predicate* pred,
                     const BoltBatch* batch) noexcept;

// Rewrite (compaction). Files smaller than target/2 are coalesced.
bool table_rewrite(TableHandle* th, const OptimizeOptions* opts) noexcept;

// Soft delete (snapshot kDelete) — predicate-based.
bool table_delete(TableHandle* th, const Predicate* pred) noexcept;

// Schema evolution. All update the current Schema in metadata.json.
bool table_add_column   (TableHandle* th, const char* name, BoltType type,
                          bool nullable) noexcept;
bool table_drop_column  (TableHandle* th, const char* name) noexcept;
bool table_rename_column(TableHandle* th, const char* from,
                          const char* to) noexcept;

// Partition / sort-order evolution.
bool table_update_partition_spec(TableHandle* th,
                                  const PartitionSpec* spec) noexcept;
bool table_update_sort_order    (TableHandle* th,
                                  const SortOrder* sort) noexcept;

// Compaction (alias for table_rewrite with target_file_size_bytes).
bool table_compact(TableHandle* th, int64_t target_file_size_bytes) noexcept;

// Snapshot lifecycle.
bool table_expire_snapshots(TableHandle* th, uint64_t older_than_ms,
                            int32_t retain_last_n) noexcept;

// Identify (and optionally delete) orphan files. `dry_run` populates out[][512]
// with up to `cap` paths; *n is the total found.
bool table_remove_orphans(TableHandle* th, uint64_t older_than_ms, bool dry_run,
                          char (*out)[512], uint32_t cap,
                          uint32_t* n) noexcept;

// Branches + tags (named snapshot refs).
bool table_create_branch(TableHandle* th, const char* name,
                         int64_t base_snapshot_id) noexcept;
bool table_create_tag   (TableHandle* th, const char* name,
                         int64_t snapshot_id) noexcept;
bool table_drop_ref     (TableHandle* th, const char* name) noexcept;

// Public accessors for tests.
const Metadata* table_metadata(const TableHandle* th) noexcept;

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
