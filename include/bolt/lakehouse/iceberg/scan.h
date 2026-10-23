// bolt/lakehouse/iceberg/scan.h — Iceberg read-side public surface (W4).
//
// Opaque table + scan handles defined in iceberg_scan.cpp; this header only
// forward-declares them in the `iceberg` sub-namespace so they don't collide
// with the Delta versions.

#pragma once

#include <cstdint>

#include "bolt/bolt_arena.h"
#include "bolt/lakehouse/catalog.h"
#include "bolt/lakehouse/format.h"
#include "bolt/lakehouse/object_store.h"

namespace bolt {

class Arena;
struct BoltBatch;
struct BoltColumn;

namespace lakehouse {
namespace iceberg {

struct TableHandle;
struct ScanHandle;
struct Metadata;

bool iceberg_table_open(TableHandle** out, Arena* arena, Catalog* catalog,
                        const char* namespace_, const char* name) noexcept;

// Open a table through a caller-supplied store instead of a catalog-rooted
// filesystem. `table_rel` is the table's key prefix in that store (it holds
// `metadata/`); the store is copied into the handle and must outlive it.
bool iceberg_table_open_store(TableHandle** out, Arena* arena,
                              const ObjectStore* store,
                              const char* table_rel) noexcept;

// Open a catalog-served table: `metadata_json` is the TableMetadata document
// a REST catalog's loadTable returned, and every file it names is fetched
// through `store` (S3, GCS, Azure, filesystem). `store->impl` is borrowed and
// must outlive the handle. `table_key_prefix` is the store key of the table's
// `location`; nullptr derives it by dropping the location's scheme and
// authority ("s3://bucket/wh/ns/t" -> "wh/ns/t").
bool iceberg_table_open_on_store(TableHandle** out, Arena* arena,
                                 const ObjectStore* store,
                                 const char* table_key_prefix,
                                 const uint8_t* metadata_json,
                                 uint32_t metadata_len) noexcept;

void iceberg_table_close(TableHandle* h) noexcept;

// The parsed metadata.json behind an open table — schemas, partition specs,
// snapshot history, and the `location` the writer recorded. Read-only and
// owned by the handle's arena; valid until that arena is reset. A caller needs
// this to pick a snapshot for time travel (`ReadOptions::snapshot_id`), which
// is otherwise unanswerable from the public surface.
const Metadata* iceberg_table_metadata(const TableHandle* h) noexcept;

bool iceberg_scan_open(ScanHandle** out, TableHandle* h,
                       const ReadOptions* opts) noexcept;

// Version-capable stores use pinned ranged Parquet reads automatically. The
// convenience open above owns an 8 MiB footer/compressed-chunk workspace and
// 1 MiB transport scratch, allocated once at scan startup. A larger file is
// allowed; a footer or individual compressed chunk exceeding workspace fails.
// Stores without either optional version operation retain the local contiguous
// reader. A partially implemented version interface is refused.
inline constexpr uint32_t kIcebergScanWorkspaceBytes = 8u << 20;
struct IcebergScanIO {
    uint8_t* workspace = nullptr;
    uint64_t workspace_bytes = 0;
    Arena* transport_scratch = nullptr;
    uint32_t request_bytes = 1u << 20;
};
// Explicit borrowed workspace for version-capable stores. Caller exclusively
// owns these disjoint buffers until close; scratch is frozen against growth at
// startup and reset per transport call. Startup probes its bounded reservation.
// Workspace and transport scratch must not overlap the table/output arena or
// each other. Configuration is copied. No fallback, retry or repin on failure.
// The handle output must be disjoint from table/config/workspace/transport
// storage; invalid output aliases are refused unchanged before startup writes.
// The borrowed origin implementation/config remains immutable and outside the
// reused buffers. Its extent is opaque; only its first byte can be checked.
bool iceberg_scan_open_with_io(ScanHandle** out, TableHandle* h,
    const ReadOptions* opts, const IcebergScanIO* io) noexcept;

// What planning cost so far (G2ICE-234). Planning must scale with the
// manifests a predicate touches, not with the table's file count; these
// counters are how a caller checks that.
struct IcebergScanStats {
    uint64_t manifests_listed;       // data manifests in the snapshot's list
    uint64_t manifests_pruned;       // skipped on partition summaries alone
    uint64_t manifests_read;         // data manifests fetched and decoded
    uint64_t delete_manifests_read;
    uint64_t entries_visited;        // data-manifest entries decoded
    uint64_t files_planned;          // live data files that passed pruning
    uint64_t metadata_bytes_read;    // manifest list + every manifest read
};
bool iceberg_scan_stats(const ScanHandle* h, IcebergScanStats* out) noexcept;

// Plan without reading data: the next live data file that passed pruning, in
// scan order. `*out_path` stays valid until the next call. Do not mix with
// iceberg_scan_next_batch on the same handle.
bool iceberg_scan_next_file(ScanHandle* h, const char** out_path,
                            bool* out_eof) noexcept;

bool iceberg_scan_next_batch(ScanHandle* h, BoltBatch* out,
                             bool* out_eof) noexcept;

// Identity of the row group that produced the batch most recently returned by
// iceberg_scan_next_batch (G2ICE-262: lets a consumer key a cache of decoded groups).
//   file_path     the data file's path as planned; valid until the next iceberg_scan_next_batch call
//   row_group     index inside that file's footer
//   file_row_base absolute position of the group's first row in the file (positional deletes)
//   decoded_rows  rows before positional deletes; emitted_rows rows in the batch after them
// Returns false before any batch was produced or on a null handle.
struct IcebergScanGroup {
    const char* file_path;
    uint32_t    row_group;
    uint32_t    _pad;
    uint64_t    file_row_base;
    uint64_t    decoded_rows;
    uint64_t    emitted_rows;
};
bool iceberg_scan_last_group(const ScanHandle* h, IcebergScanGroup* out) noexcept;

// G2ICE-262 — let a caller SUPPLY a decoded row group instead of decoding it (a read-through cache hit).
// `provide` is called once per row group, before decoding, with the group's identity. Returning true means it filled
// cols[0..ncols) (flat columns whose buffers it allocated from `arena`, the scan's scratch arena) with the group's
// rows AS THE SCAN WOULD EMIT THEM: already after positional deletes. It must also set
//   *decoded_rows  rows of the group BEFORE deletes (the scan advances its file row base by this), and
//   *emitted_rows  rows in cols (== the column length).
// The scan then neither decodes the group nor re-applies deletes to it. Returning false decodes normally.
// Only valid for a scan that emits every file column (no source column selection). Copied; nullptr clears.
struct IcebergGroupProvider {
    bool (*provide)(void* ctx, const char* file_path, uint32_t row_group, Arena* arena, BoltColumn* cols,
                    uint32_t ncols, int64_t* decoded_rows, int64_t* emitted_rows) noexcept;
    void* ctx;
};
void iceberg_scan_set_group_provider(ScanHandle* h, const IcebergGroupProvider* provider) noexcept;

void iceberg_scan_close(ScanHandle* h) noexcept;

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
