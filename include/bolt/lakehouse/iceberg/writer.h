// bolt/lakehouse/iceberg/writer.h — W5 Iceberg write public surface.
//
// Location-based open (filesystem path or URI); per-table opaque
// TableHandle. Append/overwrite/delete/rewrite/expire/refs/views — the full
// W5 surface. Bounded caps: 256 manifests/snapshot, 64 snapshots, 16 refs.
//
// Tiger Style: PODs, ≥2 asserts/fn, no exceptions, no smart pointers, all
// allocations through the caller-owned Arena, atomic commits via
// ObjectStore put_if_absent (never a head-then-put emulation).

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
    // G2ICE-283: the format version table_create writes, 0 or 2 = v2, 3 = v3 (row lineage; deletes are deletion vectors).
    uint8_t      format_version;
    uint8_t      _pad1[1];
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

enum class CommitObjectKind : uint8_t { kNone = 0, kManifest, kManifestList, kMetadata, kDataFile };

// Evidence of an uncertain conditional write, copied out of the table arena.
// Keys are store-relative; object_location and table_uuid bind the candidate
// to its table. Reconcile with the EXACT bytes (SHA-256 + length), table UUID,
// version and snapshot identity, not HEAD/size alone. A manifest/list candidate
// means metadata publication was not attempted. A metadata candidate MAY be
// visible. A data-file candidate means no manifest/metadata was attempted;
// its retained local source is available through iceberg/write_io.h.
// Absence at one read is not permission to retry a still-in-flight PUT.
struct CommitCandidate {
    CommitObjectKind kind;
    uint8_t _pad[7];
    uint64_t length;
    int64_t base_metadata_version;
    int64_t candidate_metadata_version;
    int64_t snapshot_id;
    char object_key[kIcebergMaxPath];
    char object_location[kIcebergMaxLocation];
    char table_uuid[kIcebergMaxUuid];
    char sha256[65];
};

// Create a new Iceberg table at canonical location `path`. Metadata uses os;
// data defaults to local files, or opt into iceberg/write_io.h before mutation.
// Writes
// metadata/v1.metadata.json + metadata/version-hint.text. Returns false on
// any IO error or capacity overflow.
bool table_create(TableHandle** out, Arena* arena, ObjectStore* os,
                  const char* path, const Schema* schema,
                  const PartitionSpec* spec, const SortOrder* sort,
                  const WriteOptions* opts) noexcept;

// Same create contract; *out is null on failure. On uncertain initial v1
// publication, out_uncertain receives the candidate despite there being no
// usable handle. Otherwise it is zeroed. No automatic retry/reconciliation.
bool table_create_ex(TableHandle** out, Arena* arena, ObjectStore* os,
                     const char* path, const Schema* schema,
                     const PartitionSpec* spec, const SortOrder* sort,
                     const WriteOptions* opts, CommitCandidate* out_uncertain) noexcept;

// Open an existing Iceberg table (reads latest metadata.json).
bool table_open(TableHandle** out, Arena* arena, ObjectStore* os,
                const char* path) noexcept;

void table_close(TableHandle* h) noexcept;

// Why the most recent metadata-committing call on `th` returned false.
// kConflict: another writer committed a newer table version first (or claimed
// a file name this commit needed). Nothing of this commit is visible; the
// handle is stale and must be reopened (table_open) before retrying.
// kFailed: any other failure (IO, capacity, refused operation).
// kUncertain: a conditional write may have succeeded. Candidate objects are
// retained. This handle refuses further mutations; reconcile externally, then
// reopen. Neither a read nor another mutation clears the uncertainty.
// kUnimplemented: a known spec case this writer does not write yet (a partitioned table, a manifest holding equality
// deletes or deletion vectors that would need rewriting, ...); the refusal names it on stderr and nothing was written.
enum class CommitError : uint8_t { kNone = 0, kConflict = 1, kFailed = 2, kUncertain = 3, kUnimplemented = 4 };
CommitError table_last_commit_error(const TableHandle* th) noexcept;
// False (and zeroes *out) unless this handle has an uncertain candidate.
bool table_uncertain_commit(const TableHandle* th, CommitCandidate* out) noexcept;

// Append path: open → write N batches → commit. Each commit emits one new
// data Parquet file + a manifest + a manifest list + a new metadata.json.
bool append_open  (AppendHandle** out, TableHandle* th) noexcept;
// Named columns may omit fields: apply write-default (else optional null).
// Explicit null never invokes a default. Buffers supplied by the caller stay borrowed.
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

// Row-level delete (G2ICE-280). `pred` is evaluated on the current snapshot's LIVE rows (every delete applied) under
// SQL three-valued logic: a NULL never matches a comparison; kIsNull / kIsNotNull test nullness. Literals: integer
// for integer/date/timestamp/boolean columns, integer or float for float columns, a byte string for string/binary
// columns (unsigned byte order). A data file whose every live row matches is removed; a file with some matching rows
// gets one v2 position-delete file naming exactly those rows; other files are untouched. One snapshot, operation
// "delete". A predicate that matches nothing commits nothing and returns true. A null predicate or one naming no
// column fails (kFailed); an unsupported column/literal pairing or a partitioned table fails with kUnimplemented.
// The commit fails with kConflict if a file it removes is no longer live (another writer committed first).
bool table_delete(TableHandle* th, const Predicate* pred) noexcept;

// table_delete(pred) plus an append of `batch`, in one snapshot with operation "overwrite" (committed even when the
// predicate matches nothing). Under FileWriteIO it is refused when it would need both a delete file and a data file.
bool table_overwrite(TableHandle* th, const Predicate* pred,
                     const BoltBatch* batch) noexcept;

// How a row delete is written (G2ICE-283). kPosition is table_delete's (file_path, pos) files. kEquality writes the
// deleted rows' key values (`equality_ids`, or the schema's identifier-field-ids when n_equality_ids is 0): one file per
// (spec, tuple) of the rows' data files, or with `global` one file under an unpartitioned spec that applies to every
// partition (one is added to the table's specs when it has none). A key column must be a top-level primitive that is not
// float, double or decimal(p > 18). A predicate delete in kEquality mode is refused (kConflict) when a key it deletes is
// also carried by a live row the predicate keeps, in the same scope: an equality delete removes every older row with
// that key. `position_row` adds the deleted row to each position delete, in the optional `row` column (field 2147483544).
enum class DeleteFileKind : uint8_t { kPosition = 0, kEquality = 1 };
struct DeleteOptions {
    DeleteFileKind kind;
    bool           position_row;
    bool           global;
    uint8_t        _pad;
    uint32_t       n_equality_ids;
    int32_t        equality_ids[limits::kMaxEqualityIds];
};
bool table_delete_with(TableHandle* th, const Predicate* pred, const DeleteOptions* opts) noexcept;
bool table_overwrite_with(TableHandle* th, const Predicate* pred, const BoltBatch* batch,
                          const DeleteOptions* opts) noexcept;

// Equality delete by key, without reading the table (an upsert's delete half): every live row whose key columns equal a
// row of `keys` is deleted. `keys` holds exactly the key columns, named as the schema names them; `opts` must be kEquality
// (its equality_ids, or the identifier fields). On a partitioned table the delete is global unless every live data file
// is under the default spec and each of its source columns is a key column; then one file is written per tuple. One
// snapshot, operation "delete"; an empty `keys` commits nothing.
bool table_delete_keys(TableHandle* th, const BoltBatch* keys, const DeleteOptions* opts) noexcept;

// Compaction: live data files smaller than target/2 are rewritten, with every delete that applies to them applied,
// into new files of at most ~target input bytes each, when there are at least two of them or one has position
// deletes. The rewritten files and the position-delete files they fully apply are removed, in one snapshot with
// operation "replace"; the rows the table returns are unchanged. Nothing to do commits nothing and returns true.
// Under FileWriteIO each output is published as it closes; the previous output's local source is released when the next
// one stages (the last keeps its source until the commit resolves), and a commit that definitely fails removes every
// published output (an uncertain one removes none). `opts` null = defaults; run_bin_pack false does nothing.
bool table_rewrite(TableHandle* th, const OptimizeOptions* opts) noexcept;

// Schema evolution. Each call is ADDITIVE (Iceberg spec): it appends a new,
// immutable Schema with a new schema-id to metadata.json and moves
// current-schema-id to it. Prior schemas -- and any snapshot committed under
// one -- are left untouched, so a time-travel read of an old snapshot still
// resolves its original shape (G2ICE-89).
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

// Compaction (table_rewrite with the target in bytes).
bool table_compact(TableHandle* th, int64_t target_file_size_bytes) noexcept;

// Snapshot lifecycle.
bool table_expire_snapshots(TableHandle* th, uint64_t older_than_ms,
                            int32_t retain_last_n) noexcept;

// Iceberg ExpireSnapshots: removes snapshots older than `older_than_ms`, but
// never the current snapshot, a branch/tag head, or the newest `retain_last`
// (>= 1) ancestors of the current snapshot. The snapshot-log is pruned the
// way Java prunes it. With `delete_files`, the expired manifest lists and every
// manifest no retained snapshot reaches are deleted after the commit; data
// files are left to table_remove_orphans.
struct ExpireOptions {
    int64_t  older_than_ms;
    uint32_t retain_last;
    bool     delete_files;
    uint8_t  _pad[3];
};
struct ExpireResult {
    uint32_t expired;
    uint32_t files_deleted;
};
bool table_expire_snapshots_ex(TableHandle* th, const ExpireOptions* o,
                               ExpireResult* r) noexcept;

// Per-table retention policy, persisted as the table properties
// history.expire.max-snapshot-age-ms / history.expire.min-snapshots-to-keep
// and applied after every commit (0 = unset; both 0 = keep everything, which
// is Iceberg's own default without an expiry job).
bool table_set_retention(TableHandle* th, int64_t max_snapshot_age_ms,
                         int32_t min_snapshots_to_keep) noexcept;

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
