// bolt/lakehouse/iceberg/manifest_avro.h — Iceberg manifest + manifest-list
// AVRO writers. The mirror of `manifest_parse_avro` / `manifest_list_parse_avro`
// in manifest.h, and verified against them: what we write, bolt reads back.
//
// Scope, stated plainly:
//
//   * **format-version 2, uncompressed (codec = null).** `manifest_write_avro`
//     is unpartitioned and REJECTS a partitioned file rather than writing it
//     wrong; `manifest_write_avro_partitioned` derives the nested partition
//     record from the spec itself, so the tuple shape and the spec cannot
//     disagree. Only integral partition results are written.
//     Readers accept a null codec; the reference writer's `deflate` is not
//     required.
//
//   * `null_value_counts`, `lower_bounds` and `upper_bounds` carry the
//     entry's `stats` when `stats.n_cols > 0` (G2ICE-135; a parquet footer
//     folds into them via `file_stats_from_parquet_meta`, G2ICE-43). Every
//     other optional repeated field (`column_sizes`, `value_counts`,
//     `nan_value_counts`, `split_offsets`, ...) is written null — legal, it
//     only forfeits the pruning those fields would allow.
//
//   * `record_count` and `file_size_in_bytes` are REQUIRED and are trusted by
//     readers without verification. `record_count` in particular answers
//     `COUNT(*)` straight from metadata, so a wrong value corrupts query
//     results silently, with no error raised. The caller owns their accuracy.
//
// Determinism: the OCF sync marker is a fixed constant, not random, so the
// same inputs produce byte-identical output. Callers that cache or compare
// synthesized manifests depend on this.

#pragma once

#include "bolt/lakehouse/iceberg/manifest.h"

#include <cstdint>

namespace bolt {
namespace lakehouse {
namespace iceberg {

// Write `n_files` manifest entries as a single-block Avro OCF into the scratch
// arena. `*out` points into the arena and lives as long as it does.
//
// `table_schema_json` is the Iceberg table schema (the `{"type":"struct",...}`
// form) copied verbatim into the OCF `schema` metadata key, where readers look
// for it. `partition_spec_id` must name an EMPTY spec; see the scope note.
//
// Returns false on: a null argument, more than `kIcebergMaxManifestEntries`
// files, a partitioned `DataFileRef`, or an arena too small for the encoding.
bool manifest_write_avro(const DataFileRef* files, uint32_t n_files,
                         int64_t snapshot_id, int64_t sequence_number,
                         const char* table_schema_json,
                         uint32_t table_schema_len,
                         int32_t partition_spec_id, Arena* scratch,
                         const uint8_t** out, uint64_t* out_len) noexcept;

// Result type of one partition-tuple field. Only the integral results are
// written: identity/truncate over int/long, bucket, and year/month/day/hour.
enum class PartitionAvroType : uint8_t {
    kInt  = 0,
    kLong = 1,
};

// The partitioned form of `manifest_write_avro`. `spec` supplies the tuple's
// field names, field-ids and transforms (in spec order); `result_types` holds
// `spec->n_fields` entries. Every file must carry exactly `spec->n_fields`
// partition values, each integral or null, with `partition[i]` belonging to
// `spec->fields[i]`. The manifest's `partition` record and its
// `partition-spec` / `partition-spec-id` metadata are derived from `spec`, so
// the three cannot disagree. A spec with zero fields writes byte-identically
// to `manifest_write_avro`.
//
// Returns false on anything `manifest_write_avro` refuses, a field name that
// is not an Avro identifier, a transform with no spelling, a string partition
// value, or a file whose tuple arity differs from the spec's.
bool manifest_write_avro_partitioned(const DataFileRef* files, uint32_t n_files,
                                     int64_t snapshot_id,
                                     int64_t sequence_number,
                                     const char* table_schema_json,
                                     uint32_t table_schema_len,
                                     const PartitionSpec* spec,
                                     const PartitionAvroType* result_types,
                                     Arena* scratch, const uint8_t** out,
                                     uint64_t* out_len) noexcept;

// Write `n_entries` manifest-list entries (the snapshot's `snap-*.avro`).
// `partitions` and `key_metadata` are written null; the six required
// added/existing/deleted file+row counts (G2ICE-49) are written from the
// matching `ManifestListEntry` fields, which the caller must have populated
// with real tallies -- these are trusted by readers (pyiceberg's
// `inspect.manifests()`, DuckDB, snapshot-summary rollups) without opening
// the manifest or its data files, so a wrong value here is a silently wrong
// COUNT(*), exactly like a data file's own `record_count`.
bool manifest_list_write_avro(const ManifestListEntry* entries,
                              uint32_t n_entries, int64_t snapshot_id,
                              int64_t sequence_number, Arena* scratch,
                              const uint8_t** out, uint64_t* out_len) noexcept;

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
