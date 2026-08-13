// bolt/lakehouse/iceberg/manifest_avro.h — Iceberg manifest + manifest-list
// AVRO writers. The mirror of `manifest_parse_avro` / `manifest_list_parse_avro`
// in manifest.h, and verified against them: what we write, bolt reads back.
//
// Scope, stated plainly:
//
//   * **format-version 2, unpartitioned, uncompressed (codec = null).** A
//     partition spec with fields is REJECTED rather than written wrong — the
//     partition tuple is a nested record whose shape comes from the spec, and
//     emitting a mismatched one produces a manifest that parses and then lies.
//     Readers accept a null codec; the reference writer's `deflate` is not
//     required.
//
//   * **Every optional repeated field is written as null** — `column_sizes`,
//     `value_counts`, `lower_bounds`, `split_offsets`, ... These carry
//     pruning statistics: a reader that has them can skip files without
//     opening them. Writing null is CORRECT (they are optional) but forfeits
//     that pruning, so a scan reads every file it is given. Measured safe
//     against pyiceberg 0.11.1, which accepts an all-null data_file and falls
//     back to reading the files. Populating them is a later, separate step.
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

// Write `n_entries` manifest-list entries (the snapshot's `snap-*.avro`).
// `partitions` and `key_metadata` are written null; the per-manifest row
// counts a reader would use for pruning are written as 0 with the file counts
// carried from `ManifestListEntry::added_files_count`.
bool manifest_list_write_avro(const ManifestListEntry* entries,
                              uint32_t n_entries, int64_t snapshot_id,
                              int64_t sequence_number, Arena* scratch,
                              const uint8_t** out, uint64_t* out_len) noexcept;

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
