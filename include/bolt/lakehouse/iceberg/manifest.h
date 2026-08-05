// bolt/lakehouse/iceberg/manifest.h — Iceberg manifest + manifest-list PODs.
//
// REAL Iceberg manifests are AVRO, and bolt now reads them: see
// `manifest_list_parse_avro` / `manifest_parse_avro` below. Proven end-to-end
// against files written by **pyiceberg 0.11.1** against a real catalog —
// format-version 2, identity-partitioned, deflate-coded — in
// tests/test_bolt_iceberg_real_avro.cpp, which asserts VALUES from the source
// table (bounds, null counts, partition values), not row counts. The
// pre-change reader fails at the header on those same bytes.
//
// The three things that had to land for that (G2FEAT-76):
//   1. `union{null, array<...>}`. Iceberg declares EVERY optional repeated
//      field that way (lower_bounds, split_offsets, partitions, ...), and the
//      schema parser used to fail closed on a container inside a union — so a
//      real manifest did not parse AT ALL. Now modelled as a nullable
//      container; an absent one still steps over its element descriptors.
//   2. Per-ELEMENT value delivery (`bolt::ingest::avro_read_ex` +
//      `AvroElemFn`). Containers were walked correctly but published null, so
//      a repeated field's N values were unreachable. FileStats and
//      equality_ids are exactly those repeated parts.
//   3. The deflate codec. `write.avro.compression-codec` defaults to
//      gzip/deflate, so real manifests are deflate-coded; bolt grew its own
//      RFC-1951 inflate (bolt/ingest/bolt_inflate.h) because zlib is an
//      optional dependency and its exact-fill API cannot be used for an Avro
//      block, which declares no decompressed size.
//
// `BOLT_ICEBERG_MANIFEST_JSON` below is **documentation, not a switch** —
// nothing reads it. The JSON entry points remain for the JSON-shaped fixtures
// the W4 tests use; new callers reading real tables want the Avro ones.
//
// G2FEAT-125 wired the Avro entry points into the SCAN (`iceberg_scan.cpp`),
// which until then still called the JSON ones — bolt could open a real table
// but not query it. The scan picks a parser from the OCF magic in the bytes it
// just read, never from this macro and never from a file extension, so a table
// may legitimately hold both forms. That end-to-end path (metadata -> manifests
// -> Parquet -> rows, values checked against pyiceberg's own scan of the same
// table) is `tests/test_bolt_iceberg_real_scan.cpp`.
//
// KNOWN GAPS in the Avro path (fail closed or degrade, never silently wrong):
//   - `PartitionValue::field_id` is the ORDINAL within the partition struct,
//     not the Avro "field-id" annotation (the flattened field table does not
//     model field-ids). The partition struct's order IS the spec's order, so
//     the ordinal is the stable join key against a PartitionSpec — which is
//     what `DataFileRef::partition_ordinal_ids` tells `partition_passes` to
//     use. Joining the ordinal against a v2 spec's field-id (>= 1000) matches
//     nothing, so before that flag existed partition pruning silently did
//     NOTHING for a real manifest.
//   - Bounds are truncated at kLakeMaxValBytes (64); `lower_len`/`upper_len`
//     report the stored length, and `DataFileRef::binary_bounds` says they are
//     Iceberg's binary single-value encoding rather than ASCII.
//   - Per-column stats past kIcebergMaxStatCols (32) are dropped — stats are a
//     pruning aid, so this degrades pruning, never correctness.
//   - `zstd`-coded OCFs are unsupported (null / deflate / snappy are).
//
// JSON Manifest-list:
//   [{"manifest_path", "manifest_length", "partition_spec_id", "content",
//     "added_snapshot_id", "added_files_count"}, ...]
//
// JSON Manifest:
//   [{"status", "snapshot_id",
//     "data_file": {"content", "file_path", "file_format", "partition",
//                   "record_count", "file_size_in_bytes",
//                   "lower_bounds", "upper_bounds", "null_value_counts",
//                   "equality_ids"}}, ...]

#pragma once

#include <cstddef>
#include <cstdint>

#include "bolt/lakehouse/iceberg/delete_file.h"
#include "bolt/lakehouse/iceberg/partition.h"
#include "bolt/lakehouse/iceberg/snapshot.h"
#include "bolt/lakehouse/iceberg/statistics.h"

#ifndef BOLT_ICEBERG_MANIFEST_JSON
#define BOLT_ICEBERG_MANIFEST_JSON 1
#endif

namespace bolt {
namespace lakehouse {
namespace iceberg {

static constexpr uint32_t kIcebergMaxManifestEntries  = 4096u;
static constexpr uint32_t kIcebergMaxManifestsPerList = 256u;
static constexpr uint32_t kIcebergMaxPartitionValues  = 8u;

enum class ManifestStatus : uint8_t {
    kExisting = 0,
    kAdded    = 1,
    kDeleted  = 2,
    kUnknown  = 3,
};

struct ManifestListEntry {
    int32_t      partition_spec_id;
    FileContent  content;
    uint8_t      _pad[3];
    int64_t      manifest_length;
    int64_t      added_snapshot_id;
    int64_t      added_files_count;
    char         manifest_path[kIcebergMaxManifestPath];
};

struct PartitionValue {
    int32_t  field_id;
    bool     is_int;
    bool     is_str;
    bool     is_null;
    uint8_t  _pad;
    int64_t  i64;
    char     str[kLakeMaxValBytes];
};

struct DataFileRef {
    ManifestStatus status;
    FileContent    content;
    uint8_t        _pad[2];
    int32_t        partition_spec_id;
    int64_t        snapshot_id;
    char           file_path[kIcebergMaxPath];
    PartitionValue partition[kIcebergMaxPartitionValues];
    uint32_t       n_partition;
    // How to READ the two fields above and `stats`, set by whichever parser
    // produced this record. Without them a consumer cannot tell an Avro record
    // from a JSON one, and both pruning paths join on exactly that difference
    // (G2FEAT-125). Carved from the former `_pad2`; the size is pinned below.
    //
    // `partition_ordinal_ids`: `PartitionValue::field_id` is the ORDINAL within
    // the partition struct, not the spec's field-id — the Avro schema's
    // "field-id" annotations are not modelled. The struct's order IS the spec's
    // order, so `partition[i]` belongs to `PartitionSpec::fields[i]`.
    //
    // `binary_bounds`: `lower`/`upper` hold Iceberg's BINARY single-value
    // encoding (little-endian, `lower_len`/`upper_len` bytes) rather than the
    // ASCII the JSON fixtures carry. 250.5 is 00 00 00 00 00 50 6F 40 — it
    // starts with a NUL, so any C-string read of it yields empty.
    bool           partition_ordinal_ids;
    bool           binary_bounds;
    uint8_t        _pad2[2];
    FileStats      stats;
    int32_t        equality_ids[8];
    uint32_t       n_equality_ids;
    uint32_t       _pad3;
};
// Pinned: the provenance flags came out of the 4 bytes of padding that used to
// follow `n_partition`, so nothing after them moved and the struct is the same
// size it was before G2FEAT-125.
static_assert(offsetof(DataFileRef, stats) ==
                  offsetof(DataFileRef, n_partition) + 8u,
              "DataFileRef layout pinned: provenance flags came from padding");

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt

namespace bolt { class Arena; }

namespace bolt {
namespace lakehouse {
namespace iceberg {

// ---------------------------------------------------------------------------
// AVRO manifest readers — the REAL Iceberg on-disk form.
//
// A manifest list (`snap-<id>-<n>-<uuid>.avro`) and a manifest
// (`<uuid>-m<n>.avro`) are Avro object container files, normally
// deflate-compressed (`write.avro.compression-codec` defaults to gzip). These
// read them directly via `bolt::ingest::avro_read_ex`, binding schema fields BY
// NAME against the flattened field table, so Iceberg v1 and v2 layouts and
// differing field ORDER all work without a positional assumption.
//
// Bounded exactly like the JSON entry points: at most `cap` entries, `*out_n`
// set to what was produced. `scratch` holds the decompressed blocks — reset or
// discard it after the call.
// ---------------------------------------------------------------------------

// Parse a manifest-list OCF into ManifestListEntry[].
bool manifest_list_parse_avro(const uint8_t* src, uint64_t len, Arena* scratch,
                              ManifestListEntry* out, uint32_t cap,
                              uint32_t* out_n) noexcept;

// Parse a manifest OCF into DataFileRef[] — including the repeated parts
// (per-column lower/upper bounds and null counts from the array<record<key,
// value>> maps, plus equality_ids), which is what per-element delivery in the
// Avro reader exists to make possible.
bool manifest_parse_avro(const uint8_t* src, uint64_t len, Arena* scratch,
                         int32_t default_spec_id,
                         DataFileRef* out, uint32_t cap,
                         uint32_t* out_n) noexcept;

bool manifest_list_parse_json(const uint8_t* src, uint32_t len, Arena* scratch,
                              ManifestListEntry* out, uint32_t cap,
                              uint32_t* out_n) noexcept;

bool manifest_parse_json(const uint8_t* src, uint32_t len, Arena* scratch,
                         int32_t default_spec_id,
                         DataFileRef* out, uint32_t cap,
                         uint32_t* out_n) noexcept;

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
