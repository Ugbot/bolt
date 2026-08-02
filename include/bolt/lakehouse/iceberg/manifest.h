// bolt/lakehouse/iceberg/manifest.h — Iceberg manifest + manifest-list PODs.
//
// IMPORTANT (W4 deviation): Iceberg manifests are Avro files with nested
// `data_file` records. W4 therefore reads manifests **as JSON arrays** —
// guarded by `BOLT_ICEBERG_MANIFEST_JSON`.
//
// STATUS UPDATE: the original blocker is GONE. `bolt::ingest::avro_read` now
// decodes nested records (they flatten positionally into dotted leaves such as
// "data_file.record_count") and walks arrays and maps, verified against an
// Iceberg-manifest-shaped OCF written by the stock Apache Avro JAVA writer —
// the same stack Iceberg's ManifestWriter uses (tests/data/
// golden_avro_manifest.avro, BoltAvroIceberg.RealJavaWrittenManifestShapeReads).
//
// ONE shape still blocks a genuine manifest, and it was MEASURED rather than
// assumed. Iceberg encodes its INT-KEYED maps — column_sizes, value_counts,
// null_value_counts, lower_bounds, upper_bounds — as array<record<key,value>>
// (Avro maps only take string keys). That writer emits an array as a POSITIVE
// element count with NO byte size, so a record element cannot be skipped: there
// is nothing to jump over and the element width is unknowable. bolt fails
// closed there rather than mis-aligning every following field
// (BoltAvroIceberg.RealJavaArrayOfRecordFailsClosed pins it against a real
// Java-written file).
//
// So this flag stays ON. Flipping it needs element-aware descent into an
// array<record> — a real increment, not a switch. TODO(W5-avro-nested-reader).
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
    uint32_t       _pad2;
    FileStats      stats;
    int32_t        equality_ids[8];
    uint32_t       n_equality_ids;
    uint32_t       _pad3;
};

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt

namespace bolt { class Arena; }

namespace bolt {
namespace lakehouse {
namespace iceberg {

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
