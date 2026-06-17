// bolt/lakehouse/delta/snapshot.h — replay actions into a live-file set.
// Tiger Style.

#pragma once

#include <cstddef>
#include <cstdint>

#include "bolt/bolt_arena.h"
#include "bolt/lakehouse/delta/log.h"
#include "bolt/lakehouse/format.h"
#include "bolt/lakehouse/object_store.h"

namespace bolt {
namespace lakehouse {
namespace delta {

struct FileStats {
    int64_t  num_records;
    bool     has_num_records;
    uint8_t  _pad[7];
    char     col_names[16][kLakeMaxColName];
    char     min_str[16][kLakeMaxValBytes];
    char     max_str[16][kLakeMaxValBytes];
    int64_t  null_counts[16];
    uint32_t n_cols;
    uint32_t _pad2;
};

struct LiveFile {
    char         path[kDeltaMaxPath];
    int64_t      size;
    PartitionKV  partition_values[kDeltaMaxPartitions];
    uint32_t     n_partition_values;
    bool         has_dv;
    uint8_t      _pad[3];
    DvDescriptor dv;
    FileStats    stats;
};

struct Snapshot {
    LiveFile*     files;
    uint32_t      n_files;
    uint32_t      n_files_cap;
    int64_t       version;
    DeltaProtocol protocol;
    DeltaMetadata metadata;
    bool          has_metadata;
    bool          has_protocol;
    uint8_t       _pad[6];
};

bool delta_snapshot_build(ObjectStore* os, const char* table_rel_prefix,
                          int64_t max_version, Arena* arena,
                          Snapshot* out) noexcept;

bool delta_parse_stats(const char* stats_json, uint32_t len, Arena* scratch,
                       FileStats* out) noexcept;

bool delta_file_passes(const LiveFile* f, const Predicate* preds,
                       uint32_t n_preds) noexcept;

}  // namespace delta
}  // namespace lakehouse
}  // namespace bolt
