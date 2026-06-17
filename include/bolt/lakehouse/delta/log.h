// bolt/lakehouse/delta/log.h — _delta_log NDJSON action parser.
// Tiger Style: PODs, fixed caps, no exceptions, Arena allocs.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "bolt/bolt_arena.h"
#include "bolt/lakehouse/format.h"
#include "bolt/lakehouse/object_store.h"

namespace bolt {
namespace lakehouse {
namespace delta {

enum class ActionKind : uint8_t {
    kUnknown = 0, kProtocol = 1, kMetadata = 2, kAdd = 3, kRemove = 4,
    kCommitInfo = 5, kCdc = 6, kDomainMetadata = 7, kTxn = 8,
};

enum class ColumnMappingMode : uint8_t { kNone = 0, kName = 1, kId = 2, };

enum class DvStorageType : uint8_t {
    kNone = 0, kUuid = 1, kInline = 2, kPath = 3,
};

static constexpr uint32_t kDeltaMaxPath        = 1024u;
static constexpr uint32_t kDeltaMaxPartitions  = 8u;
static constexpr uint32_t kDeltaMaxStatsBytes  = 4096u;
static constexpr uint32_t kDeltaMaxDvInline    = 1024u;
static constexpr uint32_t kDeltaMaxSchemaBytes = 8192u;

struct DvDescriptor {
    DvStorageType type;
    uint8_t       _pad[7];
    char          uuid_or_path[kDeltaMaxPath];
    uint8_t       inline_bytes[kDeltaMaxDvInline];
    uint32_t      inline_len;
    int64_t       size_in_bytes;
    int64_t       cardinality;
    int64_t       offset;
};

struct PartitionKV {
    char key[kLakeMaxColName];
    char value[kLakeMaxValBytes];
    bool is_null;
    uint8_t _pad[7];
};

struct DeltaAdd {
    char         path[kDeltaMaxPath];
    int64_t      size;
    int64_t      modification_time;
    bool         data_change;
    uint8_t      _pad[7];
    PartitionKV  partition_values[kDeltaMaxPartitions];
    uint32_t     n_partition_values;
    uint32_t     _pad2;
    char         stats_json[kDeltaMaxStatsBytes];
    uint32_t     stats_len;
    bool         has_dv;
    uint8_t      _pad3[3];
    DvDescriptor dv;
};

struct DeltaRemove {
    char    path[kDeltaMaxPath];
    int64_t deletion_timestamp;
    bool    data_change;
    uint8_t _pad[7];
};

struct DeltaProtocol {
    int32_t min_reader_version;
    int32_t min_writer_version;
    bool    has_column_mapping;
    uint8_t _pad[7];
};

struct DeltaMetadata {
    char              id[64];
    char              name[kLakeMaxColName];
    char              schema_string[kDeltaMaxSchemaBytes];
    uint32_t          schema_len;
    char              partition_columns[kLakeMaxPartCols][kLakeMaxColName];
    uint32_t          n_partition_columns;
    int64_t           created_time;
    ColumnMappingMode column_mapping;
    uint8_t           _pad[7];
};

struct DeltaCommitInfo {
    int64_t timestamp;
    char    operation[64];
    bool    has_timestamp;
    uint8_t _pad[7];
};

struct DeltaAction {
    ActionKind kind;
    uint8_t    _pad[7];
    int64_t    version;
    union {
        DeltaAdd        add;
        DeltaRemove     rem;
        DeltaProtocol   protocol;
        DeltaMetadata   metadata;
        DeltaCommitInfo commit_info;
    };
};

using ActionFn = bool (*)(void* ctx, const DeltaAction* a) noexcept;

bool delta_log_parse_commit(const uint8_t* src, uint64_t src_len,
                            int64_t version, Arena* scratch,
                            void* ctx, ActionFn cb) noexcept;

bool delta_log_walk_all(ObjectStore* os, const char* table_rel_prefix,
                        int64_t max_version, Arena* scratch,
                        void* ctx, ActionFn cb) noexcept;

bool delta_log_version_for_timestamp(ObjectStore* os,
                                     const char* table_rel_prefix,
                                     int64_t timestamp_ms, Arena* scratch,
                                     int64_t* out_version) noexcept;

}  // namespace delta
}  // namespace lakehouse
}  // namespace bolt
