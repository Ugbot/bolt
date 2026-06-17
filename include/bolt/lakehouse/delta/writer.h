// bolt/lakehouse/delta/writer.h — W3 Delta write public surface.
//
// Append + UPDATE + DELETE + MERGE + RESTORE + VACUUM + checkpoint over the
// existing W2 read primitives (snapshot/log/handle/ObjectStore + ParquetWriter).
//
// Tiger Style: PODs, fixed caps, ≥2 asserts/fn, ≤70 line fns, no exceptions,
// all alloc via Arena, all commits via os_put + head_object conflict check.

#pragma once

#include <cassert>
#include <cstdint>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/lakehouse/catalog.h"
#include "bolt/lakehouse/format.h"
#include "bolt/lakehouse/handle.h"
#include "bolt/lakehouse/object_store.h"

namespace bolt {
namespace lakehouse {
namespace delta {

static constexpr uint32_t kWriteMaxPartCols = 8u;
static constexpr uint32_t kWriteMaxColName  = 64u;
static constexpr int32_t  kWriteDefaultFileMb   = 128;
static constexpr int32_t  kWriteDefaultRgRows   = 65536;
static constexpr int32_t  kWriteDefaultCkptIv   = 10;
static constexpr uint32_t kWriteMaxBatches      = 1024u;
static constexpr uint32_t kWriteMaxAssignments  = 16u;

struct WriteOptions {
    char        partition_columns[kWriteMaxPartCols][kWriteMaxColName];
    uint32_t    n_partition_columns;
    Compression compression;
    uint8_t     _pad0[3];
    int32_t     target_file_size_mb;
    int32_t     target_row_group_rows;
    bool        emit_stats;
    bool        enable_deletion_vectors;
    uint8_t     _pad1[2];
    int32_t     checkpoint_interval;
};

inline void write_options_init(WriteOptions* o) noexcept {
    assert(o != nullptr);
    if (o == nullptr) return;
    uint8_t* p = reinterpret_cast<uint8_t*>(o);
    for (uint32_t i = 0; i < sizeof(WriteOptions); ++i) p[i] = 0;
    o->compression             = Compression::kSnappy;
    o->target_file_size_mb     = kWriteDefaultFileMb;
    o->target_row_group_rows   = kWriteDefaultRgRows;
    o->emit_stats              = true;
    o->checkpoint_interval     = kWriteDefaultCkptIv;
}

// One column = one literal value (UPDATE SET expr).
struct Assignment {
    char  column[kWriteMaxColName];
    BoltType type;
    uint8_t _pad[3];
    int64_t  i64;
    double   f64;
    char     str[kLakeMaxValBytes];
    uint32_t str_len;
    uint32_t _pad2;
};

// MERGE source row stored row-by-row. POD; bounded by 1M source rows enforced
// by table_merge.
struct MergeRow {
    int64_t key;        // ON key value (scalar Int64 only for v1).
    int64_t value;      // payload (single Int64 column for v1).
    bool    has_value;
    uint8_t _pad[7];
};

enum class MergeWhenMatched : uint8_t { kUpdate = 0, kDelete = 1, kNone = 2 };
enum class MergeWhenNotMatched : uint8_t { kInsert = 0, kNone = 1 };

struct MergeSpec {
    char             key_column[kWriteMaxColName];
    char             value_column[kWriteMaxColName];
    const MergeRow*  source;
    uint32_t         n_source;
    MergeWhenMatched    matched_action;
    MergeWhenNotMatched not_matched_action;
    uint8_t          _pad[2];
};

struct OptimizeOptions {
    char     zorder_columns[4][kWriteMaxColName];
    uint32_t n_zorder_columns;
    int32_t  target_file_size_mb;
    bool     run_bin_pack;
    uint8_t  _pad[3];
};

inline void optimize_options_init(OptimizeOptions* o) noexcept {
    assert(o != nullptr);
    if (o == nullptr) return;
    uint8_t* p = reinterpret_cast<uint8_t*>(o);
    for (uint32_t i = 0; i < sizeof(OptimizeOptions); ++i) p[i] = 0;
    o->target_file_size_mb = kWriteDefaultFileMb;
    o->run_bin_pack = true;
}

// ---------------------------------------------------------------------------
// Public surface
// ---------------------------------------------------------------------------

bool delta_table_create(TableHandle** out, Arena* arena, Catalog* cat,
                        const char* ns, const char* name,
                        const BoltSchema* schema,
                        const WriteOptions* opts) noexcept;

struct AppendHandle;

bool delta_append_open(AppendHandle** out, TableHandle* th) noexcept;
bool delta_append_write(AppendHandle* ah, const BoltBatch* batch) noexcept;
bool delta_append_commit(AppendHandle* ah) noexcept;
void delta_append_close(AppendHandle* ah) noexcept;

bool delta_table_update(TableHandle* th, const Predicate* pred,
                        const Assignment* assigns, uint32_t n) noexcept;

bool delta_table_delete(TableHandle* th, const Predicate* pred) noexcept;

bool delta_table_merge(TableHandle* th, const MergeSpec* spec) noexcept;

bool delta_table_restore(TableHandle* th, int64_t target_version) noexcept;

}  // namespace delta
}  // namespace lakehouse
}  // namespace bolt
