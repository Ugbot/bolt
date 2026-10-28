// bolt/lakehouse/format.h — TableFormat enum + Predicate + Schema PODs.
//
// Shared format-agnostic shapes for the lakehouse trait. Deliberately
// minimal: lakehouse callers compose their own ReadOptions / WriteOptions
// from these primitives. Tiger Style: PODs, fixed caps, no exceptions.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "bolt/bolt_types.h"

namespace bolt {
namespace lakehouse {

enum class TableFormat : uint8_t {
    kDelta   = 0,
    kIceberg = 1,
};

enum class Compression : uint8_t {
    kNone   = 0,
    kSnappy = 1,
    kGzip   = 2,
    kZstd   = 3,
    kLz4    = 4,
};

enum class PredicateOp : uint8_t {
    kEq = 0,
    kNe = 1,
    kLt = 2,
    kLe = 3,
    kGt = 4,
    kGe = 5,
    kIsNull    = 6,
    kIsNotNull = 7,
};

static constexpr uint32_t kLakeMaxColName  = 64u;
static constexpr uint32_t kLakeMaxValBytes = 64u;
static constexpr uint32_t kLakeMaxPredicates = 16u;
// Width of the column_prune_plan helper's fixed arrays — the
// source-schema width a lakehouse scan must be able to project/prune,
// coherently with what the Parquet reader beneath it can decode. Raised
// from 64 for G2FEAT-47-class real-data widths (ClickBench `hits` is a
// real 105-flat-leaf-column table; 64 rejected `column_prune_plan` outright
// even with no projection requested). All arrays sized by this constant
// (ReadOptions::projection, ColumnPruneResult::keep,
// RowGroupStats::{min_i64,max_i64,null_count}) are arena- or
// caller-allocated PODs, never stack-local in a hot loop — see
// delta_scan.cpp::ScanHandle / iceberg_scan.cpp::ScanHandle, both
// arena-allocated via `arena->allocate_array<ScanHandle>(1)`. This is an
// assert-ceiling-style cap: `column_prune_plan` rejects (returns false)
// rather than truncating when `n_src > kLakeMaxProjection`.
static constexpr uint32_t kLakeMaxProjection = 128u;
static constexpr uint32_t kLakeMaxPartCols  = 32u;
static constexpr uint32_t kLakeMaxLiveFiles = 4096u;
static constexpr uint32_t kLakeMaxRowGroups = 4096u;
static constexpr uint32_t kLakeMaxCommits   = 512u;
static constexpr uint32_t kLakeMaxExprBytes = 128u;

struct PredicateValue {
    BoltType type;
    uint8_t  _pad[3];
    int64_t  i64;
    double   f64;
    char     str[kLakeMaxValBytes];
    uint32_t str_len;
    uint32_t _pad2;
};

struct Predicate {
    char           column[kLakeMaxColName];
    PredicateOp    op;
    uint8_t        _pad[7];
    PredicateValue value;
};

// Legacy keeps existing ID>=0 / -1-latest behavior. Exact selects the signed
// int64 literally, including -1. Empty selects explicit absence (ID zero and
// timestamp -1), never latest. Iceberg pinned readers require Exact or Empty.
enum class SnapshotSelection : uint8_t { Legacy = 0, Exact = 1, Empty = 2 };

struct ReadOptions {
    int64_t   snapshot_id;
    int64_t   timestamp_ms;
    Predicate predicates[kLakeMaxPredicates];
    uint32_t  n_predicates;
    char      projection[kLakeMaxProjection][kLakeMaxColName];
    uint32_t  n_projection;
    bool      include_deletes;
    bool      include_partition_columns;
    SnapshotSelection snapshot_selection;
    // Iceberg only: append the `_pos` metadata column (Int64, the row's 0-based
    // position in its data file, counted before deletes) after the table columns.
    bool      emit_row_position;
    // Iceberg only (G2ICE-300): append the spec's reserved metadata columns after the table columns, in this order:
    // _file (Utf8, the data file's path as its manifest records it), _pos (Int64), _spec_id (Int64, the file's
    // partition spec), _deleted (Int64 0/1: false, deleted rows are not read). Exclusive with emit_row_position.
    bool      emit_metadata_columns;
    // Iceberg only, with emit_row_position (G2ICE-346): also append `_row_id` and `_last_updated_sequence_number` (Int64) after
    // `_pos`, each the value the data file wrote, else inherited: the file's first_row_id + `_pos` (null when the file has none)
    // and the file's data sequence number (spec "Row Lineage").
    bool      emit_row_lineage;
    uint8_t   _pad[2];
};

inline void read_options_init(ReadOptions* o) noexcept {
    assert(o != nullptr);
    if (o == nullptr) return;
    uint8_t* p = reinterpret_cast<uint8_t*>(o);
    for (uint32_t i = 0; i < sizeof(ReadOptions); ++i) p[i] = 0;
    o->snapshot_id  = -1;
    o->timestamp_ms = -1;
}

}  // namespace lakehouse
}  // namespace bolt
