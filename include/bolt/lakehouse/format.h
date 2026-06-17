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
static constexpr uint32_t kLakeMaxProjection = 64u;
static constexpr uint32_t kLakeMaxPartCols  = 8u;
static constexpr uint32_t kLakeMaxLiveFiles = 4096u;
static constexpr uint32_t kLakeMaxRowGroups = 4096u;
static constexpr uint32_t kLakeMaxCommits   = 65536u;
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

struct ReadOptions {
    int64_t   snapshot_id;
    int64_t   timestamp_ms;
    Predicate predicates[kLakeMaxPredicates];
    uint32_t  n_predicates;
    char      projection[kLakeMaxProjection][kLakeMaxColName];
    uint32_t  n_projection;
    bool      include_deletes;
    bool      include_partition_columns;
    uint8_t   _pad[6];
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
