// bolt/lakehouse/delta/cdf.h — Change Data Feed reader (skeleton).

#pragma once

#include <cstddef>
#include <cstdint>

#include "bolt/bolt_arena.h"
#include "bolt/lakehouse/delta/log.h"
#include "bolt/lakehouse/object_store.h"

namespace bolt {
namespace lakehouse {
namespace delta {

enum class CdfKind : uint8_t {
    kInsert = 0, kUpdatePreimage = 1, kUpdatePostimage = 2, kDelete = 3,
};

struct CdfFile {
    char    path[kDeltaMaxPath];
    int64_t commit_version;
    int64_t commit_timestamp;
    CdfKind kind;
    uint8_t _pad[7];
};

static constexpr uint32_t kDeltaMaxCdfFiles = 1024u;

struct CdfFileSet {
    CdfFile* files;
    uint32_t n_files;
    uint32_t n_files_cap;
};

bool delta_cdf_list(ObjectStore* os, const char* table_rel_prefix,
                    Arena* arena, CdfFileSet* out) noexcept;

}  // namespace delta
}  // namespace lakehouse
}  // namespace bolt
