// bolt/lakehouse/delta/checkpoint.h — Parquet checkpoint discovery.

#pragma once

#include <cstddef>
#include <cstdint>

#include "bolt/bolt_arena.h"
#include "bolt/lakehouse/delta/log.h"
#include "bolt/lakehouse/object_store.h"

namespace bolt {
namespace lakehouse {
namespace delta {

struct CheckpointInfo {
    int64_t version;
    char    rel_path[kDeltaMaxPath];
    bool    present;
    uint8_t _pad[7];
};

bool delta_checkpoint_discover(ObjectStore* os, const char* table_rel_prefix,
                               int64_t max_version, Arena* scratch,
                               CheckpointInfo* out) noexcept;

bool delta_checkpoint_replay(ObjectStore* os, const CheckpointInfo* cp,
                             Arena* arena, void* ctx, ActionFn cb) noexcept;

}  // namespace delta
}  // namespace lakehouse
}  // namespace bolt
