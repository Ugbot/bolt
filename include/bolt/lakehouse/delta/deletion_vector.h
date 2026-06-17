// bolt/lakehouse/delta/deletion_vector.h — load + apply Delta DVs.

#pragma once

#include <cstddef>
#include <cstdint>

#include "bolt/bolt_arena.h"
#include "bolt/ingest/bolt_roaring.h"
#include "bolt/lakehouse/delta/log.h"
#include "bolt/lakehouse/object_store.h"

namespace bolt {
namespace lakehouse {
namespace delta {

struct DeletionVector {
    bolt::ingest::RoaringBitmap bitmap;
    bool                        present;
    uint8_t                     _pad[7];
};

bool delta_dv_load(ObjectStore* os, const char* table_rel_prefix,
                   const DvDescriptor* dv, Arena* arena,
                   DeletionVector* out) noexcept;

bool delta_dv_contains(const DeletionVector* dv, uint64_t row_index) noexcept;

}  // namespace delta
}  // namespace lakehouse
}  // namespace bolt
