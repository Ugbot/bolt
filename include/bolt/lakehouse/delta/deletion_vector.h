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

// Serialise `sorted_deleted_rows` (strictly increasing, file-local 0-based
// row indices; n_rows > 0) as a new Delta deletion vector file under
// `table_rel_prefix` and fill `out` with its DvDescriptor (storageType 'u',
// offset=1, real sizeInBytes/cardinality) — ready to attach to a NEW `add`
// action for the same physical parquet file. Does not touch any existing
// add/remove log entry or the physical data file; the caller is responsible
// for the remove(old identity)+add(new identity, this DV) commit and for
// merging with any pre-existing DV (deletion vectors only grow — callers
// that need "add these rows to what's already deleted" must load the old DV
// first and pass the UNION here). G2ICE-80.
bool delta_dv_write(ObjectStore* os, const char* table_rel_prefix,
                    const uint32_t* sorted_deleted_rows, uint64_t n_rows,
                    Arena* arena, DvDescriptor* out) noexcept;

}  // namespace delta
}  // namespace lakehouse
}  // namespace bolt
