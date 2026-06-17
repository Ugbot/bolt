// bolt/lakehouse/delta/optimize.h — OPTIMIZE (bin-pack + Z-ORDER), VACUUM,
// checkpoint-write surface for W3.
//
// Tiger Style: PODs, no exceptions, Arena allocs.

#pragma once

#include <cstdint>

#include "bolt/bolt_arena.h"
#include "bolt/lakehouse/delta/writer.h"
#include "bolt/lakehouse/handle.h"

namespace bolt {
namespace lakehouse {
namespace delta {

static constexpr uint32_t kVacuumMaxDeleted = 1024u;
static constexpr uint32_t kVacuumPathBytes  = 512u;

// Pack small files (< 0.5 * target) into larger files. Z-ORDER on up to 4 cols
// via per-column bit-interleave (identity for non-numeric / unknown columns).
// Emits one commit with Add/Remove actions.
bool delta_table_optimize(TableHandle* th,
                          const OptimizeOptions* opts) noexcept;

// VACUUM: delete files older than retention_hours and not referenced by any
// live snapshot. dry_run reports the list without deleting.
bool delta_table_vacuum(TableHandle* th, uint64_t retention_hours,
                        bool dry_run,
                        char (*out_deleted)[kVacuumPathBytes],
                        uint32_t cap, uint32_t* n) noexcept;

// Write a Parquet checkpoint at <version>.checkpoint.parquet aggregating the
// live snapshot's Add/Protocol/Metadata actions. Updates _last_checkpoint.
// In W3 the checkpoint body is a minimal aggregate JSON wrapped in a Parquet
// stub (single varbinary column "actions") — sufficient for round-trip
// discovery via _last_checkpoint without a full schema mirror.
bool delta_checkpoint_write(TableHandle* th, int64_t version,
                            Arena* scratch) noexcept;

}  // namespace delta
}  // namespace lakehouse
}  // namespace bolt
