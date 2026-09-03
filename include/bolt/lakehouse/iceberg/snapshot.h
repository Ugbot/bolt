// bolt/lakehouse/iceberg/snapshot.h — Iceberg snapshot record POD.
//
// One entry of metadata.json `snapshots[]`. Tiger Style: POD, noexcept.

#pragma once

#include <cstdint>

namespace bolt {
namespace lakehouse {
namespace iceberg {

static constexpr uint32_t kIcebergMaxSnapshots = 64u;
static constexpr uint32_t kIcebergMaxManifestPath = 1024u;

enum class SnapshotOp : uint8_t {
    kAppend     = 0,
    kReplace    = 1,
    kOverwrite  = 2,
    kDelete     = 3,
    kUnknown    = 4,
};

struct Snapshot {
    int64_t     snapshot_id;
    int64_t     parent_snapshot_id;
    int64_t     timestamp_ms;
    int64_t     sequence_number;
    SnapshotOp  op;
    uint8_t     _pad0[3];
    // G2ICE-50 — the schema the snapshot was COMMITTED under. pyiceberg's own
    // golden metadata carries it on every snapshot and bolt emitted none, so a
    // time-travel read of a pre-evolution snapshot had nothing to bind its
    // columns by except the CURRENT schema. Carried per snapshot rather than
    // derived from `Metadata::current_schema_id` at emit time for exactly that
    // reason: deriving it would silently relabel every historical snapshot
    // with the newest schema the moment a column was added. Takes 4 of the 7
    // pad bytes, so the struct's size and layout are unchanged.
    int32_t     schema_id;
    char        manifest_list[kIcebergMaxManifestPath];
};

struct Metadata;   // fwd

const Snapshot* snapshot_by_id(const Metadata* m, int64_t snapshot_id) noexcept;
const Snapshot* snapshot_at_timestamp(const Metadata* m, int64_t ts_ms) noexcept;
bool snapshot_resolve(const Metadata* m, int64_t explicit_id, int64_t ts_ms,
                      Snapshot* out) noexcept;

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
