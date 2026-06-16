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
    uint8_t     _pad[7];
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
