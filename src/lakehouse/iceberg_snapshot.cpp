// bolt/lakehouse/iceberg_snapshot.cpp — snapshot resolution.

#include "bolt/lakehouse/iceberg/metadata.h"
#include "bolt/lakehouse/iceberg/snapshot.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace lakehouse {
namespace iceberg {

const Snapshot* snapshot_by_id(const Metadata* m, int64_t snapshot_id) noexcept {
    assert(m != nullptr);
    for (uint32_t i = 0; i < m->n_snapshots; ++i) {
        if (m->snapshots[i].snapshot_id == snapshot_id)
            return &m->snapshots[i];
    }
    return nullptr;
}

const Snapshot* snapshot_at_timestamp(const Metadata* m,
                                      int64_t ts_ms) noexcept {
    assert(m != nullptr);
    const Snapshot* best = nullptr;
    int64_t best_ts = -1;
    for (uint32_t i = 0; i < m->n_snapshots; ++i) {
        const Snapshot& s = m->snapshots[i];
        if (s.timestamp_ms <= ts_ms && s.timestamp_ms > best_ts) {
            best = &s; best_ts = s.timestamp_ms;
        }
    }
    return best;
}

bool snapshot_resolve(const Metadata* m, int64_t explicit_id, int64_t ts_ms,
                      Snapshot* out) noexcept {
    assert(m != nullptr && out != nullptr);
    const Snapshot* p = nullptr;
    if (explicit_id >= 0) {
        p = snapshot_by_id(m, explicit_id);
    } else if (ts_ms >= 0) {
        p = snapshot_at_timestamp(m, ts_ms);
    } else if (m->current_snapshot_id >= 0) {
        p = snapshot_by_id(m, m->current_snapshot_id);
    }
    if (p == nullptr) return false;
    *out = *p;
    return true;
}

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
