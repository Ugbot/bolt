// bolt/lakehouse/dv_inline_filter.cpp — apply Delta DV roaring bitmap to
// a selection vector before downstream operators see the rows.
//
// `delta::delta_dv_contains` is the per-row probe. We loop branchlessly
// over the input selection vector and copy survivors into the output.
// The DV is row-indexed within the *file* (Delta semantics); the caller
// passes the file-local row offset of `sel_in[0]` (`base_row`) so the
// roaring lookup uses the right key.

#include "bolt/lakehouse/scan_optimizer.h"

#include <cassert>

#include "bolt/lakehouse/delta/deletion_vector.h"

namespace bolt {
namespace lakehouse {

// Returns the new selection-vector length. `sel_out` may alias `sel_in`.
uint32_t dv_inline_filter(
    const uint32_t* sel_in, uint32_t sel_len,
    uint64_t base_row,
    const delta::DeletionVector* dv,
    uint32_t* sel_out) noexcept {
    assert(sel_in != nullptr);
    assert(sel_out != nullptr);
    assert(sel_len <= kScanMorselMaxRows);
    if (sel_in == nullptr || sel_out == nullptr) return 0;
    if (dv == nullptr || !dv->present) {
        // Passthrough.
        if (sel_out != sel_in) {
            for (uint32_t i = 0; i < sel_len; ++i) sel_out[i] = sel_in[i];
        }
        return sel_len;
    }
    uint32_t kept = 0;
    for (uint32_t i = 0; i < sel_len; ++i) {
        const uint64_t row = base_row + static_cast<uint64_t>(sel_in[i]);
        const bool deleted = delta::delta_dv_contains(dv, row);
        sel_out[kept] = sel_in[i];
        kept += (deleted ? 0u : 1u);
    }
    return kept;
}

}  // namespace lakehouse
}  // namespace bolt
