// Internal to the parquet reader TUs: one leaf column decoded slot by slot,
// the input every record-assembly path (LIST and nested) is built from.
#pragma once

#include <cstdint>

#include "bolt/bolt_column.h"
#include "bolt/ingest/bolt_parquet_meta.h"

namespace bolt {

class Arena;

namespace ingest {
namespace parquet {

// rep/def hold one level per leaf slot across the row-group range. `values`
// holds one entry per VALUE POSITION -- a slot whose def reaches the leaf's
// innermost repeated ancestor (every slot when the leaf has none) -- valid
// iff def == max_def; validity is null when none is null.
struct PqLeafSlots {
    uint32_t*  rep;
    uint32_t*  def;
    int64_t    n_slots;
    int64_t    n_rows;
    int64_t    n_values;
    BoltColumn values;
};

// Every level is checked against max_rep/max_def, rows against the footer,
// and every chunk must open a row. All storage from `arena`.
bool pq_decode_leaf_slots(const uint8_t* buf, uint64_t len, const PqMeta* meta,
                          uint32_t g0, uint32_t g1, uint32_t col, Arena* arena,
                          PqLeafSlots* out) noexcept;

}  // namespace parquet
}  // namespace ingest
}  // namespace bolt
