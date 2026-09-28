// bolt_column_limits.h — the column-count caps (limits review L6).
//
// kMaxColumns is the one column-count ceiling across bolt and everything built
// on it: a validation bound only. Schemas, batches, Parquet metadata, wire
// frames, Arrow IPC streams and CSV schemas are sized to their real column
// count, so a narrow table pays nothing for it. Every other column cap in
// marbledb/chukonu/Gestalt2 derives from it. 32,768 leaves headroom under the
// uint16 column indices (and their 0xFFFF sentinels) that plan payloads and
// the Parquet reader use.
//
// The two below are fast-path capacities of fixed-size structs, never
// refusals: a wider shape takes the general path.
#pragma once

#include <cstdint>

namespace bolt {

inline constexpr uint32_t kMaxColumns = 32768;
static_assert(kMaxColumns < 0xFFFFu, "column indices are uint16 with a 0xFFFF sentinel");

// RowView (zero-copy point get; fixed C-ABI size): a wider row returns "no
// view" and the caller takes the copying get() path.
inline constexpr uint32_t kRowViewMaxColumns = 256;

namespace ingest {
// Fields of one Arrow Struct in the IPC writer (IpcFieldDesc::n_children is
// a byte). Top-level columns are bounded by kMaxColumns only.
inline constexpr uint16_t kIpcMaxStructChildren = 255;
}  // namespace ingest

}  // namespace bolt
