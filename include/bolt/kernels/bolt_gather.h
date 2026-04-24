// bolt_gather.h — Columnar-to-row gather kernels.
//
// RULES: No exceptions. No RTTI. No heap. Caller supplies all buffers.
//
// `gather_row` is the coalesced "pack one row from column-major storage
// into a contiguous row buffer" kernel. Callers pre-compute `col_offset_in_row`
// at schema-init time; the kernel does one pointer arithmetic + memcpy
// per column. For fixed-width schemas with identically-sized columns the
// inner loop vectorises to vmovdqu stores on hosts with AVX2.
//
// Motivation. Storage engines with a columnar memtable (MarbleDB,
// ClickHouse's MergeTree-like paths) have to materialise a single row
// worth of data when answering a point-get. Naive "one arena bump per
// column" creates N×64-byte-aligned bumps for an N-column row (wasted
// alignment padding). Naive "allocate row_bytes then memcpy per column"
// is correct but hand-rolled; callers repeat the same inner loop
// everywhere. This kernel centralises it.

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>
#include <cstring>

namespace bolt {

// Gather one row from column-major storage into a packed row buffer.
//
// Inputs:
//   col_data[c]           — base pointer for column c's cell array.
//   col_size[c]           — fixed element width in bytes (≤ 64).
//   col_offset_in_row[c]  — byte offset for column c inside the row buffer.
//   ncols                 — number of columns; ≤ the allocated arrays.
//   row_idx               — row index within each column array.
//   dst                   — destination row buffer; caller guarantees
//                           it has room for at least
//                           max(col_offset_in_row[c] + col_size[c]) bytes.
//
// Output: dst filled with one packed row, columns placed at their
// respective offsets. Preserves validity ordering: unchanged inputs
// produce unchanged outputs.
//
// Cost: N memcpys of 1..64 bytes each. On uniform-width-column rows the
// compiler collapses to a fixed-width unrolled loop; mixed widths pay
// one branchless memcpy call per column.
BOLT_FORCE_INLINE void gather_row(
    const void* const*  BOLT_RESTRICT col_data,
    const uint16_t*     BOLT_RESTRICT col_size,
    const uint16_t*     BOLT_RESTRICT col_offset_in_row,
    uint32_t                          ncols,
    uint32_t                          row_idx,
    void*               BOLT_RESTRICT dst) noexcept {
    assert(col_data != nullptr);
    assert(col_size != nullptr);
    assert(col_offset_in_row != nullptr);
    assert(dst != nullptr);
    uint8_t* const dst_bytes = static_cast<uint8_t*>(dst);
    for (uint32_t c = 0; c < ncols; ++c) {
        const uint32_t tsz = col_size[c];
        const uint8_t* src = static_cast<const uint8_t*>(col_data[c])
                           + static_cast<size_t>(row_idx) * tsz;
        std::memcpy(dst_bytes + col_offset_in_row[c], src, tsz);
    }
}

// Same kernel but projects only a subset of columns indicated by
// `sel_cols[0..n_sel)`. The row buffer's layout is caller-defined via
// `col_offset_in_row` (indexed by the SELECTED column's position in the
// destination, not by the source schema). For a full projection callers
// can pass `sel_cols = nullptr` — the kernel forwards to the variant
// above.
BOLT_FORCE_INLINE void gather_row_sel(
    const void* const*  BOLT_RESTRICT col_data,
    const uint16_t*     BOLT_RESTRICT col_size,
    const uint16_t*     BOLT_RESTRICT col_offset_in_row,
    const uint16_t*                   sel_cols,     // may be nullptr
    uint32_t                          n_out_cols,
    uint32_t                          row_idx,
    void*               BOLT_RESTRICT dst) noexcept {
    assert(col_data != nullptr);
    assert(col_size != nullptr);
    assert(col_offset_in_row != nullptr);
    assert(dst != nullptr);
    if (sel_cols == nullptr) {
        gather_row(col_data, col_size, col_offset_in_row,
                   n_out_cols, row_idx, dst);
        return;
    }
    uint8_t* const dst_bytes = static_cast<uint8_t*>(dst);
    for (uint32_t i = 0; i < n_out_cols; ++i) {
        const uint32_t src_col = sel_cols[i];
        const uint32_t tsz = col_size[src_col];
        const uint8_t* src = static_cast<const uint8_t*>(col_data[src_col])
                           + static_cast<size_t>(row_idx) * tsz;
        std::memcpy(dst_bytes + col_offset_in_row[i], src, tsz);
    }
}

}  // namespace bolt
