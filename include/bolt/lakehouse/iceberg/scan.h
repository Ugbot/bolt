// bolt/lakehouse/iceberg/scan.h — Iceberg read-side public surface (W4).
//
// Opaque table + scan handles defined in iceberg_scan.cpp; this header only
// forward-declares them in the `iceberg` sub-namespace so they don't collide
// with the Delta versions.

#pragma once

#include <cstdint>

#include "bolt/bolt_arena.h"
#include "bolt/lakehouse/catalog.h"
#include "bolt/lakehouse/format.h"
#include "bolt/lakehouse/object_store.h"

namespace bolt {

class Arena;
struct BoltBatch;

namespace lakehouse {
namespace iceberg {

struct TableHandle;
struct ScanHandle;

bool iceberg_table_open(TableHandle** out, Arena* arena, Catalog* catalog,
                        const char* namespace_, const char* name) noexcept;

void iceberg_table_close(TableHandle* h) noexcept;

bool iceberg_scan_open(ScanHandle** out, TableHandle* h,
                       const ReadOptions* opts) noexcept;

bool iceberg_scan_next_batch(ScanHandle* h, BoltBatch* out,
                             bool* out_eof) noexcept;

void iceberg_scan_close(ScanHandle* h) noexcept;

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
