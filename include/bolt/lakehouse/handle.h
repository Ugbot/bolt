// bolt/lakehouse/handle.h — opaque TableHandle / ScanHandle.
// Tiger Style: PODs, no exceptions, no smart pointers.

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

struct TableHandle;
struct ScanHandle;

bool delta_table_open(TableHandle** out, Arena* arena, Catalog* catalog,
                      const char* namespace_, const char* name) noexcept;
void delta_table_close(TableHandle* h) noexcept;

bool delta_scan_open(ScanHandle** out, TableHandle* h,
                     const ReadOptions* opts) noexcept;
bool delta_scan_next_batch(ScanHandle* h, BoltBatch* out,
                           bool* out_eof) noexcept;
void delta_scan_close(ScanHandle* h) noexcept;

}  // namespace lakehouse
}  // namespace bolt
