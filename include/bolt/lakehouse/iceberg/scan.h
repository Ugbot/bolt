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
struct Metadata;

bool iceberg_table_open(TableHandle** out, Arena* arena, Catalog* catalog,
                        const char* namespace_, const char* name) noexcept;

// Open a catalog-served table: `metadata_json` is the TableMetadata document
// a REST catalog's loadTable returned, and every file it names is fetched
// through `store` (S3, GCS, Azure, filesystem). `store->impl` is borrowed and
// must outlive the handle. `table_key_prefix` is the store key of the table's
// `location`; nullptr derives it by dropping the location's scheme and
// authority ("s3://bucket/wh/ns/t" -> "wh/ns/t").
bool iceberg_table_open_on_store(TableHandle** out, Arena* arena,
                                 const ObjectStore* store,
                                 const char* table_key_prefix,
                                 const uint8_t* metadata_json,
                                 uint32_t metadata_len) noexcept;

void iceberg_table_close(TableHandle* h) noexcept;

// The parsed metadata.json behind an open table — schemas, partition specs,
// snapshot history, and the `location` the writer recorded. Read-only and
// owned by the handle's arena; valid until that arena is reset. A caller needs
// this to pick a snapshot for time travel (`ReadOptions::snapshot_id`), which
// is otherwise unanswerable from the public surface.
const Metadata* iceberg_table_metadata(const TableHandle* h) noexcept;

bool iceberg_scan_open(ScanHandle** out, TableHandle* h,
                       const ReadOptions* opts) noexcept;

bool iceberg_scan_next_batch(ScanHandle* h, BoltBatch* out,
                             bool* out_eof) noexcept;

void iceberg_scan_close(ScanHandle* h) noexcept;

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
