// bolt/lakehouse/iceberg/view.h — W5 Iceberg view support.
//
// Iceberg views are a SQL-text + schema saved alongside tables. The on-disk
// shape is metadata/<v>.metadata.json with a top-level "view-version" record
// instead of "snapshots". Bounded caps + Tiger Style throughout.

#pragma once

#include <cassert>
#include <cstdint>

#include "bolt/bolt_arena.h"
#include "bolt/lakehouse/iceberg/metadata.h"
#include "bolt/lakehouse/iceberg/writer.h"
#include "bolt/lakehouse/object_store.h"

namespace bolt {
namespace lakehouse {
namespace iceberg {

static constexpr uint32_t kIcebergMaxSqlDialect = 32u;
static constexpr uint32_t kIcebergMaxSql        = 4096u;

// Create a new Iceberg view rooted at `path`. Writes
// metadata/v1.metadata.json with a view-version record.
bool view_create(TableHandle** out, Arena* arena, ObjectStore* os,
                 const char* path, const char* sql_dialect, const char* sql,
                 const Schema* schema) noexcept;

// Replace the SQL of an existing view (bumps version).
bool view_replace(TableHandle* th, const char* sql_dialect,
                  const char* sql) noexcept;

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
