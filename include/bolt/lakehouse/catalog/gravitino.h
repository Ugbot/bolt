// bolt/lakehouse/catalog/gravitino.h — Apache Gravitino metalake/catalog gateway.
//
// Gravitino is a multi-catalog gateway (bridges Hive, Iceberg, JDBC, Kafka).
// Its REST shape is metalake → catalog → schema → table. The CatalogVT maps:
//   list_namespaces → list catalogs in the configured metalake
//   list_tables     → list tables in {catalog}/schemas/{schema}
//   create/drop     → table CRUD under a schema
// The metalake + catalog + schema CRUD calls below are Gravitino-native and
// shaped here for testability.
//
// Tiger Style: PODs, ≥2 asserts/fn, no exceptions, no smart pointers, bounded.
#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/lakehouse/catalog.h"
#include "bolt/lakehouse/catalog/secret.h"
#include "bolt/net/bolt_http_client.h"

namespace bolt {
namespace lakehouse {
namespace gravitino {

struct Config {
    char   base_url[512];
    Secret bearer_token;
    char   metalake[128];
    char   schema[128];      // default schema for table ops (e.g. "default")
    char   provider[32];     // "hive" | "lakehouse-iceberg" | "jdbc-mysql" | ...
};

bool open(Catalog* out, bolt::Arena* arena, const Config* cfg) noexcept;

// --- Gravitino-native CRUD (return kCat* codes) ---

// Metalake CRUD: /api/metalakes.
int gravitino_create_metalake(Catalog* cat, const char* name,
                              const char* comment) noexcept;
int gravitino_drop_metalake(Catalog* cat, const char* name) noexcept;

// Catalog CRUD: /api/metalakes/{ml}/catalogs.
int gravitino_create_catalog(Catalog* cat, const char* name, const char* type,
                             const char* provider) noexcept;
int gravitino_drop_catalog(Catalog* cat, const char* name) noexcept;

// Schema CRUD: /api/metalakes/{ml}/catalogs/{cat}/schemas.
int gravitino_create_schema(Catalog* cat, const char* catalog,
                            const char* schema, const char* comment) noexcept;
int gravitino_drop_schema(Catalog* cat, const char* catalog,
                          const char* schema) noexcept;

// Load a single table → metadata, copying its `name` field into out[0..cap).
int gravitino_load_table(Catalog* cat, const char* catalog, const char* schema,
                         const char* table, char* out_name,
                         uint32_t cap) noexcept;

// --- Request-shaping helpers (white-box; stable for unit tests) ---

bool gravitino_build_create_metalake(const char* base_url, const char* bearer,
                                     const char* name, const char* comment,
                                     bolt::net::HttpRequest* out, char* body_buf,
                                     uint32_t body_cap) noexcept;

bool gravitino_build_create_catalog(const char* base_url, const char* bearer,
                                    const char* metalake, const char* name,
                                    const char* type, const char* provider,
                                    bolt::net::HttpRequest* out, char* body_buf,
                                    uint32_t body_cap) noexcept;

bool gravitino_build_create_schema(const char* base_url, const char* bearer,
                                   const char* metalake, const char* catalog,
                                   const char* schema, const char* comment,
                                   bolt::net::HttpRequest* out, char* body_buf,
                                   uint32_t body_cap) noexcept;

bool gravitino_build_create_table(const char* base_url, const char* bearer,
                                  const char* metalake, const char* catalog,
                                  const char* schema, const char* table,
                                  bolt::net::HttpRequest* out, char* body_buf,
                                  uint32_t body_cap) noexcept;

}  // namespace gravitino
}  // namespace lakehouse
}  // namespace bolt
