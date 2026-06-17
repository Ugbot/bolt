// bolt/lakehouse/catalog/unity.h — Databricks Unity Catalog REST.
//
// Distinct from the Apache Iceberg REST catalog (iceberg_rest.h): Unity is
// Databricks-flavoured, with a 3-level namespace (catalog -> schema -> table),
// a `data_source_format` table model, volumes, external locations, storage
// credentials, functions, registered models, permissions (GRANT), Delta
// Sharing (shares + recipients), lineage and tags. All under the
// /api/2.1/unity-catalog base path. Auth is a workspace Bearer token (PAT or
// OAuth2 — both are carried in the same Authorization: Bearer header).
//
// The `Catalog` vtable binding (open / list / create / drop) lives in
// catalog_unity.cpp. This header adds the FULL Unity client surface as
// white-box request/body builders so the wire shape is unit-testable without
// a live workspace. HTTPS + Bearer. Never logs the token.
//
// Tiger Style: PODs, fixed-cap buffers, ≥2 asserts/fn, ≤70-line fns, no
// exceptions, no smart pointers, no RTTI, arena-owned response bodies.
#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/lakehouse/catalog.h"
#include "bolt/lakehouse/catalog/secret.h"
#include "bolt/net/bolt_http_client.h"

#include <cstdint>

namespace bolt {
namespace lakehouse {
namespace unity {

// --------------------------------------------------------------------------
// Config + Impl (white-box; stable for unit tests, like iceberg_rest::Impl).
// --------------------------------------------------------------------------

struct Config {
    char   base_url[512];        // e.g. "https://dbc-xxx.cloud.databricks.com"
    Secret bearer_token;         // PAT or OAuth2 workspace token (Bearer header)
};

struct Impl {
    Config       cfg;
    bolt::Arena* arena;
};

// Bind `out` to a freshly-opened Unity catalog (the 6-fn Catalog vtable).
// Allocates impl from `arena`.
bool open(Catalog* out, bolt::Arena* arena, const Config* cfg) noexcept;

// --------------------------------------------------------------------------
// Bounded enums + caps for the white-box builders.
// --------------------------------------------------------------------------

// data_source_format ∈ {DELTA, PARQUET, ICEBERG, CSV, JSON}.
enum class DataFormat : uint8_t {
    kDelta = 0, kParquet = 1, kIceberg = 2, kCsv = 3, kJson = 4,
};

// Volume type — managed (UC-owned location) or external (caller location).
enum class VolumeType : uint8_t { kManaged = 0, kExternal = 1 };

// Privilege for a GRANT/REVOKE on a securable.
enum class Privilege : uint8_t {
    kSelect = 0, kModify = 1, kUse = 2, kCreate = 3, kAll = 4,
};

// Securable kind for the /permissions/{securable}/{full_name} path.
enum class Securable : uint8_t {
    kCatalog = 0, kSchema = 1, kTable = 2, kVolume = 3,
    kExternalLocation = 4, kFunction = 5, kShare = 6,
};

static constexpr uint32_t kUnityMaxFullName = 256u;   // catalog.schema.table

// --------------------------------------------------------------------------
// full_name encoding. Unity addresses securables by dotted full name:
//   catalog                         (1 level)
//   catalog.schema                  (2 levels)
//   catalog.schema.table            (3 levels)
// Builds the dotted name into out[0..cap). Empty levels are skipped. Returns
// false on overflow or when all levels are empty.
// --------------------------------------------------------------------------
bool uc_build_full_name(const char* catalog, const char* schema,
                        const char* table, char* out, uint32_t cap) noexcept;

// Inject the Authorization: Bearer header onto `req`. Returns false on
// overflow. Never logs the token.
bool uc_add_auth(const Impl* impl, bolt::net::HttpRequest* req) noexcept;

// --------------------------------------------------------------------------
// Catalog CRUD — /api/2.1/unity-catalog/catalogs
// --------------------------------------------------------------------------
bool uc_build_list_catalogs_request(const Impl* impl,
                                    bolt::net::HttpRequest* out) noexcept;
bool uc_build_get_catalog_request(const Impl* impl, const char* name,
                                  bolt::net::HttpRequest* out) noexcept;
// Body buffer is caller-owned; builder fills `body`+`*body_len` and points
// `out->body` at it. POST /catalogs with {"name":..,"comment":..}.
bool uc_build_create_catalog_request(const Impl* impl, const char* name,
                                     const char* comment, char* body,
                                     uint32_t body_cap,
                                     bolt::net::HttpRequest* out) noexcept;
bool uc_build_delete_catalog_request(const Impl* impl, const char* name,
                                     bolt::net::HttpRequest* out) noexcept;

// --------------------------------------------------------------------------
// Schema CRUD — /api/2.1/unity-catalog/schemas
// --------------------------------------------------------------------------
bool uc_build_list_schemas_request(const Impl* impl, const char* catalog,
                                   bolt::net::HttpRequest* out) noexcept;
bool uc_build_create_schema_request(const Impl* impl, const char* catalog,
                                    const char* name, char* body,
                                    uint32_t body_cap,
                                    bolt::net::HttpRequest* out) noexcept;
bool uc_build_delete_schema_request(const Impl* impl, const char* full_name,
                                    bolt::net::HttpRequest* out) noexcept;

// --------------------------------------------------------------------------
// Table CRUD — /api/2.1/unity-catalog/tables
// --------------------------------------------------------------------------
bool uc_build_list_tables_request(const Impl* impl, const char* catalog,
                                  const char* schema,
                                  bolt::net::HttpRequest* out) noexcept;
bool uc_build_get_table_request(const Impl* impl, const char* full_name,
                                bolt::net::HttpRequest* out) noexcept;
bool uc_build_create_table_request(const Impl* impl, const char* catalog,
                                   const char* schema, const char* name,
                                   DataFormat fmt, const char* storage_location,
                                   char* body, uint32_t body_cap,
                                   bolt::net::HttpRequest* out) noexcept;
bool uc_build_delete_table_request(const Impl* impl, const char* full_name,
                                   bolt::net::HttpRequest* out) noexcept;

// --------------------------------------------------------------------------
// Volume CRUD — /api/2.1/unity-catalog/volumes
// --------------------------------------------------------------------------
bool uc_build_create_volume_request(const Impl* impl, const char* catalog,
                                    const char* schema, const char* name,
                                    VolumeType vtype, const char* storage_location,
                                    char* body, uint32_t body_cap,
                                    bolt::net::HttpRequest* out) noexcept;
bool uc_build_delete_volume_request(const Impl* impl, const char* full_name,
                                    bolt::net::HttpRequest* out) noexcept;

// --------------------------------------------------------------------------
// External locations + storage credentials.
// --------------------------------------------------------------------------
bool uc_build_create_external_location_request(
    const Impl* impl, const char* name, const char* url,
    const char* credential_name, char* body, uint32_t body_cap,
    bolt::net::HttpRequest* out) noexcept;
bool uc_build_create_storage_credential_request(
    const Impl* impl, const char* name, const char* iam_role_arn,
    char* body, uint32_t body_cap, bolt::net::HttpRequest* out) noexcept;

// --------------------------------------------------------------------------
// Functions + registered models.
// --------------------------------------------------------------------------
bool uc_build_list_functions_request(const Impl* impl, const char* catalog,
                                     const char* schema,
                                     bolt::net::HttpRequest* out) noexcept;
bool uc_build_list_models_request(const Impl* impl, const char* catalog,
                                  const char* schema,
                                  bolt::net::HttpRequest* out) noexcept;

// --------------------------------------------------------------------------
// Permissions — /api/2.1/unity-catalog/permissions/{securable}/{full_name}.
// PATCH body: {"changes":[{"principal":..,"add":["SELECT",..]}]}.
// --------------------------------------------------------------------------
bool uc_build_get_permissions_request(const Impl* impl, Securable sec,
                                      const char* full_name,
                                      bolt::net::HttpRequest* out) noexcept;
bool uc_build_grant_request(const Impl* impl, Securable sec,
                            const char* full_name, const char* principal,
                            const Privilege* privs, uint32_t n_privs,
                            char* body, uint32_t body_cap,
                            bolt::net::HttpRequest* out) noexcept;

// --------------------------------------------------------------------------
// Delta Sharing — shares + recipients.
// --------------------------------------------------------------------------
bool uc_build_create_share_request(const Impl* impl, const char* name,
                                   const char* comment, char* body,
                                   uint32_t body_cap,
                                   bolt::net::HttpRequest* out) noexcept;
bool uc_build_update_share_permissions_request(
    const Impl* impl, const char* share, const char* principal,
    Privilege priv, char* body, uint32_t body_cap,
    bolt::net::HttpRequest* out) noexcept;
bool uc_build_create_recipient_request(const Impl* impl, const char* name,
                                       const char* comment, char* body,
                                       uint32_t body_cap,
                                       bolt::net::HttpRequest* out) noexcept;

// --------------------------------------------------------------------------
// Lineage + tags.
// --------------------------------------------------------------------------
bool uc_build_table_lineage_request(const Impl* impl, const char* full_name,
                                    bolt::net::HttpRequest* out) noexcept;
bool uc_build_column_lineage_request(const Impl* impl, const char* full_name,
                                     const char* column,
                                     bolt::net::HttpRequest* out) noexcept;
bool uc_build_set_table_tag_request(const Impl* impl, const char* full_name,
                                    const char* key, const char* value,
                                    char* body, uint32_t body_cap,
                                    bolt::net::HttpRequest* out) noexcept;

}  // namespace unity
}  // namespace lakehouse
}  // namespace bolt
