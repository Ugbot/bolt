// bolt/lakehouse/catalog/glue.h — AWS Glue Data Catalog.
// JSON-over-HTTPS at glue.<region>.amazonaws.com, SigV4-signed.
#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/lakehouse/catalog.h"
#include "bolt/lakehouse/catalog/secret.h"
#include "bolt/net/bolt_http_client.h"

namespace bolt {
namespace lakehouse {
namespace glue {

struct Config {
    char   region[32];
    Secret access_key;
    Secret secret_key;
    Secret session_token;   // may be empty
};

bool open(Catalog* out, bolt::Arena* arena, const Config* cfg) noexcept;

// Internal impl (visible for tests).
struct Impl {
    Config       cfg;
    bolt::Arena* arena;
};

// Sign a Glue HTTP request in place (adds Authorization, X-Amz-Date, …).
// Uses current epoch as the date.
bool glue_sign_request(Impl* impl, bolt::net::HttpRequest* req) noexcept;

// Same, but with a fixed amz_date for reproducibility.
bool glue_sign_request_with_date(Impl* impl, bolt::net::HttpRequest* req,
                                 const char* amz_date) noexcept;

// ---------------------------------------------------------------------------
// G2CLUS-178: extended Glue surface — Database/Table CRUD beyond the Catalog
// VT, Partition CRUD, Schema registry, and Lake Formation grants. Each call
// POSTs `application/x-amz-json-1.1` to glue.<region>.amazonaws.com with an
// `X-Amz-Target: AWSGlue.<Op>` (or `AWSLakeFormation.<Op>`) header, SigV4-
// signed. Bodies are caller-supplied NUL-terminated JSON; responses land in
// `resp` (arena-owned). Return: kCatOk / kCatNotFound / kCatExists /
// kCatBadArg / kCatIoError. PODs, ≥2 asserts/fn, no exceptions.
// ---------------------------------------------------------------------------

// Generic typed POST: signs + sends one Glue/LakeFormation JSON-1.1 call.
// `target` is the full X-Amz-Target value. `body`/`blen` is the request JSON.
int glue_post(Impl* impl, const char* target, const char* body, uint32_t blen,
              bolt::net::HttpResponse* resp) noexcept;

// Build the host "glue.<region>.amazonaws.com" into out[0..cap).
void glue_make_host(const Config* cfg, char* out, uint32_t cap) noexcept;

// --- Database (Get / Delete; Create + GetDatabases live on the Catalog VT). --
int glue_get_database(Impl* impl, const char* db,
                      bolt::net::HttpResponse* resp) noexcept;
int glue_delete_database(Impl* impl, const char* db,
                         bolt::net::HttpResponse* resp) noexcept;

// --- Table (Get / Update / Search). ---------------------------------------
int glue_get_table(Impl* impl, const char* db, const char* table,
                   bolt::net::HttpResponse* resp) noexcept;
// `table_input_json` is the full TableInput object body, e.g.
// "{\"Name\":\"t\",\"StorageDescriptor\":{\"Columns\":[]}}".
int glue_update_table(Impl* impl, const char* db, const char* table_input_json,
                      bolt::net::HttpResponse* resp) noexcept;
int glue_search_tables(Impl* impl, const char* search_text,
                       bolt::net::HttpResponse* resp) noexcept;

// --- Partition CRUD. ------------------------------------------------------
int glue_get_partitions(Impl* impl, const char* db, const char* table,
                        const char* expression_or_null,
                        bolt::net::HttpResponse* resp) noexcept;
// `partition_inputs_json` is a JSON array of PartitionInput objects.
int glue_batch_create_partition(Impl* impl, const char* db, const char* table,
                                const char* partition_inputs_json,
                                bolt::net::HttpResponse* resp) noexcept;
// `entries_json` is a JSON array of BatchUpdatePartitionRequestEntry objects.
int glue_batch_update_partition(Impl* impl, const char* db, const char* table,
                                const char* entries_json,
                                bolt::net::HttpResponse* resp) noexcept;

// --- Schema registry. -----------------------------------------------------
// `schema_definition` is the raw Avro/JSON schema string (it is JSON-escaped).
int glue_create_schema(Impl* impl, const char* registry_name,
                       const char* schema_name, const char* data_format,
                       const char* schema_definition,
                       bolt::net::HttpResponse* resp) noexcept;
int glue_register_schema_version(Impl* impl, const char* schema_arn,
                                 const char* schema_definition,
                                 bolt::net::HttpResponse* resp) noexcept;

// --- Lake Formation grants (X-Amz-Target = AWSLakeFormation.<Op>). ---------
// `principal_arn` is a DataLakePrincipal identifier; `resource_json` is the
// Resource object; `permissions_json` is a JSON array of permission strings.
int glue_grant_permissions(Impl* impl, const char* principal_arn,
                           const char* resource_json,
                           const char* permissions_json,
                           bolt::net::HttpResponse* resp) noexcept;
int glue_list_permissions(Impl* impl, const char* principal_arn,
                          bolt::net::HttpResponse* resp) noexcept;

}  // namespace glue
}  // namespace lakehouse
}  // namespace bolt
