// bolt/src/lakehouse/catalog_glue_ext.cpp — G2CLUS-178.
//
// Extended AWS Glue Data Catalog surface: Database/Table CRUD beyond the
// Catalog VT (GetDatabase / DeleteDatabase / GetTable / UpdateTable /
// SearchTables), Partition CRUD (GetPartitions / BatchCreatePartition /
// BatchUpdatePartition), Schema registry (CreateSchema /
// RegisterSchemaVersion), and Lake Formation grants (GrantPermissions /
// ListPermissions).
//
// Every call is a `application/x-amz-json-1.1` POST to the Glue (or Lake
// Formation) endpoint with an `X-Amz-Target` header, SigV4-signed. The
// Glue calls reuse the public glue_sign_request_with_date helper; Lake
// Formation signs inline because it carries a different service name + host.
//
// Tiger Style: PODs, ≥2 asserts/fn, ≤70-line fns, fixed buffers with
// static_assert caps, no exceptions, no smart pointers, no secrets logged.

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "bolt/lakehouse/catalog/glue.h"

#include "bolt/crypto/sigv4.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <ctime>

namespace bolt {
namespace lakehouse {
namespace glue {

namespace {

// Bounded request-body buffer. Glue/LakeFormation control-plane bodies are
// small; partition/schema bodies arrive pre-built from the caller, so we only
// wrap them with a fixed envelope here.
static constexpr uint32_t kGlueMaxBody = 64u * 1024u;
static_assert(kGlueMaxBody <= bolt::net::kHttpMaxBody, "body within HTTP cap");
static constexpr uint32_t kGlueMaxHost = 256u;
static_assert(kGlueMaxHost <= bolt::crypto::kSigV4MaxHost, "host within SigV4 cap");

// Format the current UTC time as "YYYYMMDDTHHMMSSZ" (16 chars + NUL).
void format_amz_date(char out[17]) noexcept {
    assert(out != nullptr);
    std::time_t t = std::time(nullptr);
    std::tm tmv;
#ifdef _WIN32
    gmtime_s(&tmv, &t);
#else
    gmtime_r(&t, &tmv);
#endif
    int n = std::snprintf(out, 17, "%04d%02d%02dT%02d%02d%02dZ",
                          tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday,
                          tmv.tm_hour, tmv.tm_min, tmv.tm_sec);
    assert(n == 16);
    (void)n;
}

// Map an HTTP status into the catalog status enum.
int status_to_cat(int http_status) noexcept {
    assert(http_status >= 0);
    if (http_status == 404) return kCatNotFound;
    if (http_status == 409) return kCatExists;
    if (http_status / 100 != 2) return kCatIoError;
    return kCatOk;
}

// Sign + send a JSON-1.1 POST against an arbitrary service/host. Used for the
// Lake Formation calls (service "lakeformation"); the Glue calls go through
// glue_post which reuses the public glue signer. Returns a kCat* code.
int signed_post(Impl* impl, const char* service, const char* host,
                const char* target, const char* body, uint32_t blen,
                bolt::net::HttpResponse* resp) noexcept {
    assert(impl != nullptr && service != nullptr && host != nullptr);
    assert(target != nullptr && body != nullptr && resp != nullptr);
    bolt::net::HttpRequest req;
    std::memset(&req, 0, sizeof(req));
    std::strncpy(req.method, "POST", sizeof(req.method) - 1u);
    std::snprintf(req.url, sizeof(req.url), "https://%s/", host);
    bolt::net::http_request_add_header(&req, "Content-Type",
                                       "application/x-amz-json-1.1");
    bolt::net::http_request_add_header(&req, "X-Amz-Target", target);
    req.body = reinterpret_cast<const uint8_t*>(body);
    req.body_len = blen;

    char body_hash[65];
    if (blen > 0) {
        bolt::crypto::sha256_hex(req.body, blen, body_hash);
    } else {
        bolt::crypto::sha256_empty_hex(body_hash);
    }
    char amz[17];
    format_amz_date(amz);
    bolt::crypto::SigV4Request s;
    std::memset(&s, 0, sizeof(s));
    s.method = "POST";
    s.host = host;
    s.path = "/";
    s.query_string = "";
    s.service = service;
    s.region = impl->cfg.region;
    s.amz_date = amz;
    s.payload_sha256 = body_hash;
    s.access_key = impl->cfg.access_key.value;
    s.secret_key = impl->cfg.secret_key.value;
    s.session_token = impl->cfg.session_token.value[0] != '\0'
                          ? impl->cfg.session_token.value : nullptr;
    char auth_hdr[bolt::crypto::kSigV4MaxAuth];
    char amz_out[24];
    bolt::crypto::SigV4Result so;
    so.auth_header = auth_hdr;
    so.auth_cap = sizeof(auth_hdr);
    so.amz_date_out = amz_out;
    so.amz_date_cap = sizeof(amz_out);
    if (!bolt::crypto::sigv4_sign(&s, &so)) return kCatIoError;
    if (!bolt::net::http_request_add_header(&req, "X-Amz-Date", amz))
        return kCatIoError;
    if (!bolt::net::http_request_add_header(&req, "X-Amz-Content-Sha256",
                                            body_hash)) return kCatIoError;
    if (s.session_token != nullptr &&
        !bolt::net::http_request_add_header(&req, "X-Amz-Security-Token",
                                            s.session_token)) return kCatIoError;
    if (!bolt::net::http_request_add_header(&req, "Authorization", auth_hdr))
        return kCatIoError;
    if (bolt::net::http_send(impl->arena, &req, resp) != 0) return kCatIoError;
    return status_to_cat(resp->status);
}

}  // namespace

void glue_make_host(const Config* cfg, char* out, uint32_t cap) noexcept {
    assert(cfg != nullptr && out != nullptr);
    assert(cap >= 8u);
    std::snprintf(out, cap, "glue.%s.amazonaws.com", cfg->region);
}

int glue_post(Impl* impl, const char* target, const char* body, uint32_t blen,
              bolt::net::HttpResponse* resp) noexcept {
    assert(impl != nullptr && target != nullptr);
    assert(body != nullptr && resp != nullptr);
    char host[kGlueMaxHost];
    glue_make_host(&impl->cfg, host, sizeof(host));

    bolt::net::HttpRequest req;
    std::memset(&req, 0, sizeof(req));
    std::strncpy(req.method, "POST", sizeof(req.method) - 1u);
    std::snprintf(req.url, sizeof(req.url), "https://%s/", host);
    bolt::net::http_request_add_header(&req, "Content-Type",
                                       "application/x-amz-json-1.1");
    bolt::net::http_request_add_header(&req, "X-Amz-Target", target);
    req.body = reinterpret_cast<const uint8_t*>(body);
    req.body_len = blen;
    char amz[17];
    format_amz_date(amz);
    if (!glue_sign_request_with_date(impl, &req, amz)) return kCatIoError;
    if (bolt::net::http_send(impl->arena, &req, resp) != 0) return kCatIoError;
    return status_to_cat(resp->status);
}

// --- Database --------------------------------------------------------------

int glue_get_database(Impl* impl, const char* db,
                      bolt::net::HttpResponse* resp) noexcept {
    assert(impl != nullptr && db != nullptr && resp != nullptr);
    char body[kGlueMaxBody];
    int n = std::snprintf(body, sizeof(body), "{\"Name\":\"%s\"}", db);
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(body)) return kCatBadArg;
    return glue_post(impl, "AWSGlue.GetDatabase", body,
                     static_cast<uint32_t>(n), resp);
}

int glue_delete_database(Impl* impl, const char* db,
                         bolt::net::HttpResponse* resp) noexcept {
    assert(impl != nullptr && db != nullptr && resp != nullptr);
    char body[kGlueMaxBody];
    int n = std::snprintf(body, sizeof(body), "{\"Name\":\"%s\"}", db);
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(body)) return kCatBadArg;
    return glue_post(impl, "AWSGlue.DeleteDatabase", body,
                     static_cast<uint32_t>(n), resp);
}

// --- Table -----------------------------------------------------------------

int glue_get_table(Impl* impl, const char* db, const char* table,
                   bolt::net::HttpResponse* resp) noexcept {
    assert(impl != nullptr && db != nullptr);
    assert(table != nullptr && resp != nullptr);
    char body[kGlueMaxBody];
    int n = std::snprintf(body, sizeof(body),
                          "{\"DatabaseName\":\"%s\",\"Name\":\"%s\"}", db, table);
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(body)) return kCatBadArg;
    return glue_post(impl, "AWSGlue.GetTable", body,
                     static_cast<uint32_t>(n), resp);
}

int glue_update_table(Impl* impl, const char* db, const char* table_input_json,
                      bolt::net::HttpResponse* resp) noexcept {
    assert(impl != nullptr && db != nullptr);
    assert(table_input_json != nullptr && resp != nullptr);
    char body[kGlueMaxBody];
    int n = std::snprintf(body, sizeof(body),
                          "{\"DatabaseName\":\"%s\",\"TableInput\":%s}",
                          db, table_input_json);
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(body)) return kCatBadArg;
    return glue_post(impl, "AWSGlue.UpdateTable", body,
                     static_cast<uint32_t>(n), resp);
}

int glue_search_tables(Impl* impl, const char* search_text,
                       bolt::net::HttpResponse* resp) noexcept {
    assert(impl != nullptr && search_text != nullptr && resp != nullptr);
    char body[kGlueMaxBody];
    int n = std::snprintf(body, sizeof(body), "{\"SearchText\":\"%s\"}",
                          search_text);
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(body)) return kCatBadArg;
    return glue_post(impl, "AWSGlue.SearchTables", body,
                     static_cast<uint32_t>(n), resp);
}

// --- Partition CRUD --------------------------------------------------------

int glue_get_partitions(Impl* impl, const char* db, const char* table,
                        const char* expression_or_null,
                        bolt::net::HttpResponse* resp) noexcept {
    assert(impl != nullptr && db != nullptr);
    assert(table != nullptr && resp != nullptr);
    char body[kGlueMaxBody];
    int n;
    if (expression_or_null != nullptr && expression_or_null[0] != '\0') {
        n = std::snprintf(body, sizeof(body),
                          "{\"DatabaseName\":\"%s\",\"TableName\":\"%s\","
                          "\"Expression\":\"%s\"}",
                          db, table, expression_or_null);
    } else {
        n = std::snprintf(body, sizeof(body),
                          "{\"DatabaseName\":\"%s\",\"TableName\":\"%s\"}",
                          db, table);
    }
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(body)) return kCatBadArg;
    return glue_post(impl, "AWSGlue.GetPartitions", body,
                     static_cast<uint32_t>(n), resp);
}

int glue_batch_create_partition(Impl* impl, const char* db, const char* table,
                                const char* partition_inputs_json,
                                bolt::net::HttpResponse* resp) noexcept {
    assert(impl != nullptr && db != nullptr && table != nullptr);
    assert(partition_inputs_json != nullptr && resp != nullptr);
    char body[kGlueMaxBody];
    int n = std::snprintf(body, sizeof(body),
                          "{\"DatabaseName\":\"%s\",\"TableName\":\"%s\","
                          "\"PartitionInputList\":%s}",
                          db, table, partition_inputs_json);
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(body)) return kCatBadArg;
    return glue_post(impl, "AWSGlue.BatchCreatePartition", body,
                     static_cast<uint32_t>(n), resp);
}

int glue_batch_update_partition(Impl* impl, const char* db, const char* table,
                                const char* entries_json,
                                bolt::net::HttpResponse* resp) noexcept {
    assert(impl != nullptr && db != nullptr && table != nullptr);
    assert(entries_json != nullptr && resp != nullptr);
    char body[kGlueMaxBody];
    int n = std::snprintf(body, sizeof(body),
                          "{\"DatabaseName\":\"%s\",\"TableName\":\"%s\","
                          "\"Entries\":%s}",
                          db, table, entries_json);
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(body)) return kCatBadArg;
    return glue_post(impl, "AWSGlue.BatchUpdatePartition", body,
                     static_cast<uint32_t>(n), resp);
}

// --- Schema registry -------------------------------------------------------

int glue_create_schema(Impl* impl, const char* registry_name,
                       const char* schema_name, const char* data_format,
                       const char* schema_definition,
                       bolt::net::HttpResponse* resp) noexcept {
    assert(impl != nullptr && schema_name != nullptr);
    assert(data_format != nullptr && schema_definition != nullptr);
    assert(resp != nullptr);
    char body[kGlueMaxBody];
    int n;
    if (registry_name != nullptr && registry_name[0] != '\0') {
        n = std::snprintf(body, sizeof(body),
                          "{\"RegistryId\":{\"RegistryName\":\"%s\"},"
                          "\"SchemaName\":\"%s\",\"DataFormat\":\"%s\","
                          "\"SchemaDefinition\":\"%s\"}",
                          registry_name, schema_name, data_format,
                          schema_definition);
    } else {
        n = std::snprintf(body, sizeof(body),
                          "{\"SchemaName\":\"%s\",\"DataFormat\":\"%s\","
                          "\"SchemaDefinition\":\"%s\"}",
                          schema_name, data_format, schema_definition);
    }
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(body)) return kCatBadArg;
    return glue_post(impl, "AWSGlue.CreateSchema", body,
                     static_cast<uint32_t>(n), resp);
}

int glue_register_schema_version(Impl* impl, const char* schema_arn,
                                 const char* schema_definition,
                                 bolt::net::HttpResponse* resp) noexcept {
    assert(impl != nullptr && schema_arn != nullptr);
    assert(schema_definition != nullptr && resp != nullptr);
    char body[kGlueMaxBody];
    int n = std::snprintf(body, sizeof(body),
                          "{\"SchemaId\":{\"SchemaArn\":\"%s\"},"
                          "\"SchemaDefinition\":\"%s\"}",
                          schema_arn, schema_definition);
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(body)) return kCatBadArg;
    return glue_post(impl, "AWSGlue.RegisterSchemaVersion", body,
                     static_cast<uint32_t>(n), resp);
}

// --- Lake Formation grants -------------------------------------------------

int glue_grant_permissions(Impl* impl, const char* principal_arn,
                           const char* resource_json,
                           const char* permissions_json,
                           bolt::net::HttpResponse* resp) noexcept {
    assert(impl != nullptr && principal_arn != nullptr);
    assert(resource_json != nullptr && permissions_json != nullptr);
    assert(resp != nullptr);
    char host[kGlueMaxHost];
    std::snprintf(host, sizeof(host), "lakeformation.%s.amazonaws.com",
                  impl->cfg.region);
    char body[kGlueMaxBody];
    int n = std::snprintf(body, sizeof(body),
                          "{\"Principal\":{\"DataLakePrincipalIdentifier\":"
                          "\"%s\"},\"Resource\":%s,\"Permissions\":%s}",
                          principal_arn, resource_json, permissions_json);
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(body)) return kCatBadArg;
    return signed_post(impl, "lakeformation", host,
                       "AWSLakeFormation.GrantPermissions", body,
                       static_cast<uint32_t>(n), resp);
}

int glue_list_permissions(Impl* impl, const char* principal_arn,
                          bolt::net::HttpResponse* resp) noexcept {
    assert(impl != nullptr && principal_arn != nullptr && resp != nullptr);
    char host[kGlueMaxHost];
    std::snprintf(host, sizeof(host), "lakeformation.%s.amazonaws.com",
                  impl->cfg.region);
    char body[kGlueMaxBody];
    int n = std::snprintf(body, sizeof(body),
                          "{\"Principal\":{\"DataLakePrincipalIdentifier\":"
                          "\"%s\"}}",
                          principal_arn);
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(body)) return kCatBadArg;
    return signed_post(impl, "lakeformation", host,
                       "AWSLakeFormation.ListPermissions", body,
                       static_cast<uint32_t>(n), resp);
}

}  // namespace glue
}  // namespace lakehouse
}  // namespace bolt
