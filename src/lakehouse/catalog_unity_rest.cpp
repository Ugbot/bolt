// bolt/src/lakehouse/catalog_unity_rest.cpp — Databricks Unity Catalog REST,
// full white-box client surface (declared in catalog/unity.h).
//
// The 6-fn Catalog vtable (open/list/create/drop) lives in catalog_unity.cpp.
// This file builds the wider Unity surface as request/body builders so the
// wire shape (PAT header, 3-level full_name, GRANT/share bodies) is unit-
// testable without a live workspace. Every fn is bounded + ≥2 asserts. Never
// logs the Bearer token.
//
// Base path: /api/2.1/unity-catalog/{catalogs,schemas,tables,volumes,
//   external-locations,storage-credentials,functions,models,permissions,
//   shares,recipients,lineage-tracking,tags}.

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "bolt/lakehouse/catalog/unity.h"

#include <cassert>
#include <cstdio>
#include <cstring>

namespace bolt {
namespace lakehouse {
namespace unity {

namespace {

constexpr char kBase[] = "/api/2.1/unity-catalog";

// Map a privilege to its UC wire token. Total over the enum.
const char* priv_str(Privilege p) noexcept {
    switch (p) {
        case Privilege::kSelect: return "SELECT";
        case Privilege::kModify: return "MODIFY";
        case Privilege::kUse:    return "USE_CATALOG";
        case Privilege::kCreate: return "CREATE";
        case Privilege::kAll:    return "ALL_PRIVILEGES";
    }
    return "SELECT";
}

// Map a securable to its /permissions/{securable} path segment. Total.
const char* securable_str(Securable s) noexcept {
    switch (s) {
        case Securable::kCatalog:          return "catalog";
        case Securable::kSchema:           return "schema";
        case Securable::kTable:            return "table";
        case Securable::kVolume:           return "volume";
        case Securable::kExternalLocation: return "external_location";
        case Securable::kFunction:         return "function";
        case Securable::kShare:            return "share";
    }
    return "table";
}

// Map a data source format to its UC wire token. Total.
const char* format_str(DataFormat f) noexcept {
    switch (f) {
        case DataFormat::kDelta:   return "DELTA";
        case DataFormat::kParquet: return "PARQUET";
        case DataFormat::kIceberg: return "ICEBERG";
        case DataFormat::kCsv:     return "CSV";
        case DataFormat::kJson:    return "JSON";
    }
    return "DELTA";
}

// Common request scaffold: method + "<base_url>/api/2.1/unity-catalog<suffix>"
// + Accept header. `suffix` already includes its leading '/'. Returns false on
// a method/url that would not fit (truncation is treated as failure).
bool begin_request(const Impl* impl, const char* method, const char* suffix,
                   bolt::net::HttpRequest* out) noexcept {
    assert(impl != nullptr && out != nullptr);
    assert(method != nullptr && suffix != nullptr);
    std::memset(out, 0, sizeof(*out));
    std::strncpy(out->method, method, sizeof(out->method) - 1u);
    int n = std::snprintf(out->url, sizeof(out->url), "%s%s%s",
                          impl->cfg.base_url, kBase, suffix);
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(out->url)) return false;
    if (!bolt::net::http_request_add_header(out, "Accept", "application/json"))
        return false;
    return uc_add_auth(impl, out);
}

// Attach a JSON body (already in `body`, length `len`) + Content-Type.
bool attach_json_body(bolt::net::HttpRequest* out, const char* body,
                      int len) noexcept {
    assert(out != nullptr && body != nullptr);
    if (len <= 0) return false;
    if (!bolt::net::http_request_add_header(out, "Content-Type",
                                            "application/json")) return false;
    out->body = reinterpret_cast<const uint8_t*>(body);
    out->body_len = static_cast<uint32_t>(len);
    return true;
}

}  // namespace

// --------------------------------------------------------------------------
// full_name + auth.
// --------------------------------------------------------------------------

bool uc_build_full_name(const char* catalog, const char* schema,
                        const char* table, char* out, uint32_t cap) noexcept {
    assert(out != nullptr);
    assert(cap > 0u && cap <= kUnityMaxFullName);
    const char* parts[3] = {catalog, schema, table};
    uint32_t pos = 0;
    bool any = false;
    for (uint32_t i = 0; i < 3u; ++i) {
        const char* p = parts[i];
        if (p == nullptr || p[0] == '\0') continue;
        if (any) {
            if (pos + 1u >= cap) return false;
            out[pos++] = '.';
        }
        uint32_t len = static_cast<uint32_t>(std::strlen(p));
        if (pos + len >= cap) return false;          // leave room for NUL
        std::memcpy(out + pos, p, len);
        pos += len;
        any = true;
    }
    if (!any) return false;
    assert(pos < cap);
    out[pos] = '\0';
    return true;
}

bool uc_add_auth(const Impl* impl, bolt::net::HttpRequest* req) noexcept {
    assert(impl != nullptr && req != nullptr);
    char buf[kSecretMax + 8u];
    int n = std::snprintf(buf, sizeof(buf), "Bearer %s",
                          impl->cfg.bearer_token.value);
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(buf)) return false;
    return bolt::net::http_request_add_header(req, "Authorization", buf);
}

// --------------------------------------------------------------------------
// Catalog CRUD.
// --------------------------------------------------------------------------

bool uc_build_list_catalogs_request(const Impl* impl,
                                    bolt::net::HttpRequest* out) noexcept {
    assert(impl != nullptr && out != nullptr);
    return begin_request(impl, "GET", "/catalogs", out);
}

bool uc_build_get_catalog_request(const Impl* impl, const char* name,
                                  bolt::net::HttpRequest* out) noexcept {
    assert(impl != nullptr && out != nullptr);
    assert(name != nullptr && name[0] != '\0');
    char suffix[kUnityMaxFullName + 16u];
    int n = std::snprintf(suffix, sizeof(suffix), "/catalogs/%s", name);
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(suffix)) return false;
    return begin_request(impl, "GET", suffix, out);
}

bool uc_build_create_catalog_request(const Impl* impl, const char* name,
                                     const char* comment, char* body,
                                     uint32_t body_cap,
                                     bolt::net::HttpRequest* out) noexcept {
    assert(impl != nullptr && out != nullptr && body != nullptr);
    assert(name != nullptr && name[0] != '\0' && body_cap > 0u);
    if (!begin_request(impl, "POST", "/catalogs", out)) return false;
    const char* c = (comment != nullptr) ? comment : "";
    int n = std::snprintf(body, body_cap,
                          "{\"name\":\"%s\",\"comment\":\"%s\"}", name, c);
    if (static_cast<uint32_t>(n) >= body_cap) return false;
    return attach_json_body(out, body, n);
}

bool uc_build_delete_catalog_request(const Impl* impl, const char* name,
                                     bolt::net::HttpRequest* out) noexcept {
    assert(impl != nullptr && out != nullptr);
    assert(name != nullptr && name[0] != '\0');
    char suffix[kUnityMaxFullName + 16u];
    int n = std::snprintf(suffix, sizeof(suffix), "/catalogs/%s", name);
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(suffix)) return false;
    return begin_request(impl, "DELETE", suffix, out);
}

// --------------------------------------------------------------------------
// Schema CRUD.
// --------------------------------------------------------------------------

bool uc_build_list_schemas_request(const Impl* impl, const char* catalog,
                                   bolt::net::HttpRequest* out) noexcept {
    assert(impl != nullptr && out != nullptr);
    assert(catalog != nullptr && catalog[0] != '\0');
    char suffix[kUnityMaxFullName + 32u];
    int n = std::snprintf(suffix, sizeof(suffix),
                          "/schemas?catalog_name=%s", catalog);
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(suffix)) return false;
    return begin_request(impl, "GET", suffix, out);
}

bool uc_build_create_schema_request(const Impl* impl, const char* catalog,
                                    const char* name, char* body,
                                    uint32_t body_cap,
                                    bolt::net::HttpRequest* out) noexcept {
    assert(impl != nullptr && out != nullptr && body != nullptr);
    assert(catalog != nullptr && name != nullptr && body_cap > 0u);
    if (!begin_request(impl, "POST", "/schemas", out)) return false;
    int n = std::snprintf(body, body_cap,
                          "{\"name\":\"%s\",\"catalog_name\":\"%s\"}",
                          name, catalog);
    if (static_cast<uint32_t>(n) >= body_cap) return false;
    return attach_json_body(out, body, n);
}

bool uc_build_delete_schema_request(const Impl* impl, const char* full_name,
                                    bolt::net::HttpRequest* out) noexcept {
    assert(impl != nullptr && out != nullptr);
    assert(full_name != nullptr && full_name[0] != '\0');
    char suffix[kUnityMaxFullName + 16u];
    int n = std::snprintf(suffix, sizeof(suffix), "/schemas/%s", full_name);
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(suffix)) return false;
    return begin_request(impl, "DELETE", suffix, out);
}

// --------------------------------------------------------------------------
// Table CRUD.
// --------------------------------------------------------------------------

bool uc_build_list_tables_request(const Impl* impl, const char* catalog,
                                  const char* schema,
                                  bolt::net::HttpRequest* out) noexcept {
    assert(impl != nullptr && out != nullptr);
    assert(catalog != nullptr && schema != nullptr);
    char suffix[kUnityMaxFullName + 48u];
    int n = std::snprintf(suffix, sizeof(suffix),
                          "/tables?catalog_name=%s&schema_name=%s",
                          catalog, schema);
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(suffix)) return false;
    return begin_request(impl, "GET", suffix, out);
}

bool uc_build_get_table_request(const Impl* impl, const char* full_name,
                                bolt::net::HttpRequest* out) noexcept {
    assert(impl != nullptr && out != nullptr);
    assert(full_name != nullptr && full_name[0] != '\0');
    char suffix[kUnityMaxFullName + 16u];
    int n = std::snprintf(suffix, sizeof(suffix), "/tables/%s", full_name);
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(suffix)) return false;
    return begin_request(impl, "GET", suffix, out);
}

bool uc_build_create_table_request(const Impl* impl, const char* catalog,
                                   const char* schema, const char* name,
                                   DataFormat fmt, const char* storage_location,
                                   char* body, uint32_t body_cap,
                                   bolt::net::HttpRequest* out) noexcept {
    assert(impl != nullptr && out != nullptr && body != nullptr);
    assert(catalog != nullptr && schema != nullptr && name != nullptr);
    if (!begin_request(impl, "POST", "/tables", out)) return false;
    const char* loc = (storage_location != nullptr) ? storage_location : "";
    int n = std::snprintf(
        body, body_cap,
        "{\"name\":\"%s\",\"catalog_name\":\"%s\",\"schema_name\":\"%s\","
        "\"table_type\":\"EXTERNAL\",\"data_source_format\":\"%s\","
        "\"storage_location\":\"%s\",\"columns\":[]}",
        name, catalog, schema, format_str(fmt), loc);
    if (n <= 0 || static_cast<uint32_t>(n) >= body_cap) return false;
    return attach_json_body(out, body, n);
}

bool uc_build_delete_table_request(const Impl* impl, const char* full_name,
                                   bolt::net::HttpRequest* out) noexcept {
    assert(impl != nullptr && out != nullptr);
    assert(full_name != nullptr && full_name[0] != '\0');
    char suffix[kUnityMaxFullName + 16u];
    int n = std::snprintf(suffix, sizeof(suffix), "/tables/%s", full_name);
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(suffix)) return false;
    return begin_request(impl, "DELETE", suffix, out);
}

// --------------------------------------------------------------------------
// Volume CRUD.
// --------------------------------------------------------------------------

bool uc_build_create_volume_request(const Impl* impl, const char* catalog,
                                    const char* schema, const char* name,
                                    VolumeType vtype, const char* storage_location,
                                    char* body, uint32_t body_cap,
                                    bolt::net::HttpRequest* out) noexcept {
    assert(impl != nullptr && out != nullptr && body != nullptr);
    assert(catalog != nullptr && schema != nullptr && name != nullptr);
    if (!begin_request(impl, "POST", "/volumes", out)) return false;
    const char* vt = (vtype == VolumeType::kManaged) ? "MANAGED" : "EXTERNAL";
    const char* loc = (storage_location != nullptr) ? storage_location : "";
    int n = std::snprintf(
        body, body_cap,
        "{\"name\":\"%s\",\"catalog_name\":\"%s\",\"schema_name\":\"%s\","
        "\"volume_type\":\"%s\",\"storage_location\":\"%s\"}",
        name, catalog, schema, vt, loc);
    if (n <= 0 || static_cast<uint32_t>(n) >= body_cap) return false;
    return attach_json_body(out, body, n);
}

bool uc_build_delete_volume_request(const Impl* impl, const char* full_name,
                                    bolt::net::HttpRequest* out) noexcept {
    assert(impl != nullptr && out != nullptr);
    assert(full_name != nullptr && full_name[0] != '\0');
    char suffix[kUnityMaxFullName + 16u];
    int n = std::snprintf(suffix, sizeof(suffix), "/volumes/%s", full_name);
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(suffix)) return false;
    return begin_request(impl, "DELETE", suffix, out);
}

// --------------------------------------------------------------------------
// External locations + storage credentials.
// --------------------------------------------------------------------------

bool uc_build_create_external_location_request(
    const Impl* impl, const char* name, const char* url,
    const char* credential_name, char* body, uint32_t body_cap,
    bolt::net::HttpRequest* out) noexcept {
    assert(impl != nullptr && out != nullptr && body != nullptr);
    assert(name != nullptr && url != nullptr && credential_name != nullptr);
    if (!begin_request(impl, "POST", "/external-locations", out)) return false;
    int n = std::snprintf(
        body, body_cap,
        "{\"name\":\"%s\",\"url\":\"%s\",\"credential_name\":\"%s\"}",
        name, url, credential_name);
    if (n <= 0 || static_cast<uint32_t>(n) >= body_cap) return false;
    return attach_json_body(out, body, n);
}

bool uc_build_create_storage_credential_request(
    const Impl* impl, const char* name, const char* iam_role_arn,
    char* body, uint32_t body_cap, bolt::net::HttpRequest* out) noexcept {
    assert(impl != nullptr && out != nullptr && body != nullptr);
    assert(name != nullptr && iam_role_arn != nullptr);
    if (!begin_request(impl, "POST", "/storage-credentials", out)) return false;
    int n = std::snprintf(
        body, body_cap,
        "{\"name\":\"%s\",\"aws_iam_role\":{\"role_arn\":\"%s\"}}",
        name, iam_role_arn);
    if (n <= 0 || static_cast<uint32_t>(n) >= body_cap) return false;
    return attach_json_body(out, body, n);
}

// --------------------------------------------------------------------------
// Functions + registered models (list only — read surface).
// --------------------------------------------------------------------------

bool uc_build_list_functions_request(const Impl* impl, const char* catalog,
                                     const char* schema,
                                     bolt::net::HttpRequest* out) noexcept {
    assert(impl != nullptr && out != nullptr);
    assert(catalog != nullptr && schema != nullptr);
    char suffix[kUnityMaxFullName + 48u];
    int n = std::snprintf(suffix, sizeof(suffix),
                          "/functions?catalog_name=%s&schema_name=%s",
                          catalog, schema);
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(suffix)) return false;
    return begin_request(impl, "GET", suffix, out);
}

bool uc_build_list_models_request(const Impl* impl, const char* catalog,
                                  const char* schema,
                                  bolt::net::HttpRequest* out) noexcept {
    assert(impl != nullptr && out != nullptr);
    assert(catalog != nullptr && schema != nullptr);
    char suffix[kUnityMaxFullName + 48u];
    int n = std::snprintf(suffix, sizeof(suffix),
                          "/models?catalog_name=%s&schema_name=%s",
                          catalog, schema);
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(suffix)) return false;
    return begin_request(impl, "GET", suffix, out);
}

// --------------------------------------------------------------------------
// Permissions — GET + PATCH (GRANT).
// --------------------------------------------------------------------------

bool uc_build_get_permissions_request(const Impl* impl, Securable sec,
                                      const char* full_name,
                                      bolt::net::HttpRequest* out) noexcept {
    assert(impl != nullptr && out != nullptr);
    assert(full_name != nullptr && full_name[0] != '\0');
    char suffix[kUnityMaxFullName + 48u];
    int n = std::snprintf(suffix, sizeof(suffix), "/permissions/%s/%s",
                          securable_str(sec), full_name);
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(suffix)) return false;
    return begin_request(impl, "GET", suffix, out);
}

bool uc_build_grant_request(const Impl* impl, Securable sec,
                            const char* full_name, const char* principal,
                            const Privilege* privs, uint32_t n_privs,
                            char* body, uint32_t body_cap,
                            bolt::net::HttpRequest* out) noexcept {
    assert(impl != nullptr && out != nullptr && body != nullptr);
    assert(full_name != nullptr && principal != nullptr);
    assert(privs != nullptr && n_privs > 0u && n_privs <= 16u);
    char suffix[kUnityMaxFullName + 48u];
    int sn = std::snprintf(suffix, sizeof(suffix), "/permissions/%s/%s",
                           securable_str(sec), full_name);
    if (sn <= 0 || static_cast<uint32_t>(sn) >= sizeof(suffix)) return false;
    if (!begin_request(impl, "PATCH", suffix, out)) return false;
    // {"changes":[{"principal":"P","add":["SELECT","MODIFY"]}]}
    uint32_t pos = 0;
    int w = std::snprintf(body, body_cap,
                          "{\"changes\":[{\"principal\":\"%s\",\"add\":[",
                          principal);
    if (w <= 0 || static_cast<uint32_t>(w) >= body_cap) return false;
    pos = static_cast<uint32_t>(w);
    for (uint32_t i = 0; i < n_privs; ++i) {
        w = std::snprintf(body + pos, body_cap - pos, "%s\"%s\"",
                          (i == 0) ? "" : ",", priv_str(privs[i]));
        if (w <= 0 || static_cast<uint32_t>(w) >= body_cap - pos) return false;
        pos += static_cast<uint32_t>(w);
    }
    w = std::snprintf(body + pos, body_cap - pos, "]}]}");
    if (w <= 0 || static_cast<uint32_t>(w) >= body_cap - pos) return false;
    pos += static_cast<uint32_t>(w);
    return attach_json_body(out, body, static_cast<int>(pos));
}

// --------------------------------------------------------------------------
// Delta Sharing — shares + recipients.
// --------------------------------------------------------------------------

bool uc_build_create_share_request(const Impl* impl, const char* name,
                                   const char* comment, char* body,
                                   uint32_t body_cap,
                                   bolt::net::HttpRequest* out) noexcept {
    assert(impl != nullptr && out != nullptr && body != nullptr);
    assert(name != nullptr && name[0] != '\0' && body_cap > 0u);
    if (!begin_request(impl, "POST", "/shares", out)) return false;
    const char* c = (comment != nullptr) ? comment : "";
    int n = std::snprintf(body, body_cap,
                          "{\"name\":\"%s\",\"comment\":\"%s\"}", name, c);
    if (n <= 0 || static_cast<uint32_t>(n) >= body_cap) return false;
    return attach_json_body(out, body, n);
}

bool uc_build_update_share_permissions_request(
    const Impl* impl, const char* share, const char* principal,
    Privilege priv, char* body, uint32_t body_cap,
    bolt::net::HttpRequest* out) noexcept {
    assert(impl != nullptr && out != nullptr && body != nullptr);
    assert(share != nullptr && principal != nullptr);
    char suffix[kUnityMaxFullName + 24u];
    int sn = std::snprintf(suffix, sizeof(suffix), "/shares/%s/permissions",
                           share);
    if (sn <= 0 || static_cast<uint32_t>(sn) >= sizeof(suffix)) return false;
    if (!begin_request(impl, "PATCH", suffix, out)) return false;
    int n = std::snprintf(
        body, body_cap,
        "{\"changes\":[{\"principal\":\"%s\",\"add\":[\"%s\"]}]}",
        principal, priv_str(priv));
    if (n <= 0 || static_cast<uint32_t>(n) >= body_cap) return false;
    return attach_json_body(out, body, n);
}

bool uc_build_create_recipient_request(const Impl* impl, const char* name,
                                       const char* comment, char* body,
                                       uint32_t body_cap,
                                       bolt::net::HttpRequest* out) noexcept {
    assert(impl != nullptr && out != nullptr && body != nullptr);
    assert(name != nullptr && name[0] != '\0' && body_cap > 0u);
    if (!begin_request(impl, "POST", "/recipients", out)) return false;
    const char* c = (comment != nullptr) ? comment : "";
    int n = std::snprintf(
        body, body_cap,
        "{\"name\":\"%s\",\"authentication_type\":\"TOKEN\",\"comment\":\"%s\"}",
        name, c);
    if (n <= 0 || static_cast<uint32_t>(n) >= body_cap) return false;
    return attach_json_body(out, body, n);
}

// --------------------------------------------------------------------------
// Lineage + tags.
// --------------------------------------------------------------------------

bool uc_build_table_lineage_request(const Impl* impl, const char* full_name,
                                    bolt::net::HttpRequest* out) noexcept {
    assert(impl != nullptr && out != nullptr);
    assert(full_name != nullptr && full_name[0] != '\0');
    char suffix[kUnityMaxFullName + 48u];
    int n = std::snprintf(suffix, sizeof(suffix),
                          "/lineage-tracking/table-lineage?table_name=%s",
                          full_name);
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(suffix)) return false;
    return begin_request(impl, "GET", suffix, out);
}

bool uc_build_column_lineage_request(const Impl* impl, const char* full_name,
                                     const char* column,
                                     bolt::net::HttpRequest* out) noexcept {
    assert(impl != nullptr && out != nullptr);
    assert(full_name != nullptr && column != nullptr);
    char suffix[kUnityMaxFullName + 64u];
    int n = std::snprintf(
        suffix, sizeof(suffix),
        "/lineage-tracking/column-lineage?table_name=%s&column_name=%s",
        full_name, column);
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(suffix)) return false;
    return begin_request(impl, "GET", suffix, out);
}

bool uc_build_set_table_tag_request(const Impl* impl, const char* full_name,
                                    const char* key, const char* value,
                                    char* body, uint32_t body_cap,
                                    bolt::net::HttpRequest* out) noexcept {
    assert(impl != nullptr && out != nullptr && body != nullptr);
    assert(full_name != nullptr && key != nullptr && value != nullptr);
    char suffix[kUnityMaxFullName + 24u];
    int sn = std::snprintf(suffix, sizeof(suffix), "/tags/tables/%s",
                           full_name);
    if (sn <= 0 || static_cast<uint32_t>(sn) >= sizeof(suffix)) return false;
    if (!begin_request(impl, "PATCH", suffix, out)) return false;
    int n = std::snprintf(body, body_cap,
                          "{\"tags\":{\"%s\":\"%s\"}}", key, value);
    if (n <= 0 || static_cast<uint32_t>(n) >= body_cap) return false;
    return attach_json_body(out, body, n);
}

}  // namespace unity
}  // namespace lakehouse
}  // namespace bolt
