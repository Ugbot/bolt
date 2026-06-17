// bolt/src/lakehouse/catalog_polaris.cpp — Polaris catalog (G2CLUS-180).
//
// Apache Polaris speaks the Iceberg REST Catalog spec under an
// `/api/catalog/v1` prefix with Polaris-flavoured OAuth2 client-credentials.
// The Iceberg-compatible surface (namespace/table/view/snapshot CRUD) is the
// shared iceberg_rest client — this file is only the thin open() wrapper plus
// the Polaris-native management API (principal roles / catalog roles) under
// `/api/management/v1`.

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "bolt/lakehouse/catalog/polaris.h"

#include "bolt/lakehouse/catalog/iceberg_rest.h"

#include <cassert>
#include <cstdio>
#include <cstring>

namespace bolt {
namespace lakehouse {
namespace polaris {

namespace {

int classify_status(int status) noexcept {
    if (status == 404) return kCatNotFound;
    if (status == 409) return kCatExists;
    if (status / 100 != 2) return kCatIoError;
    return kCatOk;
}

void add_bearer(bolt::net::HttpRequest* req, const char* token) noexcept {
    assert(req != nullptr && token != nullptr);
    char buf[600];
    int n = std::snprintf(buf, sizeof(buf), "Bearer %s", token);
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(buf)) return;
    bolt::net::http_request_add_header(req, "Authorization", buf);
}

}  // namespace

bool open(Catalog* out, bolt::Arena* arena, const Config* cfg) noexcept {
    assert(out != nullptr);
    assert(arena != nullptr && cfg != nullptr);
    iceberg_rest::Config irc;
    std::memset(&irc, 0, sizeof(irc));
    std::strncpy(irc.base_url, cfg->base_url, sizeof(irc.base_url) - 1u);
    std::strncpy(irc.prefix, kPolarisCatalogPrefix, sizeof(irc.prefix) - 1u);
    irc.auth_mode = iceberg_rest::kAuthOAuth2;   // Polaris client-credentials
    irc.oauth_client_id = cfg->oauth_client_id;
    irc.oauth_client_secret = cfg->oauth_client_secret;
    std::strncpy(irc.oauth_token_url, cfg->oauth_token_url,
                 sizeof(irc.oauth_token_url) - 1u);
    std::strncpy(irc.oauth_scope, cfg->oauth_scope, sizeof(irc.oauth_scope) - 1u);
    return iceberg_rest::open(out, arena, &irc);
}

bool polaris_build_create_principal_role(const char* base_url,
                                         const char* bearer_token,
                                         const char* role_name,
                                         bolt::net::HttpRequest* out,
                                         char* body_buf,
                                         uint32_t body_cap) noexcept {
    assert(base_url != nullptr && role_name != nullptr);
    assert(out != nullptr && body_buf != nullptr);
    std::memset(out, 0, sizeof(*out));
    std::strncpy(out->method, "POST", sizeof(out->method) - 1u);
    int u = std::snprintf(out->url, sizeof(out->url),
                          "%s/api/management/v1/principal-roles", base_url);
    if (u <= 0 || static_cast<uint32_t>(u) >= sizeof(out->url)) return false;
    if (!bolt::net::http_request_add_header(out, "Content-Type",
                                            "application/json")) return false;
    if (!bolt::net::http_request_add_header(out, "Accept", "application/json"))
        return false;
    if (bearer_token != nullptr && bearer_token[0] != '\0')
        add_bearer(out, bearer_token);
    int b = std::snprintf(body_buf, body_cap,
                          "{\"principalRole\":{\"name\":\"%s\"}}", role_name);
    if (b <= 0 || static_cast<uint32_t>(b) >= body_cap) return false;
    out->body = reinterpret_cast<const uint8_t*>(body_buf);
    out->body_len = static_cast<uint32_t>(b);
    return true;
}

bool polaris_build_grant_catalog_role(const char* base_url,
                                      const char* bearer_token,
                                      const char* principal_role,
                                      const char* catalog,
                                      const char* catalog_role,
                                      bolt::net::HttpRequest* out,
                                      char* body_buf,
                                      uint32_t body_cap) noexcept {
    assert(base_url != nullptr && principal_role != nullptr);
    assert(catalog != nullptr && catalog_role != nullptr);
    assert(out != nullptr && body_buf != nullptr);
    std::memset(out, 0, sizeof(*out));
    std::strncpy(out->method, "PUT", sizeof(out->method) - 1u);
    int u = std::snprintf(out->url, sizeof(out->url),
                          "%s/api/management/v1/principal-roles/%s/"
                          "catalog-roles/%s", base_url, principal_role, catalog);
    if (u <= 0 || static_cast<uint32_t>(u) >= sizeof(out->url)) return false;
    if (!bolt::net::http_request_add_header(out, "Content-Type",
                                            "application/json")) return false;
    if (!bolt::net::http_request_add_header(out, "Accept", "application/json"))
        return false;
    if (bearer_token != nullptr && bearer_token[0] != '\0')
        add_bearer(out, bearer_token);
    int b = std::snprintf(body_buf, body_cap,
                          "{\"catalogRole\":{\"name\":\"%s\"}}", catalog_role);
    if (b <= 0 || static_cast<uint32_t>(b) >= body_cap) return false;
    out->body = reinterpret_cast<const uint8_t*>(body_buf);
    out->body_len = static_cast<uint32_t>(b);
    return true;
}

int polaris_create_principal_role(Catalog* cat, const char* role_name) noexcept {
    assert(cat != nullptr && role_name != nullptr);
    const char* base = iceberg_rest::base_url_of(cat);
    if (base == nullptr) return kCatBadArg;
    char token[512];
    int tr = iceberg_rest::resolve_bearer(cat, token, sizeof(token));
    if (tr != kCatOk) return tr;
    bolt::net::HttpRequest req;
    char body[256];
    if (!polaris_build_create_principal_role(base, token, role_name, &req, body,
                                              sizeof(body))) return kCatBadArg;
    bolt::Arena* arena = static_cast<iceberg_rest::Impl*>(cat->impl)->arena;
    bolt::net::HttpResponse resp;
    if (bolt::net::http_send(arena, &req, &resp) != 0) return kCatIoError;
    return classify_status(resp.status);
}

int polaris_grant_catalog_role(Catalog* cat, const char* principal_role,
                               const char* catalog,
                               const char* catalog_role) noexcept {
    assert(cat != nullptr && principal_role != nullptr);
    assert(catalog != nullptr && catalog_role != nullptr);
    const char* base = iceberg_rest::base_url_of(cat);
    if (base == nullptr) return kCatBadArg;
    char token[512];
    int tr = iceberg_rest::resolve_bearer(cat, token, sizeof(token));
    if (tr != kCatOk) return tr;
    bolt::net::HttpRequest req;
    char body[256];
    if (!polaris_build_grant_catalog_role(base, token, principal_role, catalog,
                                          catalog_role, &req, body,
                                          sizeof(body))) return kCatBadArg;
    bolt::Arena* arena = static_cast<iceberg_rest::Impl*>(cat->impl)->arena;
    bolt::net::HttpResponse resp;
    if (bolt::net::http_send(arena, &req, &resp) != 0) return kCatIoError;
    return classify_status(resp.status);
}

}  // namespace polaris
}  // namespace lakehouse
}  // namespace bolt
