// bolt/lakehouse/catalog/polaris.h — Apache Polaris catalog (Iceberg REST + RBAC).
//
// Polaris (Snowflake's open catalog) speaks the Apache Iceberg REST Catalog
// spec under an `/api/catalog/v1` prefix, with a Polaris-specific OAuth2
// client-credentials flow. `open()` binds the full Iceberg REST client (F.1)
// so namespace/table/view/snapshot CRUD all reuse that implementation; the
// only Polaris-native surface is the management API (principal roles / catalog
// roles) under `/api/management/v1`.
//
// Tiger Style: PODs, ≥2 asserts/fn, no exceptions, no smart pointers, bounded.
#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/lakehouse/catalog.h"
#include "bolt/lakehouse/catalog/secret.h"
#include "bolt/net/bolt_http_client.h"

namespace bolt {
namespace lakehouse {
namespace polaris {

// Polaris REST prefix for the Iceberg-compatible catalog surface.
static constexpr const char* kPolarisCatalogPrefix = "api/catalog/v1";

struct Config {
    char   base_url[512];        // "https://host[:port]" — NO trailing slash
    Secret oauth_client_id;
    Secret oauth_client_secret;
    char   oauth_token_url[512]; // Polaris token endpoint
    char   oauth_scope[256];     // optional (e.g. "PRINCIPAL_ROLE:ALL")
};

// Bind `out` to a Polaris catalog backed by the Iceberg REST client. The
// returned Catalog routes list/load/create/commit/drop through the shared
// iceberg_rest implementation with the Polaris prefix + OAuth2 auth.
bool open(Catalog* out, bolt::Arena* arena, const Config* cfg) noexcept;

// --- Polaris management API (RBAC) ---
//
// These hit `/api/management/v1`, NOT the Iceberg catalog surface. They are
// shaped here and sent via the bolt HTTP client with the resolved OAuth2
// bearer token.

// Create a principal role: POST /api/management/v1/principal-roles.
int polaris_create_principal_role(Catalog* cat, const char* role_name) noexcept;

// Grant a catalog role to a principal role:
// PUT /api/management/v1/principal-roles/{pr}/catalog-roles/{catalog}.
int polaris_grant_catalog_role(Catalog* cat, const char* principal_role,
                               const char* catalog,
                               const char* catalog_role) noexcept;

// --- Request-shaping helpers (white-box; stable for unit tests) ---

// Build the create-principal-role request into `out` (no send). `body_buf`
// receives the JSON body. Returns false on overflow / bad arg.
bool polaris_build_create_principal_role(const char* base_url,
                                         const char* bearer_token,
                                         const char* role_name,
                                         bolt::net::HttpRequest* out,
                                         char* body_buf,
                                         uint32_t body_cap) noexcept;

// Build the grant-catalog-role request into `out` (no send). Returns false on
// overflow / bad arg.
bool polaris_build_grant_catalog_role(const char* base_url,
                                      const char* bearer_token,
                                      const char* principal_role,
                                      const char* catalog,
                                      const char* catalog_role,
                                      bolt::net::HttpRequest* out,
                                      char* body_buf,
                                      uint32_t body_cap) noexcept;

}  // namespace polaris
}  // namespace lakehouse
}  // namespace bolt
