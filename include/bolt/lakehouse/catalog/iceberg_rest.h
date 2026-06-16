// bolt/lakehouse/catalog/iceberg_rest.h — Apache Iceberg REST Catalog.
// Speaks the spec at /v1/{prefix}/namespaces and /v1/{prefix}/namespaces/{ns}/tables.
// Auth modes: none, bearer, oauth2 client-credentials, sigv4 (Glue-style).
#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/lakehouse/catalog.h"
#include "bolt/lakehouse/catalog/secret.h"
#include "bolt/net/bolt_http_client.h"

#include <cstdint>

namespace bolt {
namespace lakehouse {
namespace iceberg_rest {

struct Config {
    char    base_url[512];          // "https://host[:port]" — NO trailing slash
    char    prefix[128];             // e.g. "" or "ws/main"
    uint8_t auth_mode;               // 0=none, 1=bearer, 2=oauth2_cc, 3=sigv4
    Secret  bearer_token;
    char    oauth_token_url[512];
    Secret  oauth_client_id;
    Secret  oauth_client_secret;
    char    aws_region[32];
    Secret  aws_access_key;
    Secret  aws_secret_key;
};

// Bind `out` to a freshly-opened catalog. Allocates impl from `arena`.
bool open(Catalog* out, bolt::Arena* arena, const Config* cfg) noexcept;

// --- Test helpers (white-box). Stable for unit tests. ---

struct Impl {
    Config        cfg;
    bolt::Arena*  arena;
    char          cached_token[512];
    int64_t       token_expiry_epoch;
};

// Build a `GET /v1/{prefix}/namespaces` request (no send).
bool irc_build_list_namespaces_request(const Impl* impl,
                                       bolt::net::HttpRequest* out) noexcept;

// Parse a `{"namespaces": [["ns1"], ...]}` body into CatalogName[].
bool irc_parse_namespaces(const uint8_t* body, uint32_t len, bolt::Arena* arena,
                          CatalogName* out, uint32_t cap,
                          uint32_t* out_n) noexcept;

}  // namespace iceberg_rest
}  // namespace lakehouse
}  // namespace bolt
