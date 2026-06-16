// bolt/src/lakehouse/catalog_polaris.cpp — Polaris catalog as Iceberg REST.

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "bolt/lakehouse/catalog/polaris.h"

#include "bolt/lakehouse/catalog/iceberg_rest.h"

#include <cassert>
#include <cstring>

namespace bolt {
namespace lakehouse {
namespace polaris {

bool open(Catalog* out, bolt::Arena* arena, const Config* cfg) noexcept {
    assert(out != nullptr);
    assert(arena != nullptr && cfg != nullptr);
    iceberg_rest::Config irc;
    std::memset(&irc, 0, sizeof(irc));
    std::strncpy(irc.base_url, cfg->base_url, sizeof(irc.base_url) - 1u);
    std::strncpy(irc.prefix, "api/catalog/v1", sizeof(irc.prefix) - 1u);
    irc.auth_mode = 2;   // oauth2 client-credentials
    irc.oauth_client_id = cfg->oauth_client_id;
    irc.oauth_client_secret = cfg->oauth_client_secret;
    std::strncpy(irc.oauth_token_url, cfg->oauth_token_url,
                 sizeof(irc.oauth_token_url) - 1u);
    return iceberg_rest::open(out, arena, &irc);
}

int polaris_create_principal_role(void* impl, const char* role_name) noexcept {
    assert(role_name != nullptr); (void)impl; (void)role_name;
    // TODO(W7+): POST /api/management/v1/principal-roles {name}.
    return kCatNotImplemented;
}

int polaris_grant_catalog_role(void* impl, const char* principal_role,
                               const char* catalog_role) noexcept {
    assert(principal_role != nullptr && catalog_role != nullptr);
    (void)impl; (void)principal_role; (void)catalog_role;
    // TODO(W7+): PUT /api/management/v1/principal-roles/{pr}/catalog-roles/{cr}.
    return kCatNotImplemented;
}

}  // namespace polaris
}  // namespace lakehouse
}  // namespace bolt
