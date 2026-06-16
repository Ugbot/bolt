// bolt/lakehouse/catalog/polaris.h — Apache Polaris catalog (Iceberg REST + RBAC).
#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/lakehouse/catalog.h"
#include "bolt/lakehouse/catalog/secret.h"

namespace bolt {
namespace lakehouse {
namespace polaris {

struct Config {
    char   base_url[512];
    Secret oauth_client_id;
    Secret oauth_client_secret;
    char   oauth_token_url[512];
};

bool open(Catalog* out, bolt::Arena* arena, const Config* cfg) noexcept;

// W7+ stubs.
int polaris_create_principal_role(void* impl, const char* role_name) noexcept;
int polaris_grant_catalog_role(void* impl, const char* principal_role,
                               const char* catalog_role) noexcept;

}  // namespace polaris
}  // namespace lakehouse
}  // namespace bolt
