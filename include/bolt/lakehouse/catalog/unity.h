// bolt/lakehouse/catalog/unity.h — Databricks Unity Catalog REST.
// HTTPS + Bearer PAT.
#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/lakehouse/catalog.h"
#include "bolt/lakehouse/catalog/secret.h"

namespace bolt {
namespace lakehouse {
namespace unity {

struct Config {
    char   base_url[512];        // e.g. "https://dbc-xxx.cloud.databricks.com"
    Secret bearer_token;
};

bool open(Catalog* out, bolt::Arena* arena, const Config* cfg) noexcept;

}  // namespace unity
}  // namespace lakehouse
}  // namespace bolt
