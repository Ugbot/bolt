// bolt/lakehouse/catalog/gravitino.h — Apache Gravitino metalake/catalog.
#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/lakehouse/catalog.h"
#include "bolt/lakehouse/catalog/secret.h"

namespace bolt {
namespace lakehouse {
namespace gravitino {

struct Config {
    char   base_url[512];
    Secret bearer_token;
    char   metalake[128];
    char   provider[32];     // "hive" | "iceberg" | "jdbc-mysql" | ...
};

bool open(Catalog* out, bolt::Arena* arena, const Config* cfg) noexcept;

}  // namespace gravitino
}  // namespace lakehouse
}  // namespace bolt
