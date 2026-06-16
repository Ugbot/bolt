// bolt/lakehouse/catalog/nessie.h — Project Nessie catalog (Iceberg REST + git-like refs).
#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/lakehouse/catalog.h"
#include "bolt/lakehouse/catalog/secret.h"

namespace bolt {
namespace lakehouse {
namespace nessie {

struct Config {
    char   base_url[512];
    Secret bearer_token;
    char   reference[128];   // e.g. "main"
};

bool open(Catalog* out, bolt::Arena* arena, const Config* cfg) noexcept;

// W7+ stubs.
int nessie_get_reference(void* impl, const char* ref_name) noexcept;
int nessie_commit(void* impl, const char* ref_name, const char* msg) noexcept;
int nessie_merge(void* impl, const char* from_ref, const char* to_ref) noexcept;

}  // namespace nessie
}  // namespace lakehouse
}  // namespace bolt
