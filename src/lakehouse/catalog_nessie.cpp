// bolt/src/lakehouse/catalog_nessie.cpp — Nessie via its Iceberg REST endpoint.

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "bolt/lakehouse/catalog/nessie.h"

#include "bolt/lakehouse/catalog/iceberg_rest.h"

#include <cassert>
#include <cstring>

namespace bolt {
namespace lakehouse {
namespace nessie {

bool open(Catalog* out, bolt::Arena* arena, const Config* cfg) noexcept {
    assert(out != nullptr);
    assert(arena != nullptr && cfg != nullptr);
    iceberg_rest::Config irc;
    std::memset(&irc, 0, sizeof(irc));
    std::strncpy(irc.base_url, cfg->base_url, sizeof(irc.base_url) - 1u);
    std::strncpy(irc.prefix, "iceberg", sizeof(irc.prefix) - 1u);
    if (cfg->bearer_token.value[0] != '\0') {
        irc.auth_mode = 1;
        irc.bearer_token = cfg->bearer_token;
    }
    return iceberg_rest::open(out, arena, &irc);
}

int nessie_get_reference(void* impl, const char* ref_name) noexcept {
    assert(ref_name != nullptr); (void)impl; (void)ref_name;
    // TODO(W7+): GET /api/v2/trees/{name}.
    return kCatNotImplemented;
}

int nessie_commit(void* impl, const char* ref_name, const char* msg) noexcept {
    assert(ref_name != nullptr && msg != nullptr);
    (void)impl; (void)ref_name; (void)msg;
    // TODO(W7+): POST /api/v2/trees/{ref}/history/commit.
    return kCatNotImplemented;
}

int nessie_merge(void* impl, const char* from_ref, const char* to_ref) noexcept {
    assert(from_ref != nullptr && to_ref != nullptr);
    (void)impl; (void)from_ref; (void)to_ref;
    // TODO(W7+): POST /api/v2/trees/{to}/history/merge.
    return kCatNotImplemented;
}

}  // namespace nessie
}  // namespace lakehouse
}  // namespace bolt
