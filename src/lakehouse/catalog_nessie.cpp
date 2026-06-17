// bolt/src/lakehouse/catalog_nessie.cpp — Nessie catalog (G2CLUS-181).
//
// Iceberg REST surface (via the shared iceberg_rest client under an /iceberg
// prefix) PLUS the Nessie-native v2 tree API for git-style workflows: commit,
// merge, transplant, compare-references. The native calls are shaped here and
// sent over the bolt HTTP client with the resolved bearer token.

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "bolt/lakehouse/catalog/nessie.h"

#include "bolt/lakehouse/catalog/iceberg_rest.h"
#include "bolt/parse/bolt_json.h"

#include <cassert>
#include <cstdio>
#include <cstring>

namespace bolt {
namespace lakehouse {
namespace nessie {

namespace {

namespace bj = bolt::parse::json;

int classify_status(int status) noexcept {
    if (status == 404) return kCatNotFound;
    if (status == 409) return kCatExists;
    if (status / 100 != 2) return kCatIoError;
    return kCatOk;
}

void add_common(bolt::net::HttpRequest* req, const char* bearer,
                bool with_json_body) noexcept {
    assert(req != nullptr);
    bolt::net::http_request_add_header(req, "Accept", "application/json");
    if (with_json_body)
        bolt::net::http_request_add_header(req, "Content-Type",
                                           "application/json");
    if (bearer != nullptr && bearer[0] != '\0') {
        char buf[600];
        int n = std::snprintf(buf, sizeof(buf), "Bearer %s", bearer);
        if (n > 0 && static_cast<uint32_t>(n) < sizeof(buf))
            bolt::net::http_request_add_header(req, "Authorization", buf);
    }
}

bolt::Arena* arena_of(Catalog* cat) noexcept {
    assert(cat != nullptr && cat->impl != nullptr);
    return static_cast<iceberg_rest::Impl*>(cat->impl)->arena;
}

// Resolve base_url + bearer token for a Nessie catalog. Returns kCat* code.
int resolve_endpoint(Catalog* cat, const char** out_base, char* token,
                     uint32_t token_cap) noexcept {
    assert(cat != nullptr && out_base != nullptr && token != nullptr);
    *out_base = iceberg_rest::base_url_of(cat);
    if (*out_base == nullptr) return kCatBadArg;
    token[0] = '\0';
    int tr = iceberg_rest::resolve_bearer(cat, token, token_cap);
    // No-auth Nessie is valid; treat kCatBadArg (auth_mode none) as "no token".
    if (tr == kCatBadArg) return kCatOk;
    return tr;
}

// Parse "hash":"..." from a single-reference response into out[0..cap).
int parse_ref_hash(const uint8_t* body, uint32_t len, bolt::Arena* arena,
                   char* out, uint32_t cap) noexcept {
    assert(body != nullptr && arena != nullptr && out != nullptr && cap > 0u);
    out[0] = '\0';
    bj::StructuralIndex idx;
    if (!bj::build_index(body, static_cast<int32_t>(len), arena, &idx))
        return kCatIoError;
    for (int32_t i = 1; i + 1 < idx.token_count; ++i) {
        const bj::Token& t = idx.tokens[i];
        if ((t.type == bj::TokenType::Key || t.type == bj::TokenType::String) &&
            t.length == 4 && std::memcmp(idx.src + t.start, "hash", 4) == 0 &&
            idx.tokens[i + 1].type == bj::TokenType::String) {
            uint32_t L = static_cast<uint32_t>(idx.tokens[i + 1].length);
            if (L >= cap) L = cap - 1u;
            std::memcpy(out, idx.src + idx.tokens[i + 1].start, L);
            out[L] = '\0';
            return kCatOk;
        }
    }
    return kCatOk;   // ref exists but no hash field — leave empty.
}

}  // namespace

bool open(Catalog* out, bolt::Arena* arena, const Config* cfg) noexcept {
    assert(out != nullptr);
    assert(arena != nullptr && cfg != nullptr);
    iceberg_rest::Config irc;
    std::memset(&irc, 0, sizeof(irc));
    std::strncpy(irc.base_url, cfg->base_url, sizeof(irc.base_url) - 1u);
    std::strncpy(irc.prefix, kNessieIcebergPrefix, sizeof(irc.prefix) - 1u);
    if (cfg->bearer_token.value[0] != '\0') {
        irc.auth_mode = iceberg_rest::kAuthBearer;
        irc.bearer_token = cfg->bearer_token;
    }
    return iceberg_rest::open(out, arena, &irc);
}

// --- Request builders ---

bool nessie_build_get_reference(const char* base_url, const char* bearer_token,
                                const char* ref_name,
                                bolt::net::HttpRequest* out) noexcept {
    assert(base_url != nullptr && ref_name != nullptr && out != nullptr);
    std::memset(out, 0, sizeof(*out));
    std::strncpy(out->method, "GET", sizeof(out->method) - 1u);
    int u = std::snprintf(out->url, sizeof(out->url),
                          "%s/api/v2/trees/%s", base_url, ref_name);
    if (u <= 0 || static_cast<uint32_t>(u) >= sizeof(out->url)) return false;
    add_common(out, bearer_token, false);
    return true;
}

bool nessie_build_commit(const char* base_url, const char* bearer_token,
                         const char* branch, const char* expected_hash,
                         const char* message, const char* operations_json,
                         bolt::net::HttpRequest* out, char* body_buf,
                         uint32_t body_cap) noexcept {
    assert(base_url != nullptr && branch != nullptr && message != nullptr);
    assert(operations_json != nullptr && out != nullptr && body_buf != nullptr);
    std::memset(out, 0, sizeof(*out));
    std::strncpy(out->method, "POST", sizeof(out->method) - 1u);
    int u;
    if (expected_hash != nullptr && expected_hash[0] != '\0') {
        u = std::snprintf(out->url, sizeof(out->url),
                          "%s/api/v2/trees/%s@%s/history/commit", base_url,
                          branch, expected_hash);
    } else {
        u = std::snprintf(out->url, sizeof(out->url),
                          "%s/api/v2/trees/%s/history/commit", base_url, branch);
    }
    if (u <= 0 || static_cast<uint32_t>(u) >= sizeof(out->url)) return false;
    add_common(out, bearer_token, true);
    int b = std::snprintf(body_buf, body_cap,
                          "{\"commitMeta\":{\"message\":\"%s\"},"
                          "\"operations\":%s}", message, operations_json);
    if (b <= 0 || static_cast<uint32_t>(b) >= body_cap) return false;
    out->body = reinterpret_cast<const uint8_t*>(body_buf);
    out->body_len = static_cast<uint32_t>(b);
    return true;
}

bool nessie_build_merge(const char* base_url, const char* bearer_token,
                        const char* from_ref, const char* from_hash,
                        const char* to_branch, bolt::net::HttpRequest* out,
                        char* body_buf, uint32_t body_cap) noexcept {
    assert(base_url != nullptr && from_ref != nullptr && to_branch != nullptr);
    assert(out != nullptr && body_buf != nullptr);
    std::memset(out, 0, sizeof(*out));
    std::strncpy(out->method, "POST", sizeof(out->method) - 1u);
    int u = std::snprintf(out->url, sizeof(out->url),
                          "%s/api/v2/trees/%s/history/merge", base_url,
                          to_branch);
    if (u <= 0 || static_cast<uint32_t>(u) >= sizeof(out->url)) return false;
    add_common(out, bearer_token, true);
    int b;
    if (from_hash != nullptr && from_hash[0] != '\0') {
        b = std::snprintf(body_buf, body_cap,
                          "{\"fromRefName\":\"%s\",\"fromHash\":\"%s\"}",
                          from_ref, from_hash);
    } else {
        b = std::snprintf(body_buf, body_cap,
                          "{\"fromRefName\":\"%s\"}", from_ref);
    }
    if (b <= 0 || static_cast<uint32_t>(b) >= body_cap) return false;
    out->body = reinterpret_cast<const uint8_t*>(body_buf);
    out->body_len = static_cast<uint32_t>(b);
    return true;
}

bool nessie_build_transplant(const char* base_url, const char* bearer_token,
                             const char* from_ref, const char* to_branch,
                             const char* const* hashes, uint32_t n,
                             bolt::net::HttpRequest* out, char* body_buf,
                             uint32_t body_cap) noexcept {
    assert(base_url != nullptr && from_ref != nullptr && to_branch != nullptr);
    assert(out != nullptr && body_buf != nullptr);
    assert(hashes != nullptr || n == 0u);
    std::memset(out, 0, sizeof(*out));
    std::strncpy(out->method, "POST", sizeof(out->method) - 1u);
    int u = std::snprintf(out->url, sizeof(out->url),
                          "%s/api/v2/trees/%s/history/transplant", base_url,
                          to_branch);
    if (u <= 0 || static_cast<uint32_t>(u) >= sizeof(out->url)) return false;
    add_common(out, bearer_token, true);
    int w = std::snprintf(body_buf, body_cap,
                          "{\"fromRefName\":\"%s\",\"hashesToTransplant\":[",
                          from_ref);
    if (w <= 0 || static_cast<uint32_t>(w) >= body_cap) return false;
    uint32_t o = static_cast<uint32_t>(w);
    for (uint32_t i = 0; i < n; ++i) {
        assert(hashes[i] != nullptr);
        int k = std::snprintf(body_buf + o, body_cap - o, "%s\"%s\"",
                              i == 0 ? "" : ",", hashes[i]);
        if (k <= 0 || o + static_cast<uint32_t>(k) >= body_cap) return false;
        o += static_cast<uint32_t>(k);
    }
    int e = std::snprintf(body_buf + o, body_cap - o, "]}");
    if (e <= 0 || o + static_cast<uint32_t>(e) >= body_cap) return false;
    out->body = reinterpret_cast<const uint8_t*>(body_buf);
    out->body_len = o + static_cast<uint32_t>(e);
    return true;
}

bool nessie_build_compare_references(const char* base_url,
                                     const char* bearer_token,
                                     const char* from_ref, const char* to_ref,
                                     bolt::net::HttpRequest* out) noexcept {
    assert(base_url != nullptr && from_ref != nullptr && to_ref != nullptr);
    assert(out != nullptr);
    std::memset(out, 0, sizeof(*out));
    std::strncpy(out->method, "GET", sizeof(out->method) - 1u);
    int u = std::snprintf(out->url, sizeof(out->url),
                          "%s/api/v2/trees/%s/diff/%s", base_url, from_ref,
                          to_ref);
    if (u <= 0 || static_cast<uint32_t>(u) >= sizeof(out->url)) return false;
    add_common(out, bearer_token, false);
    return true;
}

// --- Public native ops ---

int nessie_get_reference(Catalog* cat, const char* ref_name, char* out_hash,
                         uint32_t cap) noexcept {
    assert(cat != nullptr && ref_name != nullptr && out_hash != nullptr);
    const char* base;
    char token[512];
    int rc = resolve_endpoint(cat, &base, token, sizeof(token));
    if (rc != kCatOk) return rc;
    bolt::net::HttpRequest req;
    if (!nessie_build_get_reference(base, token, ref_name, &req))
        return kCatBadArg;
    bolt::net::HttpResponse resp;
    if (bolt::net::http_send(arena_of(cat), &req, &resp) != 0) return kCatIoError;
    int sc = classify_status(resp.status);
    if (sc != kCatOk) return sc;
    return parse_ref_hash(resp.body, resp.body_len, arena_of(cat), out_hash, cap);
}

int nessie_commit(Catalog* cat, const char* branch, const char* expected_hash,
                  const char* message, const char* operations_json) noexcept {
    assert(cat != nullptr && branch != nullptr && message != nullptr);
    assert(operations_json != nullptr);
    const char* base;
    char token[512];
    int rc = resolve_endpoint(cat, &base, token, sizeof(token));
    if (rc != kCatOk) return rc;
    bolt::net::HttpRequest req;
    const uint32_t kBody = 64u * 1024u;
    char* body = static_cast<char*>(arena_of(cat)->allocate(kBody, 16u));
    if (body == nullptr) return kCatIoError;
    if (!nessie_build_commit(base, token, branch, expected_hash, message,
                             operations_json, &req, body, kBody))
        return kCatBadArg;
    bolt::net::HttpResponse resp;
    if (bolt::net::http_send(arena_of(cat), &req, &resp) != 0) return kCatIoError;
    return classify_status(resp.status);
}

int nessie_merge(Catalog* cat, const char* from_ref, const char* from_hash,
                 const char* to_branch) noexcept {
    assert(cat != nullptr && from_ref != nullptr && to_branch != nullptr);
    const char* base;
    char token[512];
    int rc = resolve_endpoint(cat, &base, token, sizeof(token));
    if (rc != kCatOk) return rc;
    bolt::net::HttpRequest req;
    char body[512];
    if (!nessie_build_merge(base, token, from_ref, from_hash, to_branch, &req,
                            body, sizeof(body))) return kCatBadArg;
    bolt::net::HttpResponse resp;
    if (bolt::net::http_send(arena_of(cat), &req, &resp) != 0) return kCatIoError;
    return classify_status(resp.status);
}

int nessie_transplant(Catalog* cat, const char* from_ref, const char* to_branch,
                      const char* const* hashes, uint32_t n) noexcept {
    assert(cat != nullptr && from_ref != nullptr && to_branch != nullptr);
    assert(hashes != nullptr || n == 0u);
    const char* base;
    char token[512];
    int rc = resolve_endpoint(cat, &base, token, sizeof(token));
    if (rc != kCatOk) return rc;
    bolt::net::HttpRequest req;
    const uint32_t kBody = 16u * 1024u;
    char* body = static_cast<char*>(arena_of(cat)->allocate(kBody, 16u));
    if (body == nullptr) return kCatIoError;
    if (!nessie_build_transplant(base, token, from_ref, to_branch, hashes, n,
                                 &req, body, kBody)) return kCatBadArg;
    bolt::net::HttpResponse resp;
    if (bolt::net::http_send(arena_of(cat), &req, &resp) != 0) return kCatIoError;
    return classify_status(resp.status);
}

int nessie_compare_references(Catalog* cat, const char* from_ref,
                              const char* to_ref, const uint8_t** out_body,
                              uint32_t* out_len) noexcept {
    assert(cat != nullptr && from_ref != nullptr && to_ref != nullptr);
    assert(out_body != nullptr && out_len != nullptr);
    *out_body = nullptr;
    *out_len = 0;
    const char* base;
    char token[512];
    int rc = resolve_endpoint(cat, &base, token, sizeof(token));
    if (rc != kCatOk) return rc;
    bolt::net::HttpRequest req;
    if (!nessie_build_compare_references(base, token, from_ref, to_ref, &req))
        return kCatBadArg;
    bolt::net::HttpResponse resp;
    if (bolt::net::http_send(arena_of(cat), &req, &resp) != 0) return kCatIoError;
    int sc = classify_status(resp.status);
    if (sc != kCatOk) return sc;
    *out_body = resp.body;
    *out_len = resp.body_len;
    return kCatOk;
}

}  // namespace nessie
}  // namespace lakehouse
}  // namespace bolt
