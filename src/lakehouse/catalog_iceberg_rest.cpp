// bolt/src/lakehouse/catalog_iceberg_rest.cpp — Iceberg REST Catalog.
//
// Apache spec: https://iceberg.apache.org/concepts/catalog/#rest-catalog
// Implements list_namespaces / list_tables / create_table / drop_table.
// table_path + commit_metadata are stubbed for W6 (TODO W7+).

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "bolt/lakehouse/catalog/iceberg_rest.h"

#include "bolt/parse/bolt_json.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

namespace bolt {
namespace lakehouse {
namespace iceberg_rest {

namespace {

namespace bj = bolt::parse::json;

void make_url(const Config* c, char* out, uint32_t cap, const char* tail) noexcept {
    assert(c != nullptr);
    assert(out != nullptr && tail != nullptr);
    if (c->prefix[0] != '\0') {
        std::snprintf(out, cap, "%s/v1/%s%s", c->base_url, c->prefix, tail);
    } else {
        std::snprintf(out, cap, "%s/v1%s", c->base_url, tail);
    }
}

int64_t now_epoch() noexcept {
    return static_cast<int64_t>(std::time(nullptr));
}

// Token-level equality of a string/key token against a literal.
bool tok_str_eq(const bj::StructuralIndex& idx, int32_t i,
                const char* s, uint32_t n) noexcept {
    if (i < 0 || i >= idx.token_count) return false;
    const bj::Token& t = idx.tokens[i];
    if (t.type != bj::TokenType::String && t.type != bj::TokenType::Key)
        return false;
    if (static_cast<uint32_t>(t.length) != n) return false;
    return std::memcmp(idx.src + t.start, s, n) == 0;
}

// Find tokens.start span for a String/Key token (caller checks type).
const uint8_t* tok_bytes(const bj::StructuralIndex& idx, int32_t i,
                         uint32_t* out_len) noexcept {
    assert(i >= 0 && i < idx.token_count);
    const bj::Token& t = idx.tokens[i];
    *out_len = static_cast<uint32_t>(t.length);
    return idx.src + t.start;
}

// Walk tokens from `cursor` past one full value (object/array balanced or
// single scalar). Returns the cursor pointing just after the value.
int32_t skip_value(const bj::StructuralIndex& idx, int32_t cursor) noexcept {
    assert(cursor >= 0);
    if (cursor >= idx.token_count) return cursor;
    bj::TokenType t = idx.tokens[cursor].type;
    if (t != bj::TokenType::BeginObject && t != bj::TokenType::BeginArray) {
        return cursor + 1;
    }
    int32_t depth = 0;
    int32_t i = cursor;
    while (i < idx.token_count) {
        bj::TokenType tt = idx.tokens[i].type;
        if (tt == bj::TokenType::BeginObject || tt == bj::TokenType::BeginArray)
            ++depth;
        else if (tt == bj::TokenType::EndObject || tt == bj::TokenType::EndArray) {
            --depth;
            if (depth == 0) return i + 1;
        }
        ++i;
    }
    return i;
}

// Fetch a fresh OAuth2 client-credentials token. Cached on impl.
bool oauth2_refresh(Impl* impl) noexcept {
    assert(impl != nullptr);
    if (impl->cached_token[0] != '\0' &&
        now_epoch() + 60 < impl->token_expiry_epoch) {
        return true;
    }
    bolt::net::HttpRequest req;
    std::memset(&req, 0, sizeof(req));
    std::strncpy(req.method, "POST", sizeof(req.method) - 1u);
    std::strncpy(req.url, impl->cfg.oauth_token_url, sizeof(req.url) - 1u);
    bolt::net::http_request_add_header(&req, "Content-Type",
                                       "application/x-www-form-urlencoded");
    char body[1024];
    int blen = std::snprintf(body, sizeof(body),
                             "grant_type=client_credentials&client_id=%s"
                             "&client_secret=%s",
                             impl->cfg.oauth_client_id.value,
                             impl->cfg.oauth_client_secret.value);
    if (blen <= 0 || static_cast<size_t>(blen) >= sizeof(body)) return false;
    req.body = reinterpret_cast<const uint8_t*>(body);
    req.body_len = static_cast<uint32_t>(blen);
    bolt::net::HttpResponse resp;
    if (bolt::net::http_send(impl->arena, &req, &resp) != 0) return false;
    if (resp.status / 100 != 2) return false;

    bj::StructuralIndex idx;
    if (!bj::build_index(resp.body, static_cast<int32_t>(resp.body_len),
                         impl->arena, &idx)) return false;
    if (idx.token_count < 2 ||
        idx.tokens[0].type != bj::TokenType::BeginObject) return false;
    int32_t i = 1;
    int64_t expires_in = 3600;
    while (i < idx.token_count && idx.tokens[i].type != bj::TokenType::EndObject) {
        if (idx.tokens[i].type != bj::TokenType::Key) { i = skip_value(idx, i); continue; }
        bool is_token = tok_str_eq(idx, i, "access_token", 12);
        bool is_exp   = tok_str_eq(idx, i, "expires_in", 10);
        ++i;
        if (i >= idx.token_count) break;
        if (is_token && idx.tokens[i].type == bj::TokenType::String) {
            uint32_t L;
            const uint8_t* p = tok_bytes(idx, i, &L);
            if (L >= sizeof(impl->cached_token)) L = sizeof(impl->cached_token) - 1u;
            std::memcpy(impl->cached_token, p, L);
            impl->cached_token[L] = '\0';
        } else if (is_exp && idx.tokens[i].type == bj::TokenType::Int64) {
            int64_t v = 0;
            bj::Iterator it{ &idx, i, 0 };
            if (bj::iter_int64(&it, &v)) expires_in = v;
        }
        i = skip_value(idx, i);
    }
    impl->token_expiry_epoch = now_epoch() + expires_in;
    return impl->cached_token[0] != '\0';
}

// Attach authentication to `req`.
bool attach_auth(Impl* impl, bolt::net::HttpRequest* req) noexcept {
    assert(impl != nullptr);
    assert(req != nullptr);
    switch (impl->cfg.auth_mode) {
        case 0: return true;
        case 1: {
            char buf[600];
            std::snprintf(buf, sizeof(buf), "Bearer %s",
                          impl->cfg.bearer_token.value);
            return bolt::net::http_request_add_header(req, "Authorization", buf);
        }
        case 2: {
            if (!oauth2_refresh(impl)) return false;
            char buf[600];
            std::snprintf(buf, sizeof(buf), "Bearer %s", impl->cached_token);
            return bolt::net::http_request_add_header(req, "Authorization", buf);
        }
        case 3:
            // TODO(W7+): SigV4 for "execute-api".
            return true;
        default: return false;
    }
}

int do_request(Impl* impl, bolt::net::HttpRequest* req,
               bolt::net::HttpResponse* out) noexcept {
    assert(impl != nullptr);
    assert(req != nullptr && out != nullptr);
    if (!attach_auth(impl, req)) return kCatIoError;
    int rc = bolt::net::http_send(impl->arena, req, out);
    if (rc != 0) return kCatIoError;
    if (out->status == 404) return kCatNotFound;
    if (out->status == 409) return kCatExists;
    if (out->status / 100 != 2) return kCatIoError;
    return kCatOk;
}

bool build_listns_req_internal(const Impl* impl,
                               bolt::net::HttpRequest* out) noexcept {
    assert(impl != nullptr);
    assert(out != nullptr);
    std::memset(out, 0, sizeof(*out));
    std::strncpy(out->method, "GET", sizeof(out->method) - 1u);
    make_url(&impl->cfg, out->url, sizeof(out->url), "/namespaces");
    bolt::net::http_request_add_header(out, "Accept", "application/json");
    if (impl->cfg.auth_mode == 1) {
        char buf[600];
        std::snprintf(buf, sizeof(buf), "Bearer %s",
                      impl->cfg.bearer_token.value);
        bolt::net::http_request_add_header(out, "Authorization", buf);
    }
    return true;
}

int vt_list_namespaces(void* impl_v, CatalogName* out, uint32_t cap,
                       uint32_t* out_n) noexcept {
    assert(impl_v != nullptr);
    assert(out != nullptr && out_n != nullptr);
    Impl* impl = static_cast<Impl*>(impl_v);
    *out_n = 0;
    bolt::net::HttpRequest req;
    build_listns_req_internal(impl, &req);
    bolt::net::HttpResponse resp;
    int rc = do_request(impl, &req, &resp);
    if (rc != kCatOk) return rc;
    if (!irc_parse_namespaces(resp.body, resp.body_len, impl->arena, out, cap,
                              out_n)) {
        return kCatIoError;
    }
    return kCatOk;
}

int vt_list_tables(void* impl_v, const char* ns, CatalogName* out,
                   uint32_t cap, uint32_t* out_n) noexcept {
    assert(impl_v != nullptr);
    assert(ns != nullptr);
    assert(out != nullptr && out_n != nullptr);
    Impl* impl = static_cast<Impl*>(impl_v);
    *out_n = 0;
    bolt::net::HttpRequest req;
    std::memset(&req, 0, sizeof(req));
    std::strncpy(req.method, "GET", sizeof(req.method) - 1u);
    char tail[256];
    std::snprintf(tail, sizeof(tail), "/namespaces/%s/tables", ns);
    make_url(&impl->cfg, req.url, sizeof(req.url), tail);
    bolt::net::http_request_add_header(&req, "Accept", "application/json");
    bolt::net::HttpResponse resp;
    int rc = do_request(impl, &req, &resp);
    if (rc != kCatOk) return rc;

    // Parse {"identifiers":[{"namespace":["ns"],"name":"t1"},...]}
    bj::StructuralIndex idx;
    if (!bj::build_index(resp.body, static_cast<int32_t>(resp.body_len),
                         impl->arena, &idx)) return kCatIoError;
    if (idx.token_count < 2 ||
        idx.tokens[0].type != bj::TokenType::BeginObject) return kCatIoError;
    int32_t i = 1;
    while (i < idx.token_count && idx.tokens[i].type != bj::TokenType::EndObject) {
        if (idx.tokens[i].type != bj::TokenType::Key) {
            i = skip_value(idx, i); continue;
        }
        bool is_ids = tok_str_eq(idx, i, "identifiers", 11);
        ++i;
        if (i >= idx.token_count) break;
        if (is_ids && idx.tokens[i].type == bj::TokenType::BeginArray) {
            ++i;
            while (i < idx.token_count &&
                   idx.tokens[i].type != bj::TokenType::EndArray) {
                if (idx.tokens[i].type != bj::TokenType::BeginObject) {
                    i = skip_value(idx, i); continue;
                }
                ++i;
                // Inside identifier object — look for "name".
                while (i < idx.token_count &&
                       idx.tokens[i].type != bj::TokenType::EndObject) {
                    if (idx.tokens[i].type != bj::TokenType::Key) {
                        i = skip_value(idx, i); continue;
                    }
                    bool is_name = tok_str_eq(idx, i, "name", 4);
                    ++i;
                    if (i >= idx.token_count) break;
                    if (is_name && idx.tokens[i].type == bj::TokenType::String &&
                        *out_n < cap) {
                        uint32_t L;
                        const uint8_t* p = tok_bytes(idx, i, &L);
                        if (L >= kCatMaxName) L = kCatMaxName - 1u;
                        std::memcpy(out[*out_n].name, p, L);
                        out[*out_n].name[L] = '\0';
                        ++(*out_n);
                    }
                    i = skip_value(idx, i);
                }
                if (i < idx.token_count) ++i;   // EndObject
            }
            if (i < idx.token_count) ++i;       // EndArray
        } else {
            i = skip_value(idx, i);
        }
    }
    return kCatOk;
}

int vt_table_path(void* impl_v, const char* ns, const char* name, char* out,
                  uint32_t cap) noexcept {
    assert(impl_v != nullptr);
    assert(ns != nullptr && name != nullptr && out != nullptr);
    (void)impl_v; (void)ns; (void)name; (void)out; (void)cap;
    // TODO(W7+): GET .../tables/{name}, parse metadata.location.
    return kCatNotImplemented;
}

int vt_create_table(void* impl_v, const char* ns, const char* name,
                    TableLayout layout) noexcept {
    assert(impl_v != nullptr);
    assert(ns != nullptr && name != nullptr);
    (void)layout;
    Impl* impl = static_cast<Impl*>(impl_v);
    bolt::net::HttpRequest req;
    std::memset(&req, 0, sizeof(req));
    std::strncpy(req.method, "POST", sizeof(req.method) - 1u);
    char tail[256];
    std::snprintf(tail, sizeof(tail), "/namespaces/%s/tables", ns);
    make_url(&impl->cfg, req.url, sizeof(req.url), tail);
    bolt::net::http_request_add_header(&req, "Content-Type", "application/json");
    bolt::net::http_request_add_header(&req, "Accept", "application/json");
    char body[512];
    int blen = std::snprintf(body, sizeof(body),
                             "{\"name\":\"%s\",\"schema\":{\"type\":\"struct\","
                             "\"fields\":[]}}", name);
    if (blen <= 0) return kCatBadArg;
    req.body = reinterpret_cast<const uint8_t*>(body);
    req.body_len = static_cast<uint32_t>(blen);
    bolt::net::HttpResponse resp;
    return do_request(impl, &req, &resp);
}

int vt_drop_table(void* impl_v, const char* ns, const char* name) noexcept {
    assert(impl_v != nullptr);
    assert(ns != nullptr && name != nullptr);
    Impl* impl = static_cast<Impl*>(impl_v);
    bolt::net::HttpRequest req;
    std::memset(&req, 0, sizeof(req));
    std::strncpy(req.method, "DELETE", sizeof(req.method) - 1u);
    char tail[256];
    std::snprintf(tail, sizeof(tail), "/namespaces/%s/tables/%s", ns, name);
    make_url(&impl->cfg, req.url, sizeof(req.url), tail);
    bolt::net::HttpResponse resp;
    return do_request(impl, &req, &resp);
}

int vt_commit_metadata(void* impl_v, const char* ns, const char* name,
                       const char* rel, const uint8_t* d, uint64_t n) noexcept {
    assert(impl_v != nullptr);
    assert(ns != nullptr && name != nullptr);
    (void)impl_v; (void)ns; (void)name; (void)rel; (void)d; (void)n;
    // TODO(W7+): POST .../tables/{name} updates.
    return kCatNotImplemented;
}

const CatalogVT kVT = {
    vt_list_namespaces, vt_list_tables, vt_table_path,
    vt_create_table,    vt_drop_table,  vt_commit_metadata,
};

}  // namespace

bool open(Catalog* out, bolt::Arena* arena, const Config* cfg) noexcept {
    assert(out != nullptr);
    assert(arena != nullptr && cfg != nullptr);
    Impl* impl = static_cast<Impl*>(arena->allocate(sizeof(Impl), alignof(Impl)));
    if (impl == nullptr) return false;
    std::memset(impl, 0, sizeof(*impl));
    impl->cfg = *cfg;
    impl->arena = arena;
    out->vt = &kVT;
    out->impl = impl;
    return true;
}

bool irc_build_list_namespaces_request(const Impl* impl,
                                       bolt::net::HttpRequest* out) noexcept {
    return build_listns_req_internal(impl, out);
}

bool irc_parse_namespaces(const uint8_t* body, uint32_t len, bolt::Arena* arena,
                          CatalogName* out, uint32_t cap,
                          uint32_t* out_n) noexcept {
    assert(body != nullptr);
    assert(arena != nullptr && out != nullptr && out_n != nullptr);
    *out_n = 0;
    bj::StructuralIndex idx;
    if (!bj::build_index(body, static_cast<int32_t>(len), arena, &idx))
        return false;
    if (idx.token_count < 2 ||
        idx.tokens[0].type != bj::TokenType::BeginObject) return false;
    int32_t i = 1;
    while (i < idx.token_count && idx.tokens[i].type != bj::TokenType::EndObject) {
        if (idx.tokens[i].type != bj::TokenType::Key) {
            i = skip_value(idx, i); continue;
        }
        bool is_ns = tok_str_eq(idx, i, "namespaces", 10);
        ++i;
        if (i >= idx.token_count) break;
        if (is_ns && idx.tokens[i].type == bj::TokenType::BeginArray) {
            ++i;
            while (i < idx.token_count &&
                   idx.tokens[i].type != bj::TokenType::EndArray) {
                if (idx.tokens[i].type != bj::TokenType::BeginArray) {
                    i = skip_value(idx, i); continue;
                }
                ++i;
                // First element is the leaf namespace name.
                if (i < idx.token_count &&
                    idx.tokens[i].type == bj::TokenType::String &&
                    *out_n < cap) {
                    uint32_t L;
                    const uint8_t* p = tok_bytes(idx, i, &L);
                    if (L >= kCatMaxName) L = kCatMaxName - 1u;
                    std::memcpy(out[*out_n].name, p, L);
                    out[*out_n].name[L] = '\0';
                    ++(*out_n);
                }
                // Walk to EndArray of this inner array.
                int32_t depth = 0;
                while (i < idx.token_count) {
                    bj::TokenType tt = idx.tokens[i].type;
                    if (tt == bj::TokenType::BeginArray ||
                        tt == bj::TokenType::BeginObject) ++depth;
                    else if (tt == bj::TokenType::EndArray ||
                             tt == bj::TokenType::EndObject) {
                        if (depth == 0) { ++i; break; }
                        --depth;
                    }
                    ++i;
                }
            }
            if (i < idx.token_count) ++i;   // EndArray
        } else {
            i = skip_value(idx, i);
        }
    }
    return true;
}

}  // namespace iceberg_rest
}  // namespace lakehouse
}  // namespace bolt
