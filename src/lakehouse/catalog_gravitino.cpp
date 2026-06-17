// bolt/src/lakehouse/catalog_gravitino.cpp — Gravitino REST catalog.

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "bolt/lakehouse/catalog/gravitino.h"

#include "bolt/net/bolt_http_client.h"
#include "bolt/parse/bolt_json.h"

#include <cassert>
#include <cstdio>
#include <cstring>

namespace bolt {
namespace lakehouse {
namespace gravitino {

namespace {

namespace bj = bolt::parse::json;

struct Impl {
    Config       cfg;
    bolt::Arena* arena;
};

const char* eff_schema(const Impl* impl) noexcept {
    assert(impl != nullptr);
    return impl->cfg.schema[0] != '\0' ? impl->cfg.schema : "default";
}

void add_auth(const Impl* impl, bolt::net::HttpRequest* req) noexcept {
    if (impl->cfg.bearer_token.value[0] == '\0') return;
    char buf[600];
    std::snprintf(buf, sizeof(buf), "Bearer %s", impl->cfg.bearer_token.value);
    bolt::net::http_request_add_header(req, "Authorization", buf);
}

// Add a Bearer header from a raw token (used by the free-standing builders).
void add_bearer_raw(bolt::net::HttpRequest* req, const char* bearer) noexcept {
    assert(req != nullptr);
    if (bearer == nullptr || bearer[0] == '\0') return;
    char buf[600];
    int n = std::snprintf(buf, sizeof(buf), "Bearer %s", bearer);
    if (n > 0 && static_cast<uint32_t>(n) < sizeof(buf))
        bolt::net::http_request_add_header(req, "Authorization", buf);
}

int do_req(Impl* impl, bolt::net::HttpRequest* req,
           bolt::net::HttpResponse* out) noexcept {
    add_auth(impl, req);
    int rc = bolt::net::http_send(impl->arena, req, out);
    if (rc != 0) return kCatIoError;
    if (out->status == 404) return kCatNotFound;
    if (out->status == 409) return kCatExists;
    if (out->status / 100 != 2) return kCatIoError;
    return kCatOk;
}

int32_t skip_value(const bj::StructuralIndex& idx, int32_t c) noexcept {
    if (c >= idx.token_count) return c;
    bj::TokenType t = idx.tokens[c].type;
    if (t != bj::TokenType::BeginObject && t != bj::TokenType::BeginArray)
        return c + 1;
    int32_t d = 0, i = c;
    while (i < idx.token_count) {
        bj::TokenType tt = idx.tokens[i].type;
        if (tt == bj::TokenType::BeginObject || tt == bj::TokenType::BeginArray) ++d;
        else if (tt == bj::TokenType::EndObject || tt == bj::TokenType::EndArray) {
            --d; if (d == 0) return i + 1;
        }
        ++i;
    }
    return i;
}

bool tok_eq(const bj::StructuralIndex& idx, int32_t i, const char* s, uint32_t n) noexcept {
    if (i < 0 || i >= idx.token_count) return false;
    const bj::Token& t = idx.tokens[i];
    if ((t.type != bj::TokenType::String && t.type != bj::TokenType::Key) ||
        static_cast<uint32_t>(t.length) != n) return false;
    return std::memcmp(idx.src + t.start, s, n) == 0;
}

bool parse_named_objects(const uint8_t* body, uint32_t len, bolt::Arena* arena,
                         const char* outer_key, uint32_t outer_key_len,
                         CatalogName* out, uint32_t cap, uint32_t* out_n) noexcept {
    *out_n = 0;
    bj::StructuralIndex idx;
    if (!bj::build_index(body, static_cast<int32_t>(len), arena, &idx))
        return false;
    if (idx.token_count < 2 ||
        idx.tokens[0].type != bj::TokenType::BeginObject) return false;
    int32_t i = 1;
    while (i < idx.token_count &&
           idx.tokens[i].type != bj::TokenType::EndObject) {
        if (idx.tokens[i].type != bj::TokenType::Key) {
            i = skip_value(idx, i); continue;
        }
        bool match = tok_eq(idx, i, outer_key, outer_key_len);
        ++i;
        if (i >= idx.token_count) break;
        if (match && idx.tokens[i].type == bj::TokenType::BeginArray) {
            ++i;
            while (i < idx.token_count &&
                   idx.tokens[i].type != bj::TokenType::EndArray) {
                if (idx.tokens[i].type != bj::TokenType::BeginObject) {
                    i = skip_value(idx, i); continue;
                }
                ++i;
                while (i < idx.token_count &&
                       idx.tokens[i].type != bj::TokenType::EndObject) {
                    if (idx.tokens[i].type != bj::TokenType::Key) {
                        i = skip_value(idx, i); continue;
                    }
                    bool is_name = tok_eq(idx, i, "name", 4);
                    ++i;
                    if (i >= idx.token_count) break;
                    if (is_name && idx.tokens[i].type == bj::TokenType::String &&
                        *out_n < cap) {
                        uint32_t L = static_cast<uint32_t>(idx.tokens[i].length);
                        if (L >= kCatMaxName) L = kCatMaxName - 1u;
                        std::memcpy(out[*out_n].name,
                                    idx.src + idx.tokens[i].start, L);
                        out[*out_n].name[L] = '\0';
                        ++(*out_n);
                    }
                    i = skip_value(idx, i);
                }
                if (i < idx.token_count) ++i;
            }
            if (i < idx.token_count) ++i;
        } else {
            i = skip_value(idx, i);
        }
    }
    return true;
}

int vt_list_namespaces(void* impl_v, CatalogName* out, uint32_t cap,
                       uint32_t* out_n) noexcept {
    assert(impl_v != nullptr); assert(out != nullptr && out_n != nullptr);
    Impl* impl = static_cast<Impl*>(impl_v);
    bolt::net::HttpRequest req;
    std::memset(&req, 0, sizeof(req));
    std::strncpy(req.method, "GET", sizeof(req.method) - 1u);
    std::snprintf(req.url, sizeof(req.url),
                  "%s/api/metalakes/%s/catalogs",
                  impl->cfg.base_url, impl->cfg.metalake);
    bolt::net::http_request_add_header(&req, "Accept", "application/json");
    bolt::net::HttpResponse resp;
    int rc = do_req(impl, &req, &resp);
    if (rc != kCatOk) return rc;
    if (!parse_named_objects(resp.body, resp.body_len, impl->arena,
                             "catalogs", 8, out, cap, out_n))
        return kCatIoError;
    return kCatOk;
}

int vt_list_tables(void* impl_v, const char* ns, CatalogName* out,
                   uint32_t cap, uint32_t* out_n) noexcept {
    assert(impl_v != nullptr); assert(ns != nullptr);
    Impl* impl = static_cast<Impl*>(impl_v);
    bolt::net::HttpRequest req;
    std::memset(&req, 0, sizeof(req));
    std::strncpy(req.method, "GET", sizeof(req.method) - 1u);
    std::snprintf(req.url, sizeof(req.url),
                  "%s/api/metalakes/%s/catalogs/%s/schemas/%s/tables",
                  impl->cfg.base_url, impl->cfg.metalake, ns, eff_schema(impl));
    bolt::net::http_request_add_header(&req, "Accept", "application/json");
    bolt::net::HttpResponse resp;
    int rc = do_req(impl, &req, &resp);
    if (rc != kCatOk) return rc;
    if (!parse_named_objects(resp.body, resp.body_len, impl->arena,
                             "tables", 6, out, cap, out_n))
        return kCatIoError;
    return kCatOk;
}

// Find the first string value for key `key` anywhere in the body. Copies into
// out[0..cap). Leaves out empty (kCatOk) if the key is absent.
int find_string_field(const uint8_t* body, uint32_t len, bolt::Arena* arena,
                      const char* key, uint32_t key_len, char* out,
                      uint32_t cap) noexcept {
    assert(body != nullptr && arena != nullptr && key != nullptr && out != nullptr);
    assert(cap > 0u);
    out[0] = '\0';
    bj::StructuralIndex idx;
    if (!bj::build_index(body, static_cast<int32_t>(len), arena, &idx))
        return kCatIoError;
    for (int32_t i = 1; i + 1 < idx.token_count; ++i) {
        if (tok_eq(idx, i, key, key_len) &&
            idx.tokens[i + 1].type == bj::TokenType::String) {
            uint32_t L = static_cast<uint32_t>(idx.tokens[i + 1].length);
            if (L >= cap) L = cap - 1u;
            std::memcpy(out, idx.src + idx.tokens[i + 1].start, L);
            out[L] = '\0';
            return kCatOk;
        }
    }
    return kCatOk;
}

int vt_table_path(void* impl_v, const char* ns, const char* name, char* out,
                  uint32_t cap) noexcept {
    assert(impl_v != nullptr);
    assert(ns != nullptr && name != nullptr && out != nullptr);
    Impl* impl = static_cast<Impl*>(impl_v);
    bolt::net::HttpRequest req;
    std::memset(&req, 0, sizeof(req));
    std::strncpy(req.method, "GET", sizeof(req.method) - 1u);
    std::snprintf(req.url, sizeof(req.url),
                  "%s/api/metalakes/%s/catalogs/%s/schemas/%s/tables/%s",
                  impl->cfg.base_url, impl->cfg.metalake, ns, eff_schema(impl),
                  name);
    bolt::net::http_request_add_header(&req, "Accept", "application/json");
    bolt::net::HttpResponse resp;
    int rc = do_req(impl, &req, &resp);
    if (rc != kCatOk) return rc;
    // Gravitino carries the on-store path in properties.location.
    return find_string_field(resp.body, resp.body_len, impl->arena,
                             "location", 8, out, cap);
}

int vt_create_table(void* impl_v, const char* ns, const char* name,
                    TableLayout layout) noexcept {
    assert(impl_v != nullptr); assert(ns != nullptr && name != nullptr);
    (void)layout;
    Impl* impl = static_cast<Impl*>(impl_v);
    bolt::net::HttpRequest req;
    std::memset(&req, 0, sizeof(req));
    std::strncpy(req.method, "POST", sizeof(req.method) - 1u);
    std::snprintf(req.url, sizeof(req.url),
                  "%s/api/metalakes/%s/catalogs/%s/schemas/%s/tables",
                  impl->cfg.base_url, impl->cfg.metalake, ns, eff_schema(impl));
    bolt::net::http_request_add_header(&req, "Content-Type", "application/json");
    bolt::net::http_request_add_header(&req, "Accept", "application/json");
    char body[512];
    int blen = std::snprintf(body, sizeof(body),
                             "{\"name\":\"%s\",\"columns\":[]}", name);
    if (blen <= 0) return kCatBadArg;
    req.body = reinterpret_cast<const uint8_t*>(body);
    req.body_len = static_cast<uint32_t>(blen);
    bolt::net::HttpResponse resp;
    return do_req(impl, &req, &resp);
}

int vt_drop_table(void* impl_v, const char* ns, const char* name) noexcept {
    assert(impl_v != nullptr); assert(ns != nullptr && name != nullptr);
    Impl* impl = static_cast<Impl*>(impl_v);
    bolt::net::HttpRequest req;
    std::memset(&req, 0, sizeof(req));
    std::strncpy(req.method, "DELETE", sizeof(req.method) - 1u);
    std::snprintf(req.url, sizeof(req.url),
                  "%s/api/metalakes/%s/catalogs/%s/schemas/%s/tables/%s",
                  impl->cfg.base_url, impl->cfg.metalake, ns, eff_schema(impl),
                  name);
    bolt::net::HttpResponse resp;
    return do_req(impl, &req, &resp);
}

int vt_commit_metadata(void* impl_v, const char* ns, const char* name,
                       const char* rel, const uint8_t* d, uint64_t n) noexcept {
    assert(impl_v != nullptr); assert(ns != nullptr && name != nullptr);
    (void)rel;
    Impl* impl = static_cast<Impl*>(impl_v);
    if (d == nullptr || n == 0u || n >= bolt::net::kHttpMaxBody) return kCatBadArg;
    bolt::net::HttpRequest req;
    std::memset(&req, 0, sizeof(req));
    std::strncpy(req.method, "PUT", sizeof(req.method) - 1u);
    std::snprintf(req.url, sizeof(req.url),
                  "%s/api/metalakes/%s/catalogs/%s/schemas/%s/tables/%s",
                  impl->cfg.base_url, impl->cfg.metalake, ns, eff_schema(impl),
                  name);
    bolt::net::http_request_add_header(&req, "Content-Type", "application/json");
    bolt::net::http_request_add_header(&req, "Accept", "application/json");
    req.body = d;
    req.body_len = static_cast<uint32_t>(n);
    bolt::net::HttpResponse resp;
    return do_req(impl, &req, &resp);
}

const CatalogVT kVT = {
    vt_list_namespaces, vt_list_tables, vt_table_path,
    vt_create_table,    vt_drop_table,  vt_commit_metadata,
};

}  // namespace

bool open(Catalog* out, bolt::Arena* arena, const Config* cfg) noexcept {
    assert(out != nullptr); assert(arena != nullptr && cfg != nullptr);
    Impl* impl = static_cast<Impl*>(arena->allocate(sizeof(Impl), alignof(Impl)));
    if (impl == nullptr) return false;
    std::memset(impl, 0, sizeof(*impl));
    impl->cfg = *cfg;
    impl->arena = arena;
    out->vt = &kVT;
    out->impl = impl;
    return true;
}

// ===========================================================================
// Request builders (white-box; testable without a live Gravitino).
// ===========================================================================

bool gravitino_build_create_metalake(const char* base_url, const char* bearer,
                                     const char* name, const char* comment,
                                     bolt::net::HttpRequest* out, char* body_buf,
                                     uint32_t body_cap) noexcept {
    assert(base_url != nullptr && name != nullptr && out != nullptr);
    assert(body_buf != nullptr);
    std::memset(out, 0, sizeof(*out));
    std::strncpy(out->method, "POST", sizeof(out->method) - 1u);
    int u = std::snprintf(out->url, sizeof(out->url), "%s/api/metalakes",
                          base_url);
    if (u <= 0 || static_cast<uint32_t>(u) >= sizeof(out->url)) return false;
    bolt::net::http_request_add_header(out, "Content-Type", "application/json");
    bolt::net::http_request_add_header(out, "Accept", "application/json");
    add_bearer_raw(out, bearer);
    int b = std::snprintf(body_buf, body_cap,
                          "{\"name\":\"%s\",\"comment\":\"%s\"}", name,
                          comment != nullptr ? comment : "");
    if (b <= 0 || static_cast<uint32_t>(b) >= body_cap) return false;
    out->body = reinterpret_cast<const uint8_t*>(body_buf);
    out->body_len = static_cast<uint32_t>(b);
    return true;
}

bool gravitino_build_create_catalog(const char* base_url, const char* bearer,
                                    const char* metalake, const char* name,
                                    const char* type, const char* provider,
                                    bolt::net::HttpRequest* out, char* body_buf,
                                    uint32_t body_cap) noexcept {
    assert(base_url != nullptr && metalake != nullptr && name != nullptr);
    assert(type != nullptr && provider != nullptr && out != nullptr);
    assert(body_buf != nullptr);
    std::memset(out, 0, sizeof(*out));
    std::strncpy(out->method, "POST", sizeof(out->method) - 1u);
    int u = std::snprintf(out->url, sizeof(out->url),
                          "%s/api/metalakes/%s/catalogs", base_url, metalake);
    if (u <= 0 || static_cast<uint32_t>(u) >= sizeof(out->url)) return false;
    bolt::net::http_request_add_header(out, "Content-Type", "application/json");
    bolt::net::http_request_add_header(out, "Accept", "application/json");
    add_bearer_raw(out, bearer);
    int b = std::snprintf(body_buf, body_cap,
                          "{\"name\":\"%s\",\"type\":\"%s\","
                          "\"provider\":\"%s\"}", name, type, provider);
    if (b <= 0 || static_cast<uint32_t>(b) >= body_cap) return false;
    out->body = reinterpret_cast<const uint8_t*>(body_buf);
    out->body_len = static_cast<uint32_t>(b);
    return true;
}

bool gravitino_build_create_schema(const char* base_url, const char* bearer,
                                   const char* metalake, const char* catalog,
                                   const char* schema, const char* comment,
                                   bolt::net::HttpRequest* out, char* body_buf,
                                   uint32_t body_cap) noexcept {
    assert(base_url != nullptr && metalake != nullptr && catalog != nullptr);
    assert(schema != nullptr && out != nullptr && body_buf != nullptr);
    std::memset(out, 0, sizeof(*out));
    std::strncpy(out->method, "POST", sizeof(out->method) - 1u);
    int u = std::snprintf(out->url, sizeof(out->url),
                          "%s/api/metalakes/%s/catalogs/%s/schemas", base_url,
                          metalake, catalog);
    if (u <= 0 || static_cast<uint32_t>(u) >= sizeof(out->url)) return false;
    bolt::net::http_request_add_header(out, "Content-Type", "application/json");
    bolt::net::http_request_add_header(out, "Accept", "application/json");
    add_bearer_raw(out, bearer);
    int b = std::snprintf(body_buf, body_cap,
                          "{\"name\":\"%s\",\"comment\":\"%s\"}", schema,
                          comment != nullptr ? comment : "");
    if (b <= 0 || static_cast<uint32_t>(b) >= body_cap) return false;
    out->body = reinterpret_cast<const uint8_t*>(body_buf);
    out->body_len = static_cast<uint32_t>(b);
    return true;
}

bool gravitino_build_create_table(const char* base_url, const char* bearer,
                                  const char* metalake, const char* catalog,
                                  const char* schema, const char* table,
                                  bolt::net::HttpRequest* out, char* body_buf,
                                  uint32_t body_cap) noexcept {
    assert(base_url != nullptr && metalake != nullptr && catalog != nullptr);
    assert(schema != nullptr && table != nullptr && out != nullptr);
    assert(body_buf != nullptr);
    std::memset(out, 0, sizeof(*out));
    std::strncpy(out->method, "POST", sizeof(out->method) - 1u);
    int u = std::snprintf(out->url, sizeof(out->url),
                          "%s/api/metalakes/%s/catalogs/%s/schemas/%s/tables",
                          base_url, metalake, catalog, schema);
    if (u <= 0 || static_cast<uint32_t>(u) >= sizeof(out->url)) return false;
    bolt::net::http_request_add_header(out, "Content-Type", "application/json");
    bolt::net::http_request_add_header(out, "Accept", "application/json");
    add_bearer_raw(out, bearer);
    int b = std::snprintf(body_buf, body_cap,
                          "{\"name\":\"%s\",\"columns\":[]}", table);
    if (b <= 0 || static_cast<uint32_t>(b) >= body_cap) return false;
    out->body = reinterpret_cast<const uint8_t*>(body_buf);
    out->body_len = static_cast<uint32_t>(b);
    return true;
}

// ===========================================================================
// Public native CRUD.
// ===========================================================================

namespace {

Impl* gimpl(Catalog* cat) noexcept {
    assert(cat != nullptr && cat->vt == &kVT && cat->impl != nullptr);
    return static_cast<Impl*>(cat->impl);
}

int send_built(Impl* impl, bolt::net::HttpRequest* req) noexcept {
    assert(impl != nullptr && req != nullptr);
    // Builders already attached the static bearer; nothing else to add.
    bolt::net::HttpResponse resp;
    if (bolt::net::http_send(impl->arena, req, &resp) != 0) return kCatIoError;
    if (resp.status == 404) return kCatNotFound;
    if (resp.status == 409) return kCatExists;
    if (resp.status / 100 != 2) return kCatIoError;
    return kCatOk;
}

int delete_path(Impl* impl, const char* url) noexcept {
    assert(impl != nullptr && url != nullptr);
    bolt::net::HttpRequest req;
    std::memset(&req, 0, sizeof(req));
    std::strncpy(req.method, "DELETE", sizeof(req.method) - 1u);
    std::strncpy(req.url, url, sizeof(req.url) - 1u);
    bolt::net::http_request_add_header(&req, "Accept", "application/json");
    bolt::net::HttpResponse resp;
    return do_req(impl, &req, &resp);
}

}  // namespace

int gravitino_create_metalake(Catalog* cat, const char* name,
                              const char* comment) noexcept {
    assert(cat != nullptr && name != nullptr);
    Impl* impl = gimpl(cat);
    bolt::net::HttpRequest req;
    char body[512];
    if (!gravitino_build_create_metalake(impl->cfg.base_url,
                                         impl->cfg.bearer_token.value, name,
                                         comment, &req, body, sizeof(body)))
        return kCatBadArg;
    return send_built(impl, &req);
}

int gravitino_drop_metalake(Catalog* cat, const char* name) noexcept {
    assert(cat != nullptr && name != nullptr);
    Impl* impl = gimpl(cat);
    char url[640];
    std::snprintf(url, sizeof(url), "%s/api/metalakes/%s", impl->cfg.base_url,
                  name);
    return delete_path(impl, url);
}

int gravitino_create_catalog(Catalog* cat, const char* name, const char* type,
                             const char* provider) noexcept {
    assert(cat != nullptr && name != nullptr && type != nullptr);
    assert(provider != nullptr);
    Impl* impl = gimpl(cat);
    bolt::net::HttpRequest req;
    char body[512];
    if (!gravitino_build_create_catalog(impl->cfg.base_url,
                                        impl->cfg.bearer_token.value,
                                        impl->cfg.metalake, name, type, provider,
                                        &req, body, sizeof(body)))
        return kCatBadArg;
    return send_built(impl, &req);
}

int gravitino_drop_catalog(Catalog* cat, const char* name) noexcept {
    assert(cat != nullptr && name != nullptr);
    Impl* impl = gimpl(cat);
    char url[640];
    std::snprintf(url, sizeof(url), "%s/api/metalakes/%s/catalogs/%s",
                  impl->cfg.base_url, impl->cfg.metalake, name);
    return delete_path(impl, url);
}

int gravitino_create_schema(Catalog* cat, const char* catalog,
                            const char* schema, const char* comment) noexcept {
    assert(cat != nullptr && catalog != nullptr && schema != nullptr);
    Impl* impl = gimpl(cat);
    bolt::net::HttpRequest req;
    char body[512];
    if (!gravitino_build_create_schema(impl->cfg.base_url,
                                       impl->cfg.bearer_token.value,
                                       impl->cfg.metalake, catalog, schema,
                                       comment, &req, body, sizeof(body)))
        return kCatBadArg;
    return send_built(impl, &req);
}

int gravitino_drop_schema(Catalog* cat, const char* catalog,
                          const char* schema) noexcept {
    assert(cat != nullptr && catalog != nullptr && schema != nullptr);
    Impl* impl = gimpl(cat);
    char url[768];
    std::snprintf(url, sizeof(url),
                  "%s/api/metalakes/%s/catalogs/%s/schemas/%s",
                  impl->cfg.base_url, impl->cfg.metalake, catalog, schema);
    return delete_path(impl, url);
}

int gravitino_load_table(Catalog* cat, const char* catalog, const char* schema,
                         const char* table, char* out_name,
                         uint32_t cap) noexcept {
    assert(cat != nullptr && catalog != nullptr && schema != nullptr);
    assert(table != nullptr && out_name != nullptr && cap > 0u);
    Impl* impl = gimpl(cat);
    bolt::net::HttpRequest req;
    std::memset(&req, 0, sizeof(req));
    std::strncpy(req.method, "GET", sizeof(req.method) - 1u);
    std::snprintf(req.url, sizeof(req.url),
                  "%s/api/metalakes/%s/catalogs/%s/schemas/%s/tables/%s",
                  impl->cfg.base_url, impl->cfg.metalake, catalog, schema, table);
    bolt::net::http_request_add_header(&req, "Accept", "application/json");
    bolt::net::HttpResponse resp;
    int rc = do_req(impl, &req, &resp);
    if (rc != kCatOk) return rc;
    return find_string_field(resp.body, resp.body_len, impl->arena, "name", 4,
                             out_name, cap);
}

}  // namespace gravitino
}  // namespace lakehouse
}  // namespace bolt
