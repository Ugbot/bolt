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

void add_auth(const Impl* impl, bolt::net::HttpRequest* req) noexcept {
    if (impl->cfg.bearer_token.value[0] == '\0') return;
    char buf[600];
    std::snprintf(buf, sizeof(buf), "Bearer %s", impl->cfg.bearer_token.value);
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
                  "%s/api/metalakes/%s/catalogs/%s/schemas/default/tables",
                  impl->cfg.base_url, impl->cfg.metalake, ns);
    bolt::net::http_request_add_header(&req, "Accept", "application/json");
    bolt::net::HttpResponse resp;
    int rc = do_req(impl, &req, &resp);
    if (rc != kCatOk) return rc;
    if (!parse_named_objects(resp.body, resp.body_len, impl->arena,
                             "tables", 6, out, cap, out_n))
        return kCatIoError;
    return kCatOk;
}

int vt_table_path(void* impl_v, const char* ns, const char* name, char* out,
                  uint32_t cap) noexcept {
    assert(impl_v != nullptr);
    assert(ns != nullptr && name != nullptr && out != nullptr); (void)impl_v; (void)ns; (void)name; (void)out; (void)cap;
    return kCatNotImplemented;
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
                  "%s/api/metalakes/%s/catalogs/%s/schemas/default/tables",
                  impl->cfg.base_url, impl->cfg.metalake, ns);
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
                  "%s/api/metalakes/%s/catalogs/%s/schemas/default/tables/%s",
                  impl->cfg.base_url, impl->cfg.metalake, ns, name);
    bolt::net::HttpResponse resp;
    return do_req(impl, &req, &resp);
}

int vt_commit_metadata(void* impl_v, const char* ns, const char* name,
                       const char* rel, const uint8_t* d, uint64_t n) noexcept {
    assert(impl_v != nullptr); assert(ns != nullptr && name != nullptr);
    (void)impl_v; (void)ns; (void)name; (void)rel; (void)d; (void)n;
    return kCatNotImplemented;
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

}  // namespace gravitino
}  // namespace lakehouse
}  // namespace bolt
