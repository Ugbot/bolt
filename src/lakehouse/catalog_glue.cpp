// bolt/src/lakehouse/catalog_glue.cpp — AWS Glue Data Catalog over SigV4.

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "bolt/lakehouse/catalog/glue.h"

#include "bolt/crypto/sigv4.h"
#include "bolt/parse/bolt_json.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <ctime>

namespace bolt {
namespace lakehouse {
namespace glue {

namespace {

namespace bj = bolt::parse::json;

void make_host(const Config* c, char* out, uint32_t cap) noexcept {
    std::snprintf(out, cap, "glue.%s.amazonaws.com", c->region);
}

void format_amz_date(char out[17]) noexcept {
    std::time_t t = std::time(nullptr);
    std::tm tmv;
#ifdef _WIN32
    gmtime_s(&tmv, &t);
#else
    gmtime_r(&t, &tmv);
#endif
    std::snprintf(out, 17, "%04d%02d%02dT%02d%02d%02dZ",
                  tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday,
                  tmv.tm_hour, tmv.tm_min, tmv.tm_sec);
}

// Sign request with the given amz_date.
bool sign_with_date(Impl* impl, bolt::net::HttpRequest* req,
                    const char* amz_date) noexcept {
    assert(impl != nullptr);
    assert(req != nullptr && amz_date != nullptr);
    char host[256];
    make_host(&impl->cfg, host, sizeof(host));
    char body_hash[65];
    if (req->body_len > 0 && req->body != nullptr) {
        bolt::crypto::sha256_hex(req->body, req->body_len, body_hash);
    } else {
        bolt::crypto::sha256_empty_hex(body_hash);
    }
    bolt::crypto::SigV4Request s;
    std::memset(&s, 0, sizeof(s));
    s.method = req->method;
    s.host = host;
    s.path = "/";
    s.query_string = "";
    s.service = "glue";
    s.region  = impl->cfg.region;
    s.amz_date = amz_date;
    s.payload_sha256 = body_hash;
    s.access_key = impl->cfg.access_key.value;
    s.secret_key = impl->cfg.secret_key.value;
    s.session_token = impl->cfg.session_token.value[0] != '\0'
                          ? impl->cfg.session_token.value : nullptr;
    char auth_hdr[bolt::crypto::kSigV4MaxAuth];
    char amz_out[24];
    bolt::crypto::SigV4Result out;
    out.auth_header  = auth_hdr;
    out.auth_cap     = sizeof(auth_hdr);
    out.amz_date_out = amz_out;
    out.amz_date_cap = sizeof(amz_out);
    if (!bolt::crypto::sigv4_sign(&s, &out)) return false;
    if (!bolt::net::http_request_add_header(req, "X-Amz-Date", amz_date)) return false;
    if (!bolt::net::http_request_add_header(req, "X-Amz-Content-Sha256", body_hash))
        return false;
    if (s.session_token != nullptr) {
        if (!bolt::net::http_request_add_header(req, "X-Amz-Security-Token",
                                                s.session_token)) return false;
    }
    if (!bolt::net::http_request_add_header(req, "Authorization", auth_hdr))
        return false;
    return true;
}

int do_call(Impl* impl, const char* target, const char* body, uint32_t blen,
            bolt::net::HttpResponse* out) noexcept {
    assert(impl != nullptr && target != nullptr);
    bolt::net::HttpRequest req;
    std::memset(&req, 0, sizeof(req));
    std::strncpy(req.method, "POST", sizeof(req.method) - 1u);
    char host[256];
    make_host(&impl->cfg, host, sizeof(host));
    std::snprintf(req.url, sizeof(req.url), "https://%s/", host);
    bolt::net::http_request_add_header(&req, "Content-Type",
                                       "application/x-amz-json-1.1");
    bolt::net::http_request_add_header(&req, "X-Amz-Target", target);
    req.body = reinterpret_cast<const uint8_t*>(body);
    req.body_len = blen;
    char amz[17];
    format_amz_date(amz);
    if (!sign_with_date(impl, &req, amz)) return kCatIoError;
    int rc = bolt::net::http_send(impl->arena, &req, out);
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

// Extract Name from each element in array under outer-key `key`.
bool parse_named_array(const uint8_t* body, uint32_t len, bolt::Arena* arena,
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
                    bool is_name = tok_eq(idx, i, "Name", 4);
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
    bolt::net::HttpResponse resp;
    int rc = do_call(impl, "AWSGlue.GetDatabases", "{}", 2, &resp);
    if (rc != kCatOk) return rc;
    if (!parse_named_array(resp.body, resp.body_len, impl->arena,
                           "DatabaseList", 12, out, cap, out_n))
        return kCatIoError;
    return kCatOk;
}

int vt_list_tables(void* impl_v, const char* ns, CatalogName* out,
                   uint32_t cap, uint32_t* out_n) noexcept {
    assert(impl_v != nullptr); assert(ns != nullptr);
    Impl* impl = static_cast<Impl*>(impl_v);
    char body[512];
    int blen = std::snprintf(body, sizeof(body),
                             "{\"DatabaseName\":\"%s\"}", ns);
    if (blen <= 0) return kCatBadArg;
    bolt::net::HttpResponse resp;
    int rc = do_call(impl, "AWSGlue.GetTables", body,
                     static_cast<uint32_t>(blen), &resp);
    if (rc != kCatOk) return rc;
    if (!parse_named_array(resp.body, resp.body_len, impl->arena,
                           "TableList", 9, out, cap, out_n))
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
    char body[1024];
    int blen = std::snprintf(body, sizeof(body),
                             "{\"DatabaseName\":\"%s\",\"TableInput\":"
                             "{\"Name\":\"%s\",\"StorageDescriptor\":"
                             "{\"Columns\":[]}}}",
                             ns, name);
    if (blen <= 0) return kCatBadArg;
    bolt::net::HttpResponse resp;
    return do_call(impl, "AWSGlue.CreateTable", body,
                   static_cast<uint32_t>(blen), &resp);
}

int vt_drop_table(void* impl_v, const char* ns, const char* name) noexcept {
    assert(impl_v != nullptr); assert(ns != nullptr && name != nullptr);
    Impl* impl = static_cast<Impl*>(impl_v);
    char body[512];
    int blen = std::snprintf(body, sizeof(body),
                             "{\"DatabaseName\":\"%s\",\"Name\":\"%s\"}",
                             ns, name);
    if (blen <= 0) return kCatBadArg;
    bolt::net::HttpResponse resp;
    return do_call(impl, "AWSGlue.DeleteTable", body,
                   static_cast<uint32_t>(blen), &resp);
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

bool glue_sign_request(Impl* impl, bolt::net::HttpRequest* req) noexcept {
    char amz[17];
    format_amz_date(amz);
    return sign_with_date(impl, req, amz);
}

bool glue_sign_request_with_date(Impl* impl, bolt::net::HttpRequest* req,
                                 const char* amz_date) noexcept {
    return sign_with_date(impl, req, amz_date);
}

}  // namespace glue
}  // namespace lakehouse
}  // namespace bolt
