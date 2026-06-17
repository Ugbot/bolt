// bolt/src/lakehouse/catalog_iceberg_rest.cpp — Iceberg REST Catalog (full spec).
//
// Apache spec: https://iceberg.apache.org/concepts/catalog/#rest-catalog
// G2CLUS-176 expanded W6 (namespace/table CRUD + bearer) to the full F.1 surface:
//   OAuth2 client-credentials + refresh, Basic, SigV4 (API Gateway); /config
//   discovery; multi-table transactions; views; snapshots + cherry-pick/rollback;
//   register table; rename; branches + tags (set/remove-snapshot-ref); statistics;
//   pagination (pageToken/pageSize); vended credentials; CDC feed.

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "bolt/lakehouse/catalog/iceberg_rest.h"

#include "bolt/crypto/sigv4.h"
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

// ---- JSON token-walk helpers ----

bool tok_str_eq(const bj::StructuralIndex& idx, int32_t i,
                const char* s, uint32_t n) noexcept {
    if (i < 0 || i >= idx.token_count) return false;
    const bj::Token& t = idx.tokens[i];
    if (t.type != bj::TokenType::String && t.type != bj::TokenType::Key)
        return false;
    if (static_cast<uint32_t>(t.length) != n) return false;
    return std::memcmp(idx.src + t.start, s, n) == 0;
}

const uint8_t* tok_bytes(const bj::StructuralIndex& idx, int32_t i,
                         uint32_t* out_len) noexcept {
    assert(i >= 0 && i < idx.token_count);
    const bj::Token& t = idx.tokens[i];
    *out_len = static_cast<uint32_t>(t.length);
    return idx.src + t.start;
}

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

void copy_str_tok(const bj::StructuralIndex& idx, int32_t i, char* out,
                  uint32_t cap) noexcept {
    assert(out != nullptr && cap > 0u);
    uint32_t L;
    const uint8_t* p = tok_bytes(idx, i, &L);
    if (L >= cap) L = cap - 1u;
    std::memcpy(out, p, L);
    out[L] = '\0';
}

// Append "?pageToken=..&pageSize=.." to a URL in place. Bounded.
bool append_paging(char* url, uint32_t cap, const char* page_token,
                   uint32_t page_size) noexcept {
    assert(url != nullptr);
    const bool has_token = page_token != nullptr && page_token[0] != '\0';
    const bool has_size  = page_size > 0u;
    if (!has_token && !has_size) return true;
    char qs[640];
    int n;
    if (has_token && has_size) {
        n = std::snprintf(qs, sizeof(qs), "?pageToken=%s&pageSize=%u",
                          page_token, page_size);
    } else if (has_token) {
        n = std::snprintf(qs, sizeof(qs), "?pageToken=%s", page_token);
    } else {
        n = std::snprintf(qs, sizeof(qs), "?pageSize=%u", page_size);
    }
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(qs)) return false;
    const size_t cur = std::strlen(url);
    if (cur + static_cast<size_t>(n) + 1u >= cap) return false;
    std::memcpy(url + cur, qs, static_cast<size_t>(n) + 1u);
    return true;
}

// Standard base64 of `src` into `out` (NUL-terminated). Returns length or 0.
uint32_t base64_std(const uint8_t* src, uint32_t n, char* out,
                    uint32_t cap) noexcept {
    assert(src != nullptr || n == 0u);
    assert(out != nullptr);
    static const char tbl[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const uint32_t need = ((n + 2u) / 3u) * 4u;
    if (need + 1u > cap) return 0u;
    uint32_t o = 0u;
    uint32_t i = 0u;
    while (i + 3u <= n) {
        uint32_t v = (uint32_t(src[i]) << 16) | (uint32_t(src[i + 1]) << 8) |
                     uint32_t(src[i + 2]);
        out[o++] = tbl[(v >> 18) & 63u];
        out[o++] = tbl[(v >> 12) & 63u];
        out[o++] = tbl[(v >> 6) & 63u];
        out[o++] = tbl[v & 63u];
        i += 3u;
    }
    const uint32_t rem = n - i;
    if (rem == 1u) {
        uint32_t v = uint32_t(src[i]) << 16;
        out[o++] = tbl[(v >> 18) & 63u];
        out[o++] = tbl[(v >> 12) & 63u];
        out[o++] = '=';
        out[o++] = '=';
    } else if (rem == 2u) {
        uint32_t v = (uint32_t(src[i]) << 16) | (uint32_t(src[i + 1]) << 8);
        out[o++] = tbl[(v >> 18) & 63u];
        out[o++] = tbl[(v >> 12) & 63u];
        out[o++] = tbl[(v >> 6) & 63u];
        out[o++] = '=';
    }
    out[o] = '\0';
    return o;
}

// substring match for "cdc" within a bounded string token.
bool token_has_cdc(const uint8_t* p, uint32_t n) noexcept {
    for (uint32_t i = 0; i + 3u <= n; ++i) {
        if (p[i] == 'c' && p[i + 1] == 'd' && p[i + 2] == 'c') return true;
    }
    return false;
}

// ---- OAuth2 ----

bool build_oauth2_body(const Config* c, char* body, uint32_t cap,
                       uint32_t* out_len) noexcept {
    assert(c != nullptr && body != nullptr && out_len != nullptr);
    int blen;
    if (c->oauth_scope[0] != '\0') {
        blen = std::snprintf(body, cap,
                             "grant_type=client_credentials&client_id=%s"
                             "&client_secret=%s&scope=%s",
                             c->oauth_client_id.value,
                             c->oauth_client_secret.value, c->oauth_scope);
    } else {
        blen = std::snprintf(body, cap,
                             "grant_type=client_credentials&client_id=%s"
                             "&client_secret=%s",
                             c->oauth_client_id.value,
                             c->oauth_client_secret.value);
    }
    if (blen <= 0 || static_cast<uint32_t>(blen) >= cap) return false;
    *out_len = static_cast<uint32_t>(blen);
    return true;
}

bool parse_oauth2_token(Impl* impl, const uint8_t* body, uint32_t len) noexcept {
    assert(impl != nullptr && body != nullptr);
    bj::StructuralIndex idx;
    if (!bj::build_index(body, static_cast<int32_t>(len), impl->arena, &idx))
        return false;
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
            copy_str_tok(idx, i, impl->cached_token, sizeof(impl->cached_token));
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
    uint32_t blen = 0;
    if (!build_oauth2_body(&impl->cfg, body, sizeof(body), &blen)) return false;
    req.body = reinterpret_cast<const uint8_t*>(body);
    req.body_len = blen;
    bolt::net::HttpResponse resp;
    if (bolt::net::http_send(impl->arena, &req, &resp) != 0) return false;
    if (resp.status / 100 != 2) return false;
    return parse_oauth2_token(impl, resp.body, resp.body_len);
}

// ---- SigV4 (REST-over-API-Gateway) ----

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

bool sigv4_sign_req(Impl* impl, bolt::net::HttpRequest* req) noexcept {
    assert(impl != nullptr && req != nullptr);
    bool https = false;
    char host[bolt::crypto::kSigV4MaxHost];
    uint16_t port = 0;
    char path[bolt::crypto::kSigV4MaxPath];
    if (!bolt::net::http_url_parse(req->url, &https, host, sizeof(host), &port,
                                   path, sizeof(path))) return false;
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
    s.path = path;
    s.query_string = "";
    s.service = impl->cfg.aws_service[0] != '\0' ? impl->cfg.aws_service
                                                 : "execute-api";
    s.region = impl->cfg.aws_region;
    char amz[17];
    format_amz_date(amz);
    s.amz_date = amz;
    s.payload_sha256 = body_hash;
    s.access_key = impl->cfg.aws_access_key.value;
    s.secret_key = impl->cfg.aws_secret_key.value;
    s.session_token = impl->cfg.aws_session_token.value[0] != '\0'
                          ? impl->cfg.aws_session_token.value : nullptr;
    char auth_hdr[bolt::crypto::kSigV4MaxAuth];
    char amz_out[24];
    bolt::crypto::SigV4Result out;
    out.auth_header = auth_hdr;
    out.auth_cap = sizeof(auth_hdr);
    out.amz_date_out = amz_out;
    out.amz_date_cap = sizeof(amz_out);
    if (!bolt::crypto::sigv4_sign(&s, &out)) return false;
    if (!bolt::net::http_request_add_header(req, "X-Amz-Date", amz)) return false;
    if (!bolt::net::http_request_add_header(req, "X-Amz-Content-Sha256",
                                            body_hash)) return false;
    if (s.session_token != nullptr &&
        !bolt::net::http_request_add_header(req, "X-Amz-Security-Token",
                                            s.session_token)) return false;
    return bolt::net::http_request_add_header(req, "Authorization", auth_hdr);
}

bool attach_basic(Impl* impl, bolt::net::HttpRequest* req) noexcept {
    assert(impl != nullptr && req != nullptr);
    char raw[kSecretMax * 2u + 1u];
    int rn = std::snprintf(raw, sizeof(raw), "%s:%s", impl->cfg.basic_user.value,
                           impl->cfg.basic_password.value);
    if (rn <= 0 || static_cast<uint32_t>(rn) >= sizeof(raw)) return false;
    char b64[((kSecretMax * 2u + 1u + 2u) / 3u) * 4u + 1u];
    if (base64_std(reinterpret_cast<const uint8_t*>(raw),
                   static_cast<uint32_t>(rn), b64, sizeof(b64)) == 0u)
        return false;
    char buf[sizeof(b64) + 8u];
    std::snprintf(buf, sizeof(buf), "Basic %s", b64);
    return bolt::net::http_request_add_header(req, "Authorization", buf);
}

bool attach_auth(Impl* impl, bolt::net::HttpRequest* req) noexcept {
    assert(impl != nullptr && req != nullptr);
    switch (impl->cfg.auth_mode) {
        case kAuthNone: return true;
        case kAuthBearer: {
            char buf[600];
            std::snprintf(buf, sizeof(buf), "Bearer %s",
                          impl->cfg.bearer_token.value);
            return bolt::net::http_request_add_header(req, "Authorization", buf);
        }
        case kAuthOAuth2: {
            if (!oauth2_refresh(impl)) return false;
            char buf[600];
            std::snprintf(buf, sizeof(buf), "Bearer %s", impl->cached_token);
            return bolt::net::http_request_add_header(req, "Authorization", buf);
        }
        case kAuthBasic:
            return attach_basic(impl, req);
        case kAuthSigV4:
            return sigv4_sign_req(impl, req);
        default: return false;
    }
}

void maybe_delegation(const Impl* impl, bolt::net::HttpRequest* req) noexcept {
    assert(impl != nullptr && req != nullptr);
    if (impl->cfg.vended_credentials != 0u) {
        bolt::net::http_request_add_header(req, "X-Iceberg-Access-Delegation",
                                           "vended-credentials");
    }
}

int classify_status(int status) noexcept {
    if (status == 404) return kCatNotFound;
    if (status == 409) return kCatExists;
    if (status / 100 != 2) return kCatIoError;
    return kCatOk;
}

int do_request(Impl* impl, bolt::net::HttpRequest* req,
               bolt::net::HttpResponse* out) noexcept {
    assert(impl != nullptr);
    assert(req != nullptr && out != nullptr);
    maybe_delegation(impl, req);
    if (!attach_auth(impl, req)) return kCatIoError;
    int rc = bolt::net::http_send(impl->arena, req, out);
    if (rc != 0) return kCatIoError;
    return classify_status(out->status);
}

void init_req(Impl* impl, bolt::net::HttpRequest* req, const char* method,
              const char* tail) noexcept {
    assert(impl != nullptr && req != nullptr && method != nullptr);
    std::memset(req, 0, sizeof(*req));
    std::strncpy(req->method, method, sizeof(req->method) - 1u);
    make_url(&impl->cfg, req->url, sizeof(req->url), tail);
    bolt::net::http_request_add_header(req, "Accept", "application/json");
}

bool build_listns_req_internal(const Impl* impl,
                               bolt::net::HttpRequest* out) noexcept {
    assert(impl != nullptr && out != nullptr);
    std::memset(out, 0, sizeof(*out));
    std::strncpy(out->method, "GET", sizeof(out->method) - 1u);
    make_url(&impl->cfg, out->url, sizeof(out->url), "/namespaces");
    bolt::net::http_request_add_header(out, "Accept", "application/json");
    if (impl->cfg.auth_mode == kAuthBearer) {
        char buf[600];
        std::snprintf(buf, sizeof(buf), "Bearer %s",
                      impl->cfg.bearer_token.value);
        bolt::net::http_request_add_header(out, "Authorization", buf);
    }
    return true;
}

// ---- /config discovery helpers ----

void apply_config_overrides(Impl* impl, const bj::StructuralIndex& idx,
                            int32_t obj_start) noexcept {
    assert(impl != nullptr);
    int32_t i = obj_start + 1;
    while (i < idx.token_count && idx.tokens[i].type != bj::TokenType::EndObject) {
        if (idx.tokens[i].type != bj::TokenType::Key) { i = skip_value(idx, i); continue; }
        bool is_prefix  = tok_str_eq(idx, i, "prefix", 6);
        bool is_default = tok_str_eq(idx, i, "default-catalog", 15);
        ++i;
        if (i >= idx.token_count) break;
        if (is_prefix && idx.tokens[i].type == bj::TokenType::String) {
            copy_str_tok(idx, i, impl->cfg.prefix, sizeof(impl->cfg.prefix));
        } else if (is_default && idx.tokens[i].type == bj::TokenType::String) {
            copy_str_tok(idx, i, impl->default_catalog,
                         sizeof(impl->default_catalog));
        }
        i = skip_value(idx, i);
    }
}

void scan_endpoints_for_cdc(Impl* impl, const bj::StructuralIndex& idx,
                            int32_t arr_start) noexcept {
    assert(impl != nullptr);
    int32_t i = arr_start + 1;
    while (i < idx.token_count && idx.tokens[i].type != bj::TokenType::EndArray) {
        if (idx.tokens[i].type == bj::TokenType::String) {
            uint32_t L;
            const uint8_t* p = tok_bytes(idx, i, &L);
            if (token_has_cdc(p, L)) impl->cdc_supported = 1u;
        }
        i = skip_value(idx, i);
    }
}

// Parse one snapshot object starting at BeginObject; returns cursor after it.
int32_t parse_one_snapshot(const bj::StructuralIndex& idx, int32_t obj,
                           SnapshotRef* s) noexcept {
    assert(s != nullptr);
    int32_t i = obj + 1;
    while (i < idx.token_count && idx.tokens[i].type != bj::TokenType::EndObject) {
        if (idx.tokens[i].type != bj::TokenType::Key) { i = skip_value(idx, i); continue; }
        bool is_id  = tok_str_eq(idx, i, "snapshot-id", 11);
        bool is_par = tok_str_eq(idx, i, "parent-snapshot-id", 18);
        bool is_seq = tok_str_eq(idx, i, "sequence-number", 15);
        bool is_op  = tok_str_eq(idx, i, "operation", 9);
        ++i;
        if (i >= idx.token_count) break;
        if (idx.tokens[i].type == bj::TokenType::Int64 && (is_id || is_par || is_seq)) {
            int64_t v = 0;
            bj::Iterator it{ &idx, i, 0 };
            if (bj::iter_int64(&it, &v)) {
                if (is_id) s->snapshot_id = v;
                else if (is_par) s->parent_snapshot_id = v;
                else s->sequence_number = v;
            }
        } else if (is_op && idx.tokens[i].type == bj::TokenType::String) {
            copy_str_tok(idx, i, s->operation, sizeof(s->operation));
        }
        i = skip_value(idx, i);
    }
    return (i < idx.token_count) ? i + 1 : i;
}

// Parse a load-table/view response's "metadata-location" into out[0..cap).
int parse_metadata_location(const uint8_t* body, uint32_t len,
                            bolt::Arena* arena, char* out,
                            uint32_t cap) noexcept {
    assert(body != nullptr && arena != nullptr && out != nullptr);
    bj::StructuralIndex idx;
    if (!bj::build_index(body, static_cast<int32_t>(len), arena, &idx))
        return kCatIoError;
    if (idx.token_count < 2) return kCatIoError;
    for (int32_t i = 1; i < idx.token_count; ++i) {
        if (tok_str_eq(idx, i, "metadata-location", 17) &&
            i + 1 < idx.token_count &&
            idx.tokens[i + 1].type == bj::TokenType::String) {
            copy_str_tok(idx, i + 1, out, cap);
            return kCatOk;
        }
    }
    out[0] = '\0';
    return kCatOk;
}

// ---- CatalogVT entry points ----

int vt_list_namespaces(void* impl_v, CatalogName* out, uint32_t cap,
                       uint32_t* out_n) noexcept {
    assert(impl_v != nullptr);
    assert(out != nullptr && out_n != nullptr);
    Impl* impl = static_cast<Impl*>(impl_v);
    *out_n = 0;
    char next_token[256];
    next_token[0] = '\0';
    uint32_t total = 0;
    for (uint32_t page = 0; page <= cap && total < cap; ++page) {
        bolt::net::HttpRequest req;
        if (!irc_build_list_namespaces_request_paged(
                impl, next_token[0] ? next_token : nullptr, 0u, &req))
            return kCatIoError;
        bolt::net::HttpResponse resp;
        int rc = do_request(impl, &req, &resp);
        if (rc != kCatOk) return rc;
        uint32_t got = 0;
        char nxt[256];
        if (!irc_parse_namespaces_paged(resp.body, resp.body_len, impl->arena,
                                        out + total, cap - total, &got, nxt,
                                        sizeof(nxt))) return kCatIoError;
        total += got;
        *out_n = total;
        if (nxt[0] == '\0') break;
        std::strncpy(next_token, nxt, sizeof(next_token) - 1u);
        next_token[sizeof(next_token) - 1u] = '\0';
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
    char tail[256];
    std::snprintf(tail, sizeof(tail), "/namespaces/%s/tables", ns);
    init_req(impl, &req, "GET", tail);
    bolt::net::HttpResponse resp;
    int rc = do_request(impl, &req, &resp);
    if (rc != kCatOk) return rc;

    bj::StructuralIndex idx;
    if (!bj::build_index(resp.body, static_cast<int32_t>(resp.body_len),
                         impl->arena, &idx)) return kCatIoError;
    if (idx.token_count < 2 ||
        idx.tokens[0].type != bj::TokenType::BeginObject) return kCatIoError;
    int32_t i = 1;
    while (i < idx.token_count && idx.tokens[i].type != bj::TokenType::EndObject) {
        if (idx.tokens[i].type != bj::TokenType::Key) { i = skip_value(idx, i); continue; }
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
                        copy_str_tok(idx, i, out[*out_n].name, kCatMaxName);
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
    return kCatOk;
}

int vt_table_path(void* impl_v, const char* ns, const char* name, char* out,
                  uint32_t cap) noexcept {
    assert(impl_v != nullptr);
    assert(ns != nullptr && name != nullptr && out != nullptr);
    Impl* impl = static_cast<Impl*>(impl_v);
    bolt::net::HttpRequest req;
    char tail[320];
    std::snprintf(tail, sizeof(tail), "/namespaces/%s/tables/%s", ns, name);
    init_req(impl, &req, "GET", tail);
    bolt::net::HttpResponse resp;
    int rc = do_request(impl, &req, &resp);
    if (rc != kCatOk) return rc;
    return parse_metadata_location(resp.body, resp.body_len, impl->arena, out, cap);
}

int vt_create_table(void* impl_v, const char* ns, const char* name,
                    TableLayout layout) noexcept {
    assert(impl_v != nullptr);
    assert(ns != nullptr && name != nullptr);
    (void)layout;
    Impl* impl = static_cast<Impl*>(impl_v);
    bolt::net::HttpRequest req;
    char tail[256];
    std::snprintf(tail, sizeof(tail), "/namespaces/%s/tables", ns);
    init_req(impl, &req, "POST", tail);
    bolt::net::http_request_add_header(&req, "Content-Type", "application/json");
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
    char tail[320];
    std::snprintf(tail, sizeof(tail), "/namespaces/%s/tables/%s", ns, name);
    init_req(impl, &req, "DELETE", tail);
    bolt::net::HttpResponse resp;
    return do_request(impl, &req, &resp);
}

int vt_commit_metadata(void* impl_v, const char* ns, const char* name,
                       const char* rel, const uint8_t* d, uint64_t n) noexcept {
    assert(impl_v != nullptr);
    assert(ns != nullptr && name != nullptr);
    (void)rel;
    Impl* impl = static_cast<Impl*>(impl_v);
    if (d == nullptr || n == 0u || n >= bolt::net::kHttpMaxBody) return kCatBadArg;
    bolt::net::HttpRequest req;
    char tail[320];
    std::snprintf(tail, sizeof(tail), "/namespaces/%s/tables/%s", ns, name);
    init_req(impl, &req, "POST", tail);
    bolt::net::http_request_add_header(&req, "Content-Type", "application/json");
    req.body = d;
    req.body_len = static_cast<uint32_t>(n);
    bolt::net::HttpResponse resp;
    return do_request(impl, &req, &resp);
}

const CatalogVT kVT = {
    vt_list_namespaces, vt_list_tables, vt_table_path,
    vt_create_table,    vt_drop_table,  vt_commit_metadata,
};

Impl* impl_of(Catalog* cat) noexcept {
    assert(cat != nullptr && cat->vt == &kVT);
    assert(cat->impl != nullptr);
    return static_cast<Impl*>(cat->impl);
}

// Submit a single update object as a one-table commit (requirements empty).
int submit_table_update(Impl* impl, const char* ns, const char* table,
                        const char* update_obj) noexcept {
    assert(impl != nullptr && ns != nullptr && table != nullptr);
    assert(update_obj != nullptr);
    bolt::net::HttpRequest req;
    char tail[360];
    std::snprintf(tail, sizeof(tail), "/namespaces/%s/tables/%s", ns, table);
    init_req(impl, &req, "POST", tail);
    bolt::net::http_request_add_header(&req, "Content-Type", "application/json");
    char body[2048];
    int blen = std::snprintf(body, sizeof(body),
                             "{\"requirements\":[],\"updates\":[%s]}", update_obj);
    if (blen <= 0 || static_cast<uint32_t>(blen) >= sizeof(body)) return kCatBadArg;
    req.body = reinterpret_cast<const uint8_t*>(body);
    req.body_len = static_cast<uint32_t>(blen);
    bolt::net::HttpResponse resp;
    return do_request(impl, &req, &resp);
}

}  // namespace

// ===========================================================================
// Public surface.
// ===========================================================================

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

int discover_config(Catalog* cat) noexcept {
    assert(cat != nullptr);
    Impl* impl = impl_of(cat);
    bolt::net::HttpRequest req;
    std::memset(&req, 0, sizeof(req));
    std::strncpy(req.method, "GET", sizeof(req.method) - 1u);
    std::snprintf(req.url, sizeof(req.url), "%s/v1/config", impl->cfg.base_url);
    bolt::net::http_request_add_header(&req, "Accept", "application/json");
    bolt::net::HttpResponse resp;
    int rc = do_request(impl, &req, &resp);
    if (rc == kCatNotFound) return kCatNotImplemented;
    if (rc != kCatOk) return rc;
    bj::StructuralIndex idx;
    if (!bj::build_index(resp.body, static_cast<int32_t>(resp.body_len),
                         impl->arena, &idx)) return kCatIoError;
    if (idx.token_count < 2 ||
        idx.tokens[0].type != bj::TokenType::BeginObject) return kCatIoError;
    int32_t i = 1;
    while (i < idx.token_count && idx.tokens[i].type != bj::TokenType::EndObject) {
        if (idx.tokens[i].type != bj::TokenType::Key) { i = skip_value(idx, i); continue; }
        bool is_def = tok_str_eq(idx, i, "defaults", 8);
        bool is_ovr = tok_str_eq(idx, i, "overrides", 9);
        bool is_eps = tok_str_eq(idx, i, "endpoints", 9);
        ++i;
        if (i >= idx.token_count) break;
        if ((is_def || is_ovr) && idx.tokens[i].type == bj::TokenType::BeginObject) {
            apply_config_overrides(impl, idx, i);
        } else if (is_eps && idx.tokens[i].type == bj::TokenType::BeginArray) {
            scan_endpoints_for_cdc(impl, idx, i);
        }
        i = skip_value(idx, i);
    }
    impl->config_discovered = 1u;
    return kCatOk;
}

uint32_t irc_base64_std(const uint8_t* src, uint32_t n, char* out,
                        uint32_t cap) noexcept {
    return base64_std(src, n, out, cap);
}

bool irc_build_list_namespaces_request(const Impl* impl,
                                       bolt::net::HttpRequest* out) noexcept {
    return build_listns_req_internal(impl, out);
}

bool irc_build_list_namespaces_request_paged(const Impl* impl,
                                             const char* page_token,
                                             uint32_t page_size,
                                             bolt::net::HttpRequest* out) noexcept {
    assert(impl != nullptr && out != nullptr);
    if (!build_listns_req_internal(impl, out)) return false;
    return append_paging(out->url, sizeof(out->url), page_token, page_size);
}

bool irc_build_oauth2_token_request(const Impl* impl,
                                    bolt::net::HttpRequest* out,
                                    char* body_buf, uint32_t body_cap) noexcept {
    assert(impl != nullptr && out != nullptr && body_buf != nullptr);
    std::memset(out, 0, sizeof(*out));
    std::strncpy(out->method, "POST", sizeof(out->method) - 1u);
    std::strncpy(out->url, impl->cfg.oauth_token_url, sizeof(out->url) - 1u);
    bolt::net::http_request_add_header(out, "Content-Type",
                                       "application/x-www-form-urlencoded");
    uint32_t blen = 0;
    if (!build_oauth2_body(&impl->cfg, body_buf, body_cap, &blen)) return false;
    out->body = reinterpret_cast<const uint8_t*>(body_buf);
    out->body_len = blen;
    return true;
}

uint32_t irc_build_tx_commit_body(const char* const* table_jsons, uint32_t n,
                                  char* out, uint32_t cap) noexcept {
    assert(out != nullptr);
    assert(table_jsons != nullptr || n == 0u);
    int w = std::snprintf(out, cap, "{\"table-changes\":[");
    if (w <= 0 || static_cast<uint32_t>(w) >= cap) return 0u;
    uint32_t o = static_cast<uint32_t>(w);
    for (uint32_t i = 0; i < n; ++i) {
        const char* sep = (i == 0) ? "" : ",";
        int k = std::snprintf(out + o, cap - o, "%s%s", sep, table_jsons[i]);
        if (k <= 0 || o + static_cast<uint32_t>(k) >= cap) return 0u;
        o += static_cast<uint32_t>(k);
    }
    int e = std::snprintf(out + o, cap - o, "]}");
    if (e <= 0 || o + static_cast<uint32_t>(e) >= cap) return 0u;
    return o + static_cast<uint32_t>(e);
}

uint32_t irc_build_set_snapshot_ref_update(const char* ref_name,
                                           const char* type,
                                           int64_t snapshot_id, char* out,
                                           uint32_t cap) noexcept {
    assert(ref_name != nullptr && type != nullptr && out != nullptr);
    int w = std::snprintf(out, cap,
                          "{\"action\":\"set-snapshot-ref\",\"ref-name\":\"%s\","
                          "\"type\":\"%s\",\"snapshot-id\":%lld}",
                          ref_name, type, static_cast<long long>(snapshot_id));
    if (w <= 0 || static_cast<uint32_t>(w) >= cap) return 0u;
    return static_cast<uint32_t>(w);
}

bool irc_parse_namespaces(const uint8_t* body, uint32_t len, bolt::Arena* arena,
                          CatalogName* out, uint32_t cap,
                          uint32_t* out_n) noexcept {
    return irc_parse_namespaces_paged(body, len, arena, out, cap, out_n, nullptr,
                                      0u);
}

bool irc_parse_namespaces_paged(const uint8_t* body, uint32_t len,
                                bolt::Arena* arena, CatalogName* out,
                                uint32_t cap, uint32_t* out_n,
                                char* out_next_token,
                                uint32_t token_cap) noexcept {
    assert(body != nullptr);
    assert(arena != nullptr && out != nullptr && out_n != nullptr);
    *out_n = 0;
    if (out_next_token != nullptr && token_cap > 0u) out_next_token[0] = '\0';
    bj::StructuralIndex idx;
    if (!bj::build_index(body, static_cast<int32_t>(len), arena, &idx))
        return false;
    if (idx.token_count < 2 ||
        idx.tokens[0].type != bj::TokenType::BeginObject) return false;
    int32_t i = 1;
    while (i < idx.token_count && idx.tokens[i].type != bj::TokenType::EndObject) {
        if (idx.tokens[i].type != bj::TokenType::Key) { i = skip_value(idx, i); continue; }
        bool is_ns  = tok_str_eq(idx, i, "namespaces", 10);
        bool is_tok = tok_str_eq(idx, i, "next-page-token", 15);
        ++i;
        if (i >= idx.token_count) break;
        if (is_tok && idx.tokens[i].type == bj::TokenType::String &&
            out_next_token != nullptr && token_cap > 0u) {
            copy_str_tok(idx, i, out_next_token, token_cap);
            i = skip_value(idx, i);
        } else if (is_ns && idx.tokens[i].type == bj::TokenType::BeginArray) {
            ++i;
            while (i < idx.token_count &&
                   idx.tokens[i].type != bj::TokenType::EndArray) {
                if (idx.tokens[i].type != bj::TokenType::BeginArray) {
                    i = skip_value(idx, i); continue;
                }
                ++i;
                if (i < idx.token_count &&
                    idx.tokens[i].type == bj::TokenType::String && *out_n < cap) {
                    copy_str_tok(idx, i, out[*out_n].name, kCatMaxName);
                    ++(*out_n);
                }
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
            if (i < idx.token_count) ++i;
        } else {
            i = skip_value(idx, i);
        }
    }
    return true;
}

bool irc_parse_snapshots(const uint8_t* body, uint32_t len, bolt::Arena* arena,
                         SnapshotRef* out, uint32_t cap,
                         uint32_t* out_n) noexcept {
    assert(body != nullptr && arena != nullptr && out != nullptr && out_n != nullptr);
    *out_n = 0;
    bj::StructuralIndex idx;
    if (!bj::build_index(body, static_cast<int32_t>(len), arena, &idx))
        return false;
    if (idx.token_count < 2) return false;
    int32_t arr = -1;
    for (int32_t i = 1; i < idx.token_count; ++i) {
        if (tok_str_eq(idx, i, "snapshots", 9) && i + 1 < idx.token_count &&
            idx.tokens[i + 1].type == bj::TokenType::BeginArray) {
            arr = i + 1; break;
        }
    }
    if (arr < 0) return true;
    int32_t i = arr + 1;
    while (i < idx.token_count && idx.tokens[i].type != bj::TokenType::EndArray &&
           *out_n < cap) {
        if (idx.tokens[i].type != bj::TokenType::BeginObject) {
            i = skip_value(idx, i); continue;
        }
        SnapshotRef* s = &out[*out_n];
        std::memset(s, 0, sizeof(*s));
        s->parent_snapshot_id = -1;
        s->sequence_number = -1;
        i = parse_one_snapshot(idx, i, s);
        ++(*out_n);
    }
    return true;
}

// ---- High-level ops ----

int commit_transaction(Catalog* cat, const char* const* table_jsons,
                       uint32_t n) noexcept {
    assert(cat != nullptr);
    assert(table_jsons != nullptr || n == 0u);
    Impl* impl = impl_of(cat);
    bolt::net::HttpRequest req;
    init_req(impl, &req, "POST", "/transactions/commit");
    bolt::net::http_request_add_header(&req, "Content-Type", "application/json");
    const uint32_t kBody = 64u * 1024u;
    char* body = static_cast<char*>(impl->arena->allocate(kBody, 16u));
    if (body == nullptr) return kCatIoError;
    uint32_t blen = irc_build_tx_commit_body(table_jsons, n, body, kBody);
    if (blen == 0u) return kCatBadArg;
    req.body = reinterpret_cast<const uint8_t*>(body);
    req.body_len = blen;
    bolt::net::HttpResponse resp;
    return do_request(impl, &req, &resp);
}

int list_snapshots(Catalog* cat, const char* ns, const char* table,
                   SnapshotRef* out, uint32_t cap, uint32_t* out_n) noexcept {
    assert(cat != nullptr && ns != nullptr && table != nullptr);
    assert(out != nullptr && out_n != nullptr);
    Impl* impl = impl_of(cat);
    *out_n = 0;
    bolt::net::HttpRequest req;
    char tail[360];
    std::snprintf(tail, sizeof(tail), "/namespaces/%s/tables/%s/snapshots", ns,
                  table);
    init_req(impl, &req, "GET", tail);
    bolt::net::HttpResponse resp;
    int rc = do_request(impl, &req, &resp);
    if (rc != kCatOk) return rc;
    if (!irc_parse_snapshots(resp.body, resp.body_len, impl->arena, out, cap,
                             out_n)) return kCatIoError;
    return kCatOk;
}

int register_table(Catalog* cat, const char* ns, const char* table,
                   const char* metadata_loc) noexcept {
    assert(cat != nullptr && ns != nullptr && table != nullptr);
    assert(metadata_loc != nullptr);
    Impl* impl = impl_of(cat);
    bolt::net::HttpRequest req;
    char tail[320];
    std::snprintf(tail, sizeof(tail), "/namespaces/%s/register", ns);
    init_req(impl, &req, "POST", tail);
    bolt::net::http_request_add_header(&req, "Content-Type", "application/json");
    char body[kCatMaxPath + 128u];
    int blen = std::snprintf(body, sizeof(body),
                             "{\"name\":\"%s\",\"metadata-location\":\"%s\"}",
                             table, metadata_loc);
    if (blen <= 0 || static_cast<uint32_t>(blen) >= sizeof(body)) return kCatBadArg;
    req.body = reinterpret_cast<const uint8_t*>(body);
    req.body_len = static_cast<uint32_t>(blen);
    bolt::net::HttpResponse resp;
    return do_request(impl, &req, &resp);
}

int rename_table(Catalog* cat, const char* from_ns, const char* from_name,
                 const char* to_ns, const char* to_name) noexcept {
    assert(cat != nullptr && from_ns != nullptr && from_name != nullptr);
    assert(to_ns != nullptr && to_name != nullptr);
    Impl* impl = impl_of(cat);
    bolt::net::HttpRequest req;
    init_req(impl, &req, "POST", "/tables/rename");
    bolt::net::http_request_add_header(&req, "Content-Type", "application/json");
    char body[1024];
    int blen = std::snprintf(body, sizeof(body),
                             "{\"source\":{\"namespace\":[\"%s\"],\"name\":\"%s\"},"
                             "\"destination\":{\"namespace\":[\"%s\"],\"name\":\"%s\"}}",
                             from_ns, from_name, to_ns, to_name);
    if (blen <= 0 || static_cast<uint32_t>(blen) >= sizeof(body)) return kCatBadArg;
    req.body = reinterpret_cast<const uint8_t*>(body);
    req.body_len = static_cast<uint32_t>(blen);
    bolt::net::HttpResponse resp;
    return do_request(impl, &req, &resp);
}

int set_snapshot_ref(Catalog* cat, const char* ns, const char* table,
                     const char* ref_name, const char* type,
                     int64_t snapshot_id) noexcept {
    assert(cat != nullptr && ref_name != nullptr && type != nullptr);
    Impl* impl = impl_of(cat);
    char upd[512];
    if (irc_build_set_snapshot_ref_update(ref_name, type, snapshot_id, upd,
                                          sizeof(upd)) == 0u) return kCatBadArg;
    return submit_table_update(impl, ns, table, upd);
}

int remove_snapshot_ref(Catalog* cat, const char* ns, const char* table,
                        const char* ref_name) noexcept {
    assert(cat != nullptr && ref_name != nullptr);
    Impl* impl = impl_of(cat);
    char upd[256];
    int w = std::snprintf(upd, sizeof(upd),
                          "{\"action\":\"remove-snapshot-ref\",\"ref-name\":\"%s\"}",
                          ref_name);
    if (w <= 0 || static_cast<uint32_t>(w) >= sizeof(upd)) return kCatBadArg;
    return submit_table_update(impl, ns, table, upd);
}

int rollback_to_snapshot(Catalog* cat, const char* ns, const char* table,
                         int64_t snapshot_id) noexcept {
    assert(cat != nullptr);
    return set_snapshot_ref(cat, ns, table, "main", "branch", snapshot_id);
}

int cherry_pick(Catalog* cat, const char* ns, const char* table,
                int64_t snapshot_id) noexcept {
    assert(cat != nullptr && ns != nullptr && table != nullptr);
    Impl* impl = impl_of(cat);
    char upd[256];
    int w = std::snprintf(upd, sizeof(upd),
                          "{\"action\":\"cherrypick-snapshot\","
                          "\"cherrypick-snapshot-id\":%lld}",
                          static_cast<long long>(snapshot_id));
    if (w <= 0 || static_cast<uint32_t>(w) >= sizeof(upd)) return kCatBadArg;
    return submit_table_update(impl, ns, table, upd);
}

int delete_snapshots(Catalog* cat, const char* ns, const char* table,
                     const int64_t* snapshot_ids, uint32_t n) noexcept {
    assert(cat != nullptr && ns != nullptr && table != nullptr);
    assert(snapshot_ids != nullptr || n == 0u);
    Impl* impl = impl_of(cat);
    char upd[2048];
    int w = std::snprintf(upd, sizeof(upd),
                          "{\"action\":\"remove-snapshots\",\"snapshot-ids\":[");
    if (w <= 0) return kCatBadArg;
    uint32_t o = static_cast<uint32_t>(w);
    for (uint32_t i = 0; i < n; ++i) {
        int k = std::snprintf(upd + o, sizeof(upd) - o, "%s%lld",
                              i == 0 ? "" : ",",
                              static_cast<long long>(snapshot_ids[i]));
        if (k <= 0 || o + static_cast<uint32_t>(k) >= sizeof(upd)) return kCatBadArg;
        o += static_cast<uint32_t>(k);
    }
    int e = std::snprintf(upd + o, sizeof(upd) - o, "]}");
    if (e <= 0 || o + static_cast<uint32_t>(e) >= sizeof(upd)) return kCatBadArg;
    return submit_table_update(impl, ns, table, upd);
}

int set_statistics(Catalog* cat, const char* ns, const char* table,
                   int64_t snapshot_id,
                   const char* statistics_file_json) noexcept {
    assert(cat != nullptr && ns != nullptr && table != nullptr);
    assert(statistics_file_json != nullptr);
    Impl* impl = impl_of(cat);
    char upd[2048];
    int w = std::snprintf(upd, sizeof(upd),
                          "{\"action\":\"set-statistics\",\"snapshot-id\":%lld,"
                          "\"statistics\":%s}",
                          static_cast<long long>(snapshot_id),
                          statistics_file_json);
    if (w <= 0 || static_cast<uint32_t>(w) >= sizeof(upd)) return kCatBadArg;
    return submit_table_update(impl, ns, table, upd);
}

int remove_statistics(Catalog* cat, const char* ns, const char* table,
                      int64_t snapshot_id) noexcept {
    assert(cat != nullptr && ns != nullptr && table != nullptr);
    Impl* impl = impl_of(cat);
    char upd[256];
    int w = std::snprintf(upd, sizeof(upd),
                          "{\"action\":\"remove-statistics\",\"snapshot-id\":%lld}",
                          static_cast<long long>(snapshot_id));
    if (w <= 0 || static_cast<uint32_t>(w) >= sizeof(upd)) return kCatBadArg;
    return submit_table_update(impl, ns, table, upd);
}

int set_partition_statistics(Catalog* cat, const char* ns, const char* table,
                             const char* partition_statistics_file_json) noexcept {
    assert(cat != nullptr && ns != nullptr && table != nullptr);
    assert(partition_statistics_file_json != nullptr);
    Impl* impl = impl_of(cat);
    char upd[2048];
    int w = std::snprintf(upd, sizeof(upd),
                          "{\"action\":\"set-partition-statistics\","
                          "\"partition-statistics\":%s}",
                          partition_statistics_file_json);
    if (w <= 0 || static_cast<uint32_t>(w) >= sizeof(upd)) return kCatBadArg;
    return submit_table_update(impl, ns, table, upd);
}

// ---- Views ----

int create_view(Catalog* cat, const char* ns, const char* name,
                const char* schema_json, const char* sql) noexcept {
    assert(cat != nullptr && ns != nullptr && name != nullptr);
    assert(schema_json != nullptr && sql != nullptr);
    Impl* impl = impl_of(cat);
    bolt::net::HttpRequest req;
    char tail[320];
    std::snprintf(tail, sizeof(tail), "/namespaces/%s/views", ns);
    init_req(impl, &req, "POST", tail);
    bolt::net::http_request_add_header(&req, "Content-Type", "application/json");
    const uint32_t kBody = 16u * 1024u;
    char* body = static_cast<char*>(impl->arena->allocate(kBody, 16u));
    if (body == nullptr) return kCatIoError;
    int blen = std::snprintf(body, kBody,
                             "{\"name\":\"%s\",\"schema\":%s,\"view-version\":{"
                             "\"version-id\":1,\"schema-id\":0,\"summary\":{},"
                             "\"default-namespace\":[\"%s\"],\"representations\":["
                             "{\"type\":\"sql\",\"sql\":\"%s\",\"dialect\":\"spark\"}]}}",
                             name, schema_json, ns, sql);
    if (blen <= 0 || static_cast<uint32_t>(blen) >= kBody) return kCatBadArg;
    req.body = reinterpret_cast<const uint8_t*>(body);
    req.body_len = static_cast<uint32_t>(blen);
    bolt::net::HttpResponse resp;
    return do_request(impl, &req, &resp);
}

int drop_view(Catalog* cat, const char* ns, const char* name) noexcept {
    assert(cat != nullptr && ns != nullptr && name != nullptr);
    Impl* impl = impl_of(cat);
    bolt::net::HttpRequest req;
    char tail[360];
    std::snprintf(tail, sizeof(tail), "/namespaces/%s/views/%s", ns, name);
    init_req(impl, &req, "DELETE", tail);
    bolt::net::HttpResponse resp;
    return do_request(impl, &req, &resp);
}

int load_view(Catalog* cat, const char* ns, const char* name, char* out_loc,
              uint32_t cap) noexcept {
    assert(cat != nullptr && ns != nullptr && name != nullptr && out_loc != nullptr);
    Impl* impl = impl_of(cat);
    bolt::net::HttpRequest req;
    char tail[360];
    std::snprintf(tail, sizeof(tail), "/namespaces/%s/views/%s", ns, name);
    init_req(impl, &req, "GET", tail);
    bolt::net::HttpResponse resp;
    int rc = do_request(impl, &req, &resp);
    if (rc != kCatOk) return rc;
    return parse_metadata_location(resp.body, resp.body_len, impl->arena,
                                   out_loc, cap);
}

int rename_view(Catalog* cat, const char* from_ns, const char* from_name,
                const char* to_ns, const char* to_name) noexcept {
    assert(cat != nullptr && from_ns != nullptr && from_name != nullptr);
    assert(to_ns != nullptr && to_name != nullptr);
    Impl* impl = impl_of(cat);
    bolt::net::HttpRequest req;
    init_req(impl, &req, "POST", "/views/rename");
    bolt::net::http_request_add_header(&req, "Content-Type", "application/json");
    char body[1024];
    int blen = std::snprintf(body, sizeof(body),
                             "{\"source\":{\"namespace\":[\"%s\"],\"name\":\"%s\"},"
                             "\"destination\":{\"namespace\":[\"%s\"],\"name\":\"%s\"}}",
                             from_ns, from_name, to_ns, to_name);
    if (blen <= 0 || static_cast<uint32_t>(blen) >= sizeof(body)) return kCatBadArg;
    req.body = reinterpret_cast<const uint8_t*>(body);
    req.body_len = static_cast<uint32_t>(blen);
    bolt::net::HttpResponse resp;
    return do_request(impl, &req, &resp);
}

int read_cdc(Catalog* cat, const char* ns, const char* table,
             const uint8_t** out_body, uint32_t* out_len) noexcept {
    assert(cat != nullptr && ns != nullptr && table != nullptr);
    assert(out_body != nullptr && out_len != nullptr);
    Impl* impl = impl_of(cat);
    *out_body = nullptr;
    *out_len = 0;
    if (impl->config_discovered != 0u && impl->cdc_supported == 0u)
        return kCatNotImplemented;
    bolt::net::HttpRequest req;
    char tail[360];
    std::snprintf(tail, sizeof(tail), "/namespaces/%s/tables/%s/cdc", ns, table);
    init_req(impl, &req, "GET", tail);
    bolt::net::HttpResponse resp;
    int rc = do_request(impl, &req, &resp);
    if (rc != kCatOk) return rc;
    *out_body = resp.body;
    *out_len = resp.body_len;
    return kCatOk;
}

}  // namespace iceberg_rest
}  // namespace lakehouse
}  // namespace bolt
