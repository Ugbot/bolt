// bolt/lakehouse/object_store_s3.cpp — S3 ObjectStore over bolt::net's
// HTTP/1.1 client, SigV4-signed. Works against AWS and S3-compatible stores
// (MinIO, R2, ...). put_if_absent is PutObject + If-None-Match: *.

#include "bolt/lakehouse/object_store.h"

#include <cstdlib>
#include <cstring>

#include "bolt/crypto/sigv4.h"
#include "object_store_http.h"

namespace bolt {
namespace lakehouse {

namespace {

using oshttp::Buf;

// url/host/path/query for one request against `key` ("" = bucket root).
struct S3Target {
    char url[net::kHttpMaxUrl];
    char host[256];
    char path[net::kHttpMaxUrl];
};

bool s3_target(const S3ObjectStore* s3, const char* key, const char* query,
               S3Target* t) noexcept {
    assert(s3 != nullptr && key != nullptr);
    assert(t != nullptr);
    char ob[kS3MaxEndpoint + kS3MaxBucket + 64u];
    char bb[kS3MaxEndpoint];
    Buf origin; oshttp::buf_init(&origin, ob, sizeof(ob));
    Buf base;   oshttp::buf_init(&base, bb, sizeof(bb));
    if (s3->endpoint[0] == '\0') {
        oshttp::buf_str(&origin, "https://");
        if (!s3->path_style) {
            oshttp::buf_str(&origin, s3->bucket);
            oshttp::buf_str(&origin, ".");
        }
        oshttp::buf_str(&origin, "s3.");
        oshttp::buf_str(&origin, s3->region);
        oshttp::buf_str(&origin, ".amazonaws.com");
    } else {
        char eb[kS3MaxEndpoint + 16u];
        Buf ep; oshttp::buf_init(&ep, eb, sizeof(eb));
        if (!oshttp::split_endpoint(s3->endpoint, &ep, &base)) return false;
        const char* hp = std::strstr(eb, "://");
        assert(hp != nullptr);
        hp += 3;
        oshttp::buf_raw(&origin, eb, static_cast<uint32_t>(hp - eb));
        if (!s3->path_style) {
            oshttp::buf_str(&origin, s3->bucket);
            oshttp::buf_str(&origin, ".");
        }
        oshttp::buf_str(&origin, hp);
    }
    Buf path; oshttp::buf_init(&path, t->path, sizeof(t->path));
    oshttp::buf_str(&path, bb);
    oshttp::buf_str(&path, "/");
    if (s3->path_style) {
        oshttp::buf_str(&path, s3->bucket);
        oshttp::buf_str(&path, "/");
    }
    oshttp::buf_pct(&path, key, /*keep_slash=*/true);
    Buf url; oshttp::buf_init(&url, t->url, sizeof(t->url));
    oshttp::buf_str(&url, ob);
    oshttp::buf_str(&url, t->path);
    if (query != nullptr && query[0] != '\0') {
        oshttp::buf_str(&url, "?");
        oshttp::buf_str(&url, query);
    }
    if (!origin.ok || !path.ok || !url.ok) return false;
    return oshttp::url_host(t->url, t->host, sizeof(t->host));
}

// Build, SigV4-sign and send one request. `if_none_match` adds the
// create-only precondition (If-None-Match: *, or GCS's signed
// x-goog-if-generation-match: 0) that makes a PUT a CAS.
int s3_send(S3ObjectStore* s3, const char* method, const char* key,
            const char* query, const uint8_t* body, uint64_t len,
            bool if_none_match, Arena* arena,
            net::HttpResponse* resp) noexcept {
    assert(s3 != nullptr && method != nullptr && key != nullptr);
    assert(arena != nullptr && resp != nullptr);
    if (len > net::kHttpMaxBody) return kOsBadArg;
    S3Target t;
    if (!s3_target(s3, key, query, &t)) return kOsBadArg;
    net::HttpRequest req;
    std::memset(&req, 0, sizeof(req));
    std::strncpy(req.method, method, sizeof(req.method) - 1u);
    std::memcpy(req.url, t.url, sizeof(req.url));
    req.body = body;
    req.body_len = static_cast<uint32_t>(len);

    char body_hash[65];
    if (len > 0) crypto::sha256_hex(body, len, body_hash);
    else         crypto::sha256_empty_hex(body_hash);
    std::tm tmv;
    oshttp::utc_now(&tmv);
    char amz_date[20];
    std::snprintf(amz_date, sizeof(amz_date), "%04d%02d%02dT%02d%02d%02dZ",
                  tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday,
                  tmv.tm_hour, tmv.tm_min, tmv.tm_sec);
    crypto::SigV4Request sr;
    std::memset(&sr, 0, sizeof(sr));
    sr.method = method;
    sr.host = t.host;
    sr.path = t.path;
    sr.query_string = query != nullptr ? query : "";
    sr.service = "s3";
    sr.region = s3->region;
    sr.amz_date = amz_date;
    sr.payload_sha256 = body_hash;
    sr.access_key = s3->access_key;
    sr.secret_key = s3->secret_key;
    sr.session_token = s3->session_token[0] != '\0' ? s3->session_token
                                                    : nullptr;
    const bool goog_cas = if_none_match &&
                          s3->create_only == kS3CreateOnlyGoogGeneration;
    if (goog_cas) sr.extra_header = "x-goog-if-generation-match:0";
    char auth[crypto::kSigV4MaxAuth];
    char amz_out[24];
    crypto::SigV4Result so;
    so.auth_header = auth;
    so.auth_cap = sizeof(auth);
    so.amz_date_out = amz_out;
    so.amz_date_cap = sizeof(amz_out);
    if (!crypto::sigv4_sign(&sr, &so)) return kOsBadArg;
    bool ok = net::http_request_add_header(&req, "X-Amz-Date", amz_date) &&
              net::http_request_add_header(&req, "X-Amz-Content-Sha256",
                                           body_hash) &&
              net::http_request_add_header(&req, "Authorization", auth);
    if (sr.session_token != nullptr) {
        ok = ok && net::http_request_add_header(&req, "X-Amz-Security-Token",
                                                sr.session_token);
    }
    if (goog_cas) {
        ok = ok && net::http_request_add_header(
                       &req, "x-goog-if-generation-match", "0");
    } else if (if_none_match) {
        ok = ok && net::http_request_add_header(&req, "If-None-Match", "*");
    }
    if (len == 0 && std::strcmp(method, "PUT") == 0) {
        ok = ok && net::http_request_add_header(&req, "Content-Length", "0");
    }
    if (!ok) return kOsBadArg;
    return net::http_send(arena, &req, resp) == 0 ? kOsOk : kOsIoError;
}

int s3_get(void* impl, const char* key, Arena* arena,
           const uint8_t** out_data, uint64_t* out_len) noexcept {
    assert(impl != nullptr && key != nullptr && arena != nullptr);
    assert(out_data != nullptr && out_len != nullptr);
    *out_data = nullptr;
    *out_len = 0;
    net::HttpResponse resp;
    const int rc = s3_send(static_cast<S3ObjectStore*>(impl), "GET", key, "",
                           nullptr, 0, false, arena, &resp);
    if (rc != kOsOk) return rc;
    const int st = oshttp::map_status(resp.status);
    if (st != kOsOk) return st;
    *out_data = resp.body;
    *out_len = resp.body_len;
    return kOsOk;
}

int s3_put(void* impl, const char* key, const uint8_t* data,
           uint64_t len) noexcept {
    assert(impl != nullptr && key != nullptr);
    assert(data != nullptr || len == 0);
    Arena scratch;
    net::HttpResponse resp;
    const int rc = s3_send(static_cast<S3ObjectStore*>(impl), "PUT", key, "",
                           data, len, false, &scratch, &resp);
    if (rc != kOsOk) return rc;
    return resp.status / 100 == 2 ? kOsOk : kOsIoError;
}

// S3 conditional writes: PutObject with If-None-Match: * fails 412 when the
// key exists. 409 (a concurrent conditional write in flight) is transient.
int s3_put_if_absent(void* impl, const char* key, const uint8_t* data,
                     uint64_t len) noexcept {
    assert(impl != nullptr && key != nullptr);
    assert(data != nullptr || len == 0);
    Arena scratch;
    net::HttpResponse resp;
    const int rc = s3_send(static_cast<S3ObjectStore*>(impl), "PUT", key, "",
                           data, len, true, &scratch, &resp);
    if (rc != kOsOk) return rc;
    if (resp.status == 412) return kOsExists;
    return resp.status / 100 == 2 ? kOsOk : kOsIoError;
}

int s3_list(void* impl, const char* prefix, ObjectEntry* out, uint32_t cap,
            uint32_t* out_n) noexcept {
    assert(impl != nullptr);
    assert(out != nullptr && out_n != nullptr);
    *out_n = 0;
    S3ObjectStore* s3 = static_cast<S3ObjectStore*>(impl);
    char token[1024] = "";
    uint32_t count = 0;
    for (uint32_t page = 0; page < 4096u && count < cap; ++page) {
        char qb[4096];
        Buf q; oshttp::buf_init(&q, qb, sizeof(qb));
        if (token[0] != '\0') {
            oshttp::buf_str(&q, "continuation-token=");
            oshttp::buf_pct(&q, token, false);
            oshttp::buf_str(&q, "&");
        }
        oshttp::buf_str(&q, "list-type=2&prefix=");
        oshttp::buf_pct(&q, prefix != nullptr ? prefix : "", false);
        if (!q.ok) return kOsBadArg;
        Arena scratch;
        net::HttpResponse resp;
        const int rc = s3_send(s3, "GET", "", qb, nullptr, 0, false,
                               &scratch, &resp);
        if (rc != kOsOk) return rc;
        if (resp.status / 100 != 2) return kOsIoError;
        const char* x = reinterpret_cast<const char*>(resp.body);
        uint32_t pos = 0;
        while (count < cap &&
               oshttp::xml_next(x, resp.body_len, &pos, "Key",
                                out[count].key, kOsMaxKey)) {
            oshttp::xml_unescape(out[count].key);
            char sz[32];
            if (!oshttp::xml_next(x, resp.body_len, &pos, "Size", sz,
                                  sizeof(sz))) {
                return kOsIoError;
            }
            out[count].size = std::strtoull(sz, nullptr, 10);
            ++count;
        }
        char trunc[8] = "";
        uint32_t tp = 0;
        (void)oshttp::xml_next(x, resp.body_len, &tp, "IsTruncated", trunc,
                               sizeof(trunc));
        if (std::strcmp(trunc, "true") != 0 &&
            std::strcmp(trunc, "True") != 0) break;
        tp = 0;
        if (!oshttp::xml_next(x, resp.body_len, &tp, "NextContinuationToken",
                              token, sizeof(token))) {
            return kOsIoError;
        }
        oshttp::xml_unescape(token);
    }
    *out_n = count;
    assert(count <= cap);
    return kOsOk;
}

int s3_delete(void* impl, const char* key) noexcept {
    assert(impl != nullptr);
    assert(key != nullptr);
    Arena scratch;
    net::HttpResponse resp;
    const int rc = s3_send(static_cast<S3ObjectStore*>(impl), "DELETE", key,
                           "", nullptr, 0, false, &scratch, &resp);
    if (rc != kOsOk) return rc;
    return oshttp::map_status(resp.status);
}

int s3_head(void* impl, const char* key, ObjectMeta* out) noexcept {
    assert(impl != nullptr);
    assert(out != nullptr && key != nullptr);
    std::memset(out, 0, sizeof(*out));
    Arena scratch;
    net::HttpResponse resp;
    const int rc = s3_send(static_cast<S3ObjectStore*>(impl), "HEAD", key, "",
                           nullptr, 0, false, &scratch, &resp);
    if (rc != kOsOk) return rc;
    const int st = oshttp::map_status(resp.status);
    if (st != kOsOk) return st;
    const char* cl = net::http_header_get(resp.headers, resp.header_count,
                                          "Content-Length");
    out->size = cl != nullptr ? std::strtoull(cl, nullptr, 10) : 0u;
    out->exists = true;
    return kOsOk;
}

const ObjectStoreVT kS3VT = {
    s3_get, s3_put, s3_list, s3_delete, s3_head, s3_put_if_absent,
};

// Bounded config-field copy. Returns false on overflow.
bool cpy(char* dst, uint32_t cap, const char* src) noexcept {
    assert(dst != nullptr);
    if (src == nullptr) { dst[0] = '\0'; return true; }
    const size_t sl = std::strlen(src);
    if (sl + 1u > cap) return false;
    std::memcpy(dst, src, sl + 1u);
    return true;
}

}  // namespace

bool s3_object_store_init(S3ObjectStore* s3, const char* bucket,
                          const char* region, const char* access_key,
                          const char* secret_key, ObjectStore* out) noexcept {
    assert(s3 != nullptr);
    assert(out != nullptr);
    if (s3 == nullptr || out == nullptr) return false;
    std::memset(s3, 0, sizeof(*s3));
    if (!cpy(s3->bucket, kS3MaxBucket, bucket) ||
        !cpy(s3->region, kS3MaxRegion, region) ||
        !cpy(s3->access_key, kS3MaxAccess, access_key) ||
        !cpy(s3->secret_key, kS3MaxSecret, secret_key)) {
        return false;
    }
    out->vt = &kS3VT;
    out->impl = s3;
    return true;
}

}  // namespace lakehouse
}  // namespace bolt
