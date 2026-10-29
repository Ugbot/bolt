// bolt/lakehouse/object_store_s3.cpp — S3 ObjectStore over bolt::net's
// HTTP/1.1 client, SigV4-signed. Works against AWS and S3-compatible stores
// (MinIO, R2, ...). put_if_absent is PutObject + If-None-Match: *.

#include "bolt/lakehouse/object_store.h"
#include "bolt/lakehouse/object_store/s3_upload.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string_view>

#include "bolt/crypto/sigv4.h"
#include "object_store_http.h"

namespace bolt {
namespace lakehouse {

namespace {

using oshttp::Buf;
static_assert(kOsStrongEtagCap == crypto::kSigV4MaxIfMatch);

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
            net::HttpResponse* resp, const char* range = nullptr,
            uint16_t timeout_ms = 0, uint32_t response_limit = 0,
            const char* if_match = nullptr) noexcept {
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
    req.timeout_ms = timeout_ms;
    req.max_response_body = response_limit;

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
    sr.if_match = if_match;
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
    if (len != 0 && std::strcmp(method, "POST") == 0)
        ok = ok && net::http_request_add_header(&req, "Content-Type", "application/xml");
    if (range != nullptr) ok = ok && net::http_request_add_header(&req, "Range", range);
    if (if_match != nullptr) ok = ok && net::http_request_add_header(&req, "If-Match", if_match);
    if (!ok) return kOsBadArg;
    return net::http_send(arena, &req, resp) == 0 ? kOsOk : kOsIoError;
}

constexpr uint32_t kVersionedControlBytes = 64u << 10;

bool header_name_equal(const char* a, const char* b) noexcept {
    assert(a != nullptr && b != nullptr);
    for (uint32_t i = 0; i < 64; ++i) {
        const unsigned char x = static_cast<unsigned char>(a[i]);
        const unsigned char y = static_cast<unsigned char>(b[i]);
        const auto lower = [](unsigned char c) { return c >= 'A' && c <= 'Z' ? c + ('a' - 'A') : c; };
        if (lower(x) != lower(y)) return false;
        if (x == 0) return true;
    }
    assert(a != nullptr);
    return false;
}
bool single_header(const net::HttpResponse& r, const char* name, const char** out) noexcept {
    assert(name != nullptr && out != nullptr);
    assert(r.header_count <= net::kHttpMaxHeaders);
    *out = nullptr;
    uint32_t count = 0;
    for (uint32_t i = 0; i < r.header_count; ++i) {
        if (!header_name_equal(r.headers[i].name, name)) continue;
        ++count;
        *out = r.headers[i].value;
    }
    return count == 1;
}
bool decimal_u64(std::string_view value, uint64_t* out) noexcept {
    assert(out != nullptr);
    *out = 0;
    if (value.empty() || value.size() > 20) return false;
    for (const char ch : value) {
        if (ch < '0' || ch > '9') return false;
        const auto digit = static_cast<uint64_t>(ch - '0');
        if (*out > (UINT64_MAX - digit) / 10u) return false;
        *out = *out * 10u + digit;
    }
    assert(value.size() <= 20);
    return true;
}
bool response_version(const net::HttpResponse& response, ObjectVersion* out) noexcept {
    assert(out != nullptr);
    assert(response.header_count <= net::kHttpMaxHeaders);
    const char* length = nullptr;
    const char* etag = nullptr;
    ObjectVersion version{};
    if (!single_header(response, "Content-Length", &length) ||
        !decimal_u64(length, &version.size) || !single_header(response, "ETag", &etag)) return false;
    const size_t n = std::strlen(etag);
    if (n >= sizeof(version.etag)) return false;
    std::memcpy(version.etag, etag, n + 1u);
    if (!os_object_version_valid(&version)) return false;
    *out = version;
    return true;
}
bool exact_content_range(const char* text, uint64_t off, uint32_t length, uint64_t size) noexcept {
    assert(text != nullptr);
    assert(length > 0 && off <= size && length <= size - off);
    std::string_view value(text);
    if (!value.starts_with("bytes ")) return false;
    value.remove_prefix(6);
    const size_t dash = value.find('-'), slash = value.find('/');
    if (dash == std::string_view::npos || slash == std::string_view::npos || slash <= dash) return false;
    uint64_t begin = 0, end = 0, total = 0;
    return decimal_u64(value.substr(0, dash), &begin) &&
        decimal_u64(value.substr(dash + 1u, slash - dash - 1u), &end) &&
        decimal_u64(value.substr(slash + 1u), &total) &&
        begin == off && end == off + length - 1u && total == size;
}
int s3_head_versioned(void* impl, const char* key, Arena* scratch, ObjectVersion* out) noexcept {
    assert(impl != nullptr && key != nullptr);
    assert(scratch != nullptr && out != nullptr);
    detail::ObjectSpan key_span{};
    const detail::ObjectSpan output{out, sizeof(*out)}, state{impl, sizeof(S3ObjectStore)};
    if (!detail::object_key_span(key, kOsMaxKey, &key_span) ||
        !detail::object_io_admit(&key_span, 1, &output, 1, scratch, &state, 1)) return kOsBadArg;
    *out = {};
    net::HttpResponse response{};
    const int rc = s3_send(static_cast<S3ObjectStore*>(impl), "HEAD", key, "", nullptr, 0,
                          false, scratch, &response, nullptr, 0, kVersionedControlBytes);
    if (rc != kOsOk) return rc;
    if (response.status == 404) return kOsNotFound;
    if (response.status != 200 || !response_version(response, out)) return kOsIoError;
    return kOsOk;
}
int s3_get_range_if_match(void* impl, const char* key, const ObjectVersion* expected,
                          uint64_t offset, uint32_t length, Arena* scratch,
                          uint8_t* dst, uint32_t* out_length) noexcept {
    assert(impl != nullptr && key != nullptr && expected != nullptr);
    assert(scratch != nullptr && out_length != nullptr && (dst != nullptr || length == 0));
    detail::ObjectSpan inputs[2]{{}, {expected, sizeof(*expected)}};
    const detail::ObjectSpan outputs[]{{dst, length > kOsVersionedRangeMaxBytes ? 0u : length},
        {out_length, sizeof(*out_length)}};
    const detail::ObjectSpan state{impl, sizeof(S3ObjectStore)};
    if (!detail::object_key_span(key, kOsMaxKey, &inputs[0]) ||
        !detail::object_io_admit(inputs, 2, outputs, 2, scratch, &state, 1)) return kOsBadArg;
    *out_length = 0;
    if (!os_object_version_valid(expected) || length > kOsVersionedRangeMaxBytes ||
        offset > expected->size || length > expected->size - offset) return kOsBadArg;
    if (length == 0) return kOsOk;
    char range[64]{};
    const int n = std::snprintf(range, sizeof(range), "bytes=%llu-%llu",
        static_cast<unsigned long long>(offset), static_cast<unsigned long long>(offset + length - 1u));
    if (n <= 0 || static_cast<size_t>(n) >= sizeof(range)) return kOsBadArg;
    net::HttpResponse response{};
    const uint32_t response_budget = length > kVersionedControlBytes ? length : kVersionedControlBytes;
    const int rc = s3_send(static_cast<S3ObjectStore*>(impl), "GET", key, "", nullptr, 0,
        false, scratch, &response, range, 0, response_budget, expected->etag);
    if (rc != kOsOk) return rc;
    if (response.status == 412) return kOsPreconditionFailed;
    if (response.status == 404) return kOsNotFound;
    const char* etag = nullptr;
    const char* content_range = nullptr;
    if (response.status != 206 || !single_header(response, "ETag", &etag) ||
        !single_header(response, "Content-Range", &content_range)) return kOsIoError;
    ObjectVersion actual{};
    const size_t tag_size = std::strlen(etag);
    if (tag_size >= sizeof(actual.etag)) return kOsIoError;
    std::memcpy(actual.etag, etag, tag_size + 1u);
    if (!os_object_version_valid(&actual)) return kOsIoError;
    if (std::strcmp(actual.etag, expected->etag) != 0) return kOsPreconditionFailed;
    if (!exact_content_range(content_range, offset, length, expected->size) ||
        response.body_len != length || response.body == nullptr) return kOsIoError;
    std::memcpy(dst, response.body, length);
    *out_length = length;
    return kOsOk;
}

struct ContiguousSource { const uint8_t* data; };
int contiguous_part(void* context, uint64_t offset, uint32_t length,
                     const uint8_t** out) noexcept {
    assert(context != nullptr && out != nullptr);
    assert(length > 0);
    *out = static_cast<ContiguousSource*>(context)->data + offset;
    return kOsOk;
}

int multipart_contiguous(S3ObjectStore* store, const char* key,
                          const uint8_t* data, uint64_t length,
                          bool if_absent) noexcept {
    assert(store != nullptr && data != nullptr);
    assert(length > net::kHttpMaxBody);
    if (length > SIZE_MAX) return kOsBadArg;
    Arena workspace;
    S3UploadOptions options;
    options.if_absent = if_absent;
    ContiguousSource contiguous{data};
    S3UploadSource source{contiguous_part, &contiguous};
    S3UploadResult result;
    return s3_upload_stream(store, key, length, &source, &options, &workspace, &result);
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

// Ranged GET. A 206 carries exactly the range; a server that ignores Range
// answers 200 with the whole object, from which the range is cut.
int s3_get_range(void* impl, const char* key, uint64_t offset, uint64_t len,
                 uint8_t* dst, uint64_t* out_len) noexcept {
    assert(impl != nullptr && key != nullptr);
    assert(out_len != nullptr && (dst != nullptr || len == 0));
    *out_len = 0;
    if (len == 0) return kOsOk;
    if (offset > UINT64_MAX - len) return kOsBadArg;
    char range[64];
    std::snprintf(range, sizeof(range), "bytes=%llu-%llu",
                  static_cast<unsigned long long>(offset),
                  static_cast<unsigned long long>(offset + len - 1u));
    Arena scratch;
    net::HttpResponse resp;
    const int rc = s3_send(static_cast<S3ObjectStore*>(impl), "GET", key, "",
                           nullptr, 0, false, &scratch, &resp, range);
    if (rc != kOsOk) return rc;
    if (resp.status == 416) return kOsOk;               // offset at/after end
    const int st = oshttp::map_status(resp.status);
    if (st != kOsOk) return st;
    const uint8_t* src = resp.body;
    uint64_t n = resp.body_len;
    if (resp.status == 200) {                           // Range ignored
        if (offset >= n) return kOsOk;
        src += offset;
        n -= offset;
    }
    if (n > len) n = len;
    std::memcpy(dst, src, static_cast<size_t>(n));
    *out_len = n;
    return kOsOk;
}

int s3_put(void* impl, const char* key, const uint8_t* data,
           uint64_t len) noexcept {
    assert(impl != nullptr && key != nullptr);
    assert(data != nullptr || len == 0);
    if (len > net::kHttpMaxBody)
        return multipart_contiguous(static_cast<S3ObjectStore*>(impl), key, data, len, false);
    Arena scratch;
    net::HttpResponse resp;
    const int rc = s3_send(static_cast<S3ObjectStore*>(impl), "PUT", key, "",
                           data, len, false, &scratch, &resp);
    if (rc != kOsOk) return rc;
    return resp.status / 100 == 2 ? kOsOk : kOsIoError;
}

// S3 conditional writes: PutObject with If-None-Match: * fails 412 when the
// key exists. 409 (a concurrent conditional write in flight), server errors
// and lost acknowledgements are uncertain; this layer never retries them.
int s3_put_if_absent(void* impl, const char* key, const uint8_t* data,
                     uint64_t len) noexcept {
    assert(impl != nullptr && key != nullptr);
    assert(data != nullptr || len == 0);
    if (len > net::kHttpMaxBody)
        return multipart_contiguous(static_cast<S3ObjectStore*>(impl), key, data, len, true);
    Arena scratch;
    net::HttpResponse resp;
    const int rc = s3_send(static_cast<S3ObjectStore*>(impl), "PUT", key, "",
                           data, len, true, &scratch, &resp);
    // Admission/signing errors precede I/O. Once the HTTP exchange is
    // attempted, a lost response cannot prove that the conditional PUT lost.
    if (rc != kOsOk) return rc == kOsIoError ? kOsWriteUncertain : rc;
    if (resp.status == 412) return kOsExists;
    // A concurrent conditional request or server error may race publication.
    if (resp.status == 409 || resp.status >= 500) return kOsWriteUncertain;
    return resp.status / 100 == 2 ? kOsOk : kOsIoError;
}

// s3_put_if_absent reporting the version the PutObject response acknowledged (its ETag, the body's length).
int s3_put_if_absent_versioned(void* impl, const char* key, const uint8_t* data, uint64_t len,
                               ObjectVersion* out) noexcept {
    assert(impl != nullptr && key != nullptr && out != nullptr);
    assert(data != nullptr || len == 0);
    *out = {};
    if (len > net::kHttpMaxBody)
        return multipart_contiguous(static_cast<S3ObjectStore*>(impl), key, data, len, true);
    Arena scratch;
    net::HttpResponse resp;
    const int rc = s3_send(static_cast<S3ObjectStore*>(impl), "PUT", key, "",
                           data, len, true, &scratch, &resp);
    if (rc != kOsOk) return rc == kOsIoError ? kOsWriteUncertain : rc;
    if (resp.status == 412) return kOsExists;
    if (resp.status == 409 || resp.status >= 500) return kOsWriteUncertain;
    if (resp.status / 100 != 2) return kOsIoError;
    const char* etag = nullptr;
    ObjectVersion v{};
    v.size = len;
    if (single_header(resp, "ETag", &etag) && std::strlen(etag) < sizeof(v.etag)) {
        std::memcpy(v.etag, etag, std::strlen(etag) + 1u);
        if (os_object_version_valid(&v)) *out = v;
    }
    return kOsOk;
}

// ListObjectsV2, one request per page, calling fn(ctx, key, size) per key; a non-kOsOk return from fn stops with it.
int s3_walk(S3ObjectStore* s3, const char* prefix, ObjectVisitFn fn, void* ctx) noexcept {
    assert(s3 != nullptr);
    assert(fn != nullptr);
    char token[1024] = "";
    char key[kOsMaxKey];
    bool more = true;
    for (uint32_t page = 0; page < 4096u && more; ++page) {
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
        while (oshttp::xml_next(x, resp.body_len, &pos, "Key", key, kOsMaxKey)) {
            oshttp::xml_unescape(key);
            char sz[32];
            if (!oshttp::xml_next(x, resp.body_len, &pos, "Size", sz,
                                  sizeof(sz))) {
                return kOsIoError;
            }
            const int v = fn(ctx, key, std::strtoull(sz, nullptr, 10));
            if (v != kOsOk) return v;
        }
        char trunc[8] = "";
        uint32_t tp = 0;
        (void)oshttp::xml_next(x, resp.body_len, &tp, "IsTruncated", trunc,
                               sizeof(trunc));
        if (std::strcmp(trunc, "true") != 0 &&
            std::strcmp(trunc, "True") != 0) { more = false; break; }
        tp = 0;
        if (!oshttp::xml_next(x, resp.body_len, &tp, "NextContinuationToken",
                              token, sizeof(token))) {
            return kOsIoError;
        }
        oshttp::xml_unescape(token);
    }
    if (more) return kOsIoError;   // page bound hit: never a short listing
    return kOsOk;
}

struct S3Fill {
    ObjectEntry* out;
    uint32_t     cap;
    uint32_t     n;
};

int s3_fill(void* ctx, const char* key, uint64_t size) noexcept {
    S3Fill* f = static_cast<S3Fill*>(ctx);
    assert(f != nullptr && key != nullptr);
    if (f->n == f->cap) return kOsBufferSmall;
    std::memcpy(f->out[f->n].key, key, std::strlen(key) + 1u);
    f->out[f->n].size = size;
    ++f->n;
    return kOsOk;
}

int s3_list(void* impl, const char* prefix, ObjectEntry* out, uint32_t cap,
            uint32_t* out_n) noexcept {
    assert(impl != nullptr);
    assert(out != nullptr && out_n != nullptr);
    S3Fill f{out, cap, 0u};
    const int rc = s3_walk(static_cast<S3ObjectStore*>(impl), prefix, &s3_fill, &f);
    *out_n = f.n;
    assert(f.n <= cap);
    return rc;
}

int s3_list_visit(void* impl, const char* prefix, ObjectVisitFn fn, void* ctx) noexcept {
    assert(impl != nullptr && fn != nullptr);
    return s3_walk(static_cast<S3ObjectStore*>(impl), prefix, fn, ctx);
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
    s3_get_range, s3_head_versioned, s3_get_range_if_match, s3_put_if_absent_versioned,
    s3_list_visit,
};

constexpr uint32_t kUploadResponseBytes = 64u * 1024u;
constexpr uint32_t kPartXmlBytes = 6u * (kS3PartEtagCap - 1u) + 90u;
struct PartReceipt { char etag[kS3PartEtagCap]; };
struct XmlCursor { const char* p; uint32_t size; uint32_t at; };

bool xml_space(char c) noexcept { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }
void xml_skip_space(XmlCursor* x) noexcept {
    assert(x != nullptr);
    assert(x->at <= x->size);
    while (x->at < x->size && xml_space(x->p[x->at])) ++x->at;
}
bool xml_take(XmlCursor* x, std::string_view text) noexcept {
    assert(x != nullptr && x->at <= x->size);
    assert(text.size() <= UINT32_MAX);
    if (text.size() > x->size - x->at ||
        std::memcmp(x->p + x->at, text.data(), text.size()) != 0) return false;
    x->at += static_cast<uint32_t>(text.size());
    return true;
}

bool xml_character(uint32_t c) noexcept {
    return c == 9 || c == 10 || c == 13 || (c >= 0x20 && c <= 0xD7FF) ||
        (c >= 0xE000 && c <= 0xFFFD) || (c >= 0x10000 && c <= 0x10FFFF);
}

bool xml_codepoint(std::string_view input, size_t* at, uint32_t* out) noexcept {
    assert(at != nullptr && *at < input.size());
    assert(out != nullptr);
    const auto first = static_cast<uint8_t>(input[(*at)++]);
    if (first < 128) { *out = first; return xml_character(first); }
    uint32_t count = 0, value = 0, minimum = 0;
    if (first >= 0xC2 && first <= 0xDF) count = 1, value = first & 31u, minimum = 0x80;
    else if (first >= 0xE0 && first <= 0xEF) count = 2, value = first & 15u, minimum = 0x800;
    else if (first >= 0xF0 && first <= 0xF4) count = 3, value = first & 7u, minimum = 0x10000;
    else return false;
    if (count > input.size() - *at) return false;
    for (uint32_t i = 0; i < count; ++i) {
        const auto next = static_cast<uint8_t>(input[(*at)++]);
        if ((next & 0xC0) != 0x80) return false;
        value = (value << 6u) | (next & 63u);
    }
    *out = value;
    return value >= minimum && xml_character(value);
}

bool xml_entity(std::string_view entity, uint32_t* out) noexcept {
    assert(out != nullptr);
    assert(entity.size() <= 12);
    if (entity == "amp") *out = '&';
    else if (entity == "lt") *out = '<';
    else if (entity == "gt") *out = '>';
    else if (entity == "quot") *out = '"';
    else if (entity == "apos") *out = '\'';
    else {
        if (entity.empty() || entity[0] != '#') return false;
        size_t at = 1;
        const uint32_t base = at < entity.size() && entity[at] == 'x' ? (++at, 16u) : 10u;
        if (at == entity.size()) return false;
        uint32_t value = 0;
        for (; at < entity.size(); ++at) {
            const char c = entity[at];
            const uint32_t digit = c >= '0' && c <= '9' ? static_cast<uint32_t>(c - '0') :
                c >= 'a' && c <= 'f' ? static_cast<uint32_t>(c - 'a' + 10) :
                c >= 'A' && c <= 'F' ? static_cast<uint32_t>(c - 'A' + 10) : 16u;
            if (digit >= base || value > (0x10FFFFu - digit) / base) return false;
            value = value * base + digit;
        }
        *out = value;
    }
    return xml_character(*out);
}

bool xml_decode(std::string_view input, char* out, uint32_t cap) noexcept {
    assert(out != nullptr || cap == 0);
    assert(input.size() <= kUploadResponseBytes);
    uint32_t written = 0;
    if (input.find("]]>") != std::string_view::npos) return false;
    for (size_t at = 0; at < input.size();) {
        uint32_t code = 0;
        if (input[at] == '&') {
            const size_t end = input.find(';', at + 1u);
            if (end == std::string_view::npos || end - at > 13u ||
                !xml_entity(input.substr(at + 1u, end - at - 1u), &code)) return false;
            at = end + 1u;
        } else if (!xml_codepoint(input, &at, &code)) return false;
        char bytes[4];
        uint32_t n = 0;
        if (code < 0x80) bytes[n++] = static_cast<char>(code);
        else {
            const uint32_t tails = code < 0x800 ? 1u : code < 0x10000 ? 2u : 3u;
            bytes[0] = static_cast<char>((tails == 1 ? 0xC0u : tails == 2 ? 0xE0u : 0xF0u) |
                                         (code >> (6u * tails)));
            for (uint32_t i = 0; i < tails; ++i)
                bytes[i + 1u] = static_cast<char>(0x80u | ((code >> (6u * (tails - i - 1u))) & 63u));
            n = tails + 1u;
        }
        if (out != nullptr) {
            if (written >= cap || n >= cap - written) return false;
            std::memcpy(out + written, bytes, n);
        }
        written += n;
    }
    if (out != nullptr) out[written] = '\0';
    return true;
}

bool xml_quoted(XmlCursor* x, std::string_view* value) noexcept {
    assert(x != nullptr && value != nullptr);
    assert(x->at <= x->size);
    if (x->at == x->size) return false;
    const char quote = x->p[x->at++];
    if (quote != '\'' && quote != '"') return false;
    const uint32_t start = x->at;
    while (x->at < x->size && x->p[x->at] != quote && x->at - start < 128u) ++x->at;
    if (x->at == x->size || x->p[x->at] != quote) return false;
    *value = std::string_view(x->p + start, x->at - start);
    ++x->at;
    return true;
}

bool xml_declaration(XmlCursor* x) noexcept {
    assert(x != nullptr);
    assert(x->at <= x->size);
    uint32_t previous = 0;
    for (uint32_t attribute = 0; attribute < 4u; ++attribute) {
        if (xml_take(x, "?>")) return previous != 0;
        if (x->at == x->size || !xml_space(x->p[x->at])) return false;
        xml_skip_space(x);
        if (xml_take(x, "?>")) return previous != 0;
        uint32_t kind = xml_take(x, "version") ? 1u : xml_take(x, "encoding") ? 2u :
            xml_take(x, "standalone") ? 3u : 0u;
        if (kind == 0 || kind <= previous || (previous == 0 && kind != 1u)) return false;
        xml_skip_space(x);
        if (!xml_take(x, "=")) return false;
        xml_skip_space(x);
        std::string_view value;
        if (!xml_quoted(x, &value)) return false;
        if ((kind == 1u && value != "1.0") ||
            (kind == 2u && value != "UTF-8" && value != "utf-8") ||
            (kind == 3u && value != "yes" && value != "no")) return false;
        previous = kind;
    }
    return false;
}

bool xml_result_open(XmlCursor* x, const char* root) noexcept {
    assert(x != nullptr && root != nullptr);
    assert(x->at == 0);
    xml_skip_space(x);
    if (xml_take(x, "<?xml")) {
        if (!xml_declaration(x)) return false;
        xml_skip_space(x);
    }
    if (!xml_take(x, "<") || !xml_take(x, root)) return false;
    if (x->at < x->size && xml_space(x->p[x->at])) {
        xml_skip_space(x);
        if (!xml_take(x, ">")) {
            if (!xml_take(x, "xmlns")) return false;
            xml_skip_space(x);
            if (!xml_take(x, "=")) return false;
            xml_skip_space(x);
            std::string_view value;
            if (!xml_quoted(x, &value) || value != "http://s3.amazonaws.com/doc/2006-03-01/") return false;
            xml_skip_space(x);
            return xml_take(x, ">");
        }
        return true;
    }
    return xml_take(x, ">");
}

// Parse the documented S3 result shape, not a substring that might occur in an
// Error or a truncated document. Unknown flat fields are validated then ignored.
bool upload_result_xml(const net::HttpResponse& response, const char* root,
                        const S3ObjectStore* store, const char* key,
                        const char* value_name, char* value, uint32_t cap) noexcept {
    assert(store != nullptr && key != nullptr);
    assert(value != nullptr && cap > 0);
    if (response.body == nullptr || response.body_len > kUploadResponseBytes) return false;
    XmlCursor x{reinterpret_cast<const char*>(response.body), response.body_len, 0};
    if (!xml_result_open(&x, root)) return false;
    uint32_t seen = 0;
    for (uint32_t field = 0; field < 64u; ++field) {
        xml_skip_space(&x);
        if (xml_take(&x, "</")) {
            if (!xml_take(&x, root) || !xml_take(&x, ">")) return false;
            xml_skip_space(&x);
            return x.at == x.size && seen == 7u && value[0] != '\0';
        }
        if (!xml_take(&x, "<")) return false;
        const uint32_t start = x.at;
        while (x.at < x.size && x.p[x.at] != '>') {
            const char c = x.p[x.at++];
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') || c == '-' || c == '_')) return false;
        }
        const std::string_view name(x.p + start, x.at - start);
        if (name.empty() || name.size() > 63u || !xml_take(&x, ">")) return false;
        if (!((name[0] >= 'a' && name[0] <= 'z') ||
              (name[0] >= 'A' && name[0] <= 'Z') || name[0] == '_')) return false;
        const uint32_t text = x.at;
        while (x.at < x.size && x.p[x.at] != '<') ++x.at;
        const std::string_view content(x.p + text, x.at - text);
        if (!xml_take(&x, "</") || !xml_take(&x, name) || !xml_take(&x, ">")) return false;
        uint32_t bit = name == "Bucket" ? 1u : name == "Key" ? 2u : name == value_name ? 4u : 0u;
        if ((seen & bit) != 0) return false;
        seen |= bit;
        if (bit == 4u) {
            if (!xml_decode(content, value, cap)) return false;
        } else if (bit != 0) {
            char decoded[kOsMaxKey];
            if (!xml_decode(content, decoded, sizeof(decoded)) ||
                std::strcmp(decoded, bit == 1u ? store->bucket : key) != 0) return false;
        } else if (!xml_decode(content, nullptr, 0)) return false;
    }
    return false;
}

struct UploadWork {
    S3ObjectStore* store;
    const char* key;
    uint64_t length;
    const S3UploadSource* source;
    const S3UploadOptions* options;
    S3UploadResult* result;
    uint32_t count;
    PartReceipt* parts;
    char* xml;
    uint32_t xml_cap;
    Arena response_arena{ArenaConfig{kUploadResponseBytes, kUploadResponseBytes}};
};

bool upload_cancelled(const UploadWork& w) noexcept {
    assert(w.options != nullptr);
    assert(w.result != nullptr);
    return w.options->cancelled != nullptr && w.options->cancelled(w.options->cancel_context);
}

bool upload_query(const char* upload_id, uint32_t part, char* out, uint32_t cap) noexcept {
    assert(upload_id != nullptr && out != nullptr);
    assert(cap > 0);
    Buf q;
    oshttp::buf_init(&q, out, cap);
    if (part != 0) {
        char number[32];
        const int n = std::snprintf(number, sizeof(number), "partNumber=%u&", part);
        if (n <= 0 || static_cast<size_t>(n) >= sizeof(number)) return false;
        oshttp::buf_str(&q, number);
    }
    oshttp::buf_str(&q, "uploadId=");
    oshttp::buf_pct(&q, upload_id, false);
    return q.ok;
}

int upload_send(UploadWork& w, const char* method, const char* query,
                 const uint8_t* body, uint64_t size, bool conditional,
                 net::HttpResponse* response) noexcept {
    assert(response != nullptr);
    assert(size <= net::kHttpMaxBody);
    w.response_arena.reset();
    const int rc = s3_send(w.store, method, w.key, query, body, size, conditional,
                          &w.response_arena, response, nullptr, w.options->timeout_ms,
                          kUploadResponseBytes);
    w.result->http_status = static_cast<uint32_t>(response->status);
    return rc;
}

int abort_upload(S3ObjectStore* store, const char* key, const char* id,
                   uint16_t timeout_ms) noexcept {
    assert(store != nullptr && key != nullptr);
    assert(id != nullptr);
    char query[net::kHttpMaxUrl];
    if (!upload_query(id, 0, query, sizeof(query))) return kOsBadArg;
    Arena scratch{ArenaConfig{kUploadResponseBytes, kUploadResponseBytes}};
    scratch.set_growth_enabled(false);
    net::HttpResponse response{};
    const int rc = s3_send(store, "DELETE", key, query, nullptr, 0, false,
                           &scratch, &response, nullptr, timeout_ms, kUploadResponseBytes);
    if (rc != kOsOk) return rc;
    return response.status == 204 || response.status == 404 ? kOsOk : kOsIoError;
}

int fail_upload(UploadWork& w, int status) noexcept {
    assert(status != kOsOk);
    assert(!w.result->completion_attempted);
    if (w.result->upload_id[0] != '\0') {
        w.result->abort_status = abort_upload(w.store, w.key, w.result->upload_id,
                                               w.options->timeout_ms);
        if (w.result->abort_status != kOsOk) w.result->remote_state_uncertain = true;
    }
    w.result->status = status;
    return status;
}

int create_upload(UploadWork& w) noexcept {
    assert(w.count > 0 && w.count <= kS3MaxUploadParts);
    assert(w.result->upload_id[0] == '\0');
    w.result->phase = S3UploadPhase::Create;
    net::HttpResponse response{};
    const int rc = upload_send(w, "POST", "uploads=", nullptr, 0, false, &response);
    if (rc != kOsOk || response.status != 200 ||
        !upload_result_xml(response, "InitiateMultipartUploadResult", w.store, w.key,
                           "UploadId", w.result->upload_id, sizeof(w.result->upload_id))) {
        // An invalid/lost initiation response can leave an unidentified orphan.
        // Never use an unvalidated partial UploadId to authorize cleanup.
        w.result->upload_id[0] = '\0';
        w.result->remote_state_uncertain = true;
        w.result->status = rc == kOsOk ? kOsIoError : rc;
        return w.result->status;
    }
    return kOsOk;
}

int upload_parts(UploadWork& w) noexcept {
    assert(w.source->read_part != nullptr);
    assert(w.parts != nullptr && w.count <= kS3MaxUploadParts);
    for (uint32_t part = 0; part < w.count; ++part) {
        if (upload_cancelled(w)) return fail_upload(w, kOsCancelled);
        const uint64_t offset = static_cast<uint64_t>(part) * w.options->part_bytes;
        const auto size = static_cast<uint32_t>(std::min<uint64_t>(w.length - offset,
                                                                  w.options->part_bytes));
        const uint8_t* data = nullptr;
        w.result->phase = S3UploadPhase::Source;
        const int source_rc = w.source->read_part(w.source->context, offset, size, &data);
        if (source_rc != kOsOk || data == nullptr)
            return fail_upload(w, source_rc != kOsOk ? source_rc : kOsBadArg);
        if (upload_cancelled(w)) return fail_upload(w, kOsCancelled);
        char query[net::kHttpMaxUrl];
        if (!upload_query(w.result->upload_id, part + 1u, query, sizeof(query)))
            return fail_upload(w, kOsBadArg);
        w.result->phase = S3UploadPhase::Part;
        net::HttpResponse response{};
        const int rc = upload_send(w, "PUT", query, data, size, false, &response);
        if (rc != kOsOk || response.status != 200)
            return fail_upload(w, rc == kOsOk ? kOsIoError : rc);
        const char* etag = net::http_header_get(response.headers, response.header_count, "ETag");
        if (etag == nullptr || etag[0] == '\0' || std::strlen(etag) >= kS3PartEtagCap)
            return fail_upload(w, kOsIoError);
        const std::string_view tag(etag);
        for (size_t at = 0; at < tag.size();) {
            uint32_t code = 0;
            if (!xml_codepoint(tag, &at, &code)) return fail_upload(w, kOsIoError);
        }
        std::memcpy(w.parts[part].etag, etag, std::strlen(etag) + 1u);
        w.result->parts_uploaded = part + 1u;
        w.result->bytes_uploaded += size;
    }
    assert(w.result->bytes_uploaded == w.length);
    return kOsOk;
}

void xml_escape(Buf* out, const char* text) noexcept {
    assert(out != nullptr && text != nullptr);
    assert(std::strlen(text) < kS3PartEtagCap);
    for (uint32_t i = 0; text[i] != '\0' && i < kS3PartEtagCap; ++i) {
        switch (text[i]) {
        case '&': oshttp::buf_str(out, "&amp;"); break;
        case '<': oshttp::buf_str(out, "&lt;"); break;
        case '>': oshttp::buf_str(out, "&gt;"); break;
        case '"': oshttp::buf_str(out, "&quot;"); break;
        case '\'': oshttp::buf_str(out, "&apos;"); break;
        default: oshttp::buf_raw(out, text + i, 1u); break;
        }
    }
}

bool completion_xml(UploadWork& w, uint32_t* size) noexcept {
    assert(size != nullptr);
    assert(w.result->parts_uploaded == w.count);
    Buf xml;
    oshttp::buf_init(&xml, w.xml, w.xml_cap);
    oshttp::buf_str(&xml, "<CompleteMultipartUpload>");
    for (uint32_t i = 0; i < w.count; ++i) {
        char number[64];
        const int n = std::snprintf(number, sizeof(number), "<Part><PartNumber>%u</PartNumber><ETag>", i + 1u);
        if (n <= 0 || static_cast<size_t>(n) >= sizeof(number)) return false;
        oshttp::buf_str(&xml, number);
        xml_escape(&xml, w.parts[i].etag);
        oshttp::buf_str(&xml, "</ETag></Part>");
    }
    oshttp::buf_str(&xml, "</CompleteMultipartUpload>");
    *size = xml.n;
    return xml.ok;
}

int complete_upload(UploadWork& w) noexcept {
    assert(w.result->parts_uploaded == w.count);
    assert(!w.result->completion_attempted);
    w.result->phase = S3UploadPhase::Complete;
    if (upload_cancelled(w)) return fail_upload(w, kOsCancelled);
    char query[net::kHttpMaxUrl];
    uint32_t size = 0;
    if (!upload_query(w.result->upload_id, 0, query, sizeof(query)) || !completion_xml(w, &size))
        return fail_upload(w, kOsBufferSmall);
    w.result->completion_attempted = true;
    net::HttpResponse response{};
    const int rc = upload_send(w, "POST", query, reinterpret_cast<uint8_t*>(w.xml), size,
                               w.options->if_absent, &response);
    if (rc == kOsOk && w.options->if_absent && response.status == 412) {
        w.result->status = kOsExists;
        return kOsExists;
    }
    char etag[kS3PartEtagCap]{};
    if (rc != kOsOk || response.status != 200 ||
        !upload_result_xml(response, "CompleteMultipartUploadResult", w.store, w.key,
                           "ETag", etag, sizeof(etag))) {
        w.result->remote_state_uncertain = true;
        w.result->status = kOsWriteUncertain;
        return kOsWriteUncertain;
    }
    w.result->phase = S3UploadPhase::Done;
    w.result->status = kOsOk;
    return kOsOk;
}

bool upload_config_valid(const S3ObjectStore* store, const char* key) noexcept {
    assert(store != nullptr);
    assert(key != nullptr);
    if (std::memchr(key, '\0', kOsMaxKey) == nullptr || key[0] == '\0') return false;
    return std::memchr(store->bucket, '\0', sizeof(store->bucket)) != nullptr &&
        store->bucket[0] != '\0' && std::memchr(store->region, '\0', sizeof(store->region)) != nullptr &&
        store->region[0] != '\0' && std::memchr(store->endpoint, '\0', sizeof(store->endpoint)) != nullptr &&
        std::memchr(store->access_key, '\0', sizeof(store->access_key)) != nullptr &&
        std::memchr(store->secret_key, '\0', sizeof(store->secret_key)) != nullptr &&
        std::memchr(store->session_token, '\0', sizeof(store->session_token)) != nullptr;
}

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

int s3_upload_stream(S3ObjectStore* store, const char* key, uint64_t length,
                     const S3UploadSource* source, const S3UploadOptions* options,
                     Arena* arena, S3UploadResult* result) noexcept {
    assert(store != nullptr && key != nullptr && arena != nullptr);
    assert(source != nullptr && options != nullptr && result != nullptr);
    *result = S3UploadResult{};
    if (!upload_config_valid(store, key) || options->part_bytes < kS3MinPartBytes ||
        options->part_bytes > kS3MaxPartBytes || options->max_parts == 0 ||
        options->max_parts > kS3MaxUploadParts || options->timeout_ms == 0 ||
        options->timeout_ms > 60000 || (length != 0 && source->read_part == nullptr)) return kOsBadArg;
    const uint64_t count = length / options->part_bytes + (length % options->part_bytes != 0);
    if (count > options->max_parts) { result->status = kOsBufferSmall; return result->status; }
    if (length != 0 && options->if_absent && store->create_only != kS3CreateOnlyIfNoneMatch) {
        result->status = kOsNotImplemented;
        return result->status;
    }
    S3Target target;
    if (!s3_target(store, key, "uploads=", &target)) return kOsBadArg;
    UploadWork work{store, key, length, source, options, result,
                    static_cast<uint32_t>(count), nullptr, nullptr, 0};
    work.response_arena.set_growth_enabled(false);
    if (work.response_arena.allocate(kUploadResponseBytes, 1) == nullptr) {
        result->status = kOsBufferSmall;
        return result->status;
    }
    work.response_arena.reset();
    if (count != 0) {
        work.parts = arena->allocate_array<PartReceipt>(static_cast<size_t>(count));
        work.xml_cap = static_cast<uint32_t>(count) * kPartXmlBytes + 64u;
        work.xml = static_cast<char*>(arena->allocate(work.xml_cap, 1));
        if (work.parts == nullptr || work.xml == nullptr) {
            result->status = kOsBufferSmall;
            return result->status;
        }
    }
    if (upload_cancelled(work)) { result->status = kOsCancelled; return result->status; }
    if (length == 0) {
        result->phase = S3UploadPhase::Complete;
        result->completion_attempted = true;
        net::HttpResponse response{};
        const int rc = upload_send(work, "PUT", "", nullptr, 0, options->if_absent, &response);
        if (rc != kOsOk) {
            result->remote_state_uncertain = true;
            result->status = kOsWriteUncertain;
        } else if (options->if_absent && response.status == 412) result->status = kOsExists;
        else result->status = response.status / 100 == 2 ? kOsOk : kOsWriteUncertain;
        if (result->status == kOsWriteUncertain) result->remote_state_uncertain = true;
        if (result->status == kOsOk) result->phase = S3UploadPhase::Done;
        return result->status;
    }
    const int created = create_upload(work);
    if (created != kOsOk) return created;
    const int uploaded = upload_parts(work);
    return uploaded == kOsOk ? complete_upload(work) : uploaded;
}

int s3_abort_multipart(S3ObjectStore* store, const char* key,
                       const char* upload_id, uint16_t timeout_ms) noexcept {
    assert(store != nullptr && key != nullptr);
    assert(upload_id != nullptr);
    if (!upload_config_valid(store, key) || timeout_ms == 0 || timeout_ms > 60000 ||
        upload_id[0] == '\0' || std::memchr(upload_id, '\0', kS3UploadIdCap) == nullptr) return kOsBadArg;
    return abort_upload(store, key, upload_id, timeout_ms);
}

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
