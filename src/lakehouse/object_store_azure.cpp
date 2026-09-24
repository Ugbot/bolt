// bolt/lakehouse/object_store_azure.cpp — Azure Blob ObjectStore.
//
// Signing per:
//   https://learn.microsoft.com/rest/api/storageservices/authorize-with-shared-key
// Transport is bolt::net's HTTP/1.1 client; each request is signed with
// build_string_to_sign + sign_shared_key (or carries the SAS token).

#include "bolt/lakehouse/object_store/azure.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "bolt/crypto/sigv4.h"   // sha256 / hmac_sha256
#include "object_store_http.h"

namespace bolt {
namespace lakehouse {
namespace azure_blob {

// ---------------------------------------------------------------------------
// Base64 codec (standard alphabet, with padding).
// ---------------------------------------------------------------------------

static const char kB64Chars[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

uint32_t base64_encode(const uint8_t* src, uint32_t n,
                       char* out, uint32_t cap) noexcept {
    assert(out != nullptr);
    assert(src != nullptr || n == 0);
    const uint32_t need = ((n + 2u) / 3u) * 4u;
    if (need + 1u > cap) return 0;
    uint32_t o = 0;
    uint32_t i = 0;
    while (i + 3u <= n) {
        const uint32_t v =
            (static_cast<uint32_t>(src[i]) << 16) |
            (static_cast<uint32_t>(src[i + 1]) << 8) |
            (static_cast<uint32_t>(src[i + 2]));
        out[o++] = kB64Chars[(v >> 18) & 0x3F];
        out[o++] = kB64Chars[(v >> 12) & 0x3F];
        out[o++] = kB64Chars[(v >> 6) & 0x3F];
        out[o++] = kB64Chars[v & 0x3F];
        i += 3u;
    }
    if (i < n) {
        const uint32_t r = n - i;
        const uint32_t v = (static_cast<uint32_t>(src[i]) << 16) |
                           (r == 2u ? (static_cast<uint32_t>(src[i + 1]) << 8) : 0u);
        out[o++] = kB64Chars[(v >> 18) & 0x3F];
        out[o++] = kB64Chars[(v >> 12) & 0x3F];
        out[o++] = r == 2u ? kB64Chars[(v >> 6) & 0x3F] : '=';
        out[o++] = '=';
    }
    out[o] = '\0';
    assert(o == need);
    return o;
}

static int b64_val(char c) noexcept {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

uint32_t base64_decode(const char* src, uint8_t* out, uint32_t cap) noexcept {
    assert(src != nullptr);
    assert(out != nullptr);
    const size_t sl = std::strlen(src);
    // Trim padding.
    size_t real = sl;
    while (real > 0 && src[real - 1] == '=') --real;
    const uint32_t need = static_cast<uint32_t>((real * 6u) / 8u);
    if (need > cap) return 0;
    uint32_t o = 0;
    uint32_t acc = 0;
    int bits = 0;
    for (size_t i = 0; i < real; ++i) {
        const int v = b64_val(src[i]);
        if (v < 0) return 0;
        acc = (acc << 6) | static_cast<uint32_t>(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out[o++] = static_cast<uint8_t>((acc >> bits) & 0xFF);
        }
    }
    assert(o == need);
    return o;
}

// ---------------------------------------------------------------------------
// StringToSign builder.
// ---------------------------------------------------------------------------

namespace {

// Append C-string + '\n'. Returns false on overflow.
bool app_line(char* out, uint32_t cap, uint32_t* pos,
              const char* s) noexcept {
    assert(out != nullptr);
    assert(pos != nullptr);
    const char* str = s == nullptr ? "" : s;
    const size_t n = std::strlen(str);
    if (*pos + n + 1u > cap) return false;
    std::memcpy(out + *pos, str, n);
    *pos += static_cast<uint32_t>(n);
    out[(*pos)++] = '\n';
    return true;
}

// Append without trailing newline. For the final canonicalized resource block
// the spec ends without a trailing '\n'.
bool app_raw(char* out, uint32_t cap, uint32_t* pos,
             const char* s) noexcept {
    assert(out != nullptr);
    assert(pos != nullptr);
    const char* str = s == nullptr ? "" : s;
    const size_t n = std::strlen(str);
    if (*pos + n > cap) return false;
    std::memcpy(out + *pos, str, n);
    *pos += static_cast<uint32_t>(n);
    return true;
}

}  // namespace

uint32_t build_string_to_sign(const char* verb,
                              const SignHeaders* hdrs,
                              const char* account,
                              const char* resource_path,
                              const CanonicalizedHeader* x_ms_headers,
                              uint32_t x_ms_count,
                              const char* query_pairs,
                              char* out, uint32_t cap) noexcept {
    assert(verb != nullptr);
    assert(hdrs != nullptr);
    assert(account != nullptr);
    assert(resource_path != nullptr);
    assert(out != nullptr);
    if (cap == 0) return 0;
    uint32_t p = 0;
    if (!app_line(out, cap, &p, verb))                       return 0;
    if (!app_line(out, cap, &p, hdrs->content_encoding))     return 0;
    if (!app_line(out, cap, &p, hdrs->content_language))     return 0;
    if (!app_line(out, cap, &p, hdrs->content_length))       return 0;
    if (!app_line(out, cap, &p, hdrs->content_md5))          return 0;
    if (!app_line(out, cap, &p, hdrs->content_type))         return 0;
    if (!app_line(out, cap, &p, hdrs->date_header))          return 0;
    if (!app_line(out, cap, &p, hdrs->if_modified_since))    return 0;
    if (!app_line(out, cap, &p, hdrs->if_match))             return 0;
    if (!app_line(out, cap, &p, hdrs->if_none_match))        return 0;
    if (!app_line(out, cap, &p, hdrs->if_unmodified_since))  return 0;
    if (!app_line(out, cap, &p, hdrs->range))                return 0;
    // Canonicalized x-ms-* headers (caller passes them already sorted).
    for (uint32_t i = 0; i < x_ms_count; ++i) {
        assert(x_ms_headers != nullptr);
        if (!app_raw(out, cap, &p, x_ms_headers[i].lc_name)) return 0;
        if (p + 1u > cap) return 0;
        out[p++] = ':';
        if (!app_raw(out, cap, &p, x_ms_headers[i].value))   return 0;
        if (p + 1u > cap) return 0;
        out[p++] = '\n';
    }
    // CanonicalizedResource: "/<account><resource_path>"
    if (p + 1u > cap) return 0;
    out[p++] = '/';
    if (!app_raw(out, cap, &p, account))                     return 0;
    if (!app_raw(out, cap, &p, resource_path))               return 0;
    if (query_pairs != nullptr && query_pairs[0] != '\0') {
        if (p + 1u > cap) return 0;
        out[p++] = '\n';
        if (!app_raw(out, cap, &p, query_pairs))             return 0;
    }
    if (p + 1u > cap) return 0;
    out[p] = '\0';
    return p;
}

bool sign_shared_key(const uint8_t* decoded_key, uint32_t decoded_key_len,
                     const char* string_to_sign, uint32_t sts_len,
                     char* out_b64, uint32_t cap) noexcept {
    assert(decoded_key != nullptr);
    assert(string_to_sign != nullptr);
    assert(out_b64 != nullptr);
    uint8_t mac[32];
    crypto::hmac_sha256(decoded_key, decoded_key_len,
                        reinterpret_cast<const uint8_t*>(string_to_sign),
                        static_cast<uint64_t>(sts_len), mac);
    const uint32_t enc = base64_encode(mac, 32u, out_b64, cap);
    return enc != 0;
}

// ---------------------------------------------------------------------------
// ObjectStore vtable over bolt::net. put_if_absent is Put Blob with
// If-None-Match: * (409 BlobAlreadyExists / 412 when the blob exists).
// ---------------------------------------------------------------------------

namespace {

using oshttp::Buf;

constexpr const char* kAzVersion = "2021-08-06";

void rfc1123_now(char* out, uint32_t cap) noexcept {
    assert(out != nullptr);
    assert(cap >= 30u);
    static const char* kDay[] = {"Sun", "Mon", "Tue", "Wed", "Thu", "Fri",
                                 "Sat"};
    static const char* kMon[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
    std::tm t;
    oshttp::utc_now(&t);
    std::snprintf(out, cap, "%s, %02d %s %04d %02d:%02d:%02d GMT",
                  kDay[t.tm_wday % 7], t.tm_mday, kMon[t.tm_mon % 12],
                  t.tm_year + 1900, t.tm_hour, t.tm_min, t.tm_sec);
}

struct AzCall {
    const char*    verb;
    const char*    key;          // "" = the container itself
    const char*    query_url;    // encoded "a=b&c=d" or ""
    const char*    query_canon;  // sorted "a:b\nc:d" or ""
    const uint8_t* body;
    uint64_t       len;
    bool           if_none_match;
    bool           block_blob;   // x-ms-blob-type: BlockBlob
};

bool az_url(const AzureBlobStore* az, const AzCall* c, char* url,
            uint32_t url_cap, char* path, uint32_t path_cap) noexcept {
    assert(az != nullptr && c != nullptr);
    assert(url != nullptr && path != nullptr);
    char ob[kAzMaxEndpoint + kAzMaxAccount + 64u];
    char bb[kAzMaxEndpoint];
    Buf origin; oshttp::buf_init(&origin, ob, sizeof(ob));
    Buf base;   oshttp::buf_init(&base, bb, sizeof(bb));
    if (az->cfg.endpoint_override[0] == '\0') {
        oshttp::buf_str(&origin, "https://");
        oshttp::buf_str(&origin, az->cfg.account);
        oshttp::buf_str(&origin, ".blob.core.windows.net");
    } else if (!oshttp::split_endpoint(az->cfg.endpoint_override, &origin,
                                       &base)) {
        return false;
    }
    Buf p; oshttp::buf_init(&p, path, path_cap);
    oshttp::buf_str(&p, bb);
    oshttp::buf_str(&p, "/");
    oshttp::buf_str(&p, az->cfg.container);
    if (c->key[0] != '\0') {
        oshttp::buf_str(&p, "/");
        oshttp::buf_pct(&p, c->key, /*keep_slash=*/true);
    }
    Buf u; oshttp::buf_init(&u, url, url_cap);
    oshttp::buf_str(&u, ob);
    oshttp::buf_str(&u, path);
    const char* sas = az->cfg.sas_token[0] == '?' ? az->cfg.sas_token + 1
                                                  : az->cfg.sas_token;
    const bool has_q = c->query_url[0] != '\0';
    if (has_q || (az->use_sas && sas[0] != '\0')) oshttp::buf_str(&u, "?");
    oshttp::buf_str(&u, c->query_url);
    if (az->use_sas && sas[0] != '\0') {
        if (has_q) oshttp::buf_str(&u, "&");
        oshttp::buf_str(&u, sas);
    }
    return origin.ok && base.ok && p.ok && u.ok;
}

// Shared Key: sign the canonical request and add the Authorization header.
bool az_sign(const AzureBlobStore* az, const AzCall* c, const char* path,
             const char* date, const char* clen,
             net::HttpRequest* req) noexcept {
    assert(az != nullptr && c != nullptr && path != nullptr);
    assert(req != nullptr);
    SignHeaders h;
    std::memset(&h, 0, sizeof(h));
    h.content_length = clen;
    h.if_none_match = c->if_none_match ? "*" : "";
    CanonicalizedHeader xh[3];
    uint32_t nx = 0;
    if (c->block_blob) xh[nx++] = {"x-ms-blob-type", "BlockBlob"};
    xh[nx++] = {"x-ms-date", date};
    xh[nx++] = {"x-ms-version", kAzVersion};
    char sts[4096];
    const uint32_t sl = build_string_to_sign(c->verb, &h, az->cfg.account,
                                             path, xh, nx, c->query_canon,
                                             sts, sizeof(sts));
    if (sl == 0) return false;
    char sig[64];
    if (!sign_shared_key(az->decoded_key, az->decoded_key_len, sts, sl, sig,
                         sizeof(sig))) {
        return false;
    }
    char auth[kAzMaxAccount + 96u];
    std::snprintf(auth, sizeof(auth), "SharedKey %s:%s", az->cfg.account, sig);
    return net::http_request_add_header(req, "Authorization", auth);
}

int az_send(AzureBlobStore* az, const AzCall* c, Arena* arena,
            net::HttpResponse* resp) noexcept {
    assert(az != nullptr && c != nullptr);
    assert(arena != nullptr && resp != nullptr);
    if (c->len > net::kHttpMaxBody) return kOsBadArg;
    net::HttpRequest req;
    std::memset(&req, 0, sizeof(req));
    char path[net::kHttpMaxUrl];
    if (!az_url(az, c, req.url, sizeof(req.url), path, sizeof(path))) {
        return kOsBadArg;
    }
    std::strncpy(req.method, c->verb, sizeof(req.method) - 1u);
    req.body = c->body;
    req.body_len = static_cast<uint32_t>(c->len);
    char date[40];
    rfc1123_now(date, sizeof(date));
    char clen[24] = "";
    if (c->len > 0) {
        std::snprintf(clen, sizeof(clen), "%llu",
                      static_cast<unsigned long long>(c->len));
    }
    bool ok = net::http_request_add_header(&req, "x-ms-date", date) &&
              net::http_request_add_header(&req, "x-ms-version", kAzVersion);
    if (c->block_blob) {
        ok = ok && net::http_request_add_header(&req, "x-ms-blob-type",
                                                "BlockBlob");
    }
    if (c->if_none_match) {
        ok = ok && net::http_request_add_header(&req, "If-None-Match", "*");
    }
    if (c->len == 0 && std::strcmp(c->verb, "PUT") == 0) {
        ok = ok && net::http_request_add_header(&req, "Content-Length", "0");
    }
    if (ok && !az->use_sas) ok = az_sign(az, c, path, date, clen, &req);
    if (!ok) return kOsBadArg;
    return net::http_send(arena, &req, resp) == 0 ? kOsOk : kOsIoError;
}

AzCall az_call(const char* verb, const char* key) noexcept {
    assert(verb != nullptr);
    assert(key != nullptr);
    AzCall c;
    std::memset(&c, 0, sizeof(c));
    c.verb = verb;
    c.key = key;
    c.query_url = "";
    c.query_canon = "";
    return c;
}

int az_get(void* impl, const char* key, Arena* arena,
           const uint8_t** out_data, uint64_t* out_len) noexcept {
    assert(impl != nullptr && key != nullptr && arena != nullptr);
    assert(out_data != nullptr && out_len != nullptr);
    *out_data = nullptr;
    *out_len = 0;
    const AzCall c = az_call("GET", key);
    net::HttpResponse resp;
    const int rc = az_send(static_cast<AzureBlobStore*>(impl), &c, arena,
                           &resp);
    if (rc != kOsOk) return rc;
    const int st = oshttp::map_status(resp.status);
    if (st != kOsOk) return st;
    *out_data = resp.body;
    *out_len = resp.body_len;
    return kOsOk;
}

int az_put_impl(void* impl, const char* key, const uint8_t* data,
                uint64_t len, bool if_none_match) noexcept {
    assert(impl != nullptr && key != nullptr);
    assert(data != nullptr || len == 0);
    AzCall c = az_call("PUT", key);
    c.body = data;
    c.len = len;
    c.if_none_match = if_none_match;
    c.block_blob = true;
    Arena scratch;
    net::HttpResponse resp;
    const int rc = az_send(static_cast<AzureBlobStore*>(impl), &c, &scratch,
                           &resp);
    if (rc != kOsOk) return rc;
    if (if_none_match && (resp.status == 409 || resp.status == 412)) {
        return kOsExists;
    }
    return resp.status / 100 == 2 ? kOsOk : kOsIoError;
}

int az_put(void* impl, const char* key, const uint8_t* data,
           uint64_t len) noexcept {
    assert(impl != nullptr);
    assert(key != nullptr);
    return az_put_impl(impl, key, data, len, false);
}

int az_put_if_absent(void* impl, const char* key, const uint8_t* data,
                     uint64_t len) noexcept {
    assert(impl != nullptr);
    assert(key != nullptr);
    return az_put_impl(impl, key, data, len, true);
}

int az_list(void* impl, const char* prefix, ObjectEntry* out,
            uint32_t cap, uint32_t* out_n) noexcept {
    assert(impl != nullptr);
    assert(out != nullptr && out_n != nullptr);
    *out_n = 0;
    AzureBlobStore* az = static_cast<AzureBlobStore*>(impl);
    const char* pre = prefix != nullptr ? prefix : "";
    char marker[1024] = "";
    uint32_t count = 0;
    for (uint32_t page = 0; page < 4096u && count < cap; ++page) {
        char qu[4096];
        char qc[4096];
        Buf u; oshttp::buf_init(&u, qu, sizeof(qu));
        Buf k; oshttp::buf_init(&k, qc, sizeof(qc));
        oshttp::buf_str(&u, "comp=list");
        oshttp::buf_str(&k, "comp:list");
        if (marker[0] != '\0') {
            oshttp::buf_str(&u, "&marker=");
            oshttp::buf_pct(&u, marker, false);
            oshttp::buf_str(&k, "\nmarker:");
            oshttp::buf_str(&k, marker);
        }
        if (pre[0] != '\0') {
            oshttp::buf_str(&u, "&prefix=");
            oshttp::buf_pct(&u, pre, false);
            oshttp::buf_str(&k, "\nprefix:");
            oshttp::buf_str(&k, pre);
        }
        oshttp::buf_str(&u, "&restype=container");
        oshttp::buf_str(&k, "\nrestype:container");
        if (!u.ok || !k.ok) return kOsBadArg;
        AzCall c = az_call("GET", "");
        c.query_url = qu;
        c.query_canon = qc;
        Arena scratch;
        net::HttpResponse resp;
        const int rc = az_send(az, &c, &scratch, &resp);
        if (rc != kOsOk) return rc;
        if (resp.status / 100 != 2) return kOsIoError;
        const char* x = reinterpret_cast<const char*>(resp.body);
        uint32_t pos = 0;
        while (count < cap &&
               oshttp::xml_next(x, resp.body_len, &pos, "Name",
                                out[count].key, kOsMaxKey)) {
            oshttp::xml_unescape(out[count].key);
            char sz[32];
            if (!oshttp::xml_next(x, resp.body_len, &pos, "Content-Length",
                                  sz, sizeof(sz))) {
                return kOsIoError;
            }
            out[count].size = std::strtoull(sz, nullptr, 10);
            ++count;
        }
        uint32_t mp = 0;
        marker[0] = '\0';
        (void)oshttp::xml_next(x, resp.body_len, &mp, "NextMarker", marker,
                               sizeof(marker));
        if (marker[0] == '\0') break;
        oshttp::xml_unescape(marker);
    }
    *out_n = count;
    assert(count <= cap);
    return kOsOk;
}

int az_delete(void* impl, const char* key) noexcept {
    assert(impl != nullptr);
    assert(key != nullptr);
    const AzCall c = az_call("DELETE", key);
    Arena scratch;
    net::HttpResponse resp;
    const int rc = az_send(static_cast<AzureBlobStore*>(impl), &c, &scratch,
                           &resp);
    if (rc != kOsOk) return rc;
    return oshttp::map_status(resp.status);
}

int az_head(void* impl, const char* key, ObjectMeta* out) noexcept {
    assert(impl != nullptr);
    assert(out != nullptr && key != nullptr);
    std::memset(out, 0, sizeof(*out));
    const AzCall c = az_call("HEAD", key);
    Arena scratch;
    net::HttpResponse resp;
    const int rc = az_send(static_cast<AzureBlobStore*>(impl), &c, &scratch,
                           &resp);
    if (rc != kOsOk) return rc;
    const int st = oshttp::map_status(resp.status);
    if (st != kOsOk) return st;
    const char* cl = net::http_header_get(resp.headers, resp.header_count,
                                          "Content-Length");
    out->size = cl != nullptr ? std::strtoull(cl, nullptr, 10) : 0u;
    out->exists = true;
    return kOsOk;
}

const ObjectStoreVT kAzureVT = {
    az_get, az_put, az_list, az_delete, az_head, az_put_if_absent,
};

}  // namespace

bool azure_blob_store_new(ObjectStore* out, Arena* arena,
                          const Config* cfg) noexcept {
    assert(out != nullptr);
    assert(arena != nullptr);
    assert(cfg != nullptr);
    if (out == nullptr || arena == nullptr || cfg == nullptr) return false;
    AzureBlobStore* az = arena->allocate_array<AzureBlobStore>(1);
    if (az == nullptr) return false;
    std::memset(az, 0, sizeof(*az));
    az->cfg = *cfg;
    az->use_sas = cfg->sas_token[0] != '\0';
    if (!az->use_sas) {
        if (cfg->account_key.len == 0) return false;
        const uint32_t dec = base64_decode(cfg->account_key.bytes,
                                           az->decoded_key,
                                           sizeof(az->decoded_key));
        if (dec == 0) return false;
        az->decoded_key_len = dec;
    }
    out->vt = &kAzureVT;
    out->impl = az;
    return true;
}

}  // namespace azure_blob
}  // namespace lakehouse
}  // namespace bolt
