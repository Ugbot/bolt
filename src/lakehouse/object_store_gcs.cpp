// bolt/lakehouse/object_store_gcs.cpp — GCS ObjectStore.
//
// Auth machinery:
//   - JWT (header.payload) construction (Tiger-Style: fixed buffers).
//   - RS256 signing via OpenSSL EVP — gated on BOLT_WITH_TLS (bolt::net
//     already links OpenSSL 3.x; we don't introduce a new dep).
//   - Service-account JSON parsing (purpose-built scan for client_email +
//     private_key — no general-purpose JSON parser).
//
// Object ops go over the JSON API via bolt::net (see the vtable below);
// the OAuth token exchange is not wired yet.

#include "bolt/lakehouse/object_store/gcs.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "object_store_http.h"

#if defined(BOLT_WITH_TLS)
#  include <openssl/bio.h>
#  include <openssl/evp.h>
#  include <openssl/pem.h>
#endif

namespace bolt {
namespace lakehouse {
namespace gcs {

// ---------------------------------------------------------------------------
// base64url encoder (no padding).
// ---------------------------------------------------------------------------

static const char kB64UrlChars[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

uint32_t base64url_encode(const uint8_t* src, uint32_t n,
                          char* out, uint32_t cap) noexcept {
    assert(out != nullptr);
    assert(src != nullptr || n == 0);
    // unpadded length: ceil(n*4/3)
    const uint32_t need = (n * 4u + 2u) / 3u;
    if (need + 1u > cap) return 0;
    uint32_t o = 0;
    uint32_t i = 0;
    while (i + 3u <= n) {
        const uint32_t v =
            (static_cast<uint32_t>(src[i]) << 16) |
            (static_cast<uint32_t>(src[i + 1]) << 8) |
            (static_cast<uint32_t>(src[i + 2]));
        out[o++] = kB64UrlChars[(v >> 18) & 0x3F];
        out[o++] = kB64UrlChars[(v >> 12) & 0x3F];
        out[o++] = kB64UrlChars[(v >> 6) & 0x3F];
        out[o++] = kB64UrlChars[v & 0x3F];
        i += 3u;
    }
    if (i < n) {
        const uint32_t r = n - i;
        const uint32_t v = (static_cast<uint32_t>(src[i]) << 16) |
                           (r == 2u ? (static_cast<uint32_t>(src[i + 1]) << 8) : 0u);
        out[o++] = kB64UrlChars[(v >> 18) & 0x3F];
        out[o++] = kB64UrlChars[(v >> 12) & 0x3F];
        if (r == 2u) out[o++] = kB64UrlChars[(v >> 6) & 0x3F];
    }
    out[o] = '\0';
    assert(o == need);
    return o;
}

// ---------------------------------------------------------------------------
// JWT header + claims builder.
// ---------------------------------------------------------------------------

uint32_t build_jwt_unsigned(const char* client_email,
                            const char* scope,
                            const char* audience,
                            uint64_t now_unix,
                            uint32_t ttl_seconds,
                            char* out, uint32_t cap) noexcept {
    assert(client_email != nullptr);
    assert(scope != nullptr);
    assert(audience != nullptr);
    assert(out != nullptr);
    // RS256 header (JSON, fixed).
    const char* hdr_json = "{\"alg\":\"RS256\",\"typ\":\"JWT\"}";
    const uint32_t hdr_len = static_cast<uint32_t>(std::strlen(hdr_json));
    char hdr_b64[64];
    const uint32_t hdr_b64_len = base64url_encode(
        reinterpret_cast<const uint8_t*>(hdr_json), hdr_len,
        hdr_b64, sizeof(hdr_b64));
    if (hdr_b64_len == 0) return 0;

    // Claims JSON.
    char claims_json[1024];
    const uint64_t exp = now_unix + static_cast<uint64_t>(ttl_seconds);
    const int n = std::snprintf(claims_json, sizeof(claims_json),
        "{\"iss\":\"%s\",\"scope\":\"%s\",\"aud\":\"%s\","
        "\"iat\":%llu,\"exp\":%llu}",
        client_email, scope, audience,
        static_cast<unsigned long long>(now_unix),
        static_cast<unsigned long long>(exp));
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(claims_json)) return 0;
    char claims_b64[1500];
    const uint32_t claims_b64_len = base64url_encode(
        reinterpret_cast<const uint8_t*>(claims_json),
        static_cast<uint32_t>(n),
        claims_b64, sizeof(claims_b64));
    if (claims_b64_len == 0) return 0;

    const uint32_t need = hdr_b64_len + 1u + claims_b64_len;
    if (need + 1u > cap) return 0;
    std::memcpy(out, hdr_b64, hdr_b64_len);
    out[hdr_b64_len] = '.';
    std::memcpy(out + hdr_b64_len + 1u, claims_b64, claims_b64_len);
    out[need] = '\0';
    return need;
}

// ---------------------------------------------------------------------------
// RS256 signer (OpenSSL EVP). When TLS is OFF, returns 0.
// ---------------------------------------------------------------------------

uint32_t sign_jwt_rs256(const char* private_key_pem,
                        uint32_t private_key_pem_len,
                        const char* unsigned_jwt, uint32_t unsigned_len,
                        char* out_b64url_sig, uint32_t cap) noexcept {
    assert(private_key_pem != nullptr);
    assert(unsigned_jwt != nullptr);
    assert(out_b64url_sig != nullptr);
#if defined(BOLT_WITH_TLS)
    BIO* bio = BIO_new_mem_buf(private_key_pem,
                               static_cast<int>(private_key_pem_len));
    if (bio == nullptr) return 0;
    EVP_PKEY* pkey = PEM_read_bio_PrivateKey(bio, nullptr, nullptr, nullptr);
    BIO_free(bio);
    if (pkey == nullptr) return 0;

    EVP_MD_CTX* mdctx = EVP_MD_CTX_new();
    if (mdctx == nullptr) { EVP_PKEY_free(pkey); return 0; }
    uint32_t result = 0;
    do {
        if (EVP_DigestSignInit(mdctx, nullptr, EVP_sha256(), nullptr, pkey)
                != 1) break;
        if (EVP_DigestSignUpdate(mdctx, unsigned_jwt,
                                 static_cast<size_t>(unsigned_len)) != 1) break;
        size_t sig_len = 0;
        if (EVP_DigestSignFinal(mdctx, nullptr, &sig_len) != 1) break;
        if (sig_len == 0 || sig_len > 1024u) break;
        uint8_t sig[1024];
        if (EVP_DigestSignFinal(mdctx, sig, &sig_len) != 1) break;
        result = base64url_encode(sig, static_cast<uint32_t>(sig_len),
                                  out_b64url_sig, cap);
    } while (false);
    EVP_MD_CTX_free(mdctx);
    EVP_PKEY_free(pkey);
    return result;
#else
    (void)private_key_pem_len;
    (void)unsigned_len;
    (void)cap;
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// Tiny purpose-built JSON scanner for service-account credentials.
// Looks for "client_email" : "..." and "private_key" : "...". Unescapes only
// "\n" → '\n' and "\\" → '\\' inside the private_key value (the two escapes
// google emits in SA JSON). Anything else is left as-is.
// ---------------------------------------------------------------------------

namespace {

// Skip whitespace.
void skip_ws(const char* s, uint32_t* i, uint32_t n) noexcept {
    assert(s != nullptr && i != nullptr);
    while (*i < n && (s[*i] == ' ' || s[*i] == '\t' || s[*i] == '\r' ||
                      s[*i] == '\n')) ++(*i);
}

// Read a JSON string starting at s[*i] (which must be '"'). On success
// advance past the closing '"' and write the contents (with \n / \\ unescape)
// into out[0..cap), NUL-terminated. Returns false on overflow or malformed.
bool read_string(const char* s, uint32_t* i, uint32_t n,
                 char* out, uint32_t cap, uint32_t* out_len) noexcept {
    assert(s != nullptr && i != nullptr && out != nullptr);
    if (*i >= n || s[*i] != '"') return false;
    ++(*i);
    uint32_t o = 0;
    while (*i < n) {
        const char c = s[*i];
        if (c == '"') {
            if (o + 1u > cap) return false;
            out[o] = '\0';
            if (out_len != nullptr) *out_len = o;
            ++(*i);
            return true;
        }
        if (c == '\\') {
            if (*i + 1u >= n) return false;
            const char esc = s[*i + 1];
            char emit;
            switch (esc) {
                case 'n':  emit = '\n'; break;
                case 't':  emit = '\t'; break;
                case 'r':  emit = '\r'; break;
                case '"':  emit = '"';  break;
                case '\\': emit = '\\'; break;
                case '/':  emit = '/';  break;
                default:   return false;
            }
            if (o + 1u >= cap) return false;
            out[o++] = emit;
            *i += 2u;
            continue;
        }
        if (o + 1u >= cap) return false;
        out[o++] = c;
        ++(*i);
    }
    return false;
}

bool match_key(const char* s, uint32_t* i, uint32_t n,
               const char* key) noexcept {
    assert(s != nullptr && i != nullptr && key != nullptr);
    skip_ws(s, i, n);
    if (*i >= n || s[*i] != '"') return false;
    const size_t kl = std::strlen(key);
    if (*i + 1u + kl + 1u > n) return false;
    if (std::memcmp(s + *i + 1u, key, kl) != 0) return false;
    if (s[*i + 1u + kl] != '"') return false;
    *i += static_cast<uint32_t>(1u + kl + 1u);
    skip_ws(s, i, n);
    if (*i >= n || s[*i] != ':') return false;
    ++(*i);
    skip_ws(s, i, n);
    return true;
}

}  // namespace

bool parse_service_account_json(const char* json, uint32_t json_len,
                                char* out_email, uint32_t email_cap,
                                char* out_key_pem, uint32_t pem_cap,
                                uint32_t* out_pem_len) noexcept {
    assert(json != nullptr);
    assert(out_email != nullptr);
    assert(out_key_pem != nullptr);
    out_email[0] = '\0';
    out_key_pem[0] = '\0';
    if (out_pem_len != nullptr) *out_pem_len = 0;
    bool got_email = false;
    bool got_key   = false;
    // Linear scan: at each position try to match either known key.
    uint32_t i = 0;
    while (i < json_len && !(got_email && got_key)) {
        // Look for a '"' that starts a key.
        if (json[i] != '"') { ++i; continue; }
        const uint32_t save = i;
        if (!got_email && match_key(json, &i, json_len, "client_email")) {
            uint32_t len = 0;
            if (!read_string(json, &i, json_len, out_email, email_cap, &len))
                return false;
            got_email = true;
            continue;
        }
        i = save;
        if (!got_key && match_key(json, &i, json_len, "private_key")) {
            uint32_t len = 0;
            if (!read_string(json, &i, json_len, out_key_pem, pem_cap, &len))
                return false;
            if (out_pem_len != nullptr) *out_pem_len = len;
            got_key = true;
            continue;
        }
        i = save + 1u;
    }
    return got_email && got_key;
}

// ---------------------------------------------------------------------------
// ObjectStore vtable over the GCS JSON API. put_if_absent is a media upload
// with ifGenerationMatch=0 (412 when the object exists). Requests carry
// `bearer_token` when set; with no credentials configured they go anonymous
// (emulators such as fake-gcs-server). Minting a token from a service
// account, and HMAC auth, are not wired: those report kOsNotImplemented.
// ---------------------------------------------------------------------------

namespace {

using oshttp::Buf;

bool gc_auth(const GcsStore* g, net::HttpRequest* req, bool* ok) noexcept {
    assert(g != nullptr && req != nullptr);
    assert(ok != nullptr);
    *ok = true;
    if (g->bearer_token[0] != '\0') {
        char h[kGcsMaxBearer + 16u];
        std::snprintf(h, sizeof(h), "Bearer %s", g->bearer_token);
        *ok = net::http_request_add_header(req, "Authorization", h);
        return true;
    }
    const bool anonymous = g->cfg.auth_mode == AuthMode::Hmac &&
                           g->cfg.hmac_access_id.len == 0;
    return anonymous;
}

// `api` is "" (metadata) or "/upload"; `obj` is null for bucket-level calls.
int gc_send(GcsStore* g, const char* verb, const char* api, const char* obj,
            const char* query, const uint8_t* body, uint64_t len,
            Arena* arena, net::HttpResponse* resp) noexcept {
    assert(g != nullptr && verb != nullptr && api != nullptr);
    assert(arena != nullptr && resp != nullptr);
    if (len > net::kHttpMaxBody) return kOsBadArg;
    net::HttpRequest req;
    std::memset(&req, 0, sizeof(req));
    bool hdr_ok = true;
    if (!gc_auth(g, &req, &hdr_ok)) return kOsNotImplemented;
    Buf u; oshttp::buf_init(&u, req.url, sizeof(req.url));
    oshttp::buf_str(&u, g->cfg.endpoint_override[0] != '\0'
                            ? g->cfg.endpoint_override
                            : "https://storage.googleapis.com");
    while (u.n > 0 && u.p[u.n - 1] == '/') u.p[--u.n] = '\0';
    oshttp::buf_str(&u, api);
    oshttp::buf_str(&u, "/storage/v1/b/");
    oshttp::buf_pct(&u, g->cfg.bucket, false);
    oshttp::buf_str(&u, "/o");
    if (obj != nullptr) {
        oshttp::buf_str(&u, "/");
        oshttp::buf_pct(&u, obj, /*keep_slash=*/false);
    }
    if (query != nullptr && query[0] != '\0') {
        oshttp::buf_str(&u, "?");
        oshttp::buf_str(&u, query);
    }
    if (!u.ok || !hdr_ok) return kOsBadArg;
    std::strncpy(req.method, verb, sizeof(req.method) - 1u);
    req.body = body;
    req.body_len = static_cast<uint32_t>(len);
    if (std::strcmp(verb, "POST") == 0) {
        bool ok = net::http_request_add_header(
            &req, "Content-Type", "application/octet-stream");
        if (len == 0) {
            ok = ok && net::http_request_add_header(&req, "Content-Length",
                                                    "0");
        }
        if (!ok) return kOsBadArg;
    }
    return net::http_send(arena, &req, resp) == 0 ? kOsOk : kOsIoError;
}

int gc_get(void* impl, const char* key, Arena* arena,
           const uint8_t** out_data, uint64_t* out_len) noexcept {
    assert(impl != nullptr && key != nullptr && arena != nullptr);
    assert(out_data != nullptr && out_len != nullptr);
    *out_data = nullptr;
    *out_len = 0;
    net::HttpResponse resp;
    const int rc = gc_send(static_cast<GcsStore*>(impl), "GET", "", key,
                           "alt=media", nullptr, 0, arena, &resp);
    if (rc != kOsOk) return rc;
    const int st = oshttp::map_status(resp.status);
    if (st != kOsOk) return st;
    *out_data = resp.body;
    *out_len = resp.body_len;
    return kOsOk;
}

int gc_upload(void* impl, const char* key, const uint8_t* data,
              uint64_t len, bool if_absent) noexcept {
    assert(impl != nullptr && key != nullptr);
    assert(data != nullptr || len == 0);
    char qb[kOsMaxKey * 3u + 64u];
    Buf q; oshttp::buf_init(&q, qb, sizeof(qb));
    oshttp::buf_str(&q, "uploadType=media&name=");
    oshttp::buf_pct(&q, key, false);
    if (if_absent) oshttp::buf_str(&q, "&ifGenerationMatch=0");
    if (!q.ok) return kOsBadArg;
    Arena scratch;
    net::HttpResponse resp;
    const int rc = gc_send(static_cast<GcsStore*>(impl), "POST", "/upload",
                           nullptr, qb, data, len, &scratch, &resp);
    if (rc != kOsOk) return rc;
    if (if_absent && resp.status == 412) return kOsExists;
    return resp.status / 100 == 2 ? kOsOk : kOsIoError;
}

int gc_put(void* impl, const char* key, const uint8_t* data,
           uint64_t len) noexcept {
    assert(impl != nullptr);
    assert(key != nullptr);
    return gc_upload(impl, key, data, len, false);
}

int gc_put_if_absent(void* impl, const char* key, const uint8_t* data,
                     uint64_t len) noexcept {
    assert(impl != nullptr);
    assert(key != nullptr);
    return gc_upload(impl, key, data, len, true);
}

// Next "field": "<string>" value at or after *pos (simple escapes only).
bool json_field(const char* s, uint32_t len, uint32_t* pos,
                const char* field, char* out, uint32_t cap) noexcept {
    assert(s != nullptr && pos != nullptr && field != nullptr);
    assert(out != nullptr && cap > 0);
    char pat[64];
    const int pl = std::snprintf(pat, sizeof(pat), "\"%s\"", field);
    if (pl <= 0) return false;
    for (uint32_t i = *pos; i + static_cast<uint32_t>(pl) <= len; ++i) {
        if (std::memcmp(s + i, pat, static_cast<size_t>(pl)) != 0) continue;
        uint32_t j = i + static_cast<uint32_t>(pl);
        while (j < len && (s[j] == ' ' || s[j] == ':' || s[j] == '\n' ||
                           s[j] == '\t' || s[j] == '\r')) ++j;
        if (j >= len || s[j] != '"') continue;
        uint32_t o = 0;
        for (++j; j < len && s[j] != '"'; ++j) {
            char c = s[j];
            if (c == '\\') {
                if (j + 1u >= len) return false;
                c = s[++j];
                if (c == 'u') return false;
                if (c == 'n') c = '\n';
                else if (c == 't') c = '\t';
            }
            if (o + 1u >= cap) return false;
            out[o++] = c;
        }
        if (j >= len) return false;
        out[o] = '\0';
        *pos = j + 1u;
        return true;
    }
    return false;
}

int gc_list(void* impl, const char* prefix, ObjectEntry* out,
            uint32_t cap, uint32_t* out_n) noexcept {
    assert(impl != nullptr);
    assert(out != nullptr && out_n != nullptr);
    *out_n = 0;
    GcsStore* g = static_cast<GcsStore*>(impl);
    char token[1024] = "";
    uint32_t count = 0;
    for (uint32_t page = 0; page < 4096u && count < cap; ++page) {
        char qb[4096];
        Buf q; oshttp::buf_init(&q, qb, sizeof(qb));
        oshttp::buf_str(&q, "fields=items(name,size),nextPageToken&prefix=");
        oshttp::buf_pct(&q, prefix != nullptr ? prefix : "", false);
        if (token[0] != '\0') {
            oshttp::buf_str(&q, "&pageToken=");
            oshttp::buf_pct(&q, token, false);
        }
        if (!q.ok) return kOsBadArg;
        Arena scratch;
        net::HttpResponse resp;
        const int rc = gc_send(g, "GET", "", nullptr, qb, nullptr, 0,
                               &scratch, &resp);
        if (rc != kOsOk) return rc;
        if (resp.status / 100 != 2) return kOsIoError;
        const char* x = reinterpret_cast<const char*>(resp.body);
        uint32_t pos = 0;
        while (count < cap && json_field(x, resp.body_len, &pos, "name",
                                         out[count].key, kOsMaxKey)) {
            char sz[32];
            if (!json_field(x, resp.body_len, &pos, "size", sz, sizeof(sz))) {
                return kOsIoError;
            }
            out[count].size = std::strtoull(sz, nullptr, 10);
            ++count;
        }
        uint32_t tp = 0;
        token[0] = '\0';
        (void)json_field(x, resp.body_len, &tp, "nextPageToken", token,
                         sizeof(token));
        if (token[0] == '\0') break;
    }
    *out_n = count;
    assert(count <= cap);
    return kOsOk;
}

int gc_delete(void* impl, const char* key) noexcept {
    assert(impl != nullptr);
    assert(key != nullptr);
    Arena scratch;
    net::HttpResponse resp;
    const int rc = gc_send(static_cast<GcsStore*>(impl), "DELETE", "", key,
                           "", nullptr, 0, &scratch, &resp);
    if (rc != kOsOk) return rc;
    return oshttp::map_status(resp.status);
}

int gc_head(void* impl, const char* key, ObjectMeta* out) noexcept {
    assert(impl != nullptr);
    assert(out != nullptr && key != nullptr);
    std::memset(out, 0, sizeof(*out));
    Arena scratch;
    net::HttpResponse resp;
    const int rc = gc_send(static_cast<GcsStore*>(impl), "GET", "", key,
                           "fields=size", nullptr, 0, &scratch, &resp);
    if (rc != kOsOk) return rc;
    const int st = oshttp::map_status(resp.status);
    if (st != kOsOk) return st;
    char sz[32] = "0";
    uint32_t pos = 0;
    (void)json_field(reinterpret_cast<const char*>(resp.body), resp.body_len,
                     &pos, "size", sz, sizeof(sz));
    out->size = std::strtoull(sz, nullptr, 10);
    out->exists = true;
    return kOsOk;
}

const ObjectStoreVT kGcsVT = {
    gc_get, gc_put, gc_list, gc_delete, gc_head, gc_put_if_absent,
};

}  // namespace

bool gcs_store_new(ObjectStore* out, Arena* arena, const Config* cfg) noexcept {
    assert(out != nullptr);
    assert(arena != nullptr);
    assert(cfg != nullptr);
    if (out == nullptr || arena == nullptr || cfg == nullptr) return false;
    GcsStore* g = arena->allocate_array<GcsStore>(1);
    if (g == nullptr) return false;
    std::memset(g, 0, sizeof(*g));
    g->cfg = *cfg;
    if (cfg->auth_mode == AuthMode::ServiceAccount) {
        if (cfg->service_account_json.len == 0) return false;
        if (!parse_service_account_json(cfg->service_account_json.bytes,
                                        cfg->service_account_json.len,
                                        g->client_email, kGcsMaxClientEmail,
                                        g->private_key_pem, kGcsMaxPrivateKey,
                                        &g->private_key_pem_len)) {
            return false;
        }
    }
    out->vt = &kGcsVT;
    out->impl = g;
    return true;
}

}  // namespace gcs
}  // namespace lakehouse
}  // namespace bolt
