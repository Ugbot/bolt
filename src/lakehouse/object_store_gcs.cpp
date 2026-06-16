// bolt/lakehouse/object_store_gcs.cpp — GCS ObjectStore.
//
// Auth machinery:
//   - JWT (header.payload) construction (Tiger-Style: fixed buffers).
//   - RS256 signing via OpenSSL EVP — gated on BOLT_WITH_TLS (bolt::net
//     already links OpenSSL 3.x; we don't introduce a new dep).
//   - Service-account JSON parsing (purpose-built scan for client_email +
//     private_key — no general-purpose JSON parser).
//
// HTTP transport (token exchange, XML API ops) is W1-stubbed.

#include "bolt/lakehouse/object_store/gcs.h"

#include <cassert>
#include <cstdio>
#include <cstring>

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
// ObjectStore vtable — transport stubbed.
// ---------------------------------------------------------------------------

namespace {

int gc_get(void* impl, const char* key, Arena* arena,
           const uint8_t** out_data, uint64_t* out_len) noexcept {
    assert(impl != nullptr);
    assert(out_data != nullptr && out_len != nullptr);
    (void)impl; (void)key; (void)arena;
    *out_data = nullptr;
    *out_len = 0;
    return kOsNotImplemented;
}
int gc_put(void* impl, const char* key, const uint8_t* data,
           uint64_t len) noexcept {
    assert(impl != nullptr);
    assert(key != nullptr);
    (void)impl; (void)key; (void)data; (void)len;
    return kOsNotImplemented;
}
int gc_list(void* impl, const char* prefix, ObjectEntry* out,
            uint32_t cap, uint32_t* out_n) noexcept {
    assert(impl != nullptr);
    assert(out_n != nullptr);
    (void)impl; (void)prefix; (void)out; (void)cap;
    *out_n = 0;
    return kOsNotImplemented;
}
int gc_delete(void* impl, const char* key) noexcept {
    assert(impl != nullptr);
    assert(key != nullptr);
    (void)impl; (void)key;
    return kOsNotImplemented;
}
int gc_head(void* impl, const char* key, ObjectMeta* out) noexcept {
    assert(impl != nullptr);
    assert(out != nullptr);
    (void)impl; (void)key;
    std::memset(out, 0, sizeof(*out));
    return kOsNotImplemented;
}

const ObjectStoreVT kGcsVT = {
    gc_get, gc_put, gc_list, gc_delete, gc_head,
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
