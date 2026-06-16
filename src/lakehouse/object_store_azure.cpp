// bolt/lakehouse/object_store_azure.cpp — Azure Blob ObjectStore.
//
// Signing is fully implemented per:
//   https://learn.microsoft.com/rest/api/storageservices/authorize-with-shared-key
// HTTP transport is W1-stubbed (kOsNotImplemented) — same as the AWS S3 impl.
// Once bolt::net grows an HTTP/1.1 client over TlsSocket, the vtable methods
// build the canonical request via build_string_to_sign + sign_shared_key,
// send it, and surface the response. The signing path is unit-tested.

#include "bolt/lakehouse/object_store/azure.h"

#include <cassert>
#include <cstdio>
#include <cstring>

#include "bolt/crypto/sigv4.h"   // sha256 / hmac_sha256

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
// ObjectStore vtable — transport stubbed; signing path proven via the
// build_string_to_sign + sign_shared_key entry points (unit-tested).
// ---------------------------------------------------------------------------

namespace {

int az_get(void* impl, const char* key, Arena* arena,
           const uint8_t** out_data, uint64_t* out_len) noexcept {
    assert(impl != nullptr);
    assert(out_data != nullptr && out_len != nullptr);
    (void)impl; (void)key; (void)arena;
    *out_data = nullptr;
    *out_len = 0;
    return kOsNotImplemented;
}

int az_put(void* impl, const char* key, const uint8_t* data,
           uint64_t len) noexcept {
    assert(impl != nullptr);
    assert(key != nullptr);
    (void)impl; (void)key; (void)data; (void)len;
    return kOsNotImplemented;
}

int az_list(void* impl, const char* prefix, ObjectEntry* out,
            uint32_t cap, uint32_t* out_n) noexcept {
    assert(impl != nullptr);
    assert(out_n != nullptr);
    (void)impl; (void)prefix; (void)out; (void)cap;
    *out_n = 0;
    return kOsNotImplemented;
}

int az_delete(void* impl, const char* key) noexcept {
    assert(impl != nullptr);
    assert(key != nullptr);
    (void)impl; (void)key;
    return kOsNotImplemented;
}

int az_head(void* impl, const char* key, ObjectMeta* out) noexcept {
    assert(impl != nullptr);
    assert(out != nullptr);
    (void)impl; (void)key;
    std::memset(out, 0, sizeof(*out));
    return kOsNotImplemented;
}

const ObjectStoreVT kAzureVT = {
    az_get, az_put, az_list, az_delete, az_head,
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
