// bolt/lakehouse/object_store/gcs.h — Google Cloud Storage object store.
//
// W7. Two auth modes:
//   - Hmac:           XML API, reuses bolt::crypto::sigv4 with service="storage".
//   - ServiceAccount: builds a JWT (RS256) over the service-account JSON
//                     credentials, exchanges it for an OAuth2 bearer token
//                     (`https://oauth2.googleapis.com/token`), caches the
//                     token with refresh 60s before expiry.
//
// RS256 signing uses OpenSSL EVP (bolt::net already links OpenSSL 3.x). When
// BOLT_WITH_TLS is OFF the JWT builder returns false at runtime — there's no
// other RSA implementation in-tree.
//
// HTTP transport is W1-stubbed (kOsNotImplemented). The deliverable here is
// the auth machinery — independently testable offline.
//
// Tiger Style: PODs, ≥2 asserts/fn, bounded everything, no heap.

#pragma once

#include <cassert>
#include <cstdint>

#include "bolt/bolt_arena.h"
#include "bolt/lakehouse/object_store.h"
#include "bolt/lakehouse/object_store/compat.h"   // for Secret

namespace bolt {
namespace lakehouse {
namespace gcs {

using s3_compat::Secret;

enum class AuthMode : uint8_t {
    Hmac           = 0,
    ServiceAccount = 1,
};

static constexpr uint32_t kGcsMaxBucket   = 96u;
static constexpr uint32_t kGcsMaxEndpoint = 256u;
// Service-account JSON is ~2.4KB (private_key alone is ~1.7KB PEM). 4KB cap.
static constexpr uint32_t kGcsMaxSaJson   = 4096u;

// A bigger secret type just for the SA-JSON.
struct ServiceAccountJson {
    char     bytes[kGcsMaxSaJson];
    uint32_t len;
};

struct Config {
    char                bucket[kGcsMaxBucket];
    AuthMode            auth_mode;
    uint8_t             _pad0[7];
    Secret              hmac_access_id;
    Secret              hmac_secret;
    ServiceAccountJson  service_account_json;
    char                endpoint_override[kGcsMaxEndpoint];  // "" → default
};

static constexpr uint32_t kGcsMaxClientEmail = 256u;
static constexpr uint32_t kGcsMaxPrivateKey  = 3072u;       // PEM
static constexpr uint32_t kGcsMaxBearer      = 2048u;

struct GcsStore {
    Config   cfg;
    char     client_email[kGcsMaxClientEmail];   // extracted from SA JSON
    char     private_key_pem[kGcsMaxPrivateKey]; // extracted from SA JSON
    uint32_t private_key_pem_len;
    char     bearer_token[kGcsMaxBearer];        // last fetched, "" = unset
    uint64_t bearer_expiry_unix;                 // 0 = no token cached
    uint8_t  _pad[4];
};

// Initialise + bind `out`. Returns false on overflow or malformed SA JSON
// (when auth_mode is ServiceAccount).
bool gcs_store_new(ObjectStore* out, Arena* arena, const Config* cfg) noexcept;

// ---------------------------------------------------------------------------
// JWT primitives — exposed for tests.
// ---------------------------------------------------------------------------

// Build a Google service-account JWT header + claims string (NUL-terminated)
// in the form "<base64url(header)>.<base64url(claims)>".
// `now_unix` lets tests fix the time; production passes time(nullptr).
// Returns bytes written (excluding NUL). Returns 0 on overflow.
uint32_t build_jwt_unsigned(const char* client_email,
                            const char* scope,
                            const char* audience,
                            uint64_t now_unix,
                            uint32_t ttl_seconds,
                            char* out, uint32_t cap) noexcept;

// Sign `unsigned_jwt` (header.payload) with RS256 using the PEM private key.
// Appends ".<base64url(signature)>" to the buffer. Returns bytes appended.
// Returns 0 if BOLT_WITH_TLS is OFF or key parsing fails.
uint32_t sign_jwt_rs256(const char* private_key_pem,
                        uint32_t private_key_pem_len,
                        const char* unsigned_jwt, uint32_t unsigned_len,
                        char* out_b64url_sig, uint32_t cap) noexcept;

// Extract `client_email` and `private_key` from a service-account JSON blob.
// Performs a tiny purpose-built JSON scan (no general parser — we just want
// two known keys). Writes NUL-terminated strings; returns false on missing
// keys, overflow, or malformed input.
bool parse_service_account_json(const char* json, uint32_t json_len,
                                char* out_email, uint32_t email_cap,
                                char* out_key_pem, uint32_t pem_cap,
                                uint32_t* out_pem_len) noexcept;

// base64url (RFC 4648 §5) — no padding, '-' and '_' alphabet.
uint32_t base64url_encode(const uint8_t* src, uint32_t n,
                          char* out, uint32_t cap) noexcept;

}  // namespace gcs
}  // namespace lakehouse
}  // namespace bolt
