// bolt/crypto/sigv4.h — AWS Signature Version 4 request signing.
//
// Format-agnostic: caller supplies the request parts, we produce the
// `Authorization` header value + the `x-amz-date` value. The signing key
// derivation and HMAC-SHA256 chain are self-contained — a software SHA-256 +
// HMAC live in the .cpp so bolt stays zero-dependency (no OpenSSL link).
//
// Reference: docs.aws.amazon.com/general/latest/gr/sigv4-create-canonical-request.html
//
// The canonical request we build is the common case used by S3 / Glue / STS:
//   METHOD \n CANONICAL_URI \n CANONICAL_QUERY \n
//   host:HOST \n x-amz-content-sha256:HASH \n x-amz-date:DATE \n
//   [x-amz-security-token:TOKEN \n] \n
//   SIGNED_HEADERS \n PAYLOAD_SHA256
// with SIGNED_HEADERS = "host;x-amz-content-sha256;x-amz-date[;x-amz-security-token]".
//
// Tiger Style: PODs, caller-owned output buffers, ≥2 asserts/fn, no heap,
// no exceptions.

#pragma once

#include <cstddef>
#include <cstdint>

namespace bolt {
namespace crypto {

// Fixed caps for the request fields (bounded everything).
static constexpr uint32_t kSigV4MaxHost   = 256u;
static constexpr uint32_t kSigV4MaxPath    = 2048u;
static constexpr uint32_t kSigV4MaxQuery   = 4096u;
static constexpr uint32_t kSigV4MaxService = 32u;
static constexpr uint32_t kSigV4MaxRegion  = 32u;
static constexpr uint32_t kSigV4MaxKey     = 256u;
static constexpr uint32_t kSigV4MaxSecret  = 256u;
static constexpr uint32_t kSigV4MaxToken   = 2048u;
// Authorization header value upper bound (signed headers + sig hex + creds).
static constexpr uint32_t kSigV4MaxAuth    = 1024u;

// One request to sign. All char* are NUL-terminated C strings the caller owns.
// `payload_sha256` is the lowercase-hex SHA-256 of the body (64 hex chars +
// NUL); for an empty body pass the well-known empty hash. `session_token` may
// be nullptr (no STS session). `amz_date` is "YYYYMMDDTHHMMSSZ"; the matching
// "YYYYMMDD" short date is derived from it.
struct SigV4Request {
    const char* method;          // "GET" / "PUT" / "POST" ...
    const char* host;            // "example.amazonaws.com"
    const char* path;            // canonical URI, e.g. "/"
    const char* query_string;    // canonical query (may be "" or nullptr)
    const char* service;         // "service" / "s3" / "glue"
    const char* region;          // "us-east-1"
    const char* amz_date;        // "YYYYMMDDTHHMMSSZ"
    const char* payload_sha256;  // 64 lowercase hex chars
    const char* access_key;      // "AKID..."
    const char* secret_key;      // signing secret
    const char* session_token;   // STS token or nullptr
};

// The signed outputs, written into caller buffers. `auth_header` is the full
// `Authorization` header VALUE (no "Authorization:" prefix). `amz_date_out`
// echoes the date the caller must also send as the `x-amz-date` header.
struct SigV4Result {
    char* auth_header;       // out: capacity auth_cap
    uint32_t auth_cap;
    char* amz_date_out;      // out: capacity >= 17
    uint32_t amz_date_cap;
};

// Lowercase-hex SHA-256 of `data` into `out_hex` (>= 65 bytes incl NUL).
void sha256_hex(const uint8_t* data, uint64_t len, char out_hex[65]) noexcept;

// The canonical empty-payload hash, copied into out_hex (>= 65 bytes).
void sha256_empty_hex(char out_hex[65]) noexcept;

// Sign `req`. Writes the Authorization header value and echoes the date.
// Returns false on a null required field or an output-buffer overflow.
bool sigv4_sign(const SigV4Request* req, SigV4Result* out) noexcept;

// ---------------------------------------------------------------------------
// Low-level primitives (exposed for tests + the object store). Self-contained.
// ---------------------------------------------------------------------------

// SHA-256 of `data` → 32-byte digest.
void sha256(const uint8_t* data, uint64_t len, uint8_t out[32]) noexcept;

// HMAC-SHA256(key, msg) → 32-byte mac.
void hmac_sha256(const uint8_t* key, uint32_t key_len,
                 const uint8_t* msg, uint64_t msg_len,
                 uint8_t out[32]) noexcept;

}  // namespace crypto
}  // namespace bolt
