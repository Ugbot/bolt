// bolt/lakehouse/object_store/azure.h — Azure Blob Storage object store.
//
// W7. Auth: Shared Key (HMAC-SHA256 over canonical StringToSign) or SAS
// (token appended verbatim). The signing path is fully implemented and
// independently testable; HTTP transport is W1-stubbed (kOsNotImplemented)
// until bolt::net grows an HTTP/1.1 client over TlsSocket.
//
// Canonical StringToSign (Shared Key, 2017-07-29+):
//   VERB \n Content-Encoding \n Content-Language \n Content-Length \n
//   Content-MD5 \n Content-Type \n Date \n If-Modified-Since \n If-Match \n
//   If-None-Match \n If-Unmodified-Since \n Range \n
//   CanonicalizedHeaders (x-ms-* sorted, lowercase) \n
//   CanonicalizedResource (/<account><path> \n key1:value1 \n key2:value2)
//
// Content-Length is the empty string when the body is zero bytes.
// `account_key` is the BASE64-DECODED key; the helper decodes it from the
// portal-style base64 string.
//
// Tiger Style: PODs only, fixed-cap buffers, ≥2 asserts/fn, no heap.

#pragma once

#include <cassert>
#include <cstdint>

#include "bolt/bolt_arena.h"
#include "bolt/lakehouse/object_store.h"
#include "bolt/lakehouse/object_store/compat.h"   // for Secret

namespace bolt {
namespace lakehouse {
namespace azure_blob {

static constexpr uint32_t kAzMaxAccount   = 64u;
static constexpr uint32_t kAzMaxContainer = 64u;
static constexpr uint32_t kAzMaxEndpoint  = 256u;
static constexpr uint32_t kAzMaxSas       = 1024u;

using s3_compat::Secret;

struct Config {
    char    account[kAzMaxAccount];
    Secret  account_key;             // base64 PORTAL string; decoded internally
    char    container[kAzMaxContainer];
    char    endpoint_override[kAzMaxEndpoint];  // "" → default
                                                //   https://<account>.blob.core.windows.net
    char    sas_token[kAzMaxSas];    // "" → use Shared Key
};

struct AzureBlobStore {
    Config   cfg;
    uint8_t  decoded_key[64];        // raw HMAC key bytes
    uint32_t decoded_key_len;
    bool     use_sas;
    uint8_t  _pad[3];
};

// Initialise + bind `out` to the Azure vtable. Returns false on overflow or
// invalid base64 in `account_key`.
bool azure_blob_store_new(ObjectStore* out, Arena* arena,
                          const Config* cfg) noexcept;

// ---------------------------------------------------------------------------
// Low-level primitives — exposed so the signing path is unit-testable.
// ---------------------------------------------------------------------------

// Headers passed into the signer. All char*'s are NUL-terminated; pass "" for
// "not present" (Content-Length is empty when body is zero bytes — caller
// supplies "" in that case).
struct SignHeaders {
    const char* content_encoding;
    const char* content_language;
    const char* content_length;       // "" for zero-byte body
    const char* content_md5;
    const char* content_type;
    const char* date_header;          // "" if x-ms-date is used (recommended)
    const char* if_modified_since;
    const char* if_match;
    const char* if_none_match;
    const char* if_unmodified_since;
    const char* range;
};

// Canonical x-ms-* header. `value` is the header value AFTER any
// leading-whitespace trim + folding of internal newlines (per Azure spec).
struct CanonicalizedHeader {
    const char* lc_name;    // lowercase, e.g. "x-ms-date"
    const char* value;
};

// Build the StringToSign (Shared Key). Returns the byte length written
// (excluding terminating NUL). Returns 0 on overflow.
uint32_t build_string_to_sign(const char* verb,
                              const SignHeaders* hdrs,
                              const char* account,
                              const char* resource_path,
                              const CanonicalizedHeader* x_ms_headers,
                              uint32_t x_ms_count,
                              const char* query_pairs,   // sorted "k:v\nk:v" or ""
                              char* out, uint32_t cap) noexcept;

// HMAC-SHA256(decoded_key, string_to_sign) → base64 signature.
// Writes NUL-terminated base64 into `out_b64`. Returns false on overflow.
bool sign_shared_key(const uint8_t* decoded_key, uint32_t decoded_key_len,
                     const char* string_to_sign, uint32_t sts_len,
                     char* out_b64, uint32_t cap) noexcept;

// Base64-decode `src` (NUL-terminated). Writes raw bytes into out[0..cap),
// returns bytes written. Returns 0 on invalid input or overflow.
uint32_t base64_decode(const char* src, uint8_t* out, uint32_t cap) noexcept;

// Base64-encode `n` bytes from `src` → NUL-terminated `out_b64`. Returns the
// encoded length (excluding NUL). Returns 0 on overflow.
uint32_t base64_encode(const uint8_t* src, uint32_t n,
                       char* out_b64, uint32_t cap) noexcept;

}  // namespace azure_blob
}  // namespace lakehouse
}  // namespace bolt
