// bolt/lakehouse/object_store/compat.h — S3-compatible endpoint helpers.
//
// W7. Pre-populate `S3ObjectStore` (from object_store.h) with the right
// region/endpoint/path-style for popular S3-API-compatible vendors:
// MinIO, Cloudflare R2, Backblaze B2, Wasabi, DigitalOcean Spaces.
//
// These are pure config builders — they wrap `s3_object_store_init` and set
// fields the bare AWS init doesn't expose. The HTTP transport in
// object_store_s3.cpp is still W1-stubbed (kOsNotImplemented); these helpers
// are a no-cost convenience that becomes useful the moment HTTP lands.
//
// Tiger Style: PODs in/out, ≥2 asserts/fn, no heap, no exceptions.

#pragma once

#include <cassert>
#include <cstdint>

#include "bolt/bolt_arena.h"
#include "bolt/lakehouse/object_store.h"

namespace bolt {
namespace lakehouse {
namespace s3_compat {

// A bounded "secret" — fixed-capacity char buffer + length. Caller owns
// storage; never heap-allocates. Used by W7 entry points so callers don't
// have to expose raw `const char*` for credentials.
static constexpr uint32_t kSecretCap = 256u;

struct Secret {
    char     bytes[kSecretCap];   // NUL-terminated
    uint32_t len;                 // strlen(bytes); 0 = empty
    uint8_t  _pad[4];
};

// Initialise a Secret from a C string. Returns false on overflow.
bool secret_set(Secret* s, const char* src) noexcept;

// ---------------------------------------------------------------------------
// MinIO — caller-supplied endpoint, region forced to "us-east-1", path-style
// addressing on (MinIO defaults to path-style).
// `endpoint` is the full URL host (e.g. "minio.local:9000" or
// "https://minio.example.com").
// ---------------------------------------------------------------------------
bool minio_store_new(ObjectStore* out, Arena* arena, const char* endpoint,
                     const char* bucket, const Secret* key,
                     const Secret* secret) noexcept;

// ---------------------------------------------------------------------------
// Cloudflare R2 — `https://<account_id>.r2.cloudflarestorage.com`,
// region "auto", virtual-hosted style.
// ---------------------------------------------------------------------------
bool r2_store_new(ObjectStore* out, Arena* arena, const char* account_id,
                  const char* bucket, const Secret* key,
                  const Secret* secret) noexcept;

// ---------------------------------------------------------------------------
// Backblaze B2 — `https://s3.<region>.backblazeb2.com`.
// `region` may be nullptr/"" → defaults to "us-west-002".
// ---------------------------------------------------------------------------
bool backblaze_store_new(ObjectStore* out, Arena* arena, const char* bucket,
                         const Secret* app_key_id,
                         const Secret* app_key) noexcept;

// ---------------------------------------------------------------------------
// Wasabi — `https://s3.<region>.wasabisys.com`.
// ---------------------------------------------------------------------------
bool wasabi_store_new(ObjectStore* out, Arena* arena, const char* region,
                      const char* bucket, const Secret* key,
                      const Secret* secret) noexcept;

// ---------------------------------------------------------------------------
// DigitalOcean Spaces — `https://<region>.digitaloceanspaces.com`.
// ---------------------------------------------------------------------------
bool digitalocean_store_new(ObjectStore* out, Arena* arena, const char* region,
                            const char* bucket, const Secret* key,
                            const Secret* secret) noexcept;

}  // namespace s3_compat
}  // namespace lakehouse
}  // namespace bolt
