// bolt/lakehouse/object_store_compat.cpp — S3-compat endpoint builders.

#include "bolt/lakehouse/object_store/compat.h"

#include <cstdio>
#include <cstring>

namespace bolt {
namespace lakehouse {
namespace s3_compat {

bool secret_set(Secret* s, const char* src) noexcept {
    assert(s != nullptr);
    if (s == nullptr) return false;
    if (src == nullptr) {
        s->bytes[0] = '\0';
        s->len = 0;
        return true;
    }
    const size_t n = std::strlen(src);
    if (n + 1u > kSecretCap) return false;
    std::memcpy(s->bytes, src, n + 1u);
    s->len = static_cast<uint32_t>(n);
    assert(s->bytes[s->len] == '\0');
    return true;
}

namespace {

// Stamp a freshly-arena-allocated S3ObjectStore + bind `out`. Returns false
// on alloc failure or s3_object_store_init overflow.
bool stamp(ObjectStore* out, Arena* arena, const char* bucket,
           const char* region, const char* endpoint,
           const Secret* key, const Secret* secret,
           bool path_style) noexcept {
    assert(out != nullptr);
    assert(arena != nullptr);
    if (out == nullptr || arena == nullptr) return false;
    if (key == nullptr || secret == nullptr) return false;
    S3ObjectStore* s3 = arena->allocate_array<S3ObjectStore>(1);
    if (s3 == nullptr) return false;
    if (!s3_object_store_init(s3, bucket, region, key->bytes, secret->bytes,
                              out)) {
        return false;
    }
    // Stamp endpoint + path-style after init (init clears these).
    if (endpoint != nullptr && endpoint[0] != '\0') {
        const size_t n = std::strlen(endpoint);
        if (n + 1u > kS3MaxEndpoint) return false;
        std::memcpy(s3->endpoint, endpoint, n + 1u);
    }
    s3->path_style = path_style;
    return true;
}

}  // namespace

bool minio_store_new(ObjectStore* out, Arena* arena, const char* endpoint,
                     const char* bucket, const Secret* key,
                     const Secret* secret) noexcept {
    assert(out != nullptr);
    assert(endpoint != nullptr);
    return stamp(out, arena, bucket, "us-east-1", endpoint, key, secret,
                 /*path_style=*/true);
}

bool r2_store_new(ObjectStore* out, Arena* arena, const char* account_id,
                  const char* bucket, const Secret* key,
                  const Secret* secret) noexcept {
    assert(out != nullptr);
    assert(account_id != nullptr);
    char endpoint[kS3MaxEndpoint];
    // https://<account>.r2.cloudflarestorage.com
    const int n = std::snprintf(endpoint, sizeof(endpoint),
                                "https://%s.r2.cloudflarestorage.com",
                                account_id);
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(endpoint)) return false;
    return stamp(out, arena, bucket, "auto", endpoint, key, secret,
                 /*path_style=*/false);
}

bool backblaze_store_new(ObjectStore* out, Arena* arena, const char* bucket,
                         const Secret* app_key_id,
                         const Secret* app_key) noexcept {
    assert(out != nullptr);
    assert(bucket != nullptr);
    const char* region = "us-west-002";
    char endpoint[kS3MaxEndpoint];
    const int n = std::snprintf(endpoint, sizeof(endpoint),
                                "https://s3.%s.backblazeb2.com", region);
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(endpoint)) return false;
    return stamp(out, arena, bucket, region, endpoint, app_key_id, app_key,
                 /*path_style=*/false);
}

bool wasabi_store_new(ObjectStore* out, Arena* arena, const char* region,
                      const char* bucket, const Secret* key,
                      const Secret* secret) noexcept {
    assert(out != nullptr);
    assert(region != nullptr && region[0] != '\0');
    char endpoint[kS3MaxEndpoint];
    const int n = std::snprintf(endpoint, sizeof(endpoint),
                                "https://s3.%s.wasabisys.com", region);
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(endpoint)) return false;
    return stamp(out, arena, bucket, region, endpoint, key, secret,
                 /*path_style=*/false);
}

bool digitalocean_store_new(ObjectStore* out, Arena* arena, const char* region,
                            const char* bucket, const Secret* key,
                            const Secret* secret) noexcept {
    assert(out != nullptr);
    assert(region != nullptr && region[0] != '\0');
    char endpoint[kS3MaxEndpoint];
    const int n = std::snprintf(endpoint, sizeof(endpoint),
                                "https://%s.digitaloceanspaces.com", region);
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(endpoint)) return false;
    return stamp(out, arena, bucket, region, endpoint, key, secret,
                 /*path_style=*/false);
}

}  // namespace s3_compat
}  // namespace lakehouse
}  // namespace bolt
