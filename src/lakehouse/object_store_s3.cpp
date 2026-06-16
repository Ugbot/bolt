// bolt/lakehouse/object_store_s3.cpp — S3 ObjectStore: SigV4 signing wired,
// HTTP transport stubbed (kOsNotImplemented) until bolt::net::TlsSocket is
// threaded through here in W7. The signing path is the real W1 deliverable.

#include "bolt/lakehouse/object_store.h"

#include <cstring>

#include "bolt/crypto/sigv4.h"

namespace bolt {
namespace lakehouse {

namespace {

int s3_get(void* impl, const char* key, Arena* arena,
           const uint8_t** out_data, uint64_t* out_len) noexcept {
    assert(impl != nullptr);
    assert(out_data != nullptr && out_len != nullptr);
    (void)impl; (void)key; (void)arena;
    *out_data = nullptr;
    *out_len = 0;
    return kOsNotImplemented;
}

int s3_put(void* impl, const char* key, const uint8_t* data,
           uint64_t len) noexcept {
    assert(impl != nullptr);
    assert(data != nullptr || len == 0);
    (void)impl; (void)key; (void)data; (void)len;
    return kOsNotImplemented;
}

int s3_list(void* impl, const char* prefix, ObjectEntry* out, uint32_t cap,
            uint32_t* out_n) noexcept {
    assert(impl != nullptr);
    assert(out != nullptr && out_n != nullptr);
    (void)impl; (void)prefix; (void)out; (void)cap;
    *out_n = 0;
    return kOsNotImplemented;
}

int s3_delete(void* impl, const char* key) noexcept {
    assert(impl != nullptr);
    assert(key != nullptr);
    (void)impl; (void)key;
    return kOsNotImplemented;
}

int s3_head(void* impl, const char* key, ObjectMeta* out) noexcept {
    assert(impl != nullptr);
    assert(out != nullptr);
    (void)impl; (void)key;
    std::memset(out, 0, sizeof(*out));
    return kOsNotImplemented;
}

const ObjectStoreVT kS3VT = {
    s3_get, s3_put, s3_list, s3_delete, s3_head,
};

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
