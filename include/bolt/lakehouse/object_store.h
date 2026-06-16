// bolt/lakehouse/object_store.h — abstract object store (vtable, not virtual).
//
// One POD `ObjectStoreVT` of function pointers + a `void* impl` handle is the
// trait. Two implementations in W1: filesystem (local paths) and S3 (SigV4
// signing wired; HTTP transport stubbed as BOLT_NOT_IMPLEMENTED). All buffers
// are caller-owned / arena-allocated — no hidden heap.
//
// Tiger Style: PODs, ≥2 asserts/fn, no exceptions, no smart pointers.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "bolt/bolt_arena.h"

namespace bolt {
namespace lakehouse {

// Result codes for object-store ops (int 0 = ok).
enum : int {
    kOsOk             = 0,
    kOsNotFound        = 1,
    kOsNotImplemented  = 2,
    kOsBadArg          = 3,
    kOsIoError         = 4,
    kOsBufferSmall     = 5,
};

static constexpr uint32_t kOsMaxKey     = 2048u;
static constexpr uint32_t kOsMaxListing  = 4096u;

// One entry returned by list_prefix.
struct ObjectEntry {
    char     key[kOsMaxKey];
    uint64_t size;
};

// HeadObject result.
struct ObjectMeta {
    uint64_t size;
    uint64_t mtime_unix;     // seconds since epoch; 0 if unknown
    bool     exists;
    uint8_t  _pad[7];
};

// The trait. `impl` carries the implementation's state (filesystem root, S3
// config, ...). Every function pointer takes that `impl` first.
struct ObjectStoreVT {
    // GET object `key`; allocates the body into `arena`. *out_data/​*out_len
    // receive the bytes. Returns kOs* code.
    int (*get_object)(void* impl, const char* key, Arena* arena,
                      const uint8_t** out_data, uint64_t* out_len) noexcept;

    // PUT object `key` with body data[0..len). Returns kOs* code.
    int (*put_object)(void* impl, const char* key,
                      const uint8_t* data, uint64_t len) noexcept;

    // List keys under `prefix` into out[0..cap); *out_n gets the count.
    int (*list_prefix)(void* impl, const char* prefix, ObjectEntry* out,
                       uint32_t cap, uint32_t* out_n) noexcept;

    // DELETE object `key`.
    int (*delete_object)(void* impl, const char* key) noexcept;

    // HEAD object `key` → meta.
    int (*head_object)(void* impl, const char* key, ObjectMeta* out) noexcept;
};

struct ObjectStore {
    const ObjectStoreVT* vt;
    void*                impl;
};

// Thin call wrappers (assert vt + dispatch).
inline int os_get(ObjectStore* s, const char* key, Arena* a,
                  const uint8_t** d, uint64_t* n) noexcept {
    assert(s != nullptr && s->vt != nullptr);
    assert(s->vt->get_object != nullptr);
    return s->vt->get_object(s->impl, key, a, d, n);
}
inline int os_put(ObjectStore* s, const char* key, const uint8_t* d,
                  uint64_t n) noexcept {
    assert(s != nullptr && s->vt != nullptr);
    assert(s->vt->put_object != nullptr);
    return s->vt->put_object(s->impl, key, d, n);
}
inline int os_list(ObjectStore* s, const char* prefix, ObjectEntry* out,
                   uint32_t cap, uint32_t* n) noexcept {
    assert(s != nullptr && s->vt != nullptr);
    assert(s->vt->list_prefix != nullptr);
    return s->vt->list_prefix(s->impl, prefix, out, cap, n);
}
inline int os_delete(ObjectStore* s, const char* key) noexcept {
    assert(s != nullptr && s->vt != nullptr);
    assert(s->vt->delete_object != nullptr);
    return s->vt->delete_object(s->impl, key);
}
inline int os_head(ObjectStore* s, const char* key, ObjectMeta* m) noexcept {
    assert(s != nullptr && s->vt != nullptr);
    assert(s->vt->head_object != nullptr);
    return s->vt->head_object(s->impl, key, m);
}

// ---------------------------------------------------------------------------
// Filesystem implementation — keys are paths relative to `root`.
// ---------------------------------------------------------------------------
static constexpr uint32_t kOsMaxRoot = 1024u;

struct FilesystemObjectStore {
    char root[kOsMaxRoot];     // base directory; keys are appended with '/'
};

// Initialise `fs` + bind `out` to the filesystem vtable. Returns false on a
// root path that is too long.
bool filesystem_object_store_init(FilesystemObjectStore* fs, const char* root,
                                  ObjectStore* out) noexcept;

// ---------------------------------------------------------------------------
// S3 implementation — config + SigV4. Transport stubbed in W1.
// ---------------------------------------------------------------------------
static constexpr uint32_t kS3MaxBucket   = 96u;
static constexpr uint32_t kS3MaxRegion    = 32u;
static constexpr uint32_t kS3MaxEndpoint  = 256u;
static constexpr uint32_t kS3MaxAccess    = 128u;
static constexpr uint32_t kS3MaxSecret    = 256u;
static constexpr uint32_t kS3MaxSession   = 2048u;

struct S3ObjectStore {
    char     bucket[kS3MaxBucket];
    char     region[kS3MaxRegion];
    char     endpoint[kS3MaxEndpoint];   // "" = AWS default
    char     access_key[kS3MaxAccess];
    char     secret_key[kS3MaxSecret];
    char     session_token[kS3MaxSession];  // "" = no STS
    uint16_t port;                          // 0 = scheme default
    bool     path_style;
    uint8_t  _pad[5];
};

// Initialise `s3` + bind `out`. The vtable's get/put/list/delete/head return
// kOsNotImplemented in W1 (signing is exercised via bolt::crypto::sigv4 + the
// s3_object_store_sign helper). Returns false on overflow of a config field.
bool s3_object_store_init(S3ObjectStore* s3, const char* bucket,
                          const char* region, const char* access_key,
                          const char* secret_key, ObjectStore* out) noexcept;

}  // namespace lakehouse
}  // namespace bolt
