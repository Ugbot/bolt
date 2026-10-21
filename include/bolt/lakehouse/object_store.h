// bolt/lakehouse/object_store.h — abstract object store (vtable, not virtual).
//
// One POD `ObjectStoreVT` of function pointers + a `void* impl` handle is the
// trait. Implementations: filesystem (local paths), S3 (SigV4 over
// bolt::net), Azure Blob and GCS (object_store/). All buffers
// are caller-owned / arena-allocated — no hidden heap.
//
// Tiger Style: PODs, ≥2 asserts/fn, no exceptions, no smart pointers.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "bolt/bolt_arena.h"
#include "bolt/lakehouse/object_store/versioned.h"

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
    kOsExists          = 6,   // put_if_absent: the key already exists
    kOsCancelled       = 7,   // cooperative cancellation before commit
    kOsPreconditionFailed = 9, // pinned object version changed; never implicitly repin
    kOsWriteUncertain  = 8,   // write may have committed; do not blindly retry
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

    // PUT object `key` with body data[0..len). Returns kOs* code. S3 uses
    // bounded multipart above 16 MiB; kOsWriteUncertain requires caller
    // reconciliation. Explicit streaming/progress: object_store/s3_upload.h.
    int (*put_object)(void* impl, const char* key,
                      const uint8_t* data, uint64_t len) noexcept;

    // List keys under `prefix` into out[0..cap); *out_n gets the count.
    int (*list_prefix)(void* impl, const char* prefix, ObjectEntry* out,
                       uint32_t cap, uint32_t* out_n) noexcept;

    // DELETE object `key`.
    int (*delete_object)(void* impl, const char* key) noexcept;

    // HEAD object `key` → meta.
    int (*head_object)(void* impl, const char* key, ObjectMeta* out) noexcept;

    // Atomically create `key` with body data[0..len) iff it does not exist.
    // Returns kOsExists (and changes nothing) when it does. This is the
    // compare-and-swap every table-format commit is built on: a HEAD followed
    // by a PUT is not one, because two writers can both see "absent". A store
    // that cannot provide the guarantee leaves this null (os_put_if_absent
    // then reports kOsNotImplemented) — it must never be emulated.
    // A lost/ambiguous acknowledgement is kOsWriteUncertain, not proof of
    // failure: preserve any objects this candidate might reference and
    // reconcile exact candidate content/identity before another attempt.
    int (*put_if_absent)(void* impl, const char* key,
                         const uint8_t* data, uint64_t len) noexcept;

    // GET bytes [offset, offset + len) of `key` into the caller's `dst`
    // (capacity `len`); *out_len gets the bytes read, short only at the end
    // of the object. One request per call, so a caller reading a large
    // object bounds its buffer and chooses its request size. Null on a store
    // without ranged reads (os_get_range reports kOsNotImplemented).
    int (*get_range)(void* impl, const char* key, uint64_t offset,
                     uint64_t len, uint8_t* dst, uint64_t* out_len) noexcept;

    // Optional strong versioned transport. No emulation via old HEAD/range.
    // Caller supplies exclusive resettable body workspace; result owns no spans.
    int (*head_versioned)(void* impl, const char* key, Arena* scratch,
                          ObjectVersion* out) noexcept = nullptr;
    // Exact206/If-Match range, <=1 MiB and wholly within expected extent.
    // Success copies exactly length; admitted failure publishes zero length.
    // Invalid aliases return BadArg without changing output. Key/descriptor/output
    // must be disjoint from scratch backing and mutable output spans.
    int (*get_range_if_match)(void* impl, const char* key, const ObjectVersion* expected,
                              uint64_t offset, uint32_t length, Arena* scratch,
                              uint8_t* dst, uint32_t* out_length) noexcept = nullptr;
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
// list_prefix contract: past `cap` a backend returns kOsBufferSmall with
// `*n == cap`, never a silently short listing. os_list_all retries with
// doubled room in `arena` so no listing size is a ceiling; the arena's
// allocation failure is the only bound.
inline int os_list_all(ObjectStore* s, const char* prefix, Arena* arena,
                       uint32_t initial_cap, ObjectEntry** out,
                       uint32_t* n) noexcept {
    assert(s != nullptr && arena != nullptr);
    assert(out != nullptr && n != nullptr && initial_cap > 0u);
    *out = nullptr;
    *n = 0;
    uint32_t cap = initial_cap;
    for (uint32_t round = 0; round < 24u; ++round) {   // bounded
        ObjectEntry* buf = arena->allocate_array<ObjectEntry>(cap);
        if (buf == nullptr) return kOsBufferSmall;
        uint32_t got = 0;
        const int rc = os_list(s, prefix, buf, cap, &got);
        if (rc == kOsOk) { *out = buf; *n = got; return kOsOk; }
        if (rc != kOsBufferSmall) return rc;
        if (cap > (UINT32_MAX / 2u)) return kOsBufferSmall;
        cap *= 2u;
    }
    return kOsBufferSmall;
}
inline int os_head(ObjectStore* s, const char* key, ObjectMeta* m) noexcept {
    assert(s != nullptr && s->vt != nullptr);
    assert(s->vt->head_object != nullptr);
    return s->vt->head_object(s->impl, key, m);
}
inline int os_put_if_absent(ObjectStore* s, const char* key, const uint8_t* d,
                            uint64_t n) noexcept {
    assert(s != nullptr && s->vt != nullptr);
    assert(key != nullptr && (d != nullptr || n == 0));
    if (s->vt->put_if_absent == nullptr) return kOsNotImplemented;
    return s->vt->put_if_absent(s->impl, key, d, n);
}

// Invalid writable aliases are rejected without output writes. Present providers
// initialize admitted outputs, including on errors; wrappers do not prewrite them.
inline int os_get_range(ObjectStore* s, const char* key, uint64_t offset,
                        uint64_t len, uint8_t* dst, uint64_t* out_len) noexcept {
    assert(s != nullptr && s->vt != nullptr);
    assert(key != nullptr && out_len != nullptr && (dst != nullptr || len == 0));
    detail::ObjectSpan key_span{};
    const detail::ObjectSpan outputs[]{{dst, static_cast<size_t>(len)}, {out_len, sizeof(*out_len)}};
    const detail::ObjectSpan protected_spans[]{{s, sizeof(*s)}, {s->vt, sizeof(*s->vt)}};
    if (len > SIZE_MAX || !detail::object_key_span(key, kOsMaxKey, &key_span) ||
        !detail::object_io_admit(&key_span, 1, outputs, 2, nullptr, protected_spans, 2)) return kOsBadArg;
    if (s->vt->get_range == nullptr) { *out_len = 0; return kOsNotImplemented; }
    return s->vt->get_range(s->impl, key, offset, len, dst, out_len);
}

inline int os_head_versioned(ObjectStore* s, const char* key, Arena* scratch,
                              ObjectVersion* out) noexcept {
    assert(s != nullptr && s->vt != nullptr && key != nullptr);
    assert(scratch != nullptr && out != nullptr);
    detail::ObjectSpan key_span{};
    const detail::ObjectSpan output{out, sizeof(*out)};
    const detail::ObjectSpan protected_spans[]{{s, sizeof(*s)}, {s->vt, sizeof(*s->vt)}};
    if (!detail::object_key_span(key, kOsMaxKey, &key_span) ||
        !detail::object_io_admit(&key_span, 1, &output, 1, scratch, protected_spans, 2)) return kOsBadArg;
    if (s->vt->head_versioned == nullptr) { *out = {}; return kOsNotImplemented; }
    return s->vt->head_versioned(s->impl, key, scratch, out);
}
inline int os_get_range_if_match(ObjectStore* s, const char* key, const ObjectVersion* expected,
                                  uint64_t offset, uint32_t length, Arena* scratch,
                                  uint8_t* dst, uint32_t* out_length) noexcept {
    assert(s != nullptr && s->vt != nullptr && key != nullptr);
    assert(scratch != nullptr && out_length != nullptr && (dst != nullptr || length == 0));
    detail::ObjectSpan inputs[2]{{}, {expected, sizeof(*expected)}};
    const detail::ObjectSpan outputs[]{{dst, length > kOsVersionedRangeMaxBytes ? 0u : length},
        {out_length, sizeof(*out_length)}};
    const detail::ObjectSpan protected_spans[]{{s, sizeof(*s)}, {s->vt, sizeof(*s->vt)}};
    if (!detail::object_key_span(key, kOsMaxKey, &inputs[0]) ||
        !detail::object_io_admit(inputs, 2, outputs, 2, scratch, protected_spans, 2)) return kOsBadArg;
    if (s->vt->get_range_if_match == nullptr) { *out_length = 0; return kOsNotImplemented; }
    return s->vt->get_range_if_match(s->impl, key, expected, offset, length, scratch, dst, out_length);
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
// S3 implementation — config + SigV4 over bolt::net's HTTP client.
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
    uint8_t  create_only;                   // kS3CreateOnly*
    uint8_t  _pad[4];
};

// How put_if_absent expresses "only if the key does not exist".
static constexpr uint8_t kS3CreateOnlyIfNoneMatch   = 0;  // If-None-Match: *
static constexpr uint8_t kS3CreateOnlyGoogGeneration = 1; // GCS XML API

// Initialise `s3` + bind `out`. `endpoint` "" targets AWS; otherwise it is
// "http[s]://host[:port][/base]" (no scheme = https). Returns false on
// overflow of a config field.
bool s3_object_store_init(S3ObjectStore* s3, const char* bucket,
                          const char* region, const char* access_key,
                          const char* secret_key, ObjectStore* out) noexcept;

}  // namespace lakehouse
}  // namespace bolt
