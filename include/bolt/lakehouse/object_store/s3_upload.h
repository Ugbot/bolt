// Bounded, synchronous S3 multipart upload. Caller owns source storage, arena
// and result. Uses ordinary SigV4 requests, never implicit retries.
#pragma once
#include "bolt/lakehouse/object_store/object_store_limits.h"

#include "bolt/lakehouse/object_store.h"

namespace bolt::lakehouse {

inline constexpr uint32_t kS3MinPartBytes = 5u * 1024u * 1024u;
inline constexpr uint32_t kS3DefaultPartBytes = 8u * 1024u * 1024u;

enum class S3UploadPhase : uint8_t { Admission, Create, Source, Part, Complete, Done };

// Return kOsOk and exactly `length` readable bytes in *data, or an error.
// Calls are forward-only, at increasing offsets. A file producer can fill one
// preallocated part buffer. The span stays immutable/readable until the next
// read_part call or upload return. Callbacks must have their own finite bound.
struct S3UploadSource {
    int (*read_part)(void* context, uint64_t offset, uint32_t length,
                     const uint8_t** data) noexcept = nullptr;
    void* context = nullptr;
};

struct S3UploadOptions {
    uint32_t part_bytes = kS3DefaultPartBytes;
    uint32_t max_parts = kS3MaxUploadParts;
    uint16_t timeout_ms = 10000;  // per request, 1..60000; synchronous DNS excluded
    bool if_absent = false;      // CompleteMultipartUpload If-None-Match: *
    bool (*cancelled)(void* context) noexcept = nullptr;
    void* cancel_context = nullptr;
};

struct S3UploadResult {
    int status = kOsBadArg;
    int abort_status = kOsNotImplemented;  // no abort attempted
    uint32_t http_status = 0;  // last observed primary response; abort is separate
    uint32_t parts_uploaded = 0;
    uint64_t bytes_uploaded = 0;
    S3UploadPhase phase = S3UploadPhase::Admission;
    bool completion_attempted = false;
    bool remote_state_uncertain = false;  // lost initiation, commit or failed cleanup
    char upload_id[kS3UploadIdCap]{};
};

// Nonempty sources use sequential parts; zero bytes use an empty PutObject.
// Allocate metadata/XML in `arena` before Create; source staging belongs to the
// caller (<=part_bytes), HTTP control bodies are capped at 64 KiB. There is no
// shared upload state: budget this workspace independently per concurrent call.
// Store configuration must remain immutable until every concurrent call returns.
// The 8 MiB default supports 78.125 GiB; 16 MiB supports 156.25 GiB, both at
// 10,000 parts. Oversize geometry is refused before remote effects.
// Cancellation is checked between phases, not inside an in-flight HTTP request.
// Pre-completion failures attempt one abort and preserve its separate outcome.
// After completion is attempted, an unverified outcome is kOsWriteUncertain:
// no automatic retry, recreation or abort. HTTP 412 is the definitive kOsExists
// conditional refusal. Lost initiation may leave an unidentified orphan upload.
// The XML reader accepts the documented flat result, optional XML declaration
// and default S3 namespace; unsupported XML extensions fail explicitly.
int s3_upload_stream(S3ObjectStore* store, const char* key, uint64_t length,
                     const S3UploadSource* source, const S3UploadOptions* options,
                     Arena* arena, S3UploadResult* result) noexcept;

// Explicit cleanup/reconciliation helper, one request only. Acknowledged abort
// (or already-absent upload) is not proof that a remote in-flight part finished
// before cleanup; callers may need an independent lifecycle/ListParts policy.
int s3_abort_multipart(S3ObjectStore* store, const char* key,
                       const char* upload_id, uint16_t timeout_ms = 10000) noexcept;

}  // namespace bolt::lakehouse
