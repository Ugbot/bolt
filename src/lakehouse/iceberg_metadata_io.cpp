#include "bolt/lakehouse/iceberg/metadata_io.h"
#include "bolt/crypto/sha256.h"
#include "bolt/parse/bolt_json.h"
#include "bolt/lakehouse/object_store/s3_uri.h"

#include <cassert>
#include <cstring>

namespace bolt::lakehouse::iceberg {
namespace {
using Status = MetadataLoadStatus;
using Span = detail::ObjectSpan;
namespace json = bolt::parse::json;

bool length_of(const char* text, uint32_t capacity, uint32_t* length) noexcept {
    assert(text != nullptr && length != nullptr);
    assert(capacity > 0);
    Span span{};
    if (!detail::object_key_span(text, capacity, &span)) return false;
    *length = static_cast<uint32_t>(span.size - 1u);
    return true;
}

Status uri_status(S3UriStatus status) noexcept {
    assert(sizeof(S3UriStatus) == 1); assert(sizeof(Status) == 1);
    if (status == S3UriStatus::Ok) return Status::Ok;
    return status == S3UriStatus::Unsupported ? Status::Unsupported : Status::BadArg;
}

Status identity_valid(const MetadataObjectIdentity& id) noexcept {
    assert(sizeof(id.object_key) == kOsMaxKey);
    assert(sizeof(id.table_uuid) == kIcebergMaxUuid);
    uint32_t root = 0, table = 0, uri = 0, key = 0, uuid = 0;
    if (!length_of(id.store_location, sizeof(id.store_location), &root) ||
        !length_of(id.table_location, sizeof(id.table_location), &table) ||
        !length_of(id.metadata_uri, sizeof(id.metadata_uri), &uri) ||
        !length_of(id.object_key, sizeof(id.object_key), &key) ||
        !length_of(id.table_uuid, sizeof(id.table_uuid), &uuid) || uuid == 0) return Status::BadArg;
    for (uint32_t i = 0; i < uuid; ++i)
        if (static_cast<unsigned char>(id.table_uuid[i]) < 32 || id.table_uuid[i] == 127)
            return Status::BadArg;
    const Span uris[]{{id.store_location, root}, {id.table_location, table}, {id.metadata_uri, uri}};
    for (const Span item : uris) {
        const Status status = uri_status(s3_uri_validate(static_cast<const char*>(item.data)));
        if (status != Status::Ok) return status;
    }
    char mapped[kIcebergMaxLocation]{};
    const Status mapped_status = uri_status(s3_uri_for_object_key(id.store_location,
        id.object_key, mapped, sizeof(mapped)));
    if (mapped_status != Status::Ok) return mapped_status;
    return std::strlen(mapped) == uri && std::memcmp(mapped, id.metadata_uri, uri) == 0
        ? Status::Ok : Status::IdentityMismatch;
}

bool spans_valid(ObjectStore* store, const MetadataObjectIdentity* id,
    const MetadataLoadLimits* limits, Arena* metadata, Arena* io,
    Metadata* out, MetadataLoadResult* result) noexcept {
    assert(store != nullptr && metadata != nullptr && io != nullptr);
    assert(id != nullptr && limits != nullptr);
    if (metadata == io || store->vt == nullptr || store->impl == nullptr) return false;
    const Span inputs[]{{id, sizeof(*id)}, {limits, sizeof(*limits)},
        {store, sizeof(*store)}, {store->vt, sizeof(*store->vt)}};
    const Span outputs[]{{out, sizeof(*out)}, {result, sizeof(*result)}};
    const Span opaque{store->impl, 1};
    if (!detail::object_io_admit(inputs, 4, outputs, 2, metadata, &opaque, 1) ||
        !detail::object_io_admit(inputs, 4, outputs, 2, io, &opaque, 1)) return false;
    const Span arenas[]{{metadata, sizeof(*metadata)}, {io, sizeof(*io)}};
    if (detail::object_spans_overlap(arenas[0], arenas[1]) ||
        metadata->overlaps_backing(io, sizeof(*io)) || io->overlaps_backing(metadata, sizeof(*metadata)) ||
        metadata->overlaps_backing(store->impl, 1) || io->overlaps_backing(store->impl, 1)) return false;
    for (const Span arena : arenas) {
        if (!detail::object_span_valid(arena) || detail::object_spans_overlap(arena, opaque)) return false;
        for (const Span input : inputs)
            if (detail::object_spans_overlap(arena, input)) return false;
    }
    return true;
}

Status range_status(int status) noexcept {
    switch (status) {
        case kOsOk: return Status::Ok;
        case kOsNotFound: return Status::NotFound;
        case kOsPreconditionFailed: return Status::Changed;
        case kOsNotImplemented: return Status::Unsupported;
        case kOsBufferSmall: return Status::Capacity;
        case kOsBadArg: return Status::BadArg;
        default: return Status::IoError;
    }
}

Status read_exact(ObjectStore* store, const MetadataObjectIdentity& id,
    uint32_t request, Arena* io, uint8_t* bytes) noexcept {
    assert(store != nullptr && bytes != nullptr);
    assert(request > 0 && request <= kMetadataLoadMaxRequestBytes);
    uint64_t offset = 0;
    const uint64_t steps = id.version.size / request + 1u;
    for (uint64_t i = 0; offset < id.version.size && i < steps; ++i) {
        const uint32_t length = static_cast<uint32_t>(id.version.size - offset < request
            ? id.version.size - offset : request);
        uint32_t got = 0;
        io->reset(); // Only per-request provider scratch; retained bytes are disjoint.
        const Status status = range_status(os_get_range_if_match(store, id.object_key,
            &id.version, offset, length, io, bytes + offset, &got));
        if (status != Status::Ok) return status;
        if (got != length) return Status::IoError;
        offset += length;
    }
    assert(offset == id.version.size);
    return Status::Ok;
}

// Existing metadata_parse copies bounded identity strings. Check the original
// root strings through the maintained SAX parser so truncation/duplicates cannot
// masquerade as the pinned UUID/location. SAX does not allocate a second tape.
struct IdentityCheck {
    const MetadataObjectIdentity* expected;
    uint32_t depth = 0, selected = 0, seen = 0;
    bool mismatch = false;
};
IdentityCheck& identity_check(void* context) noexcept {
    assert(context != nullptr);
    auto& check = *static_cast<IdentityCheck*>(context);
    assert(check.expected != nullptr);
    return check;
}
bool json_begin(void* context) noexcept {
    auto& check = identity_check(context);
    if (check.depth == 1 && check.selected != 0) return false;
    ++check.depth; return true;
}
bool json_end(void* context) noexcept {
    auto& check = identity_check(context);
    if (check.depth == 0) return false;
    --check.depth; return true;
}
bool json_key(void* context, const char* key, int32_t length) noexcept {
    auto& check = identity_check(context);
    if (check.depth != 1) return true;
    // SAX exposes raw slices: an escaped key may alias either identity key
    // after JSON decoding. Refuse every escaped root key in this subset.
    if (length < 0 || std::memchr(key, '\\', static_cast<size_t>(length)) != nullptr)
        return false;
    check.selected = length == 8 && std::memcmp(key, "location", 8) == 0 ? 1u :
        length == 10 && std::memcmp(key, "table-uuid", 10) == 0 ? 2u : 0u;
    return check.selected == 0 || (check.seen & check.selected) == 0;
}
bool json_string(void* context, const char* value, int32_t length) noexcept {
    auto& check = identity_check(context);
    if (check.depth != 1 || check.selected == 0) return true;
    if (length < 0 || std::memchr(value, '\\', static_cast<size_t>(length)) != nullptr)
        return false;
    const char* expected = check.selected == 1 ? check.expected->table_location : check.expected->table_uuid;
    if (static_cast<size_t>(length) != std::strlen(expected) ||
        std::memcmp(value, expected, static_cast<size_t>(length)) != 0) {
        check.mismatch = true; return false;
    }
    check.seen |= check.selected; check.selected = 0; return true;
}
bool json_scalar(void* context) noexcept {
    const auto& check = identity_check(context);
    return check.depth != 1 || check.selected == 0;
}
bool json_integer(void* context, int64_t, const char*, int32_t) noexcept { return json_scalar(context); }
bool json_float(void* context, double, const char*, int32_t) noexcept { return json_scalar(context); }
bool json_boolean(void* context, bool) noexcept { return json_scalar(context); }

Status original_identity(const uint8_t* bytes, uint32_t length,
                          const MetadataObjectIdentity& identity) noexcept {
    assert(bytes != nullptr && length > 0);
    assert(length < (1u << 30));
    IdentityCheck check{&identity};
    json::SaxHandler handler{&check, json_begin, json_end, json_begin, json_end,
        json_key, json_string, json_integer, json_float, json_boolean, json_scalar};
    if (!json::sax_parse(bytes, static_cast<int32_t>(length), &handler) || check.seen != 3)
        return check.mismatch ? Status::IdentityMismatch : Status::DecodeFailed;
    return Status::Ok;
}
Status decode_exact(const MetadataObjectIdentity& pinned, const uint8_t* bytes,
    uint32_t length, Arena* metadata, Metadata* staged, Metadata* out,
    MetadataLoadResult* result) noexcept {
    assert(bytes != nullptr && metadata != nullptr); assert(staged != nullptr && out != nullptr && result != nullptr);
    auto status = original_identity(bytes, length, pinned);
    if (status != Status::Ok) return status;
    if (!metadata_parse(bytes, length, metadata, staged) ||
        (staged->format_version != 1 && staged->format_version != 2)) return Status::DecodeFailed;
    if (std::strcmp(staged->location, pinned.table_location) != 0 ||
        std::strcmp(staged->table_uuid, pinned.table_uuid) != 0) return Status::IdentityMismatch;
    MetadataLoadResult copied{}; copied.object = pinned;
    crypto::Sha256State hash{}; uint8_t digest[32]{};
    if (!crypto::sha256_stream_init(&hash) || !crypto::sha256_stream_update(&hash, bytes, length) ||
        !crypto::sha256_stream_final(&hash, digest)) return Status::DecodeFailed;
    constexpr char hex[] = "0123456789abcdef";
    for (uint32_t i = 0; i < 32; ++i) {
        copied.sha256[2u * i] = hex[digest[i] >> 4]; copied.sha256[2u * i + 1u] = hex[digest[i] & 15];
    }
    *out = *staged; *result = copied;
    assert(out->arena == metadata && result->sha256[64] == 0);
    return Status::Ok;
}

} // namespace

MetadataLoadStatus metadata_load_exact(ObjectStore* store,
    const MetadataObjectIdentity* id, const MetadataLoadLimits* limits,
    Arena* metadata, Arena* io, Metadata* out, MetadataLoadResult* result) noexcept {
    assert(store != nullptr && id != nullptr && limits != nullptr);
    assert(metadata != nullptr && io != nullptr && out != nullptr && result != nullptr);
    if (store == nullptr || id == nullptr || limits == nullptr || metadata == nullptr ||
        io == nullptr || out == nullptr || result == nullptr) return Status::BadArg;
    if (!spans_valid(store, id, limits, metadata, io, out, result) ||
        metadata->growth_enabled() || io->growth_enabled()) return Status::BadArg;
    if (!os_object_version_valid(&id->version) || id->version.size == 0 ||
        limits->max_bytes == 0 || limits->max_bytes >= (1u << 30) ||
        limits->request_bytes == 0 || limits->request_bytes > kMetadataLoadMaxRequestBytes)
        return Status::BadArg;
    Status status = identity_valid(*id);
    if (status != Status::Ok) return status;
    if (store->vt->get_range_if_match == nullptr) return Status::Unsupported;
    if (id->version.size > limits->max_bytes || id->version.size >= (1u << 30) ||
        id->version.size > SIZE_MAX) return Status::Capacity;
    auto* bytes = metadata->allocate_array<uint8_t>(static_cast<size_t>(id->version.size));
    auto* staged = metadata->allocate_array<Metadata>(1);
    if (bytes == nullptr || staged == nullptr) return Status::Capacity;
    const MetadataObjectIdentity pinned = *id;
    status = read_exact(store, pinned, limits->request_bytes, io, bytes);
    if (status != Status::Ok) return status;
    const auto length = static_cast<uint32_t>(pinned.version.size);
    return decode_exact(pinned, bytes, length, metadata, staged, out, result);
}

MetadataLoadStatus metadata_validate_exact(const MetadataObjectIdentity* id,
    const MetadataLoadLimits* limits, const uint8_t* bytes, uint64_t length,
    Arena* metadata, Metadata* out, MetadataLoadResult* result) noexcept {
    assert(id != nullptr && limits != nullptr && bytes != nullptr);
    assert(metadata != nullptr && out != nullptr && result != nullptr);
    if (id == nullptr || limits == nullptr || bytes == nullptr || metadata == nullptr ||
        out == nullptr || result == nullptr || length > SIZE_MAX) return Status::BadArg;
    const Span inputs[]{{id, sizeof(*id)}, {limits, sizeof(*limits)}, {bytes, static_cast<size_t>(length)}};
    const Span outputs[]{{out, sizeof(*out)}, {result, sizeof(*result)}};
    const Span arena_object{metadata, sizeof(*metadata)};
    if (!detail::object_io_admit(inputs, 3, outputs, 2, nullptr, &arena_object, 1) ||
        !detail::object_io_admit(inputs, 3, outputs, 2, metadata, nullptr, 0) ||
        metadata->growth_enabled()) return Status::BadArg;
    if (!os_object_version_valid(&id->version) || !length || !limits->max_bytes ||
        limits->max_bytes >= (1u << 30) || !limits->request_bytes ||
        limits->request_bytes > kMetadataLoadMaxRequestBytes) return Status::BadArg;
    if (length != id->version.size) return Status::IdentityMismatch;
    auto status = identity_valid(*id); if (status != Status::Ok) return status;
    if (length > limits->max_bytes || length >= (1u << 30)) return Status::Capacity;
    auto* retained = metadata->allocate_array<uint8_t>(static_cast<size_t>(length));
    auto* staged = metadata->allocate_array<Metadata>(1);
    if (retained == nullptr || staged == nullptr) return Status::Capacity;
    std::memcpy(retained, bytes, static_cast<size_t>(length));
    return decode_exact(*id, retained, static_cast<uint32_t>(length), metadata, staged, out, result);
}
} // namespace bolt::lakehouse::iceberg
