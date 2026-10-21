#include "bolt/lakehouse/iceberg/metadata_io.h"
#include "bolt/crypto/sha256.h"
#include "bolt/parse/bolt_json.h"
#include "object_store_http.h"

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

bool path_components(const char* text, uint32_t length) noexcept {
    assert(text != nullptr);
    assert(length < kOsMaxKey);
    if (length == 0) return false;
    uint32_t start = 0;
    for (uint32_t i = 0; i <= length; ++i) {
        const auto c = static_cast<unsigned char>(text[i]);
        if ((c < 32 || c == 127 || c == '\\') && i != length) return false;
        if (i != length && c != '/') continue;
        const uint32_t n = i - start;
        if (n == 0 || (n == 1 && text[start] == '.') ||
            (n == 2 && text[start] == '.' && text[start + 1] == '.')) return false;
        start = i + 1u;
    }
    return true;
}

int upper_hex(unsigned char c) noexcept {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// Recognize the maintained encoder's canonical spelling; never normalize it.
Status canonical_s3(const char* uri, uint32_t length) noexcept {
    assert(uri != nullptr);
    assert(length < kIcebergMaxLocation);
    if (std::strncmp(uri, "s3://", 5) != 0) return Status::Unsupported;
    uint32_t end = 5;
    while (end < length && uri[end] != '/') ++end;
    const uint32_t bucket = end - 5u;
    if (bucket < 3 || bucket > 63 || uri[5] == '-' || uri[5] == '.' ||
        uri[end - 1u] == '-' || uri[end - 1u] == '.') return Status::BadArg;
    for (uint32_t i = 5; i < end; ++i) {
        const char c = uri[i];
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '.'))
            return Status::BadArg;
    }
    if (end == length) return Status::Ok; // Bucket-only store root is canonical.
    const char* path = uri + end + 1u;
    const uint32_t n = length - end - 1u;
    if (!path_components(path, n)) return Status::BadArg;
    for (uint32_t i = 0; i < n; ++i) {
        const auto c = static_cast<unsigned char>(path[i]);
        if (oshttp::unreserved(c) || c == '/') continue;
        if (c != '%' || n - i < 3) return Status::BadArg;
        const int hi = upper_hex(static_cast<unsigned char>(path[i + 1u]));
        const int lo = upper_hex(static_cast<unsigned char>(path[i + 2u]));
        if (hi < 0 || lo < 0) return Status::BadArg;
        const auto decoded = static_cast<unsigned char>(hi * 16 + lo);
        if (decoded < 32 || decoded == 127 || decoded == '\\' || decoded == '/' ||
            oshttp::unreserved(decoded)) return Status::BadArg;
        i += 2u;
    }
    return Status::Ok;
}

Status identity_valid(const MetadataObjectIdentity& id) noexcept {
    assert(sizeof(id.object_key) == kOsMaxKey);
    assert(sizeof(id.table_uuid) == kIcebergMaxUuid);
    uint32_t root = 0, table = 0, uri = 0, key = 0, uuid = 0;
    if (!length_of(id.store_location, sizeof(id.store_location), &root) ||
        !length_of(id.table_location, sizeof(id.table_location), &table) ||
        !length_of(id.metadata_uri, sizeof(id.metadata_uri), &uri) ||
        !length_of(id.object_key, sizeof(id.object_key), &key) ||
        !length_of(id.table_uuid, sizeof(id.table_uuid), &uuid) || uuid == 0 ||
        !path_components(id.object_key, key)) return Status::BadArg;
    for (uint32_t i = 0; i < uuid; ++i)
        if (static_cast<unsigned char>(id.table_uuid[i]) < 32 || id.table_uuid[i] == 127)
            return Status::BadArg;
    const Span uris[]{{id.store_location, root}, {id.table_location, table}, {id.metadata_uri, uri}};
    for (const Span item : uris) {
        const Status status = canonical_s3(static_cast<const char*>(item.data),
                                           static_cast<uint32_t>(item.size));
        if (status != Status::Ok) return status;
    }
    char mapped[kIcebergMaxLocation]{};
    oshttp::Buf out{}; oshttp::buf_init(&out, mapped, sizeof(mapped));
    oshttp::buf_str(&out, id.store_location);
    oshttp::buf_str(&out, "/");
    oshttp::buf_pct(&out, id.object_key, true);
    if (!out.ok) return Status::BadArg;
    return out.n == uri && std::memcmp(mapped, id.metadata_uri, uri) == 0
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
    status = original_identity(bytes, length, pinned);
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
} // namespace bolt::lakehouse::iceberg
