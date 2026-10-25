#include "bolt/lakehouse/object_store/s3_uri.h"
#include <cassert>
#include <cstring>

namespace bolt::lakehouse {
namespace {
using Status = S3UriStatus;
using Span = detail::ObjectSpan;

bool text_span(const char* text, Span* out) noexcept {
    assert(out != nullptr); assert(kOsMaxKey > 1);
    return detail::object_key_span(text, kOsMaxKey, out);
}

bool components(const char* text, uint32_t length) noexcept {
    assert(text != nullptr); assert(length < kOsMaxKey);
    if (length == 0) return false;
    uint32_t start = 0;
    for (uint32_t i = 0; i <= length; ++i) {
        const auto c = static_cast<unsigned char>(text[i]);
        if (i != length && (c < 32 || c == 127 || c == '\\')) return false;
        if (i != length && c != '/') continue;
        const uint32_t n = i - start;
        if (n == 0 || (n == 1 && text[start] == '.') ||
            (n == 2 && text[start] == '.' && text[start + 1] == '.')) return false;
        start = i + 1;
    }
    return true;
}


// G2ICE-342: an Iceberg path names its S3 key byte for byte, as Iceberg-Java's S3FileIO (S3URI) and pyiceberg's
// pyarrow FileIO read it: "%25" is the three bytes '%' '2' '5', never an escape; '+', ' ', UTF-8 and sub-delims are
// literal. Refused (no reader could agree on them): control bytes, '\\', '?' and '#' (Java strips a query and fragment).
bool literal_key_byte(unsigned char c) noexcept {
    return c >= 32 && c != 127 && c != '\\' && c != '?' && c != '#';
}

Status canonical(const char* uri, uint32_t length) noexcept {
    assert(uri != nullptr); assert(length < kOsMaxKey);
    if (length < 5 || std::memcmp(uri, "s3://", 5) != 0) return Status::Unsupported;
    uint32_t end = 5;
    while (end < length && uri[end] != '/') ++end;
    const uint32_t bucket = end - 5;
    if (bucket < 3 || bucket > 63 || uri[5] == '-' || uri[5] == '.' ||
        uri[end - 1] == '-' || uri[end - 1] == '.') return Status::BadArg;
    for (uint32_t i = 5; i < end; ++i) {
        const char c = uri[i];
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '.'))
            return Status::BadArg;
    }
    if (end == length) return Status::Ok;
    const char* path = uri + end + 1;
    const uint32_t n = length - end - 1;
    if (!components(path, n)) return Status::BadArg;
    for (uint32_t i = 0; i < n; ++i)                                      // bounded by n
        if (!literal_key_byte(static_cast<unsigned char>(path[i]))) return Status::BadArg;
    return Status::Ok;
}

bool admit(const char* a, const char* b, char* out, uint32_t capacity,
           Span (&inputs)[2]) noexcept {
    assert(kOsMaxKey > 0); assert(sizeof(Span) >= sizeof(void*));
    if (capacity == 0 || capacity > kOsMaxKey || !text_span(a, &inputs[0]) ||
        !text_span(b, &inputs[1])) return false;
    const Span output{out, capacity};
    return detail::object_io_admit(inputs, 2, &output, 1, nullptr, nullptr, 0);
}
} // namespace

S3UriStatus s3_uri_validate(const char* uri) noexcept {
    assert(kOsMaxKey > 5); assert(kOsMaxKey <= UINT32_MAX);
    Span span{};
    return text_span(uri, &span) ? canonical(uri, static_cast<uint32_t>(span.size - 1))
                                 : Status::BadArg;
}

// G2ICE-300: Hadoop's s3a:// and s3n:// name the same bucket and key as s3:// (Iceberg-Java's S3FileIO and HadoopFileIO
// read all three); a recorded path in either is mapped exactly as its s3:// spelling. Only the recorded path takes
// the alias: a configured root stays canonical.
static S3UriStatus s3_alias_object_key(const char* root, const char* uri, char* out, uint32_t capacity) noexcept {
    assert(uri != nullptr);
    char canon[kOsMaxKey];
    const size_t n = std::strlen(uri);
    if (n - 1u >= sizeof(canon)) return Status::Capacity;
    std::memcpy(canon, "s3", 2);
    std::memcpy(canon + 2, uri + 3, n - 3u + 1u);                        // "s3a://x" -> "s3://x"
    return s3_uri_object_key(root, canon, out, capacity);
}

S3UriStatus s3_uri_object_key(const char* root, const char* uri,
                            char* out, uint32_t capacity) noexcept {
    assert(kOsMaxKey > 5); assert(sizeof(char) == 1);
    if (uri != nullptr && (std::strncmp(uri, "s3a://", 6) == 0 || std::strncmp(uri, "s3n://", 6) == 0))
        return s3_alias_object_key(root, uri, out, capacity);
    Span inputs[2]{};
    if (!admit(root, uri, out, capacity, inputs)) return Status::BadArg;
    const auto root_len = static_cast<uint32_t>(inputs[0].size - 1);
    const auto uri_len = static_cast<uint32_t>(inputs[1].size - 1);
    auto status = canonical(root, root_len);
    if (status == Status::Ok) status = canonical(uri, uri_len);
    if (status != Status::Ok) return status;
    if (uri_len <= root_len || std::memcmp(root, uri, root_len) != 0 || uri[root_len] != '/')
        return Status::OutsideRoot;
    const char* key = uri + root_len + 1;
    const uint32_t bytes = uri_len - root_len - 1;                       // G2ICE-342: the key is the path, literally
    if (bytes + 1 > capacity) return Status::Capacity;
    std::memcpy(out, key, bytes);
    out[bytes] = 0;
    assert(bytes < capacity);
    return Status::Ok;
}

S3UriStatus s3_uri_for_object_key(const char* root, const char* key,
                                char* out, uint32_t capacity) noexcept {
    assert(kOsMaxKey > 5); assert(sizeof(char) == 1);
    Span inputs[2]{};
    if (!admit(root, key, out, capacity, inputs)) return Status::BadArg;
    const auto root_len = static_cast<uint32_t>(inputs[0].size - 1);
    const auto key_len = static_cast<uint32_t>(inputs[1].size - 1);
    const auto status = canonical(root, root_len);
    if (status != Status::Ok) return status;
    if (!components(key, key_len)) return Status::BadArg;
    for (uint32_t i = 0; i < key_len; ++i)                                // bounded by key_len
        if (!literal_key_byte(static_cast<unsigned char>(key[i]))) return Status::BadArg;
    const uint32_t bytes = root_len + 1 + key_len;                       // G2ICE-342: root + '/' + the key, literally
    if (bytes + 1 > capacity) return Status::Capacity;
    std::memcpy(out, root, root_len);
    out[root_len] = '/';
    std::memcpy(out + root_len + 1, key, key_len);
    out[bytes] = 0;
    return Status::Ok;
}
} // namespace bolt::lakehouse
