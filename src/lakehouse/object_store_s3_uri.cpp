#include "bolt/lakehouse/object_store/s3_uri.h"
#include "object_store_http.h"
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

int upper_hex(unsigned char c) noexcept {
    assert(sizeof(unsigned char) == 1); assert(kOsMaxKey < UINT32_MAX);
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// G2ICE-300: RFC 3986 sub-delims and ':' / '@' are literal path characters (Spark writes partition directories such as
// "p=1"; Iceberg-Java's S3FileIO maps the path to the key byte for byte). They stay literal: never decoded, never aliases.
bool sub_delim(unsigned char c) noexcept {
    return c == '!' || c == '$' || c == '&' || c == '\'' || c == '(' || c == ')' || c == '*' || c == '+' || c == ',' ||
           c == ';' || c == '=' || c == ':' || c == '@';
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
    for (uint32_t i = 0; i < n; ++i) {
        const auto c = static_cast<unsigned char>(path[i]);
        if (oshttp::unreserved(c) || c == '/' || sub_delim(c)) continue;
        if (c != '%' || n - i < 3) return Status::BadArg;
        const int hi = upper_hex(static_cast<unsigned char>(path[i + 1]));
        const int lo = upper_hex(static_cast<unsigned char>(path[i + 2]));
        if (hi < 0 || lo < 0) return Status::BadArg;
        const auto decoded = static_cast<unsigned char>(hi * 16 + lo);
        if (decoded < 32 || decoded == 127 || decoded == '\\' || decoded == '/' ||
            oshttp::unreserved(decoded)) return Status::BadArg;
        i += 2;
    }
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
    const uint32_t encoded = uri_len - root_len - 1;
    uint32_t bytes = 0;
    for (uint32_t i = 0; i < encoded; ++i, ++bytes) if (key[i] == '%') i += 2;
    if (bytes + 1 > capacity) return Status::Capacity;
    uint32_t written = 0;
    for (uint32_t i = 0; i < encoded; ++i) {
        if (key[i] != '%') { out[written++] = key[i]; continue; }
        const int hi = upper_hex(static_cast<unsigned char>(key[i + 1]));
        const int lo = upper_hex(static_cast<unsigned char>(key[i + 2]));
        out[written++] = static_cast<char>(hi * 16 + lo); i += 2;
    }
    out[written] = 0;
    assert(written == bytes); assert(written < capacity);
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
    uint32_t bytes = root_len + 1;
    for (uint32_t i = 0; i < key_len; ++i) {
        const auto c = static_cast<unsigned char>(key[i]);
        bytes += oshttp::unreserved(c) || c == '/' ? 1u : 3u;
    }
    if (bytes + 1 > capacity) return Status::Capacity;
    oshttp::Buf buffer{}; oshttp::buf_init(&buffer, out, capacity);
    oshttp::buf_str(&buffer, root); oshttp::buf_str(&buffer, "/");
    oshttp::buf_pct(&buffer, key, true);
    assert(buffer.ok); assert(buffer.n == bytes);
    return Status::Ok;
}
} // namespace bolt::lakehouse
