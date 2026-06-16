// bolt/lakehouse/iceberg_transform.cpp — Iceberg partition transforms.
//
// Murmur3-32 (seed 0) for bucket(). Tiger Style: noexcept, fixed-size.

#include "bolt/lakehouse/iceberg/transform.h"

#include <cassert>
#include <cstdint>
#include <cstring>

namespace bolt {
namespace lakehouse {
namespace iceberg {

namespace {

inline uint32_t rotl32(uint32_t x, int8_t r) noexcept {
    return (x << r) | (x >> (32 - r));
}

}  // namespace

uint32_t murmur3_32(const void* key, uint32_t len, uint32_t seed) noexcept {
    assert(key != nullptr || len == 0);
    assert(len < (1u << 30));
    const uint8_t* data = static_cast<const uint8_t*>(key);
    const uint32_t nblocks = len / 4u;
    uint32_t h = seed;
    const uint32_t c1 = 0xcc9e2d51u;
    const uint32_t c2 = 0x1b873593u;
    for (uint32_t i = 0; i < nblocks; ++i) {
        uint32_t k = 0;
        std::memcpy(&k, data + i * 4u, 4u);
        k *= c1;
        k = rotl32(k, 15);
        k *= c2;
        h ^= k;
        h = rotl32(h, 13);
        h = h * 5u + 0xe6546b64u;
    }
    uint32_t k1 = 0;
    const uint8_t* tail = data + nblocks * 4u;
    const uint32_t rem = len & 3u;
    if (rem == 3) k1 ^= static_cast<uint32_t>(tail[2]) << 16;
    if (rem >= 2) k1 ^= static_cast<uint32_t>(tail[1]) << 8;
    if (rem >= 1) {
        k1 ^= static_cast<uint32_t>(tail[0]);
        k1 *= c1;
        k1 = rotl32(k1, 15);
        k1 *= c2;
        h ^= k1;
    }
    h ^= len;
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

int32_t bucket_i64(int64_t value, int32_t N) noexcept {
    assert(N > 0);
    if (N <= 0) return 0;
    uint8_t buf[8];
    for (uint32_t i = 0; i < 8; ++i) {
        buf[i] = static_cast<uint8_t>((value >> (i * 8)) & 0xff);
    }
    const uint32_t h = murmur3_32(buf, 8u, 0u);
    return static_cast<int32_t>((h & 0x7fffffffu) % static_cast<uint32_t>(N));
}

int32_t bucket_str(const char* s, uint32_t len, int32_t N) noexcept {
    assert(s != nullptr || len == 0);
    assert(N > 0);
    if (N <= 0) return 0;
    const uint32_t h = murmur3_32(s, len, 0u);
    return static_cast<int32_t>((h & 0x7fffffffu) % static_cast<uint32_t>(N));
}

int64_t truncate_i64(int64_t value, int32_t W) noexcept {
    assert(W > 0);
    if (W <= 0) return value;
    const int64_t w = static_cast<int64_t>(W);
    const int64_t r = ((value % w) + w) % w;
    return value - r;
}

namespace {

bool starts_with(const char* s, uint32_t len, const char* prefix) noexcept {
    assert(s != nullptr && prefix != nullptr);
    const uint32_t pl = static_cast<uint32_t>(std::strlen(prefix));
    if (len < pl) return false;
    return std::memcmp(s, prefix, pl) == 0;
}

bool parse_paren_int(const char* s, uint32_t len, const char* prefix,
                     int32_t* out) noexcept {
    assert(s != nullptr && prefix != nullptr && out != nullptr);
    const uint32_t pl = static_cast<uint32_t>(std::strlen(prefix));
    if (len < pl + 3u) return false;
    if (std::memcmp(s, prefix, pl) != 0) return false;
    if (s[pl] != '[') return false;
    if (s[len - 1u] != ']') return false;
    int32_t v = 0;
    for (uint32_t i = pl + 1u; i + 1u < len; ++i) {
        const char c = s[i];
        if (c < '0' || c > '9') return false;
        v = v * 10 + (c - '0');
    }
    *out = v;
    return true;
}

}  // namespace

Transform transform_parse(const char* s, uint32_t len) noexcept {
    assert(s != nullptr || len == 0);
    Transform t{};
    t.kind = TransformKind::kUnknown;
    t.param = 0;
    if (len == 0) return t;
    if (len == 8 && std::memcmp(s, "identity", 8) == 0) {
        t.kind = TransformKind::kIdentity; return t;
    }
    if (len == 4 && std::memcmp(s, "year", 4) == 0) {
        t.kind = TransformKind::kYear; return t;
    }
    if (len == 5 && std::memcmp(s, "month", 5) == 0) {
        t.kind = TransformKind::kMonth; return t;
    }
    if (len == 3 && std::memcmp(s, "day", 3) == 0) {
        t.kind = TransformKind::kDay; return t;
    }
    if (len == 4 && std::memcmp(s, "hour", 4) == 0) {
        t.kind = TransformKind::kHour; return t;
    }
    if (len == 4 && std::memcmp(s, "void", 4) == 0) {
        t.kind = TransformKind::kVoid; return t;
    }
    if (starts_with(s, len, "bucket")) {
        int32_t n = 0;
        if (parse_paren_int(s, len, "bucket", &n)) {
            t.kind = TransformKind::kBucket; t.param = n; return t;
        }
    }
    if (starts_with(s, len, "truncate")) {
        int32_t w = 0;
        if (parse_paren_int(s, len, "truncate", &w)) {
            t.kind = TransformKind::kTruncate; t.param = w; return t;
        }
    }
    return t;
}

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
