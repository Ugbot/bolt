// Iceberg v3 deletion vectors: see bolt/lakehouse/iceberg/deletion_vector.h.
#include "bolt/lakehouse/iceberg/deletion_vector.h"

#include <cassert>
#include <cstring>

#include "bolt/ingest/bolt_deflate.h"

namespace bolt {
namespace lakehouse {
namespace iceberg {

namespace {

uint32_t be32(const uint8_t* p) noexcept {
    return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}

void put_be32(uint8_t* p, uint32_t v) noexcept {
    p[0] = static_cast<uint8_t>(v >> 24); p[1] = static_cast<uint8_t>(v >> 16);
    p[2] = static_cast<uint8_t>(v >> 8);  p[3] = static_cast<uint8_t>(v);
}

}  // namespace

bool iceberg_dv_decode(const uint8_t* blob, uint64_t blob_len, Arena* arena,
                       bolt::ingest::RoaringBitmap64* out) noexcept {
    assert(arena != nullptr && out != nullptr);
    if (blob == nullptr || blob_len < 4u + 4u + 8u + 4u) return false;
    const uint64_t body = be32(blob);                       // magic + bitmap
    if (body < 4u + 8u || 4u + body + 4u != blob_len) return false;
    const uint8_t* m = blob + 4;
    if (m[0] != kDvMagic0 || m[1] != kDvMagic1 || m[2] != kDvMagic2 || m[3] != kDvMagic3) return false;
    if (bolt::ingest::crc32_ieee(m, body, 0u) != be32(blob + 4 + body)) return false;
    if (!bolt::ingest::roaring_deserialize_r64(m + 4, body - 4u, arena, out)) return false;
    return true;
}

bool iceberg_dv_contains(const bolt::ingest::RoaringBitmap64* dv, uint64_t pos) noexcept {
    assert(dv != nullptr);
    return dv != nullptr && bolt::ingest::roaring_contains64(dv, pos);
}

bool iceberg_dv_encode(const uint32_t* sorted_unique_positions, uint64_t n, Arena* arena,
                       const uint8_t** out_blob, uint64_t* out_len) noexcept {
    assert(arena != nullptr && out_blob != nullptr && out_len != nullptr);
    if (arena == nullptr || out_blob == nullptr || out_len == nullptr || (n != 0 && sorted_unique_positions == nullptr)) return false;
    const uint64_t cap = bolt::ingest::roaring_serialize_r64_bound(n);
    uint8_t* buf = arena->allocate_array<uint8_t>(static_cast<size_t>(4 + 4 + cap + 4));
    if (buf == nullptr) return false;
    uint64_t used = 0;
    if (!bolt::ingest::roaring_serialize_r64_single_bucket(sorted_unique_positions, n, arena, buf + 8, cap, &used)) return false;
    const uint64_t body = 4u + used;
    put_be32(buf, static_cast<uint32_t>(body));
    buf[4] = kDvMagic0; buf[5] = kDvMagic1; buf[6] = kDvMagic2; buf[7] = kDvMagic3;
    put_be32(buf + 4 + body, bolt::ingest::crc32_ieee(buf + 4, body, 0u));
    *out_blob = buf;
    *out_len = 4u + body + 4u;
    return true;
}

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
