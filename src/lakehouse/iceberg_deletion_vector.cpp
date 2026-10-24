// Iceberg v3 deletion vectors: see bolt/lakehouse/iceberg/deletion_vector.h.
#include "bolt/lakehouse/iceberg/deletion_vector.h"

#include <cassert>
#include <cstdio>
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

bool iceberg_dv_positions(const bolt::ingest::RoaringBitmap64* dv, Arena* scratch, uint64_t* dst, uint64_t cap,
                          uint64_t* out_n) noexcept {
    assert(dv != nullptr && scratch != nullptr && out_n != nullptr);
    uint64_t n = 0;
    for (uint32_t b = 0; b < dv->n_buckets; ++b) {                    // bounded: the bitmap's buckets
        const bolt::ingest::RoaringBucket& k = dv->buckets[b];
        const uint64_t c = bolt::ingest::roaring_cardinality(&k.map);
        if (n + c > cap) return false;
        uint32_t* lo = scratch->allocate_array<uint32_t>(static_cast<size_t>(c > 0u ? c : 1u));
        uint64_t got = 0;
        if (lo == nullptr || !bolt::ingest::roaring_to_sorted_array(&k.map, lo, c, &got) || got != c) return false;
        for (uint64_t i = 0; i < c; ++i) dst[n + i] = (static_cast<uint64_t>(k.high32) << 32) | lo[i];   // bounded: c
        n += c;
    }
    *out_n = n;
    return true;
}

namespace {

// Appends `s` with JSON escapes for `"` and `\`; false past cap.
bool json_put(char* buf, uint64_t cap, uint64_t* pos, const char* s, bool escape) noexcept {
    assert(buf != nullptr && pos != nullptr && s != nullptr);
    for (const char* p = s; *p != '\0'; ++p) {                        // bounded: the string
        const bool esc = escape && (*p == '"' || *p == '\\');
        if (*pos + (esc ? 2u : 1u) >= cap) return false;
        if (esc) buf[(*pos)++] = '\\';
        buf[(*pos)++] = *p;
    }
    return true;
}

}  // namespace

bool puffin_write_deletion_vectors(const DvBlobIn* blobs, uint32_t n_blobs, const char* created_by, Arena* arena,
                                   const uint8_t** out, uint64_t* out_len, DvBlobPlacement* placed) noexcept {
    assert(arena != nullptr && out != nullptr && out_len != nullptr);
    if (arena == nullptr || out == nullptr || out_len == nullptr || (n_blobs != 0u && (blobs == nullptr || placed == nullptr)))
        return false;
    const uint8_t** bb = arena->allocate_array<const uint8_t*>(n_blobs + 1u);
    uint64_t* bl = arena->allocate_array<uint64_t>(n_blobs + 1u);
    if (bb == nullptr || bl == nullptr) return false;
    uint64_t body = 4u, json_cap = 128u + (created_by != nullptr ? std::strlen(created_by) * 2u : 0u);
    for (uint32_t i = 0; i < n_blobs; ++i) {                          // bounded: n_blobs
        if (blobs[i].referenced_data_file == nullptr ||
            !iceberg_dv_encode(blobs[i].positions, blobs[i].n, arena, &bb[i], &bl[i])) return false;
        placed[i].offset = static_cast<int64_t>(body);
        placed[i].length = static_cast<int64_t>(bl[i]);
        body += bl[i];
        json_cap += 256u + 2u * std::strlen(blobs[i].referenced_data_file);
    }
    char* json = arena->allocate_array<char>(static_cast<size_t>(json_cap));
    if (json == nullptr) return false;
    uint64_t jp = 0;
    bool ok = json_put(json, json_cap, &jp, "{\"blobs\":[", false);
    for (uint32_t i = 0; ok && i < n_blobs; ++i) {                    // bounded: n_blobs
        char num[160];
        std::snprintf(num, sizeof(num), "%s{\"type\":\"deletion-vector-v1\",\"fields\":[2147483645],\"snapshot-id\":-1,"
                      "\"sequence-number\":-1,\"offset\":%lld,\"length\":%lld,\"properties\":{\"referenced-data-file\":\"",
                      i == 0u ? "" : ",", static_cast<long long>(placed[i].offset), static_cast<long long>(placed[i].length));
        char card[64];
        std::snprintf(card, sizeof(card), "\",\"cardinality\":\"%llu\"}}", static_cast<unsigned long long>(blobs[i].n));
        ok = json_put(json, json_cap, &jp, num, false) && json_put(json, json_cap, &jp, blobs[i].referenced_data_file, true) &&
             json_put(json, json_cap, &jp, card, false);
    }
    ok = ok && json_put(json, json_cap, &jp, "],\"properties\":{\"created-by\":\"", false) &&
         json_put(json, json_cap, &jp, created_by != nullptr ? created_by : "bolt", true) &&
         json_put(json, json_cap, &jp, "\"}}", false);
    if (!ok) return false;
    const uint64_t total = body + 4u + jp + 4u + 4u + 4u;
    uint8_t* f = arena->allocate_array<uint8_t>(static_cast<size_t>(total));
    if (f == nullptr) return false;
    static const uint8_t kMagic[4] = {0x50, 0x46, 0x41, 0x31};      // "PFA1"
    std::memcpy(f, kMagic, 4);
    for (uint32_t i = 0; i < n_blobs; ++i) std::memcpy(f + placed[i].offset, bb[i], bl[i]);   // bounded: n_blobs
    uint8_t* t = f + body;
    std::memcpy(t, kMagic, 4);
    std::memcpy(t + 4, json, jp);
    const uint32_t plen = static_cast<uint32_t>(jp);
    for (int b = 0; b < 4; ++b) t[4 + jp + b] = static_cast<uint8_t>(plen >> (8 * b));   // LE payload size
    std::memset(t + 8 + jp, 0, 4);                                    // flags: payload not compressed
    std::memcpy(t + 12 + jp, kMagic, 4);
    *out = f;
    *out_len = total;
    assert(total >= 12u);
    return true;
}

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
