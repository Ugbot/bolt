// bolt/lakehouse/iceberg/deletion_vector.h — Iceberg v3 deletion vectors (Puffin `deletion-vector-v1` blobs).
//
// Blob layout (spec, Puffin "deletion-vector-v1"): u32 BE length of (magic + bitmap), magic D1 D3 39 64, the 64-bit portable
// roaring bitmap (u64 LE bucket count, then per bucket a u32 LE high key and a 32-bit portable bitmap), u32 BE CRC-32 of
// (magic + bitmap). Manifests locate a DV by the containing file, `content_offset` and `content_size_in_bytes`.
#pragma once

#include <cstdint>

#include "bolt/bolt_arena.h"
#include "bolt/ingest/bolt_roaring.h"

namespace bolt {
namespace lakehouse {
namespace iceberg {

static constexpr uint32_t kDvMagic0 = 0xD1u, kDvMagic1 = 0xD3u, kDvMagic2 = 0x39u, kDvMagic3 = 0x64u;

// Decode and verify a whole blob (length field, magic, CRC). False on any mismatch or truncation; never a partial set.
bool iceberg_dv_decode(const uint8_t* blob, uint64_t blob_len, Arena* arena,
                       bolt::ingest::RoaringBitmap64* out) noexcept;

bool iceberg_dv_contains(const bolt::ingest::RoaringBitmap64* dv, uint64_t pos) noexcept;

// Encode strictly increasing file positions (each < 2^32; larger positions need more buckets and are refused) as a blob.
// Returns the blob in `arena`. Used by the v3 write path and by tests.
bool iceberg_dv_encode(const uint32_t* sorted_unique_positions, uint64_t n, Arena* arena,
                       const uint8_t** out_blob, uint64_t* out_len) noexcept;

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
