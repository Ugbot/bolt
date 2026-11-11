// bolt/lakehouse/iceberg/puffin.h — Puffin statistics files (G2ICE-450):
// footer blobs, apache-datasketches-theta-v1 NDV, and a table column's NDV
// from its snapshot's statistics file.
#pragma once

#include <cstdint>

#include "bolt/lakehouse/iceberg/metadata.h"

namespace bolt { class Arena; }

namespace bolt {
namespace lakehouse {
namespace iceberg {

struct TableHandle;

inline constexpr const char* kThetaBlobType = "apache-datasketches-theta-v1";

// Parse a Puffin file's footer payload (the UTF-8 JSON FileMetadata) into
// arena-owned blob records. False on malformed JSON.
bool puffin_footer_parse(const uint8_t* json, uint32_t len, Arena* a, BlobMetadata** out,
                         uint32_t* n) noexcept;

// The footer blobs of a whole Puffin file (magic, payload size, flags,
// magic). False on a malformed file or a compressed footer payload.
bool puffin_file_blobs(const uint8_t* file, uint64_t len, Arena* a, BlobMetadata** out,
                       uint32_t* n) noexcept;

// Distinct-value estimate of a serialized compact theta sketch (DataSketches
// serial version 3, or 4 compressed). False on any other layout.
bool theta_sketch_estimate(const uint8_t* blob, uint64_t len, double* out) noexcept;

// NDV of `field_id` at `snapshot_id` from the table's statistics: a theta
// blob over that one field, its `ndv` property, else its sketch read from the
// statistics file when the file is at most `max_file_bytes`. False when the
// table has no such blob or the file is over the budget / unreadable.
bool iceberg_table_column_ndv(TableHandle* h, int64_t snapshot_id, int32_t field_id,
                              uint64_t max_file_bytes, Arena* scratch, int64_t* out) noexcept;

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
