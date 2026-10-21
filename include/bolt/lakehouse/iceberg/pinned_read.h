// Open a retained metadata generation over an admitted immutable object graph.
#pragma once
#include "bolt/lakehouse/iceberg/scan.h"
#include "bolt/lakehouse/iceberg/metadata_io.h"
#include "bolt/lakehouse/object_store/pinned.h"

namespace bolt::lakehouse::iceberg {
struct PinnedTableRead {
    MetadataObjectIdentity object{};
    MetadataLoadLimits metadata_limits{};
    char sha256[65]{}; // expected complete retained JSON digest, lowercase hex
    int64_t snapshot_id = 0;
    SnapshotSelection selection = SnapshotSelection::Empty;
    // Explicit startup budgets for each scan's private footer/manifest arenas.
    uint32_t parquet_metadata_bytes = 0;
    uint32_t manifest_bytes = 0;
};

// No discovery or origin I/O. Validates/copies retained JSON with the owned
// metadata validator and checks the complete digest and directory metadata pin.
// Namespace must equal object.store_location; every referenced URI subsequently
// maps exactly once against that root, never relative to the table location.
// Exact selects the literal signed snapshot ID; Empty requires ID zero and
// absence of current-snapshot-id. Legacy selection is refused.
//
// source and its immutable directory outlive all handles; they are confined to
// the calling thread. Owner and metadata arenas are distinct, exclusively owned,
// frozen with finite backing at startup, disjoint from all inputs/source/IO and
// outputs. Metadata arena retains exact bytes/parsed arrays; owner holds handles
// and decoded batches. Each scan allocates finite private backing at startup.
// Opaque origin private storage is caller-excluded. Failure preserves *out;
// admitted failure can consume arena space. No ownership transfer.
//
// Null read options use the admitted selection. Supplied options must select the
// same snapshot explicitly, without timestamp fallback. Flat data fields bind
// by Iceberg/Parquet field IDs, including renamed/reordered physical columns.
// Current typed materialization admits boolean/int/long/float/double/string/
// binary/date and microsecond time/timestamp/timestamptz. Decimal/fixed/UUID,
// nested fields, missing-field defaults and type promotions refuse explicitly.
// Position deletes use reserved field IDs; equality deletes refuse. Supplied
// projection/predicate/include-deletes/include-partition-columns options also
// refuse until their strict semantics are implemented. No silent option ignore.
// Bolt batches retain the existing decoder's internal lanes (float widened to
// Float64, boolean to Int64, byte arrays to Utf8 storage). Metadata retains the
// original Iceberg logical declarations; exact external type egress is separate.
bool iceberg_table_open_pinned(TableHandle** out, Arena* owner, Arena* metadata,
    PinnedObjectStore* source, const PinnedTableRead* read,
    const uint8_t* metadata_json, uint64_t metadata_length) noexcept;
} // namespace bolt::lakehouse::iceberg
