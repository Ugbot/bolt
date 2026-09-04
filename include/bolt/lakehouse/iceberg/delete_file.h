// bolt/lakehouse/iceberg/delete_file.h — Iceberg V2 delete-file references.
//
// V2 introduces equality deletes (content=1) and position deletes (content=2).
// W4 implements:
//   - position deletes:  (file_path, pos) sorted by file_path → binary search.
//   - equality deletes:  single-int64 equality column (Iceberg's most common
//                        case for upserts); multi-col TODO(W5).
// Tiger Style: PODs, fixed caps, noexcept.

#pragma once

#include <cstdint>

#include "bolt/lakehouse/iceberg/statistics.h"

namespace bolt {
namespace lakehouse {
namespace iceberg {

static constexpr uint32_t kIcebergMaxPath = 1024u;
static constexpr uint32_t kIcebergMaxDeleteFiles = 256u;

// Iceberg v2 `data_file.content`. THE ORDER IS THE SPEC'S, NOT ALPHABETICAL:
// 0 = data, 1 = POSITION deletes, 2 = EQUALITY deletes. bolt had 1 and 2
// swapped, which is a mislabelling no in-tree test could see because the read
// path declined BOTH kinds identically -- a position-delete file bolt wrote
// was read by pyiceberg as an equality delete and rejected outright
// ("PyIceberg does not yet support equality deletes"), i.e. the delete was
// silently not applied. Only an external reader can catch a wrong constant.
enum class FileContent : uint8_t {
    kData            = 0,
    kPositionDeletes = 1,
    kEqualityDeletes = 2,
};

// `manifest_file.content` is a DIFFERENT, two-valued enum: 0 = a manifest of
// data files, 1 = a manifest of delete files (of either kind). Spelled out
// here so it is never confused with FileContent, which is what happens when
// the two share names.
enum class ManifestContent : uint8_t {
    kDataManifest   = 0,
    kDeleteManifest = 1,
};

struct DeleteFileRef {
    FileContent content;
    uint8_t     _pad[3];
    int32_t     spec_id;
    char        path[kIcebergMaxPath];
    int64_t     file_size_in_bytes;
    int64_t     record_count;
    int32_t     equality_ids[8];
    uint32_t    n_equality_ids;
    uint32_t    _pad2;
};

struct PositionDeleteEntry {
    char     file_path[kIcebergMaxPath];
    int64_t  pos;
};

struct EqualityDeleteI64 {
    int32_t field_id;
    int32_t _pad;
    int64_t value;
};

struct PositionDeleteSet {
    PositionDeleteEntry* entries;
    uint32_t             n;
    uint32_t             cap;
};

struct EqualityDeleteSetI64 {
    EqualityDeleteI64* entries;
    uint32_t           n;
    uint32_t           cap;
};

bool position_delete_set_init(PositionDeleteSet* s, PositionDeleteEntry* buf,
                              uint32_t cap) noexcept;
bool position_delete_set_contains(const PositionDeleteSet* s,
                                  const char* file_path,
                                  int64_t pos) noexcept;

bool equality_delete_set_init(EqualityDeleteSetI64* s, EqualityDeleteI64* buf,
                              uint32_t cap) noexcept;
bool equality_delete_set_contains(const EqualityDeleteSetI64* s,
                                  int32_t field_id, int64_t value) noexcept;

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
