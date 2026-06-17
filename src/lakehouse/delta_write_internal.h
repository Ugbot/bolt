// bolt/lakehouse/delta_write_internal.h — shared helpers used by the W3
// write translation units (update/delete/merge/restore/vacuum/optimize/
// checkpoint_write). Internal to src/lakehouse — not in include/.

#pragma once

#include <cstdint>

#include "bolt/lakehouse/catalog.h"
#include "bolt/lakehouse/object_store.h"

namespace bolt {
class Arena;
namespace lakehouse {

// TableHandle layout — MUST be byte-identical to the definition in
// delta_scan.cpp. Single ODR definition lives here; both delta_scan.cpp
// and the W3 writer TUs include this header.
namespace delta_writer_detail {
constexpr uint32_t kMaxFsRootInTable = 1024u;
constexpr uint32_t kMaxNsName        = 128u;
}  // namespace delta_writer_detail

struct TableHandle {
    Arena*                arena;
    Catalog*              catalog;
    char                  namespace_[delta_writer_detail::kMaxNsName];
    char                  name[delta_writer_detail::kMaxNsName];
    char                  table_rel[delta_writer_detail::kMaxFsRootInTable];
    FilesystemObjectStore fs_store;
    ObjectStore           os;
    bool                  os_ready;
    uint8_t               _pad[7];
};

namespace delta {

// Append a single commit body atomically. On a concurrent winner, returns
// false without altering inout_base_version. On success, bumps it.
bool delta_writer_commit_raw(TableHandle* th, const char* body,
                             uint32_t body_len,
                             int64_t* inout_base_version) noexcept;

int64_t delta_writer_latest_version(TableHandle* th) noexcept;

uint64_t delta_writer_now_ms() noexcept;

uint32_t delta_writer_escape_json(char* dst, uint32_t cap, uint32_t off,
                                   const char* src) noexcept;

bool delta_writer_log_key(const char* table_rel, int64_t version, char* out,
                           uint32_t cap) noexcept;

}  // namespace delta
}  // namespace lakehouse
}  // namespace bolt
