// Optional staged data transport. Metadata still uses the table ObjectStore.
#pragma once
#include "bolt/lakehouse/iceberg/writer.h"

namespace bolt::lakehouse::iceberg {
inline constexpr uint32_t kIcebergStagedPathCap = 4096;
struct StagedDataFile {
    CommitCandidate candidate{};
    char local_path[kIcebergStagedPathCap]{};
    // SHA256 of uint64-LE-length-prefixed UTF8 strings: version marker
    // "bolt.iceberg.data.v1", key, location, UUID, lowercase SHA; then uint64-LE
    // length/base-version/candidate-version/snapshot. Local path is excluded.
    char operation_id[65]{};
    int cleanup_status = kOsNotImplemented; // local cleanup, distinct from commit
    int remote_cleanup_status = kOsNotImplemented;
    bool source_retained = false;
    bool published = false; // definite data ACK; metadata may still be uncertain
};
struct FileWriteIO {
    char staging_root[kIcebergStagedPathCap]{};
    uint64_t max_staged_file_bytes = 0; // explicit nonzero caller budget, no default
    void* context = nullptr;
    // Synchronous, called once with exact closed immutable bytes/identity.
    // The file pointer is borrowed only during this call; copy any receipt.
    // Callback/context are borrowed and thread-confined. It must publish only
    // if absent. Return WriteUncertain for ANY ambiguous remote outcome, even
    // failed cleanup/initiation; no implicit retries/rekeys. Persist prepared
    // source/receipt before external effects if restart reconciliation needs it.
    int (*publish_if_absent)(void*, const StagedDataFile*) noexcept = nullptr;
};
// Copy once after create/open, before any subsequent mutation/append_open.
// Requires a writer handle from table_create/table_open, not a scan handle.
// Private existing absolute staging root; caller excludes external mutation.
// Cap 4..min(INT64_MAX,SIZE_MAX,kSha256MaxInputBytes); encoder heap, retained
// aggregate disk usage and concurrency need independent caller budgets.
bool table_set_file_write_io(TableHandle*, const FileWriteIO*) noexcept;
// Last data receipt, including cleanup outcomes after a definite ACK/refusal.
// A retained source fences the next file-producing mutation on this handle.
// Checked close/flush is completeness, not fsync/crash durability. Caller owns
// durable prepared-receipt admission and retained-source recovery/inventory.
// Invalid writable aliases refuse unchanged. No clearing/retry/reconciliation.
bool table_staged_data_file(const TableHandle*, StagedDataFile*) noexcept;
// Same receipt only while the table has sticky uncertainty (including later
// manifest/metadata uncertainty). The earlier data receipt survives unchanged.
bool table_uncertain_data_file(const TableHandle*, StagedDataFile*) noexcept;
}  // namespace bolt::lakehouse::iceberg
