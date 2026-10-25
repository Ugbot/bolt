// bolt/lakehouse/iceberg_summary_types.inc — snapshot summary counters (spec Appendix F, "Optional Snapshot Summary
// Fields"; G2ICE-280). Included inside iceberg_writer.cpp before TableHandle. Not a standalone TU.
//
// Deltas describe what one commit added and removed and are written only when non-zero (as Iceberg-Java does);
// totals describe the snapshot's live files and are written whenever they are known.

enum SumKey : uint32_t {
    kSumAddedDataFiles = 0, kSumDeletedDataFiles, kSumAddedDeleteFiles, kSumAddedEqDeleteFiles,
    kSumRemovedEqDeleteFiles, kSumAddedPosDeleteFiles, kSumRemovedPosDeleteFiles, kSumRemovedDeleteFiles,
    kSumAddedRecords, kSumDeletedRecords, kSumAddedFilesSize, kSumRemovedFilesSize, kSumAddedPosDeletes,
    kSumRemovedPosDeletes, kSumAddedEqDeletes, kSumRemovedEqDeletes,
    kSumFirstTotal,
    kSumTotalRecords = kSumFirstTotal, kSumTotalFilesSize, kSumTotalDataFiles, kSumTotalDeleteFiles,
    kSumTotalPosDeletes, kSumTotalEqDeletes,
    kSumKeys
};

constexpr const char* kSumNames[kSumKeys] = {
    "added-data-files", "deleted-data-files", "added-delete-files", "added-equality-delete-files",
    "removed-equality-delete-files", "added-position-delete-files", "removed-position-delete-files",
    "removed-delete-files", "added-records", "deleted-records", "added-files-size", "removed-files-size",
    "added-position-deletes", "removed-position-deletes", "added-equality-deletes", "removed-equality-deletes",
    "total-records", "total-files-size", "total-data-files", "total-delete-files", "total-position-deletes",
    "total-equality-deletes",
};

struct SnapSummary {
    int64_t v[kSumKeys];
    int64_t changed_partitions;
    bool    has_totals;
    bool    has_lineage;      // v3: first_row_id / added_rows below are the snapshot's row lineage (G2ICE-283)
    uint8_t _pad[6];
    int64_t first_row_id;
    int64_t added_rows;
};

// One snapshot this handle committed, so a later re-emit of metadata.json keeps its summary.
struct SumRec {
    int64_t     snapshot_id;
    SnapSummary s;
};
