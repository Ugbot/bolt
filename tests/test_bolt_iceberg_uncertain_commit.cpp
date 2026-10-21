// A store accepts a conditional object, then loses its acknowledgement.
// The Iceberg writer must preserve every possibly referenced object.
#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_port.h"
#include "bolt/crypto/sigv4.h"
#include "bolt/lakehouse/catalog.h"
#include "bolt/lakehouse/iceberg/scan.h"
#include "bolt/lakehouse/iceberg/view.h"
#include "bolt/lakehouse/iceberg/writer.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {
using namespace bolt::lakehouse;
using namespace bolt::lakehouse::iceberg;

enum class Fault { None, AcceptedLoss, UnacceptedLoss, Refused };

struct FaultStore {
    FilesystemObjectStore fs{};
    ObjectStore underlying{};
    ObjectStore store{};
    Fault fault = Fault::None;
    std::string match;
    std::string fail_get;
    int list_status = kOsOk;
    uint32_t conditional_calls = 0;
    uint32_t deletes = 0;
    uint32_t injected = 0;

    static FaultStore& self(void* p) { return *static_cast<FaultStore*>(p); }
    static int get(void* p, const char* k, bolt::Arena* a, const uint8_t** d,
                   uint64_t* n) noexcept {
        auto& s = self(p);
        if (s.fail_get == k) return kOsIoError;
        return os_get(&s.underlying, k, a, d, n);
    }
    static int put(void* p, const char* k, const uint8_t* d, uint64_t n) noexcept {
        return os_put(&self(p).underlying, k, d, n);
    }
    static int list(void* p, const char* k, ObjectEntry* e, uint32_t c,
                    uint32_t* n) noexcept {
        auto& s = self(p);
        if (s.list_status != kOsOk) return s.list_status;
        return os_list(&s.underlying, k, e, c, n);
    }
    static int remove(void* p, const char* k) noexcept {
        auto& s = self(p);
        ++s.deletes;
        return os_delete(&s.underlying, k);
    }
    static int head(void* p, const char* k, ObjectMeta* m) noexcept {
        return os_head(&self(p).underlying, k, m);
    }
    static int create(void* p, const char* k, const uint8_t* d, uint64_t n) noexcept {
        auto& s = self(p);
        ++s.conditional_calls;
        if (s.fault != Fault::None && std::strncmp(k, s.match.c_str(), s.match.size()) == 0) {
            ++s.injected;
            if (s.fault == Fault::Refused) return kOsExists;
            if (s.fault == Fault::UnacceptedLoss) return kOsWriteUncertain;
            const int rc = os_put_if_absent(&s.underlying, k, d, n);
            return rc == kOsOk ? kOsWriteUncertain : rc;
        }
        return os_put_if_absent(&s.underlying, k, d, n);
    }
    bool init(const char* root) {
        static const ObjectStoreVT vt{get, put, list, remove, head, create, nullptr};
        if (!filesystem_object_store_init(&fs, root, &underlying)) return false;
        store = ObjectStore{&vt, this};
        return true;
    }
};

class IcebergUncertainCommit : public ::testing::Test {
protected:
    bolt::Arena arena;
    std::string base, root;
    bool owns_root = false;
    FaultStore store;
    SchemaField field{};
    Schema schema{};
    TableHandle* table = nullptr;
    bolt::BoltBatch batch{};
    std::array<int64_t, 4> values{{7, -19, 123456789, 0}};

    void SetUp() override {
        static std::atomic<uint32_t> next{0};
        const std::string name = "bolt_iceberg_uncertain_" + std::to_string(bolt_getpid()) +
                                 "_" + std::to_string(next.fetch_add(1));
        base = (std::filesystem::temp_directory_path() / name).generic_string();
        root = (std::filesystem::path(base) / "default" / "t").generic_string();
        std::error_code ec;
        ASSERT_TRUE(std::filesystem::create_directory(base, ec));
        owns_root = true;
        ASSERT_TRUE(std::filesystem::create_directories(root, ec));
        ASSERT_FALSE(ec);
        ASSERT_TRUE(store.init(root.c_str()));
        field.id = 1;
        field.required = true;
        std::strcpy(field.name, "id");
        std::strcpy(field.type, "long");
        schema.fields = &field;
        schema.cap_fields = 1;
        schema.n_fields = 1;
        bolt::BoltBatch::init_empty(&batch);
        batch.arena = &arena;
        ASSERT_TRUE(bolt::BoltBatch::alloc_columns(&batch, &arena, 1));
        ASSERT_TRUE(batch.schema.add_field("id", bolt::BoltType::Int64, false));
        batch.num_rows = static_cast<int64_t>(values.size());
        batch.num_cols = 1;
        auto& col = batch.columns[batch.read_epoch][0];
        col.type = bolt::BoltType::Int64;
        col.length = batch.num_rows;
        col.data = values.data();
    }
    void TearDown() override {
        if (!owns_root) return;
        std::error_code ec;
        std::filesystem::remove_all(base, ec);
        EXPECT_FALSE(ec);
    }
    void create_table() {
        ASSERT_TRUE(table_create(&table, &arena, &store.store, root.c_str(), &schema,
                                 nullptr, nullptr, nullptr));
        ASSERT_NE(table, nullptr);
    }
    AppendHandle* stage() {
        AppendHandle* append = nullptr;
        if (!append_open(&append, table) || !append_write(append, &batch)) return nullptr;
        return append;
    }
    uint32_t count_files(const char* prefix) {
        std::array<ObjectEntry, 16> entries{};
        uint32_t count = 0;
        EXPECT_EQ(os_list(&store.underlying, prefix, entries.data(),
                          static_cast<uint32_t>(entries.size()), &count), kOsOk);
        return count;
    }
    void check_receipt(const CommitCandidate& receipt, CommitObjectKind kind) {
        EXPECT_EQ(receipt.kind, kind);
        EXPECT_EQ(receipt.base_metadata_version, 1);
        EXPECT_EQ(receipt.candidate_metadata_version, 2);
        EXPECT_STREQ(receipt.table_uuid, table_metadata(table)->table_uuid);
        EXPECT_EQ(std::string(receipt.object_location), root + "/" + receipt.object_key);
        bolt::Arena scratch;
        const uint8_t* data = nullptr;
        uint64_t length = 0;
        ASSERT_EQ(os_get(&store.underlying, receipt.object_key, &scratch, &data, &length), kOsOk);
        EXPECT_EQ(length, receipt.length);
        char hash[65]{};
        bolt::crypto::sha256_hex(data, length, hash);
        EXPECT_STREQ(hash, receipt.sha256);
    }
    void scan_values(int64_t skip_position = -1) {
        FilesystemCatalog fs{};
        Catalog catalog{};
        ASSERT_TRUE(filesystem_catalog_init(&fs, base.c_str(), &catalog));
        bolt::Arena scratch;
        TableHandle* read = nullptr;
        ASSERT_TRUE(iceberg_table_open(&read, &scratch, &catalog, "default", "t"));
        ReadOptions options{};
        read_options_init(&options);
        ScanHandle* scan = nullptr;
        ASSERT_TRUE(iceberg_scan_open(&scan, read, &options));
        std::vector<int64_t> got;
        bool eof = false;
        for (uint32_t i = 0; i < 16u && !eof; ++i) {
            bolt::BoltBatch output{};
            ASSERT_TRUE(iceberg_scan_next_batch(scan, &output, &eof));
            if (eof) break;
            const auto* ids = static_cast<const int64_t*>(output.columns[output.read_epoch][0].data);
            ASSERT_LE(output.num_rows, static_cast<int64_t>(values.size()));
            got.insert(got.end(), ids, ids + output.num_rows);
        }
        ASSERT_TRUE(eof);
        std::vector<int64_t> expected(values.begin(), values.end());
        if (skip_position >= 0) {
            ASSERT_LT(skip_position, static_cast<int64_t>(expected.size()));
            expected.erase(expected.begin() + static_cast<std::ptrdiff_t>(skip_position));
        }
        std::sort(got.begin(), got.end());
        std::sort(expected.begin(), expected.end());
        EXPECT_EQ(got, expected);
        iceberg_scan_close(scan);
        iceberg_table_close(read);
    }
};

TEST_F(IcebergUncertainCommit, AcceptedMetadataLossPreservesReferencedObjectsAndRows) {
    create_table();
    ASSERT_NE(table, nullptr);
    AppendHandle* append = stage();
    ASSERT_NE(append, nullptr);
    store.fault = Fault::AcceptedLoss;
    store.match = "metadata/v2.metadata.json";
    EXPECT_FALSE(append_commit(append));
    ASSERT_EQ(table_last_commit_error(table), CommitError::kUncertain);
    CommitCandidate receipt{};
    ASSERT_TRUE(table_uncertain_commit(table, &receipt));
    check_receipt(receipt, CommitObjectKind::kMetadata);
    EXPECT_GT(receipt.snapshot_id, 0);
    EXPECT_EQ(store.injected, 1u);
    EXPECT_EQ(store.deletes, 0u);
    EXPECT_EQ(count_files("data/"), 1u);
    EXPECT_EQ(count_files("metadata/manifest-"), 1u);
    EXPECT_EQ(count_files("metadata/snap-"), 1u);
    scan_values();  // Reopen through the store, not the writer's rolled-back handle.
}

TEST_F(IcebergUncertainCommit, UnacceptedMetadataLossStillRetainsCandidatesAndRefusesReuse) {
    create_table();
    ASSERT_NE(table, nullptr);
    AppendHandle* append = stage();
    ASSERT_NE(append, nullptr);
    store.fault = Fault::UnacceptedLoss;
    store.match = "metadata/v2.metadata.json";
    EXPECT_FALSE(append_commit(append));
    CommitCandidate before{};
    ASSERT_TRUE(table_uncertain_commit(table, &before));
    EXPECT_EQ(before.kind, CommitObjectKind::kMetadata);
    EXPECT_EQ(std::strlen(before.sha256), 64u);
    ObjectMeta absent{};
    ASSERT_EQ(os_head(&store.underlying, before.object_key, &absent), kOsNotFound);
    EXPECT_FALSE(absent.exists);
    const uint32_t calls = store.conditional_calls;
    EXPECT_FALSE(append_commit(append));
    EXPECT_FALSE(append_write(append, &batch));
    AppendHandle* refused = nullptr;
    EXPECT_FALSE(append_open(&refused, table));
    EXPECT_EQ(refused, nullptr);
    EXPECT_FALSE(table_overwrite(table, nullptr, &batch));
    EXPECT_FALSE(table_delete(table, nullptr));
    EXPECT_FALSE(table_rewrite(table, nullptr));
    EXPECT_FALSE(table_compact(table, 1024));
    EXPECT_FALSE(table_add_column(table, "new", bolt::BoltType::Int64, false));
    EXPECT_FALSE(table_drop_column(table, "id"));
    EXPECT_FALSE(table_rename_column(table, "id", "renamed"));
    PartitionSpec spec{};
    SortOrder order{};
    EXPECT_FALSE(table_update_partition_spec(table, &spec));
    EXPECT_FALSE(table_update_sort_order(table, &order));
    PositionDeleteEntry deletion{};
    std::strcpy(deletion.file_path, "unpublished.parquet");
    EXPECT_FALSE(table_delete_positions(table, &deletion, 1));
    EXPECT_FALSE(table_create_branch(table, "branch", 1));
    EXPECT_FALSE(table_create_tag(table, "tag", 1));
    EXPECT_FALSE(table_drop_ref(table, "tag"));
    EXPECT_FALSE(table_set_retention(table, 1, 1));
    EXPECT_FALSE(table_expire_snapshots(table, 1, 1));
    EXPECT_FALSE(view_replace(table, "sql", "select 1"));
    uint32_t orphans = 0;
    EXPECT_FALSE(table_remove_orphans(table, 0, false, nullptr, 0, &orphans));
    EXPECT_EQ(store.conditional_calls, calls);
    EXPECT_EQ(store.deletes, 0u);
    EXPECT_EQ(count_files("data/"), 1u);
    CommitCandidate after{};
    ASSERT_TRUE(table_uncertain_commit(table, &after));
    EXPECT_STREQ(before.sha256, after.sha256);
    EXPECT_STREQ(before.object_key, after.object_key);
    EXPECT_EQ(table_last_commit_error(table), CommitError::kUncertain);
}

TEST_F(IcebergUncertainCommit, UncertainManifestAndListPreserveCandidateData) {
    create_table();
    ASSERT_NE(table, nullptr);
    AppendHandle* append = stage();
    ASSERT_NE(append, nullptr);
    store.fault = Fault::AcceptedLoss;
    store.match = "metadata/manifest-";
    EXPECT_FALSE(append_commit(append));
    CommitCandidate receipt{};
    ASSERT_TRUE(table_uncertain_commit(table, &receipt));
    check_receipt(receipt, CommitObjectKind::kManifest);
    EXPECT_EQ(store.deletes, 0u);
    EXPECT_EQ(count_files("data/"), 1u);
    EXPECT_EQ(count_files("metadata/manifest-"), 1u);
}

TEST_F(IcebergUncertainCommit, UncertainManifestListDoesNotDeleteItsManifest) {
    create_table();
    ASSERT_NE(table, nullptr);
    AppendHandle* append = stage();
    ASSERT_NE(append, nullptr);
    store.fault = Fault::AcceptedLoss;
    store.match = "metadata/snap-";
    EXPECT_FALSE(append_commit(append));
    CommitCandidate receipt{};
    ASSERT_TRUE(table_uncertain_commit(table, &receipt));
    check_receipt(receipt, CommitObjectKind::kManifestList);
    EXPECT_EQ(store.deletes, 0u);
    EXPECT_EQ(count_files("data/"), 1u);
    EXPECT_EQ(count_files("metadata/manifest-"), 1u);
    EXPECT_EQ(count_files("metadata/snap-"), 1u);
}

TEST_F(IcebergUncertainCommit, DefinitiveMetadataConflictStillCleansUnpublishedObjects) {
    create_table();
    ASSERT_NE(table, nullptr);
    AppendHandle* append = stage();
    ASSERT_NE(append, nullptr);
    store.fault = Fault::Refused;
    store.match = "metadata/v2.metadata.json";
    EXPECT_FALSE(append_commit(append));
    EXPECT_EQ(table_last_commit_error(table), CommitError::kConflict);
    CommitCandidate receipt{};
    EXPECT_FALSE(table_uncertain_commit(table, &receipt));
    EXPECT_EQ(store.deletes, 3u);
    EXPECT_EQ(count_files("data/"), 0u);
    EXPECT_EQ(count_files("metadata/manifest-"), 0u);
    EXPECT_EQ(count_files("metadata/snap-"), 0u);
}

TEST_F(IcebergUncertainCommit, AcceptedDeleteMetadataLossPreservesDeleteAndPriorData) {
    create_table();
    ASSERT_NE(table, nullptr);
    AppendHandle* append = stage();
    ASSERT_NE(append, nullptr);
    PositionDeleteEntry deletion{};
    ASSERT_TRUE(append_commit_ex(append, deletion.file_path, sizeof(deletion.file_path), nullptr));
    deletion.pos = 1;
    store.fault = Fault::AcceptedLoss;
    store.match = "metadata/v3.metadata.json";
    EXPECT_FALSE(table_delete_positions(table, &deletion, 1));
    CommitCandidate receipt{};
    ASSERT_TRUE(table_uncertain_commit(table, &receipt));
    EXPECT_EQ(receipt.kind, CommitObjectKind::kMetadata);
    EXPECT_EQ(receipt.base_metadata_version, 2);
    EXPECT_EQ(receipt.candidate_metadata_version, 3);
    EXPECT_EQ(store.deletes, 0u);
    EXPECT_EQ(count_files("data/"), 2u);
    EXPECT_EQ(count_files("metadata/manifest-"), 2u);
    EXPECT_EQ(count_files("metadata/snap-"), 2u);
    scan_values(1);
}

TEST_F(IcebergUncertainCommit, AcceptedExpiryMetadataLossDoesNotDeletePriorSnapshotObjects) {
    create_table();
    ASSERT_NE(table, nullptr);
    AppendHandle* append = stage();
    ASSERT_NE(append, nullptr);
    ASSERT_TRUE(append_commit(append));
    ASSERT_TRUE(append_write(append, &batch));
    ASSERT_TRUE(append_commit(append));
    store.fault = Fault::AcceptedLoss;
    store.match = "metadata/v4.metadata.json";
    ExpireOptions options{};
    options.older_than_ms = INT64_MAX;
    options.retain_last = 1;
    options.delete_files = true;
    EXPECT_FALSE(table_expire_snapshots_ex(table, &options, nullptr));
    CommitCandidate receipt{};
    ASSERT_TRUE(table_uncertain_commit(table, &receipt));
    EXPECT_EQ(receipt.kind, CommitObjectKind::kMetadata);
    EXPECT_EQ(receipt.base_metadata_version, 3);
    EXPECT_EQ(receipt.candidate_metadata_version, 4);
    EXPECT_EQ(store.deletes, 0u);
    EXPECT_EQ(count_files("data/"), 2u);
    EXPECT_EQ(count_files("metadata/manifest-"), 2u);
    EXPECT_EQ(count_files("metadata/snap-"), 2u);
    TableHandle* reopened = nullptr;
    ASSERT_TRUE(table_open(&reopened, &arena, &store.underlying, root.c_str()));
    EXPECT_EQ(table_metadata(reopened)->n_snapshots, 1u);
    EXPECT_EQ(table_metadata(reopened)->current_snapshot_id, receipt.snapshot_id);
}

TEST_F(IcebergUncertainCommit, AcceptedInitialMetadataLossReturnsReceiptWithoutHandle) {
    store.fault = Fault::AcceptedLoss;
    store.match = "metadata/v1.metadata.json";
    CommitCandidate receipt{};
    EXPECT_FALSE(table_create_ex(&table, &arena, &store.store, root.c_str(), &schema,
                                 nullptr, nullptr, nullptr, &receipt));
    EXPECT_EQ(table, nullptr);
    EXPECT_EQ(receipt.kind, CommitObjectKind::kMetadata);
    EXPECT_EQ(receipt.base_metadata_version, 0);
    EXPECT_EQ(receipt.candidate_metadata_version, 1);
    EXPECT_STREQ(receipt.object_key, "metadata/v1.metadata.json");
    EXPECT_EQ(std::strlen(receipt.sha256), 64u);
    EXPECT_EQ(store.deletes, 0u);
    TableHandle* reopened = nullptr;
    ASSERT_TRUE(table_open(&reopened, &arena, &store.underlying, root.c_str()));
    EXPECT_STREQ(receipt.table_uuid, table_metadata(reopened)->table_uuid);
}

TEST_F(IcebergUncertainCommit, OversizeReceiptLocationRefusesBeforeConditionalWrite) {
    const std::string location = "/" + std::string(kIcebergMaxLocation - 4u, 'a');
    CommitCandidate receipt{};
    EXPECT_FALSE(table_create_ex(&table, &arena, &store.store, location.c_str(), &schema,
                                 nullptr, nullptr, nullptr, &receipt));
    EXPECT_EQ(table, nullptr);
    EXPECT_EQ(receipt.kind, CommitObjectKind::kNone);
    EXPECT_EQ(store.conditional_calls, 0u);
    EXPECT_EQ(store.deletes, 0u);
}

TEST_F(IcebergUncertainCommit, ReaderTreatsExistingStaleHintAsAdvisory) {
    create_table();
    ASSERT_NE(table, nullptr);
    AppendHandle* append = stage();
    ASSERT_NE(append, nullptr);
    ASSERT_TRUE(append_commit(append));
    const uint8_t hint = '1';
    ASSERT_EQ(os_put(&store.underlying, "metadata/version-hint.text", &hint, 1), kOsOk);
    scan_values();
}

TEST_F(IcebergUncertainCommit, ReaderFindsRetainedVersionBeyondGapAfterStaleHint) {
    create_table();
    ASSERT_NE(table, nullptr);
    AppendHandle* append = stage();
    ASSERT_NE(append, nullptr);
    ASSERT_TRUE(append_commit(append));
    ASSERT_TRUE(table_create_tag(table, "retained", table_metadata(table)->current_snapshot_id));
    ASSERT_EQ(os_delete(&store.underlying, "metadata/v2.metadata.json"), kOsOk);
    const uint8_t hint = '1';
    ASSERT_EQ(os_put(&store.underlying, "metadata/version-hint.text", &hint, 1), kOsOk);
    scan_values();
}

TEST_F(IcebergUncertainCommit, ReaderHintIoErrorsFailClosedInsteadOfSelectingAnOlderView) {
    create_table();
    ASSERT_NE(table, nullptr);
    AppendHandle* append = stage();
    ASSERT_NE(append, nullptr);
    ASSERT_TRUE(append_commit(append));
    const uint8_t hint = '1';
    ASSERT_EQ(os_put(&store.underlying, "metadata/version-hint.text", &hint, 1), kOsOk);
    for (const char* key : {"default/t/metadata/version-hint.text", "default/t/metadata/v1.metadata.json"}) {
        SCOPED_TRACE(key);
        FaultStore reader;
        ASSERT_TRUE(reader.init(base.c_str()));
        reader.fail_get = key;
        bolt::Arena scratch;
        TableHandle* read = nullptr;
        EXPECT_FALSE(iceberg_table_open_store(&read, &scratch, &reader.store, "default/t"));
    }
}

TEST_F(IcebergUncertainCommit, ReaderListingErrorCannotReturnTheHintedSnapshot) {
    create_table();
    ASSERT_NE(table, nullptr);
    AppendHandle* append = stage();
    ASSERT_NE(append, nullptr);
    ASSERT_TRUE(append_commit(append));
    FaultStore reader;
    ASSERT_TRUE(reader.init(base.c_str()));
    reader.list_status = kOsIoError;
    bolt::Arena scratch;
    TableHandle* read = nullptr;
    EXPECT_FALSE(iceberg_table_open_store(&read, &scratch, &reader.store, "default/t"));
}
}  // namespace
