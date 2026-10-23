#include "bolt/bolt_column.h"
#include "bolt/bolt_port.h"
#include "bolt/crypto/sha256.h"
#include "bolt/crypto/sigv4.h"
#include "bolt/io/bolt_file_io.h"
#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt/ingest/bolt_parquet_source.h"
#include "bolt/lakehouse/iceberg/write_io.h"
#include <gtest/gtest.h>
#include <array>
#include <atomic>
#include <bit>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <new>
#include <string>
#include <vector>

namespace {
using namespace bolt::lakehouse;
using namespace bolt::lakehouse::iceberg;
namespace pq = bolt::ingest::parquet;
template <size_t N>
bool schema_field_ids(const uint8_t* data, uint64_t size, std::array<int64_t, N>* ids) {
    if (size < 12) return false;
    ids->fill(INT64_MIN);  // Distinguish an omitted field from field ID zero.
    uint32_t length = 0;
    for (uint32_t i = 0; i < 4; ++i) length |= static_cast<uint32_t>(data[size - 8u + i]) << (8u * i);
    if (length > size - 12u) return false;
    pq::TcCursor cursor{data + size - 8u - length, data + size - 8u};
    int16_t field = 0; uint8_t type = 0;
    for (uint32_t step = 0; step < 32 && pq::tc_field(&cursor, &field, &type); ++step) {
        if (field != 2) { if (!pq::tc_skip(&cursor, type, 0)) return false; continue; }
        uint8_t element = 0; uint32_t count = 0;
        if (type != pq::kTcList || !pq::tc_list(&cursor, &element, &count) || element != pq::kTcStruct || count != N) return false;
        for (uint32_t col = 0; col < count; ++col) {
            int16_t schema_field = 0; uint8_t schema_type = 0;
            for (uint32_t f = 0; f < 32 && pq::tc_field(&cursor, &schema_field, &schema_type); ++f) {
                if (schema_field == 9) {
                    if (schema_type != pq::kTcI32 || !pq::tc_zigzag(&cursor, &(*ids)[col])) return false;
                } else if (!pq::tc_skip(&cursor, schema_type, 0)) return false;
            }
        }
        return true;
    }
    return false;
}
struct FileSource {
    bolt::io::ReadFile file{};
    uint32_t largest = 0;
    uint64_t calls = 0;
    ~FileSource() { bolt::io::file_close(&file); }
    static pq::ParquetSourceStatus read(void* context, uint64_t offset, uint32_t length, uint8_t* dst) noexcept {
        auto& source = *static_cast<FileSource*>(context);
        if (length == 0 || length > 4096 || source.calls >= 100000u) return pq::ParquetSourceStatus::BadArg;
        ++source.calls; source.largest = source.largest < length ? length : source.largest;
        uint64_t got = 0;
        return bolt::io::file_pread(source.file, offset, dst, length, &got) == bolt::io::FileIoStatus::kOk && got == length ?
            pq::ParquetSourceStatus::Ok : pq::ParquetSourceStatus::IoError;
    }
};
struct Store {
    FilesystemObjectStore fs{};
    ObjectStore origin{}, adapter{};
    bool metadata_loss = false;
    uint32_t creates = 0, deletes = 0;
    static Store& self(void* p) { return *static_cast<Store*>(p); }
    static int get(void* p, const char* k, bolt::Arena* a, const uint8_t** d, uint64_t* n) noexcept {
        return os_get(&self(p).origin, k, a, d, n);
    }
    static int put(void* p, const char* k, const uint8_t* d, uint64_t n) noexcept { return os_put(&self(p).origin, k, d, n); }
    static int list(void* p, const char* k, ObjectEntry* e, uint32_t c, uint32_t* n) noexcept {
        return os_list(&self(p).origin, k, e, c, n);
    }
    static int remove(void* p, const char* k) noexcept { auto& s = self(p); ++s.deletes; return os_delete(&s.origin, k); }
    static int head(void* p, const char* k, ObjectMeta* m) noexcept { return os_head(&self(p).origin, k, m); }
    static int create(void* p, const char* k, const uint8_t* d, uint64_t n) noexcept {
        auto& s = self(p); ++s.creates;
        const int rc = os_put_if_absent(&s.origin, k, d, n);
        return s.metadata_loss && std::strcmp(k, "metadata/v2.metadata.json") == 0 && rc == kOsOk ? kOsWriteUncertain : rc;
    }
    bool init(const char* root) {
        static const ObjectStoreVT vt{get, put, list, remove, head, create, nullptr};
        if (!filesystem_object_store_init(&fs, root, &origin)) return false;
        adapter = {&vt, this}; return true;
    }
};
struct Publisher {
    std::filesystem::path remote;
    int status = kOsOk;
    bool accepted_loss = false, obstruct_cleanup = false;
    uint32_t calls = 0;
    StagedDataFile received{};
    static int publish(void* context, const StagedDataFile* file) noexcept {
        auto& self = *static_cast<Publisher*>(context); ++self.calls; self.received = *file;
        if (file->candidate.kind != CommitObjectKind::kDataFile || file->candidate.length < 12 ||
            std::strlen(file->candidate.sha256) != 64 || std::strlen(file->operation_id) != 64 ||
            !file->source_retained || file->published) return kOsBadArg;
        if (self.status != kOsOk && !self.accepted_loss) {
            if (self.status == kOsExists) {
                std::error_code ec;
                const auto destination = self.remote / file->candidate.object_key;
                (void)std::filesystem::create_directories(destination.parent_path(), ec);
                if (ec) return kOsIoError;
                FILE* winner = std::fopen(destination.generic_string().c_str(), "wb");
                if (winner == nullptr) return kOsIoError;
                const bool wrote = std::fwrite("winner", 1, 6, winner) == 6;
                const bool closed = std::fclose(winner) == 0;
                if (!wrote || !closed) return kOsIoError;
            }
            return self.status;
        }
        std::error_code ec;
        const auto destination = self.remote / file->candidate.object_key;
        (void)std::filesystem::create_directories(destination.parent_path(), ec);
        if (ec) return kOsIoError;
        if (!std::filesystem::copy_file(file->local_path, destination, std::filesystem::copy_options::none, ec))
            return std::filesystem::exists(destination) ? kOsExists : kOsIoError;
        if (self.obstruct_cleanup) {
            const auto marker = (std::filesystem::path(file->local_path).parent_path() / "keep").generic_string();
            FILE* out = std::fopen(marker.c_str(), "wb");
            if (out == nullptr) return kOsWriteUncertain;
            if (std::fclose(out) != 0) return kOsWriteUncertain;
        }
        return self.accepted_loss ? kOsWriteUncertain : kOsOk;
    }
};
class IcebergWriteIO : public ::testing::Test {
protected:
    bolt::Arena arena;
    std::filesystem::path root, remote, staging;
    bool owns = false;
    Store store;
    Publisher publisher;
    SchemaField field{}; Schema schema{};
    TableHandle* table = nullptr;
    bolt::BoltBatch batch{};
    std::vector<int64_t> values;
    void SetUp() override {
        static std::atomic<uint32_t> next{0};
        root = std::filesystem::temp_directory_path() / ("bolt_iceberg_stage_" +
            std::to_string(bolt_getpid()) + "_" + std::to_string(next.fetch_add(1)));
        std::error_code ec; ASSERT_TRUE(std::filesystem::create_directory(root, ec)); owns = true;
        remote = root / "remote"; staging = root / "staging";
        ASSERT_TRUE(std::filesystem::create_directory(remote, ec));
        ASSERT_TRUE(std::filesystem::create_directory(staging, ec)); ASSERT_FALSE(ec);
        const auto path = remote.generic_string(); ASSERT_TRUE(store.init(path.c_str()));
        publisher.remote = remote;
        field.id = 7; field.required = true;  // Nonordinal source ID.
        std::strcpy(field.name, "id"); std::strcpy(field.type, "long");
        schema.fields = &field; schema.n_fields = 1; schema.cap_fields = 1;
        ASSERT_TRUE(table_create(&table, &arena, &store.adapter, "s3://bucket/persisted/table", &schema,
            nullptr, nullptr, nullptr));
        values = {7, -19, 123456789, 0};
        bolt::BoltBatch::init_empty(&batch); batch.arena = &arena;
        ASSERT_TRUE(bolt::BoltBatch::alloc_columns(&batch, &arena, 1));
        ASSERT_TRUE(batch.schema.add_field("id", bolt::BoltType::Int64, false));
        batch.num_cols = 1; refresh_batch();
    }
    void TearDown() override {
        if (!owns) return;
        std::error_code ec; std::filesystem::remove_all(root, ec); EXPECT_FALSE(ec);
    }
    void refresh_batch() {
        batch.num_rows = static_cast<int64_t>(values.size());
        auto& col = batch.columns[batch.read_epoch][0];
        col.type = bolt::BoltType::Int64; col.length = batch.num_rows; col.data = values.data();
        col.type_size_bytes = sizeof(int64_t); col.format = bolt::ColumnFormat::Flat; col.stats.all_valid = true;
    }
    FileWriteIO config(uint64_t cap = 64u << 20) {
        FileWriteIO options{};
        const auto path = staging.generic_string();
        std::memcpy(options.staging_root, path.c_str(), path.size() + 1u);
        options.max_staged_file_bytes = cap; options.context = &publisher; options.publish_if_absent = Publisher::publish;
        return options;
    }
    bool append(uint32_t groups = 1) {
        AppendHandle* append_handle = nullptr;
        if (!append_open(&append_handle, table)) return false;
        for (uint32_t i = 0; i < groups; ++i) if (!append_write(append_handle, &batch)) return false;
        return append_commit(append_handle);
    }
    void check_data_field_id(const char* key, int64_t expected) {
        bolt::Arena bytes; const uint8_t* data = nullptr; uint64_t length = 0;
        ASSERT_EQ(os_get(&store.origin, key, &bytes, &data, &length), kOsOk);
        std::array<int64_t, 2> ids{};
        ASSERT_TRUE(schema_field_ids(data, length, &ids));
        EXPECT_EQ(ids[0], INT64_MIN);  // The root has no Iceberg column ID.
        EXPECT_EQ(ids[1], expected);  // Physical SchemaElement field 9.
    }
    void check_source(const StagedDataFile& file) {
        ASSERT_EQ(file.candidate.kind, CommitObjectKind::kDataFile);
        EXPECT_STREQ(file.candidate.table_uuid, table_metadata(table)->table_uuid);
        EXPECT_EQ(std::string(file.candidate.object_location), std::string("s3://bucket/persisted/table/") + file.candidate.object_key);
        bolt::Arena bytes; const uint8_t* data = nullptr; uint64_t length = 0;
        ASSERT_EQ(os_get(&store.origin, file.candidate.object_key, &bytes, &data, &length), kOsOk);
        ASSERT_EQ(length, file.candidate.length);
        char hash[65]{}; bolt::crypto::sha256_hex(data, length, hash);
        EXPECT_STREQ(hash, file.candidate.sha256);
        std::vector<uint8_t> canonical;
        const auto number = [&canonical](uint64_t value) {
            for (uint32_t i = 0; i < 8; ++i) canonical.push_back(static_cast<uint8_t>(value >> (i * 8u)));
        };
        for (const char* text : {"bolt.iceberg.data.v1", file.candidate.object_key,
             file.candidate.object_location, file.candidate.table_uuid, file.candidate.sha256}) {
            number(std::strlen(text));
            canonical.insert(canonical.end(), text, text + std::strlen(text));
        }
        number(file.candidate.length); number(static_cast<uint64_t>(file.candidate.base_metadata_version));
        number(static_cast<uint64_t>(file.candidate.candidate_metadata_version));
        number(static_cast<uint64_t>(file.candidate.snapshot_id));
        bolt::crypto::sha256_hex(canonical.data(), canonical.size(), hash);
        EXPECT_STREQ(hash, file.operation_id);
    }
};
TEST_F(IcebergWriteIO, ExplicitConfigurationIsCopiedAndLocksAtFirstMutation) {
    auto options = config(0); EXPECT_FALSE(table_set_file_write_io(table, &options));
    options = config(); ASSERT_TRUE(table_set_file_write_io(table, &options));
    std::strcpy(options.staging_root, "/invalid/not-used"); options.publish_if_absent = nullptr;
    EXPECT_FALSE(table_set_file_write_io(table, &options));
    ASSERT_TRUE(append()); EXPECT_EQ(publisher.calls, 1u);
    StagedDataFile file{}; ASSERT_TRUE(table_staged_data_file(table, &file));
    EXPECT_TRUE(file.published); EXPECT_FALSE(file.source_retained); EXPECT_EQ(file.cleanup_status, kOsOk);
    EXPECT_FALSE(std::filesystem::exists(file.local_path)); check_source(file);
}
TEST_F(IcebergWriteIO, LateOrUnboundedConfigurationRefusesBeforeSourceEffects) {
    auto options = config(UINT64_MAX); EXPECT_FALSE(table_set_file_write_io(table, &options));
    options = config(); std::strcpy(options.staging_root, "relative/stage");
    EXPECT_FALSE(table_set_file_write_io(table, &options));
    options = config(); std::memset(options.staging_root, 'x', sizeof(options.staging_root));
    EXPECT_FALSE(table_set_file_write_io(table, &options));
    AppendHandle* append_handle = nullptr; ASSERT_TRUE(append_open(&append_handle, table));
    options = config(); EXPECT_FALSE(table_set_file_write_io(table, &options));
    append_close(append_handle);
    EXPECT_EQ(publisher.calls, 0u); EXPECT_TRUE(std::filesystem::is_empty(staging));
}
TEST_F(IcebergWriteIO, TinySpoolCapNeverPublishesOrLeavesPartialStage) {
    auto options = config(4); ASSERT_TRUE(table_set_file_write_io(table, &options));
    const uint32_t before = store.creates;
    EXPECT_FALSE(append()); EXPECT_EQ(table_last_commit_error(table), CommitError::kFailed);
    EXPECT_EQ(publisher.calls, 0u); EXPECT_EQ(store.creates, before);
    EXPECT_TRUE(std::filesystem::is_empty(staging));
}
TEST_F(IcebergWriteIO, AcceptedLostDataAckRetainsExactCandidateAndFencesEveryMutation) {
    auto options = config(); ASSERT_TRUE(table_set_file_write_io(table, &options));
    publisher.accepted_loss = true;
    const uint32_t before = store.creates;
    EXPECT_FALSE(append()); EXPECT_EQ(table_last_commit_error(table), CommitError::kUncertain);
    EXPECT_EQ(store.creates, before); EXPECT_EQ(store.deletes, 0u);
    StagedDataFile file{}; ASSERT_TRUE(table_uncertain_data_file(table, &file));
    EXPECT_TRUE(file.source_retained); EXPECT_FALSE(file.published); EXPECT_TRUE(std::filesystem::exists(file.local_path));
    EXPECT_STREQ(file.operation_id, publisher.received.operation_id); check_source(file);
    CommitCandidate candidate{}; ASSERT_TRUE(table_uncertain_commit(table, &candidate));
    EXPECT_EQ(candidate.kind, CommitObjectKind::kDataFile); EXPECT_STREQ(candidate.sha256, file.candidate.sha256);
    EXPECT_FALSE(append()); EXPECT_FALSE(table_overwrite(table, nullptr, &batch));
    PositionDeleteEntry entry{}; std::strcpy(entry.file_path, file.candidate.object_location); entry.pos = 0;
    EXPECT_FALSE(table_delete_positions(table, &entry, 1)); EXPECT_FALSE(table_delete(table, nullptr));
    EXPECT_FALSE(table_rewrite(table, nullptr)); EXPECT_FALSE(table_compact(table, 1024));
    EXPECT_FALSE(table_add_column(table, "blocked", bolt::BoltType::Int64, false));
    ExpireOptions expire{}; expire.retain_last = 1; ExpireResult result{};
    EXPECT_FALSE(table_expire_snapshots_ex(table, &expire, &result));
    uint32_t count = 0; EXPECT_FALSE(table_remove_orphans(table, UINT64_MAX, false, nullptr, 0, &count));
    EXPECT_EQ(publisher.calls, 1u); EXPECT_EQ(store.creates, before); EXPECT_EQ(store.deletes, 0u);
    table_close(table); EXPECT_TRUE(std::filesystem::exists(file.local_path));
}
TEST_F(IcebergWriteIO, AcknowledgedDataReceiptSurvivesLostMetadataAckWithoutCleanup) {
    auto options = config(); ASSERT_TRUE(table_set_file_write_io(table, &options)); store.metadata_loss = true;
    EXPECT_FALSE(append()); EXPECT_EQ(table_last_commit_error(table), CommitError::kUncertain);
    CommitCandidate metadata{}; ASSERT_TRUE(table_uncertain_commit(table, &metadata));
    EXPECT_EQ(metadata.kind, CommitObjectKind::kMetadata);
    StagedDataFile data{}; ASSERT_TRUE(table_uncertain_data_file(table, &data));
    EXPECT_EQ(data.candidate.kind, CommitObjectKind::kDataFile); EXPECT_TRUE(data.published);
    EXPECT_TRUE(data.source_retained); EXPECT_TRUE(std::filesystem::exists(data.local_path));
    EXPECT_EQ(data.candidate.snapshot_id, metadata.snapshot_id); EXPECT_EQ(store.deletes, 0u); check_source(data);
}
TEST_F(IcebergWriteIO, DefiniteDataConflictNeverDeletesCompetingObject) {
    auto options = config(); ASSERT_TRUE(table_set_file_write_io(table, &options)); publisher.status = kOsExists;
    const uint32_t before = store.creates;
    EXPECT_FALSE(append()); EXPECT_EQ(table_last_commit_error(table), CommitError::kConflict);
    EXPECT_EQ(publisher.calls, 1u); EXPECT_EQ(store.creates, before); EXPECT_EQ(store.deletes, 0u);
    StagedDataFile file{}; ASSERT_TRUE(table_staged_data_file(table, &file));
    EXPECT_FALSE(file.source_retained); EXPECT_EQ(file.cleanup_status, kOsOk);
    bolt::Arena bytes; const uint8_t* data = nullptr; uint64_t size = 0;
    ASSERT_EQ(os_get(&store.origin, file.candidate.object_key, &bytes, &data, &size), kOsOk);
    ASSERT_EQ(size, 6u); EXPECT_EQ(std::memcmp(data, "winner", 6), 0);
}
TEST_F(IcebergWriteIO, CleanupFailureRetainsSourceButDoesNotRelabelAcknowledgedCommit) {
    auto options = config(); ASSERT_TRUE(table_set_file_write_io(table, &options)); publisher.obstruct_cleanup = true;
    ASSERT_TRUE(append()); EXPECT_EQ(table_last_commit_error(table), CommitError::kNone);
    StagedDataFile file{}; ASSERT_TRUE(table_staged_data_file(table, &file));
    EXPECT_TRUE(file.published); EXPECT_TRUE(file.source_retained); EXPECT_EQ(file.cleanup_status, kOsIoError);
    EXPECT_TRUE(std::filesystem::exists(file.local_path));
    EXPECT_FALSE(append()); EXPECT_EQ(publisher.calls, 1u);
    StagedDataFile same{}; ASSERT_TRUE(table_staged_data_file(table, &same)); EXPECT_STREQ(same.operation_id, file.operation_id);
    EXPECT_FALSE(table_uncertain_data_file(table, &same));
}
TEST_F(IcebergWriteIO, AppendAndPositionDeleteBothUseStagedPublisher) {
    auto options = config(); ASSERT_TRUE(table_set_file_write_io(table, &options));
    ASSERT_TRUE(append()); ASSERT_EQ(publisher.calls, 1u);
    StagedDataFile first{}; ASSERT_TRUE(table_staged_data_file(table, &first));
    check_data_field_id(first.candidate.object_key, 7);
    PositionDeleteEntry entry{}; std::strcpy(entry.file_path, first.candidate.object_location); entry.pos = 1;
    ASSERT_TRUE(table_delete_positions(table, &entry, 1)); ASSERT_EQ(publisher.calls, 2u);
    StagedDataFile deletion{}; ASSERT_TRUE(table_staged_data_file(table, &deletion)); check_source(deletion);
    bolt::Arena bytes; const uint8_t* data = nullptr; uint64_t size = 0;
    ASSERT_EQ(os_get(&store.origin, deletion.candidate.object_key, &bytes, &data, &size), kOsOk);
    pq::PqMeta meta{}; ASSERT_TRUE(pq::parquet_read_meta(data, size, &bytes, &meta));
    ASSERT_EQ(meta.n_columns, 2u);
    std::array<int64_t, 3> ids{}; ASSERT_TRUE(schema_field_ids(data, size, &ids));
    EXPECT_EQ(ids[1], 2147483546); EXPECT_EQ(ids[2], 2147483545);
    std::array<bolt::BoltColumn, 2> decoded{}; int64_t rows = 0;
    ASSERT_TRUE(pq::parquet_read_row_group(data, size, &meta, 0, &bytes, decoded.data(), &rows));
    ASSERT_EQ(rows, 1); EXPECT_EQ(static_cast<const int64_t*>(decoded[1].data)[0], 1);
    const uint8_t* path = nullptr; int32_t path_length = 0; decoded[0].utf8_at(0, &path, &path_length);
    EXPECT_EQ(std::string(reinterpret_cast<const char*>(path), static_cast<size_t>(path_length)), entry.file_path);
}
TEST_F(IcebergWriteIO, PhysicalSourceFieldIdsSurviveRenameAndAreNotReusedAfterDrop) {
    auto options = config(); ASSERT_TRUE(table_set_file_write_io(table, &options));
    ASSERT_TRUE(append());
    StagedDataFile original{}; ASSERT_TRUE(table_staged_data_file(table, &original));
    check_data_field_id(original.candidate.object_key, 7);
    ASSERT_TRUE(table_rename_column(table, "id", "renamed"));
    std::strcpy(batch.schema.fields[0].name, "renamed");
    ASSERT_TRUE(append());
    StagedDataFile renamed{}; ASSERT_TRUE(table_staged_data_file(table, &renamed));
    check_data_field_id(renamed.candidate.object_key, 7);
    ASSERT_TRUE(table_drop_column(table, "renamed"));
    ASSERT_TRUE(table_add_column(table, "id", bolt::BoltType::Int64, false));
    const Schema* current = metadata_current_schema(table_metadata(table));
    ASSERT_NE(current, nullptr); ASSERT_EQ(current->n_fields, 1u);
    ASSERT_EQ(current->fields[0].id, 8);
    std::strcpy(batch.schema.fields[0].name, "id");
    ASSERT_TRUE(append());
    StagedDataFile replacement{}; ASSERT_TRUE(table_staged_data_file(table, &replacement));
    check_data_field_id(replacement.candidate.object_key, 8);
    // Earlier immutable sources keep their IDs after the current schema changes.
    check_data_field_id(original.candidate.object_key, 7);
    check_data_field_id(renamed.candidate.object_key, 7);
}
TEST_F(IcebergWriteIO, RealParquetLargerThanSixteenMiBPublishesAndReadsWithBoundedWorkspace) {
    constexpr uint32_t groups = 40, group_rows = 65536;
    values.resize(groups * group_rows);
    for (uint64_t i = 0; i < values.size(); ++i) {
        uint64_t bits = (i + 1u) * UINT64_C(0x9e3779b97f4a7c15);
        bits = (bits ^ (bits >> 30u)) * UINT64_C(0xbf58476d1ce4e5b9);
        bits = (bits ^ (bits >> 27u)) * UINT64_C(0x94d049bb133111eb);
        values[i] = std::bit_cast<int64_t>(bits ^ (bits >> 31u));
    }
    auto options = config(); ASSERT_TRUE(table_set_file_write_io(table, &options));
    std::vector<bolt::BoltBatch> batches(groups);
    AppendHandle* append_handle = nullptr; ASSERT_TRUE(append_open(&append_handle, table));
    for (uint32_t group = 0; group < groups; ++group) {
        auto& input = batches[group]; bolt::BoltBatch::init_empty(&input); input.arena = &arena;
        ASSERT_TRUE(bolt::BoltBatch::alloc_columns(&input, &arena, 1));
        ASSERT_TRUE(input.schema.add_field("id", bolt::BoltType::Int64, false));
        input.num_cols = 1; input.num_rows = group_rows;
        auto& col = input.columns[input.read_epoch][0];
        col.type = bolt::BoltType::Int64; col.type_size_bytes = sizeof(int64_t);
        col.format = bolt::ColumnFormat::Flat; col.stats.all_valid = true;
        col.length = group_rows; col.data = values.data() + static_cast<size_t>(group) * group_rows;
        ASSERT_TRUE(append_write(append_handle, &input));
    }
    ASSERT_TRUE(append_commit(append_handle)); ASSERT_EQ(publisher.calls, 1u);
    StagedDataFile receipt{}; ASSERT_TRUE(table_staged_data_file(table, &receipt));
    ASSERT_GT(receipt.candidate.length, 16u << 20); ASSERT_LE(receipt.candidate.length, options.max_staged_file_bytes);
    EXPECT_FALSE(receipt.source_retained);
    const auto path = (remote / receipt.candidate.object_key).generic_string();
    FileSource file; file.file.handle = -1;
    ASSERT_EQ(bolt::io::file_open_read(path.c_str(), false, &file.file), bolt::io::FileIoStatus::kOk);
    pq::ParquetByteSource source{file.file.size, &file, FileSource::read};
    std::vector<uint8_t> buffer(1u << 20);
    pq::ParquetSourceWorkspace workspace{buffer.data(), buffer.size(), 4096};
    // The metadata parser's first reservation is 128 columns +4096 chunks,
    // independently of this file's actual width/group count. Include its
    // staged PqMeta and alignment; this is not a file/body/workspace limit.
    constexpr size_t metadata_bytes = sizeof(pq::PqMeta) + 128u * sizeof(pq::PqColumn) +
        4096u * sizeof(pq::PqChunk) + 3u * 64u;
    static_assert(metadata_bytes > (4u << 20));
    bolt::Arena metadata{{metadata_bytes, metadata_bytes, 64, false}}; metadata.set_growth_enabled(false);
    pq::PqMeta meta{};
    ASSERT_EQ(pq::parquet_read_meta_source(&source, &workspace, &metadata, &meta), pq::ParquetSourceStatus::Ok);
    ASSERT_EQ(meta.n_row_groups, groups);
    bolt::Arena output{{2u << 20, 2u << 20, 64, false}}; output.set_growth_enabled(false);
    const uint16_t projection = 0;
    for (uint32_t group = 0; group < meta.n_row_groups; ++group) {
        output.reset(); bolt::BoltColumn column{}; int64_t rows = 0;
        ASSERT_EQ(pq::parquet_read_row_group_cols_source(&source, &workspace, &meta, group,
            &projection, 1, &output, &column, &rows), pq::ParquetSourceStatus::Ok);
        ASSERT_EQ(rows, group_rows);
        const auto* actual = static_cast<const int64_t*>(column.data);
        for (uint32_t row = 0; row < group_rows; ++row)
            ASSERT_EQ(actual[row], values[static_cast<size_t>(group) * group_rows + row]);
    }
    EXPECT_LE(file.largest, 4096u); EXPECT_GT(file.calls, 4096u);
}
TEST_F(IcebergWriteIO, UnacceptedDataLossStillRetainsEvidenceWithoutRetryOrCleanup) {
    auto options = config(); ASSERT_TRUE(table_set_file_write_io(table, &options)); publisher.status = kOsWriteUncertain;
    const uint32_t before = store.creates;
    EXPECT_FALSE(append()); EXPECT_EQ(table_last_commit_error(table), CommitError::kUncertain);
    StagedDataFile receipt{}; ASSERT_TRUE(table_uncertain_data_file(table, &receipt));
    EXPECT_TRUE(receipt.source_retained); EXPECT_TRUE(std::filesystem::exists(receipt.local_path));
    EXPECT_FALSE(std::filesystem::exists(remote / receipt.candidate.object_key));
    EXPECT_FALSE(append()); EXPECT_EQ(publisher.calls, 1u);
    EXPECT_EQ(store.creates, before); EXPECT_EQ(store.deletes, 0u);
}
TEST_F(IcebergWriteIO, ReceiptOutputsCannotOverwritePrivateTableOrArenaState) {
    auto options = config(); ASSERT_TRUE(table_set_file_write_io(table, &options));
    EXPECT_FALSE(table_staged_data_file(table, reinterpret_cast<StagedDataFile*>(table)));
    EXPECT_FALSE(table_uncertain_data_file(table, reinterpret_cast<StagedDataFile*>(&arena)));
    ASSERT_TRUE(append()); EXPECT_EQ(publisher.calls, 1u);
}
TEST_F(IcebergWriteIO, ReceiptOutputsCannotOverwriteBorrowedObjectStoreOrVtable) {
    for (const bool uncertain : {false, true}) {
        SCOPED_TRACE(uncertain);
        alignas(StagedDataFile) std::array<uint8_t, sizeof(StagedDataFile)> handle_bytes{}, vt_bytes{};
        auto* vt = new (vt_bytes.data()) ObjectStoreVT(*store.adapter.vt);
        auto* borrowed = new (handle_bytes.data()) ObjectStore{vt, store.adapter.impl};
        TableHandle* opened = nullptr;
        ASSERT_TRUE(table_open(&opened, &arena, borrowed, "s3://bucket/persisted/table"));
        table = opened;
        auto options = config(); ASSERT_TRUE(table_set_file_write_io(table, &options));
        publisher.accepted_loss = uncertain;
        ASSERT_EQ(append(), !uncertain);
        const auto original_handle = handle_bytes, original_vt = vt_bytes;
        auto* handle_output = reinterpret_cast<StagedDataFile*>(handle_bytes.data());
        auto* vt_output = reinterpret_cast<StagedDataFile*>(vt_bytes.data());
        EXPECT_FALSE(table_staged_data_file(table, handle_output));
        EXPECT_FALSE(table_uncertain_data_file(table, handle_output));
        EXPECT_FALSE(table_staged_data_file(table, vt_output));
        EXPECT_FALSE(table_uncertain_data_file(table, vt_output));
        EXPECT_EQ(handle_bytes, original_handle); EXPECT_EQ(vt_bytes, original_vt);
        StagedDataFile intact{}; ASSERT_TRUE(table_staged_data_file(table, &intact));
        EXPECT_EQ(intact.source_retained, uncertain);
    }
}
}  // namespace
