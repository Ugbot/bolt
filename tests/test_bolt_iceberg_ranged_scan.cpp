// Public Iceberg scans over a versioned origin which refuses whole Parquet GETs.
#include <gtest/gtest.h>
#include "bolt/bolt_column.h"
#include "bolt/lakehouse/iceberg/scan.h"
#include "bolt/lakehouse/iceberg/writer.h"
#include <array>
#include <bit>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <new>
#include <string>
#include <vector>

namespace {
using namespace bolt;
using namespace bolt::lakehouse;
using namespace bolt::lakehouse::iceberg;
struct Origin {
    FilesystemObjectStore fs{};
    ObjectStore base{}, store{};
    ObjectStoreVT vt{};
    uint32_t heads = 0, ranges = 0, whole_parquet = 0, largest = 0;
    uint32_t change_at = 0, short_at = 0;
    bool weak = false;
    static bool parquet(const char* key) { return std::strstr(key, ".parquet") != nullptr; }
    static int get(void* ptr, const char* key, Arena* a, const uint8_t** body, uint64_t* n) noexcept {
        auto* self = static_cast<Origin*>(ptr);
        if (parquet(key)) { ++self->whole_parquet; return kOsIoError; }
        return os_get(&self->base, key, a, body, n);
    }
    static int list(void* ptr, const char* key, ObjectEntry* entries, uint32_t cap, uint32_t* n) noexcept {
        return os_list(&static_cast<Origin*>(ptr)->base, key, entries, cap, n);
    }
    static int head(void* ptr, const char* key, Arena*, ObjectVersion* version) noexcept {
        auto* self = static_cast<Origin*>(ptr);
        ++self->heads;
        *version = ObjectVersion{};
        ObjectMeta meta{};
        const int rc = os_head(&self->base, key, &meta);
        version->size = meta.size;
        std::strcpy(version->etag, self->weak ? "W/\"fixture\"" : "\"fixture\"");
        return rc;
    }
    static int range(void* ptr, const char* key, const ObjectVersion* expected,
        uint64_t offset, uint32_t length, Arena*, uint8_t* dst, uint32_t* got) noexcept {
        auto* self = static_cast<Origin*>(ptr);
        ++self->ranges;
        self->largest = length > self->largest ? length : self->largest;
        if (std::strcmp(expected->etag, "\"fixture\"") != 0) return kOsBadArg;
        if (self->change_at != 0 && self->ranges >= self->change_at) return kOsPreconditionFailed;
        uint64_t actual = 0;
        const int rc = os_get_range(&self->base, key, offset, length, dst, &actual);
        *got = static_cast<uint32_t>(actual);
        if (self->ranges == self->short_at && *got > 0) --*got;
        return rc;
    }
    bool init(const char* root) {
        if (!filesystem_object_store_init(&fs, root, &base)) return false;
        vt = ObjectStoreVT{};
        vt.get_object = get; vt.list_prefix = list;
        vt.head_versioned = head; vt.get_range_if_match = range;
        store = {&vt, this};
        return true;
    }
};

struct Fixture {
    Arena arena;
    Arena io{{1u << 20, 1u << 20, 64, false}};
    std::array<uint8_t, 4096> workspace{};
    Origin origin;
    TableHandle* table = nullptr;
    ScanHandle* scan = nullptr;
    bool open(uint32_t request = 64, uint64_t capacity = 4096) {
        const std::string root = std::string(BOLT_TEST_DATA_DIR) + "/golden_iceberg_table";
        if (!origin.init(root.c_str()) ||
            !iceberg_table_open_store(&table, &arena, &origin.store, "db/trades")) return false;
        IcebergScanIO config{workspace.data(), capacity, &io, request};
        return iceberg_scan_open_with_io(&scan, table, nullptr, &config);
    }
};

void exact_golden(ScanHandle* scan) {
    const char* symbols[]{"", "AAPL", "AAPL", "MSFT", "MSFT", "GOOG", "GOOG", "AAPL", "TSLA", "TSLA"};
    const double prices[]{0,101.5,102.25,310,311.75,140,0,103,250.5,251};
    const bool active[]{false,true,false,true,true,false,false,true,false,true};
    std::array<bool,10> seen{};
    uint32_t count = 0;
    bool finished = false;
    for (uint32_t guard = 0; guard < 32; ++guard) {
        BoltBatch batch{}; bool eof = false;
        ASSERT_TRUE(iceberg_scan_next_batch(scan, &batch, &eof));
        if (eof) { finished = true; break; }
        const auto* cols = batch.columns[batch.read_epoch];
        ASSERT_EQ(batch.num_cols, 4u);
        for (int64_t row = 0; row < batch.num_rows; ++row) {
            const int64_t id = cols[0].typed_data<int64_t>()[row];
            ASSERT_GE(id, 1); ASSERT_LE(id, 9);
            const auto slot = static_cast<size_t>(id);
            ASSERT_FALSE(seen[slot]); seen[slot] = true; ++count;
            const uint8_t* str = nullptr; int32_t len = 0;
            cols[1].utf8_at(row, &str, &len);
            EXPECT_EQ(std::string(reinterpret_cast<const char*>(str), static_cast<size_t>(len)), symbols[id]);
            EXPECT_EQ(cols[2].is_null(row), id == 6);
            if (id != 6) EXPECT_DOUBLE_EQ(cols[2].typed_data<double>()[row], prices[id]);
            EXPECT_EQ(cols[3].is_null(row), id == 5);
            if (id != 5) EXPECT_EQ(cols[3].typed_data<int64_t>()[row] != 0, active[id]);
        }
    }
    EXPECT_EQ(count, 9u);
    EXPECT_TRUE(finished);
}

int64_t large_value(uint64_t row) {
    uint64_t n = row + UINT64_C(0x9e3779b97f4a7c15);
    n = (n ^ (n >> 30u)) * UINT64_C(0xbf58476d1ce4e5b9);
    n = (n ^ (n >> 27u)) * UINT64_C(0x94d049bb133111eb);
    return std::bit_cast<int64_t>(n ^ (n >> 31u));
}

bool write_local_table(const std::string& root, Arena* arena, ObjectStore* store,
    int64_t rows, uint32_t group_rows, TableHandle** out, char* data_path, bool scrambled = false) {
    SchemaField field{};
    field.id = 1; field.required = true;
    std::strcpy(field.name,"id"); std::strcpy(field.type,"long");
    Schema schema{}; schema.fields = &field; schema.cap_fields = 1; schema.n_fields = 1;
    WriteOptions options{}; write_options_init(&options);
    if (!table_create(out,arena,store,(root+"/table").c_str(),&schema,nullptr,nullptr,&options)) return false;
    std::vector<int64_t> values(static_cast<size_t>(rows));
    for (int64_t i = 0; i < rows; ++i) values[static_cast<size_t>(i)] = scrambled ? large_value(static_cast<uint64_t>(i)) : i * 37 + 11;
    const uint32_t groups = static_cast<uint32_t>((rows + group_rows - 1)/group_rows);
    if (groups > 256) return false;
    std::vector<BoltBatch> batches(groups);
    AppendHandle* append = nullptr;
    if (!append_open(&append,*out)) return false;
    bool ok = true;
    for (uint32_t group = 0; group < groups && ok; ++group) {
        auto& batch = batches[group]; BoltBatch::init_empty(&batch);
        const int64_t offset = static_cast<int64_t>(group)*group_rows;
        const int64_t count = rows-offset < group_rows ? rows-offset : group_rows;
        ok = BoltBatch::alloc_columns(&batch,arena,1) && batch.schema.add_field("id",BoltType::Int64,false);
        if (!ok) break;
        batch.num_rows = count; batch.num_cols = 1; batch.arena = arena;
        auto& column = batch.columns[batch.read_epoch][0];
        column = BoltColumn::make_empty(); column.type = BoltType::Int64;
        column.format = ColumnFormat::Flat; column.type_size_bytes = sizeof(int64_t);
        column.length = count; column.data = values.data()+offset; column.stats.all_valid = true;
        ok = append_write(append,&batch);
    }
    ok = ok && append_commit_ex(append,data_path,kIcebergMaxPath,nullptr);
    append_close(append);
    return ok;
}

std::string temp_root(const char* label) {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() /
        (std::string("bolt-ranged-") + label + "-" + std::to_string(nonce));
    std::error_code error;
    if (!std::filesystem::create_directory(path,error) || error) return {};
    std::fprintf(stdout,"[ranged scan] owned fixture=%s; preserved on failure\n",path.string().c_str());
    return path.generic_string();
}

bool write_store(const std::string& root, FilesystemObjectStore* fs, ObjectStore* store) {
    std::error_code error;
    if (!std::filesystem::create_directory(root+"/table",error) || error) return false;
    return filesystem_object_store_init(fs,(root+"/table").c_str(),store);
}
}  // namespace

TEST(IcebergRangedScan, IndependentGoldenValuesWithTinyRequestsAndNoWholeDataGet) {
    Fixture f;
    ASSERT_TRUE(f.open());
    exact_golden(f.scan);
    EXPECT_EQ(f.origin.heads, 5u);
    EXPECT_EQ(f.origin.whole_parquet, 0u);
    EXPECT_GT(f.origin.ranges, 20u);
    EXPECT_LE(f.origin.largest, 64u);
    iceberg_scan_close(f.scan);
}

TEST(IcebergRangedScan, ExistingOpenAutomaticallyUsesVersionedSource) {
    Fixture f;
    ASSERT_TRUE(f.open());
    iceberg_scan_close(f.scan);
    ASSERT_TRUE(iceberg_scan_open(&f.scan, f.table, nullptr));
    exact_golden(f.scan);
    EXPECT_EQ(f.origin.whole_parquet, 0u);
    EXPECT_EQ(f.origin.heads, 5u);
    iceberg_scan_close(f.scan);
}

TEST(IcebergRangedScan, ChangedFooterIsTerminalWithoutFallbackOrRepin) {
    Fixture f; ASSERT_TRUE(f.open(1u << 20));
    f.origin.change_at = 2;
    BoltBatch batch{}; bool eof = true;
    EXPECT_FALSE(iceberg_scan_next_batch(f.scan, &batch, &eof)); EXPECT_FALSE(eof);
    EXPECT_EQ(f.origin.heads, 1u); EXPECT_EQ(f.origin.ranges, 2u);
    EXPECT_FALSE(iceberg_scan_next_batch(f.scan, &batch, &eof));
    EXPECT_EQ(f.origin.heads, 1u); EXPECT_EQ(f.origin.ranges, 2u);
    EXPECT_EQ(f.origin.whole_parquet, 0u);
    iceberg_scan_close(f.scan);
}

TEST(IcebergRangedScan, ChangedDataAfterFooterIsTerminal) {
    Fixture f; ASSERT_TRUE(f.open(1u << 20));
    f.origin.change_at = 4;
    BoltBatch batch{}; bool eof = false;
    EXPECT_FALSE(iceberg_scan_next_batch(f.scan, &batch, &eof));
    EXPECT_FALSE(iceberg_scan_next_batch(f.scan, &batch, &eof));
    EXPECT_EQ(f.origin.heads, 1u); EXPECT_EQ(f.origin.ranges, 4u);
    EXPECT_EQ(f.origin.whole_parquet, 0u);
    iceberg_scan_close(f.scan);
}

TEST(IcebergRangedScan, ShortSuccessAndWeakHeadRefuse) {
    for (uint32_t mode = 0; mode < 2; ++mode) {
        Fixture f; ASSERT_TRUE(f.open(1u << 20));
        f.origin.short_at = mode == 0 ? 3u : 0u;
        f.origin.weak = mode == 1;
        BoltBatch batch{}; bool eof = false;
        EXPECT_FALSE(iceberg_scan_next_batch(f.scan, &batch, &eof));
        EXPECT_FALSE(iceberg_scan_next_batch(f.scan, &batch, &eof));
        EXPECT_EQ(f.origin.heads, 1u); EXPECT_EQ(f.origin.whole_parquet, 0u);
        EXPECT_EQ(f.origin.ranges, mode == 0 ? 3u : 0u);
        iceberg_scan_close(f.scan);
    }
}

TEST(IcebergRangedScan, WorkspaceCapacityRefusalDoesNotDownloadFile) {
    Fixture f; ASSERT_TRUE(f.open(64, 16));
    BoltBatch batch{}; bool eof = false;
    EXPECT_FALSE(iceberg_scan_next_batch(f.scan, &batch, &eof));
    EXPECT_EQ(f.origin.heads, 1u); EXPECT_EQ(f.origin.ranges, 2u);
    EXPECT_EQ(f.origin.whole_parquet, 0u);
    iceberg_scan_close(f.scan);
}

TEST(IcebergRangedScan, BorrowedWorkspaceAndTransportCannotAliasOutputArena) {
    Fixture f; ASSERT_TRUE(f.open()); iceberg_scan_close(f.scan);
    auto* bytes = f.arena.allocate_array<uint8_t>(4096);
    ASSERT_NE(bytes, nullptr);
    IcebergScanIO config{bytes,4096,&f.io,64};
    EXPECT_FALSE(iceberg_scan_open_with_io(&f.scan,f.table,nullptr,&config));
    config = {f.workspace.data(),4096,&f.arena,64};
    EXPECT_FALSE(iceberg_scan_open_with_io(&f.scan,f.table,nullptr,&config));
    EXPECT_EQ(f.origin.heads, 0u); EXPECT_EQ(f.origin.ranges, 0u);
}

TEST(IcebergRangedScan, PartialVersionInterfaceRefusesBeforeReadingData) {
    Fixture f; ASSERT_TRUE(f.open()); iceberg_scan_close(f.scan);
    f.origin.vt.get_range_if_match = nullptr;
    EXPECT_FALSE(iceberg_scan_open(&f.scan,f.table,nullptr));
    EXPECT_EQ(f.origin.heads, 0u); EXPECT_EQ(f.origin.whole_parquet, 0u);
}

TEST(IcebergRangedScan, HandleOutputCannotAliasBorrowedWorkspaceOrTransport) {
    Fixture f; ASSERT_TRUE(f.open()); iceberg_scan_close(f.scan);
    alignas(void*) std::array<uint8_t,4096> workspace{};
    auto** aliased = new (workspace.data()) ScanHandle*(f.scan);
    const auto* original = *aliased;
    IcebergScanIO config{workspace.data(),workspace.size(),&f.io,64};
    EXPECT_FALSE(iceberg_scan_open_with_io(aliased,f.table,nullptr,&config));
    EXPECT_EQ(*aliased,original);
    void* p = f.io.allocate(sizeof(ScanHandle*),alignof(ScanHandle*));
    ASSERT_NE(p,nullptr);
    aliased = new (p) ScanHandle*(f.scan);
    config = {f.workspace.data(),f.workspace.size(),&f.io,64};
    EXPECT_FALSE(iceberg_scan_open_with_io(aliased,f.table,nullptr,&config));
    EXPECT_EQ(*aliased,original);
    EXPECT_EQ(f.origin.heads,0u); EXPECT_EQ(f.origin.ranges,0u);
}

TEST(IcebergRangedScan, InsufficientFrozenTransportRefusesAtStartupWithoutIo) {
    Fixture f; ASSERT_TRUE(f.open()); iceberg_scan_close(f.scan);
    Arena small{{4096,4096,64,false}};
    IcebergScanIO config{f.workspace.data(),f.workspace.size(),&small,64};
    EXPECT_FALSE(iceberg_scan_open_with_io(&f.scan,f.table,nullptr,&config));
    EXPECT_EQ(f.scan,nullptr); EXPECT_FALSE(small.growth_enabled());
    EXPECT_EQ(f.origin.heads,0u); EXPECT_EQ(f.origin.ranges,0u);
}

TEST(IcebergRangedScan, PerCallOutputsCannotAliasSourceWorkspaceOrTransport) {
    Fixture f; ASSERT_TRUE(f.open());
    BoltBatch batch{}; bool eof = true;
    auto* aliased_eof = new (f.workspace.data()+128) bool(true);
    EXPECT_FALSE(iceberg_scan_next_batch(f.scan,&batch,aliased_eof));
    EXPECT_TRUE(*aliased_eof);
    void* p = f.io.allocate(sizeof(bool),alignof(bool)); ASSERT_NE(p,nullptr);
    aliased_eof = new (p) bool(true);
    EXPECT_FALSE(iceberg_scan_next_batch(f.scan,&batch,aliased_eof));
    EXPECT_TRUE(*aliased_eof);
    EXPECT_EQ(f.origin.heads,0u); EXPECT_EQ(f.origin.ranges,0u);
    // Admission failures do not consume the scan or modify independent outputs.
    EXPECT_TRUE(eof);
    exact_golden(f.scan);
    iceberg_scan_close(f.scan);
}

TEST(IcebergRangedScan, KnownVtableAndArenaObjectsAreProtectedFromWritableAliases) {
    Fixture f; ASSERT_TRUE(f.open()); iceberg_scan_close(f.scan);
    const auto callback = f.origin.vt.get_object;
    auto** aliased = reinterpret_cast<ScanHandle**>(&f.origin.vt);
    IcebergScanIO config{f.workspace.data(),f.workspace.size(),&f.io,64};
    EXPECT_FALSE(iceberg_scan_open_with_io(aliased,f.table,nullptr,&config));
    EXPECT_EQ(f.origin.vt.get_object,callback);
    config.workspace = reinterpret_cast<uint8_t*>(&f.origin.vt);
    config.workspace_bytes = sizeof(f.origin.vt);
    EXPECT_FALSE(iceberg_scan_open_with_io(&f.scan,f.table,nullptr,&config));
    EXPECT_EQ(f.origin.vt.get_object,callback);
    config = {f.workspace.data(),f.workspace.size(),&f.io,64};
    ASSERT_TRUE(iceberg_scan_open_with_io(&f.scan,f.table,nullptr,&config));
    std::array<uint8_t,sizeof(Arena)> saved{};
    std::memcpy(saved.data(),&f.arena,sizeof(Arena));
    BoltBatch batch{};
    EXPECT_FALSE(iceberg_scan_next_batch(f.scan,&batch,reinterpret_cast<bool*>(&f.arena)));
    EXPECT_EQ(std::memcmp(saved.data(),&f.arena,sizeof(Arena)),0);
    EXPECT_EQ(f.origin.heads,0u); EXPECT_EQ(f.origin.ranges,0u);
    iceberg_scan_close(f.scan);
}

TEST(IcebergRangedScan, PositionDeletesAcrossGroupsUsePinnedRangesAndExactAbsolutePositions) {
    const auto root = temp_root("deletes"); ASSERT_FALSE(root.empty());
    Arena write_arena, read_arena;
    Origin origin; ASSERT_TRUE(origin.init(root.c_str()));
    FilesystemObjectStore write_fs{}; ObjectStore write_os{};
    ASSERT_TRUE(write_store(root,&write_fs,&write_os));
    TableHandle* writer = nullptr;
    PositionDeleteEntry deletes[2]{};
    ASSERT_TRUE(write_local_table(root,&write_arena,&write_os,12,4,&writer,deletes[0].file_path));
    std::strcpy(deletes[1].file_path,deletes[0].file_path);
    deletes[0].pos = 1; deletes[1].pos = 7;
    ASSERT_TRUE(table_delete_positions(writer,deletes,2));
    TableHandle* table = nullptr; ScanHandle* scan = nullptr;
    ASSERT_TRUE(iceberg_table_open_store(&table,&read_arena,&origin.store,"table"));
    ASSERT_TRUE(iceberg_scan_open(&scan,table,nullptr));
    std::array<bool,12> seen{}; uint32_t count = 0; bool finished = false;
    for (uint32_t guard = 0; guard < 16; ++guard) {
        BoltBatch batch{}; bool eof = false;
        ASSERT_TRUE(iceberg_scan_next_batch(scan,&batch,&eof));
        if (eof) { finished = true; break; }
        for (int64_t row = 0; row < batch.num_rows; ++row) {
            const auto value = batch.columns[batch.read_epoch][0].typed_data<int64_t>()[row];
            ASSERT_EQ((value-11)%37,0); const int64_t index = (value-11)/37;
            ASSERT_GE(index,0); ASSERT_LT(index,12);
            const auto slot = static_cast<size_t>(index);
            ASSERT_FALSE(seen[slot]);
            ASSERT_NE(index,1); ASSERT_NE(index,7); seen[slot] = true; ++count;
        }
    }
    EXPECT_EQ(count,10u); EXPECT_EQ(origin.heads,2u); EXPECT_EQ(origin.whole_parquet,0u);
    EXPECT_TRUE(finished);
    iceberg_scan_close(scan);
    origin.heads = origin.ranges = 0;
    origin.change_at = 4;
    EXPECT_FALSE(iceberg_scan_open(&scan,table,nullptr));
    EXPECT_EQ(scan,nullptr); EXPECT_EQ(origin.heads,1u); EXPECT_EQ(origin.ranges,4u);
    EXPECT_EQ(origin.whole_parquet,0u);
    table_close(writer);
    if (!::testing::Test::HasFailure()) { std::error_code error; std::filesystem::remove_all(root,error); EXPECT_FALSE(error); }
}

TEST(IcebergRangedScan, RealEncodedFileAboveSixteenMiBUsesBoundedWorkspaceAndExactRows) {
    const auto root = temp_root("large"); ASSERT_FALSE(root.empty());
    Arena write_arena, read_arena;
    Origin origin; ASSERT_TRUE(origin.init(root.c_str()));
    FilesystemObjectStore write_fs{}; ObjectStore write_os{};
    ASSERT_TRUE(write_store(root,&write_fs,&write_os));
    TableHandle* writer = nullptr;
    char data_path[kIcebergMaxPath]{};
    constexpr int64_t rows = 150 * 16384;
    ASSERT_TRUE(write_local_table(root,&write_arena,&write_os,rows,16384,&writer,data_path,true));
    std::error_code error;
    const auto file_bytes = std::filesystem::file_size(data_path,error);
    ASSERT_FALSE(error); ASSERT_GT(file_bytes,16u << 20);
    TableHandle* table = nullptr; ScanHandle* scan = nullptr;
    ASSERT_TRUE(iceberg_table_open_store(&table,&read_arena,&origin.store,"table"));
    std::vector<uint8_t> workspace(256u << 10);
    Arena io{{1u << 20,1u << 20,64,false}};
    IcebergScanIO config{workspace.data(),workspace.size(),&io,4096};
    ASSERT_TRUE(iceberg_scan_open_with_io(&scan,table,nullptr,&config));
    int64_t count = 0; bool finished = false;
    for (uint32_t guard = 0; guard < 4096; ++guard) {
        BoltBatch batch{}; bool eof = false;
        ASSERT_TRUE(iceberg_scan_next_batch(scan,&batch,&eof));
        if (eof) { finished = true; break; }
        for (int64_t row = 0; row < batch.num_rows; ++row) {
            ASSERT_EQ(batch.columns[batch.read_epoch][0].typed_data<int64_t>()[row],large_value(static_cast<uint64_t>(count)));
            ++count;
        }
    }
    EXPECT_EQ(count,rows); EXPECT_EQ(origin.heads,1u); EXPECT_EQ(origin.whole_parquet,0u);
    EXPECT_TRUE(finished);
    EXPECT_LE(origin.largest,4096u);
    iceberg_scan_close(scan); table_close(writer);
    if (!::testing::Test::HasFailure()) { std::filesystem::remove_all(root,error); EXPECT_FALSE(error); }
}
