#include "bolt/ingest/bolt_parquet_source.h"
#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt/ingest/bolt_parquet_nested.h"
#include "bolt/ingest/bolt_parquet_write.h"
#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <new>
#include <string>
#include <vector>

namespace {
using namespace bolt::ingest::parquet;
using Status = ParquetSourceStatus;

struct MemorySource {
    const uint8_t* data = nullptr;
    uint64_t length = 0;
    uint64_t calls = 0, bytes = 0;
    uint32_t largest_request = 0;
    uint64_t fail_call = UINT64_MAX;
    Status failure = Status::IoError;
    bool short_read = false;
    static Status read(void* context, uint64_t offset, uint32_t length, uint8_t* out) noexcept {
        auto& s = *static_cast<MemorySource*>(context);
        ++s.calls;
        s.largest_request = std::max(s.largest_request, length);
        if (s.calls == s.fail_call) return s.failure;
        if (offset > s.length || length > s.length - offset || length == 0 ||
            length > kParquetSourceMaxReadBytes || s.calls > 100000u) return Status::IoError;
        if (s.short_read) {
            std::memcpy(out, s.data + offset, length - 1u);
            return Status::IoError;  // Adapter must reject a short transport read.
        }
        std::memcpy(out, s.data + offset, length);
        s.bytes += length;
        return Status::Ok;
    }
    ParquetByteSource source() { return ParquetByteSource{length, this, &read}; }
};

int64_t value_at(uint64_t row) { return static_cast<int64_t>(row * 131u) - 123456789; }

bool integer_file(uint32_t groups, uint32_t rows, std::vector<uint8_t>* file) {
    bolt::Arena input;
    bolt::BoltBatch batch{};
    bolt::BoltBatch::init_empty(&batch);
    if (!bolt::BoltBatch::alloc_columns(&batch, &input, 1)) return false;
    if (!batch.schema.add_field("id", bolt::BoltType::Int64, false)) return false;
    batch.num_rows = rows;
    std::vector<int64_t> values(rows);
    auto& col = batch.columns[batch.read_epoch][0];
    col = bolt::BoltColumn::make_empty();
    col.type = bolt::BoltType::Int64;
    col.type_size_bytes = sizeof(int64_t);
    col.format = bolt::ColumnFormat::Flat;
    col.length = rows;
    col.data = values.data();
    col.stats.all_valid = true;
    ParquetWriteOpts options{};
    options.n_columns = 1;
    std::strcpy(options.columns[0].name, "id");
    options.columns[0].type = bolt::BoltType::Int64;
    auto* writer = parquet_write_open_mem(&options);
    if (writer == nullptr) return false;
    bool ok = true;
    for (uint32_t group = 0; group < groups; ++group) {
        for (uint32_t row = 0; row < rows; ++row)
            values[row] = value_at(static_cast<uint64_t>(group) * rows + row);
        if (!parquet_write_row_group(writer, &batch)) { ok = false; break; }
    }
    bolt::Arena output;
    const uint8_t* bytes = nullptr;
    uint64_t length = 0;
    const bool closed = parquet_write_close_mem(writer, &output, &bytes, &length);
    if (!ok || !closed || length > SIZE_MAX) return false;
    file->assign(bytes, bytes + static_cast<size_t>(length));
    return true;
}

constexpr uint32_t kMixedRows = 64;
constexpr const char* kTexts[] = {"long dictionary value alpha", "long dictionary value beta",
    "long dictionary value gamma", "long dictionary value delta"};

bool mixed_file(std::vector<uint8_t>* file) {
    bolt::Arena input;
    bolt::BoltBatch batch{};
    bolt::BoltBatch::init_empty(&batch);
    if (!bolt::BoltBatch::alloc_columns(&batch, &input, 2)) return false;
    if (!batch.schema.add_field("id", bolt::BoltType::Int64, true) ||
        !batch.schema.add_field("text", bolt::BoltType::Utf8, true)) return false;
    batch.num_rows = kMixedRows;
    std::array<int64_t, kMixedRows> ids{};
    std::array<bolt::StringView, kMixedRows> strings{};
    std::array<char, kMixedRows * 64u> spill{};
    std::array<uint8_t, kMixedRows / 8u> valid{};
    uint32_t used = 0;
    for (uint32_t row = 0; row < kMixedRows; ++row) {
        const char* text = kTexts[row % 4u];
        const size_t length = std::strlen(text);
        strings[row] = bolt::StringView::from_cstr(text);
        strings[row].ref.buf_idx = 0;
        strings[row].ref.offset = used;
        std::memcpy(spill.data() + used, text, length);
        used += static_cast<uint32_t>(length);
        if (row % 5u != 0)
            valid[row / 8u] = static_cast<uint8_t>(valid[row / 8u] | (1u << (row % 8u)));
    }
    auto* cols = batch.columns[batch.read_epoch];
    for (uint32_t i = 0; i < 2; ++i) {
        cols[i] = bolt::BoltColumn::make_empty();
        cols[i].format = bolt::ColumnFormat::Flat;
        cols[i].length = kMixedRows;
        cols[i].validity = valid.data();
        cols[i].stats.all_valid = false;
    }
    cols[0].type = bolt::BoltType::Int64;
    cols[0].type_size_bytes = sizeof(int64_t);
    cols[0].data = ids.data();
    cols[1].type = bolt::BoltType::Utf8;
    cols[1].type_size_bytes = sizeof(bolt::StringView);
    cols[1].data = strings.data();
    cols[1].str_overflow_base = spill.data();
    ParquetWriteOpts options{};
    options.n_columns = 2;
    options.use_dictionary = true;
    options.compression = 1;
    std::strcpy(options.columns[0].name, "id");
    options.columns[0].type = bolt::BoltType::Int64;
    std::strcpy(options.columns[1].name, "text");
    options.columns[1].type = bolt::BoltType::Utf8;
    options.columns[0].nullable = options.columns[1].nullable = true;
    auto* writer = parquet_write_open_mem(&options);
    if (writer == nullptr) return false;
    bool ok = true;
    for (uint32_t group = 0; group < 2; ++group) {
        for (uint32_t row = 0; row < kMixedRows; ++row)
            ids[row] = static_cast<int64_t>(group) * 1000 + static_cast<int64_t>(row) * 3;
        if (!parquet_write_row_group(writer, &batch)) { ok = false; break; }
    }
    const uint8_t* data = nullptr;
    uint64_t length = 0;
    const bool closed = parquet_write_close_mem(writer, &input, &data, &length);
    if (!ok || !closed || length > SIZE_MAX) return false;
    file->assign(data, data + static_cast<size_t>(length));
    return true;
}

void mixed_values(const bolt::BoltColumn* cols, uint32_t group) {
    for (uint32_t row = 0; row < kMixedRows; ++row) {
        ASSERT_EQ(cols[0].is_null(row), row % 5u == 0);
        ASSERT_EQ(cols[1].is_null(row), row % 5u == 0);
        if (row % 5u == 0) continue;
        const uint8_t* bytes = nullptr;
        int32_t length = 0;
        cols[0].utf8_at(row, &bytes, &length);
        ASSERT_NE(bytes, nullptr);
        ASSERT_EQ(std::string(reinterpret_cast<const char*>(bytes), static_cast<size_t>(length)),
                  kTexts[row % 4u]);
        ASSERT_EQ(static_cast<const int64_t*>(cols[1].data)[row], group * 1000u + row * 3u);
    }
}

class ParquetSource : public ::testing::Test {
protected:
    std::vector<uint8_t> file;
    std::vector<uint8_t> buffer = std::vector<uint8_t>(256u * 1024u);
    bolt::Arena metadata;
    MemorySource memory;
    ParquetByteSource source{};
    ParquetSourceWorkspace workspace{};
    PqMeta* meta = nullptr;
    void bind() {
        memory.data = file.data();
        memory.length = file.size();
        source = memory.source();
        workspace = ParquetSourceWorkspace{buffer.data(), buffer.size(), 4096};
        meta = metadata.allocate_array<PqMeta>(1);
        ASSERT_NE(meta, nullptr);
        *meta = PqMeta{};
    }
    void open_small() {
        ASSERT_TRUE(integer_file(2, 32, &file));
        bind();
        ASSERT_NE(meta, nullptr);
        ASSERT_EQ(parquet_read_meta_source(&source, &workspace, &metadata, meta), Status::Ok);
    }
};

TEST_F(ParquetSource, ReadsOverSixteenMiBWithSmallReusedWorkspaceAndExactValues) {
    constexpr uint32_t groups = 300, rows = 8192;
    ASSERT_TRUE(integer_file(groups, rows, &file));
    ASSERT_GT(file.size(), 16u * 1024u * 1024u);
    bind();
    ASSERT_NE(meta, nullptr);
    ASSERT_LT(workspace.capacity, file.size() / 16u);
    ASSERT_EQ(parquet_read_meta_source(&source, &workspace, &metadata, meta), Status::Ok);
    ASSERT_EQ(meta->n_row_groups, groups);
    ASSERT_STREQ(meta->columns[0].name, "id");
    std::memset(buffer.data(), 0xA5, buffer.size());
    ASSERT_STREQ(meta->columns[0].name, "id");
    const uint16_t projection = 0;
    bolt::Arena output;
    uint64_t seen = 0;
    for (uint32_t group = 0; group < groups; ++group) {
        output.reset();
        bolt::BoltColumn column{};
        int64_t count = -1;
        ASSERT_EQ(parquet_read_row_group_cols_source(&source, &workspace, meta, group,
            &projection, 1, &output, &column, &count), Status::Ok);
        ASSERT_EQ(count, rows);
        const auto* data = static_cast<const int64_t*>(column.data);
        ASSERT_NE(data, nullptr);
        for (uint32_t row = 0; row < rows; ++row)
            ASSERT_EQ(data[row], value_at(static_cast<uint64_t>(group) * rows + row));
        seen += static_cast<uint64_t>(count);
    }
    EXPECT_EQ(seen, static_cast<uint64_t>(groups) * rows);
    EXPECT_LE(memory.largest_request, workspace.request_bytes);
    EXPECT_GT(memory.calls, groups);
    EXPECT_LT(memory.bytes, file.size() + workspace.capacity);
}

TEST_F(ParquetSource, ChangedAndReadFailuresPropagateWithoutPublishedMetadataOrRetry) {
    ASSERT_TRUE(integer_file(1, 16, &file));
    bind();
    ASSERT_NE(meta, nullptr);
    for (Status failure : {Status::Changed, Status::IoError, Status::Capacity}) {
        meta->num_rows = -17;
        memory.fail_call = memory.calls + 2u;
        memory.failure = failure;
        const uint64_t calls = memory.calls;
        EXPECT_EQ(parquet_read_meta_source(&source, &workspace, &metadata, meta), failure);
        EXPECT_EQ(meta->num_rows, -17);
        EXPECT_EQ(memory.calls, calls + 2u);
    }
}

TEST_F(ParquetSource, ShortReadAndInvalidSourceExtentFailWithoutOutput) {
    ASSERT_TRUE(integer_file(1, 16, &file));
    bind();
    ASSERT_NE(meta, nullptr);
    meta->num_rows = -31;
    memory.short_read = true;
    EXPECT_EQ(parquet_read_meta_source(&source, &workspace, &metadata, meta), Status::IoError);
    EXPECT_EQ(meta->num_rows, -31);
    memory.short_read = false;
    source.size = UINT64_MAX;
    EXPECT_EQ(parquet_read_meta_source(&source, &workspace, &metadata, meta), Status::IoError);
    EXPECT_EQ(meta->num_rows, -31);
}

TEST_F(ParquetSource, FooterAndChunkBudgetsRefuseBeforeBodyReads) {
    open_small();
    ASSERT_NE(meta, nullptr);
    const uint32_t footer = static_cast<uint32_t>(file[file.size() - 8u]) |
        (static_cast<uint32_t>(file[file.size() - 7u]) << 8u) |
        (static_cast<uint32_t>(file[file.size() - 6u]) << 16u) |
        (static_cast<uint32_t>(file[file.size() - 5u]) << 24u);
    workspace.capacity = footer - 1u;
    const uint64_t calls = memory.calls;
    const int64_t prior_rows = meta->num_rows;
    EXPECT_EQ(parquet_read_meta_source(&source, &workspace, &metadata, meta), Status::Capacity);
    EXPECT_EQ(memory.calls, calls + 2u);
    EXPECT_EQ(meta->num_rows, prior_rows);
    ASSERT_GT(meta->chunks[0].total_compressed_size, 1);
    workspace.capacity = static_cast<uint64_t>(meta->chunks[0].total_compressed_size) - 1u;
    const uint16_t projection = 0;
    bolt::Arena output;
    bolt::BoltColumn column{};
    column.length = -42;
    int64_t count = -1;
    const uint64_t before = memory.calls;
    EXPECT_EQ(parquet_read_row_group_cols_source(&source, &workspace, meta, 0,
        &projection, 1, &output, &column, &count), Status::Capacity);
    EXPECT_EQ(memory.calls, before);
    EXPECT_EQ(column.length, -42);
    EXPECT_EQ(count, 0);
}

TEST_F(ParquetSource, ChangedDuringChunkReadPreservesOutputAndStopsImmediately) {
    open_small();
    ASSERT_NE(meta, nullptr);
    const uint16_t projection = 0;
    bolt::Arena output;
    bolt::BoltColumn column{};
    column.length = -42;
    for (Status failure : {Status::Changed, Status::IoError, Status::Capacity}) {
        memory.fail_call = memory.calls + 1u;
        memory.failure = failure;
        const uint64_t calls = memory.calls;
        int64_t count = -1;
        EXPECT_EQ(parquet_read_row_group_cols_source(&source, &workspace, meta, 0,
            &projection, 1, &output, &column, &count), failure);
        EXPECT_EQ(memory.calls, calls + 1u);
        EXPECT_EQ(column.length, -42);
        EXPECT_EQ(count, 0);
    }
}

TEST_F(ParquetSource, OverflowingChunkExtentAndOversizeRequestRefuseBeforeIo) {
    open_small();
    ASSERT_NE(meta, nullptr);
    const uint16_t projection = 0;
    bolt::Arena output;
    bolt::BoltColumn column{};
    column.length = -42;
    int64_t count = -1;
    meta->chunks[0].data_page_offset = INT64_MAX;
    meta->chunks[0].dictionary_page_offset = 0;
    meta->chunks[0].total_compressed_size = INT64_MAX;
    const uint64_t before = memory.calls;
    EXPECT_EQ(parquet_read_row_group_cols_source(&source, &workspace, meta, 0,
        &projection, 1, &output, &column, &count), Status::DecodeFailed);
    EXPECT_EQ(memory.calls, before);
    EXPECT_EQ(column.length, -42);
    workspace.request_bytes = kParquetSourceMaxReadBytes + 1u;
    EXPECT_EQ(parquet_read_meta_source(&source, &workspace, &metadata, meta), Status::BadArg);
    EXPECT_EQ(memory.calls, before);
}

TEST_F(ParquetSource, DictionaryNullableUtf8AndProjectedOutputsOwnTheirBytes) {
    ASSERT_TRUE(mixed_file(&file));
    bind();
    ASSERT_NE(meta, nullptr);
    ASSERT_EQ(parquet_read_meta_source(&source, &workspace, &metadata, meta), Status::Ok);
    ASSERT_EQ(meta->n_columns, 2u);
    ASSERT_GT(meta->chunks[1].dictionary_page_offset, 0);
    const uint16_t projection[] = {1, 0};
    bolt::Arena output;
    std::array<bolt::BoltColumn, 2> first{}, second{};
    int64_t count = 0;
    ASSERT_EQ(parquet_read_row_group_cols_source(&source, &workspace, meta, 0,
        projection, 2, &output, first.data(), &count), Status::Ok);
    ASSERT_EQ(count, kMixedRows);
    std::memset(buffer.data(), 0xA5, buffer.size());
    mixed_values(first.data(), 0);
    ASSERT_EQ(parquet_read_row_group_cols_source(&source, &workspace, meta, 1,
        projection, 2, &output, second.data(), &count), Status::Ok);
    ASSERT_EQ(count, kMixedRows);
    mixed_values(second.data(), 1);
    mixed_values(first.data(), 0);
    ASSERT_STREQ(meta->columns[1].name, "text");
}

TEST_F(ParquetSource, LaterColumnFailureNeverPublishesEarlierColumn) {
    ASSERT_TRUE(mixed_file(&file));
    bind();
    ASSERT_NE(meta, nullptr);
    ASSERT_EQ(parquet_read_meta_source(&source, &workspace, &metadata, meta), Status::Ok);
    workspace.request_bytes = kParquetSourceMaxReadBytes;
    const uint16_t projection[] = {0, 1};
    bolt::Arena output;
    std::array<bolt::BoltColumn, 2> columns{};
    columns[0].length = -12;
    columns[1].length = -34;
    const uint64_t calls = memory.calls;
    memory.fail_call = calls + 2u;
    memory.failure = Status::Changed;
    int64_t rows = -1;
    EXPECT_EQ(parquet_read_row_group_cols_source(&source, &workspace, meta, 0,
        projection, 2, &output, columns.data(), &rows), Status::Changed);
    EXPECT_EQ(memory.calls, calls + 2u);
    EXPECT_EQ(columns[0].length, -12);
    EXPECT_EQ(columns[1].length, -34);
    EXPECT_EQ(rows, 0);
}

TEST_F(ParquetSource, ExistingPageCrcValidationSurvivesTheSourceBridge) {
    for (const char* name : {"page_crc_ok.parquet", "page_crc_bad.parquet", "page_crc_absent.parquet"}) {
        SCOPED_TRACE(name);
        const std::string path = std::string(BOLT_TEST_DATA_DIR) + "/" + name;
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        ASSERT_TRUE(input.good());
        const auto size = input.tellg();
        ASSERT_GT(size, 0);
        ASSERT_LE(size, 1024 * 1024);
        file.resize(static_cast<size_t>(size));
        input.seekg(0);
        ASSERT_TRUE(input.read(reinterpret_cast<char*>(file.data()),
                               static_cast<std::streamsize>(file.size())));
        bind();
        ASSERT_NE(meta, nullptr);
        ASSERT_EQ(parquet_read_meta_source(&source, &workspace, &metadata, meta), Status::Ok);
        const uint16_t projection = 0;
        bolt::Arena output;
        bolt::BoltColumn column{};
        column.length = -19;
        int64_t count = -1;
        const Status status = parquet_read_row_group_cols_source(&source, &workspace, meta, 0,
            &projection, 1, &output, &column, &count);
        if (std::strcmp(name, "page_crc_bad.parquet") == 0) {
            EXPECT_EQ(status, Status::DecodeFailed);
            EXPECT_EQ(column.length, -19);
            EXPECT_EQ(count, 0);
        } else {
            ASSERT_EQ(status, Status::Ok);
            ASSERT_EQ(count, 5);
            const auto* values = static_cast<const int64_t*>(column.data);
            ASSERT_NE(values, nullptr);
            for (int64_t row = 0; row < count; ++row) EXPECT_EQ(values[row], (row + 1) * 10);
        }
    }
}

TEST_F(ParquetSource, MalformedFooterAndZeroRequestNeverPublishMetadata) {
    ASSERT_TRUE(integer_file(1, 16, &file));
    bind();
    ASSERT_NE(meta, nullptr);
    meta->num_rows = -77;
    file[0] = 'X';
    EXPECT_EQ(parquet_read_meta_source(&source, &workspace, &metadata, meta), Status::DecodeFailed);
    EXPECT_EQ(meta->num_rows, -77);
    file[0] = 'P';
    std::memset(file.data() + file.size() - 8u, 0xFF, 4);
    const uint64_t before = memory.calls;
    EXPECT_EQ(parquet_read_meta_source(&source, &workspace, &metadata, meta), Status::DecodeFailed);
    EXPECT_EQ(memory.calls, before + 2u);
    EXPECT_EQ(meta->num_rows, -77);
    workspace.request_bytes = 0;
    const uint64_t calls = memory.calls;
    EXPECT_EQ(parquet_read_meta_source(&source, &workspace, &metadata, meta), Status::BadArg);
    EXPECT_EQ(memory.calls, calls);
}

TEST_F(ParquetSource, FlatStrictUsesTheSameBoundedFooterRequests) {
    ASSERT_TRUE(integer_file(2,32,&file));
    bind(); ASSERT_NE(meta,nullptr);
    ASSERT_EQ(parquet_read_meta_source_flat_strict(&source,&workspace,&metadata,meta),Status::Ok);
    EXPECT_EQ(meta->num_rows,64); EXPECT_EQ(meta->n_columns,1u); EXPECT_EQ(meta->n_row_groups,2u);
    EXPECT_STREQ(meta->columns[0].name,"id");
    const auto calls=memory.calls, bytes=memory.bytes;
    memory.calls=0; memory.bytes=0;
    ASSERT_EQ(parquet_read_meta_source(&source,&workspace,&metadata,meta),Status::Ok);
    EXPECT_EQ(memory.calls,calls); EXPECT_EQ(memory.bytes,bytes);
    EXPECT_LE(memory.largest_request,workspace.request_bytes);
}

TEST_F(ParquetSource, FlatStrictPreservesOutputsOnTransportAndArgumentRefusal) {
    ASSERT_TRUE(integer_file(1,16,&file));
    bind(); ASSERT_NE(meta,nullptr);
    meta->num_rows=-77;
    for (const auto failure : {Status::IoError,Status::Changed,Status::Capacity}) {
        memory.fail_call=memory.calls+3; memory.failure=failure;
        EXPECT_EQ(parquet_read_meta_source_flat_strict(&source,&workspace,&metadata,meta),failure);
        EXPECT_EQ(meta->num_rows,-77);
    }
    const auto calls=memory.calls;
    workspace.request_bytes=0;
    EXPECT_EQ(parquet_read_meta_source_flat_strict(&source,&workspace,&metadata,meta),Status::BadArg);
    EXPECT_EQ(memory.calls,calls); EXPECT_EQ(meta->num_rows,-77);
}

TEST_F(ParquetSource, FrozenOutputCapacityAndAliasedWorkspaceRefuseBeforeIo) {
    open_small();
    ASSERT_NE(meta, nullptr);
    const uint16_t projection = 0;
    bolt::Arena tiny{{64, 64}};
    tiny.set_growth_enabled(false);
    bolt::BoltColumn column{};
    column.length = -123;
    int64_t count = -1;
    const uint64_t before = memory.calls;
    EXPECT_EQ(parquet_read_row_group_cols_source(&source, &workspace, meta, 0,
        &projection, 1, &tiny, &column, &count), Status::Capacity);
    EXPECT_EQ(memory.calls, before);
    EXPECT_EQ(column.length, -123);
    EXPECT_EQ(count, 0);
    auto* alias = static_cast<uint8_t*>(metadata.allocate(1024));
    ASSERT_NE(alias, nullptr);
    ParquetSourceWorkspace aliased{alias, 1024, 1024};
    EXPECT_EQ(parquet_read_meta_source(&source, &aliased, &metadata, meta), Status::BadArg);
    EXPECT_EQ(memory.calls, before);
}

TEST_F(ParquetSource, MetadataOutputAndSourceDescriptorsCannotAliasWorkspace) {
    open_small();
    ASSERT_NE(meta, nullptr);
    bolt::Arena separate;
    const int64_t original = meta->num_rows;
    const uint64_t before = memory.calls;
    ParquetSourceWorkspace output_alias{reinterpret_cast<uint8_t*>(meta), sizeof(*meta), 1024};
    EXPECT_EQ(parquet_read_meta_source(&source, &output_alias, &separate, meta), Status::BadArg);
    EXPECT_EQ(meta->num_rows, original);
    ParquetSourceWorkspace source_alias{reinterpret_cast<uint8_t*>(&source), sizeof(source), 16};
    EXPECT_EQ(parquet_read_meta_source(&source, &source_alias, &separate, meta), Status::BadArg);
    EXPECT_EQ(source.size, file.size());
    ParquetSourceWorkspace self_alias{};
    self_alias.bytes = reinterpret_cast<uint8_t*>(&self_alias);
    self_alias.capacity = sizeof(self_alias);
    self_alias.request_bytes = 16;
    EXPECT_EQ(parquet_read_meta_source(&source, &self_alias, &separate, meta), Status::BadArg);
    ParquetSourceWorkspace arena_alias{reinterpret_cast<uint8_t*>(&separate), sizeof(separate), 16};
    EXPECT_EQ(parquet_read_meta_source(&source, &arena_alias, &separate, meta), Status::BadArg);
    EXPECT_EQ(memory.calls, before);
}

TEST_F(ParquetSource, InputMetadataAndItsOwnedArraysCannotAliasWorkspace) {
    open_small();
    ASSERT_NE(meta, nullptr);
    struct TestSpan { uint8_t* data; uint64_t length; };
    const TestSpan spans[]{{reinterpret_cast<uint8_t*>(meta), sizeof(*meta)},
        {reinterpret_cast<uint8_t*>(meta->columns), sizeof(PqColumn) * meta->n_columns},
        {reinterpret_cast<uint8_t*>(meta->chunks), sizeof(PqChunk) * meta->n_chunks}};
    bolt::Arena output;
    const uint16_t projection = 0;
    bolt::BoltColumn column{};
    column.length = -11;
    const uint64_t before = memory.calls;
    const int64_t original = meta->num_rows;
    for (const auto& span : spans) {
        ParquetSourceWorkspace alias{span.data, span.length, 1024};
        int64_t count = 19;
        EXPECT_EQ(parquet_read_row_group_cols_source(&source, &alias, meta, 0,
            &projection, 1, &output, &column, &count), Status::BadArg);
        EXPECT_EQ(count, 19);
        EXPECT_EQ(column.length, -11);
        EXPECT_EQ(meta->num_rows, original);
        EXPECT_EQ(memory.calls, before);
    }
}

TEST_F(ParquetSource, ColumnIndicesAndRowCountDescriptorsCannotAliasWorkspace) {
    open_small();
    ASSERT_NE(meta, nullptr);
    alignas(64) std::array<uint8_t, 4096> storage{};
    ParquetSourceWorkspace alias{storage.data(), storage.size(), 1024};
    const uint16_t projection = 0;
    bolt::Arena output;
    const uint64_t before = memory.calls;
    auto* overlapped_column = new (storage.data()) bolt::BoltColumn{};
    overlapped_column->length = -13;
    int64_t count = 29;
    EXPECT_EQ(parquet_read_row_group_cols_source(&source, &alias, meta, 0,
        &projection, 1, &output, overlapped_column, &count), Status::BadArg);
    EXPECT_EQ(overlapped_column->length, -13);
    EXPECT_EQ(count, 29);
    bolt::BoltColumn column{};
    column.length = -14;
    auto* overlapped_index = new (storage.data()) uint16_t(0);
    EXPECT_EQ(parquet_read_row_group_cols_source(&source, &alias, meta, 0,
        overlapped_index, 1, &output, &column, &count), Status::BadArg);
    EXPECT_EQ(*overlapped_index, 0);
    auto* overlapped_rows = new (storage.data()) int64_t(31);
    EXPECT_EQ(parquet_read_row_group_cols_source(&source, &alias, meta, 0,
        &projection, 1, &output, &column, overlapped_rows), Status::BadArg);
    EXPECT_EQ(*overlapped_rows, 31);
    EXPECT_EQ(column.length, -14);
    EXPECT_EQ(memory.calls, before);
}

TEST_F(ParquetSource, WritableDescriptorsCannotAliasMetadataOrProjection) {
    open_small();
    ASSERT_NE(meta, nullptr);
    const uint16_t projection = 0;
    bolt::Arena output;
    bolt::BoltColumn column{};
    const int64_t original = meta->num_rows;
    const uint64_t before = memory.calls;
    EXPECT_EQ(parquet_read_row_group_cols_source(&source, &workspace, meta, 0,
        &projection, 1, &output, &column, &meta->num_rows), Status::BadArg);
    EXPECT_EQ(meta->num_rows, original);
    int64_t count = 43;
    EXPECT_EQ(parquet_read_row_group_cols_source(&source, &workspace, meta, 0,
        &projection, 1, &output, reinterpret_cast<bolt::BoltColumn*>(meta), &count), Status::BadArg);
    EXPECT_EQ(meta->num_rows, original);
    EXPECT_EQ(count, 43);
    auto* indices_in_output = reinterpret_cast<const uint16_t*>(&column.length);
    column.length = 0;
    EXPECT_EQ(parquet_read_row_group_cols_source(&source, &workspace, meta, 0,
        indices_in_output, 1, &output, &column, &count), Status::BadArg);
    EXPECT_EQ(column.length, 0);
    EXPECT_EQ(count, 43);
    EXPECT_EQ(memory.calls, before);
}

TEST_F(ParquetSource, OverflowedPointerSpansRefuseBeforeIo) {
    open_small();
    ASSERT_NE(meta, nullptr);
    const uint64_t before = memory.calls;
    const int64_t original = meta->num_rows;
    ParquetSourceWorkspace bad{reinterpret_cast<uint8_t*>(UINTPTR_MAX - 3u), 16, 16};
    EXPECT_EQ(parquet_read_meta_source(&source, &bad, &metadata, meta), Status::BadArg);
    auto* bad_output = reinterpret_cast<PqMeta*>(UINTPTR_MAX - 7u);
    EXPECT_EQ(parquet_read_meta_source(&source, &workspace, &metadata, bad_output), Status::BadArg);
    EXPECT_EQ(meta->num_rows, original);
    EXPECT_EQ(memory.calls, before);
}
}  // namespace

TEST_F(ParquetSource, NestedStrictStagesOwnedTreeWithNoAdditionalFooterRequests) {
    ASSERT_TRUE(integer_file(2,32,&file));bind();ASSERT_NE(meta,nullptr);
    PqSchemaTree tree{};
    ASSERT_EQ(parquet_read_meta_source_nested_strict(&source,&workspace,&metadata,meta,&tree),Status::Ok);
    const auto calls=memory.calls,bytes=memory.bytes;
    EXPECT_EQ(tree.n_top,1u);ASSERT_NE(tree.nodes,nullptr);EXPECT_STREQ(tree.nodes[1].name,"id");
    EXPECT_FALSE(metadata.overlaps_backing(workspace.bytes,static_cast<size_t>(workspace.capacity)));
    std::memset(workspace.bytes,0xA5,static_cast<size_t>(workspace.capacity));
    EXPECT_STREQ(tree.nodes[1].name,"id");EXPECT_STREQ(meta->columns[0].name,"id");
    memory.calls=0;memory.bytes=0;PqSchemaTree ordinary{};
    ASSERT_EQ(parquet_read_meta_source_nested(&source,&workspace,&metadata,meta,&ordinary),Status::Ok);
    EXPECT_EQ(memory.calls,calls);EXPECT_EQ(memory.bytes,bytes);
    EXPECT_LE(memory.largest_request,workspace.request_bytes);
}
TEST_F(ParquetSource, NestedStrictPreservesBothOutputsOnFailureAndRejectsAliasesBeforeIo) {
    ASSERT_TRUE(integer_file(1,16,&file));bind();ASSERT_NE(meta,nullptr);
    meta->num_rows=-77;PqSchemaTree tree{};tree.n_top=91;
    for(const auto failure:{Status::IoError,Status::Changed,Status::Capacity}) {
        memory.fail_call=memory.calls+3;memory.failure=failure;
        EXPECT_EQ(parquet_read_meta_source_nested_strict(&source,&workspace,&metadata,meta,&tree),failure);
        EXPECT_EQ(meta->num_rows,-77);EXPECT_EQ(tree.n_top,91u);EXPECT_EQ(tree.nodes,nullptr);
    }
    const auto calls=memory.calls;
    auto* alias=reinterpret_cast<PqSchemaTree*>(workspace.bytes);
    EXPECT_EQ(parquet_read_meta_source_nested_strict(&source,&workspace,&metadata,meta,alias),Status::BadArg);
    EXPECT_EQ(memory.calls,calls);EXPECT_EQ(meta->num_rows,-77);EXPECT_EQ(tree.n_top,91u);
}
