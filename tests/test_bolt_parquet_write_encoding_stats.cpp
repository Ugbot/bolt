// G2PQ-24: ColumnMetaData.encoding_stats fixtures. Assertions live in
// scripts/parquet_encoding_stats_check.py (pyarrow doesn't expose this
// field, so it decodes the thrift directly and cross-checks page counts
// against the independently-parsed OffsetIndex).

#include "bolt/ingest/bolt_parquet_write.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"

namespace {

using namespace bolt::ingest::parquet;

void put_int64_col(bolt::Arena* a, bolt::BoltBatch* b, std::uint16_t slot,
                   const char* name, std::int64_t n) {
    b->schema.add_field(name, bolt::BoltType::Int64, false);
    bolt::BoltColumn& c = b->columns[b->read_epoch][slot];
    c = bolt::BoltColumn::make_flat_alloc(n, bolt::BoltType::Int64, a);
    ASSERT_NE(c.data, nullptr);
    auto* p = static_cast<std::int64_t*>(c.data);
    for (std::int64_t i = 0; i < n; ++i) p[i] = i * 3 + 1;
}

void put_dict_utf8_col(bolt::Arena* a, bolt::BoltBatch* b, std::uint16_t slot,
                       const char* name, std::int64_t n) {
    b->schema.add_field(name, bolt::BoltType::Utf8, false);
    bolt::BoltColumn& c = b->columns[b->read_epoch][slot];
    c = bolt::BoltColumn::make_empty();
    c.length = n;
    c.format = bolt::ColumnFormat::Flat;
    c.type = bolt::BoltType::Utf8;
    c.type_size_bytes = sizeof(bolt::StringView);
    auto* svs = static_cast<bolt::StringView*>(
        a->allocate(static_cast<std::size_t>(n) * sizeof(bolt::StringView),
                    alignof(bolt::StringView)));
    ASSERT_NE(svs, nullptr);
    std::memset(svs, 0, static_cast<std::size_t>(n) * sizeof(bolt::StringView));
    c.data = svs;
    c.stats.all_valid = true;
    for (std::int64_t i = 0; i < n; ++i) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "v%d", static_cast<int>(i % 5));
        svs[i] = bolt::StringView::from_cstr(buf);
    }
}

}  // namespace

TEST(BoltParquetWriteEncodingStats, PlainOnePageFixture) {
    const std::int64_t kRows = 100;
    bolt::Arena arena;
    bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(batch, nullptr);
    bolt::BoltBatch::init_empty(batch);
    batch->num_cols = 1;
    batch->num_rows = kRows;
    bolt::BoltBatch::alloc_columns(batch, &arena, 1);
    put_int64_col(&arena, batch, 0, "plain_one_page", kRows);

    ParquetWriteOpts opts{};
    opts.n_columns = 1;
    opts.compression = 0;
    opts.emit_statistics = true;
    opts.use_dictionary = false;
    std::strncpy(opts.columns[0].name, "plain_one_page",
                sizeof(opts.columns[0].name));
    opts.columns[0].type = bolt::BoltType::Int64;
    opts.columns[0].nullable = false;

    const std::string path = "test_bolt_parquet_write_encoding_stats_plain.parquet";
    ParquetWriter* w = parquet_write_open(path.c_str(), &opts);
    ASSERT_NE(w, nullptr);
    ASSERT_TRUE(parquet_write_row_group(w, batch));
    ASSERT_TRUE(parquet_write_close(w));
}

TEST(BoltParquetWriteEncodingStats, DictionaryOnePageFixture) {
    const std::int64_t kRows = 100;
    bolt::Arena arena;
    bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(batch, nullptr);
    bolt::BoltBatch::init_empty(batch);
    batch->num_cols = 1;
    batch->num_rows = kRows;
    bolt::BoltBatch::alloc_columns(batch, &arena, 1);
    put_dict_utf8_col(&arena, batch, 0, "dict_col", kRows);

    ParquetWriteOpts opts{};
    opts.n_columns = 1;
    opts.compression = 0;
    opts.emit_statistics = true;
    opts.use_dictionary = true;
    std::strncpy(opts.columns[0].name, "dict_col", sizeof(opts.columns[0].name));
    opts.columns[0].type = bolt::BoltType::Utf8;
    opts.columns[0].nullable = false;

    const std::string path = "test_bolt_parquet_write_encoding_stats_dict.parquet";
    ParquetWriter* w = parquet_write_open(path.c_str(), &opts);
    ASSERT_NE(w, nullptr);
    ASSERT_TRUE(parquet_write_row_group(w, batch));
    ASSERT_TRUE(parquet_write_close(w));
}

TEST(BoltParquetWriteEncodingStats, MultiPageFixtureWithPageIndex) {
    const std::int64_t kRows = 20000;
    bolt::Arena arena;
    bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(batch, nullptr);
    bolt::BoltBatch::init_empty(batch);
    batch->num_cols = 1;
    batch->num_rows = kRows;
    bolt::BoltBatch::alloc_columns(batch, &arena, 1);
    put_int64_col(&arena, batch, 0, "multi_page", kRows);

    ParquetWriteOpts opts{};
    opts.n_columns = 1;
    opts.compression = 0;
    opts.emit_statistics = true;
    opts.emit_page_index = true;
    opts.data_page_target_bytes = 4096;
    std::strncpy(opts.columns[0].name, "multi_page", sizeof(opts.columns[0].name));
    opts.columns[0].type = bolt::BoltType::Int64;
    opts.columns[0].nullable = false;

    const std::string path = "test_bolt_parquet_write_encoding_stats_multipage.parquet";
    ParquetWriter* w = parquet_write_open(path.c_str(), &opts);
    ASSERT_NE(w, nullptr);
    ASSERT_TRUE(parquet_write_row_group(w, batch));
    ASSERT_TRUE(parquet_write_close(w));
}
