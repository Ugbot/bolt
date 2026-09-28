// ParquetWriteColumn::ts_local: TIMESTAMP WITHOUT TIME ZONE
// (isAdjustedToUTC=false, MICROS, no ConvertedType). The default stays the
// UTC-adjusted TIMESTAMP_MICROS form. Read back through bolt's footer parser.

#include "pq_test_columns.h"
#include "bolt/ingest/bolt_parquet_write.h"
#include "bolt/ingest/bolt_parquet_meta.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"

namespace {

using namespace bolt::ingest::parquet;

void write_and_parse(bool ts_local, PqMeta* m, std::vector<PqChunk>* chunks,
                     bolt::Arena* out_arena) {
    bolt::Arena arena;
    bolt::BoltBatch* b = arena.allocate_array<bolt::BoltBatch>(1);
    bolt::BoltBatch::init_empty(b);
    b->num_cols = 1;
    b->num_rows = 2;
    bolt::BoltBatch::alloc_columns(b, &arena, 1);
    b->schema.add_field("t", bolt::BoltType::Timestamp, false);
    bolt::BoltColumn& c = b->columns[b->read_epoch][0];
    c = bolt::BoltColumn::make_flat_alloc(2, bolt::BoltType::Timestamp, &arena);
    auto* p = static_cast<std::int64_t*>(c.data);
    p[0] = -1; p[1] = 1700000000123456;

    ParquetWriteOpts opts{};
    opts.n_columns = 1;
    std::strncpy(opts.columns[0].name, "t", sizeof(opts.columns[0].name));
    opts.columns[0].type = bolt::BoltType::Timestamp;
    opts.columns[0].ts_local = ts_local;
    ParquetWriter* w = parquet_write_open_mem(&opts);
    ASSERT_NE(w, nullptr);
    ASSERT_TRUE(parquet_write_row_group(w, b));
    const std::uint8_t* out = nullptr;
    std::uint64_t out_len = 0;
    ASSERT_TRUE(parquet_write_close_mem(w, out_arena, &out, &out_len));
    std::memset(m, 0, sizeof(*m));
    m->chunks = chunks->data();
    m->chunks_cap = static_cast<std::uint32_t>(chunks->size());
    m->columns = pq_test_columns();
    m->columns_cap = kPqTestColumns;
    std::uint64_t off = 0;
    std::uint32_t len = 0;
    ASSERT_TRUE(pq_locate_footer(out, out_len, &off, &len));
    ASSERT_TRUE(pq_parse_file_meta(out + off, len, m));
}

}  // namespace

TEST(BoltParquetWriteTsLocal, LocalTimestampIsNotUtcAdjusted) {
    bolt::Arena oa;
    PqMeta m;
    std::vector<PqChunk> chunks(8);
    write_and_parse(true, &m, &chunks, &oa);
    ASSERT_EQ(m.n_columns, 1u);
    EXPECT_EQ(m.columns[0].logical, static_cast<int32_t>(PqLogical::Timestamp));
    EXPECT_EQ(m.columns[0].ts_utc, 0u);
    EXPECT_EQ(m.columns[0].time_unit, 2);
    EXPECT_EQ(m.columns[0].converted, -1);
}

TEST(BoltParquetWriteTsLocal, DefaultStaysUtcAdjusted) {
    bolt::Arena oa;
    PqMeta m;
    std::vector<PqChunk> chunks(8);
    write_and_parse(false, &m, &chunks, &oa);
    ASSERT_EQ(m.n_columns, 1u);
    // Only the legacy TIMESTAMP_MICROS ConvertedType (UTC-adjusted).
    EXPECT_EQ(m.columns[0].converted, 10);
}
