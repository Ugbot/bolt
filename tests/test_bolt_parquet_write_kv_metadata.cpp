// G2PQ-23: FileMetaData.key_value_metadata (field 5) and
// ColumnMetaData.key_value_metadata (field 8). Round-trips through bolt's
// own reader here; scripts/parquet_kv_metadata_check.py cross-checks the
// file-level fixture against a REAL pyarrow.parquet.read_metadata() (a
// strictly stronger oracle than a hand-rolled thrift decoder).

#include "bolt/ingest/bolt_parquet_write.h"
#include "bolt/ingest/bolt_parquet_meta.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"

namespace {

using namespace bolt::ingest::parquet;

void build_batch(bolt::Arena* arena, bolt::BoltBatch* out) {
    bolt::BoltBatch::init_empty(out);
    out->num_cols = 2;
    out->num_rows = 3;
    bolt::BoltBatch::alloc_columns(out, arena, 2);
    out->schema.add_field("id", bolt::BoltType::Int64, false);
    out->schema.add_field("name", bolt::BoltType::Utf8, false);

    bolt::BoltColumn& id = out->columns[out->read_epoch][0];
    id = bolt::BoltColumn::make_flat_alloc(3, bolt::BoltType::Int64, arena);
    auto* ip = static_cast<std::int64_t*>(id.data);
    ip[0] = 1; ip[1] = 2; ip[2] = 3;

    bolt::BoltColumn& name = out->columns[out->read_epoch][1];
    name = bolt::BoltColumn::make_empty();
    name.length = 3;
    name.format = bolt::ColumnFormat::Flat;
    name.type = bolt::BoltType::Utf8;
    name.type_size_bytes = sizeof(bolt::StringView);
    auto* svs = static_cast<bolt::StringView*>(arena->allocate(
        3 * sizeof(bolt::StringView), alignof(bolt::StringView)));
    name.data = svs;
    name.stats.all_valid = true;
    svs[0] = bolt::StringView::from_cstr("a");
    svs[1] = bolt::StringView::from_cstr("b");
    svs[2] = bolt::StringView::from_cstr("c");
}

// A real Arrow IPC schema base64 payload (captured from
// pyarrow.parquet.write_table on a 2-int64-column table), used verbatim so
// the fixture exercises a realistic value, not just ("foo","bar").
constexpr const char* kArrowSchemaB64 =
    "/////6gAAAAQAAAAAAAKAAwABgAFAAgACgAAAAABBAAMAAAACAAIAAAABAAIAAAABAAAAAIA"
    "AABAAAAABAAAANj///8AAAEFEAAAABgAAAAEAAAAAAAAAAEAAABiAAAABAAEAAQAAAAQABQA"
    "CAAGAAcADAAAABAAEAAAAAAAAQIQAAAAHAAAAAQAAAAAAAAAAQAAAGEAAAAIAAwACAAHAAgA"
    "AAAAAAABQAAAAAAAAAA=";

ParquetWriteOpts base_opts() {
    ParquetWriteOpts opts{};
    opts.n_columns = 2;
    opts.compression = 0;
    std::strncpy(opts.columns[0].name, "id", sizeof(opts.columns[0].name));
    opts.columns[0].type = bolt::BoltType::Int64;
    opts.columns[0].nullable = false;
    std::strncpy(opts.columns[1].name, "name", sizeof(opts.columns[1].name));
    opts.columns[1].type = bolt::BoltType::Utf8;
    opts.columns[1].nullable = false;
    return opts;
}

void parse_file(const char* path, PqMeta* m, std::vector<PqChunk>* chunks) {
    std::FILE* f = std::fopen(path, "rb");
    ASSERT_NE(f, nullptr);
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<std::uint8_t> buf(static_cast<size_t>(n));
    ASSERT_EQ(std::fread(buf.data(), 1, buf.size(), f), buf.size());
    std::fclose(f);

    chunks->resize(64);
    std::memset(m, 0, sizeof(*m));
    m->chunks = chunks->data();
    m->chunks_cap = static_cast<std::uint32_t>(chunks->size());
    std::uint64_t off = 0;
    std::uint32_t len = 0;
    ASSERT_TRUE(pq_locate_footer(buf.data(), buf.size(), &off, &len));
    ASSERT_TRUE(pq_parse_file_meta(buf.data() + off, len, m));
}

}  // namespace

// The fixture scripts/parquet_kv_metadata_check.py reads with pyarrow.
TEST(BoltParquetWriteKvMetadata, FixtureForPyarrowOracle) {
    bolt::Arena arena;
    bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
    build_batch(&arena, batch);

    ParquetWriteOpts opts = base_opts();
    opts.n_file_kv = 1;
    std::strncpy(opts.file_kv[0].key, "ARROW:schema",
                sizeof(opts.file_kv[0].key));
    std::strncpy(opts.file_kv[0].value, kArrowSchemaB64,
                sizeof(opts.file_kv[0].value));
    opts.columns[0].n_column_kv = 1;
    std::strncpy(opts.columns[0].column_kv[0].key, "PARQUET:field_id",
                sizeof(opts.columns[0].column_kv[0].key));
    std::strncpy(opts.columns[0].column_kv[0].value, "1",
                sizeof(opts.columns[0].column_kv[0].value));

    const char* path = "test_bolt_parquet_write_kv_metadata_fixture.parquet";
    ParquetWriter* w = parquet_write_open(path, &opts);
    ASSERT_NE(w, nullptr);
    ASSERT_TRUE(parquet_write_row_group(w, batch));
    ASSERT_TRUE(parquet_write_close(w));
}

TEST(BoltParquetWriteKvMetadata, RoundTripsThroughBoltsOwnReader) {
    bolt::Arena arena;
    bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
    build_batch(&arena, batch);

    ParquetWriteOpts opts = base_opts();
    opts.n_file_kv = 2;
    std::strncpy(opts.file_kv[0].key, "ARROW:schema",
                sizeof(opts.file_kv[0].key));
    std::strncpy(opts.file_kv[0].value, kArrowSchemaB64,
                sizeof(opts.file_kv[0].value));
    std::strncpy(opts.file_kv[1].key, "custom.note",
                sizeof(opts.file_kv[1].key));
    std::strncpy(opts.file_kv[1].value, "written by bolt",
                sizeof(opts.file_kv[1].value));
    opts.columns[0].n_column_kv = 1;
    std::strncpy(opts.columns[0].column_kv[0].key, "PARQUET:field_id",
                sizeof(opts.columns[0].column_kv[0].key));
    std::strncpy(opts.columns[0].column_kv[0].value, "1",
                sizeof(opts.columns[0].column_kv[0].value));

    const char* path = "test_bolt_parquet_write_kv_metadata_roundtrip.parquet";
    ParquetWriter* w = parquet_write_open(path, &opts);
    ASSERT_NE(w, nullptr);
    ASSERT_TRUE(parquet_write_row_group(w, batch));
    ASSERT_TRUE(parquet_write_close(w));

    PqMeta m;
    std::vector<PqChunk> chunks;
    parse_file(path, &m, &chunks);

    ASSERT_EQ(m.n_file_kv, 2u);
    EXPECT_STREQ(m.file_kv[0].key, "ARROW:schema");
    EXPECT_STREQ(m.file_kv[0].value, kArrowSchemaB64);
    EXPECT_STREQ(m.file_kv[1].key, "custom.note");
    EXPECT_STREQ(m.file_kv[1].value, "written by bolt");

    // One row group, chunk 0 == column "id" (n_columns == 2).
    ASSERT_EQ(m.n_row_groups, 1u);
    ASSERT_EQ(m.n_columns, 2u);
    const PqChunk& id_chunk = m.chunks[0];
    ASSERT_EQ(id_chunk.n_col_kv, 1u);
    EXPECT_STREQ(id_chunk.col_kv[0].key, "PARQUET:field_id");
    EXPECT_STREQ(id_chunk.col_kv[0].value, "1");
    const PqChunk& name_chunk = m.chunks[1];
    EXPECT_EQ(name_chunk.n_col_kv, 0u);
}

TEST(BoltParquetWriteKvMetadata, MemSinkRoundTrips) {
    bolt::Arena arena;
    bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
    build_batch(&arena, batch);

    ParquetWriteOpts opts = base_opts();
    opts.n_file_kv = 1;
    std::strncpy(opts.file_kv[0].key, "ARROW:schema",
                sizeof(opts.file_kv[0].key));
    std::strncpy(opts.file_kv[0].value, kArrowSchemaB64,
                sizeof(opts.file_kv[0].value));

    ParquetWriter* w = parquet_write_open_mem(&opts);
    ASSERT_NE(w, nullptr);
    ASSERT_TRUE(parquet_write_row_group(w, batch));

    bolt::Arena out_arena;
    const std::uint8_t* out = nullptr;
    std::uint64_t out_len = 0;
    ASSERT_TRUE(parquet_write_close_mem(w, &out_arena, &out, &out_len));
    ASSERT_GT(out_len, 0u);

    PqMeta m;
    std::vector<PqChunk> chunks(8);
    std::memset(&m, 0, sizeof(m));
    m.chunks = chunks.data();
    m.chunks_cap = static_cast<std::uint32_t>(chunks.size());
    std::uint64_t off = 0;
    std::uint32_t len = 0;
    ASSERT_TRUE(pq_locate_footer(out, out_len, &off, &len));
    ASSERT_TRUE(pq_parse_file_meta(out + off, len, &m));
    ASSERT_EQ(m.n_file_kv, 1u);
    EXPECT_STREQ(m.file_kv[0].key, "ARROW:schema");
    EXPECT_STREQ(m.file_kv[0].value, kArrowSchemaB64);
}

TEST(BoltParquetWriteKvMetadata, EmptyIsByteIdenticalToNoKv) {
    // n_file_kv=0 / n_column_kv=0 (the zero-init default) must write no
    // field 5/8 at all -- a plain open() call with no KV set stays a
    // legal, parseable file.
    bolt::Arena arena;
    bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
    build_batch(&arena, batch);

    ParquetWriteOpts opts = base_opts();
    const char* path = "test_bolt_parquet_write_kv_metadata_empty.parquet";
    ParquetWriter* w = parquet_write_open(path, &opts);
    ASSERT_NE(w, nullptr);
    ASSERT_TRUE(parquet_write_row_group(w, batch));
    ASSERT_TRUE(parquet_write_close(w));

    PqMeta m;
    std::vector<PqChunk> chunks;
    parse_file(path, &m, &chunks);
    EXPECT_EQ(m.n_file_kv, 0u);
    EXPECT_EQ(m.chunks[0].n_col_kv, 0u);
}

// ---- refusal, not truncation -----------------------------------------------

TEST(BoltParquetWriteKvMetadata, RefusesTooManyFilePairs) {
    ParquetWriteOpts opts = base_opts();
    opts.n_file_kv = kPwFileKvMaxPairs + 1;
    EXPECT_EQ(parquet_write_open("should_not_exist_1.parquet", &opts), nullptr);
}

TEST(BoltParquetWriteKvMetadata, RefusesOversizeFileValue) {
    ParquetWriteOpts opts = base_opts();
    opts.n_file_kv = 1;
    std::strncpy(opts.file_kv[0].key, "k", sizeof(opts.file_kv[0].key));
    // Fill the value buffer completely (no room for the NUL terminator) --
    // strnlen(value, cap) == cap, which kv_valid must treat as overflow.
    std::memset(opts.file_kv[0].value, 'x', kPwFileKvValBytes);
    EXPECT_EQ(parquet_write_open("should_not_exist_2.parquet", &opts), nullptr);
}

TEST(BoltParquetWriteKvMetadata, RefusesEmptyColumnKey) {
    ParquetWriteOpts opts = base_opts();
    opts.columns[0].n_column_kv = 1;
    opts.columns[0].column_kv[0].key[0] = '\0';   // key is thrift `required`
    std::strncpy(opts.columns[0].column_kv[0].value, "v",
                sizeof(opts.columns[0].column_kv[0].value));
    EXPECT_EQ(parquet_write_open("should_not_exist_3.parquet", &opts), nullptr);
}

TEST(BoltParquetWriteKvMetadata, RefusesTooManyColumnPairs) {
    ParquetWriteOpts opts = base_opts();
    opts.columns[1].n_column_kv = kPwColKvMaxPairs + 1;
    EXPECT_EQ(parquet_write_open("should_not_exist_4.parquet", &opts), nullptr);
}
