// RowGroup completeness: file_offset / total_compressed_size / ordinal
// (G2PQ-19 tracker id, spec item B8).
//
// RowGroup fields 5/6/7 are all `optional` in parquet.thrift, so their
// absence is legal -- but bolt's own writer has never populated them, which
// leaves a reader that wants to plan around them (seek straight to a row
// group's first page, budget an I/O read without walking every ColumnChunk,
// or report a row group's position in the file) with nothing to read.
//
// pyarrow's Python binding does not expose these three fields at all
// (pyarrow.parquet.RowGroupMetaData only surfaces num_rows/total_byte_size/
// sorting_columns/column(i) -- confirmed by inspection, not assumed), so it
// cannot serve as an oracle here. The independent check instead cross-
// validates two values that are encoded and decoded through entirely
// separate thrift structs: RowGroup.total_compressed_size (field 6, summary)
// must equal the sum of that row group's own ColumnMetaData.
// total_compressed_size (field 7, per-chunk) -- if the writer's summary and
// the per-chunk detail ever disagreed, that would be a real silent-misread
// bug, not a formatting nit.

#include "bolt/ingest/bolt_parquet_write.h"
#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt/ingest/bolt_parquet_meta.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"

namespace {

using namespace bolt::ingest::parquet;

std::vector<std::uint8_t> slurp_file(const char* path) {
    std::vector<std::uint8_t> v;
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) return v;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    v.resize(static_cast<std::size_t>(n));
    const std::size_t got = std::fread(v.data(), 1, v.size(), f);
    std::fclose(f);
    if (got != v.size()) v.clear();
    return v;
}

std::string tmp_path(const char* tag) {
    std::string s = "test_bolt_parquet_write_rowgroup_meta_";
    s += tag;
    s += ".parquet";
    return s;
}

void build_int64_batch(bolt::Arena* arena, std::int64_t n,
                       bolt::BoltBatch* out) {
    bolt::BoltBatch::init_empty(out);
    out->num_cols = 1;
    out->num_rows = n;
    bolt::BoltBatch::alloc_columns(out, arena, 1);
    out->schema.add_field("v", bolt::BoltType::Int64, false);
    bolt::BoltColumn& v = out->columns[out->read_epoch][0];
    v = bolt::BoltColumn::make_flat_alloc(n, bolt::BoltType::Int64, arena);
    ASSERT_NE(v.data, nullptr);
    auto* p = static_cast<std::int64_t*>(v.data);
    for (std::int64_t i = 0; i < n; ++i) p[i] = i * 7 + 3;
}

// Parses the footer and returns the parsed PqMeta; `chunks` must outlive it
// (PqMeta::chunks is a borrowed pointer, per the reader's own contract).
void parse_footer(const std::vector<std::uint8_t>& buf, PqMeta* m,
                  std::vector<PqChunk>* chunks) {
    chunks->resize(kPqMaxColumns * 64);
    std::memset(m, 0, sizeof(*m));
    m->chunks = chunks->data();
    m->chunks_cap = static_cast<std::uint32_t>(chunks->size());
    std::uint64_t off = 0;
    std::uint32_t len = 0;
    ASSERT_TRUE(pq_locate_footer(buf.data(), buf.size(), &off, &len));
    ASSERT_TRUE(pq_parse_file_meta(buf.data() + off, len, m));
}

}  // namespace

TEST(BoltParquetWriteRowGroupMeta, SingleRowGroupFieldsPopulated) {
    const std::int64_t kRows = 200;
    bolt::Arena arena;
    bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(batch, nullptr);
    build_int64_batch(&arena, kRows, batch);

    ParquetWriteOpts opts{};
    opts.n_columns = 1;
    opts.compression = 0;
    opts.emit_statistics = true;
    std::strncpy(opts.columns[0].name, "v", sizeof(opts.columns[0].name));
    opts.columns[0].type = bolt::BoltType::Int64;
    opts.columns[0].nullable = false;

    const std::string path = tmp_path("single");
    ParquetWriter* w = parquet_write_open(path.c_str(), &opts);
    ASSERT_NE(w, nullptr);
    ASSERT_TRUE(parquet_write_row_group(w, batch));
    ASSERT_TRUE(parquet_write_close(w));

    const auto buf = slurp_file(path.c_str());
    ASSERT_FALSE(buf.empty());
    PqMeta m{};
    std::vector<PqChunk> chunks;
    parse_footer(buf, &m, &chunks);

    ASSERT_EQ(m.n_row_groups, 1u);
    const PqRowGroup& rg = m.row_groups[0];

    // file_offset (field 5): the one and only row group's first page starts
    // immediately after the 4-byte "PAR1" magic -- nothing else precedes it.
    EXPECT_EQ(rg.file_offset, 4);

    // ordinal (field 7): the only row group in the file is index 0.
    EXPECT_EQ(rg.ordinal, 0);

    // total_compressed_size (field 6): must equal the independently-decoded
    // sum of this row group's own ColumnMetaData.total_compressed_size
    // fields (field 7 WITHIN ColumnChunk.meta_data -- a different struct,
    // parsed by a different code path in parse_column_chunk).
    ASSERT_EQ(rg.chunk_count, 1u);
    std::int64_t sum_chunk_compressed = 0;
    for (std::uint32_t c = 0; c < rg.chunk_count; ++c) {
        sum_chunk_compressed += chunks[rg.chunk_off + c].total_compressed_size;
    }
    EXPECT_GT(sum_chunk_compressed, 0);
    EXPECT_EQ(rg.total_compressed_size, sum_chunk_compressed);

    std::remove(path.c_str());
}

TEST(BoltParquetWriteRowGroupMeta, MultipleRowGroupsOrdinalAndOffsetsAreConsistent) {
    // row_group_max_rows splits ONE parquet_write_row_group call's batch
    // into consecutive row groups -- 350 rows / 100-per-group -> 4 groups
    // (100, 100, 100, 50), the shape most bolt-written multi-row-group files
    // actually take (G2FEAT-24).
    const std::int64_t kRows = 350;
    const std::uint32_t kRowGroupMaxRows = 100;
    bolt::Arena arena;
    bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(batch, nullptr);
    build_int64_batch(&arena, kRows, batch);

    ParquetWriteOpts opts{};
    opts.n_columns = 1;
    opts.compression = 0;
    opts.emit_statistics = true;
    opts.row_group_max_rows = kRowGroupMaxRows;
    std::strncpy(opts.columns[0].name, "v", sizeof(opts.columns[0].name));
    opts.columns[0].type = bolt::BoltType::Int64;
    opts.columns[0].nullable = false;

    const std::string path = tmp_path("multi");
    ParquetWriter* w = parquet_write_open(path.c_str(), &opts);
    ASSERT_NE(w, nullptr);
    ASSERT_TRUE(parquet_write_row_group(w, batch));
    ASSERT_TRUE(parquet_write_close(w));

    const auto buf = slurp_file(path.c_str());
    ASSERT_FALSE(buf.empty());
    PqMeta m{};
    std::vector<PqChunk> chunks;
    parse_footer(buf, &m, &chunks);

    ASSERT_EQ(m.n_row_groups, 4u);
    std::uint64_t footer_off = 0;
    std::uint32_t footer_len = 0;
    ASSERT_TRUE(pq_locate_footer(buf.data(), buf.size(), &footer_off, &footer_len));

    std::int64_t prev_file_offset = -1;
    std::int64_t total_rows = 0;
    for (std::uint32_t g = 0; g < m.n_row_groups; ++g) {
        const PqRowGroup& rg = m.row_groups[g];
        total_rows += rg.num_rows;

        // ordinal (field 7) must exactly track position in the file.
        EXPECT_EQ(rg.ordinal, static_cast<std::int16_t>(g)) << "row group " << g;

        // file_offset (field 5) must be inside the row-group region of the
        // file (past the magic, before the footer) and strictly increasing
        // -- each row group's first page comes after the previous one's.
        EXPECT_GT(rg.file_offset, prev_file_offset) << "row group " << g;
        EXPECT_LT(rg.file_offset, static_cast<std::int64_t>(footer_off))
            << "row group " << g;
        prev_file_offset = rg.file_offset;
        if (g == 0) EXPECT_EQ(rg.file_offset, 4);

        // total_compressed_size (field 6) cross-checked against the
        // independently-parsed per-chunk field, same as the single-row-group
        // case above.
        std::int64_t sum_chunk_compressed = 0;
        for (std::uint32_t c = 0; c < rg.chunk_count; ++c) {
            sum_chunk_compressed +=
                chunks[rg.chunk_off + c].total_compressed_size;
        }
        EXPECT_GT(sum_chunk_compressed, 0) << "row group " << g;
        EXPECT_EQ(rg.total_compressed_size, sum_chunk_compressed)
            << "row group " << g;
    }
    EXPECT_EQ(total_rows, kRows);

    std::remove(path.c_str());
}
