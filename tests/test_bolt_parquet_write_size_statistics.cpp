// G2PQ-25: ColumnMetaData.size_statistics (field 16) fixtures. pyarrow does
// not expose this field (checked directly against pyarrow 21's
// ColumnChunkMetaData), so scripts/parquet_size_statistics_check.py holds
// the assertions, hand-decoding thrift the same self-contained way
// scripts/parquet_encoding_stats_check.py does for field 13.

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

std::vector<std::uint8_t> slurp(const char* path) {
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

// Round-trip check for the READER half (G2PQ-25 item 4): re-parse the file
// bolt just wrote and hand back column 0's PqChunk. The Python oracle checks
// the WRITTEN bytes independent of bolt's own reader; this checks the reader
// agrees with itself, which the oracle alone cannot prove.
bool read_back_chunk0(const char* path, bolt::Arena* a,
                      bolt::ingest::parquet::PqChunk* out) {
    const auto buf = slurp(path);
    if (buf.empty()) return false;
    bolt::ingest::parquet::PqMeta meta{};
    if (!bolt::ingest::parquet::parquet_read_meta(buf.data(), buf.size(), a,
                                                  &meta)) {
        return false;
    }
    if (meta.n_chunks == 0) return false;
    *out = meta.chunks[0];
    return true;
}

using namespace bolt::ingest::parquet;

// ---- flat column fixtures --------------------------------------------------

void put_utf8_col(bolt::Arena* a, bolt::BoltBatch* b, std::uint16_t slot,
                  const char* name, const std::vector<std::string>& vals) {
    const std::int64_t n = static_cast<std::int64_t>(vals.size());
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
        svs[i] = bolt::StringView::from_cstr(vals[static_cast<std::size_t>(i)].c_str());
    }
}

// Deterministic per-row length, restated in scripts/parquet_size_statistics_check.py.
std::string flat_plain_val(int i) {
    return std::string(static_cast<std::size_t>((i % 9) + 1), 'x');
}

TEST(BoltParquetWriteSizeStatistics, FlatUtf8PlainBytesKnownNoHistogram) {
    const int kRows = 50;
    std::vector<std::string> vals;
    for (int i = 0; i < kRows; ++i) vals.push_back(flat_plain_val(i));

    bolt::Arena arena;
    bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(batch, nullptr);
    bolt::BoltBatch::init_empty(batch);
    batch->num_cols = 1;
    batch->num_rows = kRows;
    bolt::BoltBatch::alloc_columns(batch, &arena, 1);
    put_utf8_col(&arena, batch, 0, "s", vals);

    ParquetWriteOpts opts{};
    opts.n_columns = 1;
    opts.compression = 0;
    opts.emit_statistics = true;
    opts.emit_size_statistics = true;
    opts.use_dictionary = false;
    std::strncpy(opts.columns[0].name, "s", sizeof(opts.columns[0].name));
    opts.columns[0].type = bolt::BoltType::Utf8;
    opts.columns[0].nullable = false;

    const std::string path = "test_bolt_parquet_write_size_statistics_flat_plain.parquet";
    ParquetWriter* w = parquet_write_open(path.c_str(), &opts);
    ASSERT_NE(w, nullptr);
    ASSERT_TRUE(parquet_write_row_group(w, batch));
    ASSERT_TRUE(parquet_write_close(w));

    std::int64_t want = 0;
    for (int i = 0; i < kRows; ++i) want += static_cast<std::int64_t>(flat_plain_val(i).size());
    bolt::Arena ra;
    PqChunk ch{};
    ASSERT_TRUE(read_back_chunk0(path.c_str(), &ra, &ch));
    EXPECT_EQ(ch.byte_array_data_bytes, want);
    EXPECT_EQ(ch.def_hist_len, 0u) << "flat column must have no histogram";
    EXPECT_EQ(ch.rep_hist_len, 0u);
}

// Distinct strings so dictionary encoding collapses 300 rows into 5 entries;
// unencoded_byte_array_data_bytes must still count all 300 LOGICAL rows, not
// the 5 distinct dictionary entries -- restated in the check script.
TEST(BoltParquetWriteSizeStatistics, FlatUtf8DictionaryBytesCountsLogicalRows) {
    const int kRows = 300;
    const char* kDistinct[5] = {"a", "bb", "ccc", "dddd", "eeeee"};
    std::vector<std::string> vals;
    for (int i = 0; i < kRows; ++i) vals.push_back(kDistinct[i % 5]);

    bolt::Arena arena;
    bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(batch, nullptr);
    bolt::BoltBatch::init_empty(batch);
    batch->num_cols = 1;
    batch->num_rows = kRows;
    bolt::BoltBatch::alloc_columns(batch, &arena, 1);
    put_utf8_col(&arena, batch, 0, "s", vals);

    ParquetWriteOpts opts{};
    opts.n_columns = 1;
    opts.compression = 0;
    opts.emit_statistics = true;
    opts.emit_size_statistics = true;
    opts.use_dictionary = true;
    std::strncpy(opts.columns[0].name, "s", sizeof(opts.columns[0].name));
    opts.columns[0].type = bolt::BoltType::Utf8;
    opts.columns[0].nullable = false;

    const std::string path = "test_bolt_parquet_write_size_statistics_flat_dict.parquet";
    ParquetWriter* w = parquet_write_open(path.c_str(), &opts);
    ASSERT_NE(w, nullptr);
    ASSERT_TRUE(parquet_write_row_group(w, batch));
    ASSERT_TRUE(parquet_write_close(w));

    std::int64_t want = 0;
    for (int i = 0; i < kRows; ++i) want += static_cast<std::int64_t>(vals[static_cast<std::size_t>(i)].size());
    bolt::Arena ra;
    PqChunk ch{};
    ASSERT_TRUE(read_back_chunk0(path.c_str(), &ra, &ch));
    EXPECT_EQ(ch.byte_array_data_bytes, want)
        << "must count all 300 logical rows, not the 5 distinct dict entries";
}

// Not BYTE_ARRAY, not a LIST leaf: size_statistics must be entirely absent --
// the spec-legal "no loss of information" omission for a flat column, never
// an all-zero histogram.
TEST(BoltParquetWriteSizeStatistics, FlatInt64NoSizeStatisticsField) {
    const int kRows = 40;
    bolt::Arena arena;
    bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(batch, nullptr);
    bolt::BoltBatch::init_empty(batch);
    batch->num_cols = 1;
    batch->num_rows = kRows;
    bolt::BoltBatch::alloc_columns(batch, &arena, 1);

    batch->schema.add_field("n", bolt::BoltType::Int64, true);
    bolt::BoltColumn& c = batch->columns[batch->read_epoch][0];
    c = bolt::BoltColumn::make_flat_alloc(kRows, bolt::BoltType::Int64, &arena);
    ASSERT_NE(c.data, nullptr);
    auto* p = static_cast<std::int64_t*>(c.data);
    const std::size_t vb = static_cast<std::size_t>((kRows + 7) / 8);
    auto* validity = static_cast<std::uint8_t*>(arena.allocate(vb, 8));
    std::memset(validity, 0xFF, vb);
    for (int i = 0; i < kRows; ++i) {
        p[i] = i;
        if (i % 7 == 0) {
            validity[i >> 3] = static_cast<std::uint8_t>(
                validity[i >> 3] & ~(1u << (i & 7)));   // null every 7th row
        }
    }
    c.validity = validity;
    c.stats.all_valid = false;

    ParquetWriteOpts opts{};
    opts.n_columns = 1;
    opts.compression = 0;
    opts.emit_statistics = true;
    opts.emit_size_statistics = true;
    std::strncpy(opts.columns[0].name, "n", sizeof(opts.columns[0].name));
    opts.columns[0].type = bolt::BoltType::Int64;
    opts.columns[0].nullable = true;

    const std::string path = "test_bolt_parquet_write_size_statistics_flat_int64_absent.parquet";
    ParquetWriter* w = parquet_write_open(path.c_str(), &opts);
    ASSERT_NE(w, nullptr);
    ASSERT_TRUE(parquet_write_row_group(w, batch));
    ASSERT_TRUE(parquet_write_close(w));

    bolt::Arena ra;
    PqChunk ch{};
    ASSERT_TRUE(read_back_chunk0(path.c_str(), &ra, &ch));
    EXPECT_EQ(ch.byte_array_data_bytes, -1) << "Int64 is never BYTE_ARRAY";
    EXPECT_EQ(ch.def_hist_len, 0u);
    EXPECT_EQ(ch.rep_hist_len, 0u);
}

// emit_size_statistics left off (default): field 16 must be entirely absent
// even for a BYTE_ARRAY column that WOULD otherwise qualify.
TEST(BoltParquetWriteSizeStatistics, DisabledOptionOmitsFieldEntirely) {
    const int kRows = 20;
    std::vector<std::string> vals;
    for (int i = 0; i < kRows; ++i) vals.push_back(flat_plain_val(i));

    bolt::Arena arena;
    bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(batch, nullptr);
    bolt::BoltBatch::init_empty(batch);
    batch->num_cols = 1;
    batch->num_rows = kRows;
    bolt::BoltBatch::alloc_columns(batch, &arena, 1);
    put_utf8_col(&arena, batch, 0, "s", vals);

    ParquetWriteOpts opts{};
    opts.n_columns = 1;
    opts.compression = 0;
    opts.emit_statistics = true;
    opts.emit_size_statistics = false;   // explicit: the option under test
    opts.use_dictionary = false;
    std::strncpy(opts.columns[0].name, "s", sizeof(opts.columns[0].name));
    opts.columns[0].type = bolt::BoltType::Utf8;
    opts.columns[0].nullable = false;

    const std::string path = "test_bolt_parquet_write_size_statistics_disabled.parquet";
    ParquetWriter* w = parquet_write_open(path.c_str(), &opts);
    ASSERT_NE(w, nullptr);
    ASSERT_TRUE(parquet_write_row_group(w, batch));
    ASSERT_TRUE(parquet_write_close(w));

    bolt::Arena ra;
    PqChunk ch{};
    ASSERT_TRUE(read_back_chunk0(path.c_str(), &ra, &ch));
    EXPECT_EQ(ch.byte_array_data_bytes, -1) << "emit_size_statistics=false";
    EXPECT_EQ(ch.def_hist_len, 0u);
    EXPECT_EQ(ch.rep_hist_len, 0u);
}

// ---- LIST-leaf fixtures ----------------------------------------------------
//
// Six fixed rows, restated exactly in the check script so the expected
// histogram/byte-sum can be hand-verified rather than re-derived from bolt's
// own code:
//   row0  NULL list                        -> 1 slot  rep0 def0
//   row1  EMPTY list                       -> 1 slot  rep0 def1
//   row2  [present]                        -> 1 slot  rep0 def3
//   row3  [NULL element]                   -> 1 slot  rep0 def2
//   row4  [present, NULL, present]         -> 3 slots rep0 def3, rep1 def2, rep1 def3
//   row5  EMPTY list                       -> 1 slot  rep0 def1
// def_hist = [1,2,2,3], rep_hist = [6,2], 8 slots total, max_def=3, max_rep=1.
// element_nullable=true throughout, so this exercises the def=2 (null
// element) vs def=3 (present element) split as well as the list-vs-empty
// split (def=0 vs def=1).

bolt::BoltColumn build_int64_list_fixture(bolt::Arena* a, std::int32_t* offs_out) {
    // Elements in flattened order: row2[0]=100, row3[0]=NULL(999 placeholder),
    // row4[0]=200 row4[1]=NULL(999) row4[2]=201.
    const std::int64_t elems[5] = {100, 999, 200, 999, 201};
    const std::uint8_t evalid[5] = {1, 0, 1, 0, 1};
    bolt::BoltColumn elem = bolt::BoltColumn::make_flat_alloc(5, bolt::BoltType::Int64, a);
    auto* ep = static_cast<std::int64_t*>(elem.data);
    for (int i = 0; i < 5; ++i) ep[i] = elems[i];
    auto* ebm = static_cast<std::uint8_t*>(a->allocate(1, 8));
    *ebm = 0;
    for (int i = 0; i < 5; ++i) {
        if (evalid[i]) *ebm = static_cast<std::uint8_t>(*ebm | (1u << i));
    }
    elem.validity = ebm;
    elem.stats.all_valid = false;

    // offsets: row0 NULL(0,0) row1 EMPTY(0,0) row2(0,1) row3(1,2) row4(2,5) row5 EMPTY(5,5)
    const std::int32_t offs[7] = {0, 0, 0, 1, 2, 5, 5};
    std::memcpy(offs_out, offs, sizeof(offs));
    auto* lval = static_cast<std::uint8_t*>(a->allocate(1, 8));
    *lval = static_cast<std::uint8_t>(0xFFu & ~(1u << 0));   // row0 NULL, rest present
    return bolt::BoltColumn::make_list(&elem, offs_out, 6, lval, a);
}

TEST(BoltParquetWriteSizeStatistics, ListInt64HistogramOnly) {
    bolt::Arena arena;
    std::int32_t offs[7];
    bolt::BoltColumn li = build_int64_list_fixture(&arena, offs);

    bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(batch, nullptr);
    bolt::BoltBatch::init_empty(batch);
    batch->num_cols = 1;
    batch->num_rows = 6;
    bolt::BoltBatch::alloc_columns(batch, &arena, 1);
    batch->schema.add_field("li", bolt::BoltType::List, true);
    batch->columns[batch->read_epoch][0] = li;

    ParquetWriteOpts opts{};
    opts.n_columns = 1;
    opts.compression = 0;
    opts.emit_statistics = true;
    opts.emit_size_statistics = true;
    std::strncpy(opts.columns[0].name, "li", sizeof(opts.columns[0].name));
    opts.columns[0].type = bolt::BoltType::List;
    opts.columns[0].nullable = true;
    opts.columns[0].element_type = bolt::BoltType::Int64;
    opts.columns[0].element_nullable = true;

    const std::string path = "test_bolt_parquet_write_size_statistics_list_int64.parquet";
    ParquetWriter* w = parquet_write_open(path.c_str(), &opts);
    ASSERT_NE(w, nullptr);
    ASSERT_TRUE(parquet_write_row_group(w, batch));
    ASSERT_TRUE(parquet_write_close(w));

    bolt::Arena ra;
    PqChunk ch{};
    ASSERT_TRUE(read_back_chunk0(path.c_str(), &ra, &ch));
    EXPECT_EQ(ch.byte_array_data_bytes, -1) << "Int64 elements are never BYTE_ARRAY";
    ASSERT_EQ(ch.def_hist_len, 4u);
    ASSERT_EQ(ch.rep_hist_len, 2u);
    const std::int64_t want_def[4] = {1, 2, 2, 3};
    const std::int64_t want_rep[2] = {6, 2};
    for (int i = 0; i < 4; ++i) EXPECT_EQ(ch.def_hist[i], want_def[i]) << "bucket " << i;
    for (int i = 0; i < 2; ++i) EXPECT_EQ(ch.rep_hist[i], want_rep[i]) << "bucket " << i;
}

bolt::BoltColumn build_utf8_list_fixture(bolt::Arena* a, std::int32_t* offs_out) {
    // Same shape as build_int64_list_fixture; present-element strings are
    // "hello" (row2), "ab" and "cde" (row4) -> byte_array_data_bytes = 10.
    const char* strs[5] = {"hello", "NULLPLACEHOLDER", "ab", "NULLPLACEHOLDER", "cde"};
    const std::uint8_t evalid[5] = {1, 0, 1, 0, 1};
    bolt::BoltColumn elem = bolt::BoltColumn::make_empty();
    elem.length = 5;
    elem.format = bolt::ColumnFormat::Flat;
    elem.type = bolt::BoltType::Utf8;
    elem.type_size_bytes = sizeof(bolt::StringView);
    auto* svs = static_cast<bolt::StringView*>(
        a->allocate(5 * sizeof(bolt::StringView), alignof(bolt::StringView)));
    std::memset(svs, 0, 5 * sizeof(bolt::StringView));
    for (int i = 0; i < 5; ++i) svs[i] = bolt::StringView::from_cstr(strs[i]);
    elem.data = svs;
    auto* ebm = static_cast<std::uint8_t*>(a->allocate(1, 8));
    *ebm = 0;
    for (int i = 0; i < 5; ++i) {
        if (evalid[i]) *ebm = static_cast<std::uint8_t>(*ebm | (1u << i));
    }
    elem.validity = ebm;
    elem.stats.all_valid = false;

    const std::int32_t offs[7] = {0, 0, 0, 1, 2, 5, 5};
    std::memcpy(offs_out, offs, sizeof(offs));
    auto* lval = static_cast<std::uint8_t*>(a->allocate(1, 8));
    *lval = static_cast<std::uint8_t>(0xFFu & ~(1u << 0));
    return bolt::BoltColumn::make_list(&elem, offs_out, 6, lval, a);
}

TEST(BoltParquetWriteSizeStatistics, ListUtf8HistogramAndBytes) {
    bolt::Arena arena;
    std::int32_t offs[7];
    bolt::BoltColumn li = build_utf8_list_fixture(&arena, offs);

    bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(batch, nullptr);
    bolt::BoltBatch::init_empty(batch);
    batch->num_cols = 1;
    batch->num_rows = 6;
    bolt::BoltBatch::alloc_columns(batch, &arena, 1);
    batch->schema.add_field("li", bolt::BoltType::List, true);
    batch->columns[batch->read_epoch][0] = li;

    ParquetWriteOpts opts{};
    opts.n_columns = 1;
    opts.compression = 0;
    opts.emit_statistics = true;
    opts.emit_size_statistics = true;
    std::strncpy(opts.columns[0].name, "li", sizeof(opts.columns[0].name));
    opts.columns[0].type = bolt::BoltType::List;
    opts.columns[0].nullable = true;
    opts.columns[0].element_type = bolt::BoltType::Utf8;
    opts.columns[0].element_nullable = true;

    const std::string path = "test_bolt_parquet_write_size_statistics_list_utf8.parquet";
    ParquetWriter* w = parquet_write_open(path.c_str(), &opts);
    ASSERT_NE(w, nullptr);
    ASSERT_TRUE(parquet_write_row_group(w, batch));
    ASSERT_TRUE(parquet_write_close(w));

    bolt::Arena ra;
    PqChunk ch{};
    ASSERT_TRUE(read_back_chunk0(path.c_str(), &ra, &ch));
    EXPECT_EQ(ch.byte_array_data_bytes, 10) << "sum of present-element string bytes only";
    ASSERT_EQ(ch.def_hist_len, 4u);
    ASSERT_EQ(ch.rep_hist_len, 2u);
    const std::int64_t want_def[4] = {1, 2, 2, 3};
    const std::int64_t want_rep[2] = {6, 2};
    for (int i = 0; i < 4; ++i) EXPECT_EQ(ch.def_hist[i], want_def[i]) << "bucket " << i;
    for (int i = 0; i < 2; ++i) EXPECT_EQ(ch.rep_hist[i], want_rep[i]) << "bucket " << i;
}

}  // namespace
