// W-PQ — Parquet footer metadata reader vs pyarrow-written golden files.
//
// golden_flat.parquet:        5 cols (int64, decimal(15,2), int32, utf8,
//                             double), 1000 rows, snappy, dictionary,
//                             row_group_size=256 -> 4 row groups.
// golden_flat_plain.parquet:  same data, UNCOMPRESSED, PLAIN (no dict).
// Plus corrupt-input fuzzing: truncations and byte flips over the footer
// region must return false, never crash (run under ASAN in CI).

#include "bolt/ingest/bolt_parquet_meta.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace bolt::ingest::parquet;

std::vector<uint8_t> slurp(const char* path) {
    std::vector<uint8_t> v;
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) return v;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    v.resize(static_cast<size_t>(n));
    const size_t got = std::fread(v.data(), 1, v.size(), f);
    std::fclose(f);
    if (got != v.size()) v.clear();
    return v;
}

std::string data_path(const char* name) {
    // Tests run from the build tree; the data dir sits next to the sources.
    return std::string(BOLT_TEST_DATA_DIR) + "/" + name;
}

void parse_ok(const std::vector<uint8_t>& buf, PqMeta* m,
              std::vector<PqChunk>* chunks) {
    chunks->resize(kPqMaxColumns * 64);
    std::memset(m, 0, sizeof(*m));
    m->chunks = chunks->data();
    m->chunks_cap = static_cast<uint32_t>(chunks->size());
    uint64_t off = 0;
    uint32_t len = 0;
    ASSERT_TRUE(pq_locate_footer(buf.data(), buf.size(), &off, &len));
    ASSERT_TRUE(pq_parse_file_meta(buf.data() + off, len, m));
}

void check_common(const PqMeta& m) {
    EXPECT_EQ(m.num_rows, 1000);
    ASSERT_EQ(m.n_columns, 5u);
    EXPECT_EQ(m.n_row_groups, 4u);   // row_group_size=256 over 1000 rows
    EXPECT_EQ(m.n_chunks, m.n_row_groups * m.n_columns);

    EXPECT_STREQ(m.columns[0].name, "id");
    EXPECT_EQ(m.columns[0].physical, PqType::Int64);
    EXPECT_STREQ(m.columns[1].name, "price");
    EXPECT_EQ(m.columns[1].physical, PqType::FixedLenByteArray);
    EXPECT_EQ(m.columns[1].scale, 2);
    EXPECT_EQ(m.columns[1].precision, 15);
    EXPECT_STREQ(m.columns[2].name, "qty");
    EXPECT_EQ(m.columns[2].physical, PqType::Int32);
    EXPECT_STREQ(m.columns[3].name, "name");
    EXPECT_EQ(m.columns[3].physical, PqType::ByteArray);
    EXPECT_STREQ(m.columns[4].name, "ratio");
    EXPECT_EQ(m.columns[4].physical, PqType::Double);

    int64_t rows = 0;
    for (uint32_t g = 0; g < m.n_row_groups; ++g) {
        rows += m.row_groups[g].num_rows;
        for (uint32_t c = 0; c < m.n_columns; ++c) {
            const PqChunk& ch = m.chunks[m.row_groups[g].chunk_off + c];
            EXPECT_EQ(ch.num_values, m.row_groups[g].num_rows);
            EXPECT_GT(ch.data_page_offset, 0);
            EXPECT_GT(ch.total_compressed_size, 0);
        }
    }
    EXPECT_EQ(rows, 1000);

    // id chunk stats: row group 0 covers ids [0, 255] — min/max present as
    // 8-byte little-endian int64 values.
    const PqChunk& id0 = m.chunks[m.row_groups[0].chunk_off + 0];
    ASSERT_EQ(id0.min_len, 8u);
    ASSERT_EQ(id0.max_len, 8u);
    int64_t mn = 0, mx = 0;
    std::memcpy(&mn, id0.min_bytes, 8);
    std::memcpy(&mx, id0.max_bytes, 8);
    EXPECT_EQ(mn, 0);
    EXPECT_EQ(mx, 255);
}

TEST(BoltParquetMeta, GoldenSnappyDict) {
    const auto buf = slurp(data_path("golden_flat.parquet").c_str());
    ASSERT_FALSE(buf.empty());
    PqMeta m{};
    std::vector<PqChunk> chunks;
    parse_ok(buf, &m, &chunks);
    check_common(m);
    EXPECT_EQ(m.chunks[0].codec, PqCodec::Snappy);
    EXPECT_GT(m.chunks[0].dictionary_page_offset, 0);
}

TEST(BoltParquetMeta, GoldenPlainUncompressed) {
    const auto buf = slurp(data_path("golden_flat_plain.parquet").c_str());
    ASSERT_FALSE(buf.empty());
    PqMeta m{};
    std::vector<PqChunk> chunks;
    parse_ok(buf, &m, &chunks);
    check_common(m);
    EXPECT_EQ(m.chunks[0].codec, PqCodec::Uncompressed);
    EXPECT_EQ(m.chunks[0].dictionary_page_offset, 0);
}

TEST(BoltParquetMeta, CorruptInputsNeverCrash) {
    const auto buf = slurp(data_path("golden_flat.parquet").c_str());
    ASSERT_FALSE(buf.empty());
    PqMeta m{};
    std::vector<PqChunk> chunks(kPqMaxColumns * 64);
    m.chunks = chunks.data();
    m.chunks_cap = static_cast<uint32_t>(chunks.size());

    uint64_t off = 0;
    uint32_t len = 0;
    // Truncations across the whole tail: locate or parse must fail or
    // succeed cleanly — never crash (ASAN guards reads).
    for (size_t cut = 0; cut <= 64; ++cut) {
        const size_t n = buf.size() - cut;
        if (pq_locate_footer(buf.data(), n, &off, &len)) {
            (void)pq_parse_file_meta(buf.data() + off, len, &m);
        }
    }
    // Byte flips across the footer slice.
    ASSERT_TRUE(pq_locate_footer(buf.data(), buf.size(), &off, &len));
    std::vector<uint8_t> mut(buf);
    const size_t step = (len > 512) ? len / 512 : 1;
    for (size_t i = 0; i < len; i += step) {
        mut[off + i] ^= 0xFF;
        if (pq_locate_footer(mut.data(), mut.size(), &off, &len)) {
            (void)pq_parse_file_meta(mut.data() + off, len, &m);
        }
        mut[off + i] ^= 0xFF;   // restore
    }
    SUCCEED();
}

}  // namespace
