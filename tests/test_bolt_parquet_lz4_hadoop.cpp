// G2PQ-18 — legacy Hadoop-framed LZ4 (parquet CompressionCodec.LZ4 = 5,
// DEPRECATED in favor of LZ4_RAW = 7 but still legal to read).
//
// Fixtures hand-constructed by scripts/make_lz4_hadoop_fixture.py: no writer
// on this box emits codec 5 (verified there against the real on-disk thrift
// bytes of a pyarrow-written file: compression="LZ4" actually writes
// LZ4_RAW=7). The fixtures are independently proven correct by a SEPARATE
// oracle: pyarrow's own LZ4_HADOOP decoder (arrow C++ maps thrift codec 5 to
// Compression::LZ4_HADOOP and can decode it, even though its Python string
// API has no way to WRITE it) reads both files back to the exact expected
// int64 sequences -- see the script's own verify path and this session's
// working notes. bolt's decoder is therefore checked against a file two
// independent codebases agree on, not merely against its own compressor.
//
//   lz4_hadoop_single.parquet   1 page, 1 Hadoop sub-block, values 10..50
//   lz4_hadoop_multi.parquet    1 page, 2 Hadoop sub-blocks, values 0..199

#include "bolt/ingest/bolt_parquet_read.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"

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
    return std::string(BOLT_TEST_DATA_DIR) + "/" + name;
}

}  // namespace

TEST(BoltParquetLz4Hadoop, SingleSubBlock) {
    const auto buf = slurp(data_path("lz4_hadoop_single.parquet").c_str());
    ASSERT_FALSE(buf.empty());
    bolt::Arena arena;
    bolt::BoltBatch* b = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(b, nullptr);
    ASSERT_TRUE(parquet_read_file(buf.data(), buf.size(), &arena, b));
    ASSERT_EQ(b->num_rows, 5);
    ASSERT_EQ(b->num_cols, 1u);
    ASSERT_EQ(b->columns[b->read_epoch][0].type, bolt::BoltType::Int64);
    const auto* v =
        static_cast<const int64_t*>(b->columns[b->read_epoch][0].data);
    const int64_t want[5] = {10, 20, 30, 40, 50};
    for (int i = 0; i < 5; ++i) EXPECT_EQ(v[i], want[i]) << "row " << i;
}

TEST(BoltParquetLz4Hadoop, TwoSubBlocks) {
    const auto buf = slurp(data_path("lz4_hadoop_multi.parquet").c_str());
    ASSERT_FALSE(buf.empty());
    bolt::Arena arena;
    bolt::BoltBatch* b = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(b, nullptr);
    ASSERT_TRUE(parquet_read_file(buf.data(), buf.size(), &arena, b));
    ASSERT_EQ(b->num_rows, 200);
    ASSERT_EQ(b->num_cols, 1u);
    ASSERT_EQ(b->columns[b->read_epoch][0].type, bolt::BoltType::Int64);
    const auto* v =
        static_cast<const int64_t*>(b->columns[b->read_epoch][0].data);
    for (int i = 0; i < 200; ++i) EXPECT_EQ(v[i], i) << "row " << i;
}

// Corrupt one byte of the FIRST Hadoop frame's declared compressed-size
// field (byte offset 21+4 in lz4_hadoop_single.parquet -- the 4 BE bytes
// right after the 4-byte decompressed-size field) so it overstates the
// sub-block's compressed length past the actual page payload. Must be
// refused, never read past the buffer or hand back garbage rows.
TEST(BoltParquetLz4Hadoop, TruncatedFrameIsRefused) {
    auto buf = slurp(data_path("lz4_hadoop_single.parquet").c_str());
    ASSERT_FALSE(buf.empty());
    // PageHeader for this fixture is 17 bytes (verified while building the
    // fixture), starting right after the 4-byte "PAR1" magic -> the Hadoop
    // frame's 8-byte prefix starts at offset 4 + 17 = 21. Byte 21..24 is the
    // BE declared_unc (0x00000028 = 40); byte 25..28 is BE declared_comp
    // (0x0000001b = 27). Corrupt the low byte of declared_comp to blow past
    // the remaining page bytes.
    ASSERT_GT(buf.size(), 29u);
    buf[28] = 0xFF;  // declared_comp now huge, page only has ~10 bytes left
    bolt::Arena arena;
    bolt::BoltBatch* b = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(b, nullptr);
    EXPECT_FALSE(parquet_read_file(buf.data(), buf.size(), &arena, b));
}
