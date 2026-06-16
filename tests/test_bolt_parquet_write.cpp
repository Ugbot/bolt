// W-PQ-W — round-trip test: writer -> reader.
//
// Three cases:
//   1) Int64 + Utf8, UNCOMPRESSED, no nulls.
//   2) Same shape, SNAPPY compression.
//   3) Nullable Int64 with mixed null / non-null values.
//
// Each case writes to a temp path, reads it back with parquet_read_file,
// and verifies every row's value (and validity for case 3) matches the
// data fed in.

#include "bolt/ingest/bolt_parquet_write.h"
#include "bolt/ingest/bolt_parquet_read.h"

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
    v.resize(static_cast<size_t>(n));
    const size_t got = std::fread(v.data(), 1, v.size(), f);
    std::fclose(f);
    if (got != v.size()) v.clear();
    return v;
}

// Build a temp path inside the test working dir. ctest runs from the
// build's test bin dir, which is always writable.
std::string tmp_path(const char* tag) {
    std::string s = "test_bolt_parquet_write_";
    s += tag;
    s += ".parquet";
    return s;
}

// Construct a BoltBatch carrying an Int64 column and a Utf8 column.
// `n` rows, deterministic values.
void build_int64_utf8_batch(bolt::Arena* arena, std::int64_t n,
                            bolt::BoltBatch* out) {
    ASSERT_NE(arena, nullptr);
    ASSERT_NE(out, nullptr);
    bolt::BoltBatch::init_empty(out);
    out->num_cols = 2;
    out->num_rows = n;
    out->schema.add_field("id", bolt::BoltType::Int64, false);
    out->schema.add_field("name", bolt::BoltType::Utf8, false);

    bolt::BoltColumn& id = out->columns[out->read_epoch][0];
    id = bolt::BoltColumn::make_flat_alloc(n, bolt::BoltType::Int64, arena);
    ASSERT_NE(id.data, nullptr);
    auto* ip = static_cast<std::int64_t*>(id.data);
    for (std::int64_t i = 0; i < n; ++i) ip[i] = i * 1000 + 7;

    bolt::BoltColumn& nm = out->columns[out->read_epoch][1];
    nm = bolt::BoltColumn::make_empty();
    nm.length = n;
    nm.format = bolt::ColumnFormat::Flat;
    nm.type = bolt::BoltType::Utf8;
    nm.type_size_bytes = sizeof(bolt::StringView);
    // Allocate StringView array.
    auto* svs = static_cast<bolt::StringView*>(
        arena->allocate(static_cast<size_t>(n) * sizeof(bolt::StringView),
                        alignof(bolt::StringView)));
    ASSERT_NE(svs, nullptr);
    std::memset(svs, 0, static_cast<size_t>(n) * sizeof(bolt::StringView));
    nm.data = svs;
    nm.stats.all_valid = true;
    for (std::int64_t i = 0; i < n; ++i) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "n%lld", static_cast<long long>(i));
        svs[i] = bolt::StringView::from_cstr(buf);
    }
}

TEST(BoltParquetWrite, Int64Utf8UncompressedRoundtrip) {
    const std::int64_t kRows = 100;
    bolt::Arena arena;
    bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(batch, nullptr);
    build_int64_utf8_batch(&arena, kRows, batch);

    ParquetWriteOpts opts{};
    opts.n_columns = 2;
    opts.row_group_target_bytes = 1u << 20;
    opts.compression = 0;          // uncompressed
    opts.emit_statistics = true;
    std::strncpy(opts.columns[0].name, "id", sizeof(opts.columns[0].name));
    opts.columns[0].type = bolt::BoltType::Int64;
    opts.columns[0].nullable = false;
    std::strncpy(opts.columns[1].name, "name", sizeof(opts.columns[1].name));
    opts.columns[1].type = bolt::BoltType::Utf8;
    opts.columns[1].nullable = false;

    const std::string path = tmp_path("plain");
    ParquetWriter* w = parquet_write_open(path.c_str(), &opts);
    ASSERT_NE(w, nullptr);
    ASSERT_TRUE(parquet_write_row_group(w, batch));
    ASSERT_TRUE(parquet_write_close(w));

    // Read it back.
    const auto buf = slurp_file(path.c_str());
    ASSERT_FALSE(buf.empty());
    bolt::Arena ra;
    bolt::BoltBatch* rb = ra.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(rb, nullptr);
    ASSERT_TRUE(parquet_read_file(buf.data(), buf.size(), &ra, rb));
    ASSERT_EQ(rb->num_rows, kRows);
    ASSERT_EQ(rb->num_cols, 2u);

    const bolt::BoltColumn* cols = rb->columns[rb->read_epoch];
    ASSERT_EQ(cols[0].type, bolt::BoltType::Int64);
    ASSERT_EQ(cols[1].type, bolt::BoltType::Utf8);
    const auto* ip = static_cast<const std::int64_t*>(cols[0].data);
    const auto* sv = static_cast<const bolt::StringView*>(cols[1].data);
    for (std::int64_t i = 0; i < kRows; ++i) {
        EXPECT_EQ(ip[i], i * 1000 + 7);
        char want[32];
        std::snprintf(want, sizeof(want), "n%lld", static_cast<long long>(i));
        const std::uint32_t wlen = static_cast<std::uint32_t>(std::strlen(want));
        ASSERT_EQ(sv[i].length, wlen) << "row " << i;
        // All names <= 12 bytes -> inline payload (prefix + inline_data).
        EXPECT_EQ(std::memcmp(&sv[i].prefix[0], want, wlen), 0) << "row " << i;
    }
    std::remove(path.c_str());
}

TEST(BoltParquetWrite, Int64Utf8SnappyRoundtrip) {
    const std::int64_t kRows = 50;
    bolt::Arena arena;
    bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(batch, nullptr);
    build_int64_utf8_batch(&arena, kRows, batch);

    ParquetWriteOpts opts{};
    opts.n_columns = 2;
    opts.row_group_target_bytes = 1u << 20;
    opts.compression = 1;          // SNAPPY
    opts.emit_statistics = false;
    std::strncpy(opts.columns[0].name, "id", sizeof(opts.columns[0].name));
    opts.columns[0].type = bolt::BoltType::Int64;
    opts.columns[0].nullable = false;
    std::strncpy(opts.columns[1].name, "name", sizeof(opts.columns[1].name));
    opts.columns[1].type = bolt::BoltType::Utf8;
    opts.columns[1].nullable = false;

    const std::string path = tmp_path("snappy");
    ParquetWriter* w = parquet_write_open(path.c_str(), &opts);
    ASSERT_NE(w, nullptr);
    ASSERT_TRUE(parquet_write_row_group(w, batch));
    ASSERT_TRUE(parquet_write_close(w));

    const auto buf = slurp_file(path.c_str());
    ASSERT_FALSE(buf.empty());
    bolt::Arena ra;
    bolt::BoltBatch* rb = ra.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(rb, nullptr);
    ASSERT_TRUE(parquet_read_file(buf.data(), buf.size(), &ra, rb));
    ASSERT_EQ(rb->num_rows, kRows);

    const auto* ip = static_cast<const std::int64_t*>(
        rb->columns[rb->read_epoch][0].data);
    const auto* sv = static_cast<const bolt::StringView*>(
        rb->columns[rb->read_epoch][1].data);
    for (std::int64_t i = 0; i < kRows; ++i) {
        EXPECT_EQ(ip[i], i * 1000 + 7);
        char want[32];
        std::snprintf(want, sizeof(want), "n%lld", static_cast<long long>(i));
        const std::uint32_t wlen = static_cast<std::uint32_t>(std::strlen(want));
        ASSERT_EQ(sv[i].length, wlen) << "row " << i;
        EXPECT_EQ(std::memcmp(&sv[i].prefix[0], want, wlen), 0) << "row " << i;
    }
    std::remove(path.c_str());
}

TEST(BoltParquetWrite, NullableInt64Roundtrip) {
    const std::int64_t kRows = 64;
    bolt::Arena arena;
    bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(batch, nullptr);
    bolt::BoltBatch::init_empty(batch);
    batch->num_cols = 1;
    batch->num_rows = kRows;
    batch->schema.add_field("v", bolt::BoltType::Int64, true);

    bolt::BoltColumn& v = batch->columns[batch->read_epoch][0];
    v = bolt::BoltColumn::make_flat_alloc(kRows, bolt::BoltType::Int64,
                                          &arena);
    ASSERT_NE(v.data, nullptr);
    auto* vp = static_cast<std::int64_t*>(v.data);
    const size_t bm_bytes = static_cast<size_t>((kRows + 7) / 8);
    v.validity = static_cast<std::uint8_t*>(arena.allocate(bm_bytes, 1));
    ASSERT_NE(v.validity, nullptr);
    std::memset(v.validity, 0, bm_bytes);
    v.validity_offset = 0;
    v.stats.all_valid = false;
    // Row i is NULL when i % 3 == 1; otherwise valid with value = i * 11.
    for (std::int64_t i = 0; i < kRows; ++i) {
        const bool valid = (i % 3) != 1;
        vp[i] = valid ? (i * 11) : 0;
        if (valid) {
            v.validity[i >> 3] = static_cast<std::uint8_t>(
                v.validity[i >> 3] | (1u << (i & 7)));
        }
    }

    ParquetWriteOpts opts{};
    opts.n_columns = 1;
    opts.row_group_target_bytes = 1u << 20;
    opts.compression = 0;
    opts.emit_statistics = true;
    std::strncpy(opts.columns[0].name, "v", sizeof(opts.columns[0].name));
    opts.columns[0].type = bolt::BoltType::Int64;
    opts.columns[0].nullable = true;

    const std::string path = tmp_path("nullable");
    ParquetWriter* w = parquet_write_open(path.c_str(), &opts);
    ASSERT_NE(w, nullptr);
    ASSERT_TRUE(parquet_write_row_group(w, batch));
    ASSERT_TRUE(parquet_write_close(w));

    const auto buf = slurp_file(path.c_str());
    ASSERT_FALSE(buf.empty());
    bolt::Arena ra;
    bolt::BoltBatch* rb = ra.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(rb, nullptr);
    ASSERT_TRUE(parquet_read_file(buf.data(), buf.size(), &ra, rb));
    ASSERT_EQ(rb->num_rows, kRows);
    ASSERT_EQ(rb->num_cols, 1u);

    const bolt::BoltColumn& rcol = rb->columns[rb->read_epoch][0];
    ASSERT_EQ(rcol.type, bolt::BoltType::Int64);
    ASSERT_NE(rcol.validity, nullptr);
    const auto* rp = static_cast<const std::int64_t*>(rcol.data);
    for (std::int64_t i = 0; i < kRows; ++i) {
        const bool exp_valid = (i % 3) != 1;
        const std::uint8_t got_valid_bit = static_cast<std::uint8_t>(
            (rcol.validity[i >> 3] >> (i & 7)) & 1u);
        EXPECT_EQ(got_valid_bit, exp_valid ? 1u : 0u) << "row " << i;
        if (exp_valid) {
            EXPECT_EQ(rp[i], i * 11) << "row " << i;
        }
    }
    std::remove(path.c_str());
}

}  // namespace
