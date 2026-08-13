// test_bolt_parquet_write_mem.cpp — the parquet MEMORY sink.
//
// The claim being tested is an EQUALITY, not a "it also works": bytes written
// through the memory sink are byte-for-byte what the file sink writes. That is
// the whole point — a synthesized file must be indistinguishable from a stored
// one, and a caller that caches a size or a checksum from one path must be able
// to trust it for the other.
//
// A "does it parse" test would not catch the failure that matters here: a sink
// that silently reorders or drops a region still parses, because parquet is
// read through its footer offsets. So the primary assertion is memcmp.

#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt/ingest/bolt_parquet_write.h"

using bolt::ingest::parquet::ParquetWriteOpts;
using bolt::ingest::parquet::ParquetWriter;
using bolt::ingest::parquet::parquet_write_close;
using bolt::ingest::parquet::parquet_write_close_mem;
using bolt::ingest::parquet::parquet_write_open;
using bolt::ingest::parquet::parquet_write_open_mem;
using bolt::ingest::parquet::parquet_write_row_group;

namespace {

std::string tmp_path(const char* tag) {
    std::string s = "bolt_pqmem_";
    s += tag;
    s += ".parquet";
    return s;
}

std::vector<std::uint8_t> slurp(const char* p) {
    std::vector<std::uint8_t> out;
    FILE* f = std::fopen(p, "rb");
    if (f == nullptr) return out;
    std::uint8_t buf[8192];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) out.insert(out.end(), buf, buf + n);
    std::fclose(f);
    return out;
}

void build_batch(bolt::Arena* arena, std::int64_t n, bolt::BoltBatch* out) {
    ASSERT_NE(arena, nullptr);
    ASSERT_NE(out, nullptr);
    bolt::BoltBatch::init_empty(out);
    out->num_cols = 2;
    out->num_rows = n;
    bolt::BoltBatch::alloc_columns(out, arena, 2);
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

ParquetWriteOpts make_opts() {
    ParquetWriteOpts opts{};
    opts.n_columns = 2;
    opts.row_group_target_bytes = 1u << 20;
    opts.compression = 0;
    opts.emit_statistics = true;
    std::strncpy(opts.columns[0].name, "id", sizeof(opts.columns[0].name) - 1);
    opts.columns[0].type = bolt::BoltType::Int64;
    opts.columns[0].nullable = false;
    std::strncpy(opts.columns[1].name, "name", sizeof(opts.columns[1].name) - 1);
    opts.columns[1].type = bolt::BoltType::Utf8;
    opts.columns[1].nullable = false;
    return opts;
}

}  // namespace

// The equality that justifies the whole sink.
TEST(ParquetWriteMem, ByteIdenticalToFileSink) {
    for (std::int64_t rows : {std::int64_t{1}, std::int64_t{100},
                              std::int64_t{5000}}) {
        bolt::Arena arena;
        bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
        ASSERT_NE(batch, nullptr);
        build_batch(&arena, rows, batch);
        ParquetWriteOpts opts = make_opts();

        const std::string path = tmp_path("file");
        ParquetWriter* fw = parquet_write_open(path.c_str(), &opts);
        ASSERT_NE(fw, nullptr);
        ASSERT_TRUE(parquet_write_row_group(fw, batch));
        ASSERT_TRUE(parquet_write_close(fw));
        const std::vector<std::uint8_t> on_disk = slurp(path.c_str());
        std::remove(path.c_str());
        ASSERT_FALSE(on_disk.empty());

        ParquetWriter* mw = parquet_write_open_mem(&opts);
        ASSERT_NE(mw, nullptr);
        ASSERT_TRUE(parquet_write_row_group(mw, batch));
        const std::uint8_t* mem = nullptr;
        std::uint64_t mem_len = 0;
        ASSERT_TRUE(parquet_write_close_mem(mw, &arena, &mem, &mem_len));
        ASSERT_NE(mem, nullptr);

        EXPECT_EQ(mem_len, on_disk.size()) << "rows=" << rows;
        ASSERT_EQ(mem_len, on_disk.size());
        EXPECT_EQ(std::memcmp(mem, on_disk.data(), mem_len), 0)
            << "memory and file sinks diverged at rows=" << rows;
    }
}

// Multiple row groups exercise the offset bookkeeping the footer depends on —
// the place a sink swap would most plausibly drift.
TEST(ParquetWriteMem, MultiRowGroupByteIdentical) {
    bolt::Arena arena;
    bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(batch, nullptr);
    build_batch(&arena, 250, batch);
    ParquetWriteOpts opts = make_opts();

    const std::string path = tmp_path("multi");
    ParquetWriter* fw = parquet_write_open(path.c_str(), &opts);
    ASSERT_NE(fw, nullptr);
    for (int i = 0; i < 3; ++i) ASSERT_TRUE(parquet_write_row_group(fw, batch));
    ASSERT_TRUE(parquet_write_close(fw));
    const std::vector<std::uint8_t> on_disk = slurp(path.c_str());
    std::remove(path.c_str());
    ASSERT_FALSE(on_disk.empty());

    ParquetWriter* mw = parquet_write_open_mem(&opts);
    ASSERT_NE(mw, nullptr);
    for (int i = 0; i < 3; ++i) ASSERT_TRUE(parquet_write_row_group(mw, batch));
    const std::uint8_t* mem = nullptr;
    std::uint64_t mem_len = 0;
    ASSERT_TRUE(parquet_write_close_mem(mw, &arena, &mem, &mem_len));
    ASSERT_EQ(mem_len, on_disk.size());
    EXPECT_EQ(std::memcmp(mem, on_disk.data(), mem_len), 0);
}

// The bytes are real parquet, read back through bolt's own reader with VALUES
// asserted — byte-equality alone would be satisfied by two identically broken
// files.
TEST(ParquetWriteMem, MemoryBytesReadBackWithCorrectValues) {
    const std::int64_t kRows = 64;
    bolt::Arena arena;
    bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(batch, nullptr);
    build_batch(&arena, kRows, batch);
    ParquetWriteOpts opts = make_opts();

    ParquetWriter* mw = parquet_write_open_mem(&opts);
    ASSERT_NE(mw, nullptr);
    ASSERT_TRUE(parquet_write_row_group(mw, batch));
    const std::uint8_t* mem = nullptr;
    std::uint64_t mem_len = 0;
    ASSERT_TRUE(parquet_write_close_mem(mw, &arena, &mem, &mem_len));

    EXPECT_EQ(std::memcmp(mem, "PAR1", 4), 0);
    EXPECT_EQ(std::memcmp(mem + mem_len - 4, "PAR1", 4), 0);

    bolt::ingest::parquet::PqMeta meta{};
    ASSERT_TRUE(bolt::ingest::parquet::parquet_read_meta(mem, mem_len, &arena,
                                                         &meta));
    EXPECT_EQ(meta.num_rows, kRows);
    EXPECT_EQ(meta.n_columns, 2u);

    bolt::BoltColumn* cols = arena.allocate_array<bolt::BoltColumn>(2);
    ASSERT_NE(cols, nullptr);
    std::int64_t got_rows = 0;
    ASSERT_TRUE(bolt::ingest::parquet::parquet_read_row_group(
        mem, mem_len, &meta, 0, &arena, cols, &got_rows));
    ASSERT_EQ(got_rows, kRows);
    const auto* ip = static_cast<const std::int64_t*>(cols[0].data);
    ASSERT_NE(ip, nullptr);
    EXPECT_EQ(ip[0], 7);
    EXPECT_EQ(ip[63], 63 * 1000 + 7);
}

// A path-opened writer has already streamed to disk; handing back an empty
// buffer would be indistinguishable from a valid zero-row file.
TEST(ParquetWriteMem, CloseMemRejectsFileWriter) {
    bolt::Arena arena;
    ParquetWriteOpts opts = make_opts();
    const std::string path = tmp_path("reject");
    ParquetWriter* fw = parquet_write_open(path.c_str(), &opts);
    ASSERT_NE(fw, nullptr);
    const std::uint8_t* mem = reinterpret_cast<const std::uint8_t*>(0x1);
    std::uint64_t mem_len = 12345;
    EXPECT_FALSE(parquet_write_close_mem(fw, &arena, &mem, &mem_len));
    // Out-params must be untouched on failure.
    EXPECT_EQ(mem, reinterpret_cast<const std::uint8_t*>(0x1));
    EXPECT_EQ(mem_len, 12345u);
    std::remove(path.c_str());
}
