// W-PQ-W — round-trip test: writer -> reader.
//
// Cases:
//   1) Int64 + Utf8, UNCOMPRESSED, no nulls.
//   2) Same shape, SNAPPY compression.
//   3) Nullable Int64 with mixed null / non-null values.
//   4) G2FEAT-21: Utf8 chunk statistics round-trip + stats_flags provenance
//      + the bolt_parquet_stats.h pruning helpers (int range decode, utf8
//      eq-exclusion, legacy-stat conservatism, no-nulls gate).
//   5) G2FEAT-21: a >64-byte string in a chunk makes the writer OMIT stats
//      (never a truncated — i.e. invalid — bound), and the reader treats
//      absent stats as unprunable.

#include "bolt/ingest/bolt_parquet_write.h"
#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt/ingest/bolt_parquet_stats.h"

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

// ---- G2FEAT-21: chunk-statistics round-trip + pruning helpers ------------

// Build a 2-col (id int64, tag utf8) batch whose tag values come from a
// caller list (cycled). Strings >12 bytes spill into a shared buffer.
void build_stats_batch(bolt::Arena* arena, std::int64_t base, std::int64_t n,
                       const char* const* tags, std::size_t n_tags,
                       bolt::BoltBatch* out) {
    ASSERT_NE(arena, nullptr);
    ASSERT_NE(out, nullptr);
    bolt::BoltBatch::init_empty(out);
    out->num_cols = 2;
    out->num_rows = n;

    bolt::BoltColumn& id = out->columns[out->read_epoch][0];
    id = bolt::BoltColumn::make_flat_alloc(n, bolt::BoltType::Int64, arena);
    ASSERT_NE(id.data, nullptr);
    auto* ip = static_cast<std::int64_t*>(id.data);
    for (std::int64_t i = 0; i < n; ++i) ip[i] = base + i;

    bolt::BoltColumn& tg = out->columns[out->read_epoch][1];
    tg = bolt::BoltColumn::make_empty();
    tg.length = n;
    tg.format = bolt::ColumnFormat::Flat;
    tg.type = bolt::BoltType::Utf8;
    tg.type_size_bytes = sizeof(bolt::StringView);
    auto* svs = static_cast<bolt::StringView*>(
        arena->allocate(static_cast<size_t>(n) * sizeof(bolt::StringView),
                        alignof(bolt::StringView)));
    ASSERT_NE(svs, nullptr);
    std::memset(svs, 0, static_cast<size_t>(n) * sizeof(bolt::StringView));
    char* spill = static_cast<char*>(
        arena->allocate(static_cast<size_t>(n) * 128u, 8));
    ASSERT_NE(spill, nullptr);
    std::size_t spill_used = 0;
    for (std::int64_t i = 0; i < n; ++i) {
        const char* s = tags[static_cast<std::size_t>(i) % n_tags];
        bolt::StringView v = bolt::StringView::from_cstr(s);
        if (!v.is_inline()) {
            const std::size_t len = std::strlen(s);
            std::memcpy(spill + spill_used, s, len);
            v.ref.buf_idx = 0;
            v.ref.offset = static_cast<std::uint32_t>(spill_used);
            spill_used += len;
        }
        svs[i] = v;
    }
    tg.data = svs;
    tg.str_overflow_base = spill;
    tg.stats.all_valid = true;
}

// Write two rowgroups with disjoint id/tag ranges and parse the footer.
void write_and_parse_stats_file(const std::string& path,
                                std::vector<std::uint8_t>* file,
                                PqMeta* meta, std::vector<PqChunk>* chunks,
                                const char* const* tags0, std::size_t n0,
                                const char* const* tags1, std::size_t n1) {
    ParquetWriteOpts opts{};
    opts.n_columns = 2;
    opts.row_group_target_bytes = 0;   // 1 write call = 1 rowgroup
    opts.compression = 0;
    opts.emit_statistics = true;
    std::strncpy(opts.columns[0].name, "id", sizeof(opts.columns[0].name));
    opts.columns[0].type = bolt::BoltType::Int64;
    opts.columns[0].nullable = false;
    std::strncpy(opts.columns[1].name, "tag", sizeof(opts.columns[1].name));
    opts.columns[1].type = bolt::BoltType::Utf8;
    opts.columns[1].nullable = false;

    ParquetWriter* w = parquet_write_open(path.c_str(), &opts);
    ASSERT_NE(w, nullptr);
    {
        bolt::Arena arena;
        bolt::BoltBatch* b = arena.allocate_array<bolt::BoltBatch>(1);
        ASSERT_NE(b, nullptr);
        build_stats_batch(&arena, 0, 100, tags0, n0, b);
        ASSERT_TRUE(parquet_write_row_group(w, b));
        build_stats_batch(&arena, 100, 100, tags1, n1, b);
        ASSERT_TRUE(parquet_write_row_group(w, b));
    }
    ASSERT_TRUE(parquet_write_close(w));

    *file = slurp_file(path.c_str());
    ASSERT_FALSE(file->empty());
    std::uint64_t moff = 0;
    std::uint32_t mlen = 0;
    ASSERT_TRUE(pq_locate_footer(file->data(), file->size(), &moff, &mlen));
    std::memset(meta, 0, sizeof(*meta));
    chunks->resize(16);
    meta->chunks = chunks->data();
    meta->chunks_cap = 16;
    ASSERT_TRUE(pq_parse_file_meta(file->data() + moff, mlen, meta));
    ASSERT_EQ(meta->n_row_groups, 2u);
    ASSERT_EQ(meta->n_columns, 2u);
}

TEST(BoltParquetStats, Utf8AndIntStatsRoundtripWithHelpers) {
    const char* tags0[] = {"apple", "banana", "avocado"};
    const char* tags1[] = {"mike", "november_delta_xx", "oscar"};  // one spilled
    const std::string path = tmp_path("stats");
    std::vector<std::uint8_t> file;
    PqMeta meta;
    std::vector<PqChunk> chunks;
    write_and_parse_stats_file(path, &file, &meta, &chunks,
                               tags0, 3, tags1, 3);

    // Int64 chunk stats decode: rg0 ids [0,99], rg1 ids [100,199].
    const PqColumn& idc = meta.columns[0];
    const PqChunk& id_rg0 = meta.chunks[meta.row_groups[0].chunk_off + 0];
    const PqChunk& id_rg1 = meta.chunks[meta.row_groups[1].chunk_off + 0];
    std::int64_t lo = 0, hi = 0;
    ASSERT_TRUE(pq_stat_range_i64(idc, id_rg0, &lo, &hi));
    EXPECT_EQ(lo, 0);
    EXPECT_EQ(hi, 99);
    ASSERT_TRUE(pq_stat_range_i64(idc, id_rg1, &lo, &hi));
    EXPECT_EQ(lo, 100);
    EXPECT_EQ(hi, 199);

    // Required column, writer left null_count absent -> still provably
    // no-null via the REQUIRED repetition.
    EXPECT_EQ(id_rg0.null_count, -1);
    EXPECT_TRUE(pq_chunk_no_nulls(idc, id_rg0));

    // Utf8 chunk stats: provenance flags set (writer emits the modern
    // fields), min/max are the lexicographic extremes incl. the spilled
    // 17-byte value.
    const PqColumn& tgc = meta.columns[1];
    const PqChunk& tg_rg0 = meta.chunks[meta.row_groups[0].chunk_off + 1];
    const PqChunk& tg_rg1 = meta.chunks[meta.row_groups[1].chunk_off + 1];
    EXPECT_NE(tg_rg0.stats_flags & kPqStatMinIsValueField, 0u);
    EXPECT_NE(tg_rg0.stats_flags & kPqStatMaxIsValueField, 0u);
    ASSERT_EQ(tg_rg0.min_len, 5u);                       // "apple"
    EXPECT_EQ(std::memcmp(tg_rg0.min_bytes, "apple", 5), 0);
    ASSERT_EQ(tg_rg0.max_len, 6u);                       // "banana"
    EXPECT_EQ(std::memcmp(tg_rg0.max_bytes, "banana", 6), 0);
    ASSERT_EQ(tg_rg1.min_len, 4u);                       // "mike"
    EXPECT_EQ(std::memcmp(tg_rg1.min_bytes, "mike", 4), 0);
    ASSERT_EQ(tg_rg1.max_len, 5u);                       // "oscar"
    EXPECT_EQ(std::memcmp(tg_rg1.max_bytes, "oscar", 5), 0);

    // Eq-exclusion: outside [min,max] on either side excludes; inside (even
    // a value not actually present) never excludes.
    const auto q = [](const char* s) {
        return reinterpret_cast<const std::uint8_t*>(s);
    };
    EXPECT_TRUE(pq_stat_utf8_excludes_eq(tg_rg0, q("aaa"), 3));
    EXPECT_TRUE(pq_stat_utf8_excludes_eq(tg_rg0, q("zzz"), 3));
    EXPECT_FALSE(pq_stat_utf8_excludes_eq(tg_rg0, q("apple"), 5));
    EXPECT_FALSE(pq_stat_utf8_excludes_eq(tg_rg0, q("azzz"), 4));  // inside
    EXPECT_TRUE(pq_stat_utf8_excludes_eq(tg_rg1, q("apple"), 5));  // < "mike"

    // Legacy-stat conservatism: identical bytes but provenance flags
    // cleared (as if parsed from the deprecated min/max fields) must never
    // exclude — old writers ordered BYTE_ARRAY stats inconsistently.
    PqChunk legacy = tg_rg0;
    legacy.stats_flags = 0;
    EXPECT_FALSE(pq_stat_utf8_excludes_eq(legacy, q("zzz"), 3));

    // No-nulls gate: an OPTIONAL column with a positive null_count is never
    // provably null-free.
    PqColumn optc = tgc;
    optc.optional = 1;
    PqChunk nullable = tg_rg0;
    nullable.null_count = 3;
    EXPECT_FALSE(pq_chunk_no_nulls(optc, nullable));
    nullable.null_count = 0;
    EXPECT_TRUE(pq_chunk_no_nulls(optc, nullable));

    // Reversed (corrupt) int stats are rejected.
    PqChunk rev = id_rg0;
    std::memcpy(rev.min_bytes, id_rg0.max_bytes, 8);
    std::memcpy(rev.max_bytes, id_rg0.min_bytes, 8);
    EXPECT_FALSE(pq_stat_range_i64(idc, rev, &lo, &hi));

    std::remove(path.c_str());
}

TEST(BoltParquetStats, OversizeUtf8ValueOmitsStats) {
    // One rowgroup value is 80 bytes: a truncated max would be an invalid
    // bound, so the writer must omit that chunk's stats entirely.
    std::string long_tag(80, 'm');
    const char* tags0[] = {"alpha", "bravo"};
    const char* tags1[] = {"kilo", long_tag.c_str(), "lima"};
    const std::string path = tmp_path("stats_long");
    std::vector<std::uint8_t> file;
    PqMeta meta;
    std::vector<PqChunk> chunks;
    write_and_parse_stats_file(path, &file, &meta, &chunks,
                               tags0, 2, tags1, 3);

    const PqChunk& tg_rg0 = meta.chunks[meta.row_groups[0].chunk_off + 1];
    const PqChunk& tg_rg1 = meta.chunks[meta.row_groups[1].chunk_off + 1];
    EXPECT_NE(tg_rg0.min_len, 0u);   // short-string rowgroup keeps stats
    EXPECT_EQ(tg_rg1.min_len, 0u);   // oversize rowgroup omits them
    EXPECT_EQ(tg_rg1.max_len, 0u);
    EXPECT_EQ(tg_rg1.stats_flags, 0u);
    // Absent stats can never exclude.
    EXPECT_FALSE(pq_stat_utf8_excludes_eq(
        tg_rg1, reinterpret_cast<const std::uint8_t*>("zzzz"), 4));

    // Value round-trip is unaffected: the 80-byte string reads back intact.
    bolt::Arena ra;
    bolt::BoltBatch* rb = ra.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(rb, nullptr);
    ASSERT_TRUE(parquet_read_file(file.data(), file.size(), &ra, rb));
    ASSERT_EQ(rb->num_rows, 200);
    const auto* sv = static_cast<const bolt::StringView*>(
        rb->columns[rb->read_epoch][1].data);
    // Row 101 in rg1 cycles to tags1[1] (the long string).
    ASSERT_EQ(sv[101].length, 80u);

    std::remove(path.c_str());
}

}  // namespace
