// G2PQ-36: bolt::ingest::parquet::parquet_read_file on a genuinely wide
// schema (105+ columns), modeled on ClickBench's real hits.parquet (105
// columns) -- the file that first exposed this bug.
//
// Root cause (see include/bolt/bolt_arena.h's kArenaMaxBlocks comment):
// parquet_read_file decodes an ENTIRE file -- every column, every row group
// -- into ONE caller-supplied bolt::Arena that lives for the whole call.
// Each column's own output buffers (data, validity, Utf8 dictionary-code
// hints, Utf8 string-overflow) is a SEPARATE arena allocation, and the
// arena's backing-block table is a hard-capped fixed-size array
// (kArenaMaxBlocks). A file with enough columns exhausts that table --
// independent of row count or of how large `max_block_size` is configured,
// because table exhaustion is about the NUMBER of large per-column
// allocations, not their individual size.
//
// This test reproduces the failure CHEAPLY (no 13 GB download, no 100M
// rows): a small `max_block_size` makes even modest per-column buffers
// force many block-table entries, so a 110-column file exhausts a 32-slot
// table at a few thousand rows in milliseconds. That is the same
// structural mechanism the real ClickBench file hits at real scale -- the
// oversize lane and the normal growing lane both gate on the identical
// `num_blocks_ >= kArenaMaxBlocks` check in bolt_arena.h, so a cheap
// small-block repro and the real large-block failure are the same bug.
//
// Injection-tested (see the ticket): reverting just the kArenaMaxBlocks
// widen (this file's own fixture unchanged) reproduces WideSchemaFileDecodes
// failing at exactly the reported symptom -- parquet_read_file returns
// false partway through -- and restoring the fix makes it pass again.

#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt/ingest/bolt_parquet_write.h"

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

// 110 > kPqMaxColumns's realistic real-world neighborhood and matches the
// ticket's "105+" bar while staying under kPqMaxColumns (128).
constexpr std::uint32_t kCols = 110;
constexpr std::int64_t  kRows = 4000;
constexpr std::int64_t  kRowGroupRows = 800;  // -> 5 row groups

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

// ClickBench's hits.parquet is dominated by narrow numeric columns with a
// substantial minority of wide Utf8 (URL/UserAgent/Referer-shaped) columns.
// Mirror that mix: every 3rd column is Utf8, and of those, most carry
// overflow-spilled (>12B) content -- the allocation shape that actually
// drives the arena's oversize lane / block growth in init_col_ctx_any.
bolt::BoltType type_of(std::uint32_t c) {
    switch (c % 6u) {
        case 0: return bolt::BoltType::Utf8;
        case 1: return bolt::BoltType::Int64;
        case 2: return bolt::BoltType::Utf8;
        case 3: return bolt::BoltType::Float64;
        case 4: return bolt::BoltType::Int32;
        default: return bolt::BoltType::Utf8;
    }
}

std::string str_val(std::uint32_t c, std::int64_t i) {
    char buf[128];
    // Long, low-cardinality-but-not-trivial strings, like a URL/UA column --
    // consistently >12B so every row spills into the overflow buffer.
    std::snprintf(buf, sizeof(buf),
                  "https://example.invalid/col%03u/resource/%06lld?x=abcdef",
                  c, static_cast<long long>(i % 5000));
    return buf;
}

std::vector<std::uint8_t> write_wide_fixture(const char* tag) {
    bolt::Arena a;
    auto* b = a.allocate_array<bolt::BoltBatch>(1);
    EXPECT_NE(b, nullptr);
    bolt::BoltBatch::init_empty(b);
    b->num_cols = kCols;
    b->num_rows = kRows;
    bolt::BoltBatch::alloc_columns(b, &a, kCols);
    char name[32];
    for (std::uint32_t c = 0; c < kCols; ++c) {
        std::snprintf(name, sizeof(name), "col%03u", c);
        const bolt::BoltType t = type_of(c);
        b->schema.add_field(name, t, (c % 4u) == 3u);
        bolt::BoltColumn& col = b->columns[b->read_epoch][c];
        if (t == bolt::BoltType::Utf8) {
            col = bolt::BoltColumn::make_empty();
            col.length = kRows;
            col.format = bolt::ColumnFormat::Flat;
            col.type = t;
            col.type_size_bytes = sizeof(bolt::StringView);
            auto* svs = static_cast<bolt::StringView*>(a.allocate(
                static_cast<std::size_t>(kRows) * sizeof(bolt::StringView),
                alignof(bolt::StringView)));
            std::memset(svs, 0, static_cast<std::size_t>(kRows) * sizeof(bolt::StringView));
            std::size_t need = 0;
            for (std::int64_t i = 0; i < kRows; ++i) {
                need += str_val(c, i).size();
            }
            auto* spill = static_cast<std::uint8_t*>(a.allocate(need, 8));
            std::size_t off = 0;
            for (std::int64_t i = 0; i < kRows; ++i) {
                const std::string s = str_val(c, i);
                svs[i].length = static_cast<std::uint32_t>(s.size());
                std::memcpy(&svs[i].prefix[0], s.data(), 4);
                std::memcpy(spill + off, s.data(), s.size());
                svs[i].ref.offset = static_cast<std::uint32_t>(off);
                off += s.size();
            }
            col.data = svs;
            col.str_overflow_base = spill;
            col.stats.all_valid = true;
        } else {
            col = bolt::BoltColumn::make_flat_alloc(kRows, t, &a);
            if (t == bolt::BoltType::Int64) {
                auto* p = static_cast<std::int64_t*>(col.data);
                for (std::int64_t i = 0; i < kRows; ++i) p[i] = i * (c + 3) - 17;
            } else if (t == bolt::BoltType::Int32) {
                auto* p = static_cast<std::int32_t*>(col.data);
                for (std::int64_t i = 0; i < kRows; ++i) {
                    p[i] = static_cast<std::int32_t>(i * 3 + c);
                }
            } else {
                auto* p = static_cast<double*>(col.data);
                for (std::int64_t i = 0; i < kRows; ++i) {
                    p[i] = static_cast<double>(i) * 0.5 - static_cast<double>(c);
                }
            }
        }
        if ((c % 4u) == 3u) {
            const std::size_t nb = static_cast<std::size_t>((kRows + 7) / 8);
            auto* bm = static_cast<std::uint8_t*>(a.allocate(nb, 8));
            std::memset(bm, 0, nb);
            for (std::int64_t i = 0; i < kRows; ++i) {
                if ((i % 5) != 0) {
                    bm[i >> 3] = static_cast<std::uint8_t>(bm[i >> 3] | (1u << (i & 7)));
                }
            }
            col.validity = bm;
            col.validity_offset = 0;
            col.stats.all_valid = false;
        }
    }

    ParquetWriteOpts o{};
    o.n_columns = kCols;
    o.compression = 1;  // SNAPPY -- matches a real ClickBench-shaped file
    o.emit_statistics = true;
    o.row_group_max_rows = kRowGroupRows;
    for (std::uint32_t c = 0; c < kCols; ++c) {
        std::snprintf(name, sizeof(name), "col%03u", c);
        std::strncpy(o.columns[c].name, name, sizeof(o.columns[c].name) - 1);
        o.columns[c].type = type_of(c);
        o.columns[c].nullable = (c % 4u) == 3u;
    }
    const std::string path =
        std::string("test_bolt_parquet_wide_schema_") + tag + ".parquet";
    ParquetWriter* w = parquet_write_open(path.c_str(), &o);
    EXPECT_NE(w, nullptr);
    if (w == nullptr) return {};
    EXPECT_TRUE(parquet_write_row_group(w, b));
    EXPECT_TRUE(parquet_write_close(w));
    return slurp_file(path.c_str());
}

}  // namespace

// The acceptance bar: a genuinely wide (110-column) real-shaped file must
// decode successfully through parquet_read_file's single-arena, whole-file
// path with a REALISTIC (default-scale) block budget.
TEST(ParquetWideSchema, WideSchemaFileDecodes) {
    const std::vector<std::uint8_t> file = write_wide_fixture("main");
    ASSERT_FALSE(file.empty());

    bolt::ArenaConfig ac;
    ac.initial_block_size = 4ull * 1024 * 1024;
    ac.max_block_size     = 64ull * 1024 * 1024;  // the library DEFAULT
    bolt::Arena arena(ac);

    auto* batch = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(batch, nullptr);

    const bool ok = parquet_read_file(file.data(), file.size(), &arena, batch,
                                      /*decode_pool=*/nullptr);
    ASSERT_TRUE(ok) << "parquet_read_file failed on a " << kCols
                    << "-column file (num_blocks=" << arena.num_blocks()
                    << ", kArenaMaxBlocks=" << bolt::kArenaMaxBlocks << ")";

    EXPECT_EQ(batch->num_cols, kCols);
    EXPECT_EQ(batch->num_rows, kRows);

    // Spot-check correctness across the width, not just "it returned true":
    // one numeric column and one Utf8 (overflow-spilled) column, each
    // touching every row group.
    const bolt::BoltColumn* cols = batch->columns[batch->read_epoch];
    for (std::uint32_t c = 0; c < kCols; c += 17u) {
        const bolt::BoltType t = type_of(c);
        ASSERT_EQ(cols[c].type, t) << "column " << c;
        if (t == bolt::BoltType::Int64) {
            const auto* p = static_cast<const std::int64_t*>(cols[c].data);
            for (std::int64_t r = 0; r < kRows; r += 977) {
                EXPECT_EQ(p[r], r * (static_cast<std::int64_t>(c) + 3) - 17)
                    << "col " << c << " row " << r;
            }
        } else if (t == bolt::BoltType::Utf8) {
            const auto* sv = static_cast<const bolt::StringView*>(cols[c].data);
            for (std::int64_t r = 0; r < kRows; r += 977) {
                const std::string expect = str_val(c, r);
                ASSERT_EQ(sv[r].length, expect.size())
                    << "col " << c << " row " << r;
                const char* p = (sv[r].length <= 12u)
                    ? &sv[r].prefix[0]
                    : reinterpret_cast<const char*>(cols[c].str_overflow_base)
                          + sv[r].ref.offset;
                EXPECT_EQ(std::string(p, sv[r].length), expect)
                    << "col " << c << " row " << r;
            }
        }
    }
}

// A tighter arena budget (small max_block_size) makes the SAME structural
// bug reproduce at trivial memory scale -- both the oversize lane and the
// normal growing lane gate on the identical `num_blocks_ >= kArenaMaxBlocks`
// check, so this is the real mechanism at model-railroad scale, not a
// different bug. Serves as a fast standing regression guard for the
// block-table cap itself, independent of the parquet reader's exact
// per-column byte-size math.
TEST(ParquetWideSchema, WideSchemaSurvivesTightBlockBudget) {
    const std::vector<std::uint8_t> file = write_wide_fixture("tight");
    ASSERT_FALSE(file.empty());

    bolt::ArenaConfig ac;
    ac.initial_block_size = 16ull * 1024;
    ac.max_block_size     = 256ull * 1024;   // 1/256 of the default
    bolt::Arena arena(ac);

    auto* batch = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(batch, nullptr);

    const bool ok = parquet_read_file(file.data(), file.size(), &arena, batch,
                                      /*decode_pool=*/nullptr);
    // This budget needs well over the OLD 32-block cap to succeed (see the
    // injection test in the ticket write-up) -- it is a real exercise of the
    // widened table, not a config that happens to still fit in 32.
    ASSERT_GT(arena.num_blocks(), 32u)
        << "test no longer exercises more than the old 32-block cap -- "
        << "tighten max_block_size";
    ASSERT_TRUE(ok) << "parquet_read_file failed under a tight block budget "
                    << "(num_blocks=" << arena.num_blocks()
                    << ", kArenaMaxBlocks=" << bolt::kArenaMaxBlocks << ")";
    EXPECT_EQ(batch->num_cols, kCols);
    EXPECT_EQ(batch->num_rows, kRows);
}
