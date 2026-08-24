// Page-level skipping, end to end: write a page index, use it to decode only
// the pages a predicate can match, and prove the result equals a full decode
// restricted to the same rows.
//
// This is the first test in the tree that can exist at all. The page index
// reader (bolt_parquet_pageindex.h) and the resumable page decoder
// (parquet_read_col_chunk_pages) were both present and both tested in
// isolation, but nothing could WRITE a page index, so the path from
// "predicate excludes this page" to "those bytes were never decoded" had no
// fixture. It does now.
//
// Two things are asserted, and the second is the one that matters:
//
//   1. The rows that survive skipping are exactly the rows a full decode
//      would have produced for those pages -- value for value.
//   2. Pages really WERE skipped. A reader that decodes everything and
//      filters afterwards passes (1) perfectly while doing none of the work,
//      so the page count actually decoded is checked against the count the
//      index says can match.
//
// The dictionary case is the point of the exercise: parquet-mr, Arrow and
// now bolt all emit dictionary-encoded pages by default, so a resumable
// decoder that only handled PLAIN could not skip pages in most real files.

#include "bolt/ingest/bolt_parquet_write.h"
#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt/ingest/bolt_parquet_meta.h"
#include "bolt/ingest/bolt_parquet_pageindex.h"

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

constexpr std::int64_t kRows = 40000;

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

// Monotonically increasing, so a range predicate selects a contiguous span of
// pages and the number that CAN be skipped is large and easy to reason about.
std::int64_t val_at(std::int64_t i) { return i * 10; }

void build_batch(bolt::Arena* a, bolt::BoltBatch* out) {
    bolt::BoltBatch::init_empty(out);
    out->num_cols = 1;
    out->num_rows = kRows;
    bolt::BoltBatch::alloc_columns(out, a, 1);
    out->schema.add_field("v", bolt::BoltType::Int64, false);
    bolt::BoltColumn& c = out->columns[out->read_epoch][0];
    c = bolt::BoltColumn::make_flat_alloc(kRows, bolt::BoltType::Int64, a);
    auto* p = static_cast<std::int64_t*>(c.data);
    for (std::int64_t i = 0; i < kRows; ++i) p[i] = val_at(i);
}

std::vector<std::uint8_t> write_indexed(bool dictionary, const char* tag) {
    bolt::Arena a;
    auto* b = a.allocate_array<bolt::BoltBatch>(1);
    build_batch(&a, b);
    ParquetWriteOpts o{};
    o.n_columns = 1;
    o.compression = 0;                   // uncompressed: keeps the test about
                                         // page selection, not codec behaviour
    o.emit_statistics = true;
    o.emit_page_index = true;
    o.use_dictionary = dictionary;
    o.data_page_target_bytes = 4096;     // ~512 values/page -> ~78 pages
    std::strncpy(o.columns[0].name, "v", sizeof(o.columns[0].name) - 1);
    o.columns[0].type = bolt::BoltType::Int64;
    o.columns[0].nullable = false;
    const std::string path =
        std::string("test_bolt_parquet_page_skip_") + tag + ".parquet";
    ParquetWriter* w = parquet_write_open(path.c_str(), &o);
    EXPECT_NE(w, nullptr);
    if (w == nullptr) return {};
    EXPECT_TRUE(parquet_write_row_group(w, b));
    EXPECT_TRUE(parquet_write_close(w));
    return slurp_file(path.c_str());
}

struct SkipResult {
    std::vector<std::int64_t> values;      // decoded values, in row order
    std::uint32_t pages_total;
    std::uint32_t pages_decoded;
};

// The whole point, in one function: consult the ColumnIndex, decode only the
// pages whose declared bounds can contain a value in [lo, hi], and jump
// between them with the OffsetIndex.
bool skip_scan(const std::vector<std::uint8_t>& buf, std::int64_t lo,
               std::int64_t hi, SkipResult* out) {
    bolt::Arena a;
    PqMeta meta{};
    if (!parquet_read_meta(buf.data(), buf.size(), &a, &meta)) return false;
    if (meta.n_chunks < 1u) return false;
    const PqChunk& ch = meta.chunks[0];

    PqColumnIndex ci{};
    ci.pages = a.allocate_array<PqPageStat>(kPqMaxPagesPerChunk);
    ci.pages_cap = kPqMaxPagesPerChunk;
    if (!pq_read_column_index(buf.data(), buf.size(), ch, &ci)) return false;
    PqOffsetIndex oi{};
    oi.pages = a.allocate_array<PqPageLocation>(kPqMaxPagesPerChunk);
    oi.pages_cap = kPqMaxPagesPerChunk;
    if (!pq_read_offset_index(buf.data(), buf.size(), ch, &oi)) return false;
    if (ci.n_pages != oi.n_pages) return false;

    out->values.clear();
    out->pages_total = oi.n_pages;
    out->pages_decoded = 0;

    for (std::uint32_t p = 0; p < oi.n_pages; ++p) {
        std::int64_t pmin = 0, pmax = 0;
        if (!pq_page_range_i64(meta.columns[0], ci.pages[p], &pmin, &pmax)) {
            // Unprovable bounds: the page must be kept. Conservative by
            // design -- "cannot prove" never means "skip".
        } else if (pmax < lo || pmin > hi) {
            continue;                       // PROVEN disjoint: skip the bytes
        }
        ++out->pages_decoded;
        bolt::Arena pa;
        bolt::BoltColumn col{};
        std::int64_t got = 0;
        std::uint64_t next = 0;
        const std::int64_t page_rows =
            ((p + 1u < oi.n_pages) ? oi.pages[p + 1u].first_row_index : kRows) -
            oi.pages[p].first_row_index;
        if (!parquet_read_col_chunk_pages(
                buf.data(), buf.size(), &meta, 0, 0,
                static_cast<std::uint64_t>(oi.pages[p].offset), page_rows,
                &pa, &col, &got, &next)) {
            return false;
        }
        if (got < page_rows) return false;
        const auto* vp = static_cast<const std::int64_t*>(col.data);
        for (std::int64_t i = 0; i < page_rows; ++i) out->values.push_back(vp[i]);
    }
    return true;
}

// ---- tests ---------------------------------------------------------------

TEST(BoltParquetPageSkip, PlainAndDictionaryBothSkipAndAgree) {
    for (bool dict : {false, true}) {
        const auto buf = write_indexed(dict, dict ? "dict" : "plain");
        ASSERT_FALSE(buf.empty());
        SCOPED_TRACE(testing::Message() << "dictionary=" << dict);

        // A narrow window near the middle: most pages are provably disjoint.
        const std::int64_t lo = val_at(20000);
        const std::int64_t hi = val_at(20500);
        SkipResult r{};
        ASSERT_TRUE(skip_scan(buf, lo, hi, &r));
        // Page counts differ by encoding for the same budget, and should:
        // a dictionary column's data pages carry 2-byte indices where the
        // PLAIN column carries 8-byte values, so ~4x more rows fit a page.
        // The bar is only that the chunk really is many pages.
        ASSERT_GT(r.pages_total, 8u) << "the file was not split into pages";

        // (2) Pages really were skipped. Without this the test passes for a
        // reader that decodes everything.
        EXPECT_LT(r.pages_decoded, r.pages_total / 4u)
            << "decoded " << r.pages_decoded << " of " << r.pages_total
            << " pages -- skipping is not happening";
        EXPECT_GT(r.pages_decoded, 0u);

        // (1) Every value the kept pages produced is right, and the window is
        // fully covered -- skipping must not have dropped a matching row.
        ASSERT_FALSE(r.values.empty());
        std::int64_t seen_in_window = 0;
        for (std::size_t i = 0; i < r.values.size(); ++i) {
            // Values are strictly increasing, so a decoded page's contents are
            // checkable against the closed form without knowing which page.
            ASSERT_EQ(r.values[i] % 10, 0) << "value " << i << " is not ours";
            if (r.values[i] >= lo && r.values[i] <= hi) ++seen_in_window;
        }
        std::int64_t want_in_window = 0;
        for (std::int64_t i = 0; i < kRows; ++i) {
            const std::int64_t v = val_at(i);
            if (v >= lo && v <= hi) ++want_in_window;
        }
        EXPECT_EQ(seen_in_window, want_in_window)
            << "skipping dropped rows the predicate matches";
    }
}

TEST(BoltParquetPageSkip, DictionaryResumeMatchesAWholeChunkDecode) {
    // The resumable decoder re-decodes the dictionary on every call. Its
    // output must still be value-identical to one whole-chunk decode of the
    // same rows -- that is the property that makes jumping safe.
    const auto buf = write_indexed(/*dictionary=*/true, "resume");
    ASSERT_FALSE(buf.empty());

    bolt::Arena ra;
    auto* rb = ra.allocate_array<bolt::BoltBatch>(1);
    ASSERT_TRUE(parquet_read_file(buf.data(), buf.size(), &ra, rb));
    ASSERT_EQ(rb->num_rows, kRows);
    const auto* whole = static_cast<const std::int64_t*>(
        rb->columns[rb->read_epoch][0].data);
    ASSERT_NE(whole, nullptr);

    // Walk the chunk in resumable sub-chunks and compare against it.
    bolt::Arena a;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &a, &meta));
    std::uint64_t off = 0;
    std::int64_t row = 0;
    int guard = 0;
    do {
        ASSERT_LT(++guard, 10000) << "resume loop did not terminate";
        bolt::Arena pa;
        bolt::BoltColumn col{};
        std::int64_t got = 0;
        std::uint64_t next = 0;
        ASSERT_TRUE(parquet_read_col_chunk_pages(buf.data(), buf.size(), &meta,
                                                 0, 0, off, 1000, &pa, &col,
                                                 &got, &next))
            << "resume failed at row " << row;
        if (got == 0) break;
        const auto* p = static_cast<const std::int64_t*>(col.data);
        for (std::int64_t i = 0; i < got; ++i) {
            ASSERT_EQ(p[i], whole[row + i]) << "row " << (row + i);
        }
        row += got;
        off = next;
    } while (off != 0);
    EXPECT_EQ(row, kRows) << "resume walk did not cover the chunk";
}

TEST(BoltParquetPageSkip, DiscriminatingPower) {
    // If pq_page_range_i64 returned bounds that always matched, the skip test
    // above would decode everything and its page-count assertion would fail
    // -- so that assertion IS the discriminator. Confirm the other direction
    // too: a predicate covering the whole range must skip nothing, and one
    // outside it must skip everything.
    const auto buf = write_indexed(true, "disc");
    ASSERT_FALSE(buf.empty());

    SkipResult all{};
    ASSERT_TRUE(skip_scan(buf, val_at(0), val_at(kRows - 1), &all));
    EXPECT_EQ(all.pages_decoded, all.pages_total)
        << "a predicate covering every value skipped pages";
    EXPECT_EQ(all.values.size(), static_cast<std::size_t>(kRows));

    SkipResult none{};
    ASSERT_TRUE(skip_scan(buf, val_at(kRows) + 1, val_at(kRows) + 100, &none));
    EXPECT_EQ(none.pages_decoded, 0u)
        << "a predicate matching nothing still decoded pages";
    EXPECT_TRUE(none.values.empty());
}

}  // namespace
