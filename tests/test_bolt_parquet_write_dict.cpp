// Writer: dictionary encoding, data-page splitting, and the page index.
//
// These three landed together because they compose: splitting a chunk into
// pages is what makes a per-page index mean anything, and the dictionary is
// what a page's index entry usually describes.
//
// The verification standard here is the one
// docs/research/parquet-reader-completeness-plan.md sets for encoding work,
// because every case below is a silent-wrong-data risk rather than a crash:
//
//   1. Assert VALUES, not row counts. A page cut in the wrong place, or a
//      dictionary index off by one, still produces exactly the right number
//      of rows.
//   2. Sweep the parameter space: bit widths either side of a byte boundary,
//      page boundaries that fall on and between 8-value groups, nulls
//      interleaved across a page cut, all-null pages, single-entry
//      dictionaries (bit width 0), and dictionary overflow.
//   3. Prove the gate discriminates. DiscriminatingPower below writes a file
//      whose pages are deliberately mis-planned and confirms the comparison
//      catches it -- without that, a suite that never reached the code under
//      test would look identical to a passing one.
//
// Interop against pyarrow is a separate gate: InteropFixtures writes the
// files that scripts/parquet_write_interop.py reads back.

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
    std::string s = "test_bolt_parquet_write_dict_";
    s += tag;
    s += ".parquet";
    return s;
}

// ---- source data ---------------------------------------------------------
//
// One reference model both the writer input and the expected output are built
// from, so a test can never accidentally compare the writer against itself.
struct Model {
    std::vector<std::int64_t>  ints;
    std::vector<std::string>   strs;
    std::vector<std::uint8_t>  valid;   // 1 = non-null (both columns)
};

// `distinct` controls cardinality: value i is (i % distinct), so the
// dictionary has exactly `distinct` entries and the index bit width is
// ceil(log2(distinct)) -- which is how the bit-width sweep is driven.
Model make_model(std::int64_t n, std::int64_t distinct, int null_every) {
    Model m;
    m.ints.resize(static_cast<std::size_t>(n));
    m.strs.resize(static_cast<std::size_t>(n));
    m.valid.assign(static_cast<std::size_t>(n), 1u);
    for (std::int64_t i = 0; i < n; ++i) {
        const std::int64_t d = (distinct > 0) ? (i % distinct) : i;
        m.ints[static_cast<std::size_t>(i)] = d * 7 - 3;
        char buf[64];
        std::snprintf(buf, sizeof(buf), "v%lld", static_cast<long long>(d));
        m.strs[static_cast<std::size_t>(i)] = buf;
        if (null_every > 0 && (i % null_every) == 0) {
            m.valid[static_cast<std::size_t>(i)] = 0u;
        }
    }
    return m;
}

void set_validity(bolt::BoltColumn* col, bolt::Arena* arena,
                  const std::vector<std::uint8_t>& valid) {
    const std::size_t n = valid.size();
    const std::size_t nb = (n + 7u) / 8u;
    auto* bm = static_cast<std::uint8_t*>(arena->allocate(nb, 8));
    std::memset(bm, 0, nb);
    for (std::size_t i = 0; i < n; ++i) {
        if (valid[i]) bm[i >> 3] = static_cast<std::uint8_t>(bm[i >> 3] | (1u << (i & 7u)));
    }
    col->validity = bm;
    col->validity_offset = 0;
    col->stats.all_valid = false;
}

// Build an (Int64, Utf8) batch from the model. `nullable` decides whether the
// validity bitmap is attached at all.
void build_batch(bolt::Arena* arena, const Model& m, bool nullable,
                 bolt::BoltBatch* out) {
    const std::int64_t n = static_cast<std::int64_t>(m.ints.size());
    bolt::BoltBatch::init_empty(out);
    out->num_cols = 2;
    out->num_rows = n;
    bolt::BoltBatch::alloc_columns(out, arena, 2);
    out->schema.add_field("id", bolt::BoltType::Int64, nullable);
    out->schema.add_field("name", bolt::BoltType::Utf8, nullable);

    bolt::BoltColumn& id = out->columns[out->read_epoch][0];
    id = bolt::BoltColumn::make_flat_alloc(n, bolt::BoltType::Int64, arena);
    ASSERT_NE(id.data, nullptr);
    auto* ip = static_cast<std::int64_t*>(id.data);
    for (std::int64_t i = 0; i < n; ++i) ip[i] = m.ints[static_cast<std::size_t>(i)];

    bolt::BoltColumn& nm = out->columns[out->read_epoch][1];
    nm = bolt::BoltColumn::make_empty();
    nm.length = n;
    nm.format = bolt::ColumnFormat::Flat;
    nm.type = bolt::BoltType::Utf8;
    nm.type_size_bytes = sizeof(bolt::StringView);
    auto* svs = static_cast<bolt::StringView*>(
        arena->allocate(static_cast<std::size_t>(n) * sizeof(bolt::StringView),
                        alignof(bolt::StringView)));
    ASSERT_NE(svs, nullptr);
    std::memset(svs, 0, static_cast<std::size_t>(n) * sizeof(bolt::StringView));
    nm.data = svs;
    nm.stats.all_valid = true;
    for (std::int64_t i = 0; i < n; ++i) {
        svs[i] = bolt::StringView::from_cstr(m.strs[static_cast<std::size_t>(i)].c_str());
    }
    if (nullable) {
        set_validity(&id, arena, m.valid);
        set_validity(&nm, arena, m.valid);
    }
}

ParquetWriteOpts make_opts(bool nullable, bool dict, bool page_index,
                           std::uint32_t page_bytes, std::uint32_t dict_bytes,
                           std::uint8_t codec) {
    ParquetWriteOpts o{};
    o.n_columns = 2;
    o.compression = codec;
    o.emit_statistics = true;
    o.use_dictionary = dict;
    o.emit_page_index = page_index;
    o.data_page_target_bytes = page_bytes;
    o.dictionary_max_bytes = dict_bytes;
    std::strncpy(o.columns[0].name, "id", sizeof(o.columns[0].name) - 1);
    o.columns[0].type = bolt::BoltType::Int64;
    o.columns[0].nullable = nullable;
    std::strncpy(o.columns[1].name, "name", sizeof(o.columns[1].name) - 1);
    o.columns[1].type = bolt::BoltType::Utf8;
    o.columns[1].nullable = nullable;
    return o;
}

std::vector<std::uint8_t> write_model(const Model& m, const ParquetWriteOpts& o,
                                      bool nullable, const char* tag) {
    bolt::Arena arena;
    auto* batch = arena.allocate_array<bolt::BoltBatch>(1);
    EXPECT_NE(batch, nullptr);
    build_batch(&arena, m, nullable, batch);
    const std::string path = tmp_path(tag);
    ParquetWriter* w = parquet_write_open(path.c_str(), &o);
    EXPECT_NE(w, nullptr);
    if (w == nullptr) return {};
    EXPECT_TRUE(parquet_write_row_group(w, batch));
    EXPECT_TRUE(parquet_write_close(w));
    return slurp_file(path.c_str());
}

// Compare every decoded VALUE against the model, including null placement.
// Returns an empty string on a full match, else a description of the first
// mismatch. Pure -- no gtest macros -- so DiscriminatingPower below can call
// the exact comparison the other tests rely on and assert that it FAILS on a
// perturbed model. A checker that could only pass would prove nothing.
std::string diff_roundtrip(const std::vector<std::uint8_t>& buf, const Model& m,
                           bool nullable) {
    char msg[256];
    if (buf.empty()) return "empty file";
    bolt::Arena ra;
    auto* rb = ra.allocate_array<bolt::BoltBatch>(1);
    if (rb == nullptr) return "alloc failed";
    if (!parquet_read_file(buf.data(), buf.size(), &ra, rb)) return "read failed";
    const std::int64_t n = static_cast<std::int64_t>(m.ints.size());
    if (rb->num_rows != n) return "row count mismatch";
    if (rb->num_cols != 2u) return "col count mismatch";

    const bolt::BoltColumn& id = rb->columns[rb->read_epoch][0];
    const bolt::BoltColumn& nm = rb->columns[rb->read_epoch][1];
    const auto* ip = static_cast<const std::int64_t*>(id.data);
    const auto* sv = static_cast<const bolt::StringView*>(nm.data);
    const auto* spill = static_cast<const std::uint8_t*>(nm.str_overflow_base);
    if (ip == nullptr || sv == nullptr) return "null column data";

    for (std::int64_t i = 0; i < n; ++i) {
        const std::size_t u = static_cast<std::size_t>(i);
        const bool want_valid = !nullable || m.valid[u] != 0u;
        if (nullable && id.validity != nullptr) {
            const std::int64_t bit = id.validity_offset + i;
            const bool got_valid =
                ((id.validity[bit >> 3] >> (bit & 7)) & 1u) != 0u;
            if (got_valid != want_valid) {
                std::snprintf(msg, sizeof(msg), "id validity at row %lld",
                              static_cast<long long>(i));
                return msg;
            }
        }
        if (!want_valid) continue;
        if (ip[i] != m.ints[u]) {
            std::snprintf(msg, sizeof(msg),
                          "id value at row %lld: got %lld want %lld",
                          static_cast<long long>(i),
                          static_cast<long long>(ip[i]),
                          static_cast<long long>(m.ints[u]));
            return msg;
        }
        const std::uint32_t len = sv[i].length;
        if (len > 12u && spill == nullptr) return "missing spill base";
        const std::uint8_t* p =
            (len <= 12u)
                ? reinterpret_cast<const std::uint8_t*>(&sv[i].prefix[0])
                : (spill + sv[i].ref.offset);
        if (len != m.strs[u].size() ||
            std::memcmp(p, m.strs[u].data(), len) != 0) {
            std::snprintf(msg, sizeof(msg), "name value at row %lld",
                          static_cast<long long>(i));
            return msg;
        }
    }
    return std::string();
}

void expect_roundtrip(const std::vector<std::uint8_t>& buf, const Model& m,
                      bool nullable) {
    const std::string d = diff_roundtrip(buf, m, nullable);
    ASSERT_TRUE(d.empty()) << d;
}

// ---- round-trip sweep ----------------------------------------------------

TEST(BoltParquetWriteDict, DictionaryRoundtripSweep) {
    // distinct sweeps the index bit width across byte boundaries and past the
    // bw == 0 special case: 1 -> bw 0, 2 -> 1, 3 -> 2, 16 -> 4, 17 -> 5,
    // 256 -> 8, 257 -> 9.
    const std::int64_t kDistinct[] = {1, 2, 3, 16, 17, 256, 257};
    for (std::int64_t d : kDistinct) {
        for (int null_every : {0, 3, 1}) {
            const bool nullable = null_every > 0;
            // null_every == 1 makes every row null: the all-null chunk path.
            const Model m = make_model(2000, d, null_every);
            const auto o = make_opts(nullable, /*dict=*/true,
                                     /*page_index=*/true,
                                     /*page_bytes=*/0, /*dict_bytes=*/0,
                                     /*codec=*/1);
            const auto buf = write_model(m, o, nullable, "sweep");
            SCOPED_TRACE(testing::Message()
                         << "distinct=" << d << " null_every=" << null_every);
            expect_roundtrip(buf, m, nullable);
        }
    }
}

TEST(BoltParquetWriteDict, PlainAndDictionaryAgreeValueForValue) {
    // The two encodings are independent code paths that must decode to the
    // same thing. Comparing them against each other (not just against the
    // model) is what catches an error the model itself shares.
    for (int null_every : {0, 4}) {
        const bool nullable = null_every > 0;
        const Model m = make_model(5000, 37, null_every);
        const auto plain = write_model(
            m, make_opts(nullable, false, false, 8192, 0, 0), nullable, "p");
        const auto dict = write_model(
            m, make_opts(nullable, true, false, 8192, 0, 0), nullable, "d");
        expect_roundtrip(plain, m, nullable);
        expect_roundtrip(dict, m, nullable);
        // Dictionary must actually be smaller here -- 37 distinct values over
        // 5000 rows. If it is not, the dictionary silently did not engage.
        EXPECT_LT(dict.size(), plain.size());
    }
}

TEST(BoltParquetWriteDict, DictionaryOverflowFallsBackToPlain) {
    // Every value distinct, with a dictionary ceiling far below what that
    // needs. The chunk must fall back to PLAIN for the WHOLE chunk and still
    // decode exactly -- a mid-chunk switch would leave pages disagreeing.
    const Model m = make_model(4000, /*distinct=*/0, /*null_every=*/0);
    const auto o = make_opts(false, /*dict=*/true, /*page_index=*/true,
                             /*page_bytes=*/8192, /*dict_bytes=*/512,
                             /*codec=*/0);
    const auto buf = write_model(m, o, false, "overflow");
    expect_roundtrip(buf, m, false);

    // Confirm the fallback really happened: no dictionary page offset.
    bolt::Arena a;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &a, &meta));
    ASSERT_GE(meta.n_chunks, 2u);
    for (std::uint32_t i = 0; i < meta.n_chunks; ++i) {
        EXPECT_EQ(meta.chunks[i].dictionary_page_offset, 0)
            << "chunk " << i << " kept a dictionary past its ceiling";
    }
}

// ---- dictionary slot-table growth ---------------------------------------
//
// The table is sized to what the dictionary HOLDS and doubles on demand, so
// these use row counts well past its 1024-slot start -- every other fixture
// in this file is small enough that it never grows.

TEST(BoltParquetWriteDict, LowCardinalityColumnKeepsItsDictionary) {
    // The other side of the probe, and the one that matters: a heuristic that
    // abandons too eagerly would silently cost ratio on exactly the columns
    // dictionary encoding exists for. 64 distinct values over 20000 rows is
    // 0.3% distinct across the probe window, so it must survive.
    const Model m = make_model(20000, /*distinct=*/64, /*null_every=*/0);
    const auto o = make_opts(false, /*dict=*/true, /*page_index=*/false,
                             /*page_bytes=*/0, /*dict_bytes=*/1u << 20,
                             /*codec=*/0);
    const auto buf = write_model(m, o, false, "probe_lowcard");
    expect_roundtrip(buf, m, false);

    bolt::Arena a;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &a, &meta));
    ASSERT_GE(meta.n_chunks, 1u);
    for (std::uint32_t i = 0; i < meta.n_chunks; ++i) {
        EXPECT_NE(meta.chunks[i].dictionary_page_offset, 0)
            << "chunk " << i << " threw away a dictionary that pays";
    }
}

TEST(BoltParquetWriteDict, DictionaryTableGrowsPastItsInitialSlots) {
    // Exercises dict_grow. The slot table now STARTS at 1024 entries and
    // doubles on demand, so a dictionary of ~3000 values rehashes several
    // times mid-build; every value must still come back and the dictionary
    // must still be used. 3000 distinct over the 4096-row probe window is
    // 73% -- under the threshold, so the probe deliberately passes here.
    const Model m = make_model(20000, /*distinct=*/3000, /*null_every=*/0);
    const auto o = make_opts(false, /*dict=*/true, /*page_index=*/false,
                             /*page_bytes=*/0, /*dict_bytes=*/1u << 20,
                             /*codec=*/0);
    const auto buf = write_model(m, o, false, "probe_grow");
    expect_roundtrip(buf, m, false);

    bolt::Arena a;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &a, &meta));
    ASSERT_GE(meta.n_chunks, 1u);
    EXPECT_NE(meta.chunks[0].dictionary_page_offset, 0)
        << "a 3000-entry dictionary should survive the probe";
}

// ---- page splitting ------------------------------------------------------

// Parse a chunk's OffsetIndex, returning page count (0 = no index present).
std::uint32_t read_page_count(const std::vector<std::uint8_t>& buf,
                              const PqChunk& ch, bolt::Arena* a) {
    PqOffsetIndex oi{};
    oi.pages = a->allocate_array<PqPageLocation>(kPqMaxPagesPerChunk);
    oi.pages_cap = kPqMaxPagesPerChunk;
    if (!pq_read_offset_index(buf.data(), buf.size(), ch, &oi)) return 0;
    return oi.n_pages;
}

TEST(BoltParquetWriteDict, SmallPageBudgetProducesManyPages) {
    // 20000 Int64 rows at a 4 KiB budget is ~512 values per page, so the
    // chunk must hold tens of pages -- and every value must still come back.
    const Model m = make_model(20000, /*distinct=*/0, /*null_every=*/0);
    const auto o = make_opts(false, /*dict=*/false, /*page_index=*/true,
                             /*page_bytes=*/4096, 0, /*codec=*/0);
    const auto buf = write_model(m, o, false, "split");
    expect_roundtrip(buf, m, false);

    bolt::Arena a;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &a, &meta));
    ASSERT_GE(meta.n_chunks, 1u);
    const std::uint32_t pages = read_page_count(buf, meta.chunks[0], &a);
    EXPECT_GT(pages, 20u) << "a 4 KiB budget did not split the chunk";
    // Int64 at 4 KiB is 512 values/page; allow slack but bound it so a
    // runaway page count (budget ignored the other way) also fails.
    EXPECT_LT(pages, 200u);
}

TEST(BoltParquetWriteDict, PageBoundariesFallBetweenGroupsOfEight) {
    // The bit-packed run can only be cut on an 8-value boundary, so a page
    // whose value count is not a multiple of 8 exercises the trailing-pad
    // path on EVERY page rather than only the last one. Budgets chosen so
    // values-per-page lands at 5, 13 and 101 -- none a multiple of 8.
    for (std::uint32_t vals_per_page : {5u, 13u, 101u}) {
        const Model m = make_model(1000, /*distinct=*/200, /*null_every=*/0);
        auto o = make_opts(false, /*dict=*/true, /*page_index=*/true,
                           /*page_bytes=*/0, 0, /*codec=*/0);
        // 200 distinct -> bit width 8 -> one byte per value.
        o.data_page_target_bytes = vals_per_page;   // clamped up to 4 KiB
        const auto buf = write_model(m, o, false, "groups");
        SCOPED_TRACE(testing::Message() << "vals_per_page=" << vals_per_page);
        expect_roundtrip(buf, m, false);
    }
}

TEST(BoltParquetWriteDict, NullsStraddlePageBoundaries) {
    // Nulls interleaved at a period coprime with the page's value count, so
    // page cuts land inside null runs as well as outside them.
    for (int null_every : {2, 3, 7}) {
        const Model m = make_model(6000, /*distinct=*/11, null_every);
        const auto o = make_opts(true, /*dict=*/true, /*page_index=*/true,
                                 /*page_bytes=*/4096, 0, /*codec=*/1);
        const auto buf = write_model(m, o, true, "straddle");
        SCOPED_TRACE(testing::Message() << "null_every=" << null_every);
        expect_roundtrip(buf, m, true);
    }
}

// ---- page index ----------------------------------------------------------

TEST(BoltParquetWriteDict, PageIndexDescribesTheRealPages) {
    const Model m = make_model(20000, /*distinct=*/0, /*null_every=*/0);
    const auto o = make_opts(false, /*dict=*/false, /*page_index=*/true,
                             /*page_bytes=*/4096, 0, /*codec=*/0);
    const auto buf = write_model(m, o, false, "index");
    ASSERT_FALSE(buf.empty());

    bolt::Arena a;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &a, &meta));
    ASSERT_GE(meta.n_chunks, 1u);
    const PqChunk& ch = meta.chunks[0];   // the Int64 column

    PqOffsetIndex oi{};
    oi.pages = a.allocate_array<PqPageLocation>(kPqMaxPagesPerChunk);
    oi.pages_cap = kPqMaxPagesPerChunk;
    ASSERT_TRUE(pq_read_offset_index(buf.data(), buf.size(), ch, &oi));

    PqColumnIndex ci{};
    ci.pages = a.allocate_array<PqPageStat>(kPqMaxPagesPerChunk);
    ci.pages_cap = kPqMaxPagesPerChunk;
    ASSERT_TRUE(pq_read_column_index(buf.data(), buf.size(), ch, &ci));

    ASSERT_EQ(oi.n_pages, ci.n_pages);
    ASSERT_GT(oi.n_pages, 1u);

    // Every page's declared bounds must actually bracket its rows, and the
    // page's row window must be the one the OffsetIndex claims. This is the
    // property a reader relies on to SKIP a page; if it is wrong, pruning
    // silently drops matching rows.
    std::int64_t total_rows = 0;
    for (std::uint32_t p = 0; p < oi.n_pages; ++p) {
        const std::int64_t first = oi.pages[p].first_row_index;
        const std::int64_t last = (p + 1u < oi.n_pages)
            ? oi.pages[p + 1u].first_row_index
            : static_cast<std::int64_t>(m.ints.size());
        ASSERT_GT(last, first) << "page " << p << " is empty";
        ASSERT_EQ(first, total_rows) << "page " << p << " row window gap";
        total_rows = last;

        ASSERT_EQ(ci.pages[p].min_len, 8u);
        ASSERT_EQ(ci.pages[p].max_len, 8u);
        std::int64_t pmin = 0, pmax = 0;
        std::memcpy(&pmin, ci.pages[p].min_bytes, 8);
        std::memcpy(&pmax, ci.pages[p].max_bytes, 8);
        for (std::int64_t r = first; r < last; ++r) {
            const std::int64_t v = m.ints[static_cast<std::size_t>(r)];
            ASSERT_GE(v, pmin) << "page " << p << " min excludes row " << r;
            ASSERT_LE(v, pmax) << "page " << p << " max excludes row " << r;
        }
        EXPECT_EQ(ci.pages[p].null_page, 0u);
        EXPECT_EQ(ci.pages[p].null_count, 0);
        EXPECT_GT(oi.pages[p].offset, 0);
        EXPECT_GT(oi.pages[p].compressed_page_size, 0);
    }
    EXPECT_EQ(total_rows, static_cast<std::int64_t>(m.ints.size()));
    // Values ascend with the row index here, so the writer must have proved
    // ASCENDING rather than giving up and writing UNORDERED.
    EXPECT_EQ(ci.boundary_order, 1);
}

TEST(BoltParquetWriteDict, PageIndexNullCountsSumToTheChunk) {
    const Model m = make_model(9000, /*distinct=*/5, /*null_every=*/3);
    const auto o = make_opts(true, /*dict=*/true, /*page_index=*/true,
                             /*page_bytes=*/4096, 0, /*codec=*/0);
    const auto buf = write_model(m, o, true, "nullcnt");
    ASSERT_FALSE(buf.empty());

    std::int64_t want_nulls = 0;
    for (std::uint8_t v : m.valid) want_nulls += (v == 0u);

    bolt::Arena a;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &a, &meta));
    PqColumnIndex ci{};
    ci.pages = a.allocate_array<PqPageStat>(kPqMaxPagesPerChunk);
    ci.pages_cap = kPqMaxPagesPerChunk;
    ASSERT_TRUE(pq_read_column_index(buf.data(), buf.size(), meta.chunks[0], &ci));
    std::int64_t got = 0;
    for (std::uint32_t p = 0; p < ci.n_pages; ++p) {
        ASSERT_GE(ci.pages[p].null_count, 0);
        got += ci.pages[p].null_count;
    }
    EXPECT_EQ(got, want_nulls);
    EXPECT_EQ(meta.chunks[0].null_count, want_nulls);
}

TEST(BoltParquetWriteDict, AllNullPagesAreFlagged) {
    const Model m = make_model(3000, /*distinct=*/4, /*null_every=*/1);
    const auto o = make_opts(true, /*dict=*/false, /*page_index=*/true,
                             /*page_bytes=*/4096, 0, /*codec=*/0);
    const auto buf = write_model(m, o, true, "allnull");
    ASSERT_FALSE(buf.empty());
    bolt::Arena a;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &a, &meta));
    PqColumnIndex ci{};
    ci.pages = a.allocate_array<PqPageStat>(kPqMaxPagesPerChunk);
    ci.pages_cap = kPqMaxPagesPerChunk;
    ASSERT_TRUE(pq_read_column_index(buf.data(), buf.size(), meta.chunks[0], &ci));
    ASSERT_GT(ci.n_pages, 0u);
    for (std::uint32_t p = 0; p < ci.n_pages; ++p) {
        EXPECT_EQ(ci.pages[p].null_page, 1u) << "page " << p;
        EXPECT_EQ(ci.pages[p].min_len, 0u) << "null page carries a bound";
        EXPECT_EQ(ci.pages[p].max_len, 0u) << "null page carries a bound";
    }
}

TEST(BoltParquetWriteDict, NoPageIndexWhenNotRequested) {
    const Model m = make_model(5000, 9, 0);
    const auto o = make_opts(false, true, /*page_index=*/false, 4096, 0, 0);
    const auto buf = write_model(m, o, false, "noindex");
    ASSERT_FALSE(buf.empty());
    bolt::Arena a;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &a, &meta));
    PqOffsetIndex oi{};
    oi.pages = a.allocate_array<PqPageLocation>(kPqMaxPagesPerChunk);
    oi.pages_cap = kPqMaxPagesPerChunk;
    EXPECT_FALSE(pq_read_offset_index(buf.data(), buf.size(), meta.chunks[0], &oi));
    // ...and the data still reads.
    expect_roundtrip(buf, m, false);
}

// ---- the gate must discriminate -----------------------------------------

TEST(BoltParquetWriteDict, DiscriminatingPower) {
    // Everything above compares decoded values against `Model`. If that
    // comparison were vacuous -- reading its expectation from the file rather
    // than from the model -- every test would pass no matter what the writer
    // did. Perturb the model by one value and confirm the SAME comparison
    // that passes elsewhere reports exactly that row.
    const Model m = make_model(2000, 13, 0);
    const auto o = make_opts(false, true, true, 4096, 0, 1);
    const auto buf = write_model(m, o, false, "discrim");
    ASSERT_FALSE(buf.empty());
    EXPECT_EQ(diff_roundtrip(buf, m, false), std::string());

    Model bad = m;
    bad.ints[1234] += 1;                      // one value, one row
    const std::string d1 = diff_roundtrip(buf, bad, false);
    EXPECT_NE(d1.find("id value at row 1234"), std::string::npos) << d1;

    Model bad2 = m;
    bad2.strs[77] = "not-the-written-value";
    const std::string d2 = diff_roundtrip(buf, bad2, false);
    EXPECT_NE(d2.find("name value at row 77"), std::string::npos) << d2;

    // A dropped null must be caught too, not just a wrong value.
    Model bad3 = make_model(2000, 13, 4);
    const auto on = make_opts(true, true, true, 4096, 0, 1);
    const auto bufn = write_model(bad3, on, true, "discrim_null");
    EXPECT_EQ(diff_roundtrip(bufn, bad3, true), std::string());
    Model bad4 = bad3;
    bad4.valid[100] = static_cast<std::uint8_t>(1u - bad4.valid[100]);
    const std::string d3 = diff_roundtrip(bufn, bad4, true);
    EXPECT_NE(d3.find("row 100"), std::string::npos) << d3;
}

// ---- interop fixtures ----------------------------------------------------

TEST(BoltParquetWriteDict, InteropFixtures) {
    // Files for scripts/parquet_write_interop.py to read with pyarrow. A
    // bolt-to-bolt round trip proves self-consistency, not interoperability:
    // a writer and reader that share a misreading of the spec agree with each
    // other perfectly. Only a reference implementation settles that.
    struct Case { const char* tag; bool dict; bool idx; bool nullable;
                  std::uint32_t page; std::uint8_t codec; std::int64_t distinct; };
    const Case cases[] = {
        {"interop_dict_snappy",   true,  true,  false, 4096, 1, 50},
        {"interop_dict_plain",    true,  false, false, 0,    0, 7},
        {"interop_dict_nullable", true,  true,  true,  4096, 1, 13},
        {"interop_split_plain",   false, true,  false, 4096, 0, 0},
        {"interop_dict_bw0",      true,  true,  false, 4096, 0, 1},
    };
    for (const Case& c : cases) {
        const Model m = make_model(8000, c.distinct, c.nullable ? 5 : 0);
        const auto o = make_opts(c.nullable, c.dict, c.idx, c.page, 0, c.codec);
        const auto buf = write_model(m, o, c.nullable, c.tag);
        SCOPED_TRACE(c.tag);
        expect_roundtrip(buf, m, c.nullable);
    }
}

}  // namespace
