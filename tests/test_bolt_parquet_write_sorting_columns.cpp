// G2PQ-26 (spec item B6): RowGroup.sorting_columns (field 4).
//
// The spec's own words: "Only legal to write when actually sorted" -- so
// this writer never trusts a caller's claim, it VERIFIES it against the row
// group's own data before emitting the field (ParquetWriteOpts::
// sorting_columns / verify_sort_claim in bolt_parquet_write.cpp). A claim
// that does not hold fails the whole parquet_write_row_group call rather
// than being silently dropped -- the single most important behaviour this
// file tests is that a FALSE claim is REJECTED, not merely that a true one
// round-trips.
//
// pyarrow DOES expose RowGroupMetaData.sorting_columns (confirmed by direct
// inspection: pyarrow 21.0.0's ParquetFile(...).metadata.row_group(i).
// sorting_columns), so it is the oracle for the positive cases -- three
// fixtures are written here and cross-checked by
// scripts/parquet_sorting_columns_check.py. The negative (false-claim)
// cases never produce a file at all (parquet_write_row_group returns
// false), so they are asserted purely in this gtest binary.

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
    return std::string("test_bolt_parquet_write_sorting_columns_") + tag +
           ".parquet";
}

void attach_validity(bolt::BoltColumn* col, bolt::Arena* arena,
                     const std::vector<std::uint8_t>& valid) {
    const std::size_t n = valid.size();
    auto* bm = static_cast<std::uint8_t*>(arena->allocate((n + 7u) / 8u, 8));
    std::memset(bm, 0, (n + 7u) / 8u);
    for (std::size_t i = 0; i < n; ++i) {
        if (valid[i]) {
            bm[i >> 3] = static_cast<std::uint8_t>(bm[i >> 3] | (1u << (i & 7u)));
        }
    }
    col->validity = bm;
    col->validity_offset = 0;
    col->stats.all_valid = false;
}

void build_int64_batch(bolt::Arena* arena, const std::vector<std::int64_t>& vals,
                       const std::vector<std::uint8_t>* valid,
                       bolt::BoltBatch* out) {
    const std::int64_t n = static_cast<std::int64_t>(vals.size());
    bolt::BoltBatch::init_empty(out);
    out->num_cols = 1;
    out->num_rows = n;
    bolt::BoltBatch::alloc_columns(out, arena, 1);
    out->schema.add_field("v", bolt::BoltType::Int64, valid != nullptr);
    bolt::BoltColumn& c = out->columns[out->read_epoch][0];
    c = bolt::BoltColumn::make_flat_alloc(n, bolt::BoltType::Int64, arena);
    ASSERT_NE(c.data, nullptr);
    auto* p = static_cast<std::int64_t*>(c.data);
    for (std::int64_t i = 0; i < n; ++i) {
        p[i] = vals[static_cast<std::size_t>(i)];
    }
    if (valid != nullptr) attach_validity(&c, arena, *valid);
}

void build_str_batch(bolt::Arena* arena, const std::vector<std::string>& vals,
                     const std::vector<std::uint8_t>* valid,
                     bolt::BoltBatch* out) {
    const std::int64_t n = static_cast<std::int64_t>(vals.size());
    bolt::BoltBatch::init_empty(out);
    out->num_cols = 1;
    out->num_rows = n;
    bolt::BoltBatch::alloc_columns(out, arena, 1);
    out->schema.add_field("v", bolt::BoltType::Utf8, valid != nullptr);
    bolt::BoltColumn& c = out->columns[out->read_epoch][0];
    c = bolt::BoltColumn::make_empty();
    c.length = n;
    c.format = bolt::ColumnFormat::Flat;
    c.type = bolt::BoltType::Utf8;
    c.type_size_bytes = sizeof(bolt::StringView);
    auto* svs = static_cast<bolt::StringView*>(
        arena->allocate(static_cast<std::size_t>(n) * sizeof(bolt::StringView),
                        alignof(bolt::StringView)));
    std::memset(svs, 0, static_cast<std::size_t>(n) * sizeof(bolt::StringView));
    std::size_t spill_need = 0;
    for (const auto& s : vals) if (s.size() > 12u) spill_need += s.size();
    auto* spill = (spill_need > 0)
        ? static_cast<std::uint8_t*>(arena->allocate(spill_need, 8)) : nullptr;
    std::size_t spill_off = 0;
    for (std::int64_t i = 0; i < n; ++i) {
        const std::string& s = vals[static_cast<std::size_t>(i)];
        svs[i].length = static_cast<std::uint32_t>(s.size());
        if (s.size() <= 12u) {
            std::memcpy(&svs[i].prefix[0], s.data(), s.size());
        } else {
            std::memcpy(&svs[i].prefix[0], s.data(), 4);
            std::memcpy(spill + spill_off, s.data(), s.size());
            svs[i].ref.offset = static_cast<std::uint32_t>(spill_off);
            spill_off += s.size();
        }
    }
    c.data = svs;
    c.str_overflow_base = spill;
    c.stats.all_valid = true;
    if (valid != nullptr) attach_validity(&c, arena, *valid);
}

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

// ===== positive: a genuinely sorted column is verified and emitted ========

TEST(BoltParquetSortingColumns, AscendingInt64AcceptedAndRoundTrips) {
    bolt::Arena arena;
    bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(batch, nullptr);
    build_int64_batch(&arena, {1, 3, 3, 7, 20, 20, 99}, nullptr, batch);

    ParquetWriteOpts opts{};
    opts.n_columns = 1;
    opts.compression = 0;
    std::strncpy(opts.columns[0].name, "v", sizeof(opts.columns[0].name));
    opts.columns[0].type = bolt::BoltType::Int64;
    opts.columns[0].nullable = false;
    opts.n_sorting_columns = 1;
    opts.sorting_columns[0] = ParquetSortingColumn{0, false, false};

    const std::string path = tmp_path("asc");
    ParquetWriter* w = parquet_write_open(path.c_str(), &opts);
    ASSERT_NE(w, nullptr);
    // The load-bearing assertion: a TRUE claim must not be refused.
    ASSERT_TRUE(parquet_write_row_group(w, batch));
    ASSERT_TRUE(parquet_write_close(w));

    const auto buf = slurp_file(path.c_str());
    ASSERT_FALSE(buf.empty());
    PqMeta m{};
    std::vector<PqChunk> chunks;
    parse_footer(buf, &m, &chunks);
    ASSERT_EQ(m.n_row_groups, 1u);
    const PqRowGroup& rg = m.row_groups[0];
    ASSERT_EQ(rg.n_sorting_columns, 1u);
    EXPECT_EQ(rg.sorting_columns[0].column_idx, 0);
    EXPECT_FALSE(rg.sorting_columns[0].descending);
    EXPECT_FALSE(rg.sorting_columns[0].nulls_first);
}

TEST(BoltParquetSortingColumns, DescendingInt64AcceptedAndRoundTrips) {
    bolt::Arena arena;
    bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(batch, nullptr);
    build_int64_batch(&arena, {99, 50, 50, 7, 3, -8}, nullptr, batch);

    ParquetWriteOpts opts{};
    opts.n_columns = 1;
    opts.compression = 0;
    std::strncpy(opts.columns[0].name, "v", sizeof(opts.columns[0].name));
    opts.columns[0].type = bolt::BoltType::Int64;
    opts.columns[0].nullable = false;
    opts.n_sorting_columns = 1;
    opts.sorting_columns[0] = ParquetSortingColumn{0, true, false};

    const std::string path = tmp_path("desc");
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
    ASSERT_EQ(rg.n_sorting_columns, 1u);
    EXPECT_EQ(rg.sorting_columns[0].column_idx, 0);
    EXPECT_TRUE(rg.sorting_columns[0].descending);
    EXPECT_FALSE(rg.sorting_columns[0].nulls_first);
}

TEST(BoltParquetSortingColumns, NullsFirstAcceptedAndRoundTrips) {
    bolt::Arena arena;
    bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(batch, nullptr);
    // Nulls (marked invalid below) sit first, then an ascending run.
    const std::vector<std::int64_t> vals   = {0, 0, 0, 1, 5, 9};
    const std::vector<std::uint8_t> valid  = {0, 0, 0, 1, 1, 1};
    build_int64_batch(&arena, vals, &valid, batch);

    ParquetWriteOpts opts{};
    opts.n_columns = 1;
    opts.compression = 0;
    std::strncpy(opts.columns[0].name, "v", sizeof(opts.columns[0].name));
    opts.columns[0].type = bolt::BoltType::Int64;
    opts.columns[0].nullable = true;
    opts.n_sorting_columns = 1;
    opts.sorting_columns[0] = ParquetSortingColumn{0, false, true};

    const std::string path = tmp_path("nullsfirst");
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
    ASSERT_EQ(rg.n_sorting_columns, 1u);
    EXPECT_EQ(rg.sorting_columns[0].column_idx, 0);
    EXPECT_FALSE(rg.sorting_columns[0].descending);
    EXPECT_TRUE(rg.sorting_columns[0].nulls_first);
}

TEST(BoltParquetSortingColumns, Utf8AscendingAccepted) {
    bolt::Arena arena;
    bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(batch, nullptr);
    build_str_batch(&arena, {"alpha", "beta", "beta", "gamma",
                             "this-string-is-longer-than-twelve-bytes"},
                    nullptr, batch);

    ParquetWriteOpts opts{};
    opts.n_columns = 1;
    opts.compression = 0;
    std::strncpy(opts.columns[0].name, "v", sizeof(opts.columns[0].name));
    opts.columns[0].type = bolt::BoltType::Utf8;
    opts.columns[0].nullable = false;
    opts.n_sorting_columns = 1;
    opts.sorting_columns[0] = ParquetSortingColumn{0, false, false};

    const std::string path = tmp_path("utf8_asc");
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
    EXPECT_EQ(m.row_groups[0].n_sorting_columns, 1u);
}

// ===== the most important tests: a FALSE claim must be REJECTED ===========

TEST(BoltParquetSortingColumns, UnsortedInt64ClaimRejected) {
    bolt::Arena arena;
    bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(batch, nullptr);
    // One inversion (7 then 3) is enough to disprove "ascending".
    build_int64_batch(&arena, {1, 2, 7, 3, 9}, nullptr, batch);

    ParquetWriteOpts opts{};
    opts.n_columns = 1;
    opts.compression = 0;
    std::strncpy(opts.columns[0].name, "v", sizeof(opts.columns[0].name));
    opts.columns[0].type = bolt::BoltType::Int64;
    opts.columns[0].nullable = false;
    opts.n_sorting_columns = 1;
    opts.sorting_columns[0] = ParquetSortingColumn{0, false, false};

    const std::string path = tmp_path("unsorted_int64");
    ParquetWriter* w = parquet_write_open(path.c_str(), &opts);
    ASSERT_NE(w, nullptr);
    // THE key assertion in this ticket: a claim that does not hold must be
    // refused loudly, not silently dropped while the row group is written
    // on with no sorting_columns entry.
    EXPECT_FALSE(parquet_write_row_group(w, batch));
    // The writer is left in an unspecified state after a failed append (the
    // documented contract) -- close is still safe to call, just not
    // meaningful to assert on further.
    parquet_write_close(w);
}

TEST(BoltParquetSortingColumns, WrongDirectionClaimRejected) {
    // Genuinely ascending data, but the claim says descending.
    bolt::Arena arena;
    bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(batch, nullptr);
    build_int64_batch(&arena, {1, 2, 3, 4, 5}, nullptr, batch);

    ParquetWriteOpts opts{};
    opts.n_columns = 1;
    opts.compression = 0;
    std::strncpy(opts.columns[0].name, "v", sizeof(opts.columns[0].name));
    opts.columns[0].type = bolt::BoltType::Int64;
    opts.columns[0].nullable = false;
    opts.n_sorting_columns = 1;
    opts.sorting_columns[0] = ParquetSortingColumn{0, /*descending=*/true, false};

    const std::string path = tmp_path("wrong_direction");
    ParquetWriter* w = parquet_write_open(path.c_str(), &opts);
    ASSERT_NE(w, nullptr);
    EXPECT_FALSE(parquet_write_row_group(w, batch));
    parquet_write_close(w);
}

TEST(BoltParquetSortingColumns, Utf8UnsortedClaimRejected) {
    bolt::Arena arena;
    bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(batch, nullptr);
    build_str_batch(&arena, {"alpha", "zeta", "beta"}, nullptr, batch);

    ParquetWriteOpts opts{};
    opts.n_columns = 1;
    opts.compression = 0;
    std::strncpy(opts.columns[0].name, "v", sizeof(opts.columns[0].name));
    opts.columns[0].type = bolt::BoltType::Utf8;
    opts.columns[0].nullable = false;
    opts.n_sorting_columns = 1;
    opts.sorting_columns[0] = ParquetSortingColumn{0, false, false};

    const std::string path = tmp_path("utf8_unsorted");
    ParquetWriter* w = parquet_write_open(path.c_str(), &opts);
    ASSERT_NE(w, nullptr);
    EXPECT_FALSE(parquet_write_row_group(w, batch));
    parquet_write_close(w);
}

TEST(BoltParquetSortingColumns, NullsFirstViolationRejected) {
    // A null AFTER a non-null violates nulls_first=true even though the
    // non-null values themselves are perfectly ascending.
    bolt::Arena arena;
    bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(batch, nullptr);
    const std::vector<std::int64_t> vals  = {0, 1, 0, 5, 9};
    const std::vector<std::uint8_t> valid = {0, 1, 0, 1, 1};
    build_int64_batch(&arena, vals, &valid, batch);

    ParquetWriteOpts opts{};
    opts.n_columns = 1;
    opts.compression = 0;
    std::strncpy(opts.columns[0].name, "v", sizeof(opts.columns[0].name));
    opts.columns[0].type = bolt::BoltType::Int64;
    opts.columns[0].nullable = true;
    opts.n_sorting_columns = 1;
    opts.sorting_columns[0] = ParquetSortingColumn{0, false, true};

    const std::string path = tmp_path("nullsfirst_violation");
    ParquetWriter* w = parquet_write_open(path.c_str(), &opts);
    ASSERT_NE(w, nullptr);
    EXPECT_FALSE(parquet_write_row_group(w, batch));
    parquet_write_close(w);
}

TEST(BoltParquetSortingColumns, NullsLastViolationRejected) {
    // nulls_first=false (nulls LAST): a non-null after a null is a
    // violation even though the non-null run itself is ascending.
    bolt::Arena arena;
    bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(batch, nullptr);
    const std::vector<std::int64_t> vals  = {1, 5, 0, 9};
    const std::vector<std::uint8_t> valid = {1, 1, 0, 1};
    build_int64_batch(&arena, vals, &valid, batch);

    ParquetWriteOpts opts{};
    opts.n_columns = 1;
    opts.compression = 0;
    std::strncpy(opts.columns[0].name, "v", sizeof(opts.columns[0].name));
    opts.columns[0].type = bolt::BoltType::Int64;
    opts.columns[0].nullable = true;
    opts.n_sorting_columns = 1;
    opts.sorting_columns[0] = ParquetSortingColumn{0, false, false};

    const std::string path = tmp_path("nullslast_violation");
    ParquetWriter* w = parquet_write_open(path.c_str(), &opts);
    ASSERT_NE(w, nullptr);
    EXPECT_FALSE(parquet_write_row_group(w, batch));
    parquet_write_close(w);
}

// Removing the verification call turns this same fixture into a false
// PASS -- see the injection test procedure in the commit message. This test
// exists so that procedure has a single, minimal, always-run case to gate
// on: it is the discriminator between "verification exists" and
// "verification is decorative".
TEST(BoltParquetSortingColumns, RejectionIsLoadBearingNotDecorative) {
    bolt::Arena arena;
    bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(batch, nullptr);
    build_int64_batch(&arena, {5, 4, 3, 2, 1}, nullptr, batch);   // descending

    ParquetWriteOpts opts{};
    opts.n_columns = 1;
    opts.compression = 0;
    std::strncpy(opts.columns[0].name, "v", sizeof(opts.columns[0].name));
    opts.columns[0].type = bolt::BoltType::Int64;
    opts.columns[0].nullable = false;
    opts.n_sorting_columns = 1;
    // Claim ASCENDING over data that is actually strictly descending.
    opts.sorting_columns[0] = ParquetSortingColumn{0, false, false};

    const std::string path = tmp_path("load_bearing");
    ParquetWriter* w = parquet_write_open(path.c_str(), &opts);
    ASSERT_NE(w, nullptr);
    ASSERT_FALSE(parquet_write_row_group(w, batch));
    parquet_write_close(w);
}

// ===== open-time validation: an unverifiable claim never reaches a row
//        group append at all ==============================================

TEST(BoltParquetSortingColumns, OpenRejectsOutOfRangeColumnIdx) {
    ParquetWriteOpts opts{};
    opts.n_columns = 1;
    opts.compression = 0;
    std::strncpy(opts.columns[0].name, "v", sizeof(opts.columns[0].name));
    opts.columns[0].type = bolt::BoltType::Int64;
    opts.n_sorting_columns = 1;
    opts.sorting_columns[0] = ParquetSortingColumn{1, false, false};  // OOB

    const std::string path = tmp_path("bad_column_idx");
    ParquetWriter* w = parquet_write_open(path.c_str(), &opts);
    EXPECT_EQ(w, nullptr);
}

TEST(BoltParquetSortingColumns, OpenRejectsUnverifiableDecimal128Type) {
    ParquetWriteOpts opts{};
    opts.n_columns = 1;
    opts.compression = 0;
    std::strncpy(opts.columns[0].name, "v", sizeof(opts.columns[0].name));
    opts.columns[0].type = bolt::BoltType::Decimal128;
    opts.columns[0].precision = 10;
    opts.columns[0].scale = 2;
    opts.n_sorting_columns = 1;
    opts.sorting_columns[0] = ParquetSortingColumn{0, false, false};

    const std::string path = tmp_path("bad_type");
    ParquetWriter* w = parquet_write_open(path.c_str(), &opts);
    EXPECT_EQ(w, nullptr);
}

TEST(BoltParquetSortingColumns, OpenAcceptsMemSinkSameValidation) {
    // parquet_write_open_mem runs the identical sorting_columns_valid gate.
    ParquetWriteOpts opts{};
    opts.n_columns = 1;
    opts.compression = 0;
    std::strncpy(opts.columns[0].name, "v", sizeof(opts.columns[0].name));
    opts.columns[0].type = bolt::BoltType::Int64;
    opts.n_sorting_columns = 1;
    opts.sorting_columns[0] = ParquetSortingColumn{5, false, false};  // OOB

    ParquetWriter* w = parquet_write_open_mem(&opts, 0);
    EXPECT_EQ(w, nullptr);
}

// ===== baseline: no claim means no field at all ============================

TEST(BoltParquetSortingColumns, NoClaimEmitsNoSortingColumnsField) {
    bolt::Arena arena;
    bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(batch, nullptr);
    build_int64_batch(&arena, {3, 1, 4, 1, 5}, nullptr, batch);  // unsorted

    ParquetWriteOpts opts{};
    opts.n_columns = 1;
    opts.compression = 0;
    std::strncpy(opts.columns[0].name, "v", sizeof(opts.columns[0].name));
    opts.columns[0].type = bolt::BoltType::Int64;
    opts.columns[0].nullable = false;
    // n_sorting_columns left at its zero-init default: no claim at all.

    const std::string path = tmp_path("no_claim");
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
    EXPECT_EQ(m.row_groups[0].n_sorting_columns, 0u);
}

// ===== per-row-group verification under row_group_max_rows splitting ======

TEST(BoltParquetSortingColumns, PartiallySortedBatchAcrossSplitRowGroupsRejected) {
    // row_group_max_rows=3 splits this ONE call into two row groups:
    // [10,20,30] (sorted) and [5,25,15] (NOT sorted). The first row group's
    // bytes may already be written by the time the second fails -- the
    // CONTRACT is that the whole call reports failure, not that no bytes
    // were touched (see the header comment on parquet_write_row_group).
    bolt::Arena arena;
    bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(batch, nullptr);
    build_int64_batch(&arena, {10, 20, 30, 5, 25, 15}, nullptr, batch);

    ParquetWriteOpts opts{};
    opts.n_columns = 1;
    opts.compression = 0;
    opts.row_group_max_rows = 3;
    std::strncpy(opts.columns[0].name, "v", sizeof(opts.columns[0].name));
    opts.columns[0].type = bolt::BoltType::Int64;
    opts.columns[0].nullable = false;
    opts.n_sorting_columns = 1;
    opts.sorting_columns[0] = ParquetSortingColumn{0, false, false};

    const std::string path = tmp_path("partial_split");
    ParquetWriter* w = parquet_write_open(path.c_str(), &opts);
    ASSERT_NE(w, nullptr);
    EXPECT_FALSE(parquet_write_row_group(w, batch));
    parquet_write_close(w);
}

TEST(BoltParquetSortingColumns, FullySortedBatchAcrossSplitRowGroupsAllVerified) {
    bolt::Arena arena;
    bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(batch, nullptr);
    build_int64_batch(&arena, {1, 2, 3, 4, 5, 6, 7}, nullptr, batch);

    ParquetWriteOpts opts{};
    opts.n_columns = 1;
    opts.compression = 0;
    opts.row_group_max_rows = 3;   // -> groups of 3, 3, 1
    std::strncpy(opts.columns[0].name, "v", sizeof(opts.columns[0].name));
    opts.columns[0].type = bolt::BoltType::Int64;
    opts.columns[0].nullable = false;
    opts.n_sorting_columns = 1;
    opts.sorting_columns[0] = ParquetSortingColumn{0, false, false};

    const std::string path = tmp_path("full_split");
    ParquetWriter* w = parquet_write_open(path.c_str(), &opts);
    ASSERT_NE(w, nullptr);
    ASSERT_TRUE(parquet_write_row_group(w, batch));
    ASSERT_TRUE(parquet_write_close(w));

    const auto buf = slurp_file(path.c_str());
    ASSERT_FALSE(buf.empty());
    PqMeta m{};
    std::vector<PqChunk> chunks;
    parse_footer(buf, &m, &chunks);
    ASSERT_EQ(m.n_row_groups, 3u);
    for (std::uint32_t g = 0; g < m.n_row_groups; ++g) {
        EXPECT_EQ(m.row_groups[g].n_sorting_columns, 1u) << "row group " << g;
    }
}
