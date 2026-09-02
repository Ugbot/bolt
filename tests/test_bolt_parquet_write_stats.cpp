// Writer chunk-level Statistics fixtures (G2PQ-21 / G2PQ-22).
//
// The stats a writer emits are consumed by OTHER readers' pruning: get the
// ordering or the NaN rules wrong and it is DuckDB / pyarrow / parquet-mr
// that silently drop matching rows, while bolt's own round-trip stays green.
// So this test only WRITES the fixture; the assertions that matter live in
// scripts/parquet_stats_check.py, which decodes the chunk Statistics and the
// FileMetaData column_orders with a from-spec thrift compact reader (never
// bolt's parser) and checks:
//
//   * BYTE_ARRAY min/max use UNSIGNED byte-wise ordering, on values with
//     bytes >= 0x80 where the signed and unsigned orderings DIFFER --
//     the exact case an ASCII-only fixture can never catch (G2PQ-21);
//   * no NaN ever appears as a min or max; an all-NaN chunk OMITS the
//     statistic rather than writing one (G2PQ-22);
//   * signed zeros are conservative: a zero min is written -0.0 and a zero
//     max +0.0, so both zeros lie inside [min, max] (G2PQ-22);
//   * +/-inf are legal, exact bounds;
//   * deprecated Statistics.min/max (fields 1/2, whose ordering semantics
//     differ) are ABSENT, never written with modern semantics;
//   * column_orders declares one TypeDefinedOrder per column.
//
// The gtest body asserts only that the fixture writes successfully; run the
// script for the independent proof.

#include "bolt/ingest/bolt_parquet_write.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"

namespace {

using namespace bolt::ingest::parquet;

void put_f64_col(bolt::Arena* a, bolt::BoltBatch* b, std::uint16_t slot,
                 const char* name, const std::vector<double>& v) {
    b->schema.add_field(name, bolt::BoltType::Float64, false);
    bolt::BoltColumn& c = b->columns[b->read_epoch][slot];
    c = bolt::BoltColumn::make_flat_alloc(
        static_cast<std::int64_t>(v.size()), bolt::BoltType::Float64, a);
    std::memcpy(c.data, v.data(), v.size() * 8u);
}

void put_str_col(bolt::Arena* a, bolt::BoltBatch* b, std::uint16_t slot,
                 const char* name, const std::vector<std::string>& vals) {
    // Binary, not Utf8: the discriminating values carry bytes >= 0x80 that
    // are not valid UTF-8, and a String-annotated column of invalid UTF-8
    // would make the fixture itself spec-illegal. BYTE_ARRAY ordering rules
    // are identical either way (unsigned byte-wise).
    b->schema.add_field(name, bolt::BoltType::Binary, false);
    bolt::BoltColumn& c = b->columns[b->read_epoch][slot];
    c = bolt::BoltColumn::make_empty();
    c.length = static_cast<std::int64_t>(vals.size());
    c.format = bolt::ColumnFormat::Flat;
    c.type = bolt::BoltType::Binary;
    c.type_size_bytes = sizeof(bolt::StringView);
    auto* svs = static_cast<bolt::StringView*>(
        a->allocate(vals.size() * sizeof(bolt::StringView),
                    alignof(bolt::StringView)));
    ASSERT_NE(svs, nullptr);
    std::memset(svs, 0, vals.size() * sizeof(bolt::StringView));
    for (std::size_t i = 0; i < vals.size(); ++i) {
        ASSERT_LE(vals[i].size(), 12u) << "keep fixture strings inline";
        svs[i].length = static_cast<std::uint32_t>(vals[i].size());
        std::memcpy(&svs[i].prefix[0], vals[i].data(), vals[i].size());
    }
    c.data = svs;
    c.stats.all_valid = true;
}

TEST(BoltParquetWriteStats, WritesStatisticsFixture) {
    const double qnan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();

    ParquetWriteOpts o{};
    o.n_columns = 5;
    o.compression = 0;             // UNCOMPRESSED: the script reads pages raw
    o.emit_statistics = true;
    o.emit_page_index = true;      // per-page bounds for the all-NaN page case
    o.use_dictionary = false;
    auto col = [&](std::uint32_t i, const char* n, bolt::BoltType t) {
        std::snprintf(o.columns[i].name, sizeof(o.columns[i].name), "%s", n);
        o.columns[i].type = t;
        o.columns[i].nullable = false;
    };
    col(0, "b", bolt::BoltType::Binary);
    col(1, "d_mixed", bolt::BoltType::Float64);
    col(2, "d_allnan", bolt::BoltType::Float64);
    col(3, "d_zero", bolt::BoltType::Float64);
    col(4, "d_inf", bolt::BoltType::Float64);

    bolt::Arena a;
    bolt::BoltBatch* b = a.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(b, nullptr);
    bolt::BoltBatch::init_empty(b);
    b->num_cols = 5;
    b->num_rows = 5;
    bolt::BoltBatch::alloc_columns(b, &a, 5);

    // Bytes >= 0x80: unsigned order says min="\x01", max="\xff\x01"; a
    // signed byte comparison would instead pick min="\x80" and max="\x7f".
    // Every one of the four bounds differs between the two orderings.
    put_str_col(&a, b, 0, "b",
                {std::string("\x01", 1), "A", std::string("\x7f", 1),
                 std::string("\x80", 1), std::string("\xff\x01", 2)});
    put_f64_col(&a, b, 1, "d_mixed", {qnan, 1.5, -2.5, qnan, 3.0});
    put_f64_col(&a, b, 2, "d_allnan", {qnan, qnan, qnan, qnan, qnan});
    put_f64_col(&a, b, 3, "d_zero", {-0.0, +0.0, -0.0, +0.0, -0.0});
    put_f64_col(&a, b, 4, "d_inf", {-inf, 0.5, inf, 0.5, 0.5});

    ParquetWriter* w =
        parquet_write_open("test_bolt_parquet_stats_fixture.parquet", &o);
    ASSERT_NE(w, nullptr);
    ASSERT_TRUE(parquet_write_row_group(w, b));
    ASSERT_TRUE(parquet_write_close(w));
}

}  // namespace
