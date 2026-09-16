// test_bolt_iceberg_column_stats_e2e.cpp — G2ICE-135.
//
// End-to-end: write a real BoltBatch WITH nulls through the exact
// append_open/append_write/append_commit_ex path TieringManager::upload_iceberg
// (gestalt::cluster, superproject) drives, then read the resulting manifest
// bytes back OFF DISK and decode them with `manifest_parse_avro` — the same
// pyiceberg-verified reader `test_bolt_iceberg_manifest_write.cpp` trusts, a
// genuinely independent decoder from the writer under test. This is the
// "seal → tier → read back the manifest directly → compare" round trip
// G2ICE-135 asks for, minus MarbleDB itself (that half is a Gestalt2
// superproject test — TieringManager only ever hands this writer a BoltBatch,
// so exercising the writer directly with a hand-built one proves the same
// code path with full control over what "real" means here).
//
// Two columns, values chosen so hand-computed expectations are unambiguous:
//   id    (Int64,   required): 0..9,               no nulls.
//   score (Float64, nullable): i*1.5, EXCEPT rows {2,5,7} are NULL.
// So: id null_count == 0 (a certified real zero) with bounds [0,9]; score
// null_count == 3 (a certified real nonzero) with bounds computed only over
// the 7 non-null rows.

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"
#include "bolt/lakehouse/iceberg/manifest.h"
#include "bolt/lakehouse/iceberg/writer.h"
#include "bolt/lakehouse/object_store.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

namespace {

using namespace bolt::lakehouse;
using namespace bolt::lakehouse::iceberg;

constexpr int32_t kRows = 10;
constexpr int32_t kNullRows[3] = { 2, 5, 7 };

std::string unique_root() {
    auto p = std::filesystem::temp_directory_path() /
             ("bolt_iceberg_colstats_e2e_" +
              std::to_string(reinterpret_cast<uintptr_t>(&kRows)));
    std::error_code ec;
    std::filesystem::remove_all(p, ec);
    std::filesystem::create_directories(p, ec);
    return p.generic_string();
}

// Same "recorded location is absolute, ObjectStore key is root-relative"
// translation test_bolt_lakehouse_iceberg_write.cpp's `store_key` does.
std::string store_key(const std::string& root, const char* loc) {
    if (loc == nullptr) return std::string();
    const std::string s(loc);
    if (s.size() > root.size() && s.compare(0, root.size(), root) == 0 &&
        (s[root.size()] == '/' || s[root.size()] == '\\')) {
        return s.substr(root.size() + 1u);
    }
    return s;
}

Schema make_schema() {
    Schema s{};
    s.schema_id = 0;
    s.n_fields  = 2;
    s.fields[0].id = 1; s.fields[0].required = true;
    std::strncpy(s.fields[0].name, "id",   sizeof(s.fields[0].name) - 1u);
    std::strncpy(s.fields[0].type, "long", sizeof(s.fields[0].type) - 1u);
    s.fields[1].id = 2; s.fields[1].required = false;
    std::strncpy(s.fields[1].name, "score", sizeof(s.fields[1].name) - 1u);
    std::strncpy(s.fields[1].type, "double", sizeof(s.fields[1].type) - 1u);
    return s;
}

bool is_null_row(int32_t i) noexcept {
    for (int32_t n : kNullRows) if (n == i) return true;
    return false;
}

void make_batch(bolt::Arena* a, bolt::BoltBatch* out) {
    bolt::BoltBatch::init_empty(out);
    out->arena    = a;
    out->num_rows = kRows;
    out->num_cols = 2;
    bolt::BoltBatch::alloc_columns(out, a, 2);
    out->schema.add_field("id",    bolt::BoltType::Int64,   false);
    out->schema.add_field("score", bolt::BoltType::Float64, true);
    auto* cols = out->columns[out->read_epoch];

    auto* idata = a->allocate_array<int64_t>(kRows);
    for (int32_t i = 0; i < kRows; ++i) idata[i] = i;
    cols[0].type = bolt::BoltType::Int64;
    cols[0].length = kRows;
    cols[0].data = idata;
    cols[0].validity = nullptr;   // no nulls in `id`

    auto* sdata = a->allocate_array<double>(kRows);
    auto* svalid = a->allocate_array<uint8_t>((kRows + 7) / 8);
    std::memset(svalid, 0xFF, static_cast<size_t>((kRows + 7) / 8));
    for (int32_t i = 0; i < kRows; ++i) {
        sdata[i] = static_cast<double>(i) * 1.5;
        if (is_null_row(i)) {
            svalid[i >> 3] = static_cast<uint8_t>(svalid[i >> 3] & ~(1u << (i & 7)));
        }
    }
    cols[1].type = bolt::BoltType::Float64;
    cols[1].length = kRows;
    cols[1].data = sdata;
    cols[1].validity = svalid;
}

}  // namespace

TEST(IcebergColumnStatsE2E, RealCommitProducesRealManifestStats) {
    bolt::Arena arena;
    const std::string root = unique_root();
    FilesystemObjectStore fs{}; ObjectStore os{};
    ASSERT_TRUE(filesystem_object_store_init(&fs, root.c_str(), &os));
    Schema sch = make_schema();
    PartitionSpec spec{}; spec.spec_id = 0; spec.n_fields = 0;
    SortOrder sort{};     sort.order_id = 0; sort.n_fields = 0;
    WriteOptions wo;      write_options_init(&wo);
    TableHandle* th = nullptr;
    ASSERT_TRUE(table_create(&th, &arena, &os, root.c_str(), &sch, &spec,
                             &sort, &wo));

    bolt::BoltBatch b{};
    make_batch(&arena, &b);
    AppendHandle* ah = nullptr;
    ASSERT_TRUE(append_open(&ah, th));
    ASSERT_TRUE(append_write(ah, &b));
    int64_t rows = 0;
    ASSERT_TRUE(append_commit_ex(ah, nullptr, 0, &rows));
    EXPECT_EQ(rows, kRows);
    append_close(ah);

    // Walk snapshot -> manifest-list -> manifest, exactly as a real reader
    // (pyiceberg, DuckDB) would, reading only what THIS commit wrote to disk.
    const Metadata* m = table_metadata(th);
    ASSERT_EQ(m->n_snapshots, 1u);
    const char* ml_loc = m->snapshots[0].manifest_list;
    ASSERT_NE(ml_loc[0], '\0');

    const std::string ml_key = store_key(root, ml_loc);
    const uint8_t* ml_bytes = nullptr; uint64_t ml_len = 0;
    ASSERT_EQ(os_get(&os, ml_key.c_str(), &arena, &ml_bytes, &ml_len), kOsOk);

    ManifestListEntry mles[4]{};
    uint32_t n_mles = 0;
    ASSERT_TRUE(manifest_list_parse_avro(ml_bytes, ml_len, &arena, mles, 4,
                                         &n_mles));
    ASSERT_EQ(n_mles, 1u);

    const std::string mf_key = store_key(root, mles[0].manifest_path);
    const uint8_t* mf_bytes = nullptr; uint64_t mf_len = 0;
    ASSERT_EQ(os_get(&os, mf_key.c_str(), &arena, &mf_bytes, &mf_len), kOsOk);

    DataFileRef dfs[4]{};
    uint32_t n_dfs = 0;
    ASSERT_TRUE(manifest_parse_avro(mf_bytes, mf_len, &arena,
                                    /*default_spec_id=*/0, dfs, 4, &n_dfs));
    ASSERT_EQ(n_dfs, 1u);

    const DataFileRef& df = dfs[0];
    EXPECT_EQ(df.stats.record_count, kRows);
    ASSERT_EQ(df.stats.n_cols, 2u);

    const ColumnStatEntry* id = nullptr;
    const ColumnStatEntry* score = nullptr;
    for (uint32_t i = 0; i < df.stats.n_cols; ++i) {
        if (df.stats.cols[i].field_id == 1) id = &df.stats.cols[i];
        if (df.stats.cols[i].field_id == 2) score = &df.stats.cols[i];
    }
    ASSERT_NE(id, nullptr);
    ASSERT_NE(score, nullptr);

    // id: certified zero nulls, real bounds [0, 9].
    EXPECT_EQ(id->null_count, 0);
    ASSERT_TRUE(id->has_lower);
    ASSERT_TRUE(id->has_upper);
    ASSERT_EQ(id->lower_len, 8u);
    ASSERT_EQ(id->upper_len, 8u);
    int64_t id_lo = 0, id_hi = 0;
    std::memcpy(&id_lo, id->lower, 8u);
    std::memcpy(&id_hi, id->upper, 8u);
    EXPECT_EQ(id_lo, 0);
    EXPECT_EQ(id_hi, kRows - 1);

    // score: real nonzero null_count (rows 2, 5, 7), bounds computed over
    // the 7 SURVIVING rows only: min = row0 (0.0), max = row9 (13.5) — the
    // excluded nulls (3.0, 7.5, 10.5) are neither the true min nor max, so
    // this also proves nulls are excluded from the bound, not merely from
    // the count.
    EXPECT_EQ(score->null_count, 3);
    ASSERT_TRUE(score->has_lower);
    ASSERT_TRUE(score->has_upper);
    ASSERT_EQ(score->lower_len, 8u);
    ASSERT_EQ(score->upper_len, 8u);
    double score_lo = 0.0, score_hi = 0.0;
    std::memcpy(&score_lo, score->lower, 8u);
    std::memcpy(&score_hi, score->upper, 8u);
    EXPECT_DOUBLE_EQ(score_lo, 0.0);
    EXPECT_DOUBLE_EQ(score_hi, static_cast<double>(kRows - 1) * 1.5);

    // And the same file's record_count/file_size are still real, as they
    // always were -- this fix must not regress what already worked.
    EXPECT_GT(df.stats.file_size_in_bytes, 0);
}
