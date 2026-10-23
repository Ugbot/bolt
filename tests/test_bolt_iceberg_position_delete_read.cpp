// G2ICE-82 — bolt's OWN Iceberg reader applies positional deletes it wrote.
//
// test_bolt_iceberg_pyiceberg_interop.cpp already proves (via an EXTERNAL
// oracle, pyiceberg) that `table_delete_positions` writes a real, spec-
// conformant v2 positional delete: id=1007 is CONFIRMED ABSENT when pyiceberg
// scans the fixture (G2ICE-91). What that test cannot prove is whether
// bolt's OWN scan applies the same delete file -- which matters because
// chukonu's PhysicalMixedScan (the tiering delete-resurrection fix, G2ICE-82)
// reads a cold Iceberg table through THIS reader, in-process, never through
// pyiceberg.
//
// BEFORE this fix: `iceberg_scan_open` declined ANY table carrying a delete
// manifest entry at all (`TODO(W5-delete-load)` in iceberg_scan.cpp) -- safe
// (never a silent wrong answer) but the scan simply failed, which is not the
// same as reading correctly.
//
// AFTER this fix: `load_position_delete_file` loads the delete file's
// (file_path, pos) pairs and `apply_position_deletes_to_group` filters
// matching rows out of every row group as it is decoded -- so the deleted
// row is genuinely absent from bolt's own scan results, and every surviving
// row (across BOTH commits, proving the per-file position-base reset and
// the cross-commit manifest carry-forward both work) is still present.
//
// This test asserts VALUES, never row counts alone -- a row-count assertion
// cannot distinguish "the deleted row is gone" from "some other row went
// missing", which is exactly the class of bug this codebase has been burned
// by before (see test_mixed_scan_delete_resurrection.cpp's own banner).

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"
#include "bolt/lakehouse/catalog.h"
#include "bolt/lakehouse/iceberg/metadata.h"
#include "bolt/lakehouse/iceberg/scan.h"
#include "bolt/lakehouse/iceberg/writer.h"
#include "bolt/lakehouse/object_store.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

using namespace bolt::lakehouse;
using namespace bolt::lakehouse::iceberg;

// Two 6-row commits, id = 1000+i, i = 0..11. Row 7 (second commit, local
// position 1) is deleted -- deliberately in the SECOND file, so the delete
// has to coexist with a manifest carried forward from the first commit,
// exactly like the pyiceberg-verified fixture in G2ICE-91.
constexpr int32_t kRowsPerCommit = 6;
constexpr int32_t kCommits       = 2;
constexpr int32_t kDeletedRow    = 7;

std::string fixture_root() {
    std::filesystem::path p = std::filesystem::temp_directory_path() /
        "bolt_iceberg_del_read" / "default" / "t";
    std::error_code ec;
    std::filesystem::remove_all(p.parent_path().parent_path(), ec);
    std::filesystem::create_directories(p, ec);
    return p.generic_string();
}

Schema make_schema() {
    Schema s{};
    static SchemaField s_fields[8]{};  // Schema::fields is caller-owned storage
    s.fields = s_fields;
    s.cap_fields = 8u;
    s.schema_id = 0;
    s.n_fields  = 1;
    s.fields[0].id = 1; s.fields[0].required = true;
    std::strncpy(s.fields[0].name, "id",   sizeof(s.fields[0].name) - 1u);
    std::strncpy(s.fields[0].type, "long", sizeof(s.fields[0].type) - 1u);
    return s;
}

void make_batch(bolt::Arena* a, bolt::BoltBatch* out, int32_t first_row,
                int32_t n) {
    bolt::BoltBatch::init_empty(out);
    out->arena    = a;
    out->num_rows = n;
    out->num_cols = 1;
    bolt::BoltBatch::alloc_columns(out, a, 1);
    out->schema.add_field("id", bolt::BoltType::Int64, false);
    auto* cols  = out->columns[out->read_epoch];
    auto* idata = a->allocate_array<int64_t>(static_cast<uint32_t>(n));
    for (int32_t k = 0; k < n; ++k) idata[k] = 1000 + first_row + k;
    cols[0].type = bolt::BoltType::Int64; cols[0].length = n; cols[0].data = idata;
}

bool contains(const int64_t* v, uint32_t n, int64_t needle) {
    for (uint32_t i = 0; i < n; ++i) if (v[i] == needle) return true;
    return false;
}

}  // namespace

TEST(IcebergPositionDeleteRead, BoltOwnReaderExcludesDeletedRow) {
    bolt::Arena arena;
    const std::string root = fixture_root();  // <tmp>/.../default/t

    // ── write side: two commits + one positional delete ─────────────────
    FilesystemObjectStore fs{}; ObjectStore os{};
    ASSERT_TRUE(filesystem_object_store_init(&fs, root.c_str(), &os));
    Schema sch = make_schema();
    PartitionSpec spec{}; spec.spec_id = 0; spec.n_fields = 0;
    SortOrder sort{};     sort.order_id = 0; sort.n_fields = 0;
    WriteOptions wo;      write_options_init(&wo);
    TableHandle* wh = nullptr;
    ASSERT_TRUE(table_create(&wh, &arena, &os, root.c_str(), &sch, &spec,
                             &sort, &wo));

    AppendHandle* ah = nullptr;
    ASSERT_TRUE(append_open(&ah, wh));
    char commit_file[kCommits][kIcebergMaxPath];
    for (int32_t c = 0; c < kCommits; ++c) {
        bolt::BoltBatch b{};
        make_batch(&arena, &b, c * kRowsPerCommit, kRowsPerCommit);
        ASSERT_TRUE(append_write(ah, &b));
        int64_t rows = 0;
        ASSERT_TRUE(append_commit_ex(ah, commit_file[c], kIcebergMaxPath, &rows));
        ASSERT_EQ(rows, kRowsPerCommit);
    }
    append_close(ah);

    PositionDeleteEntry pd{};
    std::strncpy(pd.file_path, commit_file[kDeletedRow / kRowsPerCommit],
                sizeof(pd.file_path) - 1u);
    pd.pos = kDeletedRow % kRowsPerCommit;
    ASSERT_TRUE(table_delete_positions(wh, &pd, 1));
    table_close(wh);

    // ── read side: bolt's OWN scan, Catalog-based (the path chukonu's
    //    PhysicalMixedScan actually uses) ─────────────────────────────────
    const std::string ns_root = std::filesystem::path(root)
        .parent_path().parent_path().generic_string();  // strip "default/t"
    bolt::lakehouse::FilesystemCatalog cat_fs{};
    bolt::lakehouse::Catalog           cat{};
    ASSERT_TRUE(bolt::lakehouse::filesystem_catalog_init(&cat_fs, ns_root.c_str(), &cat));

    bolt::Arena read_arena;
    TableHandle* rh = nullptr;
    ASSERT_TRUE(iceberg_table_open(&rh, &read_arena, &cat, "default", "t"));

    ReadOptions ro{}; read_options_init(&ro);
    ScanHandle* sh = nullptr;
    // THE FIX under test: pre-fix this returned false (declined -- a table
    // with a delete manifest entry was refused outright).
    ASSERT_TRUE(iceberg_scan_open(&sh, rh, &ro))
        << "iceberg_scan_open declined a table carrying a real positional "
           "delete -- the read-side loader/apply path regressed";

    int64_t seen[32]; uint32_t n_seen = 0;
    IcebergScanGroup none{};
    EXPECT_FALSE(iceberg_scan_last_group(sh, &none)) << "no group before the first batch";
    uint32_t groups_seen = 0;
    for (int guard = 0; guard < 64; ++guard) {
        bolt::BoltBatch batch{};
        bool eof = false;
        ASSERT_TRUE(iceberg_scan_next_batch(sh, &batch, &eof));
        if (eof) break;
        ASSERT_GT(batch.num_rows, 0);
        // G2ICE-262: the batch's row-group identity, including pre-delete counts.
        IcebergScanGroup g{};
        ASSERT_TRUE(iceberg_scan_last_group(sh, &g));
        ASSERT_NE(g.file_path, nullptr);
        EXPECT_EQ(g.row_group, 0u) << "each fixture file has one row group";
        EXPECT_EQ(g.file_row_base, 0u);
        EXPECT_EQ(g.decoded_rows, static_cast<uint64_t>(kRowsPerCommit));
        EXPECT_EQ(g.emitted_rows, static_cast<uint64_t>(batch.num_rows));
        const bool is_deleted_file = std::strstr(g.file_path, commit_file[kDeletedRow / kRowsPerCommit]) != nullptr ||
                                     std::strstr(commit_file[kDeletedRow / kRowsPerCommit], g.file_path) != nullptr;
        EXPECT_EQ(g.emitted_rows, is_deleted_file ? static_cast<uint64_t>(kRowsPerCommit - 1)
                                                  : static_cast<uint64_t>(kRowsPerCommit));
        ++groups_seen;
        const auto& col = batch.columns[batch.read_epoch][0];
        const auto* vals = static_cast<const int64_t*>(col.data);
        for (int64_t r = 0; r < batch.num_rows; ++r) {
            ASSERT_LT(n_seen, 32u);
            seen[n_seen++] = vals[r];
        }
    }
    EXPECT_EQ(groups_seen, static_cast<uint32_t>(kCommits));
    iceberg_scan_close(sh);
    iceberg_table_close(rh);

    // ── VALUE assertions ─────────────────────────────────────────────────
    const int64_t deleted_id = 1000 + kDeletedRow;
    EXPECT_FALSE(contains(seen, n_seen, deleted_id))
        << "TIERING/ICEBERG DELETE-RESURRECTION: id=" << deleted_id
        << " was positionally deleted but bolt's own scan still returned it";
    for (int32_t i = 0; i < kCommits * kRowsPerCommit; ++i) {
        if (i == kDeletedRow) continue;
        EXPECT_TRUE(contains(seen, n_seen, 1000 + i))
            << "surviving row id=" << (1000 + i) << " missing from scan";
    }
    EXPECT_EQ(n_seen, static_cast<uint32_t>(kCommits * kRowsPerCommit - 1));
}

// A table with NO delete files at all must be completely unaffected -- the
// fast path (pos_dels.n == 0) must not change behaviour for the common case.
TEST(IcebergPositionDeleteRead, NoDeletesUnaffected) {
    bolt::Arena arena;
    std::filesystem::path p = std::filesystem::temp_directory_path() /
        "bolt_iceberg_del_read_none" / "default" / "t";
    std::error_code ec;
    std::filesystem::remove_all(p.parent_path().parent_path(), ec);
    std::filesystem::create_directories(p, ec);
    const std::string root = p.generic_string();

    FilesystemObjectStore fs{}; ObjectStore os{};
    ASSERT_TRUE(filesystem_object_store_init(&fs, root.c_str(), &os));
    Schema sch = make_schema();
    PartitionSpec spec{}; spec.spec_id = 0; spec.n_fields = 0;
    SortOrder sort{};     sort.order_id = 0; sort.n_fields = 0;
    WriteOptions wo;      write_options_init(&wo);
    TableHandle* wh = nullptr;
    ASSERT_TRUE(table_create(&wh, &arena, &os, root.c_str(), &sch, &spec,
                             &sort, &wo));
    AppendHandle* ah = nullptr;
    ASSERT_TRUE(append_open(&ah, wh));
    bolt::BoltBatch b{};
    make_batch(&arena, &b, 0, kRowsPerCommit);
    ASSERT_TRUE(append_write(ah, &b));
    ASSERT_TRUE(append_commit(ah));
    append_close(ah);
    table_close(wh);

    const std::string ns_root = p.parent_path().parent_path().generic_string();
    bolt::lakehouse::FilesystemCatalog cat_fs{};
    bolt::lakehouse::Catalog           cat{};
    ASSERT_TRUE(bolt::lakehouse::filesystem_catalog_init(&cat_fs, ns_root.c_str(), &cat));
    bolt::Arena read_arena;
    TableHandle* rh = nullptr;
    ASSERT_TRUE(iceberg_table_open(&rh, &read_arena, &cat, "default", "t"));
    ReadOptions ro{}; read_options_init(&ro);
    ScanHandle* sh = nullptr;
    ASSERT_TRUE(iceberg_scan_open(&sh, rh, &ro));

    uint32_t n_seen = 0;
    for (int guard = 0; guard < 64; ++guard) {
        bolt::BoltBatch batch{};
        bool eof = false;
        ASSERT_TRUE(iceberg_scan_next_batch(sh, &batch, &eof));
        if (eof) break;
        n_seen += static_cast<uint32_t>(batch.num_rows);
    }
    iceberg_scan_close(sh);
    iceberg_table_close(rh);
    EXPECT_EQ(n_seen, static_cast<uint32_t>(kRowsPerCommit));
}

// G2ICE-262: a supplied (cache-hit) row group replaces decoding and is NOT re-filtered by deletes, yet the scan's
// file-position accounting still advances by the pre-delete row count. Cached groups are captured post-delete.
namespace {
struct Captured {
    std::string path;
    uint32_t row_group;
    int64_t decoded, emitted;
    std::vector<int64_t> ids;
};
struct Serve {
    const std::vector<Captured>* groups;
    bool only_deleted_file;       // serve only groups whose emitted != decoded
    uint32_t served = 0, declined = 0;
};
bool provide(void* ctx, const char* path, uint32_t rg, bolt::Arena* arena, bolt::BoltColumn* cols, uint32_t ncols,
             int64_t* decoded, int64_t* emitted) noexcept {
    auto* sv = static_cast<Serve*>(ctx);
    for (const auto& g : *sv->groups) {
        if (g.row_group != rg || g.path != path) continue;
        if (sv->only_deleted_file && g.emitted == g.decoded) { ++sv->declined; return false; }
        if (ncols != 1) return false;
        auto* d = arena->allocate_array<int64_t>(static_cast<uint32_t>(g.ids.size()));
        for (size_t i = 0; i < g.ids.size(); ++i) d[i] = g.ids[i];
        cols[0].type = bolt::BoltType::Int64;
        cols[0].length = static_cast<int64_t>(g.ids.size());
        cols[0].data = d;
        *decoded = g.decoded;
        *emitted = g.emitted;
        ++sv->served;
        return true;
    }
    ++sv->declined;
    return false;
}
}  // namespace

TEST(IcebergPositionDeleteRead, SuppliedGroupsReplaceDecodingAndKeepPositionAccounting) {
    bolt::Arena arena;
    const std::string root = fixture_root();
    FilesystemObjectStore fs{}; ObjectStore os{};
    ASSERT_TRUE(filesystem_object_store_init(&fs, root.c_str(), &os));
    Schema sch = make_schema();
    PartitionSpec spec{}; SortOrder sort{};
    WriteOptions wo; write_options_init(&wo);
    TableHandle* wh = nullptr;
    ASSERT_TRUE(table_create(&wh, &arena, &os, root.c_str(), &sch, &spec, &sort, &wo));
    AppendHandle* ah = nullptr;
    ASSERT_TRUE(append_open(&ah, wh));
    char commit_file[kCommits][kIcebergMaxPath];
    for (int32_t c = 0; c < kCommits; ++c) {
        bolt::BoltBatch b{};
        make_batch(&arena, &b, c * kRowsPerCommit, kRowsPerCommit);
        ASSERT_TRUE(append_write(ah, &b));
        int64_t rows = 0;
        ASSERT_TRUE(append_commit_ex(ah, commit_file[c], kIcebergMaxPath, &rows));
    }
    append_close(ah);
    PositionDeleteEntry pd{};
    std::strncpy(pd.file_path, commit_file[kDeletedRow / kRowsPerCommit], sizeof(pd.file_path) - 1u);
    pd.pos = kDeletedRow % kRowsPerCommit;
    ASSERT_TRUE(table_delete_positions(wh, &pd, 1));
    table_close(wh);

    const std::string ns_root = std::filesystem::path(root).parent_path().parent_path().generic_string();
    bolt::lakehouse::FilesystemCatalog cat_fs{};
    bolt::lakehouse::Catalog cat{};
    ASSERT_TRUE(bolt::lakehouse::filesystem_catalog_init(&cat_fs, ns_root.c_str(), &cat));

    auto scan = [&](const IcebergGroupProvider* prov, std::vector<Captured>* capture, std::vector<int64_t>* ids) {
        bolt::Arena ra;
        TableHandle* rh = nullptr;
        ASSERT_TRUE(iceberg_table_open(&rh, &ra, &cat, "default", "t"));
        ReadOptions ro{}; read_options_init(&ro);
        ScanHandle* sh = nullptr;
        ASSERT_TRUE(iceberg_scan_open(&sh, rh, &ro));
        if (prov != nullptr) iceberg_scan_set_group_provider(sh, prov);
        for (int guard = 0; guard < 64; ++guard) {
            bolt::BoltBatch batch{};
            bool eof = false;
            ASSERT_TRUE(iceberg_scan_next_batch(sh, &batch, &eof));
            if (eof) break;
            IcebergScanGroup g{};
            ASSERT_TRUE(iceberg_scan_last_group(sh, &g));
            const auto* v = static_cast<const int64_t*>(batch.columns[batch.read_epoch][0].data);
            Captured c{g.file_path, g.row_group, static_cast<int64_t>(g.decoded_rows), static_cast<int64_t>(g.emitted_rows), {}};
            for (int64_t r = 0; r < batch.num_rows; ++r) { c.ids.push_back(v[r]); ids->push_back(v[r]); }
            if (capture != nullptr) capture->push_back(std::move(c));
        }
        iceberg_scan_close(sh);
        iceberg_table_close(rh);
    };

    std::vector<Captured> groups;
    std::vector<int64_t> plain;
    scan(nullptr, &groups, &plain);
    ASSERT_EQ(groups.size(), 2u);
    ASSERT_EQ(plain.size(), static_cast<size_t>(kCommits * kRowsPerCommit - 1));

    {   // every group supplied: identical rows, nothing decoded, no delete re-applied
        Serve sv{&groups, false};
        const IcebergGroupProvider prov{&provide, &sv};
        std::vector<int64_t> ids;
        scan(&prov, nullptr, &ids);
        EXPECT_EQ(sv.served, 2u);
        EXPECT_EQ(ids, plain);
    }
    {   // only the group that had a delete supplied; the other decodes: still identical
        Serve sv{&groups, true};
        const IcebergGroupProvider prov{&provide, &sv};
        std::vector<int64_t> ids;
        scan(&prov, nullptr, &ids);
        EXPECT_EQ(sv.served, 1u);
        EXPECT_EQ(sv.declined, 1u);
        EXPECT_EQ(ids, plain);
    }
}
