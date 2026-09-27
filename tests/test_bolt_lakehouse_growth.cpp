// LIMITS L5 (G2CHK-336) — Iceberg metadata has no count ceilings.
//
// Snapshots, manifests per list, schema fields and partition values grow;
// expiry and the per-table retention policy are the only things that shrink
// history; scan metadata is charged to lake_metadata_budget_mb and exhausting
// it is ResourceExhausted, never a short or empty scan.

#include "bolt/bolt_arena.h"
#include "bolt/bolt_budget.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_port.h"
#include "bolt/bolt_resource.h"
#include "bolt/bolt_types.h"
#include "bolt/lakehouse/catalog.h"
#include "bolt/lakehouse/delta/snapshot.h"
#include "bolt/lakehouse/iceberg/manifest.h"
#include "bolt/lakehouse/iceberg/manifest_avro.h"
#include "bolt/lakehouse/iceberg/metadata.h"
#include "bolt/lakehouse/iceberg/scan.h"
#include "bolt/lakehouse/iceberg/snapshot.h"
#include "bolt/lakehouse/iceberg/writer.h"
#include "bolt/lakehouse/object_store.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

using namespace bolt::lakehouse;
using namespace bolt::lakehouse::iceberg;

std::string fresh_dir(const char* tag) {
    auto p = std::filesystem::temp_directory_path() /
             ("bolt_l5_" + std::string(tag) + "_" +
              std::to_string(static_cast<long long>(bolt_getpid())));
    std::error_code ec;
    std::filesystem::remove_all(p, ec);
    std::filesystem::create_directories(p / "default" / "t", ec);
    return p.generic_string();
}

Schema one_col_schema() {
    static SchemaField f[1]{};
    Schema s{};
    s.fields = f;
    s.cap_fields = 1u;
    s.n_fields = 1;
    f[0].id = 1;
    f[0].required = true;
    std::strncpy(f[0].name, "id", sizeof(f[0].name) - 1u);
    std::strncpy(f[0].type, "long", sizeof(f[0].type) - 1u);
    return s;
}

void make_batch(bolt::Arena* a, bolt::BoltBatch* out, int64_t first, int32_t n) {
    bolt::BoltBatch::init_empty(out);
    out->arena = a;
    out->num_rows = n;
    out->num_cols = 1;
    bolt::BoltBatch::alloc_columns(out, a, 1);
    out->schema.add_field("id", bolt::BoltType::Int64, false);
    auto* cols = out->columns[out->read_epoch];
    auto* d = a->allocate_array<int64_t>(static_cast<uint32_t>(n));
    for (int32_t k = 0; k < n; ++k) d[k] = first + k;
    cols[0].type = bolt::BoltType::Int64;
    cols[0].length = n;
    cols[0].data = d;
}

struct Table {
    bolt::Arena arena;
    FilesystemObjectStore fs{};
    ObjectStore os{};
    std::string ns_root;
    std::string root;
    TableHandle* th = nullptr;
};

void create_table(Table* t, const char* tag) {
    t->ns_root = fresh_dir(tag);
    t->root = t->ns_root + "/default/t";
    ASSERT_TRUE(filesystem_object_store_init(&t->fs, t->root.c_str(), &t->os));
    Schema sch = one_col_schema();
    PartitionSpec spec{};
    SortOrder sort{};
    WriteOptions wo;
    write_options_init(&wo);
    ASSERT_TRUE(table_create(&t->th, &t->arena, &t->os, t->root.c_str(), &sch,
                             &spec, &sort, &wo));
}

bool append(Table* t, int64_t first, int32_t n) {
    AppendHandle* ah = nullptr;
    if (!append_open(&ah, t->th)) return false;
    bolt::BoltBatch b{};
    make_batch(&t->arena, &b, first, n);
    const bool ok = append_write(ah, &b) && append_commit(ah);
    append_close(ah);
    return ok;
}

struct ScanTotals {
    bool    ok = false;
    int64_t rows = 0;
    int64_t sum = 0;
};

// bolt's own reader, through the catalog path chukonu uses.
ScanTotals scan_totals(const std::string& ns_root, int64_t snapshot_id) {
    ScanTotals r;
    FilesystemCatalog cat_fs{};
    Catalog cat{};
    if (!filesystem_catalog_init(&cat_fs, ns_root.c_str(), &cat)) return r;
    bolt::Arena a;
    iceberg::TableHandle* h = nullptr;
    if (!iceberg_table_open(&h, &a, &cat, "default", "t")) return r;
    ReadOptions ro{};
    read_options_init(&ro);
    ro.snapshot_id = snapshot_id;
    ScanHandle* sh = nullptr;
    if (!iceberg_scan_open(&sh, h, &ro)) return r;
    for (uint32_t guard = 0; guard < 1000000u; ++guard) {
        bolt::BoltBatch b{};
        bool eof = false;
        if (!iceberg_scan_next_batch(sh, &b, &eof)) { iceberg_scan_close(sh); return r; }
        if (eof) break;
        const auto* v = static_cast<const int64_t*>(b.columns[b.read_epoch][0].data);
        for (int64_t i = 0; i < b.num_rows; ++i) r.sum += v[i];
        r.rows += b.num_rows;
    }
    iceberg_scan_close(sh);
    r.ok = true;
    return r;
}

}  // namespace

// The writer refused the 65th snapshot (kIcebergMaxSnapshots) and a manifest
// list past 256 entries. 300 appends now commit, and bolt's reader returns
// every row, for the current snapshot and for a time-travel read.
TEST(LakeGrowth, WriterCommitsPastOldSnapshotAndManifestCaps) {
    Table t;
    create_table(&t, "writer300");
    ASSERT_NE(t.th, nullptr);
    constexpr int32_t kCommits = 300;
    int64_t expect_sum = 0;
    for (int32_t c = 0; c < kCommits; ++c) {
        ASSERT_TRUE(append(&t, c * 10, 2)) << "commit " << c;
        expect_sum += c * 10 + c * 10 + 1;
    }
    const Metadata* m = table_metadata(t.th);
    ASSERT_EQ(m->n_snapshots, static_cast<uint32_t>(kCommits));
    const int64_t tenth = m->snapshots[9].snapshot_id;

    const ScanTotals cur = scan_totals(t.ns_root, -1);
    ASSERT_TRUE(cur.ok);
    EXPECT_EQ(cur.rows, 2 * kCommits);
    EXPECT_EQ(cur.sum, expect_sum);

    const ScanTotals tt = scan_totals(t.ns_root, tenth);
    ASSERT_TRUE(tt.ok);
    EXPECT_EQ(tt.rows, 20);
}

// Expiry keeps the current snapshot and its newest ancestors, deletes only
// what no retained snapshot reaches, and loses no live row. The policy is a
// table property: it survives a reopen and runs after every commit.
TEST(LakeGrowth, RetentionPolicyExpiresHistoryNotData) {
    Table t;
    create_table(&t, "retain");
    ASSERT_NE(t.th, nullptr);
    for (int32_t c = 0; c < 30; ++c) ASSERT_TRUE(append(&t, c * 10, 2));
    const int64_t first_id = table_metadata(t.th)->snapshots[0].snapshot_id;
    ASSERT_TRUE(table_set_retention(t.th, 0, 5));
    const Metadata* m = table_metadata(t.th);
    EXPECT_EQ(m->n_snapshots, 5u);
    EXPECT_EQ(m->n_snapshot_log, 5u);
    EXPECT_NE(snapshot_by_id(m, m->current_snapshot_id), nullptr);

    const ScanTotals cur = scan_totals(t.ns_root, -1);
    ASSERT_TRUE(cur.ok);
    EXPECT_EQ(cur.rows, 60);
    // An expired id is gone, loudly.
    EXPECT_FALSE(scan_totals(t.ns_root, first_id).ok);

    table_close(t.th);
    bolt::Arena a2;
    TableHandle* th2 = nullptr;
    ASSERT_TRUE(table_open(&th2, &a2, &t.os, t.root.c_str()));
    EXPECT_EQ(table_metadata(th2)->retention_min_snapshots, 5);
    t.th = th2;
    for (int32_t c = 30; c < 40; ++c) ASSERT_TRUE(append(&t, c * 10, 2));
    EXPECT_EQ(table_metadata(th2)->n_snapshots, 5u);
    const ScanTotals after = scan_totals(t.ns_root, -1);
    ASSERT_TRUE(after.ok);
    EXPECT_EQ(after.rows, 80);
}

// The old expiry could expire the current snapshot and move current back.
TEST(LakeGrowth, ExpireNeverDropsTheCurrentSnapshot) {
    Table t;
    create_table(&t, "expire_cur");
    ASSERT_NE(t.th, nullptr);
    for (int32_t c = 0; c < 4; ++c) ASSERT_TRUE(append(&t, c * 10, 2));
    const int64_t cur = table_metadata(t.th)->current_snapshot_id;
    ExpireOptions o{};
    o.older_than_ms = INT64_MAX;
    o.retain_last = 0;
    o.delete_files = true;
    ExpireResult r{};
    ASSERT_TRUE(table_expire_snapshots_ex(t.th, &o, &r));
    EXPECT_EQ(r.expired, 3u);
    EXPECT_GE(r.files_deleted, 3u);
    EXPECT_EQ(table_metadata(t.th)->current_snapshot_id, cur);
    const ScanTotals s = scan_totals(t.ns_root, -1);
    ASSERT_TRUE(s.ok);
    EXPECT_EQ(s.rows, 8);
}

// A 300-column schema (the old array held 256) round-trips through
// metadata.json emit and parse.
TEST(LakeGrowth, WideSchemaRoundTrips) {
    bolt::Arena a;
    Metadata* m = a.allocate_array<Metadata>(1);
    ASSERT_NE(m, nullptr);
    metadata_init(m, &a, nullptr);
    m->format_version = 2;
    std::strcpy(m->location, "/tmp/wide");
    Schema* s = metadata_push_schema(m);
    ASSERT_NE(s, nullptr);
    for (int32_t i = 0; i < 300; ++i) {
        SchemaField f{};
        f.id = i + 1;
        std::snprintf(f.name, sizeof(f.name), "c%03d", i);
        std::strcpy(f.type, "long");
        ASSERT_TRUE(schema_push_field(s, &a, nullptr, &f));
    }
    m->n_specs = 1;
    m->n_sort_orders = 1;
    const uint8_t* js = nullptr;
    uint64_t jl = 0;
    ASSERT_TRUE(metadata_json_emit_plain(m, &a, &js, &jl));
    Metadata* back = a.allocate_array<Metadata>(1);
    ASSERT_TRUE(metadata_parse(js, static_cast<uint32_t>(jl), &a, back));
    ASSERT_EQ(back->n_schemas, 1u);
    ASSERT_EQ(back->schemas[0].n_fields, 300u);
    EXPECT_STREQ(back->schemas[0].fields[299].name, "c299");
    EXPECT_EQ(back->schemas[0].fields[299].id, 300);
}

// Twelve partition columns (the old DataFileRef held 8) survive a manifest
// write and read.
TEST(LakeGrowth, TwelvePartitionValuesRoundTrip) {
    bolt::Arena a;
    PartitionSpec spec{};
    spec.spec_id = 0;
    spec.n_fields = 12;
    PartitionAvroType types[12];
    for (uint32_t i = 0; i < 12u; ++i) {
        spec.fields[i].source_id = static_cast<int32_t>(i + 1);
        spec.fields[i].field_id = static_cast<int32_t>(1000 + i);
        spec.fields[i].transform = transform_parse("identity", 8);
        std::snprintf(spec.fields[i].name, sizeof(spec.fields[i].name), "p%u", i);
        types[i] = PartitionAvroType::kLong;
    }
    std::vector<DataFileRef> files(3);
    for (uint32_t f = 0; f < 3u; ++f) {
        DataFileRef& d = files[f];
        std::memset(&d, 0, sizeof(d));
        d.status = ManifestStatus::kAdded;
        d.content = FileContent::kData;
        std::snprintf(d.file_path, sizeof(d.file_path), "/w/t/data/%u.parquet", f);
        d.n_partition = 12;
        for (uint32_t i = 0; i < 12u; ++i) {
            d.partition[i].field_id = static_cast<int32_t>(i);
            d.partition[i].is_int = true;
            d.partition[i].i64 = static_cast<int64_t>(f * 100 + i);
        }
        d.stats.record_count = 1;
    }
    static const char kSch[] =
        "{\"type\":\"struct\",\"schema-id\":0,\"fields\":[]}";
    const uint8_t* body = nullptr;
    uint64_t len = 0;
    ASSERT_TRUE(manifest_write_avro_partitioned(files.data(), 3, 1, 1, kSch,
                                                sizeof(kSch) - 1u, &spec,
                                                types, &a, &body, &len));
    DataFileRef* out = nullptr;
    uint32_t n = 0;
    ASSERT_TRUE(manifest_parse_avro_grow(body, len, &a, 0, 1u, &out, &n));
    ASSERT_EQ(n, 3u);
    ASSERT_EQ(out[2].n_partition, 12u);
    EXPECT_EQ(out[2].partition[11].i64, 211);
}

namespace {

// A table whose single manifest names `n_files` data files that do not exist:
// the scan must fail, and whether it fails on the budget or on the first file
// read says whether the metadata was charged.
void write_synthetic_table(ObjectStore* os, bolt::Arena* a, uint32_t n_files,
                           std::string* metadata_json) {
    std::vector<DataFileRef> files(n_files);
    const std::string pad(110, 'x');
    for (uint32_t i = 0; i < n_files; ++i) {
        DataFileRef& d = files[i];
        std::memset(&d, 0, sizeof(d));
        d.status = ManifestStatus::kAdded;
        d.content = FileContent::kData;
        std::snprintf(d.file_path, sizeof(d.file_path),
                      "s3://b/t/data/%s-%06u.parquet", pad.c_str(), i);
        d.stats.record_count = 1;
    }
    static const char kSch[] =
        "{\"type\":\"struct\",\"schema-id\":0,\"fields\":[{\"id\":1,"
        "\"name\":\"id\",\"required\":true,\"type\":\"long\"}]}";
    const uint8_t* mb = nullptr;
    uint64_t ml = 0;
    ASSERT_TRUE(manifest_write_avro(files.data(), n_files, 1, 1, kSch,
                                    sizeof(kSch) - 1u, 0, a, &mb, &ml));
    ASSERT_EQ(os_put(os, "t/metadata/m.avro", mb, ml), kOsOk);
    ManifestListEntry e{};
    e.content = ManifestContent::kDataManifest;
    e.manifest_length = static_cast<int64_t>(ml);
    e.added_snapshot_id = 1;
    e.added_files_count = n_files;
    std::strcpy(e.manifest_path, "s3://b/t/metadata/m.avro");
    const uint8_t* lb = nullptr;
    uint64_t ll = 0;
    ASSERT_TRUE(manifest_list_write_avro(&e, 1, 1, 1, a, &lb, &ll));
    ASSERT_EQ(os_put(os, "t/metadata/snap.avro", lb, ll), kOsOk);
    *metadata_json =
        "{\"format-version\":2,\"table-uuid\":\"u\",\"location\":\"s3://b/t\","
        "\"current-snapshot-id\":1,\"current-schema-id\":0,"
        "\"schemas\":[" + std::string(kSch) + "],"
        "\"snapshots\":[{\"snapshot-id\":1,\"timestamp-ms\":1,"
        "\"sequence-number\":1,"
        "\"manifest-list\":\"s3://b/t/metadata/snap.avro\"}]}";
}

bool first_batch_fails(ObjectStore* os, const std::string& mj) {
    bolt::Arena a;
    iceberg::TableHandle* h = nullptr;
    EXPECT_TRUE(iceberg_table_open_on_store(
        &h, &a, os, "t", reinterpret_cast<const uint8_t*>(mj.data()),
        static_cast<uint32_t>(mj.size())));
    if (h == nullptr) return false;
    ScanHandle* sh = nullptr;
    EXPECT_TRUE(iceberg_scan_open(&sh, h, nullptr));
    if (sh == nullptr) return false;
    bolt::BoltBatch b{};
    bool eof = false;
    const bool ok = iceberg_scan_next_batch(sh, &b, &eof);
    iceberg_scan_close(sh);
    return !ok;
}

}  // namespace

TEST(LakeGrowth, MetadataBudgetExhaustionIsResourceExhausted) {
    const std::string dir = fresh_dir("budget");
    FilesystemObjectStore fs{};
    ObjectStore os{};
    ASSERT_TRUE(filesystem_object_store_init(&fs, dir.c_str(), &os));
    bolt::Arena a;
    std::string mj;
    write_synthetic_table(&os, &a, 20000u, &mj);
    auto& lt = bolt::bolt_limits();
    const uint32_t id = static_cast<uint32_t>(bolt::bolt_limits_id::lake_metadata_budget_mb);
    const uint64_t saved = bolt::limits_value(lt, id);

    // 20,000 tasks of ~150 bytes do not fit in 1 MiB.
    ASSERT_TRUE(bolt::limits_set(&lt, id, 1u, nullptr, 0));
    bolt::clear_resource_exhausted();
    EXPECT_TRUE(first_batch_fails(&os, mj));
    const bolt::ResourceExhausted re = bolt::last_resource_exhausted();
    ASSERT_NE(re.knob, nullptr);
    EXPECT_STREQ(re.knob, "lake_metadata_budget_mb");

    // Under the default budget the same scan gets as far as the first
    // (missing) data file.
    ASSERT_TRUE(bolt::limits_set(&lt, id, saved, nullptr, 0));
    bolt::clear_resource_exhausted();
    EXPECT_TRUE(first_batch_fails(&os, mj));
    EXPECT_EQ(bolt::last_resource_exhausted().knob, nullptr);
}

// ---- Delta ------------------------------------------------------------------

namespace {

void put_commit(ObjectStore* os, uint32_t v, const std::string& body) {
    char name[96];
    std::snprintf(name, sizeof(name), "t/_delta_log/%020u.json", v);
    ASSERT_EQ(os_put(os, name, reinterpret_cast<const uint8_t*>(body.data()),
                     body.size()),
              kOsOk);
}

std::string add_line(uint32_t f) {
    return R"({"add":{"path":"f)" + std::to_string(f) +
           R"(.parquet","size":1,"modificationTime":0,"dataChange":true,"partitionValues":{}}})"
           "\n";
}

}  // namespace

// 20,000 adds over 40 commits, then 5,000 removes: the snapshot holds exactly
// the 15,000 survivors (live files used to stop at 4,096; replay was a linear
// path search per action).
TEST(LakeGrowth, DeltaSnapshotHoldsEveryLiveFile) {
    const std::string dir = fresh_dir("delta");
    FilesystemObjectStore fs{};
    ObjectStore os{};
    ASSERT_TRUE(filesystem_object_store_init(&fs, dir.c_str(), &os));
    constexpr uint32_t kCommits = 40, kPer = 500, kRemoved = 5000;
    for (uint32_t c = 0; c < kCommits; ++c) {
        std::string body;
        for (uint32_t i = 0; i < kPer; ++i) body += add_line(c * kPer + i);
        put_commit(&os, c, body);
    }
    std::string rem;
    for (uint32_t f = 0; f < kRemoved * 2; f += 2) {   // every even file < 10,000
        rem += R"({"remove":{"path":"f)" + std::to_string(f) +
               R"(.parquet","deletionTimestamp":0,"dataChange":true}})" "\n";
    }
    put_commit(&os, kCommits, rem);
    bolt::Arena a;
    delta::Snapshot snap{};
    ASSERT_TRUE(delta::delta_snapshot_build(&os, "t", -1, &a, &snap));
    ASSERT_EQ(snap.n_files, kCommits * kPer - kRemoved);
    std::vector<uint8_t> seen(kCommits * kPer, 0);
    for (uint32_t i = 0; i < snap.n_files; ++i) {
        const unsigned f = static_cast<unsigned>(
            std::strtoul(snap.files[i].path + 1, nullptr, 10));
        ASSERT_LT(f, kCommits * kPer);
        EXPECT_EQ(seen[f], 0) << "duplicate " << snap.files[i].path;
        seen[f] = 1;
        EXPECT_TRUE(f >= kRemoved * 2 || (f % 2) == 1) << snap.files[i].path;
    }
    // Re-adding a removed path revives it once.
    put_commit(&os, kCommits + 1, add_line(0) + add_line(0));
    bolt::Arena a2;
    delta::Snapshot s2{};
    ASSERT_TRUE(delta::delta_snapshot_build(&os, "t", -1, &a2, &s2));
    EXPECT_EQ(s2.n_files, kCommits * kPer - kRemoved + 1u);
}

// A log missing a commit (retention removed it; checkpoint decode is not
// implemented) is refused rather than replayed without that commit's actions.
TEST(LakeGrowth, DeltaLogWithAGapIsRefused) {
    const std::string dir = fresh_dir("delta_gap");
    FilesystemObjectStore fs{};
    ObjectStore os{};
    ASSERT_TRUE(filesystem_object_store_init(&fs, dir.c_str(), &os));
    put_commit(&os, 0, add_line(0));
    put_commit(&os, 2, add_line(2));
    bolt::Arena a;
    delta::Snapshot snap{};
    EXPECT_FALSE(delta::delta_snapshot_build(&os, "t", -1, &a, &snap));
    put_commit(&os, 1, add_line(1));
    bolt::Arena a2;
    delta::Snapshot s2{};
    ASSERT_TRUE(delta::delta_snapshot_build(&os, "t", -1, &a2, &s2));
    EXPECT_EQ(s2.n_files, 3u);
}
