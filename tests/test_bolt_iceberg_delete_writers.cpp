// G2ICE-283 — equality-delete files (predicate, by key, partition-scoped and global) and position deletes carrying
// the deleted `row`. Assertions are on the rows bolt's own scan returns and on the manifests that produced them. The
// external verdict (DuckDB and Spark read the rows; pyiceberg the metadata, and pyarrow the `row` column) is
// scripts/iceberg_delete_writers_interop.py and Gestalt2's verify_g2ice283_bolt_delete_writers_spark.py, which drive
// IcebergDeleteWritersFixture below.

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"
#include "bolt/lakehouse/iceberg/manifest.h"
#include "bolt/lakehouse/iceberg/manifest_avro.h"
#include "bolt/lakehouse/iceberg/metadata.h"
#include "bolt/lakehouse/iceberg/scan.h"
#include "bolt/lakehouse/iceberg/transform.h"
#include "bolt/lakehouse/iceberg/writer.h"
#include "bolt/lakehouse/object_store.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace bolt::lakehouse;
using namespace bolt::lakehouse::iceberg;

constexpr int64_t kNullK = INT64_MIN;

struct Row {
    int64_t id;
    int64_t k;
    std::string s;
    bool operator<(const Row& o) const { return id < o.id; }
    bool operator==(const Row& o) const { return id == o.id && k == o.k && s == o.s; }
};

std::string fresh_root(const std::string& base, const char* tag) {
    std::filesystem::path p = std::filesystem::path(base) / tag;
    std::error_code ec;
    std::filesystem::remove_all(p, ec);
    std::filesystem::create_directories(p, ec);
    return p.generic_string();
}

std::string temp_base() { return (std::filesystem::temp_directory_path() / "bolt_iceberg_delete_writers").generic_string(); }

// id (long, required, the identifier field), k (long), s (string).
Schema make_schema() {
    Schema s{};
    static SchemaField f[8]{};
    static int32_t ident[1] = {1};
    s.fields = f;
    s.cap_fields = 8u;
    s.n_fields = 3;
    s.identifier_ids = ident;
    s.n_identifier = 1;
    f[0].id = 1; f[0].required = true;  std::strcpy(f[0].name, "id"); std::strcpy(f[0].type, "long");
    f[1].id = 2; f[1].required = false; std::strcpy(f[1].name, "k");  std::strcpy(f[1].type, "long");
    f[2].id = 3; f[2].required = false; std::strcpy(f[2].name, "s");  std::strcpy(f[2].type, "string");
    return s;
}

std::string row_s(int64_t id) { return "row-" + std::to_string(id) + (id % 3 == 0 ? "-with-a-long-spilled-tail" : ""); }

int64_t base_k(int64_t id) {
    if (id == 3) return kNullK;
    if (id < 10) return 0;
    if (id < 20) return 1;
    return id < 25 ? 1 : 2;
}

std::vector<Row> base_rows(int64_t lo, int64_t hi) {
    std::vector<Row> v;
    for (int64_t id = lo; id < hi; ++id) v.push_back(Row{id, base_k(id), row_s(id)});
    return v;
}

// A Utf8 column of `vals` over arena storage (long values spill).
bolt::BoltColumn utf8_col(bolt::Arena* a, const std::vector<std::string>& vals) {
    const size_t n = vals.size();
    auto* svs = a->allocate_array<bolt::StringView>(n + 1u);
    size_t total = 0;
    for (const auto& v : vals) total += v.size();
    auto* spill = a->allocate_array<uint8_t>(total + 1u);
    uint32_t off = 0;
    for (size_t i = 0; i < n; ++i) {
        std::memcpy(spill + off, vals[i].data(), vals[i].size());
        svs[i] = bolt::StringView::from_cstr(vals[i].c_str());
        if (vals[i].size() > 12u) { svs[i].ref.buf_idx = 0; svs[i].ref.offset = off; }
        off += static_cast<uint32_t>(vals[i].size());
    }
    bolt::BoltColumn c{};
    c.type = bolt::BoltType::Utf8;
    c.length = static_cast<int64_t>(n);
    c.data = svs;
    c.str_overflow_base = spill;
    return c;
}

bolt::BoltColumn i64_col(bolt::Arena* a, const std::vector<int64_t>& v, bool nullable) {
    const size_t n = v.size();
    auto* d = a->allocate_array<int64_t>(n + 1u);
    auto* valid = static_cast<uint8_t*>(a->allocate_zeroed(n / 8u + 2u));
    bool any_null = false;
    for (size_t i = 0; i < n; ++i) {
        d[i] = v[i] == kNullK ? 0 : v[i];
        if (v[i] != kNullK) valid[i >> 3] |= static_cast<uint8_t>(1u << (i & 7u)); else any_null = true;
    }
    bolt::BoltColumn c = bolt::BoltColumn::make_flat(d, nullable && any_null ? valid : nullptr, static_cast<int64_t>(n),
                                                     bolt::BoltType::Int64);
    c.stats.all_valid = !any_null;
    return c;
}

void make_batch(bolt::Arena* a, bolt::BoltBatch* b, const std::vector<Row>& rows) {
    bolt::BoltBatch::init_empty(b);
    b->arena = a;
    b->num_rows = static_cast<int64_t>(rows.size());
    b->num_cols = 3;
    ASSERT_TRUE(bolt::BoltBatch::alloc_columns(b, a, 3));
    b->schema.add_field("id", bolt::BoltType::Int64, false);
    b->schema.add_field("k", bolt::BoltType::Int64, true);
    b->schema.add_field("s", bolt::BoltType::Utf8, true);
    std::vector<int64_t> ids, ks;
    std::vector<std::string> ss;
    for (const Row& r : rows) { ids.push_back(r.id); ks.push_back(r.k); ss.push_back(r.s); }
    auto* cols = b->columns[b->read_epoch];
    cols[0] = i64_col(a, ids, false);
    cols[1] = i64_col(a, ks, true);
    cols[2] = utf8_col(a, ss);
}

// A one-column key batch `name` of `vals`.
void key_batch(bolt::Arena* a, bolt::BoltBatch* b, const char* name, const std::vector<int64_t>& vals) {
    bolt::BoltBatch::init_empty(b);
    b->arena = a;
    b->num_rows = static_cast<int64_t>(vals.size());
    b->num_cols = 1;
    ASSERT_TRUE(bolt::BoltBatch::alloc_columns(b, a, 1));
    b->schema.add_field(name, bolt::BoltType::Int64, false);
    b->columns[b->read_epoch][0] = i64_col(a, vals, false);
}

struct Table {
    bolt::Arena arena;
    std::string root;
    FilesystemObjectStore fs{};
    ObjectStore os{};
    TableHandle* th = nullptr;
    std::vector<std::string> files;
};

PartitionField pf(int32_t source, const char* transform, const char* name) {
    PartitionField f{};
    f.source_id = source;
    f.transform = transform_parse(transform, static_cast<uint32_t>(std::strlen(transform)));
    std::strncpy(f.name, name, sizeof(f.name) - 1u);
    return f;
}

void create_table(Table* t, const std::string& root, const PartitionField* part = nullptr, uint8_t fv = 2) {
    t->root = root;
    ASSERT_TRUE(filesystem_object_store_init(&t->fs, root.c_str(), &t->os));
    Schema sch = make_schema();
    WriteOptions wo; write_options_init(&wo);
    wo.format_version = fv;
    PartitionSpec spec{};
    if (part != nullptr) spec.fields[spec.n_fields++] = *part;
    ASSERT_TRUE(table_create(&t->th, &t->arena, &t->os, root.c_str(), &sch, part != nullptr ? &spec : nullptr, nullptr, &wo));
}

void append_rows(Table* t, const std::vector<Row>& rows) {
    AppendHandle* ah = nullptr;
    ASSERT_TRUE(append_open(&ah, t->th));
    bolt::BoltBatch b{};
    make_batch(&t->arena, &b, rows);
    ASSERT_TRUE(append_write(ah, &b));
    char path[kIcebergMaxPath];
    int64_t n = 0;
    ASSERT_TRUE(append_commit_ex(ah, path, sizeof(path), &n));
    append_close(ah);
    t->files.push_back(path);
}

void delete_position(Table* t, const std::string& file, int64_t pos) {
    PositionDeleteEntry e{};
    std::strncpy(e.file_path, file.c_str(), sizeof(e.file_path) - 1u);
    e.pos = pos;
    ASSERT_TRUE(table_delete_positions(t->th, &e, 1));
}

// Three files (ids 0..9, 10..19, 20..29), then position deletes of ids 15 and 22 (unpartitioned base only).
void build_base(Table* t, const std::string& root, const PartitionField* part = nullptr, uint8_t fv = 2) {
    create_table(t, root, part, fv);
    append_rows(t, base_rows(0, 10));
    append_rows(t, base_rows(10, 20));
    append_rows(t, base_rows(20, 30));
    if (part == nullptr) {
        delete_position(t, t->files[1], 5);
        delete_position(t, t->files[2], 2);
    }
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string latest_metadata_path(const std::string& root) {
    const std::string hint = read_file(root + "/metadata/version-hint.text");
    return root + "/metadata/v" + std::to_string(std::stoll(hint)) + ".metadata.json";
}

std::vector<Row> scan_rows(const std::string& root) {
    bolt::Arena a;
    FilesystemObjectStore fs{}; ObjectStore os{};
    EXPECT_TRUE(filesystem_object_store_init(&fs, root.c_str(), &os));
    const std::string meta = read_file(latest_metadata_path(root));
    TableHandle* rh = nullptr;
    EXPECT_TRUE(iceberg_table_open_on_store(&rh, &a, &os, "", reinterpret_cast<const uint8_t*>(meta.data()),
                                            static_cast<uint32_t>(meta.size())));
    std::vector<Row> out;
    if (rh == nullptr) return out;
    ReadOptions ro; read_options_init(&ro);
    ScanHandle* sh = nullptr;
    EXPECT_TRUE(iceberg_scan_open(&sh, rh, &ro));
    if (sh == nullptr) return out;
    for (int guard = 0; guard < 10000; ++guard) {
        bolt::BoltBatch b{};
        bool eof = false;
        EXPECT_TRUE(iceberg_scan_next_batch(sh, &b, &eof));
        if (eof) break;
        const auto* cols = b.columns[b.read_epoch];
        for (int64_t r = 0; r < b.num_rows; ++r) {
            Row row{};
            row.id = static_cast<const int64_t*>(cols[0].data)[r];
            row.k = cols[1].is_null(r) ? kNullK : static_cast<const int64_t*>(cols[1].data)[r];
            const uint8_t* d = nullptr; int32_t dl = 0;
            cols[2].utf8_at(r, &d, &dl);
            row.s.assign(reinterpret_cast<const char*>(d), static_cast<size_t>(dl));
            out.push_back(row);
        }
    }
    iceberg_scan_close(sh);
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<Row> model(std::vector<Row> all, const std::function<bool(const Row&)>& gone) {
    std::vector<Row> v;
    for (const Row& r : all) if (!gone(r)) v.push_back(r);
    std::sort(v.begin(), v.end());
    return v;
}

Predicate pred_i64(const char* col, PredicateOp op, int64_t v) {
    Predicate p{};
    std::strncpy(p.column, col, sizeof(p.column) - 1u);
    p.op = op;
    p.value.type = bolt::BoltType::Int64;
    p.value.i64 = v;
    return p;
}

Predicate pred_str(const char* col, const std::string& v) {
    Predicate p{};
    std::strncpy(p.column, col, sizeof(p.column) - 1u);
    p.op = PredicateOp::kEq;
    p.value.type = bolt::BoltType::Utf8;
    std::memcpy(p.value.str, v.data(), v.size());
    p.value.str_len = static_cast<uint32_t>(v.size());
    return p;
}

DeleteOptions eq_opts(std::initializer_list<int32_t> ids = {}, bool global = false) {
    DeleteOptions o{};
    o.kind = DeleteFileKind::kEquality;
    o.global = global;
    for (int32_t id : ids) o.equality_ids[o.n_equality_ids++] = id;
    return o;
}

// The delete files of the current snapshot, with their content, spec, tuple width and equality ids.
struct DelFile { FileContent content; int32_t spec; uint32_t n_part; std::vector<int32_t> eq; int64_t rows; std::string path;
                 bool dv; std::string ref; };

std::string store_key(const std::string& root, const std::string& loc) {
    if (loc.size() > root.size() && loc.compare(0, root.size(), root) == 0) return loc.substr(root.size() + 1u);
    return loc;
}

std::vector<DelFile> delete_files(const Table& t) {
    std::vector<DelFile> out;
    const Metadata* m = table_metadata(t.th);
    const Snapshot* cur = nullptr;
    for (uint32_t i = 0; i < m->n_snapshots; ++i) if (m->snapshots[i].snapshot_id == m->current_snapshot_id) cur = &m->snapshots[i];
    if (cur == nullptr) return out;
    bolt::Arena a;
    const std::string ml = read_file(t.root + "/" + store_key(t.root, cur->manifest_list));
    ManifestListEntry* mle = nullptr; uint32_t nm = 0;
    EXPECT_TRUE(manifest_list_parse_avro_grow(reinterpret_cast<const uint8_t*>(ml.data()), ml.size(), &a, 4u, &mle, &nm));
    for (uint32_t i = 0; i < nm; ++i) {
        const std::string body = read_file(t.root + "/" + store_key(t.root, mle[i].manifest_path));
        DataFileRef* e = nullptr; uint32_t ne = 0;
        EXPECT_TRUE(manifest_parse_avro_grow(reinterpret_cast<const uint8_t*>(body.data()), body.size(), &a,
                                             mle[i].partition_spec_id, 4u, &e, &ne));
        for (uint32_t j = 0; j < ne; ++j) {
            if (e[j].content == FileContent::kData || e[j].status == ManifestStatus::kDeleted) continue;
            DelFile d{e[j].content, e[j].partition_spec_id, e[j].n_partition, {}, e[j].stats.record_count, e[j].file_path,
                      e[j].is_dv, e[j].referenced_data_file};
            for (uint32_t k = 0; k < e[j].n_equality_ids && k < limits::kMaxEqualityIds; ++k) d.eq.push_back(e[j].equality_ids[k]);
            if (d.spec == 0 && mle[i].partition_spec_id != 0) d.spec = mle[i].partition_spec_id;
            out.push_back(d);
        }
    }
    return out;
}

size_t count_eq(const std::vector<DelFile>& v) {
    return static_cast<size_t>(std::count_if(v.begin(), v.end(), [](const DelFile& d) { return d.content == FileContent::kEqualityDeletes; }));
}

// ---- the fixture tables, shared by the unit tests and IcebergDeleteWritersFixture ----

void write_eq_pred(Table* t, const std::string& root) {
    build_base(t, root);
    Predicate p = pred_i64("k", PredicateOp::kEq, 1);
    DeleteOptions o = eq_opts();                                          // the identifier field: id
    ASSERT_TRUE(table_delete_with(t->th, &p, &o));
    Predicate s = pred_str("s", row_s(27));
    DeleteOptions os = eq_opts({3});                                      // a string key
    ASSERT_TRUE(table_delete_with(t->th, &s, &os));
}

void write_eq_scoped(Table* t, const std::string& root) {
    PartitionField k = pf(2, "identity", "k");
    build_base(t, root, &k);
    Predicate p = pred_i64("id", PredicateOp::kLt, 12);                   // partitions k=0, k=NULL and k=1
    DeleteOptions o = eq_opts({1});
    ASSERT_TRUE(table_delete_with(t->th, &p, &o));
}

void write_eq_global(Table* t, const std::string& root) {
    PartitionField k = pf(2, "identity", "k");
    build_base(t, root, &k);
    Predicate p = pred_i64("id", PredicateOp::kGe, 25);
    DeleteOptions o = eq_opts({1}, true);
    ASSERT_TRUE(table_delete_with(t->th, &p, &o));
}

void write_keys(Table* t, const std::string& root) {
    build_base(t, root);
    bolt::BoltBatch kb{};
    key_batch(&t->arena, &kb, "id", {1, 2, 28, 99});
    DeleteOptions o = eq_opts();
    ASSERT_TRUE(table_delete_keys(t->th, &kb, &o));
}

void write_keys_bucketed(Table* t, const std::string& root) {
    PartitionField b = pf(1, "bucket[4]", "id_bucket");
    build_base(t, root, &b);
    bolt::BoltBatch kb{};
    key_batch(&t->arena, &kb, "id", {4, 17, 26});
    DeleteOptions o = eq_opts();
    ASSERT_TRUE(table_delete_keys(t->th, &kb, &o));
}

void write_keys_global(Table* t, const std::string& root) {
    PartitionField k = pf(2, "identity", "k");
    build_base(t, root, &k);
    bolt::BoltBatch kb{};
    key_batch(&t->arena, &kb, "id", {5, 13, 27});
    DeleteOptions o = eq_opts();
    ASSERT_TRUE(table_delete_keys(t->th, &kb, &o));
}

std::vector<Row> upsert_rows() {
    std::vector<Row> v;
    for (int64_t id = 0; id < 5; ++id) v.push_back(Row{id, 7, "new-" + row_s(id)});
    return v;
}

void write_upsert(Table* t, const std::string& root) {
    build_base(t, root);
    bolt::BoltBatch b{};
    make_batch(&t->arena, &b, upsert_rows());
    Predicate p = pred_i64("id", PredicateOp::kLt, 5);
    DeleteOptions o = eq_opts();
    ASSERT_TRUE(table_overwrite_with(t->th, &p, &b, &o));
}

void write_pos_row(Table* t, const std::string& root) {
    build_base(t, root);
    Predicate p = pred_i64("k", PredicateOp::kEq, 1);
    DeleteOptions o{};
    o.kind = DeleteFileKind::kPosition;
    o.position_row = true;
    ASSERT_TRUE(table_delete_with(t->th, &p, &o));
    Predicate q = pred_i64("id", PredicateOp::kLe, 2);
    ASSERT_TRUE(table_delete_with(t->th, &q, &o));
}

void write_pos_row_partitioned(Table* t, const std::string& root) {
    PartitionField k = pf(2, "identity", "k");
    build_base(t, root, &k);
    Predicate p = pred_i64("id", PredicateOp::kGe, 23);                   // partitions k=1 (23, 24) and k=2 (25..29)
    DeleteOptions o{};
    o.kind = DeleteFileKind::kPosition;
    o.position_row = true;
    ASSERT_TRUE(table_delete_with(t->th, &p, &o));
}

bool gone_pos(const Row& r) { return r.id == 15 || r.id == 22; }

// v3: the base (position deletes of ids 15 and 22 become DVs), a predicate delete merging into C's DV, a positions delete,
// a delete removing file A's last rows, and an equality delete.
void write_v3(Table* t, const std::string& root) {
    build_base(t, root, nullptr, 3);
    Predicate p = pred_i64("id", PredicateOp::kGe, 27);                   // C: 27..29 join 22 in C's one DV
    ASSERT_TRUE(table_delete(t->th, &p));
    delete_position(t, t->files[0], 4);                                   // A: id 4
    Predicate low = pred_i64("id", PredicateOp::kLt, 10);                 // every live row of A: the file goes
    DeleteOptions none{};
    ASSERT_TRUE(table_delete_with(t->th, &low, &none));
    Predicate k = pred_i64("id", PredicateOp::kEq, 18);
    DeleteOptions o = eq_opts();
    ASSERT_TRUE(table_delete_with(t->th, &k, &o));
    std::vector<Row> add{Row{100, 7, row_s(100)}, Row{101, 7, row_s(101)}};
    append_rows(t, add);
}

bool gone_v3(const Row& r) { return r.id == 15 || r.id == 22 || r.id >= 27 || r.id < 10 || r.id == 18; }

std::string metadata_text(const Table& t) { return read_file(latest_metadata_path(t.root)); }

}  // namespace

TEST(IcebergDeleteWriters, EqualityDeleteByPredicateUsesTheIdentifierAndAStringKey) {
    Table t;
    write_eq_pred(&t, fresh_root(temp_base(), "eq_pred"));
    EXPECT_EQ(scan_rows(t.root), model(base_rows(0, 30), [](const Row& r) { return gone_pos(r) || r.k == 1 || r.id == 27; }));
    const auto d = delete_files(t);
    ASSERT_EQ(count_eq(d), 2u);
    for (const DelFile& f : d) {
        if (f.content != FileContent::kEqualityDeletes) continue;
        EXPECT_TRUE(f.eq == std::vector<int32_t>{1} || f.eq == std::vector<int32_t>{3});
        EXPECT_EQ(f.rows, f.eq[0] == 1 ? 13 : 1) << "k=1 holds ids 10..24 less the deleted 15 and 22";
    }
}

TEST(IcebergDeleteWriters, EqualityDeleteRefusesAKeyAKeptRowShares) {
    Table t;
    build_base(&t, fresh_root(temp_base(), "eq_conflict"));
    const int64_t cur = table_metadata(t.th)->current_snapshot_id;
    Predicate p = pred_i64("id", PredicateOp::kEq, 4);
    DeleteOptions o = eq_opts({2});                                       // k = 0 is carried by ids 0..9
    EXPECT_FALSE(table_delete_with(t.th, &p, &o));
    EXPECT_EQ(table_last_commit_error(t.th), CommitError::kFailed);
    EXPECT_EQ(table_metadata(t.th)->current_snapshot_id, cur) << "nothing was committed";
    Predicate all0 = pred_i64("k", PredicateOp::kEq, 0);                  // every k = 0 row: allowed
    EXPECT_TRUE(table_delete_with(t.th, &all0, &o));
    EXPECT_EQ(scan_rows(t.root), model(base_rows(0, 30), [](const Row& r) { return gone_pos(r) || r.k == 0; }));
}

TEST(IcebergDeleteWriters, EqualityDeleteIsScopedToEachDataFilesPartition) {
    Table t;
    write_eq_scoped(&t, fresh_root(temp_base(), "eq_scoped"));
    EXPECT_EQ(scan_rows(t.root), model(base_rows(0, 30), [](const Row& r) { return r.id < 12; }));
    const auto d = delete_files(t);
    EXPECT_EQ(count_eq(d), 3u) << "one file per tuple: k=0, k=NULL, k=1";
    for (const DelFile& f : d) EXPECT_EQ(f.n_part, 1u);
}

TEST(IcebergDeleteWriters, GlobalEqualityDeleteAddsAnUnpartitionedSpec) {
    Table t;
    write_eq_global(&t, fresh_root(temp_base(), "eq_global"));
    EXPECT_EQ(scan_rows(t.root), model(base_rows(0, 30), [](const Row& r) { return r.id >= 25; }));
    const Metadata* m = table_metadata(t.th);
    ASSERT_EQ(m->n_specs, 2u);
    const PartitionSpec* def = nullptr;
    for (uint32_t i = 0; i < m->n_specs; ++i) if (m->specs[i].spec_id == m->current_spec_id) def = &m->specs[i];
    ASSERT_NE(def, nullptr);
    EXPECT_EQ(def->n_fields, 1u) << "the default spec is unchanged";
    const auto d = delete_files(t);
    ASSERT_EQ(count_eq(d), 1u);
    EXPECT_NE(d[0].spec, m->current_spec_id);
    EXPECT_EQ(d[0].n_part, 0u);
}

TEST(IcebergDeleteWriters, DeleteByKeysWithoutReading) {
    Table t;
    write_keys(&t, fresh_root(temp_base(), "keys"));
    EXPECT_EQ(scan_rows(t.root), model(base_rows(0, 30), [](const Row& r) { return gone_pos(r) || r.id == 1 || r.id == 2 || r.id == 28; }));
    const auto d = delete_files(t);
    ASSERT_EQ(count_eq(d), 1u);
}

TEST(IcebergDeleteWriters, DeleteByKeysIsScopedWhenThePartitionSourceIsAKey) {
    Table t;
    write_keys_bucketed(&t, fresh_root(temp_base(), "keys_bucketed"));
    EXPECT_EQ(scan_rows(t.root), model(base_rows(0, 30), [](const Row& r) { return r.id == 4 || r.id == 17 || r.id == 26; }));
    const auto d = delete_files(t);
    ASSERT_GE(count_eq(d), 1u);
    for (const DelFile& f : d) EXPECT_EQ(f.n_part, 1u) << "each file carries its bucket";
    EXPECT_EQ(table_metadata(t.th)->n_specs, 1u) << "no unpartitioned spec was needed";
}

TEST(IcebergDeleteWriters, DeleteByKeysIsGlobalWhenThePartitionSourceIsNotAKey) {
    Table t;
    write_keys_global(&t, fresh_root(temp_base(), "keys_global"));
    EXPECT_EQ(scan_rows(t.root), model(base_rows(0, 30), [](const Row& r) { return r.id == 5 || r.id == 13 || r.id == 27; }));
    const auto d = delete_files(t);
    ASSERT_EQ(count_eq(d), 1u);
    EXPECT_EQ(d[0].n_part, 0u);
}

TEST(IcebergDeleteWriters, EqualityOverwriteIsAnUpsert) {
    Table t;
    write_upsert(&t, fresh_root(temp_base(), "upsert"));
    std::vector<Row> want = model(base_rows(0, 30), [](const Row& r) { return gone_pos(r) || r.id < 5; });
    for (const Row& r : upsert_rows()) want.push_back(r);
    std::sort(want.begin(), want.end());
    EXPECT_EQ(scan_rows(t.root), want) << "the added rows share the delete's sequence number, so its keys spare them";
}

TEST(IcebergDeleteWriters, PositionDeletesCarryTheRow) {
    Table t;
    write_pos_row(&t, fresh_root(temp_base(), "pos_row"));
    EXPECT_EQ(scan_rows(t.root), model(base_rows(0, 30), [](const Row& r) { return gone_pos(r) || r.k == 1 || r.id <= 2; }));
}

TEST(IcebergDeleteWriters, PositionDeletesCarryTheRowPerPartition) {
    Table t;
    write_pos_row_partitioned(&t, fresh_root(temp_base(), "pos_row_part"));
    EXPECT_EQ(scan_rows(t.root), model(base_rows(0, 30), [](const Row& r) { return r.id >= 23; }));
    const auto d = delete_files(t);
    size_t pos = 0;
    for (const DelFile& f : d) if (f.content == FileContent::kPositionDeletes) { ++pos; EXPECT_EQ(f.n_part, 1u); }
    EXPECT_EQ(pos, 1u) << "k=2 (25..29) is a whole file, removed; k=1 keeps a position delete of 23 and 24";
}

TEST(IcebergDeleteWriters, KeysThatCannotCompareExactlyAreRefused) {
    Table t;
    build_base(&t, fresh_root(temp_base(), "refusals"));
    Predicate p = pred_i64("id", PredicateOp::kEq, 4);
    DeleteOptions unknown = eq_opts({42});
    EXPECT_FALSE(table_delete_with(t.th, &p, &unknown));
    DeleteOptions twice = eq_opts({1, 1});
    EXPECT_FALSE(table_delete_with(t.th, &p, &twice));
    bolt::BoltBatch kb{};
    key_batch(&t.arena, &kb, "wrong_name", {1});
    DeleteOptions o = eq_opts();
    EXPECT_FALSE(table_delete_keys(t.th, &kb, &o));
    DeleteOptions pos{};
    EXPECT_FALSE(table_delete_keys(t.th, &kb, &pos)) << "a key delete is an equality delete";
    EXPECT_EQ(scan_rows(t.root), model(base_rows(0, 30), gone_pos));
}

TEST(IcebergDeleteWriters, V3DeletesAreOneMergedDeletionVectorPerFileAndAppendsCarryRowLineage) {
    Table t;
    write_v3(&t, fresh_root(temp_base(), "v3"));
    std::vector<Row> want = model(base_rows(0, 30), gone_v3);
    want.push_back(Row{100, 7, row_s(100)});
    want.push_back(Row{101, 7, row_s(101)});
    std::sort(want.begin(), want.end());
    EXPECT_EQ(scan_rows(t.root), want);
    const auto d = delete_files(t);
    size_t dvs = 0, pos = 0, eq = 0;
    std::vector<std::string> refs;
    for (const DelFile& f : d) {
        if (f.dv) { ++dvs; refs.push_back(f.ref); }
        else if (f.content == FileContent::kPositionDeletes) ++pos;
        else ++eq;
    }
    EXPECT_EQ(pos, 0u) << "a v3 writer writes no position-delete files";
    EXPECT_EQ(eq, 1u);
    std::sort(refs.begin(), refs.end());
    std::vector<std::string> want_refs{t.files[1], t.files[2]};
    std::sort(want_refs.begin(), want_refs.end());
    EXPECT_EQ(refs, want_refs) << "one DV per file: B (15) and C (22, 27..29); A was removed";
    for (const DelFile& f : d) if (f.dv && f.ref == t.files[2]) EXPECT_EQ(f.rows, 4) << "C's DV holds 22 merged with 27..29";
    const std::string md = metadata_text(t);
    EXPECT_NE(md.find("\"format-version\":3"), std::string::npos);
    EXPECT_NE(md.find("\"next-row-id\":32"), std::string::npos) << "30 base rows then 2 appended (deletes assign no ids)";
    EXPECT_NE(md.find("\"first-row-id\":30,\"added-rows\":2"), std::string::npos) << md.substr(0, 400);
}

TEST(IcebergDeleteWriters, V3RefusesWhatItCannotWriteByName) {
    Table t;
    build_base(&t, fresh_root(temp_base(), "v3_refusals"), nullptr, 3);
    Predicate p = pred_i64("id", PredicateOp::kEq, 4);
    DeleteOptions row{};
    row.position_row = true;
    EXPECT_FALSE(table_delete_with(t.th, &p, &row)) << "DVs carry no row";
    EXPECT_EQ(table_last_commit_error(t.th), CommitError::kUnimplemented);
    EXPECT_FALSE(table_compact(t.th, 64ll << 20)) << "compaction must keep row ids (G2ICE-346)";
    EXPECT_EQ(scan_rows(t.root), model(base_rows(0, 30), gone_pos));
}

// A key of ten columns (past the eight the manifest model kept before) round-trips through the manifest and applies.
TEST(IcebergDeleteWriters, WideKeysRoundTrip) {
    Table t;
    t.root = fresh_root(temp_base(), "wide_key");
    ASSERT_TRUE(filesystem_object_store_init(&t.fs, t.root.c_str(), &t.os));
    Schema s{};
    static SchemaField f[10]{};
    s.fields = f; s.cap_fields = 10u; s.n_fields = 10;
    for (uint32_t i = 0; i < 10u; ++i) {
        f[i] = SchemaField{};
        f[i].id = static_cast<int32_t>(i + 1u);
        f[i].required = true;
        std::snprintf(f[i].name, sizeof(f[i].name), "c%u", i);
        std::strcpy(f[i].type, "long");
    }
    WriteOptions wo; write_options_init(&wo);
    ASSERT_TRUE(table_create(&t.th, &t.arena, &t.os, t.root.c_str(), &s, nullptr, nullptr, &wo));
    bolt::BoltBatch b{};
    bolt::BoltBatch::init_empty(&b);
    b.arena = &t.arena; b.num_rows = 4; b.num_cols = 10;
    ASSERT_TRUE(bolt::BoltBatch::alloc_columns(&b, &t.arena, 10));
    for (uint32_t c = 0; c < 10u; ++c) {
        b.schema.add_field(f[c].name, bolt::BoltType::Int64, false);
        b.columns[b.read_epoch][c] = i64_col(&t.arena, {0, 1, 2, c == 9u ? 0 : 3}, false);   // rows 0 and 3 differ in c9 only
    }
    AppendHandle* ah = nullptr;
    ASSERT_TRUE(append_open(&ah, t.th));
    ASSERT_TRUE(append_write(ah, &b));
    ASSERT_TRUE(append_commit(ah));
    append_close(ah);
    Predicate p = pred_i64("c0", PredicateOp::kEq, 0);
    DeleteOptions o = eq_opts({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    ASSERT_TRUE(table_delete_with(t.th, &p, &o));
    const auto d = delete_files(t);
    ASSERT_EQ(d.size(), 1u);
    EXPECT_EQ(d[0].eq.size(), 10u);
    // The scan must keep the c0 = 3 row: an 8-column key would have deleted it too.
    bolt::Arena a;
    FilesystemObjectStore fs{}; ObjectStore os{};
    ASSERT_TRUE(filesystem_object_store_init(&fs, t.root.c_str(), &os));
    const std::string meta = read_file(latest_metadata_path(t.root));
    TableHandle* rh = nullptr;
    ASSERT_TRUE(iceberg_table_open_on_store(&rh, &a, &os, "", reinterpret_cast<const uint8_t*>(meta.data()),
                                            static_cast<uint32_t>(meta.size())));
    ReadOptions ro; read_options_init(&ro);
    ScanHandle* sh = nullptr;
    ASSERT_TRUE(iceberg_scan_open(&sh, rh, &ro));
    std::vector<int64_t> c0;
    for (int guard = 0; guard < 100; ++guard) {
        bolt::BoltBatch out{};
        bool eof = false;
        ASSERT_TRUE(iceberg_scan_next_batch(sh, &out, &eof));
        if (eof) break;
        for (int64_t r = 0; r < out.num_rows; ++r) c0.push_back(static_cast<const int64_t*>(out.columns[out.read_epoch][0].data)[r]);
    }
    iceberg_scan_close(sh);
    std::sort(c0.begin(), c0.end());
    EXPECT_EQ(c0, (std::vector<int64_t>{1, 2, 3}));
}

// Writes the tables scripts/iceberg_delete_writers_interop.py reads (BOLT_ICEBERG_DELETE_DIR); skips when it is unset.
TEST(IcebergDeleteWritersFixture, WriteTables) {
    const char* dir = std::getenv("BOLT_ICEBERG_DELETE_DIR");
    if (dir == nullptr || dir[0] == '\0') GTEST_SKIP() << "BOLT_ICEBERG_DELETE_DIR unset";
    const std::string base(dir);
    { Table t; write_eq_pred(&t, fresh_root(base, "eq_pred_tbl")); }
    { Table t; write_eq_scoped(&t, fresh_root(base, "eq_scoped_tbl")); }
    { Table t; write_eq_global(&t, fresh_root(base, "eq_global_tbl")); }
    { Table t; write_keys(&t, fresh_root(base, "keys_tbl")); }
    { Table t; write_keys_bucketed(&t, fresh_root(base, "keys_bucketed_tbl")); }
    { Table t; write_keys_global(&t, fresh_root(base, "keys_global_tbl")); }
    { Table t; write_upsert(&t, fresh_root(base, "upsert_tbl")); }
    { Table t; write_pos_row(&t, fresh_root(base, "pos_row_tbl")); }
    { Table t; write_pos_row_partitioned(&t, fresh_root(base, "pos_row_part_tbl")); }
    { Table t; write_v3(&t, fresh_root(base, "v3_tbl")); }
}
