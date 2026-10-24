// G2ICE-280 — row-level Iceberg operations: table_delete(pred), table_overwrite(pred, batch), table_compact.
//
// Every assertion is on VALUES read back through bolt's own scan, plus the manifest state that produced them: which
// entries are DELETED/EXISTING, which delete files exist, and whether the snapshot summary counters agree with a
// recount of the manifests made here. The external verdict (pyiceberg + DuckDB) is
// scripts/iceberg_dml_interop.py, which drives IcebergDmlFixture below.

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"
#include "bolt/lakehouse/iceberg/manifest.h"
#include "bolt/lakehouse/iceberg/manifest_avro.h"
#include "bolt/lakehouse/iceberg/metadata.h"
#include "bolt/lakehouse/iceberg/scan.h"
#include "bolt/lakehouse/iceberg/writer.h"
#include "bolt/lakehouse/object_store.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace bolt::lakehouse;
using namespace bolt::lakehouse::iceberg;

constexpr int64_t kNullK = INT64_MIN;   // marks a NULL k in the row model

// Row model: id (long, required), k (long, optional), s (string, optional).
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

std::string temp_base() {
    return (std::filesystem::temp_directory_path() / "bolt_iceberg_dml").generic_string();
}

Schema make_schema() {
    Schema s{};
    static SchemaField f[8]{};
    s.fields = f;
    s.cap_fields = 8u;
    s.n_fields = 3;
    f[0].id = 1; f[0].required = true;  std::strcpy(f[0].name, "id"); std::strcpy(f[0].type, "long");
    f[1].id = 2; f[1].required = false; std::strcpy(f[1].name, "k");  std::strcpy(f[1].type, "long");
    f[2].id = 3; f[2].required = false; std::strcpy(f[2].name, "s");  std::strcpy(f[2].type, "string");
    return s;
}

std::string row_s(int64_t id) { return "row-" + std::to_string(id) + (id % 3 == 0 ? "-with-a-long-spilled-tail" : ""); }

// A batch of `rows` (ids/k/s as the model says).
void make_batch(bolt::Arena* a, bolt::BoltBatch* b, const std::vector<Row>& rows) {
    const int64_t n = static_cast<int64_t>(rows.size());
    bolt::BoltBatch::init_empty(b);
    b->arena = a;
    b->num_rows = n;
    b->num_cols = 3;
    ASSERT_TRUE(bolt::BoltBatch::alloc_columns(b, a, 3));
    b->schema.add_field("id", bolt::BoltType::Int64, false);
    b->schema.add_field("k", bolt::BoltType::Int64, true);
    b->schema.add_field("s", bolt::BoltType::Utf8, true);
    auto* ids = a->allocate_array<int64_t>(static_cast<size_t>(n) + 1u);
    auto* ks = a->allocate_array<int64_t>(static_cast<size_t>(n) + 1u);
    auto* kv = static_cast<uint8_t*>(a->allocate_zeroed(static_cast<size_t>(n) / 8u + 2u));
    auto* svs = a->allocate_array<bolt::StringView>(static_cast<size_t>(n) + 1u);
    size_t spill_len = 0;
    for (const Row& r : rows) spill_len += r.s.size();
    auto* spill = a->allocate_array<uint8_t>(spill_len + 1u);
    uint32_t off = 0;
    bool any_null = false;
    for (int64_t i = 0; i < n; ++i) {
        const Row& r = rows[static_cast<size_t>(i)];
        ids[i] = r.id;
        ks[i] = r.k == kNullK ? 0 : r.k;
        if (r.k != kNullK) kv[i >> 3] |= static_cast<uint8_t>(1u << (i & 7)); else any_null = true;
        std::memcpy(spill + off, r.s.data(), r.s.size());
        svs[i] = bolt::StringView::from_cstr(r.s.c_str());
        if (r.s.size() > 12u) { svs[i].ref.buf_idx = 0; svs[i].ref.offset = off; }
        off += static_cast<uint32_t>(r.s.size());
    }
    auto* cols = b->columns[b->read_epoch];
    cols[0] = bolt::BoltColumn::make_flat(ids, nullptr, n, bolt::BoltType::Int64);
    cols[1] = bolt::BoltColumn::make_flat(ks, any_null ? kv : nullptr, n, bolt::BoltType::Int64);
    cols[1].stats.all_valid = !any_null;
    cols[2].type = bolt::BoltType::Utf8;
    cols[2].length = n;
    cols[2].data = svs;
    cols[2].str_overflow_base = spill;
}

// k for the three base files: A (0..9) k=0 except id 3 NULL, B (10..19) k=1, C (20..29) k=1 below 25 else 2.
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

struct Table {
    bolt::Arena arena;
    std::string root;
    FilesystemObjectStore fs{};
    ObjectStore os{};
    TableHandle* th = nullptr;
    std::vector<std::string> files;   // data files in commit order (absolute)
};

void create_table(Table* t, const std::string& root) {
    t->root = root;
    ASSERT_TRUE(filesystem_object_store_init(&t->fs, root.c_str(), &t->os));
    Schema sch = make_schema();
    WriteOptions wo; write_options_init(&wo);
    ASSERT_TRUE(table_create(&t->th, &t->arena, &t->os, root.c_str(), &sch, nullptr, nullptr, &wo));
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

// The three base files plus position deletes of id 15 (B pos 5) and id 22 (C pos 2).
void build_base(Table* t, const std::string& root) {
    create_table(t, root);
    append_rows(t, base_rows(0, 10));
    append_rows(t, base_rows(10, 20));
    append_rows(t, base_rows(20, 30));
    delete_position(t, t->files[1], 5);
    delete_position(t, t->files[2], 2);
}

std::string store_key(const std::string& root, const std::string& loc) {
    if (loc.size() > root.size() && loc.compare(0, root.size(), root) == 0) return loc.substr(root.size() + 1u);
    return loc;
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

// Every live row of the current snapshot, through bolt's scan.
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

std::vector<Row> model(const std::vector<Row>& all, bool (*gone)(const Row&)) {
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

Predicate pred_str(const char* col, PredicateOp op, const std::string& v) {
    Predicate p{};
    std::strncpy(p.column, col, sizeof(p.column) - 1u);
    p.op = op;
    p.value.type = bolt::BoltType::Utf8;
    std::memcpy(p.value.str, v.data(), v.size());
    p.value.str_len = static_cast<uint32_t>(v.size());
    return p;
}

// Manifest state of one snapshot, recounted from its manifest list and manifests.
struct Entry { ManifestStatus status; FileContent content; int64_t snapshot_id, rows, size; std::string path; };
struct Recount {
    std::vector<Entry> entries;
    std::map<std::string, int64_t> c;   // summary keys recomputed from the entries
};

Recount recount(const std::string& root, const Snapshot& snap) {
    Recount r;
    bolt::Arena a;
    const std::string ml = read_file(root + "/" + store_key(root, snap.manifest_list));
    ManifestListEntry* mle = nullptr; uint32_t nm = 0;
    EXPECT_TRUE(manifest_list_parse_avro_grow(reinterpret_cast<const uint8_t*>(ml.data()), ml.size(), &a, 4u, &mle, &nm));
    for (uint32_t m = 0; m < nm; ++m) {
        const std::string body = read_file(root + "/" + store_key(root, mle[m].manifest_path));
        DataFileRef* e = nullptr; uint32_t ne = 0;
        EXPECT_TRUE(manifest_parse_avro_grow(reinterpret_cast<const uint8_t*>(body.data()), body.size(), &a,
                                             mle[m].partition_spec_id, 4u, &e, &ne));
        for (uint32_t i = 0; i < ne; ++i)
            r.entries.push_back(Entry{e[i].status, e[i].content, e[i].snapshot_id, e[i].stats.record_count,
                                      e[i].stats.file_size_in_bytes, e[i].file_path});
    }
    auto& c = r.c;
    for (const char* k : {"added-data-files", "deleted-data-files", "added-delete-files", "removed-delete-files",
                          "added-position-delete-files", "removed-position-delete-files", "added-records",
                          "deleted-records", "added-position-deletes", "removed-position-deletes", "added-files-size",
                          "removed-files-size", "total-records", "total-data-files", "total-delete-files",
                          "total-position-deletes", "total-equality-deletes", "total-files-size"})
        c[k] = 0;
    for (const Entry& e : r.entries) {
        const bool data = e.content == FileContent::kData;
        const bool mine = e.snapshot_id == snap.snapshot_id;
        if (e.status == ManifestStatus::kAdded && mine) {
            c[data ? "added-data-files" : "added-delete-files"] += 1;
            if (!data) { c["added-position-delete-files"] += 1; c["added-position-deletes"] += e.rows; }
            if (data) c["added-records"] += e.rows;
            c["added-files-size"] += e.size;
        }
        if (e.status == ManifestStatus::kDeleted && mine) {
            c[data ? "deleted-data-files" : "removed-delete-files"] += 1;
            if (!data) { c["removed-position-delete-files"] += 1; c["removed-position-deletes"] += e.rows; }
            if (data) c["deleted-records"] += e.rows;
            c["removed-files-size"] += e.size;
        }
        if (e.status == ManifestStatus::kDeleted) continue;
        c["total-files-size"] += e.size;
        if (data) { c["total-data-files"] += 1; c["total-records"] += e.rows; }
        else      { c["total-delete-files"] += 1; c["total-position-deletes"] += e.rows; }
    }
    return r;
}

// The snapshot's summary as metadata.json records it (string values).
std::map<std::string, std::string> summary_of(const std::string& root, int64_t snapshot_id) {
    std::map<std::string, std::string> out;
    const std::string doc = read_file(latest_metadata_path(root));
    const std::string key = "\"snapshot-id\":" + std::to_string(snapshot_id) + ",";
    const size_t at = doc.find(key);
    if (at == std::string::npos) return out;
    const size_t sum = doc.find("\"summary\":{", at);
    const size_t end = doc.find('}', sum);
    std::string body = doc.substr(sum + 11, end - sum - 11);
    size_t p = 0;
    while (p < body.size()) {
        const size_t k0 = body.find('"', p);
        if (k0 == std::string::npos) break;
        const size_t k1 = body.find('"', k0 + 1);
        const size_t v0 = body.find('"', k1 + 1);
        const size_t v1 = body.find('"', v0 + 1);
        out[body.substr(k0 + 1, k1 - k0 - 1)] = body.substr(v0 + 1, v1 - v0 - 1);
        p = v1 + 1;
    }
    return out;
}

// Every counter the recount derives must be in the summary with that value (absent means 0 for a delta).
void expect_summary_matches(const std::string& root, const Snapshot& snap) {
    const Recount r = recount(root, snap);
    const auto s = summary_of(root, snap.snapshot_id);
    for (const auto& [k, v] : r.c) {
        const bool total = k.rfind("total-", 0) == 0;
        auto it = s.find(k);
        if (it == s.end()) {
            EXPECT_FALSE(total) << k << " missing from snapshot " << snap.snapshot_id;
            EXPECT_EQ(v, 0) << k << " missing but the manifests count " << v;
            continue;
        }
        EXPECT_EQ(std::stoll(it->second), v) << k << " of snapshot " << snap.snapshot_id;
    }
}

const Snapshot& current(const TableHandle* th) {
    const Metadata* m = table_metadata(th);
    for (uint32_t i = 0; i < m->n_snapshots; ++i)
        if (m->snapshots[i].snapshot_id == m->current_snapshot_id) return m->snapshots[i];
    return m->snapshots[0];
}

std::string op_of(const std::string& root, int64_t id) { return summary_of(root, id)["operation"]; }

const Entry* find_entry(const Recount& r, const std::string& path, ManifestStatus st) {
    for (const Entry& e : r.entries) if (e.path == path && e.status == st) return &e;
    return nullptr;
}

}  // namespace

TEST(IcebergDml, DeleteDropsWholeFilesWritesPositionDeletesForPartialOnesAndLeavesTheRest) {
    Table t;
    build_base(&t, fresh_root(temp_base(), "delete"));
    const std::vector<Row> all = base_rows(0, 30);
    const uint32_t before = table_metadata(t.th)->n_snapshots;
    Predicate p = pred_i64("k", PredicateOp::kEq, 1);
    ASSERT_TRUE(table_delete(t.th, &p));
    ASSERT_EQ(table_last_commit_error(t.th), CommitError::kNone);
    ASSERT_EQ(table_metadata(t.th)->n_snapshots, before + 1u);
    EXPECT_EQ(scan_rows(t.root), model(all, [](const Row& r) { return r.id == 15 || r.id == 22 || r.k == 1; }));
    const Snapshot& s = current(t.th);
    EXPECT_EQ(op_of(t.root, s.snapshot_id), "delete");
    const Recount rc = recount(t.root, s);
    // B (every live row matched) is dropped; A (no match) and C (partial) stay live.
    ASSERT_NE(find_entry(rc, t.files[1], ManifestStatus::kDeleted), nullptr);
    EXPECT_EQ(find_entry(rc, t.files[1], ManifestStatus::kDeleted)->snapshot_id, s.snapshot_id);
    EXPECT_NE(find_entry(rc, t.files[0], ManifestStatus::kAdded), nullptr) << "A's manifest is carried untouched";
    EXPECT_NE(find_entry(rc, t.files[2], ManifestStatus::kAdded), nullptr);
    EXPECT_EQ(rc.c.at("added-position-deletes"), 4) << "C's ids 20,21,23,24 (22 was already deleted)";
    EXPECT_EQ(rc.c.at("deleted-data-files"), 1);
    EXPECT_EQ(rc.c.at("deleted-records"), 10);
    expect_summary_matches(t.root, s);
}

TEST(IcebergDml, DeleteMatchingNothingCommitsNothing) {
    Table t;
    build_base(&t, fresh_root(temp_base(), "nomatch"));
    const uint32_t before = table_metadata(t.th)->n_snapshots;
    const int64_t cur = table_metadata(t.th)->current_snapshot_id;
    Predicate p = pred_i64("k", PredicateOp::kEq, 99);
    ASSERT_TRUE(table_delete(t.th, &p));
    EXPECT_EQ(table_metadata(t.th)->n_snapshots, before);
    EXPECT_EQ(table_metadata(t.th)->current_snapshot_id, cur);
    Predicate bad = pred_i64("nope", PredicateOp::kEq, 1);
    EXPECT_FALSE(table_delete(t.th, &bad)) << "an unknown column is refused, not treated as matching nothing";
}

TEST(IcebergDml, NullsNeverMatchAComparisonAndIsNullFindsThem) {
    Table t;
    build_base(&t, fresh_root(temp_base(), "nulls"));
    const std::vector<Row> all = base_rows(0, 30);
    Predicate ne = pred_i64("k", PredicateOp::kNe, 0);   // A's id 3 is NULL: not deleted by k != 0
    ASSERT_TRUE(table_delete(t.th, &ne));
    EXPECT_EQ(scan_rows(t.root), model(all, [](const Row& r) { return r.id == 15 || r.id == 22 || (r.k != kNullK && r.k != 0); }));
    Predicate isnull{};
    std::strcpy(isnull.column, "k");
    isnull.op = PredicateOp::kIsNull;
    ASSERT_TRUE(table_delete(t.th, &isnull));
    EXPECT_EQ(scan_rows(t.root), model(all, [](const Row& r) { return r.id >= 10 || r.k == kNullK; }));
    expect_summary_matches(t.root, current(t.th));
}

TEST(IcebergDml, StringPredicateComparesBytesIncludingSpilledValues) {
    Table t;
    build_base(&t, fresh_root(temp_base(), "string"));
    const std::vector<Row> all = base_rows(0, 30);
    Predicate p = pred_str("s", PredicateOp::kGe, "row-27");   // "row-27-with-..." (spilled) and row-28, row-29
    ASSERT_TRUE(table_delete(t.th, &p));
    EXPECT_EQ(scan_rows(t.root), model(all, [](const Row& r) { return r.id == 15 || r.id == 22 || r.s >= "row-27"; }));
    Predicate eq = pred_str("s", PredicateOp::kEq, row_s(12));   // a spilled value
    ASSERT_TRUE(table_delete(t.th, &eq));
    EXPECT_EQ(scan_rows(t.root), model(all, [](const Row& r) { return r.id == 15 || r.id == 22 || r.id == 12 || r.s >= "row-27"; }));
}

TEST(IcebergDml, OverwriteDeletesAndAppendsInOneSnapshot) {
    Table t;
    build_base(&t, fresh_root(temp_base(), "overwrite"));
    std::vector<Row> add;
    for (int64_t id = 100; id < 105; ++id) add.push_back(Row{id, 7, row_s(id)});
    bolt::BoltBatch b{};
    make_batch(&t.arena, &b, add);
    const uint32_t before = table_metadata(t.th)->n_snapshots;
    Predicate p = pred_i64("k", PredicateOp::kEq, 1);
    ASSERT_TRUE(table_overwrite(t.th, &p, &b));
    ASSERT_EQ(table_metadata(t.th)->n_snapshots, before + 1u);
    std::vector<Row> all = base_rows(0, 30);
    all.insert(all.end(), add.begin(), add.end());
    EXPECT_EQ(scan_rows(t.root), model(all, [](const Row& r) { return r.id == 15 || r.id == 22 || r.k == 1; }));
    const Snapshot& s = current(t.th);
    EXPECT_EQ(op_of(t.root, s.snapshot_id), "overwrite");
    const Recount rc = recount(t.root, s);
    EXPECT_EQ(rc.c.at("added-data-files"), 1);
    EXPECT_EQ(rc.c.at("added-records"), 5);
    EXPECT_EQ(rc.c.at("deleted-data-files"), 1);
    expect_summary_matches(t.root, s);
}

TEST(IcebergDml, CompactionAppliesPendingDeletesAndKeepsTheRows) {
    Table t;
    create_table(&t, fresh_root(temp_base(), "compact"));
    std::vector<Row> all;
    for (int f = 0; f < 4; ++f) {
        const std::vector<Row> rows = base_rows(f * 5, f * 5 + 5);
        append_rows(&t, rows);
        all.insert(all.end(), rows.begin(), rows.end());
    }
    delete_position(&t, t.files[0], 1);   // id 1
    delete_position(&t, t.files[2], 4);   // id 14
    Predicate p = pred_i64("id", PredicateOp::kEq, 17);
    ASSERT_TRUE(table_delete(t.th, &p));  // a third position-delete file
    auto gone = [](const Row& r) { return r.id == 1 || r.id == 14 || r.id == 17; };
    const std::vector<Row> want = model(all, gone);
    ASSERT_EQ(scan_rows(t.root), want);
    ASSERT_TRUE(table_compact(t.th, 64ll << 20));
    ASSERT_EQ(table_last_commit_error(t.th), CommitError::kNone);
    const Snapshot& s = current(t.th);
    EXPECT_EQ(op_of(t.root, s.snapshot_id), "replace");
    EXPECT_EQ(scan_rows(t.root), want) << "compaction changed the table's rows";
    const Recount rc = recount(t.root, s);
    EXPECT_EQ(rc.c.at("deleted-data-files"), 4);
    EXPECT_EQ(rc.c.at("added-data-files"), 1);
    EXPECT_EQ(rc.c.at("added-records"), 17);
    EXPECT_EQ(rc.c.at("removed-delete-files"), 3) << "every position delete is now applied";
    EXPECT_EQ(rc.c.at("total-delete-files"), 0);
    EXPECT_EQ(rc.c.at("total-records"), 17);
    expect_summary_matches(t.root, s);
    // A table of one big file is left alone: nothing to coalesce, no deletes.
    const uint32_t n = table_metadata(t.th)->n_snapshots;
    ASSERT_TRUE(table_compact(t.th, 64ll << 20));
    EXPECT_EQ(table_metadata(t.th)->n_snapshots, n);
}

TEST(IcebergDml, CompactionKeepsADeleteFileThatStillAppliesToAKeptFile) {
    Table t;
    create_table(&t, fresh_root(temp_base(), "compact_keep"));
    append_rows(&t, base_rows(0, 5));
    append_rows(&t, base_rows(5, 10));
    append_rows(&t, base_rows(10, 3010));   // big: not a candidate
    // One delete file naming a row of file 0 (rewritten) and a row of file 2 (kept).
    PositionDeleteEntry e[2]{};
    std::strcpy(e[0].file_path, t.files[0].c_str()); e[0].pos = 0;
    std::strcpy(e[1].file_path, t.files[2].c_str()); e[1].pos = 0;
    ASSERT_TRUE(table_delete_positions(t.th, e, 2));
    const std::vector<Row> want = model(base_rows(0, 3010), [](const Row& r) { return r.id == 0 || r.id == 10; });
    const Recount before = recount(t.root, current(t.th));
    const Entry* big = find_entry(before, t.files[2], ManifestStatus::kAdded);
    ASSERT_NE(big, nullptr);
    ASSERT_TRUE(table_compact(t.th, 2 * big->size));   // candidates: size < big->size
    const Snapshot& s = current(t.th);
    EXPECT_EQ(op_of(t.root, s.snapshot_id), "replace");
    EXPECT_EQ(scan_rows(t.root), want);
    const Recount rc = recount(t.root, s);
    EXPECT_EQ(rc.c.at("deleted-data-files"), 2);
    EXPECT_EQ(rc.c.at("removed-delete-files"), 0) << "the delete file still deletes a row of the kept file";
    EXPECT_EQ(rc.c.at("total-delete-files"), 1);
    EXPECT_NE(find_entry(rc, t.files[2], ManifestStatus::kAdded), nullptr);
    expect_summary_matches(t.root, s);
}

TEST(IcebergDml, ACommitOnAFileAnotherWriterRemovedConflicts) {
    Table t;
    build_base(&t, fresh_root(temp_base(), "conflict"));
    TableHandle* stale = nullptr;
    bolt::Arena a2;
    ASSERT_TRUE(table_open(&stale, &a2, &t.os, t.root.c_str()));
    Predicate p = pred_i64("k", PredicateOp::kEq, 1);
    ASSERT_TRUE(table_delete(t.th, &p));           // removes B
    const int64_t cur = table_metadata(t.th)->current_snapshot_id;
    EXPECT_FALSE(table_delete(stale, &p));
    EXPECT_EQ(table_last_commit_error(stale), CommitError::kConflict);
    TableHandle* stale2 = nullptr;
    bolt::Arena a3;
    ASSERT_TRUE(table_open(&stale2, &a3, &t.os, t.root.c_str()));
    Predicate all = pred_i64("id", PredicateOp::kGe, 0);
    ASSERT_TRUE(table_delete(t.th, &all));         // removes every file
    EXPECT_FALSE(table_compact(stale2, 64ll << 20));
    EXPECT_EQ(table_last_commit_error(stale2), CommitError::kConflict);
    // Nothing of the losing commits is visible.
    TableHandle* fresh = nullptr;
    bolt::Arena a4;
    ASSERT_TRUE(table_open(&fresh, &a4, &t.os, t.root.c_str()));
    EXPECT_NE(table_metadata(fresh)->current_snapshot_id, cur);
    EXPECT_EQ(table_metadata(fresh)->current_snapshot_id, table_metadata(t.th)->current_snapshot_id);
    EXPECT_TRUE(scan_rows(t.root).empty());
}

TEST(IcebergDml, SummariesSurviveAReopenedHandle) {
    Table t;
    build_base(&t, fresh_root(temp_base(), "reopen"));
    const std::vector<int64_t> ids = [&] {
        std::vector<int64_t> v;
        const Metadata* m = table_metadata(t.th);
        for (uint32_t i = 0; i < m->n_snapshots; ++i) v.push_back(m->snapshots[i].snapshot_id);
        return v;
    }();
    TableHandle* h = nullptr;
    bolt::Arena a2;
    ASSERT_TRUE(table_open(&h, &a2, &t.os, t.root.c_str()));
    Predicate p = pred_i64("k", PredicateOp::kEq, 2);
    ASSERT_TRUE(table_delete(h, &p));
    for (int64_t id : ids) {
        EXPECT_FALSE(summary_of(t.root, id)["total-records"].empty()) << "snapshot " << id << " lost its summary";
    }
    for (uint32_t i = 0; i < table_metadata(h)->n_snapshots; ++i) expect_summary_matches(t.root, table_metadata(h)->snapshots[i]);
}

TEST(IcebergDml, NullPredicateIsRefused) {
    Table t;
    build_base(&t, fresh_root(temp_base(), "nullpred"));
    EXPECT_FALSE(table_delete(t.th, nullptr));
    EXPECT_EQ(table_last_commit_error(t.th), CommitError::kFailed);
}

// Writes the tables scripts/iceberg_dml_interop.py reads (BOLT_ICEBERG_DML_DIR); skips when it is unset.
TEST(IcebergDmlFixture, WriteTables) {
    const char* dir = std::getenv("BOLT_ICEBERG_DML_DIR");
    if (dir == nullptr || dir[0] == '\0') GTEST_SKIP() << "BOLT_ICEBERG_DML_DIR unset";
    const std::string base(dir);
    {
        Table t;
        build_base(&t, fresh_root(base, "delete_tbl"));
        Predicate p = pred_i64("k", PredicateOp::kEq, 1);
        ASSERT_TRUE(table_delete(t.th, &p));
        Predicate none = pred_i64("k", PredicateOp::kEq, 99);
        ASSERT_TRUE(table_delete(t.th, &none));
        Predicate s = pred_str("s", PredicateOp::kEq, row_s(27));
        ASSERT_TRUE(table_delete(t.th, &s));
    }
    {
        Table t;
        build_base(&t, fresh_root(base, "overwrite_tbl"));
        std::vector<Row> add;
        for (int64_t id = 100; id < 105; ++id) add.push_back(Row{id, 7, row_s(id)});
        bolt::BoltBatch b{};
        make_batch(&t.arena, &b, add);
        Predicate p = pred_i64("k", PredicateOp::kEq, 1);
        ASSERT_TRUE(table_overwrite(t.th, &p, &b));
    }
    {
        Table t;
        create_table(&t, fresh_root(base, "compact_tbl"));
        for (int f = 0; f < 4; ++f) append_rows(&t, base_rows(f * 5, f * 5 + 5));
        delete_position(&t, t.files[0], 1);
        delete_position(&t, t.files[2], 4);
        Predicate p = pred_i64("id", PredicateOp::kEq, 17);
        ASSERT_TRUE(table_delete(t.th, &p));
        ASSERT_TRUE(table_compact(t.th, 64ll << 20));
    }
}
