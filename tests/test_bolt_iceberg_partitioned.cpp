// G2ICE-292 — partitioned Iceberg tables: append splits rows into one data file per partition tuple, the tuple and
// spec id are recorded in the manifest entries (with the manifest list's partition summaries), delete / overwrite /
// compaction work per partition, specs evolve, and bolt's scan prunes on the tuples.
//
// Every value assertion reads the table back through bolt's own scan and compares against a model computed here;
// expected partition values are recomputed from the rows with the transform kernels (whose own spec vectors are in
// test_bolt_iceberg_transforms.cpp). The external verdict (pyiceberg + DuckDB, pyiceberg's own transforms, pyiceberg
// plan_files pruning) is scripts/iceberg_partition_interop.py, which drives IcebergPartitionFixture below.

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"
#include "bolt/lakehouse/iceberg/manifest.h"
#include "bolt/lakehouse/iceberg/manifest_avro.h"
#include "bolt/lakehouse/iceberg/metadata.h"
#include "bolt/lakehouse/iceberg/partition_values.h"
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

constexpr int64_t kNull = INT64_MIN;   // a NULL in the row model
constexpr int64_t kDayUs = INT64_C(86400000000);

// id long (required), k long, s string, d date, ts timestamp, i int.
struct Row {
    int64_t id, k;
    std::string s;      // "\x01" = NULL
    int64_t d, ts, i;
    bool operator<(const Row& o) const { return id < o.id; }
    bool operator==(const Row& o) const {
        return id == o.id && k == o.k && s == o.s && d == o.d && ts == o.ts && i == o.i;
    }
};
const std::string kNullS = "\x01";

Row make_row(int64_t id) {
    Row r{};
    r.id = id;
    r.k = id % 7 == 3 ? kNull : id % 4;
    r.s = id % 11 == 5 ? kNullS : (id % 3 == 0 ? "caf\xC3\xA9-" : "row-") + std::to_string(id % 9) + (id % 2 ? "-tail-long-enough-to-spill" : "");
    r.d = id % 13 == 7 ? kNull : -400 + id * 37;                          // dates on both sides of the epoch
    r.ts = id % 17 == 9 ? kNull : (r.d == kNull ? 0 : r.d) * kDayUs + (id % 24) * INT64_C(3600000000) - 1;
    r.i = id % 19 == 2 ? kNull : static_cast<int64_t>(-50 + id * 3);
    return r;
}

std::vector<Row> rows_of(int64_t lo, int64_t hi) {
    std::vector<Row> v;
    for (int64_t id = lo; id < hi; ++id) v.push_back(make_row(id));
    return v;
}

std::string fresh_root(const std::string& base, const char* tag) {
    std::filesystem::path p = std::filesystem::path(base) / tag;
    std::error_code ec;
    std::filesystem::remove_all(p, ec);
    std::filesystem::create_directories(p, ec);
    return p.generic_string();
}

std::string temp_base() { return (std::filesystem::temp_directory_path() / "bolt_iceberg_part").generic_string(); }

Schema make_schema() {
    Schema s{};
    static SchemaField f[8]{};
    s.fields = f;
    s.cap_fields = 8u;
    s.n_fields = 6;
    const char* names[] = {"id", "k", "s", "d", "ts", "i"};
    const char* types[] = {"long", "long", "string", "date", "timestamp", "int"};
    for (uint32_t i = 0; i < 6u; ++i) {
        f[i] = SchemaField{};
        f[i].id = static_cast<int32_t>(i + 1u);
        f[i].required = i == 0u;
        std::strcpy(f[i].name, names[i]);
        std::strcpy(f[i].type, types[i]);
    }
    return s;
}

template <typename T>
void fixed_col(bolt::Arena* a, bolt::BoltColumn* c, const std::vector<Row>& rows, int64_t Row::*m, bolt::BoltType t) {
    const int64_t n = static_cast<int64_t>(rows.size());
    auto* v = a->allocate_array<T>(static_cast<size_t>(n) + 1u);
    auto* valid = static_cast<uint8_t*>(a->allocate_zeroed(static_cast<size_t>(n) / 8u + 2u));
    bool any_null = false;
    for (int64_t r = 0; r < n; ++r) {
        const int64_t x = rows[static_cast<size_t>(r)].*m;
        v[r] = x == kNull ? T{} : static_cast<T>(x);
        if (x != kNull) valid[r >> 3] |= static_cast<uint8_t>(1u << (r & 7)); else any_null = true;
    }
    *c = bolt::BoltColumn::make_flat(v, any_null ? valid : nullptr, n, t);
    c->stats.all_valid = !any_null;
}

void make_batch(bolt::Arena* a, bolt::BoltBatch* b, const std::vector<Row>& rows) {
    const int64_t n = static_cast<int64_t>(rows.size());
    bolt::BoltBatch::init_empty(b);
    b->arena = a;
    b->num_rows = n;
    b->num_cols = 6;
    ASSERT_TRUE(bolt::BoltBatch::alloc_columns(b, a, 6));
    b->schema.add_field("id", bolt::BoltType::Int64, false);
    b->schema.add_field("k", bolt::BoltType::Int64, true);
    b->schema.add_field("s", bolt::BoltType::Utf8, true);
    b->schema.add_field("d", bolt::BoltType::Date32, true);
    b->schema.add_field("ts", bolt::BoltType::Timestamp, true);
    b->schema.add_field("i", bolt::BoltType::Int32, true);
    auto* cols = b->columns[b->read_epoch];
    fixed_col<int64_t>(a, &cols[0], rows, &Row::id, bolt::BoltType::Int64);
    fixed_col<int64_t>(a, &cols[1], rows, &Row::k, bolt::BoltType::Int64);
    fixed_col<int32_t>(a, &cols[3], rows, &Row::d, bolt::BoltType::Date32);
    fixed_col<int64_t>(a, &cols[4], rows, &Row::ts, bolt::BoltType::Timestamp);
    fixed_col<int32_t>(a, &cols[5], rows, &Row::i, bolt::BoltType::Int32);
    auto* svs = a->allocate_array<bolt::StringView>(static_cast<size_t>(n) + 1u);
    auto* valid = static_cast<uint8_t*>(a->allocate_zeroed(static_cast<size_t>(n) / 8u + 2u));
    size_t spill_len = 0;
    for (const Row& r : rows) spill_len += r.s.size();
    auto* spill = a->allocate_array<uint8_t>(spill_len + 1u);
    uint32_t off = 0;
    bool any_null = false;
    for (int64_t i = 0; i < n; ++i) {
        const std::string& s = rows[static_cast<size_t>(i)].s;
        if (s == kNullS) { any_null = true; svs[i] = bolt::StringView{}; continue; }
        valid[i >> 3] |= static_cast<uint8_t>(1u << (i & 7));
        std::memcpy(spill + off, s.data(), s.size());
        svs[i] = bolt::StringView::from_cstr(s.c_str());
        if (s.size() > 12u) { svs[i].ref.buf_idx = 0; svs[i].ref.offset = off; }
        off += static_cast<uint32_t>(s.size());
    }
    cols[2].type = bolt::BoltType::Utf8;
    cols[2].length = n;
    cols[2].data = svs;
    cols[2].validity = any_null ? valid : nullptr;
    cols[2].stats.all_valid = !any_null;
    cols[2].str_overflow_base = spill;
}

struct Table {
    bolt::Arena arena;
    std::string root;
    FilesystemObjectStore fs{};
    ObjectStore os{};
    TableHandle* th = nullptr;
};

PartitionField pf(int32_t source, const char* transform, const char* name) {
    PartitionField f{};
    f.source_id = source;
    f.field_id = 0;                                    // let the writer number it
    f.transform = transform_parse(transform, static_cast<uint32_t>(std::strlen(transform)));
    std::strncpy(f.name, name, sizeof(f.name) - 1u);
    return f;
}

PartitionSpec spec_of(std::initializer_list<PartitionField> fields) {
    PartitionSpec s{};
    s.spec_id = 0;
    for (const PartitionField& f : fields) s.fields[s.n_fields++] = f;
    return s;
}

void create_table(Table* t, const std::string& root, const PartitionSpec& spec) {
    t->root = root;
    ASSERT_TRUE(filesystem_object_store_init(&t->fs, root.c_str(), &t->os));
    Schema sch = make_schema();
    WriteOptions wo; write_options_init(&wo);
    ASSERT_TRUE(table_create(&t->th, &t->arena, &t->os, root.c_str(), &sch, &spec, nullptr, &wo));
}

void append_rows(Table* t, const std::vector<Row>& rows, uint32_t n_batches = 1) {
    AppendHandle* ah = nullptr;
    ASSERT_TRUE(append_open(&ah, t->th));
    const size_t per = (rows.size() + n_batches - 1u) / n_batches;
    for (size_t at = 0; at < rows.size(); at += per) {
        auto* b = new (t->arena.allocate(sizeof(bolt::BoltBatch), alignof(bolt::BoltBatch))) bolt::BoltBatch();
        make_batch(&t->arena, b, std::vector<Row>(rows.begin() + static_cast<long>(at),
                                                  rows.begin() + static_cast<long>(std::min(rows.size(), at + per))));
        ASSERT_TRUE(append_write(ah, b));
    }
    ASSERT_TRUE(append_commit(ah)) << static_cast<int>(table_last_commit_error(t->th));
    append_close(ah);
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

std::string store_key(const std::string& root, const std::string& loc) {
    if (loc.size() > root.size() && loc.compare(0, root.size(), root) == 0) return loc.substr(root.size() + 1u);
    return loc;
}

// The live manifest entries of the current snapshot, with the manifest-list entry each came from.
struct LiveFile {
    DataFileRef ref;
    ManifestListEntry mle;
};

std::vector<LiveFile> live_files(Table* t, bool include_deleted = false) {
    std::vector<LiveFile> out;
    const Metadata* m = table_metadata(t->th);
    const Snapshot* snap = nullptr;
    for (uint32_t i = 0; i < m->n_snapshots; ++i)
        if (m->snapshots[i].snapshot_id == m->current_snapshot_id) snap = &m->snapshots[i];
    if (snap == nullptr) return out;
    bolt::Arena a;
    const std::string ml = read_file(t->root + "/" + store_key(t->root, snap->manifest_list));
    ManifestListEntry* mle = nullptr; uint32_t nm = 0;
    EXPECT_TRUE(manifest_list_parse_avro_grow(reinterpret_cast<const uint8_t*>(ml.data()), ml.size(), &a, 4u, &mle, &nm));
    for (uint32_t i = 0; i < nm; ++i) {
        const std::string body = read_file(t->root + "/" + store_key(t->root, mle[i].manifest_path));
        DataFileRef* e = nullptr; uint32_t ne = 0;
        EXPECT_TRUE(manifest_parse_avro_grow(reinterpret_cast<const uint8_t*>(body.data()), body.size(), &a,
                                             mle[i].partition_spec_id, 4u, &e, &ne));
        for (uint32_t j = 0; j < ne; ++j)
            if (include_deleted || e[j].status != ManifestStatus::kDeleted) out.push_back(LiveFile{e[j], mle[i]});
    }
    return out;
}

Row read_row(const bolt::BoltColumn* cols, int64_t r) {
    Row row{};
    auto get = [&](int c, int w) -> int64_t {
        if (cols[c].is_null(r)) return kNull;
        return w == 8 ? static_cast<const int64_t*>(cols[c].data)[r] : static_cast<const int32_t*>(cols[c].data)[r];
    };
    row.id = get(0, 8); row.k = get(1, 8); row.d = get(3, 4); row.ts = get(4, 8); row.i = get(5, 4);
    if (cols[2].is_null(r)) row.s = kNullS;
    else {
        const uint8_t* d = nullptr; int32_t dl = 0;
        cols[2].utf8_at(r, &d, &dl);
        row.s.assign(reinterpret_cast<const char*>(d), static_cast<size_t>(dl));
    }
    return row;
}

// Rows the scan returns, grouped by the data file they came from. `preds` filter files only (never rows).
std::map<std::string, std::vector<Row>> scan_by_file(const std::string& root, const Predicate* preds = nullptr,
                                                     uint32_t n_preds = 0) {
    std::map<std::string, std::vector<Row>> out;
    bolt::Arena a;
    FilesystemObjectStore fs{}; ObjectStore os{};
    EXPECT_TRUE(filesystem_object_store_init(&fs, root.c_str(), &os));
    const std::string meta = read_file(latest_metadata_path(root));
    TableHandle* rh = nullptr;
    EXPECT_TRUE(iceberg_table_open_on_store(&rh, &a, &os, "", reinterpret_cast<const uint8_t*>(meta.data()),
                                            static_cast<uint32_t>(meta.size())));
    if (rh == nullptr) return out;
    ReadOptions ro; read_options_init(&ro);
    for (uint32_t i = 0; i < n_preds; ++i) ro.predicates[i] = preds[i];
    ro.n_predicates = n_preds;
    ScanHandle* sh = nullptr;
    EXPECT_TRUE(iceberg_scan_open(&sh, rh, &ro));
    if (sh == nullptr) return out;
    for (int guard = 0; guard < 100000; ++guard) {
        bolt::BoltBatch b{};
        bool eof = false;
        EXPECT_TRUE(iceberg_scan_next_batch(sh, &b, &eof));
        if (eof) break;
        IcebergScanGroup g{};
        EXPECT_TRUE(iceberg_scan_last_group(sh, &g));
        auto& v = out[g.file_path];
        for (int64_t r = 0; r < b.num_rows; ++r) v.push_back(read_row(b.columns[b.read_epoch], r));
    }
    iceberg_scan_close(sh);
    return out;
}

std::vector<Row> scan_rows(const std::string& root) {
    std::vector<Row> all;
    for (auto& [p, v] : scan_by_file(root)) all.insert(all.end(), v.begin(), v.end());
    std::sort(all.begin(), all.end());
    return all;
}

std::vector<Row> model(std::vector<Row> v, bool (*gone)(const Row&)) {
    v.erase(std::remove_if(v.begin(), v.end(), gone), v.end());
    std::sort(v.begin(), v.end());
    return v;
}

// The expected partition value of `r` for one field, computed with the kernels directly; "N" = null.
std::string expect_value(const PartitionField& f, const Row& r) {
    const TransformKind k = f.transform.kind;
    const int32_t p = f.transform.param;
    auto ints = [&](int64_t v) { return v == kNull ? std::string("N") : std::to_string(v); };
    switch (f.source_id) {
        case 1: case 2: {
            const int64_t v = f.source_id == 1 ? r.id : r.k;
            if (v == kNull || k == TransformKind::kVoid) return "N";
            if (k == TransformKind::kIdentity) return ints(v);
            if (k == TransformKind::kBucket) return ints(bucket_i64(v, p));
            return ints(truncate_i64(v, p));
        }
        case 6:
            if (r.i == kNull || k == TransformKind::kVoid) return "N";
            if (k == TransformKind::kIdentity) return ints(r.i);
            if (k == TransformKind::kBucket) return ints(bucket_i32(static_cast<int32_t>(r.i), p));
            return ints(truncate_i32(static_cast<int32_t>(r.i), p));
        case 3: {
            if (r.s == kNullS || k == TransformKind::kVoid) return "N";
            if (k == TransformKind::kIdentity) return "s:" + r.s;
            if (k == TransformKind::kBucket) return ints(bucket_str(r.s.data(), static_cast<uint32_t>(r.s.size()), p));
            return "s:" + r.s.substr(0, truncate_utf8_len(r.s.data(), static_cast<uint32_t>(r.s.size()), p));
        }
        case 4: case 5: {
            const int64_t v = f.source_id == 4 ? r.d : r.ts;
            const TimeUnit u = f.source_id == 4 ? TimeUnit::kDays : TimeUnit::kMicros;
            if (v == kNull || k == TransformKind::kVoid) return "N";
            switch (k) {
                case TransformKind::kIdentity: return ints(v);
                case TransformKind::kYear: return ints(transform_year(v, u));
                case TransformKind::kMonth: return ints(transform_month(v, u));
                case TransformKind::kDay: return ints(transform_day(v, u));
                case TransformKind::kHour: return ints(transform_hour(v, u));
                case TransformKind::kBucket: return ints(bucket_i64(v, p));
                default: return "?";
            }
        }
        default: return "?";
    }
}

std::string recorded_value(const PartitionValue& v) {
    if (v.is_null) return "N";
    if (v.is_str) return "s:" + std::string(v.str, partition_value_bytes_len(v));
    return std::to_string(v.i64);
}

std::string expect_tuple(const PartitionSpec& spec, const Row& r) {
    std::string t;
    for (uint32_t i = 0; i < spec.n_fields; ++i) t += expect_value(spec.fields[i], r) + "|";
    return t;
}

std::string recorded_tuple(const DataFileRef& f) {
    std::string t;
    for (uint32_t i = 0; i < f.n_partition; ++i) t += recorded_value(f.partition[i]) + "|";
    return t;
}

const PartitionSpec* spec_by_id(const Table& t, int32_t id) {
    const Metadata* m = table_metadata(t.th);
    for (uint32_t i = 0; i < m->n_specs; ++i) if (m->specs[i].spec_id == id) return &m->specs[i];
    return nullptr;
}

// Every live data file: its recorded tuple equals the tuple of every row it holds, no tuple has two files from one
// append, and its manifest list entry's summaries bound it.
void expect_files_match_rows(Table* t) {
    const auto by_file = scan_by_file(t->root);
    for (const LiveFile& lf : live_files(t)) {
        if (lf.ref.content != FileContent::kData) continue;
        const PartitionSpec* spec = spec_by_id(*t, lf.ref.partition_spec_id);
        ASSERT_NE(spec, nullptr);
        ASSERT_EQ(lf.ref.n_partition, spec->n_fields) << lf.ref.file_path;
        EXPECT_EQ(lf.mle.partition_spec_id, lf.ref.partition_spec_id);
        auto it = by_file.find(lf.ref.file_path);
        if (it == by_file.end()) continue;                                 // every row deleted
        for (const Row& r : it->second)
            EXPECT_EQ(expect_tuple(*spec, r), recorded_tuple(lf.ref)) << lf.ref.file_path << " id " << r.id;
    }
}

}  // namespace

TEST(IcebergPartitioned, SpecFieldIdsAreAssignedFrom1000) {
    Table t;
    create_table(&t, fresh_root(temp_base(), "ids"), spec_of({pf(2, "identity", "k"), pf(1, "bucket[4]", "id_bucket")}));
    const Metadata* m = table_metadata(t.th);
    ASSERT_EQ(m->n_specs, 1u);
    EXPECT_EQ(m->specs[0].fields[0].field_id, 1000);
    EXPECT_EQ(m->specs[0].fields[1].field_id, 1001);
    const std::string doc = read_file(latest_metadata_path(t.root));
    EXPECT_NE(doc.find("\"last-partition-id\":1001"), std::string::npos);
    Table bad;
    bad.root = fresh_root(temp_base(), "bad");
    ASSERT_TRUE(filesystem_object_store_init(&bad.fs, bad.root.c_str(), &bad.os));
    Schema sch = make_schema();
    WriteOptions wo; write_options_init(&wo);
    PartitionSpec unknown = spec_of({pf(2, "zorder", "z")});
    EXPECT_FALSE(table_create(&bad.th, &bad.arena, &bad.os, bad.root.c_str(), &sch, &unknown, nullptr, &wo));
    PartitionSpec undefined = spec_of({pf(3, "hour", "h")});                // hour of a string: not defined
    EXPECT_FALSE(table_create(&bad.th, &bad.arena, &bad.os, bad.root.c_str(), &sch, &undefined, nullptr, &wo));
}

TEST(IcebergPartitioned, AppendWritesOneFilePerTupleAndRecordsIt) {
    Table t;
    const PartitionSpec spec = spec_of({pf(2, "identity", "k"), pf(3, "truncate[4]", "s_trunc"), pf(4, "year", "d_year")});
    create_table(&t, fresh_root(temp_base(), "append"), spec);
    const std::vector<Row> all = rows_of(0, 120);
    append_rows(&t, all, 3);                                               // tuples span batches
    EXPECT_EQ(scan_rows(t.root), model(all, [](const Row&) { return false; }));
    std::set<std::string> want;
    for (const Row& r : all) want.insert(expect_tuple(table_metadata(t.th)->specs[0], r));
    const auto files = live_files(&t);
    std::set<std::string> got;
    for (const LiveFile& f : files) EXPECT_TRUE(got.insert(recorded_tuple(f.ref)).second) << "two files of one tuple";
    EXPECT_EQ(got, want);
    expect_files_match_rows(&t);
    // The manifest list summarises the manifest: every field's lower <= every file's value <= upper.
    ASSERT_FALSE(files.empty());
    const ManifestListEntry& mle = files[0].mle;
    ASSERT_EQ(mle.n_partitions, 3u);
    EXPECT_TRUE(mle.partitions[0].contains_null);                          // k is null for id % 7 == 3
    int64_t lo = 0, hi = 0;
    ASSERT_EQ(mle.partitions[0].lower_len, 8u);
    std::memcpy(&lo, mle.partitions[0].lower, 8); std::memcpy(&hi, mle.partitions[0].upper, 8);
    EXPECT_EQ(lo, 0); EXPECT_EQ(hi, 3);
    EXPECT_EQ(std::string(mle.partitions[1].lower, mle.partitions[1].lower_len), "caf\xC3\xA9");   // 4 code points
    int32_t ylo = 0, yhi = 0;
    std::memcpy(&ylo, mle.partitions[2].lower, 4); std::memcpy(&yhi, mle.partitions[2].upper, 4);
    EXPECT_EQ(ylo, -2); EXPECT_EQ(yhi, transform_year(-400 + 119 * 37, TimeUnit::kDays));
}

TEST(IcebergPartitioned, EveryTransformOnEverySourceType) {
    const char* specs[][2] = {
        {"identity", "1"}, {"bucket[5]", "1"}, {"truncate[10]", "1"}, {"bucket[3]", "6"}, {"truncate[7]", "6"},
        {"identity", "6"}, {"bucket[4]", "3"}, {"truncate[2]", "3"}, {"identity", "3"}, {"year", "4"},
        {"month", "4"}, {"day", "4"}, {"bucket[6]", "4"}, {"year", "5"}, {"month", "5"}, {"day", "5"},
        {"hour", "5"}, {"bucket[8]", "5"}, {"identity", "4"}, {"void", "2"},
    };
    for (const auto& sp : specs) {
        Table t;
        const int32_t src = std::atoi(sp[1]);
        const PartitionSpec spec = spec_of({pf(src, sp[0], "p")});
        create_table(&t, fresh_root(temp_base(), "every"), spec);
        const std::vector<Row> all = rows_of(0, 60);
        append_rows(&t, all, 2);
        EXPECT_EQ(scan_rows(t.root), model(all, [](const Row&) { return false; })) << sp[0] << " on " << src;
        expect_files_match_rows(&t);
    }
}

namespace {

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

// Every position-delete file's spec and tuple is that of a live data file, and no two delete files of one commit
// share a (spec, tuple).
void expect_deletes_scoped(Table* t) {
    std::set<std::string> data_tuples;
    std::map<int64_t, std::set<std::string>> per_snapshot;
    for (const LiveFile& lf : live_files(t)) {
        const std::string key = std::to_string(lf.ref.partition_spec_id) + "/" + recorded_tuple(lf.ref);
        if (lf.ref.content == FileContent::kData) { data_tuples.insert(key); continue; }
        ASSERT_EQ(lf.mle.content, ManifestContent::kDeleteManifest);
        EXPECT_EQ(lf.mle.partition_spec_id, lf.ref.partition_spec_id);
        EXPECT_TRUE(per_snapshot[lf.ref.snapshot_id].insert(key).second) << "two delete files of one partition";
    }
    for (const auto& [snap, keys] : per_snapshot)
        for (const std::string& k : keys) EXPECT_TRUE(data_tuples.count(k)) << "delete file of " << k << " has no data file";
}

uint32_t count_content(Table* t, FileContent c) {
    uint32_t n = 0;
    for (const LiveFile& lf : live_files(t)) n += lf.ref.content == c ? 1u : 0u;
    return n;
}

}  // namespace

TEST(IcebergPartitioned, DeleteOverwriteAndCompactWorkPerPartition) {
    Table t;
    const PartitionSpec spec = spec_of({pf(2, "identity", "k"), pf(1, "bucket[3]", "id_b")});
    create_table(&t, fresh_root(temp_base(), "dml"), spec);
    std::vector<Row> all = rows_of(0, 40);
    append_rows(&t, rows_of(0, 20));
    append_rows(&t, rows_of(20, 40));
    // k = 1 holds whole files: they are removed, no delete file.
    Predicate k1 = pred_i64("k", PredicateOp::kEq, 1);
    ASSERT_TRUE(table_delete(t.th, &k1));
    EXPECT_EQ(count_content(&t, FileContent::kPositionDeletes), 0u);
    auto gone = [](const Row& r) { return r.k == 1; };
    EXPECT_EQ(scan_rows(t.root), model(all, gone));
    // i < 25 (ids < 25) matches all of the first append's files and parts of the second's, in several partitions:
    // one delete file per partition.
    Predicate ilt = pred_i64("i", PredicateOp::kLt, 25);
    ASSERT_TRUE(table_delete(t.th, &ilt)) << static_cast<int>(table_last_commit_error(t.th));
    auto gone2 = [](const Row& r) { return r.k == 1 || (r.i != kNull && r.i < 25); };
    EXPECT_EQ(scan_rows(t.root), model(all, gone2));
    EXPECT_GT(count_content(&t, FileContent::kPositionDeletes), 1u);
    expect_deletes_scoped(&t);
    // Overwrite: delete s = 'row-4' (partial) and add rows that land in several partitions.
    const std::vector<Row> add = rows_of(100, 112);
    bolt::BoltBatch b{};
    make_batch(&t.arena, &b, add);
    Predicate s4 = pred_str("s", PredicateOp::kEq, "row-4");
    ASSERT_TRUE(table_overwrite(t.th, &s4, &b));
    all.insert(all.end(), add.begin(), add.end());
    auto gone3 = [](const Row& r) { return (r.id < 100 && (r.k == 1 || (r.i != kNull && r.i < 25))) || r.s == "row-4"; };
    EXPECT_EQ(scan_rows(t.root), model(all, gone3));
    expect_deletes_scoped(&t);
    expect_files_match_rows(&t);
    // Compaction merges within each (spec, tuple) only, applies the deletes, keeps the rows.
    ASSERT_TRUE(table_compact(t.th, 64ll << 20)) << static_cast<int>(table_last_commit_error(t.th));
    EXPECT_EQ(scan_rows(t.root), model(all, gone3));
    expect_files_match_rows(&t);
    std::map<std::string, uint32_t> per_tuple;
    for (const LiveFile& lf : live_files(&t))
        if (lf.ref.content == FileContent::kData) ++per_tuple[recorded_tuple(lf.ref)];
    for (const auto& [tuple, n] : per_tuple) EXPECT_EQ(n, 1u) << tuple << " was not compacted into one file";
    EXPECT_EQ(count_content(&t, FileContent::kPositionDeletes), 0u) << "every delete was applied by the rewrite";
}

TEST(IcebergPartitioned, PositionDeletesByPathAreScopedToTheirPartition) {
    Table t;
    create_table(&t, fresh_root(temp_base(), "posdel"), spec_of({pf(2, "identity", "k")}));
    append_rows(&t, rows_of(0, 30));
    std::vector<PositionDeleteEntry> dels;
    for (const LiveFile& lf : live_files(&t)) {
        PositionDeleteEntry e{};
        std::strncpy(e.file_path, lf.ref.file_path, sizeof(e.file_path) - 1u);
        e.pos = 0;
        dels.push_back(e);
    }
    ASSERT_GT(dels.size(), 1u);
    ASSERT_TRUE(table_delete_positions(t.th, dels.data(), static_cast<uint32_t>(dels.size())));
    EXPECT_EQ(count_content(&t, FileContent::kPositionDeletes), static_cast<uint32_t>(dels.size()));
    expect_deletes_scoped(&t);
    // The first row (in file order) of each k is gone.
    std::set<int64_t> first;
    std::set<int64_t> seen;
    for (const Row& r : rows_of(0, 30)) if (seen.insert(r.k).second) first.insert(r.id);
    const std::vector<Row> got = scan_rows(t.root);
    EXPECT_EQ(got.size(), 30u - first.size());
    for (const Row& r : got) EXPECT_FALSE(first.count(r.id)) << r.id;
    PositionDeleteEntry stray{};
    std::strncpy(stray.file_path, "/no/such/file.parquet", sizeof(stray.file_path) - 1u);
    EXPECT_FALSE(table_delete_positions(t.th, &stray, 1)) << "a delete of a file not in the table names no partition";
}

TEST(IcebergPartitioned, SpecEvolutionKeepsEachFileOnItsSpec) {
    Table t;
    create_table(&t, fresh_root(temp_base(), "evolve"), spec_of({pf(2, "identity", "k")}));
    std::vector<Row> all = rows_of(0, 30);
    append_rows(&t, all);
    const PartitionSpec next = spec_of({pf(2, "void", "k"), pf(4, "month", "d_month")});
    ASSERT_TRUE(table_update_partition_spec(t.th, &next));
    const Metadata* m = table_metadata(t.th);
    ASSERT_EQ(m->n_specs, 2u);
    EXPECT_EQ(m->current_spec_id, 1);
    EXPECT_EQ(m->specs[1].fields[0].field_id, 1001) << "void(k) is not equivalent to identity(k): a new id";
    EXPECT_EQ(m->specs[1].fields[1].field_id, 1002);
    const std::vector<Row> more = rows_of(30, 60);
    append_rows(&t, more);
    all.insert(all.end(), more.begin(), more.end());
    EXPECT_EQ(scan_rows(t.root), model(all, [](const Row&) { return false; }));
    uint32_t spec0 = 0, spec1 = 0;
    for (const LiveFile& lf : live_files(&t)) {
        (lf.ref.partition_spec_id == 0 ? spec0 : spec1) += 1u;
        if (lf.ref.partition_spec_id == 1) EXPECT_TRUE(lf.ref.partition[0].is_null) << "a void field is always null";
    }
    EXPECT_GT(spec0, 1u);
    EXPECT_GT(spec1, 1u);
    expect_files_match_rows(&t);
    const std::string doc = read_file(latest_metadata_path(t.root));
    EXPECT_NE(doc.find("\"last-partition-id\":1002"), std::string::npos);
    // Delete and compact across both specs: each file keeps its spec, deletes carry their data file's.
    Predicate ilt = pred_i64("i", PredicateOp::kLt, 40);
    ASSERT_TRUE(table_delete(t.th, &ilt));
    expect_deletes_scoped(&t);
    auto gone = [](const Row& r) { return r.i != kNull && r.i < 40; };
    EXPECT_EQ(scan_rows(t.root), model(all, gone));
    append_rows(&t, rows_of(60, 75));
    all = model([&] { auto v = all; auto x = rows_of(60, 75); v.insert(v.end(), x.begin(), x.end()); return v; }(), gone);
    ASSERT_TRUE(table_compact(t.th, 64ll << 20));
    EXPECT_EQ(scan_rows(t.root), all);
    expect_files_match_rows(&t);
    // Going back to an equivalent spec makes it the default instead of adding a third.
    const PartitionSpec back = spec_of({pf(2, "identity", "k")});
    ASSERT_TRUE(table_update_partition_spec(t.th, &back));
    EXPECT_EQ(table_metadata(t.th)->n_specs, 2u);
    EXPECT_EQ(table_metadata(t.th)->current_spec_id, 0);
}
