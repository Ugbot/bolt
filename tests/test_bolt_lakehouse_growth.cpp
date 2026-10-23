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
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

using namespace bolt::lakehouse;
using namespace bolt::lakehouse::iceberg;

// Observational filesystem trait wrapper: returns/capabilities are unchanged.
// Native error is sampled immediately at trait return, not inside backend cleanup.
struct DiagnosticStore {
    struct Event {
        const char* operation = "";
        const char* phase = "init";
        char key[kOsMaxKey]{};
        int status = 0, posix_error = 0;
        unsigned long native_error = 0;
    };
    static constexpr unsigned kTrace = 12u;
    ObjectStore delegate{};
    ObjectStore exposed{};
    ObjectStoreVT vt{};
    Event events[kTrace]{};
    Event first_io{};
    bool have_io = false;
    bool log_deletes = false;
    unsigned calls = 0u;
    int writer = -1, commit = -1, attempt = 0;
    int64_t observed_version = -1, base_version = -1;
    const char* phase = "init";
    static void before() noexcept {
        errno = 0;
#if defined(_WIN32)
        SetLastError(ERROR_SUCCESS);
#endif
    }
    int after(const char* operation, const char* key, int result) noexcept {
        const int saved_errno = errno;
#if defined(_WIN32)
        const unsigned long saved_native = GetLastError();
#else
        const unsigned long saved_native = 0;
#endif
        auto& event = events[calls++ % kTrace];
        event.operation = operation; event.phase = phase; event.status = result;
        event.posix_error = saved_errno; event.native_error = saved_native;
        std::strncpy(event.key, key, sizeof(event.key) - 1u); event.key[sizeof(event.key) - 1u] = 0;
        if (result != kOsOk && result != kOsNotFound && result != kOsExists && !have_io) {
            first_io = event; have_io = true;
        }
        if (result == kOsOk && (std::strcmp(operation, "get") == 0 || std::strcmp(operation, "put_if_absent") == 0) &&
            std::strncmp(key, "metadata/v", 10u) == 0) {
            char* end = nullptr;
            const auto version = std::strtoll(key + 10u, &end, 10);
            if (end != key + 10u && std::strcmp(end, ".metadata.json") == 0) observed_version = version;
        }
        if (log_deletes && std::strcmp(operation, "delete") == 0) this->event(event);
        errno = saved_errno;
#if defined(_WIN32)
        SetLastError(saved_native);
#endif
        return result;
    }
    static DiagnosticStore& self(void* p) { return *static_cast<DiagnosticStore*>(p); }
    static int get(void* p, const char* key, bolt::Arena* arena, const uint8_t** data, uint64_t* size) noexcept {
        auto& s = self(p); before(); const int rc = s.delegate.vt->get_object(s.delegate.impl, key, arena, data, size);
        return s.after("get", key, rc);
    }
    static int put(void* p, const char* key, const uint8_t* data, uint64_t size) noexcept {
        auto& s = self(p); before(); const int rc = s.delegate.vt->put_object(s.delegate.impl, key, data, size);
        return s.after("put", key, rc);
    }
    static int list(void* p, const char* key, ObjectEntry* out, uint32_t cap, uint32_t* count) noexcept {
        auto& s = self(p); before(); const int rc = s.delegate.vt->list_prefix(s.delegate.impl, key, out, cap, count);
        return s.after("list", key, rc);
    }
    static int remove(void* p, const char* key) noexcept {
        auto& s = self(p); before(); const int rc = s.delegate.vt->delete_object(s.delegate.impl, key);
        return s.after("delete", key, rc);
    }
    static int head(void* p, const char* key, ObjectMeta* out) noexcept {
        auto& s = self(p); before(); const int rc = s.delegate.vt->head_object(s.delegate.impl, key, out);
        return s.after("head", key, rc);
    }
    static int conditional(void* p, const char* key, const uint8_t* data, uint64_t size) noexcept {
        auto& s = self(p); before(); const int rc = s.delegate.vt->put_if_absent(s.delegate.impl, key, data, size);
        return s.after("put_if_absent", key, rc);
    }
    static int range(void* p, const char* key, uint64_t offset, uint64_t size, uint8_t* out, uint64_t* got) noexcept {
        auto& s = self(p); before(); const int rc = s.delegate.vt->get_range(s.delegate.impl, key, offset, size, out, got);
        return s.after("get_range", key, rc);
    }
    static int versioned(void* p, const char* key, bolt::Arena* scratch, ObjectVersion* out) noexcept {
        auto& s = self(p); before(); const int rc = s.delegate.vt->head_versioned(s.delegate.impl, key, scratch, out);
        return s.after("head_versioned", key, rc);
    }
    static int matched(void* p, const char* key, const ObjectVersion* version, uint64_t offset,
        uint32_t size, bolt::Arena* scratch, uint8_t* out, uint32_t* got) noexcept {
        auto& s = self(p); before(); const int rc = s.delegate.vt->get_range_if_match(s.delegate.impl, key, version, offset, size, scratch, out, got);
        return s.after("get_range_if_match", key, rc);
    }
    void bind(ObjectStore* store) {
        delegate = *store; vt = *delegate.vt;
        vt.get_object = get; vt.put_object = put; vt.list_prefix = list;
        vt.delete_object = remove; vt.head_object = head;
        if (delegate.vt->put_if_absent) vt.put_if_absent = conditional;
        if (delegate.vt->get_range) vt.get_range = range;
        if (delegate.vt->head_versioned) vt.head_versioned = versioned;
        if (delegate.vt->get_range_if_match) vt.get_range_if_match = matched;
        exposed = ObjectStore{&vt, this}; *store = exposed;
    }
    void event(const Event& e) const {
        std::fprintf(stderr, "[iceberg-diag] writer=%d c/round=%d attempt=%d phase=%s op=%s key=%.2047s status=%d post_return_errno=%d post_return_native=%lu\n",
            writer, commit, attempt, e.phase, e.operation, e.key, e.status, e.posix_error, e.native_error);
    }
    void dump(TableHandle* table) const {
        const Metadata* meta = table ? table_metadata(table) : nullptr;
        const Snapshot* current = meta ? snapshot_by_id(meta, meta->current_snapshot_id) : nullptr;
        std::fprintf(stderr, "[iceberg-diag] writer=%d c/round=%d attempt=%d phase=%s CommitError=%u observed_metadata_version=%lld opened_base_version=%lld snapshot=%lld parent_snapshot=%lld retained=%u parent_manifest_list=%.2047s\n",
            writer, commit, attempt, phase, static_cast<unsigned>(table ? table_last_commit_error(table) : CommitError::kFailed),
            static_cast<long long>(observed_version), static_cast<long long>(base_version),
            static_cast<long long>(meta ? meta->current_snapshot_id : -1),
            static_cast<long long>(current ? current->parent_snapshot_id : -1),
            meta ? meta->n_snapshots : 0u, current ? current->manifest_list : "");
        if (have_io) event(first_io);
        const unsigned count = calls < kTrace ? calls : kTrace;
        for (unsigned i = 0u; i < count; ++i) event(events[(calls - count + i) % kTrace]);
    }
};


std::string fresh_dir(const char* tag) {
    std::error_code ec;
    auto p = std::filesystem::temp_directory_path(ec);
    if (ec) { std::fprintf(stderr, "[iceberg-diag] phase=temp_directory status=%d\n", ec.value()); return {}; }
    p /= "bolt_l5_" + std::string(tag) + "_" +
         std::to_string(static_cast<long long>(bolt_getpid()));
    std::filesystem::remove_all(p, ec);
    if (ec) { std::fprintf(stderr, "[iceberg-diag] phase=remove_fixture status=%d\n", ec.value()); return {}; }
    std::filesystem::create_directories(p / "default" / "t", ec);
    if (ec) { std::fprintf(stderr, "[iceberg-diag] phase=create_fixture status=%d\n", ec.value()); return {}; }
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

bool make_batch(bolt::Arena* a, bolt::BoltBatch* out, int64_t first, int32_t n) {
    bolt::BoltBatch::init_empty(out);
    out->arena = a;
    out->num_rows = n;
    out->num_cols = 1;
    if (!bolt::BoltBatch::alloc_columns(out, a, 1)) return false;
    out->schema.add_field("id", bolt::BoltType::Int64, false);
    auto* cols = out->columns[out->read_epoch];
    auto* d = a->allocate_array<int64_t>(static_cast<uint32_t>(n));
    if (!d) return false;
    for (int32_t k = 0; k < n; ++k) d[k] = first + k;
    cols[0].type = bolt::BoltType::Int64;
    cols[0].length = n;
    cols[0].data = d;
    return true;
}

struct Table {
    bolt::Arena arena;
    FilesystemObjectStore fs{};
    ObjectStore os{};
    DiagnosticStore diagnostic;
    const char* tag = "";
    bool reopened = false;
    std::string ns_root;
    std::string root;
    TableHandle* th = nullptr;
};

void create_table(Table* t, const char* tag) {
    t->tag = tag;
    t->ns_root = fresh_dir(tag);
    ASSERT_FALSE(t->ns_root.empty());
    t->root = t->ns_root + "/default/t";
    ASSERT_TRUE(filesystem_object_store_init(&t->fs, t->root.c_str(), &t->os));
    t->diagnostic.bind(&t->os);
    t->diagnostic.log_deletes = std::strcmp(tag, "retain") == 0;
    t->diagnostic.phase = "create";
    Schema sch = one_col_schema();
    PartitionSpec spec{};
    SortOrder sort{};
    WriteOptions wo;
    write_options_init(&wo);
    ASSERT_TRUE(table_create(&t->th, &t->arena, &t->os, t->root.c_str(), &sch,
                             &spec, &sort, &wo));
    t->diagnostic.base_version = t->diagnostic.observed_version;
}

void parent_context(Table* t) {
    const Metadata* metadata = table_metadata(t->th);
    const Snapshot* parent = snapshot_by_id(metadata, metadata->current_snapshot_id);
    t->diagnostic.dump(t->th);
    if (!parent || parent->manifest_list[0] == 0) return;
    std::string key(parent->manifest_list);
    const std::string prefix = t->root + "/";
    if (key.compare(0u, prefix.size(), prefix) == 0) key.erase(0u, prefix.size());
    ObjectMeta object{};
    const int status = os_head(&t->os, key.c_str(), &object);
    std::fprintf(stderr, "[iceberg-diag] c=%d phase=%s parent_probe_key=%.2047s status=%d exists=%d bytes=%llu\n",
        t->diagnostic.commit, t->diagnostic.phase, key.c_str(), status, object.exists ? 1 : 0,
        static_cast<unsigned long long>(object.size));
}

bool append(Table* t, int64_t first, int32_t n) {
    auto& diagnostic = t->diagnostic;
    diagnostic.commit = static_cast<int>(first / 10);
    diagnostic.phase = t->reopened ? "after_reopen.before_append" : "before_reopen.before_append";
    if (std::strcmp(t->tag, "retain") == 0) parent_context(t);
    AppendHandle* ah = nullptr;
    auto fail = [&] {
        diagnostic.dump(t->th);
        if (ah) append_close(ah);
        return false;
    };
    diagnostic.phase = t->reopened ? "after_reopen.append_open" : "append_open";
    if (!append_open(&ah, t->th)) return fail();
    bolt::BoltBatch b{};
    diagnostic.phase = "batch_alloc";
    if (!make_batch(&t->arena, &b, first, n)) return fail();
    diagnostic.phase = t->reopened ? "after_reopen.write" : "write";
    if (!append_write(ah, &b)) return fail();
    diagnostic.phase = t->reopened ? "after_reopen.commit" : "commit";
    const bool ok = append_commit(ah);
    if (!ok) diagnostic.dump(t->th);
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
    t.diagnostic.phase = "set_retention";
    const bool retained = table_set_retention(t.th, 0, 5);
    if (!retained) t.diagnostic.dump(t.th);
    ASSERT_TRUE(retained);
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
    t.diagnostic.phase = "reopen";
    const bool reopened = table_open(&th2, &a2, &t.os, t.root.c_str());
    if (!reopened) t.diagnostic.dump(th2);
    ASSERT_TRUE(reopened);
    t.diagnostic.base_version = t.diagnostic.observed_version;
    t.reopened = true;
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
    ASSERT_FALSE(dir.empty());
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
    ASSERT_FALSE(dir.empty());
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
    ASSERT_FALSE(dir.empty());
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
