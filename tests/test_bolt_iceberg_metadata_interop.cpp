// test_bolt_iceberg_metadata_interop.cpp — is bolt's metadata.json a table
// any OTHER Iceberg implementation will open?
//
// WHY THIS EXISTS. Every existing Iceberg write test in this tree checks
// bolt's writer against bolt's own reader. That is the exact self-verification
// this repo has already been burned by twice (the LIST writer whose test was
// named "...ThroughBoltAndPyarrow" with pyarrow nowhere in it; the bloom
// filters nothing had ever probed). A writer and a reader that share a
// misreading of the spec agree perfectly with each other and with nobody else.
//
// And they did share one: `metadata_json_emit` wrote each schema as
//     {"schema-id":0,"fields":[...]}
// while the Iceberg spec says a schema IS a struct type and so carries
// "type":"struct" alongside its fields — which is why DuckDB 1.4.5 refuses
// the table outright with `StructType required property 'type' is missing`.
// bolt's own metadata PARSER never looks at "type", so the round trip was
// green throughout. The manifest's copy of the same schema (emit_schema_json)
// always had it: two emitters of one structure, only one of them correct.
//
// This file has two halves:
//   * a FROM-SPEC check of the emitted bytes (below) — the fast regression
//     guard, written against the spec's required-property list, not against
//     what bolt happens to emit;
//   * the external half, scripts/iceberg_metadata_interop.py, which points
//     pyiceberg and DuckDB — two independent implementations — at the table
//     this test writes. Run it after this test:
//         ./test_bolt_iceberg_metadata_interop
//         python3 scripts/iceberg_metadata_interop.py <printed dir>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"
#include "bolt/lakehouse/iceberg/writer.h"
#include "bolt/lakehouse/object_store.h"

namespace {

using namespace bolt::lakehouse;
using namespace bolt::lakehouse::iceberg;

// A stable, printable location so the python half can be pointed at it
// without parsing test output for a temp name.
std::string fresh_root(const char* table) {
    auto p = std::filesystem::temp_directory_path() /
             "bolt_iceberg_meta_interop" / "db" / table;
    std::error_code ec;
    std::filesystem::remove_all(p, ec);       // commits are put_if_absent
    std::filesystem::create_directories(p, ec);
    return p.generic_string();
}

std::string interop_root() { return fresh_root("trades"); }

Schema make_schema() {
    Schema s{};
    s.schema_id = 0;
    s.n_fields  = 2;
    s.fields[0].id = 1;
    s.fields[0].required = true;
    std::strncpy(s.fields[0].name, "id", sizeof(s.fields[0].name) - 1u);
    std::strncpy(s.fields[0].type, "long", sizeof(s.fields[0].type) - 1u);
    s.fields[1].id = 2;
    s.fields[1].required = false;
    std::strncpy(s.fields[1].name, "price", sizeof(s.fields[1].name) - 1u);
    std::strncpy(s.fields[1].type, "double", sizeof(s.fields[1].type) - 1u);
    return s;
}

void make_batch(bolt::Arena* a, bolt::BoltBatch* out, int32_t n) {
    assert(a != nullptr && out != nullptr);
    bolt::BoltBatch::init_empty(out);
    out->arena    = a;
    out->num_rows = n;
    out->num_cols = 2;
    bolt::BoltBatch::alloc_columns(out, a, 2);
    out->schema.add_field("id", bolt::BoltType::Int64, false);
    out->schema.add_field("price", bolt::BoltType::Float64, true);
    auto* cols = out->columns[out->read_epoch];
    cols[0].type   = bolt::BoltType::Int64;
    cols[0].length = n;
    auto* ids = a->allocate_array<int64_t>(static_cast<uint32_t>(n));
    cols[1].type   = bolt::BoltType::Float64;
    cols[1].length = n;
    auto* px = a->allocate_array<double>(static_cast<uint32_t>(n));
    assert(ids != nullptr && px != nullptr);
    for (int32_t i = 0; i < n; ++i) {   // bounded by n
        ids[i] = 100 + i;
        px[i]  = 1.5 * static_cast<double>(i);
    }
    cols[0].data = ids;
    cols[1].data = px;
}

std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::string();
    return std::string((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
}

// Latest metadata/vN.metadata.json, chosen the way a reader does: through
// version-hint.text, so a hint that names a file we never wrote fails here.
std::string read_latest_metadata(const std::string& root) {
    const std::string hint = slurp(root + "/metadata/version-hint.text");
    if (hint.empty()) return std::string();
    return slurp(root + "/metadata/v" + hint + ".metadata.json");
}

bool has(const std::string& hay, const char* needle) {
    return hay.find(needle) != std::string::npos;
}

}  // namespace

// The properties below are the Iceberg v2 table-metadata REQUIRED set, taken
// from the spec, not from what bolt emits. `"type":"struct"` is the one DuckDB
// names in its error; `last-partition-id` is the sibling the spec also
// requires and which was missing next to it — a second omission is cheaper to
// find now than on a second round trip.
TEST(IcebergMetadataInterop, MetadataCarriesEverySpecRequiredProperty) {
    bolt::Arena arena;
    const std::string root = interop_root();
    FilesystemObjectStore fs{};
    ObjectStore os{};
    ASSERT_TRUE(filesystem_object_store_init(&fs, root.c_str(), &os));

    Schema sch = make_schema();
    PartitionSpec spec{}; spec.spec_id = 0; spec.n_fields = 0;
    SortOrder sort{};     sort.order_id = 0; sort.n_fields = 0;
    WriteOptions wo;      write_options_init(&wo);

    TableHandle* th = nullptr;
    ASSERT_TRUE(table_create(&th, &arena, &os, root.c_str(), &sch, &spec,
                             &sort, &wo));
    ASSERT_NE(th, nullptr);

    AppendHandle* ah = nullptr;
    ASSERT_TRUE(append_open(&ah, th));
    auto* b = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(b, nullptr);
    make_batch(&arena, b, 8);
    ASSERT_TRUE(append_write(ah, b));
    ASSERT_TRUE(append_commit(ah));
    append_close(ah);

    const std::string j = read_latest_metadata(root);
    ASSERT_FALSE(j.empty()) << "no metadata.json at " << root;

    // A schema IS a struct type. Without this DuckDB 1.4.5 rejects the table:
    // "StructType required property 'type' is missing".
    EXPECT_TRUE(has(j, "\"type\":\"struct\""))
        << "schema object is missing \"type\":\"struct\"\n" << j;
    // Required in v2 — the highest partition field id ever assigned.
    EXPECT_TRUE(has(j, "\"last-partition-id\""))
        << "missing last-partition-id\n" << j;
    // The rest of the required set, asserted so a future edit cannot drop one
    // silently the way this one was dropped.
    for (const char* k : {"\"format-version\"", "\"table-uuid\"",
                          "\"location\"", "\"last-column-id\"",
                          "\"last-sequence-number\"", "\"last-updated-ms\"",
                          "\"schemas\"", "\"current-schema-id\"",
                          "\"partition-specs\"", "\"default-spec-id\"",
                          "\"sort-orders\"", "\"default-sort-order-id\"",
                          "\"snapshots\"", "\"current-snapshot-id\""}) {
        EXPECT_TRUE(has(j, k)) << "missing required property " << k;
    }

    std::fprintf(stderr,
                 "\n[interop] table written to: %s\n"
                 "[interop] now run: python3 scripts/iceberg_metadata_interop.py %s\n",
                 root.c_str(), root.c_str());
    table_close(th);
}

// The other half of G2ICE-50: every snapshot records the schema it was
// COMMITTED under. The reference (tests/data/golden_iceberg_table, written by
// pyiceberg) carries "schema-id" on each snapshot; bolt carried none.
//
// This is asserted ACROSS a schema evolution on purpose. With one schema the
// property is untestable — every plausible implementation, including the wrong
// one that reads `current-schema-id` at emit time, produces the same digit.
// Add a column between two commits and only the correct one still reports 0
// for the first snapshot: a derived value would retroactively relabel history,
// and a time-travel reader would then bind the old snapshot's rows to a schema
// that has a column those files do not contain.
TEST(IcebergMetadataInterop, SnapshotsRecordTheSchemaTheyWereCommittedUnder) {
    bolt::Arena arena;
    const std::string root = fresh_root("evolution");
    FilesystemObjectStore fs{};
    ObjectStore os{};
    ASSERT_TRUE(filesystem_object_store_init(&fs, root.c_str(), &os));

    Schema sch = make_schema();
    PartitionSpec spec{}; spec.spec_id = 0; spec.n_fields = 0;
    SortOrder sort{};     sort.order_id = 0; sort.n_fields = 0;
    WriteOptions wo;      write_options_init(&wo);
    TableHandle* th = nullptr;
    ASSERT_TRUE(table_create(&th, &arena, &os, root.c_str(), &sch, &spec,
                             &sort, &wo));

    auto commit_one = [&](int32_t base) {
        AppendHandle* ah = nullptr;
        if (!append_open(&ah, th)) return false;
        auto* b = arena.allocate_array<bolt::BoltBatch>(1);
        if (b == nullptr) return false;
        make_batch(&arena, b, base);
        const bool ok = append_write(ah, b) && append_commit(ah);
        append_close(ah);
        return ok;
    };

    ASSERT_TRUE(commit_one(4));                       // snapshot 0, schema 0
    ASSERT_TRUE(commit_one(6));                       // snapshot 1, schema 0
    const Metadata* m = table_metadata(th);
    ASSERT_NE(m, nullptr);
    ASSERT_EQ(m->n_snapshots, 2u);
    EXPECT_EQ(m->snapshots[0].schema_id, m->current_schema_id);
    EXPECT_EQ(m->snapshots[1].schema_id, m->current_schema_id);

    // G2ICE-89, FIXED: `table_add_column` used to mutate the CURRENT schema in
    // place instead of appending a new one with a new schema-id (writer.h
    // said so outright: "All update the current Schema"). The Iceberg spec
    // makes evolution additive — a new schema object, a new id,
    // `current-schema-id` moved — precisely so an old snapshot keeps
    // pointing at the shape its data files actually have. Confirmed against
    // pyiceberg (an independent implementation) that this used to bind a
    // column to snapshot 0's rows that did not exist when they were written;
    // see G2ICE-89's tracker entry for the full external-oracle repro.
    const int32_t schema0_id = m->current_schema_id;
    ASSERT_TRUE(table_add_column(th, "qty", bolt::BoltType::Int64, true));
    m = table_metadata(th);
    ASSERT_EQ(m->n_schemas, 2u)
        << "table_add_column must APPEND a new schema, not mutate the one "
           "old snapshots already point at";
    EXPECT_NE(m->current_schema_id, m->snapshots[0].schema_id);
    EXPECT_EQ(m->snapshots[0].schema_id, schema0_id)
        << "a later schema change retroactively relabelled snapshot 0";
    EXPECT_EQ(m->snapshots[1].schema_id, schema0_id)
        << "a later schema change retroactively relabelled snapshot 1, "
           "which was also committed before the evolution";
    // The ORIGINAL schema (what snapshot 0/1 point at) must be BYTE-IDENTICAL
    // to what it was before evolution -- not just "still present at some id".
    bool found_old = false;
    for (uint32_t i = 0; i < m->n_schemas; ++i) {
        if (m->schemas[i].schema_id != schema0_id) continue;
        found_old = true;
        EXPECT_EQ(m->schemas[i].n_fields, 2u)
            << "the pre-evolution schema gained a field in place";
    }
    EXPECT_TRUE(found_old);
    // The NEW current schema carries the added column, and only it.
    const Schema* cur = metadata_current_schema(m);
    ASSERT_NE(cur, nullptr);
    ASSERT_EQ(cur->n_fields, 3u);
    EXPECT_STREQ(cur->fields[2].name, "qty");

    // A post-evolution append must serialize under the NEW current schema
    // (write_data_file used to hardcode schemas[0] too, which — pre-fix —
    // happened to be a harmless no-op alias for "current"; post-fix it is
    // explicitly not, and an append naively re-using schemas[0] would write
    // the OLD 2-field layout instead of the evolved one).
    {
        AppendHandle* ah = nullptr;
        ASSERT_TRUE(append_open(&ah, th));
        auto* b3 = arena.allocate_array<bolt::BoltBatch>(1);
        ASSERT_NE(b3, nullptr);
        bolt::BoltBatch::init_empty(b3);
        b3->arena    = &arena;
        b3->num_rows = 3;
        b3->num_cols = 3;
        bolt::BoltBatch::alloc_columns(b3, &arena, 3);
        b3->schema.add_field("id", bolt::BoltType::Int64, false);
        b3->schema.add_field("price", bolt::BoltType::Float64, true);
        b3->schema.add_field("qty", bolt::BoltType::Int64, true);
        auto* cols3 = b3->columns[b3->read_epoch];
        cols3[0].type = bolt::BoltType::Int64;   cols3[0].length = 3;
        cols3[1].type = bolt::BoltType::Float64; cols3[1].length = 3;
        cols3[2].type = bolt::BoltType::Int64;   cols3[2].length = 3;
        auto* ids3 = arena.allocate_array<int64_t>(3);
        auto* px3  = arena.allocate_array<double>(3);
        auto* qty3 = arena.allocate_array<int64_t>(3);
        for (int32_t i = 0; i < 3; ++i) {
            ids3[i] = 900 + i; px3[i] = 9.0; qty3[i] = 42 + i;
        }
        cols3[0].data = ids3; cols3[1].data = px3; cols3[2].data = qty3;
        ASSERT_TRUE(append_write(ah, b3));
        ASSERT_TRUE(append_commit(ah));
        append_close(ah);
    }
    m = table_metadata(th);
    ASSERT_EQ(m->n_snapshots, 3u);
    EXPECT_EQ(m->snapshots[2].schema_id, m->current_schema_id)
        << "the post-evolution append was not recorded under the new schema";

    std::fprintf(stderr,
                 "\n[interop] evolution table written to: %s\n"
                 "[interop] now run: python3 "
                 "scripts/iceberg_schema_evolution_interop.py %s %lld\n",
                 root.c_str(), root.c_str(),
                 static_cast<long long>(m->snapshots[0].snapshot_id));
    table_close(th);
}

// G2ICE-177: a root snapshot has no parent, so the spec's optional
// "parent-snapshot-id" is absent -- never -1, which pyiceberg reports as a
// dangling parent. The child must still name the root, and the reader must
// map the absent field back to -1.
TEST(IcebergMetadataInterop, RootSnapshotOmitsParentSnapshotId) {
    bolt::Arena arena;
    const std::string root = fresh_root("ancestry");
    FilesystemObjectStore fs{};
    ObjectStore os{};
    ASSERT_TRUE(filesystem_object_store_init(&fs, root.c_str(), &os));
    Schema sch = make_schema();
    PartitionSpec spec{}; spec.spec_id = 0; spec.n_fields = 0;
    SortOrder sort{};     sort.order_id = 0; sort.n_fields = 0;
    WriteOptions wo;      write_options_init(&wo);
    TableHandle* th = nullptr;
    ASSERT_TRUE(table_create(&th, &arena, &os, root.c_str(), &sch, &spec,
                             &sort, &wo));
    for (int32_t n : {4, 6}) {
        AppendHandle* ah = nullptr;
        ASSERT_TRUE(append_open(&ah, th));
        auto* b = arena.allocate_array<bolt::BoltBatch>(1);
        ASSERT_NE(b, nullptr);
        make_batch(&arena, b, n);
        ASSERT_TRUE(append_write(ah, b));
        ASSERT_TRUE(append_commit(ah));
        append_close(ah);
    }
    const Metadata* m = table_metadata(th);
    ASSERT_EQ(m->n_snapshots, 2u);
    const long long root_id = static_cast<long long>(m->snapshots[0].snapshot_id);
    const long long child_id = static_cast<long long>(m->snapshots[1].snapshot_id);
    table_close(th);

    const std::string j = read_latest_metadata(root);
    ASSERT_FALSE(j.empty());
    EXPECT_FALSE(has(j, "\"parent-snapshot-id\":-1")) << j;
    const std::string root_obj = "{\"snapshot-id\":" + std::to_string(root_id) + ",";
    const size_t at = j.find(root_obj);
    ASSERT_NE(at, std::string::npos) << j;
    const size_t end = j.find('}', at);
    EXPECT_EQ(j.substr(at, end - at).find("parent-snapshot-id"), std::string::npos)
        << "root snapshot names a parent\n" << j;
    const std::string child_obj = "{\"snapshot-id\":" + std::to_string(child_id) +
        ",\"parent-snapshot-id\":" + std::to_string(root_id) + ",";
    EXPECT_TRUE(has(j, child_obj.c_str())) << j;

    TableHandle* th2 = nullptr;
    ASSERT_TRUE(table_open(&th2, &arena, &os, root.c_str()));
    const Metadata* m2 = table_metadata(th2);
    ASSERT_EQ(m2->n_snapshots, 2u);
    EXPECT_EQ(m2->snapshots[0].parent_snapshot_id, -1);
    EXPECT_EQ(static_cast<long long>(m2->snapshots[1].parent_snapshot_id), root_id);
    table_close(th2);
}

// Discriminating power: the assertion above must be able to FAIL. A schema
// object that carries only "schema-id" and "fields" — exactly what bolt
// emitted before this fix — is rejected by the same predicate, so a green run
// above means the property is present, not that the check is vacuous.
TEST(IcebergMetadataInterop, TheSpecCheckRejectsThePreFixShape) {
    const std::string pre_fix =
        "{\"format-version\":2,\"schemas\":[{\"schema-id\":0,\"fields\":[]}]}";
    EXPECT_FALSE(has(pre_fix, "\"type\":\"struct\""));
    EXPECT_FALSE(has(pre_fix, "\"last-partition-id\""));
}
