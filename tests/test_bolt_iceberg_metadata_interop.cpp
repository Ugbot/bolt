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

    // FOUND WHILE WRITING THIS TEST, NOT FIXED HERE, AND PINNED SO A FIX IS
    // NOTICED: `table_add_column` mutates the CURRENT schema in place instead
    // of appending a new one with a new schema-id (writer.h says so outright:
    // "All update the current Schema"). The Iceberg spec makes evolution
    // additive — a new schema object, a new id, `current-schema-id` moved —
    // precisely so an old snapshot keeps pointing at the shape its data files
    // actually have. As it stands, a time-travel read of snapshot 0 binds a
    // column that did not exist when those files were written.
    //
    // The consequence for the property above is that the recorded-vs-derived
    // distinction is currently UNOBSERVABLE from outside: with one schema id
    // for the table's whole life, a wrong implementation reading
    // `current-schema-id` at emit time produces identical bytes. The recording
    // is still done correctly at the commit site rather than derived at emit
    // time, because doing it the other way would have to be undone the moment
    // evolution is fixed. When it is, this expectation flips and the
    // discriminating assertion beneath it becomes live.
    ASSERT_TRUE(table_add_column(th, "qty", bolt::BoltType::Int64, true));
    m = table_metadata(th);
    EXPECT_EQ(m->n_schemas, 1u)
        << "table_add_column now allocates a new schema — GOOD: delete this "
           "pin and enable the assertion below";
    if (m->n_schemas > 1u) {
        EXPECT_NE(m->current_schema_id, m->snapshots[0].schema_id);
        EXPECT_EQ(m->snapshots[0].schema_id, 0)
            << "a later schema change retroactively relabelled snapshot 0";
    }
    table_close(th);
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
