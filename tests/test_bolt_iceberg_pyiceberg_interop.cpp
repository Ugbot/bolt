// G2ICE-84 — write an Iceberg table for an EXTERNAL reader to verify.
//
// WHY THIS EXISTS, AND WHY IT DOES NOT ASSERT MUCH ITSELF.
//
// bolt has a writer AND a reader for Iceberg, and every existing write test
// closes the loop through bolt's own reader. That proves close to nothing
// about the TABLE: `iceberg_scan.cpp`'s `read_ref` deliberately tries three
// different rebasings of every recorded path (location-relative, strip-root,
// join-table-rel), so it resolves a manifest list whether the writer recorded
// it as `metadata/snap-1.avro` or as `/abs/warehouse/t/metadata/snap-1.avro`.
// A reader that forgiving cannot detect a writer that records the wrong thing.
// pyiceberg is not forgiving: it resolves `manifest-list` LITERALLY, so a
// relative one is opened relative to the reader's CWD and the table fails to
// scan from anywhere except the directory it happened to be written from.
// That is the exact bug this fixture exists to expose, and only an outside
// reader can see it.
//
// The same argument applies to field ids. Iceberg binds a data file's columns
// to schema fields by FIELD ID, taken from the parquet `field_id` on each
// SchemaElement. bolt's parquet writer does not emit them (that is L1's file
// set, deliberately untouched here), so the table must instead carry
// `schema.name-mapping.default`, which is precisely the mechanism the spec
// defines for field-id-less files. bolt's own reader binds columns
// POSITIONALLY and so is blind to whether either exists.
//
// So this test writes the fixture and asserts only what it can honestly
// assert in-process (the recorded locations are absolute and name real
// files). The value-level verdict comes from
// `scripts/iceberg_pyiceberg_interop.py`, which reads the same directory with
// pyiceberg and compares against the model re-derived from the generating
// rules below.
//
//     ./test_bolt_iceberg_pyiceberg_interop            # writes the fixture
//     python3 scripts/iceberg_pyiceberg_interop.py <dir>
//
// The fixture directory is printed on stdout and can be pinned with
// BOLT_ICEBERG_INTEROP_DIR so the two halves agree without parsing output.

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"
#include "bolt/lakehouse/iceberg/metadata.h"
#include "bolt/lakehouse/iceberg/writer.h"
#include "bolt/lakehouse/object_store.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

namespace {

using namespace bolt::lakehouse;
using namespace bolt::lakehouse::iceberg;

// The generating rules the python oracle re-derives independently.
//   row i (0-based, over 12 rows written as two 6-row commits):
//       id    = 1000 + i
//       score = i * 0.5 - 3.0
constexpr int32_t kRowsPerCommit = 6;
constexpr int32_t kCommits       = 2;

std::string fixture_root() {
    const char* env = std::getenv("BOLT_ICEBERG_INTEROP_DIR");
    std::filesystem::path p;
    if (env != nullptr && env[0] != '\0') {
        p = std::filesystem::path(env);
    } else {
        p = std::filesystem::temp_directory_path() / "bolt_iceberg_interop";
    }
    std::error_code ec;
    std::filesystem::remove_all(p, ec);
    std::filesystem::create_directories(p, ec);
    return p.generic_string();
}

Schema make_schema() {
    Schema s{};
    s.schema_id = 0;
    s.n_fields  = 2;
    s.fields[0].id = 1; s.fields[0].required = true;
    std::strncpy(s.fields[0].name, "id",   sizeof(s.fields[0].name) - 1u);
    std::strncpy(s.fields[0].type, "long", sizeof(s.fields[0].type) - 1u);
    s.fields[1].id = 2; s.fields[1].required = true;
    std::strncpy(s.fields[1].name, "score", sizeof(s.fields[1].name) - 1u);
    std::strncpy(s.fields[1].type, "double", sizeof(s.fields[1].type) - 1u);
    return s;
}

void make_batch(bolt::Arena* a, bolt::BoltBatch* out, int32_t first_row,
                int32_t n) {
    bolt::BoltBatch::init_empty(out);
    out->arena    = a;
    out->num_rows = n;
    out->num_cols = 2;
    bolt::BoltBatch::alloc_columns(out, a, 2);
    out->schema.add_field("id",    bolt::BoltType::Int64,   false);
    out->schema.add_field("score", bolt::BoltType::Float64, false);
    auto* cols = out->columns[out->read_epoch];
    auto* idata = a->allocate_array<int64_t>(static_cast<uint32_t>(n));
    auto* sdata = a->allocate_array<double>(static_cast<uint32_t>(n));
    for (int32_t k = 0; k < n; ++k) {
        const int32_t i = first_row + k;
        idata[k] = 1000 + i;
        sdata[k] = static_cast<double>(i) * 0.5 - 3.0;
    }
    cols[0].type = bolt::BoltType::Int64;   cols[0].length = n; cols[0].data = idata;
    cols[1].type = bolt::BoltType::Float64; cols[1].length = n; cols[1].data = sdata;
}

bool is_absolute_path(const char* p) {
    if (p == nullptr || p[0] == '\0') return false;
    if (p[0] == '/') return true;                       // POSIX
    if (std::strstr(p, "://") != nullptr) return true;  // scheme://host/...
    // Windows drive letter.
    return (p[1] == ':' && (p[2] == '/' || p[2] == '\\'));
}

}  // namespace

TEST(IcebergPyIcebergInterop, WriteFixture) {
    bolt::Arena arena;
    const std::string root = fixture_root();
    FilesystemObjectStore fs{}; ObjectStore os{};
    ASSERT_TRUE(filesystem_object_store_init(&fs, root.c_str(), &os));
    Schema sch = make_schema();
    PartitionSpec spec{}; spec.spec_id = 0; spec.n_fields = 0;
    SortOrder sort{};     sort.order_id = 0; sort.n_fields = 0;
    WriteOptions wo;      write_options_init(&wo);
    TableHandle* th = nullptr;
    ASSERT_TRUE(table_create(&th, &arena, &os, root.c_str(), &sch, &spec,
                             &sort, &wo));

    AppendHandle* ah = nullptr;
    ASSERT_TRUE(append_open(&ah, th));
    for (int32_t c = 0; c < kCommits; ++c) {
        bolt::BoltBatch b{};
        make_batch(&arena, &b, c * kRowsPerCommit, kRowsPerCommit);
        ASSERT_TRUE(append_write(ah, &b));
        ASSERT_TRUE(append_commit(ah));
    }
    append_close(ah);

    const Metadata* m = table_metadata(th);
    ASSERT_EQ(m->n_snapshots, static_cast<uint32_t>(kCommits));

    // What this test CAN check in-process: every recorded manifest-list is an
    // absolute location that actually names a file. A relative one resolves
    // against the reader's CWD, which is the failure pyiceberg reports as
    // "FileNotFoundError: snap-<id>.avro".
    for (uint32_t i = 0; i < m->n_snapshots; ++i) {
        const char* ml = m->snapshots[i].manifest_list;
        EXPECT_TRUE(is_absolute_path(ml)) << "manifest-list not absolute: " << ml;
        EXPECT_TRUE(std::filesystem::exists(std::filesystem::path(ml)))
            << "manifest-list names no file: " << ml;
    }

    std::printf("[fixture] %s\n", root.c_str());
}
