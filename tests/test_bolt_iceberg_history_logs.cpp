// G2ICE-6 — metadata.json `snapshot-log` and `metadata-log`.
//
// The in-process half: what bolt writes, reopens and prunes. The external
// verdict (pyiceberg history / snapshot_as_of_timestamp, DuckDB time travel)
// is scripts/iceberg_pyiceberg_interop.py over the interop fixture.

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_port.h"
#include "bolt/bolt_types.h"
#include "bolt/lakehouse/iceberg/metadata.h"
#include "bolt/lakehouse/iceberg/snapshot.h"
#include "bolt/lakehouse/iceberg/writer.h"
#include "bolt/lakehouse/object_store.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace {

using namespace bolt::lakehouse;
using namespace bolt::lakehouse::iceberg;

std::string fresh_root(const char* tag) {
    auto p = std::filesystem::temp_directory_path() /
             ("bolt_iceberg_history_" + std::string(tag) + "_" +
              std::to_string(static_cast<long long>(bolt_getpid())));
    std::error_code ec;
    std::filesystem::remove_all(p, ec);
    std::filesystem::create_directories(p, ec);
    return p.generic_string();
}

Schema make_schema() {
    Schema s{};
    s.schema_id = 0;
    s.n_fields  = 1;
    s.fields[0].id = 1; s.fields[0].required = true;
    std::strncpy(s.fields[0].name, "id",   sizeof(s.fields[0].name) - 1u);
    std::strncpy(s.fields[0].type, "long", sizeof(s.fields[0].type) - 1u);
    return s;
}

void make_batch(bolt::Arena* a, bolt::BoltBatch* out, int64_t first, int32_t n) {
    bolt::BoltBatch::init_empty(out);
    out->arena    = a;
    out->num_rows = n;
    out->num_cols = 1;
    bolt::BoltBatch::alloc_columns(out, a, 1);
    out->schema.add_field("id", bolt::BoltType::Int64, false);
    auto* cols = out->columns[out->read_epoch];
    auto* d = a->allocate_array<int64_t>(static_cast<uint32_t>(n));
    for (int32_t k = 0; k < n; ++k) d[k] = first + k;
    cols[0].type = bolt::BoltType::Int64; cols[0].length = n; cols[0].data = d;
}

struct Fixture {
    bolt::Arena arena;
    FilesystemObjectStore fs{};
    ObjectStore os{};
    std::string root;
};

void open_store(Fixture* f, const char* tag) {
    f->root = fresh_root(tag);
    ASSERT_TRUE(filesystem_object_store_init(&f->fs, f->root.c_str(), &f->os));
}

TableHandle* create(Fixture* f) {
    Schema sch = make_schema();
    PartitionSpec spec{}; SortOrder sort{};
    WriteOptions wo; write_options_init(&wo);
    TableHandle* th = nullptr;
    EXPECT_TRUE(table_create(&th, &f->arena, &f->os, f->root.c_str(), &sch,
                             &spec, &sort, &wo));
    return th;
}

bool append_rows(Fixture* f, TableHandle* th, int64_t first) {
    AppendHandle* ah = nullptr;
    if (!append_open(&ah, th)) return false;
    bolt::BoltBatch b{};
    make_batch(&f->arena, &b, first, 3);
    if (!append_write(ah, &b)) return false;
    const bool ok = append_commit(ah);
    append_close(ah);
    return ok;
}

std::string version_path(const std::string& root, int v) {
    return root + "/metadata/v" + std::to_string(v) + ".metadata.json";
}

std::string slurp(const std::string& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in),
                       std::istreambuf_iterator<char>());
}

// Parse a committed version straight off disk — what a reader would see.
bool load_version(Fixture* f, int v, Metadata* out) {
    const std::string body = slurp(version_path(f->root, v));
    if (body.empty()) return false;
    return metadata_parse(reinterpret_cast<const uint8_t*>(body.data()),
                          static_cast<uint32_t>(body.size()), &f->arena, out);
}

}  // namespace

// create + 2 appends + 1 delete = v1..v4. v4 must carry a snapshot-log naming
// the three snapshots in commit order and a metadata-log naming v1..v3, each
// stamped with that file's own last-updated-ms.
TEST(IcebergHistoryLogs, EmittedOnEveryCommit) {
    Fixture f; open_store(&f, "emit");
    TableHandle* th = create(&f);
    ASSERT_NE(th, nullptr);
    ASSERT_TRUE(append_rows(&f, th, 0));
    ASSERT_TRUE(append_rows(&f, th, 100));
    ASSERT_TRUE(table_delete(th, nullptr));

    const std::string raw = slurp(version_path(f.root, 4));
    EXPECT_NE(raw.find("\"snapshot-log\":["), std::string::npos);
    EXPECT_NE(raw.find("\"metadata-log\":["), std::string::npos);

    static Metadata m; ASSERT_TRUE(load_version(&f, 4, &m));
    ASSERT_EQ(m.n_snapshots, 3u);
    ASSERT_EQ(m.n_snapshot_log, 3u);
    for (uint32_t i = 0; i < 3u; ++i) {
        EXPECT_EQ(m.snapshot_log[i].snapshot_id, m.snapshots[i].snapshot_id);
        EXPECT_EQ(m.snapshot_log[i].timestamp_ms, m.snapshots[i].timestamp_ms);
    }
    ASSERT_EQ(m.n_metadata_log, 3u);
    int64_t last_ts = 0;
    for (uint32_t i = 0; i < 3u; ++i) {
        const std::string want = version_path(f.root, static_cast<int>(i) + 1);
        EXPECT_EQ(std::string(m.metadata_log[i].metadata_file), want);
        EXPECT_TRUE(std::filesystem::exists(want));
        static Metadata older;
        ASSERT_TRUE(load_version(&f, static_cast<int>(i) + 1, &older));
        EXPECT_EQ(m.metadata_log[i].timestamp_ms, older.last_updated_ms);
        EXPECT_GE(m.metadata_log[i].timestamp_ms, last_ts);
        last_ts = m.metadata_log[i].timestamp_ms;
    }
    EXPECT_LE(last_ts, m.last_updated_ms);

    // v1 supersedes nothing and has no snapshot yet.
    static Metadata v1; ASSERT_TRUE(load_version(&f, 1, &v1));
    EXPECT_EQ(v1.n_metadata_log, 0u);
    EXPECT_EQ(v1.n_snapshot_log, 0u);
}

// Reopening from disk must carry both logs forward, not restart them.
TEST(IcebergHistoryLogs, ReopenCarriesLogsForward) {
    Fixture f; open_store(&f, "reopen");
    TableHandle* th = create(&f);
    ASSERT_NE(th, nullptr);
    ASSERT_TRUE(append_rows(&f, th, 0));
    ASSERT_TRUE(append_rows(&f, th, 100));
    static Metadata before; ASSERT_TRUE(load_version(&f, 3, &before));

    TableHandle* th2 = nullptr;
    ASSERT_TRUE(table_open(&th2, &f.arena, &f.os, f.root.c_str()));
    ASSERT_TRUE(append_rows(&f, th2, 200));

    static Metadata m; ASSERT_TRUE(load_version(&f, 4, &m));
    ASSERT_EQ(m.n_metadata_log, 3u);
    for (uint32_t i = 0; i < 2u; ++i) {
        EXPECT_EQ(m.metadata_log[i].timestamp_ms,
                  before.metadata_log[i].timestamp_ms);
        EXPECT_STREQ(m.metadata_log[i].metadata_file,
                     before.metadata_log[i].metadata_file);
    }
    EXPECT_EQ(std::string(m.metadata_log[2].metadata_file),
              version_path(f.root, 3));
    EXPECT_EQ(m.metadata_log[2].timestamp_ms, before.last_updated_ms);
    ASSERT_EQ(m.n_snapshot_log, 3u);
    EXPECT_EQ(m.snapshot_log[2].snapshot_id, m.current_snapshot_id);
}

// A commit that loses the CAS must leave the loser's handle exactly as it was.
TEST(IcebergHistoryLogs, LostCommitLeavesLogsUntouched) {
    Fixture f; open_store(&f, "conflict");
    TableHandle* th = create(&f);
    ASSERT_NE(th, nullptr);
    ASSERT_TRUE(append_rows(&f, th, 0));
    TableHandle* a = nullptr; TableHandle* b = nullptr;
    ASSERT_TRUE(table_open(&a, &f.arena, &f.os, f.root.c_str()));
    ASSERT_TRUE(table_open(&b, &f.arena, &f.os, f.root.c_str()));
    ASSERT_TRUE(append_rows(&f, a, 100));
    const Metadata* mb = table_metadata(b);
    const uint32_t nsl = mb->n_snapshot_log, nml = mb->n_metadata_log;
    const int64_t upd = mb->last_updated_ms;
    EXPECT_FALSE(append_rows(&f, b, 200));
    EXPECT_EQ(table_last_commit_error(b), CommitError::kConflict);
    EXPECT_EQ(mb->n_snapshot_log, nsl);
    EXPECT_EQ(mb->n_metadata_log, nml);
    EXPECT_EQ(mb->last_updated_ms, upd);
}

// Expiry removes the expired snapshots' log entries (and everything older).
TEST(IcebergHistoryLogs, ExpiryPrunesSnapshotLog) {
    Fixture f; open_store(&f, "expire");
    TableHandle* th = create(&f);
    ASSERT_NE(th, nullptr);
    ASSERT_TRUE(append_rows(&f, th, 0));
    ASSERT_TRUE(append_rows(&f, th, 100));
    ASSERT_TRUE(append_rows(&f, th, 200));
    const int64_t survivor = table_metadata(th)->current_snapshot_id;
    ASSERT_TRUE(table_expire_snapshots(th, UINT64_C(1) << 62, 1));
    static Metadata m; ASSERT_TRUE(load_version(&f, 5, &m));
    ASSERT_EQ(m.n_snapshots, 1u);
    ASSERT_EQ(m.n_snapshot_log, 1u);
    EXPECT_EQ(m.snapshot_log[0].snapshot_id, survivor);
    EXPECT_EQ(m.n_metadata_log, 4u);
}

// Both logs are bounded: the oldest entries go, the newest stay.
TEST(IcebergHistoryLogs, LogsAreBoundedOldestDropped) {
    static Metadata m{};
    for (uint32_t i = 0; i < kIcebergMaxSnapshotLog + 5u; ++i) {
        metadata_snapshot_log_push(&m, 1000 + i, 7000 + i);
    }
    ASSERT_EQ(m.n_snapshot_log, kIcebergMaxSnapshotLog);
    EXPECT_EQ(m.snapshot_log[0].snapshot_id, 7005);
    EXPECT_EQ(m.snapshot_log[kIcebergMaxSnapshotLog - 1u].snapshot_id,
              7000 + static_cast<int64_t>(kIcebergMaxSnapshotLog) + 4);
    for (uint32_t i = 0; i < kIcebergMaxMetadataLog + 3u; ++i) {
        MetadataLogEntry e{};
        e.timestamp_ms = i;
        std::snprintf(e.metadata_file, sizeof(e.metadata_file), "v%u", i + 1u);
        metadata_metadata_log_push(&m, &e);
    }
    ASSERT_EQ(m.n_metadata_log, kIcebergMaxMetadataLog);
    EXPECT_STREQ(m.metadata_log[0].metadata_file, "v4");
}

// Time travel by timestamp answers "what was CURRENT at T": a snapshot that
// exists but never became current (a branch commit) must not be returned.
TEST(IcebergHistoryLogs, AsOfTimestampFollowsSnapshotLog) {
    static Metadata m{};
    m.n_snapshots = 2;
    m.snapshots[0].snapshot_id = 11; m.snapshots[0].timestamp_ms = 100;
    m.snapshots[1].snapshot_id = 22; m.snapshots[1].timestamp_ms = 200;
    metadata_snapshot_log_push(&m, 100, 11);
    const Snapshot* s = snapshot_at_timestamp(&m, 250);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(s->snapshot_id, 11);
    EXPECT_EQ(snapshot_at_timestamp(&m, 50), nullptr);
}

// Tables bolt wrote before the log existed: parse derives it from snapshots[].
TEST(IcebergHistoryLogs, LegacyMetadataDerivesSnapshotLog) {
    const char* json =
        "{\"format-version\":2,\"snapshots\":["
        "{\"snapshot-id\":5,\"timestamp-ms\":10},"
        "{\"snapshot-id\":6,\"timestamp-ms\":20}]}";
    bolt::Arena arena;
    static Metadata m;
    ASSERT_TRUE(metadata_parse(reinterpret_cast<const uint8_t*>(json),
                               static_cast<uint32_t>(std::strlen(json)),
                               &arena, &m));
    ASSERT_EQ(m.n_snapshot_log, 2u);
    EXPECT_EQ(m.snapshot_log[1].snapshot_id, 6);
    EXPECT_EQ(m.snapshot_log[1].timestamp_ms, 20);
    EXPECT_EQ(m.n_metadata_log, 0u);
}
