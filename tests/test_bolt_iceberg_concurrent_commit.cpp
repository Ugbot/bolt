// G2ICE-163 — concurrent Iceberg committers against one table.
//
// Several independent TableHandles (threads, or forked processes) commit to
// the same table. The property: every append that REPORTS success is in the
// final table exactly once, and a writer that loses the race gets a conflict
// instead of clobbering the winner. Verified through bolt's catalog scan here
// and through pyiceberg + DuckDB by scripts/iceberg_concurrent_commit_oracle.py
// over the fixture the multi-process case leaves behind.
//
// Generating rule (re-derived independently by the oracle script):
//   writer w, commit c, row r  ->  id = w * 1'000'000 + c * 1000 + r

#include "bolt/bolt_arena.h"
#include "bolt/bolt_port.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"
#include "bolt/lakehouse/catalog.h"
#include "bolt/lakehouse/iceberg/metadata.h"
#include "bolt/lakehouse/iceberg/scan.h"
#include "bolt/lakehouse/iceberg/writer.h"
#include "bolt/lakehouse/object_store.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cerrno>
#include "bolt/lakehouse/iceberg/snapshot.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#endif

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
        CommitCandidate uncertain{};
        if (table && table_uncertain_commit(table, &uncertain))
            std::fprintf(stderr, "[iceberg-diag] writer=%d c/round=%d uncertain candidate key=%.255s kind=%d base_version=%lld candidate_version=%lld length=%llu\n",
                writer, commit, uncertain.object_key, static_cast<int>(uncertain.kind),
                static_cast<long long>(uncertain.base_metadata_version),
                static_cast<long long>(uncertain.candidate_metadata_version),
                static_cast<unsigned long long>(uncertain.length));
        if (have_io) event(first_io);
        const unsigned count = calls < kTrace ? calls : kTrace;
        for (unsigned i = 0u; i < count; ++i) event(events[(calls - count + i) % kTrace]);
    }
};


constexpr int32_t kRowsPerCommit = 4;

int64_t row_id(int32_t w, int32_t c, int32_t r) {
    return static_cast<int64_t>(w) * 1000000 + static_cast<int64_t>(c) * 1000 + r;
}

// <base>/default/t, so FilesystemCatalog(<base>) resolves ("default", "t").
std::string table_root(const std::string& base) {
    return (std::filesystem::path(base) / "default" / "t").generic_string();
}

std::string fresh_base(const char* tag) {
    std::error_code ec;
    std::filesystem::path p;
    const char* env = std::getenv("BOLT_ICEBERG_CONCURRENT_DIR");
    if (env != nullptr && env[0] != '\0' && std::strcmp(tag, "procs") == 0) {
        p = std::filesystem::path(env);
    } else {
        p = std::filesystem::temp_directory_path(ec);
        if (ec) { std::fprintf(stderr, "[iceberg-diag] phase=temp_directory status=%d\n", ec.value()); return {}; }
        p /= "bolt_iceberg_concurrent_" + std::string(tag) + "_" +
             std::to_string(static_cast<long long>(bolt_getpid()));
    }
    std::filesystem::remove_all(p, ec);
    if (ec) { std::fprintf(stderr, "[iceberg-diag] phase=remove_fixture status=%d\n", ec.value()); return {}; }
    std::filesystem::create_directories(p / "default" / "t", ec);
    if (ec) { std::fprintf(stderr, "[iceberg-diag] phase=create_fixture status=%d\n", ec.value()); return {}; }
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

bool make_batch(bolt::Arena* a, bolt::BoltBatch* out, int32_t w, int32_t c) {
    bolt::BoltBatch::init_empty(out);
    out->arena    = a;
    out->num_rows = kRowsPerCommit;
    out->num_cols = 1;
    if (!bolt::BoltBatch::alloc_columns(out, a, 1)) return false;
    out->schema.add_field("id", bolt::BoltType::Int64, false);
    auto* cols  = out->columns[out->read_epoch];
    auto* idata = a->allocate_array<int64_t>(kRowsPerCommit);
    if (!idata) return false;
    for (int32_t r = 0; r < kRowsPerCommit; ++r) idata[r] = row_id(w, c, r);
    cols[0].type = bolt::BoltType::Int64;
    cols[0].length = kRowsPerCommit;
    cols[0].data = idata;
    return true;
}

bool create_table(const std::string& root) {
    bolt::Arena arena;
    FilesystemObjectStore fs{}; ObjectStore os{};
    if (!filesystem_object_store_init(&fs, root.c_str(), &os)) return false;
    Schema sch = make_schema();
    PartitionSpec spec{}; spec.spec_id = 0; spec.n_fields = 0;
    SortOrder sort{};     sort.order_id = 0; sort.n_fields = 0;
    WriteOptions wo;      write_options_init(&wo);
    TableHandle* th = nullptr;
    if (!table_create(&th, &arena, &os, root.c_str(), &sch, &spec, &sort, &wo))
        return false;
    table_close(th);
    return true;
}

enum class Attempt { kCommitted, kConflict, kError };

// One commit attempt on a FRESHLY opened handle — the TieringManager shape.
// `ready` (optional) is a barrier spun on between staging and committing so
// several writers can be lined up on the same base version.
Attempt try_commit(const std::string& root, int32_t w, int32_t c,
                   std::atomic<int32_t>* ready, int32_t n_ready_target, int32_t attempt = 0) {
    bolt::Arena arena;
    FilesystemObjectStore fs{}; ObjectStore os{};
    DiagnosticStore diagnostic;
    diagnostic.writer = w; diagnostic.commit = c; diagnostic.attempt = attempt;
    TableHandle* th = nullptr; AppendHandle* ah = nullptr;
    auto fail = [&](const char* phase) {
        diagnostic.phase = phase; diagnostic.dump(th);
        if (ah) append_close(ah);
        if (th) table_close(th);
        return Attempt::kError;
    };
    if (!filesystem_object_store_init(&fs, root.c_str(), &os)) {
        diagnostic.after("store_init", "", kOsIoError); return fail("store_init");
    }
    diagnostic.bind(&os);
    diagnostic.phase = "open";
    if (!table_open(&th, &arena, &os, root.c_str())) return fail("open");
    diagnostic.base_version = diagnostic.observed_version;
    diagnostic.phase = "append_open";
    if (!append_open(&ah, th)) return fail("append_open");
    bolt::BoltBatch b{};
    if (!make_batch(&arena, &b, w, c)) return fail("batch_alloc");
    diagnostic.phase = "write";
    if (!append_write(ah, &b)) return fail("write");
    if (ready != nullptr) {
        ready->fetch_add(1, std::memory_order_acq_rel);
        for (uint32_t spin = 0; spin < (1u << 30); ++spin) {
            if (ready->load(std::memory_order_acquire) >= n_ready_target) break;
        }
    }
    diagnostic.phase = "commit";
    const bool ok = append_commit(ah);
    const CommitError err = table_last_commit_error(th);
    if (!ok && err != CommitError::kConflict) diagnostic.dump(th);
    append_close(ah); table_close(th);
    if (ok) {
        EXPECT_EQ(err, CommitError::kNone);
        return Attempt::kCommitted;
    }
    return err == CommitError::kConflict ? Attempt::kConflict : Attempt::kError;
}

// Every id bolt's catalog scan returns, plus the snapshot count.
bool scan_all(const std::string& base, std::vector<int64_t>* ids,
              uint32_t* n_snapshots) {
    FilesystemCatalog cat_fs{};
    Catalog cat{};
    if (!filesystem_catalog_init(&cat_fs, base.c_str(), &cat)) return false;
    bolt::Arena arena;
    TableHandle* rh = nullptr;
    if (!iceberg_table_open(&rh, &arena, &cat, "default", "t")) return false;
    *n_snapshots = iceberg_table_metadata(rh)->n_snapshots;
    ReadOptions ro{}; read_options_init(&ro);
    ScanHandle* sh = nullptr;
    if (!iceberg_scan_open(&sh, rh, &ro)) return false;
    for (uint32_t guard = 0; guard < 100000u; ++guard) {       // bounded
        bolt::BoltBatch batch{};
        bool eof = false;
        if (!iceberg_scan_next_batch(sh, &batch, &eof)) return false;
        if (eof) break;
        const auto& col = batch.columns[batch.read_epoch][0];
        const auto* vals = static_cast<const int64_t*>(col.data);
        for (int64_t r = 0; r < batch.num_rows; ++r) ids->push_back(vals[r]);
    }
    iceberg_scan_close(sh);
    iceberg_table_close(rh);
    return true;
}

std::vector<int64_t> expected_ids(const std::vector<std::pair<int32_t, int32_t>>& wc) {
    std::vector<int64_t> v;
    for (const auto& p : wc)
        for (int32_t r = 0; r < kRowsPerCommit; ++r)
            v.push_back(row_id(p.first, p.second, r));
    std::sort(v.begin(), v.end());
    return v;
}

}  // namespace

// Two handles opened on the same version; the second one to commit is stale.
// It must be refused with a conflict and the first writer's rows must be the
// table's rows, bit for bit. Repeated so the two commits land in the same
// millisecond often: that is where the old snapshot-id mint (now_ms * 1000 +
// n_snapshots) handed both the same id and the loser's data file overwrote
// the winner's already-committed one.
TEST(IcebergConcurrentCommit, StaleHandleIsRefusedAndWinnerSurvives) {
    for (int32_t round = 0; round < 60; ++round) {
        const std::string base = fresh_base("stale");
        ASSERT_FALSE(base.empty());
        const std::string root = table_root(base);
        ASSERT_TRUE(create_table(root));

        bolt::Arena arena_a, arena_b;
        FilesystemObjectStore fs_a{}, fs_b{}; ObjectStore os_a{}, os_b{};
        ASSERT_TRUE(filesystem_object_store_init(&fs_a, root.c_str(), &os_a));
        ASSERT_TRUE(filesystem_object_store_init(&fs_b, root.c_str(), &os_b));
        TableHandle* a = nullptr; TableHandle* b = nullptr;
        ASSERT_TRUE(table_open(&a, &arena_a, &os_a, root.c_str()));
        ASSERT_TRUE(table_open(&b, &arena_b, &os_b, root.c_str()));
        AppendHandle* aa = nullptr; AppendHandle* ab = nullptr;
        ASSERT_TRUE(append_open(&aa, a));
        ASSERT_TRUE(append_open(&ab, b));
        bolt::BoltBatch ba{}, bb{};
        ASSERT_TRUE(make_batch(&arena_a, &ba, 1, round));
        ASSERT_TRUE(make_batch(&arena_b, &bb, 2, round));
        ASSERT_TRUE(append_write(aa, &ba));
        ASSERT_TRUE(append_write(ab, &bb));

        ASSERT_TRUE(append_commit(aa)) << "round " << round;
        EXPECT_FALSE(append_commit(ab))
            << "round " << round << ": a handle opened before another "
               "writer's commit committed on top of it as if it were current";
        EXPECT_EQ(table_last_commit_error(b), CommitError::kConflict)
            << "round " << round;

        std::vector<int64_t> ids; uint32_t n_snaps = 0;
        ASSERT_TRUE(scan_all(base, &ids, &n_snaps));
        std::sort(ids.begin(), ids.end());
        EXPECT_EQ(n_snaps, 1u) << "round " << round;
        ASSERT_EQ(ids, expected_ids({{1, round}}))
            << "round " << round << ": the committed writer's rows were "
               "replaced — the loser overwrote a file the winner committed";
    }
}

// N threads line up on the same base version and commit at once, round after
// round. At most one per round may win; the losers must see a conflict; the
// table must hold exactly the winners' rows.
TEST(IcebergConcurrentCommit, RacingThreadsAtMostOneWinnerPerVersion) {
    const std::string base = fresh_base("race");
    ASSERT_FALSE(base.empty());
    const std::string root = table_root(base);
    ASSERT_TRUE(create_table(root));
    constexpr int32_t kThreads = 6;
    constexpr int32_t kRounds  = 25;
    std::vector<std::pair<int32_t, int32_t>> winners;
    int32_t conflicts = 0;
    for (int32_t round = 0; round < kRounds; ++round) {
        std::atomic<int32_t> ready{0};
        Attempt res[kThreads];
        std::vector<std::thread> ts;
        for (int32_t w = 0; w < kThreads; ++w) {
            ts.emplace_back([&, w] {
                res[w] = try_commit(root, w, round, &ready, kThreads);
            });
        }
        for (auto& t : ts) t.join();
        int32_t won = 0;
        for (int32_t w = 0; w < kThreads; ++w) {
            ASSERT_NE(res[w], Attempt::kError) << "round " << round << " w " << w;
            if (res[w] == Attempt::kCommitted) { ++won; winners.push_back({w, round}); }
            else ++conflicts;
        }
        EXPECT_LE(won, 1) << "round " << round << ": " << won
                          << " writers all committed on the same base version";
        EXPECT_GE(won, 1) << "round " << round << ": nobody committed";
    }
    std::vector<int64_t> ids; uint32_t n_snaps = 0;
    ASSERT_TRUE(scan_all(base, &ids, &n_snaps));
    std::sort(ids.begin(), ids.end());
    EXPECT_EQ(n_snaps, winners.size());
    EXPECT_EQ(ids, expected_ids(winners))
        << "reported-successful commits are missing from (or duplicated in) "
           "the table";
    EXPECT_GT(conflicts, 0);
}

// Retry-on-conflict writers: every append eventually lands, none is lost.
// Run as threads here and as separate processes below.
namespace {
constexpr int32_t kWriters = 4;
constexpr int32_t kCommitsPerWriter = 12;

int32_t writer_loop(const std::string& root, int32_t w) {
    for (int32_t c = 0; c < kCommitsPerWriter; ++c) {
        bool done = false;
        for (int32_t attempt = 0; attempt < 2000 && !done; ++attempt) {  // bounded
            const Attempt a = try_commit(root, w, c, nullptr, 0, attempt);
            if (a == Attempt::kCommitted) done = true;
            else if (a == Attempt::kError) return 1;
        }
        if (!done) return 2;
    }
    return 0;
}

std::vector<std::pair<int32_t, int32_t>> all_commits() {
    std::vector<std::pair<int32_t, int32_t>> v;
    for (int32_t w = 0; w < kWriters; ++w)
        for (int32_t c = 0; c < kCommitsPerWriter; ++c) v.push_back({w, c});
    return v;
}
}  // namespace

TEST(IcebergConcurrentCommit, RetryingThreadsLoseNothing) {
    const std::string base = fresh_base("threads");
    ASSERT_FALSE(base.empty());
    const std::string root = table_root(base);
    ASSERT_TRUE(create_table(root));
    int32_t rc[kWriters] = {};
    std::vector<std::thread> ts;
    for (int32_t w = 0; w < kWriters; ++w)
        ts.emplace_back([&, w] { rc[w] = writer_loop(root, w); });
    for (auto& t : ts) t.join();
    for (int32_t w = 0; w < kWriters; ++w) ASSERT_EQ(rc[w], 0) << "writer " << w;
    std::vector<int64_t> ids; uint32_t n_snaps = 0;
    ASSERT_TRUE(scan_all(base, &ids, &n_snaps));
    std::sort(ids.begin(), ids.end());
    EXPECT_EQ(n_snaps, static_cast<uint32_t>(kWriters * kCommitsPerWriter));
    EXPECT_EQ(ids, expected_ids(all_commits()));
}

#if !defined(_WIN32)
// Separate PROCESSES: fork gives each writer the parent's memory image —
// including any lazily-initialised "process-unique" seed — so this is also
// the case where a cached seed would make two writers mint identical names.
// Leaves the table at BOLT_ICEBERG_CONCURRENT_DIR for the external oracle.
TEST(IcebergConcurrentCommit, RetryingProcessesLoseNothing) {
    const std::string base = fresh_base("procs");
    ASSERT_FALSE(base.empty());
    const std::string root = table_root(base);
    ASSERT_TRUE(create_table(root));
    pid_t pids[kWriters];
    for (int32_t w = 0; w < kWriters; ++w) {
        const pid_t p = fork();
        ASSERT_GE(p, 0);
        if (p == 0) _exit(writer_loop(root, w));
        pids[w] = p;
    }
    for (int32_t w = 0; w < kWriters; ++w) {
        int st = 0;
        ASSERT_EQ(waitpid(pids[w], &st, 0), pids[w]);
        ASSERT_TRUE(WIFEXITED(st));
        EXPECT_EQ(WEXITSTATUS(st), 0) << "writer process " << w;
    }
    std::vector<int64_t> ids; uint32_t n_snaps = 0;
    ASSERT_TRUE(scan_all(base, &ids, &n_snaps));
    std::sort(ids.begin(), ids.end());
    EXPECT_EQ(n_snaps, static_cast<uint32_t>(kWriters * kCommitsPerWriter));
    EXPECT_EQ(ids, expected_ids(all_commits()));
    std::printf("fixture: %s\n", root.c_str());
}
#endif
