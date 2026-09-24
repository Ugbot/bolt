// G2ICE-176 — concurrent Delta committers against one table.
//
// Independent TableHandles line up on the same base version and commit at
// once. The property: every commit that REPORTS success is in the final
// table exactly once, and at most one writer wins each _delta_log version.
//
// Row rule: writer w, round c, row r  ->  id = w * 1'000'000 + c * 1000 + r

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_port.h"
#include "bolt/lakehouse/catalog.h"
#include "bolt/lakehouse/delta/snapshot.h"
#include "bolt/lakehouse/delta/writer.h"
#include "bolt/lakehouse/handle.h"
#include "bolt/lakehouse/object_store.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include "../src/lakehouse/delta_write_internal.h"

namespace {

using namespace bolt::lakehouse;
namespace dl = bolt::lakehouse::delta;

constexpr int32_t kRowsPerCommit = 4;

int64_t row_id(int32_t w, int32_t c, int32_t r) {
    return static_cast<int64_t>(w) * 1000000 + static_cast<int64_t>(c) * 1000 + r;
}

std::string fresh_root(const char* tag) {
    auto p = std::filesystem::temp_directory_path() /
             ("bolt_delta_concurrent_" + std::string(tag) + "_" +
              std::to_string(static_cast<long long>(bolt_getpid())));
    std::error_code ec;
    std::filesystem::remove_all(p, ec);
    std::filesystem::create_directories(p, ec);
    return p.generic_string();
}

void fill_batch(bolt::Arena* a, bolt::BoltBatch* b, int32_t w, int32_t c) {
    bolt::BoltBatch::init_empty(b);
    b->num_cols = 1;
    b->num_rows = kRowsPerCommit;
    bolt::BoltBatch::alloc_columns(b, a, 1);
    b->schema.add_field("id", bolt::BoltType::Int64, false);
    bolt::BoltColumn& col = b->columns[b->read_epoch][0];
    col = bolt::BoltColumn::make_flat_alloc(kRowsPerCommit,
                                            bolt::BoltType::Int64, a);
    auto* ip = static_cast<int64_t*>(col.data);
    for (int32_t r = 0; r < kRowsPerCommit; ++r) ip[r] = row_id(w, c, r);
}

bool create_table(const std::string& root) {
    FilesystemCatalog fc; Catalog cat;
    if (!filesystem_catalog_init(&fc, root.c_str(), &cat)) return false;
    bolt::Arena arena;
    bolt::BoltSchema schema;
    schema.add_field("id", bolt::BoltType::Int64, false);
    dl::WriteOptions wo;
    dl::write_options_init(&wo);
    wo.compression = Compression::kNone;
    TableHandle* th = nullptr;
    if (!dl::delta_table_create(&th, &arena, &cat, "ns", "t", &schema, &wo))
        return false;
    delta_table_close(th);
    return true;
}

// One append on a freshly opened handle (the TieringManager shape). `ready`
// is a barrier between staging and committing so every writer holds the
// same base version when it commits.
bool try_commit(const std::string& root, int32_t w, int32_t c,
                std::atomic<int32_t>* ready, int32_t n_ready) {
    FilesystemCatalog fc; Catalog cat;
    if (!filesystem_catalog_init(&fc, root.c_str(), &cat)) return false;
    bolt::Arena arena;
    TableHandle* th = nullptr;
    if (!delta_table_open(&th, &arena, &cat, "ns", "t")) return false;
    dl::AppendHandle* ah = nullptr;
    if (!dl::delta_append_open(&ah, th)) return false;
    bolt::Arena ba;
    bolt::BoltBatch b{};
    fill_batch(&ba, &b, w, c);
    if (!dl::delta_append_write(ah, &b)) return false;
    ready->fetch_add(1, std::memory_order_acq_rel);
    for (uint32_t spin = 0; spin < (1u << 30); ++spin) {   // bounded
        if (ready->load(std::memory_order_acquire) >= n_ready) break;
    }
    const bool ok = dl::delta_append_commit(ah);
    dl::delta_append_close(ah);
    delta_table_close(th);
    return ok;
}

std::vector<int64_t> scan_ids(const std::string& root) {
    FilesystemCatalog fc; Catalog cat;
    if (!filesystem_catalog_init(&fc, root.c_str(), &cat)) return {};
    bolt::Arena arena;
    TableHandle* th = nullptr;
    if (!delta_table_open(&th, &arena, &cat, "ns", "t")) return {};
    ScanHandle* sh = nullptr;
    if (!delta_scan_open(&sh, th, nullptr)) return {};
    std::vector<int64_t> out;
    for (uint32_t guard = 0; guard < 100000u; ++guard) {   // bounded
        bolt::BoltBatch batch{};
        bool eof = false;
        if (!delta_scan_next_batch(sh, &batch, &eof) || eof) break;
        const auto* ip = static_cast<const int64_t*>(
            batch.columns[batch.read_epoch][0].data);
        for (int64_t i = 0; i < batch.num_rows; ++i) out.push_back(ip[i]);
    }
    delta_scan_close(sh);
    delta_table_close(th);
    std::sort(out.begin(), out.end());
    return out;
}

int64_t latest_version(const std::string& root) {
    FilesystemObjectStore fs; ObjectStore os;
    if (!filesystem_object_store_init(&fs, root.c_str(), &os)) return -1;
    bolt::Arena a;
    dl::Snapshot snap{};
    if (!dl::delta_snapshot_build(&os, "ns/t", -1, &a, &snap)) return -1;
    return snap.version;
}

}  // namespace

TEST(DeltaConcurrentCommit, RacingThreadsAtMostOneWinnerPerVersion) {
    constexpr int32_t kWriters = 6;
    constexpr int32_t kRounds  = 40;
    const std::string root = fresh_root("threads");
    ASSERT_TRUE(create_table(root));

    std::vector<std::pair<int32_t, int32_t>> winners;
    int32_t multi_winner_rounds = 0;
    for (int32_t c = 0; c < kRounds; ++c) {
        std::atomic<int32_t> ready{0};
        std::atomic<bool> won[kWriters];
        std::vector<std::thread> ts;
        for (int32_t w = 0; w < kWriters; ++w) {
            won[w].store(false);
            ts.emplace_back([&, w] {
                won[w].store(try_commit(root, w, c, &ready, kWriters));
            });
        }
        for (auto& t : ts) t.join();
        int32_t n_won = 0;
        for (int32_t w = 0; w < kWriters; ++w) {
            if (won[w].load()) { ++n_won; winners.emplace_back(w, c); }
        }
        if (n_won > 1) ++multi_winner_rounds;
    }
    EXPECT_EQ(multi_winner_rounds, 0)
        << "several writers reported committing the same base version";

    std::vector<int64_t> expected;
    for (const auto& p : winners)
        for (int32_t r = 0; r < kRowsPerCommit; ++r)
            expected.push_back(row_id(p.first, p.second, r));
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(scan_ids(root), expected)
        << "a reported commit is missing from, or duplicated in, the table";
    EXPECT_EQ(latest_version(root), static_cast<int64_t>(winners.size()));

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

// The raw commit path (DELETE/UPDATE/MERGE/RESTORE/OPTIMIZE/DV) races too.
TEST(DeltaConcurrentCommit, RawCommitAtMostOneWinnerPerVersion) {
    constexpr int32_t kWriters = 6;
    constexpr int32_t kRounds  = 40;
    const std::string root = fresh_root("raw");
    ASSERT_TRUE(create_table(root));

    int32_t n_committed = 0, multi_winner_rounds = 0;
    for (int32_t c = 0; c < kRounds; ++c) {
        std::atomic<int32_t> ready{0};
        std::atomic<bool> won[kWriters];
        std::vector<std::thread> ts;
        for (int32_t w = 0; w < kWriters; ++w) {
            won[w].store(false);
            ts.emplace_back([&, w] {
                FilesystemCatalog fc; Catalog cat;
                if (!filesystem_catalog_init(&fc, root.c_str(), &cat)) return;
                bolt::Arena arena;
                TableHandle* th = nullptr;
                if (!delta_table_open(&th, &arena, &cat, "ns", "t")) return;
                int64_t base = dl::delta_writer_latest_version(th);
                char body[128];
                const int n = std::snprintf(body, sizeof(body),
                    "{\"commitInfo\":{\"timestamp\":%d,\"operation\":\"W%d\"}}\n",
                    c, w);
                ready.fetch_add(1, std::memory_order_acq_rel);
                for (uint32_t spin = 0; spin < (1u << 30); ++spin) {
                    if (ready.load(std::memory_order_acquire) >= kWriters) break;
                }
                won[w].store(dl::delta_writer_commit_raw(
                    th, body, static_cast<uint32_t>(n), &base));
                delta_table_close(th);
            });
        }
        for (auto& t : ts) t.join();
        int32_t n_won = 0;
        for (int32_t w = 0; w < kWriters; ++w) n_won += won[w].load() ? 1 : 0;
        if (n_won > 1) ++multi_winner_rounds;
        n_committed += n_won;
    }
    EXPECT_EQ(multi_winner_rounds, 0);
    EXPECT_EQ(latest_version(root), static_cast<int64_t>(n_committed));

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

// A store without an atomic create must refuse the commit, never fall back
// to a racy head-then-put.
TEST(DeltaConcurrentCommit, StoreWithoutAtomicCreateRefusesCommit) {
    const std::string root = fresh_root("nocas");
    ASSERT_TRUE(create_table(root));

    FilesystemCatalog fc; Catalog cat;
    ASSERT_TRUE(filesystem_catalog_init(&fc, root.c_str(), &cat));
    bolt::Arena arena;
    TableHandle* th = nullptr;
    ASSERT_TRUE(delta_table_open(&th, &arena, &cat, "ns", "t"));
    ObjectStoreVT vt = *th->os.vt;
    vt.put_if_absent = nullptr;
    th->os.vt = &vt;

    dl::AppendHandle* ah = nullptr;
    ASSERT_TRUE(dl::delta_append_open(&ah, th));
    bolt::Arena ba;
    bolt::BoltBatch b{};
    fill_batch(&ba, &b, 0, 0);
    ASSERT_TRUE(dl::delta_append_write(ah, &b));
    EXPECT_FALSE(dl::delta_append_commit(ah));
    dl::delta_append_close(ah);

    int64_t base = 0;
    const char body[] = "{\"commitInfo\":{\"timestamp\":0}}\n";
    EXPECT_FALSE(dl::delta_writer_commit_raw(th, body, sizeof(body) - 1u, &base));
    EXPECT_EQ(base, 0);
    delta_table_close(th);
    EXPECT_EQ(latest_version(root), 0);

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}
