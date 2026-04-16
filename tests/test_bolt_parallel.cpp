// test_bolt_parallel.cpp — GTest suite for morsel-parallel kernel wrappers.
//
// Covers parallel_filter_gt, parallel_sum, parallel_groupby_sum_int64.
// Each kernel has a serial-fallback test and a parallel-matches-serial test.

#include <gtest/gtest.h>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_scheduler.h"
#include "bolt/bolt_types.h"
#include "bolt/join/bolt_groupby.h"
#include "bolt/kernels/bolt_numeric.h"
#include "bolt/kernels/bolt_parallel.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <thread>
#include <vector>

using namespace bolt;
using namespace bolt::kernels;

namespace {

// Build a small scheduler config with a tiny grain so even modest test sizes
// fan out across multiple morsels (and thus exercise the parallel path).
SchedulerConfig make_test_cfg() noexcept {
    SchedulerConfig cfg;
    const unsigned hw = std::thread::hardware_concurrency();
    cfg.num_workers = (hw >= 4) ? 4u : (hw >= 2 ? hw : 2u);
    cfg.grain_bytes = 4096;  // small grain → many morsels for a few-thousand-row test
    return cfg;
}

}  // namespace

// ---------------------------------------------------------------------------
// parallel_filter_gt
// ---------------------------------------------------------------------------

TEST(BoltParallel, FilterGtSerialFallback) {
    std::vector<int32_t> data(500);
    for (int i = 0; i < 500; ++i) data[i] = i;
    std::vector<int32_t> out_par(500), out_ser(500);

    int64_t n_par = parallel_filter_gt<int32_t>(
        /*sched=*/nullptr, data.data(), 500, 100, out_par.data(), nullptr);
    int64_t n_ser = filter_gt<int32_t>(data.data(), 500, 100, out_ser.data());

    EXPECT_EQ(n_par, n_ser);
    EXPECT_EQ(n_par, 399);
    for (int64_t i = 0; i < n_par; ++i) EXPECT_EQ(out_par[i], out_ser[i]);
}

TEST(BoltParallel, FilterGtParallelMatchesSerial) {
    constexpr int64_t N = 10000;
    std::vector<int32_t> data(N);
    for (int64_t i = 0; i < N; ++i) data[i] = static_cast<int32_t>((i * 2654435761u) & 0xFFFF);

    std::vector<int32_t> out_ser(N + 32, 0);
    int64_t n_ser = filter_gt<int32_t>(data.data(), N, 30000, out_ser.data());

    Scheduler sched{};
    ASSERT_TRUE(sched.init(make_test_cfg()));

    Arena arena;
    std::vector<int32_t> out_par(N + 32, 0);
    int64_t n_par = parallel_filter_gt<int32_t>(
        &sched, data.data(), N, 30000, out_par.data(), &arena);

    // Per-morsel scatter + serial compaction yields ascending indices by
    // construction (morsels processed in index order during the final copy).
    EXPECT_EQ(n_par, n_ser);
    for (int64_t i = 0; i < n_par; ++i) EXPECT_EQ(out_par[i], out_ser[i]);

    sched.shutdown();
}

TEST(BoltParallel, ParallelFilterDeterministicOrder) {
    constexpr int64_t N = 8000;
    std::vector<int32_t> data(N);
    for (int64_t i = 0; i < N; ++i) data[i] = static_cast<int32_t>(i);

    Scheduler sched{};
    ASSERT_TRUE(sched.init(make_test_cfg()));

    Arena a1, a2;
    std::vector<int32_t> out1(N, 0), out2(N, 0);

    int64_t n1 = parallel_filter_gt<int32_t>(&sched, data.data(), N, 100, out1.data(), &a1);
    int64_t n2 = parallel_filter_gt<int32_t>(&sched, data.data(), N, 100, out2.data(), &a2);

    EXPECT_EQ(n1, n2);
    for (int64_t i = 0; i < n1; ++i) EXPECT_EQ(out1[i], out2[i]);
    // Also check ascending.
    for (int64_t i = 1; i < n1; ++i) EXPECT_LT(out1[i - 1], out1[i]);

    sched.shutdown();
}

// ---------------------------------------------------------------------------
// parallel_sum
// ---------------------------------------------------------------------------

TEST(BoltParallel, SumSerialFallback) {
    std::vector<int64_t> data(1000);
    for (int64_t i = 0; i < 1000; ++i) data[i] = i + 1;
    int64_t expected = 1000 * 1001 / 2;

    int64_t s_par = parallel_sum<int64_t>(nullptr, data.data(), 1000);
    int64_t s_ser = sum<int64_t>(data.data(), 1000);
    EXPECT_EQ(s_par, expected);
    EXPECT_EQ(s_par, s_ser);
}

TEST(BoltParallel, SumParallelMatchesSerial) {
    constexpr int64_t N = 20000;
    std::vector<int64_t> data(N);
    for (int64_t i = 0; i < N; ++i) data[i] = (i % 7) - 3;

    int64_t s_ser = sum<int64_t>(data.data(), N);

    Scheduler sched{};
    ASSERT_TRUE(sched.init(make_test_cfg()));

    int64_t s_par = parallel_sum<int64_t>(&sched, data.data(), N);
    EXPECT_EQ(s_par, s_ser);

    sched.shutdown();
}

// ---------------------------------------------------------------------------
// parallel_groupby_sum_int64
// ---------------------------------------------------------------------------

namespace {

void build_keys_values(int64_t n, uint32_t num_distinct,
                       std::vector<int64_t>& keys,
                       std::vector<int64_t>& values) {
    keys.resize(n);
    values.resize(n);
    for (int64_t i = 0; i < n; ++i) {
        keys[i]   = static_cast<int64_t>(i % num_distinct);
        values[i] = i + 1;  // deterministic, non-trivial sums
    }
}

std::map<int64_t, int64_t> to_map(const BoltColumn& keys, const BoltColumn& sums) {
    std::map<int64_t, int64_t> out;
    const int64_t n = keys.length;
    const int64_t* k = static_cast<const int64_t*>(keys.data);
    const int64_t* s = static_cast<const int64_t*>(sums.data);
    for (int64_t i = 0; i < n; ++i) out[k[i]] = s[i];
    return out;
}

}  // namespace

TEST(BoltParallel, GroupbySerialFallback) {
    std::vector<int64_t> keys, values;
    build_keys_values(500, 5, keys, values);

    Arena arena_s, arena_p;
    BoltColumn key_col = BoltColumn::make_flat(keys.data(), nullptr, 500, BoltType::Int64);
    BoltColumn val_col = BoltColumn::make_flat(values.data(), nullptr, 500, BoltType::Int64);

    BoltColumn out_k_s{}, out_s_s{}, out_k_p{}, out_s_p{};
    ASSERT_TRUE(groupby_sum_int64(key_col, val_col, &arena_s, &out_k_s, &out_s_s));
    ASSERT_TRUE(parallel_groupby_sum_int64(nullptr, key_col, val_col,
                                           &arena_p, &out_k_p, &out_s_p));
    EXPECT_EQ(to_map(out_k_s, out_s_s), to_map(out_k_p, out_s_p));
}

TEST(BoltParallel, GroupbyParallelMatchesSerial) {
    constexpr int64_t N = 10000;
    std::vector<int64_t> keys, values;
    build_keys_values(N, 10, keys, values);

    Arena arena_s;
    BoltColumn key_col = BoltColumn::make_flat(keys.data(), nullptr, N, BoltType::Int64);
    BoltColumn val_col = BoltColumn::make_flat(values.data(), nullptr, N, BoltType::Int64);

    BoltColumn out_k_s{}, out_s_s{};
    ASSERT_TRUE(groupby_sum_int64(key_col, val_col, &arena_s, &out_k_s, &out_s_s));
    auto ref = to_map(out_k_s, out_s_s);

    Scheduler sched{};
    ASSERT_TRUE(sched.init(make_test_cfg()));

    Arena arena_p;
    BoltColumn out_k_p{}, out_s_p{};
    ASSERT_TRUE(parallel_groupby_sum_int64(&sched, key_col, val_col,
                                           &arena_p, &out_k_p, &out_s_p));
    EXPECT_EQ(to_map(out_k_p, out_s_p), ref);
    EXPECT_EQ(static_cast<size_t>(out_k_p.length), ref.size());

    sched.shutdown();
}

TEST(BoltParallel, GroupbyConstantKeysFastPath) {
    constexpr int64_t N = 1000;
    std::vector<int64_t> values(N);
    for (int64_t i = 0; i < N; ++i) values[i] = i;

    BoltColumn key_col = BoltColumn::make_constant<int64_t>(42, N, BoltType::Int64);
    BoltColumn val_col = BoltColumn::make_flat(values.data(), nullptr, N, BoltType::Int64);

    Scheduler sched{};
    ASSERT_TRUE(sched.init(make_test_cfg()));

    Arena arena;
    BoltColumn out_k{}, out_s{};
    ASSERT_TRUE(parallel_groupby_sum_int64(&sched, key_col, val_col,
                                           &arena, &out_k, &out_s));
    ASSERT_EQ(out_k.length, 1);
    ASSERT_EQ(out_s.length, 1);
    EXPECT_EQ(static_cast<const int64_t*>(out_k.data)[0], 42);

    int64_t expected = N * (N - 1) / 2;
    EXPECT_EQ(static_cast<const int64_t*>(out_s.data)[0], expected);

    sched.shutdown();
}
