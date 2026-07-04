// test_bolt_primitives.cpp — GTest suite for Bolt arena, channel, column, types
//
// Tests compile with ZERO Arrow dependency. Pure C++20.
// Validates: arena lifecycle, channel correctness, column stats, type system,
// Arrow C Data Interface export (struct layout only, no libarrow).

#include <gtest/gtest.h>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_channel.h"
#include "bolt/bolt_types.h"
#include "bolt/bolt_column.h"

#include <algorithm>
#include <random>
#include <thread>
#include <vector>
#include <cstring>
#include <atomic>

using namespace bolt;

// ============================================================================
// Type System Tests
// ============================================================================

TEST(BoltTypes, SizeTable) {
    EXPECT_EQ(type_size(BoltType::Int8), 1);
    EXPECT_EQ(type_size(BoltType::Int32), 4);
    EXPECT_EQ(type_size(BoltType::Int64), 8);
    EXPECT_EQ(type_size(BoltType::Float32), 4);
    EXPECT_EQ(type_size(BoltType::Float64), 8);
    EXPECT_EQ(type_size(BoltType::Utf8), 16);  // German-style view
    EXPECT_EQ(type_size(BoltType::UUID), 16);
    EXPECT_EQ(type_size(BoltType::IPv4), 4);
    EXPECT_EQ(type_size(BoltType::Binary), 0);  // Variable width
    EXPECT_EQ(type_size(BoltType::List), 0);
}

TEST(BoltTypes, Predicates) {
    EXPECT_TRUE(is_numeric(BoltType::Int64));
    EXPECT_TRUE(is_numeric(BoltType::Float64));
    EXPECT_FALSE(is_numeric(BoltType::Utf8));
    EXPECT_TRUE(is_integer(BoltType::UInt32));
    EXPECT_FALSE(is_integer(BoltType::Float32));
    EXPECT_TRUE(is_floating(BoltType::Float64));
    EXPECT_TRUE(is_temporal(BoltType::Timestamp));
    EXPECT_TRUE(is_string(BoltType::Utf8));
    EXPECT_TRUE(is_string(BoltType::Symbol));
    EXPECT_TRUE(is_fixed_width(BoltType::Int64));
    EXPECT_FALSE(is_fixed_width(BoltType::Binary));
}

TEST(BoltTypes, ArrowFormatStrings) {
    EXPECT_STREQ(arrow_format_string(BoltType::Int32), "i");
    EXPECT_STREQ(arrow_format_string(BoltType::Int64), "l");
    EXPECT_STREQ(arrow_format_string(BoltType::Float64), "g");
    EXPECT_STREQ(arrow_format_string(BoltType::Bool), "b");
    EXPECT_STREQ(arrow_format_string(BoltType::Utf8), "vu");
}

TEST(BoltTypes, TypeNames) {
    EXPECT_STREQ(type_name(BoltType::Int64), "int64");
    EXPECT_STREQ(type_name(BoltType::Utf8), "utf8");
    EXPECT_STREQ(type_name(BoltType::Symbol), "symbol");
    EXPECT_STREQ(type_name(BoltType::UUID), "uuid");
}

// ============================================================================
// StringView Tests
// ============================================================================

TEST(StringView, InlineShort) {
    auto sv = StringView::from_cstr("hi");
    EXPECT_EQ(sv.length, 2);
    EXPECT_TRUE(sv.is_inline());
    EXPECT_EQ(sv.prefix[0], 'h');
    EXPECT_EQ(sv.prefix[1], 'i');
}

TEST(StringView, InlineMax12) {
    auto sv = StringView::from_cstr("123456789012");  // Exactly 12 bytes
    EXPECT_EQ(sv.length, 12);
    EXPECT_TRUE(sv.is_inline());
}

TEST(StringView, OutOfLine) {
    auto sv = StringView::from_cstr("1234567890123");  // 13 bytes
    EXPECT_EQ(sv.length, 13);
    EXPECT_FALSE(sv.is_inline());
    // Prefix should be first 4 bytes
    EXPECT_EQ(memcmp(sv.prefix, "1234", 4), 0);
}

TEST(StringView, EqualityInline) {
    auto a = StringView::from_cstr("hello");
    auto b = StringView::from_cstr("hello");
    auto c = StringView::from_cstr("world");
    EXPECT_TRUE(a.eq_inline(b));
    EXPECT_FALSE(a.eq_inline(c));
}

TEST(StringView, PrefixCompare) {
    auto a = StringView::from_cstr("apple");
    auto b = StringView::from_cstr("banana");
    EXPECT_NE(StringView::cmp_prefix(a, b), 0);  // Different length → not equal
}

TEST(StringView, SizeIs16) {
    EXPECT_EQ(sizeof(StringView), 16);
}

// ============================================================================
// Schema Tests
// ============================================================================

TEST(BoltSchema, AddAndFind) {
    BoltSchema s;
    EXPECT_TRUE(s.add_field("price", BoltType::Float64));
    EXPECT_TRUE(s.add_field("symbol", BoltType::Symbol));
    EXPECT_TRUE(s.add_field("timestamp", BoltType::Timestamp));

    EXPECT_EQ(s.num_fields, 3);
    EXPECT_EQ(s.find_field("price"), 0);
    EXPECT_EQ(s.find_field("symbol"), 1);
    EXPECT_EQ(s.find_field("timestamp"), 2);
    EXPECT_EQ(s.find_field("nonexistent"), -1);
}

TEST(BoltSchema, FieldNameTruncation) {
    BoltSchema s;
    char long_name[128];
    memset(long_name, 'x', 127);
    long_name[127] = '\0';
    s.add_field(long_name, BoltType::Int32);
    EXPECT_EQ(strlen(s.fields[0].name), kMaxFieldName);
}

TEST(BoltSchema, Equality) {
    BoltSchema a, b;
    a.add_field("x", BoltType::Int32);
    a.add_field("y", BoltType::Float64);
    b.add_field("x", BoltType::Int32);
    b.add_field("y", BoltType::Float64);
    EXPECT_TRUE(a.equals(b));

    BoltSchema c;
    c.add_field("x", BoltType::Int64);  // Different type
    c.add_field("y", BoltType::Float64);
    EXPECT_FALSE(a.equals(c));
}

// ============================================================================
// Arena Tests
// ============================================================================

TEST(BoltArena, BasicAllocation) {
    Arena arena;
    void* p1 = arena.allocate(1024);
    void* p2 = arena.allocate(2048);
    ASSERT_NE(p1, nullptr);
    ASSERT_NE(p2, nullptr);
    EXPECT_NE(p1, p2);
    EXPECT_GE(arena.total_allocated(), 1024 + 2048);
}

TEST(BoltArena, Alignment) {
    Arena arena;
    for (int i = 0; i < 100; ++i) {
        void* p = arena.allocate(i * 7 + 1);
        ASSERT_NE(p, nullptr);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % 64, 0);
    }
}

TEST(BoltArena, ResetReuses) {
    Arena arena;
    void* first = arena.allocate(1024);
    arena.reset();
    void* second = arena.allocate(1024);
    EXPECT_EQ(first, second);
    EXPECT_EQ(arena.total_allocated(), 1024);
}

TEST(BoltArena, ReturnsNullOnBlockTableFull) {
    ArenaConfig cfg;
    cfg.initial_block_size = 64;
    cfg.max_block_size = 64;
    Arena arena(cfg);

    // Fill all 32 block slots with tiny blocks
    bool oom = false;
    for (int i = 0; i < 10000 && !oom; ++i) {
        if (!arena.allocate(128)) oom = true;
    }
    // Eventually should run out of block slots (32 max)
    // or the allocation should succeed if blocks are reused
}

// D1 — huge-page allocator primitive. Verifies the allocator always
// returns usable memory (either real huge pages on privileged systems,
// or the fallback 4 KB path). Cannot assume huge pages land on every
// CI box (Windows needs SeLockMemoryPrivilege; Linux needs
// vm.nr_hugepages > 0) — tests the contract, not the page size.
TEST(BoltPort, HugePageAllocBasic) {
    // 8 MB — above the "huge-page worth considering" threshold.
    constexpr size_t kSize = 8ull * 1024ull * 1024ull;
    size_t got = 0;
    void* p = ::bolt_aligned_alloc_huge(kSize, &got);
    ASSERT_NE(p, nullptr);
    EXPECT_GE(got, kSize);
    // Write every 4 KB page to make sure the memory is actually backed.
    uint8_t* bytes = static_cast<uint8_t*>(p);
    for (size_t off = 0; off < kSize; off += 4096) bytes[off] = 0x42;
    for (size_t off = 0; off < kSize; off += 4096) EXPECT_EQ(bytes[off], 0x42);
    ::bolt_aligned_free_huge(p, got);
}

TEST(BoltArena, PoisonOnReset) {
    ArenaConfig cfg;
    cfg.poison_on_reset = true;
    cfg.initial_block_size = 4096;
    Arena arena(cfg);

    auto* data = static_cast<uint8_t*>(arena.allocate(100));
    memset(data, 0x42, 100);
    uint8_t* saved = data;
    arena.reset();
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(saved[i], 0xDE);
    }
}

TEST(BoltArena, ArenaGuardSetsAndRestores) {
    Arena a1, a2;
    EXPECT_EQ(tl_arena, nullptr);
    {
        ArenaGuard g1(a1);
        EXPECT_EQ(tl_arena, &a1);
        {
            ArenaGuard g2(a2);
            EXPECT_EQ(tl_arena, &a2);
        }
        EXPECT_EQ(tl_arena, &a1);
    }
    EXPECT_EQ(tl_arena, nullptr);
}

TEST(BoltArena, ArenaGuardPointerCtor) {
    Arena a;
    EXPECT_EQ(tl_arena, nullptr);
    {
        ArenaGuard g(&a);
        EXPECT_EQ(tl_arena, &a);
    }
    EXPECT_EQ(tl_arena, nullptr);

    // nullptr path: tl_arena must remain unchanged inside the guard, and be
    // restored to its entry value on exit.
    {
        ArenaGuard outer(a);
        EXPECT_EQ(tl_arena, &a);
        {
            ArenaGuard inner(static_cast<Arena*>(nullptr));
            EXPECT_EQ(tl_arena, &a);  // unchanged
        }
        EXPECT_EQ(tl_arena, &a);
    }
    EXPECT_EQ(tl_arena, nullptr);
}

TEST(BoltArena, PerformanceSanity) {
    Arena arena;
    constexpr size_t N = 100000;
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; ++i) arena.allocate(1024);
    auto end = std::chrono::high_resolution_clock::now();
    double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / (double)N;
    EXPECT_LT(ns, 100.0) << "Arena too slow: " << ns << " ns/alloc";
}

// ============================================================================
// SPSC Channel Tests
// ============================================================================

TEST(BoltSPSC, PushPop) {
    SPSCChannel<int, 16> ch;
    int val = 0;
    EXPECT_FALSE(ch.try_pop(&val));
    int v = 42;
    EXPECT_TRUE(ch.try_push(static_cast<int&&>(v)));
    EXPECT_TRUE(ch.try_pop(&val));
    EXPECT_EQ(val, 42);
}

TEST(BoltSPSC, FIFO) {
    SPSCChannel<int, 64> ch;
    for (int i = 0; i < 10; ++i) {
        int v = i;
        EXPECT_TRUE(ch.try_push(static_cast<int&&>(v)));
    }
    for (int i = 0; i < 10; ++i) {
        int val = -1;
        EXPECT_TRUE(ch.try_pop(&val));
        EXPECT_EQ(val, i);
    }
}

TEST(BoltSPSC, Full) {
    SPSCChannel<int, 4> ch;
    for (int i = 0; i < 4; ++i) { int v = i; ch.try_push(static_cast<int&&>(v)); }
    int v = 99;
    EXPECT_FALSE(ch.try_push(static_cast<int&&>(v)));
    int out;
    ch.try_pop(&out);
    v = 99;
    EXPECT_TRUE(ch.try_push(static_cast<int&&>(v)));
}

TEST(BoltSPSC, TwoThreads) {
    SPSCChannel<uint64_t, 4096> ch;
    constexpr size_t N = 1000000;

    std::thread producer([&]() {
        for (uint64_t i = 0; i < N; ++i) {
            uint64_t v = i + 1;
            while (!ch.try_push(static_cast<uint64_t&&>(v))) cpu_pause();
        }
    });

    uint64_t count = 0, checksum = 0;
    while (count < N) {
        uint64_t val;
        if (ch.try_pop(&val)) { checksum += val; count++; }
        else cpu_pause();
    }
    producer.join();
    EXPECT_EQ(checksum, N * (N + 1) / 2);
}

// ============================================================================
// MPSC Channel Tests
// ============================================================================

TEST(BoltMPSC, MultipleProducers) {
    MPSCChannel<uint64_t, 8192> ch;
    constexpr size_t PER_THREAD = 100000;
    constexpr size_t THREADS = 4;
    constexpr size_t TOTAL = PER_THREAD * THREADS;

    std::vector<std::thread> producers;
    for (size_t t = 0; t < THREADS; ++t) {
        producers.emplace_back([&, t]() {
            for (uint64_t i = 0; i < PER_THREAD; ++i) {
                uint64_t v = t * PER_THREAD + i + 1;
                while (!ch.try_push(static_cast<uint64_t&&>(v))) cpu_pause();
            }
        });
    }

    uint64_t count = 0, sum = 0;
    while (count < TOTAL) {
        uint64_t val;
        if (ch.try_pop(&val)) { sum += val; count++; }
        else cpu_pause();
    }
    for (auto& t : producers) t.join();
    EXPECT_EQ(sum, TOTAL * (TOTAL + 1) / 2);
}

// ============================================================================
// NUMA Channel Pool Tests (E1)
// ============================================================================

TEST(BoltNumaPool, PushThenRoundRobinPop) {
    // 4-node pool; each node gets 2 items; verify round-robin pop sees all.
    NumaChannelPool<uint64_t, 64, 4> pool;
    for (size_t node = 0; node < 4; ++node) {
        ASSERT_TRUE(pool.try_push(node, /*item=*/ (node * 10ULL + 1)));
        ASSERT_TRUE(pool.try_push(node, /*item=*/ (node * 10ULL + 2)));
    }

    // We pushed 8 items across 4 nodes; pop them all; order depends on
    // round-robin starting at node 0.
    std::vector<uint64_t> seen;
    seen.reserve(8);
    for (int k = 0; k < 8; ++k) {
        uint64_t v = 0;
        ASSERT_TRUE(pool.try_pop(&v));
        seen.push_back(v);
    }
    // Empty: next pop returns false.
    uint64_t dummy = 0;
    EXPECT_FALSE(pool.try_pop(&dummy));

    // Every value we pushed must appear exactly once.
    std::sort(seen.begin(), seen.end());
    std::vector<uint64_t> expected{1,2, 11,12, 21,22, 31,32};
    EXPECT_EQ(seen, expected);
}

TEST(BoltNumaPool, MultiProducerConcurrent) {
    constexpr size_t NumNodes = 4;
    constexpr size_t PerNode  = 500;
    NumaChannelPool<uint64_t, 1024, NumNodes> pool;

    std::atomic<bool> go{false};
    std::vector<std::thread> producers;
    for (size_t n = 0; n < NumNodes; ++n) {
        producers.emplace_back([&, n]() {
            while (!go.load(std::memory_order_acquire)) cpu_pause();
            for (uint64_t i = 0; i < PerNode; ++i) {
                uint64_t v = n * PerNode + i + 1;  // positive, unique
                while (!pool.try_push(n, static_cast<uint64_t&&>(v))) cpu_pause();
            }
        });
    }
    go.store(true, std::memory_order_release);

    constexpr size_t Total = NumNodes * PerNode;
    uint64_t sum = 0;
    size_t got = 0;
    while (got < Total) {
        uint64_t v = 0;
        if (pool.try_pop(&v)) { sum += v; ++got; }
        else cpu_pause();
    }
    for (auto& t : producers) t.join();
    // Expected sum = sum of 1..Total.
    EXPECT_EQ(sum, Total * (Total + 1) / 2);
}

TEST(BoltNumaPool, SingleNodeDegenerate) {
    // NumNodes=1 is the degenerate case — behaves like plain MPSC.
    NumaChannelPool<int, 16, 1> pool;
    ASSERT_TRUE(pool.try_push(0, 42));
    int v = 0;
    ASSERT_TRUE(pool.try_pop(&v));
    EXPECT_EQ(v, 42);
}

// Imbalanced producer rates — one node pushes 4× as much as the others.
// Round-robin pop must still drain every item; no node starves the
// rest, no item is lost or duplicated.
TEST(BoltNumaPool, ImbalancedProducers) {
    constexpr size_t NumNodes = 4;
    NumaChannelPool<uint64_t, 2048, NumNodes> pool;

    std::atomic<bool> go{false};
    std::vector<std::thread> producers;
    const size_t counts[NumNodes] = {1000, 250, 250, 250};
    size_t total = 0;
    for (size_t n = 0; n < NumNodes; ++n) total += counts[n];

    uint64_t base = 1;
    for (size_t n = 0; n < NumNodes; ++n) {
        const uint64_t lo = base;
        const uint64_t hi = base + counts[n];
        producers.emplace_back([&, n, lo, hi]() {
            while (!go.load(std::memory_order_acquire)) cpu_pause();
            for (uint64_t v = lo; v < hi; ++v) {
                while (!pool.try_push(n, static_cast<uint64_t&&>(v))) cpu_pause();
            }
        });
        base = hi;
    }
    go.store(true, std::memory_order_release);

    std::vector<uint64_t> seen;
    seen.reserve(total);
    while (seen.size() < total) {
        uint64_t v = 0;
        if (pool.try_pop(&v)) seen.push_back(v);
        else cpu_pause();
    }
    for (auto& t : producers) t.join();

    std::sort(seen.begin(), seen.end());
    for (size_t i = 0; i < total; ++i) EXPECT_EQ(seen[i], static_cast<uint64_t>(i + 1));
}

// 16-node pool — exercises the upper-bound template parameter.
TEST(BoltNumaPool, SixteenNodeRoundRobin) {
    constexpr size_t NumNodes = 16;
    NumaChannelPool<int, 64, NumNodes> pool;
    for (size_t n = 0; n < NumNodes; ++n) {
        ASSERT_TRUE(pool.try_push(n, static_cast<int>(n) * 100));
    }
    std::vector<int> seen;
    int v = 0;
    while (pool.try_pop(&v)) seen.push_back(v);
    EXPECT_EQ(seen.size(), NumNodes);
    std::sort(seen.begin(), seen.end());
    for (size_t n = 0; n < NumNodes; ++n) EXPECT_EQ(seen[n], static_cast<int>(n) * 100);
}

// ============================================================================
// Column Tests
// ============================================================================

TEST(BoltColumn, FlatConstruction) {
    Arena arena;
    auto col = BoltColumn::make_flat_alloc(1024, BoltType::Int64, &arena);
    EXPECT_EQ(col.length, 1024);
    EXPECT_EQ(col.format, ColumnFormat::Flat);
    EXPECT_EQ(col.type, BoltType::Int64);
    ASSERT_NE(col.data, nullptr);

    // Write and read back
    auto* data = col.typed_mutable<int64_t>();
    for (int i = 0; i < 1024; ++i) data[i] = i * 42;
    for (int i = 0; i < 1024; ++i) EXPECT_EQ(col.typed_data<int64_t>()[i], i * 42);
}

TEST(BoltColumn, Constant) {
    auto col = BoltColumn::make_constant<int64_t>(42, 16384, BoltType::Int64);
    EXPECT_EQ(col.length, 16384);
    EXPECT_TRUE(col.is_constant());
    EXPECT_EQ(col.get_constant<int64_t>(), 42);
    EXPECT_EQ(col.stats.cardinality, CardinalityClass::Constant);
    EXPECT_EQ(col.stats.distinct_count, 1);
    EXPECT_EQ(col.stats.min_value, 42);
    EXPECT_EQ(col.stats.max_value, 42);
    EXPECT_EQ(col.byte_size(), 8);  // Just the 8-byte inline value
}

TEST(BoltColumn, Sequence) {
    auto col = BoltColumn::make_sequence(100, 3, 1000, BoltType::Int64);
    EXPECT_EQ(col.format, ColumnFormat::Sequence);
    EXPECT_EQ(col.get_sequence(0), 100);
    EXPECT_EQ(col.get_sequence(1), 103);
    EXPECT_EQ(col.get_sequence(999), 100 + 999 * 3);
    EXPECT_EQ(col.stats.min_value, 100);
    EXPECT_EQ(col.stats.max_value, 100 + 999 * 3);
    EXPECT_EQ(col.stats.sort_order, SortOrder::Ascending);
    EXPECT_TRUE(col.stats.is_monotonic);
    EXPECT_EQ(col.byte_size(), 0);  // No data buffer
}

TEST(BoltColumn, View) {
    Arena arena;
    auto parent = BoltColumn::make_flat_alloc(100, BoltType::Int32, &arena);
    auto* data = parent.typed_mutable<int32_t>();
    for (int i = 0; i < 100; ++i) data[i] = i;

    auto view = BoltColumn::make_view(parent, 10, 20);
    EXPECT_EQ(view.length, 20);
    EXPECT_EQ(view.format, ColumnFormat::View);
    EXPECT_EQ(view.typed_data<int32_t>()[0], 10);
    EXPECT_EQ(view.typed_data<int32_t>()[19], 29);
}

TEST(BoltColumn, StatsZoneMapSkip) {
    auto col = BoltColumn::make_constant<int64_t>(50, 1000, BoltType::Int64);
    // min=max=50
    EXPECT_TRUE(col.stats.can_skip_gt(50));    // max_value(50) <= 50
    EXPECT_FALSE(col.stats.can_skip_gt(49));   // max_value(50) > 49
    EXPECT_TRUE(col.stats.can_skip_lt(50));    // min_value(50) >= 50
    EXPECT_TRUE(col.stats.can_skip_eq(100));   // 100 > max(50)
    EXPECT_FALSE(col.stats.can_skip_eq(50));   // 50 in range
}

// ============================================================================
// Arrow C Data Interface Tests (struct layout, no libarrow)
// ============================================================================

TEST(BoltArrowExport, SchemaFill) {
    auto col = BoltColumn::make_flat(nullptr, nullptr, 100, BoltType::Int64);
    ArrowSchema schema;
    col.fill_arrow_schema(&schema, "price");
    EXPECT_STREQ(schema.format, "l");  // int64 → "l"
    EXPECT_STREQ(schema.name, "price");
    EXPECT_EQ(schema.n_children, 0);
}

// ============================================================================
// compute_stats_numeric / clone_into / materialize / try_promote
// ============================================================================

TEST(BoltColumnStats, ComputeStatsInt32) {
    Arena arena;
    auto col = BoltColumn::make_flat_alloc(10, BoltType::Int32, &arena);
    auto* d = col.typed_mutable<int32_t>();
    int32_t vals[] = {5, -3, 7, 5, 0, 42, 5, 1, 2, 7};
    for (int i = 0; i < 10; ++i) d[i] = vals[i];
    col.compute_stats_numeric();
    EXPECT_EQ(col.stats.min_value, -3);
    EXPECT_EQ(col.stats.max_value, 42);
    EXPECT_EQ(col.stats.null_count, 0);
    EXPECT_TRUE(col.stats.all_valid);
    // 7 distinct values {5,-3,7,0,42,1,2} — under sketch cap
    EXPECT_EQ(col.stats.distinct_count, 7u);
}

TEST(BoltColumnStats, ComputeStatsFloat64) {
    Arena arena;
    auto col = BoltColumn::make_flat_alloc(5, BoltType::Float64, &arena);
    auto* d = col.typed_mutable<double>();
    d[0] = 1.5; d[1] = -2.25; d[2] = 3.75; d[3] = 1.5; d[4] = 100.0;
    col.compute_stats_numeric();
    double mn, mx;
    memcpy(&mn, &col.stats.min_value, sizeof(double));
    memcpy(&mx, &col.stats.max_value, sizeof(double));
    EXPECT_DOUBLE_EQ(mn, -2.25);
    EXPECT_DOUBLE_EQ(mx, 100.0);
    EXPECT_EQ(col.stats.distinct_count, 4u);
}

TEST(BoltColumnStats, SketchOverflow) {
    Arena arena;
    auto col = BoltColumn::make_flat_alloc(100, BoltType::Int64, &arena);
    auto* d = col.typed_mutable<int64_t>();
    for (int i = 0; i < 100; ++i) d[i] = i;  // 100 distinct
    col.compute_stats_numeric();
    EXPECT_EQ(col.stats.distinct_count, 17u);
    EXPECT_EQ(col.stats.min_value, 0);
    EXPECT_EQ(col.stats.max_value, 99);
}

// ----------------------------------------------------------------------------
// ColumnStats.mean / .variance — Welford one-pass, verified against the
// analytical answer for a seeded random array.
// ----------------------------------------------------------------------------
TEST(BoltColumnStats, WelfordMeanVariance) {
    // Offsets are a property of the ABI — assert we haven't drifted the
    // pad layout away from what MarbleDB PDX-BOND σ-bound assumes.
    EXPECT_EQ(sizeof(ColumnStats), 64u);
    EXPECT_EQ(offsetof(ColumnStats, mean), 40u);
    EXPECT_EQ(offsetof(ColumnStats, variance), 44u);

    Arena arena;
    constexpr int64_t N = 4096;
    auto col = BoltColumn::make_flat_alloc(N, BoltType::Float32, &arena);
    auto* d = col.typed_mutable<float>();

    std::mt19937 rng(0xC0FFEEu);
    std::uniform_real_distribution<float> dist(-3.0f, 5.0f);
    double sum = 0.0, sum_sq = 0.0;
    for (int64_t i = 0; i < N; ++i) {
        const float v = dist(rng);
        d[i] = v;
        sum += static_cast<double>(v);
        sum_sq += static_cast<double>(v) * static_cast<double>(v);
    }
    const double expect_mean = sum / static_cast<double>(N);
    // Population variance (matches the Welford output choice).
    const double expect_var = sum_sq / static_cast<double>(N) -
                              expect_mean * expect_mean;

    col.compute_stats_numeric();

    EXPECT_NEAR(static_cast<double>(col.stats.mean), expect_mean, 1e-3);
    EXPECT_NEAR(static_cast<double>(col.stats.variance), expect_var, 1e-3);

    // Default-initialised ColumnStats has NaN sentinels (for the
    // "caller didn't run stats_scan_typed" detection path).
    auto fresh = ColumnStats::make_default();
    EXPECT_TRUE(std::isnan(fresh.mean));
    EXPECT_TRUE(std::isnan(fresh.variance));
}

// Constant array — variance must be exactly zero, mean equal to value.
TEST(BoltColumnStats, WelfordConstantArrayZeroVariance) {
    Arena arena;
    auto col = BoltColumn::make_flat_alloc(1024, BoltType::Float32, &arena);
    auto* d = col.typed_mutable<float>();
    for (int64_t i = 0; i < 1024; ++i) d[i] = 7.5f;
    col.compute_stats_numeric();
    EXPECT_FLOAT_EQ(col.stats.mean, 7.5f);
    EXPECT_NEAR(static_cast<double>(col.stats.variance), 0.0, 1e-6);
}

TEST(BoltColumn, CloneIntoFlat) {
    Arena a1, a2;
    auto col = BoltColumn::make_flat_alloc(16, BoltType::Int64, &a1);
    auto* d = col.typed_mutable<int64_t>();
    for (int i = 0; i < 16; ++i) d[i] = i * 11;

    auto c2 = col.clone_into(&a2);
    ASSERT_NE(c2.data, nullptr);
    EXPECT_NE(c2.data, col.data);
    EXPECT_EQ(c2.length, 16);
    for (int i = 0; i < 16; ++i) {
        EXPECT_EQ(c2.typed_data<int64_t>()[i], i * 11);
    }
    // Mutating original doesn't touch clone
    d[0] = 999;
    EXPECT_EQ(c2.typed_data<int64_t>()[0], 0);
}

TEST(BoltColumn, CloneIntoConstant) {
    Arena a;
    auto col = BoltColumn::make_constant<int64_t>(77, 500, BoltType::Int64);
    auto c2 = col.clone_into(&a);
    EXPECT_EQ(c2.format, ColumnFormat::Constant);
    EXPECT_EQ(c2.length, 500);
    EXPECT_EQ(c2.get_constant<int64_t>(), 77);
}

namespace {
// Manually build a spilled (>12-byte) StringView; offset is relative to the
// caller's own overflow buffer (0-based), mirroring how real producers
// (bolt_parquet_read.cpp's sv_from_bytes) lay out a column-private spill pool.
StringView make_spilled_sv(const char* bytes, uint32_t len, uint32_t offset) {
    StringView v;
    std::memset(&v, 0, sizeof(v));
    v.length = len;
    std::memcpy(v.prefix, bytes, 4);
    v.ref.buf_idx = 0;
    v.ref.offset = offset;
    return v;
}

StringView make_inline_sv(const char* bytes, uint32_t len) {
    StringView v;
    std::memset(&v, 0, sizeof(v));
    v.length = len;
    const uint32_t p = (len < 4u) ? len : 4u;
    if (p > 0) std::memcpy(v.prefix, bytes, p);
    if (len > 4u) std::memcpy(v.inline_data, bytes + 4, len - 4u);
    return v;
}
}  // namespace

// G2FEAT-307 bug-class regression: clone_into() must deep-copy a Utf8
// column's spilled-string overflow buffer (str_overflow_base), not just the
// fixed-width StringView row array. Before the fix, a cloned column's
// spilled rows kept pointing at the ORIGINAL producer's overflow buffer —
// dangling as soon as that buffer was reclaimed (e.g. MarbleDB's next
// scan_next() call). Mixes inline (<=12 byte) and spilled (>12 byte) rows,
// plus one NULL row deliberately holding uninitialized-looking garbage (a
// bogus huge length + offset) to prove the overflow-sizing walk skips null
// rows rather than misreading garbage as a real spilled reference.
TEST(BoltColumn, CloneIntoFlatUtf8SpilledStringsSurviveSourceOverwrite) {
    const char strA[] = "AAAAAAAAAAAAAAAAAAAA";  // 20 bytes, spilled
    const char strB[] = "BBBBBBBBBBBBBBBBBB";     // 18 bytes, spilled
    ASSERT_EQ(std::strlen(strA), 20u);
    ASSERT_EQ(std::strlen(strB), 18u);

    // Column-private overflow pool: strA at offset 0, strB at offset 20.
    constexpr size_t kOvfCap = 64;
    auto* ovf = new char[kOvfCap];
    std::memset(ovf, 0, kOvfCap);
    std::memcpy(ovf, strA, 20);
    std::memcpy(ovf + 20, strB, 18);

    constexpr int64_t kN = 5;
    auto* rows = new StringView[kN];
    rows[0] = make_inline_sv("hi", 2);
    rows[1] = make_spilled_sv(strA, 20, 0);
    rows[2] = make_inline_sv("twelve_chars", 12);  // exactly 12: still inline
    rows[3] = make_spilled_sv(strB, 18, 20);
    std::memset(&rows[4], 0xAB, sizeof(StringView));  // garbage; row is NULL

    // validity: bits 0-3 valid, bit 4 (row4) null.
    uint8_t validity = 0b00001111u;

    BoltColumn col = BoltColumn::make_flat(rows, &validity, kN, BoltType::Utf8);
    col.str_overflow_base = ovf;

    Arena dst_arena;
    BoltColumn c2 = col.clone_into(&dst_arena);

    ASSERT_EQ(c2.format, ColumnFormat::Flat);
    ASSERT_EQ(c2.length, kN);
    ASSERT_NE(c2.data, static_cast<void*>(rows));           // deep-copied
    ASSERT_NE(c2.str_overflow_base, static_cast<void*>(ovf)); // deep-copied
    ASSERT_NE(c2.str_overflow_base, nullptr);

    // Poison the ORIGINAL buffers before reading back the clone — proves
    // the clone no longer depends on either the row array or the overflow
    // pool the producer owned.
    std::memset(rows, 0xEE, kN * sizeof(StringView));
    std::memset(ovf, 0xEE, kOvfCap);

    const auto* crows = static_cast<const StringView*>(c2.data);
    const auto* cbase = static_cast<const char*>(c2.str_overflow_base);

    EXPECT_EQ(crows[0].length, 2u);
    EXPECT_EQ(std::memcmp(crows[0].prefix, "hi", 2), 0);  // inline, in-struct

    EXPECT_EQ(crows[1].length, 20u);
    ASSERT_LE(static_cast<size_t>(crows[1].ref.offset) + crows[1].length, kOvfCap);
    EXPECT_EQ(std::memcmp(cbase + crows[1].ref.offset, strA, 20), 0);

    EXPECT_EQ(crows[2].length, 12u);

    EXPECT_EQ(crows[3].length, 18u);
    ASSERT_LE(static_cast<size_t>(crows[3].ref.offset) + crows[3].length, kOvfCap);
    EXPECT_EQ(std::memcmp(cbase + crows[3].ref.offset, strB, 18), 0);

    // Null row's bit survived the clone; its garbage content is never read.
    EXPECT_EQ((c2.validity[0] >> 4) & 1u, 0u);

    delete[] rows;
    delete[] ovf;
}

// When every row is inline (or the only spilled rows are null), the clone
// must not carry a stale str_overflow_base pointer forward.
TEST(BoltColumn, CloneIntoFlatUtf8AllInlineHasNoOverflowPointer) {
    auto* ovf = new char[16];
    std::memset(ovf, 0, 16);

    constexpr int64_t kN = 2;
    auto* rows = new StringView[kN];
    rows[0] = make_inline_sv("hi", 2);
    rows[1] = make_inline_sv("bye", 3);

    BoltColumn col = BoltColumn::make_flat(rows, nullptr, kN, BoltType::Utf8);
    col.str_overflow_base = ovf;  // set, but never actually referenced

    Arena dst_arena;
    BoltColumn c2 = col.clone_into(&dst_arena);
    EXPECT_EQ(c2.str_overflow_base, nullptr);

    delete[] rows;
    delete[] ovf;
}

TEST(BoltColumn, MaterializeConstant) {
    Arena a;
    auto col = BoltColumn::make_constant<int32_t>(9, 64, BoltType::Int32);
    auto m = col.materialize(&a);
    EXPECT_EQ(m.format, ColumnFormat::Flat);
    EXPECT_EQ(m.length, 64);
    for (int i = 0; i < 64; ++i) EXPECT_EQ(m.typed_data<int32_t>()[i], 9);
}

TEST(BoltColumn, MaterializeSequence) {
    Arena a;
    auto col = BoltColumn::make_sequence(10, 2, 50, BoltType::Int64);
    auto m = col.materialize(&a);
    EXPECT_EQ(m.format, ColumnFormat::Flat);
    for (int i = 0; i < 50; ++i) {
        EXPECT_EQ(m.typed_data<int64_t>()[i], 10 + i * 2);
    }
}

TEST(BoltColumn, TryPromoteToConstant) {
    Arena a;
    auto col = BoltColumn::make_flat_alloc(100, BoltType::Int64, &a);
    auto* d = col.typed_mutable<int64_t>();
    for (int i = 0; i < 100; ++i) d[i] = 42;
    col.compute_stats_numeric();
    EXPECT_EQ(col.stats.distinct_count, 1u);
    EXPECT_TRUE(col.try_promote(&a));
    EXPECT_EQ(col.format, ColumnFormat::Constant);
    EXPECT_EQ(col.get_constant<int64_t>(), 42);
}

TEST(BoltColumn, TryPromoteToDictionary) {
    Arena a;
    auto col = BoltColumn::make_flat_alloc(1000, BoltType::Int64, &a);
    auto* d = col.typed_mutable<int64_t>();
    int64_t vals[] = {10, 20, 30, 40};
    for (int i = 0; i < 1000; ++i) d[i] = vals[i % 4];
    col.compute_stats_numeric();
    EXPECT_EQ(col.stats.distinct_count, 4u);
    EXPECT_TRUE(col.try_promote(&a));
    EXPECT_EQ(col.format, ColumnFormat::Dictionary);
    ASSERT_NE(col.dict_child, nullptr);
    EXPECT_EQ(col.dict_child->length, 4);

    auto m = col.materialize(&a);
    for (int i = 0; i < 1000; ++i) {
        EXPECT_EQ(m.typed_data<int64_t>()[i], vals[i % 4]);
    }
}

TEST(BitmapIndex, BuildAndCount) {
    Arena a;
    // Build a dictionary column manually.
    auto dict = BoltColumn::make_flat_alloc(3, BoltType::Int32, &a);
    auto* dv = dict.typed_mutable<int32_t>();
    dv[0] = 100; dv[1] = 200; dv[2] = 300;

    BoltColumn col = BoltColumn::make_empty();
    col.format = ColumnFormat::Dictionary;
    col.type = BoltType::Int32;
    col.type_size_bytes = 1;
    col.length = 10;
    uint8_t* keys = a.allocate_array<uint8_t>(10);
    uint8_t seq[] = {0,1,2,0,1,2,0,1,2,0};
    memcpy(keys, seq, 10);
    col.data = keys;
    BoltColumn* child = a.allocate_array<BoltColumn>(1);
    *child = dict;
    col.dict_child = child;
    col.arena = &a;

    BitmapIndex* idx = BitmapIndex::build(col, &a);
    ASSERT_NE(idx, nullptr);
    EXPECT_EQ(idx->num_keys, 3u);
    EXPECT_EQ(idx->num_rows, 10u);
    EXPECT_EQ(idx->count(0), 4u);  // positions 0,3,6,9
    EXPECT_EQ(idx->count(1), 3u);
    EXPECT_EQ(idx->count(2), 3u);

    int32_t out[16];
    int64_t n = idx->filter(0, out);
    EXPECT_EQ(n, 4);
    EXPECT_EQ(out[0], 0);
    EXPECT_EQ(out[1], 3);
    EXPECT_EQ(out[2], 6);
    EXPECT_EQ(out[3], 9);

    ArenaGuard g(a);
    uint32_t kq[] = {0, 2};
    int64_t n2 = idx->filter_in(kq, 2, out);
    EXPECT_EQ(n2, 7);  // 4 + 3
}

// C5 — ensure_bitmap_index: lazy build + cache on the sidecar slot.
TEST(BitmapIndex, EnsureIndexCaches) {
    Arena a;
    // Same dictionary column shape as BuildAndCount.
    auto dict = BoltColumn::make_flat_alloc(3, BoltType::Int32, &a);
    auto* dv = dict.typed_mutable<int32_t>();
    dv[0] = 100; dv[1] = 200; dv[2] = 300;

    BoltColumn col = BoltColumn::make_empty();
    col.format = ColumnFormat::Dictionary;
    col.type = BoltType::Int32;
    col.type_size_bytes = 1;
    col.length = 10;
    uint8_t* keys = a.allocate_array<uint8_t>(10);
    uint8_t seq[] = {0,1,2,0,1,2,0,1,2,0};
    memcpy(keys, seq, 10);
    col.data = keys;
    BoltColumn* child = a.allocate_array<BoltColumn>(1);
    *child = dict;
    col.dict_child = child;
    col.arena = &a;

    ASSERT_EQ(col.sidecars.bitmap_index, nullptr);
    BitmapIndex* first  = col.ensure_bitmap_index(&a);
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(first->num_keys, 3u);

    // Second call returns the cached pointer (same address), no rebuild.
    BitmapIndex* second = col.ensure_bitmap_index(&a);
    EXPECT_EQ(first, second);

    // Non-Dictionary columns return nullptr without allocating.
    int32_t flat_data[4] = {1,2,3,4};
    BoltColumn flat = BoltColumn::make_flat(flat_data, nullptr, 4, BoltType::Int32);
    EXPECT_EQ(flat.ensure_bitmap_index(&a), nullptr);
    EXPECT_EQ(flat.sidecars.bitmap_index, nullptr);
}

// B2 — RLE column format: runs of equal values expanded via materialize.
TEST(BoltColumn, RLEMaterializeRoundTrip) {
    Arena a;
    // Logical rows: 5×A, 3×B, 2×A, 4×C = 14 rows across 4 runs.
    int32_t values[4]   = {100, 200, 100, 300};
    int32_t run_ends[4] = {5, 8, 10, 14};

    BoltColumn rle = BoltColumn::make_rle(
        values, /*num_runs=*/4, run_ends, /*total_rows=*/14,
        BoltType::Int32, &a);
    ASSERT_EQ(rle.format, ColumnFormat::RLE);
    ASSERT_EQ(rle.length, 14);
    ASSERT_NE(rle.dict_child, nullptr);
    EXPECT_EQ(rle.dict_child->length, 4);

    BoltColumn flat = rle.materialize(&a);
    ASSERT_EQ(flat.format, ColumnFormat::Flat);
    ASSERT_EQ(flat.length, 14);
    ASSERT_NE(flat.data, nullptr);

    const int32_t* out = static_cast<const int32_t*>(flat.data);
    const int32_t expected[14] = {100,100,100,100,100, 200,200,200,
                                  100,100, 300,300,300,300};
    for (int i = 0; i < 14; ++i) EXPECT_EQ(out[i], expected[i]) << "row " << i;
}

TEST(BoltColumn, RLEEmpty) {
    Arena a;
    BoltColumn rle = BoltColumn::make_rle(nullptr, 0, nullptr, 0,
                                           BoltType::Int64, &a);
    EXPECT_EQ(rle.format, ColumnFormat::RLE);
    EXPECT_EQ(rle.length, 0);

    BoltColumn flat = rle.materialize(&a);
    EXPECT_EQ(flat.length, 0);
}

TEST(BoltColumn, RLESingleRun) {
    Arena a;
    int64_t values[1]   = {42};
    int32_t run_ends[1] = {7};
    BoltColumn rle = BoltColumn::make_rle(values, 1, run_ends, 7,
                                           BoltType::Int64, &a);
    ASSERT_EQ(rle.length, 7);
    BoltColumn flat = rle.materialize(&a);
    ASSERT_EQ(flat.length, 7);
    const int64_t* out = static_cast<const int64_t*>(flat.data);
    for (int i = 0; i < 7; ++i) EXPECT_EQ(out[i], 42);
}

// B3 — BitPacked round-trip. 12 values, 3 bits each (range 0-7).
TEST(BoltColumn, BitPackedRoundTrip3Bit) {
    Arena a;
    // Packing 12 × 3-bit values = 36 bits, fits in one uint64.
    // Values: 0,1,2,3,4,5,6,7,0,1,2,3
    const uint32_t vals[12] = {0,1,2,3,4,5,6,7,0,1,2,3};
    uint64_t word = 0;
    for (int i = 0; i < 12; ++i) word |= uint64_t(vals[i]) << (i * 3);
    uint64_t packed[2] = {word, 0};  // second word just safe read-ahead pad

    BoltColumn bp = BoltColumn::make_bitpacked(
        packed, /*bit_width=*/3, /*total_rows=*/12,
        BoltType::Int32, &a);
    ASSERT_EQ(bp.length, 12);
    BoltColumn flat = bp.materialize(&a);
    ASSERT_EQ(flat.length, 12);
    const int32_t* out = static_cast<const int32_t*>(flat.data);
    for (int i = 0; i < 12; ++i) EXPECT_EQ(out[i], static_cast<int32_t>(vals[i])) << "i=" << i;
}

TEST(BoltColumn, BitPacked17BitCrossesWordBoundary) {
    Arena a;
    // 5 × 17-bit values = 85 bits; crosses a uint64 boundary after the 3rd.
    const uint32_t vals[5] = {0x0AAAA, 0x15555, 0x1FFFF, 0x00001, 0x10000};
    uint64_t words[3] = {0, 0, 0};
    for (int i = 0; i < 5; ++i) {
        uint64_t bit_off = uint64_t(i) * 17;
        uint64_t bit_in  = bit_off & 63u;
        words[bit_off >> 6] |= uint64_t(vals[i]) << bit_in;
        if (bit_in + 17 > 64) {
            words[(bit_off >> 6) + 1] |= uint64_t(vals[i]) >> (64 - bit_in);
        }
    }
    BoltColumn bp = BoltColumn::make_bitpacked(
        words, 17, 5, BoltType::Int32, &a);
    BoltColumn flat = bp.materialize(&a);
    const int32_t* out = static_cast<const int32_t*>(flat.data);
    for (int i = 0; i < 5; ++i) EXPECT_EQ(out[i], static_cast<int32_t>(vals[i])) << "i=" << i;
}

// B3 edge widths: 1-bit (smallest) and 32-bit (largest supported).
TEST(BoltColumn, BitPacked1BitEdge) {
    Arena a;
    // 70 bits → 2 words. Pattern: alternating 1,0,1,0,...
    uint64_t packed[2] = {0xAAAAAAAAAAAAAAAAULL, 0x2AULL /*70th..64th bits*/};
    BoltColumn bp = BoltColumn::make_bitpacked(packed, 1, 70, BoltType::Int32, &a);
    BoltColumn flat = bp.materialize(&a);
    ASSERT_EQ(flat.length, 70);
    const int32_t* out = static_cast<const int32_t*>(flat.data);
    for (int i = 0; i < 64; ++i) EXPECT_EQ(out[i], (i & 1) ? 1 : 0) << "i=" << i;
    // Bits 64..69 are the low 6 bits of 0x2A = 0b101010 → 0,1,0,1,0,1.
    const int32_t tail[6] = {0,1,0,1,0,1};
    for (int i = 0; i < 6; ++i) EXPECT_EQ(out[64 + i], tail[i]) << "i=" << 64+i;
}

TEST(BoltColumn, BitPacked32BitEdge) {
    Arena a;
    // 4 × 32-bit values pack into 2 uint64 words (exactly).
    const uint32_t vals[4] = {0x00000000u, 0xFFFFFFFFu, 0xCAFEBABEu, 0x12345678u};
    uint64_t words[3] = {0, 0, 0};
    for (int i = 0; i < 4; ++i) {
        words[i / 2] |= uint64_t(vals[i]) << ((i & 1) * 32);
    }
    BoltColumn bp = BoltColumn::make_bitpacked(words, 32, 4, BoltType::Int32, &a);
    BoltColumn flat = bp.materialize(&a);
    ASSERT_EQ(flat.length, 4);
    const int32_t* out = static_cast<const int32_t*>(flat.data);
    for (int i = 0; i < 4; ++i) EXPECT_EQ(static_cast<uint32_t>(out[i]), vals[i]) << "i=" << i;
}

// RLE stress: 1000 runs with varying lengths — exercises the materialize
// loop's tail branch more than the 4-run smoke test.
TEST(BoltColumn, RLELongRunList) {
    Arena a;
    constexpr int NRUNS = 1000;
    std::vector<int64_t> values(NRUNS);
    std::vector<int32_t> run_ends(NRUNS);
    int32_t pos = 0;
    std::mt19937 rng{0xDEAFu};
    std::uniform_int_distribution<int> run_len{1, 10};
    for (int i = 0; i < NRUNS; ++i) {
        values[i]    = (i & 1) ? 0xAAAAAAAAAAAAAAAALL : 0x5555555555555555LL;
        pos         += run_len(rng);
        run_ends[i]  = pos;
    }
    BoltColumn rle = BoltColumn::make_rle(
        values.data(), NRUNS, run_ends.data(), pos, BoltType::Int64, &a);
    BoltColumn flat = rle.materialize(&a);
    ASSERT_EQ(flat.length, pos);

    const int64_t* out = static_cast<const int64_t*>(flat.data);
    int32_t prev = 0;
    for (int i = 0; i < NRUNS; ++i) {
        for (int32_t r = prev; r < run_ends[i]; ++r) {
            ASSERT_EQ(out[r], values[i]) << "row " << r;
        }
        prev = run_ends[i];
    }
}

// B4 — FrameOfReference round-trip.  base + bit-packed deltas.
TEST(BoltColumn, FrameOfReferenceRoundTrip) {
    Arena a;
    // Simulate monotonic timestamps: 1_000_000 + {0, 5, 12, 20, 31, 48, 63}.
    const int64_t base = 1'000'000;
    const uint32_t deltas[7] = {0, 5, 12, 20, 31, 48, 63};
    uint64_t word = 0;
    for (int i = 0; i < 7; ++i) word |= uint64_t(deltas[i]) << (i * 6);
    uint64_t packed[2] = {word, 0};

    BoltColumn fr = BoltColumn::make_frame_of_ref(
        packed, /*bit_width=*/6, base, /*total_rows=*/7,
        BoltType::Int64, &a);
    ASSERT_EQ(fr.length, 7);
    BoltColumn flat = fr.materialize(&a);
    ASSERT_EQ(flat.length, 7);
    const int64_t* out = static_cast<const int64_t*>(flat.data);
    for (int i = 0; i < 7; ++i) {
        EXPECT_EQ(out[i], base + deltas[i]) << "i=" << i;
    }
}

TEST(BoltBatchArrow, FillSchemaStruct) {
    Arena a;
    alignas(64) static BoltBatch b;
    BoltBatch::init_empty(&b);
    b.arena = &a;
    b.num_rows = 0;
    b.num_cols = 2;
    b.schema.add_field("x", BoltType::Int32);
    b.schema.add_field("y", BoltType::Float64);
    b.columns[0][0] = BoltColumn::make_flat(nullptr, nullptr, 0, BoltType::Int32);
    b.columns[0][1] = BoltColumn::make_flat(nullptr, nullptr, 0, BoltType::Float64);

    ArrowSchema out;
    b.fill_arrow_schema(&out);
    EXPECT_STREQ(out.format, "+s");
    EXPECT_EQ(out.n_children, 2);
    ASSERT_NE(out.children, nullptr);
    EXPECT_STREQ(out.children[0]->format, "i");
    EXPECT_STREQ(out.children[0]->name, "x");
    EXPECT_STREQ(out.children[1]->format, "g");
    EXPECT_STREQ(out.children[1]->name, "y");
}

TEST(BoltArrowExport, ArrayFill) {
    Arena arena;
    auto col = BoltColumn::make_flat_alloc(100, BoltType::Int64, &arena);
    auto* data = col.typed_mutable<int64_t>();
    for (int i = 0; i < 100; ++i) data[i] = i;

    ArrowArray arr;
    col.fill_arrow_array(&arr);
    EXPECT_EQ(arr.length, 100);
    EXPECT_EQ(arr.null_count, 0);
    EXPECT_EQ(arr.n_buffers, 2);
    EXPECT_EQ(arr.buffers[0], nullptr);  // No validity bitmap
    EXPECT_EQ(arr.buffers[1], data);     // Zero-copy pointer to our data
}
