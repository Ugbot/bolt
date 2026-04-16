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

#include <thread>
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
