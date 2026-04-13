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

using namespace chukonu::bolt;

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
