// test_bolt_arrow.cpp — Hand-rolled Arrow C Data Interface consumer.
//
// No libarrow. No pyarrow. These tests verify that our ArrowSchema/ArrowArray
// exports conform to the C Data Interface spec closely enough that an external
// consumer (who only sees the structs defined in bolt_types.h) can reconstruct
// the original data.
//
// Spec: https://arrow.apache.org/docs/format/CDataInterface.html

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"

using namespace bolt;

namespace {

// Build a Flat column of `length` rows with values produced by `gen(i)`.
template <typename T>
BoltColumn build_flat(Arena* arena, int64_t length, BoltType type,
                      T (*gen)(int64_t)) {
    BoltColumn c = BoltColumn::make_flat_alloc(length, type, arena);
    T* p = static_cast<T*>(c.data);
    for (int64_t i = 0; i < length; ++i) p[i] = gen(i);
    return c;
}

}  // namespace

// ----------------------------------------------------------------------------
// Schema export
// ----------------------------------------------------------------------------

TEST(BoltArrow, SchemaInt32) {
    Arena arena;
    BoltColumn c = build_flat<int32_t>(
        &arena, 100, BoltType::Int32,
        [](int64_t i) -> int32_t { return static_cast<int32_t>(i); });
    ArrowSchema s;
    c.fill_arrow_schema(&s, "c0");
    EXPECT_STREQ(s.format, "i");
    EXPECT_EQ(s.n_children, 0);
    EXPECT_NE(s.release, nullptr);
    s.release(&s);
}

TEST(BoltArrow, SchemaInt64) {
    Arena arena;
    BoltColumn c = build_flat<int64_t>(
        &arena, 100, BoltType::Int64,
        [](int64_t i) -> int64_t { return i; });
    ArrowSchema s;
    c.fill_arrow_schema(&s, "c0");
    EXPECT_STREQ(s.format, "l");
    EXPECT_EQ(s.n_children, 0);
    EXPECT_NE(s.release, nullptr);
    s.release(&s);
}

TEST(BoltArrow, SchemaFloat64) {
    Arena arena;
    BoltColumn c = build_flat<double>(
        &arena, 100, BoltType::Float64,
        [](int64_t i) -> double { return static_cast<double>(i); });
    ArrowSchema s;
    c.fill_arrow_schema(&s, "c0");
    EXPECT_STREQ(s.format, "g");
    EXPECT_EQ(s.n_children, 0);
    EXPECT_NE(s.release, nullptr);
    s.release(&s);
}

// ----------------------------------------------------------------------------
// Array export round-trip
// ----------------------------------------------------------------------------

TEST(BoltArrow, ArrayInt32RoundTrip) {
    Arena arena;
    BoltColumn c = build_flat<int32_t>(
        &arena, 100, BoltType::Int32,
        [](int64_t i) -> int32_t { return static_cast<int32_t>(i * 7); });
    ArrowArray a;
    c.fill_arrow_array(&a);
    EXPECT_EQ(a.length, 100);
    EXPECT_EQ(a.null_count, 0);
    EXPECT_EQ(a.offset, 0);
    EXPECT_EQ(a.n_buffers, 2);
    EXPECT_EQ(a.n_children, 0);

    const int32_t* vals = static_cast<const int32_t*>(a.buffers[1]);
    ASSERT_NE(vals, nullptr);
    for (int64_t i = 0; i < 100; ++i) {
        EXPECT_EQ(vals[i], static_cast<int32_t>(i * 7));
    }
    a.release(&a);
}

TEST(BoltArrow, ArrayInt64RoundTrip) {
    Arena arena;
    BoltColumn c = build_flat<int64_t>(
        &arena, 100, BoltType::Int64,
        [](int64_t i) -> int64_t { return i * 7; });
    ArrowArray a;
    c.fill_arrow_array(&a);
    EXPECT_EQ(a.length, 100);
    EXPECT_EQ(a.null_count, 0);
    EXPECT_EQ(a.offset, 0);
    EXPECT_EQ(a.n_buffers, 2);

    const int64_t* vals = static_cast<const int64_t*>(a.buffers[1]);
    ASSERT_NE(vals, nullptr);
    for (int64_t i = 0; i < 100; ++i) {
        EXPECT_EQ(vals[i], i * 7);
    }
    a.release(&a);
}

TEST(BoltArrow, ArrayWithNullsValidityBitmap) {
    // Arrow validity bitmap: little-endian, bit i of byte (i>>3) is row i.
    // Bit = 1 means valid, 0 means null. Every other row is null → 50 nulls.
    Arena arena;
    constexpr int64_t N = 100;
    BoltColumn c = BoltColumn::make_flat_alloc(N, BoltType::Int32, &arena);
    ASSERT_NE(c.data, nullptr);
    int32_t* data = static_cast<int32_t*>(c.data);
    for (int64_t i = 0; i < N; ++i) data[i] = static_cast<int32_t>(i);

    // Validity bitmap: (N + 7) / 8 bytes = 13
    const size_t vbytes = (N + 7) / 8;
    uint8_t* validity = static_cast<uint8_t*>(arena.allocate(vbytes));
    ASSERT_NE(validity, nullptr);
    std::memset(validity, 0, vbytes);
    for (int64_t i = 0; i < N; ++i) {
        if ((i & 1) == 0) {  // even rows valid
            validity[i >> 3] |= static_cast<uint8_t>(1u << (i & 7));
        }
    }
    c.validity = validity;
    c.stats.all_valid = false;
    c.stats.null_count = 50;

    ArrowArray a;
    c.fill_arrow_array(&a);
    EXPECT_EQ(a.length, N);
    EXPECT_EQ(a.null_count, 50);
    EXPECT_EQ(a.n_buffers, 2);

    const uint8_t* vout = static_cast<const uint8_t*>(a.buffers[0]);
    ASSERT_NE(vout, nullptr);
    for (size_t b = 0; b < vbytes; ++b) EXPECT_EQ(vout[b], validity[b]);
    // Spot-check bit semantics: row 0 valid, row 1 null.
    EXPECT_EQ(vout[0] & 0x1u, 0x1u);
    EXPECT_EQ(vout[0] & 0x2u, 0x0u);
    a.release(&a);
}

// ----------------------------------------------------------------------------
// Batch schema as struct
// ----------------------------------------------------------------------------

namespace {

void build_mixed_batch(BoltBatch* b, Arena* arena) {
    BoltBatch::init_empty(b);
    b->num_rows = 10;
    b->num_cols = 3;
    b->arena = arena;
    BoltBatch::alloc_columns(b, arena, 3);  // G2FEAT-47: size columns[2]
    b->schema.num_fields = 3;

    const BoltType types[3] = {BoltType::Int32, BoltType::Int64, BoltType::Float64};
    const char* names[3] = {"a", "b", "c"};

    for (uint32_t c = 0; c < 3; ++c) {
        BoltField& f = b->schema.fields[c];
        std::memset(&f, 0, sizeof(f));
        f.set_name(names[c]);
        f.type = types[c];
        f.nullable = false;

        BoltColumn col = BoltColumn::make_flat_alloc(10, types[c], arena);
        if (types[c] == BoltType::Int32) {
            int32_t* p = static_cast<int32_t*>(col.data);
            for (int64_t i = 0; i < 10; ++i) p[i] = static_cast<int32_t>(i);
        } else if (types[c] == BoltType::Int64) {
            int64_t* p = static_cast<int64_t*>(col.data);
            for (int64_t i = 0; i < 10; ++i) p[i] = i * 100;
        } else {
            double* p = static_cast<double*>(col.data);
            for (int64_t i = 0; i < 10; ++i) p[i] = static_cast<double>(i) * 0.5;
        }
        b->columns[0][c] = col;
        b->columns[1][c] = col;
    }
}

}  // namespace

TEST(BoltArrow, BatchSchemaStruct) {
    Arena arena;
    BoltBatch batch;
    build_mixed_batch(&batch, &arena);

    ArrowSchema s;
    batch.fill_arrow_schema(&s);
    EXPECT_STREQ(s.format, "+s");
    EXPECT_EQ(s.n_children, 3);
    ASSERT_NE(s.children, nullptr);

    EXPECT_STREQ(s.children[0]->format, "i");
    EXPECT_STREQ(s.children[1]->format, "l");
    EXPECT_STREQ(s.children[2]->format, "g");
    EXPECT_STREQ(s.children[0]->name, "a");
    EXPECT_STREQ(s.children[1]->name, "b");
    EXPECT_STREQ(s.children[2]->name, "c");
    s.release(&s);
}

// ----------------------------------------------------------------------------
// Release idempotency
// ----------------------------------------------------------------------------

TEST(BoltArrow, ReleaseIsCallableTwice) {
    Arena arena;
    BoltColumn c = build_flat<int32_t>(
        &arena, 8, BoltType::Int32,
        [](int64_t i) -> int32_t { return static_cast<int32_t>(i); });
    ArrowSchema s;
    c.fill_arrow_schema(&s, "c0");
    ASSERT_NE(s.release, nullptr);
    s.release(&s);
    s.release(&s);  // must not crash

    ArrowArray a;
    c.fill_arrow_array(&a);
    ASSERT_NE(a.release, nullptr);
    a.release(&a);
    a.release(&a);  // must not crash
    SUCCEED();
}

// ----------------------------------------------------------------------------
// BatchToBatch — external-consumer reconstruction via C Data Interface only
// ----------------------------------------------------------------------------

namespace {

// Map an Arrow format string back to a BoltType (the subset we export here).
BoltType type_from_format(const char* fmt) noexcept {
    if (!fmt) return BoltType::NA;
    if (std::strcmp(fmt, "i") == 0) return BoltType::Int32;
    if (std::strcmp(fmt, "l") == 0) return BoltType::Int64;
    if (std::strcmp(fmt, "g") == 0) return BoltType::Float64;
    if (std::strcmp(fmt, "f") == 0) return BoltType::Float32;
    return BoltType::NA;
}

// Consumer: take one column's ArrowSchema + ArrowArray and rebuild a
// BoltColumn in a fresh arena by copying the data buffer out of a.buffers[1].
BoltColumn consume_column(const ArrowSchema* s, const ArrowArray* a,
                          Arena* dst_arena) noexcept {
    BoltType t = type_from_format(s->format);
    BoltColumn c = BoltColumn::make_flat_alloc(a->length, t, dst_arena);
    size_t tsz = type_size(t);
    const void* src = a->buffers[1];
    std::memcpy(c.data, src, static_cast<size_t>(a->length) * tsz);
    return c;
}

}  // namespace

TEST(BoltArrow, BatchToBatch) {
    Arena src_arena, dst_arena;
    BoltBatch src;
    build_mixed_batch(&src, &src_arena);

    // Producer side: export schema + per-column arrays.
    ArrowSchema schema;
    src.fill_arrow_schema(&schema);
    ASSERT_EQ(schema.n_children, 3);

    // Note: fill_arrow_array uses a thread_local buffers array, so each
    // ArrowArray's buffers[] is only valid until the next call. Consume
    // immediately after filling, one column at a time.
    BoltBatch dst;
    BoltBatch::init_empty(&dst);
    dst.num_rows = src.num_rows;
    dst.num_cols = static_cast<uint32_t>(schema.n_children);
    dst.arena = &dst_arena;
    BoltBatch::alloc_columns(&dst, &dst_arena,
                             static_cast<uint32_t>(schema.n_children));  // G2FEAT-47
    dst.schema.num_fields = dst.num_cols;

    for (uint32_t i = 0; i < dst.num_cols; ++i) {
        BoltField& f = dst.schema.fields[i];
        std::memset(&f, 0, sizeof(f));
        f.set_name(schema.children[i]->name);
        f.type = type_from_format(schema.children[i]->format);

        ArrowArray a;
        src.col(i).fill_arrow_array(&a);
        BoltColumn rebuilt = consume_column(schema.children[i], &a, &dst_arena);
        a.release(&a);
        dst.columns[0][i] = rebuilt;
        dst.columns[1][i] = rebuilt;
    }

    // Compare column-for-column.
    ASSERT_EQ(dst.num_cols, src.num_cols);
    ASSERT_EQ(dst.num_rows, src.num_rows);
    for (uint32_t i = 0; i < src.num_cols; ++i) {
        const BoltColumn& sc = src.col(i);
        const BoltColumn& dc = dst.col(i);
        ASSERT_EQ(sc.type, dc.type);
        ASSERT_EQ(sc.length, dc.length);
        size_t tsz = type_size(sc.type);
        EXPECT_EQ(std::memcmp(sc.data, dc.data,
                              static_cast<size_t>(sc.length) * tsz), 0);
    }

    schema.release(&schema);
}
