// test_bolt_wire_stream.cpp — Coverage for the streaming wire-format writer.
//
// Strategy:
//   1. Build a reference BoltBatch the conventional way.
//   2. Serialise it via bolt::wire::bolt_wire_serialize (the existing path).
//   3. Stream the same data via wire_stream_* and compare byte-identical
//      output.
//   4. Deserialise the streamed bytes via bolt_wire_deserialize and confirm
//      the round-trip recovers the original column values.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"
#include "bolt/wire/bolt_wire.h"
#include "bolt/wire/bolt_wire_stream.h"

using namespace bolt;

namespace {

// Populate a BoltBatch with `ncols` Flat int32 columns of `nrows` rows each.
// Values: col[c][r] = c * 1000 + r.  No validity bitmap (all valid).
void build_int32_batch(BoltBatch* b, Arena* arena,
                       uint32_t ncols, int64_t nrows) {
    BoltBatch::init_empty(b);
    b->num_rows = nrows;
    b->num_cols = ncols;
    b->arena    = arena;
    b->schema.num_fields = ncols;

    for (uint32_t c = 0; c < ncols; ++c) {
        char name[8];
        std::snprintf(name, sizeof(name), "c%u", c);
        BoltField& f = b->schema.fields[c];
        std::memset(&f, 0, sizeof(f));
        f.set_name(name);
        f.type     = BoltType::Int32;
        f.nullable = false;

        BoltColumn col = BoltColumn::make_flat_alloc(nrows, BoltType::Int32, arena);
        ASSERT_NE(col.data, nullptr);
        int32_t* p = static_cast<int32_t*>(col.data);
        for (int64_t r = 0; r < nrows; ++r) {
            p[r] = static_cast<int32_t>(c * 1000 + r);
        }
        b->columns[0][c] = col;
        b->columns[1][c] = col;
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Byte-identical vs the non-streaming serializer.
// ---------------------------------------------------------------------------
TEST(BoltWireStream, ByteIdenticalVsBoltWireSerialize) {
    constexpr uint32_t ncols = 3;
    constexpr int64_t  nrows = 100;

    Arena arena_src;
    BoltBatch src;
    build_int32_batch(&src, &arena_src, ncols, nrows);

    // Reference: full-batch serialize.
    const size_t need = wire::bolt_wire_size(&src);
    ASSERT_GT(need, 0u);
    std::vector<uint8_t> ref_buf(need, 0);
    const size_t written_ref = wire::bolt_wire_serialize(
        &src, ref_buf.data(), ref_buf.size());
    ASSERT_EQ(written_ref, need);

    // Streaming path.
    std::vector<uint8_t> str_buf(need, 0);
    WireStream s;
    ASSERT_TRUE(wire_stream_begin_file(&s, str_buf.data(), str_buf.size(),
                                       src.schema.fields, ncols));
    for (uint32_t c = 0; c < ncols; ++c) {
        const BoltColumn& col = src.col(c);
        const size_t raw_len = static_cast<size_t>(col.length) * sizeof(int32_t);
        ASSERT_TRUE(wire_stream_append_column(&s, &col, col.data, raw_len))
            << "append failed on column " << c;
    }
    const size_t written_str = wire_stream_finalize(&s);
    ASSERT_EQ(written_str, need);

    // Byte-identical output.
    ASSERT_EQ(ref_buf.size(), str_buf.size());
    for (size_t i = 0; i < ref_buf.size(); ++i) {
        ASSERT_EQ(ref_buf[i], str_buf[i])
            << "byte mismatch at offset " << i;
    }
}

// ---------------------------------------------------------------------------
// Deserialise the streamed output via the existing reader; verify round trip.
// ---------------------------------------------------------------------------
TEST(BoltWireStream, RoundTripViaBoltWireDeserialize) {
    constexpr uint32_t ncols = 4;
    constexpr int64_t  nrows = 73;

    Arena arena_src, arena_dst;
    BoltBatch src;
    build_int32_batch(&src, &arena_src, ncols, nrows);

    const size_t need = wire::bolt_wire_size(&src);
    std::vector<uint8_t> buf(need, 0);

    WireStream s;
    ASSERT_TRUE(wire_stream_begin_file(&s, buf.data(), buf.size(),
                                       src.schema.fields, ncols));
    for (uint32_t c = 0; c < ncols; ++c) {
        const BoltColumn& col = src.col(c);
        const size_t raw_len = static_cast<size_t>(col.length) * sizeof(int32_t);
        ASSERT_TRUE(wire_stream_append_column(&s, &col, col.data, raw_len));
    }
    const size_t written = wire_stream_finalize(&s);
    ASSERT_EQ(written, need);

    BoltBatch dst;
    ASSERT_TRUE(wire::bolt_wire_deserialize(buf.data(), buf.size(),
                                            &dst, &arena_dst));
    ASSERT_EQ(dst.num_rows, src.num_rows);
    ASSERT_EQ(dst.num_cols, src.num_cols);
    for (uint32_t c = 0; c < ncols; ++c) {
        const BoltColumn& sc = src.col(c);
        const BoltColumn& dc = dst.col(c);
        ASSERT_EQ(dc.length, sc.length);
        ASSERT_EQ(dc.type,   sc.type);
        const int32_t* sp = static_cast<const int32_t*>(sc.data);
        const int32_t* dp = static_cast<const int32_t*>(dc.data);
        for (int64_t r = 0; r < sc.length; ++r) {
            ASSERT_EQ(sp[r], dp[r]) << "col " << c << " row " << r;
        }
    }
}

// ---------------------------------------------------------------------------
// Zero columns — degenerate-but-valid.
// ---------------------------------------------------------------------------
TEST(BoltWireStream, ZeroColumns) {
    std::vector<uint8_t> buf(1024, 0);
    WireStream s;
    ASSERT_TRUE(wire_stream_begin_file(&s, buf.data(), buf.size(),
                                       nullptr, 0));
    const size_t total = wire_stream_finalize(&s);
    // Header (32) aligned to 64 == 64 for an empty file (no schema, no descs).
    // bolt_wire_size on a zero-column batch returns header + 0 + 0 + align = 64.
    Arena arena_src;
    BoltBatch empty;
    BoltBatch::init_empty(&empty);
    empty.num_rows = 0;
    empty.num_cols = 0;
    empty.arena = &arena_src;
    const size_t expect = wire::bolt_wire_size(&empty);
    EXPECT_EQ(total, expect);
}

// ---------------------------------------------------------------------------
// Out-of-order append / too many columns: the stream must reject cleanly.
// ---------------------------------------------------------------------------
TEST(BoltWireStream, RejectAppendBeyondDeclaredColumns) {
    constexpr uint32_t ncols = 1;
    constexpr int64_t  nrows = 4;

    Arena arena_src;
    BoltBatch src;
    build_int32_batch(&src, &arena_src, ncols, nrows);

    std::vector<uint8_t> buf(4096, 0);
    WireStream s;
    ASSERT_TRUE(wire_stream_begin_file(&s, buf.data(), buf.size(),
                                       src.schema.fields, ncols));

    const BoltColumn& col = src.col(0);
    const size_t raw_len = static_cast<size_t>(col.length) * sizeof(int32_t);
    ASSERT_TRUE(wire_stream_append_column(&s, &col, col.data, raw_len));

    // Extra append past the declared column count must fail.
    EXPECT_FALSE(wire_stream_append_column(&s, &col, col.data, raw_len));
    EXPECT_FALSE(s.ok);
}

// ---------------------------------------------------------------------------
// Incomplete finalize (fewer columns appended than declared) returns 0.
// ---------------------------------------------------------------------------
TEST(BoltWireStream, IncompleteFinalizeReturnsZero) {
    constexpr uint32_t ncols = 2;
    std::vector<uint8_t> buf(4096, 0);

    BoltField fields[ncols];
    std::memset(fields, 0, sizeof(fields));
    fields[0].set_name("a"); fields[0].type = BoltType::Int32;
    fields[1].set_name("b"); fields[1].type = BoltType::Int64;

    WireStream s;
    ASSERT_TRUE(wire_stream_begin_file(&s, buf.data(), buf.size(),
                                       fields, ncols));
    // Finalize without any append: still ok but cols_written != num_cols.
    EXPECT_EQ(wire_stream_finalize(&s), 0u);
    EXPECT_FALSE(s.ok);
}

// ---------------------------------------------------------------------------
// int64 + multi-type column: make sure type_size dispatch is correct.
// ---------------------------------------------------------------------------
TEST(BoltWireStream, MixedInt32Int64Columns) {
    constexpr int64_t nrows = 17;

    Arena arena_src, arena_dst;
    BoltBatch src;
    BoltBatch::init_empty(&src);
    src.num_rows = nrows;
    src.num_cols = 2;
    src.arena    = &arena_src;
    src.schema.num_fields = 2;

    BoltField& f0 = src.schema.fields[0];
    std::memset(&f0, 0, sizeof(f0));
    f0.set_name("i32"); f0.type = BoltType::Int32; f0.nullable = false;
    BoltField& f1 = src.schema.fields[1];
    std::memset(&f1, 0, sizeof(f1));
    f1.set_name("i64"); f1.type = BoltType::Int64; f1.nullable = false;

    BoltColumn c0 = BoltColumn::make_flat_alloc(nrows, BoltType::Int32, &arena_src);
    BoltColumn c1 = BoltColumn::make_flat_alloc(nrows, BoltType::Int64, &arena_src);
    int32_t* p0 = static_cast<int32_t*>(c0.data);
    int64_t* p1 = static_cast<int64_t*>(c1.data);
    for (int64_t r = 0; r < nrows; ++r) {
        p0[r] = static_cast<int32_t>(r + 1);
        p1[r] = static_cast<int64_t>(r) * 1'000'000;
    }
    src.columns[0][0] = c0; src.columns[1][0] = c0;
    src.columns[0][1] = c1; src.columns[1][1] = c1;

    const size_t need = wire::bolt_wire_size(&src);
    std::vector<uint8_t> ref(need, 0), str(need, 0);
    ASSERT_EQ(wire::bolt_wire_serialize(&src, ref.data(), ref.size()), need);

    WireStream s;
    ASSERT_TRUE(wire_stream_begin_file(&s, str.data(), str.size(),
                                       src.schema.fields, 2));
    ASSERT_TRUE(wire_stream_append_column(&s, &c0, c0.data,
                                          static_cast<size_t>(nrows) * sizeof(int32_t)));
    ASSERT_TRUE(wire_stream_append_column(&s, &c1, c1.data,
                                          static_cast<size_t>(nrows) * sizeof(int64_t)));
    ASSERT_EQ(wire_stream_finalize(&s), need);

    ASSERT_EQ(ref.size(), str.size());
    for (size_t i = 0; i < ref.size(); ++i) {
        ASSERT_EQ(ref[i], str[i]) << "mismatch at byte " << i;
    }

    // Round-trip.
    BoltBatch dst;
    ASSERT_TRUE(wire::bolt_wire_deserialize(str.data(), str.size(),
                                            &dst, &arena_dst));
    ASSERT_EQ(dst.num_cols, 2u);
    ASSERT_EQ(dst.num_rows, nrows);
    const int32_t* dp0 = static_cast<const int32_t*>(dst.col(0).data);
    const int64_t* dp1 = static_cast<const int64_t*>(dst.col(1).data);
    for (int64_t r = 0; r < nrows; ++r) {
        EXPECT_EQ(dp0[r], p0[r]);
        EXPECT_EQ(dp1[r], p1[r]);
    }
}

// ---------------------------------------------------------------------------
// Declared raw_len mismatch must be rejected.
// ---------------------------------------------------------------------------
TEST(BoltWireStream, RejectWrongRawLen) {
    constexpr uint32_t ncols = 1;
    constexpr int64_t  nrows = 8;

    Arena arena_src;
    BoltBatch src;
    build_int32_batch(&src, &arena_src, ncols, nrows);

    std::vector<uint8_t> buf(4096, 0);
    WireStream s;
    ASSERT_TRUE(wire_stream_begin_file(&s, buf.data(), buf.size(),
                                       src.schema.fields, ncols));

    const BoltColumn& col = src.col(0);
    // Off by one byte — must reject.
    const size_t bad_len = static_cast<size_t>(col.length) * sizeof(int32_t) - 1;
    EXPECT_FALSE(wire_stream_append_column(&s, &col, col.data, bad_len));
    EXPECT_FALSE(s.ok);
}
