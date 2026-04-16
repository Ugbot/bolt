// test_bolt_wire.cpp — GTest coverage for the Bolt wire format.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"
#include "bolt/wire/bolt_wire.h"

using namespace bolt;

namespace {

// Populate a BoltBatch with `ncols` Flat int32 columns of `nrows` rows each.
// Values: col[c][r] = c * 1000 + r.
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
        f.type = BoltType::Int32;
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

TEST(BoltWire, SizeSerializeRoundTrip) {
    Arena arena_src, arena_dst;
    BoltBatch src;
    build_int32_batch(&src, &arena_src, /*ncols=*/3, /*nrows=*/100);

    const size_t need = wire::bolt_wire_size(&src);
    ASSERT_GT(need, 0u);

    std::vector<uint8_t> buf(need, 0);
    const size_t written = wire::bolt_wire_serialize(&src, buf.data(), buf.size());
    ASSERT_EQ(written, need);

    // Data region offset must be 64-byte aligned.
    uint32_t data_off = 0;
    std::memcpy(&data_off, buf.data() + 28, sizeof(data_off));
    EXPECT_EQ(data_off % 64u, 0u);

    BoltBatch dst;
    ASSERT_TRUE(wire::bolt_wire_deserialize(buf.data(), buf.size(),
                                            &dst, &arena_dst));
    EXPECT_EQ(dst.num_rows, src.num_rows);
    EXPECT_EQ(dst.num_cols, src.num_cols);

    for (uint32_t c = 0; c < src.num_cols; ++c) {
        const BoltColumn& sc = src.col(c);
        const BoltColumn& dc = dst.col(c);
        ASSERT_EQ(dc.length, sc.length);
        ASSERT_EQ(dc.type, sc.type);
        const int32_t* sp = static_cast<const int32_t*>(sc.data);
        const int32_t* dp = static_cast<const int32_t*>(dc.data);
        for (int64_t r = 0; r < sc.length; ++r) {
            EXPECT_EQ(dp[r], sp[r]) << "col=" << c << " row=" << r;
        }
    }
}

TEST(BoltWire, WrongMagicRejected) {
    Arena arena_src, arena_dst;
    BoltBatch src;
    build_int32_batch(&src, &arena_src, 2, 16);

    const size_t need = wire::bolt_wire_size(&src);
    std::vector<uint8_t> buf(need, 0);
    ASSERT_EQ(wire::bolt_wire_serialize(&src, buf.data(), buf.size()), need);

    // Corrupt the magic.
    buf[0] = 'X';

    BoltBatch dst;
    EXPECT_FALSE(wire::bolt_wire_deserialize(buf.data(), buf.size(),
                                             &dst, &arena_dst));
}

TEST(BoltWire, ZeroCapacityFails) {
    Arena arena;
    BoltBatch src;
    build_int32_batch(&src, &arena, 1, 8);

    uint8_t dummy = 0;
    EXPECT_EQ(wire::bolt_wire_serialize(&src, &dummy, 0), 0u);
}

TEST(BoltWire, TooShortBufferFails) {
    Arena arena_src, arena_dst;
    BoltBatch src;
    build_int32_batch(&src, &arena_src, 2, 8);

    const size_t need = wire::bolt_wire_size(&src);
    std::vector<uint8_t> buf(need, 0);
    ASSERT_EQ(wire::bolt_wire_serialize(&src, buf.data(), buf.size()), need);

    BoltBatch dst;
    EXPECT_FALSE(wire::bolt_wire_deserialize(buf.data(), 16,
                                             &dst, &arena_dst));
}

TEST(BoltWire, DataOffsetAligned) {
    Arena arena;
    BoltBatch src;
    build_int32_batch(&src, &arena, /*ncols=*/5, /*nrows=*/37);

    const size_t need = wire::bolt_wire_size(&src);
    std::vector<uint8_t> buf(need, 0);
    ASSERT_EQ(wire::bolt_wire_serialize(&src, buf.data(), buf.size()), need);

    uint32_t data_off = 0;
    std::memcpy(&data_off, buf.data() + 28, sizeof(data_off));
    EXPECT_EQ(data_off % 64u, 0u);
}
