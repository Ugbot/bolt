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
    BoltBatch::alloc_columns(b, arena, ncols);  // G2FEAT-47: size columns[2]
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

// ---------------------------------------------------------------------------
// VarBinary round-trip (wire v2). Mixed-shape batch: Int64 PK + Utf8
// VarBinary body. Includes an empty-string row so the offsets[i] ==
// offsets[i+1] sentinel path is exercised.
// ---------------------------------------------------------------------------
TEST(BoltWire, VarBinaryRoundTrip) {
    Arena arena_src, arena_dst;

    BoltBatch src;
    BoltBatch::init_empty(&src);
    src.arena    = &arena_src;
    src.num_cols = 2;
    src.num_rows = 4;
    src.schema.num_fields = 2;
    BoltBatch::alloc_columns(&src, &arena_src, 2);  // G2FEAT-47: size columns[2]
    {
        BoltField& f0 = src.schema.fields[0];
        std::memset(&f0, 0, sizeof(f0));
        std::memcpy(f0.name, "id", 2);
        f0.type = BoltType::Int64;
        BoltField& f1 = src.schema.fields[1];
        std::memset(&f1, 0, sizeof(f1));
        std::memcpy(f1.name, "body", 4);
        f1.type = BoltType::Utf8;
    }

    int64_t* ids = static_cast<int64_t*>(arena_src.allocate(4 * 8, 64));
    ids[0] = 100; ids[1] = 200; ids[2] = 300; ids[3] = 400;
    BoltColumn c0 = BoltColumn::make_flat(ids, nullptr, 4, BoltType::Int64);
    src.columns[0][0] = c0;
    src.columns[1][0] = c0;

    const char* bodies[4] = { "alpha", "", "carrot_juice", "z" };
    int32_t lens[4]       = { 5, 0, 12, 1 };
    int32_t* offs = static_cast<int32_t*>(arena_src.allocate(5 * 4, 64));
    offs[0] = 0;
    for (int i = 0; i < 4; ++i) offs[i + 1] = offs[i] + lens[i];
    uint8_t* pool = static_cast<uint8_t*>(
        arena_src.allocate(static_cast<size_t>(offs[4] > 0 ? offs[4] : 1), 64));
    {
        int32_t cur = 0;
        for (int i = 0; i < 4; ++i) {
            if (lens[i] > 0) {
                std::memcpy(pool + cur, bodies[i], lens[i]);
                cur += lens[i];
            }
        }
    }
    BoltColumn c1 = BoltColumn::make_var_binary(
        pool, nullptr, offs, 4, BoltType::Utf8, &arena_src);
    ASSERT_EQ(c1.format, ColumnFormat::VarBinary);
    src.columns[0][1] = c1;
    src.columns[1][1] = c1;

    const size_t need = wire::bolt_wire_size(&src);
    ASSERT_GT(need, 0u);
    std::vector<uint8_t> buf(need, 0);
    ASSERT_EQ(wire::bolt_wire_serialize(&src, buf.data(), buf.size()), need);

    BoltBatch dst;
    BoltBatch::init_empty(&dst);
    ASSERT_TRUE(
        wire::bolt_wire_deserialize(buf.data(), buf.size(), &dst, &arena_dst));
    EXPECT_EQ(dst.num_cols, 2u);
    EXPECT_EQ(dst.num_rows, 4);

    const BoltColumn& d0 = dst.columns[dst.read_epoch][0];
    EXPECT_EQ(d0.format, ColumnFormat::Flat);
    const int64_t* dids = static_cast<const int64_t*>(d0.data);
    EXPECT_EQ(dids[0], 100); EXPECT_EQ(dids[3], 400);

    const BoltColumn& d1 = dst.columns[dst.read_epoch][1];
    EXPECT_EQ(d1.format, ColumnFormat::VarBinary);
    EXPECT_EQ(d1.length, 4);
    for (int i = 0; i < 4; ++i) {
        const uint8_t* bp = nullptr;
        int32_t        bl = 0;
        d1.var_binary_at(i, &bp, &bl);
        EXPECT_EQ(bl, lens[i]);
        if (lens[i] > 0) {
            EXPECT_EQ(0, std::memcmp(bp, bodies[i], lens[i]));
        }
    }
}

TEST(BoltWire, VersionStampIsThree) {
    Arena arena;
    BoltBatch src;
    build_int32_batch(&src, &arena, 1, 1);
    const size_t need = wire::bolt_wire_size(&src);
    std::vector<uint8_t> buf(need, 0);
    ASSERT_EQ(wire::bolt_wire_serialize(&src, buf.data(), buf.size()), need);
    uint32_t version = 0;
    std::memcpy(&version, buf.data() + 4, sizeof(version));
    EXPECT_EQ(version, 3u);
}

// ---------------------------------------------------------------------------
// Flat Utf8 round-trip (wire v3). Prior to this fix, `is_supported_format_pair`
// excluded (Utf8, Flat) entirely, so `bolt_wire_size()` returned 0 for any
// batch carrying a Flat-format Utf8 column — which is what every real
// Parquet-decoded Utf8 column looks like (`bolt::ingest::parquet` always
// emits Flat StringViews, never VarBinary). Mixes inline (<=12 byte) and
// spilled (>12 byte) values, plus a null row holding garbage-looking bytes,
// to exercise the same row-walk `flat_utf8_sizes`/`clone_into` share.
// ---------------------------------------------------------------------------
TEST(BoltWire, FlatUtf8RoundTrip) {
    Arena arena_src, arena_dst;

    BoltBatch src;
    BoltBatch::init_empty(&src);
    src.arena    = &arena_src;
    src.num_cols = 1;
    src.num_rows = 5;
    src.schema.num_fields = 1;
    BoltBatch::alloc_columns(&src, &arena_src, 1);  // G2FEAT-47: size columns[2]
    {
        BoltField& f0 = src.schema.fields[0];
        std::memset(&f0, 0, sizeof(f0));
        std::memcpy(f0.name, "phrase", 6);
        f0.type = BoltType::Utf8;
    }

    const char* strA = "AAAAAAAAAAAAAAAAAAAA";  // 20 bytes, spilled
    const char* strB = "BBBBBBBBBBBBBBBBBB";     // 18 bytes, spilled
    ASSERT_EQ(std::strlen(strA), 20u);
    ASSERT_EQ(std::strlen(strB), 18u);

    constexpr size_t kOvfCap = 64;
    auto* ovf = static_cast<char*>(arena_src.allocate(kOvfCap, 1));
    std::memset(ovf, 0, kOvfCap);
    std::memcpy(ovf, strA, 20);
    std::memcpy(ovf + 20, strB, 18);

    constexpr int64_t kN = 5;
    auto* rows = arena_src.allocate_array<StringView>(kN);

    auto make_inline = [](const char* bytes, uint32_t len) {
        StringView v; std::memset(&v, 0, sizeof(v)); v.length = len;
        const uint32_t p = (len < 4u) ? len : 4u;
        if (p > 0) std::memcpy(v.prefix, bytes, p);
        if (len > 4u) std::memcpy(v.inline_data, bytes + 4, len - 4u);
        return v;
    };
    auto make_spilled = [](const char* bytes, uint32_t len, uint32_t off) {
        StringView v; std::memset(&v, 0, sizeof(v)); v.length = len;
        std::memcpy(v.prefix, bytes, 4);
        v.ref.buf_idx = 0; v.ref.offset = off;
        return v;
    };

    rows[0] = make_inline("hi", 2);
    rows[1] = make_spilled(strA, 20, 0);
    rows[2] = make_inline("twelve_chars", 12);
    rows[3] = make_spilled(strB, 18, 20);
    std::memset(&rows[4], 0xAB, sizeof(StringView));  // garbage; row is NULL

    uint8_t* validity = static_cast<uint8_t*>(arena_src.allocate(1, 1));
    *validity = 0b00001111u;  // bits 0-3 valid, bit 4 (row 4) null

    BoltColumn c0 = BoltColumn::make_flat(rows, validity, kN, BoltType::Utf8);
    c0.str_overflow_base = ovf;
    src.columns[0][0] = c0;
    src.columns[1][0] = c0;

    const size_t need = wire::bolt_wire_size(&src);
    ASSERT_GT(need, 0u);
    std::vector<uint8_t> buf(need, 0);
    ASSERT_EQ(wire::bolt_wire_serialize(&src, buf.data(), buf.size()), need);

    BoltBatch dst;
    BoltBatch::init_empty(&dst);
    ASSERT_TRUE(
        wire::bolt_wire_deserialize(buf.data(), buf.size(), &dst, &arena_dst));
    EXPECT_EQ(dst.num_cols, 1u);
    EXPECT_EQ(dst.num_rows, kN);

    const BoltColumn& d0 = dst.columns[dst.read_epoch][0];
    EXPECT_EQ(d0.format, ColumnFormat::Flat);
    EXPECT_EQ(d0.type, BoltType::Utf8);
    ASSERT_NE(d0.str_overflow_base, nullptr);

    const auto* drows = static_cast<const StringView*>(d0.data);
    const auto* dbase = static_cast<const char*>(d0.str_overflow_base);

    EXPECT_EQ(drows[0].length, 2u);
    EXPECT_EQ(std::memcmp(drows[0].prefix, "hi", 2), 0);

    EXPECT_EQ(drows[1].length, 20u);
    EXPECT_EQ(std::memcmp(dbase + drows[1].ref.offset, strA, 20), 0);

    EXPECT_EQ(drows[2].length, 12u);

    EXPECT_EQ(drows[3].length, 18u);
    EXPECT_EQ(std::memcmp(dbase + drows[3].ref.offset, strB, 18), 0);

    // Null row's validity bit survived the round-trip.
    ASSERT_NE(d0.validity, nullptr);
    EXPECT_EQ((d0.validity[0] >> 4) & 1u, 0u);
}

// G2FEAT-311: a chunked/windowed writer (e.g. chukonu's streaming
// Parquet->MarbleDB loader) serializes a Flat-Utf8 column that is a SLICE
// of a larger one -- `data` pointer advanced past row 0, `str_overflow_base`
// left as the ORIGINAL (absolute) buffer. Before the fix, flat_utf8_sizes
// measured spilled bytes as max(ref.offset+length) with an implicit "the
// column starts at offset 0 of its own buffer" assumption, so a late slice
// (whose rows reference large absolute offsets) serialized a b2 spanning
// EVERY earlier row's bytes too -- unboundedly growing with the slice's
// start position, not with its own row count. This proves a late 2-row
// slice of a 4-row column wire-sizes/serializes proportionally to its own
// 2 rows, and round-trips to the correct string content.
TEST(BoltWire, FlatUtf8SlicedViewRoundTrip) {
    Arena arena_src, arena_dst;

    // Build a FULL 4-row column: two early spilled rows (small absolute
    // offsets) and two late spilled rows (large absolute offsets, as if
    // many earlier rows' worth of string data already precedes them).
    const char* strEarly0 = "EARLY_ROW_ZERO_SPILLED_TEXT";       // 28 bytes
    const char* strEarly1 = "EARLY_ROW_ONE_SPILLED_TEXT_TOO";    // 30 bytes
    const char* strLate0  = "LATE_ROW_TWO_SPILLED_TEXT_HERE";    // 30 bytes
    const char* strLate1  = "LATE_ROW_THREE_SPILLED_TEXT_END";   // 31 bytes
    const uint32_t lenEarly0 = static_cast<uint32_t>(std::strlen(strEarly0));
    const uint32_t lenEarly1 = static_cast<uint32_t>(std::strlen(strEarly1));
    const uint32_t lenLate0  = static_cast<uint32_t>(std::strlen(strLate0));
    const uint32_t lenLate1  = static_cast<uint32_t>(std::strlen(strLate1));

    // Simulate a MUCH larger prefix (as if thousands of earlier rows'
    // spilled bytes already occupy this space) before the two "late" rows'
    // actual bytes -- large enough that pre-fix code (which would copy
    // [0, max_end)) would produce a payload orders of magnitude bigger
    // than these two rows' own ~60 bytes.
    constexpr uint32_t kFakePrefixBytes = 200000;
    const uint32_t off_early0 = 0;
    const uint32_t off_early1 = off_early0 + lenEarly0;
    const uint32_t off_late0  = kFakePrefixBytes;
    const uint32_t off_late1  = off_late0 + lenLate0;

    const uint32_t total_ovf = off_late1 + lenLate1;
    auto* ovf = static_cast<char*>(arena_src.allocate(total_ovf, 1));
    std::memset(ovf, 0, total_ovf);
    std::memcpy(ovf + off_early0, strEarly0, lenEarly0);
    std::memcpy(ovf + off_early1, strEarly1, lenEarly1);
    std::memcpy(ovf + off_late0, strLate0, lenLate0);
    std::memcpy(ovf + off_late1, strLate1, lenLate1);

    auto make_spilled = [](const char* bytes, uint32_t len, uint32_t off) {
        StringView v; std::memset(&v, 0, sizeof(v)); v.length = len;
        std::memcpy(v.prefix, bytes, 4);
        v.ref.buf_idx = 0; v.ref.offset = off;
        return v;
    };

    constexpr int64_t kFullN = 4;
    auto* full_rows = arena_src.allocate_array<StringView>(kFullN);
    full_rows[0] = make_spilled(strEarly0, lenEarly0, off_early0);
    full_rows[1] = make_spilled(strEarly1, lenEarly1, off_early1);
    full_rows[2] = make_spilled(strLate0, lenLate0, off_late0);
    full_rows[3] = make_spilled(strLate1, lenLate1, off_late1);

    // The SLICE: rows [2, 4) only. `data` advances past rows 0-1;
    // `str_overflow_base` stays the SAME absolute buffer (unchanged) --
    // exactly how chukonu's row-group chunking builds a chunk view.
    BoltBatch src;
    BoltBatch::init_empty(&src);
    src.arena    = &arena_src;
    src.num_cols = 1;
    src.num_rows = 2;
    src.schema.num_fields = 1;
    BoltBatch::alloc_columns(&src, &arena_src, 1);  // G2FEAT-47: size columns[2]
    {
        BoltField& f0 = src.schema.fields[0];
        std::memset(&f0, 0, sizeof(f0));
        std::memcpy(f0.name, "phrase", 6);
        f0.type = BoltType::Utf8;
    }
    BoltColumn c0 = BoltColumn::make_flat(full_rows + 2, /*validity=*/nullptr,
                                          /*length=*/2, BoltType::Utf8);
    c0.str_overflow_base = ovf;   // unchanged absolute base
    src.columns[0][0] = c0;
    src.columns[1][0] = c0;

    const size_t need = wire::bolt_wire_size(&src);
    ASSERT_GT(need, 0u);
    // The whole point of the fix: a 2-row slice must wire-size proportional
    // to its OWN ~61 bytes of spilled content, not the ~200 KB fake prefix
    // that precedes it in the original buffer.
    EXPECT_LT(need, 4096u) << "sliced Flat-Utf8 column over-counted spilled "
                              "bytes from rows outside the slice (min-offset "
                              "rebasing regressed)";

    std::vector<uint8_t> buf(need, 0);
    ASSERT_EQ(wire::bolt_wire_serialize(&src, buf.data(), buf.size()), need);

    BoltBatch dst;
    BoltBatch::init_empty(&dst);
    ASSERT_TRUE(
        wire::bolt_wire_deserialize(buf.data(), buf.size(), &dst, &arena_dst));
    EXPECT_EQ(dst.num_rows, 2);

    const BoltColumn& d0 = dst.columns[dst.read_epoch][0];
    ASSERT_NE(d0.str_overflow_base, nullptr);
    const auto* drows = static_cast<const StringView*>(d0.data);
    const auto* dbase = static_cast<const char*>(d0.str_overflow_base);

    EXPECT_EQ(drows[0].length, lenLate0);
    EXPECT_EQ(std::memcmp(dbase + drows[0].ref.offset, strLate0, lenLate0), 0);
    EXPECT_EQ(drows[1].length, lenLate1);
    EXPECT_EQ(std::memcmp(dbase + drows[1].ref.offset, strLate1, lenLate1), 0);
}
