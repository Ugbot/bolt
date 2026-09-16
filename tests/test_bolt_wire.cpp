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

// ===========================================================================
// G2ICE-141 — narrowed zero-fill differential suite.
//
// bolt_wire_serialize used to open with `memset(buf, 0, total)` over the
// ENTIRE wire image (~3MB for a real ingest batch) — measured at ~29% of the
// synchronous ingest hot path — even though every payload byte is immediately
// overwritten. The fix narrows zeroing to the metadata region + alignment
// pads + skipped-source spans only.
//
// Differential harness: serialize the same batch into a 0x00-prefilled buffer
// AND a 0xAA-poisoned buffer. Any byte of [0, total) that the serializer
// neither writes nor zeroes shows through as 0xAA and the two images differ.
// The old whole-buffer memset made the 0x00 image definitionally correct, so
// byte-equality here proves the narrowed zero-fill reproduces the old
// serializer's output EXACTLY, for every column-type/nullability shape.
// These tests were written (and pass) against the pre-fix serializer first.
// ===========================================================================

namespace {

// Serialize `src` twice — zeroed buffer vs poisoned buffer — and require
// byte-identical images. Returns the image for further inspection.
std::vector<uint8_t> serialize_differential(BoltBatch* src) {
    const size_t need = wire::bolt_wire_size(src);
    EXPECT_GT(need, 0u);
    if (need == 0) return {};

    std::vector<uint8_t> clean(need, 0x00);
    std::vector<uint8_t> poison(need, 0xAA);
    const size_t w0 = wire::bolt_wire_serialize(src, clean.data(), need);
    const size_t w1 = wire::bolt_wire_serialize(src, poison.data(), need);
    EXPECT_EQ(w0, need);
    EXPECT_EQ(w1, need);
    EXPECT_EQ(0, std::memcmp(clean.data(), poison.data(), need))
        << "wire image depends on prior buffer contents: some byte is "
           "neither written nor zeroed (uninitialized-leak / determinism "
           "regression)";
    return clean;
}

StringView sv_make_inline(const char* bytes, uint32_t len) {
    StringView v; std::memset(&v, 0, sizeof(v)); v.length = len;
    const uint32_t p = (len < 4u) ? len : 4u;
    if (p > 0) std::memcpy(v.prefix, bytes, p);
    if (len > 4u) std::memcpy(v.inline_data, bytes + 4, len - 4u);
    return v;
}
StringView sv_make_spilled(const char* bytes, uint32_t len, uint32_t off) {
    StringView v; std::memset(&v, 0, sizeof(v)); v.length = len;
    std::memcpy(v.prefix, bytes, 4);
    v.ref.buf_idx = 0; v.ref.offset = off;
    return v;
}

void set_field(BoltBatch* b, uint32_t i, const char* name, BoltType t,
               bool nullable, uint32_t fixed_size = 0) {
    BoltField& f = b->schema.fields[i];
    std::memset(&f, 0, sizeof(f));
    f.set_name(name);
    f.type = t;
    f.nullable = nullable;
    f.fixed_size = fixed_size;
}

}  // namespace

// Fixed-width, non-nullable — the hot ingest shape the memset was pure waste
// for (int64 ts / uint64 pk / float64 val).
TEST(BoltWireZeroFill, FixedWidthNonNullableDifferential) {
    Arena arena_src, arena_dst;
    BoltBatch src;
    BoltBatch::init_empty(&src);
    src.arena = &arena_src; src.num_cols = 3; src.num_rows = 100;
    src.schema.num_fields = 3;
    BoltBatch::alloc_columns(&src, &arena_src, 3);
    set_field(&src, 0, "ts",  BoltType::Int64,   false);
    set_field(&src, 1, "pk",  BoltType::UInt64,  false);
    set_field(&src, 2, "val", BoltType::Float64, false);
    const BoltType kTypes[3] = { BoltType::Int64, BoltType::UInt64,
                                 BoltType::Float64 };
    for (uint32_t c = 0; c < 3; ++c) {
        BoltColumn col = BoltColumn::make_flat_alloc(100, kTypes[c], &arena_src);
        ASSERT_NE(col.data, nullptr);
        auto* p = static_cast<uint64_t*>(col.data);
        for (int64_t r = 0; r < 100; ++r) p[r] = c * 100000ull + r * 7ull;
        src.columns[0][c] = col; src.columns[1][c] = col;
    }

    std::vector<uint8_t> img = serialize_differential(&src);
    ASSERT_FALSE(img.empty());

    BoltBatch dst; BoltBatch::init_empty(&dst);
    ASSERT_TRUE(wire::bolt_wire_deserialize(img.data(), img.size(), &dst,
                                            &arena_dst));
    EXPECT_EQ(dst.num_rows, 100);
    for (uint32_t c = 0; c < 3; ++c) {
        const auto* sp = static_cast<const uint64_t*>(src.col(c).data);
        const auto* dp = static_cast<const uint64_t*>(dst.col(c).data);
        ASSERT_NE(dp, nullptr);
        EXPECT_EQ(0, std::memcmp(sp, dp, 100 * 8)) << "col=" << c;
    }
}

// Fixed-width nullable — validity bitmap span (b0) + its alignment pad,
// odd row count so the bitmap has trailing slack bits.
TEST(BoltWireZeroFill, FixedWidthNullableDifferential) {
    Arena arena_src, arena_dst;
    BoltBatch src;
    BoltBatch::init_empty(&src);
    src.arena = &arena_src; src.num_cols = 1; src.num_rows = 13;
    src.schema.num_fields = 1;
    BoltBatch::alloc_columns(&src, &arena_src, 1);
    set_field(&src, 0, "n", BoltType::Int32, true);
    BoltColumn col = BoltColumn::make_flat_alloc(13, BoltType::Int32, &arena_src);
    ASSERT_NE(col.data, nullptr);
    auto* p = static_cast<int32_t*>(col.data);
    for (int64_t r = 0; r < 13; ++r) p[r] = static_cast<int32_t>(r * 3 - 7);
    auto* validity = static_cast<uint8_t*>(arena_src.allocate(2, 1));
    validity[0] = 0b10110101u; validity[1] = 0b00010110u;  // rows 8..12 only
    col.validity = validity;
    col.stats.all_valid = false;
    src.columns[0][0] = col; src.columns[1][0] = col;

    std::vector<uint8_t> img = serialize_differential(&src);
    ASSERT_FALSE(img.empty());

    BoltBatch dst; BoltBatch::init_empty(&dst);
    ASSERT_TRUE(wire::bolt_wire_deserialize(img.data(), img.size(), &dst,
                                            &arena_dst));
    const BoltColumn& d0 = dst.col(0);
    ASSERT_NE(d0.validity, nullptr);
    EXPECT_EQ(d0.validity[0], validity[0]);
    EXPECT_EQ(d0.validity[1], validity[1]);
    EXPECT_EQ(0, std::memcmp(d0.data, p, 13 * 4));
}

// All-null column — validity present and entirely zero.
TEST(BoltWireZeroFill, AllNullColumnDifferential) {
    Arena arena_src, arena_dst;
    BoltBatch src;
    BoltBatch::init_empty(&src);
    src.arena = &arena_src; src.num_cols = 1; src.num_rows = 9;
    src.schema.num_fields = 1;
    BoltBatch::alloc_columns(&src, &arena_src, 1);
    set_field(&src, 0, "dead", BoltType::Int64, true);
    BoltColumn col = BoltColumn::make_flat_alloc(9, BoltType::Int64, &arena_src);
    ASSERT_NE(col.data, nullptr);
    std::memset(col.data, 0x5C, 9 * 8);  // deterministic filler under nulls
    auto* validity = static_cast<uint8_t*>(arena_src.allocate(2, 1));
    validity[0] = 0; validity[1] = 0;
    col.validity = validity;
    col.stats.all_valid = false;
    src.columns[0][0] = col; src.columns[1][0] = col;

    std::vector<uint8_t> img = serialize_differential(&src);
    ASSERT_FALSE(img.empty());
    BoltBatch dst; BoltBatch::init_empty(&dst);
    ASSERT_TRUE(wire::bolt_wire_deserialize(img.data(), img.size(), &dst,
                                            &arena_dst));
    ASSERT_NE(dst.col(0).validity, nullptr);
    EXPECT_EQ(dst.col(0).validity[0], 0u);
    EXPECT_EQ(dst.col(0).validity[1], 0u);
}

// Bool — bit-packed b1 with trailing slack bits in the last byte.
TEST(BoltWireZeroFill, BoolDifferential) {
    Arena arena_src, arena_dst;
    BoltBatch src;
    BoltBatch::init_empty(&src);
    src.arena = &arena_src; src.num_cols = 1; src.num_rows = 13;
    src.schema.num_fields = 1;
    BoltBatch::alloc_columns(&src, &arena_src, 1);
    set_field(&src, 0, "flag", BoltType::Bool, false);
    BoltColumn col = BoltColumn::make_empty();
    auto* bits = static_cast<uint8_t*>(arena_src.allocate(2, 1));
    bits[0] = 0b01011010u; bits[1] = 0b00000101u;  // slack bits deliberately 0
    col.data = bits; col.length = 13; col.format = ColumnFormat::Flat;
    col.type = BoltType::Bool; col.type_size_bytes = 0;
    src.columns[0][0] = col; src.columns[1][0] = col;

    std::vector<uint8_t> img = serialize_differential(&src);
    ASSERT_FALSE(img.empty());
    BoltBatch dst; BoltBatch::init_empty(&dst);
    ASSERT_TRUE(wire::bolt_wire_deserialize(img.data(), img.size(), &dst,
                                            &arena_dst));
    const auto* dbits = static_cast<const uint8_t*>(dst.col(0).data);
    ASSERT_NE(dbits, nullptr);
    EXPECT_EQ(dbits[0], bits[0]);
    EXPECT_EQ(dbits[1], bits[1]);
}

// Embedding — runtime stride (dim*4), odd dim so b1 lands off-alignment and
// the trailing pad must be zeroed.
TEST(BoltWireZeroFill, EmbeddingDifferential) {
    Arena arena_src, arena_dst;
    constexpr uint32_t kD = 95u;   // 380 B/row — not a multiple of 64
    constexpr int64_t  kN = 7;
    BoltColumn col = BoltColumn::make_flat_vector_alloc(kN, kD, &arena_src);
    ASSERT_NE(col.data, nullptr);
    auto* f = static_cast<float*>(col.data);
    for (int64_t i = 0; i < kN * kD; ++i) f[i] = static_cast<float>(i) * 0.25f;

    BoltBatch src;
    BoltBatch::init_empty(&src);
    src.arena = &arena_src; src.num_cols = 1; src.num_rows = kN;
    src.schema.num_fields = 1;
    BoltBatch::alloc_columns(&src, &arena_src, 1);
    set_field(&src, 0, "emb", BoltType::Embedding, false, kD);
    src.columns[0][0] = col; src.columns[1][0] = col;

    std::vector<uint8_t> img = serialize_differential(&src);
    ASSERT_FALSE(img.empty());
    BoltBatch dst; BoltBatch::init_empty(&dst);
    ASSERT_TRUE(wire::bolt_wire_deserialize(img.data(), img.size(), &dst,
                                            &arena_dst));
    EXPECT_EQ(dst.col(0).vector_dim(), kD);
    EXPECT_EQ(0, std::memcmp(dst.col(0).data, f,
                             static_cast<size_t>(kN) * kD * 4));
}

// VarBinary (nullable) — offsets (b1) + payload (b2) spans + pads.
TEST(BoltWireZeroFill, VarBinaryNullableDifferential) {
    Arena arena_src, arena_dst;
    BoltBatch src;
    BoltBatch::init_empty(&src);
    src.arena = &arena_src; src.num_cols = 1; src.num_rows = 4;
    src.schema.num_fields = 1;
    BoltBatch::alloc_columns(&src, &arena_src, 1);
    set_field(&src, 0, "body", BoltType::Utf8, true);

    const char* bodies[4] = { "alpha", "", "supercalifragilistic", "z" };
    int32_t lens[4]       = { 5, 0, 20, 1 };
    auto* offs = static_cast<int32_t*>(arena_src.allocate(5 * 4, 64));
    offs[0] = 0;
    for (int i = 0; i < 4; ++i) offs[i + 1] = offs[i] + lens[i];
    auto* pool = static_cast<uint8_t*>(arena_src.allocate(64, 64));
    { int32_t cur = 0;
      for (int i = 0; i < 4; ++i) {
          if (lens[i] > 0) { std::memcpy(pool + cur, bodies[i], lens[i]); cur += lens[i]; } } }
    auto* validity = static_cast<uint8_t*>(arena_src.allocate(1, 1));
    validity[0] = 0b00001101u;  // row 1 null
    BoltColumn c0 = BoltColumn::make_var_binary(pool, validity, offs, 4,
                                                BoltType::Utf8, &arena_src);
    ASSERT_EQ(c0.format, ColumnFormat::VarBinary);
    src.columns[0][0] = c0; src.columns[1][0] = c0;

    std::vector<uint8_t> img = serialize_differential(&src);
    ASSERT_FALSE(img.empty());
    BoltBatch dst; BoltBatch::init_empty(&dst);
    ASSERT_TRUE(wire::bolt_wire_deserialize(img.data(), img.size(), &dst,
                                            &arena_dst));
    const BoltColumn& d0 = dst.col(0);
    for (int i = 0; i < 4; ++i) {
        const uint8_t* bp = nullptr; int32_t bl = 0;
        d0.var_binary_at(i, &bp, &bl);
        EXPECT_EQ(bl, lens[i]);
        if (lens[i] > 0) EXPECT_EQ(0, std::memcmp(bp, bodies[i], lens[i]));
    }
    ASSERT_NE(d0.validity, nullptr);
    EXPECT_EQ(d0.validity[0] & 0x0Fu, 0b00001101u);
}

// Zero-row batch with a VarBinary column whose dict_child is NULL: b1 is
// still 4 bytes (offsets[0]) but the copy is SKIPPED — the old full memset
// silently supplied offsets[0]=0, so the narrowed zero-fill must too.
TEST(BoltWireZeroFill, ZeroRowVarBinaryNullDictDifferential) {
    Arena arena_src, arena_dst;
    BoltBatch src;
    BoltBatch::init_empty(&src);
    src.arena = &arena_src; src.num_cols = 2; src.num_rows = 0;
    src.schema.num_fields = 2;
    BoltBatch::alloc_columns(&src, &arena_src, 2);
    set_field(&src, 0, "k", BoltType::Int32, false);
    set_field(&src, 1, "v", BoltType::Utf8, false);

    BoltColumn c0 = BoltColumn::make_empty();
    c0.format = ColumnFormat::Flat; c0.type = BoltType::Int32;
    c0.type_size_bytes = 4; c0.length = 0;
    src.columns[0][0] = c0; src.columns[1][0] = c0;

    BoltColumn c1 = BoltColumn::make_empty();
    c1.format = ColumnFormat::VarBinary; c1.type = BoltType::Utf8;
    c1.length = 0;                       // dict_child stays NULL
    src.columns[0][1] = c1; src.columns[1][1] = c1;

    std::vector<uint8_t> img = serialize_differential(&src);
    ASSERT_FALSE(img.empty());

    // The skipped offsets span must read back as offsets[0] == 0, exactly
    // as the old whole-buffer memset guaranteed.
    BoltBatch dst; BoltBatch::init_empty(&dst);
    ASSERT_TRUE(wire::bolt_wire_deserialize(img.data(), img.size(), &dst,
                                            &arena_dst));
    EXPECT_EQ(dst.num_rows, 0);
    EXPECT_EQ(dst.num_cols, 2u);
    const BoltColumn& d1 = dst.col(1);
    EXPECT_EQ(d1.format, ColumnFormat::VarBinary);
    EXPECT_EQ(d1.length, 0);
}

// Flat Utf8 — StringView rows (inline + spilled + null-with-garbage) and the
// spilled-bytes b2 span, both whole-column and sliced (min-offset rebase).
TEST(BoltWireZeroFill, FlatUtf8Differential) {
    Arena arena_src, arena_dst;
    const char* strA = "AAAAAAAAAAAAAAAAAAAA";  // 20 B, spilled
    const char* strB = "BBBBBBBBBBBBBBBBBB";    // 18 B, spilled
    auto* ovf = static_cast<char*>(arena_src.allocate(64, 1));
    std::memset(ovf, 0, 64);
    std::memcpy(ovf, strA, 20);
    std::memcpy(ovf + 20, strB, 18);

    constexpr int64_t kN = 5;
    auto* rows = arena_src.allocate_array<StringView>(kN);
    rows[0] = sv_make_inline("hi", 2);
    rows[1] = sv_make_spilled(strA, 20, 0);
    rows[2] = sv_make_inline("twelve_chars", 12);
    rows[3] = sv_make_spilled(strB, 18, 20);
    std::memset(&rows[4], 0xAB, sizeof(StringView));  // garbage; row NULL

    auto* validity = static_cast<uint8_t*>(arena_src.allocate(1, 1));
    *validity = 0b00001111u;

    BoltBatch src;
    BoltBatch::init_empty(&src);
    src.arena = &arena_src; src.num_cols = 1; src.num_rows = kN;
    src.schema.num_fields = 1;
    BoltBatch::alloc_columns(&src, &arena_src, 1);
    set_field(&src, 0, "phrase", BoltType::Utf8, true);
    BoltColumn c0 = BoltColumn::make_flat(rows, validity, kN, BoltType::Utf8);
    c0.str_overflow_base = ovf;
    src.columns[0][0] = c0; src.columns[1][0] = c0;

    std::vector<uint8_t> img = serialize_differential(&src);
    ASSERT_FALSE(img.empty());
    BoltBatch dst; BoltBatch::init_empty(&dst);
    ASSERT_TRUE(wire::bolt_wire_deserialize(img.data(), img.size(), &dst,
                                            &arena_dst));
    const auto* drows = static_cast<const StringView*>(dst.col(0).data);
    const auto* dbase = static_cast<const char*>(dst.col(0).str_overflow_base);
    ASSERT_NE(drows, nullptr);
    ASSERT_NE(dbase, nullptr);
    EXPECT_EQ(drows[1].length, 20u);
    EXPECT_EQ(0, std::memcmp(dbase + drows[1].ref.offset, strA, 20));
    EXPECT_EQ(drows[3].length, 18u);
    EXPECT_EQ(0, std::memcmp(dbase + drows[3].ref.offset, strB, 18));
}

// Zero-row, zero-column edge.
TEST(BoltWireZeroFill, EmptyBatchDifferential) {
    Arena arena_src, arena_dst;
    BoltBatch src;
    BoltBatch::init_empty(&src);
    src.arena = &arena_src; src.num_cols = 0; src.num_rows = 0;
    src.schema.num_fields = 0;
    std::vector<uint8_t> img = serialize_differential(&src);
    ASSERT_FALSE(img.empty());
    BoltBatch dst; BoltBatch::init_empty(&dst);
    ASSERT_TRUE(wire::bolt_wire_deserialize(img.data(), img.size(), &dst,
                                            &arena_dst));
    EXPECT_EQ(dst.num_cols, 0u);
    EXPECT_EQ(dst.num_rows, 0);
}

// Kitchen sink: every supported column family in ONE batch, plus structural
// checks that the metadata pads (descriptor tail bytes, desc→data alignment
// gap) are zero in the poisoned image.
TEST(BoltWireZeroFill, MixedKitchenSinkDifferential) {
    Arena arena_src, arena_dst;
    constexpr int64_t kN = 5;
    BoltBatch src;
    BoltBatch::init_empty(&src);
    src.arena = &arena_src; src.num_cols = 5; src.num_rows = kN;
    src.schema.num_fields = 5;
    BoltBatch::alloc_columns(&src, &arena_src, 5);

    // col0: int64 non-nullable
    set_field(&src, 0, "ts", BoltType::Int64, false);
    BoltColumn c0 = BoltColumn::make_flat_alloc(kN, BoltType::Int64, &arena_src);
    for (int64_t r = 0; r < kN; ++r)
        static_cast<int64_t*>(c0.data)[r] = 1'000'000 + r;
    src.columns[0][0] = c0; src.columns[1][0] = c0;

    // col1: int32 nullable
    set_field(&src, 1, "n", BoltType::Int32, true);
    BoltColumn c1 = BoltColumn::make_flat_alloc(kN, BoltType::Int32, &arena_src);
    for (int64_t r = 0; r < kN; ++r)
        static_cast<int32_t*>(c1.data)[r] = static_cast<int32_t>(r * 11);
    auto* v1 = static_cast<uint8_t*>(arena_src.allocate(1, 1));
    *v1 = 0b00010101u;
    c1.validity = v1; c1.stats.all_valid = false;
    src.columns[0][1] = c1; src.columns[1][1] = c1;

    // col2: bool
    set_field(&src, 2, "flag", BoltType::Bool, false);
    BoltColumn c2 = BoltColumn::make_empty();
    auto* bits = static_cast<uint8_t*>(arena_src.allocate(1, 1));
    *bits = 0b00010110u;
    c2.data = bits; c2.length = kN; c2.format = ColumnFormat::Flat;
    c2.type = BoltType::Bool; c2.type_size_bytes = 0;
    src.columns[0][2] = c2; src.columns[1][2] = c2;

    // col3: VarBinary Utf8
    set_field(&src, 3, "tag", BoltType::Utf8, false);
    const char* tags[kN] = { "red", "", "green", "deep_ultraviolet", "b" };
    int32_t tlens[kN]    = { 3, 0, 5, 16, 1 };
    auto* offs = static_cast<int32_t*>(arena_src.allocate((kN + 1) * 4, 64));
    offs[0] = 0;
    for (int i = 0; i < kN; ++i) offs[i + 1] = offs[i] + tlens[i];
    auto* pool = static_cast<uint8_t*>(arena_src.allocate(64, 64));
    { int32_t cur = 0;
      for (int i = 0; i < kN; ++i) {
          if (tlens[i] > 0) { std::memcpy(pool + cur, tags[i], tlens[i]); cur += tlens[i]; } } }
    BoltColumn c3 = BoltColumn::make_var_binary(pool, nullptr, offs, kN,
                                                BoltType::Utf8, &arena_src);
    src.columns[0][3] = c3; src.columns[1][3] = c3;

    // col4: Embedding dim 33 (999-mod-64 stride → real pad)
    constexpr uint32_t kD = 33u;
    set_field(&src, 4, "emb", BoltType::Embedding, false, kD);
    BoltColumn c4 = BoltColumn::make_flat_vector_alloc(kN, kD, &arena_src);
    ASSERT_NE(c4.data, nullptr);
    for (int64_t i = 0; i < kN * kD; ++i)
        static_cast<float*>(c4.data)[i] = static_cast<float>(i) * -0.5f;
    src.columns[0][4] = c4; src.columns[1][4] = c4;

    std::vector<uint8_t> img = serialize_differential(&src);
    ASSERT_FALSE(img.empty());

    // Structural: descriptor tail bytes [49,56) and the desc→data alignment
    // gap must be zero (they are metadata slack, never source data).
    const uint32_t schema_off = 32u;
    const uint32_t desc_off = schema_off + 5u * 72u;
    uint32_t data_off = 0;
    std::memcpy(&data_off, img.data() + 28, 4);
    for (uint32_t i = 0; i < 5; ++i) {
        const uint8_t* d = img.data() + desc_off + i * 56u;
        for (uint32_t b = 49; b < 56; ++b)
            EXPECT_EQ(d[b], 0u) << "desc " << i << " tail byte " << b;
    }
    for (uint32_t off = desc_off + 5u * 56u; off < data_off; ++off)
        EXPECT_EQ(img[off], 0u) << "desc->data gap byte @" << off;

    // Round-trip value checks across all five columns.
    BoltBatch dst; BoltBatch::init_empty(&dst);
    ASSERT_TRUE(wire::bolt_wire_deserialize(img.data(), img.size(), &dst,
                                            &arena_dst));
    EXPECT_EQ(dst.num_cols, 5u);
    EXPECT_EQ(dst.num_rows, kN);
    EXPECT_EQ(0, std::memcmp(dst.col(0).data, c0.data, kN * 8));
    EXPECT_EQ(0, std::memcmp(dst.col(1).data, c1.data, kN * 4));
    ASSERT_NE(dst.col(1).validity, nullptr);
    EXPECT_EQ(dst.col(1).validity[0] & 0x1Fu, 0b00010101u);
    EXPECT_EQ(static_cast<const uint8_t*>(dst.col(2).data)[0], *bits);
    for (int i = 0; i < kN; ++i) {
        const uint8_t* bp = nullptr; int32_t bl = 0;
        dst.col(3).var_binary_at(i, &bp, &bl);
        EXPECT_EQ(bl, tlens[i]);
        if (tlens[i] > 0) EXPECT_EQ(0, std::memcmp(bp, tags[i], tlens[i]));
    }
    EXPECT_EQ(dst.col(4).vector_dim(), kD);
    EXPECT_EQ(0, std::memcmp(dst.col(4).data, c4.data,
                             static_cast<size_t>(kN) * kD * 4));
}

// Large variable-width payload — a single multi-KB spilled string, so the b2
// span crosses many 64-byte lines and ends off-alignment.
TEST(BoltWireZeroFill, LargeSpilledUtf8Differential) {
    Arena arena_src, arena_dst;
    constexpr size_t kBig = 100'003;  // prime → guaranteed off-alignment tail
    auto* big = static_cast<char*>(arena_src.allocate(kBig, 1));
    for (size_t i = 0; i < kBig; ++i)
        big[i] = static_cast<char>('a' + (i % 26));

    auto* rows = arena_src.allocate_array<StringView>(1);
    rows[0] = sv_make_spilled(big, static_cast<uint32_t>(kBig), 0);

    BoltBatch src;
    BoltBatch::init_empty(&src);
    src.arena = &arena_src; src.num_cols = 1; src.num_rows = 1;
    src.schema.num_fields = 1;
    BoltBatch::alloc_columns(&src, &arena_src, 1);
    set_field(&src, 0, "blob", BoltType::Utf8, false);
    BoltColumn c0 = BoltColumn::make_flat(rows, nullptr, 1, BoltType::Utf8);
    c0.str_overflow_base = big;
    src.columns[0][0] = c0; src.columns[1][0] = c0;

    std::vector<uint8_t> img = serialize_differential(&src);
    ASSERT_FALSE(img.empty());
    BoltBatch dst; BoltBatch::init_empty(&dst);
    ASSERT_TRUE(wire::bolt_wire_deserialize(img.data(), img.size(), &dst,
                                            &arena_dst));
    const auto* drows = static_cast<const StringView*>(dst.col(0).data);
    const auto* dbase = static_cast<const char*>(dst.col(0).str_overflow_base);
    ASSERT_NE(drows, nullptr);
    ASSERT_NE(dbase, nullptr);
    EXPECT_EQ(drows[0].length, kBig);
    EXPECT_EQ(0, std::memcmp(dbase + drows[0].ref.offset, big, kBig));
}
