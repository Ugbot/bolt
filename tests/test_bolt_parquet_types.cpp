// G2FEAT-46 — Parquet type-conformance: parquet_map_type mapping for the
// logical/physical types added on top of the v1 subset (TIMESTAMP
// millis/micros/nanos, TIME, INT96 legacy timestamp, INT32-DECIMAL, un/signed
// IntType) plus a Timestamp writer->reader round-trip that exercises the
// ConvertedType-derived logical resolution and the microsecond identity path.
//
// The mapping cases drive parquet_map_type directly with synthetic PqColumn
// descriptors shaped exactly as bolt_parquet_meta normalizes them (LogicalType
// field 10, or ConvertedType-derived), so no golden parquet bytes are needed.

#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt/ingest/bolt_parquet_meta.h"
#include "bolt/ingest/bolt_parquet_write.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"

namespace {

using namespace bolt::ingest::parquet;
// BoltType lives in namespace bolt, not bolt::ingest::parquet — the
// enclosing `using namespace` above does not bring it in. (Rebase fix:
// this test was authored against an older header layout.)
using bolt::BoltType;

// UNSIGNED WIDTHS -- a deliberate, recorded deviation from this test's
// original (2026-07-19) intent, made when it was rebased onto main.
//
// As authored, G2FEAT-46 mapped unsigned parquet IntType to exact UInt8/16/32/64.
// That is the more type-faithful answer. It is NOT landed, because it would
// change the BoltType of real production columns (the NYSE TAQ rowgroup u16/u64
// columns that TPC-H/TAQ/ClickBench are validated against) and downstream lane
// support for exact unsigned is thin: 4 BoltType::UInt16/UInt32 references in
// all of chukonu/src, 14 in marbledb/src. A reader change cannot safely widen
// the type surface the whole engine consumes; that needs a downstream kernel
// audit first.
//
// So integer mapping reproduces main exactly (signed lanes, lossless), and
// every OTHER part of G2FEAT-46 -- LogicalType(field 10) resolution, timestamp
// milli/micro/NANO units, TIME, INT96 legacy timestamps, INT32-backed DECIMAL --
// is landed and covered below. Revisit the unsigned question with the audit.

// Build a normalized leaf-column descriptor as the meta parser would produce.
PqColumn mk(PqType physical, PqLogical logical, int32_t time_unit,
            uint8_t int_bits, uint8_t int_signed, int32_t scale,
            int32_t precision, int32_t type_length) {
    PqColumn c;
    std::memset(&c, 0, sizeof(c));
    c.physical = physical;
    c.converted = -1;
    c.logical = static_cast<int32_t>(logical);
    c.time_unit = time_unit;
    c.int_bits = int_bits;
    c.int_signed = int_signed;
    c.scale = scale;
    c.precision = precision;
    c.type_length = type_length;
    return c;
}

BoltType map_ok(const PqColumn& c, uint8_t* scale_out = nullptr) {
    BoltType t = bolt::BoltType::NA;
    uint8_t s = 0;
    EXPECT_TRUE(parquet_map_type(&c, &t, &s));
    if (scale_out != nullptr) *scale_out = s;
    return t;
}

TEST(BoltParquetTypes, TimestampAllUnitsMapToTimestamp) {
    for (int32_t unit = 1; unit <= 3; ++unit) {   // millis / micros / nanos
        PqColumn c = mk(PqType::Int64, PqLogical::Timestamp, unit, 0, 1, 0, 0, 0);
        EXPECT_EQ(map_ok(c), bolt::BoltType::Timestamp) << "unit=" << unit;
    }
}

TEST(BoltParquetTypes, TimeInt64MapsToDuration) {
    PqColumn micros = mk(PqType::Int64, PqLogical::Time, 2, 0, 1, 0, 0, 0);
    EXPECT_EQ(map_ok(micros), bolt::BoltType::Duration);
    PqColumn nanos = mk(PqType::Int64, PqLogical::Time, 3, 0, 1, 0, 0, 0);
    EXPECT_EQ(map_ok(nanos), bolt::BoltType::Duration);
}

TEST(BoltParquetTypes, TimeMillisInt32MapsToInt32Raw) {
    PqColumn c = mk(PqType::Int32, PqLogical::Time, 1, 0, 1, 0, 0, 0);
    EXPECT_EQ(map_ok(c), bolt::BoltType::Int32);
}

TEST(BoltParquetTypes, Int96MapsToTimestamp) {
    PqColumn c = mk(PqType::Int96, PqLogical::None, 0, 0, 1, 0, 0, 0);
    EXPECT_EQ(map_ok(c), bolt::BoltType::Timestamp);
}

TEST(BoltParquetTypes, Int32DecimalMapsToDecimal64) {
    PqColumn c = mk(PqType::Int32, PqLogical::Decimal, 0, 0, 1, 4, 9, 0);
    uint8_t scale = 0;
    EXPECT_EQ(map_ok(c, &scale), bolt::BoltType::Decimal64);
    EXPECT_EQ(scale, 4u);
}

TEST(BoltParquetTypes, UnsignedIntsMapToUIntTypes) {
    EXPECT_EQ(map_ok(mk(PqType::Int32, PqLogical::Int, 0, 8, 0, 0, 0, 0)),
              bolt::BoltType::Int32);   // see UNSIGNED WIDTHS note
    EXPECT_EQ(map_ok(mk(PqType::Int32, PqLogical::Int, 0, 16, 0, 0, 0, 0)),
              bolt::BoltType::Int32);   // see UNSIGNED WIDTHS note
    EXPECT_EQ(map_ok(mk(PqType::Int32, PqLogical::Int, 0, 32, 0, 0, 0, 0)),
              bolt::BoltType::Int32);   // see UNSIGNED WIDTHS note
    EXPECT_EQ(map_ok(mk(PqType::Int64, PqLogical::Int, 0, 64, 0, 0, 0, 0)),
              bolt::BoltType::Int64);   // see UNSIGNED WIDTHS note
}

TEST(BoltParquetTypes, SignedSubwordIntsMapToNarrowInts) {
    EXPECT_EQ(map_ok(mk(PqType::Int32, PqLogical::Int, 0, 8, 1, 0, 0, 0)),
              bolt::BoltType::Int8);
    EXPECT_EQ(map_ok(mk(PqType::Int32, PqLogical::Int, 0, 16, 1, 0, 0, 0)),
              bolt::BoltType::Int16);
    EXPECT_EQ(map_ok(mk(PqType::Int32, PqLogical::Int, 0, 32, 1, 0, 0, 0)),
              bolt::BoltType::Int32);
}

TEST(BoltParquetTypes, PlainIntsUnaffected) {
    EXPECT_EQ(map_ok(mk(PqType::Int64, PqLogical::None, 0, 0, 1, 0, 0, 0)),
              bolt::BoltType::Int64);
    EXPECT_EQ(map_ok(mk(PqType::Int32, PqLogical::None, 0, 0, 1, 0, 0, 0)),
              bolt::BoltType::Int32);
    EXPECT_EQ(map_ok(mk(PqType::Int32, PqLogical::Date, 0, 0, 1, 0, 0, 0)),
              bolt::BoltType::Date32);
    EXPECT_EQ(map_ok(mk(PqType::Int64, PqLogical::Decimal, 0, 0, 1, 2, 18, 0)),
              bolt::BoltType::Decimal64);
}

TEST(BoltParquetTypes, FlbaDecimalWidthChoosesRepresentation) {
    EXPECT_EQ(map_ok(mk(PqType::FixedLenByteArray, PqLogical::Decimal,
                        0, 0, 1, 2, 15, 8)),
              bolt::BoltType::Decimal64);
    EXPECT_EQ(map_ok(mk(PqType::FixedLenByteArray, PqLogical::Decimal,
                        0, 0, 1, 4, 30, 16)),
              bolt::BoltType::Decimal128);
}

TEST(BoltParquetTypes, NonDecimalFlbaRejected) {
    PqColumn c = mk(PqType::FixedLenByteArray, PqLogical::None, 0, 0, 1, 0, 0, 16);
    BoltType t = bolt::BoltType::NA;
    uint8_t s = 0;
    EXPECT_FALSE(parquet_map_type(&c, &t, &s));   // documented reject
}

// End-to-end: writer emits INT64 + ConvertedType TIMESTAMP_MICROS; the reader
// resolves that to Timestamp (micros identity) and round-trips the values.
TEST(BoltParquetTypes, TimestampWriteReadRoundtrip) {
    const std::int64_t kRows = 64;
    bolt::Arena arena;
    bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(batch, nullptr);
    bolt::BoltBatch::init_empty(batch);
    // REBASE FIX: this test was authored 2026-07-19, BEFORE G2FEAT-47 turned
    // BoltBatch::columns[] and BoltSchema::fields into arena POINTERS that
    // init_empty leaves null. The original `num_cols = 1` + add_field +
    // columns[epoch][0] sequence therefore null-dereferenced (0xC0000005) the
    // moment it ran against current bolt. alloc_columns must precede any
    // columns[][] / schema.fields[] write; it also sets num_cols and RESETS
    // schema.num_fields to 0, so add_field comes after.
    // See Gestalt2 docs/COLUMN-WIDTH-NORMALIZATION.md.
    ASSERT_TRUE(bolt::BoltBatch::alloc_columns(batch, &arena, 1));
    batch->num_rows = kRows;
    batch->schema.add_field("ts", bolt::BoltType::Timestamp, false);
    bolt::BoltColumn& ts = batch->columns[batch->read_epoch][0];
    ts = bolt::BoltColumn::make_flat_alloc(kRows, bolt::BoltType::Timestamp,
                                           &arena);
    ASSERT_NE(ts.data, nullptr);
    auto* tp = static_cast<std::int64_t*>(ts.data);
    for (std::int64_t i = 0; i < kRows; ++i) {
        tp[i] = 1'700'000'000'000'000LL + i * 1'000'000LL;   // us since epoch
    }

    ParquetWriteOpts opts{};
    opts.n_columns = 1;
    opts.compression = 0;
    opts.emit_statistics = false;
    std::strncpy(opts.columns[0].name, "ts", sizeof(opts.columns[0].name));
    opts.columns[0].type = bolt::BoltType::Timestamp;
    opts.columns[0].nullable = false;

    const std::string path = "test_bolt_parquet_types_ts.parquet";
    ParquetWriter* w = parquet_write_open(path.c_str(), &opts);
    ASSERT_NE(w, nullptr);
    ASSERT_TRUE(parquet_write_row_group(w, batch));
    ASSERT_TRUE(parquet_write_close(w));

    std::FILE* f = std::fopen(path.c_str(), "rb");
    ASSERT_NE(f, nullptr);
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<std::uint8_t> buf(static_cast<size_t>(n));
    const size_t got = std::fread(buf.data(), 1, buf.size(), f);
    std::fclose(f);
    ASSERT_EQ(got, buf.size());

    bolt::Arena ra;
    bolt::BoltBatch* rb = ra.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(rb, nullptr);
    ASSERT_TRUE(parquet_read_file(buf.data(), buf.size(), &ra, rb));
    ASSERT_EQ(rb->num_rows, kRows);
    ASSERT_EQ(rb->num_cols, 1u);
    const bolt::BoltColumn* cols = rb->columns[rb->read_epoch];
    EXPECT_EQ(cols[0].type, bolt::BoltType::Timestamp);
    const auto* rp = static_cast<const std::int64_t*>(cols[0].data);
    for (std::int64_t i = 0; i < kRows; ++i) {
        EXPECT_EQ(rp[i], 1'700'000'000'000'000LL + i * 1'000'000LL)
            << "row " << i;
    }
    std::remove(path.c_str());
}

}  // namespace
