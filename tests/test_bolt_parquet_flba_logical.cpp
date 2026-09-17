// FIXED_LEN_BYTE_ARRAY beyond DECIMAL (G2PQ-16): UUID, FLOAT16, the
// deprecated ConvertedType.INTERVAL, and raw/unannotated FLBA.
//
// Fixtures are written by REAL, independent implementations (DuckDB for
// UUID + INTERVAL, pyarrow for FLOAT16 + unannotated FixedSizeBinary --
// see scripts/make_flba_logical_fixtures.py), never by bolt's own writer:
// a fixture bolt wrote could only prove bolt self-consistent with itself.
// Expected byte values below were read back from the fixtures with
// pyarrow, independently of bolt.

#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt/ingest/bolt_parquet_meta.h"

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

std::vector<std::uint8_t> slurp(const char* path) {
    std::vector<std::uint8_t> v;
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) return v;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    v.resize(static_cast<std::size_t>(n));
    const std::size_t got = std::fread(v.data(), 1, v.size(), f);
    std::fclose(f);
    if (got != v.size()) v.clear();
    return v;
}

std::string data_path(const char* name) {
#ifdef BOLT_TEST_DATA_DIR
    return std::string(BOLT_TEST_DATA_DIR) + "/" + name;
#else
    return std::string("tests/data/") + name;
#endif
}

int find_col(const PqMeta* m, const char* name) {
    for (std::uint32_t c = 0; c < m->n_columns; ++c) {
        if (std::strcmp(m->columns[c].name, name) == 0) return static_cast<int>(c);
    }
    return -1;
}

bool row_valid(const bolt::BoltColumn& col, int64_t i) {
    if (col.validity == nullptr) return true;
    return ((col.validity[i >> 3] >> (i & 7)) & 1u) != 0u;
}

const std::uint8_t* row_ptr(const bolt::BoltColumn& col, int64_t i) {
    return static_cast<const std::uint8_t*>(col.data) +
           static_cast<std::size_t>(i) * col.type_size_bytes;
}

void expect_bytes(const std::uint8_t* got, const char* hex, std::size_t n,
                  const char* what) {
    for (std::size_t i = 0; i < n; ++i) {
        unsigned v = 0;
        std::sscanf(hex + i * 2, "%2x", &v);
        EXPECT_EQ(got[i], static_cast<std::uint8_t>(v))
            << what << " byte " << i;
    }
}

void expect_zero(const std::uint8_t* got, std::size_t off, std::size_t n,
                 const char* what) {
    for (std::size_t i = off; i < off + n; ++i) {
        EXPECT_EQ(got[i], 0u) << what << " padding byte " << i;
    }
}

// ---- DuckDB fixture: UUID + deprecated INTERVAL ----------------------------

TEST(BoltParquetFlbaLogical, UuidAnnotationMapsToNativeUuidType) {
    const auto buf = slurp(data_path("golden_flba_duckdb.parquet").c_str());
    ASSERT_FALSE(buf.empty()) << "fixture missing";
    bolt::Arena a;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &a, &meta));

    const int idc = find_col(&meta, "id");
    ASSERT_GE(idc, 0);
    EXPECT_EQ(meta.columns[idc].logical, static_cast<int32_t>(PqLogical::Uuid));

    bolt::BoltType t;
    std::uint8_t scale = 0;
    ASSERT_TRUE(parquet_map_type(&meta.columns[idc], &t, &scale));
    EXPECT_EQ(t, bolt::BoltType::UUID);
    // Self-describing type -- no redundant BoltLogical tag (see bolt_types.h
    // BoltLogical::Uuid comment).
    EXPECT_EQ(parquet_map_logical(&meta.columns[idc]), bolt::BoltLogical::None);
}

TEST(BoltParquetFlbaLogical, IntervalConvertedTypeMapsToFixedSizeBinary) {
    const auto buf = slurp(data_path("golden_flba_duckdb.parquet").c_str());
    ASSERT_FALSE(buf.empty());
    bolt::Arena a;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &a, &meta));

    const int ivc = find_col(&meta, "iv");
    ASSERT_GE(ivc, 0);
    // DuckDB writes INTERVAL via the DEPRECATED ConvertedType (=21), not the
    // modern LogicalType union (which has no member for it) -- confirm the
    // normalizer actually resolved it, not merely left it "no logical type".
    EXPECT_EQ(meta.columns[ivc].converted, 21);
    EXPECT_EQ(meta.columns[ivc].logical, static_cast<int32_t>(PqLogical::Interval));
    EXPECT_EQ(meta.columns[ivc].type_length, 12);

    bolt::BoltType t;
    std::uint8_t scale = 0;
    ASSERT_TRUE(parquet_map_type(&meta.columns[ivc], &t, &scale));
    EXPECT_EQ(t, bolt::BoltType::FixedSizeBinary);
    EXPECT_EQ(parquet_map_logical(&meta.columns[ivc]), bolt::BoltLogical::Interval);
}

TEST(BoltParquetFlbaLogical, DuckdbFileDecodesUuidAndIntervalBytesExactly) {
    const auto buf = slurp(data_path("golden_flba_duckdb.parquet").c_str());
    ASSERT_FALSE(buf.empty());
    bolt::Arena a;
    auto* b = a.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(b, nullptr);
    ASSERT_TRUE(parquet_read_file(buf.data(), buf.size(), &a, b));
    ASSERT_EQ(b->num_rows, 5);

    bolt::Arena ma;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &ma, &meta));
    const int idc = find_col(&meta, "id");
    const int ivc = find_col(&meta, "iv");
    ASSERT_GE(idc, 0);
    ASSERT_GE(ivc, 0);

    const bolt::BoltColumn& idcol = b->columns[b->read_epoch][idc];
    const bolt::BoltColumn& ivcol = b->columns[b->read_epoch][ivc];
    EXPECT_EQ(idcol.type, bolt::BoltType::UUID);
    EXPECT_EQ(idcol.type_size_bytes, 16);
    EXPECT_EQ(idcol.fixed_width, 0)
        << "fixed_width is a FixedSizeBinary-only field; UUID is self-sized";
    EXPECT_EQ(ivcol.type, bolt::BoltType::FixedSizeBinary);
    EXPECT_EQ(ivcol.type_size_bytes, 16);
    EXPECT_EQ(ivcol.fixed_width, 12);

    // row 1's id is NULL, row 2's iv is NULL (see the fixture generator).
    EXPECT_TRUE(row_valid(idcol, 0));
    EXPECT_FALSE(row_valid(idcol, 1));
    EXPECT_TRUE(row_valid(idcol, 2));
    expect_bytes(row_ptr(idcol, 0), "00000000000040008000000000000000", 16, "id[0]");
    expect_bytes(row_ptr(idcol, 2), "00000002000040008000000000000002", 16, "id[2]");
    expect_bytes(row_ptr(idcol, 3), "00000003000040008000000000000003", 16, "id[3]");
    expect_bytes(row_ptr(idcol, 4), "00000004000040008000000000000004", 16, "id[4]");

    EXPECT_TRUE(row_valid(ivcol, 0));
    EXPECT_TRUE(row_valid(ivcol, 1));
    EXPECT_FALSE(row_valid(ivcol, 2));
    // months=1 days=2 millis=1000, then zero-padded from 12 to the 16-byte
    // FixedSizeBinary slot.
    expect_bytes(row_ptr(ivcol, 1), "0100000002000000e8030000", 12, "iv[1]");
    expect_zero(row_ptr(ivcol, 1), 12, 4, "iv[1]");
    expect_bytes(row_ptr(ivcol, 3), "0300000006000000b80b0000", 12, "iv[3]");
    expect_bytes(row_ptr(ivcol, 4), "0400000008000000a00f0000", 12, "iv[4]");
    // months=0 days=0 millis=0 is a legitimate (non-null) all-zero interval.
    expect_bytes(row_ptr(ivcol, 0), "000000000000000000000000", 12, "iv[0]");
}

// ---- pyarrow fixture: FLOAT16 + unannotated raw FLBA -----------------------

TEST(BoltParquetFlbaLogical, Float16AnnotationMapsToNativeFloat16Type) {
    const auto buf = slurp(data_path("golden_flba_pyarrow.parquet").c_str());
    ASSERT_FALSE(buf.empty()) << "fixture missing";
    bolt::Arena a;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &a, &meta));

    const int hc = find_col(&meta, "h");
    ASSERT_GE(hc, 0);
    EXPECT_EQ(meta.columns[hc].logical, static_cast<int32_t>(PqLogical::Float16));
    EXPECT_EQ(meta.columns[hc].type_length, 2);

    bolt::BoltType t;
    std::uint8_t scale = 0;
    ASSERT_TRUE(parquet_map_type(&meta.columns[hc], &t, &scale));
    EXPECT_EQ(t, bolt::BoltType::Float16);
    EXPECT_EQ(parquet_map_logical(&meta.columns[hc]), bolt::BoltLogical::None);
}

TEST(BoltParquetFlbaLogical, RawUnannotatedFlbaMapsToFixedSizeBinary) {
    const auto buf = slurp(data_path("golden_flba_pyarrow.parquet").c_str());
    ASSERT_FALSE(buf.empty());
    bolt::Arena a;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &a, &meta));

    const int rc = find_col(&meta, "raw");
    ASSERT_GE(rc, 0);
    // The defining property of this case: NO annotation at all.
    EXPECT_EQ(meta.columns[rc].converted, -1);
    EXPECT_EQ(meta.columns[rc].logical, static_cast<int32_t>(PqLogical::None));
    EXPECT_EQ(meta.columns[rc].type_length, 5);

    bolt::BoltType t;
    std::uint8_t scale = 0;
    ASSERT_TRUE(parquet_map_type(&meta.columns[rc], &t, &scale));
    EXPECT_EQ(t, bolt::BoltType::FixedSizeBinary);
}

TEST(BoltParquetFlbaLogical, PyarrowFileDecodesFloat16AndRawFlbaBytesExactly) {
    const auto buf = slurp(data_path("golden_flba_pyarrow.parquet").c_str());
    ASSERT_FALSE(buf.empty());
    bolt::Arena a;
    auto* b = a.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(b, nullptr);
    ASSERT_TRUE(parquet_read_file(buf.data(), buf.size(), &a, b));
    ASSERT_EQ(b->num_rows, 5);

    bolt::Arena ma;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &ma, &meta));
    const int hc = find_col(&meta, "h");
    const int rc = find_col(&meta, "raw");
    ASSERT_GE(hc, 0);
    ASSERT_GE(rc, 0);

    const bolt::BoltColumn& hcol = b->columns[b->read_epoch][hc];
    const bolt::BoltColumn& rcol = b->columns[b->read_epoch][rc];
    EXPECT_EQ(hcol.type, bolt::BoltType::Float16);
    EXPECT_EQ(hcol.type_size_bytes, 2);
    EXPECT_EQ(rcol.type, bolt::BoltType::FixedSizeBinary);
    EXPECT_EQ(rcol.type_size_bytes, 16);
    EXPECT_EQ(rcol.fixed_width, 5);

    // h: 1.5, -2.0, 0.0, NaN, NULL -- raw IEEE-754 half bit patterns, no
    // widening (matches bolt's existing Arrow "e" format convention).
    EXPECT_TRUE(row_valid(hcol, 0));
    expect_bytes(row_ptr(hcol, 0), "003e", 2, "h[0]=1.5");
    expect_bytes(row_ptr(hcol, 1), "00c0", 2, "h[1]=-2.0");
    expect_bytes(row_ptr(hcol, 2), "0000", 2, "h[2]=0.0");
    expect_bytes(row_ptr(hcol, 3), "007e", 2, "h[3]=NaN");
    EXPECT_FALSE(row_valid(hcol, 4));

    // raw: "AAAAA","BBBBB",NULL,"DDDDD","EEEEE" -- literal bytes, zero-padded
    // from 5 to the 16-byte FixedSizeBinary slot.
    EXPECT_TRUE(row_valid(rcol, 0));
    expect_bytes(row_ptr(rcol, 0), "4141414141", 5, "raw[0]");
    expect_zero(row_ptr(rcol, 0), 5, 11, "raw[0]");
    expect_bytes(row_ptr(rcol, 1), "4242424242", 5, "raw[1]");
    EXPECT_FALSE(row_valid(rcol, 2));
    expect_bytes(row_ptr(rcol, 3), "4444444444", 5, "raw[3]");
    expect_bytes(row_ptr(rcol, 4), "4545454545", 5, "raw[4]");
}

// ---- boundary / negative cases (no fixture needed) -------------------------
// Spec-mandated width checks: UUID must be FIXED[16], FLOAT16 must be
// FIXED[2], INTERVAL must be FIXED[12] -- a malformed writer that annotates
// the wrong width is refused, never silently truncated or over-read. A raw
// FLBA wider than 16 bytes exceeds bolt's one-storage-slot v1 ceiling (the
// same ceiling DECIMAL already has) and is refused too.

PqColumn make_col(PqType phys, int32_t type_length, int32_t converted,
                  PqLogical logical) {
    PqColumn c{};
    c.physical = phys;
    c.type_length = type_length;
    c.converted = converted;
    c.logical = static_cast<int32_t>(logical);
    c.optional = 1;
    return c;
}

TEST(BoltParquetFlbaLogical, WrongWidthUuidIsRefused) {
    PqColumn c = make_col(PqType::FixedLenByteArray, 15, -1, PqLogical::Uuid);
    bolt::BoltType t;
    std::uint8_t scale = 0;
    EXPECT_FALSE(parquet_map_type(&c, &t, &scale));
}

TEST(BoltParquetFlbaLogical, WrongWidthFloat16IsRefused) {
    PqColumn c = make_col(PqType::FixedLenByteArray, 3, -1, PqLogical::Float16);
    bolt::BoltType t;
    std::uint8_t scale = 0;
    EXPECT_FALSE(parquet_map_type(&c, &t, &scale));
}

TEST(BoltParquetFlbaLogical, WrongWidthIntervalIsRefused) {
    PqColumn c = make_col(PqType::FixedLenByteArray, 11, 21, PqLogical::Interval);
    bolt::BoltType t;
    std::uint8_t scale = 0;
    EXPECT_FALSE(parquet_map_type(&c, &t, &scale));
}

TEST(BoltParquetFlbaLogical, RawFlbaOverSixteenBytesIsRefused) {
    PqColumn c = make_col(PqType::FixedLenByteArray, 17, -1, PqLogical::None);
    bolt::BoltType t;
    std::uint8_t scale = 0;
    EXPECT_FALSE(parquet_map_type(&c, &t, &scale));
}

TEST(BoltParquetFlbaLogical, RawFlbaSixteenBytesIsAccepted) {
    PqColumn c = make_col(PqType::FixedLenByteArray, 16, -1, PqLogical::None);
    bolt::BoltType t;
    std::uint8_t scale = 0;
    ASSERT_TRUE(parquet_map_type(&c, &t, &scale));
    EXPECT_EQ(t, bolt::BoltType::FixedSizeBinary);
}

TEST(BoltParquetFlbaLogical, ZeroWidthRawFlbaIsRefused) {
    PqColumn c = make_col(PqType::FixedLenByteArray, 0, -1, PqLogical::None);
    bolt::BoltType t;
    std::uint8_t scale = 0;
    EXPECT_FALSE(parquet_map_type(&c, &t, &scale));
}

}  // namespace
