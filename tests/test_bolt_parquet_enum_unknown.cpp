// ENUM / UNKNOWN logical types (G2PQ-17, part of the G2PQ-8 reader-
// conformance epic's A5 slice).
//
// ENUM: ConvertedType.ENUM (legacy, field 6 = 4) and/or LogicalType.ENUM
// (modern, union field 4 -- an empty EnumType struct) annotate a BYTE_ARRAY
// column as a small closed set of string values. Physically identical to
// STRING/UTF8 -- the annotation only changes *intent*, not bytes -- so this
// stays Utf8 storage with a BoltLogical::Enum tag, exactly the JSON/BSON
// precedent (see bolt_types.h).
//
// UNKNOWN: LogicalType.UNKNOWN (union field 11 -- an empty NullType struct)
// annotates a column whose physical type was "guessed" because every value
// is null. No compatible ConvertedType exists for it (parquet.thrift says
// so explicitly). Allowed over ANY physical type; this suite exercises it
// over INT32 (what pyarrow's `pa.null()` writes) and Int64 (bolt's own
// writer).
//
// Neither pyarrow nor any other Python parquet library can write ENUM
// (verified: no `pa.enum_()`, and dictionary_encode() writes plain STRING),
// so the ENUM legacy-ConvertedType-only fixture is hand-built from the
// thrift compact protocol spec directly (scripts/make_enum_unknown_fixture.py,
// following the make_page_crc_fixture.py precedent). pyarrow DOES support
// UNKNOWN via `pa.null()`, so that fixture is a real external-writer oracle.
//
// VARIANT (parquet-format 2.11's binary variant encoding) is explicitly OUT
// OF SCOPE for this ticket -- see the G2PQ-17 tracker note.

#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt/ingest/bolt_parquet_write.h"
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

const char* kEnumValues[5] = {"ACTIVE", "INACTIVE", "ACTIVE", "PENDING",
                              "ACTIVE"};

}  // namespace

// ---- ENUM: hand-crafted legacy ConvertedType-only fixture -----------------

TEST(BoltParquetEnumUnknown, ReadsLegacyConvertedEnumWithoutRejection) {
    const auto buf = slurp(data_path("enum_converted_legacy.parquet").c_str());
    ASSERT_FALSE(buf.empty()) << "fixture missing -- run "
                                 "scripts/make_enum_unknown_fixture.py";

    bolt::Arena ma;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &ma, &meta));
    ASSERT_EQ(meta.n_columns, 1u);
    // The legacy shape carries ONLY ConvertedType (field 6 = 4), no
    // LogicalType union at all -- this pins derive_logical_from_converted's
    // kCtEnum fallback specifically.
    EXPECT_EQ(meta.columns[0].converted, 4);
    EXPECT_EQ(meta.columns[0].logical, static_cast<std::int32_t>(PqLogical::Enum));
    EXPECT_EQ(parquet_map_logical(&meta.columns[0]), bolt::BoltLogical::Enum);

    // Schema mapping must succeed unconditionally -- the "minimum bar":
    // an ENUM-annotated BYTE_ARRAY column reads, it is never rejected.
    bolt::BoltType t{};
    std::uint8_t scale = 0;
    EXPECT_TRUE(parquet_map_type(&meta.columns[0], &t, &scale));
    EXPECT_EQ(t, bolt::BoltType::Utf8);

    bolt::Arena a;
    auto* b = a.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(b, nullptr);
    ASSERT_TRUE(parquet_read_file(buf.data(), buf.size(), &a, b))
        << "an ENUM-annotated column must never be refused";
    ASSERT_EQ(b->num_rows, 5);
    const bolt::BoltColumn& col = b->columns[b->read_epoch][0];
    EXPECT_EQ(col.type, bolt::BoltType::Utf8);
    EXPECT_EQ(col.logical, bolt::BoltLogical::Enum)
        << "the ENUM annotation did not survive decode";
    const auto* sv = static_cast<const bolt::StringView*>(col.data);
    ASSERT_NE(sv, nullptr);
    for (int i = 0; i < 5; ++i) {
        const std::string want = kEnumValues[i];
        const std::uint32_t len = sv[i].length;
        ASSERT_EQ(len, want.size()) << "row " << i;
        ASSERT_EQ(0, std::memcmp(&sv[i].prefix[0], want.data(),
                                 (len <= 12u) ? len : 4))
            << "row " << i;
    }
}

// ---- UNKNOWN: pyarrow-written external-oracle fixture ---------------------

TEST(BoltParquetEnumUnknown, ReadsPyarrowUnknownAnnotationAllNull) {
    const auto buf = slurp(data_path("unknown_null_pyarrow.parquet").c_str());
    ASSERT_FALSE(buf.empty()) << "fixture missing -- run "
                                 "scripts/make_enum_unknown_fixture.py";

    bolt::Arena ma;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &ma, &meta));
    const int u = find_col(&meta, "u");
    const int n = find_col(&meta, "n");
    ASSERT_GE(u, 0);
    ASSERT_GE(n, 0);
    EXPECT_EQ(meta.columns[u].physical, PqType::Int32);
    EXPECT_EQ(meta.columns[u].logical, static_cast<std::int32_t>(PqLogical::Unknown));
    EXPECT_EQ(parquet_map_logical(&meta.columns[u]), bolt::BoltLogical::Unknown);
    // The companion "n" column carries no annotation at all.
    EXPECT_EQ(meta.columns[n].logical, static_cast<std::int32_t>(PqLogical::None));

    bolt::BoltType t{};
    std::uint8_t scale = 0;
    EXPECT_TRUE(parquet_map_type(&meta.columns[u], &t, &scale))
        << "an UNKNOWN-annotated column must never be refused";
    EXPECT_EQ(t, bolt::BoltType::Int32);

    bolt::Arena a;
    auto* b = a.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(b, nullptr);
    ASSERT_TRUE(parquet_read_file(buf.data(), buf.size(), &a, b));
    ASSERT_EQ(b->num_rows, 7);
    const bolt::BoltColumn& uc = b->columns[b->read_epoch][u];
    EXPECT_EQ(uc.logical, bolt::BoltLogical::Unknown);
    // Every value must be reported null -- either a fully-null validity
    // bitmap, or (dense all-null representation) no data at all.
    for (int i = 0; i < 7; ++i) {
        const bool valid = (uc.validity != nullptr) &&
            (((uc.validity[i >> 3] >> (i & 7)) & 1u) != 0u);
        EXPECT_FALSE(valid) << "row " << i << " of an UNKNOWN column must be null";
    }
    const bolt::BoltColumn& nc = b->columns[b->read_epoch][n];
    const auto* nv = static_cast<const std::int64_t*>(nc.data);
    ASSERT_NE(nv, nullptr);
    for (int i = 0; i < 7; ++i) EXPECT_EQ(nv[i], i) << "companion column row " << i;
}

// ---- ENUM: bolt's own writer round-trips the annotation --------------------

void build_enum_batch(bolt::Arena* a, const std::vector<std::string>& vals,
                      bolt::BoltBatch* out) {
    const std::int64_t n = static_cast<std::int64_t>(vals.size());
    bolt::BoltBatch::init_empty(out);
    out->num_cols = 1;
    out->num_rows = n;
    bolt::BoltBatch::alloc_columns(out, a, 1);
    out->schema.add_field("status", bolt::BoltType::Utf8, false);
    bolt::BoltColumn& c = out->columns[out->read_epoch][0];
    c = bolt::BoltColumn::make_empty();
    c.length = n;
    c.format = bolt::ColumnFormat::Flat;
    c.type = bolt::BoltType::Utf8;
    c.logical = bolt::BoltLogical::Enum;
    c.type_size_bytes = sizeof(bolt::StringView);
    auto* svs = static_cast<bolt::StringView*>(
        a->allocate(static_cast<std::size_t>(n) * sizeof(bolt::StringView),
                    alignof(bolt::StringView)));
    std::memset(svs, 0, static_cast<std::size_t>(n) * sizeof(bolt::StringView));
    for (std::int64_t i = 0; i < n; ++i) {
        const std::string& s = vals[static_cast<std::size_t>(i)];
        ASSERT_LE(s.size(), 12u) << "test fixture keeps values inline";
        svs[i].length = static_cast<std::uint32_t>(s.size());
        std::memcpy(&svs[i].prefix[0], s.data(), s.size());
    }
    c.data = svs;
    c.stats.all_valid = true;
}

TEST(BoltParquetEnumUnknown, WritesEnumAnnotationAndRoundTrips) {
    std::vector<std::string> vals(kEnumValues, kEnumValues + 5);
    bolt::Arena a;
    auto* b = a.allocate_array<bolt::BoltBatch>(1);
    build_enum_batch(&a, vals, b);

    ParquetWriteOpts o{};
    o.n_columns = 1;
    o.compression = 0;
    std::strncpy(o.columns[0].name, "status", sizeof(o.columns[0].name) - 1);
    o.columns[0].type = bolt::BoltType::Utf8;
    o.columns[0].nullable = false;
    o.columns[0].logical = static_cast<std::uint8_t>(bolt::BoltLogical::Enum);

    const char* path = "test_bolt_parquet_enum_written.parquet";
    ParquetWriter* w = parquet_write_open(path, &o);
    ASSERT_NE(w, nullptr);
    ASSERT_TRUE(parquet_write_row_group(w, b));
    ASSERT_TRUE(parquet_write_close(w));

    const auto buf = slurp(path);
    ASSERT_FALSE(buf.empty());
    bolt::Arena ma;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &ma, &meta));
    ASSERT_EQ(meta.n_columns, 1u);
    // Both fields must round-trip: field 6 (ConvertedType) for old readers,
    // field 10 (LogicalType) for modern ones -- the spec's own compat rule.
    EXPECT_EQ(meta.columns[0].converted, 4);
    EXPECT_EQ(parquet_map_logical(&meta.columns[0]), bolt::BoltLogical::Enum)
        << "the annotation did not survive the round trip";

    bolt::Arena ra;
    auto* rb = ra.allocate_array<bolt::BoltBatch>(1);
    ASSERT_TRUE(parquet_read_file(buf.data(), buf.size(), &ra, rb));
    ASSERT_EQ(rb->num_rows, 5);
    const bolt::BoltColumn& col = rb->columns[rb->read_epoch][0];
    EXPECT_EQ(col.logical, bolt::BoltLogical::Enum);
    const auto* sv = static_cast<const bolt::StringView*>(col.data);
    ASSERT_NE(sv, nullptr);
    for (int i = 0; i < 5; ++i) {
        const std::string& want = vals[static_cast<std::size_t>(i)];
        ASSERT_EQ(sv[i].length, want.size()) << "row " << i;
        ASSERT_EQ(0, std::memcmp(&sv[i].prefix[0], want.data(), want.size()))
            << "row " << i;
    }
}

// ---- UNKNOWN: bolt's own writer round-trips an all-null column ------------

TEST(BoltParquetEnumUnknown, WritesUnknownAnnotationAndRoundTrips) {
    constexpr std::int64_t kN = 6;
    bolt::Arena a;
    auto* b = a.allocate_array<bolt::BoltBatch>(1);
    bolt::BoltBatch::init_empty(b);
    b->num_cols = 1;
    b->num_rows = kN;
    bolt::BoltBatch::alloc_columns(b, &a, 1);
    b->schema.add_field("u", bolt::BoltType::Int64, true);
    bolt::BoltColumn& c = b->columns[b->read_epoch][0];
    c = bolt::BoltColumn::make_flat_alloc(kN, bolt::BoltType::Int64, &a);
    ASSERT_NE(c.data, nullptr);
    c.logical = bolt::BoltLogical::Unknown;
    const std::size_t nb = static_cast<std::size_t>((kN + 7) / 8);
    auto* bm = static_cast<std::uint8_t*>(a.allocate(nb, 8));
    std::memset(bm, 0, nb);       // every bit 0 -- every row null
    c.validity = bm;
    c.validity_offset = 0;
    c.stats.all_valid = false;

    ParquetWriteOpts o{};
    o.n_columns = 1;
    o.compression = 0;
    std::strncpy(o.columns[0].name, "u", sizeof(o.columns[0].name) - 1);
    o.columns[0].type = bolt::BoltType::Int64;
    o.columns[0].nullable = true;
    o.columns[0].logical = static_cast<std::uint8_t>(bolt::BoltLogical::Unknown);

    const char* path = "test_bolt_parquet_unknown_written.parquet";
    ParquetWriter* w = parquet_write_open(path, &o);
    ASSERT_NE(w, nullptr);
    ASSERT_TRUE(parquet_write_row_group(w, b));
    ASSERT_TRUE(parquet_write_close(w));

    const auto buf = slurp(path);
    ASSERT_FALSE(buf.empty());
    bolt::Arena ma;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &ma, &meta));
    ASSERT_EQ(meta.n_columns, 1u);
    // UNKNOWN has no compatible ConvertedType (parquet.thrift), so field 6
    // must stay unset -- only the LogicalType union carries it.
    EXPECT_EQ(meta.columns[0].converted, -1);
    EXPECT_EQ(parquet_map_logical(&meta.columns[0]), bolt::BoltLogical::Unknown)
        << "the annotation did not survive the round trip";

    bolt::Arena ra;
    auto* rb = ra.allocate_array<bolt::BoltBatch>(1);
    ASSERT_TRUE(parquet_read_file(buf.data(), buf.size(), &ra, rb));
    ASSERT_EQ(rb->num_rows, kN);
    const bolt::BoltColumn& col = rb->columns[rb->read_epoch][0];
    EXPECT_EQ(col.logical, bolt::BoltLogical::Unknown);
    for (std::int64_t i = 0; i < kN; ++i) {
        const bool valid = (col.validity != nullptr) &&
            (((col.validity[i >> 3] >> (i & 7)) & 1u) != 0u);
        EXPECT_FALSE(valid) << "row " << i;
    }
}

// ---- writer-side guards -----------------------------------------------

TEST(BoltParquetEnumUnknown, ImpossibleAnnotationsRejectedAtOpen) {
    // ENUM on a non-Utf8 column is a caller bug (same rule as JSON/String).
    {
        ParquetWriteOpts o{};
        o.n_columns = 1;
        std::strncpy(o.columns[0].name, "n", sizeof(o.columns[0].name) - 1);
        o.columns[0].type = bolt::BoltType::Int64;
        o.columns[0].logical = static_cast<std::uint8_t>(bolt::BoltLogical::Enum);
        ParquetWriter* w =
            parquet_write_open("test_bolt_parquet_enum_bad.parquet", &o);
        EXPECT_EQ(w, nullptr);
        if (w != nullptr) parquet_write_close(w);
    }
    // UNKNOWN on a non-nullable column can never hold "every value null".
    {
        ParquetWriteOpts o{};
        o.n_columns = 1;
        std::strncpy(o.columns[0].name, "u", sizeof(o.columns[0].name) - 1);
        o.columns[0].type = bolt::BoltType::Int64;
        o.columns[0].nullable = false;
        o.columns[0].logical = static_cast<std::uint8_t>(bolt::BoltLogical::Unknown);
        ParquetWriter* w =
            parquet_write_open("test_bolt_parquet_unknown_bad.parquet", &o);
        EXPECT_EQ(w, nullptr);
        if (w != nullptr) parquet_write_close(w);
    }
    // The ceiling raised to admit Enum(5)/Unknown(6) must still reject
    // anything past it -- a regression guard for the raised bound itself.
    {
        ParquetWriteOpts o{};
        o.n_columns = 1;
        std::strncpy(o.columns[0].name, "s", sizeof(o.columns[0].name) - 1);
        o.columns[0].type = bolt::BoltType::Utf8;
        o.columns[0].logical = 7u;
        ParquetWriter* w =
            parquet_write_open("test_bolt_parquet_logical_oob.parquet", &o);
        EXPECT_EQ(w, nullptr);
        if (w != nullptr) parquet_write_close(w);
    }
    // Positive control: Unknown (6, the new ceiling) on a nullable column IS
    // accepted -- proves the raise didn't over-reject.
    {
        ParquetWriteOpts o{};
        o.n_columns = 1;
        std::strncpy(o.columns[0].name, "u2", sizeof(o.columns[0].name) - 1);
        o.columns[0].type = bolt::BoltType::Int64;
        o.columns[0].nullable = true;
        o.columns[0].logical = static_cast<std::uint8_t>(bolt::BoltLogical::Unknown);
        ParquetWriter* w =
            parquet_write_open("test_bolt_parquet_unknown_ok.parquet", &o);
        EXPECT_NE(w, nullptr);
        if (w != nullptr) parquet_write_close(w);
    }
}
