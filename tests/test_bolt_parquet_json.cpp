// JSON / BSON / VARIANT: logical refinements, not new physical types.
//
// A JSON column's bytes ARE a string; what distinguishes it is an annotation
// saying they should be parsed as a document. Arrow models exactly that --
// pyarrow reports such a column as `extension<arrow.json>` over string
// storage -- and parquet does too, annotating a BYTE_ARRAY with a JSON
// logical type. bolt carries it as BoltColumn::logical beside the type.
//
// The alternative, a `BoltType::Json` that is StringView-shaped, would fail
// every one of the ~70 `type == BoltType::Utf8` equality tests across bolt,
// chukonu and marbledb -- silently, because those are equality tests and not
// exhaustive switches. So the property most worth pinning here is the one
// that makes this design safe: a JSON column is still a Utf8 column to
// everything that has not been taught otherwise.

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

constexpr int kN = 300;

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

std::string doc(int i) {
    char b[128];
    std::snprintf(b, sizeof(b), "{\"id\":%d,\"tag\":\"t%d\",\"nested\":{\"v\":%d}}",
                  i, i % 7, i * 3);
    return b;
}

int find_col(const PqMeta* m, const char* name) {
    for (std::uint32_t c = 0; c < m->n_columns; ++c) {
        if (std::strcmp(m->columns[c].name, name) == 0) return static_cast<int>(c);
    }
    return -1;
}

// ---- reading a pyarrow-written JSON column --------------------------------

TEST(BoltParquetJson, ReadsPyarrowJsonAnnotation) {
    const auto buf = slurp(data_path("golden_json.parquet").c_str());
    ASSERT_FALSE(buf.empty()) << "fixture missing";
    bolt::Arena a;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &a, &meta));

    const int j = find_col(&meta, "j");
    const int plain = find_col(&meta, "plain");
    const int n = find_col(&meta, "n");
    ASSERT_GE(j, 0);
    ASSERT_GE(plain, 0);
    ASSERT_GE(n, 0);

    // The annotation is recognised...
    EXPECT_EQ(parquet_map_logical(&meta.columns[j]), bolt::BoltLogical::Json);
    // ...and only where it is present. A plain UTF8 string column must NOT
    // come back as JSON, or the flag means nothing.
    EXPECT_EQ(parquet_map_logical(&meta.columns[plain]), bolt::BoltLogical::None);
    EXPECT_EQ(parquet_map_logical(&meta.columns[n]), bolt::BoltLogical::None);

    // Storage is unchanged: still a Utf8 column.
    bolt::BoltType t;
    std::uint8_t scale = 0;
    ASSERT_TRUE(parquet_map_type(&meta.columns[j], &t, &scale));
    EXPECT_EQ(t, bolt::BoltType::Utf8) << "JSON must keep string STORAGE";
}

TEST(BoltParquetJson, JsonColumnDecodesAsAnOrdinaryUtf8Column) {
    // The safety property this whole design rests on: a consumer that knows
    // nothing about JSON sees a normal string column and reads it correctly.
    const auto buf = slurp(data_path("golden_json.parquet").c_str());
    ASSERT_FALSE(buf.empty());
    bolt::Arena a;
    auto* b = a.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(b, nullptr);
    ASSERT_TRUE(parquet_read_file(buf.data(), buf.size(), &a, b));
    ASSERT_EQ(b->num_rows, kN);

    bolt::Arena ma;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &ma, &meta));
    const int j = find_col(&meta, "j");
    ASSERT_GE(j, 0);

    const bolt::BoltColumn& col = b->columns[b->read_epoch][j];
    EXPECT_EQ(col.type, bolt::BoltType::Utf8);
    EXPECT_EQ(col.logical, bolt::BoltLogical::Json)
        << "the decoded column lost its annotation";
    const auto* sv = static_cast<const bolt::StringView*>(col.data);
    const auto* sp = static_cast<const std::uint8_t*>(col.str_overflow_base);
    ASSERT_NE(sv, nullptr);

    for (int i = 0; i < kN; ++i) {
        const bool want_null = (i % 13) == 0;
        const bool valid = (col.validity == nullptr) ||
            (((col.validity[i >> 3] >> (i & 7)) & 1u) != 0u);
        ASSERT_EQ(valid, !want_null) << "row " << i << " nullness";
        if (want_null) continue;
        const std::string want = doc(i);
        const std::uint32_t len = sv[i].length;
        const std::uint8_t* p =
            (len <= 12u) ? reinterpret_cast<const std::uint8_t*>(&sv[i].prefix[0])
                         : (sp + sv[i].ref.offset);
        ASSERT_EQ(len, want.size()) << "row " << i;
        ASSERT_EQ(0, std::memcmp(p, want.data(), len)) << "row " << i;
    }
}

// ---- writing the annotation -----------------------------------------------

void build_str_batch(bolt::Arena* a, const std::vector<std::string>& vals,
                     bolt::BoltBatch* out) {
    const std::int64_t n = static_cast<std::int64_t>(vals.size());
    bolt::BoltBatch::init_empty(out);
    out->num_cols = 1;
    out->num_rows = n;
    bolt::BoltBatch::alloc_columns(out, a, 1);
    out->schema.add_field("j", bolt::BoltType::Utf8, false);
    bolt::BoltColumn& c = out->columns[out->read_epoch][0];
    c = bolt::BoltColumn::make_empty();
    c.length = n;
    c.format = bolt::ColumnFormat::Flat;
    c.type = bolt::BoltType::Utf8;
    c.logical = bolt::BoltLogical::Json;
    c.type_size_bytes = sizeof(bolt::StringView);
    auto* svs = static_cast<bolt::StringView*>(
        a->allocate(static_cast<std::size_t>(n) * sizeof(bolt::StringView),
                    alignof(bolt::StringView)));
    std::memset(svs, 0, static_cast<std::size_t>(n) * sizeof(bolt::StringView));
    std::size_t need = 0;
    for (const auto& s : vals) if (s.size() > 12u) need += s.size();
    auto* spill = need ? static_cast<std::uint8_t*>(a->allocate(need, 8)) : nullptr;
    std::size_t off = 0;
    for (std::int64_t i = 0; i < n; ++i) {
        const std::string& s = vals[static_cast<std::size_t>(i)];
        svs[i].length = static_cast<std::uint32_t>(s.size());
        if (s.size() <= 12u) {
            std::memcpy(&svs[i].prefix[0], s.data(), s.size());
        } else {
            std::memcpy(&svs[i].prefix[0], s.data(), 4);
            std::memcpy(spill + off, s.data(), s.size());
            svs[i].ref.offset = static_cast<std::uint32_t>(off);
            off += s.size();
        }
    }
    c.data = svs;
    c.str_overflow_base = spill;
    c.stats.all_valid = true;
}

TEST(BoltParquetJson, WritesAnnotationAndRoundTrips) {
    std::vector<std::string> vals;
    for (int i = 0; i < kN; ++i) vals.push_back(doc(i));
    bolt::Arena a;
    auto* b = a.allocate_array<bolt::BoltBatch>(1);
    build_str_batch(&a, vals, b);

    ParquetWriteOpts o{};
    o.n_columns = 1;
    o.compression = 1;
    o.emit_statistics = true;
    std::strncpy(o.columns[0].name, "j", sizeof(o.columns[0].name) - 1);
    o.columns[0].type = bolt::BoltType::Utf8;
    o.columns[0].nullable = false;
    o.columns[0].logical = static_cast<std::uint8_t>(bolt::BoltLogical::Json);

    const char* path = "test_bolt_parquet_json_written.parquet";
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
    EXPECT_EQ(parquet_map_logical(&meta.columns[0]), bolt::BoltLogical::Json)
        << "the annotation did not survive the round trip";

    bolt::Arena ra;
    auto* rb = ra.allocate_array<bolt::BoltBatch>(1);
    ASSERT_TRUE(parquet_read_file(buf.data(), buf.size(), &ra, rb));
    ASSERT_EQ(rb->num_rows, kN);
    const bolt::BoltColumn& col = rb->columns[rb->read_epoch][0];
    EXPECT_EQ(col.logical, bolt::BoltLogical::Json);
    const auto* sv = static_cast<const bolt::StringView*>(col.data);
    const auto* sp = static_cast<const std::uint8_t*>(col.str_overflow_base);
    for (int i = 0; i < kN; ++i) {
        const std::uint32_t len = sv[i].length;
        const std::uint8_t* p =
            (len <= 12u) ? reinterpret_cast<const std::uint8_t*>(&sv[i].prefix[0])
                         : (sp + sv[i].ref.offset);
        ASSERT_EQ(len, vals[i].size()) << "row " << i;
        ASSERT_EQ(0, std::memcmp(p, vals[i].data(), len)) << "row " << i;
    }
}

TEST(BoltParquetJson, UnannotatedStringStaysUnannotated) {
    // Discriminating check for the writer: without the flag, no JSON logical
    // type appears. If it were emitted unconditionally the test above would
    // pass while the option did nothing.
    std::vector<std::string> vals;
    for (int i = 0; i < 50; ++i) vals.push_back(doc(i));
    bolt::Arena a;
    auto* b = a.allocate_array<bolt::BoltBatch>(1);
    build_str_batch(&a, vals, b);
    ParquetWriteOpts o{};
    o.n_columns = 1;
    o.compression = 0;
    std::strncpy(o.columns[0].name, "j", sizeof(o.columns[0].name) - 1);
    o.columns[0].type = bolt::BoltType::Utf8;
    o.columns[0].logical = static_cast<std::uint8_t>(bolt::BoltLogical::None);
    const char* path = "test_bolt_parquet_json_plain.parquet";
    ParquetWriter* w = parquet_write_open(path, &o);
    ASSERT_NE(w, nullptr);
    ASSERT_TRUE(parquet_write_row_group(w, b));
    ASSERT_TRUE(parquet_write_close(w));
    const auto buf = slurp(path);
    ASSERT_FALSE(buf.empty());
    bolt::Arena ma;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &ma, &meta));
    EXPECT_EQ(parquet_map_logical(&meta.columns[0]), bolt::BoltLogical::None);
}

TEST(BoltParquetJson, ImpossibleAnnotationIsRejectedAtOpen) {
    // JSON on an integer column is a caller bug. Writing the file anyway
    // would produce a schema that lies about its own bytes.
    ParquetWriteOpts o{};
    o.n_columns = 1;
    std::strncpy(o.columns[0].name, "n", sizeof(o.columns[0].name) - 1);
    o.columns[0].type = bolt::BoltType::Int64;
    o.columns[0].logical = static_cast<std::uint8_t>(bolt::BoltLogical::Json);
    ParquetWriter* w = parquet_write_open("test_bolt_parquet_json_bad.parquet", &o);
    EXPECT_EQ(w, nullptr);
    if (w != nullptr) parquet_write_close(w);

    // An out-of-range logical byte is refused, not read as an enum.
    ParquetWriteOpts o2{};
    o2.n_columns = 1;
    std::strncpy(o2.columns[0].name, "s", sizeof(o2.columns[0].name) - 1);
    o2.columns[0].type = bolt::BoltType::Utf8;
    o2.columns[0].logical = 99u;
    ParquetWriter* w2 = parquet_write_open("test_bolt_parquet_json_bad2.parquet", &o2);
    EXPECT_EQ(w2, nullptr);
    if (w2 != nullptr) parquet_write_close(w2);
}

// ---- VARIANT --------------------------------------------------------------

TEST(BoltParquetJson, VariantColumnShape) {
    // A VARIANT is a two-field struct of {metadata, value} binaries -- the
    // shape parquet's VARIANT logical type and Spark's variant encoding both
    // use. Checked here as a column-model property; the parquet reader
    // surfaces such a group as its two leaf Binary columns today, which is
    // what those two children are.
    bolt::Arena a;
    const std::int64_t n = 8;
    bolt::BoltColumn md = bolt::BoltColumn::make_flat_alloc(n, bolt::BoltType::Int64, &a);
    bolt::BoltColumn vl = bolt::BoltColumn::make_flat_alloc(n, bolt::BoltType::Int64, &a);
    md.type = bolt::BoltType::Binary;
    vl.type = bolt::BoltType::Binary;

    bolt::BoltColumn v = bolt::BoltColumn::make_variant(&md, &vl, n, nullptr, &a);
    ASSERT_NE(v.data, nullptr);
    EXPECT_TRUE(v.is_nested());
    EXPECT_EQ(v.type, bolt::BoltType::Variant);
    EXPECT_EQ(v.logical, bolt::BoltLogical::Variant);
    EXPECT_EQ(v.child_count(), 2);
    EXPECT_EQ(v.length, n);
    ASSERT_NE(v.variant_metadata(), nullptr);
    ASSERT_NE(v.variant_value(), nullptr);
    EXPECT_EQ(v.variant_metadata()->data, md.data);
    EXPECT_EQ(v.variant_value()->data, vl.data);
    // A variant is not a list: it has one value per row, so no offsets.
    EXPECT_EQ(v.list_offsets(), nullptr);
    EXPECT_EQ(v.list_element(), nullptr);
    // And the accessors refuse a non-variant column rather than reinterpreting
    // whatever is in `data`.
    EXPECT_EQ(md.variant_metadata(), nullptr);
    EXPECT_EQ(md.child_count(), 0);
    EXPECT_EQ(md.child_at(0), nullptr);
}

TEST(BoltParquetJson, NestedStructShape) {
    bolt::Arena a;
    const std::int64_t n = 5;
    bolt::BoltColumn f0 = bolt::BoltColumn::make_flat_alloc(n, bolt::BoltType::Int64, &a);
    bolt::BoltColumn f1 = bolt::BoltColumn::make_flat_alloc(n, bolt::BoltType::Float64, &a);
    bolt::BoltColumn fields[2] = {f0, f1};
    bolt::BoltColumn st = bolt::BoltColumn::make_struct(fields, 2, n, nullptr, &a);
    ASSERT_NE(st.data, nullptr);
    EXPECT_EQ(st.type, bolt::BoltType::Struct);
    EXPECT_EQ(st.child_count(), 2);
    EXPECT_EQ(st.child_at(0)->type, bolt::BoltType::Int64);
    EXPECT_EQ(st.child_at(1)->type, bolt::BoltType::Float64);
    EXPECT_EQ(st.child_at(2), nullptr);       // bounded
    EXPECT_EQ(st.child_at(-1), nullptr);
    EXPECT_EQ(st.list_offsets(), nullptr);    // a struct has no offsets
}

}  // namespace
