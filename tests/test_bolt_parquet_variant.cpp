// parquet_read_variant_column (G2ICE-310): a shredded VARIANT group read back as {metadata, value} in the variant encoding.
//
// The file is written by bolt's nested writer with the shape Iceberg-Java's shredding writer emits:
//   optional group v (VARIANT) { required binary metadata; optional binary value;
//                                optional group typed_value { required group a { optional binary value; optional int64 typed_value; }
//                                                             required group b { optional binary value; optional binary (STRING) typed_value; } } }
// and every rebuilt value is compared byte for byte with the encoding VariantEncoding.md prescribes. The Spark 4 / Iceberg-Java oracle
// is Gestalt2's scripts/verify_g2ice310_shredded.py.

#include "bolt/ingest/bolt_parquet_variant.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/ingest/bolt_parquet_meta.h"
#include "bolt/ingest/bolt_parquet_nested.h"
#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt/ingest/bolt_parquet_write.h"

namespace {

using namespace bolt::ingest::parquet;
using bolt::BoltColumn;
using bolt::BoltType;
using Bytes = std::vector<std::uint8_t>;

constexpr int kRows = 4;

std::uint8_t* bits(bolt::Arena* a, const std::vector<bool>& valid) {
    auto* bm = static_cast<std::uint8_t*>(a->allocate_zeroed(valid.size() / 8 + 1, 1));
    for (std::size_t r = 0; r < valid.size(); ++r) if (valid[r]) bm[r >> 3] |= static_cast<std::uint8_t>(1u << (r & 7));
    return bm;
}

// A VarBinary leaf (std::nullopt-like: an empty optional is NULL).
BoltColumn bytes_col(bolt::Arena* a, BoltType t, const std::vector<std::pair<bool, Bytes>>& vals) {
    std::size_t total = 1;
    for (const auto& v : vals) total += v.second.size();
    auto* pool = static_cast<std::uint8_t*>(a->allocate_zeroed(total, 1));
    auto* offs = a->allocate_array<std::int32_t>(vals.size() + 1);
    std::int32_t at = 0;
    std::vector<bool> valid;
    for (std::size_t r = 0; r < vals.size(); ++r) {
        offs[r] = at;
        if (!vals[r].second.empty()) std::memcpy(pool + at, vals[r].second.data(), vals[r].second.size());
        at += static_cast<std::int32_t>(vals[r].second.size());
        valid.push_back(vals[r].first);
    }
    offs[vals.size()] = at;
    auto* oc = a->allocate_array<BoltColumn>(1);
    *oc = BoltColumn::make_flat(offs, nullptr, static_cast<std::int64_t>(vals.size()) + 1, BoltType::Int32);
    BoltColumn c = BoltColumn::make_empty();
    c.format = bolt::ColumnFormat::VarBinary;
    c.type = t;
    c.data = pool;
    c.dict_child = oc;
    c.length = static_cast<std::int64_t>(vals.size());
    c.validity = bits(a, valid);
    return c;
}

BoltColumn i64_col(bolt::Arena* a, const std::vector<std::pair<bool, std::int64_t>>& vals) {
    BoltColumn c = BoltColumn::make_flat_alloc(static_cast<std::int64_t>(vals.size()), BoltType::Int64, a);
    std::vector<bool> valid;
    for (std::size_t r = 0; r < vals.size(); ++r) {
        std::memcpy(static_cast<std::uint8_t*>(c.data) + r * 8, &vals[r].second, 8);
        valid.push_back(vals[r].first);
    }
    c.validity = bits(a, valid);
    return c;
}

Bytes str(const std::string& s) { return Bytes(s.begin(), s.end()); }
Bytes dict(const std::string& keys) {   // version 1, offset size 1, one-character keys
    Bytes d{0x01, static_cast<std::uint8_t>(keys.size())};
    for (std::size_t i = 0; i <= keys.size(); ++i) d.push_back(static_cast<std::uint8_t>(i));
    for (char k : keys) d.push_back(static_cast<std::uint8_t>(k));
    return d;
}

ParquetNestedNode node(const char* name, BoltType t, bool nullable, std::uint16_t first, std::uint16_t n) {
    ParquetNestedNode x{};
    std::snprintf(x.name, sizeof(x.name), "%s", name);
    x.type = t;
    x.nullable = nullable;
    x.first_child = first;
    x.n_children = n;
    return x;
}

// Rows: 0 fully shredded {a: 7, b: "hi"}; 1 partially shredded {a: "x" (in a.value), c: true (residual)}, b missing;
// 2 not shredded (value = int8 5); 3 NULL.
Bytes write_file(bolt::Arena* a) {
    std::vector<ParquetNestedNode> t;
    t.push_back(node("v", BoltType::Struct, true, 1, 3));               // 0
    t[0].logical = static_cast<std::uint8_t>(bolt::BoltLogical::Variant);
    t[0].has_field_id = true;
    t[0].field_id = 2;
    t.push_back(node("metadata", BoltType::Binary, false, 0, 0));       // 1
    t.push_back(node("value", BoltType::Binary, true, 0, 0));           // 2
    t.push_back(node("typed_value", BoltType::Struct, true, 4, 2));     // 3
    t.push_back(node("a", BoltType::Struct, false, 6, 2));              // 4
    t.push_back(node("b", BoltType::Struct, false, 8, 2));              // 5
    t.push_back(node("value", BoltType::Binary, true, 0, 0));           // 6
    t.push_back(node("typed_value", BoltType::Int64, true, 0, 0));      // 7
    t.push_back(node("value", BoltType::Binary, true, 0, 0));           // 8
    t.push_back(node("typed_value", BoltType::Utf8, true, 0, 0));       // 9
    const Bytes residual{0x02, 0x01, 0x02, 0x00, 0x01, 0x04};           // object {c (id 2): true}
    BoltColumn md = bytes_col(a, BoltType::Binary, {{true, dict("ab")}, {true, dict("abc")}, {true, dict("")}, {true, dict("")}});
    BoltColumn val = bytes_col(a, BoltType::Binary, {{false, {}}, {true, residual}, {true, {0x0C, 0x05}}, {false, {}}});
    BoltColumn av = bytes_col(a, BoltType::Binary, {{false, {}}, {true, {0x05, 'x'}}, {false, {}}, {false, {}}});
    BoltColumn at = i64_col(a, {{true, 7}, {false, 0}, {false, 0}, {false, 0}});
    BoltColumn bv = bytes_col(a, BoltType::Binary, {{false, {}}, {false, {}}, {false, {}}, {false, {}}});
    BoltColumn bt = bytes_col(a, BoltType::Utf8, {{true, str("hi")}, {false, {}}, {false, {}}, {false, {}}});
    BoltColumn ak[2] = {av, at}, bk[2] = {bv, bt};
    BoltColumn fa = BoltColumn::make_struct(ak, 2, kRows, nullptr, a), fb = BoltColumn::make_struct(bk, 2, kRows, nullptr, a);
    BoltColumn tk[2] = {fa, fb};
    BoltColumn typed = BoltColumn::make_struct(tk, 2, kRows, bits(a, {true, true, false, false}), a);
    BoltColumn gk[3] = {md, val, typed};
    BoltColumn g = BoltColumn::make_struct(gk, 3, kRows, bits(a, {true, true, true, false}), a);
    ParquetWriteOpts o{};
    o.n_columns = 1;
    o.columns[0].type = BoltType::Struct;
    std::snprintf(o.columns[0].name, sizeof(o.columns[0].name), "v");
    o.columns[0].nullable = true;
    o.columns[0].nested = t.data();
    o.columns[0].n_nested = static_cast<std::uint32_t>(t.size());
    bolt::BoltBatch b;
    bolt::BoltBatch::init_empty(&b);
    b.arena = a;
    EXPECT_TRUE(bolt::BoltBatch::alloc_columns(&b, a, 1));
    b.schema.add_field("v", BoltType::Struct, true);
    b.columns[0][0] = b.columns[1][0] = g;
    b.num_rows = kRows;
    ParquetWriter* w = parquet_write_open_mem(&o);
    EXPECT_NE(w, nullptr);
    if (w == nullptr) return {};
    EXPECT_TRUE(parquet_write_row_group(w, &b));
    const std::uint8_t* out = nullptr;
    std::uint64_t n = 0;
    EXPECT_TRUE(parquet_write_close_mem(w, a, &out, &n));
    return Bytes(out, out + n);
}

Bytes value_at(const BoltColumn& c, std::int64_t r) {
    const std::uint8_t* p = nullptr;
    std::int32_t n = 0;
    c.utf8_at_or_empty(r, &p, &n);
    return Bytes(p, p + (n > 0 ? n : 0));
}

TEST(BoltParquetVariant, ShreddedGroupRebuildsTheVariantEncoding) {
    bolt::Arena a;
    const Bytes file = write_file(&a);
    ASSERT_FALSE(file.empty());
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(file.data(), file.size(), &a, &meta));
    PqSchemaTree tree{};
    ASSERT_TRUE(parquet_read_schema_tree(file.data(), file.size(), &a, &tree));
    ASSERT_TRUE(pq_variant_group_ok(&tree, tree.top[0]));
    ASSERT_TRUE(pq_variant_group_shredded(&tree, tree.top[0]));
    BoltColumn out;
    std::int64_t rows = 0;
    ASSERT_TRUE(parquet_read_variant_column(file.data(), file.size(), &meta, &tree, 0, 0, &a, &out, &rows));
    ASSERT_EQ(rows, kRows);
    ASSERT_TRUE(out.is_variant());
    const BoltColumn& v = *out.variant_value();
    // {a: int64 7, b: "hi"}: ids 0, 1 in key order, offsets 0 / 9 / 12.
    const Bytes r0{0x02, 0x02, 0x00, 0x01, 0x00, 0x09, 0x0C, 0x18, 7, 0, 0, 0, 0, 0, 0, 0, 0x09, 'h', 'i'};
    EXPECT_EQ(value_at(v, 0), r0);
    // {a: "x", c: true}: a from its field's value, c from the residual object.
    const Bytes r1{0x02, 0x02, 0x00, 0x02, 0x00, 0x02, 0x03, 0x05, 'x', 0x04};
    EXPECT_EQ(value_at(v, 1), r1);
    EXPECT_EQ(value_at(v, 2), (Bytes{0x0C, 0x05}));
    EXPECT_TRUE(out.is_null(3));
    EXPECT_EQ(value_at(*out.variant_metadata(), 1), dict("abc"));
}

TEST(BoltParquetVariant, UnshreddedGroupIsTakenAsIs) {
    bolt::Arena a;
    std::vector<ParquetNestedNode> t;
    t.push_back(node("v", BoltType::Struct, true, 1, 2));
    t[0].logical = static_cast<std::uint8_t>(bolt::BoltLogical::Variant);
    t.push_back(node("metadata", BoltType::Binary, false, 0, 0));
    t.push_back(node("value", BoltType::Binary, false, 0, 0));
    BoltColumn md = bytes_col(&a, BoltType::Binary, {{true, dict("")}, {true, dict("")}});
    BoltColumn val = bytes_col(&a, BoltType::Binary, {{true, {0x0C, 0x05}}, {true, {0x00}}});
    BoltColumn gk[2] = {md, val};
    BoltColumn g = BoltColumn::make_struct(gk, 2, 2, nullptr, &a);
    ParquetWriteOpts o{};
    o.n_columns = 1;
    o.columns[0].type = BoltType::Struct;
    std::snprintf(o.columns[0].name, sizeof(o.columns[0].name), "v");
    o.columns[0].nullable = true;
    o.columns[0].nested = t.data();
    o.columns[0].n_nested = static_cast<std::uint32_t>(t.size());
    bolt::BoltBatch b;
    bolt::BoltBatch::init_empty(&b);
    b.arena = &a;
    ASSERT_TRUE(bolt::BoltBatch::alloc_columns(&b, &a, 1));
    b.schema.add_field("v", BoltType::Struct, true);
    b.columns[0][0] = b.columns[1][0] = g;
    b.num_rows = 2;
    ParquetWriter* w = parquet_write_open_mem(&o);
    ASSERT_NE(w, nullptr);
    ASSERT_TRUE(parquet_write_row_group(w, &b));
    const std::uint8_t* out = nullptr;
    std::uint64_t n = 0;
    ASSERT_TRUE(parquet_write_close_mem(w, &a, &out, &n));
    PqMeta meta{};
    PqSchemaTree tree{};
    ASSERT_TRUE(parquet_read_meta(out, n, &a, &meta));
    ASSERT_TRUE(parquet_read_schema_tree(out, n, &a, &tree));
    EXPECT_FALSE(pq_variant_group_shredded(&tree, tree.top[0]));
    BoltColumn col;
    std::int64_t rows = 0;
    ASSERT_TRUE(parquet_read_variant_column(out, n, &meta, &tree, 0, 0, &a, &col, &rows));
    EXPECT_EQ(value_at(*col.variant_value(), 0), (Bytes{0x0C, 0x05}));
    EXPECT_EQ(value_at(*col.variant_value(), 1), (Bytes{0x00}));
}

}  // namespace
