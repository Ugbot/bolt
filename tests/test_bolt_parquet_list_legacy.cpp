// Legacy 2-level LIST shapes (G2PQ-14).
//
// LogicalTypes.md's backward-compatibility rules: a `repeated` field NOT
// wrapped in the modern 3-level `group (LIST) { repeated group list { … } }`
// convention is still a list. Hive, Impala, pre-1.0 parquet-mr and
// Thrift-derived writers all emit these shapes. Misreading one is NOT a clean
// refusal -- the file opens and the nesting is silently misassigned -- so
// every case below asserts VALUES, and the fixtures were certified by pyarrow
// (scripts/make_legacy_list_fixtures.py, which injection-tests its own
// oracle) before bolt was ever pointed at them.
//
// The four spec disambiguation cases:
//   1. a repeated field with no LIST annotation is a list of its own type
//      (legacy2_bare.parquet: `repeated int64 nums` at the root);
//   2. inside a LIST group, a repeated LEAF is the element
//      (legacy2_annotated.parquet: `repeated int32 element`);
//   3. a repeated GROUP with more than one field IS the element
//      (legacy2_multifield.parquet: element = struct<a,b>);
//   4. a single-field repeated group named `array` (or `<name>_tuple`) IS
//      the element (legacy2_array.parquet: element = struct<x>).
//
// Cases 3/4 read per-leaf here (bolt's column model), so what the reader must
// get right is the LEVEL interpretation: list_def/rep_def anchored at the
// repeated node, elements never conflated with the 3-level middle. Injecting
// the unconditional 3-level assumption into the schema walk (deriving
// list_def/rep_def only from repeated ANCESTOR groups, ignoring a repeated
// leaf) fails the bare and annotated cases -- proven during development by
// reverting the `se.repeated` special case in bolt_parquet_meta.cpp.

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

int find_col(const PqMeta* m, const char* path) {
    for (std::uint32_t c = 0; c < m->n_columns; ++c) {
        if (std::strcmp(m->columns[c].name, path) == 0) return static_cast<int>(c);
    }
    return -1;
}

// One row of the shared model: valid=false is a NULL list; empty vector with
// valid=true is an EMPTY list. The two are different values.
struct Row {
    bool valid;
    std::vector<std::int64_t> elems;
};

// Read leaf `path` as a list column (single row group in every fixture) and
// assert it equals `want` value-for-value, including the null/empty split.
void check_list(const char* fixture, const char* path,
                const std::vector<Row>& want) {
    SCOPED_TRACE(testing::Message() << fixture << " :: " << path);
    const auto buf = slurp(data_path(fixture).c_str());
    ASSERT_FALSE(buf.empty()) << "fixture missing -- run "
                                 "scripts/make_legacy_list_fixtures.py";
    bolt::Arena a;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &a, &meta));
    ASSERT_EQ(meta.n_row_groups, 1u);
    const int ci = find_col(&meta, path);
    ASSERT_GE(ci, 0) << "leaf not found in schema walk";

    bolt::Arena ga;
    bolt::BoltColumn col{};
    std::int64_t rows = 0;
    ASSERT_TRUE(parquet_read_list_column(buf.data(), buf.size(), &meta, 0,
                                         static_cast<std::uint16_t>(ci), &ga,
                                         &col, &rows));
    ASSERT_TRUE(col.is_nested());
    ASSERT_EQ(col.type, bolt::BoltType::List);
    ASSERT_EQ(rows, static_cast<std::int64_t>(want.size()));
    const std::int32_t* offs = col.list_offsets();
    const bolt::BoltColumn* elem = col.list_element();
    ASSERT_NE(offs, nullptr);
    ASSERT_NE(elem, nullptr);
    ASSERT_EQ(offs[0], 0);
    // Element lane follows parquet_map_type: INT32 -> Int32, INT64 -> Int64.
    ASSERT_TRUE(elem->type == bolt::BoltType::Int32 ||
                elem->type == bolt::BoltType::Int64)
        << static_cast<int>(elem->type);
    ASSERT_NE(elem->data, nullptr);
    auto elem_at = [&](std::int64_t i) -> std::int64_t {
        if (elem->type == bolt::BoltType::Int32) {
            return static_cast<const std::int32_t*>(elem->data)[i];
        }
        return static_cast<const std::int64_t*>(elem->data)[i];
    };

    for (std::size_t r = 0; r < want.size(); ++r) {
        SCOPED_TRACE(testing::Message() << "row " << r);
        const bool valid =
            (col.validity == nullptr) ||
            (((col.validity[r >> 3] >> (r & 7)) & 1u) != 0u);
        if (!want[r].valid) {
            EXPECT_FALSE(valid) << "a NULL list decoded as present";
            EXPECT_EQ(offs[r + 1] - offs[r], 0);
            continue;
        }
        ASSERT_TRUE(valid) << "a present list decoded as NULL";
        ASSERT_EQ(offs[r + 1] - offs[r],
                  static_cast<std::int32_t>(want[r].elems.size()))
            << "list length";
        for (std::size_t j = 0; j < want[r].elems.size(); ++j) {
            EXPECT_EQ(elem_at(offs[r] + static_cast<std::int64_t>(j)),
                      want[r].elems[j]) << "element " << j;
        }
    }
}

// The shared 6-row model of the annotated / equiv3 fixtures.
std::vector<Row> annotated_rows() {
    return {{true, {10, 11, 12}},
            {false, {}},
            {true, {}},
            {true, {13}},
            {true, {14, 15}},
            {false, {}}};
}

// ---- case 1: bare repeated leaf, no LIST annotation -----------------------

TEST(BoltParquetListLegacy, BareRepeatedFieldIsAList) {
    // Schema-walk levels first: everything downstream keys off these. A bare
    // repeated leaf IS the repeated node, so the list anchors at the root.
    const auto buf = slurp(data_path("legacy2_bare.parquet").c_str());
    ASSERT_FALSE(buf.empty());
    bolt::Arena a;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &a, &meta));
    const int ni = find_col(&meta, "nums");
    ASSERT_GE(ni, 0);
    EXPECT_EQ(meta.columns[ni].max_rep, 1u);
    EXPECT_EQ(meta.columns[ni].max_def, 1u);
    EXPECT_EQ(meta.columns[ni].list_def, 0u);  // cannot be NULL, only empty
    EXPECT_EQ(meta.columns[ni].rep_def, 1u);

    check_list("legacy2_bare.parquet", "nums",
               {{true, {1, 2, 3}},
                {true, {}},
                {true, {7}},
                {true, {8, 9}},
                {true, {}},
                {true, {42}}});
}

TEST(BoltParquetListLegacy, BareRepeatedFileScalarSiblingStillReads) {
    // The scalar column AFTER the repeated leaf must stay chunk-aligned.
    const auto buf = slurp(data_path("legacy2_bare.parquet").c_str());
    ASSERT_FALSE(buf.empty());
    bolt::Arena a;
    bolt::BoltBatch* batch = a.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(batch, nullptr);
    ASSERT_TRUE(parquet_read_file(buf.data(), buf.size(), &a, batch));
    int id = -1;
    for (std::uint16_t c = 0; c < batch->schema.num_fields; ++c) {
        if (std::strcmp(batch->schema.fields[c].name, "id") == 0) id = c;
    }
    ASSERT_GE(id, 0);
    const bolt::BoltColumn* col = &batch->columns[batch->read_epoch][id];
    const auto* p = static_cast<const std::int64_t*>(col->data);
    ASSERT_NE(p, nullptr);
    ASSERT_EQ(col->length, 6);
    for (int i = 0; i < 6; ++i) EXPECT_EQ(p[i], 100 + i);
}

// ---- case 2: 2-level repeated leaf inside a LIST group --------------------

TEST(BoltParquetListLegacy, TwoLevelAnnotatedList) {
    const auto buf = slurp(data_path("legacy2_annotated.parquet").c_str());
    ASSERT_FALSE(buf.empty());
    bolt::Arena a;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &a, &meta));
    const int li = find_col(&meta, "l.element");
    ASSERT_GE(li, 0);
    EXPECT_EQ(meta.columns[li].max_rep, 1u);
    EXPECT_EQ(meta.columns[li].max_def, 2u);
    EXPECT_EQ(meta.columns[li].list_def, 1u);
    EXPECT_EQ(meta.columns[li].rep_def, 2u);

    check_list("legacy2_annotated.parquet", "l.element", annotated_rows());
}

TEST(BoltParquetListLegacy, TwoLevelMatchesEquivalentThreeLevel) {
    // The tracker's acceptance bar verbatim: the legacy file reads with the
    // SAME values as the pyarrow-written 3-level file of the same rows.
    check_list("legacy2_equiv3.parquet", "l.list.element", annotated_rows());
    check_list("legacy2_annotated.parquet", "l.element", annotated_rows());
}

// ---- case 3: multi-field repeated group IS the element --------------------

TEST(BoltParquetListLegacy, MultiFieldRepeatedGroupIsTheElement) {
    // list<struct<a,b>>: both leaves share offsets and list validity; the
    // values interleave per row exactly as written.
    check_list("legacy2_multifield.parquet", "l.element.a",
               {{true, {1, 3}},
                {false, {}},
                {true, {}},
                {true, {5}},
                {true, {7, 9}},
                {false, {}}});
    check_list("legacy2_multifield.parquet", "l.element.b",
               {{true, {2, 4}},
                {false, {}},
                {true, {}},
                {true, {6}},
                {true, {8, 10}},
                {false, {}}});
}

// ---- case 4: single-field repeated group named `array` --------------------

TEST(BoltParquetListLegacy, ArrayNamedRepeatedGroupIsTheElement) {
    // `array` naming: the group is the element (struct<x>), not a 3-level
    // middle. Per-leaf the level arithmetic must still place x's values in
    // the right rows with the null/empty split intact.
    check_list("legacy2_array.parquet", "l.array.x",
               {{true, {20, 21}},
                {false, {}},
                {true, {}},
                {true, {22}},
                {true, {23, 24, 25}},
                {false, {}}});
}

}  // namespace
