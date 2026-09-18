// G2PQ-15: nested repetition, max_rep >= 2 -- list<list<T>>, a MAP whose
// VALUE is itself a LIST, and three levels of nesting (list<list<list<T>>>).
//
// test_bolt_parquet_list.cpp already covers max_rep == 1 (a single repeated
// ancestor) exhaustively; this file is specifically the case that used to be
// refused: MORE than one repeated ancestor between the schema root and the
// leaf, which needs one offset array per nesting level instead of one.
//
// The fixtures are written by PYARROW, not by bolt (scripts/
// make_list_nested_fixtures.py), and the expected values are re-derived here
// from the SAME closed form the generator used -- so this compares bolt
// against the ecosystem's reading of the spec, not against itself. Assert
// VALUES, not row counts: a level bug at an intermediate nesting level
// preserves row counts and total element counts while moving values between
// rows, exactly the failure class this ticket exists to catch.

#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt/ingest/bolt_parquet_meta.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"

namespace {

using namespace bolt::ingest::parquet;

constexpr int kN = 500;

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

// ---- the closed form, restated from scripts/make_list_nested_fixtures.py --

int outer_len(int i) { return (i % 4) + 1; }
// 0 = NULL inner list, 1 = EMPTY inner list, 2 = has values.
int inner_kind(int i, int j) {
    const int s = i + j;
    if (s % 13 == 0) return 0;
    if (s % 7 == 0) return 1;
    return 2;
}
int inner_len(int i, int j) { return (i + j) % 3 + 1; }
std::int64_t leaf_value(int i, int j, int k) {
    return static_cast<std::int64_t>(i) * 1000 + j * 10 + k;
}
bool leaf_is_null(int i, int j, int k) { return (i + j + k) % 17 == 0; }

// A value at any nesting depth: either a NULL (list or leaf), a leaf int64,
// or a (possibly empty) list of deeper NestedValues. This is the SAME shape
// BoltColumn::make_list nests, generalized to a model the test can build
// independently and diff against bolt's assembly.
struct NestedValue {
    bool is_null = false;
    bool is_leaf = false;
    std::int64_t leaf_value = 0;
    std::vector<NestedValue> elems;

    static NestedValue Null() { NestedValue v; v.is_null = true; return v; }
    static NestedValue Leaf(std::int64_t x) {
        NestedValue v; v.is_leaf = true; v.leaf_value = x; return v;
    }
    static NestedValue LeafNull() {
        NestedValue v; v.is_leaf = true; v.is_null = true; return v;
    }
    static NestedValue List(std::vector<NestedValue> e) {
        NestedValue v; v.elems = std::move(e); return v;
    }
};

// One inner list<int64> at (i, j), per inner_kind/inner_len/leaf_value above.
NestedValue model_inner(int i, int j) {
    const int kind = inner_kind(i, j);
    if (kind == 0) return NestedValue::Null();
    if (kind == 1) return NestedValue::List({});
    std::vector<NestedValue> leaves;
    for (int k = 0; k < inner_len(i, j); ++k) {
        leaves.push_back(leaf_is_null(i, j, k)
                              ? NestedValue::LeafNull()
                              : NestedValue::Leaf(leaf_value(i, j, k)));
    }
    return NestedValue::List(std::move(leaves));
}

// column "c": list<list<int64>>, row i.
NestedValue model_c(int i) {
    if (i % 19 == 0) return NestedValue::Null();
    if (i % 11 == 0) return NestedValue::List({});
    std::vector<NestedValue> outer;
    for (int j = 0; j < outer_len(i); ++j) outer.push_back(model_inner(i, j));
    return NestedValue::List(std::move(outer));
}

// column "c3": list<list<list<int64>>>, row i -- see make_list_nested_fixtures
// .build_c3 for the derivation (each outer element is an independent
// model_inner-shaped sub-row keyed by i*31+j).
NestedValue model_mid(int mid_key, int m) {
    const int kind = inner_kind(mid_key, m);
    if (kind == 0) return NestedValue::Null();
    if (kind == 1) return NestedValue::List({});
    std::vector<NestedValue> leaves;
    for (int k = 0; k < inner_len(mid_key, m); ++k) {
        leaves.push_back(leaf_is_null(mid_key, m, k)
                              ? NestedValue::LeafNull()
                              : NestedValue::Leaf(leaf_value(mid_key, m, k)));
    }
    return NestedValue::List(std::move(leaves));
}
NestedValue model_c3(int i) {
    if (i % 19 == 0) return NestedValue::Null();
    if (i % 11 == 0) return NestedValue::List({});
    std::vector<NestedValue> outer;
    for (int j = 0; j < outer_len(i); ++j) {
        const int mid_key = i * 31 + j;
        const int kind = inner_kind(i, j);
        if (kind == 0) { outer.push_back(NestedValue::Null()); continue; }
        if (kind == 1) { outer.push_back(NestedValue::List({})); continue; }
        std::vector<NestedValue> mid;
        for (int m = 0; m < inner_len(i, j); ++m) mid.push_back(model_mid(mid_key, m));
        outer.push_back(NestedValue::List(std::move(mid)));
    }
    return NestedValue::List(std::move(outer));
}

// column "mp": map<string, list<int64>>, VALUE side only (the key leaf is
// max_rep == 1, unaffected by this ticket and covered by
// BoltParquetList.MapLeavesReadAsLists already). Row i's key_value entries
// are indexed j = 0..(i%4)-1; value j uses inner_kind/inner_len at (i, j+1)
// per make_list_nested_fixtures.build_map_of_list.
NestedValue model_mp_value_entry(int i, int j) { return model_inner(i, j + 1); }
NestedValue model_mp_row(int i) {
    if (i % 23 == 0) return NestedValue::Null();
    const int nkeys = i % 4;
    std::vector<NestedValue> entries;
    for (int j = 0; j < nkeys; ++j) entries.push_back(model_mp_value_entry(i, j));
    return NestedValue::List(std::move(entries));
}

// ---- generic recursive checker --------------------------------------------
//
// `col` is a BoltColumn at some nesting level (List, or the terminal Int64
// leaf); `idx` is this value's GLOBAL index within that level's flat
// element/row array. Walking down mirrors exactly how build_list_column
// wrapped the levels: col.list_element() at level k is level k+1's column,
// and offs[idx]/offs[idx+1] is idx's child range there.
void check_value(const bolt::BoltColumn& col, std::int64_t idx,
                 const NestedValue& expected) {
    ASSERT_GE(idx, 0);
    const bool valid = (col.validity == nullptr) ||
        (((col.validity[idx >> 3] >> (idx & 7)) & 1u) != 0u);
    ASSERT_EQ(valid, !expected.is_null)
        << "idx " << idx << " is_leaf=" << expected.is_leaf;
    if (expected.is_null) return;
    if (expected.is_leaf) {
        ASSERT_EQ(col.type, bolt::BoltType::Int64);
        const auto* vals = static_cast<const std::int64_t*>(col.data);
        ASSERT_NE(vals, nullptr);
        EXPECT_EQ(vals[idx], expected.leaf_value) << "idx " << idx;
        return;
    }
    ASSERT_EQ(col.type, bolt::BoltType::List);
    const std::int32_t* offs = col.list_offsets();
    const auto* child = col.list_element();
    ASSERT_NE(offs, nullptr);
    ASSERT_NE(child, nullptr);
    const std::int64_t lo = offs[idx], hi = offs[idx + 1];
    ASSERT_GE(hi, lo);
    ASSERT_EQ(hi - lo, static_cast<std::int64_t>(expected.elems.size()))
        << "idx " << idx << " child count";
    for (std::size_t k = 0; k < expected.elems.size(); ++k) {
        check_value(*child, lo + static_cast<std::int64_t>(k), expected.elems[k]);
    }
}

}  // namespace

// ---- the levels the schema walk derived for TWO DIFFERENT shapes ----------

TEST(BoltParquetListNested, SchemaWalkDerivesLevelsForListOfList) {
    // Hand-derived from the spec's def/rep accumulation rules against the
    // REAL schema pyarrow wrote for "c" (dumped via pyarrow.parquet at
    // fixture-generation time):
    //   optional group c (LIST) { repeated group list {
    //     optional group element (LIST) { repeated group list {
    //       optional int64 element } } } }
    // c: def=1,rep=0 (optional, not itself repeated)
    // outer list: def=1+1=2, rep=1        -> level1 (list_def=1, rep_def=2)
    // element (LIST-typed, optional): def=2+1=3, rep=1 (unchanged, not repeated)
    // inner list: def=3+1=4, rep=2        -> level2 (list_def=3, rep_def=4)
    // leaf (optional int64): def=4+1=5, rep=2 (unchanged, not repeated)
    const auto buf = slurp(data_path("golden_list_nested2.parquet").c_str());
    ASSERT_FALSE(buf.empty()) << "fixture missing -- run "
                                 "scripts/make_list_nested_fixtures.py";
    bolt::Arena a;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &a, &meta));
    const int ci = find_col(&meta, "c.list.element.list.element");
    ASSERT_GE(ci, 0);
    const PqColumn& col = meta.columns[ci];
    EXPECT_EQ(col.max_rep, 2u);
    EXPECT_EQ(col.max_def, 5u);
    EXPECT_EQ(col.list_defs[0], 1u);
    EXPECT_EQ(col.rep_defs[0], 2u);
    EXPECT_EQ(col.list_defs[1], 3u);
    EXPECT_EQ(col.rep_defs[1], 4u);
    // The scalar fields mirror the INNERMOST level (see PqColumn comment).
    EXPECT_EQ(col.list_def, 3u);
    EXPECT_EQ(col.rep_def, 4u);
}

TEST(BoltParquetListNested, SchemaWalkDerivesLevelsForMapOfList) {
    // A DIFFERENT schema shape reaching the same max_rep == 2: a MAP's
    // key_value is the level-1 repeated ancestor (not a "list" node named
    // "list"), and "value" being itself LIST-annotated opens level 2 without
    // a synthetic "element" wrapper the way list<list<T>> uses one. Proves
    // the schema walk is not keyed to one particular node-naming pattern.
    //   optional group mp (Map) { repeated group key_value {
    //     required binary key; optional group value (LIST) {
    //       repeated group list { optional int64 element } } } }
    const auto buf = slurp(data_path("golden_list_nested2.parquet").c_str());
    ASSERT_FALSE(buf.empty());
    bolt::Arena a;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &a, &meta));
    const int vi = find_col(&meta, "mp.key_value.value.list.element");
    ASSERT_GE(vi, 0);
    const PqColumn& col = meta.columns[vi];
    EXPECT_EQ(col.max_rep, 2u);
    EXPECT_EQ(col.list_defs[0], 1u);
    EXPECT_EQ(col.rep_defs[0], 2u);
    EXPECT_EQ(col.list_defs[1], 3u);
    EXPECT_EQ(col.rep_defs[1], 4u);
    // The key leaf is unaffected by G2PQ-15 (max_rep == 1) -- checked here
    // only to prove the two leaves' thresholds do not collide/alias.
    const int ki = find_col(&meta, "mp.key_value.key");
    ASSERT_GE(ki, 0);
    EXPECT_EQ(meta.columns[ki].max_rep, 1u);
    EXPECT_EQ(meta.columns[ki].list_defs[0], 1u);
    EXPECT_EQ(meta.columns[ki].rep_defs[0], 2u);
}

// ---- value correctness: list<list<int64>>, both encodings -----------------

TEST(BoltParquetListNested, ListOfListMatchesPyarrowModel) {
    for (const char* fixture :
         {"golden_list_nested2.parquet", "golden_list_nested2_dict.parquet"}) {
        SCOPED_TRACE(fixture);
        const auto buf = slurp(data_path(fixture).c_str());
        ASSERT_FALSE(buf.empty()) << fixture;
        // parquet_read_file's output columns are indexed identically to
        // PqMeta::columns (one slot per LEAF), and a nested list's
        // registered field name is its full dotted leaf path -- same
        // convention BoltParquetList.ReadFileReturnsListColumns relies on
        // for max_rep == 1 -- so find the leaf's index via the metadata
        // walk, then index the SAME slot in the assembled batch.
        bolt::Arena ma;
        PqMeta meta{};
        ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &ma, &meta));
        const int ci = find_col(&meta, "c.list.element.list.element");
        ASSERT_GE(ci, 0);

        bolt::Arena ba;
        auto* batch = ba.allocate_array<bolt::BoltBatch>(1);
        ASSERT_TRUE(parquet_read_file(buf.data(), buf.size(), &ba, batch))
            << fixture;
        const bolt::BoltColumn& c = batch->columns[batch->read_epoch][ci];
        ASSERT_EQ(c.type, bolt::BoltType::List);
        ASSERT_EQ(c.length, kN);
        for (int i = 0; i < kN; ++i) {
            SCOPED_TRACE(testing::Message() << "row " << i);
            check_value(c, i, model_c(i));
        }
    }
}

// ---- value correctness: three levels of nesting ----------------------------

TEST(BoltParquetListNested, ThreeLevelsOfNestingMatchesPyarrowModel) {
    const auto buf = slurp(data_path("golden_list_nested2.parquet").c_str());
    ASSERT_FALSE(buf.empty());
    bolt::Arena ma;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &ma, &meta));
    const int c3i =
        find_col(&meta, "c3.list.element.list.element.list.element");
    ASSERT_GE(c3i, 0);

    bolt::Arena ba;
    auto* batch = ba.allocate_array<bolt::BoltBatch>(1);
    ASSERT_TRUE(parquet_read_file(buf.data(), buf.size(), &ba, batch));
    const bolt::BoltColumn& c3 = batch->columns[batch->read_epoch][c3i];
    ASSERT_EQ(c3.type, bolt::BoltType::List);
    ASSERT_EQ(c3.length, kN);
    for (int i = 0; i < kN; ++i) {
        SCOPED_TRACE(testing::Message() << "row " << i);
        check_value(c3, i, model_c3(i));
    }
}

// ---- value correctness: a MAP whose value is itself a LIST -----------------

TEST(BoltParquetListNested, MapOfListMatchesPyarrowModel) {
    // Read per row group via parquet_read_list_column, exactly like the
    // existing MapLeavesReadAsLists test, so this also exercises the
    // row-group-range entry point (not just parquet_read_file's whole-file
    // path) for max_rep == 2.
    const auto buf = slurp(data_path("golden_list_nested2.parquet").c_str());
    ASSERT_FALSE(buf.empty());
    bolt::Arena a;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &a, &meta));
    const int vi = find_col(&meta, "mp.key_value.value.list.element");
    ASSERT_GE(vi, 0);

    int row = 0;
    for (std::uint32_t g = 0; g < meta.n_row_groups; ++g) {
        bolt::Arena ga;
        bolt::BoltColumn col{};
        std::int64_t rows = 0;
        ASSERT_TRUE(parquet_read_list_column(buf.data(), buf.size(), &meta, g,
                                             static_cast<std::uint16_t>(vi),
                                             &ga, &col, &rows));
        for (std::int64_t r = 0; r < rows; ++r, ++row) {
            SCOPED_TRACE(testing::Message() << "row " << row);
            check_value(col, r, model_mp_row(row));
        }
    }
    EXPECT_EQ(row, kN);
}

// ---- edge cases called out explicitly by the ticket ------------------------

TEST(BoltParquetListNested, EmptyNullAndNullElementEdgeCasesExist) {
    // Sanity that the closed form actually exercises every edge case the
    // ticket names, so the tests above are not silently vacuous over them:
    // a NULL outer list, an EMPTY outer list, a NULL inner list (an outer
    // ELEMENT whose own list value is null), an EMPTY inner list, and a
    // list-of-empty-lists row (every present outer element has an empty
    // inner list).
    bool saw_null_outer = false, saw_empty_outer = false;
    bool saw_null_inner = false, saw_empty_inner = false;
    bool saw_all_empty_inner_row = false;
    for (int i = 0; i < kN; ++i) {
        const NestedValue v = model_c(i);
        if (v.is_null) { saw_null_outer = true; continue; }
        if (v.elems.empty()) { saw_empty_outer = true; continue; }
        bool all_empty = true;
        for (const auto& e : v.elems) {
            if (e.is_null) saw_null_inner = true;
            else if (e.elems.empty()) saw_empty_inner = true;
            if (!(e.elems.empty() && !e.is_null)) all_empty = false;
        }
        if (all_empty) saw_all_empty_inner_row = true;
    }
    EXPECT_TRUE(saw_null_outer);
    EXPECT_TRUE(saw_empty_outer);
    EXPECT_TRUE(saw_null_inner);
    EXPECT_TRUE(saw_empty_inner);
    EXPECT_TRUE(saw_all_empty_inner_row);
}

// ---- refusal past the supported depth --------------------------------------

TEST(BoltParquetListNested, RealFileTenLevelsDeepRefusedNotMisassembled) {
    // A genuinely legal parquet file (pyarrow wrote it) nested 10 levels of
    // LIST deep -- past kPqMaxRepLevels (8). Whichever guard catches it first
    // (the pre-existing 16-deep schema-tree stack, or this ticket's own
    // max_rep cap -- the verbose 3-level LIST encoding costs ~2 schema levels
    // per repetition level, so the older cap fires first for this exact
    // encoding shape), the file must be refused cleanly, never crash and
    // never silently reshape/truncate the data.
    const auto buf = slurp(data_path("golden_list_too_deep.parquet").c_str());
    ASSERT_FALSE(buf.empty()) << "fixture missing -- run "
                                 "scripts/make_list_nested_fixtures.py";
    bolt::Arena a;
    PqMeta meta{};
    // Either the schema walk itself refuses, or (if it somehow didn't) the
    // file-level read must -- never a silent success reading past the cap.
    bool ok = parquet_read_meta(buf.data(), buf.size(), &a, &meta);
    if (ok) {
        bolt::Arena ba;
        auto* batch = ba.allocate_array<bolt::BoltBatch>(1);
        ok = parquet_read_file(buf.data(), buf.size(), &ba, batch);
    }
    EXPECT_FALSE(ok) << "10 levels of nested repetition must be refused";
}

TEST(BoltParquetListNested, MaxRepPastCapRefusedDirectly) {
    // Unit-level, no file needed: parquet_read_list_column's own cap check
    // (kPqMaxRepLevels) must refuse BEFORE ever touching build_list_column,
    // regardless of whether a real pyarrow encoding can reach this exact
    // max_rep without also tripping the older schema-depth cap first (see
    // RealFileTenLevelsDeepRefusedNotMisassembled above -- it generally
    // cannot, for the verbose 3-level LIST shape). A hand-built PqColumn
    // with max_rep one past the cap is enough to prove THIS guard fires.
    // parquet_read_list_column's cap check runs before touching row groups or
    // chunks at all (both return false first on out-of-range col/row_group),
    // so the meta below needs only enough to pass THOSE bounds checks and
    // reach the column: n_columns/n_row_groups and the one PqColumn's
    // max_rep. PqMeta::chunks is a caller-owned out-of-line pointer, deliber-
    // ately left null -- it is never dereferenced on this path.
    bolt::Arena a;
    PqMeta meta{};
    meta.n_columns = 1;
    meta.n_row_groups = 1;
    std::strcpy(meta.columns[0].name, "d");
    meta.columns[0].physical = PqType::Int64;
    meta.columns[0].max_rep = static_cast<std::uint8_t>(kPqMaxRepLevels + 1u);
    meta.columns[0].max_def = static_cast<std::uint8_t>(kPqMaxRepLevels + 1u);
    bolt::BoltColumn col{};
    std::int64_t rows = 0;
    // A non-null, non-empty buf: the cap check must fire before anything
    // ever tries to read page bytes out of it, so its contents don't matter.
    const std::uint8_t dummy_buf[1] = {0};
    EXPECT_FALSE(parquet_read_list_column(dummy_buf, 1u, &meta, 0, 0, &a, &col,
                                          &rows));
}
