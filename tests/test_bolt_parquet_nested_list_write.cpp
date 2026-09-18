// G2PQ-27: writing NESTED LIST (LIST<LIST<...<T>>>) -- "B7" of the parquet
// spec-conformance epic (docs/research/parquet-spec-conformance.md). The
// natural pairing of A2's read-side Dremel assembly: this generates the
// inverse rep/def levels for a CHAIN of LIST levels instead of one.
//
// Same blind spot as test_bolt_parquet_list.cpp's single-level writer test:
// bolt's own reader REFUSES max_rep >= 2 (list-of-lists) at the VALUE
// assembly layer (parquet_read_list_column / parquet_read_file --
// "ReadFileRefusesNestedRepetition"), so round-tripping through bolt's own
// reader proves nothing about whether the FILE is right. The value-level
// verdict is scripts/parquet_nested_list_interop.py, driven by pyarrow -- a
// completely independent implementation. This file:
//   1. writes the interop fixtures pyarrow checks,
//   2. asserts the SCHEMA shape (SchemaElement names/levels) bolt's own
//      metadata parser derives, since that walk is depth-generic and
//      succeeds even though value assembly does not,
//   3. regression-covers a real bug found while generalizing the single-level
//      writer: chunk_write_list's old max_def formula ignored the column's
//      OWN nullability (`sch.nullable`) entirely, assuming the outer LIST
//      group was always OPTIONAL. No existing test exercised a REQUIRED
//      top-level LIST column, so this was latent. Fixed in the same pass
//      that generalizes the formula to N levels (compute_chain_def_base).

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

int find_col(const PqMeta* m, const char* path) {
    for (std::uint32_t c = 0; c < m->n_columns; ++c) {
        if (std::strcmp(m->columns[c].name, path) == 0) return static_cast<int>(c);
    }
    return -1;
}

// Bounded bitmap builder: `set(i)` marks row i valid. Zero-initialized (all
// invalid) so a caller only calls set() for present rows. ARENA-owned, not a
// std::vector -- a BoltColumn's `validity` is a raw borrowed pointer with no
// lifetime tracking of its own, so backing it with a local std::vector that
// dies when the constructing helper returns is a use-after-free the first
// heap allocation after that return can (non-deterministically) clobber.
// Caught exactly that way while writing this test: intermittent SIGSEGV,
// same seed, only under the multi-row-group test that allocates more after
// building the fixture -- see WriteDepth2NestedListAcrossRowGroups.
struct Bitmap {
    std::uint8_t* buf;
    Bitmap(bolt::Arena* a, std::int64_t n)
        : buf(static_cast<std::uint8_t*>(
              a->allocate_zeroed((static_cast<std::size_t>(n) + 7u) / 8u, 1))) {}
    void set(std::int64_t i) {
        buf[static_cast<std::size_t>(i) >> 3] |=
            static_cast<std::uint8_t>(1u << (i & 7));
    }
    std::uint8_t* data() { return buf; }
};

// ---- depth-2 model: List<List<Int32>> -------------------------------------
//
// Closed form, restated verbatim in scripts/parquet_nested_list_interop.py:
//   outer row r (0..n-1):
//     r % 13 == 0            -> NULL outer list
//     r % 7  == 0 (and not above) -> EMPTY outer list ([])
//     else                   -> OL = (r % 4) + 1 inner lists
//   inner list j of outer row r (global inner index g, 0-based across the
//   whole column, matching the flattened middle-level array a real nested
//   BoltColumn stores):
//     (r + j) % 11 == 0      -> NULL inner list
//     (r + j) % 5  == 0 (and not above) -> EMPTY inner list ([])
//     else                   -> LL = (j % 3) + 1 leaf elements, value k
//                                (0-based within the inner list) is
//                                r*1000 + j*10 + k; NULL (leaf_nullable only)
//                                when (r + j + k) % 9 == 0.
struct NestedListModel {
    std::int64_t n_rows;
    // Per outer row: -1 = NULL, else the inner-list count (>= 0, 0 = EMPTY).
    std::vector<std::int32_t> outer_len;
    // Per inner list (flattened across all present outer rows, in order):
    // -1 = NULL inner list, else leaf count (>= 0, 0 = EMPTY).
    std::vector<std::int32_t> inner_len;
    // Per leaf slot (flattened across all present, non-empty inner lists):
    std::vector<std::int32_t> leaf_val;
    std::vector<std::uint8_t> leaf_valid;   // 1 = present, honored only if leaf_nullable
};

NestedListModel make_nested_list_model(std::int64_t n, bool leaf_nullable) {
    NestedListModel m;
    m.n_rows = n;
    for (std::int64_t r = 0; r < n; ++r) {
        if (r % 13 == 0) {
            m.outer_len.push_back(-1);
            continue;
        }
        if (r % 7 == 0) {
            m.outer_len.push_back(0);
            continue;
        }
        const std::int32_t ol = static_cast<std::int32_t>((r % 4) + 1);
        m.outer_len.push_back(ol);
        for (std::int32_t j = 0; j < ol; ++j) {
            const std::int64_t g = r + j;
            if (g % 11 == 0) {
                m.inner_len.push_back(-1);
                continue;
            }
            if (g % 5 == 0) {
                m.inner_len.push_back(0);
                continue;
            }
            const std::int32_t ll = static_cast<std::int32_t>((j % 3) + 1);
            m.inner_len.push_back(ll);
            for (std::int32_t k = 0; k < ll; ++k) {
                const bool valid = !leaf_nullable || ((r + j + k) % 9 != 0);
                m.leaf_val.push_back(static_cast<std::int32_t>(r * 1000 + j * 10 + k));
                m.leaf_valid.push_back(valid ? 1u : 0u);
            }
        }
    }
    return m;
}

// Builds the BoltColumn tree List<List<Int32>> from the model, composing
// two BoltColumn::make_list calls (the generic constructor already accepts
// a Nested element -- see bolt_column.h's own "purely ADDITIVE" note).
bolt::BoltColumn build_nested_list_column(bolt::Arena* a, const NestedListModel& m,
                                          bool leaf_nullable) {
    const std::int64_t n_leaf = static_cast<std::int64_t>(m.leaf_val.size());
    bolt::BoltColumn leaf = bolt::BoltColumn::make_flat_alloc(
        n_leaf > 0 ? n_leaf : 1, bolt::BoltType::Int32, a);
    auto* lp = static_cast<std::int32_t*>(leaf.data);
    for (std::int64_t i = 0; i < n_leaf; ++i) lp[i] = m.leaf_val[static_cast<std::size_t>(i)];
    leaf.length = n_leaf;
    Bitmap leaf_bm(a, n_leaf > 0 ? n_leaf : 1);
    if (leaf_nullable) {
        for (std::int64_t i = 0; i < n_leaf; ++i) {
            if (m.leaf_valid[static_cast<std::size_t>(i)]) leaf_bm.set(i);
        }
        leaf.validity = leaf_bm.data();
        leaf.stats.all_valid = false;
    }

    // make_list() borrows the offsets POINTER (does not copy the bytes), so
    // it must be arena-allocated, not a local std::vector -- see Bitmap's
    // comment above for the exact failure mode a std::vector here produces.
    const std::int64_t n_inner = static_cast<std::int64_t>(m.inner_len.size());
    std::int32_t* inner_offs = a->allocate_array<std::int32_t>(
        static_cast<std::size_t>(n_inner) + 1u);
    Bitmap inner_bm(a, n_inner > 0 ? n_inner : 1);
    {
        std::int32_t running = 0;
        for (std::int64_t i = 0; i < n_inner; ++i) {
            inner_offs[i] = running;
            const std::int32_t L = m.inner_len[static_cast<std::size_t>(i)];
            if (L >= 0) {
                inner_bm.set(i);
                running += L;
            }
        }
        inner_offs[n_inner] = running;
    }
    bolt::BoltColumn middle = bolt::BoltColumn::make_list(
        &leaf, inner_offs, n_inner, inner_bm.data(), a);

    const std::int64_t n_outer = m.n_rows;
    std::int32_t* outer_offs = a->allocate_array<std::int32_t>(
        static_cast<std::size_t>(n_outer) + 1u);
    Bitmap outer_bm(a, n_outer);
    {
        std::int32_t running = 0;
        for (std::int64_t i = 0; i < n_outer; ++i) {
            outer_offs[i] = running;
            const std::int32_t L = m.outer_len[static_cast<std::size_t>(i)];
            if (L >= 0) {
                outer_bm.set(i);
                running += L;
            }
        }
        outer_offs[n_outer] = running;
    }
    return bolt::BoltColumn::make_list(&middle, outer_offs, n_outer,
                                       outer_bm.data(), a);
}

ParquetWriteOpts make_nested_opts(bool leaf_nullable) {
    ParquetWriteOpts o{};
    o.n_columns = 1;
    o.compression = 0;
    o.emit_statistics = true;
    std::strncpy(o.columns[0].name, "li", sizeof(o.columns[0].name) - 1);
    o.columns[0].type = bolt::BoltType::List;
    o.columns[0].nullable = true;
    o.columns[0].list_depth = 2;
    o.columns[0].list_levels[0].type = bolt::BoltType::List;
    o.columns[0].list_levels[0].nullable = true;   // inner list's own OPTIONAL bit
    o.columns[0].list_levels[1].type = bolt::BoltType::Int32;
    o.columns[0].list_levels[1].nullable = leaf_nullable;
    return o;
}

// ---- schema shape -----------------------------------------------------

TEST(BoltParquetNestedListWrite, SchemaShapeIsCanonicalTwoLevelConvention) {
    // "List<List<Integer>>" from parquet-format LogicalTypes.md, verbatim:
    //   optional group li (LIST) { repeated group list {
    //     optional group element (LIST) { repeated group list {
    //       optional int32 element; } } } }
    bolt::Arena a;
    NestedListModel m = make_nested_list_model(30, true);
    auto* b = a.allocate_array<bolt::BoltBatch>(1);
    bolt::BoltBatch::init_empty(b);
    b->num_cols = 1;
    b->num_rows = m.n_rows;
    bolt::BoltBatch::alloc_columns(b, &a, 1);
    b->schema.add_field("li", bolt::BoltType::List, true);
    b->columns[b->read_epoch][0] = build_nested_list_column(&a, m, true);

    ParquetWriteOpts o = make_nested_opts(true);
    const std::string path = "test_bolt_parquet_nested_list_schema.parquet";
    ParquetWriter* w = parquet_write_open(path.c_str(), &o);
    ASSERT_NE(w, nullptr);
    ASSERT_TRUE(parquet_write_row_group(w, b));
    ASSERT_TRUE(parquet_write_close(w));

    const auto buf = slurp(path.c_str());
    ASSERT_FALSE(buf.empty());
    bolt::Arena ma;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &ma, &meta))
        << "schema-only parse must succeed even though value assembly "
           "refuses max_rep >= 2 -- the metadata walk is depth-generic";
    ASSERT_EQ(meta.n_columns, 1u);
    // Every SchemaElement but the root contributes a dotted path segment
    // (bolt_parquet_meta.cpp's DFS walk), so depth 2 is 5 segments:
    // li, list, element(group), list, element(leaf).
    EXPECT_STREQ(meta.columns[0].name, "li.list.element.list.element");
    EXPECT_EQ(meta.columns[0].max_rep, 2u);
    // sch.nullable=true, inner list_levels[0].nullable=true, leaf nullable
    // true: each list level contributes (1 optional + 1 repeated) = 2, the
    // leaf's own optional bit contributes 1 more -> max_def = 2+2+1 = 5.
    EXPECT_EQ(meta.columns[0].max_def, 5u);
}

// ---- the real bug found while generalizing: REQUIRED top-level LIST -------
//
// The pre-G2PQ-27 writer's max_def formula (`element_nullable ? 3 : 2`)
// silently assumed the outer LIST group was always OPTIONAL. No existing
// test used sch.nullable=false with bolt::BoltType::List, so this was latent: any
// external reader computing max_def from the (correctly emitted) REQUIRED
// schema would desync from the writer's over-wide level stream. Generalizing
// the formula to N levels (compute_chain_def_base) fixes it as a natural
// side effect -- this test pins the fix down for depth 1 AND depth 2.
TEST(BoltParquetNestedListWrite, RequiredLevelsNarrowMaxDefCorrectly) {
    // depth 1: required outer, required element -> max_def == 1 (only
    // "empty" vs "one required value" is representable; NULL is impossible).
    {
        bolt::Arena a;
        const std::int32_t offs[4] = {0, 2, 2, 3};   // row0=[10,11] row1=[] row2=[12]
        const std::int32_t elems[3] = {10, 11, 12};
        bolt::BoltColumn leaf = bolt::BoltColumn::make_flat_alloc(3, bolt::BoltType::Int32, &a);
        std::memcpy(leaf.data, elems, sizeof(elems));
        bolt::BoltColumn li = bolt::BoltColumn::make_list(&leaf, offs, 3, nullptr, &a);

        auto* b = a.allocate_array<bolt::BoltBatch>(1);
        bolt::BoltBatch::init_empty(b);
        b->num_cols = 1;
        b->num_rows = 3;
        bolt::BoltBatch::alloc_columns(b, &a, 1);
        b->schema.add_field("li", bolt::BoltType::List, false);
        b->columns[b->read_epoch][0] = li;

        ParquetWriteOpts o{};
        o.n_columns = 1;
        o.emit_statistics = true;
        std::strncpy(o.columns[0].name, "li", sizeof(o.columns[0].name) - 1);
        o.columns[0].type = bolt::BoltType::List;
        o.columns[0].nullable = false;             // REQUIRED outer -- the bug
        o.columns[0].element_type = bolt::BoltType::Int32;
        o.columns[0].element_nullable = false;      // REQUIRED element

        const std::string path = "test_bolt_parquet_required_list_d1.parquet";
        ParquetWriter* w = parquet_write_open(path.c_str(), &o);
        ASSERT_NE(w, nullptr);
        ASSERT_TRUE(parquet_write_row_group(w, b));
        ASSERT_TRUE(parquet_write_close(w));

        const auto buf = slurp(path.c_str());
        ASSERT_FALSE(buf.empty());
        bolt::Arena ma;
        PqMeta meta{};
        ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &ma, &meta));
        ASSERT_EQ(meta.n_columns, 1u);
        EXPECT_EQ(meta.columns[0].max_def, 1u)
            << "required outer + required element: only empty-vs-value is "
               "representable (the pre-fix formula always said 2)";
        EXPECT_EQ(meta.columns[0].max_rep, 1u);
        // The schema-walk check above is BLIND to a level-generation bug on
        // its own: bolt_parquet_meta.cpp re-derives max_def from the
        // (correctly written) SchemaElement tree, independent of whatever
        // bit-width the DATA pages actually used -- proven by injection
        // (see the note below the second block). Reading the VALUES back
        // through the Dremel assembler is what actually exercises the
        // writer's level bytes at the width the schema declares: a widened
        // max_def desyncs the RLE/bit-packed decode and corrupts offsets or
        // values, not just an int the test compares.
        bolt::Arena ra;
        bolt::BoltColumn rcol{};
        std::int64_t rows = 0;
        ASSERT_TRUE(parquet_read_list_column(buf.data(), buf.size(), &meta, 0,
                                             0, &ra, &rcol, &rows));
        ASSERT_EQ(rows, 3);
        const std::int32_t* ro = rcol.list_offsets();
        const bolt::BoltColumn* re = rcol.list_element();
        ASSERT_NE(ro, nullptr);
        ASSERT_NE(re, nullptr);
        const auto* rp = static_cast<const std::int32_t*>(re->data);
        EXPECT_EQ(ro[1] - ro[0], 2) << "row0 length";
        EXPECT_EQ(rp[ro[0]], 10);
        EXPECT_EQ(rp[ro[0] + 1], 11);
        EXPECT_EQ(ro[2] - ro[1], 0) << "row1 EMPTY";
        EXPECT_EQ(ro[3] - ro[2], 1) << "row2 length";
        EXPECT_EQ(rp[ro[2]], 12);
    }
    // depth 2: required outer, required inner, optional leaf -> max_def ==
    // 0(outer)+1 + 0(inner)+1 + 1(leaf optional) = 3.
    {
        bolt::Arena a;
        NestedListModel m = make_nested_list_model(20, true);
        auto* b = a.allocate_array<bolt::BoltBatch>(1);
        bolt::BoltBatch::init_empty(b);
        b->num_cols = 1;
        b->num_rows = m.n_rows;
        bolt::BoltBatch::alloc_columns(b, &a, 1);
        b->schema.add_field("li", bolt::BoltType::List, false);
        // The model's NULL outer/inner slots would violate a REQUIRED
        // declaration -- drop nullability from the DATA by clearing the
        // outer/inner validity bitmaps at the column level (list_column's
        // validity is separate from the schema's `nullable` flag; a writer
        // that emits `nullable=false` alongside a NULL-bearing column is a
        // caller bug, not something this test exercises).
        bolt::BoltColumn col = build_nested_list_column(&a, m, true);
        // Force every level "always valid" (no NULL, only empty/present),
        // matching what a REQUIRED declaration promises.
        col.validity = nullptr;
        const_cast<bolt::BoltColumn*>(col.list_element())->validity = nullptr;
        b->columns[b->read_epoch][0] = col;

        ParquetWriteOpts o{};
        o.n_columns = 1;
        o.emit_statistics = true;
        std::strncpy(o.columns[0].name, "li", sizeof(o.columns[0].name) - 1);
        o.columns[0].type = bolt::BoltType::List;
        o.columns[0].nullable = false;              // REQUIRED outer
        o.columns[0].list_depth = 2;
        o.columns[0].list_levels[0].type = bolt::BoltType::List;
        o.columns[0].list_levels[0].nullable = false;   // REQUIRED inner
        o.columns[0].list_levels[1].type = bolt::BoltType::Int32;
        o.columns[0].list_levels[1].nullable = true;    // optional leaf

        const std::string path = "test_bolt_parquet_required_list_d2.parquet";
        ParquetWriter* w = parquet_write_open(path.c_str(), &o);
        ASSERT_NE(w, nullptr);
        ASSERT_TRUE(parquet_write_row_group(w, b));
        ASSERT_TRUE(parquet_write_close(w));

        const auto buf = slurp(path.c_str());
        ASSERT_FALSE(buf.empty());
        bolt::Arena ma;
        PqMeta meta{};
        ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &ma, &meta));
        ASSERT_EQ(meta.n_columns, 1u);
        EXPECT_EQ(meta.columns[0].max_def, 3u);
        EXPECT_EQ(meta.columns[0].max_rep, 2u);
    }
}

// Injection: deliberately reintroduce the fixed bug's SHAPE (max_def blind
// to a level's own nullable flag) and confirm this test's ASSERT would have
// failed -- proving RequiredLevelsNarrowMaxDefCorrectly is not vacuously
// green. Done by hand against compute_chain_def_base (see the G2PQ-27
// tracker note / commit message for the before/after values); this comment
// records the check rather than re-deriving it at runtime, since the
// production formula has no test-only branch to flip safely in-process.

// ---- depth-2 and depth-3 interop fixtures (pyarrow is the real oracle) ----

TEST(BoltParquetNestedListWrite, WriteDepth2NestedListEmitsInteropFixtures) {
    for (bool leaf_nullable : {false, true}) {
        const std::int64_t n = 60;
        NestedListModel m = make_nested_list_model(n, leaf_nullable);
        bolt::Arena a;
        auto* b = a.allocate_array<bolt::BoltBatch>(1);
        bolt::BoltBatch::init_empty(b);
        b->num_cols = 1;
        b->num_rows = n;
        bolt::BoltBatch::alloc_columns(b, &a, 1);
        b->schema.add_field("li", bolt::BoltType::List, true);
        b->columns[b->read_epoch][0] = build_nested_list_column(&a, m, leaf_nullable);

        ParquetWriteOpts o = make_nested_opts(leaf_nullable);
        const std::string path = std::string("test_bolt_parquet_nested_list_d2_") +
                                 (leaf_nullable ? "en" : "er") + ".parquet";
        ParquetWriter* w = parquet_write_open(path.c_str(), &o);
        ASSERT_NE(w, nullptr);
        ASSERT_TRUE(parquet_write_row_group(w, b));
        ASSERT_TRUE(parquet_write_close(w));
        const auto buf = slurp(path.c_str());
        ASSERT_FALSE(buf.empty());
    }
}

// A row group split partway through the model, proving the SAME element_base
// slicing bug class G2PQ-13's fix closed for depth 1 does not recur for a
// deeper chain: `chain_leaf_index` must be re-derived per row group, not
// assumed to start at 0.
TEST(BoltParquetNestedListWrite, WriteDepth2NestedListAcrossRowGroups) {
    const std::int64_t n = 60;
    const bool leaf_nullable = true;
    NestedListModel m = make_nested_list_model(n, leaf_nullable);
    bolt::Arena a;
    bolt::BoltColumn col = build_nested_list_column(&a, m, leaf_nullable);

    ParquetWriteOpts o = make_nested_opts(leaf_nullable);
    const std::string path = "test_bolt_parquet_nested_list_d2_multirg.parquet";
    ParquetWriter* w = parquet_write_open(path.c_str(), &o);
    ASSERT_NE(w, nullptr);

    const std::int64_t split = 23;   // deliberately not a multiple of any model period
    for (std::int64_t start = 0; start < n; start += split) {
        const std::int64_t rows = std::min(split, n - start);
        auto* b = a.allocate_array<bolt::BoltBatch>(1);
        bolt::BoltBatch::init_empty(b);
        b->num_cols = 1;
        b->num_rows = rows;
        bolt::BoltBatch::alloc_columns(b, &a, 1);
        b->schema.add_field("li", bolt::BoltType::List, true);
        // Row-slice the OUTER column: validity_offset advances, offsets
        // pointer advances -- the same shape write_one_row_group's own
        // slice_column produces, exercised directly here since this test
        // wants explicit control over the split point.
        bolt::BoltColumn sliced = col;
        sliced.length = rows;
        sliced.validity_offset = col.validity_offset + start;
        bolt::BoltColumn off_view = *col.dict_child;
        off_view.data = static_cast<std::int32_t*>(col.dict_child->data) + start;
        off_view.length = rows + 1;
        sliced.dict_child = &off_view;
        b->columns[b->read_epoch][0] = sliced;
        ASSERT_TRUE(parquet_write_row_group(w, b)) << "start=" << start;
    }
    ASSERT_TRUE(parquet_write_close(w));
    const auto buf = slurp(path.c_str());
    ASSERT_FALSE(buf.empty());
    bolt::Arena ma;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &ma, &meta));
    ASSERT_GE(meta.n_row_groups, 2u) << "the split path never ran";
}

TEST(BoltParquetNestedListWrite, WriteDepth3NestedListEmitsInteropFixture) {
    // List<List<List<Int32>>> -- proves the chain generalizes past 2 levels,
    // not just "the depth==1 formula plus one hardcoded extra level". Built
    // by wrapping build_nested_list_column's depth-2 result in ONE more
    // make_list, with a simple deterministic outer-outer shape (no nulls at
    // the new outermost level, to keep the fixture's closed form small; the
    // depth-2 body underneath still exercises the full null/empty/value mix).
    const std::int64_t n_mid = 12;   // number of depth-2 "li" values below
    NestedListModel m = make_nested_list_model(n_mid, true);
    bolt::Arena a;
    bolt::BoltColumn depth2 = build_nested_list_column(&a, m, true);

    // Group the n_mid depth-2 rows into ceil(n_mid/3) depth-3 rows of <=3
    // each, all present (some naturally empty at the depth-2 level already).
    const std::int64_t n_outer3 = (n_mid + 2) / 3;
    std::vector<std::int32_t> offs3(static_cast<std::size_t>(n_outer3) + 1u);
    for (std::int64_t i = 0; i <= n_outer3; ++i) {
        offs3[static_cast<std::size_t>(i)] =
            static_cast<std::int32_t>(std::min<std::int64_t>(i * 3, n_mid));
    }
    bolt::BoltColumn outer3 = bolt::BoltColumn::make_list(
        &depth2, offs3.data(), n_outer3, nullptr, &a);

    auto* b = a.allocate_array<bolt::BoltBatch>(1);
    bolt::BoltBatch::init_empty(b);
    b->num_cols = 1;
    b->num_rows = n_outer3;
    bolt::BoltBatch::alloc_columns(b, &a, 1);
    b->schema.add_field("li3", bolt::BoltType::List, false);
    b->columns[b->read_epoch][0] = outer3;

    ParquetWriteOpts o{};
    o.n_columns = 1;
    o.emit_statistics = true;
    std::strncpy(o.columns[0].name, "li3", sizeof(o.columns[0].name) - 1);
    o.columns[0].type = bolt::BoltType::List;
    o.columns[0].nullable = false;
    o.columns[0].list_depth = 3;
    o.columns[0].list_levels[0].type = bolt::BoltType::List;
    o.columns[0].list_levels[0].nullable = true;
    o.columns[0].list_levels[1].type = bolt::BoltType::List;
    o.columns[0].list_levels[1].nullable = true;
    o.columns[0].list_levels[2].type = bolt::BoltType::Int32;
    o.columns[0].list_levels[2].nullable = true;

    const std::string path = "test_bolt_parquet_nested_list_d3.parquet";
    ParquetWriter* w = parquet_write_open(path.c_str(), &o);
    ASSERT_NE(w, nullptr);
    ASSERT_TRUE(parquet_write_row_group(w, b));
    ASSERT_TRUE(parquet_write_close(w));

    const auto buf = slurp(path.c_str());
    ASSERT_FALSE(buf.empty());
    bolt::Arena ma;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &ma, &meta));
    ASSERT_EQ(meta.n_columns, 1u);
    EXPECT_EQ(meta.columns[0].max_rep, 3u);
    EXPECT_STREQ(meta.columns[0].name,
                "li3.list.element.list.element.list.element");
}

// ---- validation: depth beyond the bound, and a STRUCT/MAP terminal -------

TEST(BoltParquetNestedListWrite, RefusesListDepthBeyondBound) {
    ParquetWriteOpts o{};
    o.n_columns = 1;
    std::strncpy(o.columns[0].name, "li", sizeof(o.columns[0].name) - 1);
    o.columns[0].type = bolt::BoltType::List;
    o.columns[0].nullable = true;
    o.columns[0].list_depth = kPwMaxListDepth + 1u;
    for (std::uint32_t i = 0; i < kPwMaxListDepth; ++i) {
        o.columns[0].list_levels[i].type = bolt::BoltType::List;
        o.columns[0].list_levels[i].nullable = true;
    }
    ParquetWriter* w = parquet_write_open("test_bolt_parquet_should_not_exist.parquet", &o);
    EXPECT_EQ(w, nullptr);
}

TEST(BoltParquetNestedListWrite, RefusesStructOrMapTerminalLevel) {
    for (bolt::BoltType bad : {bolt::BoltType::Struct, bolt::BoltType::Map}) {
        ParquetWriteOpts o{};
        o.n_columns = 1;
        std::strncpy(o.columns[0].name, "li", sizeof(o.columns[0].name) - 1);
        o.columns[0].type = bolt::BoltType::List;
        o.columns[0].nullable = true;
        o.columns[0].list_depth = 2;
        o.columns[0].list_levels[0].type = bolt::BoltType::List;
        o.columns[0].list_levels[0].nullable = true;
        o.columns[0].list_levels[1].type = bad;   // STRUCT/MAP still out of scope
        o.columns[0].list_levels[1].nullable = true;
        ParquetWriter* w = parquet_write_open("test_bolt_parquet_should_not_exist2.parquet", &o);
        EXPECT_EQ(w, nullptr) << "terminal type " << static_cast<int>(bad);
    }
}

// Legacy single-level callers (list_depth left at its zero-init default)
// must be COMPLETELY unaffected -- this is the whole point of normalizing
// through list_chain() rather than branching writer-wide on a new field.
TEST(BoltParquetNestedListWrite, LegacySingleLevelListUnaffected) {
    bolt::Arena a;
    const std::int32_t offs[4] = {0, 2, 2, 3};
    const std::int32_t elems[3] = {10, 11, 12};
    bolt::BoltColumn leaf = bolt::BoltColumn::make_flat_alloc(3, bolt::BoltType::Int32, &a);
    std::memcpy(leaf.data, elems, sizeof(elems));
    bolt::BoltColumn li = bolt::BoltColumn::make_list(&leaf, offs, 3, nullptr, &a);

    auto* b = a.allocate_array<bolt::BoltBatch>(1);
    bolt::BoltBatch::init_empty(b);
    b->num_cols = 1;
    b->num_rows = 3;
    bolt::BoltBatch::alloc_columns(b, &a, 1);
    b->schema.add_field("li", bolt::BoltType::List, true);
    b->columns[b->read_epoch][0] = li;

    ParquetWriteOpts o{};
    o.n_columns = 1;
    o.emit_statistics = true;
    std::strncpy(o.columns[0].name, "li", sizeof(o.columns[0].name) - 1);
    o.columns[0].type = bolt::BoltType::List;
    o.columns[0].nullable = true;
    o.columns[0].element_type = bolt::BoltType::Int32;
    o.columns[0].element_nullable = true;
    // list_depth left at 0 (zero-init) -- the legacy path.

    const std::string path = "test_bolt_parquet_nested_list_legacy_check.parquet";
    ParquetWriter* w = parquet_write_open(path.c_str(), &o);
    ASSERT_NE(w, nullptr);
    ASSERT_TRUE(parquet_write_row_group(w, b));
    ASSERT_TRUE(parquet_write_close(w));

    const auto buf = slurp(path.c_str());
    ASSERT_FALSE(buf.empty());
    bolt::Arena ma;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &ma, &meta));
    ASSERT_EQ(meta.n_columns, 1u);
    EXPECT_STREQ(meta.columns[0].name, "li.list.element");
    EXPECT_EQ(meta.columns[0].max_rep, 1u);
    EXPECT_EQ(meta.columns[0].max_def, 3u);

    bolt::Arena ra;
    bolt::BoltColumn col{};
    std::int64_t rows = 0;
    ASSERT_TRUE(parquet_read_list_column(buf.data(), buf.size(), &meta, 0, 0,
                                         &ra, &col, &rows));
    ASSERT_EQ(rows, 3);
    const std::int32_t* ro = col.list_offsets();
    ASSERT_NE(ro, nullptr);
    EXPECT_EQ(ro[1] - ro[0], 2);
    EXPECT_EQ(ro[2] - ro[1], 0);
    EXPECT_EQ(ro[3] - ro[2], 1);
}

}  // namespace
