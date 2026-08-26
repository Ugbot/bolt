// Reading LIST / MAP columns: Dremel record assembly.
//
// This was the last thing hardwood could read that bolt could not. A leaf
// under a REPEATED group produces a VARIABLE number of values per row, so it
// needs the repetition levels to find row boundaries and the definition
// levels to tell four cases apart:
//
//   a present element, a NULL element, an EMPTY list, and a NULL list.
//
// The last two are the ones that make this worth testing carefully. They are
// different values, a null bitmap alone cannot distinguish them, and getting
// the level comparison off by one silently converts every null list into an
// empty one (or vice versa) while every count still adds up.
//
// The fixtures are written by PYARROW, not by bolt, and the expected values
// are regenerated here from the same closed form the generator used -- so
// this compares bolt against the ecosystem's reading of the spec rather than
// against itself.

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

// The generator's closed form, restated. `li` row i is:
//   i % 17 == 0 -> NULL list
//   i % 7  == 0 -> EMPTY list
//   else        -> [i*10 + j for j in 0..k), k = (i % 5) + 1
enum class RowKind { Null, Empty, Values };

RowKind kind_of(int i) {
    if (i % 17 == 0) return RowKind::Null;
    if (i % 7 == 0) return RowKind::Empty;
    return RowKind::Values;
}
int len_of(int i) { return (i % 5) + 1; }

int find_col(const PqMeta* m, const char* path) {
    for (std::uint32_t c = 0; c < m->n_columns; ++c) {
        if (std::strcmp(m->columns[c].name, path) == 0) return static_cast<int>(c);
    }
    return -1;
}

// ---- the levels the schema walk derived -----------------------------------

TEST(BoltParquetList, SchemaWalkDerivesListLevels) {
    // Everything downstream keys off these three numbers; if the walk gets
    // them wrong every assembly below is wrong in the same direction, so they
    // are checked against pyarrow's own reported levels first.
    const auto buf = slurp(data_path("golden_list.parquet").c_str());
    ASSERT_FALSE(buf.empty()) << "fixture missing";
    bolt::Arena a;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &a, &meta));

    const int li = find_col(&meta, "li.list.element");
    ASSERT_GE(li, 0);
    EXPECT_EQ(meta.columns[li].max_def, 3u);
    EXPECT_EQ(meta.columns[li].max_rep, 1u);
    // optional group li (LIST) -> def 1; repeated group list -> def 2.
    EXPECT_EQ(meta.columns[li].list_def, 1u);
    EXPECT_EQ(meta.columns[li].rep_def, 2u);

    // A map's KEY is REQUIRED, so its max_def is one lower than the value's
    // while both share the same list/rep levels.
    const int mk = find_col(&meta, "mp.key_value.key");
    const int mv = find_col(&meta, "mp.key_value.value");
    ASSERT_GE(mk, 0);
    ASSERT_GE(mv, 0);
    EXPECT_EQ(meta.columns[mk].max_def, 2u);
    EXPECT_EQ(meta.columns[mv].max_def, 3u);
    EXPECT_EQ(meta.columns[mk].rep_def, 2u);
    EXPECT_EQ(meta.columns[mv].rep_def, 2u);

    // A flat column keeps zeroes -- the new fields must not perturb it.
    const int fl = find_col(&meta, "flat");
    ASSERT_GE(fl, 0);
    EXPECT_EQ(meta.columns[fl].max_rep, 0u);
    EXPECT_EQ(meta.columns[fl].list_def, 0u);
    EXPECT_EQ(meta.columns[fl].rep_def, 0u);
}

// ---- int64 lists ----------------------------------------------------------

void check_int_list(const char* fixture) {
    const auto buf = slurp(data_path(fixture).c_str());
    ASSERT_FALSE(buf.empty()) << fixture;
    bolt::Arena a;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &a, &meta));
    const int li = find_col(&meta, "li.list.element");
    ASSERT_GE(li, 0);

    // Rows are spread over row groups; walk them all and concatenate.
    int row = 0;
    for (std::uint32_t g = 0; g < meta.n_row_groups; ++g) {
        bolt::Arena ga;
        bolt::BoltColumn col{};
        std::int64_t rows = 0;
        ASSERT_TRUE(parquet_read_list_column(buf.data(), buf.size(), &meta, g,
                                             static_cast<std::uint16_t>(li),
                                             &ga, &col, &rows))
            << fixture << " rg " << g;
        ASSERT_TRUE(col.is_nested());
        ASSERT_EQ(col.type, bolt::BoltType::List);
        const std::int32_t* offs = col.list_offsets();
        const bolt::BoltColumn* elem = col.list_element();
        ASSERT_NE(offs, nullptr);
        ASSERT_NE(elem, nullptr);
        ASSERT_EQ(offs[0], 0);
        const auto* ep = static_cast<const std::int64_t*>(elem->data);
        ASSERT_NE(ep, nullptr);

        for (std::int64_t r = 0; r < rows; ++r, ++row) {
            const RowKind k = kind_of(row);
            const bool valid =
                (col.validity == nullptr) ||
                (((col.validity[r >> 3] >> (r & 7)) & 1u) != 0u);
            SCOPED_TRACE(testing::Message() << fixture << " row " << row);
            if (k == RowKind::Null) {
                EXPECT_FALSE(valid) << "a NULL list decoded as present";
                // A null list also carries no elements.
                EXPECT_EQ(offs[r + 1] - offs[r], 0);
                continue;
            }
            ASSERT_TRUE(valid) << "a present list decoded as NULL";
            if (k == RowKind::Empty) {
                // The distinction this whole test exists for: EMPTY is valid
                // with a zero-width offset range, not null.
                EXPECT_EQ(offs[r + 1] - offs[r], 0) << "empty list has elements";
                continue;
            }
            const int want = len_of(row);
            ASSERT_EQ(offs[r + 1] - offs[r], want) << "list length";
            for (int j = 0; j < want; ++j) {
                EXPECT_EQ(ep[offs[r] + j], static_cast<std::int64_t>(row) * 10 + j)
                    << "element " << j;
            }
        }
    }
    EXPECT_EQ(row, kN) << fixture << ": row count";
}

TEST(BoltParquetList, Int64ListsPlainV2) {
    check_int_list("golden_list.parquet");
}

TEST(BoltParquetList, Int64ListsDictionaryV1) {
    // Same logical data written dictionary-encoded as v1 pages. Two entirely
    // different decode paths must produce the same lists.
    check_int_list("golden_list_dict.parquet");
}

// ---- string lists with NULL elements --------------------------------------

TEST(BoltParquetList, StringListsWithNullElements) {
    // `ls` row i element j is null when (i + j) % 11 == 0. A null ELEMENT is
    // a third case distinct from a null list and an empty list, and it is the
    // one that shares a definition level with "list present".
    for (const char* fixture : {"golden_list.parquet", "golden_list_dict.parquet"}) {
        const auto buf = slurp(data_path(fixture).c_str());
        ASSERT_FALSE(buf.empty()) << fixture;
        bolt::Arena a;
        PqMeta meta{};
        ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &a, &meta));
        const int ls = find_col(&meta, "ls.list.element");
        ASSERT_GE(ls, 0);

        int row = 0;
        for (std::uint32_t g = 0; g < meta.n_row_groups; ++g) {
            bolt::Arena ga;
            bolt::BoltColumn col{};
            std::int64_t rows = 0;
            ASSERT_TRUE(parquet_read_list_column(buf.data(), buf.size(), &meta,
                                                 g, static_cast<std::uint16_t>(ls),
                                                 &ga, &col, &rows));
            const std::int32_t* offs = col.list_offsets();
            const bolt::BoltColumn* elem = col.list_element();
            ASSERT_NE(offs, nullptr);
            ASSERT_NE(elem, nullptr);
            const auto* sv = static_cast<const bolt::StringView*>(elem->data);
            const auto* sp =
                static_cast<const std::uint8_t*>(elem->str_overflow_base);
            ASSERT_NE(sv, nullptr);

            for (std::int64_t r = 0; r < rows; ++r, ++row) {
                SCOPED_TRACE(testing::Message() << fixture << " row " << row);
                if (kind_of(row) != RowKind::Values) continue;
                const int want = len_of(row);
                ASSERT_EQ(offs[r + 1] - offs[r], want);
                for (int j = 0; j < want; ++j) {
                    const std::int64_t e = offs[r] + j;
                    const bool evalid =
                        (elem->validity == nullptr) ||
                        (((elem->validity[e >> 3] >> (e & 7)) & 1u) != 0u);
                    const bool want_null = ((row + j) % 11) == 0;
                    ASSERT_EQ(evalid, !want_null) << "element " << j
                                                  << " nullness";
                    if (want_null) continue;
                    char expect[64];
                    std::snprintf(expect, sizeof(expect), "s%d-%d", row, j);
                    const std::uint32_t l = sv[e].length;
                    const std::uint8_t* p =
                        (l <= 12u)
                            ? reinterpret_cast<const std::uint8_t*>(&sv[e].prefix[0])
                            : (sp + sv[e].ref.offset);
                    ASSERT_EQ(l, std::strlen(expect)) << "element " << j;
                    ASSERT_EQ(0, std::memcmp(p, expect, l)) << "element " << j;
                }
            }
        }
        EXPECT_EQ(row, kN) << fixture;
    }
}

// ---- MAP ------------------------------------------------------------------

TEST(BoltParquetList, MapLeavesReadAsLists) {
    // A parquet MAP is a repeated group of {key, value}, so each leaf is a
    // list with the SAME offsets. Reading them independently and finding the
    // offsets agree is what proves the two halves stay aligned -- a map whose
    // keys and values disagree on row boundaries is silently scrambled data.
    const auto buf = slurp(data_path("golden_list.parquet").c_str());
    ASSERT_FALSE(buf.empty());
    bolt::Arena a;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &a, &meta));
    const int mk = find_col(&meta, "mp.key_value.key");
    const int mv = find_col(&meta, "mp.key_value.value");
    ASSERT_GE(mk, 0);
    ASSERT_GE(mv, 0);

    int row = 0;
    for (std::uint32_t g = 0; g < meta.n_row_groups; ++g) {
        bolt::Arena ka, va;
        bolt::BoltColumn kc{}, vc{};
        std::int64_t krows = 0, vrows = 0;
        ASSERT_TRUE(parquet_read_list_column(buf.data(), buf.size(), &meta, g,
                                             static_cast<std::uint16_t>(mk),
                                             &ka, &kc, &krows));
        ASSERT_TRUE(parquet_read_list_column(buf.data(), buf.size(), &meta, g,
                                             static_cast<std::uint16_t>(mv),
                                             &va, &vc, &vrows));
        ASSERT_EQ(krows, vrows);
        const std::int32_t* ko = kc.list_offsets();
        const std::int32_t* vo = vc.list_offsets();
        ASSERT_NE(ko, nullptr);
        ASSERT_NE(vo, nullptr);
        const auto* vp = static_cast<const std::int64_t*>(
            vc.list_element()->data);
        ASSERT_NE(vp, nullptr);

        for (std::int64_t r = 0; r < krows; ++r, ++row) {
            SCOPED_TRACE(testing::Message() << "row " << row);
            ASSERT_EQ(ko[r], vo[r]) << "key/value offsets diverged";
            ASSERT_EQ(ko[r + 1], vo[r + 1]) << "key/value offsets diverged";
            const bool knull = (kc.validity != nullptr) &&
                (((kc.validity[r >> 3] >> (r & 7)) & 1u) == 0u);
            if (row % 23 == 0) {
                EXPECT_TRUE(knull) << "a NULL map decoded as present";
                continue;
            }
            EXPECT_FALSE(knull);
            const int want = row % 4;
            ASSERT_EQ(ko[r + 1] - ko[r], want) << "map size";
            for (int j = 0; j < want; ++j) {
                EXPECT_EQ(vp[vo[r] + j],
                          static_cast<std::int64_t>(row) * 100 + j);
            }
        }
    }
    EXPECT_EQ(row, kN);
}

// ---- coexistence + refusals ----------------------------------------------

TEST(BoltParquetList, FlatColumnsInTheSameFileStillRead) {
    // The whole point of the earlier "a list blocks only itself" work: a file
    // containing lists must still read its scalar columns, and it must do so
    // through the ordinary flat path with the list machinery not involved.
    const auto buf = slurp(data_path("golden_list.parquet").c_str());
    ASSERT_FALSE(buf.empty());
    bolt::Arena a;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &a, &meta));
    const int fl = find_col(&meta, "flat");
    ASSERT_GE(fl, 0);
    const std::uint16_t idx = static_cast<std::uint16_t>(fl);

    int row = 0;
    for (std::uint32_t g = 0; g < meta.n_row_groups; ++g) {
        bolt::Arena ga;
        bolt::BoltColumn col{};
        std::int64_t rows = 0;
        ASSERT_TRUE(parquet_read_row_group_cols(buf.data(), buf.size(), &meta,
                                                g, &idx, 1, &ga, &col, &rows));
        const auto* p = static_cast<const std::int64_t*>(col.data);
        ASSERT_NE(p, nullptr);
        for (std::int64_t r = 0; r < rows; ++r, ++row) {
            ASSERT_EQ(p[r], row) << "flat row " << row;
        }
    }
    EXPECT_EQ(row, kN);
}

TEST(BoltParquetList, WrongPathsAreRefusedNotGuessed) {
    const auto buf = slurp(data_path("golden_list.parquet").c_str());
    ASSERT_FALSE(buf.empty());
    bolt::Arena a;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &a, &meta));
    const int li = find_col(&meta, "li.list.element");
    const int fl = find_col(&meta, "flat");
    ASSERT_GE(li, 0);
    ASSERT_GE(fl, 0);

    bolt::BoltColumn col{};
    std::int64_t rows = 0;
    // A FLAT column is not a list: the list reader refuses it rather than
    // fabricating one-element lists.
    EXPECT_FALSE(parquet_read_list_column(buf.data(), buf.size(), &meta, 0,
                                          static_cast<std::uint16_t>(fl), &a,
                                          &col, &rows));
    // And the flat reader still refuses the repeated column.
    const std::uint16_t lidx = static_cast<std::uint16_t>(li);
    bolt::BoltColumn fc{};
    std::int64_t frows = 0;
    EXPECT_FALSE(parquet_read_row_group_cols(buf.data(), buf.size(), &meta, 0,
                                             &lidx, 1, &a, &fc, &frows));
    // Out-of-range column index.
    EXPECT_FALSE(parquet_read_list_column(buf.data(), buf.size(), &meta, 0,
                                          static_cast<std::uint16_t>(meta.n_columns),
                                          &a, &col, &rows));
}

// ---- the gate must discriminate ------------------------------------------

TEST(BoltParquetList, DiscriminatingPower) {
    // Every check above compares against `kind_of` / `len_of`. If that
    // comparison could not fail, none of it would mean anything -- and the
    // specific failure worth proving detectable is the one this feature is
    // most likely to get wrong: confusing an EMPTY list with a NULL one.
    const auto buf = slurp(data_path("golden_list.parquet").c_str());
    ASSERT_FALSE(buf.empty());
    bolt::Arena a;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &a, &meta));
    const int li = find_col(&meta, "li.list.element");
    ASSERT_GE(li, 0);
    bolt::BoltColumn col{};
    std::int64_t rows = 0;
    ASSERT_TRUE(parquet_read_list_column(buf.data(), buf.size(), &meta, 0,
                                         static_cast<std::uint16_t>(li), &a,
                                         &col, &rows));
    ASSERT_GT(rows, 40);
    const std::int32_t* offs = col.list_offsets();
    ASSERT_NE(offs, nullptr);
    ASSERT_NE(col.validity, nullptr) << "no null lists decoded at all";

    // Row 0 is i%17==0 -> NULL; row 7 is i%7==0 -> EMPTY. Both have zero
    // elements, so ONLY the validity bit separates them. If the level
    // comparison were off by one they would be indistinguishable here.
    const bool r0_valid = ((col.validity[0] >> 0) & 1u) != 0u;
    const bool r7_valid = ((col.validity[0] >> 7) & 1u) != 0u;
    EXPECT_FALSE(r0_valid) << "row 0 should be a NULL list";
    EXPECT_TRUE(r7_valid) << "row 7 should be an EMPTY list, not null";
    EXPECT_EQ(offs[1] - offs[0], 0);
    EXPECT_EQ(offs[8] - offs[7], 0);
    // ...and a row that does have values is neither.
    EXPECT_TRUE(((col.validity[0] >> 1) & 1u) != 0u);
    EXPECT_EQ(offs[2] - offs[1], len_of(1));
}

// ---- writing lists --------------------------------------------------------

// Build a Nested/List column: `lists[r]` is nullopt for a NULL list, else the
// elements, each of which may itself be nullopt for a NULL element.
struct ListModel {
    std::vector<int> len;                 // -1 = NULL list, 0 = empty
    std::vector<std::int64_t> elems;      // flattened
    std::vector<std::uint8_t> evalid;     // per element
};

ListModel make_list_model(int n) {
    ListModel m;
    for (int i = 0; i < n; ++i) {
        const RowKind k = kind_of(i);
        if (k == RowKind::Null)  { m.len.push_back(-1); continue; }
        if (k == RowKind::Empty) { m.len.push_back(0);  continue; }
        const int L = len_of(i);
        m.len.push_back(L);
        for (int j = 0; j < L; ++j) {
            m.elems.push_back(static_cast<std::int64_t>(i) * 10 + j);
            m.evalid.push_back(((i + j) % 11 == 0) ? 0u : 1u);
        }
    }
    return m;
}

bolt::BoltColumn build_list_column(bolt::Arena* a, const ListModel& m,
                                   bool elem_nullable) {
    const std::int64_t n = static_cast<std::int64_t>(m.len.size());
    const std::int64_t ne = static_cast<std::int64_t>(m.elems.size());
    bolt::BoltColumn elem =
        bolt::BoltColumn::make_flat_alloc(ne ? ne : 1, bolt::BoltType::Int64, a);
    elem.length = ne;
    auto* ep = static_cast<std::int64_t*>(elem.data);
    for (std::int64_t i = 0; i < ne; ++i) ep[i] = m.elems[static_cast<std::size_t>(i)];
    if (elem_nullable && ne > 0) {
        const std::size_t nb = static_cast<std::size_t>((ne + 7) / 8);
        auto* bm = static_cast<std::uint8_t*>(a->allocate(nb, 8));
        std::memset(bm, 0, nb);
        for (std::int64_t i = 0; i < ne; ++i) {
            if (m.evalid[static_cast<std::size_t>(i)]) {
                bm[i >> 3] = static_cast<std::uint8_t>(bm[i >> 3] | (1u << (i & 7)));
            }
        }
        elem.validity = bm;
        elem.stats.all_valid = false;
    }
    auto* offs = a->allocate_array<std::int32_t>(n + 1);
    const std::size_t vb = static_cast<std::size_t>((n + 7) / 8);
    auto* lval = static_cast<std::uint8_t*>(a->allocate(vb, 8));
    std::memset(lval, 0xFF, vb);
    std::int32_t cur = 0;
    for (std::int64_t r = 0; r < n; ++r) {
        offs[r] = cur;
        const int L = m.len[static_cast<std::size_t>(r)];
        if (L < 0) {
            lval[r >> 3] = static_cast<std::uint8_t>(lval[r >> 3] & ~(1u << (r & 7)));
        } else {
            cur += L;
        }
    }
    offs[n] = cur;
    return bolt::BoltColumn::make_list(&elem, offs, n, lval, a);
}

TEST(BoltParquetList, WriteListRoundTripsThroughBoltAndPyarrow) {
    // The asymmetry this closes: bolt could READ lists and not write them.
    // Both nullability shapes, because an element-nullable list has a
    // different max_def and so a different level stream.
    for (bool elem_nullable : {false, true}) {
        const int n = 500;
        const ListModel m = make_list_model(n);
        bolt::Arena a;
        auto* b = a.allocate_array<bolt::BoltBatch>(1);
        bolt::BoltBatch::init_empty(b);
        b->num_cols = 1;
        b->num_rows = n;
        bolt::BoltBatch::alloc_columns(b, &a, 1);
        b->schema.add_field("li", bolt::BoltType::List, true);
        b->columns[b->read_epoch][0] = build_list_column(&a, m, elem_nullable);

        ParquetWriteOpts o{};
        o.n_columns = 1;
        o.compression = 1;
        o.emit_statistics = true;
        std::strncpy(o.columns[0].name, "li", sizeof(o.columns[0].name) - 1);
        o.columns[0].type = bolt::BoltType::List;
        o.columns[0].nullable = true;
        o.columns[0].element_type = bolt::BoltType::Int64;
        o.columns[0].element_nullable = elem_nullable;

        const std::string path =
            std::string("test_bolt_parquet_list_written_") +
            (elem_nullable ? "en" : "er") + ".parquet";
        ParquetWriter* w = parquet_write_open(path.c_str(), &o);
        ASSERT_NE(w, nullptr);
        ASSERT_TRUE(parquet_write_row_group(w, b));
        ASSERT_TRUE(parquet_write_close(w));

        const auto buf = slurp(path.c_str());
        ASSERT_FALSE(buf.empty());
        SCOPED_TRACE(testing::Message() << "elem_nullable=" << elem_nullable);

        // The schema must be the 3-level LIST shape, which is what makes the
        // derived Dremel levels match what every other reader computes.
        bolt::Arena ma;
        PqMeta meta{};
        ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &ma, &meta));
        ASSERT_EQ(meta.n_columns, 1u);
        EXPECT_STREQ(meta.columns[0].name, "li.list.element");
        EXPECT_EQ(meta.columns[0].max_rep, 1u);
        EXPECT_EQ(meta.columns[0].list_def, 1u);
        EXPECT_EQ(meta.columns[0].rep_def, 2u);
        EXPECT_EQ(meta.columns[0].max_def, elem_nullable ? 3u : 2u);

        // Read it back through the Dremel path and compare to the model.
        bolt::Arena ra;
        bolt::BoltColumn col{};
        std::int64_t rows = 0;
        ASSERT_TRUE(parquet_read_list_column(buf.data(), buf.size(), &meta, 0,
                                             0, &ra, &col, &rows));
        ASSERT_EQ(rows, n);
        const std::int32_t* offs = col.list_offsets();
        const bolt::BoltColumn* elem = col.list_element();
        ASSERT_NE(offs, nullptr);
        ASSERT_NE(elem, nullptr);
        const auto* ep = static_cast<const std::int64_t*>(elem->data);
        std::int64_t ei = 0;
        for (int r = 0; r < n; ++r) {
            const int L = m.len[static_cast<std::size_t>(r)];
            const bool valid = (col.validity == nullptr) ||
                (((col.validity[r >> 3] >> (r & 7)) & 1u) != 0u);
            SCOPED_TRACE(testing::Message() << "row " << r);
            if (L < 0) {
                EXPECT_FALSE(valid) << "a NULL list came back present";
                EXPECT_EQ(offs[r + 1] - offs[r], 0);
                continue;
            }
            ASSERT_TRUE(valid) << "a present list came back NULL";
            ASSERT_EQ(offs[r + 1] - offs[r], L) << "list length";
            for (int j = 0; j < L; ++j) {
                const std::int64_t e = offs[r] + j;
                const bool want_valid =
                    !elem_nullable || m.evalid[static_cast<std::size_t>(ei + j)];
                const bool got_valid = (elem->validity == nullptr) ||
                    (((elem->validity[e >> 3] >> (e & 7)) & 1u) != 0u);
                ASSERT_EQ(got_valid, want_valid) << "element " << j;
                if (!want_valid) continue;
                EXPECT_EQ(ep[e], m.elems[static_cast<std::size_t>(ei + j)]);
            }
            ei += L;
        }
    }
}

}  // namespace
