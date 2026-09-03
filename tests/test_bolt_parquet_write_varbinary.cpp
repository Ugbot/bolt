// G2ICE-87 — the parquet writer must accept BOTH physical Utf8 layouts, and
// must never leave a footerless file behind.
//
// WHY THIS EXISTS. `BoltType::Utf8` legitimately occurs in two layouts:
//
//   * `ColumnFormat::Flat` — a StringView array, inline for <= 12 bytes else
//     spilled into `str_overflow_base`. What the parquet reader and every
//     compute operator produce, and the only one this writer used to read.
//   * `ColumnFormat::VarBinary` — int32 offsets (in `dict_child`) over a
//     packed byte pool (in `data`). What MarbleDB produces: `txn_batch`
//     serializes Utf8 that way and `bolt_wire_parse` rebuilds the same
//     format, so EVERY Utf8 column read out of a sealed SST block is
//     VarBinary.
//
// Reading the second as the first is a silent garbage read (the hazard
// `BoltColumn::utf8_at` exists for). In practice it took the "spilled with no
// spill base" exit and failed the row group AFTER the fixed-width columns'
// pages were already on disk — so Iceberg tiering produced files that pyarrow
// rejected with "Parquet magic bytes not found in footer", and every commit
// failed. Four data files, none readable, no snapshot.
//
// These fixtures are also read back by pyarrow — bolt reading its own bytes
// proves almost nothing about a writer, a point this tree has had to learn
// twice (the LIST writer's "…ThroughBoltAndPyarrow" test had no pyarrow in
// it; the bloom filters were never probed by anything until DuckDB was
// pointed at them). Run:
//
//     ./test_bolt_parquet_write_varbinary        # writes the fixtures
//     python3 scripts/parquet_varbinary_interop.py <dir-containing-them>

#include "bolt/ingest/bolt_parquet_write.h"
#include "bolt/ingest/bolt_parquet_read.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"

namespace bp = bolt::ingest::parquet;
using bolt::Arena;
using bolt::BoltBatch;
using bolt::BoltColumn;
using bolt::BoltType;
using bolt::ColumnFormat;
using bolt::StringView;

namespace {

constexpr int64_t kRows = 300;

// The model, stated once as generating RULES so the python oracle can
// re-derive it independently rather than trusting anything bolt emitted.
//   row i is NULL   when i % 23 == 0
//   row i is EMPTY  when i % 11 == 0 (and not null)
//   otherwise       "row-<i>-" followed by (i % 30) 'x' characters,
//                   which straddles the 12-byte inline/spill boundary.
bool model_is_null(int64_t i) noexcept { return (i % 23) == 0; }

std::string model_value(int64_t i) {
    if (model_is_null(i)) return std::string();
    if ((i % 11) == 0) return std::string();
    std::string s = "row-" + std::to_string(i) + "-";
    s.append(static_cast<std::size_t>(i % 30), 'x');
    return s;
}

// --- column builders -------------------------------------------------------

// VarBinary: int32 offsets over a packed pool. The MarbleDB shape.
BoltColumn make_varbinary_col(Arena* a, const std::vector<std::string>& vals,
                              const uint8_t* validity) {
    const int64_t n = static_cast<int64_t>(vals.size());
    int32_t* offs = a->allocate_array<int32_t>(n + 1);
    std::vector<uint8_t> pool;
    offs[0] = 0;
    for (int64_t i = 0; i < n; ++i) {
        pool.insert(pool.end(), vals[static_cast<std::size_t>(i)].begin(),
                    vals[static_cast<std::size_t>(i)].end());
        offs[i + 1] = static_cast<int32_t>(pool.size());
    }
    uint8_t* data = nullptr;
    if (!pool.empty()) {
        data = a->allocate_array<uint8_t>(pool.size());
        std::memcpy(data, pool.data(), pool.size());
    }
    return BoltColumn::make_var_binary(data, const_cast<uint8_t*>(validity),
                                       offs, n, BoltType::Utf8, a);
}

// Flat: StringView array + spill buffer. The parquet-reader shape.
BoltColumn make_flat_col(Arena* a, const std::vector<std::string>& vals,
                         const uint8_t* validity) {
    const int64_t n = static_cast<int64_t>(vals.size());
    StringView* sv = a->allocate_array<StringView>(n);
    std::vector<uint8_t> spill;
    for (int64_t i = 0; i < n; ++i) {
        const std::string& s = vals[static_cast<std::size_t>(i)];
        std::memset(&sv[i], 0, sizeof(StringView));
        sv[i].length = static_cast<uint32_t>(s.size());
        if (s.size() <= 12u) {
            if (!s.empty()) std::memcpy(&sv[i].prefix[0], s.data(), s.size());
        } else {
            std::memcpy(&sv[i].prefix[0], s.data(), 4);
            sv[i].ref.buf_idx = 0;
            sv[i].ref.offset = static_cast<uint32_t>(spill.size());
            spill.insert(spill.end(), s.begin(), s.end());
        }
    }
    uint8_t* base = nullptr;
    if (!spill.empty()) {
        base = a->allocate_array<uint8_t>(spill.size());
        std::memcpy(base, spill.data(), spill.size());
    }
    BoltColumn c = BoltColumn::make_flat(sv, const_cast<uint8_t*>(validity), n,
                                         BoltType::Utf8);
    c.str_overflow_base = base;
    return c;
}

uint8_t* make_validity(Arena* a, int64_t n) {
    const int64_t nb = (n + 7) / 8;
    uint8_t* v = a->allocate_array<uint8_t>(nb);
    std::memset(v, 0, static_cast<std::size_t>(nb));
    for (int64_t i = 0; i < n; ++i) {
        if (!model_is_null(i)) {
            v[i >> 3] = static_cast<uint8_t>(v[i >> 3] | (1u << (i & 7)));
        }
    }
    return v;
}

// A 3-column batch shaped like a MarbleDB SST block: two fixed-width columns
// then the string. The ORDER matters — the fixed columns' pages reach the
// file before the string column is encoded, which is why a late string
// failure leaves a footerless file rather than an empty one.
BoltBatch* build_batch(Arena* a, bool var_binary, bool nullable) {
    std::vector<std::string> vals;
    vals.reserve(static_cast<std::size_t>(kRows));
    for (int64_t i = 0; i < kRows; ++i) vals.push_back(model_value(i));

    int64_t* ts = a->allocate_array<int64_t>(kRows);
    int64_t* id = a->allocate_array<int64_t>(kRows);
    for (int64_t i = 0; i < kRows; ++i) { ts[i] = 1000 + i; id[i] = i; }

    uint8_t* validity = nullable ? make_validity(a, kRows) : nullptr;

    BoltBatch* b = a->allocate_array<BoltBatch>(1);
    BoltBatch::init_empty(b);
    EXPECT_TRUE(BoltBatch::alloc_columns(b, a, 3));
    b->num_rows = kRows;
    b->columns[0][0] = BoltColumn::make_flat(ts, nullptr, kRows, BoltType::Int64);
    b->columns[0][1] = BoltColumn::make_flat(id, nullptr, kRows, BoltType::Int64);
    b->columns[0][2] = var_binary ? make_varbinary_col(a, vals, validity)
                                  : make_flat_col(a, vals, validity);
    for (uint32_t c = 0; c < 3; ++c) b->columns[1][c] = b->columns[0][c];
    return b;
}

bp::ParquetWriteOpts make_opts(uint32_t rg_max_rows) {
    bp::ParquetWriteOpts po{};
    po.n_columns = 3;
    std::snprintf(po.columns[0].name, sizeof(po.columns[0].name), "ts");
    po.columns[0].type = BoltType::Int64;
    po.columns[0].nullable = false;
    std::snprintf(po.columns[1].name, sizeof(po.columns[1].name), "id");
    po.columns[1].type = BoltType::Int64;
    po.columns[1].nullable = false;
    std::snprintf(po.columns[2].name, sizeof(po.columns[2].name), "content");
    po.columns[2].type = BoltType::Utf8;
    po.columns[2].nullable = true;
    po.row_group_target_bytes = 1u << 20;
    po.row_group_max_rows = rg_max_rows;
    po.compression = 1;              // SNAPPY, the writer's own default
    po.emit_statistics = true;
    return po;
}

bool file_exists(const char* path) noexcept {
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) return false;
    std::fclose(f);
    return true;
}

std::vector<uint8_t> slurp(const char* path) {
    std::vector<uint8_t> out;
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) return out;
    uint8_t buf[4096];
    std::size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) {
        out.insert(out.end(), buf, buf + n);
    }
    std::fclose(f);
    return out;
}

// Read every row of column 2 back through bolt's reader. Necessary but NOT
// sufficient evidence — scripts/parquet_varbinary_interop.py is the oracle.
void read_back_strings(const char* path, std::vector<std::string>* out,
                       std::vector<bool>* is_null) {
    out->clear();
    is_null->clear();
    const std::vector<uint8_t> buf = slurp(path);
    ASSERT_FALSE(buf.empty());
    Arena arena;
    bp::PqMeta meta{};
    ASSERT_TRUE(bp::parquet_read_meta(buf.data(), buf.size(), &arena, &meta));
    ASSERT_EQ(meta.n_columns, 3u);
    for (uint32_t rg = 0; rg < meta.n_row_groups; ++rg) {
        Arena rg_arena;
        BoltColumn cols[3];
        int64_t n_rows = 0;
        ASSERT_TRUE(bp::parquet_read_row_group(buf.data(), buf.size(), &meta,
                                               rg, &rg_arena, cols, &n_rows));
        const BoltColumn& c = cols[2];
        for (int64_t r = 0; r < n_rows; ++r) {
            const bool null = c.is_null(r);
            is_null->push_back(null);
            if (null) { out->push_back(std::string()); continue; }
            const uint8_t* p = nullptr;
            int32_t len = 0;
            c.utf8_at(r, &p, &len);
            out->push_back(std::string(reinterpret_cast<const char*>(p),
                                       static_cast<std::size_t>(len)));
        }
    }
}

void expect_matches_model(const std::vector<std::string>& vals,
                          const std::vector<bool>& nulls) {
    ASSERT_EQ(vals.size(), static_cast<std::size_t>(kRows));
    ASSERT_EQ(nulls.size(), static_cast<std::size_t>(kRows));
    for (int64_t i = 0; i < kRows; ++i) {
        const std::size_t k = static_cast<std::size_t>(i);
        EXPECT_EQ(nulls[k], model_is_null(i)) << "row " << i;
        if (!model_is_null(i)) {
            EXPECT_EQ(vals[k], model_value(i)) << "row " << i;
        }
    }
}

}  // namespace

// A VarBinary Utf8 column — the layout every MarbleDB SST block carries —
// writes a complete, readable parquet file. PRE-FIX this failed at
// parquet_write_row_group and left a footerless file. Fixture for pyarrow.
TEST(ParquetWriteVarBinary, VarBinaryUtf8WritesCompleteFile) {
    Arena arena;
    BoltBatch* b = build_batch(&arena, /*var_binary=*/true, /*nullable=*/true);
    ASSERT_EQ(b->columns[0][2].format, ColumnFormat::VarBinary);

    const char* path = "pw_varbinary.parquet";
    const bp::ParquetWriteOpts po = make_opts(0u);
    bp::ParquetWriter* w = bp::parquet_write_open(path, &po);
    ASSERT_NE(w, nullptr);
    EXPECT_TRUE(bp::parquet_write_row_group(w, b));
    EXPECT_TRUE(bp::parquet_write_close(w));
    ASSERT_TRUE(file_exists(path));

    std::vector<std::string> vals;
    std::vector<bool> nulls;
    read_back_strings(path, &vals, &nulls);
    expect_matches_model(vals, nulls);
}

// The Flat layout keeps working — the fix must be additive. Fixture for
// pyarrow, and the byte-level reference for the equivalence test below.
TEST(ParquetWriteVarBinary, FlatUtf8StillWritesCompleteFile) {
    Arena arena;
    BoltBatch* b = build_batch(&arena, /*var_binary=*/false, /*nullable=*/true);
    ASSERT_EQ(b->columns[0][2].format, ColumnFormat::Flat);

    const char* path = "pw_flat.parquet";
    const bp::ParquetWriteOpts po = make_opts(0u);
    bp::ParquetWriter* w = bp::parquet_write_open(path, &po);
    ASSERT_NE(w, nullptr);
    EXPECT_TRUE(bp::parquet_write_row_group(w, b));
    EXPECT_TRUE(bp::parquet_write_close(w));

    std::vector<std::string> vals;
    std::vector<bool> nulls;
    read_back_strings(path, &vals, &nulls);
    expect_matches_model(vals, nulls);
}

// The layout is a MEMORY representation, not a file-format choice: the same
// logical column written from either layout must produce the same FILE. This
// is the strongest statement available without leaving bolt, and it is what
// rules out "VarBinary works but encodes something subtly different".
TEST(ParquetWriteVarBinary, BothLayoutsProduceIdenticalBytes) {
    Arena a1, a2;
    BoltBatch* vb = build_batch(&a1, /*var_binary=*/true, /*nullable=*/true);
    BoltBatch* fl = build_batch(&a2, /*var_binary=*/false, /*nullable=*/true);
    const bp::ParquetWriteOpts po = make_opts(0u);

    const char* p_vb = "pw_eq_varbinary.parquet";
    const char* p_fl = "pw_eq_flat.parquet";
    bp::ParquetWriter* w1 = bp::parquet_write_open(p_vb, &po);
    ASSERT_NE(w1, nullptr);
    ASSERT_TRUE(bp::parquet_write_row_group(w1, vb));
    ASSERT_TRUE(bp::parquet_write_close(w1));
    bp::ParquetWriter* w2 = bp::parquet_write_open(p_fl, &po);
    ASSERT_NE(w2, nullptr);
    ASSERT_TRUE(bp::parquet_write_row_group(w2, fl));
    ASSERT_TRUE(bp::parquet_write_close(w2));

    const std::vector<uint8_t> b1 = slurp(p_vb);
    const std::vector<uint8_t> b2 = slurp(p_fl);
    ASSERT_FALSE(b1.empty());
    EXPECT_EQ(b1.size(), b2.size());
    EXPECT_TRUE(b1 == b2);
}

// Row-group splitting slices a column at a row offset. A VarBinary column
// cannot be advanced by a flat stride (its `data` is a byte pool, the row
// mapping lives in the offsets), so without the VarBinary branch in
// slice_column every group past the first emits row 0's strings onwards —
// a wrong-VALUE bug that preserves every row count. Fixture for pyarrow.
TEST(ParquetWriteVarBinary, VarBinarySurvivesRowGroupSplitting) {
    Arena arena;
    BoltBatch* b = build_batch(&arena, /*var_binary=*/true, /*nullable=*/true);

    const char* path = "pw_varbinary_split.parquet";
    const bp::ParquetWriteOpts po = make_opts(64u);   // 300 rows -> 5 groups
    bp::ParquetWriter* w = bp::parquet_write_open(path, &po);
    ASSERT_NE(w, nullptr);
    ASSERT_TRUE(bp::parquet_write_row_group(w, b));
    ASSERT_TRUE(bp::parquet_write_close(w));

    Arena meta_arena;
    bp::PqMeta meta{};
    const std::vector<uint8_t> raw = slurp(path);
    ASSERT_TRUE(bp::parquet_read_meta(raw.data(), raw.size(), &meta_arena, &meta));
    EXPECT_GT(meta.n_row_groups, 1u);

    std::vector<std::string> vals;
    std::vector<bool> nulls;
    read_back_strings(path, &vals, &nulls);
    expect_matches_model(vals, nulls);
}

// A Utf8 column in a layout the encoders cannot read is REFUSED before any
// bytes reach the sink. Reinterpreting it as a StringView array would be a
// silent garbage read; guessing is never the right answer here.
TEST(ParquetWriteVarBinary, UnreadableUtf8LayoutIsRefusedWithoutWriting) {
    Arena arena;
    BoltBatch* b = build_batch(&arena, /*var_binary=*/false, /*nullable=*/false);
    b->columns[0][2].format = ColumnFormat::Dictionary;   // neither layout
    b->columns[1][2].format = ColumnFormat::Dictionary;

    const char* path = "pw_bad_layout.parquet";
    const bp::ParquetWriteOpts po = make_opts(0u);
    bp::ParquetWriter* w = bp::parquet_write_open(path, &po);
    ASSERT_NE(w, nullptr);
    EXPECT_FALSE(bp::parquet_write_row_group(w, b));
    EXPECT_FALSE(bp::parquet_write_close(w));
    EXPECT_FALSE(file_exists(path)) << "a refused write must leave no file";
}

// A file with no footer is NOT a parquet file, and leaving one on disk is
// worse than leaving nothing: it has the right name and extension, a
// manifest can end up pointing at it, and the only tool that notices is
// whatever tries to read it much later. The failure here happens LATE — the
// two int64 columns' pages are already in the file — which is exactly the
// shape Iceberg tiering produced.
TEST(ParquetWriteVarBinary, FailedCloseRemovesThePartialFile) {
    Arena arena;
    BoltBatch* b = build_batch(&arena, /*var_binary=*/true, /*nullable=*/false);
    // Offsets claim bytes, pool is gone: passes the layout check, fails inside
    // the string encoder after the fixed-width columns have been written.
    b->columns[0][2].data = nullptr;
    b->columns[1][2].data = nullptr;

    const char* path = "pw_partial.parquet";
    const bp::ParquetWriteOpts po = make_opts(0u);
    bp::ParquetWriter* w = bp::parquet_write_open(path, &po);
    ASSERT_NE(w, nullptr);
    EXPECT_FALSE(bp::parquet_write_row_group(w, b));
    EXPECT_FALSE(bp::parquet_write_close(w));
    EXPECT_FALSE(file_exists(path))
        << "close must not leave a footerless file masquerading as parquet";
}
