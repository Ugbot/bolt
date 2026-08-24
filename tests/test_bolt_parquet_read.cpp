// W-PQ increment 2 — Parquet page reader vs pyarrow-written golden files.
//
// golden_flat.parquet:        5 cols (int64 id, decimal(15,2) FLBA price,
//                             int32 qty, utf8 name, double ratio),
//                             1000 rows, SNAPPY + dictionary,
//                             row_group_size=256 -> 4 row groups.
// golden_flat_plain.parquet:  same data, UNCOMPRESSED, PLAIN (no dict).
//
// Golden data law (all 1000 rows verified):
//   id[i]    == i
//   price[i] == Decimal(i)/100  -> Decimal64 mantissa i, scale 2
//   qty[i]   == i % 50
//   name[i]  == "item-%04d-with-a-long-suffix" if i%7==0 (spilled, 28 ch)
//               else "n%d" (inline <= 12)
//   ratio[i] == i * 0.25        (exact in binary)
//
// Regenerate goldens (documented generator):
//   python - <<EOF
//   import decimal, pyarrow as pa, pyarrow.parquet as pq
//   n = 1000
//   t = pa.table({
//     "id":    pa.array(range(n), pa.int64()),
//     "price": pa.array([decimal.Decimal(i)/100 for i in range(n)],
//                       pa.decimal128(15, 2)),
//     "qty":   pa.array([i % 50 for i in range(n)], pa.int32()),
//     "name":  pa.array(["item-%04d-with-a-long-suffix" % i if i % 7 == 0
//                        else "n%d" % i for i in range(n)], pa.string()),
//     "ratio": pa.array([i * 0.25 for i in range(n)], pa.float64()),
//   })
//   pq.write_table(t, "golden_flat.parquet", row_group_size=256,
//                  compression="snappy", use_dictionary=True)
//   pq.write_table(t, "golden_flat_plain.parquet", row_group_size=256,
//                  compression="none", use_dictionary=False)
//   EOF
//
// Plus corrupt-page fuzzing: truncations and byte flips over the page
// region must return false or decode cleanly — never crash (ASAN in CI).

#include "bolt/ingest/bolt_parquet_read.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"

namespace {

using namespace bolt::ingest::parquet;

std::vector<uint8_t> slurp(const char* path) {
    std::vector<uint8_t> v;
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) return v;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    v.resize(static_cast<size_t>(n));
    const size_t got = std::fread(v.data(), 1, v.size(), f);
    std::fclose(f);
    if (got != v.size()) v.clear();
    return v;
}

std::string data_path(const char* name) {
    return std::string(BOLT_TEST_DATA_DIR) + "/" + name;
}

// Resolve a StringView's bytes (inline <= 12 chars live in prefix +
// inline_data, which are contiguous; longer views spill).
const char* sv_bytes(const bolt::StringView& sv, const void* base) {
    if (sv.length <= 12u) return &sv.prefix[0];
    return static_cast<const char*>(base) + sv.ref.offset;
}

void expect_name(int64_t i, const bolt::StringView& sv, const void* base) {
    char want[64];
    if (i % 7 == 0) {
        std::snprintf(want, sizeof(want),
                      "item-%04lld-with-a-long-suffix", (long long)i);
    } else {
        std::snprintf(want, sizeof(want), "n%lld", (long long)i);
    }
    const uint32_t wlen = static_cast<uint32_t>(std::strlen(want));
    ASSERT_EQ(sv.length, wlen) << "row " << i;
    ASSERT_EQ(std::memcmp(sv_bytes(sv, base), want, wlen), 0) << "row " << i;
    // The spill law itself: long form must spill, short form must not.
    EXPECT_EQ(sv.length > 12u, i % 7 == 0) << "row " << i;
}

// All-row value verification for a batch holding rows [first, first+n).
void verify_rows(const bolt::BoltColumn* cols, int64_t first, int64_t n) {
    ASSERT_EQ(cols[0].type, bolt::BoltType::Int64);
    ASSERT_EQ(cols[1].type, bolt::BoltType::Decimal64);
    EXPECT_EQ(cols[1].decimal_scale, 2);
    ASSERT_EQ(cols[2].type, bolt::BoltType::Int32);
    ASSERT_EQ(cols[3].type, bolt::BoltType::Utf8);
    ASSERT_EQ(cols[4].type, bolt::BoltType::Float64);
    const auto* id    = static_cast<const int64_t*>(cols[0].data);
    const auto* price = static_cast<const int64_t*>(cols[1].data);
    const auto* qty   = static_cast<const int32_t*>(cols[2].data);
    const auto* name  = static_cast<const bolt::StringView*>(cols[3].data);
    const auto* ratio = static_cast<const double*>(cols[4].data);
    for (int64_t r = 0; r < n; ++r) {
        const int64_t i = first + r;
        ASSERT_EQ(id[r], i);
        ASSERT_EQ(price[r], i) << "price mantissa, row " << i;
        ASSERT_EQ(qty[r], static_cast<int32_t>(i % 50));
        ASSERT_EQ(ratio[r], static_cast<double>(i) * 0.25);
        expect_name(i, name[r], cols[3].str_overflow_base);
    }
    // No nulls in the goldens: the all-valid shortcut must have kept the
    // bitmap unallocated even though pyarrow marks columns OPTIONAL.
    for (int c = 0; c < 5; ++c) {
        EXPECT_EQ(cols[c].validity, nullptr) << "col " << c;
        EXPECT_TRUE(cols[c].stats.all_valid) << "col " << c;
    }
}

void read_and_verify(const char* file) {
    const auto buf = slurp(data_path(file).c_str());
    ASSERT_FALSE(buf.empty());
    bolt::Arena arena;
    bolt::BoltBatch* b = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(b, nullptr);
    ASSERT_TRUE(parquet_read_file(buf.data(), buf.size(), &arena, b));
    ASSERT_EQ(b->num_rows, 1000);
    ASSERT_EQ(b->num_cols, 5u);
    verify_rows(b->columns[b->read_epoch], 0, 1000);
    // Schema travels with the batch (bolt_wire reads types from it).
    ASSERT_EQ(b->schema.num_fields, 5u);
    EXPECT_STREQ(b->schema.fields[0].name, "id");
    EXPECT_EQ(b->schema.fields[1].type, bolt::BoltType::Decimal64);
}

TEST(BoltParquetRead, GoldenSnappyDictAllValues) {
    read_and_verify("golden_flat.parquet");
}

TEST(BoltParquetRead, GoldenPlainUncompressedAllValues) {
    read_and_verify("golden_flat_plain.parquet");
}

TEST(BoltParquetRead, RowGroupGranularRead) {
    const auto buf = slurp(data_path("golden_flat.parquet").c_str());
    ASSERT_FALSE(buf.empty());
    bolt::Arena arena;
    PqMeta* meta = static_cast<PqMeta*>(
        arena.allocate(sizeof(PqMeta), alignof(PqMeta)));
    ASSERT_NE(meta, nullptr);
    std::memset(meta, 0, sizeof(*meta));
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &arena, meta));
    ASSERT_EQ(meta->n_row_groups, 4u);
    // Decode each group independently (the future parallel unit) and
    // verify the rows land with group-local offsets.
    int64_t first = 0;
    for (uint32_t g = 0; g < meta->n_row_groups; ++g) {
        bolt::BoltColumn cols[kPqMaxColumns];
        int64_t rows = 0;
        ASSERT_TRUE(parquet_read_row_group(buf.data(), buf.size(), meta, g,
                                           &arena, cols, &rows));
        ASSERT_EQ(rows, meta->row_groups[g].num_rows);
        verify_rows(cols, first, rows);
        first += rows;
    }
    ASSERT_EQ(first, 1000);
    // Out-of-range group is rejected.
    bolt::BoltColumn cols[kPqMaxColumns];
    int64_t rows = 0;
    EXPECT_FALSE(parquet_read_row_group(buf.data(), buf.size(), meta,
                                        meta->n_row_groups, &arena, cols,
                                        &rows));
}

// G2FEAT-8: projection-pushdown variant decodes exactly the requested
// columns, indexed by PROJECTION position (incl. out-of-order), and matches
// the full-decode values byte-for-byte.
TEST(BoltParquetRead, RowGroupProjectedRead) {
    const auto buf = slurp(data_path("golden_flat.parquet").c_str());
    ASSERT_FALSE(buf.empty());
    bolt::Arena arena;
    PqMeta* meta = static_cast<PqMeta*>(
        arena.allocate(sizeof(PqMeta), alignof(PqMeta)));
    ASSERT_NE(meta, nullptr);
    std::memset(meta, 0, sizeof(*meta));
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &arena, meta));
    ASSERT_EQ(meta->n_columns, 5u);
    for (uint32_t g = 0; g < meta->n_row_groups; ++g) {
        // Reference: full decode.
        bolt::BoltColumn full[kPqMaxColumns];
        int64_t full_rows = 0;
        ASSERT_TRUE(parquet_read_row_group(buf.data(), buf.size(), meta, g,
                                           &arena, full, &full_rows));
        // Projection {ratio, id, name} — out-of-order + Utf8 + skips 2 cols.
        const uint16_t proj[3] = {4, 0, 3};
        bolt::BoltColumn cols[3];
        int64_t rows = 0;
        ASSERT_TRUE(parquet_read_row_group_cols(buf.data(), buf.size(), meta,
                                                g, proj, 3, &arena, cols,
                                                &rows));
        ASSERT_EQ(rows, full_rows);
        ASSERT_EQ(cols[0].type, bolt::BoltType::Float64);   // ratio at slot 0
        ASSERT_EQ(cols[1].type, bolt::BoltType::Int64);     // id at slot 1
        ASSERT_EQ(cols[2].type, bolt::BoltType::Utf8);      // name at slot 2
        const auto* p_ratio = static_cast<const double*>(cols[0].data);
        const auto* f_ratio = static_cast<const double*>(full[4].data);
        const auto* p_id    = static_cast<const int64_t*>(cols[1].data);
        const auto* f_id    = static_cast<const int64_t*>(full[0].data);
        const auto* p_name  = static_cast<const bolt::StringView*>(cols[2].data);
        for (int64_t r = 0; r < rows; ++r) {
            ASSERT_EQ(p_ratio[r], f_ratio[r]) << "rg " << g << " row " << r;
            ASSERT_EQ(p_id[r], f_id[r]) << "rg " << g << " row " << r;
        }
        // Utf8 content check via the existing helper (projected decode owns
        // its own overflow buffer — verify content, not pointers).
        for (int64_t r = 0; r < rows; ++r) {
            const int64_t i = static_cast<int64_t>(f_id[r]);
            expect_name(i, p_name[r], cols[2].str_overflow_base);
        }
    }
    // Rejections: n_idx == 0, out-of-range column, out-of-range group.
    bolt::BoltColumn cols[3];
    int64_t rows = 0;
    const uint16_t bad_col[1] = {5};
    const uint16_t ok_col[1]  = {0};
    EXPECT_FALSE(parquet_read_row_group_cols(buf.data(), buf.size(), meta, 0,
                                             ok_col, 0, &arena, cols, &rows));
    EXPECT_FALSE(parquet_read_row_group_cols(buf.data(), buf.size(), meta, 0,
                                             bad_col, 1, &arena, cols, &rows));
    EXPECT_FALSE(parquet_read_row_group_cols(buf.data(), buf.size(), meta,
                                             meta->n_row_groups, ok_col, 1,
                                             &arena, cols, &rows));
}

TEST(BoltParquetRead, TypeMapRejectsOutsideSubset) {
    PqColumn c{};
    bolt::BoltType t;
    uint8_t s = 0;
    c.physical = PqType::Boolean;                  // -> Int64 0/1 (G2FEAT-346)
    ASSERT_TRUE(parquet_map_type(&c, &t, &s));
    EXPECT_EQ(t, bolt::BoltType::Int64);
    c.physical = PqType::Float;                    // -> Float64 (widened)
    ASSERT_TRUE(parquet_map_type(&c, &t, &s));
    EXPECT_EQ(t, bolt::BoltType::Float64);
    // Int96 used to be OUTSIDE the supported subset and was asserted rejected
    // here. G2FEAT-46 added legacy-INT96-timestamp support (decoded to
    // microseconds), so it is now INSIDE the subset -- the assertion was stale,
    // not the code. Kept as a positive case so this test still pins the
    // boundary rather than silently dropping the type.
    c.physical = PqType::Int96;
    ASSERT_TRUE(parquet_map_type(&c, &t, &s));
    EXPECT_EQ(t, bolt::BoltType::Timestamp);
    c.physical = PqType::FixedLenByteArray;        // FLBA without DECIMAL
    c.converted = -1;
    c.type_length = 16;
    EXPECT_FALSE(parquet_map_type(&c, &t, &s));
    c.converted = 5;                               // DECIMAL p>18 -> Dec128
    c.precision = 38;
    c.scale = 4;
    ASSERT_TRUE(parquet_map_type(&c, &t, &s));
    EXPECT_EQ(t, bolt::BoltType::Decimal128);
    EXPECT_EQ(s, 4);
    c.physical = PqType::Int64;                    // int64 DECIMAL mantissa
    c.scale = 2;
    ASSERT_TRUE(parquet_map_type(&c, &t, &s));
    EXPECT_EQ(t, bolt::BoltType::Decimal64);
    EXPECT_EQ(s, 2);
}

// ---- G2FEAT-346: KX NYSE TAQ rowgroup-shaped types ------------------------
//
// golden_taq_types.parquet: 9 cols, 1000 rows, UNCOMPRESSED, data page
// v1, row_group_size=256 -> 4 row groups. Column shapes mirror the real
// KX TAQ rowgroup files (parquet-cpp-arrow, UNCOMPRESSED, dict + PLAIN):
//   f32_plain  FLOAT PLAIN            == i * 0.5           -> Float64
//   f32_dict   FLOAT dict-encoded     == (i % 7) * 0.25    -> Float64
//   f32_null   FLOAT dict, nulls      == null if i%11==0 else i*0.5
//   flag       BOOLEAN PLAIN          == (i % 3 == 0)      -> Int64 0/1
//   flag_null  BOOLEAN PLAIN, nulls   == null if i%5==0 else (i%2==0)
//   u16        INT32 (UINT_16)        == i                 -> Int32
//   u64        INT64 (UINT_64)        == (1<<63) + i (bit pattern)
//   dur        INT64 (duration[ns])   == i * 1000000007
//   s_dict     BYTE_ARRAY dict        == "SYM%d" % (i % 5) -> Utf8
// golden_taq_types_rlebool.parquet: flag/flag_null written with
// column_encoding="RLE" (RLE-encoded BOOLEAN data pages) + u16 PLAIN.
//
// Regenerate goldens (documented generator):
//   python gen_taq_types.py  — inline here for the record:
//     import pyarrow as pa, pyarrow.parquet as pq
//     n = 1000
//     t = pa.table({
//       "f32_plain": pa.array([i*0.5 for i in range(n)], pa.float32()),
//       "f32_dict":  pa.array([(i%7)*0.25 for i in range(n)], pa.float32()),
//       "f32_null":  pa.array([None if i%11==0 else i*0.5
//                              for i in range(n)], pa.float32()),
//       "flag":      pa.array([i%3==0 for i in range(n)], pa.bool_()),
//       "flag_null": pa.array([None if i%5==0 else (i%2==0)
//                              for i in range(n)], pa.bool_()),
//       "u16":       pa.array([i%65536 for i in range(n)], pa.uint16()),
//       "u64":       pa.array([(1<<63)+i for i in range(n)], pa.uint64()),
//       "dur":       pa.array([i*1000000007 for i in range(n)],
//                             pa.duration('ns')),
//       "s_dict":    pa.array(["SYM%d" % (i%5) for i in range(n)],
//                             pa.string()),
//     })
//     pq.write_table(t, "golden_taq_types.parquet", row_group_size=256,
//                    compression="none", data_page_version="1.0",
//                    use_dictionary=["f32_dict","f32_null","s_dict"])
//     pq.write_table(t.select(["flag","flag_null","u16"]),
//                    "golden_taq_types_rlebool.parquet", row_group_size=256,
//                    compression="none", use_dictionary=False,
//                    data_page_version="1.0",
//                    column_encoding={"flag":"RLE","flag_null":"RLE",
//                                     "u16":"PLAIN"})

bool row_valid(const bolt::BoltColumn& col, int64_t i) {
    if (col.validity == nullptr) return true;
    return ((col.validity[i >> 3] >> (i & 7)) & 1u) != 0u;
}

TEST(BoltParquetRead, TaqTypesFloatBoolWidening) {
    const auto buf = slurp(data_path("golden_taq_types.parquet").c_str());
    ASSERT_FALSE(buf.empty());
    bolt::Arena arena;
    bolt::BoltBatch* b = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(b, nullptr);
    ASSERT_TRUE(parquet_read_file(buf.data(), buf.size(), &arena, b));
    ASSERT_EQ(b->num_rows, 1000);
    ASSERT_EQ(b->num_cols, 9u);
    const bolt::BoltColumn* cols = b->columns[b->read_epoch];
    ASSERT_EQ(cols[0].type, bolt::BoltType::Float64);   // f32_plain widened
    ASSERT_EQ(cols[1].type, bolt::BoltType::Float64);   // f32_dict widened
    ASSERT_EQ(cols[2].type, bolt::BoltType::Float64);   // f32_null widened
    ASSERT_EQ(cols[3].type, bolt::BoltType::Int64);     // flag 0/1
    ASSERT_EQ(cols[4].type, bolt::BoltType::Int64);     // flag_null 0/1
    ASSERT_EQ(cols[5].type, bolt::BoltType::Int32);     // u16
    ASSERT_EQ(cols[6].type, bolt::BoltType::Int64);     // u64 bit pattern
    ASSERT_EQ(cols[7].type, bolt::BoltType::Int64);     // duration[ns]
    ASSERT_EQ(cols[8].type, bolt::BoltType::Utf8);      // dict string
    const auto* f32p = static_cast<const double*>(cols[0].data);
    const auto* f32d = static_cast<const double*>(cols[1].data);
    const auto* f32n = static_cast<const double*>(cols[2].data);
    const auto* flag  = static_cast<const int64_t*>(cols[3].data);
    const auto* flagn = static_cast<const int64_t*>(cols[4].data);
    const auto* u16  = static_cast<const int32_t*>(cols[5].data);
    const auto* u64c = static_cast<const int64_t*>(cols[6].data);
    const auto* dur  = static_cast<const int64_t*>(cols[7].data);
    const auto* sd   = static_cast<const bolt::StringView*>(cols[8].data);
    ASSERT_NE(cols[2].validity, nullptr);
    ASSERT_NE(cols[4].validity, nullptr);
    for (int64_t i = 0; i < 1000; ++i) {
        ASSERT_EQ(f32p[i], static_cast<double>(static_cast<float>(i) * 0.5f))
            << "row " << i;
        ASSERT_EQ(f32d[i], static_cast<double>(i % 7) * 0.25) << "row " << i;
        ASSERT_EQ(row_valid(cols[2], i), i % 11 != 0) << "row " << i;
        if (i % 11 != 0) {
            ASSERT_EQ(f32n[i],
                      static_cast<double>(static_cast<float>(i) * 0.5f))
                << "row " << i;
        }
        ASSERT_EQ(flag[i], (i % 3 == 0) ? 1 : 0) << "row " << i;
        ASSERT_EQ(row_valid(cols[4], i), i % 5 != 0) << "row " << i;
        if (i % 5 != 0) {
            ASSERT_EQ(flagn[i], (i % 2 == 0) ? 1 : 0) << "row " << i;
        }
        ASSERT_EQ(u16[i], static_cast<int32_t>(i));
        ASSERT_EQ(u64c[i],
                  static_cast<int64_t>((uint64_t{1} << 63) +
                                       static_cast<uint64_t>(i)));
        ASSERT_EQ(dur[i], i * 1000000007);
        char want[16];
        std::snprintf(want, sizeof(want), "SYM%lld", (long long)(i % 5));
        ASSERT_EQ(sd[i].length, std::strlen(want)) << "row " << i;
        ASSERT_EQ(std::memcmp(sv_bytes(sd[i], cols[8].str_overflow_base),
                              want, sd[i].length), 0) << "row " << i;
    }
}

TEST(BoltParquetRead, TaqTypesRleBooleanDataPages) {
    const auto buf =
        slurp(data_path("golden_taq_types_rlebool.parquet").c_str());
    ASSERT_FALSE(buf.empty());
    bolt::Arena arena;
    bolt::BoltBatch* b = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(b, nullptr);
    ASSERT_TRUE(parquet_read_file(buf.data(), buf.size(), &arena, b));
    ASSERT_EQ(b->num_rows, 1000);
    ASSERT_EQ(b->num_cols, 3u);
    const bolt::BoltColumn* cols = b->columns[b->read_epoch];
    ASSERT_EQ(cols[0].type, bolt::BoltType::Int64);
    ASSERT_EQ(cols[1].type, bolt::BoltType::Int64);
    const auto* flag  = static_cast<const int64_t*>(cols[0].data);
    const auto* flagn = static_cast<const int64_t*>(cols[1].data);
    ASSERT_NE(cols[1].validity, nullptr);
    for (int64_t i = 0; i < 1000; ++i) {
        ASSERT_EQ(flag[i], (i % 3 == 0) ? 1 : 0) << "row " << i;
        ASSERT_EQ(row_valid(cols[1], i), i % 5 != 0) << "row " << i;
        if (i % 5 != 0) {
            ASSERT_EQ(flagn[i], (i % 2 == 0) ? 1 : 0) << "row " << i;
        }
    }
}

// Corrupt-page fuzzing: flips and truncations must never crash. The
// result may legitimately stay `true` when a flip lands in value bytes —
// the contract is "false or clean decode", enforced by ASAN.
TEST(BoltParquetRead, CorruptPagesNeverCrash) {
    const auto buf = slurp(data_path("golden_flat.parquet").c_str());
    ASSERT_FALSE(buf.empty());
    bolt::Arena arena;
    bolt::BoltBatch* b = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(b, nullptr);
    uint64_t meta_off = 0;
    uint32_t meta_len = 0;
    ASSERT_TRUE(pq_locate_footer(buf.data(), buf.size(), &meta_off,
                                 &meta_len));
    // 1) whole-file truncations (kills pages and/or the footer).
    for (size_t cut = 1; cut <= buf.size(); cut += 977) {
        arena.reset();
        (void)parquet_read_file(buf.data(), buf.size() - cut, &arena, b);
    }
    // 2) dense byte flips over the first 2 KiB (page headers + first
    //    pages) and strided flips across the whole page region.
    std::vector<uint8_t> mut(buf);
    const size_t page_end = static_cast<size_t>(meta_off);
    for (size_t i = 4; i < page_end; i = (i < 2048) ? i + 1 : i + 31) {
        mut[i] ^= 0xFF;
        arena.reset();
        (void)parquet_read_file(mut.data(), mut.size(), &arena, b);
        mut[i] ^= 0xFF;                            // restore
    }
    // 3) zero out each chunk's page region in turn (simulates torn IO).
    for (size_t z = 4; z + 64 < page_end; z += 512) {
        std::memset(mut.data() + z, 0, 64);
        arena.reset();
        (void)parquet_read_file(mut.data(), mut.size(), &arena, b);
        std::memcpy(mut.data() + z, buf.data() + z, 64);
    }
    SUCCEED();
}

// ---- G2FEAT-49: resumable single-column page decode -------------------------

// The single Utf8 (ByteArray) column in a golden file (-1 if none).
int find_utf8_col(const PqMeta* meta) {
    for (uint32_t c = 0; c < meta->n_columns; ++c) {
        bolt::BoltType t;
        uint8_t s = 0;
        if (parquet_map_type(&meta->columns[c], &t, &s) &&
            t == bolt::BoltType::Utf8) {
            return static_cast<int>(c);
        }
    }
    return -1;
}

PqMeta* parse_meta(const std::vector<uint8_t>& buf, bolt::Arena* arena) {
    PqMeta* meta = static_cast<PqMeta*>(
        arena->allocate(sizeof(PqMeta), alignof(PqMeta)));
    if (meta == nullptr) return nullptr;
    std::memset(meta, 0, sizeof(*meta));
    if (!parquet_read_meta(buf.data(), buf.size(), arena, meta)) return nullptr;
    return meta;
}

// Multi-page resumption: decode a chunk whole (reference) vs via small
// sub-chunks walking next_off. Values must match row-for-row (VALUE-level, not
// raw offsets — each sub-chunk owns its overflow) and >1 sub-chunk must be
// produced (golden_pageindex: ~100-row pages, 8-10 per chunk).
TEST(BoltParquetPageRange, MultiPageResumeByteExact) {
    const auto buf = slurp(data_path("golden_pageindex.parquet").c_str());
    ASSERT_FALSE(buf.empty());
    bolt::Arena arena;
    PqMeta* meta = parse_meta(buf, &arena);
    ASSERT_NE(meta, nullptr);
    const int uc = find_utf8_col(meta);
    ASSERT_GE(uc, 0);
    const uint16_t col = static_cast<uint16_t>(uc);
    for (uint32_t g = 0; g < meta->n_row_groups; ++g) {
        const uint16_t proj[1] = {col};
        bolt::BoltColumn whole[1];
        int64_t whole_rows = 0;
        ASSERT_TRUE(parquet_read_row_group_cols(buf.data(), buf.size(), meta, g,
                                                proj, 1, &arena, whole,
                                                &whole_rows));
        const auto* w = static_cast<const bolt::StringView*>(whole[0].data);
        uint64_t off = 0;
        int64_t seen = 0;
        int nchunks = 0;
        for (int guard = 0; guard < 100000; ++guard) {
            bolt::BoltColumn sub;
            int64_t srows = 0;
            uint64_t next = 0;
            ASSERT_TRUE(parquet_read_col_chunk_pages(buf.data(), buf.size(),
                                                     meta, g, col, off, 128,
                                                     &arena, &sub, &srows,
                                                     &next));
            if (srows == 0) {
                ASSERT_EQ(next, 0u);
                break;
            }
            ++nchunks;
            ASSERT_EQ(sub.type, bolt::BoltType::Utf8);
            const auto* s = static_cast<const bolt::StringView*>(sub.data);
            for (int64_t r = 0; r < srows; ++r) {
                ASSERT_LT(seen + r, whole_rows);
                const bolt::StringView& a = w[seen + r];
                const bolt::StringView& b = s[r];
                ASSERT_EQ(a.length, b.length) << "g" << g << " row " << seen + r;
                ASSERT_EQ(std::memcmp(sv_bytes(a, whole[0].str_overflow_base),
                                      sv_bytes(b, sub.str_overflow_base),
                                      a.length),
                          0)
                    << "g" << g << " row " << seen + r;
            }
            seen += srows;
            off = next;
            if (next == 0) break;
        }
        ASSERT_EQ(seen, whole_rows) << "rg " << g;
        ASSERT_GT(nchunks, 1) << "rg " << g << ": expected multi-page resumption";
    }
}

// Spill path: golden_flat_plain name has 28-char (>12B) spilled strings; the
// per-sub-chunk overflow (sized to page uncompressed bytes) must reproduce them.
TEST(BoltParquetPageRange, SpilledStringsByteExact) {
    const auto buf = slurp(data_path("golden_flat_plain.parquet").c_str());
    ASSERT_FALSE(buf.empty());
    bolt::Arena arena;
    PqMeta* meta = parse_meta(buf, &arena);
    ASSERT_NE(meta, nullptr);
    const int uc = find_utf8_col(meta);
    ASSERT_GE(uc, 0);
    const uint16_t col = static_cast<uint16_t>(uc);
    int64_t first = 0;
    for (uint32_t g = 0; g < meta->n_row_groups; ++g) {
        uint64_t off = 0;
        int64_t seen = 0;
        bool saw_spill = false;
        for (int guard = 0; guard < 100000; ++guard) {
            bolt::BoltColumn sub;
            int64_t srows = 0;
            uint64_t next = 0;
            ASSERT_TRUE(parquet_read_col_chunk_pages(buf.data(), buf.size(),
                                                     meta, g, col, off, 50,
                                                     &arena, &sub, &srows,
                                                     &next));
            if (srows == 0) break;
            const auto* s = static_cast<const bolt::StringView*>(sub.data);
            for (int64_t r = 0; r < srows; ++r) {
                const int64_t i = first + seen + r;
                expect_name(i, s[r], sub.str_overflow_base);   // golden law
                if (s[r].length > 12u) saw_spill = true;
            }
            seen += srows;
            off = next;
            if (next == 0) break;
        }
        ASSERT_EQ(seen, meta->row_groups[g].num_rows) << "rg " << g;
        first += seen;
        EXPECT_TRUE(saw_spill) << "rg " << g << ": expected >12B spilled strings";
    }
    ASSERT_EQ(first, 1000);
}

// Fail-closed: a dictionary-encoded chunk (golden_flat name) is rejected in v1.
// Was DictChunkFailClosed, pinning the resumable decoder's v1 "PLAIN only"
// limitation. That limitation is gone: dictionary encoding is what
// parquet-mr, Arrow and bolt's own writer emit by DEFAULT, so a PLAIN-only
// resumable decoder could not skip pages in most real files, which is the
// one thing it exists for. The boundary is still worth a test, so this now
// asserts the capability instead of the refusal -- and asserts VALUES, since
// a dictionary chunk resumed from the wrong place still returns plausible
// strings.
TEST(BoltParquetPageRange, DictChunkResumesAndMatchesWholeChunk) {
    const auto buf = slurp(data_path("golden_flat.parquet").c_str());
    ASSERT_FALSE(buf.empty());
    bolt::Arena arena;
    PqMeta* meta = parse_meta(buf, &arena);
    ASSERT_NE(meta, nullptr);
    const int uc = find_utf8_col(meta);
    ASSERT_GE(uc, 0);
    // The chunk really is dictionary encoded -- otherwise this test would
    // silently be exercising the PLAIN path it used to reject.
    ASSERT_GT(meta->chunks[uc].dictionary_page_offset, 0);

    // Whole-file decode of the same column, as the reference.
    bolt::Arena wa;
    bolt::BoltBatch* whole = wa.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(whole, nullptr);
    ASSERT_TRUE(parquet_read_file(buf.data(), buf.size(), &wa, whole));
    const bolt::BoltColumn& wc = whole->columns[whole->read_epoch][uc];
    const auto* wsv = static_cast<const bolt::StringView*>(wc.data);
    const auto* wsp = static_cast<const uint8_t*>(wc.str_overflow_base);
    ASSERT_NE(wsv, nullptr);

    // parquet_read_col_chunk_pages walks ONE row group's chunk, and this
    // fixture has four (256/256/256/232). Comparing the walk against the
    // whole FILE's row count would fail for a perfectly correct reader.
    const int64_t rg_rows = meta->row_groups[0].num_rows;
    ASSERT_GT(rg_rows, 0);
    ASSERT_LE(rg_rows, whole->num_rows);

    uint64_t off = 0;
    int64_t row = 0;
    int guard = 0;
    do {
        ASSERT_LT(++guard, 1000) << "resume loop did not terminate";
        bolt::Arena pa;
        bolt::BoltColumn sub;
        int64_t srows = 0;
        uint64_t next = 0;
        ASSERT_TRUE(parquet_read_col_chunk_pages(
            buf.data(), buf.size(), meta, 0, static_cast<uint16_t>(uc), off,
            128, &pa, &sub, &srows, &next)) << "resume failed at row " << row;
        if (srows == 0) break;
        const auto* sv = static_cast<const bolt::StringView*>(sub.data);
        const auto* sp = static_cast<const uint8_t*>(sub.str_overflow_base);
        ASSERT_NE(sv, nullptr);
        for (int64_t i = 0; i < srows; ++i) {
            const uint32_t wl = wsv[row + i].length;
            const uint32_t sl = sv[i].length;
            ASSERT_EQ(sl, wl) << "row " << (row + i) << " length";
            const uint8_t* wp = (wl <= 12u)
                ? reinterpret_cast<const uint8_t*>(&wsv[row + i].prefix[0])
                : (wsp + wsv[row + i].ref.offset);
            const uint8_t* pp = (sl <= 12u)
                ? reinterpret_cast<const uint8_t*>(&sv[i].prefix[0])
                : (sp + sv[i].ref.offset);
            ASSERT_EQ(0, std::memcmp(pp, wp, sl)) << "row " << (row + i);
        }
        row += srows;
        off = next;
    } while (off != 0);
    EXPECT_EQ(row, rg_rows) << "resume walk did not cover the chunk";
}

// ---- G2FEAT-111: UINT_32 above 2^31-1 -------------------------------------
//
// golden_uint32.parquet: 8 cols, 1000 rows, UNCOMPRESSED, data page v1,
// row_group_size=250 -> 4 row groups. Every uint32 column deliberately
// straddles the signed-int32 boundary, which is the case the pre-G2FEAT-111
// mapping (INT32-physical unsigned -> BoltType::Int32) could not represent:
// a value above 2^31-1 read back NEGATIVE. u8/u16/i32/u64 ride along as
// regression guards that this fix changed nothing else.
//
//   u32_plain  INT32 (UINT_32) PLAIN      == BIG[i % 7]
//   u32_hi     INT32 (UINT_32) PLAIN      == 4294967295 - i  (all > 2^31-1)
//   u32_dict   INT32 (UINT_32) dict       == BIG[(i//97) % 7]
//   u32_null   INT32 (UINT_32), nulls     == null if i%5==0 else BIG[i % 7]
//   u8_c       INT32 (UINT_8)             == i % 251          -> Int32
//   u16_c      INT32 (UINT_16)            == (i*37) % 65521   -> Int32
//   i32_c      INT32 signed               == (-1)^i * i*1000  -> Int32
//   u64_c      INT64 (UINT_64)            == 2^64-1 - i       -> Int64 (bit
//                                            pattern; see the u64 note below)
// with BIG = [0, 1, 2147483647, 2147483648, 3000000000, 4000000000,
//             4294967295].
//
// Every expected value below is DuckDB 1.5.4's answer on this exact file,
// not bolt's own output.
//
// Regenerate (documented generator):
//   import pyarrow as pa, pyarrow.parquet as pq
//   N = 1000
//   BIG = [0,1,2147483647,2147483648,3000000000,4000000000,4294967295]
//   t = pa.table({
//     "u32_plain": pa.array([BIG[i%7] for i in range(N)], pa.uint32()),
//     "u32_hi":    pa.array([4294967295-i for i in range(N)], pa.uint32()),
//     "u32_dict":  pa.array([BIG[(i//97)%7] for i in range(N)], pa.uint32()),
//     "u32_null":  pa.array([None if i%5==0 else BIG[i%7]
//                            for i in range(N)], pa.uint32()),
//     "u8_c":      pa.array([i%251 for i in range(N)], pa.uint8()),
//     "u16_c":     pa.array([(i*37)%65521 for i in range(N)], pa.uint16()),
//     "i32_c":     pa.array([(-1)**i*(i*1000) for i in range(N)], pa.int32()),
//     "u64_c":     pa.array([18446744073709551615-i for i in range(N)],
//                           pa.uint64()),
//   })
//   pq.write_table(t, "golden_uint32.parquet", row_group_size=250,
//                  compression="none", version="2.6",
//                  data_page_version="1.0", use_dictionary=["u32_dict"],
//                  write_statistics=True)
//
// u64 SCOPE NOTE (deliberately NOT asserted as a value): INT64-physical
// UINT_64 has no wider signed lane to widen into, so bolt keeps it on Int64
// and a u64 above 2^63-1 stays a raw bit pattern. That is the same class of
// gap G2FEAT-111 fixes for u32, but it is not fixable by widening — it needs
// a real BoltType::UInt64 lane and a downstream kernel audit (see the
// UNSIGNED WIDTHS note in test_bolt_parquet_types.cpp). golden_taq_types'
// u64 column already pins the bit-pattern behaviour; u64_c is here only so
// this fixture proves the fix did not disturb it.
TEST(BoltParquetRead, Uint32AboveSignedRange) {
    const auto buf = slurp(data_path("golden_uint32.parquet").c_str());
    ASSERT_FALSE(buf.empty());
    bolt::Arena arena;
    bolt::BoltBatch* b = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(b, nullptr);
    ASSERT_TRUE(parquet_read_file(buf.data(), buf.size(), &arena, b));
    ASSERT_EQ(b->num_rows, 1000);
    ASSERT_EQ(b->num_cols, 8u);
    const bolt::BoltColumn* cols = b->columns[b->read_epoch];

    // The mapping itself: u32 must land on a lane that HOLDS every value.
    EXPECT_EQ(cols[0].type, bolt::BoltType::Int64) << "u32_plain";
    EXPECT_EQ(cols[1].type, bolt::BoltType::Int64) << "u32_hi";
    EXPECT_EQ(cols[2].type, bolt::BoltType::Int64) << "u32_dict";
    EXPECT_EQ(cols[3].type, bolt::BoltType::Int64) << "u32_null";
    // Unchanged: u8/u16 fit signed 32 losslessly, signed i32 is exact.
    EXPECT_EQ(cols[4].type, bolt::BoltType::Int32) << "u8_c";
    EXPECT_EQ(cols[5].type, bolt::BoltType::Int32) << "u16_c";
    EXPECT_EQ(cols[6].type, bolt::BoltType::Int32) << "i32_c";
    EXPECT_EQ(cols[7].type, bolt::BoltType::Int64) << "u64_c";
    ASSERT_EQ(cols[0].type_size_bytes, 8u);
    ASSERT_EQ(cols[1].type_size_bytes, 8u);
    ASSERT_EQ(cols[2].type_size_bytes, 8u);
    ASSERT_EQ(cols[3].type_size_bytes, 8u);

    static const int64_t kBig[7] = {0, 1, 2147483647LL, 2147483648LL,
                                    3000000000LL, 4000000000LL, 4294967295LL};
    const auto* up = static_cast<const int64_t*>(cols[0].data);
    const auto* uh = static_cast<const int64_t*>(cols[1].data);
    const auto* ud = static_cast<const int64_t*>(cols[2].data);
    const auto* un = static_cast<const int64_t*>(cols[3].data);
    const auto* u8c = static_cast<const int32_t*>(cols[4].data);
    const auto* u16c = static_cast<const int32_t*>(cols[5].data);
    const auto* i32c = static_cast<const int32_t*>(cols[6].data);
    const auto* u64c = static_cast<const uint64_t*>(cols[7].data);
    ASSERT_NE(cols[3].validity, nullptr);

    int64_t sum_p = 0, sum_h = 0, sum_d = 0, sum_n = 0;
    int64_t sum_u8 = 0, sum_u16 = 0, sum_i32 = 0;
    int64_t n_valid = 0, n_above = 0;
    for (int64_t i = 0; i < 1000; ++i) {          // bounded: fixture rows
        ASSERT_EQ(up[i], kBig[i % 7]) << "u32_plain row " << i;
        ASSERT_EQ(uh[i], 4294967295LL - i) << "u32_hi row " << i;
        ASSERT_EQ(ud[i], kBig[(i / 97) % 7]) << "u32_dict row " << i;
        ASSERT_EQ(row_valid(cols[3], i), i % 5 != 0) << "u32_null row " << i;
        ASSERT_EQ(u8c[i], static_cast<int32_t>(i % 251)) << "u8_c row " << i;
        ASSERT_EQ(u16c[i], static_cast<int32_t>((i * 37) % 65521))
            << "u16_c row " << i;
        ASSERT_EQ(i32c[i], static_cast<int32_t>((i % 2 == 0 ? 1 : -1) * i * 1000))
            << "i32_c row " << i;
        ASSERT_EQ(u64c[i], 18446744073709551615ull - static_cast<uint64_t>(i))
            << "u64_c row " << i;
        sum_p += up[i];
        sum_h += uh[i];
        sum_d += ud[i];
        sum_u8 += u8c[i];
        sum_u16 += u16c[i];
        sum_i32 += i32c[i];
        if (i % 5 != 0) { sum_n += un[i]; ++n_valid; }
        if (up[i] > 2147483647LL) ++n_above;
    }
    // DuckDB 1.5.4 on this exact file.
    EXPECT_EQ(sum_p, 2225065679218LL);
    EXPECT_EQ(sum_h, 4294966795500LL);
    EXPECT_EQ(sum_d, 1784954078623LL);
    EXPECT_EQ(sum_n, 1782400027021LL);
    EXPECT_EQ(n_valid, 800);
    EXPECT_EQ(n_above, 571);
    EXPECT_EQ(sum_u8, 124506);
    EXPECT_EQ(sum_u16, 18481500);
    EXPECT_EQ(sum_i32, -500000);
}

}  // namespace
