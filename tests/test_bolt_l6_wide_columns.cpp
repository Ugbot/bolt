// test_bolt_l6_wide_columns.cpp — limits review L6: one column ceiling.
//
// Every bolt column surface takes any width up to bolt::kMaxColumns, sized to
// the data: Parquet write (wide_columns) -> Parquet read (PqMeta sized to the
// file), the bolt wire codec and WireStream, the Arrow IPC writer and the CSV
// parser. Before L6 these stopped at 256 / 128 / 256 / 64 / 1,024 columns and
// the Parquet reader could not open a file its own writer had produced.
//
// The files this writes (l6_wide_bolt.parquet, l6_wide.arrows) are checked by
// a real reader in scripts/l6_wide_columns_check.py, which also hands back a
// pyarrow-written file for ReadsPyarrowWideFile.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"
#include "bolt/ingest/bolt_arrow_ipc.h"
#include "bolt/ingest/bolt_csv.h"
#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt/ingest/bolt_parquet_write.h"
#include "bolt/wire/bolt_wire.h"
#include "bolt/wire/bolt_wire_stream.h"

namespace {

using namespace bolt::ingest::parquet;

constexpr std::uint32_t kWide = 4096;
constexpr std::int64_t  kRows = 300;
constexpr std::uint32_t kRowGroupRows = 100;   // 3 row groups -> 12,288 chunks

// The IPC writer has no Int32 mapping, so its stream uses Int64 there.
bool g_ipc_types = false;

bolt::BoltType type_of(std::uint32_t c) {
    if (g_ipc_types && (c % 4u) == 3u) return bolt::BoltType::Int64;
    switch (c % 4u) {
        case 0:  return bolt::BoltType::Int64;
        case 1:  return bolt::BoltType::Float64;
        case 2:  return bolt::BoltType::Utf8;
        default: return bolt::BoltType::Int32;
    }
}
bool nullable_of(std::uint32_t c) { return (c % 5u) == 4u; }
bool is_null(std::uint32_t c, std::int64_t r) {
    return nullable_of(c) && ((r + c) % 7) == 0;
}
std::int64_t i64_val(std::uint32_t c, std::int64_t r) {
    return r * 1000003 - static_cast<std::int64_t>(c) * 7;
}
double f64_val(std::uint32_t c, std::int64_t r) {
    return static_cast<double>(r) * 0.25 + static_cast<double>(c);
}
std::int32_t i32_val(std::uint32_t c, std::int64_t r) {
    return static_cast<std::int32_t>(r * 3 - static_cast<std::int64_t>(c));
}
std::string str_val(std::uint32_t c, std::int64_t r) {
    char buf[64];
    // Short values stay inline; every third row spills past 12 bytes.
    if (r % 3 == 0) {
        std::snprintf(buf, sizeof(buf), "long-value-c%u-r%lld",
                      c, static_cast<long long>(r));
    } else {
        std::snprintf(buf, sizeof(buf), "c%ur%lld", c, static_cast<long long>(r));
    }
    return buf;
}

void set_valid(std::uint8_t* bm, std::int64_t r) {
    bm[r >> 3] = static_cast<std::uint8_t>(bm[r >> 3] | (1u << (r & 7)));
}

// A kWide x kRows batch following the formulas above.
bolt::BoltBatch* make_wide_batch(bolt::Arena* a, std::uint32_t ncols) {
    auto* b = a->allocate_array<bolt::BoltBatch>(1);
    if (b == nullptr) return nullptr;
    bolt::BoltBatch::init_empty(b);
    if (!bolt::BoltBatch::alloc_columns(b, a, ncols)) return nullptr;
    b->num_rows = kRows;
    char name[32];
    for (std::uint32_t c = 0; c < ncols; ++c) {
        std::snprintf(name, sizeof(name), "col%05u", c);
        const bolt::BoltType t = type_of(c);
        b->schema.add_field(name, t, nullable_of(c));
        bolt::BoltColumn& col = b->columns[b->read_epoch][c];
        if (t == bolt::BoltType::Utf8) {
            col = bolt::BoltColumn::make_empty();
            col.length = kRows;
            col.format = bolt::ColumnFormat::Flat;
            col.type = t;
            col.type_size_bytes = sizeof(bolt::StringView);
            auto* svs = static_cast<bolt::StringView*>(a->allocate(
                static_cast<std::size_t>(kRows) * sizeof(bolt::StringView),
                alignof(bolt::StringView)));
            auto* spill = static_cast<std::uint8_t*>(a->allocate(64u * kRows, 8));
            if (svs == nullptr || spill == nullptr) return nullptr;
            std::memset(svs, 0, static_cast<std::size_t>(kRows) * sizeof(bolt::StringView));
            std::uint32_t off = 0;
            for (std::int64_t r = 0; r < kRows; ++r) {
                const std::string s = str_val(c, r);
                bolt::StringView& v = svs[r];
                v.length = static_cast<std::uint32_t>(s.size());
                if (s.size() <= 12u) {
                    std::memcpy(&v.prefix[0], s.data(), s.size());
                } else {
                    std::memcpy(&v.prefix[0], s.data(), 4);
                    std::memcpy(spill + off, s.data(), s.size());
                    v.ref.offset = off;
                    off += static_cast<std::uint32_t>(s.size());
                }
            }
            col.data = svs;
            col.str_overflow_base = spill;
            col.stats.all_valid = true;
        } else {
            col = bolt::BoltColumn::make_flat_alloc(kRows, t, a);
            if (col.data == nullptr) return nullptr;
            for (std::int64_t r = 0; r < kRows; ++r) {
                if (t == bolt::BoltType::Int64) {
                    static_cast<std::int64_t*>(col.data)[r] = i64_val(c, r);
                } else if (t == bolt::BoltType::Float64) {
                    static_cast<double*>(col.data)[r] = f64_val(c, r);
                } else {
                    static_cast<std::int32_t*>(col.data)[r] = i32_val(c, r);
                }
            }
        }
        if (nullable_of(c)) {
            const std::size_t nb = static_cast<std::size_t>((kRows + 7) / 8);
            auto* bm = static_cast<std::uint8_t*>(a->allocate(nb, 8));
            if (bm == nullptr) return nullptr;
            std::memset(bm, 0, nb);
            for (std::int64_t r = 0; r < kRows; ++r) {
                if (!is_null(c, r)) set_valid(bm, r);
            }
            col.validity = bm;
            col.validity_offset = 0;
            col.stats.all_valid = false;
        }
    }
    return b;
}

std::string sv_str(const bolt::BoltColumn& col, std::int64_t r) {
    const auto& v = static_cast<const bolt::StringView*>(col.data)[r];
    const char* p = (v.length <= 12u)
        ? &v.prefix[0]
        : reinterpret_cast<const char*>(col.str_overflow_base) + v.ref.offset;
    return std::string(p, v.length);
}

bool valid_at(const bolt::BoltColumn& col, std::int64_t r) {
    if (col.validity == nullptr) return true;
    const std::int64_t i = r + col.validity_offset;
    return (col.validity[i >> 3] >> (i & 7)) & 1u;
}

// Every cell of every column of `b` against the formulas (Int32 columns may
// come back widened by the reader; compare by value).
void expect_wide_values(const bolt::BoltBatch* b, std::uint32_t ncols) {
    ASSERT_EQ(b->num_cols, ncols);
    ASSERT_EQ(b->num_rows, kRows);
    std::uint32_t bad = 0;
    for (std::uint32_t c = 0; c < ncols && bad < 10; ++c) {
        const bolt::BoltColumn& col = b->col(c);
        for (std::int64_t r = 0; r < kRows && bad < 10; ++r) {
            if (is_null(c, r)) {
                if (valid_at(col, r)) { ++bad; ADD_FAILURE() << "c" << c << " r" << r << " not null"; }
                continue;
            }
            if (!valid_at(col, r)) { ++bad; ADD_FAILURE() << "c" << c << " r" << r << " null"; continue; }
            const bolt::BoltType t = type_of(c);
            bool ok = true;
            if (t == bolt::BoltType::Int64) {
                ok = static_cast<const std::int64_t*>(col.data)[r] == i64_val(c, r);
            } else if (t == bolt::BoltType::Float64) {
                ok = static_cast<const double*>(col.data)[r] == f64_val(c, r);
            } else if (t == bolt::BoltType::Utf8) {
                ok = sv_str(col, r) == str_val(c, r);
            } else if (col.type == bolt::BoltType::Int32) {
                ok = static_cast<const std::int32_t*>(col.data)[r] == i32_val(c, r);
            } else {
                ok = static_cast<const std::int64_t*>(col.data)[r] == i32_val(c, r);
            }
            if (!ok) { ++bad; ADD_FAILURE() << "value mismatch c" << c << " r" << r; }
        }
    }
    EXPECT_EQ(bad, 0u);
}

std::vector<std::uint8_t> slurp(const char* path) {
    std::vector<std::uint8_t> v;
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) return v;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    v.resize(static_cast<std::size_t>(n));
    if (std::fread(v.data(), 1, v.size(), f) != v.size()) v.clear();
    std::fclose(f);
    return v;
}

bool write_wide_parquet(const bolt::BoltBatch* b, std::uint32_t ncols,
                        const char* path) {
    ParquetWriteOpts o{};
    std::vector<ParquetWriteColumn> wide(ncols);
    ParquetWriteColumn* cols = pw_opts_columns(&o, ncols, wide.data());
    if (cols == nullptr) return false;
    o.compression = 1;
    o.emit_statistics = true;
    o.row_group_max_rows = kRowGroupRows;
    for (std::uint32_t c = 0; c < ncols; ++c) {
        std::snprintf(cols[c].name, sizeof(cols[c].name), "col%05u", c);
        cols[c].type = type_of(c);
        cols[c].nullable = nullable_of(c);
    }
    ParquetWriter* w = parquet_write_open(path, &o);
    if (w == nullptr) return false;
    if (!parquet_write_row_group(w, b)) { (void)parquet_write_close(w); return false; }
    return parquet_write_close(w);
}

}  // namespace

// Writer and reader agree at 4,096 columns (the reader used to stop at 128).
TEST(L6WideColumns, ParquetRoundTrip4096) {
    bolt::Arena a;
    bolt::BoltBatch* b = make_wide_batch(&a, kWide);
    ASSERT_NE(b, nullptr);
    ASSERT_TRUE(write_wide_parquet(b, kWide, "l6_wide_bolt.parquet"));
    const std::vector<std::uint8_t> file = slurp("l6_wide_bolt.parquet");
    ASSERT_FALSE(file.empty());

    bolt::Arena ra;
    auto* meta = ra.allocate_array<PqMeta>(1);
    ASSERT_NE(meta, nullptr);
    std::memset(meta, 0, sizeof(*meta));
    ASSERT_TRUE(parquet_read_meta(file.data(), file.size(), &ra, meta));
    EXPECT_EQ(meta->n_columns, kWide);
    EXPECT_EQ(meta->n_row_groups, 3u);
    EXPECT_EQ(meta->n_chunks, 3u * kWide);
    EXPECT_STREQ(meta->columns[kWide - 1].name, "col04095");

    auto* out = ra.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(out, nullptr);
    ASSERT_TRUE(parquet_read_file(file.data(), file.size(), &ra, out, nullptr));
    expect_wide_values(out, kWide);
}

// A narrow schema still takes the inline fast path, and a wide one that does
// not provide wide_columns is refused rather than read past the inline array.
TEST(L6WideColumns, WriterInlineVsWide) {
    ParquetWriteOpts o{};
    o.n_columns = kPwInlineColumns + 1u;
    for (std::uint32_t c = 0; c < kPwInlineColumns; ++c) {
        o.columns[c].type = bolt::BoltType::Int64;
        std::snprintf(o.columns[c].name, sizeof(o.columns[c].name), "c%u", c);
    }
    EXPECT_EQ(parquet_write_open_mem(&o), nullptr);
    std::vector<ParquetWriteColumn> big(bolt::kMaxColumns + 1u);
    EXPECT_EQ(pw_opts_columns(&o, bolt::kMaxColumns + 1u, big.data()), nullptr);
    EXPECT_NE(pw_opts_columns(&o, 3, nullptr), nullptr);   // inline: no storage
    EXPECT_EQ(o.wide_columns, nullptr);
}

// The ceiling is enforced in release too: a batch past it is refused.
TEST(L6WideColumns, BatchPastCeilingRefused) {
    bolt::Arena a;
    bolt::BoltBatch b;
    bolt::BoltBatch::init_empty(&b);
    EXPECT_FALSE(bolt::BoltBatch::alloc_columns(&b, &a, bolt::kMaxColumns + 1u));
    EXPECT_EQ(b.num_cols, 0u);
    EXPECT_TRUE(bolt::BoltBatch::alloc_columns(&b, &a, bolt::kMaxColumns));
    EXPECT_EQ(b.num_cols, bolt::kMaxColumns);
}

// bolt wire frames and WireStream carry 4,096 columns (both stopped at 256).
TEST(L6WideColumns, WireRoundTrip4096) {
    bolt::Arena a;
    bolt::BoltBatch* b = make_wide_batch(&a, kWide);
    ASSERT_NE(b, nullptr);
    const std::size_t sz = bolt::wire::bolt_wire_size(b);
    ASSERT_GT(sz, 0u);
    std::vector<std::uint8_t> buf(sz);
    ASSERT_EQ(bolt::wire::bolt_wire_serialize(b, buf.data(), buf.size()), sz);
    bolt::Arena ra;
    auto* out = ra.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(out, nullptr);
    ASSERT_TRUE(bolt::wire::bolt_wire_deserialize(buf.data(), buf.size(), out, &ra));
    expect_wide_values(out, kWide);

    // WireStream: fixed-width columns only (Phase 1), so an Int64-only frame.
    std::vector<bolt::BoltField> fields(kWide);
    std::vector<std::int64_t> vals(static_cast<std::size_t>(kRows));
    for (std::uint32_t c = 0; c < kWide; ++c) {
        std::memset(&fields[c], 0, sizeof(bolt::BoltField));
        fields[c].set_name(("s" + std::to_string(c)).c_str());
        fields[c].type = bolt::BoltType::Int64;
    }
    const std::size_t cap = 1u << 26;
    std::vector<std::uint8_t> sbuf(cap);
    bolt::WireStream s{};
    ASSERT_TRUE(bolt::wire_stream_begin_file(&s, sbuf.data(), cap, fields.data(), kWide));
    for (std::uint32_t c = 0; c < kWide; ++c) {
        for (std::int64_t r = 0; r < kRows; ++r) vals[static_cast<std::size_t>(r)] = i64_val(c, r);
        bolt::BoltColumn col = bolt::BoltColumn::make_empty();
        col.type = bolt::BoltType::Int64;
        col.format = bolt::ColumnFormat::Flat;
        col.length = kRows;
        ASSERT_TRUE(bolt::wire_stream_append_column(&s, &col, vals.data(),
                                                    vals.size() * sizeof(std::int64_t)));
    }
    const std::size_t n = bolt::wire_stream_finalize(&s);
    ASSERT_GT(n, 0u);
    auto* sout = ra.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(sout, nullptr);
    ASSERT_TRUE(bolt::wire::bolt_wire_deserialize(sbuf.data(), n, sout, &ra));
    ASSERT_EQ(sout->num_cols, kWide);
    for (std::uint32_t c = 0; c < kWide; c += 97u) {
        const auto* p = static_cast<const std::int64_t*>(sout->col(c).data);
        for (std::int64_t r = 0; r < kRows; r += 13) EXPECT_EQ(p[r], i64_val(c, r));
    }
}

// Arrow IPC stream of 4,096 columns (the writer stopped at 64); pyarrow reads
// it back in scripts/l6_wide_columns_check.py.
TEST(L6WideColumns, ArrowIpc4096) {
    bolt::Arena a;
    g_ipc_types = true;
    bolt::BoltBatch* b = make_wide_batch(&a, kWide);
    ASSERT_NE(b, nullptr);
    std::vector<bolt::BoltType> types(kWide);
    std::vector<std::string> names(kWide);
    std::vector<const char*> name_ptrs(kWide);
    for (std::uint32_t c = 0; c < kWide; ++c) {
        types[c] = type_of(c);
        names[c] = b->schema.fields[c].name;
        name_ptrs[c] = names[c].c_str();
    }
    std::FILE* f = std::fopen("l6_wide.arrows", "wb");
    ASSERT_NE(f, nullptr);
    auto* w = static_cast<bolt::ingest::ArrowIpcWriter*>(
        std::calloc(1, sizeof(bolt::ingest::ArrowIpcWriter)));
    ASSERT_NE(w, nullptr);
    ASSERT_TRUE(bolt::ingest::arrow_ipc_open(w, f, types.data(), name_ptrs.data(),
                                             static_cast<std::uint16_t>(kWide)));
    EXPECT_TRUE(bolt::ingest::arrow_ipc_write_batch(w, b));
    EXPECT_TRUE(bolt::ingest::arrow_ipc_close(w));
    EXPECT_EQ(w->mem, nullptr);
    std::fclose(f);
    std::free(w);
    g_ipc_types = false;
    EXPECT_GT(slurp("l6_wide.arrows").size(), static_cast<std::size_t>(kWide) * 100u);
}

// A Struct wider than its one-byte child count is refused, never truncated.
TEST(L6WideColumns, ArrowIpcStructChildrenBound) {
    std::vector<bolt::ingest::FieldSpec> kids(bolt::ingest::kIpcMaxStructChildren + 1u);
    for (auto& k : kids) { k = bolt::ingest::FieldSpec{}; k.type = bolt::BoltType::Int64; }
    bolt::ingest::FieldSpec top{};
    top.type = bolt::BoltType::Struct;
    top.n_children = static_cast<std::uint16_t>(kids.size());
    top.children = kids.data();
    std::FILE* f = std::tmpfile();
    ASSERT_NE(f, nullptr);
    bolt::ingest::ArrowIpcWriter w{};
    EXPECT_FALSE(bolt::ingest::arrow_ipc_open_nested(&w, f, &top, 1));
    EXPECT_EQ(w.mem, nullptr);
    top.n_children = bolt::ingest::kIpcMaxStructChildren;
    EXPECT_TRUE(bolt::ingest::arrow_ipc_open_nested(&w, f, &top, 1));
    EXPECT_TRUE(bolt::ingest::arrow_ipc_close(&w));
    std::fclose(f);
}

// CSV past the inline schema (the parser stopped at 1,024 columns).
TEST(L6WideColumns, Csv2000) {
    constexpr std::uint32_t kCsv = 2000;
    std::string text;
    for (std::int64_t r = 0; r < 3; ++r) {
        for (std::uint32_t c = 0; c < kCsv; ++c) {
            if (c != 0) text += ',';
            text += std::to_string(i64_val(c, r));
        }
        text += '\n';
    }
    std::vector<bolt::BoltType> types(kCsv, bolt::BoltType::Int64);
    bolt::ingest::CsvSchema s{};
    s.num_cols = kCsv;
    s.delimiter = ',';
    EXPECT_FALSE(bolt::ingest::csv_schema_width_ok(s));   // inline cannot hold it
    s.wide_types = types.data();
    ASSERT_TRUE(bolt::ingest::csv_schema_width_ok(s));
    bolt::Arena a;
    bolt::BoltBatch out;
    ASSERT_TRUE(bolt::ingest::parse_csv(text.data(), text.size(), s, &a, &out));
    ASSERT_EQ(out.num_cols, kCsv);
    ASSERT_EQ(out.num_rows, 3);
    for (std::uint32_t c = 0; c < kCsv; c += 37u) {
        const auto* p = static_cast<const std::int64_t*>(out.col(c).data);
        for (std::int64_t r = 0; r < 3; ++r) EXPECT_EQ(p[r], i64_val(c, r));
    }
}

// A pyarrow-written file of any width reads (scripts/l6_wide_columns_check.py
// sets BOLT_L6_PYARROW_WIDE; each column c is int64 with value r*c).
TEST(L6WideColumns, ReadsPyarrowWideFile) {
    const char* path = std::getenv("BOLT_L6_PYARROW_WIDE");
    if (path == nullptr) GTEST_SKIP() << "run via scripts/l6_wide_columns_check.py";
    const std::vector<std::uint8_t> file = slurp(path);
    ASSERT_FALSE(file.empty());
    bolt::Arena ra;
    auto* out = ra.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(out, nullptr);
    ASSERT_TRUE(parquet_read_file(file.data(), file.size(), &ra, out, nullptr));
    ASSERT_GT(out->num_cols, 4096u);
    for (std::uint32_t c = 0; c < out->num_cols; ++c) {
        const auto* p = static_cast<const std::int64_t*>(out->col(c).data);
        ASSERT_NE(p, nullptr);
        for (std::int64_t r = 0; r < out->num_rows; ++r) {
            ASSERT_EQ(p[r], r * static_cast<std::int64_t>(c)) << "c" << c << " r" << r;
        }
    }
    std::printf("read pyarrow file: %u columns x %lld rows\n", out->num_cols,
                static_cast<long long>(out->num_rows));
}
