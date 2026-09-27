// test_bolt_parquet_null_slots.cpp — WI-2: a NULL row's payload slot is
// defined. The reader decodes into an arena whose memory was pre-dirtied with
// 0xCC, so a decoder that skips a NULL row leaves garbage a consumer would read
// as a ~3.4 GiB string length (G2ICE-226's utf8_to_var_binary SIGSEGV). Every
// NULL slot must come back zero: plain and dictionary Utf8 (inline and
// spilled), plus fixed-width Int64 / Float64. Also: utf8_at_or_empty yields an
// empty span for a NULL row, and Arrow var_at never reads a NULL slot.

#include <cstdint>
#include <cstdio>
#include <cstring>

#include <gtest/gtest.h>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_arrow.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_null_slot.h"
#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt/ingest/bolt_parquet_write.h"

using bolt::ingest::parquet::ParquetWriteOpts;
using bolt::ingest::parquet::ParquetWriter;
using bolt::ingest::parquet::parquet_write_close_mem;
using bolt::ingest::parquet::parquet_write_open_mem;
using bolt::ingest::parquet::parquet_write_row_group;

namespace {

constexpr std::int64_t kRows = 40;

bool is_null_row(std::int64_t i) { return (i % 3) == 1; }

void set_valid(std::uint8_t* bm, std::int64_t i) {
    bm[i >> 3] = static_cast<std::uint8_t>(bm[i >> 3] | (1u << (i & 7)));
}

// id Int64 (nullable), v Float64 (nullable), s Utf8 (nullable, inline and
// spilled values drawn from 3 distinct strings so dictionary encoding holds).
void build_batch(bolt::Arena* arena, bolt::BoltBatch* out) {
    bolt::BoltBatch::init_empty(out);
    out->num_cols = 3;
    out->num_rows = kRows;
    bolt::BoltBatch::alloc_columns(out, arena, 3);
    out->schema.add_field("id", bolt::BoltType::Int64, true);
    out->schema.add_field("v", bolt::BoltType::Float64, true);
    out->schema.add_field("s", bolt::BoltType::Utf8, true);
    const std::size_t vb = (kRows + 7) / 8;
    auto* bm = static_cast<std::uint8_t*>(arena->allocate(vb, 8));
    ASSERT_NE(bm, nullptr);
    std::memset(bm, 0, vb);
    for (std::int64_t i = 0; i < kRows; ++i) if (!is_null_row(i)) set_valid(bm, i);

    bolt::BoltColumn& id = out->columns[out->read_epoch][0];
    id = bolt::BoltColumn::make_flat_alloc(kRows, bolt::BoltType::Int64, arena);
    bolt::BoltColumn& v = out->columns[out->read_epoch][1];
    v = bolt::BoltColumn::make_flat_alloc(kRows, bolt::BoltType::Float64, arena);
    ASSERT_NE(id.data, nullptr);
    ASSERT_NE(v.data, nullptr);
    for (std::int64_t i = 0; i < kRows; ++i) {
        static_cast<std::int64_t*>(id.data)[i] = 100 + i;
        static_cast<double*>(v.data)[i] = 0.5 + static_cast<double>(i);
    }
    bolt::BoltColumn& s = out->columns[out->read_epoch][2];
    s = bolt::BoltColumn::make_empty();
    s.length = kRows;
    s.format = bolt::ColumnFormat::Flat;
    s.type = bolt::BoltType::Utf8;
    s.type_size_bytes = sizeof(bolt::StringView);
    static char kLong[] = "a-string-long-enough-to-spill";
    static const char* kVals[3] = {"aa", "bb", kLong};
    auto* svs = static_cast<bolt::StringView*>(arena->allocate(
        kRows * sizeof(bolt::StringView), alignof(bolt::StringView)));
    ASSERT_NE(svs, nullptr);
    std::memset(svs, 0, kRows * sizeof(bolt::StringView));
    s.data = svs;
    s.str_overflow_base = kLong;
    for (std::int64_t i = 0; i < kRows; ++i) {
        const char* p = kVals[i % 3];
        const auto n = static_cast<std::uint32_t>(std::strlen(p));
        if (n <= 12) {
            svs[i] = bolt::StringView::from_cstr(p);
        } else {
            svs[i].length = n;
            std::memcpy(svs[i].prefix, p, 4);
            svs[i].ref.buf_idx = 0;
            svs[i].ref.offset = 0;
        }
    }
    for (bolt::BoltColumn* c : {&id, &v, &s}) {
        c->validity = bm;
        c->stats.all_valid = false;
    }
}

ParquetWriteOpts make_opts(bool dict) {
    ParquetWriteOpts o{};
    o.n_columns = 3;
    o.row_group_target_bytes = 1u << 20;
    o.use_dictionary = dict;
    const char* names[3] = {"id", "v", "s"};
    const bolt::BoltType types[3] = {bolt::BoltType::Int64, bolt::BoltType::Float64,
                                     bolt::BoltType::Utf8};
    for (int c = 0; c < 3; ++c) {
        std::strncpy(o.columns[c].name, names[c], sizeof(o.columns[c].name) - 1);
        o.columns[c].type = types[c];
        o.columns[c].nullable = true;
    }
    return o;
}

// Zero bytes normally; 0xA5 under BOLT_NULL_POISON=1. Never the 0xCC dirt.
bool slot_is_filler(const void* p, std::size_t n) {
    const auto* b = static_cast<const std::uint8_t*>(p);
    for (std::size_t i = 0; i < n; ++i) if (b[i] != bolt::null_slot_byte()) return false;
    return true;
}

void check_round_trip(bool dict) {
    bolt::Arena wa;
    auto* batch = wa.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(batch, nullptr);
    build_batch(&wa, batch);
    ParquetWriteOpts opts = make_opts(dict);
    ParquetWriter* w = parquet_write_open_mem(&opts);
    ASSERT_NE(w, nullptr);
    ASSERT_TRUE(parquet_write_row_group(w, batch));
    const std::uint8_t* mem = nullptr;
    std::uint64_t mem_len = 0;
    ASSERT_TRUE(parquet_write_close_mem(w, &wa, &mem, &mem_len));

    bolt::Arena ra;
    void* dirty = ra.allocate(8u << 20, 64);          // dirty the arena, then
    ASSERT_NE(dirty, nullptr);                        // hand the bytes back
    std::memset(dirty, 0xCC, 8u << 20);
    ra.reset();
    bolt::ingest::parquet::PqMeta meta{};
    ASSERT_TRUE(bolt::ingest::parquet::parquet_read_meta(mem, mem_len, &ra, &meta));
    auto* cols = ra.allocate_array<bolt::BoltColumn>(3);
    ASSERT_NE(cols, nullptr);
    std::int64_t rows = 0;
    ASSERT_TRUE(bolt::ingest::parquet::parquet_read_row_group(mem, mem_len, &meta, 0,
                                                              &ra, cols, &rows));
    ASSERT_EQ(rows, kRows);
    for (std::int64_t i = 0; i < kRows; ++i) {
        for (int c = 0; c < 3; ++c) ASSERT_EQ(cols[c].is_null(i), is_null_row(i)) << i;
        const auto* sv = static_cast<const bolt::StringView*>(cols[2].data) + i;
        const uint8_t* p = nullptr;
        int32_t len = -1;
        cols[2].utf8_at_or_empty(i, &p, &len);
        if (is_null_row(i)) {
            EXPECT_TRUE(slot_is_filler(static_cast<const std::int64_t*>(cols[0].data) + i,
                                       8)) << i;
            EXPECT_TRUE(slot_is_filler(static_cast<const double*>(cols[1].data) + i, 8))
                << i;
            EXPECT_TRUE(slot_is_filler(sv, sizeof(*sv))) << "row " << i;
            EXPECT_EQ(len, 0);
            ASSERT_NE(p, nullptr);
            const char* ap = nullptr;
            int32_t alen = -1;
            ASSERT_TRUE(bolt::arrow::detail::var_at(cols[2], i, &ap, &alen));
            EXPECT_EQ(alen, 0);
        } else {
            const char* want = (i % 3) == 0 ? "aa" : "a-string-long-enough-to-spill";
            ASSERT_EQ(len, static_cast<int32_t>(std::strlen(want))) << i;
            EXPECT_EQ(std::memcmp(p, want, static_cast<size_t>(len)), 0) << i;
            EXPECT_EQ(static_cast<const std::int64_t*>(cols[0].data)[i], 100 + i);
        }
    }
}

}  // namespace

TEST(ParquetNullSlots, PlainNullSlotsAreZeroInDirtyArena) { check_round_trip(false); }

TEST(ParquetNullSlots, DictionaryNullSlotsAreZeroInDirtyArena) { check_round_trip(true); }

TEST(ParquetNullSlots, NullSlotsFillRewritesOnlyNullRows) {
    bolt::Arena a;
    bolt::BoltColumn c = bolt::BoltColumn::make_flat_alloc(8, bolt::BoltType::Int64, &a);
    ASSERT_NE(c.data, nullptr);
    std::uint8_t bm[1] = {0x0F};                      // rows 4..7 NULL
    c.validity = bm;
    c.stats.all_valid = false;
    for (int i = 0; i < 8; ++i) static_cast<std::int64_t*>(c.data)[i] = -1;
    bolt::null_slots_fill(&c);
    for (int i = 0; i < 8; ++i) {
        const std::int64_t want = (i < 4) ? -1 : (bolt::null_poison_enabled()
                                                      ? static_cast<std::int64_t>(
                                                            0xA5A5A5A5A5A5A5A5ull)
                                                      : 0);
        EXPECT_EQ(static_cast<std::int64_t*>(c.data)[i], want) << i;
    }
}
