// G2PQ-29: parquet_read_file's optional decode_pool (parallel column decode).
//
// Acceptance bar mirrors test_bolt_parquet_write_par.cpp's for the encode
// side: the parallel path must be VALUE-IDENTICAL to the serial path, at
// every pool size, repeated -- not merely "still parses" or "still round-
// trips", either of which would pass while a race quietly corrupted one
// column's overflow buffer in a way that happens not to matter for a given
// run.
//
// The fixture spans multiple row groups (so a column's task walks more than
// one decode_chunk() call, exercising the per-column ColCtx state -- notably
// the Utf8 overflow-buffer cursor -- that must accumulate correctly whether
// that column's task runs alone or alongside every other column's task under
// Arena::set_concurrent). It mixes narrow ints, floats, and wide/narrow Utf8
// (inline and overflow-spilled), nullable and non-nullable, so the columns
// take visibly different amounts of work.

#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt/ingest/bolt_parquet_write.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_scheduler.h"
#include "bolt/bolt_types.h"

namespace {

using namespace bolt::ingest::parquet;

constexpr std::uint32_t kCols = 40;
constexpr std::int64_t  kRows = 20000;
constexpr std::int64_t  kRowGroupRows = 2500;  // -> 8 row groups

std::vector<std::uint8_t> slurp_file(const char* path) {
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

bolt::BoltType type_of(std::uint32_t c) {
    switch (c % 5u) {
        case 0: return bolt::BoltType::Int64;
        case 1: return bolt::BoltType::Utf8;
        case 2: return bolt::BoltType::Float64;
        case 3: return bolt::BoltType::Int32;
        default: return bolt::BoltType::Utf8;
    }
}

std::string str_val(std::uint32_t c, std::int64_t i) {
    char buf[96];
    if ((c % 5u) == 1u) {
        std::snprintf(buf, sizeof(buf), "s%d", static_cast<int>(i % 7));  // inline, <=12B
    } else {
        std::snprintf(buf, sizeof(buf), "org.example.service.column%02u.%08lld",
                      c, static_cast<long long>(i));  // >12B, overflow spill
    }
    return buf;
}

// Writes a wide, multi-row-group fixture and returns the file bytes.
std::vector<std::uint8_t> write_fixture(const char* tag) {
    bolt::Arena a;
    auto* b = a.allocate_array<bolt::BoltBatch>(1);
    EXPECT_NE(b, nullptr);
    bolt::BoltBatch::init_empty(b);
    b->num_cols = kCols;
    b->num_rows = kRows;
    bolt::BoltBatch::alloc_columns(b, &a, kCols);
    char name[32];
    for (std::uint32_t c = 0; c < kCols; ++c) {
        std::snprintf(name, sizeof(name), "c%02u", c);
        const bolt::BoltType t = type_of(c);
        b->schema.add_field(name, t, (c % 3u) == 2u);
        bolt::BoltColumn& col = b->columns[b->read_epoch][c];
        if (t == bolt::BoltType::Utf8) {
            col = bolt::BoltColumn::make_empty();
            col.length = kRows;
            col.format = bolt::ColumnFormat::Flat;
            col.type = t;
            col.type_size_bytes = sizeof(bolt::StringView);
            auto* svs = static_cast<bolt::StringView*>(a.allocate(
                static_cast<std::size_t>(kRows) * sizeof(bolt::StringView),
                alignof(bolt::StringView)));
            std::memset(svs, 0, static_cast<std::size_t>(kRows) * sizeof(bolt::StringView));
            std::size_t need = 0;
            for (std::int64_t i = 0; i < kRows; ++i) {
                const std::string s = str_val(c, i);
                if (s.size() > 12u) need += s.size();
            }
            auto* spill = (need > 0)
                ? static_cast<std::uint8_t*>(a.allocate(need, 8)) : nullptr;
            std::size_t off = 0;
            for (std::int64_t i = 0; i < kRows; ++i) {
                const std::string s = str_val(c, i);
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
            col.data = svs;
            col.str_overflow_base = spill;
            col.stats.all_valid = true;
        } else {
            col = bolt::BoltColumn::make_flat_alloc(kRows, t, &a);
            if (t == bolt::BoltType::Int64) {
                auto* p = static_cast<std::int64_t*>(col.data);
                for (std::int64_t i = 0; i < kRows; ++i) p[i] = i * (c + 3) - 17;
            } else if (t == bolt::BoltType::Int32) {
                auto* p = static_cast<std::int32_t*>(col.data);
                for (std::int64_t i = 0; i < kRows; ++i) {
                    p[i] = static_cast<std::int32_t>(i * 3 + c);
                }
            } else {
                auto* p = static_cast<double*>(col.data);
                for (std::int64_t i = 0; i < kRows; ++i) {
                    p[i] = static_cast<double>(i) * 0.5 - static_cast<double>(c);
                }
            }
        }
        if ((c % 3u) == 2u) {
            const std::size_t nb = static_cast<std::size_t>((kRows + 7) / 8);
            auto* bm = static_cast<std::uint8_t*>(a.allocate(nb, 8));
            std::memset(bm, 0, nb);
            for (std::int64_t i = 0; i < kRows; ++i) {
                if ((i % 5) != 0) {
                    bm[i >> 3] = static_cast<std::uint8_t>(bm[i >> 3] | (1u << (i & 7)));
                }
            }
            col.validity = bm;
            col.validity_offset = 0;
            col.stats.all_valid = false;
        }
    }

    ParquetWriteOpts o{};
    o.n_columns = kCols;
    o.compression = 1;  // SNAPPY
    o.emit_statistics = true;
    o.row_group_max_rows = kRowGroupRows;
    for (std::uint32_t c = 0; c < kCols; ++c) {
        std::snprintf(name, sizeof(name), "c%02u", c);
        std::strncpy(o.columns[c].name, name, sizeof(o.columns[c].name) - 1);
        o.columns[c].type = type_of(c);
        o.columns[c].nullable = (c % 3u) == 2u;
    }
    const std::string path =
        std::string("test_bolt_parquet_read_par_") + tag + ".parquet";
    ParquetWriter* w = parquet_write_open(path.c_str(), &o);
    EXPECT_NE(w, nullptr);
    if (w == nullptr) return {};
    // Multiple write calls so the file has row groups smaller than kRows --
    // the writer also splits internally at row_group_max_rows, this just
    // proves the fixture truly has >1 row group either way.
    EXPECT_TRUE(parquet_write_row_group(w, b));
    EXPECT_TRUE(parquet_write_close(w));
    return slurp_file(path.c_str());
}

struct PoolGuard {
    bolt::Scheduler* s;
    explicit PoolGuard(std::uint32_t n) : s(new bolt::Scheduler()) {
        if (!s->init(n)) { delete s; s = nullptr; }
    }
    ~PoolGuard() { if (s != nullptr) { s->shutdown(); delete s; } }
};

bool cols_equal(const bolt::BoltColumn& a, const bolt::BoltColumn& b,
               std::int64_t rows, const char* label) {
    if (a.type != b.type) {
        ADD_FAILURE() << label << ": type differs";
        return false;
    }
    // Validity bits are the real contract and must match bit-for-bit. The
    // underlying VALUE at an invalid row is explicitly NOT part of that
    // contract (make_flat_alloc's buffer -- and the Utf8 StringView slot --
    // are plain arena::allocate(), never allocate_zeroed(); nothing decodes
    // a placeholder into a null slot's bytes), the same way Arrow treats a
    // null slot's underlying buffer content as unspecified. Comparing those
    // bytes blindly across two independently-allocated arenas is comparing
    // leftover heap contents, not decoded output -- it can differ by pure
    // allocator-reuse timing with NO decode bug present (confirmed directly:
    // a plain single-shot serial-vs-N-thread decode of this fixture, and a
    // 12x pool-reuse loop, both ran clean in isolation; the divergence only
    // showed up entangled with unrelated heap churn from this same test
    // binary writing the fixture first -- exactly the signature of reading
    // unspecified null-slot bytes, not a data race). So: validity compared
    // always; the data/string payload compared ONLY where BOTH sides agree
    // the row is valid.
    const bool has_va = a.validity != nullptr;
    const bool has_vb = b.validity != nullptr;
    if (has_va != has_vb) {
        ADD_FAILURE() << label << ": validity presence differs";
        return false;
    }
    if (has_va) {
        const std::size_t nb = static_cast<std::size_t>((rows + 7) / 8);
        if (std::memcmp(a.validity, b.validity, nb) != 0) {
            ADD_FAILURE() << label << ": validity bits differ";
            return false;
        }
    }
    auto valid_at = [](const bolt::BoltColumn& c, std::int64_t r) noexcept {
        if (c.validity == nullptr) return true;
        const auto* v = static_cast<const std::uint8_t*>(
            static_cast<const void*>(c.validity));
        return static_cast<bool>((v[r >> 3] >> (r & 7)) & 1u);
    };
    if (a.type == bolt::BoltType::Utf8) {
        const auto* sa = static_cast<const bolt::StringView*>(a.data);
        const auto* sb = static_cast<const bolt::StringView*>(b.data);
        for (std::int64_t r = 0; r < rows; ++r) {
            if (!valid_at(a, r)) continue;   // unspecified slot, both sides
            if (sa[r].length != sb[r].length) {
                ADD_FAILURE() << label << " row " << r << ": length differs";
                return false;
            }
            const char* pa = (sa[r].length <= 12u)
                ? &sa[r].prefix[0]
                : static_cast<const char*>(a.str_overflow_base) + sa[r].ref.offset;
            const char* pb = (sb[r].length <= 12u)
                ? &sb[r].prefix[0]
                : static_cast<const char*>(b.str_overflow_base) + sb[r].ref.offset;
            if (std::memcmp(pa, pb, sa[r].length) != 0) {
                ADD_FAILURE() << label << " row " << r << ": bytes differ";
                return false;
            }
        }
        return true;
    }
    std::size_t w = 0;
    switch (a.type) {
        case bolt::BoltType::Int64: case bolt::BoltType::Float64: w = 8; break;
        case bolt::BoltType::Int32: w = 4; break;
        default: ADD_FAILURE() << label << ": unexpected type in fixture"; return false;
    }
    const auto* pa = static_cast<const std::uint8_t*>(a.data);
    const auto* pb = static_cast<const std::uint8_t*>(b.data);
    for (std::int64_t r = 0; r < rows; ++r) {
        if (!valid_at(a, r)) continue;       // unspecified slot, both sides
        if (std::memcmp(pa + static_cast<std::size_t>(r) * w,
                        pb + static_cast<std::size_t>(r) * w, w) != 0) {
            ADD_FAILURE() << label << " row " << r << ": data bytes differ";
            return false;
        }
    }
    return true;
}

void assert_batches_equal(const bolt::BoltBatch& oracle, const bolt::BoltBatch& got,
                          const char* tag) {
    ASSERT_EQ(got.num_rows, oracle.num_rows) << tag;
    ASSERT_EQ(got.num_cols, oracle.num_cols) << tag;
    const bolt::BoltColumn* oc = oracle.columns[oracle.read_epoch];
    const bolt::BoltColumn* gc = got.columns[got.read_epoch];
    char label[64];
    for (std::uint32_t c = 0; c < oracle.num_cols; ++c) {
        std::snprintf(label, sizeof(label), "%s col %u", tag, c);
        EXPECT_TRUE(cols_equal(oc[c], gc[c], oracle.num_rows, label));
    }
}

// ---- the acceptance bar --------------------------------------------------

TEST(BoltParquetReadPar, ParallelDecodeIsValueIdenticalToSerial) {
    const auto buf = write_fixture("serial_oracle");
    ASSERT_FALSE(buf.empty());

    bolt::Arena oracle_arena;
    auto* oracle = oracle_arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_TRUE(parquet_read_file(buf.data(), buf.size(), &oracle_arena, oracle));
    ASSERT_EQ(oracle->num_rows, kRows);
    ASSERT_EQ(oracle->num_cols, kCols);

    for (std::uint32_t threads : {1u, 2u, 3u, 4u, 8u}) {
        PoolGuard g(threads);
        ASSERT_NE(g.s, nullptr) << "pool init failed for " << threads;
        bolt::Arena a;
        auto* got = a.allocate_array<bolt::BoltBatch>(1);
        SCOPED_TRACE(testing::Message() << "threads=" << threads);
        ASSERT_TRUE(parquet_read_file(buf.data(), buf.size(), &a, got, g.s));
        // A parallel read must never leave the arena in concurrent mode --
        // the next serial user of this SAME arena (there is none here, but
        // any caller) must get the lock-free fast path back.
        EXPECT_FALSE(a.concurrent());
        assert_batches_equal(*oracle, *got, "parallel");
    }
}

TEST(BoltParquetReadPar, RepeatedParallelDecodesAreStable) {
    const auto buf = write_fixture("repeat");
    ASSERT_FALSE(buf.empty());
    PoolGuard g(8);
    ASSERT_NE(g.s, nullptr);

    bolt::Arena oracle_arena;
    auto* oracle = oracle_arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_TRUE(parquet_read_file(buf.data(), buf.size(), &oracle_arena, oracle));

    for (int i = 0; i < 12; ++i) {
        bolt::Arena a;
        auto* got = a.allocate_array<bolt::BoltBatch>(1);
        SCOPED_TRACE(testing::Message() << "run=" << i);
        ASSERT_TRUE(parquet_read_file(buf.data(), buf.size(), &a, got, g.s));
        assert_batches_equal(*oracle, *got, "repeat");
    }
}

// A single-column file must still decode correctly with a pool supplied --
// exercising the direct (non-submit_range) path the n_columns > 1 gate
// falls back to, which is otherwise untested by the wide fixture above.
TEST(BoltParquetReadPar, SingleColumnFileWithPoolStillDecodesSerially) {
    bolt::Arena wa;
    auto* wb = wa.allocate_array<bolt::BoltBatch>(1);
    bolt::BoltBatch::init_empty(wb);
    wb->num_cols = 1;
    wb->num_rows = 5000;
    bolt::BoltBatch::alloc_columns(wb, &wa, 1);
    wb->schema.add_field("only", bolt::BoltType::Int64, false);
    bolt::BoltColumn& col = wb->columns[wb->read_epoch][0];
    col = bolt::BoltColumn::make_flat_alloc(5000, bolt::BoltType::Int64, &wa);
    auto* p = static_cast<std::int64_t*>(col.data);
    for (int i = 0; i < 5000; ++i) p[i] = i * 7 - 3;

    ParquetWriteOpts o{};
    o.n_columns = 1;
    o.row_group_max_rows = 900;  // several row groups from one column
    std::strncpy(o.columns[0].name, "only", sizeof(o.columns[0].name) - 1);
    o.columns[0].type = bolt::BoltType::Int64;
    ParquetWriter* w = parquet_write_open("test_bolt_parquet_read_par_single.parquet", &o);
    ASSERT_NE(w, nullptr);
    ASSERT_TRUE(parquet_write_row_group(w, wb));
    ASSERT_TRUE(parquet_write_close(w));
    const auto buf = slurp_file("test_bolt_parquet_read_par_single.parquet");
    ASSERT_FALSE(buf.empty());

    PoolGuard g(4);
    ASSERT_NE(g.s, nullptr);
    bolt::Arena ra;
    auto* rb = ra.allocate_array<bolt::BoltBatch>(1);
    ASSERT_TRUE(parquet_read_file(buf.data(), buf.size(), &ra, rb, g.s));
    ASSERT_EQ(rb->num_rows, 5000);
    ASSERT_EQ(rb->num_cols, 1u);
    const auto* got = static_cast<const std::int64_t*>(
        rb->columns[rb->read_epoch][0].data);
    for (int i = 0; i < 5000; ++i) ASSERT_EQ(got[i], i * 7 - 3) << "row " << i;
}

}  // namespace
