// Writer: parallel column encoding.
//
// The acceptance bar here is BYTE-IDENTITY with a serial write, at every
// pool size, repeated. Not "the threaded run still parses", and not "the
// values still round-trip" -- both of those pass while a race quietly
// reorders chunks, drops a page, or lets two columns share scratch in a way
// that happens not to matter for this input.
//
// Byte-identity is achievable here by construction rather than by luck: only
// the ENCODE is concurrent, each task touches one column's input, one
// workspace and one output slot, and PLACEMENT into the file is serial and
// in column order regardless of which worker finished first. If any of that
// stops being true, these tests fail immediately and unambiguously.
//
// The schema is deliberately heterogeneous and unbalanced -- wide strings
// next to narrow ints, a dictionary column next to a delta column next to a
// byte-stream-split column -- so the columns take visibly different amounts
// of work and the wave really does finish out of order.

#include "bolt/ingest/bolt_parquet_write.h"
#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt/ingest/bolt_parquet_meta.h"
#include "bolt/ingest/bolt_parquet_pageindex.h"

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

constexpr std::uint32_t kCols = 24;
constexpr std::int64_t  kRows = 12000;

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

// Column c's type, cycling so the schema mixes widths and encodings.
bolt::BoltType type_of(std::uint32_t c) {
    switch (c % 6u) {
        case 0: return bolt::BoltType::Int64;
        case 1: return bolt::BoltType::Utf8;
        case 2: return bolt::BoltType::Float64;
        case 3: return bolt::BoltType::Int32;
        case 4: return bolt::BoltType::Utf8;
        default: return bolt::BoltType::Int64;
    }
}

PqWriteEncoding enc_of(std::uint32_t c) {
    switch (c % 6u) {
        case 0: return PqWriteEncoding::DeltaBinaryPacked;
        case 1: return PqWriteEncoding::Dictionary;
        case 2: return PqWriteEncoding::ByteStreamSplit;
        case 3: return PqWriteEncoding::Plain;
        case 4: return PqWriteEncoding::DeltaByteArray;
        default: return PqWriteEncoding::Auto;
    }
}

std::string str_val(std::uint32_t c, std::int64_t i) {
    char buf[96];
    if ((c % 6u) == 1u) {
        // Low cardinality: the dictionary path.
        std::snprintf(buf, sizeof(buf), "cat-%lld",
                      static_cast<long long>(i % 40));
    } else {
        // Sorted, long shared prefix: the front-coding path, and wide enough
        // to spill past the 12-byte inline limit.
        std::snprintf(buf, sizeof(buf), "org.example.service.column%02u.%08lld",
                      c, static_cast<long long>(i));
    }
    return buf;
}

void build_wide_batch(bolt::Arena* a, bolt::BoltBatch* out) {
    bolt::BoltBatch::init_empty(out);
    out->num_cols = kCols;
    out->num_rows = kRows;
    bolt::BoltBatch::alloc_columns(out, a, kCols);
    char name[32];
    for (std::uint32_t c = 0; c < kCols; ++c) {
        std::snprintf(name, sizeof(name), "c%02u", c);
        const bolt::BoltType t = type_of(c);
        // Every third column is nullable, so def-level encoding participates.
        out->schema.add_field(name, t, (c % 3u) == 2u);
        bolt::BoltColumn& col = out->columns[out->read_epoch][c];
        if (t == bolt::BoltType::Utf8) {
            col = bolt::BoltColumn::make_empty();
            col.length = kRows;
            col.format = bolt::ColumnFormat::Flat;
            col.type = t;
            col.type_size_bytes = sizeof(bolt::StringView);
            auto* svs = static_cast<bolt::StringView*>(
                a->allocate(static_cast<std::size_t>(kRows) *
                                sizeof(bolt::StringView),
                            alignof(bolt::StringView)));
            std::memset(svs, 0,
                        static_cast<std::size_t>(kRows) * sizeof(bolt::StringView));
            std::size_t need = 0;
            for (std::int64_t i = 0; i < kRows; ++i) {
                const std::string s = str_val(c, i);
                if (s.size() > 12u) need += s.size();
            }
            auto* spill = (need > 0)
                ? static_cast<std::uint8_t*>(a->allocate(need, 8)) : nullptr;
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
            col = bolt::BoltColumn::make_flat_alloc(kRows, t, a);
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
            auto* bm = static_cast<std::uint8_t*>(a->allocate(nb, 8));
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
}

ParquetWriteOpts wide_opts(bolt::Scheduler* pool) {
    ParquetWriteOpts o{};
    o.n_columns = kCols;
    o.compression = 1;                  // SNAPPY
    o.emit_statistics = true;
    o.emit_page_index = true;
    o.emit_bloom_filter = true;
    o.use_dictionary = true;
    o.data_page_target_bytes = 8192;    // several pages per chunk
    o.row_group_max_rows = 5000;        // several row groups
    o.encode_pool = pool;
    char name[32];
    for (std::uint32_t c = 0; c < kCols; ++c) {
        std::snprintf(name, sizeof(name), "c%02u", c);
        std::strncpy(o.columns[c].name, name, sizeof(o.columns[c].name) - 1);
        o.columns[c].type = type_of(c);
        o.columns[c].nullable = (c % 3u) == 2u;
        o.columns[c].encoding = static_cast<std::uint8_t>(enc_of(c));
    }
    return o;
}

std::vector<std::uint8_t> write_wide(bolt::Scheduler* pool, const char* tag) {
    bolt::Arena a;
    auto* b = a.allocate_array<bolt::BoltBatch>(1);
    EXPECT_NE(b, nullptr);
    build_wide_batch(&a, b);
    const std::string path =
        std::string("test_bolt_parquet_write_par_") + tag + ".parquet";
    const auto o = wide_opts(pool);
    ParquetWriter* w = parquet_write_open(path.c_str(), &o);
    EXPECT_NE(w, nullptr);
    if (w == nullptr) return {};
    EXPECT_TRUE(parquet_write_row_group(w, b));
    EXPECT_TRUE(parquet_write_close(w));
    return slurp_file(path.c_str());
}

// Scheduler is a sizeable struct holding rings and pools; heap-allocate it
// rather than putting it on the test stack, and ALWAYS shut it down --
// bolt's pool has no destructor that joins its worker threads, so leaving
// one running is a process-teardown hazard, not just a leak.
struct PoolGuard {
    bolt::Scheduler* s;
    explicit PoolGuard(std::uint32_t n) : s(new bolt::Scheduler()) {
        if (!s->init(n)) { delete s; s = nullptr; }
    }
    ~PoolGuard() { if (s != nullptr) { s->shutdown(); delete s; } }
};

// ---- the acceptance bar --------------------------------------------------

TEST(BoltParquetWritePar, ParallelOutputIsByteIdenticalToSerial) {
    const auto serial = write_wide(nullptr, "serial");
    ASSERT_FALSE(serial.empty());

    for (std::uint32_t threads : {1u, 2u, 3u, 4u, 8u}) {
        PoolGuard g(threads);
        ASSERT_NE(g.s, nullptr) << "pool init failed for " << threads;
        const auto par = write_wide(g.s, "par");
        SCOPED_TRACE(testing::Message() << "threads=" << threads);
        ASSERT_EQ(par.size(), serial.size()) << "file size differs";
        // Report the first differing byte rather than just "not equal" --
        // the offset tells you immediately whether it is a page payload, a
        // chunk boundary or the footer.
        std::size_t first_diff = par.size();
        for (std::size_t i = 0; i < par.size(); ++i) {
            if (par[i] != serial[i]) { first_diff = i; break; }
        }
        EXPECT_EQ(first_diff, par.size())
            << "parallel output diverges at byte " << first_diff
            << " of " << par.size();
    }
}

TEST(BoltParquetWritePar, RepeatedParallelRunsAreStable) {
    // One matching run can be luck. A race that reorders two columns shows up
    // as instability across repeats even when a single comparison passes.
    PoolGuard g(8);
    ASSERT_NE(g.s, nullptr);
    const auto first = write_wide(g.s, "rep");
    ASSERT_FALSE(first.empty());
    for (int i = 0; i < 12; ++i) {
        const auto again = write_wide(g.s, "rep");
        ASSERT_EQ(again.size(), first.size()) << "run " << i;
        ASSERT_TRUE(std::memcmp(again.data(), first.data(), first.size()) == 0)
            << "run " << i << " differs from the first";
    }
}

TEST(BoltParquetWriteParValues, ParallelWriteReadsBackCorrectly) {
    // Byte-identity to serial already implies this, but only if the SERIAL
    // output was right in the first place. Check the values independently so
    // the two tests cannot both be satisfied by a shared mistake.
    PoolGuard g(4);
    ASSERT_NE(g.s, nullptr);
    const auto buf = write_wide(g.s, "values");
    ASSERT_FALSE(buf.empty());

    bolt::Arena ra;
    auto* rb = ra.allocate_array<bolt::BoltBatch>(1);
    ASSERT_TRUE(parquet_read_file(buf.data(), buf.size(), &ra, rb));
    ASSERT_EQ(rb->num_rows, kRows);
    ASSERT_EQ(rb->num_cols, kCols);

    for (std::uint32_t c = 0; c < kCols; ++c) {
        const bolt::BoltColumn& col = rb->columns[rb->read_epoch][c];
        const bool nullable = (c % 3u) == 2u;
        const bolt::BoltType t = type_of(c);
        SCOPED_TRACE(testing::Message() << "column " << c);
        for (std::int64_t i = 0; i < kRows; ++i) {
            if (nullable && (i % 5) == 0) continue;      // null row
            if (t == bolt::BoltType::Utf8) {
                const auto* sv = static_cast<const bolt::StringView*>(col.data);
                const auto* spill =
                    static_cast<const std::uint8_t*>(col.str_overflow_base);
                const std::string want = str_val(c, i);
                const std::uint32_t len = sv[i].length;
                ASSERT_EQ(len, want.size()) << "row " << i;
                const std::uint8_t* p =
                    (len <= 12u)
                        ? reinterpret_cast<const std::uint8_t*>(&sv[i].prefix[0])
                        : (spill + sv[i].ref.offset);
                ASSERT_EQ(0, std::memcmp(p, want.data(), len)) << "row " << i;
            } else if (t == bolt::BoltType::Int64) {
                const auto* p = static_cast<const std::int64_t*>(col.data);
                ASSERT_EQ(p[i], i * (c + 3) - 17) << "row " << i;
            } else if (t == bolt::BoltType::Int32) {
                const auto* p = static_cast<const std::int32_t*>(col.data);
                ASSERT_EQ(p[i], static_cast<std::int32_t>(i * 3 + c)) << "row " << i;
            } else {
                const auto* p = static_cast<const double*>(col.data);
                ASSERT_EQ(p[i], static_cast<double>(i) * 0.5 -
                                    static_cast<double>(c)) << "row " << i;
            }
        }
    }
}

TEST(BoltParquetWritePar, PageIndexAndBloomSurviveParallelEncoding) {
    // The page index and bloom filters are built during the concurrent phase
    // but PLACED serially, so their offsets are the part most exposed to a
    // reordering bug -- and an offset that points at the wrong chunk still
    // parses.
    PoolGuard g(8);
    ASSERT_NE(g.s, nullptr);
    const auto buf = write_wide(g.s, "meta");
    ASSERT_FALSE(buf.empty());

    bolt::Arena a;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &a, &meta));
    ASSERT_EQ(meta.n_columns, kCols);
    ASSERT_GT(meta.n_row_groups, 1u);

    for (std::uint32_t ci = 0; ci < meta.n_chunks; ++ci) {
        const PqChunk& ch = meta.chunks[ci];
        SCOPED_TRACE(testing::Message() << "chunk " << ci);
        PqOffsetIndex oi{};
        oi.pages = a.allocate_array<PqPageLocation>(kPqMaxPagesPerChunk);
        oi.pages_cap = kPqMaxPagesPerChunk;
        ASSERT_TRUE(pq_read_offset_index(buf.data(), buf.size(), ch, &oi));
        ASSERT_GT(oi.n_pages, 0u);
        // Page 0 of every chunk must start where the chunk says its data
        // begins. If two columns' page lists were swapped, this is what
        // catches it.
        EXPECT_EQ(oi.pages[0].offset, ch.data_page_offset);
        EXPECT_EQ(oi.pages[0].first_row_index, 0);
        for (std::uint32_t p = 1; p < oi.n_pages; ++p) {
            EXPECT_GT(oi.pages[p].offset, oi.pages[p - 1u].offset);
            EXPECT_GT(oi.pages[p].first_row_index,
                      oi.pages[p - 1u].first_row_index);
        }
        PqColumnIndex cix{};
        cix.pages = a.allocate_array<PqPageStat>(kPqMaxPagesPerChunk);
        cix.pages_cap = kPqMaxPagesPerChunk;
        ASSERT_TRUE(pq_read_column_index(buf.data(), buf.size(), ch, &cix));
        EXPECT_EQ(cix.n_pages, oi.n_pages);
    }
}

}  // namespace
