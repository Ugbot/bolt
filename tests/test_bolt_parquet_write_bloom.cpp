// Writer: split-block bloom filters, checked against bolt's own probe.
//
// A bloom filter is a claim that a value is ABSENT. bolt already had the
// reader (pq_read_bloom / pq_bloom_may_contain) and no writer, so nothing in
// the tree ever exercised the two against each other. That pairing is the
// whole risk: a builder that disagrees with the probe by a single bit turns
// a pruning optimisation into silently dropped rows, and it would look
// exactly like a working filter on any test that only asks "does it parse".
//
// So the properties asserted are the two that matter:
//
//   NO FALSE NEGATIVES -- every value actually written must probe present.
//     This is the correctness property; violating it drops real rows.
//   FALSE POSITIVES BOUNDED -- a large disjoint set must probe absent at
//     roughly the configured rate. This is the usefulness property; a filter
//     with every bit set never returns false and prunes nothing, which no
//     "does it parse" check can distinguish from a working one.

#include "bolt/ingest/bolt_parquet_write.h"
#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt/ingest/bolt_parquet_meta.h"
#include "bolt/ingest/bolt_parquet_bloom.h"

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

std::string tmp_path(const char* tag) {
    return std::string("test_bolt_parquet_write_bloom_") + tag + ".parquet";
}

void build_i64_batch(bolt::Arena* a, const std::vector<std::int64_t>& v,
                     bolt::BoltBatch* out) {
    const std::int64_t n = static_cast<std::int64_t>(v.size());
    bolt::BoltBatch::init_empty(out);
    out->num_cols = 1;
    out->num_rows = n;
    bolt::BoltBatch::alloc_columns(out, a, 1);
    out->schema.add_field("v", bolt::BoltType::Int64, false);
    bolt::BoltColumn& c = out->columns[out->read_epoch][0];
    c = bolt::BoltColumn::make_flat_alloc(n, bolt::BoltType::Int64, a);
    std::memcpy(c.data, v.data(), static_cast<std::size_t>(n) * 8u);
}

void build_str_batch(bolt::Arena* a, const std::vector<std::string>& vals,
                     bolt::BoltBatch* out) {
    const std::int64_t n = static_cast<std::int64_t>(vals.size());
    bolt::BoltBatch::init_empty(out);
    out->num_cols = 1;
    out->num_rows = n;
    bolt::BoltBatch::alloc_columns(out, a, 1);
    out->schema.add_field("v", bolt::BoltType::Utf8, false);
    bolt::BoltColumn& c = out->columns[out->read_epoch][0];
    c = bolt::BoltColumn::make_empty();
    c.length = n;
    c.format = bolt::ColumnFormat::Flat;
    c.type = bolt::BoltType::Utf8;
    c.type_size_bytes = sizeof(bolt::StringView);
    auto* svs = static_cast<bolt::StringView*>(
        a->allocate(static_cast<std::size_t>(n) * sizeof(bolt::StringView),
                    alignof(bolt::StringView)));
    std::memset(svs, 0, static_cast<std::size_t>(n) * sizeof(bolt::StringView));
    std::size_t need = 0;
    for (const auto& s : vals) if (s.size() > 12u) need += s.size();
    auto* spill = (need > 0)
        ? static_cast<std::uint8_t*>(a->allocate(need, 8)) : nullptr;
    std::size_t off = 0;
    for (std::int64_t i = 0; i < n; ++i) {
        const std::string& s = vals[static_cast<std::size_t>(i)];
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
    c.data = svs;
    c.str_overflow_base = spill;
    c.stats.all_valid = true;
}

ParquetWriteOpts bloom_opts(bolt::BoltType t, bool dict, double fpp = 0.0,
                            std::uint32_t max_bytes = 0) {
    ParquetWriteOpts o{};
    o.n_columns = 1;
    o.compression = 1;
    o.emit_statistics = true;
    o.emit_bloom_filter = true;
    o.use_dictionary = dict;
    o.bloom_fpp = fpp;
    o.bloom_max_bytes = max_bytes;
    std::strncpy(o.columns[0].name, "v", sizeof(o.columns[0].name) - 1);
    o.columns[0].type = t;
    o.columns[0].nullable = false;
    return o;
}

std::vector<std::uint8_t> write(bolt::BoltBatch* b, const ParquetWriteOpts& o,
                                const char* tag) {
    const std::string path = tmp_path(tag);
    ParquetWriter* w = parquet_write_open(path.c_str(), &o);
    EXPECT_NE(w, nullptr);
    if (w == nullptr) return {};
    EXPECT_TRUE(parquet_write_row_group(w, b));
    EXPECT_TRUE(parquet_write_close(w));
    return slurp_file(path.c_str());
}

// ---- the two properties --------------------------------------------------

TEST(BoltParquetWriteBloom, NoFalseNegativesInt64) {
    std::vector<std::int64_t> v(20000);
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = static_cast<std::int64_t>(i) * 2654435761ll - 999;
    }
    bolt::Arena a;
    auto* b = a.allocate_array<bolt::BoltBatch>(1);
    build_i64_batch(&a, v, b);
    const auto buf = write(b, bloom_opts(bolt::BoltType::Int64, false),
                           "i64_fn");
    ASSERT_FALSE(buf.empty());

    bolt::Arena ma;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &ma, &meta));
    ASSERT_GE(meta.n_chunks, 1u);
    PqBloomFilter bf{};
    ASSERT_TRUE(pq_read_bloom(buf.data(), buf.size(), meta.chunks[0], &bf));
    ASSERT_GE(bf.n_bytes, 32u);

    // The correctness property. One miss here means a reader pruning on an
    // equality predicate would skip a chunk that really contains the value.
    for (std::size_t i = 0; i < v.size(); ++i) {
        ASSERT_TRUE(pq_bloom_may_contain(bf, pq_bloom_hash_i64(v[i])))
            << "false negative at row " << i << " value " << v[i];
    }
}

TEST(BoltParquetWriteBloom, FalsePositiveRateIsNearTheTarget) {
    // A filter with every bit set has no false negatives either, and prunes
    // nothing. Only measuring the false-positive rate separates a working
    // filter from a saturated one.
    const std::size_t kN = 20000;
    std::vector<std::int64_t> v(kN);
    for (std::size_t i = 0; i < kN; ++i) {
        v[i] = static_cast<std::int64_t>(i) * 2;      // all even
    }
    bolt::Arena a;
    auto* b = a.allocate_array<bolt::BoltBatch>(1);
    build_i64_batch(&a, v, b);
    const auto buf = write(b, bloom_opts(bolt::BoltType::Int64, false, 0.05),
                           "i64_fpp");
    ASSERT_FALSE(buf.empty());
    bolt::Arena ma;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &ma, &meta));
    PqBloomFilter bf{};
    ASSERT_TRUE(pq_read_bloom(buf.data(), buf.size(), meta.chunks[0], &bf));

    // Probe a disjoint set: every odd value in the same range, none written.
    std::size_t hits = 0, probes = 0;
    for (std::size_t i = 0; i < kN; ++i) {
        const std::int64_t absent = static_cast<std::int64_t>(i) * 2 + 1;
        ++probes;
        if (pq_bloom_may_contain(bf, pq_bloom_hash_i64(absent))) ++hits;
    }
    const double rate = static_cast<double>(hits) / static_cast<double>(probes);
    // Target 0.05. The bound is loose deliberately -- this is a statistical
    // property over one sample, and a tight bound would make the test flaky
    // for no benefit. What it must catch is the two real failure modes:
    // a saturated filter (rate near 1.0) and a filter so large the sizing
    // formula is wrong by orders of magnitude (rate near 0 with a huge
    // bitset). Both are far outside this window.
    EXPECT_LT(rate, 0.20) << "filter prunes almost nothing (rate " << rate << ")";
    EXPECT_GT(rate, 0.0005) << "suspiciously perfect: rate " << rate;
}

TEST(BoltParquetWriteBloom, NoFalseNegativesUtf8) {
    std::vector<std::string> vals;
    char buf[64];
    for (std::size_t i = 0; i < 15000; ++i) {
        // Mix inline (<=12 byte) and spilled values -- the bloom hashes the
        // raw bytes with no length prefix, and the inline/spill split is
        // where a writer most easily hashes the wrong bytes.
        std::snprintf(buf, sizeof(buf),
                      (i % 3 == 0) ? "k%zu" : "a-much-longer-key-%zu", i);
        vals.push_back(buf);
    }
    bolt::Arena a;
    auto* b = a.allocate_array<bolt::BoltBatch>(1);
    build_str_batch(&a, vals, b);
    const auto fb = write(b, bloom_opts(bolt::BoltType::Utf8, true), "u8_fn");
    ASSERT_FALSE(fb.empty());
    bolt::Arena ma;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(fb.data(), fb.size(), &ma, &meta));
    PqBloomFilter bf{};
    ASSERT_TRUE(pq_read_bloom(fb.data(), fb.size(), meta.chunks[0], &bf));
    for (std::size_t i = 0; i < vals.size(); ++i) {
        const auto* p = reinterpret_cast<const std::uint8_t*>(vals[i].data());
        ASSERT_TRUE(pq_bloom_may_contain(
            bf, pq_bloom_hash_bytes(p, static_cast<std::uint32_t>(vals[i].size()))))
            << "false negative at row " << i << " value " << vals[i];
    }
    // And a disjoint set is mostly excluded.
    std::size_t hits = 0;
    for (std::size_t i = 0; i < 15000; ++i) {
        std::snprintf(buf, sizeof(buf), "absent-key-%zu", i);
        const auto* p = reinterpret_cast<const std::uint8_t*>(buf);
        if (pq_bloom_may_contain(
                bf, pq_bloom_hash_bytes(p, static_cast<std::uint32_t>(std::strlen(buf))))) {
            ++hits;
        }
    }
    EXPECT_LT(hits, 3000u) << "utf8 filter prunes almost nothing";
}

TEST(BoltParquetWriteBloom, DataStillReadsWithFiltersInterleaved) {
    // The filters are written between row groups, so they sit inside the
    // byte range a naive reader might walk as pages. total_compressed_size
    // must NOT include them, or a page walk runs off the end of the chunk.
    std::vector<std::int64_t> v(30000);
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = static_cast<std::int64_t>(i) * 3 - 7;
    }
    bolt::Arena a;
    auto* b = a.allocate_array<bolt::BoltBatch>(1);
    build_i64_batch(&a, v, b);
    auto o = bloom_opts(bolt::BoltType::Int64, true);
    o.row_group_max_rows = 7000;     // several row groups, filters between
    o.data_page_target_bytes = 4096; // several pages per chunk
    o.emit_page_index = true;
    const auto buf = write(b, o, "interleaved");
    ASSERT_FALSE(buf.empty());

    bolt::Arena ra;
    auto* rb = ra.allocate_array<bolt::BoltBatch>(1);
    ASSERT_TRUE(parquet_read_file(buf.data(), buf.size(), &ra, rb));
    ASSERT_EQ(rb->num_rows, static_cast<std::int64_t>(v.size()));
    const auto* p = static_cast<const std::int64_t*>(
        rb->columns[rb->read_epoch][0].data);
    ASSERT_NE(p, nullptr);
    for (std::size_t i = 0; i < v.size(); ++i) {
        ASSERT_EQ(p[i], v[i]) << "value at row " << i;
    }

    // Every row group's filter must be found and be sound for its own rows.
    bolt::Arena ma;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &ma, &meta));
    ASSERT_GT(meta.n_chunks, 1u);
    std::int64_t row = 0;
    for (std::uint32_t c = 0; c < meta.n_chunks; ++c) {
        PqBloomFilter bf{};
        ASSERT_TRUE(pq_read_bloom(buf.data(), buf.size(), meta.chunks[c], &bf))
            << "chunk " << c << " has no bloom filter";
        const std::int64_t n = meta.chunks[c].num_values;
        for (std::int64_t i = 0; i < n; ++i) {
            ASSERT_TRUE(pq_bloom_may_contain(
                bf, pq_bloom_hash_i64(v[static_cast<std::size_t>(row + i)])))
                << "chunk " << c << " false negative at row " << (row + i);
        }
        row += n;
    }
    EXPECT_EQ(row, static_cast<std::int64_t>(v.size()));
}

TEST(BoltParquetWriteBloom, BoolIsSkippedAndAbsenceIsClean) {
    // BOOLEAN gets no filter (a two-valued domain makes one useless), and a
    // chunk without one must report absence rather than a corrupt parse.
    bolt::Arena a;
    auto* b = a.allocate_array<bolt::BoltBatch>(1);
    const std::int64_t n = 1000;
    bolt::BoltBatch::init_empty(b);
    b->num_cols = 1;
    b->num_rows = n;
    bolt::BoltBatch::alloc_columns(b, &a, 1);
    b->schema.add_field("v", bolt::BoltType::Bool, false);
    bolt::BoltColumn& c = b->columns[b->read_epoch][0];
    c = bolt::BoltColumn::make_flat_alloc(n, bolt::BoltType::Bool, &a);
    auto* bp = static_cast<std::uint8_t*>(c.data);
    for (std::int64_t i = 0; i < n; ++i) bp[i] = static_cast<std::uint8_t>(i & 1);

    const auto buf = write(b, bloom_opts(bolt::BoltType::Bool, false), "bool");
    ASSERT_FALSE(buf.empty());
    bolt::Arena ma;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &ma, &meta));
    EXPECT_EQ(meta.chunks[0].bloom_filter_offset, 0);
    PqBloomFilter bf{};
    EXPECT_FALSE(pq_read_bloom(buf.data(), buf.size(), meta.chunks[0], &bf));

    bolt::Arena ra;
    auto* rb = ra.allocate_array<bolt::BoltBatch>(1);
    ASSERT_TRUE(parquet_read_file(buf.data(), buf.size(), &ra, rb));
    ASSERT_EQ(rb->num_rows, n);
}

TEST(BoltParquetWriteBloom, NotEmittedWhenNotRequested) {
    std::vector<std::int64_t> v(500);
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<std::int64_t>(i);
    bolt::Arena a;
    auto* b = a.allocate_array<bolt::BoltBatch>(1);
    build_i64_batch(&a, v, b);
    auto o = bloom_opts(bolt::BoltType::Int64, false);
    o.emit_bloom_filter = false;
    const auto buf = write(b, o, "off");
    ASSERT_FALSE(buf.empty());
    bolt::Arena ma;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &ma, &meta));
    EXPECT_EQ(meta.chunks[0].bloom_filter_offset, 0);
    PqBloomFilter bf{};
    EXPECT_FALSE(pq_read_bloom(buf.data(), buf.size(), meta.chunks[0], &bf));
}

TEST(BoltParquetWriteBloom, FilterSizeTracksTheTargetRate) {
    // A tighter fpp must produce a bigger filter; if the sizing formula were
    // ignored, every filter would come out the same size and the fpp option
    // would be decoration.
    std::vector<std::int64_t> v(50000);
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<std::int64_t>(i);
    std::uint32_t prev = 0;
    for (double fpp : {0.25, 0.05, 0.001}) {
        bolt::Arena a;
        auto* b = a.allocate_array<bolt::BoltBatch>(1);
        build_i64_batch(&a, v, b);
        const auto buf = write(
            b, bloom_opts(bolt::BoltType::Int64, false, fpp, 64u << 20), "size");
        ASSERT_FALSE(buf.empty());
        bolt::Arena ma;
        PqMeta meta{};
        ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &ma, &meta));
        PqBloomFilter bf{};
        ASSERT_TRUE(pq_read_bloom(buf.data(), buf.size(), meta.chunks[0], &bf));
        EXPECT_GT(bf.n_bytes, prev) << "fpp " << fpp << " did not grow the filter";
        prev = bf.n_bytes;
    }
}

TEST(BoltParquetWriteBloom, CapIsHonoured) {
    // An absurd NDV must not allocate unboundedly; the cap wins and the
    // filter stays sound (no false negatives), just more crowded.
    std::vector<std::int64_t> v(50000);
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = static_cast<std::int64_t>(i) * 7919;
    }
    bolt::Arena a;
    auto* b = a.allocate_array<bolt::BoltBatch>(1);
    build_i64_batch(&a, v, b);
    const auto buf = write(
        b, bloom_opts(bolt::BoltType::Int64, false, 0.0001, /*cap=*/4096u),
        "cap");
    ASSERT_FALSE(buf.empty());
    bolt::Arena ma;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &ma, &meta));
    PqBloomFilter bf{};
    ASSERT_TRUE(pq_read_bloom(buf.data(), buf.size(), meta.chunks[0], &bf));
    EXPECT_LE(bf.n_bytes, 4096u);
    for (std::size_t i = 0; i < v.size(); ++i) {
        ASSERT_TRUE(pq_bloom_may_contain(bf, pq_bloom_hash_i64(v[i])))
            << "capped filter dropped row " << i;
    }
}

// ---- the gate must discriminate -----------------------------------------

TEST(BoltParquetWriteBloom, DiscriminatingPower) {
    // If the probe returned true unconditionally, NoFalseNegatives would
    // pass no matter what the builder wrote. Show that the same probe used
    // there does return false for a value that was never inserted -- against
    // a filter deliberately sized generously so a false positive is
    // implausible.
    std::vector<std::int64_t> v(64);
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<std::int64_t>(i);
    bolt::Arena a;
    auto* b = a.allocate_array<bolt::BoltBatch>(1);
    build_i64_batch(&a, v, b);
    const auto buf = write(
        b, bloom_opts(bolt::BoltType::Int64, false, 0.0001, 1u << 20), "disc");
    ASSERT_FALSE(buf.empty());
    bolt::Arena ma;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &ma, &meta));
    PqBloomFilter bf{};
    ASSERT_TRUE(pq_read_bloom(buf.data(), buf.size(), meta.chunks[0], &bf));

    for (std::size_t i = 0; i < v.size(); ++i) {
        EXPECT_TRUE(pq_bloom_may_contain(bf, pq_bloom_hash_i64(v[i])));
    }
    std::size_t absent_reported = 0;
    for (std::int64_t k = 1000; k < 2000; ++k) {
        if (!pq_bloom_may_contain(bf, pq_bloom_hash_i64(k))) ++absent_reported;
    }
    // 64 values in a 1 MiB filter: essentially all 1000 probes must return
    // false. If the probe could not return false, this is 0.
    EXPECT_GT(absent_reported, 950u);
}

}  // namespace
