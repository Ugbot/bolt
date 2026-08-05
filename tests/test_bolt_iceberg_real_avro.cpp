// test_bolt_iceberg_real_avro.cpp — read a REAL Iceberg table's metadata.
//
// PROVENANCE OF THE FIXTURES (this matters — a synthetic Avro file we wrote
// ourselves would prove nothing about real-world compatibility):
//   data/golden_iceberg_real_manifest_list.avro
//   data/golden_iceberg_real_manifest.avro
//   data/golden_iceberg_real_manifest_nulls.avro
// were produced by **pyiceberg 0.11.1** driving a real SqlCatalog warehouse —
// `catalog.create_table(...)` + two `table.append(pyarrow.Table)` calls — not
// by any bolt code. They are byte-for-byte what an Iceberg engine wrote:
// format-version 2, identity-partitioned on `sym`, `"avro.codec":"deflate"`.
//
// The source table (db.trades, schema id:long, sym:string, price:double,
// active:boolean; partitioned by identity(sym)):
//   append #1 -> 6 rows: AAPL(101.5,102.25) MSFT(310.0,311.75)
//                        GOOG(140.0, NULL price)      => 3 data files
//   append #2 -> 3 rows: AAPL(103.0) TSLA(250.5,251.0) => 2 data files
//
// Every expectation below is a VALUE from that table, not a row count, so a
// decoder that mis-aligns fields cannot pass by accident.

#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/ingest/bolt_avro.h"
#include "bolt/ingest/bolt_inflate.h"
#include "bolt/lakehouse/iceberg/manifest.h"

namespace {

using namespace bolt;
using namespace bolt::lakehouse::iceberg;

std::vector<uint8_t> load(const char* name) {
    std::string p = std::string(BOLT_TEST_DATA_DIR) + "/" + name;
    FILE* f = std::fopen(p.c_str(), "rb");
    if (f == nullptr) return {};
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(static_cast<size_t>(n > 0 ? n : 0));
    if (n > 0) {
        size_t got = std::fread(v.data(), 1, static_cast<size_t>(n), f);
        v.resize(got);
    }
    std::fclose(f);
    return v;
}

// Iceberg stores bounds in its binary single-value encoding: little-endian.
double bound_double(const ColumnStatEntry& c, bool upper) {
    const char* p = upper ? c.upper : c.lower;
    double d = 0.0;
    std::memcpy(&d, p, sizeof(d));
    return d;
}
int64_t bound_i64(const ColumnStatEntry& c, bool upper) {
    const char* p = upper ? c.upper : c.lower;
    int64_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    return v;
}
const ColumnStatEntry* col_of(const FileStats& st, int32_t field_id) {
    for (uint32_t i = 0; i < st.n_cols; ++i) {
        if (st.cols[i].field_id == field_id) return &st.cols[i];
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// The codec real Iceberg actually uses.
// ---------------------------------------------------------------------------

TEST(IcebergRealAvro, ManifestsAreDeflateCoded) {
    // Pins the premise of this whole file: if these fixtures were ever
    // regenerated as uncompressed Avro, the deflate path would silently stop
    // being covered and this test says so.
    auto raw = load("golden_iceberg_real_manifest.avro");
    ASSERT_FALSE(raw.empty());
    bolt::Arena arena;
    bolt::ingest::AvroHeader hdr{};
    uint64_t body = 0;
    ASSERT_TRUE(bolt::ingest::avro_read_header(raw.data(), raw.size(), &arena,
                                               &hdr, &body));
    EXPECT_EQ(hdr.codec, bolt::ingest::AvroCodec::kDeflate);
}

TEST(IcebergRealAvro, InflateReportsExactSizeAndRejectsGarbage) {
    // A zero-capacity call is a pure size probe; the reported size must be the
    // exact amount a retry needs (the property decompress_block relies on).
    auto raw = load("golden_iceberg_real_manifest.avro");
    ASSERT_FALSE(raw.empty());
    // Locate the first block payload via the header, then inflate it directly.
    bolt::Arena arena;
    bolt::ingest::AvroHeader hdr{};
    uint64_t off = 0;
    ASSERT_TRUE(bolt::ingest::avro_read_header(raw.data(), raw.size(), &arena,
                                               &hdr, &off));
    // block := count:long size:long payload sync
    auto rl = [&](uint64_t* o) {
        uint64_t r = 0, sh = 0;
        for (;;) {
            uint8_t b = raw[*o]; ++(*o);
            r |= static_cast<uint64_t>(b & 0x7Fu) << sh;
            if ((b & 0x80u) == 0u) break;
            sh += 7;
        }
        return static_cast<int64_t>((r >> 1) ^ (~(r & 1u) + 1u));
    };
    (void)rl(&off);                       // object count
    const int64_t bsz = rl(&off);
    ASSERT_GT(bsz, 0);

    uint64_t need = 0;
    int rc = bolt::ingest::inflate_raw(raw.data() + off,
                                       static_cast<uint64_t>(bsz), nullptr, 0,
                                       &need);
    ASSERT_EQ(rc, bolt::ingest::kInflateNeedMore);
    ASSERT_GT(need, 0u);

    std::vector<uint8_t> exact(static_cast<size_t>(need));
    uint64_t got = 0;
    rc = bolt::ingest::inflate_raw(raw.data() + off, static_cast<uint64_t>(bsz),
                                   exact.data(), exact.size(), &got);
    EXPECT_EQ(rc, bolt::ingest::kInflateOk);
    EXPECT_EQ(got, need);                 // the probe was exact, not a guess

    // Corrupt input must fail, never crash or silently return junk.
    std::vector<uint8_t> bad(static_cast<size_t>(bsz), 0xFFu);
    uint64_t n2 = 0;
    EXPECT_NE(bolt::ingest::inflate_raw(bad.data(), bad.size(), exact.data(),
                                        exact.size(), &n2),
              bolt::ingest::kInflateOk);
}

// ---------------------------------------------------------------------------
// Manifest list.
// ---------------------------------------------------------------------------

TEST(IcebergRealAvro, ManifestListParsesRealValues) {
    auto raw = load("golden_iceberg_real_manifest_list.avro");
    ASSERT_FALSE(raw.empty());
    bolt::Arena arena;
    ManifestListEntry ents[8];
    uint32_t n = 0;
    ASSERT_TRUE(manifest_list_parse_avro(raw.data(), raw.size(), &arena, ents,
                                         8, &n));
    // The current snapshot's list references both snapshots' manifests.
    ASSERT_EQ(n, 2u);

    // NOTE: snapshot ids, manifest lengths and file paths are assigned by the
    // writer and change every time the fixtures are regenerated, so they are
    // deliberately NOT hard-coded here. What IS reproducible is the shape the
    // source table dictates — and the cross-file invariant below ties the list
    // to the manifest more strongly than a magic number would.
    EXPECT_EQ(ents[0].added_files_count, 2);   // append #2 -> AAPL, TSLA
    EXPECT_EQ(ents[1].added_files_count, 3);   // append #1 -> AAPL, GOOG, MSFT
    for (uint32_t i = 0; i < n; ++i) {
        EXPECT_EQ(ents[i].partition_spec_id, 0);
        EXPECT_GT(ents[i].manifest_length, 0);
        EXPECT_NE(ents[i].added_snapshot_id, 0);
        EXPECT_NE(std::strstr(ents[i].manifest_path, ".avro"), nullptr);
    }
    // Two distinct appends => two distinct snapshots.
    EXPECT_NE(ents[0].added_snapshot_id, ents[1].added_snapshot_id);
}

TEST(IcebergRealAvro, ManifestListAndManifestAgreeOnSnapshotId) {
    // Cross-file invariant: the snapshot that ADDED a manifest (read from the
    // manifest list) is the snapshot stamped on that manifest's entries (read
    // from the manifest itself). Two independently-decoded files, two
    // different schemas — they can only agree if both decode correctly.
    auto list_raw = load("golden_iceberg_real_manifest_list.avro");
    auto man_raw  = load("golden_iceberg_real_manifest.avro");
    ASSERT_FALSE(list_raw.empty());
    ASSERT_FALSE(man_raw.empty());
    bolt::Arena arena;

    ManifestListEntry ents[8];
    uint32_t nl = 0;
    ASSERT_TRUE(manifest_list_parse_avro(list_raw.data(), list_raw.size(),
                                         &arena, ents, 8, &nl));
    ASSERT_EQ(nl, 2u);

    DataFileRef* files = arena.allocate_array<DataFileRef>(8);
    ASSERT_NE(files, nullptr);
    uint32_t nf = 0;
    ASSERT_TRUE(manifest_parse_avro(man_raw.data(), man_raw.size(), &arena, 0,
                                    files, 8, &nf));
    ASSERT_EQ(nf, 2u);
    // golden_iceberg_real_manifest.avro is the newest snapshot's manifest,
    // which is ents[0] — and its added_files_count must equal what it holds.
    EXPECT_EQ(static_cast<int64_t>(nf), ents[0].added_files_count);
    for (uint32_t i = 0; i < nf; ++i) {
        EXPECT_EQ(files[i].snapshot_id, ents[0].added_snapshot_id);
    }
}

// ---------------------------------------------------------------------------
// Manifest entries — including the repeated per-column statistics, which is
// the part per-element delivery exists for.
// ---------------------------------------------------------------------------

TEST(IcebergRealAvro, ManifestParsesDataFilesAndPartitions) {
    auto raw = load("golden_iceberg_real_manifest.avro");
    ASSERT_FALSE(raw.empty());
    bolt::Arena arena;
    DataFileRef* files = arena.allocate_array<DataFileRef>(8);
    ASSERT_NE(files, nullptr);
    uint32_t n = 0;
    ASSERT_TRUE(manifest_parse_avro(raw.data(), raw.size(), &arena, 0, files, 8,
                                    &n));
    ASSERT_EQ(n, 2u);                       // append #2 wrote AAPL + TSLA

    // Rows are ordered by partition; find them by partition value so the test
    // does not depend on writer ordering.
    const DataFileRef* aapl = nullptr;
    const DataFileRef* tsla = nullptr;
    for (uint32_t i = 0; i < n; ++i) {
        ASSERT_EQ(files[i].n_partition, 1u);          // identity(sym)
        ASSERT_TRUE(files[i].partition[0].is_str);
        if (std::strcmp(files[i].partition[0].str, "AAPL") == 0) aapl = &files[i];
        if (std::strcmp(files[i].partition[0].str, "TSLA") == 0) tsla = &files[i];
    }
    ASSERT_NE(aapl, nullptr);
    ASSERT_NE(tsla, nullptr);

    EXPECT_EQ(aapl->status, ManifestStatus::kAdded);
    EXPECT_NE(aapl->snapshot_id, 0);            // writer-assigned; see above
    EXPECT_EQ(aapl->stats.record_count, 1);          // one AAPL row in append #2
    EXPECT_EQ(tsla->stats.record_count, 2);          // two TSLA rows
    EXPECT_GT(aapl->stats.file_size_in_bytes, 0);
    EXPECT_NE(std::strstr(tsla->file_path, "sym=TSLA"), nullptr);
    EXPECT_NE(std::strstr(tsla->file_path, ".parquet"), nullptr);
}

TEST(IcebergRealAvro, ManifestParsesPerColumnBoundsFromArrayOfRecord) {
    auto raw = load("golden_iceberg_real_manifest.avro");
    ASSERT_FALSE(raw.empty());
    bolt::Arena arena;
    DataFileRef* files = arena.allocate_array<DataFileRef>(8);
    ASSERT_NE(files, nullptr);
    uint32_t n = 0;
    ASSERT_TRUE(manifest_parse_avro(raw.data(), raw.size(), &arena, 0, files, 8,
                                    &n));
    ASSERT_EQ(n, 2u);
    const DataFileRef* tsla = nullptr;
    for (uint32_t i = 0; i < n; ++i) {
        if (std::strcmp(files[i].partition[0].str, "TSLA") == 0) tsla = &files[i];
    }
    ASSERT_NE(tsla, nullptr);

    // Iceberg field ids from the schema: 1=id, 2=sym, 3=price, 4=active.
    // Stats exist at all only because array<record<key,value>> elements are
    // now DELIVERED rather than skipped.
    ASSERT_EQ(tsla->stats.n_cols, 4u);

    const ColumnStatEntry* id = col_of(tsla->stats, 1);
    ASSERT_NE(id, nullptr);
    ASSERT_TRUE(id->has_lower);
    ASSERT_TRUE(id->has_upper);
    EXPECT_EQ(id->lower_len, 8u);                     // long => 8 raw bytes
    EXPECT_EQ(bound_i64(*id, false), 8);              // TSLA rows are id 8, 9
    EXPECT_EQ(bound_i64(*id, true), 9);

    const ColumnStatEntry* sym = col_of(tsla->stats, 2);
    ASSERT_NE(sym, nullptr);
    EXPECT_EQ(std::string(sym->lower, sym->lower_len), "TSLA");
    EXPECT_EQ(std::string(sym->upper, sym->upper_len), "TSLA");

    // The case a NUL-terminated bound would have destroyed: 250.5 encodes to
    // 00 00 00 00 00 50 6F 40, whose first byte is NUL.
    const ColumnStatEntry* price = col_of(tsla->stats, 3);
    ASSERT_NE(price, nullptr);
    EXPECT_EQ(price->lower_len, 8u);
    EXPECT_DOUBLE_EQ(bound_double(*price, false), 250.5);
    EXPECT_DOUBLE_EQ(bound_double(*price, true), 251.0);

    const ColumnStatEntry* active = col_of(tsla->stats, 4);
    ASSERT_NE(active, nullptr);
    EXPECT_EQ(active->lower_len, 1u);
    EXPECT_EQ(active->lower[0], 0);                   // false
    EXPECT_EQ(active->upper[0], 1);                   // true

    // Nothing was null in append #2.
    EXPECT_EQ(id->null_count, 0);
    EXPECT_EQ(price->null_count, 0);
}

TEST(IcebergRealAvro, ManifestReportsRealNullCounts) {
    // append #1's GOOG partition holds (140.0, NULL) — so price has one null
    // and its bounds still cover only the non-null value.
    auto raw = load("golden_iceberg_real_manifest_nulls.avro");
    ASSERT_FALSE(raw.empty());
    bolt::Arena arena;
    DataFileRef* files = arena.allocate_array<DataFileRef>(8);
    ASSERT_NE(files, nullptr);
    uint32_t n = 0;
    ASSERT_TRUE(manifest_parse_avro(raw.data(), raw.size(), &arena, 0, files, 8,
                                    &n));
    ASSERT_EQ(n, 3u);                                  // AAPL, GOOG, MSFT

    const DataFileRef* goog = nullptr;
    for (uint32_t i = 0; i < n; ++i) {
        if (std::strcmp(files[i].partition[0].str, "GOOG") == 0) goog = &files[i];
    }
    ASSERT_NE(goog, nullptr);
    EXPECT_EQ(goog->stats.record_count, 2);

    const ColumnStatEntry* price = col_of(goog->stats, 3);
    ASSERT_NE(price, nullptr);
    EXPECT_EQ(price->null_count, 1);                   // the NULL price
    EXPECT_DOUBLE_EQ(bound_double(*price, false), 140.0);
    EXPECT_DOUBLE_EQ(bound_double(*price, true), 140.0);

    const ColumnStatEntry* sym = col_of(goog->stats, 2);
    ASSERT_NE(sym, nullptr);
    EXPECT_EQ(sym->null_count, 0);
}

// ---------------------------------------------------------------------------
// Discriminating controls — these must FAIL on a decoder that guesses.
// ---------------------------------------------------------------------------

TEST(IcebergRealAvro, TruncatedManifestIsRejectedNotGuessed) {
    auto raw = load("golden_iceberg_real_manifest.avro");
    ASSERT_FALSE(raw.empty());
    bolt::Arena arena;
    DataFileRef* files = arena.allocate_array<DataFileRef>(8);
    ASSERT_NE(files, nullptr);
    // Chop the file at a range of points; every one must fail closed, and none
    // may read out of bounds (ASAN/`/RTC` would catch that in CI).
    for (size_t cut = raw.size() / 4; cut < raw.size(); cut += raw.size() / 8) {
        uint32_t n = 0;
        const bool ok = manifest_parse_avro(raw.data(), cut, &arena, 0, files, 8,
                                            &n);
        EXPECT_FALSE(ok) << "truncation at " << cut << " was accepted";
    }
}

TEST(IcebergRealAvro, ManifestListRejectsAManifestFile) {
    // The two schemas are different records; binding by NAME means feeding the
    // wrong file yields no manifest_path and an explicit failure, rather than
    // positionally misreading one as the other.
    auto raw = load("golden_iceberg_real_manifest.avro");
    ASSERT_FALSE(raw.empty());
    bolt::Arena arena;
    ManifestListEntry ents[8];
    uint32_t n = 0;
    EXPECT_FALSE(manifest_list_parse_avro(raw.data(), raw.size(), &arena, ents,
                                          8, &n));
}

TEST(IcebergRealAvro, CapIsRespected) {
    auto raw = load("golden_iceberg_real_manifest_nulls.avro");
    ASSERT_FALSE(raw.empty());
    bolt::Arena arena;
    DataFileRef* files = arena.allocate_array<DataFileRef>(2);
    ASSERT_NE(files, nullptr);
    uint32_t n = 0;
    ASSERT_TRUE(manifest_parse_avro(raw.data(), raw.size(), &arena, 0, files, 1,
                                    &n));
    EXPECT_EQ(n, 1u);                                  // 3 available, cap 1
}

}  // namespace
