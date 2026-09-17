// PageHeader.crc verification (G2PQ-19).
//
// The spec defines PageHeader.crc (field 4, optional) as the CRC32 of the
// COMPRESSED page data, "used to indicate the corruption of the page" --
// see docs/research/parquet-spec-conformance.md A7. Most writers never emit
// it (parquet-mr's default writer doesn't; bolt's own writer doesn't
// either), so its absence must never be an error -- but when a writer DID
// emit it, a mismatch means real corruption and a conformant reader must
// REFUSE rather than silently decode (possibly wrong) bytes.
//
// Fixtures (scripts/make_page_crc_fixture.py, following the exact
// hand-thrift-construction precedent make_legacy_list_fixtures.py set for
// G2PQ-14, since pyarrow has no way to opt into writing page CRCs):
//   page_crc_ok.parquet      correct crc -- must read the real values
//   page_crc_bad.parquet     stored crc does not match the on-disk bytes
//                            (one byte flipped after the crc was computed)
//                            -- must be REFUSED
//   page_crc_absent.parquet  no crc field at all -- the common case, must
//                            read exactly like page_crc_ok.parquet
//
// The fixture script's own verify()/--inject self-test already proves (a)
// page_crc_ok.parquet is genuinely valid parquet (pyarrow reads the real
// values) and (b) pyarrow itself does NOT check PageHeader.crc (it silently
// returns the corrupted value from page_crc_bad.parquet without complaint)
// -- so this bolt-side test is the only place in the toolchain that can
// catch this class of corruption, which is exactly why it needs a test.

#include "bolt/ingest/bolt_parquet_read.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
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

std::string data_path(const char* name) {
#ifdef BOLT_TEST_DATA_DIR
    return std::string(BOLT_TEST_DATA_DIR) + "/" + name;
#else
    return std::string("tests/data/") + name;
#endif
}

const std::int64_t kWant[5] = {10, 20, 30, 40, 50};

}  // namespace

TEST(ParquetPageCrc, CorrectCrcReadsRealValues) {
    const auto buf = slurp(data_path("page_crc_ok.parquet").c_str());
    ASSERT_FALSE(buf.empty()) << "fixture missing -- run "
                                 "scripts/make_page_crc_fixture.py";
    bolt::Arena arena;
    bolt::BoltBatch* b = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(b, nullptr);
    ASSERT_TRUE(parquet_read_file(buf.data(), buf.size(), &arena, b))
        << "a correct PageHeader.crc must never cause a refusal";
    ASSERT_EQ(b->num_rows, 5);
    ASSERT_EQ(b->num_cols, 1u);
    const auto& col = b->columns[b->read_epoch][0];
    ASSERT_EQ(col.type, bolt::BoltType::Int64);
    const auto* v = static_cast<const std::int64_t*>(col.data);
    ASSERT_NE(v, nullptr);
    for (int i = 0; i < 5; ++i) EXPECT_EQ(v[i], kWant[i]) << "row " << i;
}

TEST(ParquetPageCrc, MismatchedCrcIsRefused) {
    const auto buf = slurp(data_path("page_crc_bad.parquet").c_str());
    ASSERT_FALSE(buf.empty()) << "fixture missing -- run "
                                 "scripts/make_page_crc_fixture.py";
    bolt::Arena arena;
    bolt::BoltBatch* b = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(b, nullptr);
    // The whole point: corrupted page bytes under a stale stored crc must
    // be refused, never silently decoded as [245, 20, 30, 40, 50] (which is
    // exactly what pyarrow does today, per the fixture script's own oracle
    // check -- pyarrow does not verify this field at all).
    EXPECT_FALSE(parquet_read_file(buf.data(), buf.size(), &arena, b))
        << "a PageHeader.crc mismatch must refuse the read, not decode "
           "corrupted bytes";
}

TEST(ParquetPageCrc, AbsentCrcFieldIsUnaffected) {
    // The overwhelmingly common case: no writer this reader has ever been
    // pointed at (including bolt's own) emits field 4 at all. Its absence
    // must never gate anything -- this is the regression guard for that.
    const auto buf = slurp(data_path("page_crc_absent.parquet").c_str());
    ASSERT_FALSE(buf.empty()) << "fixture missing -- run "
                                 "scripts/make_page_crc_fixture.py";
    bolt::Arena arena;
    bolt::BoltBatch* b = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(b, nullptr);
    ASSERT_TRUE(parquet_read_file(buf.data(), buf.size(), &arena, b))
        << "a page with no crc field must read exactly as if this feature "
           "did not exist";
    ASSERT_EQ(b->num_rows, 5);
    const auto& col = b->columns[b->read_epoch][0];
    const auto* v = static_cast<const std::int64_t*>(col.data);
    ASSERT_NE(v, nullptr);
    for (int i = 0; i < 5; ++i) EXPECT_EQ(v[i], kWant[i]) << "row " << i;
}
