// test_bolt_arrow_ipc.cpp — Arrow IPC stream writer (G2ARROW-10).
//
// Structural gates run here; the VALUE gate is the pyarrow oracle
// (scripts/arrow_ipc_check.py) over the fixture this test writes —
// pyarrow.ipc.open_stream() must read the bolt-written bytes and every
// value must match the generating rule re-derived in python. The
// fixture path is <BOLT_TEST_DATA_DIR>/../out/arrow_ipc_fixture.arrows
// unless BOLT_ARROW_IPC_OUT overrides it.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"
#include "bolt/ingest/bolt_arrow_ipc.h"

using bolt::BoltBatch;
using bolt::BoltColumn;
using bolt::BoltType;
using bolt::ColumnFormat;
using bolt::StringView;
using namespace bolt::ingest;

namespace {

// The one deterministic generating rule; the python oracle re-derives
// the same values independently.
constexpr std::int64_t kRows = 100;

std::int64_t int_val(std::int64_t i) { return i * 3 - 50; }
double dbl_val(std::int64_t i) { return static_cast<double>(i) * 0.5 - 10.0; }
// Row i string: "s<i>" for short rows; every 7th row is a long spilled
// (>12 byte) string; every 13th row is NULL.
bool row_is_null(std::int64_t i) { return (i % 13) == 0; }
std::string str_val(std::int64_t i) {
    if ((i % 7) == 0) {
        return "spilled-string-value-" + std::to_string(i) + "-padpadpad";
    }
    return "s" + std::to_string(i);
}

StringView make_view(const char* p, std::uint32_t len, char* pool,
                     std::uint32_t* pool_used) {
    StringView v{};
    v.length = len;
    const std::uint32_t np = (len < 4u) ? len : 4u;
    std::memcpy(v.prefix, p, np);
    if (len <= 12) {
        if (len > 4) std::memcpy(v.inline_data, p + 4, len - 4);
        return v;
    }
    v.ref.buf_idx = 0;
    v.ref.offset = *pool_used;
    std::memcpy(pool + *pool_used, p, len);
    *pool_used += len;
    return v;
}

struct Fixture {
    std::int64_t i64[kRows];
    double       f64[kRows];
    StringView   views[kRows];
    char         pool[8192];
    std::uint32_t pool_used = 0;
    std::uint8_t validity[(kRows + 7) / 8];
    bolt::Arena  arena;   // owns the batch's dynamic column arrays
    BoltBatch*   batch;   // heap: a stack BoltBatch is this repo's trap

    Fixture() {
        std::memset(validity, 0xFF, sizeof(validity));
        for (std::int64_t i = 0; i < kRows; ++i) {
            i64[i] = int_val(i);
            f64[i] = dbl_val(i);
            if (row_is_null(i)) {
                validity[i >> 3] &= static_cast<std::uint8_t>(
                    ~(1u << (i & 7)));
                views[i] = StringView{};   // length 0
            } else {
                const std::string s = str_val(i);
                views[i] = make_view(s.data(),
                                     static_cast<std::uint32_t>(s.size()),
                                     pool, &pool_used);
            }
        }
        batch = static_cast<BoltBatch*>(std::calloc(1, sizeof(BoltBatch)));
        BoltBatch::init_empty(batch);
        const bool cols_ok = BoltBatch::alloc_columns(batch, &arena, 3);
        assert(cols_ok);
        (void)cols_ok;
        batch->num_rows = kRows;
        batch->num_cols = 3;
        BoltColumn& c0 = batch->columns[batch->read_epoch][0];
        c0.type = BoltType::Int64;   c0.format = ColumnFormat::Flat;
        c0.data = i64; c0.length = kRows; c0.type_size_bytes = 8;
        BoltColumn& c1 = batch->columns[batch->read_epoch][1];
        c1.type = BoltType::Float64; c1.format = ColumnFormat::Flat;
        c1.data = f64; c1.length = kRows; c1.type_size_bytes = 8;
        BoltColumn& c2 = batch->columns[batch->read_epoch][2];
        c2.type = BoltType::Utf8;    c2.format = ColumnFormat::Flat;
        c2.data = views; c2.length = kRows;
        c2.str_overflow_base = pool;
        c2.validity = validity;
    }
    ~Fixture() { std::free(batch); }
};

std::string fixture_path() {
    const char* env = std::getenv("BOLT_ARROW_IPC_OUT");
    if (env != nullptr && env[0] != '\0') return env;
    // Default: cwd (the build dir under ctest) — never the source tree.
    return "arrow_ipc_fixture.arrows";
}

}  // namespace

TEST(ArrowIpc, RejectsUnsupportedTypeAtOpen) {
    auto* w = static_cast<ArrowIpcWriter*>(
        std::calloc(1, sizeof(ArrowIpcWriter)));
    ASSERT_NE(w, nullptr);
    std::FILE* f = std::tmpfile();
    ASSERT_NE(f, nullptr);
    const BoltType bad[2] = {BoltType::Int64, BoltType::Bool};
    EXPECT_FALSE(arrow_ipc_open(w, f, bad, nullptr, 2));
    // Fail closed AT open: nothing was written.
    std::fflush(f);
    EXPECT_EQ(std::ftell(f), 0);
    std::fclose(f);
    std::free(w);
}

TEST(ArrowIpc, SchemaMismatchFailsClosed) {
    auto* w = static_cast<ArrowIpcWriter*>(
        std::calloc(1, sizeof(ArrowIpcWriter)));
    std::FILE* f = std::tmpfile();
    ASSERT_NE(f, nullptr);
    const BoltType tys[1] = {BoltType::Float64};
    ASSERT_TRUE(arrow_ipc_open(w, f, tys, nullptr, 1));
    Fixture fx;                       // batch is (Int64, Float64, Utf8)
    EXPECT_FALSE(arrow_ipc_write_batch(w, fx.batch));   // col-count drift
    EXPECT_FALSE(arrow_ipc_close(w)); // failure is latched
    std::fclose(f);
    std::free(w);
}

TEST(ArrowIpc, StreamStructureAndEos) {
    auto* w = static_cast<ArrowIpcWriter*>(
        std::calloc(1, sizeof(ArrowIpcWriter)));
    std::FILE* f = std::tmpfile();
    ASSERT_NE(f, nullptr);
    Fixture fx;
    const BoltType tys[3] = {BoltType::Int64, BoltType::Float64,
                             BoltType::Utf8};
    const char* names[3] = {"ints", "floats", "strs"};
    ASSERT_TRUE(arrow_ipc_open(w, f, tys, names, 3));
    ASSERT_TRUE(arrow_ipc_write_batch(w, fx.batch));
    ASSERT_TRUE(arrow_ipc_close(w));

    const long total = std::ftell(f);
    ASSERT_GT(total, 16);
    EXPECT_EQ(total % 8, 0);          // every message is 8-aligned
    std::rewind(f);
    std::uint32_t head[2] = {0, 0};
    ASSERT_EQ(std::fread(head, 4, 2, f), 2u);
    EXPECT_EQ(head[0], 0xFFFFFFFFu);  // continuation marker
    EXPECT_EQ(head[1] % 8, 0u);       // padded metadata length
    ASSERT_EQ(std::fseek(f, total - 8, SEEK_SET), 0);
    std::uint32_t tail[2] = {0, 0};
    ASSERT_EQ(std::fread(tail, 4, 2, f), 2u);
    EXPECT_EQ(tail[0], 0xFFFFFFFFu);  // end-of-stream marker
    EXPECT_EQ(tail[1], 0u);
    std::fclose(f);
    std::free(w);
}

// Writes the fixture the pyarrow oracle (scripts/arrow_ipc_check.py)
// validates value-for-value. Two batches so the oracle also proves
// multi-batch framing.
TEST(ArrowIpc, WritesPyarrowOracleFixture) {
    const std::string path = fixture_path();
    std::FILE* f = std::fopen(path.c_str(), "wb");
    ASSERT_NE(f, nullptr) << path;
    auto* w = static_cast<ArrowIpcWriter*>(
        std::calloc(1, sizeof(ArrowIpcWriter)));
    Fixture fx;
    const BoltType tys[3] = {BoltType::Int64, BoltType::Float64,
                             BoltType::Utf8};
    const char* names[3] = {"ints", "floats", "strs"};
    ASSERT_TRUE(arrow_ipc_open(w, f, tys, names, 3));
    ASSERT_TRUE(arrow_ipc_write_batch(w, fx.batch));   // batch 1: full rule
    ASSERT_TRUE(arrow_ipc_write_batch(w, fx.batch));   // batch 2: same again
    ASSERT_TRUE(arrow_ipc_close(w));
    std::fclose(f);
    std::free(w);
    ::testing::Test::RecordProperty("fixture", path);
}
