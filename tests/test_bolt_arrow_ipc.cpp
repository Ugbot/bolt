// test_bolt_arrow_ipc.cpp — Arrow IPC stream writer (G2ARROW-10, widened
// to Bool/Date32/Decimal128/Binary by G2ARROW-20).
//
// Structural gates run here; the VALUE gate is the pyarrow oracle
// (scripts/arrow_ipc_check.py) over the fixtures this test writes —
// pyarrow.ipc.open_stream() must read the bolt-written bytes and every
// value must match the generating rule re-derived in python. The
// Int64/Float64/Utf8 fixture path is
// <BOLT_TEST_DATA_DIR>/../out/arrow_ipc_fixture.arrows unless
// BOLT_ARROW_IPC_OUT overrides it; the wider (+Bool/Date32/Binary/
// Decimal128) fixture is BOLT_ARROW_IPC_WIDE_OUT, default
// arrow_ipc_wide_fixture.arrows.

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

// ---------------------------------------------------------------------
// Wide-types fixture (G2ARROW-20): Bool, Date32, Binary, Decimal128 —
// each with its own null pattern, sharing the same kRows/int_val rule so
// the python oracle can re-derive every column independently.
// ---------------------------------------------------------------------

bool bool_val(std::int64_t i) { return (i % 3) == 0; }
bool bool_is_null(std::int64_t i) { return (i % 11) == 0; }

std::int32_t date_val(std::int64_t i) {
    return static_cast<std::int32_t>(i * 5 - 200);   // spans negative (pre-epoch)
}
bool date_is_null(std::int64_t i) { return (i % 17) == 0; }

constexpr std::uint8_t kDecimalScale = 4;
std::int64_t decimal_mantissa(std::int64_t i) { return (i - 50) * 100000007LL; }
bool decimal_is_null(std::int64_t i) { return (i % 19) == 0; }

// Little-endian two's-complement 128-bit encode of a (sign-extended)
// int64 mantissa — exactly Arrow's Decimal128 wire layout, and the same
// layout bolt itself stores a Decimal128 column's 16-byte rows in.
void write_decimal128_le(std::uint8_t out[16], std::int64_t mantissa) {
    const std::uint64_t lo = static_cast<std::uint64_t>(mantissa);
    const std::int64_t hi_signed = (mantissa < 0) ? -1 : 0;   // sign extend
    const std::uint64_t hi = static_cast<std::uint64_t>(hi_signed);
    std::memcpy(out, &lo, 8);
    std::memcpy(out + 8, &hi, 8);
}

// Raw (deliberately non-UTF-8) byte content: lone continuation bytes and
// invalid lead bytes that would fail any UTF-8 validator, proving the
// writer treats Binary payloads as opaque bytes. Every 7th row is long
// enough to force the StringView-spill path, matching str_val()'s own
// inline-vs-spilled split.
std::string binary_val(std::int64_t i) {
    std::string s;
    s.push_back(static_cast<char>(0xFF));
    s.push_back(static_cast<char>(0xFE));
    s.push_back(static_cast<char>(0x00));
    s.push_back(static_cast<char>(i & 0xFF));
    s.push_back(static_cast<char>(0x80));
    s.push_back(static_cast<char>(0xC0));
    s.push_back(static_cast<char>(0xC1));
    if ((i % 7) == 0) {
        for (int k = 0; k < 10; ++k) {
            s.push_back(static_cast<char>(0x80 + (k & 0x3F)));
        }
    }
    return s;
}
bool binary_is_null(std::int64_t i) { return (i % 23) == 0; }

struct WideFixture {
    std::int64_t  i64[kRows];
    double        f64[kRows];
    StringView    strs[kRows];
    std::uint8_t  bools[kRows];       // byte-packed, bolt's native Bool layout
    std::int32_t  dates[kRows];
    StringView    bins[kRows];
    std::uint8_t  decimals[kRows][16];
    char          str_pool[8192];
    std::uint32_t str_pool_used = 0;
    char          bin_pool[8192];
    std::uint32_t bin_pool_used = 0;
    std::uint8_t  strs_validity[(kRows + 7) / 8];
    std::uint8_t  bools_validity[(kRows + 7) / 8];
    std::uint8_t  dates_validity[(kRows + 7) / 8];
    std::uint8_t  bins_validity[(kRows + 7) / 8];
    std::uint8_t  decimals_validity[(kRows + 7) / 8];
    bolt::Arena   arena;
    BoltBatch*    batch;

    WideFixture() {
        std::memset(strs_validity, 0xFF, sizeof(strs_validity));
        std::memset(bools_validity, 0xFF, sizeof(bools_validity));
        std::memset(dates_validity, 0xFF, sizeof(dates_validity));
        std::memset(bins_validity, 0xFF, sizeof(bins_validity));
        std::memset(decimals_validity, 0xFF, sizeof(decimals_validity));
        for (std::int64_t i = 0; i < kRows; ++i) {
            i64[i] = int_val(i);
            f64[i] = dbl_val(i);
            if (row_is_null(i)) {
                strs_validity[i >> 3] &= static_cast<std::uint8_t>(~(1u << (i & 7)));
                strs[i] = StringView{};
            } else {
                const std::string s = str_val(i);
                strs[i] = make_view(s.data(), static_cast<std::uint32_t>(s.size()),
                                    str_pool, &str_pool_used);
            }
            bools[i] = bool_val(i) ? 1 : 0;
            if (bool_is_null(i)) {
                bools_validity[i >> 3] &= static_cast<std::uint8_t>(~(1u << (i & 7)));
            }
            dates[i] = date_val(i);
            if (date_is_null(i)) {
                dates_validity[i >> 3] &= static_cast<std::uint8_t>(~(1u << (i & 7)));
            }
            if (binary_is_null(i)) {
                bins_validity[i >> 3] &= static_cast<std::uint8_t>(~(1u << (i & 7)));
                bins[i] = StringView{};
            } else {
                const std::string b = binary_val(i);
                bins[i] = make_view(b.data(), static_cast<std::uint32_t>(b.size()),
                                    bin_pool, &bin_pool_used);
            }
            write_decimal128_le(decimals[i], decimal_mantissa(i));
            if (decimal_is_null(i)) {
                decimals_validity[i >> 3] &= static_cast<std::uint8_t>(~(1u << (i & 7)));
            }
        }
        batch = static_cast<BoltBatch*>(std::calloc(1, sizeof(BoltBatch)));
        BoltBatch::init_empty(batch);
        const bool cols_ok = BoltBatch::alloc_columns(batch, &arena, 7);
        assert(cols_ok);
        (void)cols_ok;
        batch->num_rows = kRows;
        batch->num_cols = 7;
        BoltColumn& c0 = batch->columns[batch->read_epoch][0];
        c0.type = BoltType::Int64; c0.format = ColumnFormat::Flat;
        c0.data = i64; c0.length = kRows; c0.type_size_bytes = 8;
        BoltColumn& c1 = batch->columns[batch->read_epoch][1];
        c1.type = BoltType::Float64; c1.format = ColumnFormat::Flat;
        c1.data = f64; c1.length = kRows; c1.type_size_bytes = 8;
        BoltColumn& c2 = batch->columns[batch->read_epoch][2];
        c2.type = BoltType::Utf8; c2.format = ColumnFormat::Flat;
        c2.data = strs; c2.length = kRows;
        c2.str_overflow_base = str_pool; c2.validity = strs_validity;
        BoltColumn& c3 = batch->columns[batch->read_epoch][3];
        c3.type = BoltType::Bool; c3.format = ColumnFormat::Flat;
        c3.data = bools; c3.length = kRows; c3.type_size_bytes = 1;
        c3.validity = bools_validity;
        BoltColumn& c4 = batch->columns[batch->read_epoch][4];
        c4.type = BoltType::Date32; c4.format = ColumnFormat::Flat;
        c4.data = dates; c4.length = kRows; c4.type_size_bytes = 4;
        c4.validity = dates_validity;
        BoltColumn& c5 = batch->columns[batch->read_epoch][5];
        c5.type = BoltType::Binary; c5.format = ColumnFormat::Flat;
        c5.data = bins; c5.length = kRows;
        c5.str_overflow_base = bin_pool; c5.validity = bins_validity;
        BoltColumn& c6 = batch->columns[batch->read_epoch][6];
        c6.type = BoltType::Decimal128; c6.format = ColumnFormat::Flat;
        c6.data = decimals; c6.length = kRows; c6.type_size_bytes = 16;
        c6.decimal_scale = kDecimalScale;
        c6.validity = decimals_validity;
    }
    ~WideFixture() { std::free(batch); }
};

std::string wide_fixture_path() {
    const char* env = std::getenv("BOLT_ARROW_IPC_WIDE_OUT");
    if (env != nullptr && env[0] != '\0') return env;
    return "arrow_ipc_wide_fixture.arrows";
}

}  // namespace

TEST(ArrowIpc, RejectsUnsupportedTypeAtOpen) {
    auto* w = static_cast<ArrowIpcWriter*>(
        std::calloc(1, sizeof(ArrowIpcWriter)));
    ASSERT_NE(w, nullptr);
    std::FILE* f = std::tmpfile();
    ASSERT_NE(f, nullptr);
    // List remains genuinely unsupported (nested types are G2ARROW-20's
    // documented follow-up) — Bool/Date32/Binary/Decimal128 moved out of
    // this test into WritesPyarrowOracleWideTypesFixture below since they
    // are now real, pyarrow-verified support, not a rejection case.
    const BoltType bad[2] = {BoltType::Int64, BoltType::List};
    EXPECT_FALSE(arrow_ipc_open(w, f, bad, nullptr, 2));
    // Fail closed AT open: nothing was written.
    std::fflush(f);
    EXPECT_EQ(std::ftell(f), 0);
    std::fclose(f);
    std::free(w);
}

TEST(ArrowIpc, Decimal128RequiresScaleAtOpen) {
    auto* w = static_cast<ArrowIpcWriter*>(
        std::calloc(1, sizeof(ArrowIpcWriter)));
    ASSERT_NE(w, nullptr);
    std::FILE* f = std::tmpfile();
    ASSERT_NE(f, nullptr);
    const BoltType tys[1] = {BoltType::Decimal128};
    // No decimal_scales array supplied: fail closed rather than default
    // to a scale that would silently misrepresent every value on read.
    EXPECT_FALSE(arrow_ipc_open(w, f, tys, nullptr, 1));
    std::fflush(f);
    EXPECT_EQ(std::ftell(f), 0);

    const std::uint8_t too_wide[1] = {39};   // > kDecimal128Precision (38)
    EXPECT_FALSE(arrow_ipc_open(w, f, tys, nullptr, 1, too_wide));
    std::fflush(f);
    EXPECT_EQ(std::ftell(f), 0);

    const std::uint8_t ok_scale[1] = {kDecimalScale};
    EXPECT_TRUE(arrow_ipc_open(w, f, tys, nullptr, 1, ok_scale));
    EXPECT_TRUE(arrow_ipc_close(w));
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

// G2ARROW-20: writes the wide-types fixture (Int64, Float64, Utf8, Bool,
// Date32, Binary, Decimal128 in one batch — a realistic mixed-type query
// result) the pyarrow oracle (scripts/arrow_ipc_check.py) validates
// value-for-value, including every column's own null pattern, negative
// Decimal128 mantissas (two's-complement sign), and non-UTF-8 Binary
// bytes.
TEST(ArrowIpc, WritesPyarrowOracleWideTypesFixture) {
    const std::string path = wide_fixture_path();
    std::FILE* f = std::fopen(path.c_str(), "wb");
    ASSERT_NE(f, nullptr) << path;
    auto* w = static_cast<ArrowIpcWriter*>(
        std::calloc(1, sizeof(ArrowIpcWriter)));
    WideFixture fx;
    const BoltType tys[7] = {
        BoltType::Int64,  BoltType::Float64, BoltType::Utf8,
        BoltType::Bool,   BoltType::Date32,  BoltType::Binary,
        BoltType::Decimal128,
    };
    const char* names[7] = {
        "ints", "floats", "strs", "bools", "dates", "bins", "decimals",
    };
    const std::uint8_t scales[7] = {0, 0, 0, 0, 0, 0, kDecimalScale};
    ASSERT_TRUE(arrow_ipc_open(w, f, tys, names, 7, scales));
    ASSERT_TRUE(arrow_ipc_write_batch(w, fx.batch));
    ASSERT_TRUE(arrow_ipc_close(w));
    std::fclose(f);
    std::free(w);
    ::testing::Test::RecordProperty("fixture", path);
}
