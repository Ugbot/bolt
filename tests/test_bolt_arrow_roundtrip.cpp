// test_bolt_arrow_roundtrip.cpp — G2ARROW-6: export -> import round-trip
// entirely WITHIN bolt (no external Arrow library involved on either side).
//
// This is Tier 2 of the G2ARROW-4 test harness. Tier 1 (test_bolt_arrow_export.cpp)
// pins the export contract in isolation; Tier 3 (the Gestalt2-superproject
// pyarrow test) validates against a real Arrow implementation. Neither of
// those proves bolt's own export and import agree with EACH OTHER -- a wrong
// but internally-consistent format string could pass both. This file builds a
// real BoltColumn, exports it via bolt::arrow::export_column, imports the
// resulting (ArrowSchema, ArrowArray) back via bolt::arrow::import_column
// (built for this ticket -- no importer existed before), and asserts
// value-for-value / byte-for-byte equality against the original, including
// validity. It also pins the release() contract end-to-end: the exported
// structs are released immediately after import and the imported column must
// remain fully valid, proving import copies rather than aliases.
#include <cstdint>
#include <cstring>
#include <string>

#include <gtest/gtest.h>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_arrow.h"
#include "bolt/bolt_column.h"

using namespace bolt;

namespace {

bool bit_valid(const uint8_t* validity, int64_t i) {
    if (validity == nullptr) return true;
    return (validity[i >> 3] & (uint8_t{1} << (i & 7))) != 0;
}

// ---- fixed-width builder: `valid` may be nullptr (no validity buffer at
// all -- the "no-null column with no validity buffer" adversarial case). ----
template <typename T>
BoltColumn make_flat_typed(Arena* a, const T* vals, const bool* valid,
                           int64_t n, BoltType type) {
    T* d = static_cast<T*>(a->allocate(static_cast<size_t>(n) * sizeof(T), 64));
    if (n > 0) std::memcpy(d, vals, static_cast<size_t>(n) * sizeof(T));
    uint8_t* v = nullptr;
    if (valid != nullptr) {
        const size_t nb = static_cast<size_t>((n + 7) / 8);
        v = static_cast<uint8_t*>(a->allocate(nb, 64));
        std::memset(v, 0xFF, nb);
        for (int64_t i = 0; i < n; ++i) {
            if (!valid[i]) v[i >> 3] &= static_cast<uint8_t>(~(1u << (i & 7)));
        }
    }
    return BoltColumn::make_flat(d, v, n, type);
}

// Round-trips `src` through export+import, releases the exported structs
// immediately (proving import copied rather than aliased), and returns the
// imported column via `out`. Returns false (with a gtest failure recorded)
// on any step failing.
bool round_trip(BoltColumn& src, int64_t n, Arena* import_arena,
                BoltColumn* out) {
    ArrowSchema s;
    ArrowArray arr;
    if (!bolt::arrow::export_column(src, n, "col", &s, &arr)) {
        ADD_FAILURE() << "export_column failed";
        return false;
    }
    if (!bolt::arrow::import_column(s, arr, import_arena, out)) {
        ADD_FAILURE() << "import_column failed";
        s.release(&s);
        arr.release(&arr);
        return false;
    }
    // Genuine copy, not a pointer hand-back: the imported buffer must not
    // alias the (about-to-be-released) source ArrowArray buffer.
    if (n > 0 && arr.n_buffers > 1 && arr.buffers[1] != nullptr) {
        EXPECT_NE(out->data, arr.buffers[1])
            << "import aliased the exported buffer instead of copying it";
    }
    s.release(&s);
    arr.release(&arr);
    // Use-after-release proof: everything read from `out` below happens
    // AFTER both structs are released. If import had borrowed rather than
    // copied, this would be a use-after-free (and would show up under ASAN).
    return true;
}

// ---- Utf8 rows: covers empty, the 12/13-byte inline/spill boundary, and
// nulls. `valid=false` rows are encoded as zero-length content by this
// test's own convention (the spec leaves null-row content undefined; this
// pins OUR chosen convention round-trips exactly, which is a meaningful
// signal that build_varlen/var_at don't smuggle garbage through it). ----
struct Row { const char* s; bool valid; };

BoltColumn make_utf8_view_rows(Arena* a, const Row* rows, int64_t n) {
    auto* views = static_cast<StringView*>(a->allocate(
        static_cast<size_t>(n) * sizeof(StringView), 64));
    std::memset(views, 0, static_cast<size_t>(n) * sizeof(StringView));
    std::string spill;
    bool any_null = false;
    for (int64_t i = 0; i < n; ++i) if (!rows[i].valid) any_null = true;
    uint8_t* validity = nullptr;
    if (any_null) {
        const size_t nb = static_cast<size_t>((n + 7) / 8);
        validity = static_cast<uint8_t*>(a->allocate(nb, 64));
        std::memset(validity, 0xFF, nb);
    }
    for (int64_t i = 0; i < n; ++i) {
        if (!rows[i].valid) {
            validity[i >> 3] &= static_cast<uint8_t>(~(1u << (i & 7)));
            continue;  // views[i] stays zeroed => length 0
        }
        const char* str = rows[i].s;
        const uint32_t len = static_cast<uint32_t>(std::strlen(str));
        views[i].length = len;
        if (len <= 12) {
            if (len > 0) std::memcpy(views[i].prefix, str, (len < 4) ? len : 4);
            if (len > 4) std::memcpy(views[i].inline_data, str + 4, len - 4);
        } else {
            std::memcpy(views[i].prefix, str, 4);
            views[i].ref.buf_idx = 0;
            views[i].ref.offset = static_cast<uint32_t>(spill.size());
            spill += str;
        }
    }
    auto* base = static_cast<char*>(a->allocate(spill.size() + 1, 64));
    std::memcpy(base, spill.data(), spill.size());
    auto c = BoltColumn::make_flat(views, validity, n, BoltType::Utf8);
    c.str_overflow_base = base;
    return c;
}

// The SECOND bolt Utf8 physical layout: VarBinary (packed offsets + bytes),
// per bolt_arrow.h's own doc comment -- both layouts must export correctly
// via var_at()'s dual dispatch, and both must import back to equal values.
BoltColumn make_utf8_varbinary_rows(Arena* a, const Row* rows, int64_t n) {
    auto* offs = a->allocate_array<int32_t>(static_cast<size_t>(n + 1));
    std::string bytes;
    offs[0] = 0;
    bool any_null = false;
    for (int64_t i = 0; i < n; ++i) if (!rows[i].valid) any_null = true;
    uint8_t* validity = nullptr;
    if (any_null) {
        const size_t nb = static_cast<size_t>((n + 7) / 8);
        validity = static_cast<uint8_t*>(a->allocate(nb, 64));
        std::memset(validity, 0xFF, nb);
    }
    for (int64_t i = 0; i < n; ++i) {
        if (!rows[i].valid) {
            validity[i >> 3] &= static_cast<uint8_t>(~(1u << (i & 7)));
            offs[i + 1] = static_cast<int32_t>(bytes.size());  // 0-length
            continue;
        }
        bytes += rows[i].s;
        offs[i + 1] = static_cast<int32_t>(bytes.size());
    }
    auto* data = static_cast<char*>(a->allocate(bytes.size() + 1, 64));
    std::memcpy(data, bytes.data(), bytes.size());
    return BoltColumn::make_var_binary(data, validity, offs, n, BoltType::Utf8, a);
}

void expect_utf8_rows_match(const BoltColumn& col, const Row* rows, int64_t n) {
    ASSERT_EQ(col.length, n);
    ASSERT_EQ(col.type, BoltType::Utf8);
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_EQ(bit_valid(col.validity, i), rows[i].valid) << "row " << i;
        const auto* views = static_cast<const StringView*>(col.data);
        const StringView& v = views[i];
        std::string got;
        if (v.is_inline()) {
            const uint32_t p = (v.length < 4u) ? v.length : 4u;
            got.append(v.prefix, p);
            if (v.length > 4u) got.append(v.inline_data, v.length - 4u);
        } else {
            const auto* base = static_cast<const char*>(col.str_overflow_base);
            got.assign(base + v.ref.offset, v.length);
        }
        const std::string expected = rows[i].valid ? std::string(rows[i].s) : std::string();
        EXPECT_EQ(got, expected) << "row " << i;
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Int64 — the type this whole gateway is measured on.
// ---------------------------------------------------------------------------

TEST(ArrowRoundTrip, Int64MixedNulls) {
    Arena src_arena, import_arena;
    const int64_t vals[6] = {-9223372036854775807LL, -1, 0, 1, 42, 9223372036854775807LL};
    const bool valid[6]   = {true, false, true, false, true, true};
    BoltColumn src = make_flat_typed<int64_t>(&src_arena, vals, valid, 6, BoltType::Int64);

    BoltColumn out;
    ASSERT_TRUE(round_trip(src, 6, &import_arena, &out));
    ASSERT_EQ(out.type, BoltType::Int64);
    ASSERT_EQ(out.length, 6);
    const auto* got = static_cast<const int64_t*>(out.data);
    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_EQ(bit_valid(out.validity, i), valid[i]) << i;
        EXPECT_EQ(got[i], vals[i]) << i;
    }
}

TEST(ArrowRoundTrip, Int64AllNull) {
    Arena src_arena, import_arena;
    const int64_t vals[4] = {1, 2, 3, 4};
    const bool valid[4]   = {false, false, false, false};
    BoltColumn src = make_flat_typed<int64_t>(&src_arena, vals, valid, 4, BoltType::Int64);

    ArrowSchema s; ArrowArray arr;
    ASSERT_TRUE(bolt::arrow::export_column(src, 4, "x", &s, &arr));
    EXPECT_EQ(arr.null_count, 4);
    BoltColumn out;
    ASSERT_TRUE(bolt::arrow::import_column(s, arr, &import_arena, &out));
    s.release(&s); arr.release(&arr);
    for (int64_t i = 0; i < 4; ++i) EXPECT_FALSE(bit_valid(out.validity, i)) << i;
}

TEST(ArrowRoundTrip, Int64NoNullNoValidityBuffer) {
    // valid == nullptr => src.validity == nullptr (no bitmap allocated at
    // all, not merely all-1s). Exercises export's `col.validity != nullptr`
    // NULLABLE-flag path and import's "no buffer[0] at all" path together.
    Arena src_arena, import_arena;
    const int64_t vals[3] = {7, 8, 9};
    BoltColumn src = make_flat_typed<int64_t>(&src_arena, vals, nullptr, 3, BoltType::Int64);
    ASSERT_EQ(src.validity, nullptr);

    ArrowSchema s; ArrowArray arr;
    ASSERT_TRUE(bolt::arrow::export_column(src, 3, "x", &s, &arr));
    EXPECT_EQ(s.flags & 2, 0);           // NOT nullable
    EXPECT_EQ(arr.buffers[0], nullptr);  // no validity buffer at all
    BoltColumn out;
    ASSERT_TRUE(bolt::arrow::import_column(s, arr, &import_arena, &out));
    s.release(&s); arr.release(&arr);
    EXPECT_EQ(out.validity, nullptr);
    const auto* got = static_cast<const int64_t*>(out.data);
    for (int64_t i = 0; i < 3; ++i) EXPECT_EQ(got[i], vals[i]);
}

TEST(ArrowRoundTrip, Int64EmptyColumn) {
    Arena src_arena, import_arena;
    BoltColumn src = make_flat_typed<int64_t>(&src_arena, nullptr, nullptr, 0, BoltType::Int64);
    BoltColumn out;
    ASSERT_TRUE(round_trip(src, 0, &import_arena, &out));
    EXPECT_EQ(out.length, 0);
}

// ---------------------------------------------------------------------------
// Float64
// ---------------------------------------------------------------------------

TEST(ArrowRoundTrip, Float64MixedNulls) {
    Arena src_arena, import_arena;
    const double vals[5] = {-1.5, 0.0, 3.14159265358979, 1e300, -1e-300};
    const bool valid[5]  = {true, false, true, true, false};
    BoltColumn src = make_flat_typed<double>(&src_arena, vals, valid, 5, BoltType::Float64);

    BoltColumn out;
    ASSERT_TRUE(round_trip(src, 5, &import_arena, &out));
    const auto* got = static_cast<const double*>(out.data);
    for (int64_t i = 0; i < 5; ++i) {
        EXPECT_EQ(bit_valid(out.validity, i), valid[i]) << i;
        // Byte-exact, not epsilon: a real round-trip must not perturb bits.
        EXPECT_EQ(std::memcmp(&got[i], &vals[i], sizeof(double)), 0) << i;
    }
}

// ---------------------------------------------------------------------------
// Utf8 — both bolt physical layouts as the SOURCE, both feeding the same
// export path; import always materializes Flat/View (the layout the rest of
// the engine consumes). Covers empty strings and the exact 12/13-byte
// inline/spill boundary per G2ARROW-6.
// ---------------------------------------------------------------------------

TEST(ArrowRoundTrip, Utf8FlatViewLayoutBoundariesAndNulls) {
    Arena src_arena, import_arena;
    const std::string s12(12, 'a');   // exactly inline (<=12)
    const std::string s13(13, 'b');   // exactly spilled (>12)
    const Row rows[] = {
        {"", true},                                              // empty, valid
        {s12.c_str(), true},                                     // 12-byte boundary
        {s13.c_str(), true},                                     // 13-byte boundary
        {"", false},                                              // null
        {"a value that is definitely longer than twelve bytes", true},
        {"z", false},                                             // null, short
        {"", false},                                              // null, empty
        {"tail", true},
    };
    const int64_t n = static_cast<int64_t>(sizeof(rows) / sizeof(rows[0]));
    BoltColumn src = make_utf8_view_rows(&src_arena, rows, n);

    BoltColumn out;
    ASSERT_TRUE(round_trip(src, n, &import_arena, &out));
    expect_utf8_rows_match(out, rows, n);
}

TEST(ArrowRoundTrip, Utf8VarBinarySourceLayoutRoundTrips) {
    // Second bolt-side layout for the SAME logical data as the test above
    // (minus the boundary values, which VarBinary has no inline/spill
    // distinction for -- it is exercised by the Flat/View test instead).
    Arena src_arena, import_arena;
    const Row rows[] = {
        {"", true},
        {"short", true},
        {"", false},
        {"a longer value that would spill under Flat/View, here just bytes", true},
        {"x", false},
    };
    const int64_t n = static_cast<int64_t>(sizeof(rows) / sizeof(rows[0]));
    BoltColumn src = make_utf8_varbinary_rows(&src_arena, rows, n);
    ASSERT_EQ(src.format, ColumnFormat::VarBinary);

    BoltColumn out;
    ASSERT_TRUE(round_trip(src, n, &import_arena, &out));
    // Imported column always materializes Flat/View regardless of the
    // source's physical layout.
    EXPECT_EQ(out.format, ColumnFormat::Flat);
    expect_utf8_rows_match(out, rows, n);
}

TEST(ArrowRoundTrip, Utf8AllNull) {
    Arena src_arena, import_arena;
    const Row rows[3] = {{"", false}, {"", false}, {"", false}};
    BoltColumn src = make_utf8_view_rows(&src_arena, rows, 3);
    BoltColumn out;
    ASSERT_TRUE(round_trip(src, 3, &import_arena, &out));
    for (int64_t i = 0; i < 3; ++i) EXPECT_FALSE(bit_valid(out.validity, i)) << i;
}

TEST(ArrowRoundTrip, Utf8NoNullNoValidityBuffer) {
    Arena src_arena, import_arena;
    const Row rows[3] = {{"a", true}, {"bb", true}, {"ccc", true}};
    BoltColumn src = make_utf8_view_rows(&src_arena, rows, 3);
    ASSERT_EQ(src.validity, nullptr);
    ArrowSchema s; ArrowArray arr;
    ASSERT_TRUE(bolt::arrow::export_column(src, 3, "x", &s, &arr));
    EXPECT_EQ(arr.buffers[0], nullptr);
    BoltColumn out;
    ASSERT_TRUE(bolt::arrow::import_column(s, arr, &import_arena, &out));
    s.release(&s); arr.release(&arr);
    EXPECT_EQ(out.validity, nullptr);
    expect_utf8_rows_match(out, rows, 3);
}

// ---------------------------------------------------------------------------
// Binary — same wire shape as Utf8 ("z" not "u"), materializes VarBinary.
// ---------------------------------------------------------------------------

TEST(ArrowRoundTrip, BinaryWithEmbeddedNulBytesAndValidity) {
    Arena src_arena, import_arena;
    // Binary payloads may contain \0 -- use explicit lengths, not strlen.
    const char b0[] = {0x00, 0x01, 0x02};
    const char b1[] = {};
    const char b2[] = {static_cast<char>(0xFF), 0x00, static_cast<char>(0xAB)};
    struct BRow { const char* p; int32_t len; bool valid; };
    const BRow rows[3] = {{b0, 3, true}, {b1, 0, true}, {b2, 3, false}};

    auto* offs = src_arena.allocate_array<int32_t>(4);
    std::string bytes;
    offs[0] = 0;
    uint8_t* validity = static_cast<uint8_t*>(src_arena.allocate(1, 64));
    *validity = 0b011;  // row0 valid, row1 valid, row2 invalid
    for (int i = 0; i < 3; ++i) {
        bytes.append(rows[i].p, static_cast<size_t>(rows[i].len));
        offs[i + 1] = static_cast<int32_t>(bytes.size());
    }
    auto* data = static_cast<char*>(src_arena.allocate(bytes.size() + 1, 64));
    std::memcpy(data, bytes.data(), bytes.size());
    BoltColumn src = BoltColumn::make_var_binary(data, validity, offs, 3,
                                                 BoltType::Binary, &src_arena);

    BoltColumn out;
    ASSERT_TRUE(round_trip(src, 3, &import_arena, &out));
    ASSERT_EQ(out.format, ColumnFormat::VarBinary);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(bit_valid(out.validity, i), rows[i].valid) << i;
        const uint8_t* p; int32_t n;
        out.var_binary_at(i, &p, &n);
        if (rows[i].valid) {
            ASSERT_EQ(n, rows[i].len) << i;
            EXPECT_EQ(std::memcmp(p, rows[i].p, static_cast<size_t>(n)), 0) << i;
        }
    }
    // Genuine copy: the imported bytes buffer differs from the source's.
    EXPECT_NE(out.data, src.data);
}

// ---------------------------------------------------------------------------
// Decimal128 — the column's real scale must survive (export's own regression
// test pins that "d:38,10" was hardcoded; import must read it back exactly).
// ---------------------------------------------------------------------------

TEST(ArrowRoundTrip, Decimal128PreservesRealScale) {
    Arena src_arena, import_arena;
    unsigned char raw[3 * 16];
    std::memset(raw, 0, sizeof(raw));
    int64_t mantissas[3] = {12345, -98765, 0};
    for (int i = 0; i < 3; ++i) std::memcpy(raw + i * 16, &mantissas[i], 8);
    const bool valid[3] = {true, false, true};
    BoltColumn src = BoltColumn::make_flat(raw, nullptr, 3, BoltType::Decimal128);
    // add validity separately since make_flat_typed<T> assumes trivial T
    const size_t nb = 1;
    auto* v = static_cast<uint8_t*>(src_arena.allocate(nb, 64));
    *v = 0b101;
    src.validity = v;
    src.decimal_scale = 7;
    src.type_size_bytes = 16;

    ArrowSchema s; ArrowArray arr;
    ASSERT_TRUE(bolt::arrow::export_column(src, 3, "amt", &s, &arr));
    EXPECT_STREQ(s.format, "d:38,7");
    BoltColumn out;
    ASSERT_TRUE(bolt::arrow::import_column(s, arr, &import_arena, &out));
    s.release(&s); arr.release(&arr);

    EXPECT_EQ(out.type, BoltType::Decimal128);
    EXPECT_EQ(out.decimal_scale, 7);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(bit_valid(out.validity, i), valid[i]) << i;
        EXPECT_EQ(std::memcmp(static_cast<const unsigned char*>(out.data) + i * 16,
                              raw + i * 16, 16), 0) << i;
    }
}

// ---------------------------------------------------------------------------
// Date32 / Timestamp — the temporal types this codebase actually uses.
// ---------------------------------------------------------------------------

TEST(ArrowRoundTrip, Date32RoundTrip) {
    Arena src_arena, import_arena;
    const int32_t vals[4] = {0, 19700, -5, 2932896};  // includes pre-epoch
    const bool valid[4]   = {true, true, false, true};
    BoltColumn src = make_flat_typed<int32_t>(&src_arena, vals, valid, 4, BoltType::Date32);
    BoltColumn out;
    ASSERT_TRUE(round_trip(src, 4, &import_arena, &out));
    EXPECT_EQ(out.type, BoltType::Date32);
    const auto* got = static_cast<const int32_t*>(out.data);
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(bit_valid(out.validity, i), valid[i]) << i;
        EXPECT_EQ(got[i], vals[i]) << i;
    }
}

TEST(ArrowRoundTrip, TimestampRoundTrip) {
    Arena src_arena, import_arena;
    const int64_t vals[3] = {0, 1755000000000000LL, -1000};
    const bool valid[3]   = {true, true, false};
    BoltColumn src = make_flat_typed<int64_t>(&src_arena, vals, valid, 3, BoltType::Timestamp);
    ArrowSchema s; ArrowArray arr;
    ASSERT_TRUE(bolt::arrow::export_column(src, 3, "ts", &s, &arr));
    EXPECT_STREQ(s.format, "tsu:");
    BoltColumn out;
    ASSERT_TRUE(bolt::arrow::import_column(s, arr, &import_arena, &out));
    s.release(&s); arr.release(&arr);
    EXPECT_EQ(out.type, BoltType::Timestamp);
    const auto* got = static_cast<const int64_t*>(out.data);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(bit_valid(out.validity, i), valid[i]) << i;
        EXPECT_EQ(got[i], vals[i]) << i;
    }
}

// ---------------------------------------------------------------------------
// Bool — bit-pack (export) then bit-unpack (import) must be lossless, not
// just non-crashing: this is the one type where the wire representation
// (Arrow bit-packed) genuinely differs from bolt's own (byte-packed).
// ---------------------------------------------------------------------------

TEST(ArrowRoundTrip, BoolBitPackRoundTrip) {
    Arena src_arena, import_arena;
    const int64_t n = 20;
    auto* d = static_cast<uint8_t*>(src_arena.allocate(n, 64));
    bool expect[20];
    for (int64_t i = 0; i < n; ++i) { expect[i] = (i % 3 == 0); d[i] = expect[i] ? 1 : 0; }
    const size_t nb = static_cast<size_t>((n + 7) / 8);
    auto* v = static_cast<uint8_t*>(src_arena.allocate(nb, 64));
    std::memset(v, 0xFF, nb);
    v[1] &= static_cast<uint8_t>(~(1u << 3));  // row 11 -> null
    BoltColumn src = BoltColumn::make_flat(d, v, n, BoltType::Bool);

    BoltColumn out;
    ASSERT_TRUE(round_trip(src, n, &import_arena, &out));
    EXPECT_EQ(out.type, BoltType::Bool);
    const auto* got = static_cast<const uint8_t*>(out.data);
    for (int64_t i = 0; i < n; ++i) {
        if (i == 11) { EXPECT_FALSE(bit_valid(out.validity, i)); continue; }
        EXPECT_TRUE(bit_valid(out.validity, i)) << i;
        EXPECT_EQ(got[i] != 0, expect[i]) << i;
    }
}

// ---------------------------------------------------------------------------
// The release() contract: no leak, no double-free, no use-after-release.
// ---------------------------------------------------------------------------

TEST(ArrowRoundTrip, ImportedColumnSurvivesSourceArenaResetAndStructRelease) {
    // Belt-and-suspenders on top of round_trip()'s own release-before-compare
    // ordering: also recycle the SOURCE bolt arena (not just release the
    // Arrow structs), so the only way this test can pass is if import copied
    // every buffer -- validity, offsets, and payload -- into its own arena.
    Arena src_arena, import_arena;
    const Row rows[3] = {{"alpha", true}, {"a value spilled past twelve bytes", true}, {"", false}};
    BoltColumn src = make_utf8_view_rows(&src_arena, rows, 3);
    ArrowSchema s; ArrowArray arr;
    ASSERT_TRUE(bolt::arrow::export_column(src, 3, "x", &s, &arr));

    src_arena.reset();  // recycle every source buffer the export copied FROM

    BoltColumn out;
    ASSERT_TRUE(bolt::arrow::import_column(s, arr, &import_arena, &out));
    s.release(&s);
    arr.release(&arr);

    expect_utf8_rows_match(out, rows, 3);
}

TEST(ArrowRoundTrip, ImportFailsOnAlreadyReleasedStruct) {
    Arena src_arena, import_arena;
    const int64_t vals[2] = {1, 2};
    BoltColumn src = make_flat_typed<int64_t>(&src_arena, vals, nullptr, 2, BoltType::Int64);
    ArrowSchema s; ArrowArray arr;
    ASSERT_TRUE(bolt::arrow::export_column(src, 2, "x", &s, &arr));
    s.release(&s);
    arr.release(&arr);
    ASSERT_EQ(s.release, nullptr);
    ASSERT_EQ(arr.release, nullptr);

    BoltColumn out;
    // Nothing left to read; must fail closed, not read freed/zeroed memory.
    EXPECT_FALSE(bolt::arrow::import_column(s, arr, &import_arena, &out));
    EXPECT_EQ(out.data, nullptr);
    EXPECT_EQ(out.length, 0);
}

TEST(ArrowRoundTrip, DoubleReleaseAfterImportIsSafe) {
    Arena src_arena, import_arena;
    const int64_t vals[2] = {5, 6};
    BoltColumn src = make_flat_typed<int64_t>(&src_arena, vals, nullptr, 2, BoltType::Int64);
    ArrowSchema s; ArrowArray arr;
    ASSERT_TRUE(bolt::arrow::export_column(src, 2, "x", &s, &arr));
    BoltColumn out;
    ASSERT_TRUE(bolt::arrow::import_column(s, arr, &import_arena, &out));

    auto* sfn = s.release;
    auto* afn = arr.release;
    sfn(&s); sfn(&s);      // release is documented safe against a repeat call
    afn(&arr); afn(&arr);
    EXPECT_EQ(s.release, nullptr);
    EXPECT_EQ(arr.release, nullptr);
    // Imported column is unaffected by any of this -- it never pointed at
    // the released structs' memory in the first place.
    const auto* got = static_cast<const int64_t*>(out.data);
    EXPECT_EQ(got[0], 5);
    EXPECT_EQ(got[1], 6);
}

TEST(ArrowRoundTrip, ImportFailsClosedOnUnrecognizedFormat) {
    Arena a;
    ArrowSchema s; ArrowArray arr;
    std::memset(&s, 0, sizeof(s));
    std::memset(&arr, 0, sizeof(arr));
    s.format = "+w:4:e";  // not a format import_column recognizes
    s.release = [](ArrowSchema* x) { x->release = nullptr; };
    arr.length = 0;
    arr.release = [](ArrowArray* x) { x->release = nullptr; };
    BoltColumn out;
    EXPECT_FALSE(bolt::arrow::import_column(s, arr, &a, &out));
    s.release(&s);
    arr.release(&arr);
}
