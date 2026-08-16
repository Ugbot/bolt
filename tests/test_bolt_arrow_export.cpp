// test_bolt_arrow_export — the C Data Interface contract for
// bolt::arrow::export_column (bolt/bolt_arrow.h).
//
// The companion test in the Gestalt2 superproject validates against REAL
// pyarrow, which is the only way to catch a wrong format string. This one
// needs no external dependency, so it runs everywhere (including Windows CI)
// and pins the parts of the spec that are checkable in-process — above all
// the release contract, which the previous export violated and whose old test
// could not fail, because "does not crash" is exactly what a no-op does.
#include <cstdint>
#include <cstring>
#include <string>

#include <gtest/gtest.h>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_arrow.h"
#include "bolt/bolt_column.h"

using namespace bolt;

namespace {

BoltColumn make_i64(Arena* a, int64_t n, bool with_nulls) {
    auto* d = static_cast<int64_t*>(a->allocate(n * 8, 64));
    for (int64_t i = 0; i < n; ++i) d[i] = 100 + i;
    uint8_t* v = nullptr;
    if (with_nulls) {
        const size_t nb = static_cast<size_t>((n + 7) / 8);
        v = static_cast<uint8_t*>(a->allocate(nb, 64));
        std::memset(v, 0xFF, nb);
        for (int64_t i = 0; i < n; i += 2) {
            v[i >> 3] &= static_cast<uint8_t>(~(1u << (i & 7)));
        }
    }
    return BoltColumn::make_flat(d, v, n, BoltType::Int64);
}

// Utf8 in the StringView layout with a >12-byte value spilled to an overflow
// base — the layout the parquet reader and hash-agg produce.
BoltColumn make_utf8_view(Arena* a, const char* const* vals, int64_t n) {
    auto* views = static_cast<StringView*>(a->allocate(n * sizeof(StringView), 64));
    std::memset(views, 0, static_cast<size_t>(n) * sizeof(StringView));
    std::string spill;
    for (int64_t i = 0; i < n; ++i) {
        const uint32_t len = static_cast<uint32_t>(std::strlen(vals[i]));
        views[i].length = len;
        if (len <= 12) {
            std::memcpy(views[i].prefix, vals[i], len);
        } else {
            std::memcpy(views[i].prefix, vals[i], 4);
            views[i].ref.buf_idx = 0;
            views[i].ref.offset = static_cast<uint32_t>(spill.size());
            spill += vals[i];
        }
    }
    auto* base = static_cast<char*>(a->allocate(spill.size() + 1, 64));
    std::memcpy(base, spill.data(), spill.size());
    auto c = BoltColumn::make_flat(views, nullptr, n, BoltType::Utf8);
    c.str_overflow_base = base;
    return c;
}

}  // namespace

// ---- the release contract (what the old export got wrong) ----

TEST(ArrowExport, ReleaseMarksStructReleasedPerSpec) {
    Arena a;
    BoltColumn c = make_i64(&a, 4, false);
    ArrowSchema s; ArrowArray arr;
    ASSERT_TRUE(bolt::arrow::export_column(c, 4, "x", &s, &arr));

    ASSERT_NE(s.release, nullptr);
    s.release(&s);
    // The spec REQUIRES this: a consumer tests `release == NULL` to know the
    // struct was released. pyarrow aborts the process when it is not cleared,
    // which made the previous export unusable for EVERY type.
    EXPECT_EQ(s.release, nullptr);

    ASSERT_NE(arr.release, nullptr);
    arr.release(&arr);
    EXPECT_EQ(arr.release, nullptr);
}

TEST(ArrowExport, ReleaseIsIdempotent) {
    Arena a;
    BoltColumn c = make_i64(&a, 4, false);
    ArrowSchema s; ArrowArray arr;
    ASSERT_TRUE(bolt::arrow::export_column(c, 4, "x", &s, &arr));
    // Save the pointers: a correct release sets `release` to NULL, so calling
    // `s.release(&s)` a second time would dereference a null function pointer
    // — the CONSUMER is required to check for NULL. What must hold is that the
    // callback ITSELF is safe if invoked again on an already-released struct.
    auto* sfn = s.release;
    auto* afn = arr.release;
    sfn(&s);   sfn(&s);
    afn(&arr); afn(&arr);
    EXPECT_EQ(s.release, nullptr);
    EXPECT_EQ(arr.release, nullptr);
}

TEST(ArrowExport, SchemaAndArrayReleaseIndependentlyInEitherOrder) {
    // They are separate structs a consumer owns separately. Sharing owned
    // state between them means whichever is released first frees memory the
    // other still points at.
    Arena a;
    BoltColumn c = make_i64(&a, 4, false);
    {
        ArrowSchema s; ArrowArray arr;
        ASSERT_TRUE(bolt::arrow::export_column(c, 4, "x", &s, &arr));
        s.release(&s);
        // Array must still be intact and readable AFTER the schema is gone.
        const auto* d = static_cast<const int64_t*>(arr.buffers[1]);
        EXPECT_EQ(d[0], 100);
        arr.release(&arr);
    }
    {
        ArrowSchema s; ArrowArray arr;
        ASSERT_TRUE(bolt::arrow::export_column(c, 4, "x", &s, &arr));
        arr.release(&arr);
        EXPECT_STREQ(s.format, "l");     // schema intact after the array went
        s.release(&s);
    }
}

// ---- per-export state (the thread_local aliasing defect) ----

TEST(ArrowExport, TwoExportsDoNotAliasEachOther) {
    Arena a;
    auto* d0 = static_cast<int64_t*>(a.allocate(4 * 8, 64));
    auto* d1 = static_cast<int64_t*>(a.allocate(4 * 8, 64));
    for (int i = 0; i < 4; ++i) { d0[i] = 100 + i; d1[i] = 900 + i; }
    auto c0 = BoltColumn::make_flat(d0, nullptr, 4, BoltType::Int64);
    auto c1 = BoltColumn::make_flat(d1, nullptr, 4, BoltType::Int64);

    ArrowSchema s0, s1; ArrowArray a0, a1;
    ASSERT_TRUE(bolt::arrow::export_column(c0, 4, "left", &s0, &a0));
    ASSERT_TRUE(bolt::arrow::export_column(c1, 4, "right", &s1, &a1));

    // Previously both buffer lists were ONE static thread_local array, so the
    // first column read back as the second.
    EXPECT_NE(a0.buffers, a1.buffers);
    EXPECT_EQ(static_cast<const int64_t*>(a0.buffers[1])[0], 100);
    EXPECT_EQ(static_cast<const int64_t*>(a1.buffers[1])[0], 900);
    EXPECT_STREQ(s0.name, "left");
    EXPECT_STREQ(s1.name, "right");

    s0.release(&s0); s1.release(&s1); a0.release(&a0); a1.release(&a1);
}

// ---- ownership: the export outlives its source arena ----

TEST(ArrowExport, SurvivesArenaReset) {
    Arena a;
    const char* vals[3] = {"alpha", "a value longer than twelve bytes", ""};
    BoltColumn c = make_utf8_view(&a, vals, 3);
    ArrowSchema s; ArrowArray arr;
    ASSERT_TRUE(bolt::arrow::export_column(c, 3, "labels", &s, &arr));

    a.reset();   // recycle every source buffer the column pointed at

    const auto* offs = static_cast<const int32_t*>(arr.buffers[1]);
    const auto* bytes = static_cast<const char*>(arr.buffers[2]);
    ASSERT_EQ(offs[0], 0);
    EXPECT_EQ(std::string(bytes + offs[0], offs[1] - offs[0]), "alpha");
    EXPECT_EQ(std::string(bytes + offs[1], offs[2] - offs[1]),
              "a value longer than twelve bytes");
    EXPECT_EQ(offs[3] - offs[2], 0);
    s.release(&s); arr.release(&arr);
}

// ---- format strings and buffer layout ----

TEST(ArrowExport, DecimalCarriesTheColumnsRealScale) {
    Arena a;
    auto* d = static_cast<unsigned char*>(a.allocate(2 * 16, 64));
    std::memset(d, 0, 32);
    int64_t m = 12345;
    std::memcpy(d, &m, 8);
    auto c = BoltColumn::make_flat(d, nullptr, 2, BoltType::Decimal128);
    c.decimal_scale = 2;
    c.type_size_bytes = 16;

    ArrowSchema s; ArrowArray arr;
    ASSERT_TRUE(bolt::arrow::export_column(c, 2, "amt", &s, &arr));
    // Was hardcoded "d:38,10" regardless of the column — a 10^8 error.
    EXPECT_STREQ(s.format, "d:38,2");
    s.release(&s); arr.release(&arr);
}

TEST(ArrowExport, Utf8IsRealArrowStringWithThreeBuffers) {
    Arena a;
    const char* vals[2] = {"ab", "a value longer than twelve bytes"};
    BoltColumn c = make_utf8_view(&a, vals, 2);
    ArrowSchema s; ArrowArray arr;
    ASSERT_TRUE(bolt::arrow::export_column(c, 2, "l", &s, &arr));

    // Was "vu" (StringView) with 2 buffers, which is invalid — and the
    // spilled-value base was never exported at all.
    EXPECT_STREQ(s.format, "u");
    EXPECT_EQ(arr.n_buffers, 3);
    const auto* offs = static_cast<const int32_t*>(arr.buffers[1]);
    EXPECT_EQ(offs[0], 0);
    EXPECT_EQ(offs[1], 2);
    EXPECT_EQ(offs[2], 2 + 32);          // monotonic, spilled value included
    s.release(&s); arr.release(&arr);
}

TEST(ArrowExport, BoolIsBitPackedNotBytePacked) {
    Arena a;
    const int64_t n = 8;
    auto* d = static_cast<uint8_t*>(a.allocate(n, 64));
    for (int64_t i = 0; i < n; ++i) d[i] = (i % 3 == 0) ? 1 : 0;
    auto c = BoltColumn::make_flat(d, nullptr, n, BoltType::Bool);

    ArrowSchema s; ArrowArray arr;
    ASSERT_TRUE(bolt::arrow::export_column(c, n, "f", &s, &arr));
    EXPECT_STREQ(s.format, "b");
    // bolt stores one BYTE per value; Arrow "b" is one BIT, LSB-first.
    const auto* bits = static_cast<const uint8_t*>(arr.buffers[1]);
    EXPECT_EQ(bits[0], 0b01001001);
    s.release(&s); arr.release(&arr);
}

// ---- null_count and fail-closed ----

TEST(ArrowExport, NullCountIsCountedFromTheBitmap) {
    Arena a;
    BoltColumn c = make_i64(&a, 8, /*with_nulls=*/true);
    // Deliberately do NOT call compute_stats_numeric(): the old export read
    // that stat, so an uncomputed column claimed zero nulls and consumers
    // silently dropped nothing.
    ArrowSchema s; ArrowArray arr;
    ASSERT_TRUE(bolt::arrow::export_column(c, 8, "x", &s, &arr));
    EXPECT_EQ(arr.null_count, 4);
    EXPECT_EQ(s.flags & 2, 2);           // ARROW_FLAG_NULLABLE
    s.release(&s); arr.release(&arr);
}

TEST(ArrowExport, UnsupportedFormatFailsClosed) {
    Arena a;
    BoltColumn c = make_i64(&a, 4, false);
    c.format = ColumnFormat::Dictionary;   // values are not resolvable as Flat
    ArrowSchema s; ArrowArray arr;
    // Was guarded only by an assert — compiled out under NDEBUG, giving a
    // heap over-read as the consumer read key bytes as int64s.
    EXPECT_FALSE(bolt::arrow::export_column(c, 4, "d", &s, &arr));
    EXPECT_EQ(s.release, nullptr);        // nothing half-built handed out
    EXPECT_EQ(arr.release, nullptr);
}

TEST(ArrowExport, UnsupportedTypeFailsClosed) {
    Arena a;
    auto* d = static_cast<int64_t*>(a.allocate(4 * 8, 64));
    auto c = BoltColumn::make_flat(d, nullptr, 4, BoltType::Embedding);
    ArrowSchema s; ArrowArray arr;
    // Embedding has no Arrow short name; exporting "" is not a valid format.
    EXPECT_FALSE(bolt::arrow::export_column(c, 4, "e", &s, &arr));
}

TEST(ArrowExport, EmptyColumnIsValid) {
    Arena a;
    BoltColumn c = make_i64(&a, 0, false);
    ArrowSchema s; ArrowArray arr;
    ASSERT_TRUE(bolt::arrow::export_column(c, 0, "x", &s, &arr));
    EXPECT_EQ(arr.length, 0);
    EXPECT_EQ(arr.null_count, 0);
    s.release(&s); arr.release(&arr);
}
