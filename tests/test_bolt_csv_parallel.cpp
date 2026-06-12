// W3 — parse_csv_parallel parity vs the serial parser.
//
// The parallel parser must be VALUE-identical to parse_csv on every
// column. Fixed-width columns compare byte-for-byte; Utf8 compares by
// CONTENT (spilled ref.offsets legitimately differ: serial packs the
// overflow buffer sequentially, parallel windows it per chunk).
// Coverage: mixed types, spilled (>12B) strings, quoted fields, CRLF
// line endings, empty fields, a final line without a trailing newline,
// header skipping, and the small-input serial fallback.

#include "bolt/ingest/bolt_csv.h"

#include "bolt/bolt_arena.h"
#include "bolt/bolt_scheduler.h"
#include "bolt/kernels/bolt_utf8.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

using namespace bolt;
using namespace bolt::ingest;

std::uint64_t splitmix64(std::uint64_t* s) noexcept {
    std::uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Build a >4 MiB CSV: int64 | decimal(2) | date | utf8 (mixed inline/
// spilled/quoted/empty) with occasional CRLF endings and NO trailing
// newline on the last row.
std::string make_csv(std::int64_t rows, bool header) {
    std::string s;
    s.reserve(static_cast<std::size_t>(rows) * 64);
    if (header) s += "a|b|c|d\n";
    std::uint64_t rng = 0xC5AA11ULL;
    char buf[96];
    for (std::int64_t i = 0; i < rows; ++i) {
        const std::uint64_t r = splitmix64(&rng);
        const int dec_cents = static_cast<int>(r % 99989);
        std::snprintf(buf, sizeof(buf), "%lld|%d.%02d|199%d-0%d-1%d|",
                      static_cast<long long>(i * 7 - 1000),
                      dec_cents / 100, dec_cents % 100,
                      static_cast<int>(r % 9), static_cast<int>(r % 8) + 1,
                      static_cast<int>(r % 9));
        s += buf;
        switch (r % 5) {
            case 0: s += "short"; break;
            case 1: s += "a-spilled-string-well-over-twelve-bytes"; break;
            case 2: s += "\"quoted,value\""; break;
            case 3: /* empty field */ break;
            default: {
                std::snprintf(buf, sizeof(buf), "row%lld-tail-padding-xyz",
                              static_cast<long long>(i));
                s += buf;
                break;
            }
        }
        if (i + 1 < rows) {
            s += (r % 7 == 0) ? "\r\n" : "\n";   // mixed CRLF
        }
        // Last row: NO trailing newline (the partial-final-line case).
    }
    return s;
}

void fill_schema(CsvSchema* cs, bool header) noexcept {
    std::memset(cs, 0, sizeof(*cs));
    cs->num_cols = 4;
    cs->delimiter = '|';
    cs->has_header = header;
    cs->col_types[0] = BoltType::Int64;
    cs->col_types[1] = BoltType::Decimal64; cs->col_scales[1] = 2;
    cs->col_types[2] = BoltType::Date32;
    cs->col_types[3] = BoltType::Utf8;
}

void expect_batches_equal(const BoltBatch& a, const BoltBatch& b) {
    ASSERT_EQ(a.num_rows, b.num_rows);
    ASSERT_EQ(a.num_cols, b.num_cols);
    const std::int64_t n = a.num_rows;
    for (std::uint32_t c = 0; c < a.num_cols; ++c) {
        const BoltColumn& ca = a.col(c);
        const BoltColumn& cb = b.col(c);
        ASSERT_EQ(ca.type, cb.type) << "col " << c;
        if (ca.type == BoltType::Utf8) {
            const auto* va = static_cast<const StringView*>(ca.data);
            const auto* vb = static_cast<const StringView*>(cb.data);
            const char* base_a = static_cast<const char*>(ca.str_overflow_base);
            const char* base_b = static_cast<const char*>(cb.str_overflow_base);
            for (std::int64_t r = 0; r < n; ++r) {
                ASSERT_EQ(va[r].length, vb[r].length) << "col " << c << " row " << r;
                const char* pa = kernels::utf8::sv_bytes(va[r], base_a);
                const char* pb = kernels::utf8::sv_bytes(vb[r], base_b);
                ASSERT_EQ(0, std::memcmp(pa, pb, va[r].length))
                    << "col " << c << " row " << r;
            }
        } else {
            const std::size_t w = type_size(ca.type);
            ASSERT_GT(w, 0u);
            ASSERT_EQ(0, std::memcmp(ca.data, cb.data,
                                     w * static_cast<std::size_t>(n)))
                << "col " << c;
        }
    }
}

TEST(BoltCsvParallel, ParityWithSerialLargeMixed) {
    // ~6.5 MiB => above the 4 MiB parallel threshold; >4 chunks.
    const std::string csv = make_csv(120000, /*header=*/true);
    ASSERT_GT(csv.size(), 4u << 20);

    CsvSchema cs{};
    fill_schema(&cs, true);

    Scheduler sched{};
    ASSERT_TRUE(sched.init(4));

    Arena a1, a2;
    BoltBatch b_serial{}, b_par{};
    ASSERT_TRUE(parse_csv(csv.data(), csv.size(), cs, &a1, &b_serial));
    ASSERT_TRUE(parse_csv_parallel(csv.data(), csv.size(), cs, &a2, &sched,
                                   &b_par));
    expect_batches_equal(b_serial, b_par);
    sched.shutdown();
}

TEST(BoltCsvParallel, SmallInputTakesSerialPath) {
    const std::string csv = make_csv(50, /*header=*/false);
    CsvSchema cs{};
    fill_schema(&cs, false);
    Scheduler sched{};
    ASSERT_TRUE(sched.init(2));
    Arena a1, a2;
    BoltBatch b_serial{}, b_par{};
    ASSERT_TRUE(parse_csv(csv.data(), csv.size(), cs, &a1, &b_serial));
    ASSERT_TRUE(parse_csv_parallel(csv.data(), csv.size(), cs, &a2, &sched,
                                   &b_par));
    expect_batches_equal(b_serial, b_par);
    sched.shutdown();
}

TEST(BoltCsvParallel, NullSchedulerFallsBackSerial) {
    const std::string csv = make_csv(2000, /*header=*/true);
    CsvSchema cs{};
    fill_schema(&cs, true);
    Arena a1, a2;
    BoltBatch b_serial{}, b_par{};
    ASSERT_TRUE(parse_csv(csv.data(), csv.size(), cs, &a1, &b_serial));
    ASSERT_TRUE(parse_csv_parallel(csv.data(), csv.size(), cs, &a2,
                                   /*sched=*/nullptr, &b_par));
    expect_batches_equal(b_serial, b_par);
}

}  // namespace
