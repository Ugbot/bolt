// test_bolt_fastlanes.cpp — B1: the FastLanes block layout behind
// ColumnFormat::BitPacked / FrameOfRef / DeltaFOR (scalar reference).

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <cstring>
#include <random>
#include <type_traits>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/kernels/bolt_fastlanes.h"

using namespace bolt;
namespace fl = bolt::fastlanes;

namespace {

// Independent transcription of the spiraldb `fastlanes` crate's pack! macro
// (tmp word, flush when the next row starts a new word): the oracle for the
// packed byte order.
template <class U>
void crate_pack(const U* in, uint32_t w, U* out) {
    constexpr uint32_t T = sizeof(U) * 8, L = 1024 / T;
    const uint8_t order[8] = {0, 4, 2, 6, 1, 5, 3, 7};
    if (w == 0) return;
    for (uint32_t lane = 0; lane < L; ++lane) {
        if (w == T) {
            for (uint32_t row = 0; row < T; ++row)
                out[L * row + lane] = in[order[row / 8] * 16 + (row % 8) * 128 + lane];
            continue;
        }
        const U mask = static_cast<U>((U(1) << w) - 1);
        U tmp = 0;
        for (uint32_t row = 0; row < T; ++row) {
            const U src = static_cast<U>(in[order[row / 8] * 16 + (row % 8) * 128 + lane] & mask);
            if (row == 0) tmp = src;
            else tmp = static_cast<U>(tmp | static_cast<U>(src << ((row * w) % T)));
            const uint32_t cur = (row * w) / T, next = ((row + 1) * w) / T;
            if (next > cur) {
                out[L * cur + lane] = tmp;
                const uint32_t rem = ((row + 1) * w) % T;
                tmp = static_cast<U>(rem == 0 ? 0 : src >> (w - rem));
            }
        }
    }
}

template <class U>
void check_layout_all_widths(uint64_t seed) {
    constexpr uint32_t T = sizeof(U) * 8, L = 1024 / T;
    std::mt19937_64 g(seed);
    for (uint32_t w = 0; w <= T; ++w) {
        std::vector<U> in(1024), mine(L * (w ? w : 1), 0xA5), oracle(L * (w ? w : 1), 0x5A),
            back(1024);
        const uint64_t m = w == 64 ? ~0ull : ((1ull << w) - 1);
        for (auto& x : in) x = static_cast<U>(g() & m);
        fl::pack_block<U>(in.data(), w, mine.data());
        crate_pack<U>(in.data(), w, oracle.data());
        if (w) ASSERT_EQ(std::memcmp(mine.data(), oracle.data(), sizeof(U) * L * w), 0)
            << "T=" << T << " W=" << w;
        fl::unpack_block<U>(mine.data(), w, back.data());
        ASSERT_EQ(back, in) << "T=" << T << " W=" << w;
        for (uint32_t i = 0; i < 1024; i += 37) ASSERT_EQ(fl::unpack_one<U>(mine.data(), w, i), in[i]);
    }
}

template <class V>
std::vector<V> make_values(int kind, int64_t n, std::mt19937_64& g) {
    std::vector<V> v(static_cast<size_t>(n));
    using U = std::make_unsigned_t<V>;
    const V lo = std::numeric_limits<V>::min(), hi = std::numeric_limits<V>::max();
    for (int64_t i = 0; i < n; ++i) {
        U x;
        switch (kind) {
            case 0: x = static_cast<U>(g()); break;                              // full range
            case 1: x = static_cast<U>(static_cast<U>(lo) + (g() % 13)); break;  // near min
            case 2: x = static_cast<U>(static_cast<U>(hi) - (g() % 300)); break; // near max
            case 3: x = static_cast<U>(i * 3 + 7); break;                        // fixed step
            case 4: x = static_cast<U>(i * 5 + static_cast<int64_t>(g() % 4)); break;  // near sorted
            default: x = static_cast<U>(1000 - i); break;                        // descending
        }
        v[static_cast<size_t>(i)] = static_cast<V>(x);
    }
    return v;
}

BoltType bolt_type_of(size_t sz, bool sgn) {
    switch (sz) {
        case 1: return sgn ? BoltType::Int8 : BoltType::UInt8;
        case 2: return sgn ? BoltType::Int16 : BoltType::UInt16;
        case 4: return sgn ? BoltType::Int32 : BoltType::UInt32;
        default: return sgn ? BoltType::Int64 : BoltType::UInt64;
    }
}

template <class V>
void column_round_trips(uint64_t seed) {
    using U = std::make_unsigned_t<V>;
    constexpr uint32_t T = sizeof(V) * 8;
    std::mt19937_64 g(seed);
    const BoltType bt = bolt_type_of(sizeof(V), std::is_signed_v<V>);
    for (int64_t n : {0, 1, 2, 63, 1023, 1024, 1025, 3000}) {
        for (int kind = 0; kind < 6; ++kind) {
            Arena a;
            const std::vector<V> v = make_values<V>(kind, n, g);
            // FrameOfRef with the chosen width; one bit less must lose data.
            int64_t ref = 0;
            uint32_t w = 0;
            fl::choose_for<V>(v.data(), n, &ref, &w);
            ASSERT_LE(w, T);
            std::vector<uint8_t> buf(fl::packed_bytes(T, n, w) + 1);
            fl::encode_for<V>(v.data(), n, ref, w, buf.data());
            BoltColumn c = BoltColumn::make_frame_of_ref(buf.data(), static_cast<uint8_t>(w), ref,
                                                        n, bt, &a);
            ASSERT_EQ(c.format, ColumnFormat::FrameOfRef);
            EXPECT_EQ(c.byte_size(), fl::packed_bytes(T, n, w));
            BoltColumn f = c.materialize(&a);
            ASSERT_EQ(f.length, n);
            if (n) ASSERT_EQ(std::memcmp(f.data, v.data(), sizeof(V) * static_cast<size_t>(n)), 0)
                << "FOR T=" << T << " n=" << n << " kind=" << kind;
            if (w > 0) {
                std::vector<V> back(static_cast<size_t>(n));
                std::vector<uint8_t> nb(fl::packed_bytes(T, n, w - 1) + 1);
                fl::encode_for<V>(v.data(), n, ref, w - 1, nb.data());
                fl::decode_for<V>(nb.data(), n, ref, w - 1, back.data());
                ASSERT_NE(back, v) << "width " << w << " is not minimal";
            }
            // DeltaFOR.
            int64_t dref = 0;
            fl::choose_delta<V>(v.data(), n, &dref, &w);
            std::vector<uint8_t> dbuf(fl::delta_bytes(T, n, w) + 1);
            fl::encode_delta_for<V>(v.data(), n, dref, w, dbuf.data());
            BoltColumn d = BoltColumn::make_delta_for(dbuf.data(), static_cast<uint8_t>(w), dref,
                                                      n, bt, &a);
            EXPECT_EQ(d.byte_size(), fl::delta_bytes(T, n, w));
            BoltColumn df = d.materialize(&a);
            ASSERT_EQ(df.length, n);
            if (n) ASSERT_EQ(std::memcmp(df.data, v.data(), sizeof(V) * static_cast<size_t>(n)), 0)
                << "DeltaFOR T=" << T << " n=" << n << " kind=" << kind;
            // BitPacked: ref 0, width of the unsigned max.
            uint64_t mx = 0;
            for (V x : v) mx = std::max<uint64_t>(mx, static_cast<U>(x));
            const uint32_t bw = fl::width_of(mx);
            std::vector<uint8_t> bbuf(fl::packed_bytes(T, n, bw) + 1);
            fl::encode_for<V>(v.data(), n, 0, bw, bbuf.data());
            BoltColumn b = BoltColumn::make_bitpacked(bbuf.data(), static_cast<uint8_t>(bw), n, bt, &a);
            BoltColumn bf = b.materialize(&a);
            if (n) ASSERT_EQ(std::memcmp(bf.data, v.data(), sizeof(V) * static_cast<size_t>(n)), 0);
        }
    }
}

}  // namespace

TEST(FastLanes, OrderAndIndexArePermutations) {
    for (int i = 0; i < 8; ++i) EXPECT_EQ(fl::kOrder[fl::kOrder[i]], i);
    for (uint32_t t : {8u, 16u, 32u, 64u}) {
        std::vector<int> seen(1024, 0);
        for (uint32_t row = 0; row < t; ++row)
            for (uint32_t lane = 0; lane < 1024 / t; ++lane) ++seen[fl::index(row, lane)];
        for (int s : seen) ASSERT_EQ(s, 1) << "T=" << t;
    }
    std::vector<int> seen(1024, 0);
    for (uint32_t i = 0; i < 1024; ++i) ++seen[fl::transpose(i)];
    for (int s : seen) ASSERT_EQ(s, 1);
    // The crate's own known transpose indices.
    const uint32_t kin[8][2] = {{0, 0}, {1, 64}, {16, 32}, {32, 16}, {48, 48}, {64, 8}, {128, 1}, {1023, 1023}};
    for (auto& k : kin) EXPECT_EQ(fl::transpose(k[0]), k[1]);
}

TEST(FastLanes, RowLaneInvertsIndex) {
    auto check = [](auto tag) {
        using U = decltype(tag);
        constexpr uint32_t T = sizeof(U) * 8;
        for (uint32_t row = 0; row < T; ++row)
            for (uint32_t lane = 0; lane < 1024 / T; ++lane) {
                uint32_t r = 0, l = 0;
                fl::row_lane<U>(fl::index(row, lane), &r, &l);
                ASSERT_EQ(r, row);
                ASSERT_EQ(l, lane);
            }
    };
    check(uint8_t{}); check(uint16_t{}); check(uint32_t{}); check(uint64_t{});
}

TEST(FastLanes, PackMatchesCrateOracleEveryWidth) {
    check_layout_all_widths<uint8_t>(1);
    check_layout_all_widths<uint16_t>(2);
    check_layout_all_widths<uint32_t>(3);
    check_layout_all_widths<uint64_t>(4);
}

TEST(FastLanes, ColumnRoundTripsEveryIntegerType) {
    column_round_trips<int8_t>(11);
    column_round_trips<uint8_t>(12);
    column_round_trips<int16_t>(13);
    column_round_trips<uint16_t>(14);
    column_round_trips<int32_t>(15);
    column_round_trips<uint32_t>(16);
    column_round_trips<int64_t>(17);
    column_round_trips<uint64_t>(18);
}

TEST(FastLanes, DeltaLanesAreContiguousRunsForT64) {
    // For T = 64 lane l is the natural run [64 l, 64 l + 64): a sorted column
    // with step 3 packs to W = 0 and the per-lane bases are 64 * 3 apart.
    std::vector<int64_t> v(1024);
    for (int i = 0; i < 1024; ++i) v[static_cast<size_t>(i)] = 100 + 3 * i;
    int64_t dref = 0;
    uint32_t w = 99;
    fl::choose_delta<int64_t>(v.data(), 1024, &dref, &w);
    EXPECT_EQ(w, 0u);
    EXPECT_EQ(dref, 3);
    std::vector<uint64_t> buf(fl::delta_bytes(64, 1024, 0) / 8);
    fl::encode_delta_for<int64_t>(v.data(), 1024, dref, 0, buf.data());
    ASSERT_EQ(buf.size(), 16u);   // just the 16 lane bases
    for (uint32_t lane = 0; lane < 16; ++lane)
        EXPECT_EQ(static_cast<int64_t>(buf[lane]), 100 + 3 * 64 * int64_t(lane) - dref);
}

TEST(FastLanes, TemporalAndDecimalTypesUseTheIntegerLayout) {
    Arena a;
    std::vector<int64_t> ts(2000);
    for (int i = 0; i < 2000; ++i) ts[static_cast<size_t>(i)] = 1'700'000'000'000'000 + 1000 * i;
    int64_t dref = 0;
    uint32_t w = 0;
    fl::choose_delta<int64_t>(ts.data(), 2000, &dref, &w);
    std::vector<uint8_t> buf(fl::delta_bytes(64, 2000, w) + 1);
    fl::encode_delta_for<int64_t>(ts.data(), 2000, dref, w, buf.data());
    for (BoltType t : {BoltType::Timestamp, BoltType::Duration, BoltType::Date64, BoltType::Decimal64}) {
        BoltColumn c = BoltColumn::make_delta_for(buf.data(), static_cast<uint8_t>(w), dref, 2000, t, &a);
        ASSERT_EQ(c.format, ColumnFormat::DeltaFOR) << int(t);
        BoltColumn f = c.materialize(&a);
        ASSERT_EQ(std::memcmp(f.data, ts.data(), 8 * 2000), 0);
    }
}

TEST(FastLanes, RefusesBadShapes) {
    Arena a;
    uint64_t words[64] = {};
    EXPECT_EQ(BoltColumn::make_bitpacked(words, 3, 10, BoltType::Float64, &a).format, ColumnFormat::Flat);
    EXPECT_EQ(BoltColumn::make_bitpacked(words, 33, 10, BoltType::Int32, &a).format, ColumnFormat::Flat);
    EXPECT_EQ(BoltColumn::make_fastlanes(ColumnFormat::BitPacked, words, 3, 5, 10,
                                         BoltType::Int32, &a).format, ColumnFormat::Flat);
    EXPECT_EQ(BoltColumn::make_delta_for(nullptr, 0, 0, 10, BoltType::Int64, &a).format,
              ColumnFormat::Flat);
    // W = 0 BitPacked needs no buffer: every value is 0.
    BoltColumn z = BoltColumn::make_bitpacked(nullptr, 0, 3000, BoltType::UInt16, &a);
    ASSERT_EQ(z.format, ColumnFormat::BitPacked);
    BoltColumn zf = z.materialize(&a);
    for (int i = 0; i < 3000; ++i) ASSERT_EQ(static_cast<const uint16_t*>(zf.data)[i], 0);
}

// The padding rule is part of the bytes (deterministic writer): a partial
// last block encodes exactly as if the last value repeated to 1,024.
TEST(FastLanes, PartialBlockPadsWithTheLastValue) {
    std::mt19937_64 g(5);
    std::vector<int32_t> v(1300), full(2048);
    for (auto& x : v) x = static_cast<int32_t>(g() % 5000);
    for (size_t i = 0; i < full.size(); ++i) full[i] = v[i < v.size() ? i : v.size() - 1];
    int64_t dref = 0;
    uint32_t w = 0;
    fl::choose_delta<int32_t>(full.data(), 2048, &dref, &w);
    std::vector<uint8_t> a(fl::delta_bytes(32, 2048, w)), b(fl::delta_bytes(32, 1300, w));
    fl::encode_delta_for<int32_t>(full.data(), 2048, dref, w, a.data());
    fl::encode_delta_for<int32_t>(v.data(), 1300, dref, w, b.data());
    EXPECT_EQ(a, b);
    int64_t dref2 = 0;
    uint32_t w2 = 0;
    fl::choose_delta<int32_t>(v.data(), 1300, &dref2, &w2);
    EXPECT_EQ(dref2, dref);
    EXPECT_EQ(w2, w);
}
