// make_view, value access and range decode for the FastLanes formats
// (BitPacked / FrameOfRef / DeltaFOR, MSEG B1).

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
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

BoltType bolt_type_of(size_t sz, bool sgn) {
    switch (sz) {
        case 1: return sgn ? BoltType::Int8 : BoltType::UInt8;
        case 2: return sgn ? BoltType::Int16 : BoltType::UInt16;
        case 4: return sgn ? BoltType::Int32 : BoltType::UInt32;
        default: return sgn ? BoltType::Int64 : BoltType::UInt64;
    }
}

template <class V>
std::vector<V> values(int64_t n, int kind, std::mt19937_64& g) {
    using U = std::make_unsigned_t<V>;
    std::vector<V> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        U x = kind == 0 ? static_cast<U>(g()) : static_cast<U>(i * 7 + int64_t(g() % 5));
        v[size_t(i)] = static_cast<V>(x);
    }
    return v;
}

// The encoded column of each format, buffers owned by the caller.
template <class V>
struct Encoded {
    std::vector<uint8_t> buf[3];
    BoltColumn col[3];
};

template <class V>
void encode_all(const std::vector<V>& v, Arena* a, Encoded<V>* e) {
    using U = std::make_unsigned_t<V>;
    constexpr uint32_t T = sizeof(V) * 8;
    const int64_t n = int64_t(v.size());
    const BoltType bt = bolt_type_of(sizeof(V), std::is_signed_v<V>);
    int64_t ref = 0, dref = 0;
    uint32_t w = 0, dw = 0;
    fl::choose_for<V>(v.data(), n, &ref, &w);
    e->buf[0].resize(fl::packed_bytes(T, n, w) + 1);
    fl::encode_for<V>(v.data(), n, ref, w, e->buf[0].data());
    e->col[0] = BoltColumn::make_frame_of_ref(e->buf[0].data(), uint8_t(w), ref, n, bt, a);
    uint64_t mx = 0;
    for (V x : v) mx = std::max<uint64_t>(mx, static_cast<U>(x));
    const uint32_t bw = fl::width_of(mx);
    e->buf[1].resize(fl::packed_bytes(T, n, bw) + 1);
    fl::encode_for<V>(v.data(), n, 0, bw, e->buf[1].data());
    e->col[1] = BoltColumn::make_bitpacked(e->buf[1].data(), uint8_t(bw), n, bt, a);
    fl::choose_delta<V>(v.data(), n, &dref, &dw);
    e->buf[2].resize(fl::delta_bytes(T, n, dw) + 1);
    fl::encode_delta_for<V>(v.data(), n, dref, dw, e->buf[2].data());
    e->col[2] = BoltColumn::make_delta_for(e->buf[2].data(), uint8_t(dw), dref, n, bt, a);
}

size_t span_bytes(const BoltColumn& parent, int64_t first, int64_t len) {
    if (len == 0) return 0;
    const int64_t blocks = (first + len - 1) / 1024 - first / 1024 + 1;
    const size_t w = size_t(parent.seq_step);
    const size_t per = 128u * (parent.format == ColumnFormat::DeltaFOR ? w + 1 : w);
    return size_t(blocks) * per;
}

template <class V>
void check_view(const BoltColumn& parent, const std::vector<V>& v, int64_t base, int64_t off,
                int64_t len, Arena* a) {
    BoltColumn s = BoltColumn::make_view(parent, off, len);
    ASSERT_EQ(s.format, parent.format);
    ASSERT_EQ(s.length, len);
    ASSERT_EQ(s.data, parent.data) << "a FastLanes view is zero-copy";
    EXPECT_EQ(s.byte_size(), span_bytes(parent, base + off, len));
    BoltColumn m = s.materialize(a);
    ASSERT_EQ(m.format, ColumnFormat::Flat);
    ASSERT_EQ(m.length, len);
    if (len) ASSERT_EQ(std::memcmp(m.data, v.data() + base + off, sizeof(V) * size_t(len)), 0)
        << "fmt=" << int(parent.format) << " T=" << sizeof(V) * 8 << " base=" << base
        << " off=" << off << " len=" << len;
    for (int64_t i = 0; i < len; i += 1 + len / 97)
        ASSERT_EQ(s.template fastlanes_value<V>(i), v[size_t(base + off + i)])
            << "fmt=" << int(parent.format) << " i=" << i;
    if (len) ASSERT_EQ(s.template fastlanes_value<V>(len - 1), v[size_t(base + off + len - 1)]);
}

template <class V>
void views_of_every_format(uint64_t seed) {
    std::mt19937_64 g(seed);
    for (int64_t n : {1, 1023, 1024, 1025, 3000, 9000}) {
        for (int kind = 0; kind < 2; ++kind) {
            Arena a;
            const std::vector<V> v = values<V>(n, kind, g);
            Encoded<V> e;
            encode_all<V>(v, &a, &e);
            for (const BoltColumn& c : e.col) {
                const int64_t fixed[][2] = {{0, n}, {0, 0}, {n - 1, 1}, {0, 1},
                                            {n / 2, n - n / 2}, {n > 1024 ? 1024 : 0, n > 1024 ? n - 1024 : n}};
                for (const auto& p : fixed) check_view<V>(c, v, 0, p[0], p[1], &a);
                for (int r = 0; r < 12; ++r) {
                    const int64_t off = int64_t(g() % uint64_t(n));
                    const int64_t len = int64_t(g() % uint64_t(n - off + 1));
                    check_view<V>(c, v, 0, off, len, &a);
                    // A view of a view addresses the parent's rows.
                    if (len > 0) {
                        BoltColumn s = BoltColumn::make_view(c, off, len);
                        const int64_t o2 = int64_t(g() % uint64_t(len));
                        const int64_t l2 = int64_t(g() % uint64_t(len - o2 + 1));
                        check_view<V>(s, v, off, o2, l2, &a);
                    }
                }
            }
        }
    }
}

}  // namespace

TEST(FastLanesView, UntransposeInvertsTranspose) {
    for (uint32_t i = 0; i < fl::kBlockValues; ++i) {
        ASSERT_EQ(fl::untranspose(fl::transpose(i)), i);
        ASSERT_EQ(fl::transpose(fl::untranspose(i)), i);
    }
}

TEST(FastLanesView, ViewsAreZeroCopyAndReadTheParentRows) {
    for (uint64_t seed = 0; seed < 4; ++seed) {
        views_of_every_format<int8_t>(seed);
        views_of_every_format<uint8_t>(seed);
        views_of_every_format<int16_t>(seed);
        views_of_every_format<uint16_t>(seed);
        views_of_every_format<int32_t>(seed);
        views_of_every_format<uint32_t>(seed);
        views_of_every_format<int64_t>(seed);
        views_of_every_format<uint64_t>(seed);
    }
}

TEST(FastLanesView, RangeDecodeMatchesFullDecode) {
    std::mt19937_64 g(7);
    const int64_t n = 5000;
    const std::vector<int32_t> v = values<int32_t>(n, 1, g);
    Arena a;
    Encoded<int32_t> e;
    encode_all<int32_t>(v, &a, &e);
    const BoltColumn& d = e.col[2];
    std::vector<int32_t> out(size_t(n), 0);
    for (int64_t first : {0, 1, 1023, 1024, 2047, 4999}) {
        const int64_t cnt = n - first;
        fl::decode_delta_for_range<int32_t>(d.data, first, cnt, d.seq_offset,
                                            uint32_t(d.seq_step), out.data());
        ASSERT_EQ(std::memcmp(out.data(), v.data() + first, sizeof(int32_t) * size_t(cnt)), 0);
        const BoltColumn& f = e.col[0];
        fl::decode_for_range<int32_t>(f.data, first, cnt, f.seq_offset, uint32_t(f.seq_step),
                                      out.data());
        ASSERT_EQ(std::memcmp(out.data(), v.data() + first, sizeof(int32_t) * size_t(cnt)), 0);
    }
}

TEST(FastLanesView, TemporalTypeViewKeepsItsMeaning) {
    std::mt19937_64 g(3);
    const std::vector<int64_t> v = values<int64_t>(2500, 1, g);
    Arena a;
    int64_t dref = 0;
    uint32_t w = 0;
    fl::choose_delta<int64_t>(v.data(), 2500, &dref, &w);
    std::vector<uint8_t> buf(fl::delta_bytes(64, 2500, w));
    fl::encode_delta_for<int64_t>(v.data(), 2500, dref, w, buf.data());
    BoltColumn c = BoltColumn::make_delta_for(buf.data(), uint8_t(w), dref, 2500,
                                              BoltType::Timestamp, &a);
    BoltColumn s = BoltColumn::make_view(c, 1500, 600);
    EXPECT_EQ(s.type, BoltType::Timestamp);
    EXPECT_EQ(s.seq_offset, c.seq_offset);
    EXPECT_EQ(s.seq_step, c.seq_step);
    EXPECT_EQ(s.fastlanes_value<int64_t>(0), v[1500]);
    EXPECT_EQ(s.fastlanes_value<int64_t>(599), v[2099]);
}
