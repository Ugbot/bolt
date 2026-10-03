// test_bolt_key_encode.cpp — canonical key bytes (MSEG §6.5): memcmp order
// of the encoding equals value order for every key kind and for composites
// with text in any position.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "bolt/kernels/bolt_key_encode.h"

namespace {

using bolt::BoltColumn;
using bolt::BoltType;
using bolt::KeyEncodeStatus;
using bolt::StringView;

int bytes_cmp(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    const size_t m = std::min(a.size(), b.size());
    const int c = m ? std::memcmp(a.data(), b.data(), m) : 0;
    if (c != 0) return c < 0 ? -1 : 1;
    return a.size() < b.size() ? -1 : (a.size() > b.size() ? 1 : 0);
}

template <typename T>
std::vector<uint8_t> enc1(T v, BoltType t) {
    T slot[2] = {v, v};
    BoltColumn c = BoltColumn::make_flat(slot, nullptr, 1, t);
    uint8_t buf[64];
    uint32_t n = 0;
    EXPECT_EQ(bolt::key_encode_cell(c, 0, buf, sizeof(buf), &n), KeyEncodeStatus::kOk);
    return std::vector<uint8_t>(buf, buf + n);
}

std::vector<uint8_t> enc_text(const std::string& s) {
    std::vector<uint8_t> out(s.size() * 2 + 2);
    const uint32_t n = bolt::key_put_text(out.data(), static_cast<uint32_t>(out.size()),
                                          reinterpret_cast<const uint8_t*>(s.data()),
                                          static_cast<uint32_t>(s.size()));
    EXPECT_GT(n, 0u);
    out.resize(n);
    return out;
}

template <typename T>
void check_ints(BoltType t, std::mt19937_64& rng) {
    std::vector<T> v = {std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), T(0), T(1)};
    for (int i = 0; i < 400; ++i) v.push_back(static_cast<T>(rng()));
    for (size_t i = 0; i < v.size(); ++i)
        for (size_t j = 0; j < 20; ++j) {
            const T a = v[i], b = v[(i * 7 + j) % v.size()];
            const int want = a < b ? -1 : (a > b ? 1 : 0);
            ASSERT_EQ(bytes_cmp(enc1(a, t), enc1(b, t)), want) << +a << " vs " << +b;
        }
}

TEST(KeyEncode, IntegersOrderAsValues) {
    std::mt19937_64 rng(7);
    check_ints<int8_t>(BoltType::Int8, rng);
    check_ints<int16_t>(BoltType::Int16, rng);
    check_ints<int32_t>(BoltType::Int32, rng);
    check_ints<int64_t>(BoltType::Int64, rng);
    check_ints<uint8_t>(BoltType::UInt8, rng);
    check_ints<uint16_t>(BoltType::UInt16, rng);
    check_ints<uint32_t>(BoltType::UInt32, rng);
    check_ints<uint64_t>(BoltType::UInt64, rng);
    check_ints<int32_t>(BoltType::Date32, rng);
    check_ints<int64_t>(BoltType::Timestamp, rng);
    EXPECT_EQ(enc1<int64_t>(-1, BoltType::Int64).size(), 8u);
}

TEST(KeyEncode, Decimal128OrdersAsTwosComplement) {
    std::mt19937_64 rng(11);
    std::vector<__int128> v = {0, -1, 1};
    for (int i = 0; i < 300; ++i) {
        const __int128 x = (static_cast<__int128>(static_cast<int64_t>(rng())) << 64) |
                           static_cast<__int128>(rng());
        v.push_back(x);
    }
    for (size_t i = 0; i < v.size(); ++i)
        for (size_t j = 0; j < v.size(); j += 7) {
            const int want = v[i] < v[j] ? -1 : (v[i] > v[j] ? 1 : 0);
            ASSERT_EQ(bytes_cmp(enc1(v[i], BoltType::Decimal128), enc1(v[j], BoltType::Decimal128)), want);
        }
}

TEST(KeyEncode, FloatsTotalOrderZeroAndNan) {
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> v = {-inf, -1e300, -2.5, -1e-300, 0.0, 1e-300, 1.0, 3.5, 1e300, inf};
    for (size_t i = 0; i + 1 < v.size(); ++i)
        EXPECT_LT(bytes_cmp(enc1(v[i], BoltType::Float64), enc1(v[i + 1], BoltType::Float64)), 0) << v[i];
    EXPECT_EQ(enc1(-0.0, BoltType::Float64), enc1(0.0, BoltType::Float64));
    const double qnan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(enc1(qnan, BoltType::Float64), enc1(-qnan, BoltType::Float64));
    EXPECT_GT(bytes_cmp(enc1(qnan, BoltType::Float64), enc1(inf, BoltType::Float64)), 0);
    EXPECT_LT(bytes_cmp(enc1(-1.5f, BoltType::Float32), enc1(0.25f, BoltType::Float32)), 0);
    EXPECT_EQ(enc1(-0.0f, BoltType::Float32), enc1(0.0f, BoltType::Float32));
}

TEST(KeyEncode, TextEscapesAndTerminates) {
    EXPECT_EQ(enc_text(""), (std::vector<uint8_t>{0, 0}));
    EXPECT_EQ(enc_text(std::string("a\0b", 3)), (std::vector<uint8_t>{'a', 0, 0xFF, 'b', 0, 0}));
    const std::vector<std::string> sorted = {"", std::string("\0", 1), std::string("\0\0", 2),
                                             std::string("\0\x01", 2), "\x01", "a",
                                             std::string("a\0", 2), std::string("a\0\0", 3),
                                             std::string("a\0b", 3), "ab", "abc", "b", "\xff"};
    for (size_t i = 0; i < sorted.size(); ++i)
        for (size_t j = 0; j < sorted.size(); ++j) {
            const int want = i < j ? -1 : (i > j ? 1 : 0);
            ASSERT_EQ(bytes_cmp(enc_text(sorted[i]), enc_text(sorted[j])), want) << i << " " << j;
        }
}

std::string rand_text(std::mt19937_64& rng) {
    std::string s(rng() % 20, 'a');
    static const char kAlpha[5] = {0, 1, 'a', 'b', static_cast<char>(0xFF)};
    for (char& c : s) c = kAlpha[rng() % 5];
    return s;
}

// (text, int64, text): composite memcmp order == tuple order.
TEST(KeyEncode, CompositeTextAnywhereOrdersAsTuples) {
    std::mt19937_64 rng(3);
    struct Tup { std::string a; int64_t b; std::string c; };
    std::vector<Tup> v;
    for (int i = 0; i < 600; ++i) v.push_back({rand_text(rng), static_cast<int64_t>(rng() % 5) - 2, rand_text(rng)});
    auto enc = [](const Tup& t) {
        std::vector<uint8_t> e = enc_text(t.a);
        const auto m = enc1(t.b, BoltType::Int64);
        e.insert(e.end(), m.begin(), m.end());
        const auto c = enc_text(t.c);
        e.insert(e.end(), c.begin(), c.end());
        return e;
    };
    for (size_t i = 0; i < v.size(); ++i)
        for (size_t j = 0; j < v.size(); j += 13) {
            const auto ta = std::tie(v[i].a, v[i].b, v[i].c);
            const auto tb = std::tie(v[j].a, v[j].b, v[j].c);
            const int want = ta < tb ? -1 : (tb < ta ? 1 : 0);
            ASSERT_EQ(bytes_cmp(enc(v[i]), enc(v[j])), want);
        }
}

TEST(KeyEncode, ColumnHelperMatchesRowsAndRefusesNull) {
    bolt::Arena ar;
    const size_t n = 50;
    std::vector<int64_t> ints(n);
    std::vector<StringView> sv(n);
    std::string over;
    std::vector<std::string> strs(n);
    for (size_t r = 0; r < n; ++r) {
        ints[r] = static_cast<int64_t>(r) * 1000 - 7;
        strs[r] = std::string(r % 30, static_cast<char>('a' + r % 3));
        if (r % 4 == 0 && !strs[r].empty()) strs[r][0] = '\0';
    }
    for (size_t r = 0; r < n; ++r) {
        std::memset(&sv[r], 0, sizeof(StringView));
        sv[r].length = static_cast<uint32_t>(strs[r].size());
        std::memcpy(sv[r].prefix, strs[r].data(), std::min<size_t>(strs[r].size(), 12));
        if (strs[r].size() > 12) { sv[r].ref.offset = static_cast<uint32_t>(over.size()); over += strs[r]; }
    }
    BoltColumn cols[2];
    cols[0] = BoltColumn::make_flat(sv.data(), nullptr, static_cast<int64_t>(n), BoltType::Utf8);
    cols[0].type_size_bytes = 16;
    cols[0].str_overflow_base = over.data();
    cols[1] = BoltColumn::make_flat(ints.data(), nullptr, static_cast<int64_t>(n), BoltType::Int64);
    const uint32_t kc[2] = {0, 1};
    BoltColumn out;
    ASSERT_EQ(bolt::key_encode_column(cols, kc, 2, static_cast<int64_t>(n), &ar, &out), KeyEncodeStatus::kOk);
    const auto* ov = static_cast<const StringView*>(out.data);
    for (size_t r = 0; r < n; ++r) {
        std::vector<uint8_t> want = enc_text(strs[r]);
        const auto m = enc1(ints[r], BoltType::Int64);
        want.insert(want.end(), m.begin(), m.end());
        const uint8_t* p = ov[r].length <= 12 ? reinterpret_cast<const uint8_t*>(ov[r].prefix)
                                              : static_cast<const uint8_t*>(out.str_overflow_base) + ov[r].ref.offset;
        ASSERT_EQ(std::vector<uint8_t>(p, p + ov[r].length), want) << r;
    }
    uint8_t valid[8] = {0xFF, 0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    cols[1].validity = valid;
    EXPECT_EQ(bolt::key_encode_column(cols, kc, 2, static_cast<int64_t>(n), &ar, &out), KeyEncodeStatus::kNull);
    BoltColumn list = BoltColumn::make_flat(ints.data(), nullptr, static_cast<int64_t>(n), BoltType::Float16);
    EXPECT_EQ(bolt::key_encode_column(&list, kc, 1, static_cast<int64_t>(n), &ar, &out), KeyEncodeStatus::kUnsupported);
}

TEST(KeyEncode, ColumnHelperRefusesOverflowPastItsCap) {
    bolt::Arena ar;
    const size_t n = 40;
    std::vector<int64_t> a(n), b(n);
    for (size_t r = 0; r < n; ++r) { a[r] = static_cast<int64_t>(r); b[r] = -static_cast<int64_t>(r); }
    BoltColumn cols[2];
    cols[0] = BoltColumn::make_flat(a.data(), nullptr, static_cast<int64_t>(n), BoltType::Int64);
    cols[1] = BoltColumn::make_flat(b.data(), nullptr, static_cast<int64_t>(n), BoltType::Int64);
    const uint32_t kc[2] = {0, 1};
    BoltColumn out;
    const uint64_t total = n * 16;   // 16-byte keys all spill to overflow
    static_assert(bolt::kKeyEncodeMaxOverflowBytes <= UINT32_MAX);
    EXPECT_EQ(bolt::key_encode_column(cols, kc, 2, static_cast<int64_t>(n), &ar, &out, total), KeyEncodeStatus::kOk);
    EXPECT_EQ(bolt::key_encode_column(cols, kc, 2, static_cast<int64_t>(n), &ar, &out, total - 1), KeyEncodeStatus::kNoRoom);
    EXPECT_EQ(bolt::key_encode_column(cols, kc, 1, static_cast<int64_t>(n), &ar, &out, 0), KeyEncodeStatus::kOk);
}

}  // namespace
