// bolt/lakehouse/iceberg_transform_values.cpp — value kernels for every Iceberg partition transform (G2ICE-292).
//
// Spec: "Partition Transforms", "Bucket Transform Details", "Truncate Transform Details", Appendix B. The Appendix B
// test vectors are asserted in tests/test_bolt_iceberg_transforms.cpp. Arithmetic that Java does in fixed-width
// signed ints (truncate) is done here in unsigned ints so the same wrap happens without undefined behaviour.

#include "bolt/lakehouse/iceberg/transform.h"

#include <cassert>
#include <cstdint>
#include <cstring>

namespace bolt {
namespace lakehouse {
namespace iceberg {

namespace {

constexpr int64_t kMicrosPerDay  = INT64_C(86400000000);
constexpr int64_t kMicrosPerHour = INT64_C(3600000000);

int64_t floor_div(int64_t a, int64_t b) noexcept {
    assert(b > 0);
    const int64_t q = a / b;
    return (a % b != 0 && a < 0) ? q - 1 : q;
}

// Proleptic Gregorian civil date of a day count (H. Hinnant, "chrono-Compatible Low-Level Date Algorithms").
void civil_from_days(int64_t z, int64_t* y, uint32_t* m) noexcept {
    assert(y != nullptr && m != nullptr);
    z += 719468;
    const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    const uint32_t doe = static_cast<uint32_t>(z - era * 146097);
    const uint32_t yoe = (doe - doe / 1460u + doe / 36524u - doe / 146096u) / 365u;
    const uint32_t doy = doe - (365u * yoe + yoe / 4u - yoe / 100u);
    const uint32_t mp = (5u * doy + 2u) / 153u;
    *m = mp < 10u ? mp + 3u : mp - 9u;
    *y = static_cast<int64_t>(yoe) + era * 400 + (*m <= 2u ? 1 : 0);
    assert(*m >= 1u && *m <= 12u);
}

int64_t days_of(int64_t v, TimeUnit u) noexcept {
    assert(u == TimeUnit::kDays || u == TimeUnit::kMicros || u == TimeUnit::kNanos);
    if (u == TimeUnit::kDays) return v;
    const int64_t us = u == TimeUnit::kNanos ? micros_from_nanos(v) : v;
    return floor_div(us, kMicrosPerDay);
}

int32_t to_i32(int64_t v) noexcept {
    // Year/month/day/hour of any representable source fit an int for every date Iceberg can store; past that the
    // value wraps exactly as Java's (int) cast would.
    return static_cast<int32_t>(static_cast<uint32_t>(static_cast<uint64_t>(v)));
}

}  // namespace

int32_t hash_long(int64_t v) noexcept {
    uint8_t buf[8];
    for (uint32_t i = 0; i < 8u; ++i) buf[i] = static_cast<uint8_t>((static_cast<uint64_t>(v) >> (i * 8u)) & 0xffu);
    const uint32_t h = murmur3_32(buf, 8u, 0u);
    assert(sizeof(buf) == 8u);
    return static_cast<int32_t>(h);
}

int32_t hash_bytes(const void* p, uint32_t len) noexcept {
    assert(p != nullptr || len == 0u);
    assert(len < (1u << 30));
    return static_cast<int32_t>(murmur3_32(p, len, 0u));
}

int32_t hash_double(double v) noexcept {
    uint64_t bits = 0;
    if (v != v) bits = UINT64_C(0x7ff8000000000000);            // every NaN pattern canonicalised
    else if (v == 0.0) bits = 0;                                 // -0.0 == 0.0
    else std::memcpy(&bits, &v, 8u);
    assert(v == v || bits == UINT64_C(0x7ff8000000000000));
    return hash_long(static_cast<int64_t>(bits));
}

int32_t hash_bool(bool v) noexcept {
    const int32_t h = hash_long(v ? 1 : 0);
    assert(v || h == hash_long(0));
    return h;
}

uint32_t decimal_min_be_bytes(int64_t hi, uint64_t lo, uint8_t out[16]) noexcept {
    assert(out != nullptr);
    uint8_t be[16];
    for (uint32_t i = 0; i < 8u; ++i) {
        be[i] = static_cast<uint8_t>((static_cast<uint64_t>(hi) >> (56u - 8u * i)) & 0xffu);
        be[8u + i] = static_cast<uint8_t>((lo >> (56u - 8u * i)) & 0xffu);
    }
    // Drop a leading byte while it is pure sign extension of the next one.
    uint32_t s = 0;
    while (s < 15u) {                                            // bounded: 15
        const bool next_neg = (be[s + 1u] & 0x80u) != 0u;
        if ((be[s] == 0x00u && !next_neg) || (be[s] == 0xffu && next_neg)) ++s;
        else break;
    }
    const uint32_t n = 16u - s;
    std::memcpy(out, be + s, n);
    assert(n >= 1u && n <= 16u);
    return n;
}

int32_t hash_decimal(int64_t hi, uint64_t lo) noexcept {
    uint8_t b[16];
    const uint32_t n = decimal_min_be_bytes(hi, lo, b);
    assert(n >= 1u);
    return hash_bytes(b, n);
}

int32_t bucket_of_hash(int32_t h, int32_t N) noexcept {
    assert(N > 0);
    if (N <= 0) return 0;
    const int32_t r = static_cast<int32_t>((static_cast<uint32_t>(h) & 0x7fffffffu) % static_cast<uint32_t>(N));
    assert(r >= 0 && r < N);
    return r;
}

int32_t bucket_i32(int32_t value, int32_t N) noexcept {
    assert(N > 0);
    return bucket_of_hash(hash_long(static_cast<int64_t>(value)), N);
}

int32_t bucket_bytes(const void* p, uint32_t len, int32_t N) noexcept {
    assert(p != nullptr || len == 0u);
    assert(N > 0);
    return bucket_of_hash(hash_bytes(p, len), N);
}

int32_t bucket_decimal(int64_t hi, uint64_t lo, int32_t N) noexcept {
    assert(N > 0);
    const int32_t b = bucket_of_hash(hash_decimal(hi, lo), N);
    assert(b >= 0);
    return b;
}

int64_t micros_from_nanos(int64_t ns) noexcept {
    const int64_t us = floor_div(ns, 1000);
    assert(us <= ns || ns < 0);
    return us;
}

int32_t truncate_i32(int32_t value, int32_t W) noexcept {
    assert(W > 0);
    if (W <= 0) return value;
    const int32_t r = static_cast<int32_t>((static_cast<int64_t>(value % W) + W) % W);
    assert(r >= 0 && r < W);
    return static_cast<int32_t>(static_cast<uint32_t>(value) - static_cast<uint32_t>(r));
}

void truncate_decimal(int64_t hi, uint64_t lo, int32_t W, int64_t* out_hi, uint64_t* out_lo) noexcept {
    assert(out_hi != nullptr && out_lo != nullptr);
    assert(W > 0);
    *out_hi = hi;
    *out_lo = lo;
    if (W <= 0) return;
    const bool neg = hi < 0;
    uint64_t mh = static_cast<uint64_t>(hi), ml = lo;
    if (neg) { ml = ~ml + 1u; mh = ~mh + (ml == 0u ? 1u : 0u); }    // |v|
    const uint64_t w = static_cast<uint64_t>(W);
    uint64_t r = mh % w;                                         // |v| mod W, 32 bits at a time: r < 2^31
    r = ((r << 32) | (ml >> 32)) % w;
    r = ((r << 32) | (ml & 0xffffffffu)) % w;
    const uint64_t rp = (neg && r != 0u) ? w - r : r;            // the positive remainder of v
    *out_lo = lo - rp;
    *out_hi = static_cast<int64_t>(static_cast<uint64_t>(hi) - (lo < rp ? 1u : 0u));
}

uint32_t truncate_utf8_len(const char* s, uint32_t len, int32_t L) noexcept {
    assert(s != nullptr || len == 0u);
    assert(L >= 0);
    uint32_t o = 0;
    for (int32_t cp = 0; cp < L && o < len; ++cp) {             // bounded by len
        const uint8_t c = static_cast<uint8_t>(s[o]);
        uint32_t w = c < 0x80u ? 1u : (c >> 5) == 0x6u ? 2u : (c >> 4) == 0xeu ? 3u : (c >> 3) == 0x1eu ? 4u : 1u;
        if (o + w > len) w = len - o;
        for (uint32_t k = 1; k < w; ++k)                         // a malformed continuation ends the sequence there
            if ((static_cast<uint8_t>(s[o + k]) & 0xc0u) != 0x80u) { w = k; break; }
        o += w;
    }
    assert(o <= len);
    return o;
}

uint32_t truncate_binary_len(uint32_t len, int32_t L) noexcept {
    assert(L >= 0);
    const uint32_t l = L < 0 ? 0u : static_cast<uint32_t>(L);
    return len < l ? len : l;
}

int32_t transform_year(int64_t v, TimeUnit u) noexcept {
    int64_t y = 0; uint32_t m = 0;
    civil_from_days(days_of(v, u), &y, &m);
    assert(m >= 1u);
    return to_i32(y - 1970);
}

int32_t transform_month(int64_t v, TimeUnit u) noexcept {
    int64_t y = 0; uint32_t m = 0;
    civil_from_days(days_of(v, u), &y, &m);
    assert(m >= 1u && m <= 12u);
    return to_i32((y - 1970) * 12 + static_cast<int64_t>(m) - 1);
}

int32_t transform_day(int64_t v, TimeUnit u) noexcept {
    const int64_t d = days_of(v, u);
    assert(u != TimeUnit::kDays || d == v);
    return to_i32(d);
}

int32_t transform_hour(int64_t v, TimeUnit u) noexcept {
    assert(u != TimeUnit::kDays);
    const int64_t us = u == TimeUnit::kNanos ? micros_from_nanos(v) : v;
    return to_i32(floor_div(us, kMicrosPerHour));
}

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
