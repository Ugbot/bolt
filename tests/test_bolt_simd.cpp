// test_bolt_simd.cpp — direct unit tests for bolt::simd wrappers.
//
// Scalar reference is computed in plain C++, then each wrapper is invoked
// and byte-compared via a memory store. Works on every ISA path because the
// reference doesn't touch intrinsics.

#include <gtest/gtest.h>

#include "bolt/bolt_port.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>

using namespace bolt::simd;

namespace {

constexpr int LI32 = bmm_lanes_i32;
constexpr int LI64 = bmm_lanes_i64;
constexpr int LF32 = bmm_lanes_f32;
constexpr int LF64 = bmm_lanes_f64;
constexpr int LI8  = bmm_lanes_i8;

// -------- helpers: store a vector to a local buffer ------------------------
struct I32Buf { alignas(64) int32_t v[LI32]; };
struct I64Buf { alignas(64) int64_t v[LI64]; };
struct F32Buf { alignas(64) float   v[LF32]; };
struct F64Buf { alignas(64) double  v[LF64]; };
struct I8Buf  { alignas(64) int8_t  v[LI8];  };

I32Buf dump_i32(bmm_vec_i32 x) noexcept { I32Buf b{}; bmm_storeu_i32(b.v, x); return b; }
I64Buf dump_i64(bmm_vec_i64 x) noexcept {
    // No bmm_store_i64 wrapper — reinterpret through i32 store (works because
    // same backing type for all ISAs).
    I64Buf b{};
#if BOLT_SIMD_AVX2
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(b.v), x);
#elif BOLT_SIMD_SSE42
    _mm_storeu_si128(reinterpret_cast<__m128i*>(b.v), x);
#elif BOLT_SIMD_NEON
    vst1q_s64(b.v, x);
#else
    for (int i = 0; i < LI64; ++i) b.v[i] = x.v[i];
#endif
    return b;
}
F64Buf dump_f64(bmm_vec_f64 x) noexcept {
    F64Buf b{};
#if BOLT_SIMD_AVX2
    _mm256_storeu_pd(b.v, x);
#elif BOLT_SIMD_SSE42
    _mm_storeu_pd(b.v, x);
#elif BOLT_SIMD_NEON
    vst1q_f64(b.v, x);
#else
    for (int i = 0; i < LF64; ++i) b.v[i] = x.v[i];
#endif
    return b;
}

// make a vector from a pointer
bmm_vec_i32 make_i32(const int32_t* p) noexcept { return bmm_loadu_i32(p); }
bmm_vec_i64 make_i64(const int64_t* p) noexcept {
#if BOLT_SIMD_AVX2
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
#elif BOLT_SIMD_SSE42
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
#elif BOLT_SIMD_NEON
    return vld1q_s64(p);
#else
    bmm_vec_i64 r; for (int i = 0; i < LI64; ++i) r.v[i] = p[i]; return r;
#endif
}
bmm_vec_f64 make_f64(const double* p) noexcept {
#if BOLT_SIMD_AVX2
    return _mm256_loadu_pd(p);
#elif BOLT_SIMD_SSE42
    return _mm_loadu_pd(p);
#elif BOLT_SIMD_NEON
    return vld1q_f64(p);
#else
    bmm_vec_f64 r; for (int i = 0; i < LF64; ++i) r.v[i] = p[i]; return r;
#endif
}

} // namespace

// ===========================================================================
// Load / store round-trips
// ===========================================================================
TEST(BoltSimd, LoadStoreI32RoundTrip) {
    alignas(64) int32_t in[LI32]; alignas(64) int32_t out[LI32];
    for (int i = 0; i < LI32; ++i) in[i] = i * 7 - 13;
    bmm_storeu_i32(out, bmm_loadu_i32(in));
    EXPECT_EQ(0, std::memcmp(in, out, sizeof(in)));
}

TEST(BoltSimd, LoadStoreF32RoundTrip) {
    alignas(64) float in[LF32]; alignas(64) float out[LF32];
    for (int i = 0; i < LF32; ++i) in[i] = static_cast<float>(i) * 1.25f - 3.5f;
    bmm_storeu_f32(out, bmm_loadu_f32(in));
    EXPECT_EQ(0, std::memcmp(in, out, sizeof(in)));
}

TEST(BoltSimd, LoadStoreF64RoundTrip) {
    alignas(64) double in[LF64]; alignas(64) double out[LF64];
    for (int i = 0; i < LF64; ++i) in[i] = static_cast<double>(i) * 1.25 - 3.5;
    auto v = make_f64(in);
    auto dump = dump_f64(v);
    EXPECT_EQ(0, std::memcmp(in, dump.v, sizeof(in)));
    (void)out;
}

TEST(BoltSimd, LoadStoreI64RoundTrip) {
    alignas(64) int64_t in[LI64];
    for (int i = 0; i < LI64; ++i) in[i] = (int64_t)i * 1000000007LL - 42;
    auto dump = dump_i64(make_i64(in));
    EXPECT_EQ(0, std::memcmp(in, dump.v, sizeof(in)));
}

// ===========================================================================
// set1 fills all lanes
// ===========================================================================
TEST(BoltSimd, Set1I32) {
    auto d = dump_i32(bmm_set1_i32(0x12345678));
    for (int i = 0; i < LI32; ++i) EXPECT_EQ(d.v[i], 0x12345678);
}
TEST(BoltSimd, Set1F32) {
    alignas(64) float out[LF32];
    bmm_storeu_f32(out, bmm_set1_f32(3.25f));
    for (int i = 0; i < LF32; ++i) EXPECT_EQ(out[i], 3.25f);
}

// ===========================================================================
// Compares
// ===========================================================================
TEST(BoltSimd, CmpGtI32Boundary) {
    alignas(64) int32_t in[LI32];
    for (int i = 0; i < LI32; ++i) in[i] = i - (LI32 / 2);   // e.g. -4..3
    auto a = bmm_loadu_i32(in);
    auto b = bmm_set1_i32(0);
    auto r = dump_i32(bmm_cmpgt_i32(a, b));
    for (int i = 0; i < LI32; ++i) {
        int32_t exp = (in[i] > 0) ? -1 : 0;
        EXPECT_EQ(r.v[i], exp);
    }
}

TEST(BoltSimd, CmpGtI32Extremes) {
    alignas(64) int32_t in[LI32];
    for (int i = 0; i < LI32; ++i) in[i] = (i & 1) ? INT32_MAX : INT32_MIN;
    auto a = bmm_loadu_i32(in);
    auto b = bmm_set1_i32(0);
    auto r = dump_i32(bmm_cmpgt_i32(a, b));
    for (int i = 0; i < LI32; ++i) {
        int32_t exp = (in[i] > 0) ? -1 : 0;
        EXPECT_EQ(r.v[i], exp);
    }
}

TEST(BoltSimd, CmpEqI32) {
    alignas(64) int32_t in[LI32];
    for (int i = 0; i < LI32; ++i) in[i] = (i % 2 == 0) ? 42 : 43;
    auto r = dump_i32(bmm_cmpeq_i32(bmm_loadu_i32(in), bmm_set1_i32(42)));
    for (int i = 0; i < LI32; ++i) EXPECT_EQ(r.v[i], (in[i] == 42) ? -1 : 0);
}

// ===========================================================================
// Movemask
// ===========================================================================
TEST(BoltSimd, MovemaskI32Known) {
    alignas(64) int32_t in[LI32];
    for (int i = 0; i < LI32; ++i) in[i] = (i & 1) ? -1 : 0;
    int m = bmm_movemask_i32(bmm_loadu_i32(in));
    int expected = 0;
    for (int i = 0; i < LI32; ++i) if (in[i] < 0) expected |= (1 << i);
    EXPECT_EQ(m, expected);
}

TEST(BoltSimd, MovemaskF32Known) {
    alignas(64) float in[LF32];
    for (int i = 0; i < LF32; ++i) in[i] = (i % 3 == 0) ? -1.0f : 1.0f;
    int m = bmm_movemask_f32(bmm_loadu_f32(in));
    int expected = 0;
    for (int i = 0; i < LF32; ++i) if (in[i] < 0) expected |= (1 << i);
    EXPECT_EQ(m, expected);
}

// ===========================================================================
// Logical and/or
// ===========================================================================
TEST(BoltSimd, AndOrBits) {
    alignas(64) int32_t a[LI32], b[LI32];
    for (int i = 0; i < LI32; ++i) { a[i] = 0x0F0F0F0F; b[i] = 0x33333333; }
    auto ra = dump_i32(bmm_and(bmm_loadu_i32(a), bmm_loadu_i32(b)));
    auto ro = dump_i32(bmm_or (bmm_loadu_i32(a), bmm_loadu_i32(b)));
    for (int i = 0; i < LI32; ++i) {
        EXPECT_EQ(ra.v[i], 0x03030303);
        EXPECT_EQ(ro.v[i], 0x3F3F3F3F);
    }
}

// ===========================================================================
// Compressstore
// ===========================================================================
TEST(BoltSimd, CompressstoreAlternating) {
    alignas(64) int32_t in[LI32];
    for (int i = 0; i < LI32; ++i) in[i] = i * 10;
    uint32_t mask = 0;
    for (int i = 0; i < LI32; ++i) if (i & 1) mask |= (1u << i);
    alignas(64) int32_t out[LI32] = {};
    uint32_t n = bmm_compressstore_i32(out, bmm_loadu_i32(in), mask);
    EXPECT_EQ(n, (uint32_t)(LI32 / 2));
    int pos = 0;
    for (int i = 0; i < LI32; ++i) if (mask & (1u << i)) {
        EXPECT_EQ(out[pos++], in[i]);
    }
}

TEST(BoltSimd, CompressstoreAllSet) {
    alignas(64) int32_t in[LI32];
    for (int i = 0; i < LI32; ++i) in[i] = 100 + i;
    uint32_t mask = (LI32 == 8) ? 0xFFu : 0xFu;
    alignas(64) int32_t out[LI32] = {};
    uint32_t n = bmm_compressstore_i32(out, bmm_loadu_i32(in), mask);
    EXPECT_EQ(n, (uint32_t)LI32);
    for (int i = 0; i < LI32; ++i) EXPECT_EQ(out[i], in[i]);
}

TEST(BoltSimd, CompressstoreNoneSet) {
    alignas(64) int32_t in[LI32] = {};
    for (int i = 0; i < LI32; ++i) in[i] = i;
    alignas(64) int32_t out[LI32] = {};
    uint32_t n = bmm_compressstore_i32(out, bmm_loadu_i32(in), 0);
    EXPECT_EQ(n, 0u);
}

// ===========================================================================
// Gather
// ===========================================================================
TEST(BoltSimd, GatherI32) {
    alignas(64) int32_t base[16];
    for (int i = 0; i < 16; ++i) base[i] = i * 11 + 1;
    alignas(64) int32_t idx[LI32];
    for (int i = 0; i < LI32; ++i) idx[i] = (LI32 - 1 - i) % 16;
    auto r = dump_i32(bmm_gather_i32(base, bmm_loadu_i32(idx)));
    for (int i = 0; i < LI32; ++i) EXPECT_EQ(r.v[i], base[idx[i]]);
}

TEST(BoltSimd, GatherI64) {
    alignas(64) int64_t base[16];
    for (int i = 0; i < 16; ++i) base[i] = (int64_t)i * 100007LL;
    alignas(64) int32_t idx[LI32];
    for (int i = 0; i < LI32; ++i) idx[i] = (i * 3) % 16;
    auto r = dump_i64(bmm_gather_i64(base, bmm_loadu_i32(idx)));
    for (int i = 0; i < LI64; ++i) EXPECT_EQ(r.v[i], base[idx[i]]);
}

// ===========================================================================
// Horizontal add
// ===========================================================================
TEST(BoltSimd, HaddI64Series) {
    alignas(64) int64_t in[LI64];
    int64_t expect = 0;
    for (int i = 0; i < LI64; ++i) { in[i] = i + 1; expect += in[i]; }
    EXPECT_EQ(bmm_hadd_i64(make_i64(in)), expect);
}

TEST(BoltSimd, HaddF64Series) {
    alignas(64) double in[LF64];
    double expect = 0.0;
    for (int i = 0; i < LF64; ++i) { in[i] = static_cast<double>(i) + 0.5; expect += in[i]; }
    double got = bmm_hadd_f64(make_f64(in));
    EXPECT_DOUBLE_EQ(got, expect);
}

// ===========================================================================
// Blend
// ===========================================================================
TEST(BoltSimd, BlendI32ByCond) {
    alignas(64) int32_t a[LI32], b[LI32], cond[LI32];
    for (int i = 0; i < LI32; ++i) {
        a[i] = 1000 + i;
        b[i] = 2000 + i;
        cond[i] = (i & 1) ? -1 : 0;
    }
    auto r = dump_i32(bmm_blend_i32(bmm_loadu_i32(a), bmm_loadu_i32(b), bmm_loadu_i32(cond)));
    for (int i = 0; i < LI32; ++i) EXPECT_EQ(r.v[i], (cond[i] ? b[i] : a[i]));
}

// ===========================================================================
// Shuffle
// ===========================================================================
TEST(BoltSimd, ShuffleI32Reverse) {
    alignas(64) int32_t in[LI32];
    for (int i = 0; i < LI32; ++i) in[i] = i + 100;
    alignas(64) int32_t idx[LI32];
    // On AVX2 path, 8-lane reverse; on 4-lane paths, reverse within a single
    // 128-bit vector (which is what our SSE/NEON/Scalar wrappers support).
    for (int i = 0; i < LI32; ++i) idx[i] = (LI32 - 1) - i;
    auto r = dump_i32(bmm_shuffle_i32(bmm_loadu_i32(in), bmm_loadu_i32(idx)));
    for (int i = 0; i < LI32; ++i) EXPECT_EQ(r.v[i], in[idx[i]]);
}

// ===========================================================================
// i8 / SwissTable-style
// ===========================================================================
TEST(BoltSimd, CmpEqI8AndMovemask) {
    alignas(64) int8_t in[LI8];
    for (int i = 0; i < LI8; ++i) in[i] = (int8_t)((i % 4 == 0) ? 0x55 : 0x00);
    auto v = bmm_loadu_i8(in);
    auto eq = bmm_cmpeq_i8(v, bmm_set1_i8((int8_t)0x55));
    int m = bmm_movemask_i8(eq);
    int expected = 0;
    for (int i = 0; i < LI8; ++i) if (in[i] == (int8_t)0x55) expected |= (1 << i);
    EXPECT_EQ(m, expected);
}

// ===========================================================================
// Popcount helper round-trip
// ===========================================================================
TEST(BoltSimd, PopcountMask) {
    EXPECT_EQ(bmm_popcount_mask(0u), 0u);
    EXPECT_EQ(bmm_popcount_mask(0xFFu), 8u);
    EXPECT_EQ(bmm_popcount_mask(0xA5A5u), 8u);
    EXPECT_EQ(bmm_popcount_mask(0xFFFFFFFFu), 32u);
}
