// test_bolt_wave2.cpp — Wave 2 coverage: Track H (SYMBOL upgrades —
// literal-resolve-once, DictionaryPool, BitmapIndex popcount
// miss-accelerator) and Track I (run-native kernels for BitPacked /
// FrameOfRef column formats).

#include <gtest/gtest.h>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_branchless.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_dictionary.h"
#include "bolt/kernels/bolt_dict_filter.h"

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

using namespace bolt;

// ============================================================================
// H1 — literal-resolve-once: dict_resolve_code + filter_eq_dict_keys
// ============================================================================

TEST(DictFilter, ResolveCodePresent) {
    const int32_t dict[4] = {100, 200, 300, 400};
    EXPECT_EQ(kernels::dict_resolve_code<int32_t>(dict, 4, 300), 2);
    EXPECT_EQ(kernels::dict_resolve_code<int32_t>(dict, 4, 100), 0);
    EXPECT_EQ(kernels::dict_resolve_code<int32_t>(dict, 4, 400), 3);
}

TEST(DictFilter, ResolveCodeMissing) {
    const int32_t dict[3] = {10, 20, 30};
    EXPECT_EQ(kernels::dict_resolve_code<int32_t>(dict, 3, 999), -1);
    EXPECT_EQ(kernels::dict_resolve_code<int32_t>(nullptr, 0, 5), -1);
}

TEST(DictFilter, FilterEqDictKeysMatchesMaterialise) {
    // Keys for a dict of size 4, 64 rows, pattern [0,1,2,3] repeating.
    constexpr int64_t N = 64;
    uint8_t keys[N];
    int32_t values[4] = {100, 200, 300, 400};
    for (int i = 0; i < N; ++i) keys[i] = static_cast<uint8_t>(i & 3);

    int32_t out[N] = {};
    int64_t n = kernels::filter_eq_dict_keys<uint8_t>(keys, N, /*code=*/2, out);
    ASSERT_EQ(n, 16);
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_EQ(keys[out[i]], 2u) << "i=" << i;
        EXPECT_EQ(values[keys[out[i]]], 300);
    }
}

TEST(DictFilter, FilterEqDictKeysNoMatch) {
    uint8_t keys[16] = {1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4};
    int32_t out[16] = {};
    int64_t n = kernels::filter_eq_dict_keys<uint8_t>(keys, 16, /*code=*/7, out);
    EXPECT_EQ(n, 0);
}

// End-to-end: resolve literal against a dict_child, then filter keys.
// Shape a caller sees when it owns a BoltColumn::Dictionary directly.
TEST(DictFilter, EndToEndLiteralResolveOnce) {
    Arena a;
    auto dict = BoltColumn::make_flat_alloc(3, BoltType::Int32, &a);
    auto* dv = dict.typed_mutable<int32_t>();
    dv[0] = 111; dv[1] = 222; dv[2] = 333;

    constexpr int64_t N = 12;
    uint8_t keys[N] = {0,1,2, 0,1,2, 0,1,2, 0,1,2};  // pattern repeats 4×

    // Caller: literal=222 → code=1 via one linear scan, then tight loop.
    int32_t code = kernels::dict_resolve_code<int32_t>(
        dv, dict.length, /*scalar=*/222);
    ASSERT_EQ(code, 1);
    int32_t out[N] = {};
    int64_t n = kernels::filter_eq_dict_keys<uint8_t>(
        keys, N, static_cast<uint32_t>(code), out);
    ASSERT_EQ(n, 4);
    EXPECT_EQ(out[0], 1);
    EXPECT_EQ(out[1], 4);
    EXPECT_EQ(out[2], 7);
    EXPECT_EQ(out[3], 10);
}

// ============================================================================
// H2 — DictionaryPool: intern + resolve + cross-morsel stable codes
// ============================================================================

TEST(DictionaryPool, InternAndResolve) {
    Arena a;
    DictionaryPool pool{};
    ASSERT_TRUE(DictionaryPool::create(&pool, /*max_codes=*/128,
                                        /*avg_bytes_per_str=*/8, &a));

    const uint32_t c1 = pool.intern("AAPL", 4);
    const uint32_t c2 = pool.intern("GOOG", 4);
    const uint32_t c3 = pool.intern("AAPL", 4);  // already interned
    EXPECT_NE(c1, kDictPoolInvalidCode);
    EXPECT_NE(c2, kDictPoolInvalidCode);
    EXPECT_EQ(c1, c3) << "intern must be idempotent";
    EXPECT_NE(c1, c2) << "distinct strings get distinct codes";

    const char* data;
    uint32_t len;
    pool.resolve(c1, &data, &len);
    ASSERT_EQ(len, 4u);
    EXPECT_EQ(memcmp(data, "AAPL", 4), 0);
    pool.resolve(c2, &data, &len);
    ASSERT_EQ(len, 4u);
    EXPECT_EQ(memcmp(data, "GOOG", 4), 0);
}

TEST(DictionaryPool, CodesStableAcrossBatches) {
    Arena a;
    DictionaryPool pool{};
    ASSERT_TRUE(DictionaryPool::create(&pool, 64, 8, &a));

    // Batch 1 interns the literal "NYSE". Batch 2 re-interns it —
    // code must match.
    uint32_t code_batch1 = pool.intern("NYSE", 4);
    for (int i = 0; i < 100; ++i) {
        char buf[16];
        int n = snprintf(buf, sizeof(buf), "SYM%03d", i);
        pool.intern(buf, static_cast<uint32_t>(n));
    }
    uint32_t code_batch2 = pool.find("NYSE", 4);
    EXPECT_EQ(code_batch1, code_batch2);
}

TEST(DictionaryPool, FindMissingReturnsInvalid) {
    Arena a;
    DictionaryPool pool{};
    ASSERT_TRUE(DictionaryPool::create(&pool, 64, 8, &a));
    pool.intern("AAA", 3);
    EXPECT_EQ(pool.find("ZZZ", 3), kDictPoolInvalidCode);
}

TEST(DictionaryPool, OverflowReturnsInvalid) {
    Arena a;
    DictionaryPool pool{};
    ASSERT_TRUE(DictionaryPool::create(&pool, /*max_codes=*/4, 4, &a));
    EXPECT_NE(pool.intern("AAA", 3), kDictPoolInvalidCode);
    EXPECT_NE(pool.intern("BBB", 3), kDictPoolInvalidCode);
    EXPECT_NE(pool.intern("CCC", 3), kDictPoolInvalidCode);
    EXPECT_NE(pool.intern("DDD", 3), kDictPoolInvalidCode);
    // 5th distinct entry hits max_codes cap.
    EXPECT_EQ(pool.intern("EEE", 3), kDictPoolInvalidCode);
    // Existing entries still resolve.
    EXPECT_NE(pool.find("CCC", 3), kDictPoolInvalidCode);
}

// ============================================================================
// H3 — BitmapIndex popcount miss-accelerator (probably_absent)
// ============================================================================

TEST(BitmapIndexPopcount, ProbablyAbsentForUnusedKeys) {
    Arena a;
    // Dict of 5 keys, but rows only use keys 0, 2, 4 — keys 1 and 3
    // must be flagged as absent.
    auto dict = BoltColumn::make_flat_alloc(5, BoltType::Int32, &a);
    auto* dv = dict.typed_mutable<int32_t>();
    for (int i = 0; i < 5; ++i) dv[i] = 100 + i;

    BoltColumn col = BoltColumn::make_empty();
    col.format = ColumnFormat::Dictionary;
    col.type = BoltType::Int32;
    col.type_size_bytes = 1;
    col.length = 9;
    uint8_t* keys = a.allocate_array<uint8_t>(9);
    const uint8_t used[9] = {0,2,4, 0,2,4, 0,2,4};
    memcpy(keys, used, 9);
    col.data = keys;
    BoltColumn* child = a.allocate_array<BoltColumn>(1);
    *child = dict;
    col.dict_child = child;
    col.arena = &a;

    BitmapIndex* idx = BitmapIndex::build(col, &a);
    ASSERT_NE(idx, nullptr);
    EXPECT_EQ(idx->num_keys, 5u);

    // Present keys (0, 2, 4) → NOT absent, count > 0.
    EXPECT_FALSE(idx->probably_absent(0));
    EXPECT_FALSE(idx->probably_absent(2));
    EXPECT_FALSE(idx->probably_absent(4));
    EXPECT_EQ(idx->count(0), 3u);
    EXPECT_EQ(idx->count(2), 3u);
    EXPECT_EQ(idx->count(4), 3u);

    // Unused keys (1, 3) → absent; out-of-range (5+) → absent too.
    EXPECT_TRUE(idx->probably_absent(1));
    EXPECT_TRUE(idx->probably_absent(3));
    EXPECT_TRUE(idx->probably_absent(999));

    // count short-circuits via popcount sidecar → 0 for unused keys.
    EXPECT_EQ(idx->count(1), 0u);
    EXPECT_EQ(idx->count(3), 0u);

    // filter() on absent key returns 0 without scanning.
    int32_t out[9] = {};
    EXPECT_EQ(idx->filter(/*key=*/1, out), 0);
    EXPECT_EQ(idx->filter(/*key=*/3, out), 0);

    // filter() on present key still returns the right rows.
    int64_t nr = idx->filter(/*key=*/2, out);
    ASSERT_EQ(nr, 3);
    EXPECT_EQ(out[0], 1);
    EXPECT_EQ(out[1], 4);
    EXPECT_EQ(out[2], 7);
}

// ============================================================================
// I1 — filter_gt_bitpacked / filter_eq_bitpacked
// ============================================================================

TEST(BitPackedKernels, FilterGtMatchesUnpackThenFilter) {
    // 20 × 5-bit values = 100 bits → 2 words.
    constexpr int64_t N = 20;
    uint32_t vals[N];
    std::mt19937 rng{0xBEEFu};
    std::uniform_int_distribution<uint32_t> dist{0, 31};
    for (auto& v : vals) v = dist(rng);

    uint64_t packed[3] = {0, 0, 0};
    for (int i = 0; i < N; ++i) {
        uint64_t bit_off = uint64_t(i) * 5;
        uint64_t bit_in  = bit_off & 63u;
        packed[bit_off >> 6] |= uint64_t(vals[i]) << bit_in;
        if (bit_in + 5 > 64) packed[(bit_off >> 6) + 1] |= uint64_t(vals[i]) >> (64 - bit_in);
    }

    const int32_t scalar = 15;
    // Reference: unpack then filter.
    int64_t ref_count = 0;
    for (int i = 0; i < N; ++i) if (int32_t(vals[i]) > scalar) ++ref_count;

    int32_t got[N] = {};
    int64_t n = branchless::filter_gt_bitpacked(packed, N, /*bw=*/5, scalar, got);
    ASSERT_EQ(n, ref_count);
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_GT(int32_t(vals[got[i]]), scalar) << "row " << got[i];
    }
}

TEST(BitPackedKernels, FilterEqMatchesExpected) {
    // Hand-constructed 12 × 3-bit values {0..7, 0,1,2,3}.
    const uint32_t vals[12] = {0,1,2,3,4,5,6,7, 0,1,2,3};
    uint64_t packed[1] = {0};
    for (int i = 0; i < 12; ++i) packed[0] |= uint64_t(vals[i]) << (i * 3);

    int32_t got[12] = {};
    int64_t n = branchless::filter_eq_bitpacked(packed, 12, /*bw=*/3, /*scalar=*/3, got);
    ASSERT_EQ(n, 2);
    EXPECT_EQ(got[0], 3);
    EXPECT_EQ(got[1], 11);
}

// ============================================================================
// I2 — sum_frame_of_ref
// ============================================================================

TEST(FrameOfRefKernel, SumMatchesMaterialisedSum) {
    // Simulate sparse-rise timestamps: base = 1_000_000, 16 × 4-bit deltas.
    constexpr int64_t N = 16;
    const int64_t base = 1'000'000;
    const uint32_t deltas[N] = {0,1,2,3,4,5,6,7, 8,9,10,11,12,13,14,15};
    uint64_t packed[2] = {0, 0};
    for (int i = 0; i < N; ++i) {
        uint64_t bit_off = uint64_t(i) * 4;
        packed[bit_off >> 6] |= uint64_t(deltas[i]) << (bit_off & 63u);
    }

    int64_t ref = 0;
    for (int i = 0; i < N; ++i) ref += base + int64_t(deltas[i]);
    int64_t got = branchless::sum_frame_of_ref(packed, N, /*bw=*/4, base);
    EXPECT_EQ(got, ref);
}

TEST(FrameOfRefKernel, ZeroLengthIsBase) {
    // Empty input: base * 0 + 0 = 0. Even though base is non-zero.
    EXPECT_EQ(branchless::sum_frame_of_ref(nullptr, 0, /*bw=*/8, 42), 0);
}
