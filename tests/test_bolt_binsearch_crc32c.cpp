// test_bolt_binsearch_crc32c.cpp — GTest suite for new primitives.
//
// Covers:
//   - bolt::kernels::binsearch / lower_bound / upper_bound (i64 / u64 / f64)
//     across empty, singleton, midpoint / boundary hits, duplicates, and a
//     1k-iteration random fuzz against std::lower_bound / std::upper_bound.
//   - bolt::io::crc32c / crc32c_update against the Castagnoli reference
//     vectors, a chain-invariant check, and a 1 MB random buffer round-trip
//     between the hardware path (on this host) and the software oracle.
//
// Zero dependency beyond GTest — matches the rest of the tests/ shape.

#include <gtest/gtest.h>

#include "bolt/kernels/bolt_binsearch.h"
#include "bolt/io/bolt_crc32c.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

using bolt::kernels::binsearch_i64;
using bolt::kernels::binsearch_u64;
using bolt::kernels::binsearch_f64;
using bolt::kernels::lower_bound_i64;
using bolt::kernels::lower_bound_u64;
using bolt::kernels::lower_bound_f64;
using bolt::kernels::upper_bound_i64;
using bolt::kernels::upper_bound_u64;
using bolt::kernels::upper_bound_f64;

// ============================================================================
// binsearch — empty / singleton
// ============================================================================

TEST(BinsearchEmpty, ReturnsNotFound) {
    // data pointer may be null when n == 0 (documented API).
    EXPECT_EQ(binsearch_i64(nullptr, 0, 42), -1);
    EXPECT_EQ(lower_bound_i64(nullptr, 0, 42), 0);
    EXPECT_EQ(upper_bound_i64(nullptr, 0, 42), 0);
}

TEST(BinsearchSingleton, HitMiss) {
    const int64_t a[1] = {7};
    EXPECT_EQ(binsearch_i64(a, 1, 7), 0);
    EXPECT_EQ(binsearch_i64(a, 1, 6), -1);
    EXPECT_EQ(binsearch_i64(a, 1, 8), -1);
    EXPECT_EQ(lower_bound_i64(a, 1, 7), 0);
    EXPECT_EQ(lower_bound_i64(a, 1, 8), 1);
    EXPECT_EQ(upper_bound_i64(a, 1, 7), 1);
    EXPECT_EQ(upper_bound_i64(a, 1, 6), 0);
}

// ============================================================================
// binsearch — midpoint / first / last
// ============================================================================

TEST(BinsearchPositions, FirstMidLast) {
    const int64_t a[7] = {1, 3, 5, 7, 9, 11, 13};
    EXPECT_EQ(binsearch_i64(a, 7, 1), 0);    // first
    EXPECT_EQ(binsearch_i64(a, 7, 7), 3);    // mid
    EXPECT_EQ(binsearch_i64(a, 7, 13), 6);   // last
    EXPECT_EQ(binsearch_i64(a, 7, 2), -1);
    EXPECT_EQ(binsearch_i64(a, 7, 14), -1);
}

// ============================================================================
// binsearch — duplicates: binsearch returns *any* match, lower/upper bracket.
// ============================================================================

TEST(BinsearchDuplicates, BracketIsInclusiveExclusive) {
    const int64_t a[] = {1, 2, 2, 2, 3, 4};
    const int64_t n   = 6;
    // lower_bound → index of first 2
    EXPECT_EQ(lower_bound_i64(a, n, 2), 1);
    // upper_bound → index past last 2
    EXPECT_EQ(upper_bound_i64(a, n, 2), 4);
    // binsearch → any index in [1, 4)
    const int64_t hit = binsearch_i64(a, n, 2);
    EXPECT_GE(hit, 1);
    EXPECT_LT(hit, 4);
    EXPECT_EQ(a[hit], 2);
}

// ============================================================================
// All-less-than-key → both bounds return n.
// ============================================================================

TEST(BinsearchAllLess, BothReturnN) {
    const int64_t a[] = {1, 2, 3};
    EXPECT_EQ(lower_bound_i64(a, 3, 100), 3);
    EXPECT_EQ(upper_bound_i64(a, 3, 100), 3);
    EXPECT_EQ(binsearch_i64(a, 3, 100), -1);
}

TEST(BinsearchAllGreater, BothReturnZero) {
    const int64_t a[] = {10, 20, 30};
    EXPECT_EQ(lower_bound_i64(a, 3, 5), 0);
    EXPECT_EQ(upper_bound_i64(a, 3, 5), 0);
    EXPECT_EQ(binsearch_i64(a, 3, 5), -1);
}

// ============================================================================
// Random fuzz: 1k sorted arrays, compare against std:: reference impls.
// ============================================================================

TEST(BinsearchFuzz, MatchesStdReference_i64) {
    std::mt19937_64 rng(0xC0FFEEu);
    std::uniform_int_distribution<int64_t> dval(-1000, 1000);
    std::uniform_int_distribution<int> dlen(0, 256);

    for (int trial = 0; trial < 1000; ++trial) {
        const int n = dlen(rng);
        std::vector<int64_t> v(n);
        for (int i = 0; i < n; ++i) v[i] = dval(rng);
        std::sort(v.begin(), v.end());

        // Probe both present and absent keys.
        const int64_t key = (n > 0 && (trial & 1)) ? v[rng() % n] : dval(rng);

        const int64_t got_lb = lower_bound_i64(v.data(), n, key);
        const int64_t exp_lb = static_cast<int64_t>(
            std::lower_bound(v.begin(), v.end(), key) - v.begin());
        EXPECT_EQ(got_lb, exp_lb) << "trial=" << trial << " n=" << n
                                  << " key=" << key;

        const int64_t got_ub = upper_bound_i64(v.data(), n, key);
        const int64_t exp_ub = static_cast<int64_t>(
            std::upper_bound(v.begin(), v.end(), key) - v.begin());
        EXPECT_EQ(got_ub, exp_ub) << "trial=" << trial << " n=" << n
                                  << " key=" << key;

        const int64_t got_bs = binsearch_i64(v.data(), n, key);
        const bool    exp_present = std::binary_search(v.begin(), v.end(), key);
        if (exp_present) {
            ASSERT_GE(got_bs, 0);
            ASSERT_LT(got_bs, static_cast<int64_t>(n));
            EXPECT_EQ(v[got_bs], key);
        } else {
            EXPECT_EQ(got_bs, -1);
        }
    }
}

TEST(BinsearchFuzz, MatchesStdReference_u64) {
    std::mt19937_64 rng(0xDEADBEEFu);
    std::uniform_int_distribution<uint64_t> dval(0u, 1000u);

    for (int trial = 0; trial < 200; ++trial) {
        const int n = static_cast<int>(rng() % 128u);
        std::vector<uint64_t> v(n);
        for (int i = 0; i < n; ++i) v[i] = dval(rng);
        std::sort(v.begin(), v.end());
        const uint64_t key = dval(rng);

        EXPECT_EQ(lower_bound_u64(v.data(), n, key),
                  static_cast<int64_t>(
                      std::lower_bound(v.begin(), v.end(), key) - v.begin()));
        EXPECT_EQ(upper_bound_u64(v.data(), n, key),
                  static_cast<int64_t>(
                      std::upper_bound(v.begin(), v.end(), key) - v.begin()));
    }
}

TEST(BinsearchFuzz, MatchesStdReference_f64) {
    std::mt19937_64 rng(0xFEEDFACEu);
    std::uniform_real_distribution<double> dval(-1000.0, 1000.0);

    for (int trial = 0; trial < 200; ++trial) {
        const int n = static_cast<int>(rng() % 128u);
        std::vector<double> v(n);
        for (int i = 0; i < n; ++i) v[i] = dval(rng);
        std::sort(v.begin(), v.end());
        const double key = dval(rng);

        EXPECT_EQ(lower_bound_f64(v.data(), n, key),
                  static_cast<int64_t>(
                      std::lower_bound(v.begin(), v.end(), key) - v.begin()));
        EXPECT_EQ(upper_bound_f64(v.data(), n, key),
                  static_cast<int64_t>(
                      std::upper_bound(v.begin(), v.end(), key) - v.begin()));
    }
}

// ============================================================================
// crc32c — reference vectors
// ============================================================================

TEST(Crc32cVectors, Standard) {
    // "123456789" → 0xE3069283 (Castagnoli reference vector, used by iSCSI,
    // Btrfs, Parquet, Google Crc32c library).
    const char s[] = "123456789";
    EXPECT_EQ(bolt::io::crc32c(s, 9), 0xE3069283u);

    // Empty buffer → 0 (seed = 0, no bytes consumed).
    EXPECT_EQ(bolt::io::crc32c("", 0), 0u);
    EXPECT_EQ(bolt::io::crc32c(nullptr, 0), 0u);

    // 32 zero bytes → 0x8A9136AA.
    const uint8_t zeros[32] = {};
    EXPECT_EQ(bolt::io::crc32c(zeros, 32), 0x8A9136AAu);
}

TEST(Crc32cVectors, SoftwarePathAlsoAgrees) {
    // Exercise the software path directly to cover platforms where the
    // hardware path would otherwise mask a bug in the slicing table.
    const char s[] = "123456789";
    EXPECT_EQ(bolt::io::crc32c_software(s, 9, 0), 0xE3069283u);
    const uint8_t zeros[32] = {};
    EXPECT_EQ(bolt::io::crc32c_software(zeros, 32, 0), 0x8A9136AAu);
    EXPECT_EQ(bolt::io::crc32c_software("", 0, 0), 0u);
}

// ============================================================================
// crc32c — chain invariant
// ============================================================================

TEST(Crc32cChain, ConcatEqualsUpdate) {
    // CRC of "Hello, world!" should match CRC of "Hello, " continued with
    // update(..., "world!").
    const char all[] = "Hello, world!";
    const char a[]   = "Hello, ";
    const char b[]   = "world!";
    const size_t la  = 7;
    const size_t lb  = 6;

    const uint32_t once  = bolt::io::crc32c(all, la + lb);
    const uint32_t step1 = bolt::io::crc32c(a, la);
    const uint32_t step2 = bolt::io::crc32c_update(step1, b, lb);
    EXPECT_EQ(once, step2);

    // Non-zero seed form (direct) matches update form.
    const uint32_t step2_direct = bolt::io::crc32c(b, lb, step1);
    EXPECT_EQ(step2, step2_direct);
}

TEST(Crc32cSeed, EmptyReturnsSeed) {
    // Empty input with non-zero seed should return seed unchanged — this
    // is the invariant that makes chained updates work.
    EXPECT_EQ(bolt::io::crc32c(nullptr, 0, 0x12345678u), 0x12345678u);
    EXPECT_EQ(bolt::io::crc32c_update(0xDEADBEEFu, nullptr, 0), 0xDEADBEEFu);
}

// ============================================================================
// crc32c — 1 MB random buffer: hardware path (if present) matches software.
// ============================================================================

TEST(Crc32cLarge, HardwareMatchesSoftware_1MB) {
    constexpr size_t kLen = 1u << 20;  // 1 MB
    std::vector<uint8_t> buf(kLen);
    std::mt19937 rng(0xA5A5A5A5u);
    for (size_t i = 0; i < kLen; ++i) buf[i] = static_cast<uint8_t>(rng());

    const uint32_t via_default = bolt::io::crc32c(buf.data(), kLen);
    const uint32_t via_soft    = bolt::io::crc32c_software(buf.data(), kLen, 0);
    EXPECT_EQ(via_default, via_soft)
        << "hardware and software paths disagree on 1 MB random buffer";

    // Split the buffer into three chunks at arbitrary offsets and verify
    // streaming update matches one-shot.
    const size_t s1 = 333333;
    const size_t s2 = 666666;
    uint32_t step = bolt::io::crc32c(buf.data(), s1);
    step = bolt::io::crc32c_update(step, buf.data() + s1, s2 - s1);
    step = bolt::io::crc32c_update(step, buf.data() + s2, kLen - s2);
    EXPECT_EQ(step, via_default);
}

// ============================================================================
// crc32c — unaligned tails (every length 0..16) sanity-check across paths.
// ============================================================================

TEST(Crc32cTails, EveryLenFromZeroTo16) {
    std::array<uint8_t, 16> buf{};
    for (size_t i = 0; i < buf.size(); ++i) buf[i] = static_cast<uint8_t>(i * 17 + 3);

    for (size_t len = 0; len <= buf.size(); ++len) {
        const uint32_t got = bolt::io::crc32c(buf.data(), len);
        const uint32_t exp = bolt::io::crc32c_software(buf.data(), len, 0);
        EXPECT_EQ(got, exp) << "len=" << len;
    }
}
