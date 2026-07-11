// bolt/kernels/bolt_prefix_scan.h — blocked parallel inclusive prefix
// sum (plan E2 phase 2: single-partition cumulative windows).
//
// Covers:
//   * serial reference correctness (int64 + float64, vs a naive fold);
//   * parallel int64 EXACT equality with the serial fold (blocked
//     integer addition is associative);
//   * parallel float64: bit-identical to the fixed-block reference
//     association (determinism — the same blocked order regardless of
//     pool size), near-equal to the serial fold within accumulated-
//     rounding tolerance, and EXACT for integer-valued doubles under
//     2^53 (the widened-Int64 window-operator case);
//   * engagement gating: below/at the one-block floor and with a null
//     scheduler the kernel reports the serial path (return false) and
//     still produces the correct fold;
//   * in-place scan (in == out);
//   * boundary shapes: exactly 2 blocks, block-misaligned tail.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_scheduler.h"
#include "bolt/kernels/bolt_prefix_scan.h"

namespace ps = bolt::kernels::prefix_scan;

namespace {

// Fixed-block reference association: what the 3-phase kernel computes,
// restated serially. Independent of any scheduler.
template <typename T>
void blocked_reference(const std::vector<T>& in, std::vector<T>& out) {
    const std::int64_t n = static_cast<std::int64_t>(in.size());
    out.resize(in.size());
    const std::int64_t B = ps::k_scan_block_rows;
    T carry = T{0};
    for (std::int64_t b0 = 0; b0 < n; b0 += B) {
        const std::int64_t b1 = (b0 + B < n) ? (b0 + B) : n;
        T acc = T{0};
        for (std::int64_t i = b0; i < b1; ++i) {
            acc += in[i];
            out[static_cast<std::size_t>(i)] = carry + acc;
        }
        carry += acc;
    }
}

}  // namespace

TEST(BoltPrefixScan, SerialReferenceInt64) {
    std::vector<std::int64_t> in{3, -1, 4, 1, -5, 9, 2, -6};
    std::vector<std::int64_t> out(in.size());
    ps::inclusive_scan_serial(in.data(),
                              static_cast<std::int64_t>(in.size()),
                              out.data());
    std::int64_t acc = 0;
    for (std::size_t i = 0; i < in.size(); ++i) {
        acc += in[i];
        EXPECT_EQ(out[i], acc) << "row " << i;
    }
}

TEST(BoltPrefixScan, ParallelInt64ExactVsSerial) {
    constexpr std::int64_t kN = 3'000'000;   // ~46 blocks
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<std::int64_t> dist(-1'000'000, 1'000'000);
    std::vector<std::int64_t> in(kN);
    for (auto& v : in) v = dist(rng);

    std::vector<std::int64_t> ser(kN), par(kN);
    ps::inclusive_scan_serial(in.data(), kN, ser.data());

    bolt::Scheduler sched;
    ASSERT_TRUE(sched.init(0));
    bolt::Arena arena;
    const bool ran_parallel =
        ps::inclusive_scan<std::int64_t>(&sched, in.data(), kN, par.data(),
                                         &arena);
    if (sched.thread_count() > 1) EXPECT_TRUE(ran_parallel);
    sched.shutdown();
    for (std::int64_t i = 0; i < kN; ++i) {
        ASSERT_EQ(par[i], ser[i]) << "row " << i;
    }
}

TEST(BoltPrefixScan, ParallelFloat64DeterministicBlockedAssociation) {
    constexpr std::int64_t kN = 1'500'000;
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> in(kN);
    for (auto& v : in) v = dist(rng);

    std::vector<double> ref;
    blocked_reference(in, ref);

    bolt::Scheduler sched;
    ASSERT_TRUE(sched.init(0));
    bolt::Arena arena;
    std::vector<double> par(kN);
    const bool ran_parallel =
        ps::inclusive_scan<double>(&sched, in.data(), kN, par.data(), &arena);
    const std::uint32_t workers = sched.thread_count();
    sched.shutdown();
    if (workers <= 1) GTEST_SKIP() << "single-core host";
    ASSERT_TRUE(ran_parallel);

    // Bit-identical to the fixed-block association (pool-size independent).
    for (std::int64_t i = 0; i < kN; ++i) {
        ASSERT_EQ(par[i], ref[static_cast<std::size_t>(i)]) << "row " << i;
    }
    // Near the strict serial fold (one extra rounding per row).
    std::vector<double> ser(kN);
    ps::inclusive_scan_serial(in.data(), kN, ser.data());
    for (std::int64_t i = 0; i < kN; i += 1013) {
        const double tol = 1e-9 * (std::fabs(ser[i]) + 1.0);
        ASSERT_NEAR(par[i], ser[i], tol) << "row " << i;
    }
}

TEST(BoltPrefixScan, ParallelFloat64ExactForIntegerValuedDoubles) {
    // The window-operator case: Int64 column widened to double (TAQ
    // SUM(size) cumulative). All intermediate sums stay integers below
    // 2^53, so the blocked association is EXACT == the serial fold.
    constexpr std::int64_t kN = 2'000'000;
    std::mt19937_64 rng(99);
    std::uniform_int_distribution<std::int64_t> dist(0, 100'000);
    std::vector<double> in(kN);
    for (auto& v : in) v = static_cast<double>(dist(rng));

    std::vector<double> ser(kN), par(kN);
    ps::inclusive_scan_serial(in.data(), kN, ser.data());

    bolt::Scheduler sched;
    ASSERT_TRUE(sched.init(0));
    bolt::Arena arena;
    ps::inclusive_scan<double>(&sched, in.data(), kN, par.data(), &arena);
    sched.shutdown();
    for (std::int64_t i = 0; i < kN; ++i) {
        ASSERT_EQ(par[i], ser[i]) << "row " << i;
    }
}

TEST(BoltPrefixScan, GatingSmallInputAndNullScheduler) {
    // n <= one block → serial path (return false), correct fold.
    const std::int64_t n = ps::k_scan_block_rows;   // exactly one block
    std::vector<std::int64_t> in(static_cast<std::size_t>(n), 2);
    std::vector<std::int64_t> out(static_cast<std::size_t>(n));

    bolt::Scheduler sched;
    ASSERT_TRUE(sched.init(0));
    bolt::Arena arena;
    EXPECT_FALSE(ps::inclusive_scan<std::int64_t>(&sched, in.data(), n,
                                                  out.data(), &arena));
    EXPECT_EQ(out[0], 2);
    EXPECT_EQ(out[static_cast<std::size_t>(n) - 1], 2 * n);

    // Null scheduler → serial path.
    EXPECT_FALSE(ps::inclusive_scan<std::int64_t>(nullptr, in.data(), n,
                                                  out.data(), nullptr));
    EXPECT_EQ(out[static_cast<std::size_t>(n) - 1], 2 * n);

    // n == 0 is legal and a no-op.
    EXPECT_FALSE(ps::inclusive_scan<std::int64_t>(&sched, nullptr, 0,
                                                  nullptr, &arena));
    sched.shutdown();
}

TEST(BoltPrefixScan, InPlaceAndMisalignedTail) {
    // 2 blocks + a ragged 12,345-row tail; scan in place (in == out).
    const std::int64_t n = 2 * ps::k_scan_block_rows + 12'345;
    std::mt19937_64 rng(1234);
    std::uniform_int_distribution<std::int64_t> dist(-50, 50);
    std::vector<std::int64_t> data(static_cast<std::size_t>(n));
    for (auto& v : data) v = dist(rng);
    std::vector<std::int64_t> ser(static_cast<std::size_t>(n));
    ps::inclusive_scan_serial(data.data(), n, ser.data());

    bolt::Scheduler sched;
    ASSERT_TRUE(sched.init(0));
    bolt::Arena arena;
    ps::inclusive_scan<std::int64_t>(&sched, data.data(), n, data.data(),
                                     &arena);
    sched.shutdown();
    for (std::int64_t i = 0; i < n; ++i) {
        ASSERT_EQ(data[static_cast<std::size_t>(i)],
                  ser[static_cast<std::size_t>(i)]) << "row " << i;
    }
}
