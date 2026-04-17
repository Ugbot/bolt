// test_bolt_fintech_phase4b.cpp — Phase 4b (log/diff-dependent) Tier 1 kernels.
//
// The 14 Phase 4 kernels that build on Phase 1 primitives (diff, log,
// sum_of_squares) plus the four order-entry guard/flag kernels. Every
// case is hand-computed on 4-8-row inputs so the expected value is
// visible in the source — no external reference dependency.

#include <gtest/gtest.h>

#include "bolt/kernels/fintech/bipower_var.h"
#include "bolt/kernels/fintech/circuit_breaker.h"
#include "bolt/kernels/fintech/corwin_schultz.h"
#include "bolt/kernels/fintech/fat_finger_guard.h"
#include "bolt/kernels/fintech/garman_klass.h"
#include "bolt/kernels/fintech/implementation_shortfall.h"
#include "bolt/kernels/fintech/log_returns.h"
#include "bolt/kernels/fintech/ofi.h"
#include "bolt/kernels/fintech/open_interest_delta.h"
#include "bolt/kernels/fintech/parkinson.h"
#include "bolt/kernels/fintech/price_band_guard.h"
#include "bolt/kernels/fintech/realized_var.h"
#include "bolt/kernels/fintech/rogers_satchell.h"
#include "bolt/kernels/fintech/stale_quote_detector.h"

#include <cmath>
#include <cstdint>

using namespace bolt::fintech;

namespace {
constexpr double kEps = 1e-12;

#ifndef BOLT_PHASE4B_M_PI
#define BOLT_PHASE4B_M_PI 3.14159265358979323846
#endif
}  // namespace

// ---------------------------------------------------------------------------
// P4.6 LogReturns
// ---------------------------------------------------------------------------
TEST(Phase4B_LogReturns, HandComputed) {
    const double px[] = {100.0, 101.0, 99.0, 102.0};
    double out[4] = {};
    const double* in[] = {px};
    LogReturnsOpDesc desc{0, 1};
    log_returns_kernel(in, 4, out, desc);
    EXPECT_NEAR(out[0], 0.0, kEps);
    EXPECT_NEAR(out[1], std::log(101.0 / 100.0), kEps);
    EXPECT_NEAR(out[2], std::log( 99.0 / 101.0), kEps);
    EXPECT_NEAR(out[3], std::log(102.0 /  99.0), kEps);
}

TEST(Phase4B_LogReturns, Empty) {
    double out[1] = {0.12345};
    const double* in[] = {out};
    LogReturnsOpDesc desc{0, 1};
    log_returns_kernel(in, 0, out, desc);
    // untouched
    EXPECT_NEAR(out[0], 0.12345, 0.0);
}

// ---------------------------------------------------------------------------
// P4.8 OFI
// ---------------------------------------------------------------------------
TEST(Phase4B_OFI, HandComputed) {
    const double buy[]  = {10.0, 5.0, 7.0, 0.0};
    const double sell[] = { 3.0, 5.0, 9.0, 4.0};
    double out[4] = {};
    const double* in[] = {buy, sell};
    OFIOpDesc desc{0, 1, 2};
    ofi_kernel(in, 4, out, desc);
    EXPECT_NEAR(out[0],  7.0, kEps);
    EXPECT_NEAR(out[1],  0.0, kEps);
    EXPECT_NEAR(out[2], -2.0, kEps);
    EXPECT_NEAR(out[3], -4.0, kEps);
}

// ---------------------------------------------------------------------------
// P4.10 ImplementationShortfall — chukonu bps form
// ---------------------------------------------------------------------------
TEST(Phase4B_ImplementationShortfall, HandComputed) {
    // bps = 10000 * (exec - bench)/bench
    const double exec[]  = {100.05, 99.95, 100.00, 100.00};
    const double bench[] = {100.00, 100.00, 100.00,   0.0};   // last => sentinel
    double out[4] = {};
    const double* in[] = {exec, bench};
    ImplementationShortfallOpDesc desc{0, 1, 2};
    implementation_shortfall_kernel(in, 4, out, desc);
    EXPECT_NEAR(out[0],  5.0, 1e-9);
    EXPECT_NEAR(out[1], -5.0, 1e-9);
    EXPECT_NEAR(out[2],  0.0, 1e-9);
    EXPECT_NEAR(out[3],  0.0, kEps);   // zero-bench sentinel
}

// ---------------------------------------------------------------------------
// P4.16 OpenInterestDelta
// ---------------------------------------------------------------------------
TEST(Phase4B_OpenInterestDelta, HandComputed) {
    const double oi[] = {1000.0, 1050.0, 1025.0, 1100.0};
    double out[4] = {};
    const double* in[] = {oi};
    OpenInterestDeltaOpDesc desc{0, 1};
    open_interest_delta_kernel(in, 4, out, desc);
    EXPECT_NEAR(out[0],   0.0, kEps);   // sentinel
    EXPECT_NEAR(out[1],  50.0, kEps);
    EXPECT_NEAR(out[2], -25.0, kEps);
    EXPECT_NEAR(out[3],  75.0, kEps);
}

// ---------------------------------------------------------------------------
// P4.17 CorwinSchultz
// ---------------------------------------------------------------------------
TEST(Phase4B_CorwinSchultz, HandComputed) {
    // Recompute the formula inline for the reference.
    const double high[] = {101.0, 102.5, 103.0};
    const double low[]  = { 99.0,  99.5, 101.0};
    double out[3] = {};
    const double* in[] = {high, low};
    CorwinSchultzOpDesc desc{0, 1, 2};
    corwin_schultz_kernel(in, 3, out, desc);
    EXPECT_NEAR(out[0], 0.0, kEps);

    const double k_const = 3.0 - 2.0 * std::sqrt(2.0);
    for (int i = 1; i < 3; ++i) {
        const double h0 = high[i - 1], l0 = low[i - 1];
        const double h1 = high[i],     l1 = low[i];
        const double a = std::log(h0 / l0), b = std::log(h1 / l1);
        const double beta = (a*a + b*b) * 0.5;
        const double h_max = (h0 > h1) ? h0 : h1;
        const double l_min = (l0 < l1) ? l0 : l1;
        const double ln_rng = std::log(h_max / l_min);
        const double gamma_v = ln_rng * ln_rng;
        const double alpha = (std::sqrt(2.0 * beta) - std::sqrt(beta)) / k_const
                             - std::sqrt(gamma_v / k_const);
        const double e = std::exp(alpha);
        double s = 2.0 * (e - 1.0) / (1.0 + e);
        if (s < 0.0) s = 0.0;
        EXPECT_NEAR(out[i], s, 1e-12);
    }
}

// ---------------------------------------------------------------------------
// P4.18 GarmanKlass
// ---------------------------------------------------------------------------
TEST(Phase4B_GarmanKlass, HandComputed) {
    const double high[]  = {101.0, 102.5};
    const double low[]   = { 99.0,  99.5};
    const double open[]  = {100.0, 100.5};
    const double close[] = {100.5, 101.0};
    double out[2] = {};
    const double* in[] = {high, low, open, close};
    GarmanKlassOpDesc desc{0, 1, 2, 3, 4};
    garman_klass_kernel(in, 2, out, desc);
    const double coeff = 2.0 * std::log(2.0) - 1.0;
    for (int i = 0; i < 2; ++i) {
        const double lh = std::log(high[i] / low[i]);
        const double lc = std::log(close[i] / open[i]);
        const double exp_v = 0.5 * lh * lh - coeff * lc * lc;
        EXPECT_NEAR(out[i], exp_v, 1e-12);
    }
}

// ---------------------------------------------------------------------------
// P4.19 Parkinson
// ---------------------------------------------------------------------------
TEST(Phase4B_Parkinson, HandComputed) {
    const double high[] = {101.0, 102.5, 100.5};
    const double low[]  = { 99.0,  99.5, 100.0};
    double out[3] = {};
    const double* in[] = {high, low};
    ParkinsonOpDesc desc{0, 1, 2};
    parkinson_kernel(in, 3, out, desc);
    const double coeff = 1.0 / (4.0 * std::log(2.0));
    for (int i = 0; i < 3; ++i) {
        const double lh = std::log(high[i] / low[i]);
        EXPECT_NEAR(out[i], coeff * lh * lh, 1e-12);
    }
}

// ---------------------------------------------------------------------------
// P4.20 RogersSatchell
// ---------------------------------------------------------------------------
TEST(Phase4B_RogersSatchell, HandComputed) {
    const double high[]  = {101.0, 102.5};
    const double low[]   = { 99.0,  99.5};
    const double open[]  = {100.0, 100.5};
    const double close[] = {100.5, 101.0};
    double out[2] = {};
    const double* in[] = {high, low, open, close};
    RogersSatchellOpDesc desc{0, 1, 2, 3, 4};
    rogers_satchell_kernel(in, 2, out, desc);
    for (int i = 0; i < 2; ++i) {
        const double h = high[i], l = low[i], o = open[i], c = close[i];
        const double expv = std::log(h / c) * std::log(h / o)
                          + std::log(l / c) * std::log(l / o);
        EXPECT_NEAR(out[i], expv, 1e-12);
    }
}

// ---------------------------------------------------------------------------
// P4.21 RealizedVar — scalar broadcast
// ---------------------------------------------------------------------------
TEST(Phase4B_RealizedVar, ScalarBroadcast) {
    const double r[] = {0.01, -0.02, 0.005, -0.015};
    double out[4] = {};
    const double* in[] = {r};
    RealizedVarOpDesc desc{0, 1};
    realized_var_kernel(in, 4, out, desc);
    double rv = 0.0;
    for (int i = 0; i < 4; ++i) rv += r[i] * r[i];
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(out[i], rv, 1e-14);
    }
}

// ---------------------------------------------------------------------------
// P4.22 BipowerVar — (π/2) · Σ|r_i||r_{i-1}| / (n-1)
// ---------------------------------------------------------------------------
TEST(Phase4B_BipowerVar, ScalarBroadcast) {
    const double r[] = {0.01, -0.02, 0.005, -0.015, 0.02};
    constexpr int64_t n = 5;
    double out[n] = {};
    const double* in[] = {r};
    BipowerVarOpDesc desc{0, 1};
    bipower_var_kernel(in, n, out, desc);
    double s = 0.0;
    for (int64_t i = 1; i < n; ++i) s += std::fabs(r[i]) * std::fabs(r[i - 1]);
    const double expv = (BOLT_PHASE4B_M_PI * 0.5) * s / static_cast<double>(n - 1);
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(out[i], expv, 1e-14);
    }
}

TEST(Phase4B_BipowerVar, DegenerateN1) {
    const double r[] = {0.01};
    double out[1] = {7.0};
    const double* in[] = {r};
    BipowerVarOpDesc desc{0, 1};
    bipower_var_kernel(in, 1, out, desc);
    EXPECT_NEAR(out[0], 0.0, kEps);
}

// ---------------------------------------------------------------------------
// P4.23 StaleQuoteDetector
// ---------------------------------------------------------------------------
TEST(Phase4B_StaleQuoteDetector, HandComputed) {
    // max_age = 100; ts-lu: 50, 100, 101, 500
    const int64_t ts[] = {1000, 1000, 1000, 1000};
    const int64_t lu[] = { 950,  900,  899,  500};
    int32_t out[4] = {};
    const int64_t* in[] = {ts, lu};
    StaleQuoteDetectorOpDesc desc{0, 1, 2, /*max_age*/ 100};
    stale_quote_detector_kernel(in, 4, out, desc);
    EXPECT_EQ(out[0], 0);  // age 50  <= 100
    EXPECT_EQ(out[1], 0);  // age 100 <= 100  (strict >)
    EXPECT_EQ(out[2], 1);  // age 101 >  100
    EXPECT_EQ(out[3], 1);  // age 500 >  100
}

// ---------------------------------------------------------------------------
// P4.24 PriceBandGuard — task-spec polarity (1 = trip)
// ---------------------------------------------------------------------------
TEST(Phase4B_PriceBandGuard, HandComputed) {
    const double band = 0.05;     // 5%
    // deviations: 0.00, 0.04, 0.05 (edge -> trip), 0.06
    const double op[] = {100.0, 104.0, 105.0, 106.0, 1.0};
    const double rp[] = {100.0, 100.0, 100.0, 100.0, 0.0};
    int32_t out[5] = {};
    const double* in[] = {op, rp};
    PriceBandGuardOpDesc desc{0, 1, 2, band};
    price_band_guard_kernel(in, 5, out, desc);
    EXPECT_EQ(out[0], 0);
    EXPECT_EQ(out[1], 0);
    EXPECT_EQ(out[2], 1);   // dev == band -> trip (>=)
    EXPECT_EQ(out[3], 1);
    EXPECT_EQ(out[4], 1);   // zero ref_px -> trip
}

// ---------------------------------------------------------------------------
// P4.25 FatFingerGuard — price-band OR notional-cap trip
// ---------------------------------------------------------------------------
TEST(Phase4B_FatFingerGuard, HandComputed) {
    const double band = 0.05;
    const double cap  = 1000.0;    // notional ceiling
    //            row 0: ok (dev 0%, notional 100)
    //            row 1: notional trip (1000 == cap → >=)
    //            row 2: price trip    (dev 10%, notional 110)
    //            row 3: both trip
    const double op[] = {100.0, 100.0, 110.0, 200.0};
    const double os[] = {  1.0,  10.0,   1.0,  20.0};
    const double rp[] = {100.0, 100.0, 100.0, 100.0};
    int32_t out[4] = {};
    const double* in[] = {op, os, rp};
    FatFingerGuardOpDesc desc{0, 1, 2, 3, band, cap};
    fat_finger_guard_kernel(in, 4, out, desc);
    EXPECT_EQ(out[0], 0);
    EXPECT_EQ(out[1], 1);
    EXPECT_EQ(out[2], 1);
    EXPECT_EQ(out[3], 1);
}

// ---------------------------------------------------------------------------
// P4.26 CircuitBreaker — batch-local, strict-greater
// ---------------------------------------------------------------------------
TEST(Phase4B_CircuitBreaker, HandComputed) {
    // Use an IEEE-exact threshold (0.125 = 2^-3) so the boundary row is
    // unambiguous — `p/ref - 1 == 0.125` is representable exactly with
    // binary-clean prices.
    const double thr = 0.125;
    // Deviations (all exact in IEEE double): 0.0, 0.0625, 0.125, 0.25.
    const double p[]  = {1.0, 1.0625, 1.125, 1.25, 0.5};
    const double rp[] = {1.0, 1.0,    1.0,   1.0,  0.0};
    int32_t out[5] = {};
    const double* in[] = {p, rp};
    CircuitBreakerOpDesc desc{0, 1, 2, thr};
    circuit_breaker_kernel(in, 5, out, desc);
    EXPECT_EQ(out[0], 0);
    EXPECT_EQ(out[1], 0);
    EXPECT_EQ(out[2], 0);  // dev == thr, strict > → no trip
    EXPECT_EQ(out[3], 1);
    EXPECT_EQ(out[4], 0);  // zero ref → chukonu sentinel (no trip)
}
