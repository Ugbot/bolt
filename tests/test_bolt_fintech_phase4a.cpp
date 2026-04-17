// test_bolt_fintech_phase4a.cpp — Phase 4a (zero-dep) Tier 1 fintech kernels.
//
// Covers the 12 Phase 4 kernels that depend only on arithmetic primitives
// (no Phase 1 diff/log/sumSq). Round-trip against hand-computed references
// on 4-8-row inputs; guards against div-by-zero / sentinel paths.
//
// Phase 4b (log/diff-dependent subset: LogReturns, OFI, Parkinson, etc.)
// lands in a separate file once Phase 1 is wired through the operator
// surface.

#include <gtest/gtest.h>

#include "bolt/kernels/fintech/arrival_price_impact.h"
#include "bolt/kernels/fintech/cip_basis.h"
#include "bolt/kernels/fintech/effective_spread.h"
#include "bolt/kernels/fintech/forward_points.h"
#include "bolt/kernels/fintech/funding_apr.h"
#include "bolt/kernels/fintech/funding_cost.h"
#include "bolt/kernels/fintech/l1_imbalance.h"
#include "bolt/kernels/fintech/microprice.h"
#include "bolt/kernels/fintech/midprice.h"
#include "bolt/kernels/fintech/perp_fair_price.h"
#include "bolt/kernels/fintech/quoted_spread.h"
#include "bolt/kernels/fintech/signed_volume.h"

#include <cmath>
#include <cstdint>

using namespace bolt::fintech;

namespace {

constexpr double kEps = 1e-12;

}  // namespace

// ---------------------------------------------------------------------------
// P4.1 Midprice
// ---------------------------------------------------------------------------
TEST(Phase4A_Midprice, TwoRow) {
    const double bid[] = {100.0, 102.0};
    const double ask[] = {101.0, 103.0};
    double out[2] = {};
    const double* in[] = {bid, ask};
    MidpriceOpDesc desc{0, 1, 2};
    midprice_kernel(in, 2, out, desc);
    EXPECT_NEAR(out[0], 100.5, kEps);
    EXPECT_NEAR(out[1], 102.5, kEps);
}

TEST(Phase4A_Midprice, EightRow) {
    const double bid[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    const double ask[] = {2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    double out[8] = {};
    const double* in[] = {bid, ask};
    MidpriceOpDesc desc{0, 1, 2};
    midprice_kernel(in, 8, out, desc);
    for (int i = 0; i < 8; ++i) {
        EXPECT_NEAR(out[i], (bid[i] + ask[i]) * 0.5, kEps);
    }
}

// ---------------------------------------------------------------------------
// P4.2 Microprice
// ---------------------------------------------------------------------------
TEST(Phase4A_Microprice, HandComputed) {
    // bid=100, ask=101, bid_sz=10, ask_sz=30:
    // (100*30 + 101*10) / (10+30) = (3000 + 1010) / 40 = 100.25
    const double bid[]    = {100.0, 100.0};
    const double ask[]    = {101.0, 101.0};
    const double bid_sz[] = { 10.0,  40.0};
    const double ask_sz[] = { 30.0,  40.0};
    double out[2] = {};
    const double* in[] = {bid, ask, bid_sz, ask_sz};
    MicropriceOpDesc desc{0, 1, 2, 3, 4};
    microprice_kernel(in, 2, out, desc);
    EXPECT_NEAR(out[0], 100.25, kEps);
    // balanced sizes -> midprice = 100.5
    EXPECT_NEAR(out[1], 100.5, kEps);
}

// ---------------------------------------------------------------------------
// P4.3 L1Imbalance — including 0/0 NaN guard
// ---------------------------------------------------------------------------
TEST(Phase4A_L1Imbalance, HandComputed) {
    const double bid_sz[] = {10.0, 20.0};
    const double ask_sz[] = {30.0, 20.0};
    double out[2] = {};
    const double* in[] = {bid_sz, ask_sz};
    L1ImbalanceOpDesc desc{0, 1, 2};
    l1_imbalance_kernel(in, 2, out, desc);
    EXPECT_NEAR(out[0], -0.5, kEps);
    EXPECT_NEAR(out[1],  0.0, kEps);
}

TEST(Phase4A_L1Imbalance, ZeroDivNaN) {
    const double bid_sz[] = {0.0};
    const double ask_sz[] = {0.0};
    double out[1] = {};
    const double* in[] = {bid_sz, ask_sz};
    L1ImbalanceOpDesc desc{0, 1, 2};
    l1_imbalance_kernel(in, 1, out, desc);
    EXPECT_TRUE(std::isnan(out[0]));
}

// ---------------------------------------------------------------------------
// P4.4 QuotedSpread
// ---------------------------------------------------------------------------
TEST(Phase4A_QuotedSpread, HandComputed) {
    const double bid[] = {99.5, 100.0, 100.5, 101.0};
    const double ask[] = {100.0, 100.25, 100.75, 101.125};
    double out[4] = {};
    const double* in[] = {bid, ask};
    QuotedSpreadOpDesc desc{0, 1, 2};
    quoted_spread_kernel(in, 4, out, desc);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(out[i], ask[i] - bid[i], kEps);
    }
}

// ---------------------------------------------------------------------------
// P4.5 EffectiveSpread
// ---------------------------------------------------------------------------
TEST(Phase4A_EffectiveSpread, HandComputed) {
    // 2 * |exec - mid|
    const double exec[] = {100.10, 99.90, 100.00, 101.50};
    const double mid[]  = {100.00, 100.00, 100.00, 100.00};
    double out[4] = {};
    const double* in[] = {exec, mid};
    EffectiveSpreadOpDesc desc{0, 1, 2};
    effective_spread_kernel(in, 4, out, desc);
    EXPECT_NEAR(out[0], 0.20, kEps);
    EXPECT_NEAR(out[1], 0.20, kEps);
    EXPECT_NEAR(out[2], 0.00, kEps);
    EXPECT_NEAR(out[3], 3.00, kEps);
}

// ---------------------------------------------------------------------------
// P4.7 SignedVolume — side in {-1, 0, +1}
// ---------------------------------------------------------------------------
TEST(Phase4A_SignedVolume, SignOfSide) {
    const int64_t side[] = {-1, 0, +1, +1, -1};
    const double  qty[]  = {100.0, 100.0, 100.0, 250.0, 75.0};
    double out[5] = {};
    SignedVolumeOpDesc desc{0, 1, 2};
    signed_volume_kernel(side, qty, 5, out, desc);
    EXPECT_NEAR(out[0], -100.0, kEps);
    EXPECT_NEAR(out[1],    0.0, kEps);
    EXPECT_NEAR(out[2], +100.0, kEps);
    EXPECT_NEAR(out[3], +250.0, kEps);
    EXPECT_NEAR(out[4],  -75.0, kEps);
}

// ---------------------------------------------------------------------------
// P4.11 ForwardPoints
// ---------------------------------------------------------------------------
TEST(Phase4A_ForwardPoints, HandComputed) {
    // T=1, rd=0.05, rf=0.02 -> forward = spot * exp(0.03) ≈ spot * 1.030455
    const double spot[]  = {100.0, 200.0};
    const double r_dom[] = {0.05, 0.05};
    const double r_for[] = {0.02, 0.02};
    double out[2] = {};
    const double* in[] = {spot, r_dom, r_for};
    ForwardPointsOpDesc desc{0, 1, 2, 3, /*t_years*/ 1.0};
    forward_points_kernel(in, 2, out, desc);
    const double expect0 = 100.0 * std::exp(0.03) - 100.0;
    const double expect1 = 200.0 * std::exp(0.03) - 200.0;
    EXPECT_NEAR(out[0], expect0, 1e-10);
    EXPECT_NEAR(out[1], expect1, 1e-10);
}

// ---------------------------------------------------------------------------
// P4.12 FundingCost = position * price * funding_rate
// ---------------------------------------------------------------------------
TEST(Phase4A_FundingCost, HandComputed) {
    const double position[] = { 10.0, -5.0, 100.0};
    const double price[]    = {100.0, 200.0,  50.0};
    const double rate[]     = {0.01, 0.01, 0.001};
    double out[3] = {};
    const double* in[] = {position, price, rate};
    FundingCostOpDesc desc{0, 1, 2, 3};
    funding_cost_kernel(in, 3, out, desc);
    EXPECT_NEAR(out[0],   10.0,  kEps);  // 10 * 100 * 0.01
    EXPECT_NEAR(out[1],  -10.0,  kEps);  // -5 * 200 * 0.01
    EXPECT_NEAR(out[2],    5.0,  kEps);  // 100 * 50 * 0.001
}

// ---------------------------------------------------------------------------
// P4.13 CIPBasis — log-form, matches chukonu
// ---------------------------------------------------------------------------
TEST(Phase4A_CIPBasis, HandComputed) {
    // If f = s*exp((rd-rf)*T) exactly, then implied == actual and basis = 0.
    const double T = 0.5;
    const double spot[]    = {100.0, 100.0};
    const double r_dom[]   = {0.04, 0.04};
    const double r_for[]   = {0.01, 0.01};
    const double fwd_ok    = 100.0 * std::exp((0.04 - 0.01) * T);
    const double fwd_skew  = fwd_ok * std::exp(0.001 * T);  // 10bp basis
    const double forward[] = {fwd_ok, fwd_skew};
    double out[2] = {};
    const double* in[] = {spot, forward, r_dom, r_for};
    CIPBasisOpDesc desc{0, 1, 2, 3, 4, T};
    cip_basis_kernel(in, 2, out, desc);
    EXPECT_NEAR(out[0], 0.0,   1e-12);
    EXPECT_NEAR(out[1], 0.001, 1e-12);
}

// ---------------------------------------------------------------------------
// P4.14 FundingAPR = rate * periods_per_year * 100
// ---------------------------------------------------------------------------
TEST(Phase4A_FundingAPR, HandComputed) {
    // 8h funding: 365 days * 3 periods/day = 1095
    const double rate[] = {0.0001, 0.0005, -0.0002};
    double out[3] = {};
    const double* in[] = {rate};
    FundingAPROpDesc desc{0, 1, /*periods_per_year*/ 1095.0};
    funding_apr_kernel(in, 3, out, desc);
    EXPECT_NEAR(out[0],  0.0001 * 1095.0 * 100.0, 1e-9);
    EXPECT_NEAR(out[1],  0.0005 * 1095.0 * 100.0, 1e-9);
    EXPECT_NEAR(out[2], -0.0002 * 1095.0 * 100.0, 1e-9);
}

// ---------------------------------------------------------------------------
// P4.15 PerpFairPrice = spot * (1 + fr * hours / 8)
// ---------------------------------------------------------------------------
TEST(Phase4A_PerpFairPrice, HandComputed) {
    const double spot[]  = {10000.0, 10000.0, 10000.0};
    const double fr[]    = {0.0001, 0.0001, -0.0002};
    const double hours[] = {8.0, 4.0, 8.0};
    double out[3] = {};
    const double* in[] = {spot, fr, hours};
    PerpFairPriceOpDesc desc{0, 1, 2, 3};
    perp_fair_price_kernel(in, 3, out, desc);
    EXPECT_NEAR(out[0], 10000.0 * (1.0 + 0.0001 * 8.0 / 8.0), 1e-9);
    EXPECT_NEAR(out[1], 10000.0 * (1.0 + 0.0001 * 4.0 / 8.0), 1e-9);
    EXPECT_NEAR(out[2], 10000.0 * (1.0 - 0.0002 * 8.0 / 8.0), 1e-9);
}

// ---------------------------------------------------------------------------
// P4.9 ArrivalPriceImpact — bps, with 1e-15 zero-mid sentinel
// ---------------------------------------------------------------------------
TEST(Phase4A_ArrivalPriceImpact, HandComputed) {
    // exec=100.05, mid=100.00 -> 10000 * 0.05 / 100 = 5.0 bps
    const double exec[] = {100.05, 99.95, 100.00};
    const double mid[]  = {100.00, 100.00, 100.00};
    double out[3] = {};
    const double* in[] = {exec, mid};
    ArrivalPriceImpactOpDesc desc{0, 1, 2};
    arrival_price_impact_kernel(in, 3, out, desc);
    EXPECT_NEAR(out[0],  5.0, 1e-9);
    EXPECT_NEAR(out[1], -5.0, 1e-9);
    EXPECT_NEAR(out[2],  0.0, 1e-9);
}

TEST(Phase4A_ArrivalPriceImpact, ZeroMidSentinel) {
    const double exec[] = {1.0};
    const double mid[]  = {0.0};
    double out[1] = {};
    const double* in[] = {exec, mid};
    ArrivalPriceImpactOpDesc desc{0, 1, 2};
    arrival_price_impact_kernel(in, 1, out, desc);
    EXPECT_NEAR(out[0], 0.0, kEps);  // chukonu sentinel, not inf
}
