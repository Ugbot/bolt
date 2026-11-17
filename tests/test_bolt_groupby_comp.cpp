// SumComp / AvgComp: Neumaier-compensated float64 fold cells whose partials merge as rows do (G2CHK-1201).
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "bolt/join/bolt_groupby.h"

namespace {

struct Cell { bolt::GbCell16 c{0, 0}; int64_t n = 0; };

bolt::GbCell16 row_cell(double x) {
    bolt::GbCell16 c{0, 0};
    std::memcpy(&c.a, &x, 8);
    return c;
}

void fold_row(bolt::AggKind k, Cell* d, double x) {
    bolt::gb_fold_combine(k, bolt::BoltType::Float64, &d->c, &d->n, row_cell(x), 1);
}

void merge(bolt::AggKind k, Cell* d, const Cell& s) {
    bolt::gb_fold_combine(k, bolt::BoltType::Float64, &d->c, &d->n, s.c, s.n);
}

double answer(bolt::AggKind k, Cell c) {
    double out = 0;
    bolt::gb_fold_emit(k, bolt::BoltType::Float64, 0, 0, c.c, &c.n, &out, 0);
    return out;
}

}  // namespace

TEST(GroupbyComp, CancellingTermsKeepTheSmallOneInAnyOrderAndAnySplit) {
    const double v[3] = {1e300, -2.25, -1e300};
    for (int perm = 0; perm < 6; ++perm) {
        static const int order[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
        for (int split = 0; split <= 3; ++split) {   // rows [0, split) in one partial, the rest in the other
            Cell a, b;
            for (int i = 0; i < 3; ++i) fold_row(bolt::AggKind::SumComp, i < split ? &a : &b, v[order[perm][i]]);
            Cell m = a;
            merge(bolt::AggKind::SumComp, &m, b);
            EXPECT_EQ(answer(bolt::AggKind::SumComp, m), -2.25) << "perm " << perm << " split " << split;
            EXPECT_EQ(m.n, 3);
        }
    }
}

TEST(GroupbyComp, AMergedSplitEqualsOneFoldOverTheRows) {
    uint64_t s = 88172645463325252ULL;
    auto rnd = [&s] { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; };
    for (int rep = 0; rep < 200; ++rep) {
        std::vector<double> xs(1 + rnd() % 40);
        for (auto& x : xs) {
            const double mag = std::ldexp(1.0, static_cast<int>(rnd() % 80) - 20);
            x = (static_cast<double>(rnd() % 2001) - 1000.0) * mag / 1000.0;
        }
        Cell whole;
        for (double x : xs) fold_row(bolt::AggKind::SumComp, &whole, x);
        const size_t cut = rnd() % (xs.size() + 1);
        Cell a, b;
        for (size_t i = 0; i < xs.size(); ++i) fold_row(bolt::AggKind::SumComp, i < cut ? &a : &b, xs[i]);
        merge(bolt::AggKind::SumComp, &a, b);
        const double got = answer(bolt::AggKind::SumComp, a), want = answer(bolt::AggKind::SumComp, whole);
        EXPECT_LE(std::fabs(got - want), 1e-12 * std::fabs(want) + 1e-300) << "rep " << rep;
    }
}

TEST(GroupbyComp, InfinitiesNaNAndSignedZeroAnswerAsASequentialSumDoes) {
    Cell z;
    fold_row(bolt::AggKind::SumComp, &z, -0.0);
    EXPECT_EQ(answer(bolt::AggKind::SumComp, z), 0.0);   // sum + compensation, as the single-state fold answers
    Cell inf;
    fold_row(bolt::AggKind::SumComp, &inf, std::numeric_limits<double>::infinity());
    fold_row(bolt::AggKind::SumComp, &inf, 5.0);
    EXPECT_EQ(answer(bolt::AggKind::SumComp, inf), std::numeric_limits<double>::infinity());
    Cell both;
    fold_row(bolt::AggKind::SumComp, &both, std::numeric_limits<double>::infinity());
    fold_row(bolt::AggKind::SumComp, &both, -std::numeric_limits<double>::infinity());
    EXPECT_TRUE(std::isnan(answer(bolt::AggKind::SumComp, both)));
    Cell nan;
    fold_row(bolt::AggKind::SumComp, &nan, 1.0);
    fold_row(bolt::AggKind::SumComp, &nan, std::nan(""));
    EXPECT_TRUE(std::isnan(answer(bolt::AggKind::SumComp, nan)));
}

TEST(GroupbyComp, AvgIsTheCompensatedSumOverTheCount) {
    Cell a, b;
    fold_row(bolt::AggKind::AvgComp, &a, 1e300);
    fold_row(bolt::AggKind::AvgComp, &b, -2.25);
    fold_row(bolt::AggKind::AvgComp, &b, -1e300);
    merge(bolt::AggKind::AvgComp, &a, b);
    EXPECT_EQ(answer(bolt::AggKind::AvgComp, a), -2.25 / 3.0);
    Cell empty;
    double out = 7;
    bolt::gb_fold_emit(bolt::AggKind::AvgComp, bolt::BoltType::Float64, 0, 0, empty.c, &empty.n, &out, 0);
    EXPECT_LE(empty.n, 0);   // nothing folded reads as NULL
}

TEST(GroupbyComp, AvgOfValuesWhoseSumLeavesDoubleRangeIsTheirMeanInAnySplit) {
    const double big = 9.988465674311579e307;
    const double vals[3][3] = {{big, big, big}, {big, -big, 0.0}, {-big, -big, -big}};
    const double want[3] = {big, 0.0, -big};
    for (int v = 0; v < 3; ++v) {
        for (int split = 0; split <= 3; ++split) {
            Cell a, b;
            for (int i = 0; i < 3; ++i) fold_row(bolt::AggKind::AvgComp, i < split ? &a : &b, vals[v][i]);
            merge(bolt::AggKind::AvgComp, &a, b);
            const double got = answer(bolt::AggKind::AvgComp, a);
            if (v == 1) EXPECT_EQ(got, 0.0) << "split " << split;
            else EXPECT_LE(std::fabs(got - want[v]), 1e-14 * std::fabs(want[v])) << "v " << v << " split " << split << " got " << got;
        }
    }
    Cell inf;   // an infinite value is the mean
    fold_row(bolt::AggKind::AvgComp, &inf, 1.0);
    fold_row(bolt::AggKind::AvgComp, &inf, std::numeric_limits<double>::infinity());
    EXPECT_EQ(answer(bolt::AggKind::AvgComp, inf), std::numeric_limits<double>::infinity());
}

TEST(GroupbyComp, AvgOverAnySplitEqualsOneFold) {
    uint64_t s = 1234567891011ULL;
    auto rnd = [&s] { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; };
    for (int rep = 0; rep < 200; ++rep) {
        std::vector<double> xs(1 + rnd() % 30);
        for (auto& x : xs) x = (static_cast<double>(rnd() % 2001) - 1000.0) * std::ldexp(1.0, static_cast<int>(rnd() % 40) - 10);
        Cell whole, a, b;
        for (double x : xs) fold_row(bolt::AggKind::AvgComp, &whole, x);
        const size_t cut = rnd() % (xs.size() + 1);
        for (size_t i = 0; i < xs.size(); ++i) fold_row(bolt::AggKind::AvgComp, i < cut ? &a : &b, xs[i]);
        merge(bolt::AggKind::AvgComp, &a, b);
        const double got = answer(bolt::AggKind::AvgComp, a), want = answer(bolt::AggKind::AvgComp, whole);
        EXPECT_LE(std::fabs(got - want), 1e-12 * std::fabs(want) + 1e-300) << "rep " << rep;
    }
}
