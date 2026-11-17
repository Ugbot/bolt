// test_promql_time_fn.cpp: the over_time folds and the frame wrappers of
// promql_time_fn.h, against literals worked out by hand (not by the kernels).

#include <gtest/gtest.h>

#include "bolt/kernels/promql.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace bp = bolt::promql;

namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();
const double kNan = std::numeric_limits<double>::quiet_NaN();

bp::TimeFnFrame frame(const int64_t* ts, double* v, int64_t n, int64_t lo, int64_t hi) {
    bp::TimeFnFrame f{};
    f.ts = ts; f.v = v; f.n = n; f.lo = lo; f.hi = hi; f.eval_ts = hi;
    f.ts_per_second = 1000.0;
    return f;
}

std::uint64_t bits(double d) { std::uint64_t b; std::memcpy(&b, &d, 8); return b; }
}  // namespace

TEST(PromqlOverTime, SumIsCompensated) {
    const double v[4] = {1e16, 1.0, -1e16, 1.0};
    EXPECT_EQ(bp::promql_sum_over_time(v, 4), 2.0);   // a plain sum gives 1
    const double inf[3] = {1.7e308, 1.7e308, -1.0};
    EXPECT_EQ(bp::promql_sum_over_time(inf, 3), kInf);
    EXPECT_TRUE(bp::promql_is_no_sample(bp::promql_sum_over_time(v, 0)));
}

TEST(PromqlOverTime, AvgSurvivesAnOverflowingSum) {
    const double v[4] = {9.99e307, 9.99e307, 9.99e307, 9.99e307};
    EXPECT_EQ(bp::promql_avg_over_time(v, 4), 9.99e307);
    const double w[3] = {1.0, 2.0, 6.0};
    EXPECT_EQ(bp::promql_avg_over_time(w, 3), 3.0);
    const double mixed[2] = {kInf, -kInf};
    EXPECT_TRUE(std::isnan(bp::promql_avg_over_time(mixed, 2)));
}

TEST(PromqlOverTime, MinMaxIgnoreNanAndKeepFirstZero) {
    const double v[4] = {kNan, 3.0, -0.0, 0.0};
    EXPECT_EQ(bp::promql_min_over_time(v, 4), -0.0);
    EXPECT_TRUE(std::signbit(bp::promql_min_over_time(v, 4)));
    EXPECT_EQ(bp::promql_max_over_time(v, 4), 3.0);
    const double nn[2] = {kNan, kNan};
    EXPECT_TRUE(std::isnan(bp::promql_min_over_time(nn, 2)));
    EXPECT_FALSE(bp::promql_is_no_sample(bp::promql_max_over_time(nn, 2)));
    const double z[2] = {0.0, -0.0};
    EXPECT_FALSE(std::signbit(bp::promql_max_over_time(z, 2)));
}

TEST(PromqlOverTime, TsOfExtremeLatestWinsATie) {
    const int64_t ts[3] = {0, 10000, 20000};
    const double v[3] = {1.0, 5.0, 1.0};
    EXPECT_EQ(bp::promql_ts_of_extreme_over_time(ts, v, 3, true, 1000.0), 20.0);
    EXPECT_EQ(bp::promql_ts_of_extreme_over_time(ts, v, 3, false, 1000.0), 10.0);
    const double nn[3] = {kNan, kNan, kNan};
    EXPECT_EQ(bp::promql_ts_of_extreme_over_time(ts, nn, 3, true, 1000.0), 20.0);
    EXPECT_EQ(bp::promql_ts_of_extreme_over_time(ts, v, 3, true, 1.0), 20000.0);
    EXPECT_EQ(bp::promql_ts_of_edge_over_time(ts, 3, false, 1000.0), 0.0);
    EXPECT_EQ(bp::promql_ts_of_edge_over_time(ts, 3, true, 1000.0), 20.0);
}

TEST(PromqlTimeFn, RateFamilyUnitsAndAbsence) {
    int64_t ts[3] = {10000, 20000, 30000};
    double v[3] = {10.0, 20.0, 30.0};
    const bp::TimeFnFrame ms = frame(ts, v, 3, 0, 40000);
    // Same samples in microseconds, ts_per_second = 1e6, give the same rate.
    int64_t tu[3] = {10000000, 20000000, 30000000};
    double vu[3] = {10.0, 20.0, 30.0};
    bp::TimeFnFrame us = frame(tu, vu, 3, 0, 40000000);
    us.ts_per_second = 1e6;
    const bp::TimeFnOut a = bp::promql_tf_rate(ms), b = bp::promql_tf_rate(us);
    ASSERT_TRUE(a.present && b.present);
    EXPECT_EQ(bits(a.value), bits(b.value));
    EXPECT_EQ(bits(a.value), bits(bp::promql_rate(ts, v, 3, 0, 40000)));
    bp::TimeFnFrame one = frame(ts, v, 1, 0, 40000);
    EXPECT_FALSE(bp::promql_tf_rate(one).present);
    EXPECT_FALSE(bp::promql_tf_irate(one).present);
    one.n = 0;
    EXPECT_FALSE(bp::promql_tf_present(one).present);
}

TEST(PromqlTimeFn, CountersAndShape) {
    int64_t ts[5] = {1000, 2000, 3000, 4000, 5000};
    double v[5] = {1.0, 2.0, 1.0, kNan, kNan};
    const bp::TimeFnFrame f = frame(ts, v, 5, 0, 6000);
    EXPECT_EQ(bp::promql_tf_resets(f).value, 1.0);
    EXPECT_EQ(bp::promql_tf_changes(f).value, 3.0);
    EXPECT_EQ(bp::promql_tf_present(f).value, 1.0);
    double w[3] = {3.0, 1.0, 2.0};
    bp::TimeFnFrame q = frame(ts, w, 3, 0, 6000);
    q.p0 = 0.5;
    EXPECT_EQ(bp::promql_tf_quantile(q).value, 2.0);
    EXPECT_TRUE(bp::promql_tf_mutates(&bp::promql_tf_quantile));
    EXPECT_FALSE(bp::promql_tf_mutates(&bp::promql_tf_rate));
    double d[3] = {1.0, 2.0, 4.0};
    bp::TimeFnFrame e = frame(ts, d, 3, 0, 6000);
    e.p0 = 0.5; e.p1 = 0.5;
    EXPECT_TRUE(bp::promql_tf_double_exp(e).present);
    e.p0 = 1.0;
    EXPECT_FALSE(bp::promql_tf_double_exp(e).present);
}
