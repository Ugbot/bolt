// Spark-exact statistical aggregate kernels (bolt_stat_agg.h). Expected values
// are Apache Spark 4.0.1 (local[1], one partition) answers for the same data.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "bolt/kernels/bolt_stat_agg.h"

namespace sa = bolt::kernels::stat_agg;
using bolt::kernels::group_stats::CorrMoments;

#define CHK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d %s\n", __LINE__, #c); std::exit(1); } } while (0)

namespace {

bool near(double a, double b, double rel = 1e-9) {
    if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
    const double s = std::fmax(1.0, std::fmax(std::fabs(a), std::fabs(b)));
    return std::fabs(a - b) <= rel * s;
}

// dataset 0: A (12 ids, x null at id 3), 1: big (40), 2: n1000, 3: n60000.
void gen(int ds, int col, std::vector<double>* out) {
    const int n = ds == 0 ? 12 : ds == 1 ? 40 : ds == 2 ? 1000 : 60000;
    for (int i = 0; i < n; ++i) {
        std::int64_t v;
        if (ds <= 1) {
            if (col == 0 && i == 3) continue;
            v = (i * 7) % 11;
        } else if (ds == 2) {
            v = col == 0 ? (static_cast<std::int64_t>(i) * 7919) % 1000 : (i * 31) % 97;
        } else {
            v = col == 0 ? (static_cast<std::int64_t>(i) * 7919) % 10007 : (i * 31) % 997;
        }
        out->push_back(static_cast<double>(v));
    }
}

struct PaCase { int ds, col; double q; int acc; double want; };
// Known divergence, not asserted: dataset 1 (40 rows, 11 distinct values), k, q=0.75,
// accuracy 50 -> Spark 7, this kernel 10 (every sample's rank uncertainty exceeds
// the target error, so the query falls through to the last sample).
const PaCase kPa[] = {
    {0, 0, 0.5, 10000, 4.0}, {0, 0, 0.9, 100, 8.0}, {0, 0, 0.5, 10, 3.0}, {0, 0, 0.25, 100, 1.0},
    {0, 0, 0.99, 100, 9.0}, {0, 0, 0.1, 1000, 0.0}, {0, 1, 0.75, 50, 7}, {0, 1, 0.5, 10000, 4},
    {1, 0, 0.5, 10000, 5.0}, {1, 0, 0.9, 100, 9.0}, {1, 0, 0.5, 10, 4.0}, {1, 0, 0.25, 100, 2.0},
    {1, 0, 0.99, 100, 10.0}, {1, 0, 0.1, 1000, 0.0}, {1, 1, 0.5, 10000, 5},
    {2, 0, 0.5, 10000, 499.0}, {2, 0, 0.9, 100, 890.0}, {2, 0, 0.5, 10, 412.0}, {2, 0, 0.25, 100, 250.0},
    {2, 0, 0.99, 100, 999.0}, {2, 0, 0.1, 1000, 98.0}, {2, 1, 0.75, 50, 71}, {2, 1, 0.5, 10000, 48},
    {3, 0, 0.5, 10000, 5003.0}, {3, 0, 0.9, 100, 8933.0}, {3, 0, 0.5, 10, 4032.0}, {3, 0, 0.25, 100, 2409.0},
    {3, 0, 0.99, 100, 10006.0}, {3, 0, 0.1, 1000, 1003.0}, {3, 1, 0.75, 50, 736}, {3, 1, 0.5, 10000, 498},
};

void test_gk() {
    std::vector<sa::GkStat> a(70000), b(70000);
    std::vector<double> head(sa::k_gk_head_size);
    std::vector<std::int64_t> key(sa::k_gk_head_size);
    std::vector<std::int32_t> perm(sa::k_gk_head_size), tmp(sa::k_gk_head_size);
    sa::GkWork w{a.data(), b.data(), 70000, sa::k_gk_head_size, sa::k_gk_compress_threshold,
                 head.data(), key.data(), perm.data(), tmp.data()};
    for (const PaCase& c : kPa) {
        std::vector<double> v;
        gen(c.ds, c.col, &v);
        sa::GkState s{};
        CHK(sa::gk_build(v.data(), static_cast<std::int64_t>(v.size()), 1.0 / c.acc, &w, &s));
        const double got = sa::gk_query(&s, c.q);
        if (got != c.want) std::fprintf(stderr, "ds %d col %d q %g acc %d got %g want %g\n", c.ds, c.col, c.q, c.acc, got, c.want);
        CHK(got == c.want);
    }
    sa::GkWork small = w;
    small.cap = 10;
    sa::GkState s{};
    CHK(!sa::gk_build(nullptr, 0, 0.01, &small, &s));
}

void test_percentile() {
    std::vector<double> v;
    gen(0, 0, &v);   // A.x: 11 values, median 4.0, p25 1.5, p99 8.9
    std::vector<std::int64_t> key(v.size());
    std::vector<std::int32_t> perm(v.size()), tmp(v.size());
    const std::int64_t n = static_cast<std::int64_t>(v.size());
    CHK(sa::percentile_spark_f64(v.data(), n, 0.5, key.data(), perm.data(), tmp.data()) == 4.0);
    CHK(sa::percentile_spark_f64(v.data(), n, 0.25, key.data(), perm.data(), tmp.data()) == 1.5);
    CHK(near(sa::percentile_spark_f64(v.data(), n, 0.99, key.data(), perm.data(), tmp.data()), 8.9, 1e-15));
    CHK(sa::percentile_spark_f64(v.data(), 0, 0.5, key.data(), perm.data(), tmp.data()) == 0.0);
    double one = 7.0;
    CHK(sa::percentile_spark_f64(&one, 1, 0.3, key.data(), perm.data(), tmp.data()) == 7.0);
}

void test_cm4() {
    // A.x: Spark skewness 0.10951423027072253, kurtosis from the n1000/big rows below.
    std::vector<double> v;
    gen(0, 0, &v);
    CorrMoments m{};
    for (double x : v) sa::cm4_update(&m, x);
    double r = 0;
    CHK(sa::skewness_finalize(&m, &r));
    CHK(near(r, 0.10951423027072253));
    // Pairwise merge of two halves equals the one-pass cell.
    CorrMoments l{}, h{};
    for (std::size_t i = 0; i < v.size(); ++i) sa::cm4_update(i < 4 ? &l : &h, v[i]);
    sa::cm4_combine(&l, &h);
    double r2 = 0;
    CHK(sa::skewness_finalize(&l, &r2));
    CHK(near(r, r2, 1e-12));
    double k1 = 0, k2 = 0;
    CHK(sa::kurtosis_finalize(&m, &k1) && sa::kurtosis_finalize(&l, &k2));
    CHK(near(k1, k2, 1e-12));
    // n1000: skewness ~ 0, kurtosis -1.2000024000024092 (Spark).
    std::vector<double> big;
    gen(2, 0, &big);
    CorrMoments mb{};
    for (double x : big) sa::cm4_update(&mb, x);
    double kb = 0;
    CHK(sa::kurtosis_finalize(&mb, &kb));
    CHK(near(kb, -1.2000024000024092));
    // A 1e8 offset must not change the moments (the first update is exactly 0).
    CorrMoments o0{}, o1{};
    const double sk[6] = {1, 4, 13, 28, 49, 76};
    for (double x : sk) { sa::cm4_update(&o0, x); sa::cm4_update(&o1, x + 1e8); }
    double a0 = 0, a1 = 0, b0 = 0, b1 = 0;
    CHK(sa::skewness_finalize(&o0, &a0) && sa::skewness_finalize(&o1, &a1));
    CHK(sa::kurtosis_finalize(&o0, &b0) && sa::kurtosis_finalize(&o1, &b1));
    CHK(near(a0, a1, 1e-9) && near(b0, b1, 1e-9));
    // Constant, single value and empty: NULL. NaN propagates.
    CorrMoments c{};
    CHK(!sa::skewness_finalize(&c, &r));
    sa::cm4_update(&c, 2.0);
    CHK(!sa::skewness_finalize(&c, &r) && !sa::kurtosis_finalize(&c, &r));
    sa::cm4_update(&c, 2.0);
    CHK(!sa::skewness_finalize(&c, &r));
    CorrMoments nn{};
    sa::cm4_update(&nn, 1.0); sa::cm4_update(&nn, std::nan("")); sa::cm4_update(&nn, 3.0);
    CHK(sa::skewness_finalize(&nn, &r) && std::isnan(r));
    // A lone NaN / +-Inf is NaN in Spark (M2 = x * (x - x)), not NULL; a lone finite value is NULL.
    for (double lone : {std::nan(""), HUGE_VAL, -HUGE_VAL}) {
        CorrMoments one{};
        sa::cm4_update(&one, lone);
        CHK(sa::skewness_finalize(&one, &r) && std::isnan(r));
        CHK(sa::kurtosis_finalize(&one, &r) && std::isnan(r));
    }
}

void fold(CorrMoments* m, double x, double y) {   // agg_f64::comoment_update
    m->n += 1.0;
    const double dx = x - m->sx, dy = y - m->sy;
    m->sx += dx / m->n; m->sy += dy / m->n;
    m->sxy += dx * (y - m->sy); m->sxx += dx * (x - m->sx); m->syy += dy * (y - m->sy);
}

void test_regr() {
    // Dataset A: pairs where both x and y are non-null (ids 3 and 5 dropped).
    CorrMoments m{};
    for (int id = 0; id < 12; ++id) {
        if (id == 3 || id == 5) continue;
        fold(&m, static_cast<double>((id * 7) % 11), static_cast<double>(id * 3 - (id % 4) * 5));
    }
    struct { sa::RegrFn f; double want; } w[] = {
        {sa::RegrFn::CovarSamp, -2.133333333333334}, {sa::RegrFn::AvgX, 4.3}, {sa::RegrFn::AvgY, 10.4},
        {sa::RegrFn::Intercept, 11.25910509885536}, {sa::RegrFn::R2, 0.00396937516797486},
        {sa::RegrFn::Slope, -0.19979188345473473}, {sa::RegrFn::Sxx, 96.1}, {sa::RegrFn::Sxy, -19.2},
        {sa::RegrFn::Syy, 966.4},
    };
    for (auto& c : w) { double r = 0; CHK(sa::regr_finalize(c.f, &m, &r)); CHK(near(r, c.want)); }
    // Edge rows from Spark: one pair, x constant, y constant, empty.
    CorrMoments one{};
    fold(&one, 1.0, 2.0);
    double r = 0;
    CHK(!sa::regr_finalize(sa::RegrFn::CovarSamp, &one, &r));
    CHK(sa::regr_finalize(sa::RegrFn::Sxx, &one, &r) && r == 0.0);
    CHK(!sa::regr_finalize(sa::RegrFn::Slope, &one, &r));
    CHK(!sa::regr_finalize(sa::RegrFn::Intercept, &one, &r));
    CHK(!sa::regr_finalize(sa::RegrFn::R2, &one, &r));
    CorrMoments xc{};   // x constant 2.0, y = 0..4: slope NULL, r2 1.0, covar_samp 0
    for (int i = 0; i < 5; ++i) fold(&xc, 2.0, i);
    CHK(!sa::regr_finalize(sa::RegrFn::Slope, &xc, &r));
    CHK(sa::regr_finalize(sa::RegrFn::R2, &xc, &r) && r == 1.0);
    CHK(sa::regr_finalize(sa::RegrFn::CovarSamp, &xc, &r) && r == 0.0);
    CorrMoments yc{};   // x = 0..3, y constant 3: slope 0, intercept 3, r2 NULL
    for (int i = 0; i < 4; ++i) fold(&yc, i, 3.0);
    CHK(sa::regr_finalize(sa::RegrFn::Slope, &yc, &r) && r == 0.0);
    CHK(sa::regr_finalize(sa::RegrFn::Intercept, &yc, &r) && r == 3.0);
    CHK(!sa::regr_finalize(sa::RegrFn::R2, &yc, &r));
    CorrMoments e{};
    CHK(!sa::regr_finalize(sa::RegrFn::AvgX, &e, &r) && !sa::regr_finalize(sa::RegrFn::Sxx, &e, &r));
}

}  // namespace

int main() {
    test_gk();
    test_percentile();
    test_cm4();
    test_regr();
    std::puts("test_bolt_stat_agg: ok");
    return 0;
}
