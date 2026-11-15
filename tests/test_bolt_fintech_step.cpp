// test_bolt_fintech_step.cpp: the one-row entry points (fintech/step.h) give the
// array kernels' answers bit for bit, over random series with NaN, infinities,
// constant runs and a single row, for any split of the series into calls.

#include <gtest/gtest.h>

#include "bolt/kernels/fintech.h"
#include "bolt/kernels/fintech/step.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <vector>

using namespace bolt::fintech;

namespace {

constexpr uint32_t K = 64;   // ring capacity of the test states
constexpr size_t kRows = 300;

struct Series {
    std::vector<double> a, b, c;
};

Series make_series(uint32_t seed, size_t n) {
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> u(-5.0, 5.0);
    Series s;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const uint32_t mode = seed % 4;
    for (size_t i = 0; i < n; ++i) {
        double x = u(g) + 100.0, y = u(g) + 50.0, z = u(g) + 75.0;
        if (mode == 1) x = y = z = 7.25;
        if (mode == 2 && g() % 17 == 0) x = nan;
        if (mode == 3 && g() % 23 == 0) y = (g() & 1) ? inf : -inf;
        s.a.push_back(x); s.b.push_back(y); s.c.push_back(z);
    }
    return s;
}

// `arr(state, series, lo, hi, out)` runs rows [lo, hi) through the array kernel; `step(state, series, i, out)` one row.
template <class State, class Init, class Arr, class Step>
void check(const char* name, size_t outs, Init init, Arr arr, Step step) {
    for (uint32_t seed = 1; seed <= 16; ++seed) {
        const size_t len = seed == 5 ? 1 : kRows;
        const Series s = make_series(seed, len);
        auto a = std::make_unique<State>();
        auto b = std::make_unique<State>();
        std::memset(static_cast<void*>(a.get()), 0, sizeof(State));
        std::memset(static_cast<void*>(b.get()), 0, sizeof(State));
        init(a.get());
        init(b.get());
        std::vector<double> want(len * outs), got(want.size());
        std::mt19937_64 g(seed * 31);
        size_t lo = 0;
        while (lo < len) {   // any split: the array kernels resume from their state
            const size_t hi = std::min(len, lo + 1 + static_cast<size_t>(g() % 40));
            arr(a.get(), s, lo, hi, want.data() + lo * outs);
            lo = hi;
        }
        for (size_t i = 0; i < len; ++i) step(b.get(), s, i, got.data() + i * outs);
        ASSERT_EQ(0, std::memcmp(want.data(), got.data(), want.size() * sizeof(double))) << name << " seed " << seed;
    }
}

}  // namespace

TEST(FintechStep, Sma) {
    check<SMAState<K>>("sma", 1, [](auto* s) { s->init(0, 0, 9); },
        [](auto* s, const Series& q, size_t lo, size_t hi, double* o) { const int64_t N = static_cast<int64_t>(hi - lo); const double* in[1] = {q.a.data() + lo}; execute_sma<K>(s, in, N, o); },
        [](auto* s, const Series& q, size_t i, double* o) { o[0] = sma_step<K>(s, q.a[i]); });
}

TEST(FintechStep, EmaEwmaRsi) {
    check<EMAState>("ema", 1, [](auto* s) { s->desc.in_col = 0; s->ema.init(12); },
        [](auto* s, const Series& q, size_t lo, size_t hi, double* o) { const int64_t N = static_cast<int64_t>(hi - lo); const double* in[1] = {q.a.data() + lo}; execute_ema(s, in, N, o); },
        [](auto* s, const Series& q, size_t i, double* o) { o[0] = ema_step(s, q.a[i]); });
    check<EWMAState>("ewma", 1, [](auto* s) { s->desc.alpha = 0.3; s->ema.alpha = 0.3; },
        [](auto* s, const Series& q, size_t lo, size_t hi, double* o) { const int64_t N = static_cast<int64_t>(hi - lo); const double* in[1] = {q.a.data() + lo}; execute_ewma(s, in, N, o); },
        [](auto* s, const Series& q, size_t i, double* o) { o[0] = ewma_step(s, q.a[i]); });
    check<RSIState>("rsi", 1, [](auto* s) { s->avg_gain.alpha = s->avg_loss.alpha = 1.0 / 14.0; rsi_reset(s); },
        [](auto* s, const Series& q, size_t lo, size_t hi, double* o) { const int64_t N = static_cast<int64_t>(hi - lo); const double* in[1] = {q.a.data() + lo}; execute_rsi(s, in, N, o); },
        [](auto* s, const Series& q, size_t i, double* o) { o[0] = rsi_step(s, q.a[i]); });
}

TEST(FintechStep, BollingerAndRolling) {
    check<BollingerBandsState<K>>("bollinger", 3, [](auto* s) { s->window = 7; s->k = 2.0; s->ring_x.init(7); s->ring_xx.init(7); },
        [](auto* s, const Series& q, size_t lo, size_t hi, double* o) { const int64_t N = static_cast<int64_t>(hi - lo);
            const double* in[1] = {q.a.data() + lo};
            std::vector<double> u(hi - lo), m(hi - lo), l(hi - lo);
            execute_bollinger_bands<K>(s, in, N, u.data(), m.data(), l.data());
            for (size_t i = 0; i < hi - lo; ++i) { o[3 * i] = u[i]; o[3 * i + 1] = m[i]; o[3 * i + 2] = l[i]; }
        },
        [](auto* s, const Series& q, size_t i, double* o) { bollinger_step<K>(s, q.a[i], o); });
    check<RollingStdState<K>>("rolling_std", 1, [](auto* s) { s->init(0, 0, 8); },
        [](auto* s, const Series& q, size_t lo, size_t hi, double* o) { const int64_t N = static_cast<int64_t>(hi - lo); execute_rolling_std<K>(s, q.a.data() + lo, N, o); },
        [](auto* s, const Series& q, size_t i, double* o) { o[0] = rolling_std_step<K>(s, q.a[i]); });
    check<RollingMinState<K>>("rolling_min", 1, [](auto* s) { s->init(0, 0, 6); },
        [](auto* s, const Series& q, size_t lo, size_t hi, double* o) { const int64_t N = static_cast<int64_t>(hi - lo); const double* in[1] = {q.a.data() + lo}; execute_rolling_min<K>(s, in, N, o); },
        [](auto* s, const Series& q, size_t i, double* o) { o[0] = rolling_min_step<K>(s, q.a[i]); });
    check<RollingMaxState<K>>("rolling_max", 1, [](auto* s) { s->init(0, 0, 6); },
        [](auto* s, const Series& q, size_t lo, size_t hi, double* o) { const int64_t N = static_cast<int64_t>(hi - lo); const double* in[1] = {q.a.data() + lo}; execute_rolling_max<K>(s, in, N, o); },
        [](auto* s, const Series& q, size_t i, double* o) { o[0] = rolling_max_step<K>(s, q.a[i]); });
    check<RollingSkewState<K>>("rolling_skew", 1, [](auto* s) { s->init(0, 0, 9); },
        [](auto* s, const Series& q, size_t lo, size_t hi, double* o) { const int64_t N = static_cast<int64_t>(hi - lo); execute_rolling_skew<K>(s, q.a.data() + lo, N, o); },
        [](auto* s, const Series& q, size_t i, double* o) { o[0] = rolling_skew_step<K>(s, q.a[i]); });
    check<RollingKurtState<K>>("rolling_kurt", 1, [](auto* s) { s->init(0, 0, 9); },
        [](auto* s, const Series& q, size_t lo, size_t hi, double* o) { const int64_t N = static_cast<int64_t>(hi - lo); execute_rolling_kurt<K>(s, q.a.data() + lo, N, o); },
        [](auto* s, const Series& q, size_t i, double* o) { o[0] = rolling_kurt_step<K>(s, q.a[i]); });
    check<RollingZScoreState<K>>("rolling_zscore", 1, [](auto* s) { s->init(0, 0, 10); },
        [](auto* s, const Series& q, size_t lo, size_t hi, double* o) { const int64_t N = static_cast<int64_t>(hi - lo); execute_rolling_zscore<K>(s, q.a.data() + lo, N, o); },
        [](auto* s, const Series& q, size_t i, double* o) { o[0] = rolling_zscore_step<K>(s, q.a[i]); });
    check<RollingCorrelationState<K>>("rolling_corr", 1, [](auto* s) { s->init(0, 1, 0, 11); },
        [](auto* s, const Series& q, size_t lo, size_t hi, double* o) { const int64_t N = static_cast<int64_t>(hi - lo); execute_rolling_correlation<K>(s, q.a.data() + lo, q.b.data() + lo, N, o); },
        [](auto* s, const Series& q, size_t i, double* o) { o[0] = rolling_corr_step<K>(s, q.a[i], q.b[i]); });
}

TEST(FintechStep, RunningKernels) {
    check<DrawdownState>("drawdown", 1, [](auto* s) { s->desc.in_col = 0; },
        [](auto* s, const Series& q, size_t lo, size_t hi, double* o) { const int64_t N = static_cast<int64_t>(hi - lo); const double* in[1] = {q.a.data() + lo}; execute_drawdown(s, in, N, o); },
        [](auto* s, const Series& q, size_t i, double* o) { o[0] = drawdown_step(s, q.a[i]); });
    check<MaxDrawdownState>("max_drawdown", 1, [](auto* s) { s->desc.in_col = 0; },
        [](auto* s, const Series& q, size_t lo, size_t hi, double* o) { const int64_t N = static_cast<int64_t>(hi - lo); const double* in[1] = {q.a.data() + lo}; execute_max_drawdown(s, in, N, o); },
        [](auto* s, const Series& q, size_t i, double* o) { o[0] = max_drawdown_step(s, q.a[i]); });
    check<MACDState>("macd", 3, [](auto* s) { s->ema_fast.init(12); s->ema_slow.init(26); s->ema_signal.init(9); },
        [](auto* s, const Series& q, size_t lo, size_t hi, double* o) { const int64_t N = static_cast<int64_t>(hi - lo);
            const double* in[1] = {q.a.data() + lo};
            std::vector<double> l(hi - lo), g(hi - lo), h(hi - lo);
            execute_macd(s, in, N, l.data(), g.data(), h.data());
            for (size_t i = 0; i < hi - lo; ++i) { o[3 * i] = l[i]; o[3 * i + 1] = g[i]; o[3 * i + 2] = h[i]; }
        },
        [](auto* s, const Series& q, size_t i, double* o) { macd_step(s, q.a[i], o); });
    check<WelfordMeanVarState>("welford", 2, [](auto* s) { s->init(0, 0, 1); },
        [](auto* s, const Series& q, size_t lo, size_t hi, double* o) { const int64_t N = static_cast<int64_t>(hi - lo);
            std::vector<double> m(hi - lo), v(hi - lo);
            execute_welford_mean_var(s, q.a.data() + lo, N, m.data(), v.data());
            for (size_t i = 0; i < hi - lo; ++i) { o[2 * i] = m[i]; o[2 * i + 1] = v[i]; }
        },
        [](auto* s, const Series& q, size_t i, double* o) { welford_step(s, q.a[i], o); });
    check<CUSUMState>("cusum", 2, [](auto* s) { s->desc.in_col = 0; s->target = 100.0; s->threshold = 3.0; },
        [](auto* s, const Series& q, size_t lo, size_t hi, double* o) { const int64_t N = static_cast<int64_t>(hi - lo);
            const double* in[1] = {q.a.data() + lo};
            std::vector<double> p(hi - lo), n(hi - lo);
            execute_cusum(s, in, N, p.data(), n.data());
            for (size_t i = 0; i < hi - lo; ++i) { o[2 * i] = p[i]; o[2 * i + 1] = n[i]; }
        },
        [](auto* s, const Series& q, size_t i, double* o) { cusum_step(s, q.a[i], o); });
    check<RiskMetricsVolState>("risk_metrics_vol", 1, [](auto* s) { s->init(0, 0, 0.94); },
        [](auto* s, const Series& q, size_t lo, size_t hi, double* o) { const int64_t N = static_cast<int64_t>(hi - lo); execute_risk_metrics_vol(s, q.a.data() + lo, N, o); },
        [](auto* s, const Series& q, size_t i, double* o) { o[0] = risk_metrics_vol_step(s, q.a[i]); });
    check<EWCOVState>("ewcov", 1, [](auto* s) { s->desc.x_col = 0; s->desc.y_col = 1; s->desc.lambda = 0.94; },
        [](auto* s, const Series& q, size_t lo, size_t hi, double* o) { const int64_t N = static_cast<int64_t>(hi - lo); const double* in[2] = {q.a.data() + lo, q.b.data() + lo}; execute_ewcov(s, in, N, o); },
        [](auto* s, const Series& q, size_t i, double* o) { o[0] = ewcov_step(s, q.a[i], q.b[i]); });
}

TEST(FintechStep, ThreeColumnKernels) {
    check<ATRState<K>>("atr", 1, [](auto* s) { s->init(0, 1, 2, 3, 9); },
        [](auto* s, const Series& q, size_t lo, size_t hi, double* o) { const int64_t N = static_cast<int64_t>(hi - lo); const double* in[3] = {q.a.data() + lo, q.b.data() + lo, q.c.data() + lo}; execute_atr<K>(s, in, N, o); },
        [](auto* s, const Series& q, size_t i, double* o) { o[0] = atr_step<K>(s, q.a[i], q.b[i], q.c[i]); });
    check<StochasticOscState<K>>("stochastic_k", 1, [](auto* s) { s->init(0, 1, 2, 3, 7); },
        [](auto* s, const Series& q, size_t lo, size_t hi, double* o) { const int64_t N = static_cast<int64_t>(hi - lo); const double* in[3] = {q.a.data() + lo, q.b.data() + lo, q.c.data() + lo}; execute_stochastic_osc<K>(s, in, N, o); },
        [](auto* s, const Series& q, size_t i, double* o) { o[0] = stochastic_k_step<K>(s, q.a[i], q.b[i], q.c[i]); });
    check<DonchianChannelState<K>>("donchian", 2, [](auto* s) { s->init(0, 1, 2, 3, 8); },
        [](auto* s, const Series& q, size_t lo, size_t hi, double* o) { const int64_t N = static_cast<int64_t>(hi - lo);
            const double* in[2] = {q.a.data() + lo, q.b.data() + lo};
            std::vector<double> u(hi - lo), l(hi - lo);
            execute_donchian_channel<K>(s, in, N, u.data(), l.data());
            for (size_t i = 0; i < hi - lo; ++i) { o[2 * i] = u[i]; o[2 * i + 1] = l[i]; }
        },
        [](auto* s, const Series& q, size_t i, double* o) { donchian_step<K>(s, q.a[i], q.b[i], o); });
}

TEST(FintechStep, LogReturnMatchesTheArrayKernel) {
    for (uint32_t seed = 1; seed <= 16; ++seed) {
        const Series s = make_series(seed, seed == 5 ? 1 : kRows);
        const size_t n = s.a.size();
        std::vector<double> want(n), got(n);
        const double* in[1] = {s.a.data()};
        LogReturnsOpDesc d{};
        log_returns_kernel(in, static_cast<int64_t>(n), want.data(), d);
        LogReturnState st;
        st.init();
        for (size_t i = 0; i < n; ++i) got[i] = log_return_step(&st, s.a[i]);
        ASSERT_EQ(0, std::memcmp(want.data(), got.data(), n * sizeof(double))) << "seed " << seed;
    }
}
