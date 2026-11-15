// bolt/kernels/fintech/step.h: one-row entry points over the states the
// array kernels use, for engines that feed rows as they arrive (a window
// operator evaluating a frame point by point). Each step runs the array
// kernel for n = 1 on the same state, so the state and the result are the
// array kernel's, bit for bit (test_bolt_fintech_step).

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/fintech.h"

#include <cmath>
#include <cstdint>

namespace bolt {
namespace fintech {

// Largest ring window the step states are sized for.
inline constexpr uint32_t kStepWindow = 2048u;

// log_returns_kernel carries no state between calls; this one does.
struct LogReturnState {
    double prev;
    bool   have_prev;
    inline void init() noexcept { prev = 0.0; have_prev = false; }
};

inline double log_return_step(LogReturnState* s, double x) noexcept {
    assert(s != nullptr);
    const double out = s->have_prev ? std::log(x / s->prev) : 0.0;
    s->prev = x;
    s->have_prev = true;
    return out;
}

template <uint32_t K>
inline double sma_step(SMAState<K>* s, double x) noexcept {
    assert(s != nullptr);
    const double* BOLT_RESTRICT in[1] = {&x};
    double out = 0.0;
    execute_sma<K>(s, in, 1, &out);
    return out;
}

inline double ema_step(EMAState* s, double x) noexcept {
    assert(s != nullptr);
    const double* BOLT_RESTRICT in[1] = {&x};
    double out = 0.0;
    execute_ema(s, in, 1, &out);
    return out;
}

inline double ewma_step(EWMAState* s, double x) noexcept {
    assert(s != nullptr);
    const double* BOLT_RESTRICT in[1] = {&x};
    double out = 0.0;
    execute_ewma(s, in, 1, &out);
    return out;
}

inline double rsi_step(RSIState* s, double x) noexcept {
    assert(s != nullptr);
    const double* BOLT_RESTRICT in[1] = {&x};
    double out = 0.0;
    execute_rsi(s, in, 1, &out);
    return out;
}

// out[0..2] = upper, middle, lower.
template <uint32_t K>
inline void bollinger_step(BollingerBandsState<K>* s, double x, double* out) noexcept {
    assert(s != nullptr && out != nullptr);
    const double* BOLT_RESTRICT in[1] = {&x};
    execute_bollinger_bands<K>(s, in, 1, out, out + 1, out + 2);
}

template <uint32_t K>
inline double rolling_std_step(RollingStdState<K>* s, double x) noexcept {
    assert(s != nullptr);
    double out = 0.0;
    execute_rolling_std<K>(s, &x, 1, &out);
    return out;
}

template <uint32_t K>
inline double rolling_min_step(RollingMinState<K>* s, double x) noexcept {
    assert(s != nullptr);
    const double* BOLT_RESTRICT in[1] = {&x};
    double out = 0.0;
    execute_rolling_min<K>(s, in, 1, &out);
    return out;
}

template <uint32_t K>
inline double rolling_max_step(RollingMaxState<K>* s, double x) noexcept {
    assert(s != nullptr);
    const double* BOLT_RESTRICT in[1] = {&x};
    double out = 0.0;
    execute_rolling_max<K>(s, in, 1, &out);
    return out;
}

template <uint32_t K>
inline double rolling_skew_step(RollingSkewState<K>* s, double x) noexcept {
    assert(s != nullptr);
    double out = 0.0;
    execute_rolling_skew<K>(s, &x, 1, &out);
    return out;
}

template <uint32_t K>
inline double rolling_kurt_step(RollingKurtState<K>* s, double x) noexcept {
    assert(s != nullptr);
    double out = 0.0;
    execute_rolling_kurt<K>(s, &x, 1, &out);
    return out;
}

template <uint32_t K>
inline double rolling_zscore_step(RollingZScoreState<K>* s, double x) noexcept {
    assert(s != nullptr);
    double out = 0.0;
    execute_rolling_zscore<K>(s, &x, 1, &out);
    return out;
}

template <uint32_t K>
inline double rolling_corr_step(RollingCorrelationState<K>* s, double a, double b) noexcept {
    assert(s != nullptr);
    double out = 0.0;
    execute_rolling_correlation<K>(s, &a, &b, 1, &out);
    return out;
}

inline double drawdown_step(DrawdownState* s, double x) noexcept {
    assert(s != nullptr);
    const double* BOLT_RESTRICT in[1] = {&x};
    double out = 0.0;
    execute_drawdown(s, in, 1, &out);
    return out;
}

inline double max_drawdown_step(MaxDrawdownState* s, double x) noexcept {
    assert(s != nullptr);
    const double* BOLT_RESTRICT in[1] = {&x};
    double out = 0.0;
    execute_max_drawdown(s, in, 1, &out);
    return out;
}

// out[0..2] = line, signal, histogram.
inline void macd_step(MACDState* s, double x, double* out) noexcept {
    assert(s != nullptr && out != nullptr);
    const double* BOLT_RESTRICT in[1] = {&x};
    execute_macd(s, in, 1, out, out + 1, out + 2);
}

// out[0..1] = mean, variance.
inline void welford_step(WelfordMeanVarState* s, double x, double* out) noexcept {
    assert(s != nullptr && out != nullptr);
    execute_welford_mean_var(s, &x, 1, out, out + 1);
}

// out[0..1] = cum_pos, cum_neg.
inline void cusum_step(CUSUMState* s, double x, double* out) noexcept {
    assert(s != nullptr && out != nullptr);
    const double* BOLT_RESTRICT in[1] = {&x};
    execute_cusum(s, in, 1, out, out + 1);
}

inline double risk_metrics_vol_step(RiskMetricsVolState* s, double x) noexcept {
    assert(s != nullptr);
    double out = 0.0;
    execute_risk_metrics_vol(s, &x, 1, &out);
    return out;
}

inline double ewcov_step(EWCOVState* s, double x, double y) noexcept {
    assert(s != nullptr);
    const double* BOLT_RESTRICT in[2] = {&x, &y};
    double out = 0.0;
    execute_ewcov(s, in, 1, &out);
    return out;
}

template <uint32_t K>
inline double atr_step(ATRState<K>* s, double high, double low, double close) noexcept {
    assert(s != nullptr);
    const double* BOLT_RESTRICT in[3] = {&high, &low, &close};
    double out = 0.0;
    execute_atr<K>(s, in, 1, &out);
    return out;
}

template <uint32_t K>
inline double stochastic_k_step(StochasticOscState<K>* s, double close, double high, double low) noexcept {
    assert(s != nullptr);
    const double* BOLT_RESTRICT in[3] = {&close, &high, &low};
    double out = 0.0;
    execute_stochastic_osc<K>(s, in, 1, &out);
    return out;
}

// out[0..1] = upper, lower.
template <uint32_t K>
inline void donchian_step(DonchianChannelState<K>* s, double high, double low, double* out) noexcept {
    assert(s != nullptr && out != nullptr);
    const double* BOLT_RESTRICT in[2] = {&high, &low};
    execute_donchian_channel<K>(s, in, 1, out, out + 1);
}

}  // namespace fintech
}  // namespace bolt
