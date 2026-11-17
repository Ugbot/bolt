// test_counter_rate.cpp: the counter-rate family with an optional start-timestamp column (G2CHK-1173):
// hand-worked cases, then the frames kernel against a naive per-frame reference over random series.

#include <gtest/gtest.h>

#include "bolt/kernels/counter_rate.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace bk = bolt::kernels;
namespace bc = bolt::kernels::counter;

namespace {

std::uint64_t bits(double d) { std::uint64_t b; std::memcpy(&b, &d, 8); return b; }

struct Rng {
    std::uint64_t s;
    std::uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    std::uint32_t below(std::uint32_t n) { return static_cast<std::uint32_t>(next() % n); }
};

// A naive reading of Prometheus's float path: walk the window's samples directly, no shared helpers.
double naive(bc::RateFn fn, const std::vector<std::int64_t>& t, const std::vector<double>& v,
             const std::vector<std::int64_t>& st, std::int64_t lo, std::int64_t hi, bool use_st) {
    std::vector<std::size_t> ix;
    for (std::size_t i = 0; i < t.size(); ++i) if (t[i] > lo && t[i] <= hi) ix.push_back(i);
    const double nosample = bolt::promql::promql_no_sample();
    auto reset = [&](std::size_t a, std::size_t b) {
        if (v[b] < v[a]) return true;
        if (!use_st) return false;
        const std::int64_t ps = st[a], cs = st[b], pt = t[a], ct = t[b];
        if (ps == cs || cs == 0 || cs >= ct) return false;
        if (cs < pt) return false;
        if (cs > pt) return true;
        if (ps > pt) return false;
        return ps != 0 && ps != pt;
    };
    if (ix.empty()) return nosample;
    const std::size_t n = ix.size();
    if (fn == bc::RateFn::Resets) {
        std::int64_t c = 0;
        for (std::size_t k = 1; k < n; ++k) c += reset(ix[k - 1], ix[k]) ? 1 : 0;
        return static_cast<double>(c);
    }
    if (fn == bc::RateFn::Irate || fn == bc::RateFn::Idelta) {
        if (n < 2) return nosample;
        const std::size_t a = ix[n - 2], b = ix[n - 1];
        const std::int64_t dt = t[b] - t[a];
        if (dt == 0) return nosample;
        const bool rate = fn == bc::RateFn::Irate;
        double r = (rate && reset(a, b)) ? v[b] : v[b] - v[a];
        if (rate) r /= static_cast<double>(dt) / 1000.0;
        return r;
    }
    const bool counter = fn != bc::RateFn::Delta, rate = fn == bc::RateFn::Rate;
    if (!use_st) {
        if (n < 2) return nosample;
        return bk::counter::rate_one(fn, t.data() + ix[0], v.data() + ix[0], nullptr, static_cast<std::int64_t>(n), lo, hi, 1000.0);
    }
    double result = v[ix[n - 1]] - v[ix[0]];
    if (counter) for (std::size_t k = 1; k < n; ++k) if (reset(ix[k - 1], ix[k])) result += v[ix[k - 1]];
    double d2s = static_cast<double>(t[ix[0]] - lo) / 1000.0;
    double d2e = static_cast<double>(hi - t[ix[n - 1]]) / 1000.0;
    double si = static_cast<double>(t[ix[n - 1]] - t[ix[0]]) / 1000.0;
    const double avg = n > 1 ? si / static_cast<double>(n - 1) : 0.0;
    const double th = avg * 1.1;
    const std::int64_t s0 = st[ix[0]];
    if (counter && s0 != 0 && s0 > lo && s0 < t[ix[0]]) {
        d2s = 0.0;
        si = static_cast<double>(t[ix[n - 1]] - s0) / 1000.0;
        result += v[ix[0]];
    } else if (n == 1) {
        return nosample;
    } else {
        if (d2s >= th) d2s = avg / 2.0;
        if (counter) {
            double z = d2s;
            if (result > 0.0 && v[ix[0]] >= 0.0) z = si * (v[ix[0]] / result);
            if (z < d2s) d2s = z;
        }
    }
    if (d2e >= th) d2e = avg / 2.0;
    double factor = 1.0;
    if (si != 0.0) factor = (si + d2s + d2e) / si;
    if (rate) { const double rs = static_cast<double>(hi - lo) / 1000.0; if (rs != 0.0) factor /= rs; }
    return result * factor;
}

}  // namespace

// The reset sum adds one term per sample whether or not it resets (G2CHK-1199); the answer must equal, bit for bit, the
// form that adds only on a reset: same summation order, and a signed zero (a -0.0 first sample, a -0.0 delta) survives.
double branching_reference(const std::vector<std::int64_t>& t, const std::vector<double>& v,
                           const std::vector<std::int64_t>& st, std::int64_t range_start, std::int64_t range_end) {
    const std::int64_t n = static_cast<std::int64_t>(t.size());
    double result = v[n - 1] - v[0];
    for (std::int64_t i = 1; i < n; ++i)
        if (v[i] < v[i - 1] || bc::st_is_reset(st[i - 1], t[i - 1], st[i], t[i])) result += v[i - 1];
    const double tps = 1000.0;
    double dts = static_cast<double>(t[0] - range_start) / tps;
    double dte = static_cast<double>(range_end - t[n - 1]) / tps;
    double sampled = static_cast<double>(t[n - 1] - t[0]) / tps;
    const double avg = sampled / static_cast<double>(n - 1);
    const double thr = avg * 1.1;
    if (st[0] != 0 && st[0] > range_start && st[0] < t[0]) {
        dts = 0.0; sampled = static_cast<double>(t[n - 1] - st[0]) / tps; result += v[0];
    } else {
        if (dts >= thr) dts = avg / 2.0;
        double dtz = dts;
        if (result > 0.0 && v[0] >= 0.0) dtz = sampled * (v[0] / result);
        if (dtz < dts) dts = dtz;
    }
    if (dte >= thr) dte = avg / 2.0;
    double factor = 1.0;
    if (sampled != 0.0) factor = (sampled + dts + dte) / sampled;
    factor /= static_cast<double>(range_end - range_start) / tps;
    return result * factor;
}

TEST(CounterRate, ResetSumIsBitIdenticalToTheBranchingForm) {
    Rng rng{0x9e3779b97f4a7c15ull};
    const double pool[] = {0.0, -0.0, 1.0, 2.5, 7.0, 1e-300, 1e300, 0.1, 0.2, 0.3, 123456789.125, 3.0, 3.0, -0.0};
    for (int round = 0; round < 4000; ++round) {
        const std::uint32_t n = 2 + rng.below(40);
        std::vector<std::int64_t> t(n), st(n);
        std::vector<double> v(n);
        std::int64_t at = 1000;
        for (std::uint32_t i = 0; i < n; ++i) {
            at += 1 + rng.below(30000);
            t[i] = at;
            v[i] = round % 3 == 0 ? pool[rng.below(sizeof(pool) / sizeof(pool[0]))] : static_cast<double>(rng.below(1000)) * 0.37;
            st[i] = (round % 5 == 0 && rng.below(4) == 0) ? static_cast<std::int64_t>(t[i] - rng.below(50000)) : 0;
        }
        const std::int64_t lo = t[0] - 1000, hi = t[n - 1] + 1000;
        const double got = bc::st_extrapolated_rate(t.data(), v.data(), st.data(), n, lo, hi, true, true, 1000.0);
        if (st[0] == 0 || !(st[0] > lo && st[0] < t[0])) {
            if (n >= 2 && t[n - 1] > t[0]) EXPECT_EQ(bits(got), bits(branching_reference(t, v, st, lo, hi))) << "round " << round;
        }
    }
}

TEST(CounterRate, StartInsideTheFrameExtrapolatesToTheStart) {
    // Samples at 10s, 20s, 30s in a (0, 30s] frame; the first declares a start at 4s: a zero sample sits there.
    const std::int64_t ts[3] = {10000, 20000, 30000};
    const double v[3] = {5.0, 15.0, 25.0};
    const std::int64_t st[3] = {4000, 4000, 4000};
    // result = 20 + 5 (the zero-at-start sample); interval = (30 - 4) s = 26; to_end = 0: factor = 26 / 26 / 30.
    const double r = bc::st_extrapolated_rate(ts, v, st, 3, 0, 30000, true, true, 1000.0);
    EXPECT_EQ(bits(r), bits(25.0 * (26.0 / 26.0 / 30.0)));
}

TEST(CounterRate, StartMovingPastThePreviousSampleIsAReset) {
    EXPECT_TRUE(bc::st_is_reset(1000, 5000, 6000, 8000));    // start 6000 after prev sample 5000: restart
    EXPECT_FALSE(bc::st_is_reset(1000, 5000, 3000, 8000));   // start before prev sample: no
    EXPECT_FALSE(bc::st_is_reset(1000, 5000, 0, 8000));      // unknown start
    EXPECT_FALSE(bc::st_is_reset(1000, 5000, 1000, 8000));   // unchanged start
    EXPECT_FALSE(bc::st_is_reset(1000, 5000, 9000, 8000));   // start after the sample itself: invalid
    const std::int64_t ts[3] = {1000, 2000, 3000};
    const double v[3] = {1.0, 2.0, 3.0};
    const std::int64_t st[3] = {0, 0, 2500};
    EXPECT_EQ(bc::st_resets(ts, v, st, 3), 1);
    EXPECT_EQ(bc::st_resets(ts, v, st, 2), 0);
}

TEST(CounterRate, InstantValueUsesTheStartForResets) {
    const std::int64_t ts[2] = {1000, 3000};
    const double v[2] = {10.0, 12.0};
    const std::int64_t st[2] = {0, 2000};   // started between the samples: the delta is the value itself
    EXPECT_EQ(bc::st_instant_value(ts, v, st, 2, true, 1000.0), 12.0 / 2.0);
    EXPECT_EQ(bc::st_instant_value(ts, v, st, 2, false, 1000.0), 2.0);
    EXPECT_TRUE(bolt::promql::promql_is_no_sample(bc::st_instant_value(ts, v, st, 1, true, 1000.0)));
}

TEST(CounterRate, NoStartColumnIsThePlainFamily) {
    const std::int64_t ts[4] = {1000, 2000, 3000, 4000};
    const double v[4] = {1.0, 3.0, 2.0, 6.0};
    for (int f = 0; f <= 5; ++f) {
        const double a = bc::rate_one(static_cast<bc::RateFn>(f), ts, v, nullptr, 4, 0, 5000, 1000.0);
        double b;
        switch (f) {
            case 0: b = bolt::promql::promql_rate(ts, v, 4, 0, 5000, 1000.0); break;
            case 1: b = bolt::promql::promql_increase(ts, v, 4, 0, 5000, 1000.0); break;
            case 2: b = bolt::promql::promql_delta(ts, v, 4, 0, 5000, 1000.0); break;
            case 3: b = bolt::promql::promql_irate(ts, v, 4, 1000.0); break;
            case 4: b = bolt::promql::promql_idelta(ts, v, 4, 1000.0); break;
            default: b = static_cast<double>(bolt::promql::promql_resets(v, 4)); break;
        }
        EXPECT_EQ(bits(a), bits(b)) << f;
    }
}

TEST(CounterRate, FramesMatchTheNaiveReference) {
    Rng rng{0x9e3779b97f4a7c15ull};
    for (int round = 0; round < 300; ++round) {
        const std::uint32_t n = 1 + rng.below(40);
        std::vector<std::int64_t> t(n), st(n);
        std::vector<double> v(n);
        std::vector<std::uint8_t> valid(n);
        std::int64_t cur = 1000 + rng.below(500);
        double val = rng.below(50);
        for (std::uint32_t i = 0; i < n; ++i) {
            cur += 500 + rng.below(4000);
            t[i] = cur;
            val = rng.below(7) == 0 ? val * 0.3 : val + rng.below(30);
            v[i] = val;
            st[i] = rng.below(3) == 0 ? 0 : (rng.below(4) == 0 ? cur - static_cast<std::int64_t>(rng.below(9000)) : 500);
            if (st[i] < 0) st[i] = 0;
            valid[i] = rng.below(9) != 0 ? 1u : 0u;
        }
        std::vector<std::uint32_t> pc(n + 1, 0);
        std::vector<std::int64_t> ct, cst;
        std::vector<double> cv;
        for (std::uint32_t i = 0; i < n; ++i) {
            pc[i + 1] = pc[i] + valid[i];
            if (valid[i]) { ct.push_back(t[i]); cv.push_back(v[i]); cst.push_back(st[i]); }
        }
        // the naive reference only sees the valid samples
        std::vector<std::int64_t> nt = ct, nst = cst;
        std::vector<double> nv = cv;
        const std::uint32_t m = 1 + rng.below(8);
        std::vector<std::uint32_t> lo(m), hi(m);
        std::vector<std::int64_t> pt(m);
        const std::int64_t range = 3000 + rng.below(20000);
        for (std::uint32_t i = 0; i < m; ++i) {
            pt[i] = t[0] + static_cast<std::int64_t>(rng.below(static_cast<std::uint32_t>(t[n - 1] - t[0] + 1)));
            std::uint32_t a = 0, b = 0;
            while (a < n && t[a] <= pt[i] - range) ++a;
            b = a;
            while (b < n && t[b] <= pt[i]) ++b;
            lo[i] = a; hi[i] = b;
        }
        bk::frame::FrameSpec spec{};
        spec.range = range;
        for (int f = 0; f <= 5; ++f) {
            for (int use = 0; use < 2; ++use) {
                std::vector<double> out(m);
                std::vector<std::uint8_t> ok(m);
                bc::rate_frames(static_cast<bc::RateFn>(f), ct.empty() ? nullptr : ct.data(), cv.data(),
                                use ? cst.data() : nullptr, pc.data(), 0, lo.data(), hi.data(), pt.data(), m, spec,
                                false, 1000.0, out.data(), ok.data());
                for (std::uint32_t i = 0; i < m; ++i) {
                    // frames hold the valid samples of rows [lo, hi): the reference windows by time over the valid ones
                    std::int64_t wlo = pt[i] - range;
                    std::vector<std::int64_t> wt, wst;
                    std::vector<double> wv;
                    for (std::uint32_t r = lo[i]; r < hi[i]; ++r) if (valid[r]) { wt.push_back(t[r]); wv.push_back(v[r]); wst.push_back(st[r]); }
                    double want = wt.empty() ? bolt::promql::promql_no_sample()
                                             : naive(static_cast<bc::RateFn>(f), wt, wv, wst, wlo, pt[i], use != 0);
                    // the frame's evaluation bounds are the grid's (pt - range, pt]
                    const bool present = !bolt::promql::promql_is_no_sample(want);
                    ASSERT_EQ(ok[i] != 0, present) << "round " << round << " f " << f << " use " << use << " i " << i;
                    if (present) ASSERT_EQ(bits(out[i]), bits(want)) << "round " << round << " f " << f << " use " << use << " i " << i;
                }
            }
        }
    }
}
