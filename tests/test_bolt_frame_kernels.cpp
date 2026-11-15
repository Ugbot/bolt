// test_bolt_frame_kernels.cpp: run offsets, frame index columns and the frame column kernels against naive
// references, over random data with NULLs, empty frames, single rows and int64 extremes (G2CHK-1182).

#include <gtest/gtest.h>

#include "bolt/kernels/bolt_frame_agg.h"
#include "bolt/kernels/bolt_frame_bounds.h"
#include "bolt/kernels/bolt_run_bounds.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

using namespace bolt::kernels;
using bolt::StringView;

namespace {

// ---- run offsets ------------------------------------------------------------------------------------------

std::size_t U(std::int64_t x) { return static_cast<std::size_t>(x); }

std::vector<std::uint32_t> naive_starts(const std::vector<std::vector<std::uint8_t>>& keys, std::uint32_t w,
                                        const std::vector<std::uint8_t>& valid, const std::vector<std::int32_t>& sel,
                                        std::int64_t n, bool is_f64) {
    std::vector<std::uint32_t> out;
    for (std::int64_t i = 0; i < n; ++i) {
        const std::size_t p = sel.empty() ? U(i) : U(sel[U(i)]);
        bool start = i == 0;
        if (i > 0) {
            const std::size_t q = sel.empty() ? U(i - 1) : U(sel[U(i - 1)]);
            const bool vp = valid.empty() || valid[p] != 0, vq = valid.empty() || valid[q] != 0;
            if (vp != vq) start = true;
            else if (vp) {
                bool same;
                if (is_f64) {
                    double a, b;
                    std::memcpy(&a, keys[p].data(), 8); std::memcpy(&b, keys[q].data(), 8);
                    same = a == b || (std::isnan(a) && std::isnan(b));
                } else {
                    same = std::memcmp(keys[p].data(), keys[q].data(), w) == 0;
                }
                start = !same;
            }
        }
        if (start) out.push_back(static_cast<std::uint32_t>(i));
    }
    return out;
}

TEST(RunBounds, FixedAndFloatKeysWithNullsAndSelection) {
    std::mt19937_64 g(7);
    for (int round = 0; round < 400; ++round) {
        const std::size_t rows = 1 + static_cast<std::size_t>(g() % 70);
        const bool use_sel = round % 3 == 0;
        const bool use_valid = round % 2 == 0;
        const bool f64 = round % 5 == 0;
        std::vector<std::vector<std::uint8_t>> cells(rows, std::vector<std::uint8_t>(8));
        std::vector<std::uint8_t> valid_bytes(rows, 1), bits((rows + 7) / 8 + 1, 0);
        std::vector<double> dcol(rows);
        std::vector<std::int64_t> icol(rows);
        for (std::size_t i = 0; i < rows; ++i) {
            const std::uint64_t r = g() % 4;
            if (f64) {
                const double d = r == 0 ? std::nan("") : (r == 1 ? 0.0 : (r == 2 ? -0.0 : 1.5));
                dcol[i] = d;
                std::memcpy(cells[i].data(), &d, 8);
            } else {
                icol[i] = static_cast<std::int64_t>(r);
                std::memcpy(cells[i].data(), &icol[i], 8);
            }
            valid_bytes[i] = use_valid ? static_cast<std::uint8_t>(g() % 4 != 0) : 1;
            if (valid_bytes[i] != 0) bits[i >> 3] = static_cast<std::uint8_t>(bits[i >> 3] | (1u << (i & 7)));
        }
        std::vector<std::int32_t> sel;
        std::int64_t n = static_cast<std::int64_t>(rows);
        if (use_sel) {
            for (std::size_t i = 0; i < rows; ++i) if (g() % 3 != 0) sel.push_back(static_cast<std::int32_t>(i));
            n = static_cast<std::int64_t>(sel.size());
            if (n == 0) continue;
        }
        run_bounds::KeyCol k{};
        k.data = f64 ? static_cast<const void*>(dcol.data()) : static_cast<const void*>(icol.data());
        k.validity = use_valid ? bits.data() : nullptr;
        k.cell_w = 8;
        k.kind = f64 ? run_bounds::KeyKind::F64 : run_bounds::KeyKind::Bytes;
        std::vector<std::uint8_t> mask(U(n), 0);
        run_bounds::adjacent_ne(k, use_sel ? sel.data() : nullptr, n, mask.data());
        std::vector<std::uint32_t> offs(U(n));
        const std::int64_t c = run_bounds::mask_starts(mask.data(), n, offs.data(), n);
        const auto want = naive_starts(cells, 8, use_valid ? valid_bytes : std::vector<std::uint8_t>{}, sel, n, f64);
        ASSERT_EQ(U(c), want.size());
        for (std::size_t i = 0; i < U(c); ++i) ASSERT_EQ(offs[i], want[i]);
    }
}

TEST(RunBounds, Utf8AndCombinedColumns) {
    const char* long_a = "aaaaaaaaaaaaaaaaaaaa";   // 20 bytes: not inline
    std::vector<char> overflow(64, 0);
    std::memcpy(overflow.data(), long_a, 20);
    std::memcpy(overflow.data() + 24, long_a, 20);
    auto mk = [&](const char* s, std::uint32_t len, std::uint32_t off) {
        StringView sv;
        std::memset(&sv, 0, sizeof(sv));
        sv.length = len;
        if (len <= 12) { std::memcpy(sv.prefix, s, len < 4 ? len : 4); if (len > 4) std::memcpy(sv.inline_data, s + 4, len - 4); }
        else { std::memcpy(sv.prefix, s, 4); sv.ref.buf_idx = 0; sv.ref.offset = off; }
        return sv;
    };
    std::vector<StringView> sv = {mk("ab", 2, 0), mk("ab", 2, 0), mk(long_a, 20, 0), mk(long_a, 20, 24), mk("", 0, 0), mk("", 0, 0), mk("abc", 3, 0)};
    std::vector<std::int32_t> second = {1, 1, 1, 2, 2, 2, 2};
    run_bounds::KeyCol a{sv.data(), nullptr, 0, 16, run_bounds::KeyKind::Utf8, overflow.data()};
    run_bounds::KeyCol b{second.data(), nullptr, 0, 4, run_bounds::KeyKind::Bytes, nullptr};
    std::vector<std::uint8_t> mask(sv.size(), 0);
    run_bounds::adjacent_ne(a, nullptr, static_cast<std::int64_t>(sv.size()), mask.data());
    run_bounds::adjacent_ne(b, nullptr, static_cast<std::int64_t>(sv.size()), mask.data());
    std::uint32_t offs[8];
    const std::int64_t c = run_bounds::mask_starts(mask.data(), static_cast<std::int64_t>(sv.size()), offs, 8);
    // rows: ab,ab | long,long (second col changes at 3) | "" ,"" | abc
    const std::vector<std::uint32_t> want = {0, 2, 3, 4, 6};
    ASSERT_EQ(static_cast<std::size_t>(c), want.size());
    for (std::size_t i = 0; i < want.size(); ++i) EXPECT_EQ(offs[i], want[i]);
    EXPECT_EQ(run_bounds::mask_starts(mask.data(), static_cast<std::int64_t>(sv.size()), offs, 2), -1);
}

// ---- frame bounds ----------------------------------------------------------------------------------------

struct NaiveBounds { std::uint32_t lo, hi; };

NaiveBounds naive_bounds(const std::vector<std::int64_t>& ts, const frame::FrameSpec& s, std::int64_t t) {
    const __int128 e = static_cast<__int128>(s.has_at != 0 ? s.at : t) - s.offset;
    const bool trailing = s.bucket == 0;
    const bool unb = s.range == INT64_MAX;
    const __int128 lo = trailing ? (unb ? -(static_cast<__int128>(1) << 100) : e - s.range) : e;
    const __int128 hi = trailing ? e : (unb ? (static_cast<__int128>(1) << 100) : e + s.range);
    std::uint32_t l = 0, h = 0;
    for (std::size_t i = 0; i < ts.size(); ++i) {
        const __int128 v = ts[i];
        const bool below = s.lo_open != 0 ? v <= lo : v < lo;
        if (below) ++l;
        const bool inside = s.hi_open != 0 ? v < hi : v <= hi;
        if (inside) ++h;
    }
    if (h < l) h = l;
    return {l, h};
}

TEST(FrameBounds, TimeAgainstNaive) {
    std::mt19937_64 g(11);
    const std::int64_t ext[] = {INT64_MIN, INT64_MIN + 1, -5, 0, 7, INT64_MAX - 1, INT64_MAX};
    for (int round = 0; round < 3000; ++round) {
        const std::uint32_t n = static_cast<std::uint32_t>(g() % 40);
        std::vector<std::int64_t> ts(n);
        for (auto& v : ts) v = (g() % 5 == 0) ? ext[g() % 7] : static_cast<std::int64_t>(g() % 60) - 10;
        std::sort(ts.begin(), ts.end());
        frame::FrameSpec s{};
        const std::int64_t ranges[] = {0, 1, 5, 30, INT64_MAX, INT64_MAX - 3};
        s.range = ranges[g() % 6];
        const std::int64_t offs[] = {0, 3, -3, INT64_MAX, INT64_MIN + 1};
        s.offset = offs[g() % 5];
        s.has_at = g() % 4 == 0;
        s.at = static_cast<std::int64_t>(g() % 50) - 5;
        s.bucket = g() % 2;
        s.lo_open = s.range == 0 ? 0 : g() % 2;
        s.hi_open = s.range == 0 ? 0 : g() % 2;
        const std::uint32_t m = static_cast<std::uint32_t>(g() % 20);
        std::vector<std::int64_t> pts(m);
        for (auto& v : pts) v = static_cast<std::int64_t>(g() % 70) - 15;
        std::sort(pts.begin(), pts.end());
        std::vector<std::uint32_t> lo(m), hi(m);
        frame::frame_bounds_time_points(ts.data(), n, s, pts.data(), m, lo.data(), hi.data());
        for (std::uint32_t i = 0; i < m; ++i) {
            const NaiveBounds w = naive_bounds(ts, s, pts[i]);
            ASSERT_EQ(lo[i], w.lo) << "round " << round << " point " << i;
            ASSERT_EQ(hi[i], w.hi) << "round " << round << " point " << i;
        }
        for (std::uint32_t i = 0; i < m; ++i) ASSERT_EQ(frame::frame_first_row(ts.data(), n, s, pts[i]), lo[i]) << "first_row round " << round;
        const std::int64_t step = 1 + static_cast<std::int64_t>(g() % 6), start = static_cast<std::int64_t>(g() % 20) - 10;
        const std::int64_t g0 = static_cast<std::int64_t>(g() % 4);
        frame::frame_bounds_time_grid(ts.data(), n, s, start, step, g0, m, lo.data(), hi.data());
        for (std::uint32_t i = 0; i < m; ++i) {
            const std::int64_t t = start + (g0 + i) * step;
            const NaiveBounds w = naive_bounds(ts, s, t);
            ASSERT_EQ(lo[i], w.lo) << "grid round " << round << " point " << i;
            ASSERT_EQ(hi[i], w.hi) << "grid round " << round << " point " << i;
        }
    }
}

TEST(FrameBounds, RowsFrames) {
    std::vector<std::uint32_t> lo(10), hi(10);
    frame::frame_bounds_rows(3, 6, 4, 0, lo.data(), hi.data());
    const std::uint32_t wl[] = {0, 1, 2, 3, 4, 5}, wh[] = {4, 5, 6, 7, 8, 9};
    for (std::size_t i = 0; i < 6; ++i) { EXPECT_EQ(lo[i], wl[i]); EXPECT_EQ(hi[i], wh[i]); }
    frame::frame_bounds_rows(5, 3, 4, 4, lo.data(), hi.data());   // floor clamps
    EXPECT_EQ(lo[0], 4u); EXPECT_EQ(hi[0], 6u);
    EXPECT_EQ(lo[2], 4u); EXPECT_EQ(hi[2], 8u);
    frame::frame_bounds_rows(2, 3, INT64_MAX, 1, lo.data(), hi.data());
    EXPECT_EQ(lo[0], 1u); EXPECT_EQ(lo[2], 1u); EXPECT_EQ(hi[2], 5u);
    frame::frame_bounds_rows(0, 0, 1, 0, lo.data(), hi.data());
}

// ---- frame kernels ---------------------------------------------------------------------------------------

struct Frames {
    std::vector<std::uint32_t> lo, hi;
};

Frames random_frames(std::mt19937_64& g, std::uint32_t n, std::uint32_t m) {
    Frames f;
    std::uint32_t l = 0, h = 0;
    for (std::uint32_t i = 0; i < m; ++i) {
        l = std::min(n, l + static_cast<std::uint32_t>(g() % 4));
        h = std::max(h, l);
        h = std::min(n, h + static_cast<std::uint32_t>(g() % 5));
        if (g() % 6 == 0) h = l;   // an empty frame (hi may then fall back to l)
        f.lo.push_back(l);
        f.hi.push_back(std::max(l, h));
        h = std::max(h, l);
    }
    for (std::uint32_t i = 1; i < m; ++i) f.hi[i] = std::max(f.hi[i], f.hi[i - 1]);   // non-decreasing, as the contract says
    return f;
}

TEST(FrameAgg, CountSumGatherDeque) {
    std::mt19937_64 g(23);
    for (int round = 0; round < 1500; ++round) {
        const std::uint32_t n = static_cast<std::uint32_t>(g() % 50), m = 1 + static_cast<std::uint32_t>(g() % 30);
        const Frames fr = random_frames(g, n, m);
        std::vector<std::int64_t> v(n);
        std::vector<std::uint8_t> valid(n);
        const bool has_nulls = round % 2 == 0;
        for (std::uint32_t i = 0; i < n; ++i) {
            const std::int64_t ext[] = {INT64_MAX, INT64_MIN, 0, 5, -5};
            v[i] = (g() % 7 == 0) ? ext[g() % 5] : static_cast<std::int64_t>(g() % 11) - 5;
            valid[i] = has_nulls ? static_cast<std::uint8_t>(g() % 3 != 0) : 1;
        }
        const std::uint8_t* vp = has_nulls ? valid.data() : nullptr;
        std::uint32_t b, e;
        frame::frame_span(fr.lo.data(), fr.hi.data(), m, &b, &e);
        const std::uint32_t len = e - b;
        std::vector<std::uint32_t> pc(len + 1), pv(len + 1), nv(len + 1), dq(len + 1);
        std::vector<__int128> ps(len + 1);
        frame::valid_prefix_count(vp != nullptr ? vp + b : nullptr, len, pc.data());
        std::vector<std::int64_t> cnt(m), sum(m);
        frame::frame_count_valid(pc.data(), b, fr.lo.data(), fr.hi.data(), m, cnt.data());
        std::vector<std::uint8_t> pres(m);
        bool fail = false;
        frame::frame_sum_int<std::int64_t>(v.data() + b, vp != nullptr ? vp + b : nullptr, len, ps.data(), pc.data(), b,
                                           fr.lo.data(), fr.hi.data(), m, sum.data(), pres.data(), &fail);
        bool want_fail = false;
        for (std::uint32_t i = 0; i < m; ++i) {
            __int128 s = 0;
            std::int64_t c = 0;
            for (std::uint32_t r = fr.lo[i]; r < fr.hi[i]; ++r) if (valid[r] != 0) { s += v[r]; ++c; }
            ASSERT_EQ(cnt[i], c);
            ASSERT_EQ(pres[i] != 0, c != 0);
            if (c != 0 && (s > INT64_MAX || s < INT64_MIN)) want_fail = true;
            else if (c != 0) ASSERT_EQ(sum[i], static_cast<std::int64_t>(s));
        }
        ASSERT_EQ(fail, want_fail);
        // first / last valid
        std::vector<std::uint32_t> pk(m);
        std::vector<std::uint8_t> pr(m);
        if (has_nulls) { frame::last_valid_scan(vp + b, len, pv.data()); frame::first_valid_scan(vp + b, len, nv.data()); }
        frame::frame_last_valid(has_nulls ? pv.data() : nullptr, b, fr.lo.data(), fr.hi.data(), m, pk.data(), pr.data());
        for (std::uint32_t i = 0; i < m; ++i) {
            std::int64_t want = -1;
            for (std::uint32_t r = fr.lo[i]; r < fr.hi[i]; ++r) if (valid[r] != 0) want = r;
            ASSERT_EQ(pr[i] != 0, want >= 0);
            if (want >= 0) ASSERT_EQ(pk[i], static_cast<std::uint32_t>(want));
        }
        frame::frame_first_valid(has_nulls ? nv.data() : nullptr, b, fr.lo.data(), fr.hi.data(), m, pk.data(), pr.data());
        for (std::uint32_t i = 0; i < m; ++i) {
            std::int64_t want = -1;
            for (std::uint32_t r = fr.lo[i]; r < fr.hi[i] && want < 0; ++r) if (valid[r] != 0) want = r;
            ASSERT_EQ(pr[i] != 0, want >= 0);
            if (want >= 0) ASSERT_EQ(pk[i], static_cast<std::uint32_t>(want));
        }
        frame::frame_edge(vp != nullptr ? vp + b : nullptr, b, fr.lo.data(), fr.hi.data(), m, true, pk.data(), pr.data());
        for (std::uint32_t i = 0; i < m; ++i) {
            const bool any = fr.lo[i] < fr.hi[i];
            ASSERT_EQ(pr[i] != 0, any && valid[fr.hi[i] - 1] != 0);
        }
        // min / max: the first of equal wins
        for (int mx = 0; mx < 2; ++mx) {
            frame::frame_extreme_pick(vp != nullptr ? vp + b : nullptr, b, fr.lo.data(), fr.hi.data(), m, mx != 0,
                                      [&](std::uint32_t x, std::uint32_t y) { return v[x] < v[y]; }, dq.data(), pk.data(), pr.data());
            for (std::uint32_t i = 0; i < m; ++i) {
                std::int64_t at = -1;
                for (std::uint32_t r = fr.lo[i]; r < fr.hi[i]; ++r) {
                    if (valid[r] == 0) continue;
                    if (at < 0 || (mx != 0 ? v[static_cast<std::size_t>(at)] < v[r] : v[r] < v[static_cast<std::size_t>(at)])) at = r;
                }
                ASSERT_EQ(pr[i] != 0, at >= 0) << "round " << round;
                if (at >= 0) ASSERT_EQ(pk[i], static_cast<std::uint32_t>(at)) << "round " << round << " max " << mx;
            }
        }
    }
}

TEST(FrameAgg, ValidityAndCastAndFill) {
    std::mt19937_64 g(5);
    for (int round = 0; round < 200; ++round) {
        const std::uint32_t n = 1 + static_cast<std::uint32_t>(g() % 70);
        const std::int64_t off = static_cast<std::int64_t>(g() % 13);
        std::vector<std::uint8_t> bytes(n), bits(U((n + off + 7) / 8 + 2), 0xA5);
        for (auto& b : bytes) b = g() % 3 != 0;
        const std::uint32_t nulls = frame::pack_validity(bytes.data(), n, bits.data(), off);
        std::uint32_t want_nulls = 0;
        for (std::uint32_t i = 0; i < n; ++i) {
            ASSERT_EQ(((bits[U((off + i) >> 3)] >> ((off + i) & 7)) & 1u) != 0, bytes[i] != 0);
            want_nulls += bytes[i] == 0;
        }
        ASSERT_EQ(nulls, want_nulls);
        std::vector<std::uint8_t> back(n);
        frame::unpack_validity(bits.data(), off, nullptr, 0, 0, back.data());
        std::vector<std::uint8_t> shifted(n);
        frame::unpack_validity(bits.data(), off, nullptr, 0, n, shifted.data());
        ASSERT_EQ(shifted, bytes);
        std::vector<std::uint8_t> ones(bits.size(), 0);
        frame::set_valid_range(ones.data(), off, n);
        for (std::uint32_t i = 0; i < n; ++i) ASSERT_TRUE(((ones[U((off + i) >> 3)] >> ((off + i) & 7)) & 1u) != 0);
        if (off > 0) ASSERT_TRUE(((ones[U((off - 1) >> 3)] >> ((off - 1) & 7)) & 1u) == 0);
    }
    const std::int32_t i32[] = {1, -2, 3};
    const std::uint8_t val[] = {1, 0, 1};
    double out[3];
    ASSERT_TRUE(frame::cast_cells_f64(bolt::BoltType::Int32, reinterpret_cast<const std::uint8_t*>(i32), val, 3, out));
    EXPECT_EQ(out[0], 1.0); EXPECT_TRUE(std::isnan(out[1])); EXPECT_EQ(out[2], 3.0);
    EXPECT_FALSE(frame::cast_cells_f64(bolt::BoltType::Utf8, nullptr, nullptr, 0, out));

    // FILL
    const std::uint32_t lo[] = {0, 0, 2, 2, 3}, hi[] = {2, 0, 3, 2, 3};   // points 1, 3, 4 are empty
    double cells[5] = {1.5, 0, 2.5, 0, 0};
    std::uint8_t valid[5] = {1, 0, 1, 0, 0};
    frame::FillPrev prev{};
    frame::frame_fill(frame::FillMode::Prev, 8, reinterpret_cast<std::uint8_t*>(cells), valid, lo, hi, 5, &prev, nullptr);
    EXPECT_EQ(cells[1], 1.5); EXPECT_EQ(cells[3], 2.5); EXPECT_EQ(cells[4], 2.5);
    EXPECT_TRUE(valid[1] && valid[3] && valid[4]);
    double c2[5] = {1.5, 0, 2.5, 0, 0};
    std::uint8_t v2[5] = {1, 0, 1, 0, 0};
    frame::FillPrev prev2{};
    const double k = 9.0;
    frame::frame_fill(frame::FillMode::Const, 8, reinterpret_cast<std::uint8_t*>(c2), v2, lo, hi, 5, &prev2, reinterpret_cast<const std::uint8_t*>(&k));
    EXPECT_EQ(c2[1], 9.0); EXPECT_TRUE(v2[1] && v2[4]); EXPECT_EQ(c2[0], 1.5);
    std::uint8_t v3[5] = {1, 1, 1, 1, 1};
    double c3[5] = {1, 2, 3, 4, 5};
    frame::FillPrev prev3{};
    frame::frame_fill(frame::FillMode::Null, 8, reinterpret_cast<std::uint8_t*>(c3), v3, lo, hi, 5, &prev3, nullptr);
    EXPECT_TRUE(v3[0] && !v3[1] && v3[2] && !v3[3] && !v3[4]);
    // a leading empty frame with Prev has nothing to copy
    const std::uint32_t lo4[] = {0, 0}, hi4[] = {0, 1};
    double c4[2] = {0, 7};
    std::uint8_t v4[2] = {0, 1};
    frame::FillPrev prev4{};
    frame::frame_fill(frame::FillMode::Prev, 8, reinterpret_cast<std::uint8_t*>(c4), v4, lo4, hi4, 2, &prev4, nullptr);
    EXPECT_FALSE(v4[0]); EXPECT_TRUE(v4[1]);
}

TEST(FrameAgg, GatherFixed) {
    const std::int32_t cells[] = {10, 20, 30, 40};
    const std::uint32_t idx[] = {3, 0, 2};
    const std::uint8_t present[] = {1, 0, 1};
    std::int32_t out[3] = {-1, -1, -1};
    frame::gather_fixed(reinterpret_cast<const std::uint8_t*>(cells), 4, idx, present, 3, reinterpret_cast<std::uint8_t*>(out));
    EXPECT_EQ(out[0], 40); EXPECT_EQ(out[1], -1); EXPECT_EQ(out[2], 30);
    const std::int64_t ts[] = {5, 6, 7};
    std::int64_t o2[3] = {0, 0, 0};
    frame::gather_i64(ts, idx + 1, present + 1, 2, o2);
    EXPECT_EQ(o2[0], 0); EXPECT_EQ(o2[1], 7);
}

}  // namespace
