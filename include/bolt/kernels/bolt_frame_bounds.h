// bolt/kernels/bolt_frame_bounds.h: frame index columns of an ordered series.
//
// Given the ts column of one series run (ascending, n rows) and m evaluation points (an arithmetic grid, or
// explicit times), write lo[m], hi[m]: the half-open row range [lo, hi) of each point's frame. One merge
// pass: the first point's pointers come from a binary search, the rest advance forward only.
//
//   e = (has_at ? at : t) - offset            (128-bit; a bound outside int64 is a fact about the frame)
//   Trailing: ts in (e - range, e]            Bucket: ts in [e, e + range)
//   lo_open / hi_open choose each end; range == INT64_MAX is unbounded on the trailing side (Trailing) or
//   the leading side (Bucket).
//
// Row-count frames: frame_bounds_rows gives [max(floor, r + 1 - N), r + 1) for the point on row r.
//
// Tiger Style: noexcept, no allocation, loops bounded by n + m.

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace kernels {
namespace frame {

struct FrameSpec {
    std::int64_t range;        // INT64_MAX = unbounded
    std::int64_t offset;
    std::int64_t at;
    std::uint8_t has_at;
    std::uint8_t bucket;       // 1 = [e, e + range), 0 = (e - range, e]
    std::uint8_t lo_open;
    std::uint8_t hi_open;
};

struct FrameBound {
    std::int64_t lo, hi;
    bool lo_inf, hi_inf;
    bool lo_above;   // the lower bound is past every int64 ts: no row qualifies
    bool hi_below;   // the upper bound is below every int64 ts: no row qualifies
};

BOLT_FORCE_INLINE FrameBound frame_bound(const FrameSpec& s, std::int64_t t) noexcept {
    FrameBound b{};
    const __int128 e = static_cast<__int128>(s.has_at != 0 ? s.at : t) - s.offset;
    const bool trailing = s.bucket == 0;
    const bool unbounded = s.range == INT64_MAX;
    const __int128 lo = trailing ? e - s.range : e;
    const __int128 hi = trailing ? e : e + s.range;
    b.lo_inf = (trailing && unbounded) || lo < INT64_MIN;
    b.hi_inf = (!trailing && unbounded) || hi > INT64_MAX;
    b.lo_above = !b.lo_inf && lo > INT64_MAX;
    b.hi_below = !b.hi_inf && hi < INT64_MIN;
    b.lo = (b.lo_inf || b.lo_above) ? 0 : static_cast<std::int64_t>(lo);
    b.hi = (b.hi_inf || b.hi_below) ? 0 : static_cast<std::int64_t>(hi);
    return b;
}

namespace detail {

struct GridPoints {
    std::int64_t start, step, g0;
    BOLT_FORCE_INLINE std::int64_t operator()(std::uint32_t i) const noexcept {
        return static_cast<std::int64_t>(static_cast<std::uint64_t>(start) +
                                         (static_cast<std::uint64_t>(g0) + i) * static_cast<std::uint64_t>(step));
    }
};

struct ListPoints {
    const std::int64_t* pt;
    BOLT_FORCE_INLINE std::int64_t operator()(std::uint32_t i) const noexcept { return pt[i]; }
};

BOLT_FORCE_INLINE bool below_lo(const FrameSpec& s, const FrameBound& b, std::int64_t t) noexcept {
    return s.lo_open != 0 ? t <= b.lo : t < b.lo;
}

BOLT_FORCE_INLINE bool in_hi(const FrameSpec& s, const FrameBound& b, std::int64_t t) noexcept {
    return b.hi_inf || (s.hi_open != 0 ? t < b.hi : t <= b.hi);
}

// First row of [0, n) not below the lower bound.
inline std::uint32_t search_lo(const std::int64_t* ts, std::uint32_t n, const FrameSpec& s, const FrameBound& b) noexcept {
    if (b.lo_above) return n;
    if (b.lo_inf) return 0;
    std::uint32_t l = 0, h = n;
    while (l < h) {
        const std::uint32_t mid = l + (h - l) / 2u;
        if (below_lo(s, b, ts[mid])) l = mid + 1u; else h = mid;
    }
    return l;
}

// First row of [from, n) past the upper bound.
inline std::uint32_t search_hi(const std::int64_t* ts, std::uint32_t n, std::uint32_t from, const FrameSpec& s,
                               const FrameBound& b) noexcept {
    if (b.hi_below) return from;
    if (b.hi_inf) return n;
    std::uint32_t l = from, h = n;
    while (l < h) {
        const std::uint32_t mid = l + (h - l) / 2u;
        if (in_hi(s, b, ts[mid])) l = mid + 1u; else h = mid;
    }
    return l;
}

template <class Pts>
inline void bounds_time(const std::int64_t* BOLT_RESTRICT ts, std::uint32_t n, const FrameSpec& s, Pts pts,
                        std::uint32_t m, std::uint32_t* BOLT_RESTRICT lo, std::uint32_t* BOLT_RESTRICT hi) noexcept {
    assert(ts != nullptr || n == 0);
    assert((lo != nullptr && hi != nullptr) || m == 0);
    std::uint32_t lp = 0, hp = 0;
    for (std::uint32_t i = 0; i < m; ++i) {
        const FrameBound b = frame_bound(s, pts(i));
        if (i == 0) {
            lp = search_lo(ts, n, s, b);
            hp = search_hi(ts, n, lp, s, b);
        } else {
            if (b.lo_above) lp = n;
            while (lp < n && !b.lo_inf && below_lo(s, b, ts[lp])) ++lp;
            if (hp < lp || b.hi_below) hp = lp;
            while (hp < n && !b.hi_below && in_hi(s, b, ts[hp])) ++hp;
        }
        lo[i] = lp;
        hi[i] = hp;
    }
}

}  // namespace detail

// First row of ts[0, n) not below the lower bound of the frame of time t (the row a frame starting at t reads
// first): the rows below it can leave the buffer. One binary search.
inline std::uint32_t frame_first_row(const std::int64_t* ts, std::uint32_t n, const FrameSpec& s, std::int64_t t) noexcept {
    assert(ts != nullptr || n == 0);
    return detail::search_lo(ts, n, s, frame_bound(s, t));
}

// Points t_i = start + (g0 + i) * step (wrapping unsigned arithmetic), i in [0, m).
inline void frame_bounds_time_grid(const std::int64_t* ts, std::uint32_t n, const FrameSpec& s, std::int64_t start,
                                   std::int64_t step, std::int64_t g0, std::uint32_t m, std::uint32_t* lo,
                                   std::uint32_t* hi) noexcept {
    detail::bounds_time(ts, n, s, detail::GridPoints{start, step, g0}, m, lo, hi);
}

// Points are explicit times, ascending (the rows' own ts for a per-row evaluation).
inline void frame_bounds_time_points(const std::int64_t* ts, std::uint32_t n, const FrameSpec& s,
                                     const std::int64_t* pt, std::uint32_t m, std::uint32_t* lo,
                                     std::uint32_t* hi) noexcept {
    detail::bounds_time(ts, n, s, detail::ListPoints{pt}, m, lo, hi);
}

// The point on row row0 + i has the frame [max(floor, row0 + i + 1 - range), row0 + i + 1); range INT64_MAX = floor.
inline void frame_bounds_rows(std::uint32_t row0, std::uint32_t m, std::int64_t range, std::uint32_t floor,
                              std::uint32_t* BOLT_RESTRICT lo, std::uint32_t* BOLT_RESTRICT hi) noexcept {
    assert(range >= 1);
    assert((lo != nullptr && hi != nullptr) || m == 0);
    for (std::uint32_t i = 0; i < m; ++i) {
        const std::int64_t end = static_cast<std::int64_t>(row0) + i + 1;
        const std::int64_t from = range == INT64_MAX ? 0 : end - range;
        const std::int64_t l = from < static_cast<std::int64_t>(floor) ? static_cast<std::int64_t>(floor) : from;
        lo[i] = static_cast<std::uint32_t>(l);
        hi[i] = static_cast<std::uint32_t>(end);
    }
}

}  // namespace frame
}  // namespace kernels
}  // namespace bolt
