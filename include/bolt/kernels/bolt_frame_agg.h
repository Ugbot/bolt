// bolt/kernels/bolt_frame_agg.h: column kernels over frame index columns.
//
// A frame is the half-open row range [lo[i], hi[i]) of one series run, i in [0, m) (see bolt_frame_bounds.h).
// Both ends are non-decreasing in i. Every kernel here is ONE call over the m frames of a run chunk:
//   prefix / scan forms  : valid_prefix_count, frame_count_valid, frame_sum_int (exact 128-bit prefix sums)
//   gather forms         : last / first valid row of the frame (scan arrays), gather_fixed
//   deque form           : frame_extreme_pick (monotonic deque, O(rows + m))
//   column plumbing      : validity bytes <-> bits, cast to f64 with NaN under NULL, fill.
// `valid` is a byte mask (1 = non-null) or nullptr for all valid. Frames index rows of the arrays handed in:
// `base` is the row the arrays' element 0 stands for (a span view), so a chunk touches only the rows its
// frames cover.
//
// Tiger Style: noexcept, no allocation (caller scratch), loops bounded by rows + m.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/bolt_types.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace bolt {
namespace kernels {
namespace frame {

inline constexpr std::uint32_t kNoRow = 0xFFFFFFFFu;

// Smallest lo and largest hi of the frames (the rows a chunk touches); m == 0 gives an empty span.
inline void frame_span(const std::uint32_t* lo, const std::uint32_t* hi, std::uint32_t m, std::uint32_t* b,
                       std::uint32_t* e) noexcept {
    assert(b != nullptr && e != nullptr);
    if (m == 0) { *b = *e = 0; return; }
    std::uint32_t bb = lo[0], ee = hi[0];
    for (std::uint32_t i = 1; i < m; ++i) {
        bb = lo[i] < bb ? lo[i] : bb;
        ee = hi[i] > ee ? hi[i] : ee;
    }
    *b = bb;
    *e = ee < bb ? bb : ee;
}

// ---- validity plumbing ---------------------------------------------------------------------------------------

// bytes[i] = bit (phys(i) + bit_off) of an Arrow bitmap; phys(i) = sel ? sel[i] : first + i. nullptr bits: all 1.
inline void unpack_validity(const std::uint8_t* bits, std::int64_t bit_off, const std::int32_t* sel, std::int64_t first,
                            std::uint32_t n, std::uint8_t* BOLT_RESTRICT bytes) noexcept {
    assert(bytes != nullptr || n == 0);
    if (bits == nullptr) { if (n != 0) std::memset(bytes, 1, n); return; }
    for (std::uint32_t i = 0; i < n; ++i) {
        const std::int64_t p = (sel != nullptr ? sel[i] : first + i) + bit_off;
        bytes[i] = static_cast<std::uint8_t>((bits[p >> 3] >> (p & 7)) & 1u);
    }
}

// Set bits [off, off + n) of `bits` to bytes[0, n) (0/1); returns the number of cleared bits.
inline std::uint32_t pack_validity(const std::uint8_t* BOLT_RESTRICT bytes, std::uint32_t n, std::uint8_t* bits,
                                   std::int64_t off) noexcept {
    assert(bits != nullptr || n == 0);
    std::uint32_t nulls = 0;
    std::uint32_t i = 0;
    while (i < n && ((off + i) & 7) != 0) {
        const std::int64_t p = off + i;
        if (bytes[i] != 0) bits[p >> 3] = static_cast<std::uint8_t>(bits[p >> 3] | (1u << (p & 7)));
        else { bits[p >> 3] = static_cast<std::uint8_t>(bits[p >> 3] & ~(1u << (p & 7))); ++nulls; }
        ++i;
    }
    for (; i + 8 <= n; i += 8) {
        std::uint8_t w = 0;
        for (std::uint32_t k = 0; k < 8; ++k) w = static_cast<std::uint8_t>(w | ((bytes[i + k] != 0 ? 1u : 0u) << k));
        bits[(off + i) >> 3] = w;
        nulls += 8u - static_cast<std::uint32_t>(__builtin_popcount(w));
    }
    for (; i < n; ++i) {
        const std::int64_t p = off + i;
        if (bytes[i] != 0) bits[p >> 3] = static_cast<std::uint8_t>(bits[p >> 3] | (1u << (p & 7)));
        else { bits[p >> 3] = static_cast<std::uint8_t>(bits[p >> 3] & ~(1u << (p & 7))); ++nulls; }
    }
    return nulls;
}

// Set bits [off, off + n) to 1.
inline void set_valid_range(std::uint8_t* bits, std::int64_t off, std::uint32_t n) noexcept {
    assert(bits != nullptr || n == 0);
    std::uint32_t i = 0;
    while (i < n && ((off + i) & 7) != 0) { const std::int64_t p = off + i; bits[p >> 3] = static_cast<std::uint8_t>(bits[p >> 3] | (1u << (p & 7))); ++i; }
    if (i + 8 <= n) { const std::uint32_t bytes = (n - i) / 8u; std::memset(bits + ((off + i) >> 3), 0xFF, bytes); i += bytes * 8u; }
    for (; i < n; ++i) { const std::int64_t p = off + i; bits[p >> 3] = static_cast<std::uint8_t>(bits[p >> 3] | (1u << (p & 7))); }
}

// ---- cells ---------------------------------------------------------------------------------------------------

template <class T>
BOLT_FORCE_INLINE double cell_f64(const std::uint8_t* c) noexcept {
    T v;
    std::memcpy(&v, c, sizeof(T));
    return static_cast<double>(v);
}

// out[i] = cells[i] as double, NaN where valid[i] == 0. False for a type with no numeric reading.
inline bool cast_cells_f64(BoltType t, const std::uint8_t* BOLT_RESTRICT cells, const std::uint8_t* valid,
                           std::uint32_t n, double* BOLT_RESTRICT out) noexcept {
    assert(out != nullptr || n == 0);
    double (*rd)(const std::uint8_t*) noexcept = nullptr;
    std::uint32_t w = 0;
    switch (t) {
        case BoltType::Int8:    rd = &cell_f64<std::int8_t>;   w = 1; break;
        case BoltType::Int16:   rd = &cell_f64<std::int16_t>;  w = 2; break;
        case BoltType::Int32:   rd = &cell_f64<std::int32_t>;  w = 4; break;
        case BoltType::Int64:   rd = &cell_f64<std::int64_t>;  w = 8; break;
        case BoltType::UInt8:   rd = &cell_f64<std::uint8_t>;  w = 1; break;
        case BoltType::UInt16:  rd = &cell_f64<std::uint16_t>; w = 2; break;
        case BoltType::UInt32:  rd = &cell_f64<std::uint32_t>; w = 4; break;
        case BoltType::UInt64:  rd = &cell_f64<std::uint64_t>; w = 8; break;
        case BoltType::Float32: rd = &cell_f64<float>;         w = 4; break;
        case BoltType::Float64: rd = &cell_f64<double>;        w = 8; break;
        default: return false;
    }
    const double nan = std::nan("");
    if (valid == nullptr) {
        for (std::uint32_t i = 0; i < n; ++i) out[i] = rd(cells + static_cast<std::size_t>(i) * w);
    } else {
        for (std::uint32_t i = 0; i < n; ++i) out[i] = valid[i] != 0 ? rd(cells + static_cast<std::size_t>(i) * w) : nan;
    }
    return true;
}

// out[i] = cells[idx[i]] (w bytes) where present[i] != 0; other slots untouched.
inline void gather_fixed(const std::uint8_t* BOLT_RESTRICT cells, std::uint32_t w, const std::uint32_t* BOLT_RESTRICT idx,
                         const std::uint8_t* BOLT_RESTRICT present, std::uint32_t m, std::uint8_t* BOLT_RESTRICT out) noexcept {
    assert(w >= 1 && w <= 16);
    assert((cells != nullptr && idx != nullptr && out != nullptr) || m == 0);
    switch (w) {
        case 8:
            for (std::uint32_t i = 0; i < m; ++i) if (present[i] != 0) std::memcpy(out + static_cast<std::size_t>(i) * 8u, cells + static_cast<std::size_t>(idx[i]) * 8u, 8);
            break;
        case 4:
            for (std::uint32_t i = 0; i < m; ++i) if (present[i] != 0) std::memcpy(out + static_cast<std::size_t>(i) * 4u, cells + static_cast<std::size_t>(idx[i]) * 4u, 4);
            break;
        default:
            for (std::uint32_t i = 0; i < m; ++i) if (present[i] != 0) std::memcpy(out + static_cast<std::size_t>(i) * w, cells + static_cast<std::size_t>(idx[i]) * w, w);
            break;
    }
}

// out[i] = ts[idx[i]] where present[i] != 0.
inline void gather_i64(const std::int64_t* BOLT_RESTRICT ts, const std::uint32_t* BOLT_RESTRICT idx,
                       const std::uint8_t* BOLT_RESTRICT present, std::uint32_t m, std::int64_t* BOLT_RESTRICT out) noexcept {
    assert((ts != nullptr && idx != nullptr && out != nullptr) || m == 0);
    for (std::uint32_t i = 0; i < m; ++i) if (present[i] != 0) out[i] = ts[idx[i]];
}

// ---- prefix forms --------------------------------------------------------------------------------------------

// pc[0] = 0, pc[k + 1] = pc[k] + (valid[k] != 0): len + 1 entries. valid == nullptr: pc[k] = k.
inline void valid_prefix_count(const std::uint8_t* BOLT_RESTRICT valid, std::uint32_t len, std::uint32_t* BOLT_RESTRICT pc) noexcept {
    assert(pc != nullptr);
    pc[0] = 0;
    if (valid == nullptr) { for (std::uint32_t k = 0; k < len; ++k) pc[k + 1] = k + 1u; return; }
    std::uint32_t acc = 0;
    for (std::uint32_t k = 0; k < len; ++k) { acc += valid[k] != 0 ? 1u : 0u; pc[k + 1] = acc; }
}

// out[i] = pc[hi[i] - base] - pc[lo[i] - base]; the rows of the arrays start at `base`.
inline void frame_count_valid(const std::uint32_t* BOLT_RESTRICT pc, std::uint32_t base, const std::uint32_t* BOLT_RESTRICT lo,
                              const std::uint32_t* BOLT_RESTRICT hi, std::uint32_t m, std::int64_t* BOLT_RESTRICT out) noexcept {
    assert((pc != nullptr && out != nullptr) || m == 0);
    for (std::uint32_t i = 0; i < m; ++i) out[i] = static_cast<std::int64_t>(pc[hi[i] - base]) - pc[lo[i] - base];
}

// Exact frame sums of an integer column: prefix sums in 128 bits (`ps` holds len + 1 entries), then
// out[i] = ps[hi] - ps[lo]. present[i] = the frame has a non-null row (pc as above). `fail` is set when any
// total leaves int64. T is the cell's integer type (UInt64 included).
template <class T>
inline void frame_sum_int(const T* BOLT_RESTRICT cells, const std::uint8_t* BOLT_RESTRICT valid, std::uint32_t len,
                          __int128* BOLT_RESTRICT ps, const std::uint32_t* BOLT_RESTRICT pc, std::uint32_t base,
                          const std::uint32_t* BOLT_RESTRICT lo, const std::uint32_t* BOLT_RESTRICT hi, std::uint32_t m,
                          std::int64_t* BOLT_RESTRICT out, std::uint8_t* BOLT_RESTRICT present, bool* fail) noexcept {
    assert(ps != nullptr && pc != nullptr && fail != nullptr);
    ps[0] = 0;
    if (valid == nullptr) { for (std::uint32_t k = 0; k < len; ++k) ps[k + 1] = ps[k] + static_cast<__int128>(cells[k]); }
    else { for (std::uint32_t k = 0; k < len; ++k) ps[k + 1] = ps[k] + (valid[k] != 0 ? static_cast<__int128>(cells[k]) : 0); }
    for (std::uint32_t i = 0; i < m; ++i) {
        const __int128 s = ps[hi[i] - base] - ps[lo[i] - base];
        const bool bad = s > INT64_MAX || s < INT64_MIN;
        present[i] = pc[hi[i] - base] != pc[lo[i] - base] ? 1u : 0u;
        out[i] = bad ? 0 : static_cast<std::int64_t>(s);
        if (bad && present[i] != 0) *fail = true;
    }
}

// ---- gather forms --------------------------------------------------------------------------------------------

// pv[k] = the last valid row index <= k (relative to the span), kNoRow if none. nv[k] = first valid >= k.
inline void last_valid_scan(const std::uint8_t* BOLT_RESTRICT valid, std::uint32_t len, std::uint32_t* BOLT_RESTRICT pv) noexcept {
    std::uint32_t cur = kNoRow;
    for (std::uint32_t k = 0; k < len; ++k) { if (valid[k] != 0) cur = k; pv[k] = cur; }
}

inline void first_valid_scan(const std::uint8_t* BOLT_RESTRICT valid, std::uint32_t len, std::uint32_t* BOLT_RESTRICT nv) noexcept {
    std::uint32_t cur = kNoRow;
    for (std::uint32_t k = len; k > 0; --k) { if (valid[k - 1] != 0) cur = k - 1; nv[k - 1] = cur; }
}

// pick[i] = last valid row in [lo, hi) (absolute row = base + span index); present[i] says there is one.
// pv == nullptr: every row is valid (the last row of the frame).
inline void frame_last_valid(const std::uint32_t* BOLT_RESTRICT pv, std::uint32_t base, const std::uint32_t* BOLT_RESTRICT lo,
                             const std::uint32_t* BOLT_RESTRICT hi, std::uint32_t m, std::uint32_t* BOLT_RESTRICT pick,
                             std::uint8_t* BOLT_RESTRICT present) noexcept {
    assert((pick != nullptr && present != nullptr) || m == 0);
    for (std::uint32_t i = 0; i < m; ++i) {
        if (hi[i] <= lo[i]) { present[i] = 0; pick[i] = 0; continue; }
        const std::uint32_t at = pv == nullptr ? hi[i] - 1u : pv[hi[i] - 1u - base];
        const bool ok = pv == nullptr || (at != kNoRow && at + base >= lo[i]);
        present[i] = ok ? 1u : 0u;
        pick[i] = ok ? (pv == nullptr ? at : at + base) : 0u;
    }
}

inline void frame_first_valid(const std::uint32_t* BOLT_RESTRICT nv, std::uint32_t base, const std::uint32_t* BOLT_RESTRICT lo,
                              const std::uint32_t* BOLT_RESTRICT hi, std::uint32_t m, std::uint32_t* BOLT_RESTRICT pick,
                              std::uint8_t* BOLT_RESTRICT present) noexcept {
    assert((pick != nullptr && present != nullptr) || m == 0);
    for (std::uint32_t i = 0; i < m; ++i) {
        if (hi[i] <= lo[i]) { present[i] = 0; pick[i] = 0; continue; }
        const std::uint32_t at = nv == nullptr ? lo[i] : nv[lo[i] - base];
        const bool ok = nv == nullptr || (at != kNoRow && at + base < hi[i]);
        present[i] = ok ? 1u : 0u;
        pick[i] = ok ? (nv == nullptr ? at : at + base) : 0u;
    }
}

// Edge row of the frame (FIRST_VALUE / LAST_VALUE): row lo (last = false) or hi - 1; present = non-empty and valid.
inline void frame_edge(const std::uint8_t* BOLT_RESTRICT valid, std::uint32_t base, const std::uint32_t* BOLT_RESTRICT lo,
                       const std::uint32_t* BOLT_RESTRICT hi, std::uint32_t m, bool last, std::uint32_t* BOLT_RESTRICT pick,
                       std::uint8_t* BOLT_RESTRICT present) noexcept {
    assert((pick != nullptr && present != nullptr) || m == 0);
    for (std::uint32_t i = 0; i < m; ++i) {
        const bool any = lo[i] < hi[i];
        const std::uint32_t at = last ? (hi[i] > 0 ? hi[i] - 1u : 0u) : lo[i];
        pick[i] = any ? at : 0u;
        present[i] = (any && (valid == nullptr || valid[at - base] != 0)) ? 1u : 0u;
    }
}

// ---- deque form ----------------------------------------------------------------------------------------------

// Frame minimum / maximum as a row pick: rows are compared by less(a, b) on absolute row numbers; the first of
// equal rows wins. `dq` holds hi_max - lo_min entries. NULL rows (valid[row - base] == 0) never enter.
template <class Less>
inline void frame_extreme_pick(const std::uint8_t* valid, std::uint32_t base, const std::uint32_t* BOLT_RESTRICT lo,
                               const std::uint32_t* BOLT_RESTRICT hi, std::uint32_t m, bool want_max, Less less,
                               std::uint32_t* BOLT_RESTRICT dq, std::uint32_t* BOLT_RESTRICT pick,
                               std::uint8_t* BOLT_RESTRICT present) noexcept {
    assert((dq != nullptr && pick != nullptr && present != nullptr) || m == 0);
    std::uint32_t head = 0, tail = 0, next = 0;
    for (std::uint32_t i = 0; i < m; ++i) {
        if (next < lo[i]) next = lo[i];
        for (; next < hi[i]; ++next) {
            if (valid != nullptr && valid[next - base] == 0) continue;
            while (tail > head && (want_max ? less(dq[tail - 1], next) : less(next, dq[tail - 1]))) --tail;
            dq[tail++] = next;
        }
        while (head < tail && dq[head] < lo[i]) ++head;
        const bool any = head < tail;
        present[i] = any ? 1u : 0u;
        pick[i] = any ? dq[head] : 0u;
    }
}

// ---- FILL ----------------------------------------------------------------------------------------------------

struct FillPrev {
    std::uint8_t cell[16];
    std::uint8_t have;
};

enum class FillMode : std::uint8_t { Null = 1, Prev = 2, Const = 3 };

// A point whose frame is empty takes the mode's value (Null: stays NULL; Prev: the last non-empty point's
// present value, none yet = NULL; Const: `cst`). A point with a frame that has a present value is remembered.
// cells = m cells of w bytes, valid = m bytes (1 = present), lo / hi = the frame columns.
inline void frame_fill(FillMode mode, std::uint32_t w, std::uint8_t* BOLT_RESTRICT cells, std::uint8_t* BOLT_RESTRICT valid,
                       const std::uint32_t* BOLT_RESTRICT lo, const std::uint32_t* BOLT_RESTRICT hi, std::uint32_t m,
                       FillPrev* prev, const std::uint8_t* cst) noexcept {
    assert(w >= 1 && w <= 16 && prev != nullptr);
    assert(mode != FillMode::Const || cst != nullptr);
    for (std::uint32_t i = 0; i < m; ++i) {
        std::uint8_t* c = cells + static_cast<std::size_t>(i) * w;
        if (lo[i] < hi[i]) {
            if (valid[i] != 0) { std::memcpy(prev->cell, c, w); prev->have = 1; }
            continue;
        }
        valid[i] = 0;
        if (mode == FillMode::Prev && prev->have != 0) { std::memcpy(c, prev->cell, w); valid[i] = 1; }
        else if (mode == FillMode::Const) { std::memcpy(c, cst, w); valid[i] = 1; }
    }
}

}  // namespace frame
}  // namespace kernels
}  // namespace bolt
