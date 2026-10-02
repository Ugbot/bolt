// bolt_rle.h — ColumnFormat::RLE encode / decode and run-granular predicates
// (MSEG K3, decision §6.4 encoding 4 / §17 K3).
//
// Layout (the in-memory BoltColumn::make_rle shape, byte for byte):
//   values   T[num_runs]        — BoltColumn::data, MSEG page b1
//   run_ends int32[num_runs]    — BoltColumn::dict_child, page b2
//   run i covers rows [run_ends[i-1], run_ends[i]), run_ends[-1] = 0;
//   run ends strictly increase and the last equals the row count.
//   validity (optional, page b0) is a per-ROW Arrow bitmap, LSB first,
//   1 = valid, offset 0 — separate from the runs.
//
// Runs split on bit inequality (so NaN, -0.0 and payload bits round-trip).
// A null row never splits a run: it joins the run around it (leading nulls
// join the first valid value's run; an all-null column is one run of T{}).
// Decode therefore returns every VALID row bit-exact and a null row's slot
// holds its run's value; nulls are carried by the validity bitmap alone.
//
// Predicates (= != < <= > >= and IN) are evaluated once per run; matching
// runs emit int32 row ids, coalesced row ranges, or a count. Null rows never
// match. A run's rows are produced by an iota fill (no validity) or by
// scanning the run's validity words with ctz — runs are never decoded.
//
// Tiger Style: POD + free functions, noexcept, no allocation; every loop is
// bounded by num_runs, the row count, or the IN-set size.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "bolt/bolt_port.h"
#include "bolt/kernels/bolt_rle_limits.h"

namespace bolt {
namespace rle {

enum class CmpOp : uint8_t { Eq = 0, Ne = 1, Lt = 2, Le = 3, Gt = 4, Ge = 5 };

namespace detail {

template <size_t N> struct BitsOf;
template <> struct BitsOf<1> { using type = uint8_t; };
template <> struct BitsOf<2> { using type = uint16_t; };
template <> struct BitsOf<4> { using type = uint32_t; };
template <> struct BitsOf<8> { using type = uint64_t; };

template <typename T>
BOLT_FORCE_INLINE typename BitsOf<sizeof(T)>::type bits(T v) noexcept {
    typename BitsOf<sizeof(T)>::type b;
    std::memcpy(&b, &v, sizeof(T));
    return b;
}

BOLT_FORCE_INLINE bool valid_at(const uint8_t* validity, int64_t row) noexcept {
    return validity == nullptr || ((validity[row >> 3] >> (row & 7)) & 1u) != 0;
}

// Validity word w (rows [64w, 64w + 64)) of a bitmap over `rows` rows;
// never reads past the bitmap's (rows + 7) / 8 bytes. Little-endian hosts.
BOLT_FORCE_INLINE uint64_t validity_word(const uint8_t* validity, int64_t w,
                                         int64_t rows) noexcept {
    const int64_t nbytes = (rows + 7) >> 3;
    const int64_t off = w << 3;
    assert(off < nbytes);
    uint64_t x = 0;
    const int64_t take = nbytes - off < 8 ? nbytes - off : 8;
    std::memcpy(&x, validity + off, static_cast<size_t>(take));
    return x;
}

// Validity word w restricted to rows [begin, end).
BOLT_FORCE_INLINE uint64_t masked_word(const uint8_t* validity, int64_t w, int64_t begin,
                                       int64_t end, int64_t rows) noexcept {
    assert(begin < end && end <= rows);
    uint64_t m = validity_word(validity, w, rows);
    const int64_t lo = w << 6;
    if (lo < begin) m &= ~0ull << (begin - lo);
    if (end - lo < 64) m &= (1ull << (end - lo)) - 1u;
    return m;
}

// Calls emit(begin, end) for each run, nulls joining the current run.
template <typename T, typename Emit>
BOLT_FORCE_INLINE int64_t scan_runs(const T* BOLT_RESTRICT values,
                                    const uint8_t* BOLT_RESTRICT validity, int64_t rows,
                                    Emit&& emit) noexcept {
    assert(values != nullptr || rows == 0);
    assert(rows >= 0 && rows <= kRleMaxRows);
    if (rows == 0) return 0;
    int64_t first = 0;
    if (validity != nullptr) {
        while (first < rows && !valid_at(validity, first)) ++first;
    }
    auto cur = first < rows ? bits(values[first]) : decltype(bits(values[0])){0};
    int64_t begin = 0;
    int64_t runs = 0;
    for (int64_t i = first + 1; i < rows; ++i) {
        const auto b = bits(values[i]);
        if (b != cur && valid_at(validity, i)) {
            if (!emit(begin, i, cur)) return -1;
            ++runs;
            begin = i;
            cur = b;
        }
    }
    if (!emit(begin, rows, cur)) return -1;
    return runs + 1;
}

// Fill out[b, e) with v in whole 64 B chunks. A chunk may run past e into
// the next run's rows (rewritten by that run) but never past the page: a
// chunk starts only below `room` = rows - K + 1; past it, an exact loop.
template <typename T>
BOLT_FORCE_INLINE void fill_run(T* BOLT_RESTRICT out, int32_t b, int32_t e, int32_t room,
                                T v) noexcept {
    constexpr int32_t K = static_cast<int32_t>(kRleFillBytes / sizeof(T));
    static_assert(K >= 1 && K * sizeof(T) == kRleFillBytes);
    assert(b < e);
    assert(room >= 0);
    T chunk[K];
    for (int32_t k = 0; k < K; ++k) chunk[k] = v;
    const int32_t lim = e < room ? e : room;
    int32_t j = b;
    while (j + K < lim) {  // bounded: j advances by 2K
        std::memcpy(out + j, chunk, kRleFillBytes);
        std::memcpy(out + j + K, chunk, kRleFillBytes);
        j += 2 * K;
    }
    if (j < lim) {
        std::memcpy(out + j, chunk, kRleFillBytes);
        j += K;
    }
    for (; j < e; ++j) out[j] = v;
}

}  // namespace detail

/// True iff `run_ends` is a well-formed run table for `rows` rows.
inline bool run_ends_valid(const int32_t* run_ends, int64_t num_runs, int64_t rows) noexcept {
    assert(num_runs >= 0);
    assert(rows >= 0);
    if (rows > kRleMaxRows) return false;
    if (num_runs == 0) return rows == 0;
    if (run_ends == nullptr || num_runs > rows) return false;
    int32_t prev = 0;
    for (int64_t i = 0; i < num_runs; ++i) {
        if (run_ends[i] <= prev) return false;
        prev = run_ends[i];
    }
    return prev == rows;
}

/// Number of runs `encode` produces for these rows.
template <typename T>
inline int64_t count_runs(const T* values, const uint8_t* validity, int64_t rows) noexcept {
    static_assert(std::is_trivially_copyable_v<T>);
    assert(values != nullptr || rows == 0);
    assert(rows >= 0 && rows <= kRleMaxRows);
    return detail::scan_runs(values, validity, rows,
                             [](int64_t, int64_t, auto) noexcept { return true; });
}

/// Encode `rows` values into `out_values` / `out_run_ends`. Returns the run
/// count, or -1 when more than `max_runs` runs are needed (nothing past
/// max_runs is written; size with count_runs, or max_runs = rows).
template <typename T>
inline int64_t encode(const T* BOLT_RESTRICT values, const uint8_t* BOLT_RESTRICT validity,
                      int64_t rows, T* BOLT_RESTRICT out_values,
                      int32_t* BOLT_RESTRICT out_run_ends, int64_t max_runs) noexcept {
    static_assert(std::is_trivially_copyable_v<T>);
    assert(rows >= 0 && rows <= kRleMaxRows);
    assert(max_runs >= 0);
    assert((out_values != nullptr && out_run_ends != nullptr) || max_runs == 0);
    int64_t n = 0;
    const int64_t r = detail::scan_runs(
        values, validity, rows, [&](int64_t, int64_t end, auto b) noexcept {
            if (n >= max_runs) return false;
            std::memcpy(out_values + n, &b, sizeof(T));
            out_run_ends[n] = static_cast<int32_t>(end);
            ++n;
            return true;
        });
    assert(r < 0 || run_ends_valid(out_run_ends, r, rows));
    return r;
}

/// Decode into `out` (rows values). False on a malformed run table.
template <typename T>
inline bool decode(const T* BOLT_RESTRICT values, const int32_t* BOLT_RESTRICT run_ends,
                   int64_t num_runs, int64_t rows, T* BOLT_RESTRICT out) noexcept {
    static_assert(std::is_trivially_copyable_v<T>);
    assert(num_runs >= 0 && rows >= 0);
    assert(out != nullptr || rows == 0);
    if (rows > kRleMaxRows || num_runs > rows) return false;
    if (num_runs == 0) return rows == 0;
    constexpr int64_t K = static_cast<int64_t>(kRleFillBytes / sizeof(T));
    const int32_t room = static_cast<int32_t>(rows - K + 1 > 0 ? rows - K + 1 : 0);
    int32_t prev = 0;
    for (int64_t i = 0; i < num_runs; ++i) {
        const int32_t end = run_ends[i];
        if (end <= prev || end > rows) return false;
        detail::fill_run(out, prev, end, room, values[i]);
        prev = end;
    }
    return prev == rows;
}

/// The run holding `row` (binary search over run ends).
inline int64_t find_run(const int32_t* run_ends, int64_t num_runs, int64_t row) noexcept {
    assert(run_ends != nullptr && num_runs > 0);
    assert(row >= 0 && row < run_ends[num_runs - 1]);
    int64_t lo = 0;
    int64_t hi = num_runs - 1;
    while (lo < hi) {  // bounded: hi - lo halves
        const int64_t mid = lo + ((hi - lo) >> 1);
        if (run_ends[mid] > row) hi = mid; else lo = mid + 1;
    }
    return lo;
}

// ---------------------------------------------------------------------------
// Predicates
// ---------------------------------------------------------------------------

namespace detail {

template <CmpOp OP, typename T>
BOLT_FORCE_INLINE bool cmp(T v, T s) noexcept {
    if constexpr (OP == CmpOp::Eq) return v == s;
    else if constexpr (OP == CmpOp::Ne) return v != s;
    else if constexpr (OP == CmpOp::Lt) return v < s;
    else if constexpr (OP == CmpOp::Le) return v <= s;
    else if constexpr (OP == CmpOp::Gt) return v > s;
    else return v >= s;
}

template <CmpOp OP, typename T> struct CmpPred {
    T s;
    BOLT_FORCE_INLINE bool operator()(T v) const noexcept { return cmp<OP>(v, s); }
};

// Membership in a strictly ascending set.
template <typename T> struct InPred {
    const T* set;
    int32_t n;
    BOLT_FORCE_INLINE bool operator()(T v) const noexcept {
        if (n <= kRleInLinearMax) {
            bool hit = false;
            for (int32_t k = 0; k < n; ++k) hit |= set[k] == v;
            return hit;
        }
        int32_t lo = 0;
        int32_t hi = n;
        while (lo < hi) {  // bounded: hi - lo halves
            const int32_t mid = lo + ((hi - lo) >> 1);
            if (set[mid] < v) lo = mid + 1; else hi = mid;
        }
        return lo < n && set[lo] == v;
    }
};

// Sinks: emit(begin, end) for every row of a matching run, honouring validity.
struct CountSink {
    int64_t n = 0;
    BOLT_FORCE_INLINE bool all(int64_t b, int64_t e) noexcept { n += e - b; return true; }
    BOLT_FORCE_INLINE bool word(int64_t, uint64_t m) noexcept {
        n += bolt_popcount64(m);
        return true;
    }
};

struct SelSink {
    int32_t* BOLT_RESTRICT sel;
    int64_t n = 0;
    BOLT_FORCE_INLINE bool all(int64_t b, int64_t e) noexcept {
        int32_t* BOLT_RESTRICT o = sel + n;
        const int32_t base = static_cast<int32_t>(b);
        const int32_t len = static_cast<int32_t>(e - b);
        for (int32_t k = 0; k < len; ++k) o[k] = base + k;
        n += len;
        return true;
    }
    BOLT_FORCE_INLINE bool word(int64_t lo, uint64_t m) noexcept {
        while (m != 0) {  // bounded: one bit cleared per step
            sel[n++] = static_cast<int32_t>(lo + bolt_ctz64(m));
            m &= m - 1;
        }
        return true;
    }
};

struct RangeSink {
    int32_t* BOLT_RESTRICT starts;
    int32_t* BOLT_RESTRICT ends;
    int64_t cap;
    int64_t n = 0;
    BOLT_FORCE_INLINE bool all(int64_t b, int64_t e) noexcept {
        if (n > 0 && ends[n - 1] == b) { ends[n - 1] = static_cast<int32_t>(e); return true; }
        if (n >= cap) return false;
        starts[n] = static_cast<int32_t>(b);
        ends[n] = static_cast<int32_t>(e);
        ++n;
        return true;
    }
    BOLT_FORCE_INLINE bool word(int64_t lo, uint64_t m) noexcept {
        while (m != 0) {  // bounded: each step clears >= 1 bit
            const int s = bolt_ctz64(m);
            const int len = bolt_ctz64(~(m >> s));
            const int stop = s + (len > 64 - s ? 64 - s : len);
            if (!all(lo + s, lo + stop)) return false;
            m = stop >= 64 ? 0 : m & (~0ull << stop);
        }
        return true;
    }
};

template <typename Sink>
BOLT_FORCE_INLINE bool emit_valid(Sink& sink, const uint8_t* validity, int64_t b, int64_t e,
                                  int64_t rows) noexcept {
    assert(b < e && e <= rows);
    const int64_t w_end = (e - 1) >> 6;
    for (int64_t w = b >> 6; w <= w_end; ++w) {
        const uint64_t m = masked_word(validity, w, b, e, rows);
        if (m != 0 && !sink.word(w << 6, m)) return false;
    }
    return true;
}

// Returns false only when the sink refused (range capacity).
template <typename T, typename Pred, typename Sink>
inline bool select_runs(const T* BOLT_RESTRICT values, const int32_t* BOLT_RESTRICT run_ends,
                        int64_t num_runs, const uint8_t* BOLT_RESTRICT validity, Pred pred,
                        Sink& sink) noexcept {
    assert(values != nullptr || num_runs == 0);
    assert(run_ends != nullptr || num_runs == 0);
    const int64_t rows = num_runs > 0 ? run_ends[num_runs - 1] : 0;
    int32_t prev = 0;
    for (int64_t i = 0; i < num_runs; ++i) {
        const int32_t end = run_ends[i];
        assert(end > prev);
        if (pred(values[i])) {
            const bool ok = validity == nullptr ? sink.all(prev, end)
                                                : emit_valid(sink, validity, prev, end, rows);
            if (!ok) return false;
        }
        prev = end;
    }
    return true;
}

// Count without validity: branch-free over runs (vectorizes).
template <typename T, typename Pred>
inline int64_t count_novalid(const T* BOLT_RESTRICT values, const int32_t* BOLT_RESTRICT run_ends,
                             int64_t num_runs, Pred pred) noexcept {
    assert(values != nullptr || num_runs == 0);
    assert(run_ends != nullptr || num_runs == 0);
    if (num_runs == 0) return 0;
    int64_t acc = pred(values[0]) ? run_ends[0] : 0;
    for (int64_t i = 1; i < num_runs; ++i) {
        const int64_t len = static_cast<int64_t>(run_ends[i]) - run_ends[i - 1];
        acc += pred(values[i]) ? len : 0;
    }
    return acc;
}

template <typename T>
inline int64_t count_cmp(const T* values, const int32_t* run_ends, int64_t num_runs, CmpOp op,
                         T s) noexcept {
    switch (op) {
        case CmpOp::Eq: return count_novalid(values, run_ends, num_runs, CmpPred<CmpOp::Eq, T>{s});
        case CmpOp::Ne: return count_novalid(values, run_ends, num_runs, CmpPred<CmpOp::Ne, T>{s});
        case CmpOp::Lt: return count_novalid(values, run_ends, num_runs, CmpPred<CmpOp::Lt, T>{s});
        case CmpOp::Le: return count_novalid(values, run_ends, num_runs, CmpPred<CmpOp::Le, T>{s});
        case CmpOp::Gt: return count_novalid(values, run_ends, num_runs, CmpPred<CmpOp::Gt, T>{s});
        case CmpOp::Ge: return count_novalid(values, run_ends, num_runs, CmpPred<CmpOp::Ge, T>{s});
    }
    assert(false && "bad CmpOp");
    return 0;
}

template <typename T, typename Sink>
inline bool select_cmp(const T* values, const int32_t* run_ends, int64_t num_runs,
                       const uint8_t* validity, CmpOp op, T s, Sink& sink) noexcept {
    switch (op) {
        case CmpOp::Eq: return select_runs(values, run_ends, num_runs, validity, CmpPred<CmpOp::Eq, T>{s}, sink);
        case CmpOp::Ne: return select_runs(values, run_ends, num_runs, validity, CmpPred<CmpOp::Ne, T>{s}, sink);
        case CmpOp::Lt: return select_runs(values, run_ends, num_runs, validity, CmpPred<CmpOp::Lt, T>{s}, sink);
        case CmpOp::Le: return select_runs(values, run_ends, num_runs, validity, CmpPred<CmpOp::Le, T>{s}, sink);
        case CmpOp::Gt: return select_runs(values, run_ends, num_runs, validity, CmpPred<CmpOp::Gt, T>{s}, sink);
        case CmpOp::Ge: return select_runs(values, run_ends, num_runs, validity, CmpPred<CmpOp::Ge, T>{s}, sink);
    }
    assert(false && "bad CmpOp");
    return false;
}

template <typename T>
inline bool in_set_valid(const T* set, int32_t n) noexcept {
    if (n < 0 || (n > 0 && set == nullptr)) return false;
    for (int32_t k = 1; k < n; ++k) {
        if (!(set[k - 1] < set[k])) return false;
    }
    return true;
}

}  // namespace detail

/// Row ids (ascending) of valid rows where `value OP scalar`. `sel` holds
/// up to run_ends[num_runs-1] entries. Returns the count.
template <typename T>
inline int64_t filter_sel(const T* values, const int32_t* run_ends, int64_t num_runs,
                          const uint8_t* validity, CmpOp op, T scalar, int32_t* sel) noexcept {
    assert(sel != nullptr || num_runs == 0);
    assert(num_runs >= 0);
    detail::SelSink sink{sel};
    detail::select_cmp(values, run_ends, num_runs, validity, op, scalar, sink);
    return sink.n;
}

/// Count of valid rows where `value OP scalar`.
template <typename T>
inline int64_t filter_count(const T* values, const int32_t* run_ends, int64_t num_runs,
                            const uint8_t* validity, CmpOp op, T scalar) noexcept {
    assert(values != nullptr || num_runs == 0);
    assert(num_runs >= 0);
    if (validity == nullptr) return detail::count_cmp(values, run_ends, num_runs, op, scalar);
    detail::CountSink sink;
    detail::select_cmp(values, run_ends, num_runs, validity, op, scalar, sink);
    return sink.n;
}

/// Coalesced half-open row ranges [starts[k], ends[k]) of valid matching
/// rows. Returns the range count, or -1 past `max_ranges` (at most num_runs
/// ranges without validity; at most (rows + 1) / 2 with it).
template <typename T>
inline int64_t filter_ranges(const T* values, const int32_t* run_ends, int64_t num_runs,
                             const uint8_t* validity, CmpOp op, T scalar, int32_t* starts,
                             int32_t* ends, int64_t max_ranges) noexcept {
    assert(max_ranges >= 0);
    assert((starts != nullptr && ends != nullptr) || max_ranges == 0);
    detail::RangeSink sink{starts, ends, max_ranges};
    if (!detail::select_cmp(values, run_ends, num_runs, validity, op, scalar, sink)) return -1;
    return sink.n;
}

/// IN over a strictly ascending `set` of `set_n` values; -1 if the set is
/// not strictly ascending. Otherwise as filter_sel.
template <typename T>
inline int64_t filter_in_sel(const T* values, const int32_t* run_ends, int64_t num_runs,
                             const uint8_t* validity, const T* set, int32_t set_n,
                             int32_t* sel) noexcept {
    assert(sel != nullptr || num_runs == 0);
    assert(num_runs >= 0);
    if (!detail::in_set_valid(set, set_n)) return -1;
    detail::SelSink sink{sel};
    detail::select_runs(values, run_ends, num_runs, validity, detail::InPred<T>{set, set_n}, sink);
    return sink.n;
}

/// IN count; -1 if the set is not strictly ascending.
template <typename T>
inline int64_t filter_in_count(const T* values, const int32_t* run_ends, int64_t num_runs,
                               const uint8_t* validity, const T* set, int32_t set_n) noexcept {
    assert(values != nullptr || num_runs == 0);
    assert(num_runs >= 0);
    if (!detail::in_set_valid(set, set_n)) return -1;
    if (validity == nullptr) {
        return detail::count_novalid(values, run_ends, num_runs, detail::InPred<T>{set, set_n});
    }
    detail::CountSink sink;
    detail::select_runs(values, run_ends, num_runs, validity, detail::InPred<T>{set, set_n}, sink);
    return sink.n;
}

/// IN ranges; -1 if the set is not strictly ascending or past max_ranges.
template <typename T>
inline int64_t filter_in_ranges(const T* values, const int32_t* run_ends, int64_t num_runs,
                                const uint8_t* validity, const T* set, int32_t set_n,
                                int32_t* starts, int32_t* ends, int64_t max_ranges) noexcept {
    assert(max_ranges >= 0);
    assert((starts != nullptr && ends != nullptr) || max_ranges == 0);
    if (!detail::in_set_valid(set, set_n)) return -1;
    detail::RangeSink sink{starts, ends, max_ranges};
    if (!detail::select_runs(values, run_ends, num_runs, validity,
                             detail::InPred<T>{set, set_n}, sink)) {
        return -1;
    }
    return sink.n;
}

}  // namespace rle
}  // namespace bolt
