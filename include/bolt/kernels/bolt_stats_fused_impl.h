// bolt_stats_fused_impl.h — internals of bolt_stats_fused.h (MSEG K7).
// Include bolt_stats_fused.h, not this file.
#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace bolt::stats::detail {

// ---- keys -------------------------------------------------------------------
// A key is the signed integer of the value's width whose order is the value
// order: ints as is, unsigned with the top bit flipped, floats as their IEEE
// total-order bits. The state keeps keys sign-extended to int64.

template <size_t W> struct IntOfWidth;
template <> struct IntOfWidth<1> { using S = int8_t;  using U = uint8_t; };
template <> struct IntOfWidth<2> { using S = int16_t; using U = uint16_t; };
template <> struct IntOfWidth<4> { using S = int32_t; using U = uint32_t; };
template <> struct IntOfWidth<8> { using S = int64_t; using U = uint64_t; };

template <class T> using KeyT = typename IntOfWidth<sizeof(T)>::S;
template <class T> using MaskT = typename IntOfWidth<sizeof(T)>::U;

BOLT_FORCE_INLINE int64_t key_f64_bits(uint64_t bits) noexcept {
    const int64_t s = static_cast<int64_t>(bits);
    return s ^ ((s >> 63) & INT64_MAX);
}
BOLT_FORCE_INLINE int32_t key_f32_bits(uint32_t bits) noexcept {
    const int32_t s = static_cast<int32_t>(bits);
    return s ^ ((s >> 31) & INT32_MAX);
}

template <class T, StatsKind K>
BOLT_FORCE_INLINE KeyT<T> key_of(T v) noexcept {
    if constexpr (K == StatsKind::kF32) {
        uint32_t b; std::memcpy(&b, &v, 4); return key_f32_bits(b);
    } else if constexpr (K == StatsKind::kF64) {
        uint64_t b; std::memcpy(&b, &v, 8); return key_f64_bits(b);
    } else if constexpr (K == StatsKind::kUnsigned) {
        using U = MaskT<T>;
        constexpr U top = static_cast<U>(U{1} << (8 * sizeof(T) - 1));
        return static_cast<KeyT<T>>(static_cast<U>(v ^ top));
    } else {
        return static_cast<KeyT<T>>(v);
    }
}

template <class T>
BOLT_FORCE_INLINE T last_val(const StatsFusedState& st) noexcept {
    T v;
    std::memcpy(&v, &st.last_raw, sizeof(T));
    return v;
}

template <class T>
BOLT_FORCE_INLINE void set_last(StatsFusedState& st, T v) noexcept {
    st.last_raw = 0;
    std::memcpy(&st.last_raw, &v, sizeof(T));
}

BOLT_FORCE_INLINE bool d128_less(D128Key a, D128Key b) noexcept {
    return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}

BOLT_FORCE_INLINE D128Key d128_load(const uint8_t* p) noexcept {
    D128Key k;
    std::memcpy(&k.lo, p, 8);
    std::memcpy(&k.hi, p + 8, 8);
    return k;
}

// ---- validity ---------------------------------------------------------------

BOLT_FORCE_INLINE bool validity_bit(const uint8_t* v, uint64_t i) noexcept {
    return ((v[i >> 3] >> (i & 7)) & 1u) != 0;
}

// True when bits [off, off + n) of `v` are all set; reads no byte past them.
BOLT_FORCE_INLINE bool validity_all_set(const uint8_t* v, uint64_t off, uint32_t n) noexcept {
    assert(v != nullptr);
    assert(n > 0);
    while (n > 0) {
        const uint32_t m = n < 56 ? n : 56;
        const uint32_t sh = static_cast<uint32_t>(off & 7);
        const uint32_t nbytes = (sh + m + 7) >> 3;
        uint64_t w = 0;
        std::memcpy(&w, v + (off >> 3), nbytes);
        const uint64_t mask = (1ull << m) - 1;
        if (((w >> sh) & mask) != mask) return false;
        off += m;
        n -= m;
    }
    return true;
}

// ---- scalar path --------------------------------------------------------------

// 128-bit add of a sign- (signed kinds) or zero-extended 64-bit value.
template <StatsKind K>
BOLT_FORCE_INLINE void add_wide(StatsFusedState& st, uint64_t u, uint32_t lane) noexcept {
    const uint64_t nl = st.sum_lo[lane] + u;
    int64_t c = static_cast<int64_t>(nl < u);
    if constexpr (K == StatsKind::kSigned) c += static_cast<int64_t>(u) >> 63;
    st.sum_hi[lane] += c;
    st.sum_lo[lane] = nl;
}

template <class T, StatsKind K>
BOLT_FORCE_INLINE void add_sum(StatsFusedState& st, T v, uint32_t lane) noexcept {
    if constexpr (K == StatsKind::kF32 || K == StatsKind::kF64) {
        st.fsum[lane] += static_cast<double>(v);
    } else if constexpr (K == StatsKind::kSigned) {
        add_wide<K>(st, static_cast<uint64_t>(static_cast<int64_t>(v)), lane);
    } else {
        add_wide<K>(st, static_cast<uint64_t>(v), lane);
    }
}

// One row, any validity: the path for prologue, tail, blocks with nulls and
// float blocks with a NaN.
template <class T, StatsKind K>
BOLT_FORCE_INLINE void scalar_row(StatsFusedState& st, const T* s, T* d, uint32_t i,
                                  bool valid, uint64_t row) noexcept {
    assert(s != nullptr);
    assert(st.kind != StatsKind::kNone && st.kind != StatsKind::kD128);
    if (!valid) {
        if (d != nullptr) d[i] = T{};
        ++st.nulls;
        return;
    }
    const T v = s[i];
    if (d != nullptr) d[i] = v;
    if constexpr (K == StatsKind::kF32 || K == StatsKind::kF64) {
        if (v != v) {
            ++st.nans;
            return;
        }
    }
    const int64_t k = key_of<T, K>(v);
    if (st.nvals != 0) {
        const T pv = last_val<T>(st);
        st.asc &= !(v < pv);
        st.desc &= !(v > pv);
    }
    set_last<T>(st, v);
    st.key_min = k < st.key_min ? k : st.key_min;
    st.key_max = k > st.key_max ? k : st.key_max;
    ++st.nvals;
    add_sum<T, K>(st, v, static_cast<uint32_t>(row & (kStatsSumLanes - 1)));
}

// ---- dense path ---------------------------------------------------------------

// Lanes of one dense group: floats use the kStatsSumLanes sum lanes; ints use
// two vectors of keys.
template <class T, StatsKind K>
inline constexpr uint32_t kDenseLanes =
    (K == StatsKind::kF32 || K == StatsKind::kF64) ? kStatsSumLanes
    : (sizeof(T) >= 4 ? 8u : 32u / static_cast<uint32_t>(sizeof(T)));

// Per-block integer sum accumulator: 32-bit for 1/2-byte values (exact over
// kStatsBlockRows rows), otherwise a wrapping uint64 read back as int64 for
// signed kinds once block_sum_exact holds.
template <class T, StatsKind K>
using BlockSumT = std::conditional_t<
    sizeof(T) <= 2, std::conditional_t<K == StatsKind::kSigned, int32_t, uint32_t>, uint64_t>;

static_assert(kStatsBlockRows <= 1024, "block sum bounds assume <= 1024 rows per block");
static_assert(kStatsBlockRows % 32 == 0, "block holds whole dense groups of every type");

// Whether a block's wrapping 64-bit lane sums are exact: every |value| < 2^53
// and 1024 * 2^53 = 2^63. Narrower types always are (1024 * 2^16 < 2^32).
template <class T, StatsKind K>
BOLT_FORCE_INLINE bool block_sum_exact(int64_t mn, int64_t mx) noexcept {
    constexpr int64_t lim = int64_t{1} << 53;
    if constexpr (sizeof(T) < 8) {
        (void)mn; (void)mx;
        return true;
    } else if constexpr (K == StatsKind::kUnsigned) {
        (void)mn;
        return mx < static_cast<int64_t>(static_cast<uint64_t>(lim) ^ (1ull << 63));
    } else {
        return mn >= -lim && mx < lim;
    }
}

template <class T, StatsKind K, uint32_t G>
BOLT_FORCE_INLINE void add_block_sums(StatsFusedState& st, const BlockSumT<T, K>* bs) noexcept {
    for (uint32_t j = 0; j < G; ++j) {
        if constexpr (K == StatsKind::kSigned)
            add_wide<K>(st, static_cast<uint64_t>(static_cast<int64_t>(bs[j])), j & (kStatsSumLanes - 1));
        else
            add_wide<K>(st, static_cast<uint64_t>(bs[j]), j & (kStatsSumLanes - 1));
    }
}

template <class T, StatsKind K>
inline constexpr bool kIsFloat = K == StatsKind::kF32 || K == StatsKind::kF64;

template <class T, StatsKind K, uint32_t G>
struct DenseAcc {
    T mn[G], mx[G];
    MaskT<T> ba[G], bd[G];
    BlockSumT<T, K> bs[G];
    double fs[kStatsSumLanes];
};

// Total-order key of a float block's numeric extreme `e`: a zero extreme is
// -0.0 (min) / +0.0 (max) when the block holds that zero.
template <class T, StatsKind K>
BOLT_FORCE_INLINE int64_t float_extreme_key(const T* s, uint32_t n, T e, bool is_min) noexcept {
    static_assert(kIsFloat<T, K>, "floats only");
    if (e != T{0}) return key_of<T, K>(e);
    const T want = is_min ? -T{0} : T{0};
    const KeyT<T> kw = key_of<T, K>(want);
    for (uint32_t i = 0; i < n; ++i)
        if (key_of<T, K>(s[i]) == kw) return kw;
    return key_of<T, K>(-want);
}

// Folds a dense block's lanes into the state; false (state untouched) when a
// float block's lane sums show a NaN (a NaN row, or inf - inf).
template <class T, StatsKind K, uint32_t G>
BOLT_FORCE_INLINE bool dense_commit(StatsFusedState& st, const T* s, uint32_t n,
                                    const DenseAcc<T, K, G>& a) noexcept {
    assert(n >= G && n % G == 0);
    MaskT<T> bad_asc = 0, bad_desc = 0;
    T emn = a.mn[0], emx = a.mx[0];
    for (uint32_t j = 0; j < G; ++j) {
        bad_asc |= a.ba[j]; bad_desc |= a.bd[j];
        emn = a.mn[j] < emn ? a.mn[j] : emn;
        emx = a.mx[j] > emx ? a.mx[j] : emx;
    }
    int64_t bmn, bmx;
    if constexpr (kIsFloat<T, K>) {
        for (uint32_t j = 0; j < kStatsSumLanes; ++j)
            if (a.fs[j] != a.fs[j]) return false;
        bmn = float_extreme_key<T, K>(s, n, emn, true);
        bmx = float_extreme_key<T, K>(s, n, emx, false);
        for (uint32_t j = 0; j < kStatsSumLanes; ++j) st.fsum[j] = a.fs[j];
    } else {
        bmn = key_of<T, K>(emn); bmx = key_of<T, K>(emx);
        if (block_sum_exact<T, K>(bmn, bmx)) add_block_sums<T, K, G>(st, a.bs);
        else for (uint32_t i = 0; i < n; ++i) add_sum<T, K>(st, s[i], i & (kStatsSumLanes - 1));
    }
    st.key_min = bmn < st.key_min ? bmn : st.key_min;
    st.key_max = bmx > st.key_max ? bmx : st.key_max;
    st.asc &= bad_asc == 0;
    st.desc &= bad_desc == 0;
    set_last<T>(st, s[n - 1]);
    st.nvals += n;
    assert(st.key_min <= st.key_max);
    return true;
}

// Extremes and sorted flags compare values natively (floats numerically; a
// NaN anywhere poisons its sum lane and the block is redone row by row).
template <class T, StatsKind K, bool kCopy, uint32_t G>
BOLT_FORCE_INLINE void dense_step(DenseAcc<T, K, G>& a, T v, T pv, uint32_t j,
                                  T* BOLT_RESTRICT d, uint32_t i, bool first) noexcept {
    using M = MaskT<T>;
    if constexpr (kCopy) d[i] = v;
    if constexpr (kIsFloat<T, K>) {
        a.mn[j] = first ? v : std::fmin(v, a.mn[j]);
        a.mx[j] = first ? v : std::fmax(v, a.mx[j]);
        a.fs[j] += static_cast<double>(v);
    } else {
        a.mn[j] = first ? v : (v < a.mn[j] ? v : a.mn[j]);
        a.mx[j] = first ? v : (v > a.mx[j] ? v : a.mx[j]);
        using B = BlockSumT<T, K>;
        const B add = std::is_same_v<B, uint64_t>
            ? static_cast<B>(static_cast<int64_t>(v)) : static_cast<B>(v);
        a.bs[j] = static_cast<B>((first ? B{0} : a.bs[j]) + add);
    }
    a.ba[j] = static_cast<M>((first ? M{0} : a.ba[j]) | static_cast<M>(M{0} - static_cast<M>(v < pv)));
    a.bd[j] = static_cast<M>((first ? M{0} : a.bd[j]) | static_cast<M>(M{0} - static_cast<M>(v > pv)));
}

// A block of n (multiple of the dense group, <= kStatsBlockRows) valid rows,
// starting on sum lane 0. False, leaving `st` untouched, when a float block
// holds a NaN.
template <class T, StatsKind K, bool kCopy>
BOLT_FORCE_INLINE bool dense_block(StatsFusedState& st, const T* BOLT_RESTRICT s,
                                   T* BOLT_RESTRICT d, uint32_t n) noexcept {
    constexpr uint32_t G = kDenseLanes<T, K>;
    assert(n > 0 && n % G == 0 && n <= kStatsBlockRows);
    assert(!kCopy || d != nullptr);
    DenseAcc<T, K, G> a;
    for (uint32_t j = 0; j < kStatsSumLanes; ++j) a.fs[j] = st.fsum[j];
    const T p0 = st.nvals ? last_val<T>(st) : s[0];
    for (uint32_t j = 0; j < G; ++j)
        dense_step<T, K, kCopy, G>(a, s[j], j ? s[j - 1] : p0, j, d, j, true);
    for (uint32_t g = G; g < n; g += G)
        for (uint32_t j = 0; j < G; ++j)
            dense_step<T, K, kCopy, G>(a, s[g + j], s[g + j - 1], j, d, g + j, false);
    return dense_commit<T, K, G>(st, s, n, a);
}

// ---- driver -------------------------------------------------------------------

struct NoExt {
    template <class T>
    BOLT_FORCE_INLINE void on_rows(const T*, uint32_t, uint64_t, bool) noexcept {}
};

template <class T, StatsKind K, bool kCopy, class Ext>
BOLT_FORCE_INLINE void scalar_range(StatsFusedState& st, const T* s, T* d,
                                    const uint8_t* val, uint64_t voff, uint32_t i0,
                                    uint32_t i1, Ext* ext) noexcept {
    assert(i0 <= i1);
    for (uint32_t i = i0; i < i1; ++i) {
        const bool valid = val == nullptr || validity_bit(val, voff + i);
        scalar_row<T, K>(st, s, kCopy ? d : nullptr, i, valid, st.rows + i);
    }
    if (ext != nullptr && i1 > i0) ext->on_rows(s + i0, i1 - i0, st.rows + i0, false);
}

template <class T, StatsKind K, bool kCopy, class Ext>
void update_typed(StatsFusedState& st, const T* s, const uint8_t* val, uint64_t voff,
                  uint32_t n, T* d, Ext* ext) noexcept {
    constexpr uint32_t L = kStatsSumLanes, G = kDenseLanes<T, K>;
    assert(n > 0 && s != nullptr);
    const uint32_t lead = static_cast<uint32_t>((L - (st.rows & (L - 1))) & (L - 1));
    const uint32_t pro = lead < n ? lead : n;
    scalar_range<T, K, kCopy>(st, s, d, val, voff, 0, pro, ext);
    uint32_t i = pro;
    while (n - i >= G) {
        const uint32_t rem = (n - i) / G * G;
        const uint32_t b = rem < kStatsBlockRows ? rem : kStatsBlockRows;
        const bool all_valid = val == nullptr || validity_all_set(val, voff + i, b);
        if (all_valid && dense_block<T, K, kCopy>(st, s + i, kCopy ? d + i : nullptr, b)) {
            if (ext != nullptr) ext->on_rows(s + i, b, st.rows + i, true);
        } else {
            scalar_range<T, K, kCopy>(st, s, d, val, voff, i, i + b, ext);
        }
        i += b;
    }
    scalar_range<T, K, kCopy>(st, s, d, val, voff, i, n, ext);
    assert(i <= n);
    st.rows += n;
}

inline void update_d128(StatsFusedState& st, const uint8_t* s, const uint8_t* val,
                        uint64_t voff, uint32_t n, uint8_t* d) noexcept {
    assert(st.kind == StatsKind::kD128);
    for (uint32_t i = 0; i < n; ++i) {
        const bool valid = val == nullptr || validity_bit(val, voff + i);
        if (!valid) {
            if (d != nullptr) std::memset(d + 16u * i, 0, 16);
            ++st.nulls;
            continue;
        }
        if (d != nullptr) std::memcpy(d + 16u * i, s + 16u * i, 16);
        const D128Key k = d128_load(s + 16u * i);
        if (st.nvals != 0) {
            st.asc &= !d128_less(k, st.d_last);
            st.desc &= !d128_less(st.d_last, k);
        }
        st.d_last = k;
        if (d128_less(k, st.d_min)) st.d_min = k;
        if (d128_less(st.d_max, k)) st.d_max = k;
        ++st.nvals;
    }
    st.rows += n;
}

template <class T, StatsKind K, class Ext>
BOLT_FORCE_INLINE void update_as(StatsFusedState& st, const void* src, const uint8_t* val,
                                 uint64_t voff, uint32_t n, void* dst, Ext* ext) noexcept {
    const T* s = static_cast<const T*>(src);
    if (dst != nullptr) update_typed<T, K, true>(st, s, val, voff, n, static_cast<T*>(dst), ext);
    else update_typed<T, K, false>(st, s, val, voff, n, static_cast<T*>(nullptr), ext);
}

template <class Ext>
inline void update_ints(StatsFusedState& st, const void* src, const uint8_t* val,
                        uint64_t voff, uint32_t n, void* dst, Ext* ext) noexcept {
    constexpr StatsKind S = StatsKind::kSigned, U = StatsKind::kUnsigned;
    const bool sgn = st.kind == S;
    assert(sgn || st.kind == U);
    switch (st.width) {
        case 1: if (sgn) update_as<int8_t, S>(st, src, val, voff, n, dst, ext);
                else update_as<uint8_t, U>(st, src, val, voff, n, dst, ext);
                break;
        case 2: if (sgn) update_as<int16_t, S>(st, src, val, voff, n, dst, ext);
                else update_as<uint16_t, U>(st, src, val, voff, n, dst, ext);
                break;
        case 4: if (sgn) update_as<int32_t, S>(st, src, val, voff, n, dst, ext);
                else update_as<uint32_t, U>(st, src, val, voff, n, dst, ext);
                break;
        default: assert(st.width == 8);
                if (sgn) update_as<int64_t, S>(st, src, val, voff, n, dst, ext);
                else update_as<uint64_t, U>(st, src, val, voff, n, dst, ext);
                break;
    }
}

// ---- finish / merge helpers -----------------------------------------------------

// Writes the value whose state key is `key` into a 16 B min/max slot.
inline void write_key(StatsKind k, uint8_t w, int64_t key, uint8_t* out) noexcept {
    assert(out != nullptr);
    assert(w == 1 || w == 2 || w == 4 || w == 8);
    std::memset(out, 0, 16);
    if (k == StatsKind::kF32) {
        const uint32_t b = static_cast<uint32_t>(key_f32_bits(static_cast<uint32_t>(key)));
        std::memcpy(out, &b, 4);
    } else if (k == StatsKind::kF64) {
        const uint64_t b = static_cast<uint64_t>(key_f64_bits(static_cast<uint64_t>(key)));
        std::memcpy(out, &b, 8);
    } else if (k == StatsKind::kUnsigned) {
        const uint64_t top = 1ull << (8 * w - 1);
        const uint64_t mask = w == 8 ? ~0ull : (1ull << (8 * w)) - 1;
        const uint64_t v = (static_cast<uint64_t>(key) ^ top) & mask;
        std::memcpy(out, &v, 8);
    } else {
        assert(k == StatsKind::kSigned);
        std::memcpy(out, &key, 8);
        if (key < 0) std::memset(out + 8, 0xFF, 8);
    }
}

inline void finish_sum(const StatsFusedState& st, FusedStats* o) noexcept {
    static_assert(kStatsSumLanes == 8, "the float fold is over exactly eight lanes");
    assert(o != nullptr);
    assert(st.kind != StatsKind::kNone);
    if (st.kind == StatsKind::kF32 || st.kind == StatsKind::kF64) {
        const double* f = st.fsum;
        const double s = ((f[0] + f[1]) + (f[2] + f[3])) + ((f[4] + f[5]) + (f[6] + f[7]));
        std::memcpy(&o->sum, &s, 8);
        if (std::isfinite(s)) o->flags |= kStatsSumValid;
        return;
    }
    if (st.kind == StatsKind::kD128) return;
    uint64_t lo = 0;
    int64_t hi = 0;
    for (uint32_t j = 0; j < kStatsSumLanes; ++j) {
        const uint64_t nl = lo + st.sum_lo[j];
        hi += st.sum_hi[j] + static_cast<int64_t>(nl < lo);
        lo = nl;
    }
    o->sum = static_cast<int64_t>(lo);
    o->sum_hi = hi;
    const bool fits = st.kind == StatsKind::kSigned
        ? hi == (static_cast<int64_t>(lo) >> 63) : hi == 0;
    if (fits) o->flags |= kStatsSumValid;
}

// Order of two 16 B min/max slots for the sorted flags: floats numerically
// (the sorted order), everything else as cmp_slot.
inline int cmp_slot(StatsKind k, const uint8_t* a, const uint8_t* b) noexcept;

inline int cmp_sorted(StatsKind k, const uint8_t* a, const uint8_t* b) noexcept {
    assert(a != nullptr && b != nullptr);
    if (k == StatsKind::kF32) {
        float x, y; std::memcpy(&x, a, 4); std::memcpy(&y, b, 4);
        return x < y ? -1 : (x > y ? 1 : 0);
    }
    if (k == StatsKind::kF64) {
        double x, y; std::memcpy(&x, a, 8); std::memcpy(&y, b, 8);
        return x < y ? -1 : (x > y ? 1 : 0);
    }
    return cmp_slot(k, a, b);
}

// -1/0/+1 comparison of two 16 B min/max slots in the type's order.
inline int cmp_slot(StatsKind k, const uint8_t* a, const uint8_t* b) noexcept {
    assert(a != nullptr && b != nullptr);
    if (k == StatsKind::kD128) {
        const D128Key x = d128_load(a), y = d128_load(b);
        return d128_less(x, y) ? -1 : (d128_less(y, x) ? 1 : 0);
    }
    if (k == StatsKind::kUnsigned) {
        uint64_t x, y; std::memcpy(&x, a, 8); std::memcpy(&y, b, 8);
        return x < y ? -1 : (x > y ? 1 : 0);
    }
    int64_t x = 0, y = 0;
    if (k == StatsKind::kF32) {
        uint32_t p, q; std::memcpy(&p, a, 4); std::memcpy(&q, b, 4);
        x = key_f32_bits(p); y = key_f32_bits(q);
    } else if (k == StatsKind::kF64) {
        uint64_t p, q; std::memcpy(&p, a, 8); std::memcpy(&q, b, 8);
        x = key_f64_bits(p); y = key_f64_bits(q);
    } else {
        std::memcpy(&x, a, 8); std::memcpy(&y, b, 8);
    }
    return x < y ? -1 : (x > y ? 1 : 0);
}

inline void merge_sum(StatsKind k, const FusedStats& a, const FusedStats& b,
                      FusedStats* o) noexcept {
    assert(o != nullptr);
    assert(k != StatsKind::kNone && a.type == b.type);
    if (k == StatsKind::kD128) return;
    if (k == StatsKind::kF32 || k == StatsKind::kF64) {
        double x, y;
        std::memcpy(&x, &a.sum, 8); std::memcpy(&y, &b.sum, 8);
        const double s = x + y;
        std::memcpy(&o->sum, &s, 8);
        if (std::isfinite(s)) o->flags |= kStatsSumValid;
        return;
    }
    const uint64_t lo = static_cast<uint64_t>(a.sum) + static_cast<uint64_t>(b.sum);
    o->sum_hi = a.sum_hi + b.sum_hi + static_cast<int64_t>(lo < static_cast<uint64_t>(a.sum));
    o->sum = static_cast<int64_t>(lo);
    const bool fits = k == StatsKind::kSigned ? o->sum_hi == (o->sum >> 63) : o->sum_hi == 0;
    if (fits) o->flags |= kStatsSumValid;
}

}  // namespace bolt::stats::detail
