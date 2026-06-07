// bolt/kernels/bolt_numeric.h — Numeric kernel matrix (Wave A4)
//
// Compile-time X-macro-expanded kernels for filter / aggregate / arithmetic /
// cast over BOLT_NUMERIC_TYPES. Everything is header-only, noexcept, branchless
// on the hot path, and uses BOLT_RESTRICT for auto-vectorization.
//
// Shape rules (Tiger Style):
//   - every function ≥2 asserts, ≤70 lines
//   - no heap; callers supply output buffers (arena-managed upstream)
//   - no exceptions; no RTTI; no virtuals
//   - compile-time type dispatch, no runtime switch in the inner loop
//
// Filter kernels emit a selection vector (int32 row indices).
// Aggregates return a single widened scalar.
// Arithmetic is two-input-one-output, write-into-existing-buffer.
// Cast narrows with saturation; widens verbatim.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/bolt_types.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_branchless.h"

#include <cstdint>
#include <cassert>
#include <cstring>
#include <limits>
#include <type_traits>

namespace bolt {
namespace kernels {

// ============================================================================
// Accumulator type mapping (widened sum / avg)
// ============================================================================
// int8/16/32 → int64; uint8/16/32 → uint64; int64/uint64 → same;
// float32 → double; float64 → double.
template <typename T> struct accum_of { using type = T; };
template <> struct accum_of<int8_t>   { using type = int64_t; };
template <> struct accum_of<int16_t>  { using type = int64_t; };
template <> struct accum_of<int32_t>  { using type = int64_t; };
template <> struct accum_of<int64_t>  { using type = int64_t; };
template <> struct accum_of<uint8_t>  { using type = uint64_t; };
template <> struct accum_of<uint16_t> { using type = uint64_t; };
template <> struct accum_of<uint32_t> { using type = uint64_t; };
template <> struct accum_of<uint64_t> { using type = uint64_t; };
template <> struct accum_of<float>    { using type = double; };
template <> struct accum_of<double>   { using type = double; };

template <typename T> using accum_t = typename accum_of<T>::type;

// ============================================================================
// Filter kernels — branchless selection-vector emit (Pirk DaMoN 2014)
// ============================================================================
// Pattern: always write the row index, advance by bool-to-int. No branch.
// Compiler auto-vectorizes under /arch:AVX2 or -march=native.

#define BOLT_DEFINE_FILTER(NAME, OP)                                           \
template <typename T>                                                          \
inline int64_t NAME(const T* BOLT_RESTRICT data, int64_t n, T scalar,          \
                    int32_t* BOLT_RESTRICT out) noexcept {                     \
    assert(data != nullptr || n == 0);                                         \
    assert(out  != nullptr || n == 0);                                         \
    assert(n >= 0);                                                            \
    int64_t count = 0;                                                         \
    for (int64_t i = 0; i < n; ++i) {                                          \
        out[count] = static_cast<int32_t>(i);                                  \
        count += (data[i] OP scalar);                                          \
    }                                                                          \
    return count;                                                              \
}

BOLT_DEFINE_FILTER(filter_gt, >)
BOLT_DEFINE_FILTER(filter_lt, <)
BOLT_DEFINE_FILTER(filter_eq, ==)
BOLT_DEFINE_FILTER(filter_ne, !=)
BOLT_DEFINE_FILTER(filter_ge, >=)
BOLT_DEFINE_FILTER(filter_le, <=)

#undef BOLT_DEFINE_FILTER

// Branching variants (used by adaptive dispatch at extreme selectivity).
template <typename T>
inline int64_t filter_gt_branching(const T* BOLT_RESTRICT data, int64_t n,
                                   T scalar, int32_t* BOLT_RESTRICT out) noexcept {
    assert(data != nullptr || n == 0);
    assert(out  != nullptr || n == 0);
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (data[i] > scalar) out[count++] = static_cast<int32_t>(i);
    }
    return count;
}

// ============================================================================
// SIMD-accelerated i32 filter_gt — uses bolt::simd::bmm_* wrappers.
// ============================================================================
// Compile-time dispatch: under AVX2/SSE4.2/NEON we emit vector compare +
// movemask + per-lane scalar emit. Under the scalar fallback the bmm_*
// wrappers expand to ordinary loops, so the output is identical to
// filter_gt<int32_t> — no branches added, no runtime cost.

inline int64_t filter_gt_i32_simd(const int32_t* BOLT_RESTRICT data, int64_t n,
                                  int32_t scalar,
                                  int32_t* BOLT_RESTRICT out) noexcept {
    assert(data != nullptr || n == 0);
    assert(out  != nullptr || n == 0);
    assert(n >= 0);

    constexpr int L = bolt::simd::bmm_lanes_i32;
    int64_t count = 0;
    int64_t i = 0;
    const auto vs = bolt::simd::bmm_set1_i32(scalar);
    for (; i + L <= n; i += L) {
        auto vd = bolt::simd::bmm_loadu_i32(data + i);
        auto vc = bolt::simd::bmm_cmpgt_i32(vd, vs);
        int mask = bolt::simd::bmm_movemask_i32(vc);
        // Branchless index emit per lane.
        for (int k = 0; k < L; ++k) {
            out[count] = static_cast<int32_t>(i + k);
            count += ((mask >> k) & 1);
        }
    }
    for (; i < n; ++i) {
        out[count] = static_cast<int32_t>(i);
        count += (data[i] > scalar);
    }
    return count;
}

// int64 / float64 specializations: route the generic filter_gt<T> entry
// point through the SIMD kernel when a vector ISA is compiled in. Under
// scalar builds the bmm_* fallbacks fold back to the branchless template,
// so behaviour is unchanged.
#if BOLT_SIMD_AVX2 || BOLT_SIMD_SSE42 || BOLT_SIMD_NEON
template <>
inline int64_t filter_gt<int64_t>(const int64_t* BOLT_RESTRICT data, int64_t n,
                                  int64_t scalar,
                                  int32_t* BOLT_RESTRICT out) noexcept {
    assert(data != nullptr || n == 0);
    assert(out  != nullptr || n == 0);
    return bolt::branchless::filter_gt_avx2_i64(data, n, scalar, out);
}
template <>
inline int64_t filter_gt<double>(const double* BOLT_RESTRICT data, int64_t n,
                                 double scalar,
                                 int32_t* BOLT_RESTRICT out) noexcept {
    assert(data != nullptr || n == 0);
    assert(out  != nullptr || n == 0);
    return bolt::branchless::filter_gt_avx2_f64(data, n, scalar, out);
}
#endif

// ============================================================================
// Aggregate kernels
// ============================================================================

template <typename T>
inline accum_t<T> sum(const T* BOLT_RESTRICT data, int64_t n) noexcept {
    assert(data != nullptr || n == 0);
    assert(n >= 0);
    using A = accum_t<T>;
    if constexpr (std::is_integral_v<T>) {
        // Integer add is associative → 4 independent accumulators give a
        // BYTE-IDENTICAL result to the serial sum (any grouping of integer
        // additions yields the same value, including on wrap-around since
        // two's-complement add is associative modulo 2^width). Splitting the
        // single serial dependency chain into 4 lets the OoO core overlap the
        // adds. The scalar tail folds the remainder into lane 0, and the final
        // combine reduces left-to-right (a0+a1+a2+a3) — order is irrelevant for
        // integers, so the bits match the old `s += data[i]` loop exactly.
        A a0 = A{0}, a1 = A{0}, a2 = A{0}, a3 = A{0};
        int64_t i = 0;
        const int64_t n4 = n & ~int64_t{3};
        for (; i < n4; i += 4) {
            a0 += static_cast<A>(data[i + 0]);
            a1 += static_cast<A>(data[i + 1]);
            a2 += static_cast<A>(data[i + 2]);
            a3 += static_cast<A>(data[i + 3]);
        }
        for (; i < n; ++i) a0 += static_cast<A>(data[i]);
        assert(i == n);
        return (a0 + a1) + (a2 + a3);
    } else {
        // DEFERRED: floating-point sum is NOT reassociated. 4-accumulator
        // splitting changes the order of FP additions, which changes the
        // last-ULP result. TPC-H sums are golden-contractual, so flipping this
        // needs a gated, separately-validated decision. Keep the serial chain.
        A s = A{0};
        for (int64_t i = 0; i < n; ++i) s += static_cast<A>(data[i]);
        return s;
    }
}

template <typename T>
inline T min(const T* BOLT_RESTRICT data, int64_t n) noexcept {
    assert(data != nullptr);
    assert(n > 0);
    if constexpr (std::is_integral_v<T>) {
        // Integer min is associative + commutative with no rounding, so 4
        // independent lanes combined left-to-right are BYTE-IDENTICAL to the
        // serial scan. (FP min via `<`-select is left serial below: NaN makes
        // the select non-associative, which could change which value/bits win
        // vs the golden serial order.)
        T m0 = data[0], m1 = data[0], m2 = data[0], m3 = data[0];
        int64_t i = 0;
        const int64_t n4 = n & ~int64_t{3};
        for (; i < n4; i += 4) {
            m0 = (data[i + 0] < m0) ? data[i + 0] : m0;
            m1 = (data[i + 1] < m1) ? data[i + 1] : m1;
            m2 = (data[i + 2] < m2) ? data[i + 2] : m2;
            m3 = (data[i + 3] < m3) ? data[i + 3] : m3;
        }
        for (; i < n; ++i) m0 = (data[i] < m0) ? data[i] : m0;
        assert(i == n);
        const T a = (m1 < m0) ? m1 : m0;
        const T b = (m3 < m2) ? m3 : m2;
        return (b < a) ? b : a;
    } else {
        T m = data[0];
        for (int64_t i = 1; i < n; ++i) m = (data[i] < m) ? data[i] : m;
        return m;
    }
}

template <typename T>
inline T max(const T* BOLT_RESTRICT data, int64_t n) noexcept {
    assert(data != nullptr);
    assert(n > 0);
    if constexpr (std::is_integral_v<T>) {
        // Integer max: same associative/commutative argument as min above →
        // BYTE-IDENTICAL to the serial scan. FP max stays serial (NaN).
        T m0 = data[0], m1 = data[0], m2 = data[0], m3 = data[0];
        int64_t i = 0;
        const int64_t n4 = n & ~int64_t{3};
        for (; i < n4; i += 4) {
            m0 = (data[i + 0] > m0) ? data[i + 0] : m0;
            m1 = (data[i + 1] > m1) ? data[i + 1] : m1;
            m2 = (data[i + 2] > m2) ? data[i + 2] : m2;
            m3 = (data[i + 3] > m3) ? data[i + 3] : m3;
        }
        for (; i < n; ++i) m0 = (data[i] > m0) ? data[i] : m0;
        assert(i == n);
        const T a = (m1 > m0) ? m1 : m0;
        const T b = (m3 > m2) ? m3 : m2;
        return (b > a) ? b : a;
    } else {
        T m = data[0];
        for (int64_t i = 1; i < n; ++i) m = (data[i] > m) ? data[i] : m;
        return m;
    }
}

template <typename T>
inline int64_t count(const T* BOLT_RESTRICT /*data*/, int64_t n) noexcept {
    assert(n >= 0);
    assert(n <= (int64_t)1 << 62);
    return n;
}

template <typename T>
inline double avg(const T* BOLT_RESTRICT data, int64_t n) noexcept {
    assert(data != nullptr);
    assert(n > 0);
    accum_t<T> s = sum<T>(data, n);
    return static_cast<double>(s) / static_cast<double>(n);
}

// ============================================================================
// Fused filter+sum kernels (Pirk DaMoN 2014, Wave F1)
// ============================================================================
// Two kernels that avoid materializing a selection vector for streaming
// filter-then-sum workloads (e.g. TPC-H Q6 style predicates):
//
//   sum_masked<T>(data, sel, sel_n)    — gather-sum over a pre-built sel vec.
//   filter_sum_gt<T>(data, n, scalar)  — fully fused; no sel vec allocated.
//
// Both are branchless on the hot path. The scalar bodies auto-vectorize
// cleanly under /arch:AVX2 or -march=native thanks to BOLT_RESTRICT; the
// int64 `sum_masked` has an explicit gather path using bolt::simd::bmm_*
// wrappers for platforms where the compiler won't emit gather on its own.

// sum_masked<T> — scalar reference, branchless, with read-ahead prefetch.
// The compiler auto-vectorizes this under AVX2/NEON to a gather + add chain.
template <typename T>
inline accum_t<T> sum_masked(const T* BOLT_RESTRICT data,
                             const int32_t* BOLT_RESTRICT sel,
                             int64_t sel_n) noexcept {
    assert(data != nullptr || sel_n == 0);
    assert(sel  != nullptr || sel_n == 0);
    assert(sel_n >= 0);
    accum_t<T> acc = accum_t<T>{0};
    constexpr int64_t kPrefetchAhead = 16;
    for (int64_t i = 0; i < sel_n; ++i) {
        if (i + kPrefetchAhead < sel_n) {
            BOLT_PREFETCH_READ(&data[sel[i + kPrefetchAhead]]);
        }
        acc += static_cast<accum_t<T>>(data[sel[i]]);
    }
    return acc;
}

// sum_masked<int64_t> — explicit AVX2/SSE gather path.
// bmm_gather_i64 consumes 4 int32 indices (low 128 bits of bmm_vec_i32)
// on AVX2, so we step by bmm_lanes_i64 (= 4 on AVX2, 2 on SSE/NEON).
// When the SIMD dispatch falls back to the portable scalar path, the
// bmm_* wrappers still expand to correct scalar code, so the result
// matches the generic template bit-for-bit.
template <>
inline int64_t sum_masked<int64_t>(const int64_t* BOLT_RESTRICT data,
                                   const int32_t* BOLT_RESTRICT sel,
                                   int64_t sel_n) noexcept {
    assert(data != nullptr || sel_n == 0);
    assert(sel  != nullptr || sel_n == 0);
    assert(sel_n >= 0);
    constexpr int L = bolt::simd::bmm_lanes_i64;
    auto vacc = bolt::simd::bmm_setzero_i64();
    int64_t i = 0;
    for (; i + L <= sel_n; i += L) {
        if (i + 32 < sel_n) {
            BOLT_PREFETCH_READ(&data[sel[i + 32]]);
        }
        auto vidx = bolt::simd::bmm_loadu_i32(sel + i);  // 4/8 int32 indices
        auto vgot = bolt::simd::bmm_gather_i64(data, vidx);
        vacc = bolt::simd::bmm_add_i64(vacc, vgot);
    }
    int64_t acc = bolt::simd::bmm_hadd_i64(vacc);
    for (; i < sel_n; ++i) acc += data[sel[i]];
    return acc;
}

// filter_sum_gt<T> — fully fused filter-then-sum, branchless.
// Per lane: acc += (data[i] > scalar) ? data[i] : 0.
// Written as a mask-multiply so there is no branch; auto-vectorizes to
// compare+and+add under AVX2 / NEON. No selection vector is materialized,
// so the working set is O(1) regardless of n.
template <typename T>
inline accum_t<T> filter_sum_gt(const T* BOLT_RESTRICT data, int64_t n,
                                T scalar) noexcept {
    assert(data != nullptr || n == 0);
    assert(n >= 0);
    using A = accum_t<T>;
    A acc = A{0};
    if constexpr (std::is_floating_point_v<T>) {
        // Float: multiply by 0.0/1.0 mask (branchless, IEEE-clean).
        // The (v > s) comparison produces a 0/1 int; cast widens to double.
        for (int64_t i = 0; i < n; ++i) {
            A cond = static_cast<A>(static_cast<int>(data[i] > scalar));
            acc += cond * static_cast<A>(data[i]);
        }
    } else {
        // Integral: widen the bool to the accumulator type, negate to get
        // all-ones mask, AND with widened data. Equivalent to predicated add.
        for (int64_t i = 0; i < n; ++i) {
            A v = static_cast<A>(data[i]);
            A cond = -static_cast<A>(data[i] > scalar);  // 0 or -1 (all-ones)
            acc += (v & cond);
        }
    }
    return acc;
}

// filter_count_gt<T> — fully fused filter-then-count, branchless.
// One compare + one 0/1 add per element. Auto-vectorises cleanly on all
// ISAs; no selection vector, no data widening beyond bool-to-int.
template <typename T>
inline int64_t filter_count_gt(const T* BOLT_RESTRICT data, int64_t n,
                               T scalar) noexcept {
    assert(data != nullptr || n == 0);
    assert(n >= 0);
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        count += static_cast<int64_t>(data[i] > scalar);
    }
    return count;
}

// filter_minmax_gt<T> — fully fused filter-then-{min,max}, branchless.
// Walks rows that pass `> scalar` and tracks the min+max of those.
// Uses a sentinel seeded to (numeric_limits::max, numeric_limits::lowest)
// so an empty match reports "no result" via the returned count=0; when
// count>0, min/max are the aggregate over matching rows.
//
// Returns the number of matching rows; writes results through `out_min`
// and `out_max` (non-null, caller-owned).
template <typename T>
inline int64_t filter_minmax_gt(const T* BOLT_RESTRICT data, int64_t n,
                                T scalar,
                                T* BOLT_RESTRICT out_min,
                                T* BOLT_RESTRICT out_max) noexcept {
    assert(data != nullptr || n == 0);
    assert(out_min != nullptr);
    assert(out_max != nullptr);
    assert(n >= 0);

    // Seed sentinels so the first accepted row wins both comparisons
    // without a leading branch. If no row passes, count=0 tells the
    // caller not to read the sentinels.
    T mn = std::numeric_limits<T>::max();
    T mx = std::numeric_limits<T>::lowest();
    int64_t count = 0;

    for (int64_t i = 0; i < n; ++i) {
        const T v  = data[i];
        const bool pass = (v > scalar);
        // Branchless update: if !pass, keep current min/max; if pass,
        // take v when it extends the range. `pass ? v : mn` is a cmov
        // on x86 / csel on ARM.
        const T v_for_min = pass ? v : mn;
        const T v_for_max = pass ? v : mx;
        mn = (v_for_min < mn) ? v_for_min : mn;
        mx = (v_for_max > mx) ? v_for_max : mx;
        count += static_cast<int64_t>(pass);
    }
    *out_min = mn;
    *out_max = mx;
    return count;
}

// ============================================================================
// Arithmetic kernels — two-input, one-output, write-into-buffer
// ============================================================================
// Caller owns `out` (arena-allocated upstream). Same shape for all four ops
// so one X-macro generates them.

#define BOLT_DEFINE_ARITH(NAME, OP)                                            \
template <typename T>                                                          \
inline void NAME(const T* BOLT_RESTRICT a, const T* BOLT_RESTRICT b,           \
                 int64_t n, T* BOLT_RESTRICT out) noexcept {                   \
    assert(a != nullptr || n == 0);                                            \
    assert(b != nullptr || n == 0);                                            \
    assert(out != nullptr || n == 0);                                          \
    assert(n >= 0);                                                            \
    for (int64_t i = 0; i < n; ++i) out[i] = static_cast<T>(a[i] OP b[i]);     \
}

BOLT_DEFINE_ARITH(add, +)
BOLT_DEFINE_ARITH(sub, -)
BOLT_DEFINE_ARITH(mul, *)

#undef BOLT_DEFINE_ARITH

// div is branchy for integer zero-check; floats get IEEE ±inf/NaN semantics.
template <typename T>
inline void div(const T* BOLT_RESTRICT a, const T* BOLT_RESTRICT b,
                int64_t n, T* BOLT_RESTRICT out) noexcept {
    assert(a != nullptr || n == 0);
    assert(b != nullptr || n == 0);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    if constexpr (std::is_integral_v<T>) {
        for (int64_t i = 0; i < n; ++i) {
            T d = b[i];
            out[i] = (d == T{0}) ? T{0} : static_cast<T>(a[i] / d);
        }
    } else {
        for (int64_t i = 0; i < n; ++i) out[i] = static_cast<T>(a[i] / b[i]);
    }
}

// ============================================================================
// Elementwise comparison kernels — column op column → int64 {0,1} column.
// Used by the column-at-a-time expression evaluator (a hash-join/agg input
// `WHERE`/`SELECT` predicate over millions of rows). BOLT_RESTRICT + a tight
// branch-predicated body auto-vectorize. Output is int64 0/1 (the engine's
// boolean column representation), so And/Or/Not compose over the same buffers.
// ~0.3-0.5 ns/row (memory-bound on two streams). Caller owns `out`.
// ============================================================================

// BRANCH-FREE: `a OP b` yields a 0/1 bool that we widen to int64 — the compiler
// lowers it to a predicated setcc, no data-dependent branch (Pirk DaMoN 2014).
#define BOLT_DEFINE_CMP(NAME, OP)                                              \
template <typename T>                                                          \
inline void NAME(const T* BOLT_RESTRICT a, const T* BOLT_RESTRICT b,           \
                 int64_t n, int64_t* BOLT_RESTRICT out) noexcept {             \
    assert(a != nullptr || n == 0);                                           \
    assert(b != nullptr || n == 0);                                           \
    assert(out != nullptr || n == 0);                                         \
    assert(n >= 0);                                                            \
    for (int64_t i = 0; i < n; ++i)                                            \
        out[i] = static_cast<int64_t>(a[i] OP b[i]);                           \
}

BOLT_DEFINE_CMP(cmp_eq, ==)
BOLT_DEFINE_CMP(cmp_ne, !=)
BOLT_DEFINE_CMP(cmp_lt, <)
BOLT_DEFINE_CMP(cmp_le, <=)
BOLT_DEFINE_CMP(cmp_gt, >)
BOLT_DEFINE_CMP(cmp_ge, >=)

#undef BOLT_DEFINE_CMP

// ============================================================================
// Boolean column kernels — over int64 {0,1} columns (nonzero == true). The
// column-at-a-time evaluator's And/Or/Not. BRANCH-FREE: normalise to 0/1 with
// `(x != 0)` (setne) then combine with BITWISE &/| (no short-circuit branch).
// ~0.3 ns/row. Caller owns `out`.
// ============================================================================

inline void logical_and(const int64_t* BOLT_RESTRICT a,
                        const int64_t* BOLT_RESTRICT b,
                        int64_t n, int64_t* BOLT_RESTRICT out) noexcept {
    assert((a != nullptr && b != nullptr && out != nullptr) || n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i)
        out[i] = static_cast<int64_t>((a[i] != 0) & (b[i] != 0));
}

inline void logical_or(const int64_t* BOLT_RESTRICT a,
                       const int64_t* BOLT_RESTRICT b,
                       int64_t n, int64_t* BOLT_RESTRICT out) noexcept {
    assert((a != nullptr && b != nullptr && out != nullptr) || n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i)
        out[i] = static_cast<int64_t>((a[i] != 0) | (b[i] != 0));
}

inline void logical_not(const int64_t* BOLT_RESTRICT a,
                        int64_t n, int64_t* BOLT_RESTRICT out) noexcept {
    assert((a != nullptr && out != nullptr) || n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) out[i] = static_cast<int64_t>(a[i] == 0);
}

// Negate a numeric column (unary minus). ~0.2 ns/row.
template <typename T>
inline void negate(const T* BOLT_RESTRICT a, int64_t n,
                   T* BOLT_RESTRICT out) noexcept {
    assert((a != nullptr && out != nullptr) || n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) out[i] = static_cast<T>(-a[i]);
}

// Broadcast a scalar across a column (literal materialization). ~0.2 ns/row.
template <typename T>
inline void broadcast(T scalar, int64_t n, T* BOLT_RESTRICT out) noexcept {
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) out[i] = scalar;
}

// ============================================================================
// Cast kernels — saturating on narrowing, IEEE round-to-nearest for int↔float
// ============================================================================

namespace detail {

// Saturate `src` (any arithmetic) into target range of TTo.
template <typename TTo, typename TFrom>
inline TTo cast_one(TFrom src) noexcept {
    if constexpr (std::is_same_v<TFrom, TTo>) {
        return src;
    } else if constexpr (std::is_floating_point_v<TFrom> && std::is_floating_point_v<TTo>) {
        // float<->double: direct cast, IEEE round-to-nearest handled by HW.
        return static_cast<TTo>(src);
    } else if constexpr (std::is_floating_point_v<TFrom> && std::is_integral_v<TTo>) {
        // Float → int: saturate at TTo limits, round-to-nearest via rint-ish.
        // We use truncation (static_cast) after range-clamp; callers wanting
        // banker's rounding should rint() upstream.
        constexpr double lo = static_cast<double>(std::numeric_limits<TTo>::lowest());
        constexpr double hi = static_cast<double>(std::numeric_limits<TTo>::max());
        double d = static_cast<double>(src);
        if (!(d == d)) return TTo{0};            // NaN → 0
        if (d <= lo) return std::numeric_limits<TTo>::lowest();
        if (d >= hi) return std::numeric_limits<TTo>::max();
        return static_cast<TTo>(d);
    } else if constexpr (std::is_integral_v<TFrom> && std::is_floating_point_v<TTo>) {
        return static_cast<TTo>(src);             // Widening to float, exact-ish
    } else {
        // Integer → integer: saturate.
        using CommonU = std::make_unsigned_t<
            std::conditional_t<(sizeof(TFrom) > sizeof(TTo)), TFrom, TTo>>;
        constexpr bool from_signed = std::is_signed_v<TFrom>;
        constexpr bool to_signed   = std::is_signed_v<TTo>;
        if constexpr (from_signed && !to_signed) {
            if (src < 0) return TTo{0};
            if (static_cast<CommonU>(src) >
                static_cast<CommonU>(std::numeric_limits<TTo>::max())) {
                return std::numeric_limits<TTo>::max();
            }
            return static_cast<TTo>(src);
        } else if constexpr (!from_signed && to_signed) {
            if (static_cast<CommonU>(src) >
                static_cast<CommonU>(std::numeric_limits<TTo>::max())) {
                return std::numeric_limits<TTo>::max();
            }
            return static_cast<TTo>(src);
        } else {
            // Same signedness. Only saturate on narrowing; widening is lossless.
            if constexpr (sizeof(TFrom) <= sizeof(TTo)) {
                return static_cast<TTo>(src);
            } else {
                if (src > static_cast<TFrom>(std::numeric_limits<TTo>::max())) {
                    return std::numeric_limits<TTo>::max();
                }
                if constexpr (from_signed) {
                    if (src < static_cast<TFrom>(std::numeric_limits<TTo>::lowest())) {
                        return std::numeric_limits<TTo>::lowest();
                    }
                }
                return static_cast<TTo>(src);
            }
        }
    }
}

}  // namespace detail

template <typename TFrom, typename TTo>
inline void cast(const TFrom* BOLT_RESTRICT src, int64_t n,
                 TTo* BOLT_RESTRICT dst) noexcept {
    assert(src != nullptr || n == 0);
    assert(dst != nullptr || n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) dst[i] = detail::cast_one<TTo, TFrom>(src[i]);
}

// ============================================================================
// Micro-adaptive filter_gt (Pirk ICDE 2025)
// ============================================================================
// If zone-map proves the entire morsel passes or none passes, short-circuit.
// Otherwise pick branching vs. branchless from estimated selectivity.
// When stats == nullptr, fall through to branchless (the default good path).

namespace detail {

template <typename T>
inline int64_t emit_identity_selection(int64_t n, int32_t* BOLT_RESTRICT out) noexcept {
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) out[i] = static_cast<int32_t>(i);
    return n;
}

template <typename T>
inline int64_t zone_min_i64(const ColumnStats* s) noexcept {
    if constexpr (std::is_floating_point_v<T>) {
        double d; memcpy(&d, &s->min_value, sizeof(double));
        return static_cast<int64_t>(d);
    } else {
        return s->min_value;
    }
}

template <typename T>
inline int64_t zone_max_i64(const ColumnStats* s) noexcept {
    if constexpr (std::is_floating_point_v<T>) {
        double d; memcpy(&d, &s->max_value, sizeof(double));
        return static_cast<int64_t>(d);
    } else {
        return s->max_value;
    }
}

}  // namespace detail

template <typename T>
inline int64_t filter_gt_adaptive(const T* BOLT_RESTRICT data, int64_t n,
                                  T scalar, int32_t* BOLT_RESTRICT out,
                                  const ColumnStats* stats) noexcept {
    assert(data != nullptr || n == 0);
    assert(out  != nullptr || n == 0);
    assert(n >= 0);

    if (stats == nullptr) return filter_gt<T>(data, n, scalar, out);

    const int64_t zmin = detail::zone_min_i64<T>(stats);
    const int64_t zmax = detail::zone_max_i64<T>(stats);
    const int64_t sc64 = static_cast<int64_t>(scalar);

    // Provably none: max <= scalar → no row is > scalar.
    if (zmax <= sc64) return 0;
    // Provably all: min > scalar → every row is > scalar.
    if (zmin > sc64)  return detail::emit_identity_selection<T>(n, out);

    const float sel = bolt::branchless::estimate_selectivity_gt(zmin, zmax, sc64);
    if (sel < bolt::branchless::kBranchlessThresholdLow ||
        sel > bolt::branchless::kBranchlessThresholdHigh) {
        return filter_gt_branching<T>(data, n, scalar, out);
    }
    return filter_gt<T>(data, n, scalar, out);
}

// ============================================================================
// X-macro explicit-instantiation surface
// ============================================================================
// These inline function templates are defined above. The explicit-instantiation
// block below forces each numeric type through the codegen pipeline during
// whichever TU #includes this header, so symbol-visibility / ODR issues
// surface at compile time rather than link time. Inline templates remain
// header-only — no duplicate definitions.

#define BOLT_KERNEL_INSTANTIATE_ONE(CTYPE)                                     \
    (void)static_cast<int64_t(*)(const CTYPE*, int64_t, CTYPE, int32_t*)>(     \
        &filter_gt<CTYPE>);                                                    \
    (void)static_cast<int64_t(*)(const CTYPE*, int64_t, CTYPE, int32_t*)>(     \
        &filter_lt<CTYPE>);                                                    \
    (void)static_cast<int64_t(*)(const CTYPE*, int64_t, CTYPE, int32_t*)>(     \
        &filter_eq<CTYPE>);                                                    \
    (void)static_cast<int64_t(*)(const CTYPE*, int64_t, CTYPE, int32_t*)>(     \
        &filter_ne<CTYPE>);                                                    \
    (void)static_cast<int64_t(*)(const CTYPE*, int64_t, CTYPE, int32_t*)>(     \
        &filter_ge<CTYPE>);                                                    \
    (void)static_cast<int64_t(*)(const CTYPE*, int64_t, CTYPE, int32_t*)>(     \
        &filter_le<CTYPE>);

// touch_kernels() — optional no-op that forces the compiler to realize every
// specialization exists. Lets downstream TUs verify the matrix compiled.
inline void touch_kernels() noexcept {
#define X(NAME, CTYPE, ENUM_VAL) BOLT_KERNEL_INSTANTIATE_ONE(CTYPE)
    BOLT_NUMERIC_TYPES
#undef X
}

#undef BOLT_KERNEL_INSTANTIATE_ONE

}  // namespace kernels
}  // namespace bolt
