// bolt_branchless.h — Branchless primitives for columnar compute kernels
//
// Branch mispredictions cost 10-20 cycles on modern x86/ARM. In a tight
// per-row loop processing 16K rows, that's 160K-320K wasted cycles per
// mispredicted branch. Branchless code eliminates this entirely.
//
// Techniques used here:
//   1. Conditional move (CMOV) — ternary operator → compiler emits cmov
//   2. Arithmetic predication — bool * value (mask multiplication)
//   3. Bitwise masking — (cmp >> 31) & value (sign bit extraction)
//   4. SIMD masking — _mm256_blendv / _mm256_and_si256 with cmp result
//   5. Selection vector compaction — branchless scatter via prefix sum
//
// RULES: No exceptions. No RTTI. No virtual. All noexcept.
// Compiles to straight-line SIMD on any modern compiler with -O2 -march=native.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/bolt_types.h"
#include <cstdint>
#include <cstring>
#include <cassert>

// SIMD dispatch is handled entirely by bolt_port.h (BOLT_SIMD_AVX2/SSE42/NEON
// flavors and bmm_* wrappers). This header no longer needs its own per-ISA
// intrinsic includes — the `#if BOLT_SIMD_*` gate below is the only one.

namespace bolt {
namespace branchless {

// ============================================================================
// Scalar Branchless Primitives
// ============================================================================

/// Branchless min/max using conditional move.
/// Compiler emits cmov on x86, csel on ARM. Zero branches.
template <typename T>
inline T bmin(T a, T b) noexcept { return (a < b) ? a : b; }

template <typename T>
inline T bmax(T a, T b) noexcept { return (a > b) ? a : b; }

/// Branchless clamp.
template <typename T>
inline T bclamp(T v, T lo, T hi) noexcept { return bmin(bmax(v, lo), hi); }

/// Branchless conditional select: returns a if cond, else b.
/// cond must be 0 or 1. Uses arithmetic multiplication.
template <typename T>
inline T bselect(bool cond, T a, T b) noexcept {
    // Ternary → cmov on x86. No branch.
    return cond ? a : b;
}

/// Branchless absolute value (integer).
inline int64_t babs(int64_t x) noexcept {
    int64_t mask = x >> 63;         // All 1s if negative, all 0s if positive
    return (x + mask) ^ mask;       // Flip bits + add 1 if negative
}

/// Branchless sign: returns -1, 0, or 1.
inline int32_t bsign(int64_t x) noexcept {
    return static_cast<int32_t>((x > 0) - (x < 0));
}

// ============================================================================
// Branchless Filter → Selection Vector
// ============================================================================

/// Branchless filter: column[i] > scalar → output_indices.
///
/// The key insight from VectorWise/Tectorwise: DON'T branch on the
/// comparison result. Instead, always write the index, but conditionally
/// advance the output pointer. The write is cheap (goes to L1 cache).
/// The branch mispredict it avoids costs 15 cycles.
///
///   output[count] = i;           // Always write (speculative)
///   count += (data[i] > scalar); // Advance only if true (no branch)
///
/// The bool-to-int conversion (data[i] > scalar) produces 0 or 1.
/// Adding 0 is a no-op. Adding 1 advances. No branch predictor involved.
///
/// With -O2 -march=native, the compiler auto-vectorizes this into
/// SIMD compare + compress-store on AVX-512, or compare + movemask + 
/// branchless scatter on AVX2.

template <typename T>
int64_t filter_gt_branchless(const T* BOLT_RESTRICT data, int64_t n,
                              T scalar,
                              int32_t* BOLT_RESTRICT out) noexcept {
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        out[count] = static_cast<int32_t>(i);
        count += (data[i] > scalar);   // 0 or 1, no branch
    }
    return count;
}

template <typename T>
int64_t filter_lt_branchless(const T* BOLT_RESTRICT data, int64_t n,
                              T scalar,
                              int32_t* BOLT_RESTRICT out) noexcept {
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        out[count] = static_cast<int32_t>(i);
        count += (data[i] < scalar);
    }
    return count;
}

template <typename T>
int64_t filter_eq_branchless(const T* BOLT_RESTRICT data, int64_t n,
                              T scalar,
                              int32_t* BOLT_RESTRICT out) noexcept {
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        out[count] = static_cast<int32_t>(i);
        count += (data[i] == scalar);
    }
    return count;
}

template <typename T>
int64_t filter_ne_branchless(const T* BOLT_RESTRICT data, int64_t n,
                              T scalar,
                              int32_t* BOLT_RESTRICT out) noexcept {
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        out[count] = static_cast<int32_t>(i);
        count += (data[i] != scalar);
    }
    return count;
}

template <typename T>
int64_t filter_ge_branchless(const T* BOLT_RESTRICT data, int64_t n,
                              T scalar,
                              int32_t* BOLT_RESTRICT out) noexcept {
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        out[count] = static_cast<int32_t>(i);
        count += (data[i] >= scalar);
    }
    return count;
}

template <typename T>
int64_t filter_le_branchless(const T* BOLT_RESTRICT data, int64_t n,
                              T scalar,
                              int32_t* BOLT_RESTRICT out) noexcept {
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        out[count] = static_cast<int32_t>(i);
        count += (data[i] <= scalar);
    }
    return count;
}

/// Branchless filter with selection vector input (composable).
/// Filters sel_in[0..sel_count) further based on predicate.
template <typename T>
int64_t filter_gt_selected_branchless(
        const T* BOLT_RESTRICT data,
        const int32_t* BOLT_RESTRICT sel_in, int64_t sel_count,
        T scalar,
        int32_t* BOLT_RESTRICT sel_out) noexcept {
    int64_t count = 0;
    for (int64_t i = 0; i < sel_count; ++i) {
        int32_t idx = sel_in[i];
        sel_out[count] = idx;
        count += (data[idx] > scalar);
    }
    return count;
}

template <typename T>
int64_t filter_eq_selected_branchless(
        const T* BOLT_RESTRICT data,
        const int32_t* BOLT_RESTRICT sel_in, int64_t sel_count,
        T scalar,
        int32_t* BOLT_RESTRICT sel_out) noexcept {
    int64_t count = 0;
    for (int64_t i = 0; i < sel_count; ++i) {
        const int32_t idx = sel_in[i];
        sel_out[count] = idx;
        count += (data[idx] == scalar);
    }
    return count;
}

// ============================================================================
// IN-list membership filters (W-J1b). Row passes iff data[i] equals ANY of
// vals[0..k). BRANCH-FREE: the k equality bits OR-accumulate (bitwise |, no
// short-circuit), then the standard speculative-write emit. k is small and
// loop-invariant (SQL IN lists; callers cap at <= 16), so the inner compare
// loop unrolls per value lane. Output indices ASCENDING (filter_eq contract).
// ============================================================================

constexpr int32_t kFilterInListMaxVals = 16;

template <typename T>
int64_t filter_in_list_branchless(
        const T* BOLT_RESTRICT data, int64_t n,
        const T* BOLT_RESTRICT vals, int32_t k,
        int32_t* BOLT_RESTRICT out) noexcept {
    assert((data != nullptr && out != nullptr) || n == 0);
    assert(vals != nullptr && k > 0 && k <= kFilterInListMaxVals);
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        const T x = data[i];
        int hit = 0;
        for (int32_t v = 0; v < k; ++v) hit |= (x == vals[v]);
        out[count] = static_cast<int32_t>(i);
        count += hit;
    }
    return count;
}

template <typename T>
int64_t filter_in_list_selected_branchless(
        const T* BOLT_RESTRICT data,
        const int32_t* BOLT_RESTRICT sel_in, int64_t sel_count,
        const T* BOLT_RESTRICT vals, int32_t k,
        int32_t* BOLT_RESTRICT sel_out) noexcept {
    assert((data != nullptr && sel_out != nullptr) || sel_count == 0);
    assert(vals != nullptr && k > 0 && k <= kFilterInListMaxVals);
    int64_t count = 0;
    for (int64_t i = 0; i < sel_count; ++i) {
        const int32_t idx = sel_in[i];
        const T x = data[idx];
        int hit = 0;
        for (int32_t v = 0; v < k; ++v) hit |= (x == vals[v]);
        sel_out[count] = idx;
        count += hit;
    }
    return count;
}

template <typename T>
int64_t filter_ne_selected_branchless(
        const T* BOLT_RESTRICT data,
        const int32_t* BOLT_RESTRICT sel_in, int64_t sel_count,
        T scalar,
        int32_t* BOLT_RESTRICT sel_out) noexcept {
    int64_t count = 0;
    for (int64_t i = 0; i < sel_count; ++i) {
        const int32_t idx = sel_in[i];
        sel_out[count] = idx;
        count += (data[idx] != scalar);
    }
    return count;
}

template <typename T>
int64_t filter_lt_selected_branchless(
        const T* BOLT_RESTRICT data,
        const int32_t* BOLT_RESTRICT sel_in, int64_t sel_count,
        T scalar,
        int32_t* BOLT_RESTRICT sel_out) noexcept {
    int64_t count = 0;
    for (int64_t i = 0; i < sel_count; ++i) {
        const int32_t idx = sel_in[i];
        sel_out[count] = idx;
        count += (data[idx] < scalar);
    }
    return count;
}

template <typename T>
int64_t filter_le_selected_branchless(
        const T* BOLT_RESTRICT data,
        const int32_t* BOLT_RESTRICT sel_in, int64_t sel_count,
        T scalar,
        int32_t* BOLT_RESTRICT sel_out) noexcept {
    int64_t count = 0;
    for (int64_t i = 0; i < sel_count; ++i) {
        const int32_t idx = sel_in[i];
        sel_out[count] = idx;
        count += (data[idx] <= scalar);
    }
    return count;
}

template <typename T>
int64_t filter_ge_selected_branchless(
        const T* BOLT_RESTRICT data,
        const int32_t* BOLT_RESTRICT sel_in, int64_t sel_count,
        T scalar,
        int32_t* BOLT_RESTRICT sel_out) noexcept {
    int64_t count = 0;
    for (int64_t i = 0; i < sel_count; ++i) {
        const int32_t idx = sel_in[i];
        sel_out[count] = idx;
        count += (data[idx] >= scalar);
    }
    return count;
}

template <typename T>
int64_t filter_eq_col_branchless(
        const T* BOLT_RESTRICT lhs,
        const T* BOLT_RESTRICT rhs,
        int64_t n,
        int32_t* BOLT_RESTRICT out) noexcept {
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        out[count] = static_cast<int32_t>(i);
        count += (lhs[i] == rhs[i]);
    }
    return count;
}

template <typename T>
int64_t filter_ne_col_branchless(
        const T* BOLT_RESTRICT lhs,
        const T* BOLT_RESTRICT rhs,
        int64_t n,
        int32_t* BOLT_RESTRICT out) noexcept {
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        out[count] = static_cast<int32_t>(i);
        count += (lhs[i] != rhs[i]);
    }
    return count;
}

template <typename T>
int64_t filter_lt_col_branchless(
        const T* BOLT_RESTRICT lhs,
        const T* BOLT_RESTRICT rhs,
        int64_t n,
        int32_t* BOLT_RESTRICT out) noexcept {
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        out[count] = static_cast<int32_t>(i);
        count += (lhs[i] < rhs[i]);
    }
    return count;
}

template <typename T>
int64_t filter_le_col_branchless(
        const T* BOLT_RESTRICT lhs,
        const T* BOLT_RESTRICT rhs,
        int64_t n,
        int32_t* BOLT_RESTRICT out) noexcept {
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        out[count] = static_cast<int32_t>(i);
        count += (lhs[i] <= rhs[i]);
    }
    return count;
}

template <typename T>
int64_t filter_gt_col_branchless(
        const T* BOLT_RESTRICT lhs,
        const T* BOLT_RESTRICT rhs,
        int64_t n,
        int32_t* BOLT_RESTRICT out) noexcept {
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        out[count] = static_cast<int32_t>(i);
        count += (lhs[i] > rhs[i]);
    }
    return count;
}

template <typename T>
int64_t filter_ge_col_branchless(
        const T* BOLT_RESTRICT lhs,
        const T* BOLT_RESTRICT rhs,
        int64_t n,
        int32_t* BOLT_RESTRICT out) noexcept {
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        out[count] = static_cast<int32_t>(i);
        count += (lhs[i] >= rhs[i]);
    }
    return count;
}

template <typename T>
int64_t filter_eq_col_selected_branchless(
        const T* BOLT_RESTRICT lhs, const T* BOLT_RESTRICT rhs,
        const int32_t* BOLT_RESTRICT sel_in, int64_t sel_count,
        int32_t* BOLT_RESTRICT sel_out) noexcept {
    int64_t count = 0;
    for (int64_t i = 0; i < sel_count; ++i) {
        const int32_t idx = sel_in[i];
        sel_out[count] = idx;
        count += (lhs[idx] == rhs[idx]);
    }
    return count;
}

template <typename T>
int64_t filter_ne_col_selected_branchless(
        const T* BOLT_RESTRICT lhs, const T* BOLT_RESTRICT rhs,
        const int32_t* BOLT_RESTRICT sel_in, int64_t sel_count,
        int32_t* BOLT_RESTRICT sel_out) noexcept {
    int64_t count = 0;
    for (int64_t i = 0; i < sel_count; ++i) {
        const int32_t idx = sel_in[i];
        sel_out[count] = idx;
        count += (lhs[idx] != rhs[idx]);
    }
    return count;
}

template <typename T>
int64_t filter_lt_col_selected_branchless(
        const T* BOLT_RESTRICT lhs, const T* BOLT_RESTRICT rhs,
        const int32_t* BOLT_RESTRICT sel_in, int64_t sel_count,
        int32_t* BOLT_RESTRICT sel_out) noexcept {
    int64_t count = 0;
    for (int64_t i = 0; i < sel_count; ++i) {
        const int32_t idx = sel_in[i];
        sel_out[count] = idx;
        count += (lhs[idx] < rhs[idx]);
    }
    return count;
}

template <typename T>
int64_t filter_le_col_selected_branchless(
        const T* BOLT_RESTRICT lhs, const T* BOLT_RESTRICT rhs,
        const int32_t* BOLT_RESTRICT sel_in, int64_t sel_count,
        int32_t* BOLT_RESTRICT sel_out) noexcept {
    int64_t count = 0;
    for (int64_t i = 0; i < sel_count; ++i) {
        const int32_t idx = sel_in[i];
        sel_out[count] = idx;
        count += (lhs[idx] <= rhs[idx]);
    }
    return count;
}

template <typename T>
int64_t filter_gt_col_selected_branchless(
        const T* BOLT_RESTRICT lhs, const T* BOLT_RESTRICT rhs,
        const int32_t* BOLT_RESTRICT sel_in, int64_t sel_count,
        int32_t* BOLT_RESTRICT sel_out) noexcept {
    int64_t count = 0;
    for (int64_t i = 0; i < sel_count; ++i) {
        const int32_t idx = sel_in[i];
        sel_out[count] = idx;
        count += (lhs[idx] > rhs[idx]);
    }
    return count;
}

template <typename T>
int64_t filter_ge_col_selected_branchless(
        const T* BOLT_RESTRICT lhs, const T* BOLT_RESTRICT rhs,
        const int32_t* BOLT_RESTRICT sel_in, int64_t sel_count,
        int32_t* BOLT_RESTRICT sel_out) noexcept {
    int64_t count = 0;
    for (int64_t i = 0; i < sel_count; ++i) {
        const int32_t idx = sel_in[i];
        sel_out[count] = idx;
        count += (lhs[idx] >= rhs[idx]);
    }
    return count;
}

// ============================================================================
// Branchless Null-Aware Filter
// ============================================================================

/// Filter with null handling. Null values never pass the predicate.
/// Uses branchless AND of (not_null) & (comparison_result).
template <typename T>
int64_t filter_gt_nullable_branchless(
        const T* BOLT_RESTRICT data,
        const uint8_t* validity,  // Arrow format: bit=1 means valid
        int64_t n, T scalar,
        int32_t* BOLT_RESTRICT out) noexcept {
    if (!validity) {
        // No nulls — fast path, skip validity check entirely
        return filter_gt_branchless(data, n, scalar, out);
    }

    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        bool is_valid = (validity[i >> 3] >> (i & 7)) & 1;
        out[count] = static_cast<int32_t>(i);
        count += (is_valid & (data[i] > scalar));  // Both must be true
    }
    return count;
}

// ============================================================================
// Branchless Aggregation
// ============================================================================

/// Branchless min across array. No branch per element.
/// Compiler emits cmov or SIMD min instruction.
template <typename T>
T aggregate_min_branchless(const T* BOLT_RESTRICT data, int64_t n) noexcept {
    assert(n > 0);
    T result = data[0];
    for (int64_t i = 1; i < n; ++i) {
        result = bmin(result, data[i]);  // cmov, no branch
    }
    return result;
}

template <typename T>
T aggregate_max_branchless(const T* BOLT_RESTRICT data, int64_t n) noexcept {
    assert(n > 0);
    T result = data[0];
    for (int64_t i = 1; i < n; ++i) {
        result = bmax(result, data[i]);  // cmov, no branch
    }
    return result;
}

/// Branchless sum with 4-way unroll for auto-vectorization.
/// Separate accumulators prevent loop-carried dependency.
template <typename T, typename AccumT = int64_t>
AccumT aggregate_sum_branchless(const T* BOLT_RESTRICT data, int64_t n) noexcept {
    AccumT a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        a0 += static_cast<AccumT>(data[i]);
        a1 += static_cast<AccumT>(data[i + 1]);
        a2 += static_cast<AccumT>(data[i + 2]);
        a3 += static_cast<AccumT>(data[i + 3]);
    }
    AccumT result = a0 + a1 + a2 + a3;
    for (; i < n; ++i) result += static_cast<AccumT>(data[i]);
    return result;
}

/// Branchless conditional sum: sum of elements where mask[i] is true.
/// No branch per element.
template <typename T, typename AccumT = int64_t>
AccumT aggregate_sum_masked(const T* BOLT_RESTRICT data,
                             const uint8_t* BOLT_RESTRICT mask,
                             int64_t n) noexcept {
    AccumT result = 0;
    for (int64_t i = 0; i < n; ++i) {
        // mask[i] is 0 or 1. Multiply selects value or zero.
        result += static_cast<AccumT>(data[i]) * mask[i];
    }
    return result;
}

// ============================================================================
// Branchless Hash Computation
// ============================================================================

/// Branchless hash finalization (Murmur3 finalizer).
/// Pure arithmetic, zero branches.
inline uint64_t hash_finalize(uint64_t h) noexcept {
    h ^= h >> 33;
    h *= 0xFF51AFD7ED558CCDUL;
    h ^= h >> 33;
    h *= 0xC4CEB9FE1A85EC53UL;
    h ^= h >> 33;
    return h;
}

/// Branchless hash for fixed-width column.
/// No branches, no type-dependent logic in the loop.
template <typename T>
void hash_column_branchless(const T* BOLT_RESTRICT data, int64_t n,
                             uint32_t* BOLT_RESTRICT hashes,
                             uint32_t seed = 0x9E3779B9) noexcept {
    for (int64_t i = 0; i < n; ++i) {
        uint64_t val = 0;
        memcpy(&val, &data[i], sizeof(T));  // Type-punning via memcpy
        hashes[i] = static_cast<uint32_t>(hash_finalize(val ^ seed));
    }
}

/// Branchless hash combine (for multi-column keys).
inline void hash_combine_branchless(const uint32_t* BOLT_RESTRICT h1,
                                     const uint32_t* BOLT_RESTRICT h2,
                                     int64_t n,
                                     uint32_t* BOLT_RESTRICT out) noexcept {
    for (int64_t i = 0; i < n; ++i) {
        // boost::hash_combine equivalent, zero branches
        out[i] = h1[i] ^ (h2[i] + 0x9E3779B9u + (h1[i] << 6) + (h1[i] >> 2));
    }
}

// ============================================================================
// Branchless Gather with Software Prefetch
// ============================================================================

/// Branchless gather: copy selected elements to contiguous output.
/// Software prefetch hides memory latency for random access patterns.
/// No branches in the loop.
///
/// The generic template uses scalar loads with a software-prefetch pipeline;
/// `int32_t` and `int64_t` are specialised below on AVX2/AVX-512 to use
/// hardware gather (`_mm256_i32gather_*`) + vectorised prefetch lookahead.
template <typename T>
void gather_branchless(const T* BOLT_RESTRICT data,
                        const int32_t* BOLT_RESTRICT indices,
                        int64_t count,
                        T* BOLT_RESTRICT output) noexcept {
    constexpr int kPrefetchDist = 16;

    int64_t i = 0;
    // Prefetch ahead
    for (; i < bmin(count, (int64_t)kPrefetchDist); ++i) {
        BOLT_PREFETCH_READ(&data[indices[i]]);
    }
    // Main loop with prefetch pipeline
    for (i = 0; i < count; ++i) {
        if (i + kPrefetchDist < count) {
            BOLT_PREFETCH_READ(&data[indices[i + kPrefetchDist]]);
        }
        output[i] = data[indices[i]];  // No branch, just load-store
    }
}

/// Width-templated POD gather: gathers `Bytes`-wide elements
/// `data[indices[i]]` into contiguous `output[i]`, for callers that
/// dispatch on a runtime element width (e.g. join/collect-sink payload
/// materialisation over Int64 / Decimal128 / StringView). Routes through
/// the generic `gather_branchless<T>` prefetch pipeline; the element type
/// is an align-1 byte POD so any source pointer (Decimal128 8-byte-aligned,
/// StringView 4-byte-aligned) is valid to reinterpret. For `Bytes == 8`
/// prefer `gather_branchless<int64_t>` directly (it has an AVX2 path).
///
/// ns/row: inherits the generic `gather_branchless<T>` floor — scalar load
/// + software prefetch; a 16-byte copy is ~2× the 8-byte element cost.
/// StringView lifetime: a gathered StringView copies its inline bytes or
/// its {buf_idx, offset} ref verbatim; spilled views still resolve against
/// the *source* column's `str_overflow_base`, so the destination column
/// must carry that same base (no deep copy here).
template <int Bytes>
BOLT_FORCE_INLINE void gather_pod(const void* BOLT_RESTRICT data,
                                  const int32_t* BOLT_RESTRICT indices,
                                  int64_t count,
                                  void* BOLT_RESTRICT output) noexcept {
    static_assert(Bytes >= 1, "gather_pod: element width must be positive");
    struct Pod { unsigned char b[Bytes]; };
    gather_branchless<Pod>(static_cast<const Pod*>(data), indices, count,
                           static_cast<Pod*>(output));
}

#if BOLT_SIMD_AVX2 || BOLT_SIMD_AVX512

// ----------------------------------------------------------------------
// Named SIMD variants — kept in source so a future JIT / runtime dispatch
// can reach for them per-CPU or per-shape without digging through git.
// Currently: `gather_simd_i32` is the default for int32_t (wins ~17%
// median on client Intel); `gather_simd_i64` is slower on client Intel
// (AVX2 hardware `_mm256_i32gather_epi64` serialises internally and
// loses ~3× to prefetched scalar loads), so the generic template stays
// the default for int64_t. See design-log "gather_branchless — SIMD
// gather vs scalar + prefetch (A2)" for workloads that would flip this.
// ----------------------------------------------------------------------

/// Hardware-gather path for int32: `_mm256_i32gather_epi32` on 8 lanes,
/// with the prefetch pipeline seeded two SIMD batches ahead. Always
/// available on AVX2+; callers who want this explicitly can invoke it.
inline void gather_simd_i32(
        const int32_t* BOLT_RESTRICT data,
        const int32_t* BOLT_RESTRICT indices,
        int64_t count,
        int32_t* BOLT_RESTRICT output) noexcept {
    assert(data != nullptr || count == 0);
    assert(indices != nullptr || count == 0);
    assert(output != nullptr || count == 0);

    using namespace bolt::simd;
    constexpr int L = bmm_lanes_i32;          // 8 on AVX2
    constexpr int64_t kAhead = L * 2;         // two SIMD batches ahead

    for (int64_t p = 0; p < bmin(count, kAhead); ++p) {
        BOLT_PREFETCH_READ(&data[indices[p]]);
    }
    int64_t i = 0;
    for (; i + L <= count; i += L) {
        const int64_t pi = i + kAhead;
        if (pi + L <= count) {
            for (int k = 0; k < L; ++k) {
                BOLT_PREFETCH_READ(&data[indices[pi + k]]);
            }
        }
        const bmm_vec_i32 vidx = bmm_loadu_i32(indices + i);
        const bmm_vec_i32 vout = bmm_gather_i32(data, vidx);
        bmm_storeu_i32(output + i, vout);
    }
    for (; i < count; ++i) output[i] = data[indices[i]];
}

/// Hardware-gather path for int64: `_mm256_i32gather_epi64` on 4 lanes.
/// Kept for future use (Zen3+, Sapphire Rapids, or JIT dispatch) — on
/// client Intel this is measurably slower than the scalar-prefetch
/// template so it is NOT the default for `gather_branchless<int64_t>`.
inline void gather_simd_i64(
        const int64_t* BOLT_RESTRICT data,
        const int32_t* BOLT_RESTRICT indices,
        int64_t count,
        int64_t* BOLT_RESTRICT output) noexcept {
    assert(data != nullptr || count == 0);
    assert(indices != nullptr || count == 0);
    assert(output != nullptr || count == 0);

    using namespace bolt::simd;
    constexpr int L = bmm_lanes_i64;          // 4 on AVX2
    constexpr int64_t kAhead = L * 4;         // 16 items ahead

    for (int64_t p = 0; p < bmin(count, kAhead); ++p) {
        BOLT_PREFETCH_READ(&data[indices[p]]);
    }
    int64_t i = 0;
    for (; i + L <= count; i += L) {
        const int64_t pi = i + kAhead;
        if (pi + L <= count) {
            for (int k = 0; k < L; ++k) {
                BOLT_PREFETCH_READ(&data[indices[pi + k]]);
            }
        }
        // Build an 8-lane i32 vector from L=4 indices; upper lanes are
        // unused by bmm_gather_i64 (consumes only the low 128 bits).
        alignas(32) int32_t idx_buf[8] = {0};
        for (int k = 0; k < L; ++k) idx_buf[k] = indices[i + k];
        const bmm_vec_i32 vidx = bmm_loadu_i32(idx_buf);
        const bmm_vec_i64 vout = bmm_gather_i64(data, vidx);
        bmm_storeu_i64(output + i, vout);
    }
    for (; i < count; ++i) output[i] = data[indices[i]];
}

/// Default for int32_t: route to the SIMD gather path (wins on AVX2).
template <>
inline void gather_branchless<int32_t>(
        const int32_t* BOLT_RESTRICT data,
        const int32_t* BOLT_RESTRICT indices,
        int64_t count,
        int32_t* BOLT_RESTRICT output) noexcept {
    gather_simd_i32(data, indices, count, output);
}

// int64_t intentionally NOT specialised — the generic scalar-prefetch
// template is faster on measured client-Intel hardware. Callers who
// know the target CPU favours hardware gather can call
// `gather_simd_i64` by name.

#endif  // BOLT_SIMD_AVX2 || BOLT_SIMD_AVX512

// ============================================================================
// Branchless Selection Vector Intersection (AND of two filters)
// ============================================================================

/// Branchless merge-intersect of two sorted selection vectors.
/// Uses conditional advance (no branch per comparison).
///
/// Classic merge intersection has a branch per element:
///   if (a[i] < b[j]) i++; else if (a[i] > b[j]) j++; else { *out++ = a[i]; i++; j++; }
///
/// Branchless version:
///   advance_a = (a[i] <= b[j]);   // 0 or 1
///   advance_b = (a[i] >= b[j]);   // 0 or 1
///   match     = (a[i] == b[j]);   // 0 or 1
///   out[count] = a[i];
///   count += match;
///   i += advance_a;
///   j += advance_b;
///
/// No branch predictor involved. Pure arithmetic.
///
/// Generic on index type (`int32_t`, `int64_t`, `uint32_t`, …) so selection
/// vectors from kernels that use wider row indices can chain without
/// materialising to i32 first.  The original `selection_intersect_branchless`
/// signature (i32-only) is preserved as an inline wrapper for callers who
/// don't want to name the type.
template <typename Idx>
inline int64_t selection_intersect_t(
        const Idx* BOLT_RESTRICT a, int64_t na,
        const Idx* BOLT_RESTRICT b, int64_t nb,
        Idx* BOLT_RESTRICT out) noexcept {
    static_assert(sizeof(Idx) <= sizeof(int64_t),
                  "selection_intersect_t: index type too wide");
    assert(a != nullptr || na == 0);
    assert(b != nullptr || nb == 0);
    assert(out != nullptr || (na == 0 || nb == 0));
    int64_t i = 0, j = 0, count = 0;
    while (i < na && j < nb) {
        Idx va = a[i], vb = b[j];
        bool match = (va == vb);
        out[count] = va;
        count += match;
        i += (va <= vb);
        j += (va >= vb);
    }
    return count;
}

template <typename Idx>
inline int64_t selection_union_t(
        const Idx* BOLT_RESTRICT a, int64_t na,
        const Idx* BOLT_RESTRICT b, int64_t nb,
        Idx* BOLT_RESTRICT out) noexcept {
    static_assert(sizeof(Idx) <= sizeof(int64_t),
                  "selection_union_t: index type too wide");
    assert(a != nullptr || na == 0);
    assert(b != nullptr || nb == 0);
    int64_t i = 0, j = 0, count = 0;
    while (i < na && j < nb) {
        Idx va = a[i], vb = b[j];
        Idx val = bmin<Idx>(va, vb);
        out[count] = val;
        count++;
        i += (va <= vb);
        j += (va >= vb);
    }
    while (i < na) out[count++] = a[i++];
    while (j < nb) out[count++] = b[j++];
    return count;
}

// Back-compat: i32 inline shims so existing call sites keep working.
inline int64_t selection_intersect_branchless(
        const int32_t* BOLT_RESTRICT a, int64_t na,
        const int32_t* BOLT_RESTRICT b, int64_t nb,
        int32_t* BOLT_RESTRICT out) noexcept {
    return selection_intersect_t<int32_t>(a, na, b, nb, out);
}

inline int64_t selection_union_branchless(
        const int32_t* BOLT_RESTRICT a, int64_t na,
        const int32_t* BOLT_RESTRICT b, int64_t nb,
        int32_t* BOLT_RESTRICT out) noexcept {
    return selection_union_t<int32_t>(a, na, b, nb, out);
}

/// Convert a 64-bit-word bitmap (Arrow format: LSB-first, bit i = row i)
/// into a dense selection vector of indices. `num_rows` bounds the valid
/// range so trailing bits past `num_rows` in the last word are ignored.
/// Writes ctz(word) derived indices — fully branchless per bit.
template <typename Idx>
inline int64_t bitmap_to_indices(const uint64_t* BOLT_RESTRICT bitmap,
                                  int64_t num_rows,
                                  Idx* BOLT_RESTRICT out) noexcept {
    static_assert(sizeof(Idx) <= sizeof(int64_t),
                  "bitmap_to_indices: index type too wide");
    assert(bitmap != nullptr || num_rows == 0);
    assert(out != nullptr || num_rows == 0);

    int64_t count = 0;
    const int64_t full_words = num_rows >> 6;
    const int     tail_bits  = static_cast<int>(num_rows & 63);

    int64_t base = 0;
    for (int64_t w = 0; w < full_words; ++w, base += 64) {
        uint64_t word = bitmap[w];
        while (word) {
            const int b = bolt_ctz64(word);
            out[count++] = static_cast<Idx>(base + b);
            word &= word - 1;
        }
    }
    if (tail_bits > 0) {
        uint64_t word = bitmap[full_words] & ((uint64_t{1} << tail_bits) - 1);
        while (word) {
            const int b = bolt_ctz64(word);
            out[count++] = static_cast<Idx>(base + b);
            word &= word - 1;
        }
    }
    return count;
}

// ============================================================================
// Branchless Bitmap Operations (for BitmapIndex)
// ============================================================================

/// Count set bits in a bitmap (popcount).
/// Uses hardware popcount if available.
inline uint32_t bitmap_popcount(const uint64_t* bitmap, uint32_t num_words) noexcept {
    uint32_t count = 0;
    for (uint32_t i = 0; i < num_words; ++i) {
#if defined(__POPCNT__) || defined(__x86_64__) || defined(_M_X64)
        count += bolt_popcount64(bitmap[i]);
#else
        // Fallback: Kernighan's method (still branchless per bit)
        uint64_t v = bitmap[i];
        while (v) { count++; v &= v - 1; }
#endif
    }
    return count;
}

/// Branchless bitmap AND (intersection of two bitmaps).
inline void bitmap_and(const uint64_t* BOLT_RESTRICT a,
                        const uint64_t* BOLT_RESTRICT b,
                        uint64_t* BOLT_RESTRICT out,
                        uint32_t num_words) noexcept {
    for (uint32_t i = 0; i < num_words; ++i) {
        out[i] = a[i] & b[i];  // Pure bitwise, zero branches
    }
}

/// Branchless bitmap OR (union of two bitmaps).
inline void bitmap_or(const uint64_t* BOLT_RESTRICT a,
                       const uint64_t* BOLT_RESTRICT b,
                       uint64_t* BOLT_RESTRICT out,
                       uint32_t num_words) noexcept {
    for (uint32_t i = 0; i < num_words; ++i) {
        out[i] = a[i] | b[i];
    }
}

// ============================================================================
// Run-native kernels (B2b) — operate on RLE-encoded columns directly, so each
// run is evaluated once regardless of run length. Complements the materialise-
// then-scan path; callers pick based on expected run length (run-native wins
// past ~4 rows/run on modern CPUs).
// ============================================================================

/// Run-native `filter_eq`: walks `num_runs` (value, run_end) pairs and emits
/// every row index where `values[i] == scalar`. O(num_runs) compares and
/// O(matching_rows) stores — no per-row compare. The per-run inner loop
/// is a tight `for` that compiles to `rep stosd`-equivalent on x86.
template <typename T>
inline int64_t filter_eq_rle(const T* BOLT_RESTRICT values,
                              const int32_t* BOLT_RESTRICT run_ends,
                              int64_t num_runs,
                              T scalar,
                              int32_t* BOLT_RESTRICT out) noexcept {
    assert(values   != nullptr || num_runs == 0);
    assert(run_ends != nullptr || num_runs == 0);
    assert(out      != nullptr || num_runs == 0);
    assert(num_runs >= 0);

    int64_t count = 0;
    int32_t prev = 0;
    for (int64_t i = 0; i < num_runs; ++i) {
        const int32_t end = run_ends[i];
        assert(end >= prev);
        if (values[i] == scalar) {
            for (int32_t r = prev; r < end; ++r) out[count++] = r;
        }
        prev = end;
    }
    return count;
}

/// Run-native `sum` over an RLE column: Σ values[i] * run_length[i].
/// Width-widened accumulator so 8-bit runs don't overflow on long runs.
template <typename T>
inline int64_t sum_rle_i64(const T* BOLT_RESTRICT values,
                            const int32_t* BOLT_RESTRICT run_ends,
                            int64_t num_runs) noexcept {
    assert(values   != nullptr || num_runs == 0);
    assert(run_ends != nullptr || num_runs == 0);
    assert(num_runs >= 0);

    int64_t acc = 0;
    int32_t prev = 0;
    for (int64_t i = 0; i < num_runs; ++i) {
        const int32_t end = run_ends[i];
        assert(end >= prev);
        acc += static_cast<int64_t>(values[i]) * (end - prev);
        prev = end;
    }
    return acc;
}

// ============================================================================
// Run-native kernels for BitPacked + FrameOfRef (I1 / I2) — decode on the
// fly into a stack scratch buffer, compare/aggregate, emit.  Avoids the
// full-column materialise traffic that a materialise-then-filter path
// would pay when the caller only needs a selection vector or a sum.
// ============================================================================

// Internal helper: unpack `count` values (≤ stride_max) from the packed
// buffer starting at logical index `start` into `out[]`. Caller ensures
// `out` is sized to `count`. Same bit-unpack shape used in
// BoltColumn::materialize's BitPacked branch.
BOLT_FORCE_INLINE void bolt_bitpacked_unpack_range(
        const uint64_t* BOLT_RESTRICT words, int64_t start, int64_t count,
        uint8_t bit_width, int32_t* BOLT_RESTRICT out) noexcept {
    assert(words != nullptr || count == 0);
    assert(out   != nullptr || count == 0);
    assert(bit_width >= 1 && bit_width <= 32);
    const uint64_t mask = (bit_width == 64) ? ~uint64_t{0}
                                            : ((uint64_t{1} << bit_width) - 1u);
    for (int64_t i = 0; i < count; ++i) {
        const uint64_t bit_off     = static_cast<uint64_t>(start + i)
                                   * static_cast<uint64_t>(bit_width);
        const uint64_t word_off    = bit_off >> 6;
        const uint64_t bit_in_word = bit_off & 63u;
        uint64_t v = words[word_off] >> bit_in_word;
        if (bit_in_word + static_cast<uint64_t>(bit_width) > 64u) {
            v |= words[word_off + 1] << (64u - bit_in_word);
        }
        out[i] = static_cast<int32_t>(v & mask);
    }
}

/// Run-native `filter_gt` over a BitPacked uint32-shaped column.
/// Unpacks 64 values at a time into a stack buffer, compares scalar-
/// branchless, emits indices. Emitted indices are absolute (0..length).
inline int64_t filter_gt_bitpacked(const uint64_t* BOLT_RESTRICT words,
                                   int64_t length, uint8_t bit_width,
                                   int32_t scalar,
                                   int32_t* BOLT_RESTRICT out) noexcept {
    assert(words != nullptr || length == 0);
    assert(out   != nullptr || length == 0);
    assert(length >= 0);
    assert(bit_width >= 1 && bit_width <= 32);
    constexpr int64_t kChunk = 64;
    alignas(64) int32_t scratch[kChunk];
    int64_t count = 0;
    for (int64_t s = 0; s < length; s += kChunk) {
        const int64_t n = (length - s < kChunk) ? (length - s) : kChunk;
        bolt_bitpacked_unpack_range(words, s, n, bit_width, scratch);
        for (int64_t i = 0; i < n; ++i) {
            out[count] = static_cast<int32_t>(s + i);
            count += (scratch[i] > scalar);
        }
    }
    return count;
}

/// Run-native `filter_eq` over a BitPacked column.
inline int64_t filter_eq_bitpacked(const uint64_t* BOLT_RESTRICT words,
                                   int64_t length, uint8_t bit_width,
                                   int32_t scalar,
                                   int32_t* BOLT_RESTRICT out) noexcept {
    assert(words != nullptr || length == 0);
    assert(out   != nullptr || length == 0);
    assert(length >= 0);
    assert(bit_width >= 1 && bit_width <= 32);
    constexpr int64_t kChunk = 64;
    alignas(64) int32_t scratch[kChunk];
    int64_t count = 0;
    for (int64_t s = 0; s < length; s += kChunk) {
        const int64_t n = (length - s < kChunk) ? (length - s) : kChunk;
        bolt_bitpacked_unpack_range(words, s, n, bit_width, scratch);
        for (int64_t i = 0; i < n; ++i) {
            out[count] = static_cast<int32_t>(s + i);
            count += (scratch[i] == scalar);
        }
    }
    return count;
}

/// Run-native `sum` over a FrameOfRef column: base * n + Σ(deltas).
/// Unpacks deltas in 64-value chunks; scalar add hoisted, so the hot
/// loop is a straight `acc += scratch[i]`.
inline int64_t sum_frame_of_ref(const uint64_t* BOLT_RESTRICT words,
                                int64_t length, uint8_t bit_width,
                                int64_t base) noexcept {
    assert(words != nullptr || length == 0);
    assert(length >= 0);
    assert(bit_width >= 1 && bit_width <= 32);
    constexpr int64_t kChunk = 64;
    alignas(64) int32_t scratch[kChunk];
    int64_t delta_sum = 0;
    for (int64_t s = 0; s < length; s += kChunk) {
        const int64_t n = (length - s < kChunk) ? (length - s) : kChunk;
        bolt_bitpacked_unpack_range(words, s, n, bit_width, scratch);
        for (int64_t i = 0; i < n; ++i) {
            delta_sum += static_cast<int64_t>(scratch[i]);
        }
    }
    return base * length + delta_sum;
}

/// Extract set bit positions to selection vector (branchless per word).
inline int64_t bitmap_to_selection(const uint64_t* bitmap, uint32_t num_words,
                                    int32_t* BOLT_RESTRICT out) noexcept {
    int64_t count = 0;
    for (uint32_t w = 0; w < num_words; ++w) {
        uint64_t bits = bitmap[w];
        int32_t base = static_cast<int32_t>(w * 64);
        while (bits) {
            // count trailing zeros = position of lowest set bit
            int pos = bolt_ctz64(bits);
            out[count++] = base + pos;
            bits &= bits - 1;  // Clear lowest set bit (Kernighan's trick)
        }
    }
    return count;
}

// ============================================================================
// SIMD Filter Kernels (ISA-agnostic via bolt::simd::bmm_*)
// ============================================================================
//
// Single ISA gate: the per-ISA dispatch lives in bolt_port.h. Scalar fallback
// is the branchless template above — no extra code path needed here.

#if BOLT_SIMD_AVX2 || BOLT_SIMD_SSE42 || BOLT_SIMD_NEON

/// SIMD branchless filter: int32 column > scalar.
/// Processes `bmm_lanes_i32` elements per iteration via compare + movemask +
/// compressstore. Keeps legacy `_avx2_` name for callers.
///
/// The hot path writes a lane-index vector `[0..L)+base` unconditionally,
/// and `bmm_compressstore_i32` lays out the matching indices contiguously
/// — one write per SIMD iteration, no per-lane ctz loop.
inline int64_t filter_gt_avx2_i32(const int32_t* BOLT_RESTRICT data, int64_t n,
                                   int32_t scalar,
                                   int32_t* BOLT_RESTRICT out) noexcept {
    assert(data != nullptr || n == 0);
    assert(out  != nullptr || n == 0);

    using namespace bolt::simd;
    const bmm_vec_i32 vscalar = bmm_set1_i32(scalar);
    constexpr int L = bmm_lanes_i32;
    int64_t count = 0;
    int64_t i = 0;

    // Main SIMD loop: L elements per iteration (8 on AVX2, 4 on SSE/NEON).
    for (; i + L <= n; i += L) {
        const bmm_vec_i32 vdata = bmm_loadu_i32(data + i);
        const bmm_vec_i32 vcmp  = bmm_cmpgt_i32(vdata, vscalar);
        const uint32_t    mask  = static_cast<uint32_t>(bmm_movemask_i32(vcmp));

        // Build per-lane row indices [i, i+1, ..., i+L-1] and compressstore
        // the subset whose mask bit is set. Single vector store (AVX2) or a
        // tight LUT permute (SSE/NEON) — no ctz loop.
        alignas(32) int32_t idx_buf[16];  // 16 >= L for every ISA.
        for (int k = 0; k < L; ++k) idx_buf[k] = static_cast<int32_t>(i) + k;
        const bmm_vec_i32 vidx = bmm_loadu_i32(idx_buf);
        count += bmm_compressstore_i32(out + count, vidx, mask);
    }

    // Scalar tail (shared with fallback).
    for (; i < n; ++i) {
        out[count] = static_cast<int32_t>(i);
        count += (data[i] > scalar);
    }
    return count;
}

/// SIMD branchless filter: int64 column > scalar.
/// Processes two `bmm_lanes_i64` groups per iteration so the combined mask
/// width exactly matches `bmm_compressstore_i32` (8 lanes on AVX2, 4 on
/// SSE/NEON). One SIMD compressstore per step — no per-mask-bit scalar loop.
inline int64_t filter_gt_avx2_i64(const int64_t* BOLT_RESTRICT data, int64_t n,
                                   int64_t scalar,
                                   int32_t* BOLT_RESTRICT out) noexcept {
    assert(data != nullptr || n == 0);
    assert(out  != nullptr || n == 0);
    assert(n >= 0);

    using namespace bolt::simd;
    const bmm_vec_i64 vscalar = bmm_set1_i64(scalar);
    constexpr int Li = bmm_lanes_i64;      // 4 on AVX2, 2 on SSE/NEON
    constexpr int Lo = bmm_lanes_i32;      // 8 on AVX2, 4 on SSE/NEON
    static_assert(Lo == 2 * Li,
                  "filter_gt_avx2_i64: i32 lanes must be 2 * i64 lanes");
    int64_t count = 0;
    int64_t i = 0;

    // Main SIMD loop: two i64 loads per step; combined mask feeds a single
    // compressstore that emits popcount(mask) contiguous i32 indices.
    for (; i + Lo <= n; i += Lo) {
        const bmm_vec_i64 vdata0 = bmm_loadu_i64(data + i);
        const bmm_vec_i64 vdata1 = bmm_loadu_i64(data + i + Li);
        const bmm_vec_i64 vcmp0  = bmm_cmpgt_i64(vdata0, vscalar);
        const bmm_vec_i64 vcmp1  = bmm_cmpgt_i64(vdata1, vscalar);
        const uint32_t    m0     = static_cast<uint32_t>(bmm_movemask_i64(vcmp0));
        const uint32_t    m1     = static_cast<uint32_t>(bmm_movemask_i64(vcmp1));
        const uint32_t    mask   = m0 | (m1 << Li);

        alignas(32) int32_t idx_buf[16];   // 16 >= Lo for every ISA
        for (int k = 0; k < Lo; ++k) idx_buf[k] = static_cast<int32_t>(i) + k;
        const bmm_vec_i32 vidx = bmm_loadu_i32(idx_buf);
        count += bmm_compressstore_i32(out + count, vidx, mask);
    }

    // Scalar tail (shared shape with the i32 kernel's tail).
    for (; i < n; ++i) {
        out[count] = static_cast<int32_t>(i);
        count += (data[i] > scalar);
    }
    return count;
}

/// SIMD branchless filter: float64 column > scalar.
/// Mirrors filter_gt_avx2_i64; uses ordered greater-than (NaN → no match).
/// Processes two `bmm_lanes_f64` groups per iteration and funnels the
/// combined mask into `bmm_compressstore_i32`.
inline int64_t filter_gt_avx2_f64(const double* BOLT_RESTRICT data, int64_t n,
                                   double scalar,
                                   int32_t* BOLT_RESTRICT out) noexcept {
    assert(data != nullptr || n == 0);
    assert(out  != nullptr || n == 0);
    assert(n >= 0);

    using namespace bolt::simd;
    const bmm_vec_f64 vscalar = bmm_set1_f64(scalar);
    constexpr int Li = bmm_lanes_f64;      // 4 on AVX2, 2 on SSE/NEON
    constexpr int Lo = bmm_lanes_i32;      // 8 on AVX2, 4 on SSE/NEON
    static_assert(Lo == 2 * Li,
                  "filter_gt_avx2_f64: i32 lanes must be 2 * f64 lanes");
    int64_t count = 0;
    int64_t i = 0;

    for (; i + Lo <= n; i += Lo) {
        const bmm_vec_f64 vdata0 = bmm_loadu_f64(data + i);
        const bmm_vec_f64 vdata1 = bmm_loadu_f64(data + i + Li);
        const bmm_vec_f64 vcmp0  = bmm_cmpgt_f64(vdata0, vscalar);
        const bmm_vec_f64 vcmp1  = bmm_cmpgt_f64(vdata1, vscalar);
        const uint32_t    m0     = static_cast<uint32_t>(bmm_movemask_f64(vcmp0));
        const uint32_t    m1     = static_cast<uint32_t>(bmm_movemask_f64(vcmp1));
        const uint32_t    mask   = m0 | (m1 << Li);

        alignas(32) int32_t idx_buf[16];
        for (int k = 0; k < Lo; ++k) idx_buf[k] = static_cast<int32_t>(i) + k;
        const bmm_vec_i32 vidx = bmm_loadu_i32(idx_buf);
        count += bmm_compressstore_i32(out + count, vidx, mask);
    }

    for (; i < n; ++i) {
        out[count] = static_cast<int32_t>(i);
        count += (data[i] > scalar);
    }
    return count;
}

/// SIMD branchless sum: int64 column.
/// Accumulates `bmm_lanes_i64` lanes in parallel, then horizontal-reduces.
inline int64_t sum_avx2_i64(const int64_t* BOLT_RESTRICT data, int64_t n) noexcept {
    assert(data != nullptr || n == 0);
    assert(n >= 0);

    using namespace bolt::simd;
    constexpr int L = bmm_lanes_i64;
    bmm_vec_i64 acc = bmm_setzero_i64();
    int64_t i = 0;
    for (; i + L <= n; i += L) {
        acc = bmm_add_i64(acc, bmm_loadu_i64(data + i));
    }
    int64_t result = bmm_hadd_i64(acc);
    for (; i < n; ++i) result += data[i];
    return result;
}

/// SIMD branchless filter: float32 column > scalar. (A5)
/// Mirrors filter_gt_avx2_f64 — two loads per step, combined mask into
/// bmm_compressstore_i32. 8 f32 lanes per load on AVX2 fill the 8-lane
/// i32 compressstore exactly; on SSE/NEON (4+4=8) the step naturally
/// doubles relative to a single load.  Uses ordered GT: NaN → no match.
inline int64_t filter_gt_avx2_f32(const float* BOLT_RESTRICT data, int64_t n,
                                   float scalar,
                                   int32_t* BOLT_RESTRICT out) noexcept {
    assert(data != nullptr || n == 0);
    assert(out  != nullptr || n == 0);
    assert(n >= 0);

    using namespace bolt::simd;
    const bmm_vec_f32 vscalar = bmm_set1_f32(scalar);
    constexpr int Li = bmm_lanes_f32;          // 8 on AVX2, 4 on SSE/NEON
    constexpr int Lo = Li + Li;                // two loads per iter
    int64_t count = 0;
    int64_t i = 0;

    for (; i + Lo <= n; i += Lo) {
        const bmm_vec_f32 v0 = bmm_loadu_f32(data + i);
        const bmm_vec_f32 v1 = bmm_loadu_f32(data + i + Li);
        const uint32_t m0 = static_cast<uint32_t>(bmm_movemask_f32(bmm_cmpgt_f32(v0, vscalar)));
        const uint32_t m1 = static_cast<uint32_t>(bmm_movemask_f32(bmm_cmpgt_f32(v1, vscalar)));
        const uint32_t mask = (m0 | (m1 << Li)) & ((1u << Lo) - 1u);

        // compressstore_i32 takes an 8-lane i32 vec; fill one-to-one on
        // AVX2 (Lo=16 > 8 there → skip the wide path) or two-to-one on
        // SSE/NEON (Lo=8 exactly matches the 4-lane i32 buf upper half).
        // The simpler portable path: emit per mask bit when Lo > 8; the
        // AVX2 path gets the SIMD win when Lo == 8.
        if constexpr (Lo <= 8) {
            alignas(32) int32_t idx_buf[16];
            for (int k = 0; k < Lo; ++k) idx_buf[k] = static_cast<int32_t>(i) + k;
            const bmm_vec_i32 vidx = bmm_loadu_i32(idx_buf);
            count += bmm_compressstore_i32(out + count, vidx, mask);
        } else {
            // AVX2 (Lo=16): split into two 8-lane compressstores.
            alignas(32) int32_t idx_buf0[16];
            alignas(32) int32_t idx_buf1[16];
            for (int k = 0; k < 8; ++k) {
                idx_buf0[k] = static_cast<int32_t>(i) + k;
                idx_buf1[k] = static_cast<int32_t>(i) + k + 8;
            }
            const uint32_t mask_lo = mask & 0xFFu;
            const uint32_t mask_hi = (mask >> 8) & 0xFFu;
            count += bmm_compressstore_i32(out + count,
                                           bmm_loadu_i32(idx_buf0), mask_lo);
            count += bmm_compressstore_i32(out + count,
                                           bmm_loadu_i32(idx_buf1), mask_hi);
        }
    }
    // Scalar tail.
    for (; i < n; ++i) {
        out[count] = static_cast<int32_t>(i);
        count += (data[i] > scalar);
    }
    return count;
}

/// SIMD branchless sum: float32 column.
/// Accumulates `bmm_lanes_f32` lanes in parallel, then horizontal-reduces.
inline double sum_avx2_f32(const float* BOLT_RESTRICT data, int64_t n) noexcept {
    assert(data != nullptr || n == 0);
    assert(n >= 0);

    using namespace bolt::simd;
    constexpr int L = bmm_lanes_f32;
    bmm_vec_f32 acc = bmm_setzero_f32();
    int64_t i = 0;
    for (; i + L <= n; i += L) {
        acc = bmm_add_f32(acc, bmm_loadu_f32(data + i));
    }
    // Widen to double for the reduction — matches the scalar semantics
    // of f64 accumulators everywhere else in the codebase.
    double result = static_cast<double>(bmm_hadd_f32(acc));
    for (; i < n; ++i) result += static_cast<double>(data[i]);
    return result;
}

#endif  // BOLT_SIMD_AVX2 || BOLT_SIMD_SSE42 || BOLT_SIMD_NEON

// ============================================================================
// Type-dispatched branchless filter (X-macro)
// ============================================================================

enum class CmpOp : uint8_t { Eq, Ne, Lt, Le, Gt, Ge };

/// Dispatch branchless filter by type. Single switch at the boundary,
/// fully specialized branchless kernel in the inner loop.
inline int64_t dispatch_filter_branchless(
        const void* data, BoltType type, int64_t n,
        CmpOp op, int64_t scalar_i64,
        int32_t* BOLT_RESTRICT out) noexcept {

    // The switch is the ONE branch — predicted after first call
    // (same column type every morsel). Inner kernel is branchless.
    switch (type) {
#define X(NAME, CPP_TYPE, BOLT_TYPE) \
        case BOLT_TYPE: { \
            CPP_TYPE scalar_val; \
            memcpy(&scalar_val, &scalar_i64, sizeof(CPP_TYPE)); \
            switch (op) { \
                case CmpOp::Gt: return filter_gt_branchless(static_cast<const CPP_TYPE*>(data), n, scalar_val, out); \
                case CmpOp::Lt: return filter_lt_branchless(static_cast<const CPP_TYPE*>(data), n, scalar_val, out); \
                case CmpOp::Eq: return filter_eq_branchless(static_cast<const CPP_TYPE*>(data), n, scalar_val, out); \
                case CmpOp::Ne: return filter_ne_branchless(static_cast<const CPP_TYPE*>(data), n, scalar_val, out); \
                case CmpOp::Ge: return filter_ge_branchless(static_cast<const CPP_TYPE*>(data), n, scalar_val, out); \
                case CmpOp::Le: return filter_le_branchless(static_cast<const CPP_TYPE*>(data), n, scalar_val, out); \
            } \
            return 0; \
        }
        BOLT_NUMERIC_TYPES
#undef X
        default: return 0;
    }
}

// ============================================================================
// Branching Kernels (for micro-adaptive selection when selectivity is extreme)
// ============================================================================

/// Branching filter — faster than branchless when selectivity < 20% or > 80%
/// because the branch predictor is accurate and the CPU can speculate past
/// the branch, effectively prefetching future iterations.
template <typename T>
int64_t filter_gt_branching(const T* BOLT_RESTRICT data, int64_t n,
                             T scalar,
                             int32_t* BOLT_RESTRICT out) noexcept {
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (data[i] > scalar) {      // Branch — predictor handles extreme selectivity
            out[count++] = static_cast<int32_t>(i);
        }
    }
    return count;
}

template <typename T>
int64_t filter_lt_branching(const T* BOLT_RESTRICT data, int64_t n,
                             T scalar,
                             int32_t* BOLT_RESTRICT out) noexcept {
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (data[i] < scalar) out[count++] = static_cast<int32_t>(i);
    }
    return count;
}

// ============================================================================
// Micro-Adaptive Filter (Pirk et al., White-Box Micro-Adaptive, ICDE 2025)
// ============================================================================

/// Estimate selectivity from column stats zone map.
/// Returns fraction in [0.0, 1.0]. Assumes uniform distribution within
/// [min_value, max_value] range (conservative but fast).
inline float estimate_selectivity_gt(int64_t col_min, int64_t col_max,
                                      int64_t scalar) noexcept {
    if (scalar >= col_max) return 0.0f;
    if (scalar < col_min) return 1.0f;
    int64_t range = col_max - col_min;
    if (range <= 0) return 0.5f;
    return static_cast<float>(col_max - scalar) / static_cast<float>(range);
}

inline float estimate_selectivity_lt(int64_t col_min, int64_t col_max,
                                      int64_t scalar) noexcept {
    if (scalar <= col_min) return 0.0f;
    if (scalar > col_max) return 1.0f;
    int64_t range = col_max - col_min;
    if (range <= 0) return 0.5f;
    return static_cast<float>(scalar - col_min) / static_cast<float>(range);
}

inline float estimate_selectivity_eq(int64_t col_min, int64_t col_max,
                                      int64_t scalar,
                                      uint32_t distinct_count) noexcept {
    if (scalar < col_min || scalar > col_max) return 0.0f;
    if (distinct_count == 0) return 0.5f;
    return 1.0f / static_cast<float>(distinct_count);
}

/// Selectivity threshold for switching between branching and branchless.
/// Below this or above (1 - this), branching is faster.
/// Pirk 2014 empirically measured ~20%. Hardware-dependent.
static constexpr float kBranchlessThresholdLow  = 0.20f;
static constexpr float kBranchlessThresholdHigh = 0.80f;

/// Micro-adaptive dispatch: choose branching or branchless based on
/// estimated selectivity from column statistics.
///
/// This is the Pirk 2025 "white-box micro-adaptive" pattern applied
/// at morsel granularity. Decision cost: two float comparisons (once
/// per morsel, not per row).
template <typename T>
int64_t filter_gt_adaptive(const T* BOLT_RESTRICT data, int64_t n,
                            T scalar,
                            int64_t col_min, int64_t col_max,
                            int32_t* BOLT_RESTRICT out) noexcept {
    int64_t scalar_i64 = 0;
    memcpy(&scalar_i64, &scalar, sizeof(T) < sizeof(int64_t) ? sizeof(T) : sizeof(int64_t));
    float sel = estimate_selectivity_gt(col_min, col_max, scalar_i64);

    if (sel < kBranchlessThresholdLow || sel > kBranchlessThresholdHigh) {
        return filter_gt_branching(data, n, scalar, out);
    }
    return filter_gt_branchless(data, n, scalar, out);
}

template <typename T>
int64_t filter_lt_adaptive(const T* BOLT_RESTRICT data, int64_t n,
                            T scalar,
                            int64_t col_min, int64_t col_max,
                            int32_t* BOLT_RESTRICT out) noexcept {
    int64_t scalar_i64 = 0;
    memcpy(&scalar_i64, &scalar, sizeof(T) < sizeof(int64_t) ? sizeof(T) : sizeof(int64_t));
    float sel = estimate_selectivity_lt(col_min, col_max, scalar_i64);

    if (sel < kBranchlessThresholdLow || sel > kBranchlessThresholdHigh) {
        return filter_lt_branching(data, n, scalar, out);
    }
    return filter_lt_branchless(data, n, scalar, out);
}

// ============================================================================
// Predicated Dual-Output Partition (Pirk et al., Database Cracking, 2014)
// ============================================================================

/// Branchless two-way partition: split data into left (< pivot) and
/// right (>= pivot) output arrays. Both outputs are written speculatively;
/// only the correct cursor advances.
///
/// This is the core primitive for:
///   - Hash join build (partition by hash)
///   - Radix partitioning
///   - Database cracking
///
/// Returns {left_count, right_count}.
struct PartitionResult {
    int64_t left_count;
    int64_t right_count;
};

template <typename T>
PartitionResult partition_predicated(const T* BOLT_RESTRICT data, int64_t n,
                                      T pivot,
                                      T* BOLT_RESTRICT left_out,
                                      T* BOLT_RESTRICT right_out) noexcept {
    int64_t l = 0, r = 0;
    for (int64_t i = 0; i < n; ++i) {
        left_out[l]  = data[i];      // Speculative write to left
        right_out[r] = data[i];      // Speculative write to right
        bool goes_left = (data[i] < pivot);
        l += goes_left;              // Advance left cursor if < pivot
        r += !goes_left;             // Advance right cursor if >= pivot
    }
    return {l, r};
}

/// Predicated partition with index tracking (for join builds where we
/// need to know which original row went where).
template <typename T>
PartitionResult partition_predicated_indexed(
        const T* BOLT_RESTRICT data, int64_t n,
        T pivot,
        T* BOLT_RESTRICT left_out,
        T* BOLT_RESTRICT right_out,
        int32_t* BOLT_RESTRICT left_indices,
        int32_t* BOLT_RESTRICT right_indices) noexcept {
    int64_t l = 0, r = 0;
    for (int64_t i = 0; i < n; ++i) {
        bool goes_left = (data[i] < pivot);
        left_out[l]     = data[i];
        right_out[r]    = data[i];
        left_indices[l] = static_cast<int32_t>(i);
        right_indices[r]= static_cast<int32_t>(i);
        l += goes_left;
        r += !goes_left;
    }
    return {l, r};
}

/// Multi-way radix partition (branchless). Partitions data into 2^radix_bits
/// buckets based on hash bits. Used for parallel hash join build.
///
/// histogram[bucket] gets the count per bucket.
/// dest[bucket] array gets the elements for that bucket.
///
/// This is the inner loop of radix-partitioned hash join build.
inline void radix_partition_scatter(
        const uint32_t* BOLT_RESTRICT hashes,
        const int32_t* BOLT_RESTRICT indices,
        int64_t n,
        uint32_t radix_bits,
        int64_t* BOLT_RESTRICT offsets,  // Current write offset per bucket
        int32_t* BOLT_RESTRICT dest_indices,
        uint32_t* BOLT_RESTRICT dest_hashes) noexcept {
    uint32_t mask = (1u << radix_bits) - 1;
    for (int64_t i = 0; i < n; ++i) {
        uint32_t bucket = hashes[i] & mask;
        int64_t pos = offsets[bucket]++;
        dest_indices[pos] = indices[i];
        dest_hashes[pos] = hashes[i];
    }
}

}  // namespace branchless
}  // namespace bolt
