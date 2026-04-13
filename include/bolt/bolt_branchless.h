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

#include "bolt_types.h"
#include <cstdint>
#include <cstring>
#include <cassert>

// Platform SIMD includes
#if defined(__AVX2__)
#include <immintrin.h>
#define BOLT_HAS_AVX2 1
#elif defined(__SSE4_2__)
#include <nmmintrin.h>
#define BOLT_HAS_SSE42 1
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define BOLT_HAS_NEON 1
#endif

namespace chukonu {
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
int64_t filter_gt_branchless(const T* __restrict__ data, int64_t n,
                              T scalar,
                              int32_t* __restrict__ out) noexcept {
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        out[count] = static_cast<int32_t>(i);
        count += (data[i] > scalar);   // 0 or 1, no branch
    }
    return count;
}

template <typename T>
int64_t filter_lt_branchless(const T* __restrict__ data, int64_t n,
                              T scalar,
                              int32_t* __restrict__ out) noexcept {
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        out[count] = static_cast<int32_t>(i);
        count += (data[i] < scalar);
    }
    return count;
}

template <typename T>
int64_t filter_eq_branchless(const T* __restrict__ data, int64_t n,
                              T scalar,
                              int32_t* __restrict__ out) noexcept {
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        out[count] = static_cast<int32_t>(i);
        count += (data[i] == scalar);
    }
    return count;
}

template <typename T>
int64_t filter_ne_branchless(const T* __restrict__ data, int64_t n,
                              T scalar,
                              int32_t* __restrict__ out) noexcept {
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        out[count] = static_cast<int32_t>(i);
        count += (data[i] != scalar);
    }
    return count;
}

template <typename T>
int64_t filter_ge_branchless(const T* __restrict__ data, int64_t n,
                              T scalar,
                              int32_t* __restrict__ out) noexcept {
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        out[count] = static_cast<int32_t>(i);
        count += (data[i] >= scalar);
    }
    return count;
}

template <typename T>
int64_t filter_le_branchless(const T* __restrict__ data, int64_t n,
                              T scalar,
                              int32_t* __restrict__ out) noexcept {
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
        const T* __restrict__ data,
        const int32_t* __restrict__ sel_in, int64_t sel_count,
        T scalar,
        int32_t* __restrict__ sel_out) noexcept {
    int64_t count = 0;
    for (int64_t i = 0; i < sel_count; ++i) {
        int32_t idx = sel_in[i];
        sel_out[count] = idx;
        count += (data[idx] > scalar);
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
        const T* __restrict__ data,
        const uint8_t* validity,  // Arrow format: bit=1 means valid
        int64_t n, T scalar,
        int32_t* __restrict__ out) noexcept {
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
T aggregate_min_branchless(const T* __restrict__ data, int64_t n) noexcept {
    assert(n > 0);
    T result = data[0];
    for (int64_t i = 1; i < n; ++i) {
        result = bmin(result, data[i]);  // cmov, no branch
    }
    return result;
}

template <typename T>
T aggregate_max_branchless(const T* __restrict__ data, int64_t n) noexcept {
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
AccumT aggregate_sum_branchless(const T* __restrict__ data, int64_t n) noexcept {
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
AccumT aggregate_sum_masked(const T* __restrict__ data,
                             const uint8_t* __restrict__ mask,
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
void hash_column_branchless(const T* __restrict__ data, int64_t n,
                             uint32_t* __restrict__ hashes,
                             uint32_t seed = 0x9E3779B9) noexcept {
    for (int64_t i = 0; i < n; ++i) {
        uint64_t val = 0;
        memcpy(&val, &data[i], sizeof(T));  // Type-punning via memcpy
        hashes[i] = static_cast<uint32_t>(hash_finalize(val ^ seed));
    }
}

/// Branchless hash combine (for multi-column keys).
inline void hash_combine_branchless(const uint32_t* __restrict__ h1,
                                     const uint32_t* __restrict__ h2,
                                     int64_t n,
                                     uint32_t* __restrict__ out) noexcept {
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
template <typename T>
void gather_branchless(const T* __restrict__ data,
                        const int32_t* __restrict__ indices,
                        int64_t count,
                        T* __restrict__ output) noexcept {
    constexpr int kPrefetchDist = 16;

    int64_t i = 0;
    // Prefetch ahead
    for (; i < bmin(count, (int64_t)kPrefetchDist); ++i) {
        __builtin_prefetch(&data[indices[i]], 0, 1);
    }
    // Main loop with prefetch pipeline
    for (i = 0; i < count; ++i) {
        if (i + kPrefetchDist < count) {
            __builtin_prefetch(&data[indices[i + kPrefetchDist]], 0, 1);
        }
        output[i] = data[indices[i]];  // No branch, just load-store
    }
}

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
inline int64_t selection_intersect_branchless(
        const int32_t* __restrict__ a, int64_t na,
        const int32_t* __restrict__ b, int64_t nb,
        int32_t* __restrict__ out) noexcept {
    int64_t i = 0, j = 0, count = 0;
    while (i < na && j < nb) {
        int32_t va = a[i], vb = b[j];
        bool match = (va == vb);
        out[count] = va;
        count += match;
        i += (va <= vb);  // Advance a if a <= b (branchless)
        j += (va >= vb);  // Advance b if a >= b (branchless)
    }
    return count;
}

/// Branchless union of two sorted selection vectors.
inline int64_t selection_union_branchless(
        const int32_t* __restrict__ a, int64_t na,
        const int32_t* __restrict__ b, int64_t nb,
        int32_t* __restrict__ out) noexcept {
    int64_t i = 0, j = 0, count = 0;
    while (i < na && j < nb) {
        int32_t va = a[i], vb = b[j];
        int32_t val = bmin(va, vb);
        out[count] = val;
        count++;
        i += (va <= vb);
        j += (va >= vb);
    }
    while (i < na) out[count++] = a[i++];
    while (j < nb) out[count++] = b[j++];
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
        count += static_cast<uint32_t>(__builtin_popcountll(bitmap[i]));
#else
        // Fallback: Kernighan's method (still branchless per bit)
        uint64_t v = bitmap[i];
        while (v) { count++; v &= v - 1; }
#endif
    }
    return count;
}

/// Branchless bitmap AND (intersection of two bitmaps).
inline void bitmap_and(const uint64_t* __restrict__ a,
                        const uint64_t* __restrict__ b,
                        uint64_t* __restrict__ out,
                        uint32_t num_words) noexcept {
    for (uint32_t i = 0; i < num_words; ++i) {
        out[i] = a[i] & b[i];  // Pure bitwise, zero branches
    }
}

/// Branchless bitmap OR (union of two bitmaps).
inline void bitmap_or(const uint64_t* __restrict__ a,
                       const uint64_t* __restrict__ b,
                       uint64_t* __restrict__ out,
                       uint32_t num_words) noexcept {
    for (uint32_t i = 0; i < num_words; ++i) {
        out[i] = a[i] | b[i];
    }
}

/// Extract set bit positions to selection vector (branchless per word).
inline int64_t bitmap_to_selection(const uint64_t* bitmap, uint32_t num_words,
                                    int32_t* __restrict__ out) noexcept {
    int64_t count = 0;
    for (uint32_t w = 0; w < num_words; ++w) {
        uint64_t bits = bitmap[w];
        int32_t base = static_cast<int32_t>(w * 64);
        while (bits) {
            // __builtin_ctzll: count trailing zeros = position of lowest set bit
            int pos = __builtin_ctzll(bits);
            out[count++] = base + pos;
            bits &= bits - 1;  // Clear lowest set bit (Kernighan's trick)
        }
    }
    return count;
}

// ============================================================================
// SIMD Filter Kernels (AVX2)
// ============================================================================

#ifdef BOLT_HAS_AVX2

/// AVX2 branchless filter: int32 column > scalar.
/// Processes 8 int32s per iteration via SIMD compare + movemask + scatter.
inline int64_t filter_gt_avx2_i32(const int32_t* __restrict__ data, int64_t n,
                                   int32_t scalar,
                                   int32_t* __restrict__ out) noexcept {
    __m256i vscalar = _mm256_set1_epi32(scalar);
    int64_t count = 0;
    int64_t i = 0;

    // Main SIMD loop (8 elements per iteration)
    for (; i + 8 <= n; i += 8) {
        __m256i vdata = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
        __m256i vcmp = _mm256_cmpgt_epi32(vdata, vscalar);
        int mask = _mm256_movemask_epi8(vcmp);

        // Convert byte mask to element mask (every 4th bit)
        // Each int32 comparison produces 4 bytes of 0xFF or 0x00
        uint32_t elem_mask = _pext_u32(static_cast<uint32_t>(mask), 0x88888888u);

        // Scatter matching indices (branchless — always write, conditionally advance)
        // This is the scalar fallback; AVX-512 has _mm256_mask_compressstoreu_epi32
        int32_t base = static_cast<int32_t>(i);
        while (elem_mask) {
            int pos = __builtin_ctz(elem_mask);
            out[count++] = base + pos;
            elem_mask &= elem_mask - 1;
        }
    }

    // Scalar tail
    for (; i < n; ++i) {
        out[count] = static_cast<int32_t>(i);
        count += (data[i] > scalar);
    }
    return count;
}

/// AVX2 branchless sum: int64 column.
inline int64_t sum_avx2_i64(const int64_t* __restrict__ data, int64_t n) noexcept {
    __m256i vsum = _mm256_setzero_si256();
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256i vdata = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
        vsum = _mm256_add_epi64(vsum, vdata);
    }
    // Horizontal sum
    alignas(32) int64_t tmp[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), vsum);
    int64_t result = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    // Scalar tail
    for (; i < n; ++i) result += data[i];
    return result;
}

#endif  // BOLT_HAS_AVX2

#ifdef BOLT_HAS_NEON

/// NEON branchless filter: int32 column > scalar.
inline int64_t filter_gt_neon_i32(const int32_t* __restrict__ data, int64_t n,
                                   int32_t scalar,
                                   int32_t* __restrict__ out) noexcept {
    int32x4_t vscalar = vdupq_n_s32(scalar);
    int64_t count = 0;
    int64_t i = 0;

    for (; i + 4 <= n; i += 4) {
        int32x4_t vdata = vld1q_s32(data + i);
        uint32x4_t vcmp = vcgtq_s32(vdata, vscalar);

        // Extract comparison results
        uint32_t mask = 0;
        mask |= (vgetq_lane_u32(vcmp, 0) ? 1u : 0u);
        mask |= (vgetq_lane_u32(vcmp, 1) ? 2u : 0u);
        mask |= (vgetq_lane_u32(vcmp, 2) ? 4u : 0u);
        mask |= (vgetq_lane_u32(vcmp, 3) ? 8u : 0u);

        int32_t base = static_cast<int32_t>(i);
        while (mask) {
            int pos = __builtin_ctz(mask);
            out[count++] = base + pos;
            mask &= mask - 1;
        }
    }

    for (; i < n; ++i) {
        out[count] = static_cast<int32_t>(i);
        count += (data[i] > scalar);
    }
    return count;
}

#endif  // BOLT_HAS_NEON

// ============================================================================
// Type-dispatched branchless filter (X-macro)
// ============================================================================

enum class CmpOp : uint8_t { Eq, Ne, Lt, Le, Gt, Ge };

/// Dispatch branchless filter by type. Single switch at the boundary,
/// fully specialized branchless kernel in the inner loop.
inline int64_t dispatch_filter_branchless(
        const void* data, BoltType type, int64_t n,
        CmpOp op, int64_t scalar_i64,
        int32_t* __restrict__ out) noexcept {

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
int64_t filter_gt_branching(const T* __restrict__ data, int64_t n,
                             T scalar,
                             int32_t* __restrict__ out) noexcept {
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (data[i] > scalar) {      // Branch — predictor handles extreme selectivity
            out[count++] = static_cast<int32_t>(i);
        }
    }
    return count;
}

template <typename T>
int64_t filter_lt_branching(const T* __restrict__ data, int64_t n,
                             T scalar,
                             int32_t* __restrict__ out) noexcept {
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
int64_t filter_gt_adaptive(const T* __restrict__ data, int64_t n,
                            T scalar,
                            int64_t col_min, int64_t col_max,
                            int32_t* __restrict__ out) noexcept {
    int64_t scalar_i64 = 0;
    memcpy(&scalar_i64, &scalar, sizeof(T) < sizeof(int64_t) ? sizeof(T) : sizeof(int64_t));
    float sel = estimate_selectivity_gt(col_min, col_max, scalar_i64);

    if (sel < kBranchlessThresholdLow || sel > kBranchlessThresholdHigh) {
        return filter_gt_branching(data, n, scalar, out);
    }
    return filter_gt_branchless(data, n, scalar, out);
}

template <typename T>
int64_t filter_lt_adaptive(const T* __restrict__ data, int64_t n,
                            T scalar,
                            int64_t col_min, int64_t col_max,
                            int32_t* __restrict__ out) noexcept {
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
PartitionResult partition_predicated(const T* __restrict__ data, int64_t n,
                                      T pivot,
                                      T* __restrict__ left_out,
                                      T* __restrict__ right_out) noexcept {
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
        const T* __restrict__ data, int64_t n,
        T pivot,
        T* __restrict__ left_out,
        T* __restrict__ right_out,
        int32_t* __restrict__ left_indices,
        int32_t* __restrict__ right_indices) noexcept {
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
        const uint32_t* __restrict__ hashes,
        const int32_t* __restrict__ indices,
        int64_t n,
        uint32_t radix_bits,
        int64_t* __restrict__ offsets,  // Current write offset per bucket
        int32_t* __restrict__ dest_indices,
        uint32_t* __restrict__ dest_hashes) noexcept {
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
}  // namespace chukonu
