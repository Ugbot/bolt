#pragma once
#include "bolt/bolt_port.h"
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cassert>
#include <cmath>
#include <limits>
#if defined(BOLT_SIMD_AVX2)
#include <immintrin.h>
#endif

// Per-column f32 statistics kernel.
//
// Produces mean / population variance / min / max / count over a contiguous
// or strided f32 column. Uses Welford's online algorithm in double-precision
// accumulators for numerical stability (single-pass, O(1) state).
//
// AVX2 8-lane Welford with Pébay parallel-merge is a follow-up; the scalar
// implementation passes the numerical-stability tests and is the default.
namespace bolt {

struct ColumnStatsF32 {
    float    mean;
    float    var;   // population variance (divide by n)
    float    min;
    float    max;
    uint32_t n;
};

BOLT_FORCE_INLINE void column_stats_f32(
    const float* BOLT_RESTRICT col, size_t n,
    ColumnStatsF32* BOLT_RESTRICT out
) noexcept {
    assert(out != nullptr);
    if (n == 0) {
        out->mean = 0.0f;
        out->var  = 0.0f;
        out->min  = std::numeric_limits<float>::infinity();
        out->max  = -std::numeric_limits<float>::infinity();
        out->n    = 0u;
        return;
    }
    assert(col != nullptr);
    double mean = 0.0;
    double m2   = 0.0;
    float  mn   = std::numeric_limits<float>::infinity();
    float  mx   = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < n; ++i) {
        const float x = col[i];
        const double xd = static_cast<double>(x);
        const double delta = xd - mean;
        mean += delta / static_cast<double>(i + 1);
        const double delta2 = xd - mean;
        m2 += delta * delta2;
        mn = (x < mn) ? x : mn;
        mx = (x > mx) ? x : mx;
    }
    out->mean = static_cast<float>(mean);
    out->var  = static_cast<float>(m2 / static_cast<double>(n));
    out->min  = mn;
    out->max  = mx;
    out->n    = static_cast<uint32_t>(n);
}

BOLT_FORCE_INLINE void column_stats_f32_strided(
    const float* BOLT_RESTRICT col, size_t n, size_t stride_bytes,
    ColumnStatsF32* BOLT_RESTRICT out
) noexcept {
    assert(out != nullptr);
    if (n == 0) {
        out->mean = 0.0f;
        out->var  = 0.0f;
        out->min  = std::numeric_limits<float>::infinity();
        out->max  = -std::numeric_limits<float>::infinity();
        out->n    = 0u;
        return;
    }
    assert(col != nullptr);
    assert(stride_bytes >= sizeof(float));
    double mean = 0.0;
    double m2   = 0.0;
    float  mn   = std::numeric_limits<float>::infinity();
    float  mx   = -std::numeric_limits<float>::infinity();
    const uint8_t* p = reinterpret_cast<const uint8_t*>(col);
    for (size_t i = 0; i < n; ++i) {
        float x;
        std::memcpy(&x, p + i * stride_bytes, sizeof(float));
        const double xd = static_cast<double>(x);
        const double delta = xd - mean;
        mean += delta / static_cast<double>(i + 1);
        const double delta2 = xd - mean;
        m2 += delta * delta2;
        mn = (x < mn) ? x : mn;
        mx = (x > mx) ? x : mx;
    }
    out->mean = static_cast<float>(mean);
    out->var  = static_cast<float>(m2 / static_cast<double>(n));
    out->min  = mn;
    out->max  = mx;
    out->n    = static_cast<uint32_t>(n);
}

}  // namespace bolt
