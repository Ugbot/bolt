// gemv_scalar.cpp -- scalar reference tier for bolt::compute's f32/f16 GEMV
// (perf pass). Compiled at BASELINE ISA (no /arch flag) so it is a genuinely
// safe runtime fallback on any CPU. Accumulates each row's dot product in f64
// (double) -- this is the numerical "truth" the SIMD tiers agree with to
// ~1e-6 relative (the SIMD tiers reorder the f32 reduction, so agreement is
// within tolerance, not bit-exact; see compute_gemv.h).

#include "gemv_kernels_internal.h"

#include <cassert>
#include <cstdint>
#include <cstring>

namespace bolt::compute::detail {

namespace {

// IEEE-754 binary16 -> binary32, pure scalar (no F16C). Handles zero,
// subnormal, normal, inf/nan. Only runs on CPUs without F16C, so correctness
// over speed.
inline float half_to_float(uint16_t h) noexcept {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0u) {
        if (mant == 0u) {
            bits = sign;  // +/- 0
        } else {
            // Subnormal: normalize into a binary32 normal.
            int32_t e = -1;
            uint32_t m = mant;
            do {
                m <<= 1;
                ++e;
            } while ((m & 0x400u) == 0u);
            m &= 0x3FFu;
            bits = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);  // inf / nan
    } else {
        bits = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

}  // namespace

void gemv_f32_scalar(const float* weight, const float* x, float* out, int32_t rows,
                     int32_t cols) noexcept {
    assert(weight != nullptr && x != nullptr && out != nullptr);
    assert(rows > 0 && cols > 0);

    for (int32_t r = 0; r < rows; ++r) {
        const float* row = weight + static_cast<int64_t>(r) * cols;
        double acc = 0.0;
        for (int32_t i = 0; i < cols; ++i) acc += static_cast<double>(row[i] * x[i]);
        out[r] = static_cast<float>(acc);
    }
}

void gemv_f16_scalar(const uint16_t* weight, const float* x, float* out, int32_t rows,
                     int32_t cols) noexcept {
    assert(weight != nullptr && x != nullptr && out != nullptr);
    assert(rows > 0 && cols > 0);

    for (int32_t r = 0; r < rows; ++r) {
        const uint16_t* row = weight + static_cast<int64_t>(r) * cols;
        double acc = 0.0;
        for (int32_t i = 0; i < cols; ++i) acc += static_cast<double>(half_to_float(row[i]) * x[i]);
        out[r] = static_cast<float>(acc);
    }
}

}  // namespace bolt::compute::detail
