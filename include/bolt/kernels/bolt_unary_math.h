// Column unary math over Float64 and truncating Int64 division / remainder by
// a literal (G2LAUNCH-152: compiled Python UDFs vectorize through these).
// Plain loops: the compiler vectorizes them (libm calls stay per element).
#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>

#include "bolt/bolt_port.h"

namespace bolt::kernels {

enum class UnaryF64 : uint8_t { Sqrt, Abs, Neg, Exp, Log, Log10, Log2, Floor, Ceil };

inline void unary_f64(UnaryF64 op, const double* BOLT_RESTRICT a, int64_t n,
                      double* BOLT_RESTRICT out) noexcept {
    assert(n >= 0);
    assert((a != nullptr && out != nullptr) || n == 0);
    switch (op) {
        case UnaryF64::Sqrt:  for (int64_t i = 0; i < n; ++i) out[i] = std::sqrt(a[i]); break;
        case UnaryF64::Abs:   for (int64_t i = 0; i < n; ++i) out[i] = std::fabs(a[i]); break;
        case UnaryF64::Neg:   for (int64_t i = 0; i < n; ++i) out[i] = -a[i]; break;
        case UnaryF64::Exp:   for (int64_t i = 0; i < n; ++i) out[i] = std::exp(a[i]); break;
        case UnaryF64::Log:   for (int64_t i = 0; i < n; ++i) out[i] = std::log(a[i]); break;
        case UnaryF64::Log10: for (int64_t i = 0; i < n; ++i) out[i] = std::log10(a[i]); break;
        case UnaryF64::Log2:  for (int64_t i = 0; i < n; ++i) out[i] = std::log2(a[i]); break;
        case UnaryF64::Floor: for (int64_t i = 0; i < n; ++i) out[i] = std::floor(a[i]); break;
        case UnaryF64::Ceil:  for (int64_t i = 0; i < n; ++i) out[i] = std::ceil(a[i]); break;
    }
}

// a / lit and a % lit truncating toward zero (C++ / SQL). lit must not be 0
// or -1 (the callers route those through their null / overflow rules).
inline void div_lit_i64(const int64_t* BOLT_RESTRICT a, int64_t n, int64_t lit,
                        int64_t* BOLT_RESTRICT out) noexcept {
    assert(lit != 0 && lit != -1);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) out[i] = a[i] / lit;
}

inline void mod_lit_i64(const int64_t* BOLT_RESTRICT a, int64_t n, int64_t lit,
                        int64_t* BOLT_RESTRICT out) noexcept {
    assert(lit != 0 && lit != -1);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) out[i] = a[i] % lit;
}

}  // namespace bolt::kernels
