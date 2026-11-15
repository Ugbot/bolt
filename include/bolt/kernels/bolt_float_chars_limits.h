// bolt_float_chars_limits.h: invariants of the float <-> text kernels (bolt_float_chars.h).
// Kind A: fixed by the output grammar, reported, never set.
#pragma once

#include <cstdint>

namespace bolt {

// Longest f64_to_chars_prom output: sign + 17 digits + '.' + "e-324".
inline constexpr uint32_t k_f64_prom_max_chars = 32;

}  // namespace bolt
