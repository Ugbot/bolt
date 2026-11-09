// bolt_csr_algo_limits.h — caps of the whole-graph CSR algorithm kernels.
// Kind A: contract bounds checked on entry, reported, never set.
#pragma once

#include <cstdint>

namespace bolt {
namespace kernels {

// CDLP iterations one csr_label_propagation call may ask for. Graphalytics
// runs ~10; this bounds a mistyped parameter, not a real workload.
inline constexpr int32_t k_csr_lp_max_iterations = 1000000;

}  // namespace kernels
}  // namespace bolt
