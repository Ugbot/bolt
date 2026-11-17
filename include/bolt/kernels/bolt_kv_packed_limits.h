// bolt_kv_packed_limits.h — bounds of the packed key-value kernels (bolt_kv_packed.h).
#pragma once

#include <cstdint>

namespace bolt {
namespace kernels {
namespace kv_packed {

// Longest key one lookup carries (kind A: the needle is built on the stack).
inline constexpr uint32_t kKeyMaxBytes = 254;

}  // namespace kv_packed
}  // namespace kernels
}  // namespace bolt
