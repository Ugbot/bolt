// bolt_vector_limits.h — the one vector-dimension ceiling.
//
// Every vector-dim check in bolt, marbledb, chukonu and Gestalt2 derives from
// here. These are validation ceilings, not allocation sizes: kernels loop over
// the runtime dim and callers size scratch from it.
//
// A stored vector row lives in one BoltColumn slot whose stride is the
// uint16 type_size_bytes, so the per-row byte width is the real bound and the
// dim ceiling depends on the element width: f32 16,383, f16 32,767, u8/i8
// 65,535. kMaxVectorDim is the widest of those (the kernel-side ceiling).

#pragma once

#include <cstdint>

namespace bolt {

inline constexpr uint32_t kMaxVectorRowBytes = 65535u;   // uint16 column stride
inline constexpr uint32_t kMaxVectorDim      = kMaxVectorRowBytes;

// Largest dim a stored vector of `elem_bytes`-wide lanes can have.
constexpr uint32_t max_vector_dim_for(uint32_t elem_bytes) noexcept {
    return elem_bytes == 0u ? 0u : kMaxVectorRowBytes / elem_bytes;
}

static_assert(max_vector_dim_for(4u) == 16383u, "f32 row stride bound");

}  // namespace bolt
