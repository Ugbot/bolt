#pragma once
// bolt_vector_traits.h — element-type traits for multi-precision vector
// kernels. The Wave 9.4 Phase ζ scaffold: a single place to map each
// supported BoltType to its storage / accumulator / stride properties,
// plus a global `QuantParams` POD for scalar-quantised columns.
//
// Lookup pattern: `ElementTraits<BoltType::Float32>::storage_t` is `float`;
// `ElementTraits<BoltType::Float16>::bytes_per_elt` is 2; etc.
//
// Existing f32-only kernels stay as-is — this header strictly ADDS new
// types alongside. f16/u8/i8 specialisations of `l2_pair`,
// `l2_vertical_accumulate`, BOND helpers, and σ-bound kernels consume
// these traits.
//
// Tiger Style — header-only, no exceptions, no virtuals; everything
// compile-time-dispatched.

#include "bolt/bolt_port.h"
#include "bolt/bolt_types.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace bolt {
namespace vec {

// Trait base — undefined; specialise per supported element type. A
// compile error here means the kernel was instantiated on an
// unsupported BoltType (helpful: instead of silently working at f32
// precision, the build fails fast at the kernel call site).
template <BoltType T> struct ElementTraits;

template <> struct ElementTraits<BoltType::Float32> {
    using storage_t = float;
    using accum_t   = float;             // FMA accumulator
    static constexpr size_t bytes_per_elt = 4;
    static constexpr bool   is_quantized  = false;
    static constexpr bool   needs_params  = false;
};

template <> struct ElementTraits<BoltType::Float16> {
    // We never use the C++23 `std::float16_t` — too new for the MSVC
    // target. The storage type is `uint16_t` (bit container); the f16
    // kernels reinterpret as native `__fp16` (NEON) or convert via
    // `_mm256_cvtph_ps` (AVX2) as needed.
    using storage_t = uint16_t;
    using accum_t   = float;             // promote to f32 for accumulation
    static constexpr size_t bytes_per_elt = 2;
    static constexpr bool   is_quantized  = false;
    static constexpr bool   needs_params  = false;
};

template <> struct ElementTraits<BoltType::UInt8> {
    using storage_t = uint8_t;
    using accum_t   = uint32_t;          // i32 squared-difference accum
    static constexpr size_t bytes_per_elt = 1;
    static constexpr bool   is_quantized  = true;
    static constexpr bool   needs_params  = true;
};

template <> struct ElementTraits<BoltType::Int8> {
    using storage_t = int8_t;
    using accum_t   = int32_t;
    static constexpr size_t bytes_per_elt = 1;
    static constexpr bool   is_quantized  = true;
    static constexpr bool   needs_params  = true;
};

// Global per-column (per-row-group) scalar-quantisation parameters.
// PDXearch's pattern — one base + scale across the entire row group;
// applied symmetrically to the query before invoking the quantised
// kernel. See `third-party/pdxearch_reference/src/include/pdxearch/
// quantizers/scalar.hpp` for the canonical layout we model on.
//
//   x_q = round((x - base) * scale)
//   x_f ≈ x_q / scale + base                       (inverse)
//   L2²_f ≈ L2²_q / (scale * scale)                (squared inverse)
//
// `scale_squared` + `inv_scale_squared` memoised so the quantised
// kernels avoid a fp divide on the hot path.
struct QuantParams {
    float base;                // shift before quantisation
    float scale;               // 1 / range; applied per-element
    float scale_squared;       // scale × scale (memoised)
    float inv_scale_squared;   // 1 / (scale × scale) (memoised)
};

static_assert(sizeof(QuantParams) == 16,
              "QuantParams layout drift");

// Stride per row for the given element type and dim. Replaces the
// `embedding_stride()` in `bolt_types.h` which hardcoded f32. Callers
// compute on-disk row stride via this function; new ColumnTypes
// (kVectorFloat16, kVectorUInt8, kVectorInt8) MUST consume it for
// every stride / offset arithmetic.
BOLT_FORCE_INLINE size_t element_stride_bytes(BoltType t,
                                              uint32_t dim) noexcept {
    assert(dim > 0u);
    switch (t) {
        case BoltType::Float32:  return static_cast<size_t>(dim) * 4u;
        case BoltType::Float16:  return static_cast<size_t>(dim) * 2u;
        case BoltType::UInt8:    return static_cast<size_t>(dim) * 1u;
        case BoltType::Int8:     return static_cast<size_t>(dim) * 1u;
        case BoltType::Embedding: return static_cast<size_t>(dim) * 4u;  // legacy
        default: return 0;       // not a vector element type
    }
}

// Compile-time helper: does this BoltType have a valid Embedding kernel
// family? Use to gate templates so unsupported types fail at the
// template instantiation site, not deep in a kernel body.
BOLT_FORCE_INLINE constexpr bool is_vector_element(BoltType t) noexcept {
    return t == BoltType::Float32 ||
           t == BoltType::Float16 ||
           t == BoltType::UInt8   ||
           t == BoltType::Int8    ||
           t == BoltType::Embedding;
}

}  // namespace vec
}  // namespace bolt
