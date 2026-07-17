// bolt_tensor.h — Device-generic tensor type (N0 substrate for boltllm's
// MoE inference engine, and any future GPU-backed consumer of Bolt).
//
// POD description of an N-dimensional (rank <= 4) tensor: a `Device` tag,
// a `DType` (plain float or a per-row-scaled quantized element format),
// fixed-size dims/strides arrays, and an arena/caller-owned data pointer.
//
// This is the substrate that boltllm::quant::QTensor
// (boltllm/include/boltllm/quant/QuantTypes.h) generalizes into: QTensor
// is a CPU-only, 2D, read-only view over a per-row-scaled quantized
// weight tensor; `bolt::Tensor` below is the SAME layout with a `Device`
// axis bolted on (so a later Vulkan/CUDA backend can reuse this type
// instead of boltllm growing a second, parallel CPU-only tensor
// hierarchy). QTensor's eventual migration onto this type is a separate,
// later ticket (BLLM-14) — this header only has to make that migration
// *possible* by carrying QTensor's exact per-row-scale convention
// forward, not redesign it.
//
// RULES (Tiger Style — matches bolt_zonemap.h / bolt_row_view.h):
//   - POD — no ctors/dtors, memcpy-safe, no exceptions, no RTTI.
//   - No allocation inside Tensor itself. `data` / `row_scales` are
//     arena- or caller-owned (see bolt_arena.h); Tensor never mallocs
//     and never frees.
//   - Fixed-rank dims[4]/strides[4] (GGML_MAX_DIMS-style), not a
//     dynamic-rank shape — matches this codebase's "hard upper bounds on
//     everything" rule (bolt_row_view.h's fixed col_ptrs[], bolt_types.h's
//     BoltSchema::fields[256]).
//
// Scope: this header defines the type only.
//   - `bolt::compute` (matmul / rmsnorm / rope dispatch over `Tensor`) is
//     BLLM-13, layered on top of this header.
//   - Vulkan/CUDA backend dispatch is separate G-stage work; `Device`
//     below only reserves the enum values so call sites can start
//     branching on device without the backends existing yet.
//
// -----------------------------------------------------------------------
// Field-for-field mapping to boltllm::quant::QTensor (see that header's
// citations for full quant-format provenance — Colibri glm.c, Apache-2.0):
//
//   QTensor::fmt        -> Tensor::dtype     DType::F32/Int8/Int4/Int2 use
//                          the SAME numeric values as
//                          QuantFmt::F32=0/Int8=1/Int4=2/Int2=3, so a
//                          QTensor::fmt byte and a Tensor::dtype byte are
//                          bit-identical. F16/BF16 are new values appended
//                          at 4/5 with no QuantFmt equivalent (GPU/compute
//                          dtypes QTensor never needed).
//   QTensor::rows       -> Tensor::dims[1]   ne[1], the outer/slower axis.
//   QTensor::cols       -> Tensor::dims[0]   ne[0], the inner/contiguous
//                          axis — ggml's nb[i] = nb[i-1]*ne[i-1]
//                          convention: dim 0 is the fastest-varying axis.
//   QTensor::stride     -> Tensor::strides[1]  Bytes between rows;
//                          computed by the exact same formula as
//                          QuantTypes.h's row_stride_bytes() (see
//                          dtype_row_stride_bytes() below).
//   QTensor::data       -> Tensor::data      Payload bytes. QTensor's view
//                          is const (`const uint8_t*`); Tensor's is
//                          mutable (`void*`) since compute writes into
//                          tensors too — a CPU/QTensor alias just
//                          reinterprets this pointer as const.
//   QTensor::row_scales -> Tensor::row_scales  Per-row f32 scale array,
//                          nullptr iff dtype_is_quantized(dtype) is false.
//   (n/a — CPU only)    -> Tensor::device     Device::CPU always, for
//                          anything QTensor could represent today.
//   (n/a — 2D only)     -> Tensor::dims[2..3] == 1 for a QTensor-
//                          equivalent view; reserved for batched/ND use.
//
// Sub-byte packed formats (Int4/Int2): a per-element byte stride is not
// an integer (0.5 / 0.25 bytes/element), so `strides[0]` cannot hold
// "bytes to advance one element" for these dtypes — QuantTypes.h never
// expresses one either, only a row stride. Convention carried forward
// here: `strides[0] == 0` is the sentinel "sub-byte packed; use
// dtype_row_stride_bytes(dtype, dims[0]) for the row size, and unpack
// within a row via the dtype's own bit layout (nibble / 2-bit-code
// order), not by advancing strides[0]". F32/F16/BF16/Int8 are
// whole-byte-per-element formats, so strides[0] is a real byte stride
// there (== dtype_element_bytes(dtype)).

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace bolt {

// ---------------------------------------------------------------------------
// Device — index only. Vulkan/CUDA dispatch is separate G-stage work; this
// ticket (N0) only reserves the enum values so `Tensor::device` has
// somewhere to point.
// ---------------------------------------------------------------------------
enum class Device : uint8_t {
    CPU    = 0,
    Vulkan = 1,
    CUDA   = 2,
    ROCm   = 3,
};

// ---------------------------------------------------------------------------
// DType — element format. F32/Int8/Int4/Int2 mirror
// boltllm::quant::QuantFmt by VALUE (see the field-mapping block above);
// F16/BF16 are Bolt-only additions with no QuantFmt equivalent.
// ---------------------------------------------------------------------------
enum class DType : uint8_t {
    F32  = 0,  // == boltllm::quant::QuantFmt::F32
    Int8 = 1,  // == boltllm::quant::QuantFmt::Int8
    Int4 = 2,  // == boltllm::quant::QuantFmt::Int4
    Int2 = 3,  // == boltllm::quant::QuantFmt::Int2
    F16  = 4,  // new — no QuantFmt equivalent
    BF16 = 5,  // new — no QuantFmt equivalent
};

inline constexpr bool dtype_is_quantized(DType dtype) noexcept {
    return dtype == DType::Int8 || dtype == DType::Int4 || dtype == DType::Int2;
}

// Whole-element byte size. Returns 0 for sub-byte-packed formats
// (Int4 = 2 elems/byte, Int2 = 4 elems/byte) — mirrors bolt_types.h's
// `kTypeSize` "0 = variable width" idiom, repurposed here as "0 =
// not expressible as a whole per-element byte stride".
inline constexpr int32_t dtype_element_bytes(DType dtype) noexcept {
    switch (dtype) {
        case DType::F32:  return 4;
        case DType::F16:  return 2;
        case DType::BF16: return 2;
        case DType::Int8: return 1;
        case DType::Int4: return 0;  // sub-byte packed
        case DType::Int2: return 0;  // sub-byte packed
    }
    return 0;
}

// Row byte stride for `count` contiguous elements of `dtype`, tightly
// packed with NO per-row padding. Identical formulas to
// boltllm::quant::row_stride_bytes() (QuantTypes.h) for the four
// overlapping formats — this function IS that one, generalized with the
// two whole-byte float dtypes QuantFmt doesn't have. Kept in lock-step
// with QuantTypes.h by convention (see field-mapping block above); if
// either changes, update both.
inline constexpr int64_t dtype_row_stride_bytes(DType dtype, int32_t count) noexcept {
    switch (dtype) {
        case DType::F32:  return static_cast<int64_t>(count) * 4;
        case DType::F16:  return static_cast<int64_t>(count) * 2;
        case DType::BF16: return static_cast<int64_t>(count) * 2;
        case DType::Int8: return static_cast<int64_t>(count);
        case DType::Int4: return static_cast<int64_t>((count + 1) / 2);
        case DType::Int2: return static_cast<int64_t>((count + 3) / 4);
    }
    return 0;
}

// Maximum tensor rank. Fixed, not dynamic — Tiger Style hard upper bound.
// ggml's GGML_MAX_DIMS is also 4; matched deliberately (see header banner).
inline constexpr int32_t kTensorMaxDims = 4;

// ---------------------------------------------------------------------------
// Tensor — device-generic N-D (rank <= 4) tensor view. POD, no ownership.
// ---------------------------------------------------------------------------
struct Tensor {
    Device  device = Device::CPU;
    DType   dtype  = DType::F32;
    // ne[] — element counts per axis, dim 0 fastest-varying (contiguous).
    int32_t dims[kTensorMaxDims]    = {0, 0, 0, 0};
    // nb[] — byte strides per axis; nb[i] = nb[i-1]*ne[i-1] for dense
    // tensors (ggml convention). strides[0] == 0 is the sub-byte-packed
    // sentinel for Int4/Int2 (see header banner).
    int64_t strides[kTensorMaxDims] = {0, 0, 0, 0};
    // Arena/caller-owned payload bytes. Tensor never allocates or frees
    // this; lifetime is the owner's responsibility (bolt::Arena or a
    // caller-managed buffer), exactly like QTensor::data.
    void*   data       = nullptr;
    // Arena/caller-owned per-row f32 scale array, length == rows().
    // nullptr iff !dtype_is_quantized(dtype), exactly like
    // QTensor::row_scales.
    float*  row_scales = nullptr;

    // ---- CPU 2D accessors — the QTensor-equivalent view -----------------
    // (rows/cols/row-stride nomenclature kept identical to QTensor so a
    // reader who knows QTensor can read this at a glance.)

    int32_t rows() const noexcept { return dims[1]; }
    int32_t cols() const noexcept { return dims[0]; }
    int64_t row_stride_bytes() const noexcept { return strides[1]; }

    const uint8_t* row_ptr(int32_t r) const noexcept {
        assert(device == Device::CPU);
        assert(r >= 0 && r < rows());
        return static_cast<const uint8_t*>(data) +
               static_cast<size_t>(r) * static_cast<size_t>(strides[1]);
    }

    uint8_t* row_ptr(int32_t r) noexcept {
        assert(device == Device::CPU);
        assert(r >= 0 && r < rows());
        return static_cast<uint8_t*>(data) +
               static_cast<size_t>(r) * static_cast<size_t>(strides[1]);
    }

    float scale(int32_t r) const noexcept {
        assert(dtype_is_quantized(dtype));
        assert(row_scales != nullptr);
        assert(r >= 0 && r < rows());
        return row_scales[r];
    }
};

// Layout sanity — kept in the header (not just the test suite) so any
// future field reordering fails the build immediately, everywhere Bolt
// is vendored, not just where its own tests happen to run.
static_assert(offsetof(Tensor, device) == 0,      "Tensor::device offset");
static_assert(offsetof(Tensor, dtype)  == 1,      "Tensor::dtype offset");
static_assert(offsetof(Tensor, dims)   == 4,      "Tensor::dims offset (4-byte aligned)");
static_assert(offsetof(Tensor, strides) == 24,    "Tensor::strides offset (8-byte aligned)");
static_assert(offsetof(Tensor, data)   == 56,     "Tensor::data offset");
static_assert(offsetof(Tensor, row_scales) == 64, "Tensor::row_scales offset");
static_assert(sizeof(Tensor) == 72,                "Tensor must be exactly 72 bytes");
static_assert(std::is_trivially_copyable<Tensor>::value,
              "Tensor must be POD/memcpy-safe (Tiger Style)");

// ---------------------------------------------------------------------------
// Factory — CPU 2D view (the QTensor-equivalent construction path). Not a
// ctor (Tiger Style: POD, no ctors) — a plain function returning a value,
// matching bolt_zonemap.h's `zone_make_empty_*()` idiom.
// ---------------------------------------------------------------------------
inline Tensor tensor_make_cpu_2d(DType dtype, int32_t rows, int32_t cols,
                                  void* data, float* row_scales = nullptr) noexcept {
    assert(rows >= 0 && cols >= 0);
    assert(data != nullptr || rows == 0 || cols == 0);
    assert(row_scales != nullptr || !dtype_is_quantized(dtype));

    Tensor t;
    t.device = Device::CPU;
    t.dtype  = dtype;
    t.dims[0] = cols;
    t.dims[1] = rows;
    t.dims[2] = 1;
    t.dims[3] = 1;

    const int64_t row_stride = dtype_row_stride_bytes(dtype, cols);
    t.strides[0] = dtype_element_bytes(dtype);  // 0 sentinel for Int4/Int2
    t.strides[1] = row_stride;
    t.strides[2] = row_stride * rows;
    t.strides[3] = t.strides[2];

    t.data       = data;
    t.row_scales = row_scales;
    return t;
}

}  // namespace bolt
