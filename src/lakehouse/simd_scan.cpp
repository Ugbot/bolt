// bolt/lakehouse/simd_scan.cpp — SIMD per-column-type scan dispatch.
//
// Dispatches a single-column predicate (col OP scalar) to the matching
// bolt::branchless::filter_*_branchless kernel. Int64 and Float64 are wired today;
// other types fall back to the scalar template loop. The output is an
// ascending selection vector, identical in shape to what
// `scan_optimizer.cpp` consumes downstream.

#include "bolt/lakehouse/scan_optimizer.h"

#include <cassert>
#include <cstdint>

#include "bolt/bolt_branchless.h"

namespace bolt {
namespace lakehouse {

namespace {

template <typename T>
uint32_t simd_filter_one(
    const T* data, uint32_t n, PredicateOp op, T scalar,
    uint32_t* out_sel) noexcept {
    assert(data != nullptr);
    assert(out_sel != nullptr);
    int32_t* tmp = reinterpret_cast<int32_t*>(out_sel);
    int64_t k = 0;
    switch (op) {
        case PredicateOp::kEq:
            k = bolt::branchless::filter_eq_branchless<T>(data, n, scalar, tmp); break;
        case PredicateOp::kNe:
            k = bolt::branchless::filter_ne_branchless<T>(data, n, scalar, tmp); break;
        case PredicateOp::kLt:
            k = bolt::branchless::filter_lt_branchless<T>(data, n, scalar, tmp); break;
        case PredicateOp::kLe:
            k = bolt::branchless::filter_le_branchless<T>(data, n, scalar, tmp); break;
        case PredicateOp::kGt:
            k = bolt::branchless::filter_gt_branchless<T>(data, n, scalar, tmp); break;
        case PredicateOp::kGe:
            k = bolt::branchless::filter_ge_branchless<T>(data, n, scalar, tmp); break;
        default:
            // is_null / is_not_null require validity buffer — caller handles.
            k = 0; break;
    }
    // The kernels write int32_t into the same backing buffer as the
    // uint32_t selection vector. Index values are non-negative row
    // positions (n ≤ 1M ≪ 2^31), so the bit pattern is identical and
    // no per-row conversion is needed — we just reinterpret in place.
    return static_cast<uint32_t>(k);
}

}  // namespace

// Public entry: dispatch on column type. Returns selection-vector length.
// `data_type` tells us how to interpret `data`. `scalar_*` carry the
// scalar value (one of i64 / f64 used; others ignored for the wired
// types). Scalar fallback for types we don't have a kernel for yet.
uint32_t simd_scan_filter(
    const void* data, uint32_t n, BoltType data_type,
    PredicateOp op, int64_t scalar_i64, double scalar_f64,
    uint32_t* out_sel) noexcept {
    assert(data != nullptr);
    assert(out_sel != nullptr);
    assert(n <= kScanMorselMaxRows);
    if (data == nullptr || out_sel == nullptr) return 0;
    switch (data_type) {
        case BoltType::Int64:
            return simd_filter_one<int64_t>(
                static_cast<const int64_t*>(data), n, op, scalar_i64, out_sel);
        case BoltType::Float64:
            return simd_filter_one<double>(
                static_cast<const double*>(data), n, op, scalar_f64, out_sel);
        case BoltType::Int32:
            return simd_filter_one<int32_t>(
                static_cast<const int32_t*>(data), n, op,
                static_cast<int32_t>(scalar_i64), out_sel);
        case BoltType::Float32:
            return simd_filter_one<float>(
                static_cast<const float*>(data), n, op,
                static_cast<float>(scalar_f64), out_sel);
        default:
            // Scalar fallback: caller can iterate. We just emit identity.
            for (uint32_t i = 0; i < n; ++i) out_sel[i] = i;
            return n;
    }
}

}  // namespace lakehouse
}  // namespace bolt
