// Indexed numeric relationship weights without an O(edges) staging buffer.
// The column is indexed by global edge ID - id_base, not CSR slot. Reading
// bytes with memcpy also supports a Float64 column stored in an Int64 lane.
// Caller owns immutable values/validity/IDs for the search lifetime. Only
// reached, label-kept edges are read; malformed IDs and NULLs refuse there.
#pragma once

#include "bolt/kernels/bolt_csr_algo.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace bolt::kernels {

template <typename Value, typename Eid>
struct CsrIndexedWeights {
    static_assert(std::is_same_v<Value, int64_t> || std::is_same_v<Value, double>);
    static_assert(std::is_same_v<Eid, int64_t> || std::is_same_v<Eid, uint32_t>);
    const void* values;
    const uint8_t* validity; // LSB-first; nullptr means all valid.
    const Eid* edge_ids;
    int64_t id_base;
    int64_t count;
};

template <typename Value, typename Eid>
inline bool csr_dj_weights_ok(const CsrIndexedWeights<Value, Eid>& w,
                              int64_t edges) noexcept {
    assert(edges >= 0);
    assert(sizeof(Value) == 8);
    return w.id_base >= 0 && w.count >= 0 &&
        static_cast<uint64_t>(w.count) <= SIZE_MAX / sizeof(Value) &&
        (edges == 0 || (w.values != nullptr && w.edge_ids != nullptr));
}

template <typename Value, typename Eid>
inline CsrAlgoStatus csr_dj_weight(const CsrIndexedWeights<Value, Eid>& w,
                                  int64_t slot, double* out) noexcept {
    assert(w.values != nullptr && w.edge_ids != nullptr && out != nullptr);
    assert(slot >= 0 && w.id_base >= 0);
    const int64_t id = static_cast<int64_t>(w.edge_ids[slot]);
    if (id < w.id_base) return CsrAlgoStatus::MalformedGraph;
    const int64_t row = id - w.id_base;
    if (row >= w.count) return CsrAlgoStatus::MalformedGraph;
    if (w.validity != nullptr && ((w.validity[row >> 3] >> (row & 7)) & 1u) == 0)
        return CsrAlgoStatus::NegativeWeight; // Same refusal as a NaN weight.
    Value value;
    std::memcpy(&value, static_cast<const uint8_t*>(w.values) +
                       static_cast<size_t>(row) * sizeof(Value), sizeof(Value));
    *out = static_cast<double>(value);
    return CsrAlgoStatus::Ok;
}

// Full searches can profit from contiguous weights despite the O(edges)
// gather. Keep that strategy explicit; NULL maps to NaN so Dijkstra retains
// its reached-edge validation semantics. Bad IDs always refuse the gather.
template <typename Value, typename Eid>
inline CsrAlgoStatus csr_gather_indexed_weights(
        const CsrIndexedWeights<Value, Eid>& weights, int64_t edges,
        double* BOLT_RESTRICT out, int64_t capacity) noexcept {
    assert(edges >= 0);
    assert(capacity >= 0);
    if (edges < 0 || capacity < edges || out == nullptr ||
        !csr_dj_weights_ok(weights, edges)) return CsrAlgoStatus::InvalidArgument;
    for (int64_t slot = 0; slot < edges; ++slot) {
        const CsrAlgoStatus status = csr_dj_weight(weights, slot, out + slot);
        if (status == CsrAlgoStatus::NegativeWeight)
            out[slot] = std::numeric_limits<double>::quiet_NaN();
        else if (status != CsrAlgoStatus::Ok) return status;
    }
    return CsrAlgoStatus::Ok;
}

} // namespace bolt::kernels
