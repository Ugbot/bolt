// bolt_partition_route.h — route rows to partitions by the values of key
// columns, in one pass (partitioned writes: Spark partitionBy, Hive layout).
//
// A row's partition key is its key cells in key_encode bytes, each prefixed
// by a presence byte (0 = NULL, 1 = value), so NULL is a partition value of
// its own. Partitions are numbered in first-seen order; the output groups
// row ids by partition (stable: each partition's rows in input order).
//
// Tiger Style: arena-owned output, open addressing sized from the caller's
// partition budget, every loop bounded by rows / keys / table slots.

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_hash.h"
#include "bolt/kernels/bolt_key_encode.h"

namespace bolt {

struct PartitionRoute {
    uint32_t        n_parts = 0;
    int64_t         n_rows = 0;
    uint32_t*       part_of_row = nullptr;  // [n_rows]
    uint32_t*       offsets = nullptr;      // [n_parts + 1] into rows
    uint32_t*       rows = nullptr;         // [n_rows] row ids grouped by partition
    uint32_t*       first_row = nullptr;    // [n_parts] a row holding the partition's values
};

enum class PartitionRouteStatus : uint8_t {
    kOk = 0,
    kUnsupported,   // a key type with no key encoding
    kTooManyParts,  // more distinct keys than max_parts
    kKeyTooLong,    // one row's key past kKeyEncodeMaxBytes
    kNoMemory,
};

namespace partition_route_detail {

inline PartitionRouteStatus encode(const BoltColumn* cols, const uint32_t* key_cols, uint32_t n_key, int64_t r,
                                   uint8_t* dst, uint32_t cap, uint32_t* len) noexcept {
    uint32_t at = 0;
    for (uint32_t i = 0; i < n_key; ++i) {                      // bounded: n_key
        const BoltColumn& c = cols[key_cols[i]];
        if (at + 1u > cap) return PartitionRouteStatus::kKeyTooLong;
        if (key_cell_null(c, r)) { dst[at++] = 0u; continue; }
        dst[at++] = 1u;
        uint32_t n = 0;
        const KeyEncodeStatus s = key_encode_cell(c, r, dst + at, cap - at, &n);
        if (s == KeyEncodeStatus::kNoRoom) return PartitionRouteStatus::kKeyTooLong;
        if (s != KeyEncodeStatus::kOk) return PartitionRouteStatus::kUnsupported;
        at += n;
    }
    *len = at;
    return PartitionRouteStatus::kOk;
}

}  // namespace partition_route_detail

// Route rows [0, n_rows) of `cols` by key_cols[0..n_key). At most max_parts
// distinct keys (kTooManyParts past it: the caller names its budget).
inline PartitionRouteStatus partition_route(const BoltColumn* cols, const uint32_t* key_cols, uint32_t n_key,
                                            int64_t n_rows, uint32_t max_parts, Arena* ar,
                                            PartitionRoute* out) noexcept {
    namespace d = partition_route_detail;
    assert(cols != nullptr && key_cols != nullptr && ar != nullptr && out != nullptr);
    assert(n_rows >= 0 && max_parts >= 1u);
    *out = PartitionRoute{};
    if (n_key == 0u || n_key > kKeyEncodeMaxCells || n_rows > int64_t{UINT32_MAX})
        return PartitionRouteStatus::kUnsupported;
    for (uint32_t i = 0; i < n_key; ++i)                        // bounded: n_key
        if (key_cell_kind(cols[key_cols[i]].type) == KeyCellKind::kNone) return PartitionRouteStatus::kUnsupported;
    const auto rows = static_cast<size_t>(n_rows);
    const uint32_t cap_parts = n_rows < int64_t{max_parts} ? static_cast<uint32_t>(n_rows) + 1u : max_parts;
    uint32_t slots = 16u;
    while (slots < cap_parts * 2u && slots < (1u << 31)) slots <<= 1u;   // bounded: 32 doublings
    auto* table = static_cast<uint32_t*>(ar->allocate(sizeof(uint32_t) * slots, 64));
    auto* hashes = ar->allocate_array<uint64_t>(cap_parts);
    auto* keys = ar->allocate_array<const uint8_t*>(cap_parts);
    auto* key_len = ar->allocate_array<uint32_t>(cap_parts);
    out->part_of_row = ar->allocate_array<uint32_t>(rows + 1u);
    out->first_row = ar->allocate_array<uint32_t>(cap_parts);
    auto* scratch = static_cast<uint8_t*>(ar->allocate(kKeyEncodeMaxBytes, 16));
    if (table == nullptr || hashes == nullptr || keys == nullptr || key_len == nullptr ||
        out->part_of_row == nullptr || out->first_row == nullptr || scratch == nullptr)
        return PartitionRouteStatus::kNoMemory;
    std::memset(table, 0xFF, sizeof(uint32_t) * slots);
    uint32_t n_parts = 0;
    for (int64_t r = 0; r < n_rows; ++r) {                      // bounded: n_rows
        uint32_t len = 0;
        const PartitionRouteStatus s = d::encode(cols, key_cols, n_key, r, scratch, kKeyEncodeMaxBytes, &len);
        if (s != PartitionRouteStatus::kOk) return s;
        const uint64_t h = hash_bytes(scratch, len);
        uint32_t at = static_cast<uint32_t>(h) & (slots - 1u);
        uint32_t part = UINT32_MAX;
        for (uint32_t probe = 0; probe < slots; ++probe) {      // bounded: table slots
            const uint32_t p = table[at];
            if (p == UINT32_MAX) break;
            if (hashes[p] == h && key_len[p] == len && std::memcmp(keys[p], scratch, len) == 0) { part = p; break; }
            at = (at + 1u) & (slots - 1u);
        }
        if (part == UINT32_MAX) {
            if (n_parts == max_parts) return PartitionRouteStatus::kTooManyParts;
            auto* k = static_cast<uint8_t*>(ar->allocate(len + 1u, 1));
            if (k == nullptr) return PartitionRouteStatus::kNoMemory;
            std::memcpy(k, scratch, len);
            part = n_parts++;
            hashes[part] = h; keys[part] = k; key_len[part] = len;
            out->first_row[part] = static_cast<uint32_t>(r);
            table[at] = part;
        }
        out->part_of_row[r] = part;
    }
    out->offsets = ar->allocate_array<uint32_t>(n_parts + 1u);
    out->rows = ar->allocate_array<uint32_t>(rows + 1u);
    if (out->offsets == nullptr || out->rows == nullptr) return PartitionRouteStatus::kNoMemory;
    std::memset(out->offsets, 0, sizeof(uint32_t) * (n_parts + 1u));
    for (int64_t r = 0; r < n_rows; ++r) ++out->offsets[out->part_of_row[r] + 1u];   // bounded: n_rows
    for (uint32_t p = 0; p < n_parts; ++p) out->offsets[p + 1u] += out->offsets[p];   // bounded: n_parts
    auto* fill = ar->allocate_array<uint32_t>(n_parts + 1u);
    if (fill == nullptr) return PartitionRouteStatus::kNoMemory;
    std::memcpy(fill, out->offsets, sizeof(uint32_t) * (n_parts + 1u));
    for (int64_t r = 0; r < n_rows; ++r) out->rows[fill[out->part_of_row[r]]++] = static_cast<uint32_t>(r);
    out->n_parts = n_parts;
    out->n_rows = n_rows;
    assert(out->offsets[n_parts] == static_cast<uint32_t>(n_rows));
    return PartitionRouteStatus::kOk;
}

}  // namespace bolt
