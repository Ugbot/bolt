// bolt/lakehouse/iceberg/eq_key.h — the canonical bytes of an equality-delete key. The scan applies an equality
// delete by these bytes and the writer (G2ICE-283) checks what a delete it writes would remove by the same bytes, so the
// two cannot disagree on what "equal" means.
#pragma once

#include <cstdint>
#include <cstring>

#include "bolt/bolt_column.h"
#include "bolt/lakehouse/iceberg/iceberg_limits.h"

namespace bolt::lakehouse::iceberg::eqkey {

inline uint64_t fnv64(const uint8_t* p, uint64_t n, uint64_t h = 1469598103934665603ull) noexcept {
    for (uint64_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }   // bounded by n
    return h;
}

// True when a column of this decoded type has a defined key equality.
inline bool type_keyable(BoltType t) noexcept {
    switch (t) {
        case BoltType::Bool: case BoltType::Int8: case BoltType::Int16: case BoltType::Int32: case BoltType::Int64:
        case BoltType::UInt8: case BoltType::UInt16: case BoltType::UInt32: case BoltType::UInt64:
        case BoltType::Date32: case BoltType::Date64: case BoltType::Timestamp: case BoltType::Duration:
        case BoltType::Decimal64: case BoltType::Float32: case BoltType::Float64: case BoltType::Utf8:
        case BoltType::Binary: return true;
        default: return false;
    }
}

// Row r's key columns: integer-like types as a signed 64-bit value, floats by value (-0.0 == 0.0), strings by bytes, NULL
// equal to NULL. False on a type with no defined equality or a key past limits::kEqKeyMax bytes.
inline bool encode(const BoltColumn* const* cols, uint32_t n, int64_t r, uint8_t* buf, uint32_t* len) noexcept {
    uint32_t o = 0;
    for (uint32_t c = 0; c < n; ++c) {                                    // bounded by n
        const BoltColumn& col = *cols[c];
        if (o + 9 > limits::kEqKeyMax || !type_keyable(col.type)) return false;
        if (col.is_null(r)) { buf[o++] = 0; continue; }
        if (col.type == BoltType::Utf8 || col.type == BoltType::Binary) {
            const uint8_t* d = nullptr; int32_t sl = 0;
            col.utf8_at(r, &d, &sl);
            if (sl < 0 || o + 5 + static_cast<uint32_t>(sl) > limits::kEqKeyMax) return false;
            buf[o++] = 3; const uint32_t u = static_cast<uint32_t>(sl); std::memcpy(buf + o, &u, 4); o += 4;
            if (sl > 0) std::memcpy(buf + o, d, static_cast<size_t>(sl));
            o += static_cast<uint32_t>(sl);
            continue;
        }
        if (col.type == BoltType::Float32 || col.type == BoltType::Float64) {
            double d = col.type == BoltType::Float32 ? static_cast<double>(static_cast<const float*>(col.data)[r])
                                                      : static_cast<const double*>(col.data)[r];
            if (d == 0.0) d = 0.0;
            buf[o++] = 2; std::memcpy(buf + o, &d, 8); o += 8;
            continue;
        }
        int64_t v = 0;
        switch (col.type_size_bytes) {
            case 1: v = static_cast<const int8_t*>(col.data)[r]; break;
            case 2: v = static_cast<const int16_t*>(col.data)[r]; break;
            case 4: v = static_cast<const int32_t*>(col.data)[r]; break;
            default: v = static_cast<const int64_t*>(col.data)[r]; break;
        }
        buf[o++] = 1; std::memcpy(buf + o, &v, 8); o += 8;
    }
    *len = o;
    return true;
}

}  // namespace bolt::lakehouse::iceberg::eqkey
