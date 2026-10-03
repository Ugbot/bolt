// bolt_key_encode.h — canonical order-preserving key bytes (MSEG §6.5, G6).
//
// One encoding for every key kind, so a key of any kinds orders under
// memcmp (shorter prefix first) exactly as its values order:
//
//   signed ints, dates, timestamps, durations, decimals (two's complement,
//   any width)            big-endian with the sign bit flipped
//   unsigned ints, IPv4   big-endian
//   bool                  one byte, 0 / 1
//   float32 / float64     total order: positive -> sign bit set, negative ->
//                         every bit inverted; -0.0 is +0.0, every NaN one NaN
//                         (above +inf)
//   UUID, FixedSizeBinary the raw bytes (a declared fixed-width hash key is
//                         this kind: no verify step)
//   Utf8 / Binary / Symbol each 0x00 written 0x00 0xFF, then 0x00 0x00
//
// A composite key is the concatenation of its cells. The text terminator is
// what lets a text cell sit anywhere in a composite: 0x00 0x00 sorts below
// every continuation (0x00 0xFF or any byte >= 0x01), so "a" < "a\0" < "ab".
// Fixed cells need no terminator. NULL is not a key value (kNull).
//
// Tiger Style: POD + free functions, noexcept, no allocation except the
// whole-column helper's caller arena; every loop bounded by a key length.

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_port.h"
#include "bolt/bolt_types.h"
#include "bolt/kernels/bolt_key_encode_limits.h"

namespace bolt {

enum class KeyEncodeStatus : uint8_t {
    kOk = 0,
    kNull,          // a key cell is NULL
    kUnsupported,   // a type / format with no key encoding
    kNoRoom,        // dst too small, or the key past kKeyEncodeMaxBytes
    kNoMemory,
};

inline constexpr uint8_t  kKeyTextEscape  = 0x00;
inline constexpr uint8_t  kKeyTextEscaped = 0xFF;   // follows an escaped 0x00
inline constexpr uint8_t  kKeyTextTerm    = 0x00;   // twice: the terminator

enum class KeyCellKind : uint8_t { kNone = 0, kSigned, kUnsigned, kBool, kFloat, kRaw, kText };

BOLT_FORCE_INLINE KeyCellKind key_cell_kind(BoltType t) noexcept {
    switch (t) {
        case BoltType::Int8: case BoltType::Int16: case BoltType::Int32: case BoltType::Int64:
        case BoltType::Date32: case BoltType::Date64: case BoltType::Timestamp:
        case BoltType::Duration: case BoltType::Decimal64: case BoltType::Decimal128:
        case BoltType::Decimal256:
            return KeyCellKind::kSigned;
        case BoltType::UInt8: case BoltType::UInt16: case BoltType::UInt32: case BoltType::UInt64:
        case BoltType::IPv4:
            return KeyCellKind::kUnsigned;
        case BoltType::Bool:    return KeyCellKind::kBool;
        case BoltType::Float32: case BoltType::Float64: return KeyCellKind::kFloat;
        case BoltType::UUID: case BoltType::FixedSizeBinary: return KeyCellKind::kRaw;
        case BoltType::Utf8: case BoltType::Binary: case BoltType::Symbol: return KeyCellKind::kText;
        default: return KeyCellKind::kNone;
    }
}

// Two's complement little-endian `src[0..w)` as order-preserving bytes.
BOLT_FORCE_INLINE void key_put_int(uint8_t* dst, const uint8_t* src, uint32_t w,
                                   bool is_signed) noexcept {
    assert(dst != nullptr && src != nullptr);
    assert(w >= 1 && w <= 32);
    for (uint32_t i = 0; i < w; ++i) dst[i] = src[w - 1 - i];   // bounded: w
    if (is_signed) dst[0] = static_cast<uint8_t>(dst[0] ^ 0x80u);
}

BOLT_FORCE_INLINE uint64_t key_f64_bits(double v) noexcept {
    if (v != v) return UINT64_MAX;                 // every NaN: one, above +inf
    if (v == 0.0) v = 0.0;                         // -0.0 is 0.0
    uint64_t b;
    std::memcpy(&b, &v, 8);
    const uint64_t out = (b >> 63) ? ~b : (b | (uint64_t{1} << 63));
    assert(out != UINT64_MAX);
    return out;
}

BOLT_FORCE_INLINE uint32_t key_f32_bits(float v) noexcept {
    if (v != v) return UINT32_MAX;
    if (v == 0.0f) v = 0.0f;
    uint32_t b;
    std::memcpy(&b, &v, 4);
    const uint32_t out = (b >> 31) ? ~b : (b | (uint32_t{1} << 31));
    assert(out != UINT32_MAX);
    return out;
}

BOLT_FORCE_INLINE uint32_t key_text_encoded_len(const uint8_t* p, uint32_t n) noexcept {
    assert(p != nullptr || n == 0);
    uint32_t z = 0;
    for (uint32_t i = 0; i < n; ++i) z += p[i] == kKeyTextEscape ? 1u : 0u;   // bounded: n
    assert(z <= n);
    return n + z + 2u;
}

// Escaped text + terminator into dst[0..cap); the bytes written, or 0 when
// cap is short (an encoded text is never empty).
inline uint32_t key_put_text(uint8_t* dst, uint32_t cap, const uint8_t* p, uint32_t n) noexcept {
    assert(dst != nullptr || cap == 0);
    assert(p != nullptr || n == 0);
    uint32_t o = 0;
    for (uint32_t i = 0; i < n; ++i) {                          // bounded: n
        if (o + 2u > cap) return 0;
        dst[o++] = p[i];
        if (p[i] == kKeyTextEscape) dst[o++] = kKeyTextEscaped;
    }
    if (o + 2u > cap) return 0;
    dst[o++] = kKeyTextTerm;
    dst[o++] = kKeyTextTerm;
    return o;
}

// Row r's string bytes (Flat StringView + overflow, VarBinary, Constant).
inline bool key_cell_text(const BoltColumn& c, int64_t r, const uint8_t** p, uint32_t* n) noexcept {
    assert(p != nullptr && n != nullptr);
    assert(r >= 0 && r < c.length);
    if (c.format == ColumnFormat::VarBinary) {
        if (c.dict_child == nullptr || c.dict_child->data == nullptr) return false;
        const auto* off = static_cast<const int32_t*>(c.dict_child->data);
        *p = static_cast<const uint8_t*>(c.data) + off[r];
        *n = static_cast<uint32_t>(off[r + 1] - off[r]);
        return true;
    }
    const StringView* v;
    if (c.format == ColumnFormat::Constant) v = reinterpret_cast<const StringView*>(c.inline_value);
    else if (c.format == ColumnFormat::Flat && c.data != nullptr) v = static_cast<const StringView*>(c.data) + r;
    else return false;
    *n = v->length;
    if (v->length <= 12u) { *p = reinterpret_cast<const uint8_t*>(v->prefix); return true; }
    if (c.str_overflow_base == nullptr) return false;
    *p = static_cast<const uint8_t*>(c.str_overflow_base) + v->ref.offset;
    return true;
}

BOLT_FORCE_INLINE bool key_cell_null(const BoltColumn& c, int64_t r) noexcept {
    assert(r >= 0 && r < c.length);
    if (c.validity == nullptr) return false;
    const int64_t b = r + (c.format == ColumnFormat::Flat ? c.validity_offset : 0);
    return ((c.validity[b >> 3] >> (b & 7)) & 1u) == 0;
}

// Fixed cell value bytes (little-endian slot) of row r, and the slot width.
inline const uint8_t* key_cell_fixed(const BoltColumn& c, int64_t r, uint32_t* w) noexcept {
    assert(w != nullptr);
    assert(r >= 0 && r < c.length);
    const uint32_t slot = c.type_size_bytes ? c.type_size_bytes : static_cast<uint32_t>(type_size(c.type));
    *w = (c.type == BoltType::FixedSizeBinary && c.fixed_width) ? c.fixed_width : slot;
    if (slot == 0 || slot > 32u) return nullptr;
    if (c.format == ColumnFormat::Constant) return slot <= 16u ? c.inline_value : nullptr;
    if (c.format != ColumnFormat::Flat || c.data == nullptr) return nullptr;
    return static_cast<const uint8_t*>(c.data) + static_cast<size_t>(r) * slot;
}

// Encoded bytes of one cell, without writing (0 = cannot encode).
inline uint32_t key_cell_len(const BoltColumn& c, int64_t r) noexcept {
    assert(r >= 0 && r < c.length);
    const KeyCellKind k = key_cell_kind(c.type);
    if (k == KeyCellKind::kNone || key_cell_null(c, r)) return 0;
    if (k == KeyCellKind::kText) {
        const uint8_t* p = nullptr;
        uint32_t n = 0;
        return key_cell_text(c, r, &p, &n) ? key_text_encoded_len(p, n) : 0;
    }
    uint32_t w = 0;
    const uint8_t* v = key_cell_fixed(c, r, &w);
    if (k == KeyCellKind::kBool) return v ? 1u : 0u;
    return v ? w : 0u;
}

// Row r of one column as a key cell into dst[0..cap); *len = bytes written.
inline KeyEncodeStatus key_encode_cell(const BoltColumn& c, int64_t r, uint8_t* dst, uint32_t cap,
                                       uint32_t* len) noexcept {
    assert(len != nullptr && (dst != nullptr || cap == 0));
    assert(r >= 0 && r < c.length);
    *len = 0;
    const KeyCellKind k = key_cell_kind(c.type);
    if (k == KeyCellKind::kNone) return KeyEncodeStatus::kUnsupported;
    if (key_cell_null(c, r)) return KeyEncodeStatus::kNull;
    if (k == KeyCellKind::kText) {
        const uint8_t* p = nullptr;
        uint32_t n = 0;
        if (!key_cell_text(c, r, &p, &n)) return KeyEncodeStatus::kUnsupported;
        *len = key_put_text(dst, cap, p, n);
        return *len ? KeyEncodeStatus::kOk : KeyEncodeStatus::kNoRoom;
    }
    uint32_t w = 0;
    const uint8_t* v = key_cell_fixed(c, r, &w);
    if (v == nullptr) return KeyEncodeStatus::kUnsupported;
    if (k == KeyCellKind::kBool) w = 1;
    if (w > cap) return KeyEncodeStatus::kNoRoom;
    if (k == KeyCellKind::kBool) {
        dst[0] = v[0] ? 1u : 0u;
    } else if (k == KeyCellKind::kFloat && w == 8) {
        double d;
        std::memcpy(&d, v, 8);
        const uint64_t b = key_f64_bits(d);
        key_put_int(dst, reinterpret_cast<const uint8_t*>(&b), 8, false);
    } else if (k == KeyCellKind::kFloat) {
        float f;
        std::memcpy(&f, v, 4);
        const uint32_t b = key_f32_bits(f);
        key_put_int(dst, reinterpret_cast<const uint8_t*>(&b), 4, false);
    } else if (k == KeyCellKind::kRaw) {
        std::memcpy(dst, v, w);
    } else {
        key_put_int(dst, v, w, k == KeyCellKind::kSigned);
    }
    *len = w;
    return KeyEncodeStatus::kOk;
}

// Row r's key: cells cols[key_cols[0..n_key)] concatenated.
inline KeyEncodeStatus key_encode_row(const BoltColumn* cols, const uint32_t* key_cols,
                                      uint32_t n_key, int64_t r, uint8_t* dst, uint32_t cap,
                                      uint32_t* len) noexcept {
    assert(cols != nullptr && key_cols != nullptr && len != nullptr);
    assert(n_key >= 1 && n_key <= kKeyEncodeMaxCells);
    uint32_t at = 0;
    for (uint32_t i = 0; i < n_key; ++i) {                      // bounded: n_key
        uint32_t n = 0;
        const KeyEncodeStatus s = key_encode_cell(cols[key_cols[i]], r, dst + at, cap - at, &n);
        if (s != KeyEncodeStatus::kOk) { *len = 0; return s; }
        at += n;
    }
    *len = at;
    return KeyEncodeStatus::kOk;
}

// Every row's key as a Flat Binary StringView column (+ overflow), arena
// owned. Two passes: lengths (into the views), then bytes.
inline KeyEncodeStatus key_encode_column(const BoltColumn* cols, const uint32_t* key_cols,
                                         uint32_t n_key, int64_t rows, Arena* ar,
                                         BoltColumn* out) noexcept {
    assert(cols != nullptr && key_cols != nullptr && ar != nullptr && out != nullptr);
    assert(rows >= 0);
    if (n_key == 0 || n_key > kKeyEncodeMaxCells) return KeyEncodeStatus::kUnsupported;
    auto* v = static_cast<StringView*>(ar->allocate_zeroed(static_cast<size_t>(rows) * sizeof(StringView) + 16, 16));
    if (v == nullptr) return KeyEncodeStatus::kNoMemory;
    uint64_t over = 0;
    for (int64_t r = 0; r < rows; ++r) {                        // bounded: rows
        uint64_t n = 0;
        for (uint32_t i = 0; i < n_key; ++i) {                  // bounded: n_key
            const BoltColumn& c = cols[key_cols[i]];
            const uint32_t l = key_cell_len(c, r);
            if (l == 0) return key_cell_null(c, r) ? KeyEncodeStatus::kNull : KeyEncodeStatus::kUnsupported;
            n += l;
        }
        if (n > kKeyEncodeMaxBytes) return KeyEncodeStatus::kNoRoom;
        v[r].length = static_cast<uint32_t>(n);
        over += n > 12u ? n : 0u;
    }
    auto* ob = static_cast<uint8_t*>(ar->allocate(static_cast<size_t>(over) + 16, 16));
    if (ob == nullptr) return KeyEncodeStatus::kNoMemory;
    uint64_t o = 0;
    for (int64_t r = 0; r < rows; ++r) {                        // bounded: rows
        const uint32_t want = v[r].length;
        uint8_t small[16];
        uint8_t* dst = want <= 12u ? small : ob + o;
        uint32_t n = 0;
        const KeyEncodeStatus s = key_encode_row(cols, key_cols, n_key, r, dst, want, &n);
        if (s != KeyEncodeStatus::kOk) return s;
        assert(n == want);
        std::memcpy(v[r].prefix, dst, n < 4u ? n : 4u);
        if (n <= 12u) { if (n > 4u) std::memcpy(v[r].inline_data, dst + 4, n - 4u); continue; }
        v[r].ref.offset = static_cast<uint32_t>(o);
        o += n;
    }
    assert(o == over);
    *out = BoltColumn::make_flat(v, nullptr, rows, BoltType::Binary);
    out->type_size_bytes = sizeof(StringView);
    out->str_overflow_base = ob;
    return KeyEncodeStatus::kOk;
}

}  // namespace bolt
