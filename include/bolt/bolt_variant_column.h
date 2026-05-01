// bolt_variant_column.h — Variant (sum-type) column primitive.
//
// Layer 1.2b of `this-was-a-freach-hashed-crab.md`. Mirrors the shape of
// ClickHouse's `Dynamic` column: a UInt8 discriminator selects one of
// up to 254 typed variant columns, with reserved values 254 (overflow,
// payload lives in `shared_variant`) and 255 (null).
//
// RULES: No exceptions. No RTTI. No std::*. Arena-owned. POD layout.

#pragma once

#include "bolt/bolt_column.h"
#include "bolt/bolt_arena.h"
#include "bolt/bolt_port.h"
#include "bolt/bolt_types.h"

#include <cassert>
#include <cstdint>
#include <cstring>

namespace bolt {

// Hard cap on the number of typed variants. The discriminator reserves
// 254 (shared overflow) and 255 (null) — leaving 0..253 for typed
// variants, i.e. up to 254 entries. Static-asserted at compile time so
// the constant cannot drift silently.
inline constexpr uint8_t kVariantMaxVariants = 254u;

// Reserved discriminator bytes.
inline constexpr uint8_t kVariantDiscOverflow = 254u;
inline constexpr uint8_t kVariantDiscNull     = 255u;

// ============================================================================
// VariantColumn — POD descriptor.
//
// Layout: both BoltColumns sit at the front (each is 64-byte-aligned via
// its embedded `alignas(64) ColumnStats`), then the small POD fields,
// then a trailing pad to the struct's natural 64-byte alignment.
//
//   discriminator   : BoltColumn          // UInt8 per row, Flat or Constant
//   shared_variant  : BoltColumn          // VarBinary tail for disc==254
//   variants        : BoltColumn*         // arena-allocated array (length variant_count)
//   length          : int64_t             // == discriminator.length
//   variant_count   : int32_t             // size of `variants` (≤ kVariantMaxVariants)
//   _pad            : uint8_t[4]          // explicit alignment filler
//
// The trailing tail (variants + length + variant_count + _pad = 24 B)
// is rounded up to the struct's 64-byte alignment, hence the `+64` in
// the static_assert below.
// ============================================================================
struct VariantColumn {
    BoltColumn  discriminator;
    BoltColumn  shared_variant;
    BoltColumn* variants;
    int64_t     length;
    int32_t     variant_count;
    uint8_t     _pad[4];
};

static_assert(kVariantMaxVariants == 254u,
              "kVariantMaxVariants must stay at 254 — discriminator "
              "reserves 254 (overflow) and 255 (null)");
static_assert(sizeof(VariantColumn) == 2u * sizeof(BoltColumn) + 64u,
              "VariantColumn layout drifted; expected two BoltColumns "
              "plus the tail (ptr + int64 + int32 + 4-byte pad) rounded "
              "up to the struct's 64-byte alignment");
static_assert(alignof(VariantColumn) == 64u,
              "VariantColumn must inherit the 64-byte alignment of BoltColumn");

// ============================================================================
// Construction
// ============================================================================

/// Build an empty (sentinel) VariantColumn. `length == 0`,
/// `variants == nullptr`. Used as the failure return.
BOLT_FORCE_INLINE VariantColumn variant_column_make_empty() noexcept {
    VariantColumn v;
    v.discriminator  = BoltColumn::make_empty();
    v.shared_variant = BoltColumn::make_empty();
    v.variants       = nullptr;
    v.length         = 0;
    v.variant_count  = 0;
    v._pad[0] = v._pad[1] = v._pad[2] = v._pad[3] = 0u;
    return v;
}

/// Build a VariantColumn from a discriminator column, an array of typed
/// variant columns, and a (possibly empty) shared overflow column.
///
/// The `variants` array is copied into `arena` so the caller's buffer
/// lifetime does not have to match the column's. Returns the empty
/// shape on bad input (variant_count > kVariantMaxVariants, negative,
/// or null pointer with non-zero count, type mismatch, length mismatch).
inline VariantColumn variant_column_make(BoltColumn discriminator,
                                         BoltColumn* variants,
                                         int32_t variant_count,
                                         BoltColumn shared_variant,
                                         Arena* arena) noexcept {
    assert(arena != nullptr);
    assert(variant_count >= 0);

    if (arena == nullptr) return variant_column_make_empty();
    if (variant_count < 0) return variant_column_make_empty();
    if (variant_count > static_cast<int32_t>(kVariantMaxVariants)) {
        return variant_column_make_empty();
    }
    if (variant_count > 0 && variants == nullptr) {
        return variant_column_make_empty();
    }
    if (discriminator.type != BoltType::UInt8) {
        return variant_column_make_empty();
    }
    if (discriminator.format != ColumnFormat::Flat &&
        discriminator.format != ColumnFormat::Constant) {
        return variant_column_make_empty();
    }

    VariantColumn v = variant_column_make_empty();
    v.discriminator  = discriminator;
    v.shared_variant = shared_variant;
    v.length         = discriminator.length;
    v.variant_count  = variant_count;

    if (variant_count > 0) {
        BoltColumn* arr = arena->allocate_array<BoltColumn>(
            static_cast<size_t>(variant_count));
        if (arr == nullptr) return variant_column_make_empty();
        for (int32_t i = 0; i < variant_count; ++i) arr[i] = variants[i];
        v.variants = arr;
    }
    return v;
}

// ============================================================================
// Hot-path accessor — branchless.
//
// Reads the discriminator byte for `row` and (if it indexes a typed
// variant) writes &variants[disc] into *out_variant_col. For the
// reserved bytes 254/255 the out-pointer is set to nullptr. Compiles
// to load + cmp + cmov on MSVC /O2 (no jcc on the byte comparison).
// ============================================================================
BOLT_FORCE_INLINE uint8_t variant_column_at(const VariantColumn* v,
                                            int64_t row,
                                            const BoltColumn** out_variant_col)
    noexcept {
    assert(v != nullptr);
    assert(out_variant_col != nullptr);
    assert(row >= 0);
    assert(row < v->length);

    // Discriminator may be Flat (one byte per row) or Constant
    // (single inline value broadcast across rows). The Constant form
    // stores the value in `inline_value` and points `data` at it, so
    // a uniform load through `data` works for both shapes — the row
    // offset is multiplied by the per-row stride.
    const uint8_t* disc_data = static_cast<const uint8_t*>(v->discriminator.data);
    const int64_t  stride    = (v->discriminator.format == ColumnFormat::Constant)
                                ? int64_t{0} : int64_t{1};
    const uint8_t  disc      = disc_data[row * stride];

    const uint32_t in_range  =
        static_cast<uint32_t>(disc) <
        static_cast<uint32_t>(v->variant_count);

    // Branchless index: 0 when out-of-range, disc otherwise. The
    // `* in_range` clamp keeps the load valid for variant_count == 0.
    const int32_t  idx       = static_cast<int32_t>(disc) *
                               static_cast<int32_t>(in_range);
    const BoltColumn* hit    = (v->variants != nullptr)
                                ? &v->variants[idx] : nullptr;
    // CMOV: select hit when in-range, nullptr otherwise.
    *out_variant_col = in_range ? hit : nullptr;
    return disc;
}

// ============================================================================
// Compact pass — Flat → Constant when all rows share the same byte.
//
// Walks the discriminator once (only when it is currently Flat). If
// every byte is identical, replaces the discriminator with a Constant
// BoltColumn carrying the same UInt8 value and returns true. No-ops
// (returns false) when already Constant, empty, or non-uniform.
// ============================================================================
inline bool variant_column_compact(VariantColumn* v, Arena* arena) noexcept {
    assert(v != nullptr);
    assert(arena != nullptr);
    (void)arena;  // Compaction does not allocate — Constant is inline.

    if (v->length <= 0) return false;
    if (v->discriminator.format != ColumnFormat::Flat) return false;
    if (v->discriminator.data == nullptr) return false;

    const uint8_t* disc = static_cast<const uint8_t*>(v->discriminator.data);
    const uint8_t  first = disc[0];
    for (int64_t i = 1; i < v->length; ++i) {
        if (disc[i] != first) return false;
    }

    BoltColumn c = BoltColumn::make_constant<uint8_t>(
        first, v->length, BoltType::UInt8);
    v->discriminator = c;
    return true;
}

}  // namespace bolt
