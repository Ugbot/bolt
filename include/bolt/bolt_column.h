// bolt_column.h — Adaptive column with stats, sidecars, C Data Interface export
//
// RULES: No exceptions. No RTTI. No smart pointers. No std::string. No std::vector.
// Arena-owned memory. Returns nullptr/false on failure. Fixed-size inline stats.
//
// Arrow compatibility: zero-copy export via ArrowArray/ArrowSchema C structs.
// No libarrow link required. Any Arrow consumer (Polars, DuckDB, Pandas) can
// read our columns directly through the C Data Interface.

#pragma once

#include "bolt/bolt_types.h"
#include "bolt/bolt_arena.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <atomic>
#include <type_traits>

namespace bolt {

// ============================================================================
// Column Statistics — exactly 64 bytes (one cache line), always inline
// ============================================================================

enum class CardinalityClass : uint8_t {
    Unknown  = 0, Constant = 1, Low = 2, Medium = 3, High = 4,
};

enum class SortOrder : uint8_t {
    Unknown = 0, Ascending = 1, Descending = 2, NearlySorted = 3, Unsorted = 4,
};

struct alignas(64) ColumnStats {
    int64_t  min_value;        // Type-punned for fixed-width types    (0..7)
    int64_t  max_value;        //                                       (8..15)
    uint32_t null_count;       //                                       (16..19)
    uint32_t distinct_count;   // Exact if < 1024, HLL approx otherwise (20..23)
    CardinalityClass cardinality;   //                                   24
    SortOrder sort_order;           //                                   25
    bool     all_valid;             //                                   26
    bool     is_monotonic;          //                                   27
    uint32_t max_string_len;        //                                   28..31
    uint32_t total_string_bytes;    //                                   32..35
    uint16_t avg_string_len;        //                                   36..37
    bool     all_inline;            //                                   38
    uint8_t  _pad_align;            // 39: align `mean` to 4-byte offset 40
    // Numeric Welford sketch — populated by `stats_scan_typed`. When
    // `stats_scan_typed` has not run yet these are NaN sentinels so
    // downstream consumers (MarbleDB PDX-BOND σ-bound) can detect the
    // "stats unavailable" case without an extra flag. 8 bytes used of
    // the original 13-byte pad region; 4 bytes remain via `_pad`.
    float    mean;                  //                                   40..43
    float    variance;              //                                   44..47
    uint8_t  _pad[4];               //                                   48..51

    bool can_skip_eq(int64_t v) const noexcept { return v < min_value || v > max_value; }
    bool can_skip_gt(int64_t v) const noexcept { return max_value <= v; }
    bool can_skip_lt(int64_t v) const noexcept { return min_value >= v; }

    void invalidate() noexcept {
        cardinality = CardinalityClass::Unknown;
        sort_order = SortOrder::Unknown;
    }

    static ColumnStats make_default() noexcept {
        ColumnStats s;
        memset(&s, 0, sizeof(s));
        s.all_valid = true;
        // Mean / variance default to NaN so callers that do not populate
        // them via `compute_stats_numeric` can detect the gap (used by
        // MarbleDB PDX-BOND σ-bound to fall back to the min/max bound).
        const uint32_t nan_bits = 0x7FC00000u;
        memcpy(&s.mean, &nan_bits, sizeof(float));
        memcpy(&s.variance, &nan_bits, sizeof(float));
        return s;
    }
};

static_assert(sizeof(ColumnStats) == 64, "ColumnStats must be one cache line");
static_assert(offsetof(ColumnStats, mean) == 40,
              "ColumnStats::mean must sit at offset 40 (4-byte-aligned pad slot)");
static_assert(offsetof(ColumnStats, variance) == 44,
              "ColumnStats::variance must sit at offset 44");

// ============================================================================
// Sidecar Index Slots — arena-allocated, ephemeral per morsel
// ============================================================================

// Per-dimension statistics for a vector / Embedding column. Sized by
// the column's `fixed_size` (== dim). Arena-allocated; the four
// parallel float arrays back PDX-BOND zone-rank scoring
// (`bond_zone_rank_scores_f32`) and the σ-bound column-level
// early-termination kernels (`sigma_bound_remaining_*_f32`).
//
// Tiger Style: POD, no constructors, fixed-size pointers, no smart
// pointers. The dim slot is replicated here so kernels can validate
// against the BoltColumn's `fixed_size` in one read.
struct VectorStats {
    float*   min;         // [dim]   per-dim minimum
    float*   max;         // [dim]   per-dim maximum
    float*   mean;        // [dim]   per-dim mean (Welford)
    float*   variance;    // [dim]   per-dim variance (Welford M2 / (n-1))
    uint32_t dim;         //         redundant copy of column.fixed_size
    uint32_t row_count;   //         row count used to compute the stats
};

static_assert(sizeof(VectorStats) == 40,
              "VectorStats POD layout drift — kernels assume 40 bytes");

struct SidecarSlots {
    void* bitmap_index;    // BitmapIndex*
    void* hash_index;      // HashIndex*
    void* sort_index;      // int32_t* permutation
    void* bloom_filter;    // BloomFilter*
    void* vector_stats;    // VectorStats* — Embedding columns only
};

// ============================================================================
// Column Format
// ============================================================================

enum class ColumnFormat : uint8_t {
    Flat       = 0,   // Contiguous typed array
    Constant   = 1,   // Single value, zero per-row storage
    Dictionary = 2,   // Keys array + values column
    Sequence   = 3,   // value[i] = offset + i * step
    View       = 4,   // Zero-copy slice of parent
    RLE        = 5,   // Run-length: `data` = values[num_runs];
                      // `dict_child` = int32 run_ends[num_runs]
                      // where run i covers [run_ends[i-1], run_ends[i]).
    BitPacked  = 6,   // Bit-packed: `data` = uint64 words, each value uses
                      // `seq_step` bits (reused from union). Width ∈ [1,32].
    FrameOfRef = 7,   // FOR = base + bit-packed deltas: `data` = delta words,
                      // `seq_offset` = base value, `seq_step` = bit width.
    VarBinary  = 8,   // Variable-width payload (Utf8 / Binary / JSONB / etc.).
                      // `data`  = flat byte buffer (concatenated payloads).
                      // `dict_child` = Flat Int32 column of length+1 offsets.
                      //                row i spans bytes [offsets[i], offsets[i+1]).
                      // `validity` = optional null bitmap (Arrow shape).
                      // `type_size_bytes` = 0 (variable width).
    Nested     = 9,   // LIST / MAP / STRUCT (and VARIANT, which is a STRUCT).
                      // `type` says which. See the note below.
};

// ---------------------------------------------------------------------------
// ColumnFormat::Nested
// ---------------------------------------------------------------------------
//
// One format carries every nested shape, distinguished by `BoltColumn::type`:
//
//   type == List   : `data` = BoltColumn[1], the ELEMENT column.
//                    `dict_child` = Flat Int32 column of `length + 1` offsets
//                    measured in ELEMENTS -- row i spans elements
//                    [offsets[i], offsets[i+1]).
//                    `validity` distinguishes a NULL list from an EMPTY one;
//                    an empty list is valid with offsets[i] == offsets[i+1].
//   type == Map    : as List, whose element column is a Struct of {key, value}.
//   type == Struct : `data` = BoltColumn[n_children], the fields, in schema
//                    order. `dict_child` is nullptr -- a struct has one value
//                    per row, so there are no offsets.
//
//   `seq_offset` (the format-specific union, unused by Nested otherwise)
//   holds n_children. `type_size_bytes` is 0: a nested column has no per-row
//   value buffer of its own.
//
// WHY THIS SHAPE. The obvious alternative -- reusing `dict_child` for the
// element column the way VarBinary reuses it for byte offsets -- does not
// work: a typed list needs offsets in ELEMENTS *and* a typed child, which is
// two things, and `dict_child` is one slot already spoken for three ways
// (dictionary values, RLE run-ends, VarBinary byte offsets). Adding a second
// pointer field would grow BoltColumn, and BoltBatch embeds
// columns[2][kMaxBatchColumns], so a pointer costs kilobytes per batch.
// Pointing `data` at a child ARRAY costs nothing: no existing format uses
// `data` as anything but a value buffer, and `type_size_bytes == 0` already
// marks a column as having no value buffer.
//
// This is purely ADDITIVE. Nothing produced a Nested column before, so no
// existing consumer can encounter one; code that switches on format falls
// through to whatever it already did for an unrecognised format.

// ============================================================================
// BoltColumn
// ============================================================================

/// Forward declaration for dictionary child
struct BoltColumn;

/// Forward declaration — full definition further below (post-BoltColumn).
struct BitmapIndex;

/// A single column. Fixed-size struct. Arena-owned data. No heap.
///
/// Supports zero-copy export to Arrow via fill_arrow_array().
struct BoltColumn {
    // --- Data pointers (interpretation depends on format) ---
    void*     data;             // Raw typed data / keys / inline constant
    uint8_t*  validity;         // Null bitmap (Arrow format: 1=valid)
    int64_t   validity_offset;  // Bit offset for views
    int64_t   length;           // Row count

    // --- Format + type ---
    ColumnFormat format;
    BoltType     type;
    uint16_t     type_size_bytes; // Cached from kTypeSize or user-specified
    uint8_t      decimal_scale;   // Decimal128/256: digits after the point
                                  // (Arrow stores scale in the type; we carry
                                  // it on the column so it travels with the
                                  // value through operators). 0 for non-decimal.
    uint8_t      _scale_pad[3];   // keep stats alignment explicit

    // --- Inline stats (always present) ---
    ColumnStats stats;

    // --- Sidecar indexes (optional, arena-allocated) ---
    SidecarSlots sidecars;

    // --- Format-specific ---
    union {
        // Constant: inline value (up to 16 bytes)
        alignas(16) uint8_t inline_value[16];
        // Sequence: offset + step
        struct { int64_t seq_offset; int64_t seq_step; };
    };

    // Dictionary: pointer to value column (arena-allocated)
    BoltColumn* dict_child;

    // Utf8 (StringView) spilled-bytes base. For a Utf8 column whose >12-char
    // values spill into a shared overflow buffer (e.g. CSV ingest), this is the
    // base pointer that resolves StringView::ref.offset. nullptr ⇒ all views
    // are inline (no spill) or the producer didn't expose a base. Consumers
    // pass it as the `spilled_base` arg to utf8::sv_bytes / sv_compare / sv_like.
    void* str_overflow_base;

    // Owning arena
    Arena* arena;

    // =====================================================================
    // Construction (all noexcept, all return by value)
    // =====================================================================

    /// Zero-initialized column
    static BoltColumn make_empty() noexcept {
        BoltColumn c;
        memset(&c, 0, sizeof(c));
        c.stats = ColumnStats::make_default();
        return c;
    }

    /// Flat column wrapping existing data (borrows pointer)
    static BoltColumn make_flat(void* data, uint8_t* validity, int64_t length,
                                BoltType type) noexcept {
        BoltColumn c = make_empty();
        c.data = data;
        c.validity = validity;
        c.length = length;
        c.format = ColumnFormat::Flat;
        c.type = type;
        c.type_size_bytes = static_cast<uint16_t>(bolt::type_size(type));
        c.stats.all_valid = (validity == nullptr);
        return c;
    }

    /// LIST column over `element`, with `length + 1` element offsets.
    /// `offsets` is borrowed (caller/arena owned) and must be non-decreasing
    /// with offsets[0] == 0 and offsets[length] == element->length.
    /// `validity` may be null (every list present). An EMPTY list is a VALID
    /// row whose offsets are equal -- that is the distinction a null bitmap
    /// alone cannot express, which is why both exist.
    static BoltColumn make_list(const BoltColumn* element,
                                const int32_t* offsets, int64_t length,
                                uint8_t* validity, Arena* arena) noexcept {
        BoltColumn c = make_empty();
        if (element == nullptr || offsets == nullptr || arena == nullptr) {
            return c;
        }
        auto* kids = static_cast<BoltColumn*>(
            arena->allocate(sizeof(BoltColumn), alignof(BoltColumn)));
        if (kids == nullptr) return c;
        kids[0] = *element;
        auto* off = static_cast<BoltColumn*>(
            arena->allocate(sizeof(BoltColumn), alignof(BoltColumn)));
        if (off == nullptr) return c;
        *off = make_flat(const_cast<int32_t*>(offsets), nullptr, length + 1,
                         BoltType::Int32);
        c.format = ColumnFormat::Nested;
        c.type = BoltType::List;
        c.type_size_bytes = 0;
        c.length = length;
        c.data = kids;
        c.dict_child = off;
        c.seq_offset = 1;                 // n_children
        c.validity = validity;
        c.stats.all_valid = (validity == nullptr);
        c.arena = arena;
        return c;
    }

    /// STRUCT column over `n` field columns, copied into arena storage.
    static BoltColumn make_struct(const BoltColumn* fields, int64_t n,
                                  int64_t length, uint8_t* validity,
                                  Arena* arena) noexcept {
        BoltColumn c = make_empty();
        if (fields == nullptr || n <= 0 || arena == nullptr) return c;
        auto* kids = static_cast<BoltColumn*>(arena->allocate(
            static_cast<size_t>(n) * sizeof(BoltColumn), alignof(BoltColumn)));
        if (kids == nullptr) return c;
        for (int64_t i = 0; i < n; ++i) kids[i] = fields[i];
        c.format = ColumnFormat::Nested;
        c.type = BoltType::Struct;
        c.type_size_bytes = 0;
        c.length = length;
        c.data = kids;
        c.dict_child = nullptr;
        c.seq_offset = n;
        c.validity = validity;
        c.stats.all_valid = (validity == nullptr);
        c.arena = arena;
        return c;
    }

    /// Nested accessors. All return nullptr / 0 on a non-nested column, so a
    /// consumer can probe without first switching on the format.
    bool is_nested() const noexcept {
        return format == ColumnFormat::Nested;
    }
    int64_t child_count() const noexcept {
        return is_nested() ? seq_offset : 0;
    }
    const BoltColumn* child_at(int64_t i) const noexcept {
        if (!is_nested() || data == nullptr) return nullptr;
        if (i < 0 || i >= seq_offset) return nullptr;
        return static_cast<const BoltColumn*>(data) + i;
    }
    /// LIST/MAP only: the `length + 1` element offsets, or nullptr.
    const int32_t* list_offsets() const noexcept {
        if (!is_nested() || dict_child == nullptr) return nullptr;
        if (type != BoltType::List && type != BoltType::Map) return nullptr;
        return static_cast<const int32_t*>(dict_child->data);
    }
    /// LIST/MAP only: the element column, or nullptr.
    const BoltColumn* list_element() const noexcept {
        if (type != BoltType::List && type != BoltType::Map) return nullptr;
        return child_at(0);
    }

    /// Flat column allocated from arena
    static BoltColumn make_flat_alloc(int64_t length, BoltType type,
                                      Arena* arena) noexcept {
        BoltColumn c = make_empty();
        size_t tsz = bolt::type_size(type);
        if (tsz == 0) return c;  // Variable-width needs different path
        c.data = arena->allocate(length * tsz);
        if (!c.data) return c;
        c.length = length;
        c.format = ColumnFormat::Flat;
        c.type = type;
        c.type_size_bytes = static_cast<uint16_t>(tsz);
        c.arena = arena;
        c.stats.all_valid = true;
        return c;
    }

    /// Flat column carrying fixed-stride f32 vectors (Embedding columns).
    /// `data` layout = N row-major float32 vectors, each of `dim` lanes
    /// (per-row stride = `dim * 4` bytes). The dimension is encoded
    /// into `type_size_bytes` so existing flat-array kernels see the
    /// correct per-row stride without a special case; the dimension
    /// can be recovered via `vector_dim()`.
    ///
    /// Returns `make_empty()` if `dim` is zero or the per-row stride
    /// would overflow `type_size_bytes` (uint16; cap ≈ 16 384 dims).
    /// Borrows `data` (does not allocate).
    static BoltColumn make_flat_vector(void* data, uint8_t* validity,
                                        int64_t length, uint32_t dim) noexcept {
        if (dim == 0u) return make_empty();
        const size_t stride = bolt::embedding_stride(dim);
        if (stride > UINT16_MAX) return make_empty();
        BoltColumn c = make_empty();
        c.data            = data;
        c.validity        = validity;
        c.length          = length;
        c.format          = ColumnFormat::Flat;
        c.type            = BoltType::Embedding;
        c.type_size_bytes = static_cast<uint16_t>(stride);
        c.stats.all_valid = (validity == nullptr);
        return c;
    }

    /// Flat vector column allocated from arena.
    static BoltColumn make_flat_vector_alloc(int64_t length, uint32_t dim,
                                              Arena* arena) noexcept {
        if (dim == 0u) return make_empty();
        const size_t stride = bolt::embedding_stride(dim);
        if (stride > UINT16_MAX) return make_empty();
        BoltColumn c = make_empty();
        c.data = arena->allocate(static_cast<size_t>(length) * stride);
        if (!c.data) return c;
        c.length          = length;
        c.format          = ColumnFormat::Flat;
        c.type            = BoltType::Embedding;
        c.type_size_bytes = static_cast<uint16_t>(stride);
        c.arena           = arena;
        c.stats.all_valid = true;
        return c;
    }

    // ------------------------------------------------------------------
    // Wave 9.4 z.5 — multi-precision vector factories.
    //
    // Same wire shape as `make_flat_vector`: borrows the caller's `data`
    // buffer (no allocation), encodes the per-row stride into
    // `type_size_bytes`, and tags the column with the precision-specific
    // BoltType. Distance kernels in `bolt::vec::` consume the matching
    // ElementTraits<T> specialisation.
    //
    // Stride caps: f16 ≤ 32 768 dims; u8/i8 ≤ 65 535 dims (limited by
    // uint16 type_size_bytes — well above the 4 096 / 1 024 head-room
    // any realistic embedding workload needs).
    // ------------------------------------------------------------------

    static BoltColumn make_flat_vector_f16(void* data, uint8_t* validity,
                                            int64_t length,
                                            uint32_t dim) noexcept {
        if (dim == 0u) return make_empty();
        const size_t stride =
            bolt::embedding_stride_for_type(BoltType::EmbeddingF16, dim);
        if (stride > UINT16_MAX) return make_empty();
        BoltColumn c = make_empty();
        c.data            = data;
        c.validity        = validity;
        c.length          = length;
        c.format          = ColumnFormat::Flat;
        c.type            = BoltType::EmbeddingF16;
        c.type_size_bytes = static_cast<uint16_t>(stride);
        c.stats.all_valid = (validity == nullptr);
        return c;
    }

    static BoltColumn make_flat_vector_u8(void* data, uint8_t* validity,
                                           int64_t length,
                                           uint32_t dim) noexcept {
        if (dim == 0u) return make_empty();
        const size_t stride =
            bolt::embedding_stride_for_type(BoltType::EmbeddingU8, dim);
        if (stride > UINT16_MAX) return make_empty();
        BoltColumn c = make_empty();
        c.data            = data;
        c.validity        = validity;
        c.length          = length;
        c.format          = ColumnFormat::Flat;
        c.type            = BoltType::EmbeddingU8;
        c.type_size_bytes = static_cast<uint16_t>(stride);
        c.stats.all_valid = (validity == nullptr);
        return c;
    }

    static BoltColumn make_flat_vector_i8(void* data, uint8_t* validity,
                                           int64_t length,
                                           uint32_t dim) noexcept {
        if (dim == 0u) return make_empty();
        const size_t stride =
            bolt::embedding_stride_for_type(BoltType::EmbeddingI8, dim);
        if (stride > UINT16_MAX) return make_empty();
        BoltColumn c = make_empty();
        c.data            = data;
        c.validity        = validity;
        c.length          = length;
        c.format          = ColumnFormat::Flat;
        c.type            = BoltType::EmbeddingI8;
        c.type_size_bytes = static_cast<uint16_t>(stride);
        c.stats.all_valid = (validity == nullptr);
        return c;
    }

    // Arena-allocating variants — mirror `make_flat_vector_alloc`.

    static BoltColumn make_flat_vector_f16_alloc(int64_t length,
                                                  uint32_t dim,
                                                  Arena* arena) noexcept {
        assert(arena != nullptr);
        if (dim == 0u) return make_empty();
        const size_t stride =
            bolt::embedding_stride_for_type(BoltType::EmbeddingF16, dim);
        if (stride > UINT16_MAX) return make_empty();
        BoltColumn c = make_empty();
        c.data = arena->allocate(static_cast<size_t>(length) * stride);
        if (!c.data) return c;
        c.length          = length;
        c.format          = ColumnFormat::Flat;
        c.type            = BoltType::EmbeddingF16;
        c.type_size_bytes = static_cast<uint16_t>(stride);
        c.arena           = arena;
        c.stats.all_valid = true;
        return c;
    }

    static BoltColumn make_flat_vector_u8_alloc(int64_t length,
                                                 uint32_t dim,
                                                 Arena* arena) noexcept {
        assert(arena != nullptr);
        if (dim == 0u) return make_empty();
        const size_t stride =
            bolt::embedding_stride_for_type(BoltType::EmbeddingU8, dim);
        if (stride > UINT16_MAX) return make_empty();
        BoltColumn c = make_empty();
        c.data = arena->allocate(static_cast<size_t>(length) * stride);
        if (!c.data) return c;
        c.length          = length;
        c.format          = ColumnFormat::Flat;
        c.type            = BoltType::EmbeddingU8;
        c.type_size_bytes = static_cast<uint16_t>(stride);
        c.arena           = arena;
        c.stats.all_valid = true;
        return c;
    }

    static BoltColumn make_flat_vector_i8_alloc(int64_t length,
                                                 uint32_t dim,
                                                 Arena* arena) noexcept {
        assert(arena != nullptr);
        if (dim == 0u) return make_empty();
        const size_t stride =
            bolt::embedding_stride_for_type(BoltType::EmbeddingI8, dim);
        if (stride > UINT16_MAX) return make_empty();
        BoltColumn c = make_empty();
        c.data = arena->allocate(static_cast<size_t>(length) * stride);
        if (!c.data) return c;
        c.length          = length;
        c.format          = ColumnFormat::Flat;
        c.type            = BoltType::EmbeddingI8;
        c.type_size_bytes = static_cast<uint16_t>(stride);
        c.arena           = arena;
        c.stats.all_valid = true;
        return c;
    }

    /// Recover the per-row dim of an Embedding column. Returns 0 for
    /// non-vector columns or for columns that haven't had their stride
    /// initialised (e.g. zero-init or wire-recovered without dim).
    /// Multi-precision aware: divides `type_size_bytes` by the element
    /// width of the column's `type` (4 for f32 / 2 for f16 / 1 for u8/i8).
    BOLT_FORCE_INLINE uint32_t vector_dim() const noexcept {
        const uint32_t bpe = bolt::embedding_element_bytes(type);
        if (bpe == 0u) return 0u;
        return static_cast<uint32_t>(type_size_bytes) / bpe;
    }

    /// Row pointer for an Embedding column (typed). No bounds check —
    /// callers assert `row < length` and `type == Embedding`.
    BOLT_FORCE_INLINE const float* vector_row(int64_t row) const noexcept {
        return reinterpret_cast<const float*>(
            static_cast<const uint8_t*>(data) +
            static_cast<size_t>(row) * type_size_bytes);
    }
    BOLT_FORCE_INLINE float* vector_row_mut(int64_t row) noexcept {
        return reinterpret_cast<float*>(
            static_cast<uint8_t*>(data) +
            static_cast<size_t>(row) * type_size_bytes);
    }

    /// Constant column
    template <typename T>
    static BoltColumn make_constant(T value, int64_t length, BoltType type) noexcept {
        static_assert(sizeof(T) <= 16, "Constant value too large");
        BoltColumn c = make_empty();
        c.length = length;
        c.format = ColumnFormat::Constant;
        c.type = type;
        c.type_size_bytes = static_cast<uint16_t>(sizeof(T));
        memcpy(c.inline_value, &value, sizeof(T));
        c.data = c.inline_value;

        memcpy(&c.stats.min_value, &value,
               sizeof(T) < sizeof(int64_t) ? sizeof(T) : sizeof(int64_t));
        c.stats.max_value = c.stats.min_value;
        c.stats.null_count = 0;
        c.stats.distinct_count = 1;
        c.stats.cardinality = CardinalityClass::Constant;
        c.stats.sort_order = SortOrder::Ascending;
        c.stats.all_valid = true;
        c.stats.is_monotonic = false;
        return c;
    }

    /// Sequence column (value[i] = offset + i * step)
    static BoltColumn make_sequence(int64_t offset, int64_t step, int64_t length,
                                    BoltType type) noexcept {
        BoltColumn c = make_empty();
        c.length = length;
        c.format = ColumnFormat::Sequence;
        c.type = type;
        c.type_size_bytes = static_cast<uint16_t>(sizeof(int64_t));
        c.seq_offset = offset;
        c.seq_step = step;

        c.stats.min_value = (step >= 0) ? offset : offset + (length - 1) * step;
        c.stats.max_value = (step >= 0) ? offset + (length - 1) * step : offset;
        c.stats.null_count = 0;
        c.stats.distinct_count = (step == 0) ? 1 : static_cast<uint32_t>(length);
        c.stats.cardinality = (step == 0) ? CardinalityClass::Constant : CardinalityClass::High;
        c.stats.sort_order = (step > 0) ? SortOrder::Ascending
                           : (step < 0) ? SortOrder::Descending
                           : SortOrder::Ascending;
        c.stats.all_valid = true;
        c.stats.is_monotonic = (step != 0);
        return c;
    }

    /// View (zero-copy slice)
    /// Run-length encoded column (B2). `values[]` and `run_ends[]` must each
    /// have `num_runs` elements and live in or outlive `arena`. Run i covers
    /// rows `[run_ends[i-1], run_ends[i])` (run_ends[-1] is treated as 0);
    /// `run_ends[num_runs-1]` must equal `total_rows`. Returns make_empty()
    /// on shape violation or OOM.
    ///
    /// Storage: `data = values`, `dict_child = &flat int32 column over run_ends`,
    /// `length = total_rows`. The `dict_child` is an int32 Flat column (created
    /// from the caller-supplied pointer; zero-copy).
    static BoltColumn make_rle(const void* values, int64_t num_runs,
                               const int32_t* run_ends, int64_t total_rows,
                               BoltType type, Arena* arena) noexcept {
        assert(arena != nullptr);
        assert(num_runs >= 0);
        assert(total_rows >= 0);
        if (num_runs == 0 && total_rows == 0) {
            BoltColumn c = make_empty();
            c.type = type;
            c.type_size_bytes = static_cast<uint16_t>(bolt::type_size(type));
            c.length = 0;
            c.format = ColumnFormat::RLE;
            c.arena = arena;
            return c;
        }
        if (values == nullptr || run_ends == nullptr) return make_empty();
        if (num_runs <= 0) return make_empty();
        // Last run end must match total_rows — trust but assert in debug.
        assert(run_ends[num_runs - 1] == static_cast<int32_t>(total_rows));

        BoltColumn c = make_empty();
        c.type = type;
        c.type_size_bytes = static_cast<uint16_t>(bolt::type_size(type));
        c.length = total_rows;
        c.format = ColumnFormat::RLE;
        c.arena = arena;
        // Borrow the values pointer; caller owns lifetime.
        c.data = const_cast<void*>(values);

        // Wrap run_ends in a child column.  Arena-allocate a BoltColumn
        // descriptor so the RLE column is self-contained.
        BoltColumn* rc = arena->allocate_array<BoltColumn>(1);
        if (!rc) return make_empty();
        *rc = BoltColumn::make_flat(
            const_cast<int32_t*>(run_ends), nullptr, num_runs,
            BoltType::Int32);
        c.dict_child = rc;

        c.stats.all_valid = true;
        return c;
    }

    /// Variable-width payload column (Layer 1.1).
    ///
    /// Borrows caller-owned buffers — does NOT copy or take ownership.
    /// `offsets` must be of length `total_rows + 1` and monotonically
    /// non-decreasing; `data` must hold at least `offsets[total_rows]` bytes.
    /// Row i spans bytes `[offsets[i], offsets[i+1])`. Empty rows have
    /// `offsets[i] == offsets[i+1]`. Null rows are indicated via the optional
    /// `validity` bitmap (Arrow convention: bit set = valid).
    ///
    /// Used by Bolt's parse/JSONB consumers and by ClickHouse-style
    /// per-path JSON subcolumns. Tiger Style: noexcept, ≥2 assertions,
    /// caller-supplied buffers, no allocation here beyond the descriptor for
    /// the offsets sub-column.
    static BoltColumn make_var_binary(void* data, uint8_t* validity,
                                       const int32_t* offsets,
                                       int64_t total_rows, BoltType type,
                                       Arena* arena) noexcept {
        assert(arena != nullptr);
        assert(total_rows >= 0);
        if (total_rows < 0) return make_empty();
        if (total_rows > 0) {
            // Inputs must be present for non-empty columns.
            if (offsets == nullptr) return make_empty();
            if (offsets[total_rows] > 0 && data == nullptr) return make_empty();
        }
        BoltColumn c = make_empty();
        c.data            = data;
        c.validity        = validity;
        c.length          = total_rows;
        c.format          = ColumnFormat::VarBinary;
        c.type            = type;
        c.type_size_bytes = 0;          // variable-width
        c.arena           = arena;

        // Wrap offsets in a child Flat Int32 column so the descriptor is
        // self-contained. Offsets array has length+1 entries by Arrow convention.
        BoltColumn* oc = arena->allocate_array<BoltColumn>(1);
        if (oc == nullptr) return make_empty();
        *oc = BoltColumn::make_flat(
            const_cast<int32_t*>(offsets), nullptr, total_rows + 1,
            BoltType::Int32);
        c.dict_child = oc;

        c.stats.all_valid = (validity == nullptr);
        return c;
    }

    /// Utf8 column in the **Flat/StringView** layout, built from an already
    /// packed `(bytes, offsets)` pair — i.e. the same physical shape the
    /// parquet reader and the hash-agg output produce, which is what every
    /// operator in the engine actually consumes.
    ///
    /// WHY THIS EXISTS (G2FEAT-77). Two incompatible physical layouts share
    /// `BoltType::Utf8`:
    ///   * `Format::Flat`/`View` — one `StringView` per row (inline ≤12 bytes,
    ///     else spilled via `str_overflow_base`);
    ///   * `Format::VarBinary` — `offsets[] + packed bytes` (`make_var_binary`).
    /// Compute, filter, sort, join and hash-agg all read Utf8 *exclusively* as
    /// `StringView` + `str_overflow_base`; none of them dispatch on `format`.
    /// A VarBinary column reaching them is therefore reinterpreted as a
    /// StringView array — a silent garbage read, not a clean failure. Producers
    /// that naturally hold packed bytes (every connector: Kafka key/value,
    /// Redis, MQTT, REST, …) should build through THIS factory so their strings
    /// are readable by the whole engine. `make_var_binary` remains valid for
    /// consumers that explicitly dispatch on `VarBinary` (e.g. collect_sink).
    ///
    /// ZERO-COPY: the caller's packed buffer becomes `str_overflow_base`
    /// verbatim, and a spilled row's `ref.offset` is simply its `offsets[i]`.
    /// Only the `StringView` array itself is allocated. `data` must therefore
    /// outlive the column — the same borrowing contract `make_var_binary` has.
    ///
    /// `offsets` has `total_rows + 1` entries, monotonically non-decreasing.
    /// Returns an empty column if the offsets exceed the `uint32` range a
    /// `StringView::ref.offset` can address.
    static BoltColumn make_utf8_from_packed(const void* data,
                                            uint8_t* validity,
                                            const int32_t* offsets,
                                            int64_t total_rows,
                                            Arena* arena) noexcept {
        assert(arena != nullptr);
        assert(total_rows >= 0);
        if (total_rows < 0) return make_empty();
        if (total_rows > 0) {
            if (offsets == nullptr) return make_empty();
            if (offsets[total_rows] > 0 && data == nullptr) return make_empty();
        }
        BoltColumn c = make_flat_alloc(total_rows, BoltType::Utf8, arena);
        if (total_rows > 0 && c.data == nullptr) return make_empty();
        const char* base = static_cast<const char*>(data);
        auto* sv = static_cast<StringView*>(c.data);
        for (int64_t i = 0; i < total_rows; ++i) {
            const int32_t start = offsets[i];
            const int32_t end   = offsets[i + 1];
            assert(end >= start && "offsets must be non-decreasing");
            const uint32_t len = static_cast<uint32_t>(end - start);
            StringView r;
            std::memset(&r, 0, sizeof(r));
            r.length = len;
            if (len <= 12u) {
                // Inline: prefix[4] then inline_data[8]; nothing to resolve.
                const uint32_t p = (len < 4u) ? len : 4u;
                if (p > 0) std::memcpy(r.prefix, base + start, p);
                if (len > 4u) std::memcpy(r.inline_data, base + start + 4, len - 4u);
            } else {
                // Spilled: point INTO the caller's packed buffer. No copy.
                if (static_cast<uint64_t>(start) > 0xFFFFFFFFull) return make_empty();
                std::memcpy(r.prefix, base + start, 4);
                r.ref.buf_idx = 0;
                r.ref.offset  = static_cast<uint32_t>(start);
            }
            sv[i] = r;
        }
        c.validity          = validity;
        c.str_overflow_base = const_cast<void*>(data);
        c.stats.all_valid   = (validity == nullptr);
        return c;
    }

    /// Variable-width payload accessor. Returns `(ptr, len)` for row `i`.
    /// Caller must verify the column is `Format::VarBinary`. Branchless: a
    /// pair of int32 loads + a pointer add. Validity is NOT checked here —
    /// caller probes the validity bitmap separately if needed.
    BOLT_FORCE_INLINE void var_binary_at(int64_t row,
                                          const uint8_t** out_data,
                                          int32_t* out_len) const noexcept {
        assert(format == ColumnFormat::VarBinary);
        assert(out_data != nullptr && out_len != nullptr);
        const int32_t* offs = static_cast<const int32_t*>(dict_child->data);
        const int32_t  start = offs[row];
        const int32_t  end   = offs[row + 1];
        *out_data = static_cast<const uint8_t*>(data) + start;
        *out_len  = end - start;
    }

    /// Format-agnostic variable-width accessor for a `Utf8`/`Binary` column.
    /// Returns `(ptr, len)` for row `i` whether the column is laid out as
    /// `VarBinary` (offsets + packed bytes) or `Flat`/`View` (StringView
    /// array, inline ≤12 bytes else spilled into `str_overflow_base`).
    ///
    /// Both layouts legitimately occur for `BoltType::Utf8` — the parquet
    /// reader and every operator produce `Flat`, connectors historically
    /// produced `VarBinary` — and reading one as the other is a silent
    /// garbage read in Release, not a crash (G2FEAT-77). Any consumer that
    /// takes bytes out of a string column it did not build itself must use
    /// this accessor rather than `var_binary_at`. Validity is NOT checked;
    /// probe the validity bitmap separately if needed.
    BOLT_FORCE_INLINE void utf8_at(int64_t row,
                                    const uint8_t** out_data,
                                    int32_t* out_len) const noexcept {
        assert(out_data != nullptr && out_len != nullptr);
        assert(row >= 0 && row < length);
        if (format == ColumnFormat::VarBinary) {
            var_binary_at(row, out_data, out_len);
            return;
        }
        assert(format == ColumnFormat::Flat || format == ColumnFormat::View);
        const auto& sv = static_cast<const StringView*>(data)[row];
        *out_len = static_cast<int32_t>(sv.length);
        if (sv.length <= 12u) {
            // Inline: first 4 bytes in `prefix`, remainder in `inline_data`,
            // and they are adjacent in the struct, so `prefix` is the base.
            *out_data = reinterpret_cast<const uint8_t*>(sv.prefix);
        } else {
            *out_data = static_cast<const uint8_t*>(str_overflow_base)
                      + sv.ref.offset;
        }
    }

    /// Bit-packed column (B3). `packed_words` holds `total_rows * bit_width`
    /// bits LSB-first, each value using `bit_width` ∈ [1,32] bits. Caller
    /// owns the buffer lifetime. `type` determines the output integer type
    /// at materialise time (must be a ≤32-bit signed or unsigned int).
    static BoltColumn make_bitpacked(const uint64_t* packed_words,
                                      uint8_t bit_width, int64_t total_rows,
                                      BoltType type, Arena* arena) noexcept {
        assert(arena != nullptr);
        assert(bit_width >= 1 && bit_width <= 32);
        assert(total_rows >= 0);
        if (total_rows > 0 && packed_words == nullptr) return make_empty();

        BoltColumn c = make_empty();
        c.type = type;
        c.type_size_bytes = static_cast<uint16_t>(bolt::type_size(type));
        c.length = total_rows;
        c.format = ColumnFormat::BitPacked;
        c.arena = arena;
        c.data = const_cast<uint64_t*>(packed_words);
        c.seq_offset = 0;  // unused for BitPacked
        c.seq_step   = static_cast<int64_t>(bit_width);
        c.stats.all_valid = true;
        return c;
    }

    /// Frame-of-reference column (B4). Logical value[i] = base + delta[i],
    /// where delta[] is bit-packed at `bit_width` bits per delta.
    static BoltColumn make_frame_of_ref(const uint64_t* packed_deltas,
                                         uint8_t bit_width, int64_t base,
                                         int64_t total_rows,
                                         BoltType type, Arena* arena) noexcept {
        assert(arena != nullptr);
        assert(bit_width >= 1 && bit_width <= 32);
        assert(total_rows >= 0);
        if (total_rows > 0 && packed_deltas == nullptr) return make_empty();

        BoltColumn c = make_empty();
        c.type = type;
        c.type_size_bytes = static_cast<uint16_t>(bolt::type_size(type));
        c.length = total_rows;
        c.format = ColumnFormat::FrameOfRef;
        c.arena = arena;
        c.data = const_cast<uint64_t*>(packed_deltas);
        c.seq_offset = base;
        c.seq_step   = static_cast<int64_t>(bit_width);
        c.stats.all_valid = true;
        return c;
    }

    static BoltColumn make_view(const BoltColumn& parent, int64_t offset,
                                int64_t length) noexcept {
        assert(offset >= 0 && offset + length <= parent.length);
        BoltColumn c = make_empty();
        c.type = parent.type;
        c.type_size_bytes = parent.type_size_bytes;
        c.length = length;

        if (parent.format == ColumnFormat::Flat || parent.format == ColumnFormat::View) {
            c.format = ColumnFormat::View;
            c.data = static_cast<uint8_t*>(parent.data) + offset * parent.type_size_bytes;
            c.validity = parent.validity;
            c.validity_offset = parent.validity_offset + offset;
        } else if (parent.format == ColumnFormat::Constant) {
            c = parent;
            c.length = length;
        } else if (parent.format == ColumnFormat::Sequence) {
            c.format = ColumnFormat::Sequence;
            c.seq_offset = parent.seq_offset + offset * parent.seq_step;
            c.seq_step = parent.seq_step;
        }

        // Inherit stats (conservative — min/max still valid for subsets)
        c.stats = parent.stats;
        return c;
    }

    // =====================================================================
    // Data access (typed, no virtual dispatch)
    // =====================================================================

    template <typename T> const T* typed_data() const noexcept {
        assert(format == ColumnFormat::Flat || format == ColumnFormat::View);
        return static_cast<const T*>(data);
    }

    template <typename T> T* typed_mutable() noexcept {
        assert(format == ColumnFormat::Flat);
        return static_cast<T*>(data);
    }

    bool is_null(int64_t i) const noexcept {
        if (!validity) return false;
        int64_t bit = validity_offset + i;
        return !(validity[bit >> 3] & (1 << (bit & 7)));
    }

    template <typename T> T get_constant() const noexcept {
        assert(format == ColumnFormat::Constant);
        T v; memcpy(&v, inline_value, sizeof(T)); return v;
    }

    int64_t get_sequence(int64_t i) const noexcept {
        assert(format == ColumnFormat::Sequence);
        return seq_offset + i * seq_step;
    }

    bool is_constant()   const noexcept { return format == ColumnFormat::Constant; }
    bool is_dictionary() const noexcept { return format == ColumnFormat::Dictionary; }
    bool has_nulls()     const noexcept { return validity && !stats.all_valid; }

    /// Byte size of this column's data (excluding sidecars)
    size_t byte_size() const noexcept {
        switch (format) {
            case ColumnFormat::Flat:
            case ColumnFormat::View:
                return static_cast<size_t>(length) * type_size_bytes
                     + (validity ? (static_cast<size_t>(length) + 7) / 8 : 0);
            case ColumnFormat::Constant:   return type_size_bytes;
            case ColumnFormat::Dictionary: return static_cast<size_t>(length) * type_size_bytes;
            case ColumnFormat::Sequence:   return 0;
            case ColumnFormat::VarBinary: {
                if (length <= 0 || dict_child == nullptr) return 0;
                const int32_t* offs = static_cast<const int32_t*>(dict_child->data);
                const size_t payload = static_cast<size_t>(offs[length]);
                const size_t offsets_bytes = static_cast<size_t>(length + 1) * sizeof(int32_t);
                const size_t valid_bytes = validity ? (static_cast<size_t>(length) + 7) / 8 : 0;
                return payload + offsets_bytes + valid_bytes;
            }
            case ColumnFormat::RLE:
            case ColumnFormat::BitPacked:
            case ColumnFormat::FrameOfRef:
                return 0;  // existing TODO; sized by their own buffers
        }
        return 0;
    }

    // =====================================================================
    // Arrow C Data Interface export (zero-copy, no libarrow link)
    // =====================================================================

    /// DEPRECATED — DO NOT USE. Superseded by bolt/bolt_arrow.h
    /// (`bolt::arrow::export_column`).
    ///
    /// Audited against pyarrow 21 on 2026-08-16: this export cannot be
    /// consumed by any real Arrow implementation. The release callback below
    /// is an empty body, and the spec REQUIRES it to set `release = NULL` to
    /// mark the struct released — pyarrow aborts the process on every import
    /// without it. Beyond that: `fill_arrow_array` stores its buffer pointers
    /// in a `static thread_local` array, so every exported column aliases the
    /// last one; Utf8 declares "vu" (StringView) while exporting two buffers
    /// and never exporting `str_overflow_base`; Bool hands byte-packed data to
    /// a bit-packed format; Decimal128 hardcodes "d:38,10" regardless of the
    /// column's real scale; and `null_count` reads a stat that is zero unless
    /// someone called compute_stats_numeric().
    ///
    /// Kept only so the existing tests that pin this behaviour still compile.
    /// New code must use bolt::arrow::export_column, which owns its buffers
    /// and is validated against real pyarrow (tests/test_arrow_pyarrow.py).
    ///
    /// Fill an ArrowSchema struct for this column's type.
    void fill_arrow_schema(ArrowSchema* out, const char* name) const noexcept {
        memset(out, 0, sizeof(ArrowSchema));
        out->format = arrow_format_string(type);
        out->name = name;
        out->flags = (validity != nullptr) ? 2 : 0;  // ARROW_FLAG_NULLABLE = 2
        out->n_children = 0;
        out->children = nullptr;
        out->dictionary = nullptr;
        out->release = &noop_release_schema;
        out->private_data = nullptr;
    }

    /// Fill an ArrowArray struct pointing to this column's buffers.
    /// Zero-copy: the ArrowArray points directly into our arena memory.
    /// Caller must ensure the BoltColumn outlives the ArrowArray consumer.
    ///
    /// For Flat columns: buffers = [validity, data]
    /// For Constant: must materialize first (caller's responsibility)
    void fill_arrow_array(ArrowArray* out) const noexcept {
        assert(format == ColumnFormat::Flat || format == ColumnFormat::View);

        memset(out, 0, sizeof(ArrowArray));
        out->length = length;
        out->null_count = stats.null_count;
        out->offset = 0;
        out->n_buffers = 2;
        out->n_children = 0;

        // Static buffer array — we store validity + data pointers
        // These must remain valid as long as the ArrowArray is in use
        static thread_local const void* tl_buffers[2];
        tl_buffers[0] = validity;  // May be nullptr (no nulls)
        tl_buffers[1] = data;
        out->buffers = tl_buffers;

        out->children = nullptr;
        out->dictionary = nullptr;
        out->release = &noop_release_array;
        out->private_data = nullptr;
    }

    // =====================================================================
    // Compute stats from current data (one-pass)
    // =====================================================================

    /// Compute min/max/null_count/cardinality for numeric Flat columns.
    /// Template dispatch via BOLT_NUMERIC_TYPES X-macro.
    void compute_stats_numeric() noexcept;

    /// Compute per-dim min/max/mean/variance for an Embedding column.
    /// Allocates a `VectorStats` POD from `arena` and stores it on
    /// `sidecars.vector_stats`. Returns `nullptr` if the column is not
    /// a Flat Embedding column, dim is zero, or the arena rejects
    /// allocation. Subsequent calls reuse the existing sidecar slot
    /// (the caller is responsible for resetting the slot when the
    /// underlying data is mutated). One-pass per-dim Welford —
    /// vertical traversal (rows outer, dim inner) so the inner loop
    /// auto-vectorises on AVX2 / NEON.
    VectorStats* compute_stats_vector(Arena* arena_in) noexcept;

    /// Try to promote format based on stats (e.g., Flat → Constant)
    bool try_promote(Arena* arena) noexcept;

    /// Deep-copy into arena
    BoltColumn clone_into(Arena* arena) const noexcept;

    /// Materialize non-Flat formats to Flat (arena-allocated)
    BoltColumn materialize(Arena* arena) const noexcept;

    /// Ensure a `BitmapIndex` is attached to this column, building it lazily
    /// on first call from `arena`. The index lives on the `bitmap_index`
    /// slot; subsequent calls return the cached pointer with no work.
    /// Returns nullptr if the column is not a Dictionary (only shape the
    /// current `BitmapIndex::build` supports) or on OOM. C5.
    BitmapIndex* ensure_bitmap_index(Arena* arena) noexcept;

    static void noop_release_schema(ArrowSchema*) noexcept {}
    static void noop_release_array(ArrowArray*) noexcept {}
};

// ============================================================================
// BoltBatch — double-buffered, COW, Venus tick-tock pattern
// ============================================================================

// Upper BOUND on a batch's column count — an assertion cap, NOT an allocation
// size. G2FEAT-47: columns are now dynamically right-sized to `num_cols` via
// `alloc_columns()`, so a wide bound costs nothing when unused. (Was 256 as a
// fixed inline `columns[2][256]` that over-allocated ~151 KB per batch.)
static constexpr uint32_t kMaxBatchColumns = 1024;

// Column cap for the FIXED-INLINE auxiliary framing structs that predate the
// dynamic-columns change and still embed `[N]` column arrays: the wire codec
// (WireStream / serialize scratch), RowView, and the Parquet write column
// table. Kept at the historical kMaxBatchColumns value (256) so those structs
// stay compact — decoupling them from the in-memory batch bound (raised to
// 1024) avoids ballooning them 4× and keeps WireStream inside its 16 KB budget.
// Consequence (non-regressing): a batch wider than kMaxFixedColumns can live in
// memory (right-sized) but cannot cross the wire / be Parquet-row-group-written
// — the exact prior behaviour, since batches were capped at 256 before.
static constexpr uint32_t kMaxFixedColumns = 256;

struct alignas(64) BoltBatch {
    // Double-buffered columns — two tick-tock epochs, each a pointer to an
    // arena-allocated array of exactly `num_cols` BoltColumn (nullptr until
    // `alloc_columns()` runs). Indexing `columns[epoch][i]` is unchanged from
    // the old fixed 2D array; only the backing storage moved to the arena.
    BoltColumn* columns[2];

    // Dirty tracking — DYNAMIC bitmask, one bit per column, sized to
    // `ceil(num_cols/64)` atomic words and arena-allocated by alloc_columns()
    // (nullptr until then). G2FEAT-47: replaces the old fixed 128-bit
    // dirty_mask_lo/hi so clone-on-write works for ANY column index up to the
    // kMaxBatchColumns bound (in-place ingestion / dataframe-style column
    // mutation on wide batches is a first-class path). The common num_cols<=64
    // case is a single word — the array owns its own cache line (64-B aligned),
    // preserving the original false-sharing isolation. num_cols in [65,128]
    // gives two words, byte-for-byte the old lo/hi behaviour.
    std::atomic<uint64_t>* dirty_mask;
    uint32_t               dirty_words;   // ceil(num_cols/64)

    // Epoch — atomic (W6.0): the tick-tock `swap()` flips these on the
    // owner/apply thread while pinned readers concurrently pick an epoch
    // to index `columns[epoch]`. Plain uint8_t raced on ARM (TSan). 1 byte,
    // align 1 — no layout/ABI change. Accesses default to seq_cst via the
    // implicit operator; hot sites may opt down to relaxed after profiling
    // (the swap uses release / reader epoch-pick uses acquire — the race pair).
    std::atomic<uint8_t> read_epoch;
    std::atomic<uint8_t> write_epoch;
    uint64_t generation[2];

    // Dimensions
    int64_t  num_rows;
    uint32_t num_cols;

    // Schema
    BoltSchema schema;

    // Arena for COW copies
    Arena* arena;

    // =====================================================================
    // Construction
    // =====================================================================

    // Initialize an already-allocated BoltBatch in place. Returning by value
    // is not possible: std::atomic members are non-copyable.
    static void init_empty(BoltBatch* b) noexcept {
        // zero all non-atomic fields from read_epoch onward (num_rows/num_cols/
        // schema/arena all live after it); then explicitly init atomics and the
        // two column pointers (which sit BEFORE read_epoch and so are untouched
        // by the memset — they MUST be nulled here so a caller that never calls
        // alloc_columns can't dereference a dangling pointer).
        memset(reinterpret_cast<char*>(b) + offsetof(BoltBatch, read_epoch),
               0, sizeof(BoltBatch) - offsetof(BoltBatch, read_epoch));
        b->columns[0]  = nullptr;
        b->columns[1]  = nullptr;
        b->dirty_mask  = nullptr;
        b->dirty_words = 0;
        b->read_epoch  = 0;
        b->write_epoch = 1;
    }

    // Arena-allocate the two per-epoch column arrays, each sized to exactly
    // `n_cols` BoltColumn, zero-initialised. Sets `num_cols` and `arena`. This
    // is the sanctioned "constructor" for a batch's storage — call it AFTER
    // init_empty and BEFORE writing any `columns[epoch][i]`.
    //
    // READ-ONLY by default (G2FEAT-47 "pay for what you use"): no dirty-mask is
    // allocated, so the batch is strictly cheaper than the old fixed layout.
    // This is correct for the vast majority of sites — scan outputs, wide
    // 105/1024-col batches, and freshly-produced operator batches that fill
    // every column once and never `mut_col` in place. Such a batch must never
    // call mut_col()/mark_dirty() (they assert dirty_mask != nullptr). Use
    // `alloc_columns_mutable` for the ingestion / dataframe-mutation / post-swap
    // clone-on-write producers instead.
    //
    // Tiger Style: this is init-time allocation (once per batch use), never a
    // hot-path grow. Pool-reused batches re-`init_empty` (nulling the pointers)
    // and re-`alloc_columns` at the width of the new use; the pool's arena is
    // reset between uses, so the previous arrays are already reclaimed — there
    // is no leak and no stale-width reuse.
    //
    // Returns false on arena OOM (batch left with num_cols==0, columns null).
    // `n_cols == 0` is legal (empty batch): leaves both pointers null.
    static bool alloc_columns(BoltBatch* b, Arena* arena, uint32_t n_cols) noexcept {
        return alloc_columns_impl(b, arena, n_cols, /*with_cow=*/false);
    }

    // Like alloc_columns, but ALSO allocates the dynamic clone-on-write dirty
    // mask (`ceil(n_cols/64)` atomic words, 64-B aligned so the common
    // single-word case owns its cache line), enabling mut_col()/swap()/COW at
    // ANY column index up to kMaxBatchColumns. For the ingestion path, in-place
    // dataframe-style column mutation, and any operator that COW-mutates an
    // existing batch after a tick-tock swap.
    static bool alloc_columns_mutable(BoltBatch* b, Arena* arena,
                                      uint32_t n_cols) noexcept {
        return alloc_columns_impl(b, arena, n_cols, /*with_cow=*/true);
    }

    static bool alloc_columns_impl(BoltBatch* b, Arena* arena, uint32_t n_cols,
                                   bool with_cow) noexcept {
        assert(b != nullptr);
        assert(n_cols <= kMaxBatchColumns);
        b->arena       = arena;
        b->num_cols    = n_cols;
        b->dirty_mask  = nullptr;
        b->dirty_words = 0;
        if (n_cols == 0) {
            b->columns[0] = nullptr;
            b->columns[1] = nullptr;
            b->schema.set_storage(nullptr, 0);   // G2FEAT-47: dynamic schema
            return true;
        }
        assert(arena != nullptr);
        BoltColumn* c0 = arena->allocate_array<BoltColumn>(n_cols);
        BoltColumn* c1 = arena->allocate_array<BoltColumn>(n_cols);
        // G2FEAT-47: right-size the embedded schema's field storage to n_cols too
        // (BoltSchema.fields is now an arena pointer, not a fixed [256] array —
        // this is what stops BoltBatch carrying ~20 KB of inline schema).
        BoltField* fbuf = arena->allocate_array<BoltField>(n_cols);
        if (c0 == nullptr || c1 == nullptr || fbuf == nullptr) {
            b->columns[0] = nullptr;
            b->columns[1] = nullptr;
            b->schema.set_storage(nullptr, 0);
            b->num_cols   = 0;
            return false;
        }
        memset(static_cast<void*>(c0), 0, sizeof(BoltColumn) * n_cols);
        memset(static_cast<void*>(c1), 0, sizeof(BoltColumn) * n_cols);
        memset(static_cast<void*>(fbuf), 0, sizeof(BoltField) * n_cols);
        b->columns[0] = c0;
        b->columns[1] = c1;
        b->schema.set_storage(fbuf, n_cols);
        if (with_cow) {
            const uint32_t dw = (n_cols + 63u) / 64u;
            void* raw = arena->allocate(sizeof(std::atomic<uint64_t>) * dw,
                                        /*align=*/64);
            if (raw == nullptr) {
                b->columns[0] = nullptr;
                b->columns[1] = nullptr;
                b->num_cols   = 0;
                return false;
            }
            auto* dm = static_cast<std::atomic<uint64_t>*>(raw);
            for (uint32_t w = 0; w < dw; ++w) {
                new (&dm[w]) std::atomic<uint64_t>(0);
            }
            b->dirty_mask  = dm;
            b->dirty_words = dw;
        }
        return true;
    }

    // =====================================================================
    // Read access (zero atomics)
    // =====================================================================

    const BoltColumn& col(uint32_t i) const noexcept {
        assert(i < num_cols);
        assert(columns[read_epoch] != nullptr &&
               "col() before alloc_columns() — missing alloc_columns at a fill site");
        return columns[read_epoch][i];
    }

    template <typename T>
    const T* col_data(uint32_t i) const noexcept {
        return col(i).typed_data<T>();
    }

    // =====================================================================
    // Write access (COW on first touch per epoch)
    // =====================================================================

    BoltColumn& mut_col(uint32_t i) noexcept {
        assert(i < num_cols);
        assert(columns[write_epoch] != nullptr &&
               "mut_col() before alloc_columns() — missing alloc_columns at a fill site");
        mark_dirty(i);
        return columns[write_epoch][i];
    }

    // =====================================================================
    // Epoch swap (~1.3ns measured)
    // =====================================================================

    void swap() noexcept {
        read_epoch ^= 1;
        write_epoch ^= 1;
        generation[write_epoch]++;
        // Clear every dirty word (bounded: dirty_words <= ceil(1024/64) = 16).
        // A read-only batch (no dirty mask) has nothing to clear — swapping one
        // is legal (epoch flip only), it simply carries no COW state.
        for (uint32_t w = 0; w < dirty_words; ++w) {
            dirty_mask[w].store(0, std::memory_order_release);
        }
    }

    // =====================================================================
    // Arrow C Data Interface export
    // =====================================================================

    /// Export entire batch as ArrowSchema + array of ArrowArrays.
    /// Zero-copy for Flat columns. Constant/Sequence must be materialized first.
    void fill_arrow_schema(ArrowSchema* out) const noexcept;

    // =====================================================================
    // IPC serialization (Bolt wire format, Arrow-layout-compatible data)
    // =====================================================================

    /// Serialize to contiguous buffer. Returns bytes written, 0 on error.
    /// Layout: header + column data (Arrow-compatible buffers).
    size_t serialize(void* out_buf, size_t buf_capacity) const noexcept;

    /// Deserialize from buffer into arena. Returns false on error.
    static bool deserialize(const void* buf, size_t buf_len,
                            BoltBatch* out, Arena* arena) noexcept;

private:
    void mark_dirty(uint32_t col_idx) noexcept {
        // The dynamic dirty mask makes clone-on-write correct for ANY column
        // index up to kMaxBatchColumns. Batch MUST have been constructed via
        // alloc_columns_mutable (dirty mask present) — a read-only batch never
        // mutates in place.
        assert(dirty_mask != nullptr &&
               "mut_col on a read-only batch — construct with alloc_columns_mutable");
        assert(col_idx < num_cols);
        const uint32_t word = col_idx >> 6;
        const uint64_t bit  = 1ULL << (col_idx & 63u);
        assert(word < dirty_words);
        const uint64_t mask = dirty_mask[word].load(std::memory_order_acquire);
        if (!(mask & bit)) {
            if (generation[read_epoch] != generation[write_epoch] && arena) {
                columns[write_epoch][col_idx] =
                    columns[read_epoch][col_idx].clone_into(arena);
            }
            dirty_mask[word].fetch_or(bit, std::memory_order_release);
        }
    }
};

// ============================================================================
// BitmapIndex (QuestDB-style, arena-allocated)
// ============================================================================

struct BitmapIndex {
    uint64_t** bitmaps;     // K bitmaps, each ceil(N/64) words
    uint32_t*  popcounts;   // K entries (H3 — miss-accelerator); 0 = key absent
    uint32_t   num_keys;
    uint32_t   num_rows;
    uint32_t   words_per_bitmap;

    /// Build from a Dictionary column's uint8/uint16/uint32 keys.
    /// Arena-allocated. Returns nullptr on failure.
    static BitmapIndex* build(const BoltColumn& col, Arena* arena) noexcept;

    /// Count rows with value == key. O(1) after build via the `popcounts`
    /// sidecar — no bitmap scan.
    uint32_t count(uint32_t key) const noexcept;

    /// H3 — O(1) "key was never seen" check. Returns true iff the key is
    /// out of range OR its popcount is 0. Lets filter dispatchers short-
    /// circuit the full bitmap walk for definitively-absent keys.
    /// QuestDB SYMBOL has no equivalent.
    BOLT_FORCE_INLINE bool probably_absent(uint32_t key) const noexcept {
        if (key >= num_keys) return true;
        return popcounts == nullptr ? false : (popcounts[key] == 0u);
    }

    /// Write matching row indices. Returns count written.
    int64_t filter(uint32_t key, int32_t* out) const noexcept;

    /// OR of multiple keys. Returns count written.
    int64_t filter_in(const uint32_t* keys, uint32_t nkeys,
                      int32_t* out) const noexcept;
};

// ============================================================================
// BoltColumn::compute_stats_numeric — one-pass min/max/null/distinct scan
// ============================================================================

namespace detail {

// Distinct sketch: up to 16 values tracked exactly; overflow → distinct_count=17.
template <typename T>
inline void stats_scan_typed(const T* BOLT_RESTRICT data,
                             const uint8_t* BOLT_RESTRICT validity,
                             int64_t n, ColumnStats* out) noexcept {
    assert(data != nullptr || n == 0);
    assert(out != nullptr);
    assert(n >= 0);

    constexpr int kSketchCap = 16;
    T sketch[kSketchCap];
    int sketch_n = 0;
    bool overflow = false;

    uint32_t null_count = 0;
    bool first = true;
    T mn = T{}, mx = T{};
    bool monotonic_asc = true;
    bool monotonic_desc = true;
    T prev = T{};

    // Welford's online mean + variance. Numerically stable single-pass.
    // `welford_n` counts valid samples; `welford_mean` is the running
    // mean; `welford_m2` is Σ(x − μ)² accumulated via
    //   δ₁ = x − μ_old;  μ_new = μ_old + δ₁ / n;  δ₂ = x − μ_new;
    //   M2 ← M2 + δ₁ · δ₂.
    // Double precision so that large-count runs don't lose low bits of
    // the variance.
    uint64_t welford_n = 0;
    double   welford_mean = 0.0;
    double   welford_m2   = 0.0;

    for (int64_t i = 0; i < n; ++i) {
        bool valid = true;
        if (validity) {
            valid = (validity[i >> 3] & (uint8_t)(1u << (i & 7))) != 0;
        }
        if (!valid) { null_count++; continue; }
        T v = data[i];
        if (first) { mn = v; mx = v; prev = v; first = false; }
        else {
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            if (v < prev) monotonic_asc = false;
            if (v > prev) monotonic_desc = false;
            prev = v;
        }
        // Welford step.
        welford_n += 1u;
        const double xd = static_cast<double>(v);
        const double delta1 = xd - welford_mean;
        welford_mean += delta1 / static_cast<double>(welford_n);
        const double delta2 = xd - welford_mean;
        welford_m2 += delta1 * delta2;

        if (!overflow) {
            bool seen = false;
            for (int k = 0; k < sketch_n; ++k) {
                if (sketch[k] == v) { seen = true; break; }
            }
            if (!seen) {
                if (sketch_n < kSketchCap) sketch[sketch_n++] = v;
                else overflow = true;
            }
        }
    }

    out->null_count = null_count;
    out->all_valid = (null_count == 0);
    if (first) {
        out->distinct_count = 0;
        out->cardinality = CardinalityClass::Unknown;
        // No valid samples → leave `mean` / `variance` at their NaN
        // sentinels from `make_default()`.
        return;
    }

    // Finalise Welford — σ² = M2 / N (population variance — chosen over
    // sample variance because the σ-bound consumer wants a closed-form
    // upper envelope of the observed data, not an unbiased estimator).
    if (welford_n > 0u) {
        out->mean = static_cast<float>(welford_mean);
        out->variance = static_cast<float>(
            welford_m2 / static_cast<double>(welford_n));
    }

    // Type-pun min/max into the 64-bit slots (floating types overlay).
    if constexpr (std::is_floating_point_v<T>) {
        double dmn = (double)mn, dmx = (double)mx;
        memcpy(&out->min_value, &dmn, sizeof(double));
        memcpy(&out->max_value, &dmx, sizeof(double));
    } else {
        out->min_value = (int64_t)mn;
        out->max_value = (int64_t)mx;
    }

    uint32_t dc = overflow ? (uint32_t)(kSketchCap + 1) : (uint32_t)sketch_n;
    out->distinct_count = dc;
    if (dc == 1)              out->cardinality = CardinalityClass::Constant;
    else if (dc <= 8)         out->cardinality = CardinalityClass::Low;
    else if (!overflow)       out->cardinality = CardinalityClass::Medium;
    else                      out->cardinality = CardinalityClass::High;

    out->is_monotonic = (monotonic_asc || monotonic_desc) && n > 1;
    if (n > 1 && monotonic_asc)       out->sort_order = SortOrder::Ascending;
    else if (n > 1 && monotonic_desc) out->sort_order = SortOrder::Descending;
    else                              out->sort_order = SortOrder::Unsorted;
}

}  // namespace detail

inline void BoltColumn::compute_stats_numeric() noexcept {
    assert(format == ColumnFormat::Flat || format == ColumnFormat::View);
    assert(length >= 0);

    const uint8_t* v = validity;
    // View columns with validity_offset != 0 are not supported here; fall back.
    if (format == ColumnFormat::View && validity_offset != 0) v = nullptr;

    switch (type) {
#define X(NAME, CTYPE, ENUM_VAL)                                           \
        case ENUM_VAL:                                                     \
            detail::stats_scan_typed<CTYPE>(                               \
                static_cast<const CTYPE*>(data), v, length, &stats);       \
            break;
        BOLT_NUMERIC_TYPES
#undef X
        // W-DEC: Decimal64 stats scan on the int64 mantissa. Mantissa
        // order == decimal order at a fixed scale, so min/max/monotonic
        // are exact; mean/variance are in mantissa units (consumers that
        // need decimal units divide by 10^scale). Deliberately NOT added
        // to BOLT_NUMERIC_TYPES — kernels opt in surface by surface so
        // unaware code falls back instead of ignoring the scale.
        case BoltType::Decimal64:
            detail::stats_scan_typed<int64_t>(
                static_cast<const int64_t*>(data), v, length, &stats);
            break;
        default:
            // Non-numeric type: leave stats untouched.
            break;
    }
}

// ============================================================================
// BoltColumn::compute_stats_vector — per-dim Welford for Embedding columns
// ============================================================================
//
// Vertical traversal: outer loop over rows, inner loop over dims. This
// keeps the inner sweep over `dim` contiguous floats, which AVX2 /
// NEON auto-vectorise. Welford recurrence per dim is numerically
// stable for f32 accumulation across millions of rows. Single pass —
// the caller is expected to invalidate the sidecar slot whenever the
// underlying data mutates.
inline VectorStats* BoltColumn::compute_stats_vector(Arena* arena_in) noexcept {
    assert(arena_in != nullptr);
    if (type != BoltType::Embedding) return nullptr;
    if (format != ColumnFormat::Flat) return nullptr;
    const uint32_t dim = vector_dim();
    if (dim == 0u || length <= 0) return nullptr;

    auto* vs = static_cast<VectorStats*>(
        arena_in->allocate(sizeof(VectorStats)));
    if (vs == nullptr) return nullptr;
    const size_t lane_bytes = static_cast<size_t>(dim) * sizeof(float);
    auto* mn = static_cast<float*>(arena_in->allocate(lane_bytes));
    auto* mx = static_cast<float*>(arena_in->allocate(lane_bytes));
    auto* mu = static_cast<float*>(arena_in->allocate(lane_bytes));
    auto* m2 = static_cast<float*>(arena_in->allocate(lane_bytes));
    if (!mn || !mx || !mu || !m2) return nullptr;

    const float* BOLT_RESTRICT row0 = vector_row(0);
    for (uint32_t d = 0; d < dim; ++d) {
        const float v = row0[d];
        mn[d] = v; mx[d] = v;
        mu[d] = v; m2[d] = 0.0f;
    }
    for (int64_t r = 1; r < length; ++r) {
        const float* BOLT_RESTRICT row = vector_row(r);
        const float inv_n = 1.0f / static_cast<float>(r + 1);
        for (uint32_t d = 0; d < dim; ++d) {
            const float v = row[d];
            if (v < mn[d]) mn[d] = v;
            if (v > mx[d]) mx[d] = v;
            const float delta = v - mu[d];
            mu[d] += delta * inv_n;
            m2[d] += delta * (v - mu[d]);
        }
    }
    const float denom = (length > 1)
        ? static_cast<float>(length - 1)
        : 1.0f;
    for (uint32_t d = 0; d < dim; ++d) m2[d] /= denom;

    vs->min       = mn;
    vs->max       = mx;
    vs->mean      = mu;
    vs->variance  = m2;
    vs->dim       = dim;
    vs->row_count = static_cast<uint32_t>(length);
    sidecars.vector_stats = vs;
    return vs;
}

// ============================================================================
// BoltColumn::clone_into
// ============================================================================

namespace detail {

// Utf8 spilled-bytes (`str_overflow_base`) sizing helper for `clone_into()`.
//
// `BoltColumn` has no "bytes actually used" field for the overflow buffer —
// `str_overflow_base`'s allocated capacity (tracked only by the producer,
// e.g. bolt_parquet_read.cpp's `ColCtx::overflow_cap`) is merely an upper
// bound. So a correct deep-copy must walk the StringView row array and find
// the highest `ref.offset + length` actually referenced by a VALID row —
// mirroring the inline-vs-spilled row walk `bolt_parquet_write.cpp`'s
// `encode_plain_byte_array` already performs for the same reason.
//
// Null rows are explicitly skipped (not merely `length <= 12` short-circuited):
// producers such as `plain_utf8()` in bolt_parquet_read.cpp `continue` on a
// null row WITHOUT writing that row's StringView slot at all, and
// `Arena::allocate()` does not zero-fill, so a null slot may hold arbitrary
// leftover bytes — including a `length` field that misreads as > 12 with a
// garbage `ref.offset`. Reading only valid rows' lengths avoids treating that
// garbage as a real spilled reference.
inline size_t utf8_overflow_used_bytes(const StringView* rows, int64_t n,
                                       const uint8_t* validity,
                                       int64_t validity_offset) noexcept {
    assert(rows != nullptr || n == 0);
    assert(n >= 0);
    size_t used = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (validity != nullptr) {
            const int64_t bit = validity_offset + i;
            const uint8_t b = (validity[bit >> 3] >> (bit & 7)) & 1u;
            if (b == 0u) continue;  // null row: slot may be uninitialized
        }
        if (rows[i].length > 12u) {
            const size_t end = static_cast<size_t>(rows[i].ref.offset) +
                               static_cast<size_t>(rows[i].length);
            if (end > used) used = end;
        }
    }
    return used;
}

// Like `utf8_overflow_used_bytes`, but ALSO reports the lowest referenced
// `ref.offset` (not just the highest end). `utf8_overflow_used_bytes` alone
// implicitly assumes `rows` is the FULL column starting at its own row 0 —
// correct for a whole-column clone, where offset 0 IS the start of
// `str_overflow_base`. It silently over-counts for a SLICE of a larger
// column (e.g. one chunk of a windowed/chunked wire-serialize): a window
// starting well past row 0 still has its rows' `ref.offset` values pointing
// at their true (large, absolute) position in the ORIGINAL overflow buffer,
// so `used = max(end)` alone reports "bytes from the start of the whole
// buffer through this window" instead of "bytes this window's own rows
// occupy" — a serializer that then copies `[0, used)` copies every earlier
// row's spilled bytes too, growing without bound as the window moves
// forward regardless of how few rows it contains. Callers that may see a
// sliced column (`bolt/wire/bolt_wire.h`'s Flat-Utf8 wire support) use this
// overload and copy `[*out_min, *out_min + *out_used)` instead.
// `*out_min`/`*out_used` are both 0 when no row in `[rows, rows+n)` spills.
inline void utf8_overflow_span(const StringView* rows, int64_t n,
                               const uint8_t* validity,
                               int64_t validity_offset,
                               size_t* out_min, size_t* out_used) noexcept {
    assert(rows != nullptr || n == 0);
    assert(n >= 0);
    assert(out_min != nullptr && out_used != nullptr);
    size_t min_off = 0, max_end = 0;
    bool any = false;
    for (int64_t i = 0; i < n; ++i) {
        if (validity != nullptr) {
            const int64_t bit = validity_offset + i;
            const uint8_t b = (validity[bit >> 3] >> (bit & 7)) & 1u;
            if (b == 0u) continue;  // null row: slot may be uninitialized
        }
        if (rows[i].length > 12u) {
            const size_t off = static_cast<size_t>(rows[i].ref.offset);
            const size_t end = off + static_cast<size_t>(rows[i].length);
            if (!any || off < min_off) min_off = off;
            if (end > max_end) max_end = end;
            any = true;
        }
    }
    *out_min  = any ? min_off : 0u;
    *out_used = any ? (max_end - min_off) : 0u;
}

}  // namespace detail

inline BoltColumn BoltColumn::clone_into(Arena* arena_in) const noexcept {
    assert(arena_in != nullptr);
    assert(length >= 0);
    BoltColumn c = *this;
    c.arena = arena_in;
    // G2FEAT-152: a NON-Dictionary column may carry an ADVISORY dictionary hint
    // on `dict_child` (the parquet reader publishes one for Utf8 so a consumer
    // can hash each dictionary entry once instead of re-hashing every row's
    // content). It points into the SOURCE arena, so a clone must DROP it rather
    // than deep-copy it: correctness never depends on the hint, and a hint
    // dangling into a reset arena is precisely the G2FEAT-307 use-after-free
    // class. The Dictionary case below owns its child for real and still
    // deep-copies it.
    if (format != ColumnFormat::Dictionary &&
        format != ColumnFormat::VarBinary) {
        c.dict_child = nullptr;
    }

    switch (format) {
        case ColumnFormat::VarBinary: {
            // A VarBinary column's OFFSETS live on `dict_child` — they are
            // load-bearing, not the advisory dictionary hint the comment
            // above describes. Before this case existed, the switch matched
            // nothing: `data` was carried over UNCOPIED (a dangling pointer
            // into the source arena the moment it reset — precisely the
            // G2FEAT-307 use-after-free class) and the blanket
            // `dict_child = nullptr` above threw the offsets away, so
            // `var_binary_at` null-dereferenced on the first read.
            //
            // Reached by every MarbleDB-scanned Utf8/Binary column: the scan
            // op deep-copies each column into the fragment's own arena
            // (G2FEAT-307), and MarbleDB hands strings back as VarBinary.
            if (length <= 0 || dict_child == nullptr ||
                dict_child->data == nullptr) {
                return make_empty();
            }
            const int32_t* offs = static_cast<const int32_t*>(dict_child->data);
            const int64_t  nbytes = static_cast<int64_t>(offs[length]);
            void* ndata = nullptr;
            if (nbytes > 0) {
                if (data == nullptr) return make_empty();
                ndata = arena_in->copy_into(data, static_cast<size_t>(nbytes));
                if (ndata == nullptr) return make_empty();
            }
            void* noffs = arena_in->copy_into(
                offs, static_cast<size_t>(length + 1) * sizeof(int32_t));
            if (noffs == nullptr) return make_empty();
            BoltColumn* oc = arena_in->allocate_array<BoltColumn>(1);
            if (oc == nullptr) return make_empty();
            *oc = BoltColumn::make_flat(static_cast<int32_t*>(noffs), nullptr,
                                        length + 1, BoltType::Int32);
            c.data       = ndata;
            c.dict_child = oc;
            break;
        }
        case ColumnFormat::Flat: {
            if (length > 0 && type_size_bytes > 0 && data) {
                size_t bytes = (size_t)length * (size_t)type_size_bytes;
                void* ndata = arena_in->copy_into(data, bytes);
                if (!ndata) return make_empty();
                c.data = ndata;
            } else {
                c.data = nullptr;
            }
            if (validity && length > 0) {
                size_t vbytes = ((size_t)length + 7) / 8;
                void* nval = arena_in->copy_into(validity, vbytes);
                if (!nval) return make_empty();
                c.validity = static_cast<uint8_t*>(nval);
                c.validity_offset = 0;
            } else {
                c.validity = nullptr;
                c.validity_offset = 0;
            }
            break;
        }
        case ColumnFormat::Constant: {
            // inline_value is part of the struct; the shallow copy already
            // captured the 16 bytes. Repoint data → our own inline_value.
            c.data = c.inline_value;
            break;
        }
        case ColumnFormat::Sequence: {
            // seq_offset/seq_step covered by shallow copy.
            c.data = nullptr;
            break;
        }
        case ColumnFormat::View: {
            // Promote view → flat by materializing, then deep-copying.
            // Caller semantics: clone should be self-contained.
            if (length > 0 && type_size_bytes > 0 && data) {
                size_t bytes = (size_t)length * (size_t)type_size_bytes;
                void* ndata = arena_in->copy_into(data, bytes);
                if (!ndata) return make_empty();
                c.data = ndata;
                c.format = ColumnFormat::Flat;
            }
            if (validity) {
                // Shift bits so validity_offset becomes 0.
                size_t vbytes = ((size_t)length + 7) / 8;
                uint8_t* nval = static_cast<uint8_t*>(
                    arena_in->allocate_zeroed(vbytes));
                if (!nval) return make_empty();
                for (int64_t i = 0; i < length; ++i) {
                    int64_t srcbit = validity_offset + i;
                    uint8_t bit = (validity[srcbit >> 3] >> (srcbit & 7)) & 1u;
                    nval[i >> 3] |= (uint8_t)(bit << (i & 7));
                }
                c.validity = nval;
                c.validity_offset = 0;
            }
            break;
        }
        case ColumnFormat::Dictionary: {
            // Copy keys buffer (sizeof determined by type_size_bytes on the
            // dictionary column's data field).
            if (length > 0 && type_size_bytes > 0 && data) {
                size_t bytes = (size_t)length * (size_t)type_size_bytes;
                void* ndata = arena_in->copy_into(data, bytes);
                if (!ndata) return make_empty();
                c.data = ndata;
            }
            if (dict_child) {
                BoltColumn* nc = arena_in->allocate_array<BoltColumn>(1);
                if (!nc) return make_empty();
                *nc = dict_child->clone_into(arena_in);
                c.dict_child = nc;
            }
            // NOTE: a Dictionary-format column's own `data` holds integer
            // keys, not StringViews, even when its logical `type` is Utf8 —
            // the real StringView payload (and str_overflow_base) lives on
            // `dict_child`, already deep-copied above. Do not run the Utf8
            // overflow walk below against key data.
            return c;
        }
    }

    // Utf8 spilled-string overflow deep-copy (G2FEAT-307 bug class): Flat
    // (including View, promoted to Flat above) and Constant are the only
    // formats whose `data`/`inline_value` are StringView-shaped. Every other
    // format either can't carry Utf8 payload directly (Dictionary handled
    // via `return c` above) or is numeric-only (Sequence/RLE/BitPacked/
    // FrameOfRef) — str_overflow_base stays whatever the shallow copy gave
    // it (normally nullptr) for those.
    if (type == BoltType::Utf8 && str_overflow_base != nullptr) {
        size_t used = 0;
        if (c.format == ColumnFormat::Flat) {
            const auto* rows = static_cast<const StringView*>(c.data);
            used = detail::utf8_overflow_used_bytes(rows, c.length, c.validity,
                                                    c.validity_offset);
        } else if (c.format == ColumnFormat::Constant) {
            const auto* row = reinterpret_cast<const StringView*>(c.inline_value);
            used = detail::utf8_overflow_used_bytes(row, 1, nullptr, 0);
        }
        if (used > 0) {
            void* nspill = arena_in->copy_into(str_overflow_base, used);
            if (!nspill) return make_empty();
            c.str_overflow_base = nspill;
        } else {
            // No row actually references the spill buffer (all inline, or
            // every referencing row was null) — do not carry a stale
            // pointer into the destination arena's lifetime.
            c.str_overflow_base = nullptr;
        }
    }
    return c;
}

// ============================================================================
// BoltColumn::materialize
// ============================================================================

inline BoltColumn BoltColumn::materialize(Arena* arena_in) const noexcept {
    assert(arena_in != nullptr);
    assert(length >= 0);

    if (format == ColumnFormat::Flat) return clone_into(arena_in);

    // For Dictionary the outer type_size_bytes is the key width, not the
    // logical value width — consult the type-size table instead.
    size_t tsz = (format == ColumnFormat::Dictionary)
        ? bolt::type_size(type)
        : (type_size_bytes ? (size_t)type_size_bytes : bolt::type_size(type));
    if (tsz == 0) return make_empty();

    BoltColumn out = make_empty();
    out.type = type;
    out.type_size_bytes = (uint16_t)tsz;
    out.length = length;
    out.format = ColumnFormat::Flat;
    out.arena = arena_in;
    out.stats = stats;

    if (length == 0) { out.data = nullptr; return out; }

    void* buf = arena_in->allocate((size_t)length * tsz);
    if (!buf) return make_empty();
    out.data = buf;

    if (format == ColumnFormat::Constant) {
        uint8_t* dst = static_cast<uint8_t*>(buf);
        for (int64_t i = 0; i < length; ++i) {
            memcpy(dst + (size_t)i * tsz, inline_value, tsz);
        }
    } else if (format == ColumnFormat::Sequence) {
        // Only Int64 (or int-convertible) sequences are supported; dispatch.
        if (tsz == 8) {
            int64_t* dst = static_cast<int64_t*>(buf);
            for (int64_t i = 0; i < length; ++i)
                dst[i] = seq_offset + i * seq_step;
        } else if (tsz == 4) {
            int32_t* dst = static_cast<int32_t*>(buf);
            for (int64_t i = 0; i < length; ++i)
                dst[i] = (int32_t)(seq_offset + i * seq_step);
        } else {
            return make_empty();
        }
    } else if (format == ColumnFormat::View) {
        memcpy(buf, data, (size_t)length * tsz);
        if (validity) {
            size_t vbytes = ((size_t)length + 7) / 8;
            uint8_t* nval = static_cast<uint8_t*>(arena_in->allocate_zeroed(vbytes));
            if (!nval) return make_empty();
            for (int64_t i = 0; i < length; ++i) {
                int64_t srcbit = validity_offset + i;
                uint8_t bit = (validity[srcbit >> 3] >> (srcbit & 7)) & 1u;
                nval[i >> 3] |= (uint8_t)(bit << (i & 7));
            }
            out.validity = nval;
        }
    } else if (format == ColumnFormat::BitPacked ||
               format == ColumnFormat::FrameOfRef) {
        // B3/B4 unpack — read `bit_width` bits at a time from `data` and
        // store each decoded value (plus base for FOR) into the Flat buffer.
        if (!data) return make_empty();
        const uint64_t* words = static_cast<const uint64_t*>(data);
        const int64_t  bw = seq_step;      // bit_width
        if (bw < 1 || bw > 32) return make_empty();
        const int64_t base = (format == ColumnFormat::FrameOfRef) ? seq_offset : 0;
        const uint64_t mask = (bw == 64) ? ~uint64_t{0}
                                         : ((uint64_t{1} << bw) - 1u);
        for (int64_t i = 0; i < length; ++i) {
            const uint64_t bit_off     = static_cast<uint64_t>(i) * static_cast<uint64_t>(bw);
            const uint64_t word_off    = bit_off >> 6;
            const uint64_t bit_in_word = bit_off & 63u;
            uint64_t v = words[word_off] >> bit_in_word;
            if (bit_in_word + static_cast<uint64_t>(bw) > 64u) {
                v |= words[word_off + 1] << (64u - bit_in_word);
            }
            v &= mask;
            const int64_t  full = base + static_cast<int64_t>(v);
            uint8_t*       dst  = static_cast<uint8_t*>(buf) + (size_t)i * tsz;
            // Narrow to the target type's width (all supported widths are ≤8).
            if      (tsz == 1) { uint8_t  t = static_cast<uint8_t>(full);  memcpy(dst, &t, 1); }
            else if (tsz == 2) { uint16_t t = static_cast<uint16_t>(full); memcpy(dst, &t, 2); }
            else if (tsz == 4) { uint32_t t = static_cast<uint32_t>(full); memcpy(dst, &t, 4); }
            else if (tsz == 8) { int64_t  t = full;                        memcpy(dst, &t, 8); }
            else return make_empty();
        }
    } else if (format == ColumnFormat::RLE) {
        // Expand runs: for each run i, memset/memcpy `values[i]` into
        // `out[run_ends[i-1] .. run_ends[i])`.  Values buffer sits at
        // `data`, run_ends at `dict_child->data` (int32, length=num_runs).
        if (!dict_child || !data) return make_empty();
        if (dict_child->format != ColumnFormat::Flat) return make_empty();
        if (dict_child->type != BoltType::Int32) return make_empty();
        const int64_t num_runs = dict_child->length;
        if (num_runs <= 0) { return out; }  // empty + early-exit keeps out.data
        const uint8_t*  vals = static_cast<const uint8_t*>(data);
        const int32_t* rends = static_cast<const int32_t*>(dict_child->data);
        uint8_t* dst = static_cast<uint8_t*>(buf);
        int32_t prev = 0;
        for (int64_t i = 0; i < num_runs; ++i) {
            const int32_t end = rends[i];
            assert(end >= prev);
            const int64_t run_len = static_cast<int64_t>(end) - prev;
            const uint8_t* src = vals + (size_t)i * tsz;
            for (int64_t r = 0; r < run_len; ++r) {
                memcpy(dst + (size_t)(prev + r) * tsz, src, tsz);
            }
            prev = end;
        }
    } else if (format == ColumnFormat::Dictionary) {
        // Expand keys → values via dict_child.
        if (!dict_child || !data) return make_empty();
        BoltColumn vals = dict_child->materialize(arena_in);
        if (!vals.data) return make_empty();
        const uint8_t* vals_bytes = static_cast<const uint8_t*>(vals.data);
        uint8_t* dst = static_cast<uint8_t*>(buf);
        // Key width determined by type_size_bytes of the dictionary column.
        uint16_t key_w = type_size_bytes;
        for (int64_t i = 0; i < length; ++i) {
            uint32_t k = 0;
            const uint8_t* kp = static_cast<const uint8_t*>(data) + (size_t)i * key_w;
            if (key_w == 1)      k = kp[0];
            else if (key_w == 2) { uint16_t t; memcpy(&t, kp, 2); k = t; }
            else if (key_w == 4) { uint32_t t; memcpy(&t, kp, 4); k = t; }
            memcpy(dst + (size_t)i * tsz, vals_bytes + (size_t)k * tsz, tsz);
        }
    }
    return out;
}

// ============================================================================
// BoltColumn::try_promote
// ============================================================================

inline bool BoltColumn::try_promote(Arena* arena_in) noexcept {
    assert(arena_in != nullptr);
    assert(length >= 0);

    if (format != ColumnFormat::Flat) return false;
    if (length <= 0) return false;
    if (stats.distinct_count == 0) return false;

    // Flat → Constant
    if (stats.distinct_count == 1 && type_size_bytes > 0 &&
        type_size_bytes <= 16 && data) {
        memcpy(inline_value, data, type_size_bytes);
        data = inline_value;
        format = ColumnFormat::Constant;
        stats.cardinality = CardinalityClass::Constant;
        return true;
    }

    // Flat → Dictionary heuristic:
    //   distinct * log2(distinct) + N * key_width < N * sizeof(T) * 0.7
    // Only attempt when distinct_count is exact (no sketch overflow) and
    // distinct <= 16 (sketch cap). Narrow niche but fits the X-macro scan.
    if (stats.distinct_count > 1 && stats.distinct_count <= 16 &&
        type_size_bytes >= 4 && data) {
        uint32_t dc = stats.distinct_count;
        double cost_dict = (double)dc * (double)type_size_bytes
                         + (double)length * 1.0; // 1-byte key (dc<=16)
        double cost_flat = (double)length * (double)type_size_bytes * 0.7;
        if (cost_dict >= cost_flat) return false;

        // Build dictionary: unique values + uint8 keys.
        uint8_t  values_buf[16 * 16];  // up to 16 entries of ≤16 bytes each
        uint32_t n_unique = 0;
        uint16_t tw = type_size_bytes;
        if (tw > 16) return false;

        uint8_t* keys = arena_in->allocate_array<uint8_t>((size_t)length);
        if (!keys) return false;

        const uint8_t* src = static_cast<const uint8_t*>(data);
        for (int64_t i = 0; i < length; ++i) {
            const uint8_t* vp = src + (size_t)i * tw;
            int found = -1;
            for (uint32_t k = 0; k < n_unique; ++k) {
                if (memcmp(values_buf + (size_t)k * tw, vp, tw) == 0) {
                    found = (int)k; break;
                }
            }
            if (found < 0) {
                if (n_unique >= 16) return false; // sketch lied; bail
                memcpy(values_buf + (size_t)n_unique * tw, vp, tw);
                found = (int)n_unique++;
            }
            keys[i] = (uint8_t)found;
        }

        // Allocate and fill dict_child (Flat column of unique values).
        BoltColumn* child = arena_in->allocate_array<BoltColumn>(1);
        if (!child) return false;
        *child = make_empty();
        void* child_data = arena_in->copy_into(values_buf, (size_t)n_unique * tw);
        if (!child_data) return false;
        child->data = child_data;
        child->length = (int64_t)n_unique;
        child->format = ColumnFormat::Flat;
        child->type = type;
        child->type_size_bytes = tw;
        child->arena = arena_in;
        child->stats.all_valid = true;
        child->stats.distinct_count = n_unique;

        // Mutate self into Dictionary view.
        data = keys;
        dict_child = child;
        format = ColumnFormat::Dictionary;
        type_size_bytes = 1;  // key width
        return true;
    }
    return false;
}

// ============================================================================
// BoltBatch::fill_arrow_schema — struct schema with N children
// ============================================================================

inline void BoltBatch::fill_arrow_schema(ArrowSchema* out) const noexcept {
    assert(out != nullptr);
    assert(arena != nullptr);
    assert(num_cols <= kMaxBatchColumns);

    memset(out, 0, sizeof(ArrowSchema));
    out->format = "+s";
    out->name = "";
    out->flags = 0;
    out->n_children = (int64_t)num_cols;

    if (num_cols == 0) {
        out->children = nullptr;
        out->release = &BoltColumn::noop_release_schema;
        return;
    }

    // child pointer array + backing child structs
    ArrowSchema** kids = arena->allocate_array<ArrowSchema*>(num_cols);
    ArrowSchema*  bodies = arena->allocate_array<ArrowSchema>(num_cols);
    assert(kids != nullptr);
    assert(bodies != nullptr);
    for (uint32_t i = 0; i < num_cols; ++i) {
        const BoltField& f = schema.fields[i];
        columns[read_epoch][i].fill_arrow_schema(&bodies[i], f.name);
        kids[i] = &bodies[i];
    }
    out->children = kids;
    out->dictionary = nullptr;
    out->release = &BoltColumn::noop_release_schema;
    out->private_data = nullptr;
}

// ============================================================================
// BitmapIndex
// ============================================================================

inline BitmapIndex* BitmapIndex::build(const BoltColumn& col,
                                       Arena* arena) noexcept {
    assert(arena != nullptr);
    assert(col.length >= 0);
    if (col.format != ColumnFormat::Dictionary) return nullptr;
    if (col.length == 0) return nullptr;
    if (!col.data) return nullptr;

    uint16_t key_w = col.type_size_bytes;
    if (key_w != 1 && key_w != 2 && key_w != 4) return nullptr;

    // K = number of distinct keys; derive from dict_child->length if available.
    uint32_t K = 0;
    if (col.dict_child && col.dict_child->length > 0) {
        K = (uint32_t)col.dict_child->length;
    } else {
        // Scan keys to find max.
        const uint8_t* kp = static_cast<const uint8_t*>(col.data);
        uint32_t maxk = 0;
        for (int64_t i = 0; i < col.length; ++i) {
            uint32_t k = 0;
            const uint8_t* p = kp + (size_t)i * key_w;
            if (key_w == 1)      k = p[0];
            else if (key_w == 2) { uint16_t t; memcpy(&t, p, 2); k = t; }
            else                 { uint32_t t; memcpy(&t, p, 4); k = t; }
            if (k > maxk) maxk = k;
        }
        K = maxk + 1;
    }
    if (K == 0) return nullptr;

    uint32_t words = (uint32_t)(((uint64_t)col.length + 63) / 64);

    BitmapIndex* idx = arena->allocate_array<BitmapIndex>(1);
    if (!idx) return nullptr;
    idx->num_keys = K;
    idx->num_rows = (uint32_t)col.length;
    idx->words_per_bitmap = words;

    uint64_t** maps = arena->allocate_array<uint64_t*>(K);
    uint32_t*  pc   = arena->allocate_array<uint32_t>(K);
    if (!maps || !pc) return nullptr;
    memset(pc, 0, (size_t)K * sizeof(uint32_t));
    for (uint32_t k = 0; k < K; ++k) {
        uint64_t* bm = static_cast<uint64_t*>(
            arena->allocate_zeroed((size_t)words * sizeof(uint64_t)));
        if (!bm) return nullptr;
        maps[k] = bm;
    }
    idx->bitmaps   = maps;
    idx->popcounts = pc;

    const uint8_t* kp = static_cast<const uint8_t*>(col.data);
    for (int64_t i = 0; i < col.length; ++i) {
        uint32_t k = 0;
        const uint8_t* p = kp + (size_t)i * key_w;
        if (key_w == 1)      k = p[0];
        else if (key_w == 2) { uint16_t t; memcpy(&t, p, 2); k = t; }
        else                 { uint32_t t; memcpy(&t, p, 4); k = t; }
        if (k < K) {
            maps[k][(uint64_t)i >> 6] |= (1ULL << ((uint64_t)i & 63));
            pc[k]++;
        }
    }
    return idx;
}

inline uint32_t BitmapIndex::count(uint32_t key) const noexcept {
    assert(bitmaps != nullptr);
    assert(words_per_bitmap > 0 || num_rows == 0);
    if (key >= num_keys) return 0;
    // H3 — O(1) via the precomputed popcount sidecar. Falls back to a
    // full bitmap scan only if popcounts wasn't built (legacy callers).
    if (popcounts != nullptr) return popcounts[key];
    const uint64_t* bm = bitmaps[key];
    uint32_t sum = 0;
    for (uint32_t w = 0; w < words_per_bitmap; ++w) {
        sum += bolt_popcount64(bm[w]);
    }
    return sum;
}

inline int64_t BitmapIndex::filter(uint32_t key, int32_t* out) const noexcept {
    assert(out != nullptr);
    assert(bitmaps != nullptr);
    if (key >= num_keys) return 0;
    // H3 — skip the whole bitmap walk for keys that were never observed.
    if (popcounts != nullptr && popcounts[key] == 0u) return 0;
    const uint64_t* bm = bitmaps[key];
    int64_t count = 0;
    for (uint32_t w = 0; w < words_per_bitmap; ++w) {
        uint64_t word = bm[w];
        uint64_t base = (uint64_t)w << 6;
        while (word) {
            int b = bolt_ctz64(word);
            int64_t row = (int64_t)(base + (uint64_t)b);
            if (row >= (int64_t)num_rows) break;
            out[count++] = (int32_t)row;
            word &= word - 1;
        }
    }
    return count;
}

inline int64_t BitmapIndex::filter_in(const uint32_t* keys, uint32_t nkeys,
                                      int32_t* out) const noexcept {
    assert(out != nullptr);
    assert(bitmaps != nullptr);
    assert(keys != nullptr || nkeys == 0);
    if (nkeys == 0 || words_per_bitmap == 0) return 0;

    Arena* a = tl_arena;
    assert(a != nullptr && "filter_in requires tl_arena for temp OR buffer");
    uint64_t* tmp = static_cast<uint64_t*>(
        a->allocate_zeroed((size_t)words_per_bitmap * sizeof(uint64_t)));
    if (!tmp) return 0;

    for (uint32_t i = 0; i < nkeys; ++i) {
        uint32_t k = keys[i];
        if (k >= num_keys) continue;
        const uint64_t* bm = bitmaps[k];
        for (uint32_t w = 0; w < words_per_bitmap; ++w) tmp[w] |= bm[w];
    }

    int64_t count = 0;
    for (uint32_t w = 0; w < words_per_bitmap; ++w) {
        uint64_t word = tmp[w];
        uint64_t base = (uint64_t)w << 6;
        while (word) {
            int b = bolt_ctz64(word);
            int64_t row = (int64_t)(base + (uint64_t)b);
            if (row >= (int64_t)num_rows) break;
            out[count++] = (int32_t)row;
            word &= word - 1;
        }
    }
    return count;
}

// ============================================================================
// BoltColumn::ensure_bitmap_index (C5)
// ============================================================================
// Lazily attach a BitmapIndex to a Dictionary column; cached in the
// `bitmap_index` slot. Callers typically hit this once per column at
// plan time and re-use across filter / semi-join probes.
//
// NOTE: default filter dispatch does NOT consult the index — callers
// opt in by calling `col.ensure_bitmap_index(arena)->filter(key, out)`
// directly. This matches the keep-code-paths-for-JIT-later policy:
// the primitive is in the binary, an explicit name reaches it; a
// future planner can choose bitmap-vs-scan per column based on
// cardinality and selectivity stats.
// ============================================================================

inline BitmapIndex* BoltColumn::ensure_bitmap_index(Arena* arena_in) noexcept {
    assert(arena_in != nullptr);
    if (sidecars.bitmap_index != nullptr) {
        return static_cast<BitmapIndex*>(sidecars.bitmap_index);
    }
    if (format != ColumnFormat::Dictionary) return nullptr;
    BitmapIndex* idx = BitmapIndex::build(*this, arena_in);
    if (idx == nullptr) return nullptr;
    sidecars.bitmap_index = idx;
    return idx;
}

// ============================================================================
// gather_to_column<T> — materialize a selection of a Flat column into a new
// arena-backed Flat column. Used by join probe paths that have (column +
// selection vector) and need a compacted column for a downstream operator.
//
// Contract:
//   - src must be Flat (for now). Other formats return an empty column and
//     trip BOLT_NYI in debug builds.
//   - sel must have `sel_n` valid int32 indices in [0, src.length).
//   - arena must be non-null; result is arena-owned.
//   - Validity IS propagated: if src has a validity bitmap and is not
//     all_valid, the output gets a freshly-packed bitmap with bit i set iff
//     src is valid at position sel[i]. src.validity_offset is honoured.
// ============================================================================
template <typename T>
inline BoltColumn gather_to_column(const BoltColumn& src,
                                   const int32_t* BOLT_RESTRICT sel,
                                   int64_t sel_n,
                                   Arena* arena) noexcept {
    assert(arena != nullptr);
    assert(sel_n >= 0);
    assert(sel != nullptr || sel_n == 0);

    if (src.format != ColumnFormat::Flat) {
        BOLT_NYI("gather_to_column: only Flat src supported");
        return BoltColumn::make_empty();
    }
    assert(static_cast<size_t>(src.type_size_bytes) == sizeof(T));

    BoltColumn out = BoltColumn::make_flat_alloc(sel_n, src.type, arena);
    if (sel_n > 0 && out.data == nullptr) return out;  // OOM — empty result.

    const T* BOLT_RESTRICT in = static_cast<const T*>(src.data);
    T* BOLT_RESTRICT dst = static_cast<T*>(out.data);
    const int64_t src_n = src.length;
    for (int64_t i = 0; i < sel_n; ++i) {
        const int32_t idx = sel[i];
        assert(idx >= 0 && static_cast<int64_t>(idx) < src_n);
        dst[i] = in[idx];
    }
    (void)src_n;

    // Propagate validity if the source actually carries nulls. If src is
    // all_valid (no nulls) or has no bitmap, skip — leaves out.validity
    // null and out.stats.all_valid = true (set by make_flat_alloc).
    if (src.validity != nullptr && !src.stats.all_valid && sel_n > 0) {
        const size_t vbytes = (static_cast<size_t>(sel_n) + 7u) / 8u;
        uint8_t* nval = static_cast<uint8_t*>(arena->allocate_zeroed(vbytes));
        if (nval != nullptr) {
            assert(src.validity_offset >= 0);
            int64_t null_count = 0;
            for (int64_t i = 0; i < sel_n; ++i) {
                const int64_t srcbit = src.validity_offset +
                                       static_cast<int64_t>(sel[i]);
                const uint8_t bit =
                    (src.validity[srcbit >> 3] >> (srcbit & 7)) & 1u;
                nval[i >> 3] |= static_cast<uint8_t>(bit << (i & 7));
                null_count += (1 - static_cast<int64_t>(bit));
            }
            out.validity = nval;
            out.validity_offset = 0;
            // null_count is uint32_t; sel_n is capped at int64 but in
            // practice arrays this big would OOM long before overflow.
            assert(null_count >= 0 && null_count <= 0xFFFFFFFFLL);
            out.stats.null_count = static_cast<uint32_t>(null_count);
            out.stats.all_valid = (null_count == 0);
        }
        // If arena OOM'd the validity bitmap: leave out.validity = null and
        // stats.all_valid = true. Callers already have to handle OOM on
        // the main data buffer; a missing validity buffer is degraded but
        // not corrupt (rows appear valid; equivalent to all_valid claim).
    }
    return out;
}

}  // namespace bolt
