// bolt_column.h — Adaptive column with stats, sidecars, C Data Interface export
//
// RULES: No exceptions. No RTTI. No smart pointers. No std::string. No std::vector.
// Arena-owned memory. Returns nullptr/false on failure. Fixed-size inline stats.
//
// Arrow compatibility: zero-copy export via ArrowArray/ArrowSchema C structs.
// No libarrow link required. Any Arrow consumer (Polars, DuckDB, Pandas) can
// read our columns directly through the C Data Interface.

#pragma once

#include "bolt_types.h"
#include "bolt_arena.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <atomic>

namespace chukonu {
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
    int64_t  min_value;        // Type-punned for fixed-width types
    int64_t  max_value;
    uint32_t null_count;
    uint32_t distinct_count;   // Exact if < 1024, HLL approx otherwise
    CardinalityClass cardinality;
    SortOrder sort_order;
    bool     all_valid;        // No nulls → skip validity bitmap
    bool     is_monotonic;
    uint32_t max_string_len;
    uint32_t total_string_bytes;
    uint16_t avg_string_len;
    bool     all_inline;       // All strings ≤ 12 bytes
    uint8_t  _pad[13];

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
        return s;
    }
};

static_assert(sizeof(ColumnStats) == 64, "ColumnStats must be one cache line");

// ============================================================================
// Sidecar Index Slots — arena-allocated, ephemeral per morsel
// ============================================================================

struct SidecarSlots {
    void* bitmap_index;    // BitmapIndex*
    void* hash_index;      // HashIndex*
    void* sort_index;      // int32_t* permutation
    void* bloom_filter;    // BloomFilter*
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
};

// ============================================================================
// BoltColumn
// ============================================================================

/// Forward declaration for dictionary child
struct BoltColumn;

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
        }
        return 0;
    }

    // =====================================================================
    // Arrow C Data Interface export (zero-copy, no libarrow link)
    // =====================================================================

    /// Fill an ArrowSchema struct for this column's type.
    /// The schema's release callback will be set to a no-op (Bolt owns memory).
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

    /// Try to promote format based on stats (e.g., Flat → Constant)
    bool try_promote(Arena* arena) noexcept;

    /// Deep-copy into arena
    BoltColumn clone_into(Arena* arena) const noexcept;

    /// Materialize non-Flat formats to Flat (arena-allocated)
    BoltColumn materialize(Arena* arena) const noexcept;

private:
    static void noop_release_schema(ArrowSchema*) noexcept {}
    static void noop_release_array(ArrowArray*) noexcept {}
};

// ============================================================================
// BoltBatch — double-buffered, COW, Venus tick-tock pattern
// ============================================================================

static constexpr uint32_t kMaxBatchColumns = 256;

struct alignas(64) BoltBatch {
    // Double-buffered columns
    BoltColumn columns[2][kMaxBatchColumns];

    // Dirty tracking (own cache line)
    alignas(64) std::atomic<uint64_t> dirty_mask_lo;  // Columns 0-63
    alignas(64) std::atomic<uint64_t> dirty_mask_hi;  // Columns 64-127
    // (Columns 128-255 rarely mutated, no atomic tracking needed)

    // Epoch
    uint8_t  read_epoch;
    uint8_t  write_epoch;
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

    static BoltBatch make_empty() noexcept {
        BoltBatch b;
        memset(&b, 0, sizeof(b));
        b.read_epoch = 0;
        b.write_epoch = 1;
        b.dirty_mask_lo.store(0, std::memory_order_relaxed);
        b.dirty_mask_hi.store(0, std::memory_order_relaxed);
        return b;
    }

    // =====================================================================
    // Read access (zero atomics)
    // =====================================================================

    const BoltColumn& col(uint32_t i) const noexcept {
        assert(i < num_cols);
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
        dirty_mask_lo.store(0, std::memory_order_release);
        dirty_mask_hi.store(0, std::memory_order_release);
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
        if (col_idx < 64) {
            uint64_t bit = 1ULL << col_idx;
            uint64_t mask = dirty_mask_lo.load(std::memory_order_acquire);
            if (!(mask & bit)) {
                if (generation[read_epoch] != generation[write_epoch] && arena) {
                    columns[write_epoch][col_idx] =
                        columns[read_epoch][col_idx].clone_into(arena);
                }
                dirty_mask_lo.fetch_or(bit, std::memory_order_release);
            }
        } else if (col_idx < 128) {
            uint64_t bit = 1ULL << (col_idx - 64);
            uint64_t mask = dirty_mask_hi.load(std::memory_order_acquire);
            if (!(mask & bit)) {
                if (generation[read_epoch] != generation[write_epoch] && arena) {
                    columns[write_epoch][col_idx] =
                        columns[read_epoch][col_idx].clone_into(arena);
                }
                dirty_mask_hi.fetch_or(bit, std::memory_order_release);
            }
        }
        // Columns 128+ don't get atomic COW tracking (rarely mutated)
    }
};

// ============================================================================
// BitmapIndex (QuestDB-style, arena-allocated)
// ============================================================================

struct BitmapIndex {
    uint64_t** bitmaps;     // K bitmaps, each ceil(N/64) words
    uint32_t   num_keys;
    uint32_t   num_rows;
    uint32_t   words_per_bitmap;

    /// Build from a Dictionary column's uint8/uint16/uint32 keys.
    /// Arena-allocated. Returns nullptr on failure.
    static BitmapIndex* build(const BoltColumn& col, Arena* arena) noexcept;

    /// Count rows with value == key
    uint32_t count(uint32_t key) const noexcept;

    /// Write matching row indices. Returns count written.
    int64_t filter(uint32_t key, int32_t* out) const noexcept;

    /// OR of multiple keys. Returns count written.
    int64_t filter_in(const uint32_t* keys, uint32_t nkeys,
                      int32_t* out) const noexcept;
};

}  // namespace bolt
}  // namespace chukonu
