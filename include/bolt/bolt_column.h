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
    RLE        = 5,   // Run-length: `data` = values[num_runs];
                      // `dict_child` = int32 run_ends[num_runs]
                      // where run i covers [run_ends[i-1], run_ends[i]).
    BitPacked  = 6,   // Bit-packed: `data` = uint64 words, each value uses
                      // `seq_step` bits (reused from union). Width ∈ [1,32].
    FrameOfRef = 7,   // FOR = base + bit-packed deltas: `data` = delta words,
                      // `seq_offset` = base value, `seq_step` = bit width.
};

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

    // Initialize an already-allocated BoltBatch in place. Returning by value
    // is not possible: std::atomic members are non-copyable.
    static void init_empty(BoltBatch* b) noexcept {
        // zero all non-atomic fields; then explicitly init atomics
        memset(reinterpret_cast<char*>(b) + offsetof(BoltBatch, read_epoch),
               0, sizeof(BoltBatch) - offsetof(BoltBatch, read_epoch));
        b->read_epoch  = 0;
        b->write_epoch = 1;
        b->dirty_mask_lo.store(0, std::memory_order_relaxed);
        b->dirty_mask_hi.store(0, std::memory_order_relaxed);
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
        return;
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
        default:
            // Non-numeric type: leave stats untouched.
            break;
    }
}

// ============================================================================
// BoltColumn::clone_into
// ============================================================================

inline BoltColumn BoltColumn::clone_into(Arena* arena_in) const noexcept {
    assert(arena_in != nullptr);
    assert(length >= 0);
    BoltColumn c = *this;
    c.arena = arena_in;

    switch (format) {
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
            break;
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
