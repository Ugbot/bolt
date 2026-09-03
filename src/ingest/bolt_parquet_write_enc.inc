// bolt_parquet_write_enc.inc — value encoders, page planning, dictionary.
//
// Included ONCE from bolt_parquet_write.cpp, inside
// namespace bolt::ingest::parquet { namespace { ... } }. Not a standalone
// translation unit and not a public header: it is a size seam, per the repo
// rule that a source file stays well under 1.5k lines.
//
// What lives here:
//   - plan_page_rows()  : how many rows fit a page's uncompressed byte budget
//   - RleHybrid         : the RLE / bit-packed hybrid encoder (parquet's
//                         "hybrid" stream) used for dictionary indices
//   - DictBuilder       : per-column-chunk dictionary construction
//   - the PLAIN value encoders, range-based ([r0, r1))
//
// Every function here is noexcept, allocation-bounded by explicit caps, and
// carries >= 2 assertions. Bit layout note: parquet's bit-packed run is
// LITTLE-ENDIAN bit order within a group of 8 values -- the exact layout
// bolt_parquet_read.cpp's unpack_le_bounded consumes. pack_le_bits below is
// its inverse and the two are pinned against each other by a round-trip test.

// ===== bit packing ========================================================

// Append `n` values of `bw` bits each, LSB-first little-endian, to `dst`.
// `n` MUST be a multiple of 8 (a parquet bit-packed run is whole groups of
// 8); the caller pads. bw is in [1, 32].
void pack_le_bits(const std::uint32_t* vals, std::uint32_t n,
                  std::uint32_t bw, std::vector<std::uint8_t>* dst) noexcept {
    assert(dst != nullptr);
    assert(vals != nullptr || n == 0u);
    assert((n & 7u) == 0u);
    assert(bw >= 1u && bw <= 32u);
    if (n == 0u) return;
    // A group of 8 values at bw bits is exactly bw bytes, so the output size
    // is known up front and the inner loop needs no capacity check.
    const std::size_t base = dst->size();
    const std::size_t add = (static_cast<std::size_t>(n) / 8u) * bw;
    dst->resize(base + add);
    std::uint8_t* out = dst->data() + base;
    std::memset(out, 0, add);

    // Branch-free accumulation: a 64-bit window absorbs up to 32 bits per
    // value and is drained a byte at a time. No per-bit loop, no conditional
    // on the value itself.
    std::uint64_t acc = 0;
    std::uint32_t acc_bits = 0;
    std::size_t o = 0;
    const std::uint64_t mask =
        (bw == 32u) ? 0xFFFFFFFFull : ((1ull << bw) - 1ull);
    for (std::uint32_t i = 0; i < n; ++i) {
        acc |= (static_cast<std::uint64_t>(vals[i]) & mask) << acc_bits;
        acc_bits += bw;
        while (acc_bits >= 8u) {
            assert(o < add);
            out[o++] = static_cast<std::uint8_t>(acc & 0xFFu);
            acc >>= 8;
            acc_bits -= 8u;
        }
    }
    // acc_bits is 0 here because n is a multiple of 8 and 8*bw is a whole
    // number of bytes; assert rather than emit a partial byte.
    assert(acc_bits == 0u);
    assert(o == add);
}

// Append `n` values of `bw` bits (1..64) each, LSB-first little-endian.
// `n` must be a multiple of 8. The 64-bit accumulator in pack_le_bits cannot
// take a 64-bit value on top of up to 7 leftover bits, so widths past 56 go
// through a per-byte loop instead. That split is not a performance
// compromise: a delta whose bit width is 57+ is a delta the size of the
// value itself, i.e. the case where delta encoding has already lost.
void pack_le_bits64(const std::uint64_t* vals, std::uint32_t n,
                    std::uint32_t bw, std::vector<std::uint8_t>* dst) noexcept {
    assert(dst != nullptr);
    assert(vals != nullptr || n == 0u);
    assert((n & 7u) == 0u);
    assert(bw >= 1u && bw <= 64u);
    if (n == 0u) return;
    const std::size_t base = dst->size();
    const std::size_t add = (static_cast<std::size_t>(n) / 8u) * bw;
    dst->resize(base + add);
    std::uint8_t* out = dst->data() + base;
    std::memset(out, 0, add);
    const std::uint64_t mask =
        (bw == 64u) ? ~0ull : ((1ull << bw) - 1ull);

    if (bw <= 56u) {
        std::uint64_t acc = 0;
        std::uint32_t acc_bits = 0;
        std::size_t o = 0;
        for (std::uint32_t i = 0; i < n; ++i) {
            acc |= (vals[i] & mask) << acc_bits;
            acc_bits += bw;
            while (acc_bits >= 8u) {
                assert(o < add);
                out[o++] = static_cast<std::uint8_t>(acc & 0xFFu);
                acc >>= 8;
                acc_bits -= 8u;
            }
        }
        assert(acc_bits == 0u);
        assert(o == add);
        return;
    }
    std::uint8_t cur = 0;
    std::uint32_t used = 0;
    std::size_t o = 0;
    for (std::uint32_t i = 0; i < n; ++i) {
        std::uint64_t v = vals[i] & mask;
        std::uint32_t left = bw;
        while (left != 0u) {                    // bounded: <= 9 iterations
            const std::uint32_t take = (8u - used < left) ? (8u - used) : left;
            const std::uint8_t bits =
                static_cast<std::uint8_t>(v & ((1ull << take) - 1ull));
            cur = static_cast<std::uint8_t>(cur | (bits << used));
            v >>= take;
            used += take;
            left -= take;
            if (used == 8u) {
                assert(o < add);
                out[o++] = cur;
                cur = 0;
                used = 0;
            }
        }
    }
    assert(used == 0u);
    assert(o == add);
}

// Minimum bit width that can represent `max_val`. 0 -> 0 (a single-value
// dictionary needs no bits at all, which parquet allows).
std::uint32_t bit_width_for(std::uint32_t max_val) noexcept {
    std::uint32_t bw = 0;
    while (max_val != 0u) { ++bw; max_val >>= 1; }
    assert(bw <= 32u);
    return bw;
}

// Bit width needed for an unsigned 64-bit magnitude.
std::uint32_t bit_width_for64(std::uint64_t v) noexcept {
    std::uint32_t bw = 0;
    while (v != 0u) { ++bw; v >>= 1; }
    assert(bw <= 64u);
    return bw;
}

// ===== RLE / bit-packed hybrid encoder ====================================

// Emits parquet's hybrid stream. Two run kinds, distinguished by the low bit
// of a varint header:
//   RLE run        : header = (run_length << 1) | 0, then the value in
//                    ceil(bw/8) bytes, little-endian.
//   bit-packed run : header = (num_groups << 1) | 1, then num_groups * 8
//                    values packed at bw bits.
//
// A bit-packed run carries whole groups of 8 values, so a run can only be cut
// on an 8-value boundary. That constraint is why this encoder only switches
// into an RLE run when the pending bit-packed buffer is already 8-aligned: a
// repeat that starts mid-group contributes its first few values to the packed
// buffer and is picked up on the next aligned position. The alternative --
// padding a short group -- would inject values that were never in the input.
// The trailing flush DOES pad, which is safe and standard: the reader is
// bounded by num_values and never looks at the pad.
struct RleHybrid {
    std::vector<std::uint8_t>* out;
    std::uint32_t              bw;
    std::vector<std::uint32_t> pending;   // 8-aligned when flushed mid-stream
};

// Flush `pending` as one or more bit-packed runs. `pad` allows the final
// flush to round up to a group boundary.
void rle_flush_packed(RleHybrid* e, bool pad) noexcept {
    assert(e != nullptr);
    assert(e->out != nullptr);
    std::size_t n = e->pending.size();
    if (n == 0u) return;
    if (pad) {
        while ((e->pending.size() & 7u) != 0u) e->pending.push_back(0u);
        n = e->pending.size();
    }
    assert((n & 7u) == 0u);
    const std::uint64_t groups = static_cast<std::uint64_t>(n) / 8u;
    hybrid_varint_emit(e->out, (groups << 1) | 1ull);
    pack_le_bits(e->pending.data(), static_cast<std::uint32_t>(n), e->bw,
                 e->out);
    e->pending.clear();
}

// Emit one RLE run of `count` copies of `val`.
void rle_emit_run(RleHybrid* e, std::uint32_t val,
                  std::uint64_t count) noexcept {
    assert(e != nullptr);
    assert(count > 0u);
    hybrid_varint_emit(e->out, count << 1);
    const std::uint32_t nbytes = (e->bw + 7u) / 8u;
    for (std::uint32_t i = 0; i < nbytes; ++i) {
        e->out->push_back(static_cast<std::uint8_t>((val >> (i * 8u)) & 0xFFu));
    }
}

// Bound on how many values sit in `pending` before a mid-stream flush. Keeps
// the encoder's scratch O(1) regardless of input length. Multiple of 8.
constexpr std::size_t kRleMaxPending = 512u;

// Encode `n` values. Single pass, greedy: a repeat of >= 8 that begins on an
// 8-aligned boundary becomes an RLE run, everything else is bit-packed.
void rle_hybrid_encode(const std::uint32_t* vals, std::size_t n,
                       std::uint32_t bw,
                       std::vector<std::uint8_t>* out) noexcept {
    assert(out != nullptr);
    assert(vals != nullptr || n == 0u);
    if (n == 0u) return;
    if (bw == 0u) {
        // Every value is 0 (a one-entry dictionary). One RLE run of zeros
        // carries the whole page and the value field is zero bytes wide.
        hybrid_varint_emit(out, static_cast<std::uint64_t>(n) << 1);
        return;
    }
    RleHybrid e{out, bw, {}};
    e.pending.reserve(kRleMaxPending);
    std::size_t i = 0;
    while (i < n) {
        // Length of the repeat starting at i, bounded by the input.
        std::size_t run = 1;
        const std::uint32_t v = vals[i];
        while (i + run < n && vals[i + run] == v) ++run;

        const bool aligned = (e.pending.size() & 7u) == 0u;
        if (run >= 8u && aligned) {
            rle_flush_packed(&e, /*pad=*/false);
            rle_emit_run(&e, v, run);
            i += run;
            continue;
        }
        e.pending.push_back(v);
        ++i;
        if (e.pending.size() >= kRleMaxPending) {
            rle_flush_packed(&e, /*pad=*/false);
        }
    }
    rle_flush_packed(&e, /*pad=*/true);
}

// ===== dictionary builder =================================================

// One column chunk's dictionary. Entries are compared by their raw value
// bytes; the PLAIN form the DICTIONARY_PAGE needs is materialised in `plain`
// as entries are added, so no second pass is required.
//
// Bounded by kPwMaxDictEntries and the caller's byte ceiling. Exceeding
// either aborts the build (`overflow`) and the caller writes the chunk PLAIN.
struct DictBuilder {
    std::vector<std::uint32_t> slots;    // open-addressed; 0 = empty, else id+1
    std::uint32_t              mask;
    std::vector<std::uint8_t>  plain;    // PLAIN-encoded distinct values
    std::vector<std::uint32_t> voff;     // per entry: offset of raw value bytes
    std::vector<std::uint32_t> vlen;     // per entry: raw value byte length
    std::uint32_t              count;
    std::uint32_t              max_bytes;
    bool                       overflow;
    bool                       var_len;  // BYTE_ARRAY: plain form has a u32 len
};

// Slot count for a chunk of `n_rows`: the next power of two at or above
// 2 * n_rows, floored at 1024 and capped at 2 * kPwMaxDictEntries. Sizing to
// the chunk rather than to the global ceiling matters -- the ceiling would
// cost an 8 MiB zero-fill per column chunk even for a 100-row chunk. Because
// the table is >= 2x the entry ceiling it can reach, the load factor stays
// <= 0.5 and the linear probe never rehashes.
// The table now STARTS here and grows on demand (see dict_grow), instead of
// being sized from the row count. n_rows is a bound on how many values are
// offered, not on how many are DISTINCT, and it is the distinct count that
// the table has to hold.
std::size_t dict_slot_count(std::int64_t n_rows) noexcept {
    assert(n_rows >= 0);
    const std::size_t cap = static_cast<std::size_t>(kPwMaxDictEntries) * 2u;
    std::size_t want = 1024u;
    if (n_rows > 0 && static_cast<std::uint64_t>(n_rows) * 2u < want) {
        want = 1024u;                       // never below the floor
    }
    assert(want >= 1024u && want <= cap);
    assert((want & (want - 1u)) == 0u);
    return want;
}

void dict_init(DictBuilder* d, std::uint32_t max_bytes, bool var_len,
               std::int64_t n_rows) noexcept {
    assert(d != nullptr);
    assert(max_bytes > 0u);
    const std::size_t slots = dict_slot_count(n_rows);
    // Reuse the buffer across chunks: assign() re-zeroes only what we size to
    // now, and capacity is retained, so a wide schema pays one growth total.
    d->slots.assign(slots, 0u);
    d->mask = static_cast<std::uint32_t>(slots - 1u);
    d->plain.clear();
    d->voff.clear();
    d->vlen.clear();
    d->count = 0;
    d->max_bytes = max_bytes;
    d->overflow = false;
    d->var_len = var_len;
}

// Double the slot table and re-place every existing entry.
//
// The table is sized to what the dictionary HOLDS, not to the row count.
// Sizing it from n_rows made every probe a miss into mostly-empty memory: a
// 400,000-row chunk asks for 1,048,576 slots = 4 MB, so a column whose
// dictionary has 64 entries did 400,000 random accesses into 4 MB of zeros
// and missed cache on essentially every one. MEASURED at 37 ns per intern
// for a lookup that virtually always HITS a 64-entry dictionary.
//
// Growth is amortised O(1) and the ceiling is unchanged, so a genuinely
// large dictionary ends up at exactly the table size it used to start at.
bool dict_grow(DictBuilder* d) noexcept {
    assert(d != nullptr);
    const std::size_t cur = static_cast<std::size_t>(d->mask) + 1u;
    const std::size_t cap = static_cast<std::size_t>(kPwMaxDictEntries) * 2u;
    if (cur >= cap) return false;
    const std::size_t ns = cur * 2u;
    assert((ns & (ns - 1u)) == 0u);
    d->slots.assign(ns, 0u);
    d->mask = static_cast<std::uint32_t>(ns - 1u);
    for (std::uint32_t id = 0; id < d->count; ++id) {
        const std::uint64_t h = pq_xxh64(d->plain.data() + d->voff[id],
                                         d->vlen[id],
                                         0x9E3779B97F4A7C15ull);
        std::uint32_t slot = static_cast<std::uint32_t>(h) & d->mask;
        // Load factor is <= 0.5 by construction, so an empty slot exists.
        for (std::uint32_t p = 0; p <= d->mask; ++p) {
            if (d->slots[slot] == 0u) { d->slots[slot] = id + 1u; break; }
            slot = (slot + 1u) & d->mask;
        }
    }
    return true;
}

// Look up (or insert) `val[0..len)`. Returns the dictionary index, or
// UINT32_MAX once the dictionary has overflowed its bounds.
std::uint32_t dict_intern(DictBuilder* d, const std::uint8_t* val,
                          std::uint32_t len) noexcept {
    assert(d != nullptr);
    assert(val != nullptr || len == 0u);
    if (d->overflow) return 0xFFFFFFFFu;
    // Keep the load factor at or under 0.5 so the linear probe stays short.
    if ((static_cast<std::uint64_t>(d->count) + 1u) * 2u >
            (static_cast<std::uint64_t>(d->mask) + 1u) &&
        !dict_grow(d)) {
        d->overflow = true;
        return 0xFFFFFFFFu;
    }
    const std::uint64_t h = pq_xxh64(val, len, 0x9E3779B97F4A7C15ull);
    std::uint32_t slot = static_cast<std::uint32_t>(h) & d->mask;
    // Open addressing, linear probe. The table is 2x the entry ceiling, so
    // the load factor never exceeds 0.5 and the probe is bounded by the
    // table size; assert that rather than trusting it.
    for (std::uint32_t probe = 0; probe <= d->mask; ++probe) {
        const std::uint32_t s = d->slots[slot];
        if (s == 0u) {
            if (d->count >= kPwMaxDictEntries) { d->overflow = true; return 0xFFFFFFFFu; }
            const std::size_t need =
                d->plain.size() + len + (d->var_len ? 4u : 0u);
            if (need > d->max_bytes) { d->overflow = true; return 0xFFFFFFFFu; }
            if (d->var_len) {
                const std::uint8_t lb[4] = {
                    static_cast<std::uint8_t>(len & 0xFFu),
                    static_cast<std::uint8_t>((len >> 8) & 0xFFu),
                    static_cast<std::uint8_t>((len >> 16) & 0xFFu),
                    static_cast<std::uint8_t>((len >> 24) & 0xFFu),
                };
                d->plain.insert(d->plain.end(), lb, lb + 4);
            }
            const std::uint32_t off = static_cast<std::uint32_t>(d->plain.size());
            d->plain.insert(d->plain.end(), val, val + len);
            d->voff.push_back(off);
            d->vlen.push_back(len);
            const std::uint32_t id = d->count++;
            d->slots[slot] = id + 1u;
            return id;
        }
        const std::uint32_t id = s - 1u;
        if (d->vlen[id] == len &&
            (len == 0u || std::memcmp(d->plain.data() + d->voff[id], val, len) == 0)) {
            return id;
        }
        slot = (slot + 1u) & d->mask;
    }
    // Unreachable at load factor <= 0.5; fail closed rather than loop.
    d->overflow = true;
    return 0xFFFFFFFFu;
}

// ===== per-row value view =================================================

// A pointer/length view of row `r`'s raw value bytes, for whichever physical
// layout the column uses. `nullptr` length-0 is a legal (empty) value.
struct RowVal {
    const std::uint8_t* p;
    std::uint32_t       n;
};

// Fixed-width row view. `elem` is the physical byte width.
inline RowVal row_val_fixed(const BoltColumn& col, std::size_t elem,
                            std::int64_t r) noexcept {
    assert(col.data != nullptr);
    assert(elem > 0u);
    const std::uint8_t* base = static_cast<const std::uint8_t*>(col.data);
    return RowVal{base + static_cast<std::size_t>(r) * elem,
                  static_cast<std::uint32_t>(elem)};
}

// BYTE_ARRAY row view. A Utf8/Binary column legitimately occurs in TWO
// physical layouts and this writer must read both (G2ICE-87):
//
//   * `ColumnFormat::Flat`/`View` -- a StringView array, inline for <= 12
//     bytes else spilled into `str_overflow_base`. What the parquet READER
//     and every compute operator produce.
//   * `ColumnFormat::VarBinary`  -- `dict_child` int32 offsets over a packed
//     byte pool in `data`. What connectors and MarbleDB produce: `txn_batch`
//     serializes Utf8 as VarBinary and `bolt_wire_parse` rebuilds that same
//     format, so EVERY Utf8 column read out of a sealed MarbleDB SST block
//     arrives here as VarBinary.
//
// Reading the second as the first is a silent garbage read, not a crash
// (the hazard `BoltColumn::utf8_at` was added for) -- an int32 offset pair
// reinterpreted as a StringView yields whatever length the offsets happen to
// spell. That is what broke Iceberg tiering: the offsets spelled a >12-byte
// length, the spill base was the packed pool, the dictionary builder bailed,
// PLAIN encoding then refused, and the row group failed AFTER the fixed-width
// columns' pages were already on disk -- leaving a footerless file that
// pyarrow diagnoses as "Parquet magic bytes not found in footer".
//
// Returns n == UINT32_MAX for a malformed column (spilled with no spill base,
// or a format that cannot carry bytes at all), which callers treat as a hard
// error. Never guesses.
inline RowVal row_val_bytes(const BoltColumn& col, std::int64_t r) noexcept {
    assert(r >= 0);
    assert(col.data != nullptr || col.format == ColumnFormat::VarBinary);
    if (col.format == ColumnFormat::VarBinary) {
        if (col.dict_child == nullptr || col.dict_child->data == nullptr) {
            return RowVal{nullptr, 0xFFFFFFFFu};
        }
        const auto* offs = static_cast<const std::int32_t*>(col.dict_child->data);
        const std::int32_t start = offs[r];
        const std::int32_t end   = offs[r + 1];
        if (end < start) return RowVal{nullptr, 0xFFFFFFFFu};
        const std::uint32_t len = static_cast<std::uint32_t>(end - start);
        if (len == 0u) return RowVal{nullptr, 0u};
        if (col.data == nullptr) return RowVal{nullptr, 0xFFFFFFFFu};
        return RowVal{static_cast<const std::uint8_t*>(col.data) + start, len};
    }
    if (col.format != ColumnFormat::Flat && col.format != ColumnFormat::View) {
        return RowVal{nullptr, 0xFFFFFFFFu};
    }
    const auto* sv = static_cast<const StringView*>(col.data);
    const auto* spill = static_cast<const std::uint8_t*>(col.str_overflow_base);
    const std::uint32_t len = sv[r].length;
    if (len == 0u) return RowVal{nullptr, 0u};
    if (len <= 12u) {
        return RowVal{reinterpret_cast<const std::uint8_t*>(&sv[r].prefix[0]),
                      len};
    }
    if (spill == nullptr) return RowVal{nullptr, 0xFFFFFFFFu};
    return RowVal{spill + sv[r].ref.offset, len};
}

// Decimal128 on disk is FLBA(16) BIG-endian two's complement; bolt stores it
// little-endian in memory. One value's worth of byte-reversal.
inline void decimal_to_be(const std::uint8_t* le, std::uint8_t* be) noexcept {
    assert(le != nullptr && be != nullptr);
    for (std::size_t k = 0; k < 16u; ++k) be[k] = le[15u - k];
}

// ===== page planning ======================================================

// Physical PLAIN byte width for a fixed-width bolt type, or 0 if the type is
// not fixed width (BYTE_ARRAY) or not supported.
std::size_t plain_elem_width(BoltType t) noexcept {
    switch (t) {
        case BoltType::Int32:
        case BoltType::Date32:
        case BoltType::Float32:    return 4u;
        case BoltType::Int64:
        case BoltType::Timestamp:
        case BoltType::Float64:    return 8u;
        case BoltType::Decimal128: return 16u;
        default:                   return 0u;
    }
}

// Exclusive end row for a data page starting at `r0`, such that the page's
// uncompressed VALUE bytes stay within `budget`. Always returns at least
// r0 + 1 so a single oversized row still forms a page instead of stalling.
//
// `bits_per_value` > 0 selects index sizing (dictionary-encoded page); the
// value bytes are then bits_per_value/8 per non-null row regardless of type.
std::int64_t plan_page_rows(const BoltColumn& col, BoltType t,
                            std::int64_t r0, std::int64_t r_end,
                            const std::uint8_t* def_bits, bool nullable,
                            std::uint64_t budget,
                            std::uint32_t bits_per_value) noexcept {
    assert(r0 < r_end);
    assert(def_bits != nullptr || !nullable);
    assert(budget > 0u);

    // Dictionary indices: constant cost per non-null row, so the row count is
    // closed-form once the non-null count is known. Scan def bits branch-free.
    if (bits_per_value != 0u) {
        const std::uint64_t cap_vals = (budget * 8u) / bits_per_value;
        if (cap_vals == 0u) return r0 + 1;
        std::uint64_t seen = 0;
        for (std::int64_t r = r0; r < r_end; ++r) {
            seen += nullable ? static_cast<std::uint64_t>(def_bits[r] != 0u) : 1u;
            if (seen > cap_vals) return (r > r0) ? r : (r0 + 1);
        }
        return r_end;
    }

    const std::size_t elem = plain_elem_width(t);
    if (t == BoltType::Bool) {
        // 1 bit per non-null value.
        const std::uint64_t cap_vals = budget * 8u;
        std::uint64_t seen = 0;
        for (std::int64_t r = r0; r < r_end; ++r) {
            seen += nullable ? static_cast<std::uint64_t>(def_bits[r] != 0u) : 1u;
            if (seen > cap_vals) return (r > r0) ? r : (r0 + 1);
        }
        return r_end;
    }
    if (elem != 0u) {
        const std::uint64_t cap_vals = budget / elem;
        if (cap_vals == 0u) return r0 + 1;
        std::uint64_t seen = 0;
        for (std::int64_t r = r0; r < r_end; ++r) {
            seen += nullable ? static_cast<std::uint64_t>(def_bits[r] != 0u) : 1u;
            if (seen > cap_vals) return (r > r0) ? r : (r0 + 1);
        }
        return r_end;
    }
    // BYTE_ARRAY: cost is data-dependent, so accumulate.
    assert(t == BoltType::Utf8 || t == BoltType::Binary);
    std::uint64_t bytes = 0;
    for (std::int64_t r = r0; r < r_end; ++r) {
        const bool valid = (!nullable) || def_bits[r] != 0u;
        if (valid) {
            const RowVal v = row_val_bytes(col, r);
            bytes += 4u + (v.n == 0xFFFFFFFFu ? 0u : v.n);
        }
        if (bytes > budget) return (r > r0) ? r : (r0 + 1);
    }
    return r_end;
}

// ===== BYTE_STREAM_SPLIT (encoding 9) =====================================

// A pure transpose: for a width-W type the page holds all byte 0s, then all
// byte 1s, and so on. The inverse of the reader's decode_byte_stream_split,
// and stream-major for the same reason -- each pass writes one output stream
// sequentially rather than scattering W streams per value.
//
// Decimal128 is byte-reversed to big-endian FIRST, because the split is
// defined over the value's ON-DISK bytes; splitting the little-endian
// in-memory form would transpose the wrong bytes and decode to garbage that
// still has the right length.
bool encode_byte_stream_split(const BoltColumn& col, BoltType t,
                              std::int64_t r0, std::int64_t r1,
                              const std::uint8_t* def_bits, bool nullable,
                              std::vector<std::uint8_t>* dst) noexcept {
    assert(dst != nullptr);
    assert(r0 <= r1);
    const std::size_t w = plain_elem_width(t);
    if (w == 0u || w > 16u) return false;
    const bool is_dec = (t == BoltType::Decimal128);
    const auto* base = static_cast<const std::uint8_t*>(col.data);
    if (base == nullptr && r0 != r1) return false;

    // Count non-null values first: the stream length is n, and a stream can
    // only be laid down once n is known.
    std::int64_t n = 0;
    for (std::int64_t r = r0; r < r1; ++r) {
        n += (!nullable) ? 1 : (def_bits[r] != 0u);
    }
    if (n == 0) return true;
    const std::size_t start = dst->size();
    dst->resize(start + static_cast<std::size_t>(n) * w);
    std::uint8_t* out = dst->data() + start;

    std::int64_t i = 0;
    for (std::int64_t r = r0; r < r1; ++r) {
        if (nullable && def_bits[r] == 0u) continue;
        const std::uint8_t* src = base + static_cast<std::size_t>(r) * w;
        std::uint8_t be[16];
        if (is_dec) { decimal_to_be(src, be); src = be; }
        for (std::size_t j = 0; j < w; ++j) {
            out[static_cast<std::size_t>(j) * static_cast<std::size_t>(n) +
                static_cast<std::size_t>(i)] = src[j];
        }
        ++i;
    }
    assert(i == n);
    return true;
}

// ===== DELTA_BINARY_PACKED (encoding 5) ===================================

// Spec shape:
//   header : uleb(block_size) uleb(miniblocks_per_block) uleb(total_values)
//            zigzag(first_value)
//   block  : zigzag(min_delta) bitwidth[miniblocks_per_block] miniblock*
//
// block_size 128 with 4 miniblocks gives 32 values per miniblock, satisfying
// both the spec ("multiple of 128" / "multiple of 32") and the reader's own
// checks.
constexpr std::uint32_t kDeltaBlockValues = 128u;
constexpr std::uint32_t kDeltaMiniBlocks  = 4u;
constexpr std::uint32_t kDeltaValsPerMini = kDeltaBlockValues / kDeltaMiniBlocks;

void delta_put_uleb(std::vector<std::uint8_t>* dst, std::uint64_t v) noexcept {
    assert(dst != nullptr);
    (void)hybrid_varint_emit(dst, v);
}

void delta_put_zigzag(std::vector<std::uint8_t>* dst, std::int64_t v) noexcept {
    assert(dst != nullptr);
    const std::uint64_t u = (static_cast<std::uint64_t>(v) << 1) ^
                            static_cast<std::uint64_t>(v >> 63);
    (void)hybrid_varint_emit(dst, u);
}

// Encode `n` int64 values. All arithmetic is done in uint64 so that a delta
// which exceeds the int64 range WRAPS rather than being undefined -- the
// decoder's `cur += min_delta + packed` wraps identically, so the round trip
// is exact for the full int64 domain. parquet-mr relies on the same property.
//
// Scratch is one caller-owned block of kDeltaBlockValues entries, so this
// allocates nothing per call beyond the output.
bool delta_encode_ints(const std::int64_t* vals, std::uint32_t n,
                       std::vector<std::uint8_t>* dst) noexcept {
    assert(dst != nullptr);
    assert(vals != nullptr || n == 0u);
    delta_put_uleb(dst, kDeltaBlockValues);
    delta_put_uleb(dst, kDeltaMiniBlocks);
    delta_put_uleb(dst, n);
    delta_put_zigzag(dst, (n > 0u) ? vals[0] : 0);
    if (n <= 1u) return true;

    std::uint64_t packed[kDeltaBlockValues];
    std::uint64_t deltas[kDeltaBlockValues];
    std::uint32_t done = 0;                       // deltas emitted so far
    const std::uint32_t n_deltas = n - 1u;
    while (done < n_deltas) {                     // bounded by n_deltas
        const std::uint32_t take =
            (n_deltas - done < kDeltaBlockValues) ? (n_deltas - done)
                                                  : kDeltaBlockValues;
        std::int64_t min_delta = 0;
        for (std::uint32_t i = 0; i < take; ++i) {
            const std::uint32_t idx = done + i + 1u;
            const std::uint64_t d = static_cast<std::uint64_t>(vals[idx]) -
                                    static_cast<std::uint64_t>(vals[idx - 1u]);
            deltas[i] = d;
            const std::int64_t sd = static_cast<std::int64_t>(d);
            if (i == 0u || sd < min_delta) min_delta = sd;
        }
        delta_put_zigzag(dst, min_delta);

        // Bit width per miniblock, then the widths array, then the data. The
        // widths array is ALWAYS kDeltaMiniBlocks bytes even when the block
        // is partial -- the reader consumes it unconditionally -- but data is
        // written only for the miniblocks it will actually read.
        const std::uint32_t used_mini =
            (take + kDeltaValsPerMini - 1u) / kDeltaValsPerMini;
        assert(used_mini >= 1u && used_mini <= kDeltaMiniBlocks);
        std::uint8_t widths[kDeltaMiniBlocks];
        std::memset(widths, 0, sizeof(widths));
        for (std::uint32_t m = 0; m < used_mini; ++m) {
            std::uint64_t maxv = 0;
            for (std::uint32_t k = 0; k < kDeltaValsPerMini; ++k) {
                const std::uint32_t i = m * kDeltaValsPerMini + k;
                // Pad the tail of a partial miniblock with min_delta, which
                // packs to 0 and so costs no bits.
                const std::uint64_t p =
                    (i < take) ? (deltas[i] - static_cast<std::uint64_t>(min_delta))
                               : 0ull;
                packed[i] = p;
                if (p > maxv) maxv = p;
            }
            widths[m] = static_cast<std::uint8_t>(bit_width_for64(maxv));
        }
        dst->insert(dst->end(), widths, widths + kDeltaMiniBlocks);
        for (std::uint32_t m = 0; m < used_mini; ++m) {
            if (widths[m] == 0u) continue;        // all deltas equal min_delta
            pack_le_bits64(packed + m * kDeltaValsPerMini, kDeltaValsPerMini,
                           widths[m], dst);
        }
        done += take;
    }
    return true;
}

// Gather rows [r0, r1)'s non-null values as int64, widening INT32-physical
// types. Returns the count, or -1 if the type is not delta-encodable.
std::int64_t gather_ints(const BoltColumn& col, BoltType t,
                         std::int64_t r0, std::int64_t r1,
                         const std::uint8_t* def_bits, bool nullable,
                         std::vector<std::int64_t>* out) noexcept {
    assert(out != nullptr);
    assert(r0 <= r1);
    out->clear();
    const std::size_t w = plain_elem_width(t);
    if (w != 4u && w != 8u) return -1;
    if (t == BoltType::Float32 || t == BoltType::Float64) return -1;
    if (col.data == nullptr && r0 != r1) return -1;
    out->reserve(static_cast<std::size_t>(r1 - r0));
    for (std::int64_t r = r0; r < r1; ++r) {
        if (nullable && def_bits[r] == 0u) continue;
        if (w == 4u) {
            std::int32_t v;
            std::memcpy(&v, static_cast<const std::uint8_t*>(col.data) + r * 4, 4);
            out->push_back(v);
        } else {
            std::int64_t v;
            std::memcpy(&v, static_cast<const std::uint8_t*>(col.data) + r * 8, 8);
            out->push_back(v);
        }
    }
    return static_cast<std::int64_t>(out->size());
}

// ===== DELTA_LENGTH_BYTE_ARRAY (6) and DELTA_BYTE_ARRAY (7) ===============

// DELTA_LENGTH_BYTE_ARRAY: a delta-packed length array, then every value's
// bytes concatenated with no per-value length prefix.
bool encode_delta_length_byte_array(const BoltColumn& col,
                                    std::int64_t r0, std::int64_t r1,
                                    const std::uint8_t* def_bits, bool nullable,
                                    std::vector<std::int64_t>* lens_scratch,
                                    std::vector<std::uint8_t>* dst) noexcept {
    assert(dst != nullptr && lens_scratch != nullptr);
    assert(r0 <= r1);
    lens_scratch->clear();
    for (std::int64_t r = r0; r < r1; ++r) {
        if (nullable && def_bits[r] == 0u) continue;
        const RowVal v = row_val_bytes(col, r);
        if (v.n == 0xFFFFFFFFu) return false;
        lens_scratch->push_back(static_cast<std::int64_t>(v.n));
    }
    if (!delta_encode_ints(lens_scratch->data(),
                           static_cast<std::uint32_t>(lens_scratch->size()),
                           dst)) {
        return false;
    }
    for (std::int64_t r = r0; r < r1; ++r) {
        if (nullable && def_bits[r] == 0u) continue;
        const RowVal v = row_val_bytes(col, r);
        if (v.n == 0xFFFFFFFFu) return false;
        if (v.n != 0u) dst->insert(dst->end(), v.p, v.p + v.n);
    }
    return true;
}

// DELTA_BYTE_ARRAY: front coding. Value i reuses the first prefix[i] bytes of
// value i-1, so only the suffix is stored. Layout is a delta-packed prefix
// length array, then a delta-packed SUFFIX length array, then the suffix
// bytes -- note that is two separate delta blocks, not a nested
// DELTA_LENGTH_BYTE_ARRAY, even though the tail looks identical.
//
// The prefix is capped at the PREDECESSOR's length: a value cannot borrow
// bytes its predecessor does not have, and the reader rejects a stream that
// claims otherwise.
bool encode_delta_byte_array(const BoltColumn& col,
                             std::int64_t r0, std::int64_t r1,
                             const std::uint8_t* def_bits, bool nullable,
                             std::vector<std::int64_t>* pre_scratch,
                             std::vector<std::int64_t>* suf_scratch,
                             std::vector<std::uint8_t>* bytes_scratch,
                             std::vector<std::uint8_t>* dst) noexcept {
    assert(dst != nullptr);
    assert(pre_scratch != nullptr && suf_scratch != nullptr);
    assert(bytes_scratch != nullptr);
    pre_scratch->clear();
    suf_scratch->clear();
    bytes_scratch->clear();
    const std::uint8_t* prev_p = nullptr;
    std::uint32_t prev_n = 0;
    for (std::int64_t r = r0; r < r1; ++r) {
        if (nullable && def_bits[r] == 0u) continue;
        const RowVal v = row_val_bytes(col, r);
        if (v.n == 0xFFFFFFFFu) return false;
        const std::uint32_t cap = (v.n < prev_n) ? v.n : prev_n;
        std::uint32_t p = 0;
        while (p < cap && v.p[p] == prev_p[p]) ++p;   // bounded by cap
        pre_scratch->push_back(p);
        suf_scratch->push_back(static_cast<std::int64_t>(v.n - p));
        if (v.n > p) {
            bytes_scratch->insert(bytes_scratch->end(), v.p + p, v.p + v.n);
        }
        prev_p = v.p;
        prev_n = v.n;
    }
    const std::uint32_t n = static_cast<std::uint32_t>(pre_scratch->size());
    if (!delta_encode_ints(pre_scratch->data(), n, dst)) return false;
    if (!delta_encode_ints(suf_scratch->data(), n, dst)) return false;
    dst->insert(dst->end(), bytes_scratch->begin(), bytes_scratch->end());
    return true;
}
