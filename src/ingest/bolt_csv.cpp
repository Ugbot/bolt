// ---------------------------------------------------------------------------
// bolt::ingest — CSV parser into BoltBatch columns.
//
// Two-pass algorithm:
//   1) SWAR-scan the buffer 8 bytes at a time, counting newlines to size
//      the column allocations exactly. Skip the header line if requested.
//   2) Walk the buffer row-by-row, locating each delimiter / newline with
//      bolt_swar_find_byte_u64 and dispatching per-column-type.
//
// No heap outside the caller arena. No exceptions. Tiger Style: ≥2 asserts
// per hot-path function, ≤70 LOC per function, helpers split freely.
// ---------------------------------------------------------------------------
#include "bolt/ingest/bolt_csv.h"
#include "bolt/parse/bolt_ascii.h"
#include "bolt/kernels/bolt_date.h"
#include "bolt/kernels/bolt_decimal.h"

#include <cassert>
#include <cstdint>
#include <cstring>

namespace bolt {
namespace ingest {

// ---------------------------------------------------------------------------
// Pass 1 — count physical lines (excluding optional header).
// ---------------------------------------------------------------------------

// Count '\n' bytes in [buf, buf+len) using 8-byte SWAR chunks.
static size_t count_newlines(const char* BOLT_RESTRICT buf, size_t len) noexcept {
    assert(buf != nullptr || len == 0);
    assert(len <= static_cast<size_t>(INT64_MAX));

    size_t count = 0;
    size_t i = 0;
    while (i + 8 <= len) {
        uint64_t word;
        std::memcpy(&word, buf + i, 8);
        // popcount over the 0x80-per-match mask gives the hit count.
        const uint64_t hits = bolt_swar_has_byte_u64(word, '\n');
        count += static_cast<size_t>(bolt_popcount64(hits >> 7));
        i += 8;
    }
    for (; i < len; ++i) {
        if (buf[i] == '\n') ++count;
    }
    return count;
}

// Row count = newlines + (1 if buffer has trailing content after the
// last '\n' or no '\n' at all). Header (if any) deducted by caller.
static int64_t compute_row_count(const char* buf, size_t len,
                                 bool has_header) noexcept {
    assert(buf != nullptr || len == 0);
    assert(len <= static_cast<size_t>(INT64_MAX));

    size_t nl = count_newlines(buf, len);
    size_t rows = nl;
    if (len > 0 && buf[len - 1] != '\n') rows += 1;
    if (has_header) {
        if (rows == 0) return 0;
        rows -= 1;
    }
    return static_cast<int64_t>(rows);
}

// Find the next '\n' starting at `pos` in [buf, buf+len). Returns index of
// the newline or `len` if none. Uses 8-byte SWAR chunks.
static size_t find_newline(const char* BOLT_RESTRICT buf, size_t len,
                           size_t pos) noexcept {
    assert(buf != nullptr || len == 0);
    assert(pos <= len);

    while (pos + 8 <= len) {
        uint64_t word;
        std::memcpy(&word, buf + pos, 8);
        const int lane = bolt_swar_find_byte_u64(word, '\n');
        if (lane >= 0) return pos + static_cast<size_t>(lane);
        pos += 8;
    }
    for (; pos < len; ++pos) {
        if (buf[pos] == '\n') return pos;
    }
    return len;
}

// Find the next `delim` or '\n', whichever is earlier, starting at `pos`.
// Returns the index of the first hit or `len` if neither is found.
static size_t find_delim_or_newline(const char* BOLT_RESTRICT buf, size_t len,
                                    size_t pos, char delim) noexcept {
    assert(buf != nullptr || len == 0);
    assert(pos <= len);
    assert(delim != '\n');

    const uint8_t db = static_cast<uint8_t>(delim);
    while (pos + 8 <= len) {
        uint64_t word;
        std::memcpy(&word, buf + pos, 8);
        const uint64_t h_delim = bolt_swar_has_byte_u64(word, db);
        const uint64_t h_nl    = bolt_swar_has_byte_u64(word, '\n');
        const uint64_t hits    = h_delim | h_nl;
        if (hits) {
            const int lane = static_cast<int>(bolt_ctz64(hits) >> 3);
            return pos + static_cast<size_t>(lane);
        }
        pos += 8;
    }
    for (; pos < len; ++pos) {
        const char c = buf[pos];
        if (c == delim || c == '\n') return pos;
    }
    return len;
}

// ---------------------------------------------------------------------------
// Utf8 inline-vs-spilled writer.
// ---------------------------------------------------------------------------

// Build a StringView over the byte range [p, p+len). If len <= 12 the view
// is fully inline. Otherwise the bytes are copied into the shared overflow
// buffer (bump-allocated cursor) and the ref fields point into it.
//
// Returns false on overflow-buffer OOM.
static bool make_view(const char* p, uint32_t len,
                      char* overflow_buf, size_t overflow_cap,
                      size_t* overflow_cursor,
                      StringView* out) noexcept {
    assert(p != nullptr || len == 0);
    assert(out != nullptr);
    assert(overflow_cursor != nullptr);

    std::memset(out, 0, sizeof(*out));
    out->length = len;
    if (len == 0) return true;

    if (len <= 12) {
        const uint32_t pfx = (len < 4) ? len : 4u;
        std::memcpy(out->prefix, p, pfx);
        if (len > 4) std::memcpy(out->inline_data, p + 4, len - 4);
        return true;
    }
    // Spilled.
    std::memcpy(out->prefix, p, 4);
    size_t cur = *overflow_cursor;
    if (cur + len > overflow_cap) return false;
    std::memcpy(overflow_buf + cur, p, len);
    out->ref.buf_idx = 0;
    out->ref.offset  = static_cast<uint32_t>(cur);
    *overflow_cursor = cur + len;
    return true;
}

// ---------------------------------------------------------------------------
// Numeric field dispatch.
// ---------------------------------------------------------------------------

// parse_int10th requires len in [3..6]. Empty or shorter fields return 0.
static int32_t field_to_int32(const char* p, int len) noexcept {
    assert(len >= 0);
    assert(p != nullptr || len == 0);
    if (len < 3 || len > 6) return 0;
    return bolt::parse::parse_int10th(p, len);
}

// parse_int requires len in [1..18]. Empty fields return 0.
static int64_t field_to_int64(const char* p, int len) noexcept {
    assert(len >= 0);
    assert(p != nullptr || len == 0);
    if (len < 1 || len > 18) return 0;
    return bolt::parse::parse_int(p, len);
}

// ---------------------------------------------------------------------------
// Date32 field — ISO YYYY-MM-DD (10 bytes, fixed width).
// ---------------------------------------------------------------------------

// Validate a 4-digit ASCII year, write into `*out`. Branch-free digit fold.
// Returns false if any character is not [0-9].
static bool fold_year4(const char* p, int32_t* out) noexcept {
    assert(p   != nullptr);
    assert(out != nullptr);
    uint32_t d0 = static_cast<uint8_t>(p[0]) - '0';
    uint32_t d1 = static_cast<uint8_t>(p[1]) - '0';
    uint32_t d2 = static_cast<uint8_t>(p[2]) - '0';
    uint32_t d3 = static_cast<uint8_t>(p[3]) - '0';
    uint32_t bad = (d0 | d1 | d2 | d3) >> 4;  // any digit > 9 sets a high bit
    *out = static_cast<int32_t>(d0 * 1000u + d1 * 100u + d2 * 10u + d3);
    return bad == 0u;
}

// Validate a 2-digit ASCII run, write into `*out`. Branch-free digit fold.
static bool fold_two(const char* p, uint32_t* out) noexcept {
    assert(p   != nullptr);
    assert(out != nullptr);
    uint32_t d0 = static_cast<uint8_t>(p[0]) - '0';
    uint32_t d1 = static_cast<uint8_t>(p[1]) - '0';
    uint32_t bad = (d0 | d1) >> 4;
    *out = d0 * 10u + d1;
    return bad == 0u;
}

// Parse "YYYY-MM-DD" -> int32 days since 1970-01-01. Returns false on bad
// format / out-of-range components. `flen` is the post-CRLF-strip length.
static bool field_to_date32(const char* p, size_t flen, int32_t* out) noexcept {
    assert(out != nullptr);
    assert(p != nullptr || flen == 0);
    if (flen != 10) return false;
    if (p[4] != '-' || p[7] != '-') return false;
    int32_t  y = 0;
    uint32_t m = 0, d = 0;
    if (!fold_year4(p, &y))        return false;
    if (!fold_two(p + 5, &m))      return false;
    if (!fold_two(p + 8, &d))      return false;
    if (m < 1u || m > 12u)         return false;
    if (d < 1u || d > 31u)         return false;
    int64_t z = bolt::kernels::date::days_from_civil(y, m, d);
    if (z < INT32_MIN || z > INT32_MAX) return false;
    *out = static_cast<int32_t>(z);
    return true;
}

// ---------------------------------------------------------------------------
// Decimal128 field — sign + integer + optional '.' + fractional digits.
// ---------------------------------------------------------------------------

using bolt::kernels::decimal::Decimal128;
using bolt::kernels::decimal::d128_neg;
using bolt::kernels::decimal::mul_u64_u64;

// Unsigned 128-bit multiply-by-10-add-digit. Branch-free on the hot path.
BOLT_FORCE_INLINE void d128u_mul10_add(Decimal128* v, uint32_t digit) noexcept {
    assert(v != nullptr);
    assert(digit < 10u);
    const uint64_t lo = static_cast<uint64_t>(v->lo);
    const uint64_t hi = static_cast<uint64_t>(v->hi);
    uint64_t prod_lo, prod_hi;
    mul_u64_u64(lo, 10ull, &prod_lo, &prod_hi);
    const uint64_t new_lo  = prod_lo + static_cast<uint64_t>(digit);
    const uint64_t carry   = (new_lo < prod_lo) ? 1ull : 0ull;
    const uint64_t new_hi  = hi * 10ull + prod_hi + carry;
    v->lo = static_cast<int64_t>(new_lo);
    v->hi = static_cast<int64_t>(new_hi);
}

// Scan a run of decimal digits [p, end). Folds into `*acc`. Returns the
// number of digits consumed. Stops at first non-digit. Branch-free per byte.
static size_t fold_digits(const char* p, const char* end,
                          Decimal128* acc) noexcept {
    assert(acc != nullptr);
    assert(end >= p);
    const char* q = p;
    while (q < end) {
        const uint32_t c = static_cast<uint8_t>(*q) - static_cast<uint32_t>('0');
        if (c > 9u) break;
        d128u_mul10_add(acc, c);
        ++q;
    }
    return static_cast<size_t>(q - p);
}

// Multiply unsigned 128-bit by a small constant (10..10^18). Branch-free.
BOLT_FORCE_INLINE void d128u_mul_small(Decimal128* v, uint64_t k) noexcept {
    assert(v != nullptr);
    assert(k > 0);
    const uint64_t lo = static_cast<uint64_t>(v->lo);
    const uint64_t hi = static_cast<uint64_t>(v->hi);
    uint64_t prod_lo, prod_hi;
    mul_u64_u64(lo, k, &prod_lo, &prod_hi);
    v->lo = static_cast<int64_t>(prod_lo);
    v->hi = static_cast<int64_t>(hi * k + prod_hi);
}

// Power-of-ten table up to 10^18 (fits uint64). Higher powers chain.
static constexpr uint64_t kPow10U64[19] = {
    1ull, 10ull, 100ull, 1000ull, 10000ull, 100000ull, 1000000ull,
    10000000ull, 100000000ull, 1000000000ull, 10000000000ull,
    100000000000ull, 1000000000000ull, 10000000000000ull,
    100000000000000ull, 1000000000000000ull, 10000000000000000ull,
    100000000000000000ull, 1000000000000000000ull,
};

// Multiply unsigned 128-bit by 10^n. n in [0..38]. Chains 10^18 steps.
static void d128u_mul_pow10(Decimal128* v, uint32_t n) noexcept {
    assert(v != nullptr);
    assert(n <= 38u);
    while (n >= 18u) {
        d128u_mul_small(v, kPow10U64[18]);
        n -= 18u;
    }
    if (n > 0u) d128u_mul_small(v, kPow10U64[n]);
}

// Parse a decimal string into a Decimal128 at the column's declared scale.
// Returns false on malformed input or (strict_scale && extra fractional digits).
// Empty field => zero.
static bool field_to_decimal128(const char* p, size_t flen, uint8_t out_scale,
                                bool strict_scale, Decimal128* out) noexcept {
    assert(out != nullptr);
    assert(p != nullptr || flen == 0);
    out->lo = 0;
    out->hi = 0;
    if (flen == 0) return true;
    if (out_scale > 38u) return false;

    const char* q   = p;
    const char* end = p + flen;
    bool neg = false;
    if (q < end && (*q == '-' || *q == '+')) { neg = (*q == '-'); ++q; }

    Decimal128 acc{0, 0};
    const size_t int_digits  = fold_digits(q, end, &acc);
    q += int_digits;
    size_t frac_digits = 0;
    size_t frac_have   = 0;
    if (q < end && *q == '.') {
        ++q;
        // Fold up to out_scale fractional digits into acc.
        const char* fs = q;
        const char* fe = end;
        const char* limit = (static_cast<size_t>(fe - fs) > out_scale)
                          ? fs + out_scale : fe;
        frac_have = fold_digits(fs, limit, &acc);
        q += frac_have;
        // Skip any remaining fractional digits (truncate) and count them.
        frac_digits = frac_have;
        while (q < end) {
            const uint32_t c = static_cast<uint8_t>(*q) - static_cast<uint32_t>('0');
            if (c > 9u) break;
            ++frac_digits;
            ++q;
        }
    }
    if (q != end)                       return false;  // garbage after digits
    if (int_digits == 0 && frac_have == 0 && frac_digits == 0) {
        // Sign with no digits at all is a parse error.
        if (neg || (p < end && *p == '+')) return false;
    }
    if (strict_scale && frac_digits > out_scale) return false;

    // Pad with zeros if we have fewer fractional digits than the column scale.
    const uint32_t pad = static_cast<uint32_t>(out_scale) -
                         static_cast<uint32_t>(frac_have);
    if (pad > 0u) d128u_mul_pow10(&acc, pad);
    *out = neg ? d128_neg(acc) : acc;
    return true;
}

// ---------------------------------------------------------------------------
// Column setup.
// ---------------------------------------------------------------------------

// Returns false on unsupported type or arena OOM.
static bool alloc_columns(const CsvSchema& schema, int64_t row_count,
                          Arena* arena, BoltColumn* out) noexcept {
    assert(arena != nullptr);
    assert(out != nullptr);

    for (uint32_t c = 0; c < schema.num_cols; ++c) {
        const BoltType t = schema.col_types[c];
        if (t != BoltType::Int32 && t != BoltType::Int64 && t != BoltType::Utf8 &&
            t != BoltType::Date32 && t != BoltType::Decimal128) {
            return false;
        }
        if (row_count == 0) {
            out[c] = BoltColumn::make_empty();
            out[c].type = t;
            out[c].format = ColumnFormat::Flat;
            out[c].type_size_bytes = static_cast<uint16_t>(bolt::type_size(t));
            out[c].arena = arena;
            continue;
        }
        out[c] = BoltColumn::make_flat_alloc(row_count, t, arena);
        if (!out[c].data) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Row parser.
// ---------------------------------------------------------------------------

// Parse a single field [f_start, f_end) into column `c` at row index `row`.
// Strips a trailing '\r' if present (so CRLF line terminators work).
// Returns false only on Utf8 overflow-buffer OOM.
static bool write_field(const CsvSchema& schema, uint32_t c, int64_t row,
                        const char* f_start, const char* f_end,
                        BoltColumn* cols,
                        char* overflow_buf, size_t overflow_cap,
                        size_t* overflow_cursor) noexcept {
    assert(f_end >= f_start);
    assert(c < schema.num_cols);

    size_t flen = static_cast<size_t>(f_end - f_start);
    if (flen > 0 && f_start[flen - 1] == '\r') flen -= 1;  // CRLF strip

    const BoltType t = schema.col_types[c];
    if (t == BoltType::Int32) {
        int32_t v = field_to_int32(f_start, static_cast<int>(flen));
        static_cast<int32_t*>(cols[c].data)[row] = v;
    } else if (t == BoltType::Int64) {
        int64_t v = field_to_int64(f_start, static_cast<int>(flen));
        static_cast<int64_t*>(cols[c].data)[row] = v;
    } else if (t == BoltType::Date32) {
        int32_t v = 0;
        if (flen != 0 && !field_to_date32(f_start, flen, &v)) return false;
        static_cast<int32_t*>(cols[c].data)[row] = v;
    } else if (t == BoltType::Decimal128) {
        Decimal128 v{0, 0};
        if (!field_to_decimal128(f_start, flen,
                                 schema.col_scales[c],
                                 schema.strict_decimal_scale, &v)) {
            return false;
        }
        static_cast<Decimal128*>(cols[c].data)[row] = v;
    } else {
        assert(t == BoltType::Utf8);
        StringView sv;
        if (!make_view(f_start, static_cast<uint32_t>(flen),
                       overflow_buf, overflow_cap, overflow_cursor, &sv)) {
            return false;
        }
        static_cast<StringView*>(cols[c].data)[row] = sv;
    }
    return true;
}

// Parse one line [line_start, line_end) into row `row` across all columns.
// Returns false on schema-column-count mismatch or Utf8 OOM.
static bool parse_line(const CsvSchema& schema, int64_t row,
                       const char* BOLT_RESTRICT buf, size_t buf_len,
                       size_t line_start, size_t line_end,
                       BoltColumn* cols,
                       char* overflow_buf, size_t overflow_cap,
                       size_t* overflow_cursor) noexcept {
    (void)buf_len;
    assert(line_start <= line_end);
    assert(line_end <= buf_len);

    size_t pos = line_start;
    uint32_t c = 0;
    while (c < schema.num_cols) {
        size_t hit = find_delim_or_newline(buf, line_end, pos, schema.delimiter);
        // Clamp — line_end is the actual terminator (or buf_len).
        if (hit > line_end) hit = line_end;

        if (!write_field(schema, c, row,
                         buf + pos, buf + hit,
                         cols, overflow_buf, overflow_cap, overflow_cursor)) {
            return false;
        }
        ++c;

        if (hit >= line_end) {
            // End of line consumed. Must have filled exactly all columns.
            return c == schema.num_cols;
        }
        // Consumed a delimiter; step past it.
        pos = hit + 1;
    }
    // Ran out of schema columns before the newline — extra fields on this
    // row. That's a mismatch we refuse to paper over.
    return pos >= line_end;
}

// ---------------------------------------------------------------------------
// Public entry point.
// ---------------------------------------------------------------------------

bool parse_csv(const char* BOLT_RESTRICT buf, size_t buf_len,
               const CsvSchema& schema,
               Arena*           arena,
               BoltBatch*       out_batch) noexcept {
    assert(arena != nullptr);
    assert(out_batch != nullptr);

    if (schema.num_cols == 0 || schema.num_cols > kCsvMaxCols) return false;
    if (schema.delimiter == '\n' || schema.delimiter == '\r') return false;

    const int64_t row_count = compute_row_count(buf, buf_len, schema.has_header);

    BoltBatch::init_empty(out_batch);
    out_batch->arena    = arena;
    out_batch->num_cols = schema.num_cols;
    out_batch->num_rows = row_count;

    BoltColumn* cols = out_batch->columns[out_batch->read_epoch];
    if (!alloc_columns(schema, row_count, arena, cols)) return false;

    // Overflow buffer for spilled Utf8 fields. Upper bound: buf_len.
    // Allocate only if at least one Utf8 column exists and we have rows.
    char* overflow_buf = nullptr;
    size_t overflow_cap = 0;
    size_t overflow_cursor = 0;
    bool any_utf8 = false;
    for (uint32_t c = 0; c < schema.num_cols; ++c) {
        if (schema.col_types[c] == BoltType::Utf8) { any_utf8 = true; break; }
    }
    if (any_utf8 && buf_len > 0 && row_count > 0) {
        overflow_cap = buf_len;
        overflow_buf = static_cast<char*>(arena->allocate(overflow_cap, 1));
        if (!overflow_buf) return false;
        // Expose the spill base on every Utf8 column so downstream operators
        // can resolve >12-char (spilled) StringViews (sv_bytes/sv_compare/
        // sv_like). buf_idx is always 0 here (single overflow buffer).
        for (uint32_t c = 0; c < schema.num_cols; ++c) {
            if (schema.col_types[c] == BoltType::Utf8) {
                cols[c].str_overflow_base = overflow_buf;
            }
        }
    }

    // Skip header if present.
    size_t pos = 0;
    if (schema.has_header && buf_len > 0) {
        const size_t nl = find_newline(buf, buf_len, 0);
        pos = (nl < buf_len) ? nl + 1 : buf_len;
    }

    int64_t row = 0;
    while (pos < buf_len && row < row_count) {
        const size_t nl = find_newline(buf, buf_len, pos);
        // line_end = nl (exclusive); if no newline, line_end = buf_len.
        if (!parse_line(schema, row, buf, buf_len, pos, nl,
                        cols, overflow_buf, overflow_cap, &overflow_cursor)) {
            return false;
        }
        pos = (nl < buf_len) ? nl + 1 : buf_len;
        ++row;
    }

    // Guard against off-by-one between pass 1 and pass 2.
    if (row != row_count) return false;
    return true;
}

}  // namespace ingest
}  // namespace bolt

// ---------------------------------------------------------------------------
// C FFI surface.
// ---------------------------------------------------------------------------

extern "C" int bolt_ingest_parse_csv_c(const char* buf, size_t buf_len,
                                       const void* schema_ptr,
                                       void* arena_ptr,
                                       void* batch_ptr) {
    if (!schema_ptr || !arena_ptr || !batch_ptr) return 0;
    const auto* schema = static_cast<const bolt::ingest::CsvSchema*>(schema_ptr);
    auto* arena  = static_cast<bolt::Arena*>(arena_ptr);
    auto* batch  = static_cast<bolt::BoltBatch*>(batch_ptr);
    return bolt::ingest::parse_csv(buf, buf_len, *schema, arena, batch) ? 1 : 0;
}
