// bolt_parquet_write.cpp — W-PQ-W: streaming Parquet writer.
// See include/bolt/ingest/bolt_parquet_write.h for scope + Tiger Style
// contract.
//
// File layout:
//   "PAR1"
//   <row group 0>
//     <column chunk 0: data page header (thrift) + data page bytes (codec)>
//     ...
//     <column chunk K-1: data page header + data page bytes>
//   <row group 1> ...
//   <thrift FileMetaData>
//   <u32 LE footer length>
//   "PAR1"
//
// Per data page bytes (PLAIN encoding):
//   [optional def-level stream prefix: u32 LE byte length + RLE/bitpack hybrid]
//     - present only when the column is OPTIONAL (max_def_level == 1).
//     - bw == 1; per row 0 = null, 1 = valid.
//   [PLAIN payload for the non-null values]
//
// Footer Thrift fields written (must match the reader's parser exactly —
// see src/ingest/bolt_parquet_meta.cpp):
//   FileMetaData {1 version i32, 2 schema list<SchemaElement>,
//                 3 num_rows i64, 4 row_groups list<RowGroup>,
//                 6 created_by binary}
//   SchemaElement (root)  {4 name "root", 5 num_children i32}
//   SchemaElement (col)   {1 type, 2 type_length (FLBA only),
//                          3 repetition, 4 name, 5 num_children=0,
//                          6 converted_type, 7 scale, 8 precision}
//   RowGroup              {1 columns list<ColumnChunk>, 2 total_byte_size,
//                          3 num_rows}
//   ColumnChunk           {3 meta_data}
//   ColumnMetaData        {1 type, 2 encodings list<i32>, 3 path_in_schema
//                          list<binary>, 4 codec, 5 num_values,
//                          6 total_uncompressed_size,
//                          7 total_compressed_size,
//                          9 data_page_offset, 12 statistics}
//   Statistics            {3 null_count, 5 max_value, 6 min_value}

#include "bolt/ingest/bolt_parquet_write.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"
#include "bolt/ingest/bolt_snappy.h"   // snappy_compress (added in this CL)

namespace bolt {
namespace ingest {
namespace parquet {

namespace {

// ===== compile-time caps ===================================================

constexpr std::uint32_t kPwMaxRowGroupBytes = 64u * 1024u * 1024u;  // 64 MiB
constexpr std::uint32_t kPwMaxRowGroups     = 4096u;
constexpr std::uint32_t kPwMaxColumns       = kMaxBatchColumns;

// Parquet enum codes we emit (parquet.thrift).
constexpr std::int32_t kPtBoolean   = 0;
constexpr std::int32_t kPtInt32     = 1;
constexpr std::int32_t kPtInt64     = 2;
constexpr std::int32_t kPtFloat     = 4;
constexpr std::int32_t kPtDouble    = 5;
constexpr std::int32_t kPtByteArray = 6;
constexpr std::int32_t kPtFlba      = 7;

constexpr std::int32_t kRepRequired = 0;
constexpr std::int32_t kRepOptional = 1;

constexpr std::int32_t kEncPlain = 0;
constexpr std::int32_t kEncRle   = 3;     // for def-level hybrid

constexpr std::int32_t kCodecUncompressed = 0;
constexpr std::int32_t kCodecSnappy       = 1;

constexpr std::int32_t kConvUtf8    = 0;
constexpr std::int32_t kConvDate    = 6;
constexpr std::int32_t kConvTsMicro = 10;
constexpr std::int32_t kConvDecimal = 5;

constexpr std::int32_t kPageData = 0;

// ===== thrift compact write codec ==========================================
//
// We emit into a std::vector<uint8_t>; the writer accumulates ~KB at a
// time and flushes through std::ofstream, so std::vector at the edge is
// well within the Tiger Style guideline (no hot per-row vector growth).

struct TcOut {
    std::vector<std::uint8_t>* buf;
};

inline void tc_put_u8(TcOut* o, std::uint8_t b) noexcept {
    assert(o != nullptr && o->buf != nullptr);
    o->buf->push_back(b);
}

inline void tc_put_varint(TcOut* o, std::uint64_t v) noexcept {
    assert(o != nullptr);
    // Bounded: <= 10 bytes for a u64.
    for (int i = 0; i < 10; ++i) {
        const std::uint8_t b = static_cast<std::uint8_t>(v & 0x7Fu);
        v >>= 7;
        if (v == 0) { tc_put_u8(o, b); return; }
        tc_put_u8(o, static_cast<std::uint8_t>(b | 0x80u));
    }
    assert(false && "varint overflow");
}

inline void tc_put_zigzag(TcOut* o, std::int64_t v) noexcept {
    assert(o != nullptr);
    const std::uint64_t u = (static_cast<std::uint64_t>(v) << 1) ^
                            static_cast<std::uint64_t>(v >> 63);
    tc_put_varint(o, u);
}

// Field header. Always uses long form (delta=0, then zigzag id) so we never
// have to track the previous field id across helpers. Cheap and trivially
// correct; readers tolerate both forms.
inline void tc_put_field(TcOut* o, std::int16_t fid, std::uint8_t type) noexcept {
    assert(o != nullptr);
    assert(type != 0u && type <= 12u);
    tc_put_u8(o, static_cast<std::uint8_t>(type & 0x0Fu));
    tc_put_zigzag(o, static_cast<std::int64_t>(fid));
}

inline void tc_put_stop(TcOut* o) noexcept {
    assert(o != nullptr);
    tc_put_u8(o, 0u);
}

inline void tc_put_list_hdr(TcOut* o, std::uint8_t elem_type,
                            std::uint32_t n) noexcept {
    assert(o != nullptr);
    if (n < 15u) {
        tc_put_u8(o, static_cast<std::uint8_t>((n << 4) | (elem_type & 0x0Fu)));
        return;
    }
    tc_put_u8(o, static_cast<std::uint8_t>(0xF0u | (elem_type & 0x0Fu)));
    tc_put_varint(o, n);
}

inline void tc_put_binary(TcOut* o, const std::uint8_t* p,
                          std::uint32_t n) noexcept {
    assert(o != nullptr);
    assert(p != nullptr || n == 0u);
    tc_put_varint(o, n);
    if (n > 0u) o->buf->insert(o->buf->end(), p, p + n);
}

inline void tc_put_string(TcOut* o, const char* s) noexcept {
    assert(o != nullptr && s != nullptr);
    const std::uint32_t n = static_cast<std::uint32_t>(std::strlen(s));
    tc_put_binary(o, reinterpret_cast<const std::uint8_t*>(s), n);
}

// Thrift compact element types we use.
constexpr std::uint8_t kFI32 = 5, kFI64 = 6, kFBinary = 8, kFList = 9,
                      kFStruct = 12, kFTrue = 1, kFFalse = 2;

// ===== bit / def-level encoding ===========================================
//
// We emit def levels as a single bit-packed run that covers all rows of
// the page. Parquet's hybrid says odd headers describe (n/8) groups of 8
// values bit-packed LE; we pad the last group with zeros. Plus a leading
// varint over the byte length of the hybrid stream itself, plus the
// data-page-level u32 LE byte length the reader expects in front.

inline std::uint64_t hybrid_varint_emit(std::vector<std::uint8_t>* dst,
                                        std::uint64_t v) noexcept {
    assert(dst != nullptr);
    std::uint64_t n = 0;
    for (int i = 0; i < 10; ++i) {
        const std::uint8_t b = static_cast<std::uint8_t>(v & 0x7Fu);
        v >>= 7;
        if (v == 0) { dst->push_back(b); ++n; return n; }
        dst->push_back(static_cast<std::uint8_t>(b | 0x80u));
        ++n;
    }
    assert(false && "hybrid varint overflow");
    return n;
}

// Encode `n` def-level bits (each 0 or 1) using one bit-packed-only run
// (header = odd: (groups<<1)|1). Pad to a multiple of 8 with zeros.
// Returns the hybrid byte length (not counting the outer u32 prefix).
bool def_levels_encode(const std::uint8_t* def_bits, std::uint32_t n,
                       std::vector<std::uint8_t>* dst) noexcept {
    assert(dst != nullptr);
    assert(def_bits != nullptr || n == 0u);
    if (n == 0u) return true;
    const std::uint32_t groups = (n + 7u) / 8u;
    if (groups > 0x7FFFFFFFu) return false;
    const std::uint64_t header =
        (static_cast<std::uint64_t>(groups) << 1) | 1ull;
    (void)hybrid_varint_emit(dst, header);
    // bw == 1: byte i packs bits for rows [i*8, i*8+8). LSB = row i*8.
    for (std::uint32_t g = 0; g < groups; ++g) {
        std::uint8_t b = 0;
        for (std::uint32_t k = 0; k < 8u; ++k) {
            const std::uint32_t row = g * 8u + k;
            if (row >= n) break;
            if (def_bits[row]) b = static_cast<std::uint8_t>(b | (1u << k));
        }
        dst->push_back(b);
    }
    return true;
}

// Read the validity bitmap one bit per row (Arrow shape, 1 = valid).
// Honors validity_offset. Returns 1 when the row is valid (or no bitmap).
inline std::uint8_t read_valid(const std::uint8_t* bm, std::int64_t off,
                               std::int64_t row) noexcept {
    if (bm == nullptr) return 1u;
    const std::int64_t bit = off + row;
    const std::uint8_t v = bm[bit >> 3];
    return static_cast<std::uint8_t>((v >> (bit & 7)) & 1u);
}

// ===== writer state =======================================================

// Per-chunk record built up while writing the chunk; flushed into the
// footer at close.
struct ChunkRec {
    std::int64_t num_values;
    std::int64_t total_unc;
    std::int64_t total_cmp;
    std::int64_t data_page_offset;
    std::int64_t null_count;
    bool         have_stats;
    bool         null_count_known;
    // Min / max as raw little-endian (or BE for Decimal128, raw bytes for
    // BYTE_ARRAY) payloads the reader can copy straight into
    // PqChunk::min_bytes / max_bytes. Sized to the reader's
    // kPqMaxStatBytes (64) so Utf8 stats round-trip; a chunk whose extreme
    // string is longer than this simply omits stats (compute_stats).
    std::uint8_t min_buf[64];
    std::uint8_t max_buf[64];
    std::uint32_t min_len;
    std::uint32_t max_len;
};

struct RowGroupRec {
    std::int64_t  num_rows;
    std::int64_t  total_byte_size;
    std::uint32_t chunk_off;     // index into ParquetWriter::chunks
    std::uint32_t chunk_count;
};

}  // namespace

// Fully-defined writer state (opaque to callers).
struct ParquetWriter {
    std::ofstream     fout;
    ParquetWriteOpts  opts;
    std::int64_t      file_pos;        // bytes written
    bool              failed;
    // Footer accumulators.
    std::vector<ChunkRec>    chunks;
    std::vector<RowGroupRec> row_groups;
    std::int64_t      total_rows;
};

namespace {

// ===== type validation ====================================================

bool type_supported(const ParquetWriteColumn& c) noexcept {
    switch (c.type) {
        case BoltType::Bool:
        case BoltType::Int32:
        case BoltType::Int64:
        case BoltType::Float32:
        case BoltType::Float64:
        case BoltType::Utf8:
        case BoltType::Binary:
        case BoltType::Date32:
        case BoltType::Timestamp:
            return true;
        case BoltType::Decimal128:
            // Tolerate p=0 (no statistics use scale) but require scale < 38.
            return c.scale <= 38u && c.precision <= 38u;
        default:
            return false;
    }
}

std::int32_t bolt_to_pq_physical(BoltType t) noexcept {
    switch (t) {
        case BoltType::Bool:       return kPtBoolean;
        case BoltType::Int32:
        case BoltType::Date32:     return kPtInt32;
        case BoltType::Int64:
        case BoltType::Timestamp:  return kPtInt64;
        case BoltType::Float32:    return kPtFloat;
        case BoltType::Float64:    return kPtDouble;
        case BoltType::Utf8:
        case BoltType::Binary:     return kPtByteArray;
        case BoltType::Decimal128: return kPtFlba;
        default:                   return kPtInt32;   // unreachable
    }
}

std::int32_t bolt_to_pq_converted(BoltType t) noexcept {
    switch (t) {
        case BoltType::Utf8:       return kConvUtf8;
        case BoltType::Date32:     return kConvDate;
        case BoltType::Timestamp:  return kConvTsMicro;
        case BoltType::Decimal128: return kConvDecimal;
        default:                   return -1;
    }
}

bool has_converted(BoltType t) noexcept {
    return bolt_to_pq_converted(t) >= 0;
}

// ===== PLAIN value encoders ===============================================

bool encode_plain_fixed(const BoltColumn& col, std::size_t elem,
                        std::int64_t n_rows,
                        const std::uint8_t* def_bits,
                        bool nullable,
                        std::vector<std::uint8_t>* dst) noexcept {
    assert(dst != nullptr);
    assert(col.data != nullptr || n_rows == 0);
    const std::uint8_t* src = static_cast<const std::uint8_t*>(col.data);
    for (std::int64_t r = 0; r < n_rows; ++r) {
        const bool valid = (!nullable) || def_bits[r] != 0;
        if (!valid) continue;
        const std::uint8_t* p = src + static_cast<std::size_t>(r) * elem;
        dst->insert(dst->end(), p, p + elem);
    }
    return true;
}

bool encode_plain_bool(const BoltColumn& col, std::int64_t n_rows,
                       const std::uint8_t* def_bits, bool nullable,
                       std::vector<std::uint8_t>* dst) noexcept {
    assert(dst != nullptr);
    assert(col.data != nullptr || n_rows == 0);
    // BoltColumn stores bool as one byte per row (see bolt_types.h line 53).
    const std::uint8_t* src = static_cast<const std::uint8_t*>(col.data);
    // Pack PLAIN bool: bit-packed LE, 8 values per byte.
    std::uint8_t pack = 0;
    std::uint32_t nb = 0;
    for (std::int64_t r = 0; r < n_rows; ++r) {
        const bool valid = (!nullable) || def_bits[r] != 0;
        if (!valid) continue;
        if (src[r] != 0u) pack = static_cast<std::uint8_t>(pack | (1u << nb));
        ++nb;
        if (nb == 8u) { dst->push_back(pack); pack = 0; nb = 0; }
    }
    if (nb != 0u) dst->push_back(pack);
    return true;
}

bool encode_plain_byte_array(const BoltColumn& col, std::int64_t n_rows,
                             const std::uint8_t* def_bits, bool nullable,
                             std::vector<std::uint8_t>* dst) noexcept {
    assert(dst != nullptr);
    assert(col.data != nullptr || n_rows == 0);
    const auto* sv = static_cast<const StringView*>(col.data);
    const auto* spill = static_cast<const std::uint8_t*>(col.str_overflow_base);
    for (std::int64_t r = 0; r < n_rows; ++r) {
        const bool valid = (!nullable) || def_bits[r] != 0;
        if (!valid) continue;
        const std::uint32_t len = sv[r].length;
        const std::uint8_t lb[4] = {
            static_cast<std::uint8_t>(len & 0xFFu),
            static_cast<std::uint8_t>((len >> 8) & 0xFFu),
            static_cast<std::uint8_t>((len >> 16) & 0xFFu),
            static_cast<std::uint8_t>((len >> 24) & 0xFFu),
        };
        dst->insert(dst->end(), lb, lb + 4);
        if (len == 0u) continue;
        if (len <= 12u) {
            // Inline payload — prefix[4] + inline_data[8] are contiguous.
            const std::uint8_t* p =
                reinterpret_cast<const std::uint8_t*>(&sv[r].prefix[0]);
            dst->insert(dst->end(), p, p + len);
        } else {
            if (spill == nullptr) return false;
            const std::uint8_t* p = spill + sv[r].ref.offset;
            dst->insert(dst->end(), p, p + len);
        }
    }
    return true;
}

// ===== def-level packing ==================================================

// Fill def_bits[0..n_rows) with 0/1 from the column's validity bitmap.
// Counts nulls into *null_count.
void fill_def_bits(const BoltColumn& col, std::int64_t n_rows,
                   std::uint8_t* def_bits,
                   std::int64_t* null_count) noexcept {
    assert(def_bits != nullptr);
    assert(null_count != nullptr);
    const std::uint8_t* bm = col.validity;
    const std::int64_t off = col.validity_offset;
    std::int64_t nulls = 0;
    for (std::int64_t r = 0; r < n_rows; ++r) {
        const std::uint8_t v = read_valid(bm, off, r);
        def_bits[r] = v;
        if (v == 0u) ++nulls;
    }
    *null_count = nulls;
}

// ===== statistics (PLAIN min/max bytes — what the reader will copy) =======

// Helpers that write little-endian fixed-width values into the dst buffer.
template <typename T>
void write_le(std::uint8_t* dst, T v) noexcept {
    std::memcpy(dst, &v, sizeof(T));
}

// Unsigned byte-wise lexicographic compare; shorter-prefix compares less.
// (The UTF8/BYTE_ARRAY sort order min_value/max_value are defined against.)
int stat_lex_cmp(const std::uint8_t* a, std::uint32_t a_len,
                 const std::uint8_t* b, std::uint32_t b_len) noexcept {
    const std::uint32_t n = (a_len < b_len) ? a_len : b_len;
    const int c = (n != 0) ? std::memcmp(a, b, n) : 0;
    if (c != 0) return c;
    return (a_len < b_len) ? -1 : (a_len > b_len ? 1 : 0);
}

// Utf8/Binary chunk stats (G2FEAT-21). Emits full min/max byte strings —
// never a truncated bound (a truncated max without the spec's
// increment-last-byte adjustment would be an INVALID upper bound). A chunk
// containing any value longer than the stat buffer (== the reader's
// kPqMaxStatBytes) therefore omits stats entirely; omission is always
// legal and the reader treats it as "cannot prune".
bool compute_stats_utf8(const BoltColumn& col, std::int64_t n_rows,
                        const std::uint8_t* def_bits, bool nullable,
                        ChunkRec* rec) noexcept {
    assert(rec != nullptr);
    assert(col.data != nullptr || n_rows == 0);
    const auto* sv    = static_cast<const StringView*>(col.data);
    const auto* spill = static_cast<const std::uint8_t*>(col.str_overflow_base);
    const std::uint8_t* mn_p = nullptr; std::uint32_t mn_n = 0;
    const std::uint8_t* mx_p = nullptr; std::uint32_t mx_n = 0;
    for (std::int64_t r = 0; r < n_rows; ++r) {
        const bool valid = (!nullable) || def_bits[r] != 0;
        if (!valid) continue;
        const std::uint32_t len = sv[r].length;
        if (len > sizeof(rec->min_buf)) return true;   // omit stats (see above)
        const std::uint8_t* p;
        if (len <= 12u) {
            // Inline payload — prefix[4] + inline_data[8] are contiguous
            // (same layout contract encode_plain_byte_array relies on).
            p = reinterpret_cast<const std::uint8_t*>(&sv[r].prefix[0]);
        } else {
            if (spill == nullptr) return true;         // can't read: omit
            p = spill + sv[r].ref.offset;
        }
        if (mn_p == nullptr || stat_lex_cmp(p, len, mn_p, mn_n) < 0) {
            mn_p = p; mn_n = len;
        }
        if (mx_p == nullptr || stat_lex_cmp(p, len, mx_p, mx_n) > 0) {
            mx_p = p; mx_n = len;
        }
    }
    if (mn_p == nullptr) return true;   // all null -> no min/max
    std::memcpy(rec->min_buf, mn_p, mn_n);
    std::memcpy(rec->max_buf, mx_p, mx_n);
    rec->min_len = mn_n;
    rec->max_len = mx_n;
    rec->have_stats = true;
    return true;
}

bool compute_stats(const BoltColumn& col, std::int64_t n_rows,
                   const std::uint8_t* def_bits, bool nullable,
                   ChunkRec* rec) noexcept {
    assert(rec != nullptr);
    rec->have_stats = false;
    if (n_rows == 0) return true;
    const auto valid_at = [&](std::int64_t r) noexcept {
        return (!nullable) || def_bits[r] != 0;
    };
    // Find first valid row.
    std::int64_t r0 = -1;
    for (std::int64_t r = 0; r < n_rows; ++r) {
        if (valid_at(r)) { r0 = r; break; }
    }
    if (r0 < 0) return true;   // all null -> no min/max
    switch (col.type) {
        case BoltType::Int32:
        case BoltType::Date32: {
            const auto* p = static_cast<const std::int32_t*>(col.data);
            std::int32_t mn = p[r0], mx = p[r0];
            for (std::int64_t r = r0 + 1; r < n_rows; ++r) {
                if (!valid_at(r)) continue;
                if (p[r] < mn) mn = p[r];
                if (p[r] > mx) mx = p[r];
            }
            write_le<std::int32_t>(rec->min_buf, mn);
            write_le<std::int32_t>(rec->max_buf, mx);
            rec->min_len = rec->max_len = 4u;
            rec->have_stats = true;
            return true;
        }
        case BoltType::Int64:
        case BoltType::Timestamp: {
            const auto* p = static_cast<const std::int64_t*>(col.data);
            std::int64_t mn = p[r0], mx = p[r0];
            for (std::int64_t r = r0 + 1; r < n_rows; ++r) {
                if (!valid_at(r)) continue;
                if (p[r] < mn) mn = p[r];
                if (p[r] > mx) mx = p[r];
            }
            write_le<std::int64_t>(rec->min_buf, mn);
            write_le<std::int64_t>(rec->max_buf, mx);
            rec->min_len = rec->max_len = 8u;
            rec->have_stats = true;
            return true;
        }
        case BoltType::Float32: {
            const auto* p = static_cast<const float*>(col.data);
            bool seen = false;
            float mn = 0.0f, mx = 0.0f;
            for (std::int64_t r = r0; r < n_rows; ++r) {
                if (!valid_at(r)) continue;
                const float v = p[r];
                if (v != v) continue;               // NaN: never a bound
                if (!seen) { mn = mx = v; seen = true; continue; }
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            if (!seen) return true;                 // all NaN -> omit stats
            // Spec recommendation: a zero min is written as -0.0 and a zero
            // max as +0.0 so both signed zeros lie inside [min, max].
            if (mn == 0.0f) mn = -0.0f;
            if (mx == 0.0f) mx = +0.0f;
            write_le<float>(rec->min_buf, mn);
            write_le<float>(rec->max_buf, mx);
            rec->min_len = rec->max_len = 4u;
            rec->have_stats = true;
            return true;
        }
        case BoltType::Float64: {
            const auto* p = static_cast<const double*>(col.data);
            bool seen = false;
            double mn = 0.0, mx = 0.0;
            for (std::int64_t r = r0; r < n_rows; ++r) {
                if (!valid_at(r)) continue;
                const double v = p[r];
                if (v != v) continue;               // NaN: never a bound
                if (!seen) { mn = mx = v; seen = true; continue; }
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            if (!seen) return true;                 // all NaN -> omit stats
            if (mn == 0.0) mn = -0.0;
            if (mx == 0.0) mx = +0.0;
            write_le<double>(rec->min_buf, mn);
            write_le<double>(rec->max_buf, mx);
            rec->min_len = rec->max_len = 8u;
            rec->have_stats = true;
            return true;
        }
        case BoltType::Bool: {
            // BoltColumn stores bool one byte per row; the Statistics
            // payload for BOOLEAN is a single PLAIN byte 0x00 / 0x01.
            const auto* p = static_cast<const std::uint8_t*>(col.data);
            std::uint8_t mn = 1u, mx = 0u;
            for (std::int64_t r = r0; r < n_rows; ++r) {
                if (!valid_at(r)) continue;
                const std::uint8_t v = (p[r] != 0u) ? 1u : 0u;
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            if (mn > mx) return true;   // unreachable (r0 valid) — safety
            rec->min_buf[0] = mn;
            rec->max_buf[0] = mx;
            rec->min_len = rec->max_len = 1u;
            rec->have_stats = true;
            return true;
        }
        case BoltType::Utf8:
        case BoltType::Binary:
            return compute_stats_utf8(col, n_rows, def_bits, nullable, rec);
        default:
            // Decimal128: min/max skipped (v1 gap, documented in the header).
            return true;
    }
}

// ===== snappy compression wrapper =========================================

bool maybe_compress(const std::uint8_t* src, std::size_t src_len,
                    std::uint8_t codec, std::vector<std::uint8_t>* dst) noexcept {
    assert(dst != nullptr);
    if (codec == 0u) {
        dst->assign(src, src + src_len);
        return true;
    }
    if (codec == 1u) {
        const std::size_t cap = snappy_max_compressed_len(src_len);
        dst->resize(cap);
        std::uint64_t out_len = 0;
        if (!snappy_compress(src, src_len, dst->data(), cap, &out_len)) {
            return false;
        }
        dst->resize(static_cast<std::size_t>(out_len));
        return true;
    }
    return false;
}

// ===== page header writer =================================================

void write_page_header(std::vector<std::uint8_t>* hdr,
                       std::int32_t unc, std::int32_t cmp,
                       std::int32_t num_values) noexcept {
    assert(hdr != nullptr);
    TcOut o{hdr};
    // PageHeader.type = DATA_PAGE
    tc_put_field(&o, 1, kFI32);
    tc_put_zigzag(&o, kPageData);
    tc_put_field(&o, 2, kFI32);
    tc_put_zigzag(&o, unc);
    tc_put_field(&o, 3, kFI32);
    tc_put_zigzag(&o, cmp);
    // DataPageHeader (field 5)
    tc_put_field(&o, 5, kFStruct);
    {
        tc_put_field(&o, 1, kFI32);
        tc_put_zigzag(&o, num_values);
        tc_put_field(&o, 2, kFI32);
        tc_put_zigzag(&o, kEncPlain);
        tc_put_field(&o, 3, kFI32);
        tc_put_zigzag(&o, kEncRle);     // def-level encoding
        tc_put_field(&o, 4, kFI32);
        tc_put_zigzag(&o, kEncRle);     // rep-level encoding (unused; flat schema)
        tc_put_stop(&o);
    }
    tc_put_stop(&o);
}

// ===== chunk writer (one column of one row group) ========================

// Build the in-memory page payload (def levels + PLAIN values), then
// compress and write [page header][compressed bytes]. Updates rec.
bool write_column_chunk(ParquetWriter* w, const BoltColumn& col,
                        const ParquetWriteColumn& sch,
                        std::int64_t n_rows, ChunkRec* rec) noexcept {
    assert(w != nullptr && rec != nullptr);
    std::memset(rec, 0, sizeof(*rec));
    rec->num_values = n_rows;
    rec->null_count = 0;

    // Validate column type matches schema.
    if (col.type != sch.type) return false;
    if (col.length < n_rows) return false;

    // Allocate def-level workspace if the column is OPTIONAL.
    std::vector<std::uint8_t> def_bits;
    const bool nullable = sch.nullable;
    if (nullable) {
        def_bits.resize(static_cast<std::size_t>(n_rows));
        fill_def_bits(col, n_rows, def_bits.data(), &rec->null_count);
    }
    // G2FEAT-24: null_count is always known — counted for OPTIONAL columns,
    // structurally 0 for REQUIRED ones — so the footer always carries it
    // (pyarrow/DuckDB surface it; bolt's pq_chunk_no_nulls accepts either
    // null_count == 0 or REQUIRED, so both readers agree).
    rec->null_count_known = true;

    // Encode PLAIN values.
    std::vector<std::uint8_t> values;
    values.reserve(static_cast<std::size_t>(n_rows) * 8u);
    bool ok = false;
    switch (sch.type) {
        case BoltType::Bool:
            ok = encode_plain_bool(col, n_rows,
                                   nullable ? def_bits.data() : nullptr,
                                   nullable, &values);
            break;
        case BoltType::Int32:
        case BoltType::Date32:
        case BoltType::Float32:
            ok = encode_plain_fixed(col, 4, n_rows,
                                    nullable ? def_bits.data() : nullptr,
                                    nullable, &values);
            break;
        case BoltType::Int64:
        case BoltType::Timestamp:
        case BoltType::Float64:
            ok = encode_plain_fixed(col, 8, n_rows,
                                    nullable ? def_bits.data() : nullptr,
                                    nullable, &values);
            break;
        case BoltType::Utf8:
        case BoltType::Binary:
            ok = encode_plain_byte_array(col, n_rows,
                                         nullable ? def_bits.data() : nullptr,
                                         nullable, &values);
            break;
        case BoltType::Decimal128:
            // FLBA 16 BE — caller supplies decimal128 already as BE 16-byte
            // values, OR as little-endian Decimal128 (Bolt's in-memory shape).
            // Bolt stores Decimal128 inline 16 bytes; spec says the on-disk
            // form must be big-endian two's complement. Convert per value.
            {
                const auto* p = static_cast<const std::uint8_t*>(col.data);
                for (std::int64_t r = 0; r < n_rows; ++r) {
                    const bool valid = (!nullable) || def_bits[r] != 0;
                    if (!valid) continue;
                    std::uint8_t be[16];
                    for (std::size_t k = 0; k < 16u; ++k) be[k] = p[r * 16 + 15 - k];
                    values.insert(values.end(), be, be + 16);
                }
                ok = true;
            }
            break;
        default:
            return false;
    }
    if (!ok) return false;

    // Stats (after def_bits computed).
    if (w->opts.emit_statistics) {
        if (!compute_stats(col, n_rows,
                           nullable ? def_bits.data() : nullptr,
                           nullable, rec)) {
            return false;
        }
    }

    // Build the data page bytes: [u32 LE def-level byte len][hybrid bytes][PLAIN values]
    std::vector<std::uint8_t> page;
    if (nullable) {
        std::vector<std::uint8_t> def_hybrid;
        if (!def_levels_encode(def_bits.data(),
                               static_cast<std::uint32_t>(n_rows),
                               &def_hybrid)) {
            return false;
        }
        const std::uint32_t dlen = static_cast<std::uint32_t>(def_hybrid.size());
        const std::uint8_t lb[4] = {
            static_cast<std::uint8_t>(dlen & 0xFFu),
            static_cast<std::uint8_t>((dlen >> 8) & 0xFFu),
            static_cast<std::uint8_t>((dlen >> 16) & 0xFFu),
            static_cast<std::uint8_t>((dlen >> 24) & 0xFFu),
        };
        page.insert(page.end(), lb, lb + 4);
        page.insert(page.end(), def_hybrid.begin(), def_hybrid.end());
    }
    page.insert(page.end(), values.begin(), values.end());

    const std::int32_t unc = static_cast<std::int32_t>(page.size());

    // Compress.
    std::vector<std::uint8_t> compressed;
    if (!maybe_compress(page.data(), page.size(), w->opts.compression,
                        &compressed)) {
        return false;
    }
    const std::int32_t cmp = static_cast<std::int32_t>(compressed.size());

    // Header.
    std::vector<std::uint8_t> hdr;
    write_page_header(&hdr, unc, cmp,
                      static_cast<std::int32_t>(n_rows));

    // Position & write.
    rec->data_page_offset = w->file_pos;
    w->fout.write(reinterpret_cast<const char*>(hdr.data()),
                  static_cast<std::streamsize>(hdr.size()));
    w->fout.write(reinterpret_cast<const char*>(compressed.data()),
                  static_cast<std::streamsize>(compressed.size()));
    if (!w->fout.good()) return false;
    w->file_pos += static_cast<std::int64_t>(hdr.size() + compressed.size());
    rec->total_unc = static_cast<std::int64_t>(hdr.size()) + unc;
    rec->total_cmp = static_cast<std::int64_t>(hdr.size()) + cmp;
    return true;
}

// ===== footer (FileMetaData) =============================================

void write_schema_element_root(TcOut* o, std::uint32_t n_children) noexcept {
    tc_put_field(o, 4, kFBinary);
    tc_put_string(o, "schema");
    tc_put_field(o, 5, kFI32);
    tc_put_zigzag(o, static_cast<std::int64_t>(n_children));
    tc_put_stop(o);
}

void write_schema_element_col(TcOut* o,
                              const ParquetWriteColumn& c) noexcept {
    tc_put_field(o, 1, kFI32);
    tc_put_zigzag(o, bolt_to_pq_physical(c.type));
    if (c.type == BoltType::Decimal128) {
        tc_put_field(o, 2, kFI32);
        tc_put_zigzag(o, 16);                // FLBA length
    }
    tc_put_field(o, 3, kFI32);
    tc_put_zigzag(o, c.nullable ? kRepOptional : kRepRequired);
    tc_put_field(o, 4, kFBinary);
    tc_put_string(o, c.name);
    tc_put_field(o, 5, kFI32);
    tc_put_zigzag(o, 0);                     // num_children
    if (has_converted(c.type)) {
        tc_put_field(o, 6, kFI32);
        tc_put_zigzag(o, bolt_to_pq_converted(c.type));
    }
    if (c.type == BoltType::Decimal128) {
        tc_put_field(o, 7, kFI32);
        tc_put_zigzag(o, c.scale);
        tc_put_field(o, 8, kFI32);
        tc_put_zigzag(o, c.precision);
    }
    tc_put_stop(o);
}

void write_statistics(TcOut* o, const ChunkRec& rec) noexcept {
    if (rec.null_count_known) {
        tc_put_field(o, 3, kFI64);
        tc_put_zigzag(o, rec.null_count);
    }
    if (rec.have_stats) {
        tc_put_field(o, 5, kFBinary);
        tc_put_binary(o, rec.max_buf, rec.max_len);
        tc_put_field(o, 6, kFBinary);
        tc_put_binary(o, rec.min_buf, rec.min_len);
    }
    tc_put_stop(o);
}

void write_column_meta(TcOut* o, const ParquetWriter* w,
                       const ParquetWriteColumn& sch,
                       const ChunkRec& rec) noexcept {
    tc_put_field(o, 1, kFI32);
    tc_put_zigzag(o, bolt_to_pq_physical(sch.type));
    // encodings: list<i32> = {PLAIN, RLE}
    tc_put_field(o, 2, kFList);
    tc_put_list_hdr(o, kFI32, 2);
    tc_put_zigzag(o, kEncPlain);
    tc_put_zigzag(o, kEncRle);
    // path_in_schema: list<binary> = {name}
    tc_put_field(o, 3, kFList);
    tc_put_list_hdr(o, kFBinary, 1);
    tc_put_string(o, sch.name);
    // codec
    tc_put_field(o, 4, kFI32);
    tc_put_zigzag(o, (w->opts.compression == 1u) ? kCodecSnappy
                                                 : kCodecUncompressed);
    tc_put_field(o, 5, kFI64);
    tc_put_zigzag(o, rec.num_values);
    tc_put_field(o, 6, kFI64);
    tc_put_zigzag(o, rec.total_unc);
    tc_put_field(o, 7, kFI64);
    tc_put_zigzag(o, rec.total_cmp);
    tc_put_field(o, 9, kFI64);
    tc_put_zigzag(o, rec.data_page_offset);
    if (w->opts.emit_statistics && (rec.have_stats || rec.null_count_known)) {
        tc_put_field(o, 12, kFStruct);
        write_statistics(o, rec);
    }
    tc_put_stop(o);
}

void write_column_chunk_struct(TcOut* o, const ParquetWriter* w,
                               const ParquetWriteColumn& sch,
                               const ChunkRec& rec) noexcept {
    // ColumnChunk: 2 = file_offset (i64, REQUIRED in parquet.thrift).
    // Deprecated by the spec ("writers should write 0") but thrift-generated
    // readers (pyarrow, DuckDB) throw INVALID_DATA when a required field is
    // absent — omitting it made every bolt-written file unreadable by them
    // (G2FEAT-24 audit find; bolt's own reader never consulted it, which is
    // why the gap was invisible to round-trip tests).
    tc_put_field(o, 2, kFI64);
    tc_put_zigzag(o, 0);
    // ColumnChunk: 3 = meta_data (struct)
    tc_put_field(o, 3, kFStruct);
    write_column_meta(o, w, sch, rec);
    tc_put_stop(o);
}

void write_row_group(TcOut* o, const ParquetWriter* w,
                     const RowGroupRec& rg) noexcept {
    // columns list<struct>
    tc_put_field(o, 1, kFList);
    tc_put_list_hdr(o, kFStruct, rg.chunk_count);
    for (std::uint32_t i = 0; i < rg.chunk_count; ++i) {
        const std::uint32_t ci = rg.chunk_off + i;
        write_column_chunk_struct(o, w, w->opts.columns[i],
                                  w->chunks[ci]);
    }
    tc_put_field(o, 2, kFI64);
    tc_put_zigzag(o, rg.total_byte_size);
    tc_put_field(o, 3, kFI64);
    tc_put_zigzag(o, rg.num_rows);
    tc_put_stop(o);
}

void write_file_metadata(std::vector<std::uint8_t>* dst,
                         const ParquetWriter* w) noexcept {
    assert(dst != nullptr && w != nullptr);
    TcOut o{dst};
    // 1 version
    tc_put_field(&o, 1, kFI32);
    tc_put_zigzag(&o, 1);
    // 2 schema: list<SchemaElement>: root + N cols
    const std::uint32_t n = w->opts.n_columns;
    tc_put_field(&o, 2, kFList);
    tc_put_list_hdr(&o, kFStruct, n + 1u);
    write_schema_element_root(&o, n);
    for (std::uint32_t i = 0; i < n; ++i) {
        write_schema_element_col(&o, w->opts.columns[i]);
    }
    // 3 num_rows
    tc_put_field(&o, 3, kFI64);
    tc_put_zigzag(&o, w->total_rows);
    // 4 row_groups: list<RowGroup>
    const std::uint32_t g = static_cast<std::uint32_t>(w->row_groups.size());
    tc_put_field(&o, 4, kFList);
    tc_put_list_hdr(&o, kFStruct, g);
    for (std::uint32_t i = 0; i < g; ++i) {
        write_row_group(&o, w, w->row_groups[i]);
    }
    // 6 created_by
    tc_put_field(&o, 6, kFBinary);
    tc_put_string(&o, "bolt-parquet-writer v1");
    // 7 column_orders: list<ColumnOrder>, one TYPE_ORDER (TypeDefinedOrder,
    // an empty struct inside the ColumnOrder union) per column. Without
    // this, parquet-cpp / pyarrow refuses to trust the modern
    // min_value/max_value statistics (`has_min_max == false`) even though
    // the bytes are present (G2FEAT-24 audit find). Declaring TYPE_ORDER is
    // correct for everything we emit stats for: signed order for
    // INT32/INT64, IEEE order with NaN skipped + signed-zero normalization
    // for FLOAT/DOUBLE, false < true for BOOLEAN, unsigned lexicographic
    // bytes for BYTE_ARRAY (compute_stats/compute_stats_utf8 above).
    // Decimal128 chunks emit no min/max, so the declaration is vacuous there.
    tc_put_field(&o, 7, kFList);
    tc_put_list_hdr(&o, kFStruct, n);
    for (std::uint32_t i = 0; i < n; ++i) {
        tc_put_field(&o, 1, kFStruct);   // ColumnOrder.TYPE_ORDER
        tc_put_stop(&o);                 // empty TypeDefinedOrder struct
        tc_put_stop(&o);                 // end ColumnOrder union
    }
    tc_put_stop(&o);
}

}  // namespace

// ===== public API =========================================================

ParquetWriter* parquet_write_open(const char* path,
                                  const ParquetWriteOpts* opts) noexcept {
    assert(path != nullptr);
    assert(opts != nullptr);
    if (path == nullptr || opts == nullptr) return nullptr;
    if (opts->n_columns == 0u || opts->n_columns > kPwMaxColumns) return nullptr;
    // Reject unsupported codecs up-front.
    if (opts->compression != 0u && opts->compression != 1u) return nullptr;
    for (std::uint32_t i = 0; i < opts->n_columns; ++i) {
        if (!type_supported(opts->columns[i])) return nullptr;
    }
    ParquetWriter* w = new (std::nothrow) ParquetWriter();
    if (w == nullptr) return nullptr;
    w->opts = *opts;
    if (w->opts.row_group_target_bytes == 0u ||
        w->opts.row_group_target_bytes > kPwMaxRowGroupBytes) {
        w->opts.row_group_target_bytes = kPwMaxRowGroupBytes;
    }
    w->file_pos = 0;
    w->failed = false;
    w->total_rows = 0;
    w->fout.open(path, std::ios::binary | std::ios::trunc);
    if (!w->fout.is_open()) {
        delete w;
        return nullptr;
    }
    static const char kMagic[4] = {'P', 'A', 'R', '1'};
    w->fout.write(kMagic, 4);
    if (!w->fout.good()) {
        delete w;
        return nullptr;
    }
    w->file_pos = 4;
    return w;
}

namespace {

// Per-row byte stride of the writer-side in-memory representation, used to
// slice a column at a row offset (G2FEAT-24 rowgroup splitting). Utf8 /
// Binary columns are arrays of StringView whose spill offsets are absolute
// into str_overflow_base, so advancing the view array alone is a valid
// slice. Returns 0 for unsupported types (rejected at open anyway).
std::size_t slice_stride(BoltType t) noexcept {
    switch (t) {
        case BoltType::Bool:       return 1u;   // byte-packed in BoltColumn
        case BoltType::Int32:
        case BoltType::Date32:
        case BoltType::Float32:    return 4u;
        case BoltType::Int64:
        case BoltType::Timestamp:
        case BoltType::Float64:    return 8u;
        case BoltType::Utf8:
        case BoltType::Binary:     return sizeof(StringView);
        case BoltType::Decimal128: return 16u;
        default:                   return 0u;
    }
}

// A shallow row-window view of `c` starting at `start` for `rows` rows.
// Data pointer advances by the type stride; the validity bitmap pointer is
// left alone and validity_offset absorbs the shift (read_valid honors it).
BoltColumn slice_column(const BoltColumn& c, BoltType t,
                        std::int64_t start, std::int64_t rows) noexcept {
    assert(start >= 0);
    assert(rows >= 0);
    BoltColumn s = c;
    s.length = rows;
    if (start > 0 && c.data != nullptr) {
        const std::size_t stride = slice_stride(t);
        s.data = static_cast<std::uint8_t*>(c.data) +
                 static_cast<std::size_t>(start) * stride;
        s.validity_offset = c.validity_offset + start;
    }
    return s;
}

// Append exactly one row group covering rows [start, start + rows) of the
// batch's columns. Bookkeeping mirrors the pre-split writer verbatim.
bool write_one_row_group(ParquetWriter* w, const BoltColumn* cols,
                         std::int64_t start, std::int64_t rows) noexcept {
    assert(w != nullptr && cols != nullptr);
    assert(rows >= 0);
    if (w->row_groups.size() >= kPwMaxRowGroups) return false;
    const std::int64_t rg_start = w->file_pos;
    RowGroupRec rg{};
    rg.num_rows = rows;
    rg.chunk_off = static_cast<std::uint32_t>(w->chunks.size());
    rg.chunk_count = w->opts.n_columns;
    for (std::uint32_t c = 0; c < w->opts.n_columns; ++c) {
        const BoltColumn sc = slice_column(cols[c], w->opts.columns[c].type,
                                           start, rows);
        ChunkRec rec{};
        if (!write_column_chunk(w, sc, w->opts.columns[c], rows, &rec)) {
            return false;
        }
        w->chunks.push_back(rec);
    }
    rg.total_byte_size = w->file_pos - rg_start;
    w->row_groups.push_back(rg);
    w->total_rows += rows;
    return true;
}

}  // namespace

bool parquet_write_row_group(ParquetWriter* w,
                             const BoltBatch* batch) noexcept {
    assert(w != nullptr);
    assert(batch != nullptr);
    if (w == nullptr || batch == nullptr || w->failed) return false;
    if (batch->num_cols != w->opts.n_columns) return false;
    const std::int64_t n_rows = batch->num_rows;
    if (n_rows < 0) return false;

    const BoltColumn* cols = batch->columns[batch->read_epoch];
    // G2FEAT-24: 0 keeps the legacy one-call-one-rowgroup contract; > 0
    // splits into consecutive row groups of at most that many rows.
    const std::int64_t cap =
        (w->opts.row_group_max_rows > 0u)
            ? static_cast<std::int64_t>(w->opts.row_group_max_rows)
            : (n_rows > 0 ? n_rows : 1);
    std::int64_t start = 0;
    do {
        const std::int64_t remain = n_rows - start;
        const std::int64_t rows = (remain < cap) ? remain : cap;
        if (!write_one_row_group(w, cols, start, rows)) {
            w->failed = true;
            return false;
        }
        start += rows;
    } while (start < n_rows);
    return true;
}

bool parquet_write_close(ParquetWriter* w) noexcept {
    assert(w != nullptr);
    if (w == nullptr) return false;
    bool ok = !w->failed && w->fout.good();
    if (ok) {
        std::vector<std::uint8_t> footer;
        write_file_metadata(&footer, w);
        const std::uint32_t flen = static_cast<std::uint32_t>(footer.size());
        w->fout.write(reinterpret_cast<const char*>(footer.data()),
                      static_cast<std::streamsize>(footer.size()));
        const std::uint8_t lb[4] = {
            static_cast<std::uint8_t>(flen & 0xFFu),
            static_cast<std::uint8_t>((flen >> 8) & 0xFFu),
            static_cast<std::uint8_t>((flen >> 16) & 0xFFu),
            static_cast<std::uint8_t>((flen >> 24) & 0xFFu),
        };
        w->fout.write(reinterpret_cast<const char*>(lb), 4);
        static const char kMagic[4] = {'P', 'A', 'R', '1'};
        w->fout.write(kMagic, 4);
        ok = w->fout.good();
    }
    w->fout.close();
    delete w;
    return ok;
}

}  // namespace parquet
}  // namespace ingest
}  // namespace bolt
