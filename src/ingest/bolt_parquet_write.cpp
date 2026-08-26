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
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"
#include "bolt/bolt_scheduler.h"
#include "bolt/ingest/bolt_snappy.h"        // snappy_compress
#include "bolt/ingest/bolt_lz4_raw.h"      // lz4_raw_compress
#include "bolt/ingest/bolt_deflate.h"      // gzip_compress
#include "bolt/ingest/bolt_parquet_bloom.h"  // pq_xxh64 (dictionary hashing)
#include "bolt/ingest/bolt_parquet_pageindex.h"  // kPqMaxPagesPerChunk

namespace bolt {
namespace ingest {
namespace parquet {

namespace {

// ===== compile-time caps ===================================================

constexpr std::uint32_t kPwMaxRowGroupBytes = 64u * 1024u * 1024u;  // 64 MiB
constexpr std::uint32_t kPwMaxRowGroups     = 4096u;
// Ceiling on concurrent column encodes. Each slot holds a dictionary, a bloom
// builder and the DELTA scratch, so this bounds peak encode memory
// independently of how wide the pool or the schema is.
constexpr std::uint32_t kPwMaxEncodeWave    = 64u;
constexpr std::uint32_t kPwMaxColumns       = kMaxFixedColumns;  // G2FEAT-47

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
constexpr std::int32_t kRepRepeated = 2;
constexpr std::int32_t kConvList    = 3;   // ConvertedType LIST

constexpr std::int32_t kEncPlain    = 0;
constexpr std::int32_t kEncRle      = 3;   // for def-level hybrid
constexpr std::int32_t kEncRleDict  = 8;   // RLE_DICTIONARY data pages
constexpr std::int32_t kEncDeltaBinaryPacked = 5;
constexpr std::int32_t kEncDeltaLenByteArray = 6;
constexpr std::int32_t kEncDeltaByteArray    = 7;
constexpr std::int32_t kEncByteStreamSplit   = 9;

constexpr std::int32_t kCodecUncompressed = 0;
constexpr std::int32_t kCodecSnappy       = 1;
// parquet.thrift CompressionCodec: LZ4 (5) is the DEPRECATED Hadoop-framed
// variant; LZ4_RAW (7) is the bare block format, which is what bolt's
// self-contained codec speaks and what modern writers emit.
constexpr std::int32_t kCodecLz4Raw       = 7;
constexpr std::int32_t kCodecGzip         = 2;

constexpr std::int32_t kConvUtf8    = 0;
constexpr std::int32_t kConvJson    = 24;
constexpr std::int32_t kConvBson    = 25;
constexpr std::int32_t kConvDate    = 6;
constexpr std::int32_t kConvTsMicro = 10;
constexpr std::int32_t kConvDecimal = 5;

constexpr std::int32_t kPageData = 0;
constexpr std::int32_t kPageDict   = 2;
constexpr std::int32_t kPageDataV2 = 3;

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

// Value encoders, page planning, the RLE/bit-packed hybrid and the
// dictionary builder. A size seam, not a public header -- see the file's
// own banner. It needs hybrid_varint_emit and the parquet constants above,
// so it is included here rather than at the top.
#include "bolt_parquet_write_enc.inc"
#include "bolt_parquet_write_bloom.inc"

// ===== writer state =======================================================

// Per-chunk record built up while writing the chunk; flushed into the
// footer at close.
// Min / max as raw little-endian (or BE for Decimal128, raw bytes for
// BYTE_ARRAY) payloads the reader can copy straight into
// PqChunk::min_bytes / max_bytes. Sized to the reader's kPqMaxStatBytes (64)
// so Utf8 stats round-trip; a range whose extreme string is longer than this
// simply omits stats (compute_stats). Shared by chunk Statistics and by the
// per-page bounds a ColumnIndex carries -- they are the same computation over
// a different row range, so they are the same code.
struct StatBuf {
    std::uint8_t  min_buf[64];
    std::uint8_t  max_buf[64];
    std::uint32_t min_len;
    std::uint32_t max_len;
    bool          have_stats;
};

// One data page, as the OffsetIndex / ColumnIndex need to describe it.
struct PageRec {
    std::int64_t  offset;       // file offset of the page header
    std::int32_t  comp_size;    // page header + compressed body
    std::int64_t  first_row;    // row index of the page within its row group
    std::int64_t  null_count;
    bool          all_null;
    StatBuf       st;
};

struct ChunkRec {
    std::int64_t num_values;
    std::int64_t total_unc;
    std::int64_t total_cmp;
    // RELATIVE to the chunk's buffer while encoding; absolute after the emit
    // pass adds the base. -1 means "no data page written yet" -- 0 cannot be
    // that sentinel any more, because the first page starts at relative 0.
    std::int64_t data_page_offset;
    std::int64_t dict_page_offset;   // -1 when the chunk is not dictionary-encoded
    std::int64_t null_count;
    bool         null_count_known;
    bool         dictionary;         // RLE_DICTIONARY data pages
    std::int32_t encoding;           // parquet Encoding code of the data pages
    StatBuf      st;
    // Page index: [page_off, page_off + page_count) into ParquetWriter::pages,
    // and where the two index structs landed in the file (filled at close).
    std::uint32_t page_off;
    std::uint32_t page_count;
    std::int64_t  col_index_off;
    std::int32_t  col_index_len;
    std::int64_t  off_index_off;
    std::int32_t  off_index_len;
    // Bloom filter: written just after the row group, so the offset is
    // patched in after the chunk itself.
    std::int64_t  bloom_off;
    std::int32_t  bloom_len;
};

// One column chunk, encoded but not yet placed in the file. Every offset in
// `rec` and `pages` is RELATIVE to the start of `bytes`; the serial emit pass
// adds the chunk's base file position exactly once.
//
// Encoding into a buffer rather than straight to the sink is what lets the
// per-column encode run off the write path at all -- and it is a better
// serial shape too, since a chunk becomes one sink append instead of two per
// page.
struct ChunkOut {
    std::vector<std::uint8_t> bytes;
    std::vector<PageRec>      pages;
    std::vector<std::uint8_t> bloom;    // serialized filter, empty if none
    ChunkRec                  rec;
    bool                      ok;
};

// All mutable scratch one column-chunk encode needs. One instance per
// concurrent encode slot, never per column: a 256-column schema would
// otherwise hold 256 dictionaries at once.
struct ChunkWorkspace {
    DictBuilder                dict;
    std::vector<std::uint32_t> dict_idx;
    // DELTA encoder scratch (length / prefix / suffix arrays, suffix bytes).
    std::vector<std::int64_t>  i64a;
    std::vector<std::int64_t>  i64b;
    std::vector<std::uint8_t>  bytes;
    BloomBuilder               bloom;
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
    // MEMORY SINK. When set, bytes accumulate in `mem` and `fout` is never
    // opened. Everything else -- page layout, offsets, footer -- is identical,
    // because `file_pos` was already the single source of truth for every
    // offset written into the metadata; the sink only decides where the bytes
    // land. That is what makes a memory-written file byte-identical to the
    // file-written one.
    bool                     to_mem;
    std::vector<std::uint8_t> mem;
    // Page index accumulator: every data page of every chunk, in write
    // order. Only populated when opts.emit_page_index is set. ChunkRec
    // carries the [page_off, page_off + page_count) window into it.
    std::vector<PageRec>     pages;
    // Encode slots. `ws` is one workspace per CONCURRENT encode (a wave), so
    // its length is the pool width, not the column count. `outs` is one
    // encoded chunk per wave slot, placed into the file serially afterwards.
    // Both are grown once and reused for the writer's lifetime.
    std::vector<ChunkWorkspace> ws;
    std::vector<ChunkOut>       outs;
    // Bloom filters for the row group currently being written. Flushed and
    // cleared at the end of each row group so live filter memory is bounded
    // by one row group's columns, not the file's.
    std::vector<std::vector<std::uint8_t>> rg_blooms;   // serialized
    std::vector<std::uint32_t>             rg_bloom_chunk;  // chunk index
};

namespace {

// Append to whichever sink is active. The ONE place that knows the difference.
bool sink_write(ParquetWriter* w, const void* p, std::size_t n) noexcept {
    assert(w != nullptr);
    assert(p != nullptr || n == 0u);
    if (n == 0u) return true;
    if (w->to_mem) {
        const std::uint8_t* b = static_cast<const std::uint8_t*>(p);
        // Allocation failure terminates rather than returning false: bolt
        // builds -fno-exceptions, so there is no throw to catch. That is the
        // SAME exposure the file path already carries -- the footer and every
        // compressed page are accumulated in std::vector here too -- so the
        // memory sink adds no new failure mode, it just holds the bytes longer.
        w->mem.insert(w->mem.end(), b, b + n);
        return true;
    }
    w->fout.write(static_cast<const char*>(p), static_cast<std::streamsize>(n));
    return w->fout.good();
}

bool sink_ok(const ParquetWriter* w) noexcept {
    assert(w != nullptr);
    return w->to_mem ? !w->failed : w->fout.good();
}

// ===== type validation ====================================================

bool type_supported_flat(const ParquetWriteColumn& c) noexcept;

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
        case BoltType::List: {
            // A LIST is only as writable as its ELEMENT, and the element must
            // itself be a flat type -- one level of nesting, mirroring what
            // the reader assembles. A list of lists is refused here rather
            // than producing a schema the reader would then decline.
            if (c.element_type == BoltType::List ||
                c.element_type == BoltType::Struct ||
                c.element_type == BoltType::Map) {
                return false;
            }
            ParquetWriteColumn e{};
            e.type = c.element_type;
            e.scale = c.scale;
            e.precision = c.precision;
            return type_supported_flat(e);
        }
        default:
            return false;
    }
}

// The flat subset, used to validate a LIST's element without recursing into
// another List arm.
bool type_supported_flat(const ParquetWriteColumn& c) noexcept {
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
                        std::int64_t r_begin, std::int64_t r_end,
                        const std::uint8_t* def_bits,
                        bool nullable,
                        std::vector<std::uint8_t>* dst) noexcept {
    assert(dst != nullptr);
    assert(r_begin <= r_end);
    assert(col.data != nullptr || r_begin == r_end);
    const std::uint8_t* src = static_cast<const std::uint8_t*>(col.data);
    for (std::int64_t r = r_begin; r < r_end; ++r) {
        const bool valid = (!nullable) || def_bits[r] != 0;
        if (!valid) continue;
        const std::uint8_t* p = src + static_cast<std::size_t>(r) * elem;
        dst->insert(dst->end(), p, p + elem);
    }
    return true;
}

bool encode_plain_bool(const BoltColumn& col, std::int64_t r_begin,
                       std::int64_t r_end,
                       const std::uint8_t* def_bits, bool nullable,
                       std::vector<std::uint8_t>* dst) noexcept {
    assert(dst != nullptr);
    assert(r_begin <= r_end);
    assert(col.data != nullptr || r_begin == r_end);
    // BoltColumn stores bool as one byte per row (see bolt_types.h line 53).
    const std::uint8_t* src = static_cast<const std::uint8_t*>(col.data);
    // Pack PLAIN bool: bit-packed LE, 8 values per byte.
    std::uint8_t pack = 0;
    std::uint32_t nb = 0;
    for (std::int64_t r = r_begin; r < r_end; ++r) {
        const bool valid = (!nullable) || def_bits[r] != 0;
        if (!valid) continue;
        if (src[r] != 0u) pack = static_cast<std::uint8_t>(pack | (1u << nb));
        ++nb;
        if (nb == 8u) { dst->push_back(pack); pack = 0; nb = 0; }
    }
    if (nb != 0u) dst->push_back(pack);
    return true;
}

bool encode_plain_byte_array(const BoltColumn& col, std::int64_t r_begin,
                             std::int64_t r_end,
                             const std::uint8_t* def_bits, bool nullable,
                             std::vector<std::uint8_t>* dst) noexcept {
    assert(dst != nullptr);
    assert(r_begin <= r_end);
    assert(col.data != nullptr || r_begin == r_end);
    const auto* sv = static_cast<const StringView*>(col.data);
    const auto* spill = static_cast<const std::uint8_t*>(col.str_overflow_base);
    for (std::int64_t r = r_begin; r < r_end; ++r) {
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
bool compute_stats_utf8(const BoltColumn& col, std::int64_t r_begin,
                        std::int64_t r_end,
                        const std::uint8_t* def_bits, bool nullable,
                        StatBuf* rec) noexcept {
    assert(rec != nullptr);
    assert(r_begin <= r_end);
    assert(col.data != nullptr || r_begin == r_end);
    const auto* sv    = static_cast<const StringView*>(col.data);
    const auto* spill = static_cast<const std::uint8_t*>(col.str_overflow_base);
    const std::uint8_t* mn_p = nullptr; std::uint32_t mn_n = 0;
    const std::uint8_t* mx_p = nullptr; std::uint32_t mx_n = 0;
    for (std::int64_t r = r_begin; r < r_end; ++r) {
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

bool compute_stats(const BoltColumn& col, std::int64_t r_begin,
                   std::int64_t r_end,
                   const std::uint8_t* def_bits, bool nullable,
                   StatBuf* rec) noexcept {
    assert(rec != nullptr);
    assert(r_begin <= r_end);
    rec->have_stats = false;
    rec->min_len = rec->max_len = 0u;
    if (r_begin == r_end) return true;
    const auto valid_at = [&](std::int64_t r) noexcept {
        return (!nullable) || def_bits[r] != 0;
    };
    // Find first valid row in the range.
    std::int64_t r0 = -1;
    for (std::int64_t r = r_begin; r < r_end; ++r) {
        if (valid_at(r)) { r0 = r; break; }
    }
    if (r0 < 0) return true;   // all null -> no min/max
    switch (col.type) {
        case BoltType::Int32:
        case BoltType::Date32: {
            const auto* p = static_cast<const std::int32_t*>(col.data);
            std::int32_t mn = p[r0], mx = p[r0];
            for (std::int64_t r = r0 + 1; r < r_end; ++r) {
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
            for (std::int64_t r = r0 + 1; r < r_end; ++r) {
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
            for (std::int64_t r = r0; r < r_end; ++r) {
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
            for (std::int64_t r = r0; r < r_end; ++r) {
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
            for (std::int64_t r = r0; r < r_end; ++r) {
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
            return compute_stats_utf8(col, r_begin, r_end, def_bits, nullable, rec);
        default:
            // Decimal128: min/max skipped (v1 gap, documented in the header).
            return true;
    }
}

// ===== compression =========================================================

// Writer codec ids (ParquetWriteOpts::compression) -> parquet CompressionCodec.
constexpr std::uint8_t kPwCodecNone   = 0;
constexpr std::uint8_t kPwCodecSnappy = 1;
constexpr std::uint8_t kPwCodecLz4Raw = 4;
constexpr std::uint8_t kPwCodecGzip   = 2;

std::int32_t pq_codec_code(std::uint8_t c) noexcept {
    switch (c) {
        case kPwCodecSnappy: return kCodecSnappy;
        case kPwCodecLz4Raw: return kCodecLz4Raw;
        case kPwCodecGzip:   return kCodecGzip;
        default:             return kCodecUncompressed;
    }
}

bool pw_codec_supported(std::uint8_t c) noexcept {
    return c == kPwCodecNone || c == kPwCodecSnappy ||
           c == kPwCodecLz4Raw || c == kPwCodecGzip;
}


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
    if (codec == kPwCodecGzip) {
        const std::size_t cap = static_cast<std::size_t>(gzip_bound(src_len));
        dst->resize(cap);
        std::uint64_t out_len = 0;
        // The DeflateState is 256 KiB, far too big for the stack and pointless
        // to reallocate per page; one per thread, reused for the process.
        // Thread-local rather than shared because parallel column encoding
        // runs this concurrently.
        static thread_local DeflateState gz_state;
        if (!gzip_compress(src, src_len, dst->data(), cap, &out_len,
                           &gz_state)) {
            return false;
        }
        dst->resize(static_cast<std::size_t>(out_len));
        return true;
    }
    if (codec == kPwCodecLz4Raw) {
        const std::size_t cap =
            static_cast<std::size_t>(lz4_raw_bound(src_len));
        dst->resize(cap);
        std::uint64_t out_len = 0;
        if (!lz4_raw_compress(src, src_len, dst->data(), cap, &out_len)) {
            return false;
        }
        dst->resize(static_cast<std::size_t>(out_len));
        return true;
    }
    return false;
}

// ColumnIndex / OffsetIndex emission. Needs the thrift codec, stat_lex_cmp,
// PageRec and ParquetWriter, so it is included here rather than at the top.
#include "bolt_parquet_write_index.inc"

// ===== page header writer =================================================

// PageHeader for a v1 DATA_PAGE. `num_values` counts rows INCLUDING nulls --
// it is the number of definition levels, which is what a reader advances the
// row cursor by.
void write_page_header(std::vector<std::uint8_t>* hdr,
                       std::int32_t unc, std::int32_t cmp,
                       std::int32_t num_values,
                       std::int32_t encoding) noexcept {
    assert(hdr != nullptr);
    assert(unc >= 0 && cmp >= 0);
    TcOut o{hdr};
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
        tc_put_zigzag(&o, encoding);
        tc_put_field(&o, 3, kFI32);
        tc_put_zigzag(&o, kEncRle);     // def-level encoding
        tc_put_field(&o, 4, kFI32);
        tc_put_zigzag(&o, kEncRle);     // rep-level encoding (unused; flat)
        tc_put_stop(&o);
    }
    tc_put_stop(&o);
}

// PageHeader for a DATA_PAGE_V2. The level byte lengths are what make this
// format worth having: a reader knows exactly how many bytes to step over
// without decompressing anything, because v2 stores the levels RAW ahead of
// the (separately compressed) values.
//
// `unc` and `cmp` both INCLUDE the level bytes -- the levels are not
// compressed, so they contribute their own length to each. Getting that wrong
// makes a reader compute a negative values length.
void write_page_header_v2(std::vector<std::uint8_t>* hdr,
                          std::int32_t unc, std::int32_t cmp,
                          std::int32_t num_values, std::int32_t num_nulls,
                          std::int32_t num_rows, std::int32_t encoding,
                          std::int32_t def_len, std::int32_t rep_len,
                          bool is_compressed) noexcept {
    assert(hdr != nullptr);
    assert(unc >= 0 && cmp >= 0);
    assert(def_len >= 0 && rep_len >= 0);
    assert(num_nulls >= 0 && num_nulls <= num_values);
    TcOut o{hdr};
    tc_put_field(&o, 1, kFI32);
    tc_put_zigzag(&o, kPageDataV2);
    tc_put_field(&o, 2, kFI32);
    tc_put_zigzag(&o, unc);
    tc_put_field(&o, 3, kFI32);
    tc_put_zigzag(&o, cmp);
    // DataPageHeaderV2 (field 8)
    tc_put_field(&o, 8, kFStruct);
    {
        tc_put_field(&o, 1, kFI32);
        tc_put_zigzag(&o, num_values);
        tc_put_field(&o, 2, kFI32);
        tc_put_zigzag(&o, num_nulls);
        tc_put_field(&o, 3, kFI32);
        tc_put_zigzag(&o, num_rows);
        tc_put_field(&o, 4, kFI32);
        tc_put_zigzag(&o, encoding);
        tc_put_field(&o, 5, kFI32);
        tc_put_zigzag(&o, def_len);
        tc_put_field(&o, 6, kFI32);
        tc_put_zigzag(&o, rep_len);
        // 7 is_compressed: a thrift compact bool inside a struct is carried
        // BY the field header type, with no separate value byte.
        tc_put_field(&o, 7, is_compressed ? kFTrue : kFFalse);
        tc_put_stop(&o);
    }
    tc_put_stop(&o);
}

// PageHeader for a DICTIONARY_PAGE. The dictionary itself is PLAIN-encoded;
// the modern encoding code for that is PLAIN (0). parquet-mr historically
// wrote PLAIN_DICTIONARY (2) here and readers accept both -- bolt's reader
// ignores the field for dictionary pages entirely (decode_dict_page keys off
// the physical type), and pyarrow accepts PLAIN.
void write_dict_page_header(std::vector<std::uint8_t>* hdr,
                            std::int32_t unc, std::int32_t cmp,
                            std::int32_t num_values) noexcept {
    assert(hdr != nullptr);
    assert(unc >= 0 && cmp >= 0);
    TcOut o{hdr};
    tc_put_field(&o, 1, kFI32);
    tc_put_zigzag(&o, kPageDict);
    tc_put_field(&o, 2, kFI32);
    tc_put_zigzag(&o, unc);
    tc_put_field(&o, 3, kFI32);
    tc_put_zigzag(&o, cmp);
    // DictionaryPageHeader (field 7)
    tc_put_field(&o, 7, kFStruct);
    {
        tc_put_field(&o, 1, kFI32);
        tc_put_zigzag(&o, num_values);
        tc_put_field(&o, 2, kFI32);
        tc_put_zigzag(&o, kEncPlain);
        tc_put_field(&o, 3, kFFalse);   // is_sorted = false
        tc_put_stop(&o);
    }
    tc_put_stop(&o);
}

// ===== chunk writer (one column of one row group) ========================

// Effective, clamped page / dictionary budgets. 0 means "the default", never
// "unlimited" -- see the header: an unlimited page overflows the int32 size
// fields in the page header and emits a corrupt file.
std::uint32_t page_budget_bytes(const ParquetWriteOpts& o) noexcept {
    std::uint32_t b = (o.data_page_target_bytes != 0u) ? o.data_page_target_bytes
                                                       : kPwDefaultPageBytes;
    if (b < kPwMinPageBytes) b = kPwMinPageBytes;
    if (b > kPwMaxPageBytes) b = kPwMaxPageBytes;
    assert(b >= kPwMinPageBytes && b <= kPwMaxPageBytes);
    return b;
}

std::uint32_t dict_budget_bytes(const ParquetWriteOpts& o) noexcept {
    const std::uint32_t b = (o.dictionary_max_bytes != 0u)
        ? o.dictionary_max_bytes : kPwDefaultDictBytes;
    assert(b > 0u);
    return b;
}

// BOOLEAN is deliberately excluded: a two-entry dictionary plus an index
// stream is strictly larger than the one-bit-per-value PLAIN form, which is
// why parquet-mr and Arrow never dictionary-encode it either.
bool dict_eligible(BoltType t) noexcept {
    switch (t) {
        case BoltType::Int32:
        case BoltType::Date32:
        case BoltType::Float32:
        case BoltType::Int64:
        case BoltType::Timestamp:
        case BoltType::Float64:
        case BoltType::Decimal128:
        case BoltType::Utf8:
        case BoltType::Binary:
            return true;
        default:
            return false;
    }
}

// Which encodings a type can carry. Consulted at open (to reject a caller's
// impossible request loudly) and when resolving Auto.
bool encoding_applies(PqWriteEncoding e, BoltType t) noexcept {
    switch (e) {
        case PqWriteEncoding::Auto:
        case PqWriteEncoding::Plain:
            return true;
        case PqWriteEncoding::Dictionary:
            return dict_eligible(t);
        case PqWriteEncoding::DeltaBinaryPacked:
            // INT32/INT64 physical only -- Date32 and Timestamp are those.
            return t == BoltType::Int32 || t == BoltType::Int64 ||
                   t == BoltType::Date32 || t == BoltType::Timestamp;
        case PqWriteEncoding::DeltaLengthByteArray:
        case PqWriteEncoding::DeltaByteArray:
            return t == BoltType::Utf8 || t == BoltType::Binary;
        case PqWriteEncoding::ByteStreamSplit:
            // FLOAT/DOUBLE, plus the fixed-width integer and FLBA types the
            // spec added in 2.9 -- which is what bolt's reader accepts.
            return t == BoltType::Float32 || t == BoltType::Float64 ||
                   t == BoltType::Int32 || t == BoltType::Int64 ||
                   t == BoltType::Date32 || t == BoltType::Timestamp ||
                   t == BoltType::Decimal128;
    }
    return false;
}

// Resolve a column's encoding. Auto is the ONLY value that consults
// use_dictionary; naming an encoding overrides it, matching Arrow.
PqWriteEncoding resolve_encoding(const ParquetWriteColumn& sch,
                                 const ParquetWriteOpts& o) noexcept {
    const PqWriteEncoding want = static_cast<PqWriteEncoding>(sch.encoding);
    if (want != PqWriteEncoding::Auto) return want;
    if (o.use_dictionary && dict_eligible(sch.type)) {
        return PqWriteEncoding::Dictionary;
    }
    return PqWriteEncoding::Plain;
}

// The parquet Encoding code a resolved encoding writes into its data pages.
std::int32_t pq_encoding_code(PqWriteEncoding e) noexcept {
    switch (e) {
        case PqWriteEncoding::Dictionary:           return kEncRleDict;
        case PqWriteEncoding::DeltaBinaryPacked:    return kEncDeltaBinaryPacked;
        case PqWriteEncoding::DeltaLengthByteArray: return kEncDeltaLenByteArray;
        case PqWriteEncoding::DeltaByteArray:       return kEncDeltaByteArray;
        case PqWriteEncoding::ByteStreamSplit:      return kEncByteStreamSplit;
        default:                                    return kEncPlain;
    }
}

// PLAIN-encode rows [r0, r1) of one column.
bool encode_plain_range(const BoltColumn& col, const ParquetWriteColumn& sch,
                        std::int64_t r0, std::int64_t r1,
                        const std::uint8_t* def_bits, bool nullable,
                        std::vector<std::uint8_t>* dst) noexcept {
    assert(dst != nullptr);
    assert(r0 <= r1);
    switch (sch.type) {
        case BoltType::Bool:
            return encode_plain_bool(col, r0, r1, def_bits, nullable, dst);
        case BoltType::Int32:
        case BoltType::Date32:
        case BoltType::Float32:
            return encode_plain_fixed(col, 4, r0, r1, def_bits, nullable, dst);
        case BoltType::Int64:
        case BoltType::Timestamp:
        case BoltType::Float64:
            return encode_plain_fixed(col, 8, r0, r1, def_bits, nullable, dst);
        case BoltType::Utf8:
        case BoltType::Binary:
            return encode_plain_byte_array(col, r0, r1, def_bits, nullable, dst);
        case BoltType::Decimal128: {
            const auto* p = static_cast<const std::uint8_t*>(col.data);
            if (p == nullptr && r0 != r1) return false;
            for (std::int64_t r = r0; r < r1; ++r) {
                if (nullable && def_bits[r] == 0u) continue;
                std::uint8_t be[16];
                decimal_to_be(p + r * 16, be);
                dst->insert(dst->end(), be, be + 16);
            }
            return true;
        }
        default:
            return false;
    }
}

// Prepend the def-level stream to `values` to form the final page payload.
// Layout (v1): [u32 LE hybrid byte length][hybrid bytes][values]. A REQUIRED
// column has no def levels and the payload is the values verbatim.
// v1 prefixes the def-level stream with its own u32 byte length; v2 does not,
// because DataPageHeaderV2 carries definition_levels_byte_length instead.
// `*out_level_bytes` reports how many leading bytes of the payload are levels
// -- v2 needs that both for the header and to know what NOT to compress.
bool build_page_payload(const std::vector<std::uint8_t>& values,
                        const std::uint8_t* def_bits, bool nullable,
                        std::int64_t r0, std::int64_t r1, bool v2,
                        std::vector<std::uint8_t>* payload,
                        std::uint32_t* out_level_bytes) noexcept {
    assert(payload != nullptr && out_level_bytes != nullptr);
    assert(r0 <= r1);
    payload->clear();
    *out_level_bytes = 0;
    if (nullable) {
        assert(def_bits != nullptr);
        std::vector<std::uint8_t> def_hybrid;
        const std::uint32_t n = static_cast<std::uint32_t>(r1 - r0);
        if (!def_levels_encode(def_bits + r0, n, &def_hybrid)) return false;
        const std::uint32_t dlen = static_cast<std::uint32_t>(def_hybrid.size());
        if (!v2) {
            const std::uint8_t lb[4] = {
                static_cast<std::uint8_t>(dlen & 0xFFu),
                static_cast<std::uint8_t>((dlen >> 8) & 0xFFu),
                static_cast<std::uint8_t>((dlen >> 16) & 0xFFu),
                static_cast<std::uint8_t>((dlen >> 24) & 0xFFu),
            };
            payload->insert(payload->end(), lb, lb + 4);
        }
        payload->insert(payload->end(), def_hybrid.begin(), def_hybrid.end());
        if (v2) *out_level_bytes = dlen;
    }
    payload->insert(payload->end(), values.begin(), values.end());
    return true;
}

// Compress `payload`, write [header][body], and fold the page's sizes into
// `rec`. `dict_page` selects the dictionary page header. Fails closed if the
// page would overflow the int32 size fields rather than truncating them --
// the bug this page-splitting work exists to remove.
// How to write one page. Grouped into a struct because v2 needs five more
// facts than v1 and threading them as positional arguments was already at the
// limit of readable.
struct PageSpec {
    bool          dict_page;
    bool          v2;
    std::int64_t  num_values;    // rows in the page, INCLUDING nulls
    std::int64_t  num_nulls;
    std::int32_t  encoding;
    std::uint32_t level_bytes;   // v2: leading payload bytes that are levels
};

bool emit_page(ParquetWriter* w, ChunkOut* out,
               const std::vector<std::uint8_t>& payload, const PageSpec& spec,
               ChunkRec* rec,
               std::int64_t* out_offset, std::int32_t* out_size) noexcept {
    assert(w != nullptr && rec != nullptr && out != nullptr);
    assert(spec.num_values >= 0 && spec.num_nulls >= 0);
    assert(spec.level_bytes <= payload.size());
    if (payload.size() > 0x7FFFFFFFull) return false;
    if (spec.num_values > 0x7FFFFFFF) return false;
    const std::int32_t unc = static_cast<std::int32_t>(payload.size());

    // v2 compresses ONLY the values: the levels are stored raw so a reader can
    // step over them without inflating anything. So the codec is applied from
    // level_bytes onward and the level bytes are copied through verbatim.
    const std::uint32_t lv = spec.v2 ? spec.level_bytes : 0u;
    std::vector<std::uint8_t> compressed;
    if (!maybe_compress(payload.data() + lv, payload.size() - lv,
                        w->opts.compression, &compressed)) {
        return false;
    }
    if (compressed.size() + lv > 0x7FFFFFFFull) return false;
    const std::int32_t cmp =
        static_cast<std::int32_t>(compressed.size() + lv);

    std::vector<std::uint8_t> hdr;
    if (spec.dict_page) {
        write_dict_page_header(&hdr, unc, cmp,
                               static_cast<std::int32_t>(spec.num_values));
    } else if (spec.v2) {
        write_page_header_v2(
            &hdr, unc, cmp, static_cast<std::int32_t>(spec.num_values),
            static_cast<std::int32_t>(spec.num_nulls),
            static_cast<std::int32_t>(spec.num_values),   // flat: rows == values
            spec.encoding, static_cast<std::int32_t>(lv), 0,
            w->opts.compression != 0u);
    } else {
        write_page_header(&hdr, unc, cmp,
                          static_cast<std::int32_t>(spec.num_values),
                          spec.encoding);
    }

    // Offsets are relative to the chunk buffer; the emit pass rebases them.
    const std::int64_t off = static_cast<std::int64_t>(out->bytes.size());
    out->bytes.insert(out->bytes.end(), hdr.begin(), hdr.end());
    if (lv != 0u) {
        out->bytes.insert(out->bytes.end(), payload.begin(),
                          payload.begin() + lv);
    }
    out->bytes.insert(out->bytes.end(), compressed.begin(), compressed.end());
    const std::size_t total = hdr.size() + compressed.size() + lv;
    rec->total_unc += static_cast<std::int64_t>(hdr.size()) + unc;
    rec->total_cmp += static_cast<std::int64_t>(hdr.size()) + cmp;
    if (out_offset != nullptr) *out_offset = off;
    if (out_size != nullptr) *out_size = static_cast<std::int32_t>(total);
    return true;
}

// Record one data page in the writer's page table (consumed at close by the
// ColumnIndex / OffsetIndex writers). Only populated when the caller asked
// for a page index; otherwise the table stays empty and costs nothing.
void note_page(ParquetWriter* w, ChunkOut* out, ChunkRec* rec,
               std::int64_t offset, std::int32_t size, std::int64_t first_row,
               std::int64_t nulls, std::int64_t n_vals,
               const StatBuf& st) noexcept {
    assert(w != nullptr && rec != nullptr && out != nullptr);
    assert(nulls >= 0 && n_vals >= 0);
    if (!w->opts.emit_page_index) return;
    PageRec pr;
    std::memset(&pr, 0, sizeof(pr));
    pr.offset = offset;
    pr.comp_size = size;
    pr.first_row = first_row;
    pr.null_count = nulls;
    pr.all_null = (n_vals > 0) && (nulls == n_vals);
    pr.st = st;
    out->pages.push_back(pr);
    ++rec->page_count;
}

// Count nulls in [r0, r1). Branch-free accumulate over the def bits.
std::int64_t count_nulls(const std::uint8_t* def_bits, bool nullable,
                         std::int64_t r0, std::int64_t r1) noexcept {
    assert(r0 <= r1);
    if (!nullable) return 0;
    assert(def_bits != nullptr);
    std::int64_t n = 0;
    for (std::int64_t r = r0; r < r1; ++r) n += (def_bits[r] == 0u);
    return n;
}

// Guard matching the reader's own per-chunk page ceiling
// (kPqMaxPagesPerChunk). At the 1 MiB default that is 64 GiB in one column
// chunk; a chunk that large should have been split into row groups. Fail
// closed rather than emit a file bolt itself cannot read back.
constexpr std::uint32_t kPwMaxPagesPerChunk = 1u << 16;

// Encode rows [r0, r1) with a non-dictionary encoding.
bool encode_direct_range(ParquetWriter* w, ChunkWorkspace* ws,
                         const BoltColumn& col,
                         const ParquetWriteColumn& sch, PqWriteEncoding enc,
                         std::int64_t r0, std::int64_t r1,
                         const std::uint8_t* def_bits, bool nullable,
                         std::vector<std::uint8_t>* dst) noexcept {
    assert(w != nullptr && dst != nullptr);
    assert(r0 <= r1);
    switch (enc) {
        case PqWriteEncoding::ByteStreamSplit:
            return encode_byte_stream_split(col, sch.type, r0, r1, def_bits,
                                            nullable, dst);
        case PqWriteEncoding::DeltaBinaryPacked: {
            if (gather_ints(col, sch.type, r0, r1, def_bits, nullable,
                            &ws->i64a) < 0) {
                return false;
            }
            return delta_encode_ints(
                ws->i64a.data(),
                static_cast<std::uint32_t>(ws->i64a.size()), dst);
        }
        case PqWriteEncoding::DeltaLengthByteArray:
            return encode_delta_length_byte_array(col, r0, r1, def_bits,
                                                  nullable, &ws->i64a,
                                                  dst);
        case PqWriteEncoding::DeltaByteArray:
            return encode_delta_byte_array(col, r0, r1, def_bits, nullable,
                                           &ws->i64a, &ws->i64b,
                                           &ws->bytes, dst);
        default:
            return encode_plain_range(col, sch, r0, r1, def_bits, nullable, dst);
    }
}

// Non-dictionary data pages for the whole chunk. Page planning uses the PLAIN
// size for every encoding: exact for PLAIN and BYTE_STREAM_SPLIT (a transpose
// does not change the byte count), and an UPPER bound for the DELTA family,
// whose whole purpose is to be smaller. So a delta page lands at or under the
// budget -- conservative, never over.
bool chunk_write_direct(ParquetWriter* w, ChunkWorkspace* ws, ChunkOut* out,
                        const BoltColumn& col,
                        const ParquetWriteColumn& sch, PqWriteEncoding enc,
                        std::int64_t n_rows,
                        const std::uint8_t* def_bits, bool nullable,
                        ChunkRec* rec) noexcept {
    assert(w != nullptr && rec != nullptr);
    assert(n_rows >= 0);
    const std::uint64_t budget = page_budget_bytes(w->opts);
    const std::int32_t enc_code = pq_encoding_code(enc);
    rec->encoding = enc_code;
    std::vector<std::uint8_t> vals, payload;
    std::int64_t r0 = 0;
    std::uint32_t pages = 0;
    for (;;) {
        if (++pages > kPwMaxPagesPerChunk) return false;
        const std::int64_t r1 = (r0 < n_rows)
            ? plan_page_rows(col, sch.type, r0, n_rows, def_bits, nullable,
                             budget, /*bits_per_value=*/0u)
            : r0;
        assert(r1 >= r0 && r1 <= n_rows);
        vals.clear();
        if (!encode_direct_range(w, ws, col, sch, enc, r0, r1, def_bits, nullable,
                                 &vals)) {
            return false;
        }
        std::uint32_t lv = 0;
        if (!build_page_payload(vals, def_bits, nullable, r0, r1,
                                w->opts.data_page_v2, &payload, &lv)) {
            return false;
        }
        StatBuf ps;
        std::memset(&ps, 0, sizeof(ps));
        if (w->opts.emit_page_index &&
            !compute_stats(col, r0, r1, def_bits, nullable, &ps)) {
            return false;
        }
        const std::int64_t page_nulls = count_nulls(def_bits, nullable, r0, r1);
        std::int64_t off = 0;
        std::int32_t sz = 0;
        PageSpec spec{false, w->opts.data_page_v2, r1 - r0, page_nulls,
                      enc_code, lv};
        if (!emit_page(w, out, payload, spec, rec, &off, &sz)) return false;
        if (rec->data_page_offset < 0) rec->data_page_offset = off;
        note_page(w, out, rec, off, sz, r0, page_nulls, r1 - r0, ps);
        r0 = r1;
        if (r0 >= n_rows) break;
    }
    return true;
}

// Intern every non-null value of [0, n_rows) into `d`, appending each row's
// index to `idx` (one entry per NON-NULL row, in row order). Returns false
// when the dictionary exceeded its bounds -- the caller then writes PLAIN.
bool dict_build_chunk(const BoltColumn& col, const ParquetWriteColumn& sch,
                      std::int64_t n_rows, const std::uint8_t* def_bits,
                      bool nullable, std::uint32_t max_bytes,
                      DictBuilder* d, std::vector<std::uint32_t>* idx) noexcept {
    assert(d != nullptr && idx != nullptr);
    assert(n_rows >= 0);
    const bool var_len = (sch.type == BoltType::Utf8 ||
                          sch.type == BoltType::Binary);
    dict_init(d, max_bytes, var_len, n_rows);
    idx->clear();
    idx->reserve(static_cast<std::size_t>(n_rows));
    const std::size_t elem = plain_elem_width(sch.type);
    const bool is_dec = (sch.type == BoltType::Decimal128);
    for (std::int64_t r = 0; r < n_rows; ++r) {
        if (nullable && def_bits[r] == 0u) continue;
        std::uint32_t id;
        if (var_len) {
            const RowVal v = row_val_bytes(col, r);
            if (v.n == 0xFFFFFFFFu) return false;   // spilled with no base
            id = dict_intern(d, v.p, v.n);
        } else if (is_dec) {
            std::uint8_t be[16];
            decimal_to_be(static_cast<const std::uint8_t*>(col.data) + r * 16,
                          be);
            id = dict_intern(d, be, 16u);
        } else {
            assert(elem != 0u);
            const RowVal v = row_val_fixed(col, elem, r);
            id = dict_intern(d, v.p, v.n);
        }
        if (id == 0xFFFFFFFFu) return false;        // overflow -> PLAIN
        idx->push_back(id);
    }
    return true;
}

// DICTIONARY_PAGE + RLE_DICTIONARY data pages for the whole chunk.
bool chunk_write_dict(ParquetWriter* w, ChunkWorkspace* ws, ChunkOut* out,
                      const BoltColumn& col,
                      const ParquetWriteColumn& sch, std::int64_t n_rows,
                      const std::uint8_t* def_bits, bool nullable,
                      const DictBuilder& d,
                      const std::vector<std::uint32_t>& idx,
                      ChunkRec* rec) noexcept {
    assert(w != nullptr && rec != nullptr);
    assert(d.count > 0u);
    rec->dictionary = true;
    // Dictionary page first: its offset is the chunk's start, and the reader
    // requires it to precede the data pages that reference it.
    std::vector<std::uint8_t> dict_payload(d.plain.begin(), d.plain.end());
    std::int64_t doff = 0;
    // A dictionary page has no v2 variant -- the format defines only
    // DATA_PAGE_V2 -- so it is written the same way regardless.
    PageSpec dspec{true, false, d.count, 0, kEncPlain, 0u};
    if (!emit_page(w, out, dict_payload, dspec, rec, &doff, nullptr)) {
        return false;
    }
    rec->dict_page_offset = doff;

    const std::uint32_t bw = bit_width_for(d.count - 1u);
    const std::uint64_t budget = page_budget_bytes(w->opts);
    // A zero-bit-width dictionary (one distinct value) still costs a run
    // header per page; size such pages by row count instead of by bits.
    const std::uint32_t bits = (bw != 0u) ? bw : 1u;

    std::vector<std::uint8_t> vals, payload;
    std::int64_t r0 = 0;
    std::size_t vi = 0;            // cursor into idx (non-null rows so far)
    std::uint32_t pages = 0;
    for (;;) {
        if (++pages > kPwMaxPagesPerChunk) return false;
        const std::int64_t r1 = (r0 < n_rows)
            ? plan_page_rows(col, sch.type, r0, n_rows, def_bits, nullable,
                             budget, bits)
            : r0;
        assert(r1 >= r0 && r1 <= n_rows);
        const std::int64_t nv = r1 - r0 - count_nulls(def_bits, nullable, r0, r1);
        assert(nv >= 0);
        assert(vi + static_cast<std::size_t>(nv) <= idx.size());
        vals.clear();
        vals.push_back(static_cast<std::uint8_t>(bw));
        rle_hybrid_encode(idx.data() + vi, static_cast<std::size_t>(nv), bw,
                          &vals);
        vi += static_cast<std::size_t>(nv);
        std::uint32_t lv = 0;
        if (!build_page_payload(vals, def_bits, nullable, r0, r1,
                                w->opts.data_page_v2, &payload, &lv)) {
            return false;
        }
        StatBuf ps;
        std::memset(&ps, 0, sizeof(ps));
        if (w->opts.emit_page_index &&
            !compute_stats(col, r0, r1, def_bits, nullable, &ps)) {
            return false;
        }
        std::int64_t off = 0;
        std::int32_t sz = 0;
        PageSpec spec{false, w->opts.data_page_v2, r1 - r0, (r1 - r0) - nv,
                      kEncRleDict, lv};
        if (!emit_page(w, out, payload, spec, rec, &off, &sz)) return false;
        if (rec->data_page_offset < 0) rec->data_page_offset = off;
        note_page(w, out, rec, off, sz, r0, (r1 - r0) - nv, r1 - r0, ps);
        r0 = r1;
        if (r0 >= n_rows) break;
    }
    assert(vi == idx.size());
    return true;
}

// Build this chunk's bloom filter and queue it for the flush that happens
// after the row group. `ndv_exact` says the dictionary count is the real
// distinct-value count; otherwise the non-null row count is used, which
// over-estimates and so over-SIZES the filter -- the safe direction, since
// an under-sized filter is over-full and its false-positive rate degrades.
bool chunk_build_bloom(ParquetWriter* w, ChunkWorkspace* ws, ChunkOut* out,
                       const BoltColumn& col,
                       const ParquetWriteColumn& sch, std::int64_t n_rows,
                       const std::uint8_t* def_bits, bool nullable,
                       bool ndv_exact, ChunkRec* rec) noexcept {
    assert(w != nullptr && rec != nullptr && out != nullptr);
    assert(n_rows >= 0);
    if (!w->opts.emit_bloom_filter) return true;
    if (!bloom_eligible(sch.type)) return true;
    const std::int64_t non_null = n_rows - rec->null_count;
    if (non_null <= 0) return true;               // nothing to assert absent
    const std::uint64_t ndv = ndv_exact
        ? static_cast<std::uint64_t>(ws->dict.count)
        : static_cast<std::uint64_t>(non_null);
    std::uint32_t cap = (w->opts.bloom_max_bytes != 0u)
        ? w->opts.bloom_max_bytes : kPwDefaultBloomBytes;
    if (cap < kPwBloomMinBytes) cap = kPwBloomMinBytes;
    const double fpp = (w->opts.bloom_fpp > 0.0 && w->opts.bloom_fpp < 1.0)
        ? w->opts.bloom_fpp : kPwDefaultBloomFpp;
    bloom_reset(&ws->bloom, bloom_optimal_bytes(ndv, fpp, cap));
    if (!bloom_build_chunk(col, sch.type, n_rows, def_bits, nullable,
                           &ws->bloom)) {
        return false;
    }
    bloom_serialize(ws->bloom, &out->bloom);
    return true;
}

// ---- LIST chunks ---------------------------------------------------------
//
// The inverse of the reader's Dremel assembly. For the 3-level shape the
// schema writer emits, list_def == 1 and rep_def == 2 and max_def == 2 or 3
// depending on element nullability, so per row:
//
//   NULL list   -> one slot, rep 0, def 0
//   EMPTY list  -> one slot, rep 0, def 1        (present, no elements)
//   n elements  -> n slots, rep 0 then 1..1,
//                  def 2 for a NULL element, 3 for a present one
//
// An empty list and a null list are different values that both occupy one
// leaf slot and carry no element; only the definition level separates them,
// which is why they are spelled out here rather than derived.
struct ListLevels {
    std::vector<std::uint8_t> rep;      // one per leaf slot
    std::vector<std::uint8_t> def;
    std::vector<std::int64_t> row_slot; // row -> first leaf slot (rows+1)
    std::int64_t n_elems;
};

bool build_list_levels(const BoltColumn& col, const ParquetWriteColumn& sch,
                       std::int64_t n_rows, ListLevels* out) noexcept {
    assert(out != nullptr);
    if (col.format != ColumnFormat::Nested) return false;
    if (col.type != BoltType::List && col.type != BoltType::Map) return false;
    const std::int32_t* offs = col.list_offsets();
    const BoltColumn* elem = col.list_element();
    if (offs == nullptr || elem == nullptr) return false;

    const std::uint8_t max_def =
        static_cast<std::uint8_t>(sch.element_nullable ? 3u : 2u);
    out->rep.clear();
    out->def.clear();
    out->row_slot.assign(static_cast<std::size_t>(n_rows) + 1u, 0);
    out->n_elems = 0;
    for (std::int64_t r = 0; r < n_rows; ++r) {
        out->row_slot[static_cast<std::size_t>(r)] =
            static_cast<std::int64_t>(out->rep.size());
        const bool present = (col.validity == nullptr) ||
            (((col.validity[(col.validity_offset + r) >> 3] >>
               ((col.validity_offset + r) & 7)) & 1u) != 0u);
        const std::int32_t lo = offs[r];
        const std::int32_t hi = offs[r + 1];
        if (hi < lo) return false;
        if (!present) {                       // NULL list
            out->rep.push_back(0u);
            out->def.push_back(0u);
            continue;
        }
        if (hi == lo) {                       // EMPTY list, present
            out->rep.push_back(0u);
            out->def.push_back(1u);
            continue;
        }
        for (std::int32_t e = lo; e < hi; ++e) {
            out->rep.push_back(static_cast<std::uint8_t>((e == lo) ? 0u : 1u));
            bool evalid = true;
            if (sch.element_nullable && elem->validity != nullptr) {
                const std::int64_t bit = elem->validity_offset + e;
                evalid = ((elem->validity[bit >> 3] >> (bit & 7)) & 1u) != 0u;
            }
            out->def.push_back(evalid ? max_def
                                      : static_cast<std::uint8_t>(2u));
            ++out->n_elems;
        }
    }
    out->row_slot[static_cast<std::size_t>(n_rows)] =
        static_cast<std::int64_t>(out->rep.size());
    return true;
}

// RLE/bit-packed hybrid for a level stream, prefixed with its u32 byte length
// (the v1 layout). Levels use the minimum bit width for their max.
bool encode_level_stream(const std::uint8_t* lv, std::size_t n,
                         std::uint32_t max_lvl,
                         std::vector<std::uint8_t>* out) noexcept {
    assert(out != nullptr);
    std::vector<std::uint32_t> tmp(n);
    for (std::size_t i = 0; i < n; ++i) tmp[i] = lv[i];
    std::vector<std::uint8_t> body;
    rle_hybrid_encode(tmp.data(), n, bit_width_for(max_lvl), &body);
    const std::uint32_t blen = static_cast<std::uint32_t>(body.size());
    const std::uint8_t pfx[4] = {
        static_cast<std::uint8_t>(blen & 0xFFu),
        static_cast<std::uint8_t>((blen >> 8) & 0xFFu),
        static_cast<std::uint8_t>((blen >> 16) & 0xFFu),
        static_cast<std::uint8_t>((blen >> 24) & 0xFFu),
    };
    out->insert(out->end(), pfx, pfx + 4);
    out->insert(out->end(), body.begin(), body.end());
    return true;
}

// One LIST column chunk. Pages are cut on ROW boundaries -- a data page must
// contain whole records, so a page may never start mid-list.
bool chunk_write_list(ParquetWriter* w, ChunkWorkspace* ws, ChunkOut* out,
                      const BoltColumn& col, const ParquetWriteColumn& sch,
                      std::int64_t n_rows, ChunkRec* rec) noexcept {
    assert(w != nullptr && rec != nullptr && out != nullptr);
    ListLevels lv;
    if (!build_list_levels(col, sch, n_rows, &lv)) return false;
    const BoltColumn* elem = col.list_element();
    const std::uint8_t max_def = static_cast<std::uint8_t>(
        sch.element_nullable ? 3u : 2u);

    // The chunk describes LEAF values, so num_values counts slots and
    // null_count counts slots carrying no present element.
    rec->num_values = static_cast<std::int64_t>(lv.rep.size());
    rec->null_count = 0;
    for (std::size_t i = 0; i < lv.def.size(); ++i) {
        rec->null_count += (lv.def[i] < max_def) ? 1 : 0;
    }
    rec->null_count_known = true;
    rec->encoding = kEncPlain;

    // Element validity as the flat encoders expect: one byte per element.
    std::vector<std::uint8_t> ebits;
    const bool enullable = sch.element_nullable && elem->validity != nullptr;
    if (enullable) {
        ebits.resize(static_cast<std::size_t>(lv.n_elems), 1u);
        for (std::int64_t e = 0; e < lv.n_elems; ++e) {
            const std::int64_t bit = elem->validity_offset + e;
            ebits[static_cast<std::size_t>(e)] =
                ((elem->validity[bit >> 3] >> (bit & 7)) & 1u) ? 1u : 0u;
        }
    }
    if (w->opts.emit_statistics) {
        ParquetWriteColumn eschema = sch;
        eschema.type = sch.element_type;
        (void)compute_stats(*elem, 0, lv.n_elems,
                            enullable ? ebits.data() : nullptr, enullable,
                            &rec->st);
    }

    const std::uint64_t budget = page_budget_bytes(w->opts);
    ParquetWriteColumn eschema = sch;
    eschema.type = sch.element_type;
    std::vector<std::uint8_t> vals, payload;
    std::int64_t r0 = 0;
    std::uint32_t pages = 0;
    for (;;) {
        if (++pages > kPwMaxPagesPerChunk) return false;
        // Grow the page a row at a time until the slot budget is met. Cutting
        // anywhere else would split a record across pages, which parquet
        // forbids.
        std::int64_t r1 = r0;
        while (r1 < n_rows) {
            const std::int64_t slots = lv.row_slot[static_cast<std::size_t>(r1 + 1)] -
                                       lv.row_slot[static_cast<std::size_t>(r0)];
            if (r1 > r0 && static_cast<std::uint64_t>(slots) * 8u > budget) break;
            ++r1;
        }
        const std::int64_t s0 = lv.row_slot[static_cast<std::size_t>(r0)];
        const std::int64_t s1 = lv.row_slot[static_cast<std::size_t>(r1)];
        // Element range for these rows: count present elements before s0.
        std::int64_t e0 = 0;
        for (std::int64_t i = 0; i < s0; ++i) e0 += (lv.def[i] >= 2u) ? 1 : 0;
        std::int64_t e1 = e0;
        for (std::int64_t i = s0; i < s1; ++i) e1 += (lv.def[i] >= 2u) ? 1 : 0;

        vals.clear();
        if (!encode_plain_range(*elem, eschema, e0, e1,
                                enullable ? ebits.data() : nullptr, enullable,
                                &vals)) {
            return false;
        }
        payload.clear();
        // v1 order: repetition levels first, then definition levels.
        if (!encode_level_stream(lv.rep.data() + s0,
                                 static_cast<std::size_t>(s1 - s0), 1u,
                                 &payload)) {
            return false;
        }
        if (!encode_level_stream(lv.def.data() + s0,
                                 static_cast<std::size_t>(s1 - s0), max_def,
                                 &payload)) {
            return false;
        }
        payload.insert(payload.end(), vals.begin(), vals.end());

        std::int64_t off = 0;
        std::int32_t sz = 0;
        std::int64_t page_nulls = 0;
        for (std::int64_t i = s0; i < s1; ++i) {
            page_nulls += (lv.def[i] < max_def) ? 1 : 0;
        }
        PageSpec spec{false, false, s1 - s0, page_nulls, kEncPlain, 0u};
        if (!emit_page(w, out, payload, spec, rec, &off, &sz)) return false;
        if (rec->data_page_offset < 0) rec->data_page_offset = off;
        StatBuf ps;
        std::memset(&ps, 0, sizeof(ps));
        note_page(w, out, rec, off, sz, r0, page_nulls, s1 - s0, ps);
        r0 = r1;
        if (r0 >= n_rows) break;
    }
    (void)ws;
    return true;
}

// Build the def levels + statistics that both encodings need, choose the
// encoding, and delegate. Fallback from dictionary to PLAIN happens here and
// only here, so a chunk's data pages never disagree about their encoding.
bool write_column_chunk(ParquetWriter* w, ChunkWorkspace* ws, ChunkOut* out,
                        const BoltColumn& col,
                        const ParquetWriteColumn& sch,
                        std::int64_t n_rows) noexcept {
    assert(w != nullptr && out != nullptr);
    assert(n_rows >= 0);
    out->bytes.clear();
    out->pages.clear();
    out->bloom.clear();
    out->ok = false;
    ChunkRec* rec = &out->rec;
    std::memset(rec, 0, sizeof(*rec));
    rec->num_values = n_rows;
    rec->data_page_offset = -1;      // relative-0 is a real offset now
    rec->dict_page_offset = -1;

    if (col.type != sch.type) return false;
    if (col.length < n_rows) return false;
    if (sch.type == BoltType::List) {
        out->ok = chunk_write_list(w, ws, out, col, sch, n_rows, rec);
        return out->ok;
    }

    std::vector<std::uint8_t> def_bits;
    const bool nullable = sch.nullable;
    if (nullable) {
        def_bits.resize(static_cast<std::size_t>(n_rows));
        fill_def_bits(col, n_rows, def_bits.data(), &rec->null_count);
    }
    // G2FEAT-24: null_count is always known -- counted for OPTIONAL columns,
    // structurally 0 for REQUIRED ones -- so the footer always carries it.
    rec->null_count_known = true;
    const std::uint8_t* db = nullable ? def_bits.data() : nullptr;

    if (w->opts.emit_statistics &&
        !compute_stats(col, 0, n_rows, db, nullable, &rec->st)) {
        return false;
    }

    const PqWriteEncoding enc = resolve_encoding(sch, w->opts);
    bool ok;
    bool dict_ndv_exact = false;
    if (enc == PqWriteEncoding::Dictionary && n_rows > 0 &&
        dict_build_chunk(col, sch, n_rows, db, nullable,
                         dict_budget_bytes(w->opts), &ws->dict, &ws->dict_idx) &&
        ws->dict.count > 0u) {
        dict_ndv_exact = true;
        ok = chunk_write_dict(w, ws, out, col, sch, n_rows, db, nullable, ws->dict,
                              ws->dict_idx, rec);
    } else if (enc == PqWriteEncoding::Dictionary) {
        // Dictionary overflowed its ceiling (or the chunk is all-null):
        // PLAIN for the whole chunk. Nothing has been written yet, so this
        // costs only the abandoned build.
        ok = chunk_write_direct(w, ws, out, col, sch, PqWriteEncoding::Plain,
                                n_rows, db, nullable, rec);
    } else {
        ok = chunk_write_direct(w, ws, out, col, sch, enc, n_rows, db, nullable,
                                rec);
    }
    if (!ok) return false;
    if (!chunk_build_bloom(w, ws, out, col, sch, n_rows, db, nullable,
                           dict_ndv_exact, rec)) {
        return false;
    }
    out->ok = true;
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
    // 6 converted_type. A JSON/BSON annotation REPLACES the type's default
    // (a JSON column is not additionally UTF8-annotated -- parquet allows one
    // ConvertedType, and JSON already implies the bytes are a string).
    const BoltLogical lg = static_cast<BoltLogical>(c.logical);
    if (lg == BoltLogical::Json || lg == BoltLogical::Bson) {
        tc_put_field(o, 6, kFI32);
        tc_put_zigzag(o, (lg == BoltLogical::Json) ? kConvJson : kConvBson);
    } else if (has_converted(c.type)) {
        tc_put_field(o, 6, kFI32);
        tc_put_zigzag(o, bolt_to_pq_converted(c.type));
    }
    if (c.type == BoltType::Decimal128) {
        tc_put_field(o, 7, kFI32);
        tc_put_zigzag(o, c.scale);
        tc_put_field(o, 8, kFI32);
        tc_put_zigzag(o, c.precision);
    }
    // 10 logicalType. Modern readers prefer this over the legacy
    // ConvertedType; both are written so old and new readers agree. The union
    // arm is an empty struct -- JSON is 12, BSON is 13 in parquet.thrift.
    if (lg == BoltLogical::Json || lg == BoltLogical::Bson) {
        tc_put_field(o, 10, kFStruct);
        tc_put_field(o, (lg == BoltLogical::Json) ? 12 : 13, kFStruct);
        tc_put_stop(o);                      // empty JsonType / BsonType
        tc_put_stop(o);                      // end LogicalType union
    }
    tc_put_stop(o);
}

// A LIST column is three SchemaElements, not one:
//
//   optional group <name> (LIST) {        // num_children = 1
//     repeated group list {               // num_children = 1
//       <optional|required> <element>;    // the leaf
//     }
//   }
//
// That exact shape is what makes bolt's own reader derive list_def = 1 and
// rep_def = 2 (and pyarrow name the leaf "<name>.list.element"). Emitting a
// 2-level legacy list instead would still parse, but every consumer would
// compute different levels.
void write_schema_elements_list(TcOut* o,
                                const ParquetWriteColumn& c) noexcept {
    assert(c.type == BoltType::List);
    // 1. the outer LIST group
    tc_put_field(o, 3, kFI32);
    tc_put_zigzag(o, c.nullable ? kRepOptional : kRepRequired);
    tc_put_field(o, 4, kFBinary);
    tc_put_string(o, c.name);
    tc_put_field(o, 5, kFI32);
    tc_put_zigzag(o, 1);                       // num_children
    tc_put_field(o, 6, kFI32);
    tc_put_zigzag(o, kConvList);
    tc_put_stop(o);
    // 2. the repeated group. Named "list" because that is what parquet-mr,
    //    Arrow and pyarrow all emit; the name is not load-bearing for
    //    correctness but it is what tools display.
    tc_put_field(o, 3, kFI32);
    tc_put_zigzag(o, kRepRepeated);
    tc_put_field(o, 4, kFBinary);
    tc_put_string(o, "list");
    tc_put_field(o, 5, kFI32);
    tc_put_zigzag(o, 1);
    tc_put_stop(o);
    // 3. the element leaf
    ParquetWriteColumn leaf{};
    leaf.type = c.element_type;
    leaf.nullable = c.element_nullable;
    leaf.precision = c.precision;
    leaf.scale = c.scale;
    leaf.logical = c.logical;
    std::strncpy(leaf.name, "element", sizeof(leaf.name) - 1);
    write_schema_element_col(o, leaf);
}

// How many SchemaElements a column contributes to the footer.
std::uint32_t schema_element_count(const ParquetWriteColumn& c) noexcept {
    return (c.type == BoltType::List) ? 3u : 1u;
}

void write_statistics(TcOut* o, const ChunkRec& rec) noexcept {
    if (rec.null_count_known) {
        tc_put_field(o, 3, kFI64);
        tc_put_zigzag(o, rec.null_count);
    }
    if (rec.st.have_stats) {
        tc_put_field(o, 5, kFBinary);
        tc_put_binary(o, rec.st.max_buf, rec.st.max_len);
        tc_put_field(o, 6, kFBinary);
        tc_put_binary(o, rec.st.min_buf, rec.st.min_len);
    }
    tc_put_stop(o);
}

void write_column_meta(TcOut* o, const ParquetWriter* w,
                       const ParquetWriteColumn& sch,
                       const ChunkRec& rec) noexcept {
    tc_put_field(o, 1, kFI32);
    tc_put_zigzag(o, bolt_to_pq_physical(
        (sch.type == BoltType::List) ? sch.element_type : sch.type));
    // encodings: the set actually used by this chunk. RLE always appears --
    // it encodes the definition levels even for a REQUIRED column's absent
    // stream, and parquet-mr lists it unconditionally. A dictionary chunk
    // adds RLE_DICTIONARY for the data pages and keeps PLAIN for the
    // dictionary page itself.
    tc_put_field(o, 2, kFList);
    if (rec.dictionary) {
        tc_put_list_hdr(o, kFI32, 3);
        tc_put_zigzag(o, kEncPlain);      // the dictionary page itself
        tc_put_zigzag(o, kEncRle);
        tc_put_zigzag(o, kEncRleDict);
    } else if (rec.encoding != kEncPlain) {
        tc_put_list_hdr(o, kFI32, 2);
        tc_put_zigzag(o, kEncRle);
        tc_put_zigzag(o, rec.encoding);
    } else {
        tc_put_list_hdr(o, kFI32, 2);
        tc_put_zigzag(o, kEncPlain);
        tc_put_zigzag(o, kEncRle);
    }
    // path_in_schema. A LIST's chunk describes its LEAF, so the path is the
    // three-segment one the schema above declares.
    tc_put_field(o, 3, kFList);
    if (sch.type == BoltType::List) {
        tc_put_list_hdr(o, kFBinary, 3);
        tc_put_string(o, sch.name);
        tc_put_string(o, "list");
        tc_put_string(o, "element");
    } else {
        tc_put_list_hdr(o, kFBinary, 1);
        tc_put_string(o, sch.name);
    }
    // codec
    tc_put_field(o, 4, kFI32);
    tc_put_zigzag(o, pq_codec_code(w->opts.compression));
    tc_put_field(o, 5, kFI64);
    tc_put_zigzag(o, rec.num_values);
    tc_put_field(o, 6, kFI64);
    tc_put_zigzag(o, rec.total_unc);
    tc_put_field(o, 7, kFI64);
    tc_put_zigzag(o, rec.total_cmp);
    tc_put_field(o, 9, kFI64);
    tc_put_zigzag(o, rec.data_page_offset);
    // 11 dictionary_page_offset. Readers (bolt's own included) take the
    // chunk's byte region to start at the dictionary page when this is set,
    // so it must be the FIRST page of the chunk and total_compressed_size
    // must span from it -- which is why emit_page folds the dictionary page
    // into the same running totals.
    if (rec.dictionary && rec.dict_page_offset > 0) {
        tc_put_field(o, 11, kFI64);
        tc_put_zigzag(o, rec.dict_page_offset);
    }
    if (w->opts.emit_statistics && (rec.st.have_stats || rec.null_count_known)) {
        tc_put_field(o, 12, kFStruct);
        write_statistics(o, rec);
    }
    // 14/15 = bloom_filter_offset / bloom_filter_length. Length is the
    // header PLUS the bitset, which is what bolt's pq_read_bloom and
    // parquet-cpp both expect to bound the slice by.
    if (rec.bloom_len > 0) {
        tc_put_field(o, 14, kFI64);
        tc_put_zigzag(o, rec.bloom_off);
        tc_put_field(o, 15, kFI32);
        tc_put_zigzag(o, rec.bloom_len);
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
    // 4/5 = offset_index_offset/length, 6/7 = column_index_offset/length.
    // These live on ColumnChunk, NOT on ColumnMetaData -- a reader looks
    // here to find the index region without parsing any page.
    if (rec.off_index_len > 0) {
        tc_put_field(o, 4, kFI64);
        tc_put_zigzag(o, rec.off_index_off);
        tc_put_field(o, 5, kFI32);
        tc_put_zigzag(o, rec.off_index_len);
    }
    if (rec.col_index_len > 0) {
        tc_put_field(o, 6, kFI64);
        tc_put_zigzag(o, rec.col_index_off);
        tc_put_field(o, 7, kFI32);
        tc_put_zigzag(o, rec.col_index_len);
    }
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
    std::uint32_t n_elems = 1u;                // the root
    for (std::uint32_t i = 0; i < n; ++i) {
        n_elems += schema_element_count(w->opts.columns[i]);
    }
    tc_put_list_hdr(&o, kFStruct, n_elems);
    write_schema_element_root(&o, n);
    for (std::uint32_t i = 0; i < n; ++i) {
        if (w->opts.columns[i].type == BoltType::List) {
            write_schema_elements_list(&o, w->opts.columns[i]);
        } else {
            write_schema_element_col(&o, w->opts.columns[i]);
        }
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
    if (!pw_codec_supported(opts->compression)) return nullptr;
    for (std::uint32_t i = 0; i < opts->n_columns; ++i) {
        if (!type_supported(opts->columns[i])) return nullptr;
        // Reject an encoding the column's type cannot carry, loudly. Quietly
        // writing PLAIN instead would hide the caller's bug in a file that
        // reads back fine.
        if (opts->columns[i].encoding > static_cast<std::uint8_t>(
                PqWriteEncoding::ByteStreamSplit)) {
            return nullptr;
        }
        // A JSON annotation on an integer column is a caller bug; refuse it
        // rather than writing a file whose schema lies about its bytes.
        {
            const BoltLogical lg =
                static_cast<BoltLogical>(opts->columns[i].logical);
            const BoltType ct = opts->columns[i].type;
            if (lg == BoltLogical::Json && ct != BoltType::Utf8) return nullptr;
            if (lg == BoltLogical::Bson && ct != BoltType::Binary &&
                ct != BoltType::Utf8) {
                return nullptr;
            }
            if (opts->columns[i].logical > 
                static_cast<std::uint8_t>(BoltLogical::Variant)) {
                return nullptr;
            }
        }
        {
            const ParquetWriteColumn& pc = opts->columns[i];
            const BoltType et = (pc.type == BoltType::List) ? pc.element_type
                                                            : pc.type;
            if (!encoding_applies(
                    static_cast<PqWriteEncoding>(pc.encoding), et)) {
                return nullptr;
            }
        }
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
    w->to_mem = false;
    w->fout.open(path, std::ios::binary | std::ios::trunc);
    if (!w->fout.is_open()) {
        delete w;
        return nullptr;
    }
    static const char kMagic[4] = {'P', 'A', 'R', '1'};
    if (!sink_write(w, kMagic, 4) || !sink_ok(w)) {
        delete w;
        return nullptr;
    }
    w->file_pos = 4;
    return w;
}

ParquetWriter* parquet_write_open_mem(const ParquetWriteOpts* opts,
                                      std::uint64_t reserve_bytes) noexcept {
    assert(opts != nullptr);
    if (opts == nullptr) return nullptr;
    if (opts->n_columns == 0u || opts->n_columns > kPwMaxColumns) return nullptr;
    if (!pw_codec_supported(opts->compression)) return nullptr;
    for (std::uint32_t i = 0; i < opts->n_columns; ++i) {
        if (!type_supported(opts->columns[i])) return nullptr;
        // Reject an encoding the column's type cannot carry, loudly. Quietly
        // writing PLAIN instead would hide the caller's bug in a file that
        // reads back fine.
        if (opts->columns[i].encoding > static_cast<std::uint8_t>(
                PqWriteEncoding::ByteStreamSplit)) {
            return nullptr;
        }
        // A JSON annotation on an integer column is a caller bug; refuse it
        // rather than writing a file whose schema lies about its bytes.
        {
            const BoltLogical lg =
                static_cast<BoltLogical>(opts->columns[i].logical);
            const BoltType ct = opts->columns[i].type;
            if (lg == BoltLogical::Json && ct != BoltType::Utf8) return nullptr;
            if (lg == BoltLogical::Bson && ct != BoltType::Binary &&
                ct != BoltType::Utf8) {
                return nullptr;
            }
            if (opts->columns[i].logical > 
                static_cast<std::uint8_t>(BoltLogical::Variant)) {
                return nullptr;
            }
        }
        {
            const ParquetWriteColumn& pc = opts->columns[i];
            const BoltType et = (pc.type == BoltType::List) ? pc.element_type
                                                            : pc.type;
            if (!encoding_applies(
                    static_cast<PqWriteEncoding>(pc.encoding), et)) {
                return nullptr;
            }
        }
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
    w->to_mem = true;
    // One allocation up front instead of a growth curve. Capped so a bad
    // estimate cannot commit an unbounded reservation.
    constexpr std::uint64_t kMaxReserve = 1ull << 31;   // 2 GiB
    if (reserve_bytes != 0u) {
        w->mem.reserve(static_cast<std::size_t>(
            reserve_bytes < kMaxReserve ? reserve_bytes : kMaxReserve));
    }
    static const char kMagic[4] = {'P', 'A', 'R', '1'};
    if (!sink_write(w, kMagic, 4)) {
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
// One wave of concurrent column encodes. `base` is the first column, `n` the
// wave width; slot i encodes column base + i using workspace i and output i.
struct EncodeWave {
    ParquetWriter*    w;
    const BoltColumn* cols;
    std::int64_t      start;
    std::int64_t      rows;
    std::uint32_t     base;
};

// Runs on a pool worker. Touches only opts (read-only for the duration of the
// write), its own column's input, and the workspace/output at its own slot --
// so no two tasks share mutable state and none of them touches file_pos, the
// chunk list or the page list. Placement does all of that, serially, after.
void encode_wave_task(void* user, std::uint32_t lo, std::uint32_t hi,
                      std::uint32_t /*thread_id*/) noexcept {
    EncodeWave* e = static_cast<EncodeWave*>(user);
    assert(e != nullptr);
    assert(lo <= hi);
    for (std::uint32_t i = lo; i < hi; ++i) {
        const std::uint32_t c = e->base + i;
        assert(c < e->w->opts.n_columns);
        const BoltColumn sc = slice_column(e->cols[c],
                                           e->w->opts.columns[c].type,
                                           e->start, e->rows);
        // Failure is recorded on the output (ChunkOut::ok) rather than
        // returned: a pool task has nowhere to return a status to, and the
        // placement pass checks every slot before writing anything.
        (void)write_column_chunk(e->w, &e->w->ws[i], &e->w->outs[i], sc,
                                 e->w->opts.columns[c], e->rows);
    }
}

// How many columns to encode at once. One workspace per concurrent encode,
// so this is also the scratch bound. Serial (1) unless a pool was supplied.
std::uint32_t encode_wave_width(const ParquetWriter* w) noexcept {
    assert(w != nullptr);
    if (w->opts.encode_pool == nullptr) return 1u;
    if (w->opts.n_columns <= 1u) return 1u;
    const std::uint32_t threads = w->opts.encode_pool->thread_count();
    std::uint32_t width = (threads > 0u) ? threads : 1u;
    if (width > w->opts.n_columns) width = w->opts.n_columns;
    if (width > kPwMaxEncodeWave) width = kPwMaxEncodeWave;
    assert(width >= 1u);
    return width;
}

// Place one encoded chunk into the file: append its bytes, then rebase every
// offset it recorded by the position they landed at. This is the ONLY place
// that turns a relative offset into an absolute one, which is what keeps the
// encode side ignorant of where in the file it will end up -- and therefore
// safe to run off the write path.
bool place_chunk(ParquetWriter* w, ChunkOut* out) noexcept {
    assert(w != nullptr && out != nullptr);
    if (!out->ok) return false;
    const std::int64_t base = w->file_pos;
    if (!sink_write(w, out->bytes.data(), out->bytes.size())) return false;
    if (!sink_ok(w)) return false;
    w->file_pos += static_cast<std::int64_t>(out->bytes.size());

    ChunkRec rec = out->rec;
    assert(rec.data_page_offset >= 0);
    rec.data_page_offset += base;
    if (rec.dictionary) {
        assert(rec.dict_page_offset >= 0);
        rec.dict_page_offset += base;
    } else {
        rec.dict_page_offset = 0;      // 0 = absent, per the footer contract
    }
    rec.page_off = static_cast<std::uint32_t>(w->pages.size());
    assert(rec.page_count == out->pages.size());
    for (std::size_t i = 0; i < out->pages.size(); ++i) {
        PageRec pr = out->pages[i];
        pr.offset += base;
        w->pages.push_back(pr);
    }
    w->chunks.push_back(rec);
    if (!out->bloom.empty()) {
        w->rg_blooms.push_back(out->bloom);
        w->rg_bloom_chunk.push_back(
            static_cast<std::uint32_t>(w->chunks.size() - 1u));
    }
    return true;
}

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

    const std::uint32_t width = encode_wave_width(w);
    if (w->ws.size() < width) w->ws.resize(width);
    if (w->outs.size() < width) w->outs.resize(width);

    // Encode a wave of columns, then place that wave IN COLUMN ORDER. The
    // placement order is what makes the bytes independent of the pool size:
    // whatever order the workers finished in, column c always lands before
    // column c+1.
    for (std::uint32_t base = 0; base < w->opts.n_columns; base += width) {
        std::uint32_t n = w->opts.n_columns - base;
        if (n > width) n = width;
        if (n == 1u || w->opts.encode_pool == nullptr) {
            for (std::uint32_t i = 0; i < n; ++i) {
                const std::uint32_t c = base + i;
                const BoltColumn sc = slice_column(cols[c],
                                                   w->opts.columns[c].type,
                                                   start, rows);
                if (!write_column_chunk(w, &w->ws[i], &w->outs[i], sc,
                                        w->opts.columns[c], rows)) {
                    return false;
                }
            }
        } else {
            // Invalidate every slot BEFORE dispatching. The slots are reused
            // across waves, and submit_range drops a task silently if its
            // payload pool is exhausted -- a slot still carrying the previous
            // wave's ok=true would then be placed into the file as that
            // column's chunk. Clearing here turns a dropped task into a clean
            // write failure instead of silently wrong bytes.
            for (std::uint32_t i = 0; i < n; ++i) w->outs[i].ok = false;
            EncodeWave e{w, cols, start, rows, base};
            // grain 1: one task per column, so a slow column cannot hold a
            // fast one hostage behind it in the same task.
            w->opts.encode_pool->submit_range(&encode_wave_task, &e, n, 1u);
            w->opts.encode_pool->wait_all();
        }
        for (std::uint32_t i = 0; i < n; ++i) {
            if (!place_chunk(w, &w->outs[i])) return false;
        }
    }

    rg.total_byte_size = w->file_pos - rg_start;
    // Flush this row group's bloom filters now (parquet-mr's AFTER_ROWGROUP
    // placement). Doing it here rather than at close is what keeps live
    // filter memory to one row group; at 256 columns the difference is real.
    // The bytes deliberately do NOT count toward any chunk's
    // total_compressed_size -- that field bounds the chunk's PAGE region,
    // and a reader walking pages past its end would fail.
    for (std::size_t i = 0; i < w->rg_blooms.size(); ++i) {
        const std::vector<std::uint8_t>& bf = w->rg_blooms[i];
        if (bf.size() > 0x7FFFFFFFull) return false;
        const std::uint32_t ci = w->rg_bloom_chunk[i];
        assert(ci < w->chunks.size());
        w->chunks[ci].bloom_off = w->file_pos;
        w->chunks[ci].bloom_len = static_cast<std::int32_t>(bf.size());
        if (!sink_write(w, bf.data(), bf.size())) return false;
        w->file_pos += static_cast<std::int64_t>(bf.size());
    }
    w->rg_blooms.clear();
    w->rg_bloom_chunk.clear();
    if (!sink_ok(w)) return false;
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

namespace {

// Footer + length + trailing magic. Shared by both close paths so a
// memory-written file cannot drift from a file-written one.
bool finish_footer(ParquetWriter* w) noexcept {
    assert(w != nullptr);
    assert(!w->failed);
    // The page index sits between the last row group and the footer, and
    // writing it fills in the ColumnChunk locators the footer then emits --
    // so it must run BEFORE write_file_metadata, not after.
    if (!write_page_index(w)) return false;
    std::vector<std::uint8_t> footer;
    write_file_metadata(&footer, w);
    const std::uint32_t flen = static_cast<std::uint32_t>(footer.size());
    if (!sink_write(w, footer.data(), footer.size())) return false;
    const std::uint8_t lb[4] = {
        static_cast<std::uint8_t>(flen & 0xFFu),
        static_cast<std::uint8_t>((flen >> 8) & 0xFFu),
        static_cast<std::uint8_t>((flen >> 16) & 0xFFu),
        static_cast<std::uint8_t>((flen >> 24) & 0xFFu),
    };
    if (!sink_write(w, lb, 4)) return false;
    static const char kMagic[4] = {'P', 'A', 'R', '1'};
    if (!sink_write(w, kMagic, 4)) return false;
    return sink_ok(w);
}

}  // namespace

bool parquet_write_close(ParquetWriter* w) noexcept {
    assert(w != nullptr);
    if (w == nullptr) return false;
    bool ok = !w->failed && sink_ok(w);
    if (ok) ok = finish_footer(w);
    w->fout.close();
    delete w;
    return ok;
}

bool parquet_write_close_mem(ParquetWriter* w, Arena* arena,
                             const std::uint8_t** out,
                             std::uint64_t* out_len) noexcept {
    assert(w != nullptr);
    assert(arena != nullptr);
    assert(out != nullptr && out_len != nullptr);
    if (w == nullptr) return false;
    if (arena == nullptr || out == nullptr || out_len == nullptr) {
        delete w;
        return false;
    }
    // A writer opened on a PATH has already streamed its bytes to disk; there
    // is nothing in `mem` to hand back, and returning an empty buffer would
    // look like a valid zero-row file.
    if (!w->to_mem) {
        delete w;
        return false;
    }
    bool ok = !w->failed && finish_footer(w);
    if (ok) {
        const std::uint64_t n = w->mem.size();
        std::uint8_t* dst = arena->allocate_array<std::uint8_t>(n);
        if (dst != nullptr) {
            std::memcpy(dst, w->mem.data(), n);
            *out     = dst;
            *out_len = n;
        } else {
            ok = false;
        }
    }
    delete w;
    return ok;
}

}  // namespace parquet
}  // namespace ingest
}  // namespace bolt
